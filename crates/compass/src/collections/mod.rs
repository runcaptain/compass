// collections/mod.rs — CollectionManager v2: named vector spaces, relationships, scoring.
//
// Each collection now has:
//   - A Tantivy FTS index (shared across all vector spaces)
//   - Multiple named USearch HNSW indices (one per vector space)
//   - A relationship store (parent-child + sibling grouping)
//   - Precomputed bitset facets for microsecond metadata faceting
//
// The manager handles: create, load on startup, ingest with batch parent resolution,
// search with full scoring pipeline, vector space CRUD, background rebuild jobs.

pub mod cloud;
pub mod rebuild;
pub mod relation_store;
pub mod relationships;
pub mod store;

use crate::embed::EmbedState;
use crate::models::*;
use crate::scoring::{self, ScoredCandidate};
use crate::search::chunk_cache::ChunkCache;
use crate::search::chunk_store::ChunkStore;
use crate::search::filter_index::{selectivity, FilterIndex};
use crate::search::filter_pushdown::FilterExpr;
use crate::search::hybrid;
use crate::search::tantivy_fts::{self, FtsState};
use crate::search::vector::{self, hnsw_ef_search_default, VectorState};
use crate::search::SearchMode;
use crate::storage::Storage;
use chrono::Utc;
use rebuild::RebuildTracker;
use relation_store::RelationStore;
use relationships::RelationshipStore;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Validate that a user-supplied name segment is safe to interpolate into
/// on-disk paths. Used for both collection names and vector-space names —
/// both end up as path components (e.g. `data/<collection>/vectors/<space>.bin`),
/// so an unconstrained value like `../../tmp/pwn` could write or delete
/// arbitrary files. `kind` is the noun used in the error message
/// ("Collection", "Vector space", ...).
pub(crate) fn validate_name_segment(
    name: &str,
    kind: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return Err(format!(
            "{kind} name '{name}' is invalid. Use letters, digits, and hyphens only (e.g. 'my-name')."
        )
        .into());
    }
    Ok(())
}

/// A loaded collection with all its search indices in memory.
/// Which manifest seqs this node has applied to its local indexes.
///
/// `contiguous` is the count of contiguously-applied seqs (fragments
/// `0..contiguous` are reflected locally); `out_of_band` holds seqs this node
/// applied AHEAD of the contiguous frontier — its own appends land locally at
/// commit time while earlier REMOTE fragments may still be unapplied, so a
/// single watermark would silently skip those remote fragments forever. The
/// refresher advances `contiguous` in seq order, draining `out_of_band`.
#[derive(Debug, Default, Clone)]
struct SeqTracker {
    contiguous: u64,
    out_of_band: std::collections::BTreeSet<u64>,
}

impl SeqTracker {
    fn starting_at(contiguous: u64) -> Self {
        Self {
            contiguous,
            ..Default::default()
        }
    }

    /// Has this seq been applied locally (either side of the frontier)?
    fn covers(&self, seq: u64) -> bool {
        seq < self.contiguous || self.out_of_band.contains(&seq)
    }

    /// Record a locally-applied seq and advance the contiguous frontier.
    fn mark(&mut self, seq: u64) {
        if seq < self.contiguous {
            return;
        }
        self.out_of_band.insert(seq);
        while self.out_of_band.remove(&self.contiguous) {
            self.contiguous += 1;
        }
    }
}

struct LoadedCollection {
    metadata: Collection,
    /// Per-space count of ingest batches since the HNSW index was last saved
    /// (saving rewrites the whole index file — O(index) per batch was a scale
    /// wall). A stale on-disk index is detected at load (size < keymap) and
    /// rebuilt from the mmap file. u32::MAX means "no mutable in-RAM index".
    hnsw_unsaved: HashMap<String, u32>,
    /// LRU stamp for lazy-attach eviction (process-monotonic tick).
    last_used: std::sync::atomic::AtomicU64,
    /// Manifest seqs applied to this node's local indexes (see [`SeqTracker`]).
    applied: SeqTracker,
    /// Cloud-mode id pool: ranges CAS-leased from `{ns}/id-alloc`. In cloud
    /// mode ids are ONLY taken from here (never from `next_id`, which becomes
    /// a diagnostic high-water mark) so attached nodes and stateless writers
    /// can never mint colliding ids.
    id_pool: std::collections::VecDeque<std::ops::Range<u64>>,
    fts: FtsState,
    /// Named vector spaces, each with its own USearch HNSW index.
    /// Arc-wrapped so search can clone cheaply and run in spawn_blocking.
    vector_spaces: HashMap<String, Arc<VectorState>>,
    /// Document relationships (parent-child + sibling groups)
    relationships: RelationshipStore,
    /// Bounded read-through cache over the disk-backed chunk store. Chunks
    /// are NOT held wholesale in RAM anymore — serving memory is O(cache
    /// budget), not O(collection). Existence checks go through the filter
    /// index universe (live ids as a treemap).
    chunk_store: ChunkCache,
    /// Disk-backed typed many-to-many chunk relations. Source of truth on disk;
    /// read on demand at search time (never rehydrated into RAM).
    relation_store: RelationStore,
    /// Soft-delete tombstones: chunk ids marked deleted. In-RAM `HashSet` for
    /// O(1) filtering at search time, rehydrated on startup from the chunk
    /// store's tombstone table. HNSW/FTS still physically contain these ids;
    /// search filters them out. Physical removal happens on rebuild/compaction.
    tombstones: std::collections::HashSet<u64>,
    /// Next auto-increment ID for new chunks
    next_id: u64,
    /// Roaring-bitmap filter index over `chunks`. Rebuilt alongside the FTS
    /// and HNSW indexes on every ingest batch, and on load from the rehydrated
    /// chunks. Powers filter-aware ANN: queries with `filters={...}` compile
    /// to a `FilterExpr`, resolve to an eligible bitmap, and route through
    /// USearch's `filtered_search`. Planned follow-up.
    filter_index: FilterIndex,
}

/// Manages all collections. Thread-safe via Arc<RwLock<...>>.
pub struct CollectionManager {
    data_dir: PathBuf,
    collections: RwLock<HashMap<String, LoadedCollection>>,
    pub rebuild_tracker: RebuildTracker,
    /// Source-of-truth storage backend. Local disk (redb/mmap, the default) or
    /// object storage. In object-storage mode, ingest also mirrors each batch
    /// into the LSM (WAL fragments + CAS manifest) so data is S3-native.
    storage: Arc<dyn Storage>,
    /// True when `storage` is a cloud object-storage backend (not local disk).
    /// Gates the LSM write path so local deployments are unaffected.
    cloud_mode: bool,
    /// Namespaces with an auto-compaction currently in flight. Single-flights
    /// background compaction so concurrent triggers don't each write (and, on
    /// CAS loss, leak) a full segment.
    compacting: Arc<std::sync::Mutex<std::collections::HashSet<String>>>,
    /// Node role (COMPASS_ROLE). Writer = durable-append-only ingest with no
    /// local indexes; Full = today's behavior. Cloud mode only.
    role: NodeRole,
    /// Stateless-writer id pools, keyed by namespace (attached collections
    /// pool on `LoadedCollection.id_pool` instead).
    writer_pools:
        tokio::sync::Mutex<HashMap<String, std::collections::VecDeque<std::ops::Range<u64>>>>,
    /// Cache of bucket collection configs for stateless-writer validation.
    bucket_configs: tokio::sync::RwLock<HashMap<String, cloud::BucketConfig>>,
    /// Lazy attach (COMPASS_LAZY_ATTACH): namespaces discovered in the bucket
    /// but not yet attached. Attach happens on first request.
    registered: tokio::sync::RwLock<std::collections::HashSet<String>>,
    /// Per-namespace attach mutexes: a request stampede on a cold namespace
    /// rebuilds ONCE, without holding the global collections lock.
    attach_locks: tokio::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    /// Lazy attach enabled (cloud mode + COMPASS_LAZY_ATTACH=true).
    lazy_attach: bool,
    /// LRU budget for attached collections (COMPASS_MAX_ATTACHED; 0 = unbounded).
    max_attached: usize,
}

/// What this node does. Parsed from `COMPASS_ROLE` (default `full`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRole {
    /// Serve reads and writes with full local indexes (default).
    Full,
    /// Durable-append-only writes; no local indexes, no read serving.
    Writer,
}

impl NodeRole {
    fn from_env() -> Self {
        match std::env::var("COMPASS_ROLE").as_deref() {
            Ok("writer") => NodeRole::Writer,
            _ => NodeRole::Full,
        }
    }
}

impl CollectionManager {
    /// Create a manager with local-disk storage (the default embedded mode).
    /// Convenience wrapper used by tests and local-only callers.
    pub async fn new(
        data_dir: &Path,
    ) -> Result<Arc<Self>, Box<dyn std::error::Error + Send + Sync>> {
        let storage: Arc<dyn Storage> =
            Arc::new(crate::storage::local::LocalDiskStorage::new(data_dir)?);
        Self::new_with_storage(data_dir, storage).await
    }

    /// Create a manager with an explicit storage backend and load existing
    /// collections. `main.rs` passes the backend selected by `COMPASS_STORAGE`.
    pub async fn new_with_storage(
        data_dir: &Path,
        storage: Arc<dyn Storage>,
    ) -> Result<Arc<Self>, Box<dyn std::error::Error + Send + Sync>> {
        let role = NodeRole::from_env();
        Self::new_with_storage_role(data_dir, storage, role).await
    }

    /// Like [`new_with_storage`] with an explicit node role (used by tests;
    /// `new_with_storage` parses `COMPASS_ROLE`).
    pub async fn new_with_storage_role(
        data_dir: &Path,
        storage: Arc<dyn Storage>,
        role: NodeRole,
    ) -> Result<Arc<Self>, Box<dyn std::error::Error + Send + Sync>> {
        let lazy = std::env::var("COMPASS_LAZY_ATTACH")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);
        let max_attached = std::env::var("COMPASS_MAX_ATTACHED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let refresh_interval_secs = std::env::var("COMPASS_REFRESH_INTERVAL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);
        Self::new_with_storage_opts(
            data_dir,
            storage,
            role,
            lazy,
            max_attached,
            refresh_interval_secs,
        )
        .await
    }

    /// Fully-explicit constructor (role + lazy-attach + LRU budget), used by
    /// tests to avoid process-global env races and by callers embedding
    /// Compass as a library.
    pub async fn new_with_storage_opts(
        data_dir: &Path,
        storage: Arc<dyn Storage>,
        role: NodeRole,
        lazy_attach: bool,
        max_attached: usize,
        refresh_interval_secs: u64,
    ) -> Result<Arc<Self>, Box<dyn std::error::Error + Send + Sync>> {
        std::fs::create_dir_all(data_dir)?;

        // Clean up any stale rebuild directories from crashes
        rebuild::cleanup_stale_rebuilds(data_dir);

        let cloud_mode = storage.backend_name() != "local-disk";
        // The writer role is meaningless without a shared bucket; force Full
        // in local mode so a stray COMPASS_ROLE can't disable local serving.
        let role = if cloud_mode { role } else { NodeRole::Full };
        if role == NodeRole::Writer {
            tracing::info!("Node role: writer (durable-append-only; no read serving)");
        }
        let manager = Arc::new(Self {
            data_dir: data_dir.to_path_buf(),
            collections: RwLock::new(HashMap::new()),
            rebuild_tracker: rebuild::new_tracker(),
            storage,
            cloud_mode,
            compacting: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
            role,
            writer_pools: tokio::sync::Mutex::new(HashMap::new()),
            bucket_configs: tokio::sync::RwLock::new(HashMap::new()),
            registered: tokio::sync::RwLock::new(std::collections::HashSet::new()),
            attach_locks: tokio::sync::Mutex::new(HashMap::new()),
            lazy_attach: cloud_mode && lazy_attach,
            max_attached,
        });

        // Writer role: no local collections, no recovery — the node serves
        // durable appends only, validated against bucket configs. Boot is
        // instant regardless of how much data lives in the bucket.
        if manager.role == NodeRole::Writer {
            return Ok(manager);
        }

        // Load existing collections from local disk.
        let names = store::list_collection_names(data_dir)?;
        for name in &names {
            if let Err(e) = manager.load_collection(name).await {
                tracing::error!("Failed to load collection '{}': {}", name, e);
            }
        }
        if !names.is_empty() {
            tracing::info!("Loaded {} collection(s) from disk", names.len());
        }

        // Cloud mode: object storage is the source of truth. Discover any
        // collections that exist in S3 but not on local disk (e.g. an ephemeral
        // node with a fresh disk after a restart) and rebuild their local
        // indexes from the manifest. This is what makes cloud mode actually
        // durable-without-local-state.
        if cloud_mode {
            match crate::storage::lsm::list_namespaces(manager.storage.as_ref()).await {
                Ok(cloud_names) => {
                    let mut recovered = 0usize;
                    for ns in &cloud_names {
                        let already = {
                            let c = manager.collections.read().await;
                            c.contains_key(ns)
                        };
                        if already {
                            continue;
                        }
                        if manager.lazy_attach {
                            // Lazy mode: register only — attach on first
                            // request. Boot cost is O(namespaces), not O(data).
                            manager.registered.write().await.insert(ns.clone());
                            continue;
                        }
                        match manager.rebuild_collection_from_storage(ns).await {
                            Ok(n) => {
                                recovered += 1;
                                tracing::info!(
                                    "Recovered collection '{}' from object storage ({} chunks)",
                                    ns,
                                    n
                                );
                            }
                            Err(e) => {
                                tracing::error!("Failed to recover '{}' from storage: {}", ns, e)
                            }
                        }
                    }
                    if recovered > 0 {
                        tracing::info!("Recovered {} collection(s) from object storage", recovered);
                    }
                    if manager.lazy_attach {
                        let n = manager.registered.read().await.len();
                        if n > 0 {
                            tracing::info!("Registered {} collection(s) for lazy attach", n);
                        }
                    }
                }
                Err(e) => tracing::error!("Could not list collections from object storage: {}", e),
            }
        }

        // Background manifest refresher: keeps this node's local indexes
        // converged with fragments written by OTHER nodes (stateless writers,
        // other serving nodes). COMPASS_REFRESH_INTERVAL seconds, default 5,
        // 0 disables. Holds only a Weak — the task dies with the manager.
        if cloud_mode {
            let interval_secs = refresh_interval_secs;
            if interval_secs > 0 {
                let weak = Arc::downgrade(&manager);
                tokio::spawn(async move {
                    loop {
                        tokio::time::sleep(std::time::Duration::from_secs(interval_secs)).await;
                        let Some(m) = weak.upgrade() else { break };
                        m.refresh_all().await;
                    }
                });
            }
        }

        Ok(manager)
    }

    /// Load a single collection from disk into memory.
    async fn load_collection(
        &self,
        name: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let metadata = store::load_metadata(&self.data_dir, name)?;
        let tantivy_dir = store::tantivy_dir(&self.data_dir, name);
        let vectors_dir = store::vectors_dir(&self.data_dir, name);

        // Open the Tantivy FTS index
        let fts = if tantivy_dir.join("meta.json").exists() {
            tantivy_fts::open_index(&tantivy_dir)?
        } else {
            tantivy_fts::build_index(&tantivy_dir, &[], 0)?
        };

        // Load each named vector space from disk
        let mut vector_spaces = HashMap::new();
        for (space_name, space_config) in &metadata.vector_spaces {
            let index_path = vectors_dir.join(format!("{}.index", space_name));
            let vecs_path = vectors_dir.join(format!("{}.bin", space_name));
            if vecs_path.exists() {
                match vector::load_vector_index(&index_path, &vecs_path, space_config.dims) {
                    Ok(vs) => {
                        vector_spaces.insert(space_name.clone(), Arc::new(vs));
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to load vector space '{}' for '{}': {}",
                            space_name,
                            name,
                            e
                        );
                    }
                }
            } else {
                // Empty vector space (no vectors yet)
                vector_spaces.insert(
                    space_name.clone(),
                    Arc::new(VectorState {
                        index: None,
                        key_to_chunk_id: Vec::new(),
                        mmap_vectors: None,
                        vectors: Vec::new(),
                        dims: space_config.dims,
                    }),
                );
            }
        }

        // Load relationship store
        let rel_path = store::collection_dir(&self.data_dir, name).join("relationships.bin");
        let relationships = RelationshipStore::load(&rel_path)?;

        // Open the persistent chunk store and rehydrate the in-memory cache.
        // The cache is what scoring / search response assembly reads from; the
        // store is what survives process restarts and crashes.
        let chunks_db = store::chunks_db_path(&self.data_dir, name);
        // Ensure the parent dir exists in case the collection has never had
        // chunks ingested yet (e.g. older deployments).
        if let Some(parent) = chunks_db.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let chunk_store = ChunkCache::new(ChunkStore::open(&chunks_db)?);
        let mut max_seen_id: u64 = 0;
        let mut rehydrated_count: usize = 0;
        let mut filter_index = FilterIndex::new();
        let tombstones_vec = chunk_store.load_tombstones()?;
        let tombstones: std::collections::HashSet<u64> = tombstones_vec.into_iter().collect();
        chunk_store.for_each(|id, chunk| {
            if id >= max_seen_id {
                max_seen_id = id;
            }
            rehydrated_count += 1;
            if !tombstones.contains(&id) {
                filter_index.insert(id, &filter_meta(&chunk));
            }
        })?;
        filter_index.finalize();
        // next_id is a MONOTONIC high-water mark that must never regress or reuse
        // an id. Take the max of: the persisted metadata.next_id (survives even
        // when the local chunk store is empty on a cold restart), and one past
        // the highest id actually seen on disk. Using chunk_count here would be a
        // bug: soft-delete makes chunk_count a *live* count, so it can be far
        // below the highest assigned id → id reuse → overwriting existing chunks.
        let from_disk = if rehydrated_count > 0 {
            max_seen_id + 1
        } else {
            0
        };
        let next_id = metadata.next_id.max(from_disk).max(metadata.chunk_count);

        let chunk_count = metadata.chunk_count;
        let relations_db = store::relations_db_path(&self.data_dir, name);
        let relation_store = RelationStore::open(&relations_db)?;
        let loaded = LoadedCollection {
            id_pool: Default::default(),
            hnsw_unsaved: HashMap::new(),
            last_used: std::sync::atomic::AtomicU64::new(next_lru_tick()),
            // Persistent-disk restart: local indexes reflect fragments
            // 0..applied_seq (persisted on every apply); the refresher applies
            // the delta instead of a full rebuild.
            applied: SeqTracker::starting_at(metadata.applied_seq),
            next_id,
            metadata,
            fts,
            vector_spaces,
            relationships,
            chunk_store,
            relation_store,
            tombstones,
            filter_index,
        };

        let mut collections = self.collections.write().await;
        collections.insert(name.to_string(), loaded);
        tracing::info!(
            "Loaded collection '{}' ({} chunks declared, {} rehydrated from disk, {} vector spaces)",
            name,
            chunk_count,
            rehydrated_count,
            collections
                .get(name)
                .map(|c| c.vector_spaces.len())
                .unwrap_or(0)
        );

        Ok(())
    }

    /// Create a new empty collection.
    pub async fn create_collection(
        &self,
        name: &str,
        vector_spaces: Option<HashMap<String, VectorSpaceConfig>>,
        embedding_dims: Option<usize>,
        config: Option<CollectionConfig>,
    ) -> Result<Collection, Box<dyn std::error::Error + Send + Sync>> {
        if self.role == NodeRole::Writer {
            return Err(
                "this node runs in writer role; create collections via a serving node".into(),
            );
        }
        validate_name_segment(name, "Collection")?;

        let collection = {
            let mut collections = self.collections.write().await;
            if collections.contains_key(name) {
                return Err(format!("Collection '{}' already exists", name).into());
            }

            // Build vector spaces config: use explicit spaces, or create a "default" space
            let spaces = vector_spaces.unwrap_or_else(|| {
                let dims = embedding_dims.unwrap_or(384);
                let mut m = HashMap::new();
                m.insert(
                    "default".to_string(),
                    VectorSpaceConfig {
                        dims,
                        model: "bge-small-en-v1.5".to_string(),
                        status: "active".to_string(),
                    },
                );
                m
            });

            let default_space = spaces.keys().next().cloned();
            let dims = spaces.values().next().map(|s| s.dims).unwrap_or(384);

            let collection = Collection {
                name: name.to_string(),
                created_at: Utc::now(),
                vector_spaces: spaces,
                default_vector_space: default_space,
                embedding_dims: dims,
                chunk_count: 0,
                next_id: 0,
                config: config.unwrap_or_default(),
                applied_seq: 0,
            };

            store::save_metadata(&self.data_dir, &collection)?;

            // Build empty FTS index
            let tantivy_dir = store::tantivy_dir(&self.data_dir, name);
            let fts = tantivy_fts::build_index(&tantivy_dir, &[], 0)?;

            // Create empty vector spaces
            let mut vs_map = HashMap::new();
            for (sname, sconfig) in &collection.vector_spaces {
                vs_map.insert(
                    sname.clone(),
                    Arc::new(VectorState {
                        index: None,
                        key_to_chunk_id: Vec::new(),
                        mmap_vectors: None,
                        vectors: Vec::new(),
                        dims: sconfig.dims,
                    }),
                );
            }

            // Open the disk-backed chunk store for the new collection. Empty
            // database file is created at <data_dir>/<name>/chunks.redb.
            let chunks_db = store::chunks_db_path(&self.data_dir, name);
            if let Some(parent) = chunks_db.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let chunk_store = ChunkCache::new(ChunkStore::open(&chunks_db)?);
            let relations_db = store::relations_db_path(&self.data_dir, name);
            let relation_store = RelationStore::open(&relations_db)?;

            let loaded = LoadedCollection {
                id_pool: Default::default(),
                hnsw_unsaved: HashMap::new(),
                last_used: std::sync::atomic::AtomicU64::new(next_lru_tick()),
                applied: SeqTracker::default(),
                metadata: collection.clone(),
                fts,
                vector_spaces: vs_map,
                relationships: RelationshipStore::new(),
                chunk_store,
                relation_store,
                tombstones: std::collections::HashSet::new(),
                next_id: 0,
                filter_index: FilterIndex::new(),
            };

            collections.insert(name.to_string(), loaded);
            collection
        }; // write lock released — never hold it across S3 round-trips.

        // Cloud mode: make the collection exist DURABLY in the bucket —
        // create-only config object + empty manifest — so a zero-ingest
        // collection is discoverable from a fresh disk and stateless writers
        // can validate against its config. On any bucket failure, roll the
        // local creation back so local and bucket state agree (= absent).
        if self.cloud_mode {
            let rollback_local = || async {
                self.collections.write().await.remove(name);
                let _ = store::delete_collection_data(&self.data_dir, name);
            };
            let bucket_cfg = cloud::BucketConfig::from_collection(&collection);
            match cloud::write_bucket_config_if_absent(self.storage.as_ref(), name, &bucket_cfg)
                .await
            {
                Ok(()) => {}
                Err(crate::storage::StorageError::AlreadyExists(_)) => {
                    rollback_local().await;
                    return Err(
                        format!("Collection '{}' already exists in object storage", name).into(),
                    );
                }
                Err(e) => {
                    rollback_local().await;
                    return Err(format!("bucket config write failed: {e}").into());
                }
            }
            match crate::storage::lsm::init_namespace(self.storage.as_ref(), name).await {
                Ok(()) => {
                    // Fresh namespace: seed the id allocator at 0 so every
                    // ingest path (attached or stateless) can claim blocks.
                    if let Err(e) =
                        crate::storage::id_alloc::seed(self.storage.as_ref(), name, 0).await
                    {
                        let _ = crate::storage::lsm::delete_namespace(self.storage.as_ref(), name)
                            .await;
                        rollback_local().await;
                        return Err(format!("id allocator seed failed: {e}").into());
                    }
                }
                Err(crate::storage::StorageError::AlreadyExists(_)) => {
                    // Data exists in the bucket without a config (pre-v0.4
                    // namespace): this create collides with real data. Remove
                    // the config we just wrote and refuse.
                    let _ = self.storage.delete(&cloud::config_key(name)).await;
                    rollback_local().await;
                    return Err(
                        format!("namespace '{}' already has data in object storage", name).into(),
                    );
                }
                Err(e) => {
                    let _ = self.storage.delete(&cloud::config_key(name)).await;
                    rollback_local().await;
                    return Err(format!("bucket manifest init failed: {e}").into());
                }
            }
        }

        tracing::info!("Created collection '{}'", name);
        Ok(collection)
    }

    pub async fn list_collections(&self) -> Vec<Collection> {
        let mut out: Vec<Collection> = {
            let collections = self.collections.read().await;
            collections.values().map(|c| c.metadata.clone()).collect()
        };
        if self.lazy_attach {
            let attached: std::collections::HashSet<String> =
                out.iter().map(|c| c.name.clone()).collect();
            let names: Vec<String> = {
                let reg = self.registered.read().await;
                reg.iter()
                    .filter(|n| !attached.contains(*n))
                    .cloned()
                    .collect()
            };
            for name in names {
                if let Ok(Some(cfg)) = cloud::read_bucket_config(self.storage.as_ref(), &name).await
                {
                    out.push(Collection {
                        name: cfg.name,
                        created_at: cfg.created_at,
                        vector_spaces: cfg.vector_spaces,
                        default_vector_space: cfg.default_vector_space,
                        embedding_dims: cfg.embedding_dims,
                        // Live counts are known only once attached.
                        chunk_count: 0,
                        next_id: 0,
                        config: cfg.config,
                        applied_seq: 0,
                    });
                }
            }
        }
        out
    }

    pub async fn get_collection(&self, name: &str) -> Option<Collection> {
        // Lazy mode: a registered-but-unattached collection attaches on its
        // first request — including a metadata read.
        let _ = self.ensure_attached(name).await;
        let collections = self.collections.read().await;
        collections.get(name).map(|c| c.metadata.clone())
    }

    pub async fn delete_collection(
        &self,
        name: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Lazy mode: the collection may be registered-but-unattached (or LRU
        // evicted) — deleting it must still purge the bucket.
        let attached = {
            let mut collections = self.collections.write().await;
            collections.remove(name).is_some()
        }; // write lock released BEFORE any filesystem/S3 work.
        let registered = self.registered.write().await.remove(name);
        if !attached && !registered {
            // Not known locally; in cloud mode it may still exist in the bucket
            // (created by another node).
            let in_bucket = self.cloud_mode
                && cloud::read_bucket_config(self.storage.as_ref(), name)
                    .await
                    .ok()
                    .flatten()
                    .is_some();
            if !in_bucket {
                return Err(format!("Collection '{}' not found", name).into());
            }
        }
        if attached {
            store::delete_collection_data(&self.data_dir, name)?;
        }
        // Purge every node-local cache tied to the namespace so a later
        // recreate can't consume stale pooled ids or stale configs.
        self.bucket_configs.write().await.remove(name);
        self.writer_pools.lock().await.remove(name);
        self.attach_locks.lock().await.remove(name);
        // Cloud mode: also purge the collection's objects from storage, so it
        // can't be resurrected from S3 on a later cold start (and so a racing
        // ingest's orphan fragment doesn't bring a "deleted" collection back).
        if self.cloud_mode {
            crate::storage::lsm::delete_namespace(self.storage.as_ref(), name)
                .await
                .map_err(|e| format!("failed to purge '{name}' from object storage: {e}"))?;
        }
        tracing::info!("Deleted collection '{}'", name);
        Ok(())
    }

    // ── Vector Space CRUD ────────────────────────────────────────────────

    /// Add a new vector space to a collection.
    pub async fn add_vector_space(
        &self,
        collection_name: &str,
        space_name: &str,
        dims: usize,
        model: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Validate `space_name` before it touches the filesystem. The name is
        // interpolated into on-disk paths (`{space_name}.bin`, `.index`,
        // `.keymap`), so an unconstrained value like `../../tmp/pwn` could
        // write or later delete arbitrary files inside the container. Same
        // character set as collection names.
        validate_name_segment(space_name, "Vector space")?;

        if self.role == NodeRole::Writer {
            return Err(
                "this node runs in writer role; manage vector spaces via a serving node".into(),
            );
        }
        self.ensure_attached(collection_name).await?;
        // Phase 1 (short read lock): preconditions only.
        {
            let collections = self.collections.read().await;
            let loaded = collections
                .get(collection_name)
                .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
            if loaded.metadata.vector_spaces.contains_key(space_name) {
                return Err(format!("Vector space '{}' already exists", space_name).into());
            }
        }

        // Phase 2 (NO lock): bucket-first CAS — the bucket config is the source
        // of truth in cloud mode; the mutate closure revalidates against the
        // LATEST doc so a racing add loses cleanly.
        if self.cloud_mode {
            cloud::cas_update_bucket_config(self.storage.as_ref(), collection_name, |cfg| {
                if cfg.vector_spaces.contains_key(space_name) {
                    return Err(format!("Vector space '{}' already exists", space_name).into());
                }
                cfg.vector_spaces.insert(
                    space_name.to_string(),
                    VectorSpaceConfig {
                        dims,
                        model: model.to_string(),
                        status: "building".to_string(),
                    },
                );
                Ok(())
            })
            .await?;
        }

        // Phase 3 (write lock): apply locally.
        let mut collections = self.collections.write().await;
        let loaded = collections
            .get_mut(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
        if !loaded.metadata.vector_spaces.contains_key(space_name) {
            loaded.metadata.vector_spaces.insert(
                space_name.to_string(),
                VectorSpaceConfig {
                    dims,
                    model: model.to_string(),
                    status: "building".to_string(),
                },
            );
            loaded.vector_spaces.insert(
                space_name.to_string(),
                Arc::new(VectorState {
                    index: None,
                    key_to_chunk_id: Vec::new(),
                    mmap_vectors: None,
                    vectors: Vec::new(),
                    dims,
                }),
            );
            store::save_metadata(&self.data_dir, &loaded.metadata)?;
        }
        Ok(())
    }

    /// Delete a vector space from a collection.
    pub async fn delete_vector_space(
        &self,
        collection_name: &str,
        space_name: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Same path-traversal guard as add_vector_space — the name flows into
        // `remove_file` calls below.
        validate_name_segment(space_name, "Vector space")?;

        if self.role == NodeRole::Writer {
            return Err(
                "this node runs in writer role; manage vector spaces via a serving node".into(),
            );
        }
        self.ensure_attached(collection_name).await?;
        // Phase 1 (short read lock): preconditions.
        {
            let collections = self.collections.read().await;
            let loaded = collections
                .get(collection_name)
                .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
            if loaded.metadata.default_vector_space.as_deref() == Some(space_name) {
                return Err("Cannot delete the default vector space. Switch default first.".into());
            }
        }

        // Phase 2 (NO lock): bucket-first CAS, revalidating against the latest doc.
        if self.cloud_mode {
            cloud::cas_update_bucket_config(self.storage.as_ref(), collection_name, |cfg| {
                if cfg.default_vector_space.as_deref() == Some(space_name) {
                    return Err(
                        "Cannot delete the default vector space. Switch default first.".into(),
                    );
                }
                cfg.vector_spaces.remove(space_name);
                Ok(())
            })
            .await?;
        }

        // Phase 3 (write lock): apply locally.
        let mut collections = self.collections.write().await;
        let loaded = collections
            .get_mut(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
        loaded.metadata.vector_spaces.remove(space_name);
        loaded.vector_spaces.remove(space_name);

        // Clean up disk files
        let vectors_dir = store::vectors_dir(&self.data_dir, collection_name);
        let _ = std::fs::remove_file(vectors_dir.join(format!("{}.index", space_name)));
        let _ = std::fs::remove_file(vectors_dir.join(format!("{}.bin", space_name)));
        let _ = std::fs::remove_file(vectors_dir.join(format!("{}.keymap", space_name)));

        store::save_metadata(&self.data_dir, &loaded.metadata)?;
        Ok(())
    }

    /// Switch the default vector space for a collection.
    pub async fn set_default_vector_space(
        &self,
        collection_name: &str,
        space_name: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.role == NodeRole::Writer {
            return Err(
                "this node runs in writer role; manage vector spaces via a serving node".into(),
            );
        }
        self.ensure_attached(collection_name).await?;
        // Phase 1 (short read lock): preconditions.
        {
            let collections = self.collections.read().await;
            let loaded = collections
                .get(collection_name)
                .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
            if !loaded.metadata.vector_spaces.contains_key(space_name) {
                return Err(format!("Vector space '{}' not found", space_name).into());
            }
        }

        // Phase 2 (NO lock): bucket-first CAS.
        if self.cloud_mode {
            cloud::cas_update_bucket_config(self.storage.as_ref(), collection_name, |cfg| {
                if !cfg.vector_spaces.contains_key(space_name) {
                    return Err(format!("Vector space '{}' not found", space_name).into());
                }
                cfg.default_vector_space = Some(space_name.to_string());
                Ok(())
            })
            .await?;
        }

        // Phase 3 (write lock): apply locally.
        let mut collections = self.collections.write().await;
        let loaded = collections
            .get_mut(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
        loaded.metadata.default_vector_space = Some(space_name.to_string());
        store::save_metadata(&self.data_dir, &loaded.metadata)?;
        Ok(())
    }

    /// Mark a vector space as active (called when rebuild completes).
    #[allow(dead_code)]
    pub async fn mark_vector_space_active(
        &self,
        collection_name: &str,
        space_name: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.role == NodeRole::Writer {
            return Err(
                "this node runs in writer role; manage vector spaces via a serving node".into(),
            );
        }
        self.ensure_attached(collection_name).await?;
        // Bucket-first status flip (NO lock during the CAS).
        if self.cloud_mode {
            cloud::cas_update_bucket_config(self.storage.as_ref(), collection_name, |cfg| {
                if let Some(space) = cfg.vector_spaces.get_mut(space_name) {
                    space.status = "active".to_string();
                }
                Ok(())
            })
            .await?;
        }

        let mut collections = self.collections.write().await;
        let loaded = collections
            .get_mut(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;

        if let Some(config) = loaded.metadata.vector_spaces.get_mut(space_name) {
            config.status = "active".to_string();
        }

        // Reload the vector index from disk
        let vectors_dir = store::vectors_dir(&self.data_dir, collection_name);
        let index_path = vectors_dir.join(format!("{}.index", space_name));
        let vecs_path = vectors_dir.join(format!("{}.bin", space_name));
        let dims = loaded
            .metadata
            .vector_spaces
            .get(space_name)
            .map(|c| c.dims)
            .unwrap_or(384);

        if vecs_path.exists() {
            let vs = vector::load_vector_index(&index_path, &vecs_path, dims)?;
            loaded
                .vector_spaces
                .insert(space_name.to_string(), Arc::new(vs));
        }

        store::save_metadata(&self.data_dir, &loaded.metadata)?;
        Ok(())
    }

    /// Get the data dir for rebuild jobs.
    pub fn vectors_dir(&self, collection_name: &str) -> PathBuf {
        store::vectors_dir(&self.data_dir, collection_name)
    }

    // ── Ingest ───────────────────────────────────────────────────────────

    /// Ingest chunks with batch parent resolution, named embeddings, and relationships.
    /// Claim an id block, migrating a pre-v0.4 namespace on first use: if the
    /// allocator object is absent, seed it from the bucket-derived high-water
    /// mark (create-only, race-safe — no new ids can be minted while the
    /// allocator is absent because every cloud ingest path requires it).
    async fn claim_ids_or_migrate(
        &self,
        ns: &str,
        count: u64,
    ) -> Result<std::ops::Range<u64>, Box<dyn std::error::Error + Send + Sync>> {
        use crate::storage::id_alloc;
        match id_alloc::claim(self.storage.as_ref(), ns, count).await {
            Ok(r) => Ok(r),
            Err(crate::storage::StorageError::NotFound(_)) => {
                let (manifest, _) =
                    crate::storage::lsm::read_manifest(self.storage.as_ref(), ns).await?;
                let mat = cloud::materialize(self.storage.as_ref(), ns, &manifest).await?;
                let start = if mat.max_id > 0 || !mat.chunks.is_empty() {
                    mat.max_id + 1
                } else {
                    0
                };
                id_alloc::seed(self.storage.as_ref(), ns, start).await?;
                Ok(id_alloc::claim(self.storage.as_ref(), ns, count).await?)
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Take `count` ids for an ATTACHED collection from its pooled blocks,
    /// refilling via CAS with the collections lock RELEASED (never hold the
    /// global lock across an S3 round-trip). Racing refills both push their
    /// ranges — nothing leaks, no extra mutex.
    async fn take_ids_cloud(
        &self,
        collection_name: &str,
        count: usize,
    ) -> Result<Vec<u64>, Box<dyn std::error::Error + Send + Sync>> {
        loop {
            {
                let mut collections = self.collections.write().await;
                let loaded = collections
                    .get_mut(collection_name)
                    .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
                let available: u64 = loaded.id_pool.iter().map(|r| r.end - r.start).sum();
                if available >= count as u64 {
                    let mut ids = Vec::with_capacity(count);
                    while ids.len() < count {
                        let front = loaded
                            .id_pool
                            .front_mut()
                            .expect("available >= count guarantees a range");
                        ids.push(front.start);
                        front.start += 1;
                        if front.start == front.end {
                            loaded.id_pool.pop_front();
                        }
                    }
                    return Ok(ids);
                }
            } // lock released before the S3 round-trip below.
            let range = self
                .claim_ids_or_migrate(collection_name, count as u64)
                .await?;
            let mut collections = self.collections.write().await;
            match collections.get_mut(collection_name) {
                Some(loaded) => loaded.id_pool.push_back(range),
                // Collection deleted mid-claim: the block leaks (gaps are fine).
                None => return Err(format!("Collection '{}' not found", collection_name).into()),
            }
        }
    }

    /// Bucket collection config, cached. `refresh` forces a re-fetch (used
    /// once on validation failure, so a just-added vector space is seen
    /// without restarting the writer).
    async fn bucket_config(
        &self,
        ns: &str,
        refresh: bool,
    ) -> Result<cloud::BucketConfig, Box<dyn std::error::Error + Send + Sync>> {
        if !refresh {
            if let Some(cfg) = self.bucket_configs.read().await.get(ns) {
                return Ok(cfg.clone());
            }
        }
        let cfg = cloud::read_bucket_config(self.storage.as_ref(), ns)
            .await?
            .ok_or_else(|| format!("Collection '{}' not found in object storage", ns))?;
        self.bucket_configs
            .write()
            .await
            .insert(ns.to_string(), cfg.clone());
        Ok(cfg)
    }

    /// Writer-role ingest: validate against the bucket config, claim ids from
    /// the shared allocator, append ONE durable WAL fragment, return. No
    /// collections lock, no local indexes — the batch becomes searchable on
    /// serving nodes after their manifest refresh (or attach).
    async fn ingest_stateless(
        &self,
        collection_name: &str,
        ingest_chunks: Vec<IngestChunk>,
        embed_state: &EmbedState,
    ) -> Result<(usize, HashMap<String, u64>, Option<u64>), Box<dyn std::error::Error + Send + Sync>>
    {
        validate_name_segment(collection_name, "Collection")?;
        let count = ingest_chunks.len();
        if count == 0 {
            return Ok((0, HashMap::new(), None));
        }
        let cfg = self.bucket_config(collection_name, false).await?;

        // Ids from the writer-side pool (same allocator as attached nodes).
        // The pool mutex is NEVER held across the S3 claim: drain what's
        // available, release, claim, push, repeat. Ids already drained are
        // kept across iterations (a failed later claim leaks them — fine).
        let mut ids: Vec<u64> = Vec::with_capacity(count);
        loop {
            {
                let mut pools = self.writer_pools.lock().await;
                let pool = pools.entry(collection_name.to_string()).or_default();
                while ids.len() < count {
                    let Some(front) = pool.front_mut() else { break };
                    if front.start < front.end {
                        ids.push(front.start);
                        front.start += 1;
                    }
                    if front.start >= front.end {
                        pool.pop_front();
                    }
                }
                if ids.len() == count {
                    break;
                }
            } // pool mutex released before the S3 round-trip.
            let need = (count - ids.len()) as u64;
            let range = self.claim_ids_or_migrate(collection_name, need).await?;
            let mut pools = self.writer_pools.lock().await;
            pools
                .entry(collection_name.to_string())
                .or_default()
                .push_back(range);
        }

        // Build chunks with the same embedding rules as the attached path,
        // validating dims against the bucket config. On a validation failure,
        // refresh the config once (a space may have just been added) before
        // rejecting — a stale cache must never poison a durable fragment.
        let build = |cfg: &cloud::BucketConfig| -> Result<
            (Vec<DocumentChunk>, HashMap<String, u64>),
            Box<dyn std::error::Error + Send + Sync>,
        > {
            let default_space = cfg
                .default_vector_space
                .clone()
                .unwrap_or_else(|| "default".into());
            let mut client_id_map: HashMap<String, u64> = HashMap::new();
            for (ic, &id) in ingest_chunks.iter().zip(ids.iter()) {
                if let Some(ref cid) = ic.client_id {
                    client_id_map.insert(cid.clone(), id);
                }
            }
            let parent_ids: Vec<Option<u64>> =
                ingest_chunks.iter().map(|ic| ic.parent_id).collect();
            let parent_refs: Vec<Option<String>> = ingest_chunks
                .iter()
                .map(|ic| ic.parent_ref.clone())
                .collect();
            let group_ids: Vec<Option<String>> =
                ingest_chunks.iter().map(|ic| ic.group_id.clone()).collect();
            let resolved = RelationshipStore::resolve_batch_refs(
                &client_id_map,
                &parent_ids,
                &parent_refs,
                &group_ids,
            );

            let mut chunks: Vec<DocumentChunk> = Vec::with_capacity(count);
            for (i, ic) in ingest_chunks.iter().enumerate() {
                let id = ids[i];
                let (parent_id, group_id) = resolved[i].clone();
                let mut embeddings = ic.embeddings.clone();
                if let Some(emb) = ic.embedding.clone() {
                    embeddings.entry(default_space.clone()).or_insert(emb);
                }
                if embeddings.is_empty() {
                    if let Ok(emb) = embed_state.embed_query(&ic.text) {
                        let expected = cfg
                            .vector_spaces
                            .get(&default_space)
                            .map(|c| c.dims)
                            .unwrap_or(cfg.embedding_dims);
                        if emb.len() == expected {
                            embeddings.insert(default_space.clone(), emb);
                        } else {
                            tracing::warn!(
                                "writer ingest: built-in embedder produces {} dims but space \
                                 '{}' expects {} — chunk {} will be FTS-only",
                                emb.len(),
                                default_space,
                                expected,
                                i
                            );
                        }
                    }
                }
                for (space_name, vec) in &embeddings {
                    let expected = cfg
                        .vector_spaces
                        .get(space_name)
                        .map(|c| c.dims)
                        .unwrap_or(cfg.embedding_dims);
                    if vec.len() != expected {
                        return Err(format!(
                            "chunk {i}: embedding for vector space '{space_name}' has {} dims, \
                             expected {expected}",
                            vec.len()
                        )
                        .into());
                    }
                }
                chunks.push(DocumentChunk {
                    id,
                    collection: collection_name.to_string(),
                    file_id: ic.file_id.clone(),
                    chunk_index: ic.chunk_index,
                    page: ic.page,
                    text: ic.text.clone(),
                    metadata: ic.metadata.clone(),
                    doc_type: ic.doc_type.clone(),
                    parent_id,
                    group_id,
                    embeddings,
                    embedding: None,
                });
            }
            Ok((chunks, client_id_map))
        };
        let (chunks, client_id_map) = match build(&cfg) {
            Ok(out) => out,
            Err(first_err) => {
                let fresh = self.bucket_config(collection_name, true).await?;
                build(&fresh).map_err(|_| first_err)?
            }
        };

        // ONE durable append; searchable on serving nodes after refresh.
        let payload = serde_json::to_vec(&chunks)?;
        let records = chunks.len() as u64;
        let seq = crate::storage::lsm::append_fragment(
            self.storage.as_ref(),
            collection_name,
            bytes::Bytes::from(payload),
            records,
        )
        .await
        .map_err(|e| format!("cloud WAL append failed: {e}"))?;
        maybe_auto_compact(
            self.storage.clone(),
            collection_name.to_string(),
            self.compacting.clone(),
        );
        tracing::info!(
            "Writer ingest: WAL fragment seq={} ({} chunks) durable for '{}'",
            seq,
            records,
            collection_name
        );
        Ok((count, client_id_map, Some(seq)))
    }

    pub async fn ingest(
        &self,
        collection_name: &str,
        ingest_chunks: Vec<IngestChunk>,
        embed_state: &EmbedState,
    ) -> Result<(usize, HashMap<String, u64>, Option<u64>), Box<dyn std::error::Error + Send + Sync>>
    {
        crate::metrics::inc(&crate::metrics::INGEST_REQUESTS_TOTAL);
        crate::metrics::add(
            &crate::metrics::INGEST_CHUNKS_TOTAL,
            ingest_chunks.len() as u64,
        );
        // Writer role: durable-append-only ingest, no local state required.
        if self.role == NodeRole::Writer {
            return self
                .ingest_stateless(collection_name, ingest_chunks, embed_state)
                .await;
        }

        let count = ingest_chunks.len();
        self.ensure_attached(collection_name).await?;

        // Cloud mode: ids come from CAS-leased blocks (storage/id_alloc.rs) so
        // they can NEVER collide with a stateless writer's ids. This happens
        // BEFORE taking the write lock (its refill path does S3 round-trips).
        // A failed ingest after this point leaks the taken ids — gaps are fine;
        // the invariant is no-reuse, not density.
        let cloud_ids: Option<Vec<u64>> = if self.cloud_mode && count > 0 {
            Some(self.take_ids_cloud(collection_name, count).await?)
        } else {
            None
        };

        let mut collections = self.collections.write().await;
        let loaded = collections
            .get_mut(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;

        // Phase 1: Assign IDs and build client_id -> chunk_id map
        let mut client_id_map: HashMap<String, u64> = HashMap::new();
        let assigned_ids: Vec<u64> = match cloud_ids {
            Some(ids) => {
                // Keep the local counter as a diagnostic high-water mark only.
                if let Some(&max) = ids.iter().max() {
                    loaded.next_id = loaded.next_id.max(max + 1);
                }
                ids
            }
            None => {
                let mut ids = Vec::with_capacity(count);
                for _ in 0..count {
                    ids.push(loaded.next_id);
                    loaded.next_id += 1;
                }
                ids
            }
        };
        for (ic, &id) in ingest_chunks.iter().zip(assigned_ids.iter()) {
            if let Some(ref cid) = ic.client_id {
                client_id_map.insert(cid.clone(), id);
            }
        }

        // Phase 2: Resolve batch parent references
        let parent_ids: Vec<Option<u64>> = ingest_chunks.iter().map(|ic| ic.parent_id).collect();
        let parent_refs: Vec<Option<String>> = ingest_chunks
            .iter()
            .map(|ic| ic.parent_ref.clone())
            .collect();
        let group_ids: Vec<Option<String>> =
            ingest_chunks.iter().map(|ic| ic.group_id.clone()).collect();
        let resolved = RelationshipStore::resolve_batch_refs(
            &client_id_map,
            &parent_ids,
            &parent_refs,
            &group_ids,
        );

        // Phase 3: Build DocumentChunks and collect embeddings per vector space
        let default_space = loaded
            .metadata
            .default_vector_space
            .clone()
            .unwrap_or_else(|| "default".into());
        let mut chunks: Vec<DocumentChunk> = Vec::with_capacity(count);
        // space_name -> Vec<(chunk_id, embedding)>
        let mut space_vectors: HashMap<String, Vec<(u64, Vec<f32>)>> = HashMap::new();
        // Deferred relationship additions (applied after the S3 append, so we can
        // release the collections lock during network I/O).
        let mut rel_adds: Vec<(u64, Option<u64>, Option<String>)> = Vec::with_capacity(count);

        for (i, ic) in ingest_chunks.into_iter().enumerate() {
            let id = assigned_ids[i];
            let (parent_id, group_id) = resolved[i].clone();

            // Collect named embeddings
            let mut embeddings = ic.embeddings;
            // Legacy: single embedding -> map to default space
            if let Some(emb) = ic.embedding {
                if !embeddings.contains_key(&default_space) {
                    embeddings.insert(default_space.clone(), emb);
                }
            }
            // If no embeddings provided at all, compute using the built-in
            // embedder — but ONLY if its output matches the default space's
            // dims (a 384-dim BGE vector in a 4-dim space would corrupt the
            // vector file). Chunks without a usable embedding stay FTS-only.
            if embeddings.is_empty() {
                if let Ok(emb) = embed_state.embed_query(&ic.text) {
                    let expected = loaded
                        .metadata
                        .vector_spaces
                        .get(&default_space)
                        .map(|c| c.dims)
                        .unwrap_or(loaded.metadata.embedding_dims);
                    if emb.len() == expected {
                        embeddings.insert(default_space.clone(), emb);
                    }
                }
            }

            // Validate USER-provided embedding lengths against each space's
            // configured dims BEFORE anything is written. One wrong-length
            // vector would silently corrupt the mmap vector file in release
            // builds (offsets shift for every vector after it).
            for (space_name, vec) in &embeddings {
                let expected = loaded
                    .metadata
                    .vector_spaces
                    .get(space_name)
                    .map(|c| c.dims)
                    .unwrap_or(loaded.metadata.embedding_dims);
                if vec.len() != expected {
                    return Err(format!(
                        "chunk {i}: embedding for vector space '{space_name}' has {} dims, \
                         expected {expected}",
                        vec.len()
                    )
                    .into());
                }
            }

            // Store embeddings by vector space for batch index building
            for (space_name, vec) in &embeddings {
                space_vectors
                    .entry(space_name.clone())
                    .or_default()
                    .push((id, vec.clone()));
            }

            let chunk = DocumentChunk {
                id,
                collection: collection_name.to_string(),
                file_id: ic.file_id,
                chunk_index: ic.chunk_index,
                page: ic.page,
                text: ic.text,
                metadata: ic.metadata,
                doc_type: ic.doc_type,
                parent_id,
                group_id: group_id.clone(),
                embeddings,
                embedding: None, // v2 uses named embeddings
            };

            // Defer applying relationships / chunk map to `loaded` until AFTER
            // the S3 append (so we can drop the lock during network I/O, #2).
            rel_adds.push((id, parent_id, group_id));
            chunks.push(chunk);
        }

        // Release the collections write lock BEFORE the S3 network round-trip, so
        // a slow S3 call doesn't stall every other collection (#2). Nothing local
        // has been mutated yet (chunks/relationships were deferred into local Vecs
        // above), so there is no state to roll back if the append fails.
        drop(collections);

        // Phase 3a (cloud): DURABLE S3 WAL append FIRST, before any local commit
        // (fixes F14 split-brain — a failed append leaves nothing local, clean retry).
        let mut appended_seq: Option<u64> = None;
        if self.cloud_mode {
            let payload = serde_json::to_vec(&chunks)?;
            let records = chunks.len() as u64;
            let seq = crate::storage::lsm::append_fragment(
                self.storage.as_ref(),
                collection_name,
                bytes::Bytes::from(payload),
                records,
            )
            .await
            .map_err(|e| format!("cloud WAL append failed, ingest not applied: {e}"))?;
            appended_seq = Some(seq);
            tracing::info!(
                "Cloud ingest: WAL fragment seq={} ({} chunks) durable for '{}'",
                seq,
                records,
                collection_name
            );
            maybe_auto_compact(
                self.storage.clone(),
                collection_name.to_string(),
                self.compacting.clone(),
            );
        }

        // Re-acquire the write lock and apply local state (durable S3 record, if
        // any, already written). Ids were pre-assigned from a monotonic counter,
        // so no concurrent ingest can collide; applying by id is order-independent.
        let mut collections = self.collections.write().await;
        let loaded = match collections.get_mut(collection_name) {
            Some(l) => {
                l.last_used
                    .store(next_lru_tick(), std::sync::atomic::Ordering::Relaxed);
                l
            }
            None => {
                // Missing from the map: either DELETED or merely LRU-EVICTED in
                // the lock gap. If the bucket still has the collection, the
                // append is healthy and durable — do NOT erase it; the next
                // attach/refresh applies it.
                if self.cloud_mode
                    && cloud::read_bucket_config(self.storage.as_ref(), collection_name)
                        .await
                        .ok()
                        .flatten()
                        .is_some()
                {
                    return Ok((count, client_id_map, appended_seq));
                }
                // Genuinely deleted: `delete_collection` purges S3, but our
                // fragment may have landed after that purge — append a
                // tombstone so a re-materialize (which would recreate a
                // manifest referencing only our orphan fragment) yields nothing.
                if self.cloud_mode {
                    if let Err(te) = crate::storage::lsm::append_tombstone(
                        self.storage.as_ref(),
                        collection_name,
                        &assigned_ids,
                    )
                    .await
                    {
                        tracing::error!(
                            "compensation (deleted-in-gap): S3 tombstone for '{}' failed: {}",
                            collection_name,
                            te
                        );
                    }
                }
                return Err(format!("Collection '{}' not found", collection_name).into());
            }
        };
        // Double-apply guard: between our S3 append and this reacquire, the
        // manifest refresher may have polled and applied OUR fragment. The
        // tracker is the single source of truth for "already reflected
        // locally" — skip the local apply if it covers our seq.
        if let Some(seq) = appended_seq {
            if loaded.applied.covers(seq) {
                tracing::debug!(
                    "ingest seq={} for '{}' already applied by refresher; skipping local apply",
                    seq,
                    collection_name
                );
                return Ok((count, client_id_map, appended_seq));
            }
        }

        // Apply all local state (chunks map, redb, FTS, HNSW, metadata, filter
        // index) in one fallible step. On ANY failure in cloud mode we've already
        // written a durable S3 fragment for these ids, so we compensate with a
        // tombstone (below) — otherwise a partial local commit + orphan S3
        // fragment would resurrect/duplicate the batch on a cold restart (F2).
        let commit_result = Self::apply_ingest_commit(
            &self.data_dir,
            collection_name,
            loaded,
            rel_adds,
            &chunks,
            space_vectors,
            count,
        );
        if commit_result.is_ok() {
            if let Some(seq) = appended_seq {
                loaded.applied.mark(seq);
                loaded.metadata.applied_seq = loaded.applied.contiguous;
            }
        }
        if let Err(e) = commit_result {
            if self.cloud_mode {
                // Compensate for the durable S3 fragment whose local commit
                // failed. Tombstone the ids in THREE places so they can never
                // resurface, on any restart path:
                //   1. local redb tombstones table — survives `load_collection`
                //      rehydrating from redb on a persistent-disk node (the
                //      normal deployment). Without this, the chunk sits in redb
                //      (insert_batch may have succeeded before FTS/HNSW failed)
                //      and would be pulled back into RAM on restart.
                //   2. the in-RAM tombstone set — masks it at query time now.
                //   3. an S3 tombstone — so a fresh-disk rebuild-from-manifest
                //      also drops it.
                if let Err(te) = loaded.chunk_store.tombstone_batch(&assigned_ids) {
                    tracing::error!(
                        "compensation: failed to write local tombstones for '{}': {} \
                         (chunk may need manual delete)",
                        collection_name,
                        te
                    );
                }
                for id in &assigned_ids {
                    loaded.tombstones.insert(*id);
                    if let Ok(Some(c)) = loaded.chunk_store.get(*id) {
                        loaded.filter_index.remove(*id, &filter_meta(&c));
                    }
                }
                drop(collections);
                if let Err(te) = crate::storage::lsm::append_tombstone(
                    self.storage.as_ref(),
                    collection_name,
                    &assigned_ids,
                )
                .await
                {
                    // Flaky S3: the orphan fragment now has no S3 tombstone. Local
                    // tombstones (redb + RAM) still mask it on THIS node; surface
                    // loudly so a fresh-disk rebuild risk is visible to operators.
                    tracing::error!(
                        "compensation: durable S3 tombstone for '{}' FAILED: {} — \
                         orphan fragment may resurrect on a fresh-disk rebuild; \
                         run POST /collections/{}/compact once S3 is healthy",
                        collection_name,
                        te,
                        collection_name
                    );
                }
            }
            return Err(e);
        }

        tracing::info!("Ingested {} chunks into '{}'", count, collection_name);

        Ok((count, client_id_map, appended_seq))
    }

    /// Apply an ingest batch's local state (chunk map, redb, FTS, HNSW, metadata,
    /// relationships, filter index). All-or-caller-compensates: any `?` failure
    /// leaves partial local state, which the caller undoes + tombstones in cloud
    /// mode. Synchronous (no `.await`) — the S3 write already happened.
    #[allow(clippy::too_many_arguments)]
    fn apply_ingest_commit(
        data_dir: &Path,
        collection_name: &str,
        loaded: &mut LoadedCollection,
        rel_adds: Vec<(u64, Option<u64>, Option<String>)>,
        chunks: &[DocumentChunk],
        space_vectors: HashMap<String, Vec<(u64, Vec<f32>)>>,
        count: usize,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        for (id, parent_id, group_id) in rel_adds {
            loaded.relationships.add(id, parent_id, group_id);
        }

        // Phase 3b: Persist chunks to the disk-backed store BEFORE updating
        // FTS/HNSW. If this write fails we error out before any index commits,
        // so we never end up with a Tantivy or HNSW index referencing chunks
        // that don't exist on disk. redb writes are atomic per batch.
        let to_persist: Vec<(u64, DocumentChunk)> =
            chunks.iter().map(|c| (c.id, c.clone())).collect();
        loaded.chunk_store.insert_batch(&to_persist)?;

        // Phase 4: Update Tantivy FTS index
        let tantivy_dir = store::tantivy_dir(data_dir, collection_name);
        loaded.fts = tantivy_fts::build_index(&tantivy_dir, chunks, loaded.metadata.chunk_count)?;

        // Phase 5: Update each vector space's HNSW index
        let vectors_dir = store::vectors_dir(data_dir, collection_name);
        for (space_name, new_vecs) in space_vectors {
            // Same fallback the ingest-time validation uses, so a space unknown
            // to metadata can't validate against one dims and build with another.
            let dims = loaded
                .metadata
                .vector_spaces
                .get(&space_name)
                .map(|c| c.dims)
                .unwrap_or(loaded.metadata.embedding_dims);

            let index_path = vectors_dir.join(format!("{}.index", space_name));
            let vecs_path = vectors_dir.join(format!("{}.bin", space_name));

            // Check if we can do incremental add (existing index + mmap vectors)
            let existing = loaded.vector_spaces.get(&space_name);
            let can_incremental = existing.map(|e| e.mmap_vectors.is_some()).unwrap_or(false);

            if can_incremental {
                // Incremental path: append to mmap file, add to HNSW, save
                let arc = loaded.vector_spaces.remove(&space_name).unwrap();
                let Ok(mut vs) = Arc::try_unwrap(arc) else {
                    // Another thread holds a reference — fall back to full rebuild
                    let existing = loaded.vector_spaces.get(&space_name);
                    let mut all_ids: Vec<u64> = existing
                        .map(|e| e.key_to_chunk_id.clone())
                        .unwrap_or_default();
                    let mut all_vecs: Vec<Vec<f32>> = existing
                        .and_then(|e| e.mmap_vectors.as_ref())
                        .map(|m| m.to_vecs())
                        .unwrap_or_default();
                    for (cid, vec) in new_vecs {
                        all_ids.push(cid);
                        all_vecs.push(vec);
                    }
                    let vs = vector::build_vector_index(
                        &index_path,
                        &vecs_path,
                        &all_ids,
                        &all_vecs,
                        dims,
                    )?;
                    loaded.vector_spaces.insert(space_name, Arc::new(vs));
                    continue;
                };

                // Save-batching state lives on the collection (the closure
                // owns only the unwrapped VectorState).
                let prev = loaded
                    .hnsw_unsaved
                    .get(&space_name)
                    .copied()
                    .unwrap_or(u32::MAX);
                let mut mutable_flag = prev != u32::MAX;
                let mut unsaved_ctr = if mutable_flag { prev } else { 0 };
                let unsaved = &mut unsaved_ctr;
                let mutable_now = &mut mutable_flag;
                // Run the fallible updates in a closure so the space is ALWAYS
                // re-inserted into `vector_spaces` afterward — an early `?` here
                // used to drop the unwrapped space entirely, silently disabling
                // semantic search on it until restart.
                let result = (|| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    // Append new vectors to mmap file
                    if let Some(ref mut mmap) = vs.mmap_vectors {
                        mmap.append(&new_vecs)?;
                    }

                    // Extend the key mapping and persist it IMMEDIATELY after
                    // the mmap append, before any HNSW work — so the two files
                    // never desync on disk (a stale keymap makes search fabricate
                    // chunk ids from raw key indexes after restart).
                    let base_key = vs.key_to_chunk_id.len();
                    for (cid, _) in &new_vecs {
                        vs.key_to_chunk_id.push(*cid);
                    }
                    let map_path = index_path.with_extension("keymap");
                    vector::save_key_map(&map_path, &vs.key_to_chunk_id)?;

                    // Add to HNSW index. The in-RAM index stays mutable across
                    // batches (first mutation loads from disk once); the FILE is
                    // rewritten only every HNSW_SAVE_EVERY batches — per-batch
                    // saves were O(index size), a scale wall. A crash between
                    // saves leaves a stale file, detected and rebuilt from the
                    // mmap at next load (vectors are already durable there).
                    const HNSW_SAVE_EVERY: u32 = 16;
                    let total = vs.key_to_chunk_id.len();
                    if total >= 1000 {
                        let index_path_str = index_path
                            .to_str()
                            .ok_or("USearch index path is not valid UTF-8")?;
                        let (index, was_fresh) = match (*mutable_now, vs.index.take()) {
                            (true, Some(idx)) => (idx, false),
                            _ => {
                                let idx = vector::create_index(dims, total)?;
                                if index_path.exists() {
                                    idx.load(index_path_str).map_err(|e| {
                                        format!("Failed to load USearch index: {}", e)
                                    })?;
                                }
                                (idx, true)
                            }
                        };
                        let threads = 128.max(rayon::current_num_threads());
                        index
                            .reserve_capacity_and_threads(total, threads)
                            .map_err(|e| format!("Reserve failed: {}", e))?;
                        for (i, (_, vec)) in new_vecs.iter().enumerate() {
                            index
                                .add((base_key + i) as u64, vec)
                                .map_err(|e| format!("Failed to add vector: {}", e))?;
                        }
                        *unsaved += 1;
                        if was_fresh || *unsaved >= HNSW_SAVE_EVERY {
                            index
                                .save(index_path_str)
                                .map_err(|e| format!("Failed to save index: {}", e))?;
                            *unsaved = 0;
                        }
                        vs.index = Some(index);
                        *mutable_now = true;
                    }
                    Ok(())
                })();
                // Space goes back in whatever happened; a partial update is
                // recoverable (caller compensates the batch), a vanished space
                // is a silent outage.
                if mutable_flag {
                    loaded.hnsw_unsaved.insert(space_name.clone(), unsaved_ctr);
                }
                loaded.vector_spaces.insert(space_name, Arc::new(vs));
                result?;
            } else {
                // Full rebuild path (first ingest or legacy data)
                let mut all_ids: Vec<u64> = existing
                    .map(|e| e.key_to_chunk_id.clone())
                    .unwrap_or_default();
                let mut all_vecs: Vec<Vec<f32>> =
                    existing.map(|e| e.vectors.clone()).unwrap_or_default();

                for (cid, vec) in new_vecs {
                    all_ids.push(cid);
                    all_vecs.push(vec);
                }

                let vs =
                    vector::build_vector_index(&index_path, &vecs_path, &all_ids, &all_vecs, dims)?;
                loaded.vector_spaces.insert(space_name, Arc::new(vs));
            }
        }

        // Phase 6: Save metadata + relationships, then rebuild the filter index.
        // Persist the advanced next_id high-water mark so ids are never reused
        // even if the local chunk store is later empty on restart.
        loaded.metadata.chunk_count += count as u64;
        loaded.metadata.next_id = loaded.next_id;
        store::save_metadata(data_dir, &loaded.metadata)?;
        let rel_path = store::collection_dir(data_dir, collection_name).join("relationships.bin");
        loaded.relationships.save(&rel_path)?;
        // Incremental: O(batch), not O(collection) — a full index rebuild here
        // made every ingest/replay cost scale with the whole collection.
        for c in chunks {
            loaded.filter_index.insert(c.id, &filter_meta(c));
        }
        loaded.filter_index.finalize();
        Ok(())
    }

    // ── Search ───────────────────────────────────────────────────────────

    /// Search with full scoring pipeline: retrieve (filter-aware) → score → return.
    ///
    /// The filter is applied INSIDE the HNSW walk via USearch's filter
    /// callback when set. Recall does not collapse on selective filters.
    /// See `docs/v0.4-filter-aware-ann.md`.
    pub async fn search(
        &self,
        collection_name: &str,
        req: &SearchRequest,
        embed_state: &EmbedState,
    ) -> Result<
        (
            Vec<(
                DocumentChunk,
                f32,
                String,
                Option<HashMap<String, MetadataValue>>,
                Option<Vec<ChunkRelation>>,
            )>,
            usize,
            u64,
            Option<ExplainPlan>,
        ),
        Box<dyn std::error::Error + Send + Sync>,
    > {
        if self.role == NodeRole::Writer {
            return Err("this node runs in writer role and does not serve queries".into());
        }
        crate::metrics::inc(&crate::metrics::SEARCH_REQUESTS_TOTAL);
        self.ensure_attached(collection_name).await?;

        // Read-your-writes: wait (bounded) until fragments up to `min_seq` are
        // applied locally, refreshing on demand. A `min_seq` beyond the
        // manifest is rejected rather than waited on forever.
        if let Some(min_seq) = req.min_seq {
            if self.cloud_mode {
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
                loop {
                    let covered = {
                        let collections = self.collections.read().await;
                        let loaded = collections
                            .get(collection_name)
                            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
                        loaded
                            .last_used
                            .store(next_lru_tick(), std::sync::atomic::Ordering::Relaxed);
                        loaded.applied.covers(min_seq)
                    };
                    if covered {
                        break;
                    }
                    let next_seq = self.refresh_collection(collection_name).await?;
                    if min_seq >= next_seq {
                        return Err(format!(
                            "min_seq {} is beyond the collection's write history ({})",
                            min_seq, next_seq
                        )
                        .into());
                    }
                    if std::time::Instant::now() >= deadline {
                        return Err(format!(
                            "timed out waiting for min_seq {min_seq} to be applied"
                        )
                        .into());
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            }
        }
        let start = std::time::Instant::now();
        let collections = self.collections.read().await;
        let loaded = collections
            .get(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
        loaded
            .last_used
            .store(next_lru_tick(), std::sync::atomic::Ordering::Relaxed);

        let mode = SearchMode::from_str_param(&req.mode);
        let rerank_k = req.top_k * 3; // fetch extra candidates for scoring

        // Determine which vector space to use
        let space_name = req
            .vector_space
            .as_deref()
            .or(loaded.metadata.default_vector_space.as_deref())
            .unwrap_or("default");

        // ── Step 0: Compile filter, resolve eligible bitmap ──────────────
        // FilterExpr::compile is cheap. eligible() is a roaring intersection
        // across the predicate bitmaps; sub-millisecond at any realistic size.
        // The bitmap routes through both FTS and semantic retrieval below.
        let filter_expr = FilterExpr::compile(&req.filters);
        let eligible = loaded.filter_index.eligible(&filter_expr);
        let universe_count = loaded.filter_index.len();
        let selectivity_val = selectivity(&eligible, universe_count);
        let filter_active = !filter_expr.is_empty();

        // Engine / candidates-inspected / ef metadata, captured for /explain.
        let mut explain_engine: Option<&'static str> = None;
        let mut explain_candidates: Option<u64> = None;

        // ── Step 1: Retrieve candidates (filter-aware) ───────────────────
        let fts_results = if matches!(mode, SearchMode::Fts | SearchMode::Hybrid) {
            let (raw, _, _) =
                tantivy_fts::search(&loaded.fts, &req.query, &HashMap::new(), rerank_k)?;
            // FTS doesn't yet have predicate pushdown; post-filter results
            // against the same eligible bitmap so the merged top-k respects
            // the filter exactly the same way the semantic path does.
            if filter_active {
                raw.into_iter()
                    .filter(|(id, _)| eligible.contains(*id))
                    .collect()
            } else {
                raw
            }
        } else {
            Vec::new()
        };

        let semantic_results = if matches!(mode, SearchMode::Semantic | SearchMode::Hybrid) {
            if let Some(vs) = loaded.vector_spaces.get(space_name) {
                let query_vec_opt: Option<Vec<f32>> = req
                    .query_vector
                    .clone()
                    .or_else(|| embed_state.embed_query(&req.query).ok());
                if let Some(query_vec) = query_vec_opt {
                    let vs_clone = vs.clone();
                    if filter_active {
                        // Filter-aware path: USearch's filter callback prunes
                        // ineligible nodes during the HNSW walk. No over-fetch,
                        // no post-filter recall collapse.
                        let eligible_clone = eligible.clone();
                        let (vr, explain) = tokio::task::spawn_blocking(move || {
                            vector::search_vectors_filtered(
                                &query_vec,
                                &vs_clone,
                                rerank_k,
                                &eligible_clone,
                            )
                        })
                        .await
                        .unwrap_or_else(|_| (Vec::new(), vector::FilteredSearchExplain::default()));
                        explain_engine = Some(if explain.used_hnsw {
                            "hnsw"
                        } else {
                            "brute_force"
                        });
                        if explain.used_hnsw {
                            explain_candidates = Some(explain.candidates_inspected);
                        }
                        vr.iter().map(|r| (r.chunk_id, r.score)).collect::<Vec<_>>()
                    } else {
                        // No filter: skip the predicate-callback overhead and
                        // use the existing unfiltered HNSW path.
                        let vr = tokio::task::spawn_blocking(move || {
                            vector::search_vectors(&query_vec, &vs_clone, rerank_k)
                        })
                        .await
                        .unwrap_or_default();
                        explain_engine = Some("hnsw");
                        vr.iter().map(|r| (r.chunk_id, r.score)).collect::<Vec<_>>()
                    }
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // ── Step 2: Merge via RRF (for hybrid) or use single-mode results ──
        let mut candidates: Vec<ScoredCandidate> = match mode {
            SearchMode::Hybrid if !fts_results.is_empty() || !semantic_results.is_empty() => {
                let (rrf_k, fts_w, sem_w) = match &req.score_weights {
                    Some(sw) => (
                        sw.rrf_k as f32,
                        sw.fts_weight as f32,
                        sw.semantic_weight as f32,
                    ),
                    None => (60.0, 1.0, 1.0),
                };
                let merged = hybrid::merge_rrf(
                    &fts_results,
                    &semantic_results,
                    rerank_k,
                    rrf_k,
                    fts_w,
                    sem_w,
                );
                merged
                    .iter()
                    .map(|r| ScoredCandidate {
                        chunk_id: r.chunk_id,
                        base_score: r.rrf_score,
                        final_score: r.rrf_score,
                        source: r.source.as_str().to_string(),
                    })
                    .collect()
            }
            SearchMode::Fts => fts_results
                .iter()
                .map(|(id, score)| ScoredCandidate {
                    chunk_id: *id,
                    base_score: *score,
                    final_score: *score,
                    source: "fts".to_string(),
                })
                .collect(),
            SearchMode::Semantic => semantic_results
                .iter()
                .map(|(id, score)| ScoredCandidate {
                    chunk_id: *id,
                    base_score: *score,
                    final_score: *score,
                    source: "semantic".to_string(),
                })
                .collect(),
            _ => Vec::new(),
        };

        // ── Step 3: Filter is already applied (filter-aware retrieval). ─
        // The bitmap pushdown happens inside both FTS post-filter and
        // USearch's filter callback above, so we no longer need a post-merge
        // `retain`. Kept as an assertion in debug builds to catch invariant
        // drift if a new retrieval path bypasses the eligibility check.
        debug_assert!(
            !filter_active || candidates.iter().all(|c| eligible.contains(c.chunk_id)),
            "filter-aware retrieval produced a candidate outside the eligible bitmap"
        );

        // ── Step 3b: Drop soft-deleted (tombstoned) chunks ──────────────
        // The HNSW/FTS indexes still physically contain deleted ids until the
        // next rebuild/compaction, so we filter them out here. Cheap O(1) set
        // membership per candidate.
        if !loaded.tombstones.is_empty() {
            candidates.retain(|c| !loaded.tombstones.contains(&c.chunk_id));
        }

        // ── Step 4: Apply scoring pipeline ──────────────────────────────
        // Resolve recency preset into a full config (explicit `recency` wins)
        let recency_config = req.recency.clone().or_else(|| {
            req.recency_preset.as_deref().and_then(|preset| {
                req.recency_field
                    .as_deref()
                    .map(|field| RecencyConfig::from_preset(preset, field.to_string()))
                    .flatten()
            })
        });

        let has_scoring =
            recency_config.is_some() || !req.boosts.is_empty() || req.relationship_boost.is_some();

        if has_scoring && !candidates.is_empty() {
            let candidate_ids: Vec<u64> = candidates.iter().map(|c| c.chunk_id).collect();
            let chunk_metadata: HashMap<u64, HashMap<String, MetadataValue>> = loaded
                .chunk_store
                .get_batch(&candidate_ids)
                .unwrap_or_default()
                .into_iter()
                .map(|chunk| (chunk.id, chunk.metadata.clone()))
                .collect();

            let candidate_ids: Vec<u64> = candidates.iter().map(|c| c.chunk_id).collect();
            let (parent_ids, sibling_map) = loaded.relationships.build_scoring_maps(&candidate_ids);

            scoring::apply_scoring_pipeline(
                &mut candidates,
                &chunk_metadata,
                &parent_ids,
                &sibling_map,
                &recency_config,
                &req.boosts,
                &req.relationship_boost,
            );
        }

        // ── Step 5: Truncate to top_k and build response ────────────────
        candidates.truncate(req.top_k);
        let total = candidates.len();

        // ── Step 5a: Parent metadata enrichment for segment hits ─────────
        // For each segment hit with a parent_id, inline the parent's top-level
        // metadata so callers (typically AI agents) avoid a second round-trip
        // to fetch source-level attributes. Parents are deduplicated: N
        // segments sharing the same parent_id pay for one HashMap lookup,
        // not N. No additional I/O; the chunk map is already in memory.
        let candidate_chunk_ids: Vec<u64> = candidates.iter().map(|c| c.chunk_id).collect();
        let parent_meta_cache =
            build_parent_metadata_cache(&candidate_chunk_ids, &loaded.chunk_store);

        // ── Step 5b: Relation enrichment (opt-in) ───────────────────────
        // When include_relations is set, fetch each hit's edges in ONE batched,
        // on-demand read from the disk-backed relation store (never resident in
        // RAM). target_status is resolved against the in-memory chunk cache:
        // "found" if the target chunk exists locally, else "missing".
        let mut relations_by_chunk: HashMap<u64, Vec<ChunkRelation>> = HashMap::new();
        if req.include_relations {
            let types = req.relation_types.as_deref();
            relations_by_chunk = loaded.relation_store.for_chunks(
                &candidate_chunk_ids,
                req.relation_direction,
                types,
            )?;
            for edges in relations_by_chunk.values_mut() {
                for edge in edges.iter_mut() {
                    edge.target_status = if loaded.filter_index.contains(edge.target_chunk_id) {
                        "found".to_string()
                    } else {
                        "missing".to_string()
                    };
                }
            }
        }

        let hits: Vec<(
            DocumentChunk,
            f32,
            String,
            Option<HashMap<String, MetadataValue>>,
            Option<Vec<ChunkRelation>>,
        )> = candidates
            .iter()
            .filter_map(|c| {
                loaded
                    .chunk_store
                    .get(c.chunk_id)
                    .ok()
                    .flatten()
                    .map(|chunk| {
                        let parent_metadata = parent_metadata_for(&chunk, &parent_meta_cache);
                        // Some(vec) when requested (possibly empty), None when not —
                        // mirrors the parent_metadata Option discipline.
                        let relations = if req.include_relations {
                            Some(
                                relations_by_chunk
                                    .get(&c.chunk_id)
                                    .cloned()
                                    .unwrap_or_default(),
                            )
                        } else {
                            None
                        };
                        (
                            chunk.clone(),
                            c.final_score,
                            c.source.clone(),
                            parent_metadata,
                            relations,
                        )
                    })
            })
            .collect();

        let took_us = start.elapsed().as_micros() as u64;

        // ── Step 6: Build /explain plan if requested ────────────────────
        let explain_plan = if req.explain {
            Some(ExplainPlan {
                filter: FilterExplain {
                    eligible_count: eligible.len(),
                    universe_count,
                    selectivity: selectivity_val,
                },
                ann: AnnExplain {
                    engine: explain_engine.unwrap_or("none").to_string(),
                    candidates_inspected: explain_candidates,
                    ef_search_used: hnsw_ef_search_default(),
                },
            })
        } else {
            None
        };

        Ok((hits, total, took_us, explain_plan))
    }

    // ── Chunk Relations ──────────────────────────────────────────────────

    /// Create a batch of chunk relations. The server mints a UUIDv4
    /// `relation_id` and stamps `created_at` for each. Self-relations
    /// (`source == target`) are rejected. Returns the created edges with their
    /// assigned ids.
    pub async fn create_relations(
        &self,
        collection_name: &str,
        new: Vec<CreateRelation>,
    ) -> Result<Vec<ChunkRelation>, Box<dyn std::error::Error + Send + Sync>> {
        // Writer role: build the edges without local state — target_status is
        // stored as "missing" and re-resolved against the live chunk set at
        // every read on serving nodes — and append ONE durable fragment.
        if self.role == NodeRole::Writer {
            let now = Utc::now();
            let mut built: Vec<ChunkRelation> = Vec::with_capacity(new.len());
            for r in new {
                if r.source_chunk_id == r.target_chunk_id {
                    return Err("A relation's source and target chunk must differ".into());
                }
                built.push(ChunkRelation {
                    relation_id: uuid::Uuid::new_v4().to_string(),
                    source_chunk_id: r.source_chunk_id,
                    target_chunk_id: r.target_chunk_id,
                    target_document_id: r.target_document_id,
                    relation_type: r.relation_type,
                    target_status: "missing".to_string(),
                    metadata: r.metadata,
                    created_at: now,
                });
            }
            if !built.is_empty() {
                let payload = serde_json::to_vec(&built)?;
                let records = built.len() as u64;
                crate::storage::lsm::append_relation_upsert(
                    self.storage.as_ref(),
                    collection_name,
                    bytes::Bytes::from(payload),
                    records,
                )
                .await
                .map_err(|e| format!("cloud relation-upsert append failed: {e}"))?;
            }
            return Ok(built);
        }
        self.ensure_attached(collection_name).await?;
        // Phase 1 (read lock): build the edges, resolving target_status against
        // the chunk map. Then release the lock BEFORE the S3 round-trip (#2/#4).
        let now = Utc::now();
        let mut built: Vec<ChunkRelation> = Vec::with_capacity(new.len());
        {
            let collections = self.collections.read().await;
            let loaded = collections
                .get(collection_name)
                .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
            for r in new {
                if r.source_chunk_id == r.target_chunk_id {
                    return Err("A relation's source and target chunk must differ".into());
                }
                built.push(ChunkRelation {
                    relation_id: uuid::Uuid::new_v4().to_string(),
                    source_chunk_id: r.source_chunk_id,
                    target_chunk_id: r.target_chunk_id,
                    target_document_id: r.target_document_id,
                    relation_type: r.relation_type,
                    target_status: if loaded.filter_index.contains(r.target_chunk_id) {
                        "found".to_string()
                    } else {
                        "missing".to_string()
                    },
                    metadata: r.metadata,
                    created_at: now,
                });
            }
        } // read lock released before S3 I/O.

        // Phase 2 (NO lock): durable S3 relation-upsert FIRST (S3-first ordering).
        let mut appended_seq: Option<u64> = None;
        if self.cloud_mode && !built.is_empty() {
            let payload = serde_json::to_vec(&built)?;
            let records = built.len() as u64;
            let seq = crate::storage::lsm::append_relation_upsert(
                self.storage.as_ref(),
                collection_name,
                bytes::Bytes::from(payload),
                records,
            )
            .await
            .map_err(|e| format!("cloud relation-upsert append failed: {e}"))?;
            appended_seq = Some(seq);
        }

        // Phase 3 (read lock): apply locally (durable S3 record already written).
        // On EITHER failure mode — collection deleted in the lock gap, or a
        // local insert error — compensate with a RelationDelete for the minted
        // ids, so the durable upsert fragment can't resurrect orphan edges on a
        // later materialize (the same discipline ingest applies to chunks).
        let apply_result: Result<(), Box<dyn std::error::Error + Send + Sync>> = {
            let mut collections = self.collections.write().await;
            match collections.get_mut(collection_name) {
                Some(loaded) => {
                    // Double-apply guard vs the manifest refresher; replay of a
                    // relation upsert is idempotent anyway (same relation_ids).
                    if appended_seq.map(|s| loaded.applied.covers(s)) == Some(true) {
                        Ok(())
                    } else {
                        let r = loaded.relation_store.insert_batch(&built);
                        if r.is_ok() {
                            // Mark only after a successful apply (see C2).
                            if let Some(seq) = appended_seq {
                                loaded.applied.mark(seq);
                                loaded.metadata.applied_seq = loaded.applied.contiguous;
                                let _ = store::save_metadata(&self.data_dir, &loaded.metadata);
                            }
                        }
                        r
                    }
                }
                None => Err(format!("Collection '{}' not found", collection_name).into()),
            }
        };
        if let Err(e) = apply_result {
            if self.cloud_mode && !built.is_empty() {
                let ids: Vec<String> = built.iter().map(|r| r.relation_id.clone()).collect();
                if let Err(te) = crate::storage::lsm::append_relation_delete(
                    self.storage.as_ref(),
                    collection_name,
                    &ids,
                )
                .await
                {
                    tracing::error!(
                        "compensation: relation-delete for '{}' failed: {} — orphan \
                         relation fragment may resurrect on rebuild",
                        collection_name,
                        te
                    );
                }
            }
            return Err(e);
        }
        Ok(built)
    }

    /// Delete a relation by id. Returns true if it existed.
    pub async fn delete_relation(
        &self,
        collection_name: &str,
        relation_id: &str,
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        // Writer role: durable relation-delete only (idempotent on replay).
        if self.role == NodeRole::Writer {
            crate::storage::lsm::append_relation_delete(
                self.storage.as_ref(),
                collection_name,
                std::slice::from_ref(&relation_id.to_string()),
            )
            .await
            .map_err(|e| format!("cloud relation-delete append failed: {e}"))?;
            return Ok(true);
        }
        self.ensure_attached(collection_name).await?;
        // Existence check under a short read lock, then release before S3 I/O.
        {
            let collections = self.collections.read().await;
            if !collections.contains_key(collection_name) {
                return Err(format!("Collection '{}' not found", collection_name).into());
            }
        }

        // Cloud mode: durable S3 relation-delete FIRST — NO lock held across the
        // S3 round-trip (#4). Replay drops the id; deleting an absent id is an
        // idempotent no-op on materialize.
        let mut appended_seq: Option<u64> = None;
        if self.cloud_mode {
            let seq = crate::storage::lsm::append_relation_delete(
                self.storage.as_ref(),
                collection_name,
                std::slice::from_ref(&relation_id.to_string()),
            )
            .await
            .map_err(|e| format!("cloud relation-delete append failed: {e}"))?;
            appended_seq = Some(seq);
        }

        // Apply locally (write lock: the seq tracker needs &mut).
        let mut collections = self.collections.write().await;
        let loaded = collections
            .get_mut(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
        if appended_seq.map(|s| loaded.applied.covers(s)) == Some(true) {
            return Ok(true); // refresher already applied our delete
        }
        let r = loaded.relation_store.delete(relation_id);
        if r.is_ok() {
            // Mark only after a successful apply (see C2).
            if let Some(seq) = appended_seq {
                loaded.applied.mark(seq);
                loaded.metadata.applied_seq = loaded.applied.contiguous;
                let _ = store::save_metadata(&self.data_dir, &loaded.metadata);
            }
        }
        r
    }

    /// List a single chunk's relations, with `target_status` resolved against
    /// the current chunk set.
    pub async fn get_chunk_relations(
        &self,
        collection_name: &str,
        chunk_id: u64,
        direction: RelationDirection,
        types: Option<&[String]>,
    ) -> Result<Vec<ChunkRelation>, Box<dyn std::error::Error + Send + Sync>> {
        if self.role == NodeRole::Writer {
            return Err("this node runs in writer role and does not serve queries".into());
        }
        self.ensure_attached(collection_name).await?;
        let collections = self.collections.read().await;
        let loaded = collections
            .get(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
        loaded
            .last_used
            .store(next_lru_tick(), std::sync::atomic::Ordering::Relaxed);
        let mut edges = loaded
            .relation_store
            .for_chunk(chunk_id, direction, types)?;
        for edge in edges.iter_mut() {
            edge.target_status = if loaded.filter_index.contains(edge.target_chunk_id) {
                "found".to_string()
            } else {
                "missing".to_string()
            };
        }
        Ok(edges)
    }

    // ── Delete (soft-delete via tombstones) ──────────────────────────────

    /// Soft-delete a set of chunk ids. Tombstones them (so they immediately
    /// vanish from search results), persists the tombstones, prunes incident
    /// relations, and — in object-storage mode — appends a tombstone WAL
    /// fragment so the deletion is durably S3-native. The vectors physically
    /// remain in the HNSW/FTS indexes until the next rebuild/compaction; search
    /// filters them out in the meantime. Returns the number newly deleted.
    /// Local tombstone apply — shared by the delete path and fragment replay.
    /// Idempotent: already-deleted / absent ids are filtered by the caller (or
    /// harmlessly re-tombstoned in redb).
    fn apply_tombstones_locally(
        data_dir: &Path,
        loaded: &mut LoadedCollection,
        apply: &[u64],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        loaded.chunk_store.tombstone_batch(apply)?;
        for id in apply {
            loaded.tombstones.insert(*id);
        }
        // Persist the corrected live count IMMEDIATELY after the tombstones —
        // before the fallible relation pruning — so a pruning error can't leave
        // chunk_count permanently overstated.
        let removed = apply.len() as u64;
        loaded.metadata.chunk_count = loaded.metadata.chunk_count.saturating_sub(removed);
        store::save_metadata(data_dir, &loaded.metadata)?;
        // Keep the filter index in step with the tombstones so `eligible` /
        // selectivity don't count deleted chunks — incrementally (O(batch)).
        for id in apply {
            if let Ok(Some(c)) = loaded.chunk_store.get(*id) {
                loaded.filter_index.remove(*id, &filter_meta(&c));
            }
        }
        // Prune relations incident on the deleted chunks (F6: propagate errors;
        // on failure the edges are orphaned but target_status reports their
        // endpoints as missing, and cloud replay prunes them independently).
        for &id in apply {
            let edges = loaded
                .relation_store
                .for_chunk(id, RelationDirection::Both, None)?;
            for e in edges {
                loaded.relation_store.delete(&e.relation_id)?;
            }
        }
        Ok(())
    }

    /// Replay one WAL fragment into the local indexes (the refresher's apply
    /// path — mirrors `cloud::materialize`'s kind dispatch exactly). MUST be
    /// idempotent: fragments may race the originating node's own local apply.
    fn apply_fragment_locally(
        data_dir: &Path,
        collection_name: &str,
        loaded: &mut LoadedCollection,
        kind: crate::storage::lsm::FragmentKind,
        payload: &[u8],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use crate::storage::lsm::FragmentKind;
        match kind {
            FragmentKind::Data => {
                let chunks: Vec<DocumentChunk> = serde_json::from_slice(payload)?;
                // Defensive dims validation: a foreign writer's stale config
                // could have let a wrong-length vector into a durable fragment;
                // appending it would corrupt the mmap file for every vector
                // after it. Quarantine (skip + loud error), never apply.
                let mut fresh: Vec<DocumentChunk> = Vec::with_capacity(chunks.len());
                'chunk: for c in chunks {
                    // Idempotent replay: skip ids already present so
                    // chunk_count can't double-count.
                    if loaded.filter_index.contains(c.id) || loaded.tombstones.contains(&c.id) {
                        continue;
                    }
                    for (space, emb) in &c.embeddings {
                        let expected = loaded
                            .metadata
                            .vector_spaces
                            .get(space)
                            .map(|v| v.dims)
                            .unwrap_or(loaded.metadata.embedding_dims);
                        if emb.len() != expected {
                            tracing::error!(
                                "replay: chunk {} in '{}' has {}-dim embedding for space '{}' \
                                 (expected {}); quarantined",
                                c.id,
                                collection_name,
                                emb.len(),
                                space,
                                expected
                            );
                            crate::metrics::inc(&crate::metrics::QUARANTINED_CHUNKS_TOTAL);
                            continue 'chunk;
                        }
                    }
                    fresh.push(c);
                }
                if fresh.is_empty() {
                    return Ok(());
                }
                let rel_adds: Vec<(u64, Option<u64>, Option<String>)> = fresh
                    .iter()
                    .map(|c| (c.id, c.parent_id, c.group_id.clone()))
                    .collect();
                let mut space_vectors: HashMap<String, Vec<(u64, Vec<f32>)>> = HashMap::new();
                for c in &fresh {
                    for (space, emb) in &c.embeddings {
                        space_vectors
                            .entry(space.clone())
                            .or_default()
                            .push((c.id, emb.clone()));
                    }
                }
                let count = fresh.len();
                Self::apply_ingest_commit(
                    data_dir,
                    collection_name,
                    loaded,
                    rel_adds,
                    &fresh,
                    space_vectors,
                    count,
                )
            }
            FragmentKind::Tombstone => {
                let ids: Vec<u64> = serde_json::from_slice(payload)?;
                let apply: Vec<u64> = ids
                    .into_iter()
                    .filter(|id| loaded.filter_index.contains(*id))
                    .collect();
                if apply.is_empty() {
                    return Ok(());
                }
                Self::apply_tombstones_locally(data_dir, loaded, &apply)
            }
            FragmentKind::RelationUpsert => {
                let rels: Vec<ChunkRelation> = serde_json::from_slice(payload)?;
                loaded.relation_store.insert_batch(&rels)
            }
            FragmentKind::RelationDelete => {
                let ids: Vec<String> = serde_json::from_slice(payload)?;
                for id in &ids {
                    loaded.relation_store.delete(id)?;
                }
                Ok(())
            }
        }
    }

    /// The per-namespace attach mutex (created on demand). Serializes every
    /// destructive local-state transition for a namespace: attach, refresh-
    /// triggered full re-attach, and LRU detach — so two of them can never run
    /// concurrently on the same live index directory.
    async fn attach_lock(&self, ns: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.attach_locks.lock().await;
        locks.entry(ns.to_string()).or_default().clone()
    }

    /// Lazy attach: make sure `ns` is attached (rebuilt from the bucket) before
    /// serving a request against it. No-op when already attached or when lazy
    /// attach is off. A request stampede on a cold namespace rebuilds ONCE via
    /// the per-namespace mutex; the global collections lock is never held
    /// across the rebuild.
    async fn ensure_attached(
        &self,
        ns: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if !self.lazy_attach {
            return Ok(());
        }
        if self.collections.read().await.contains_key(ns) {
            return Ok(());
        }
        let lock = self.attach_lock(ns).await;
        let _guard = lock.lock().await;
        // Double-check under the attach mutex: a racer may have attached.
        if self.collections.read().await.contains_key(ns) {
            return Ok(());
        }
        // Confirm the namespace exists in the bucket. Check the registry first
        // (boot-time discovery), then the bucket itself — a collection created
        // by ANOTHER node after our boot is attachable too.
        let known = self.registered.read().await.contains(ns);
        if !known {
            validate_name_segment(ns, "Collection")?;
            let exists = cloud::read_bucket_config(self.storage.as_ref(), ns)
                .await?
                .is_some()
                || self
                    .storage
                    .exists(&format!("{ns}/manifest"))
                    .await
                    .unwrap_or(false);
            if !exists {
                // Don't leak an attach-lock entry per garbage name probed.
                self.attach_locks.lock().await.remove(ns);
                return Err(format!("Collection '{}' not found", ns).into());
            }
            self.registered.write().await.insert(ns.to_string());
        }
        let start = std::time::Instant::now();
        let n = self.rebuild_collection_from_storage(ns).await?;
        crate::metrics::inc(&crate::metrics::ATTACH_TOTAL);
        crate::metrics::add(
            &crate::metrics::ATTACH_SECONDS_SUM_MILLIS,
            start.elapsed().as_millis() as u64,
        );
        tracing::info!(
            "Attached '{}' on demand ({} chunks in {:.2}s)",
            ns,
            n,
            start.elapsed().as_secs_f64()
        );
        self.maybe_evict_lru(ns).await;
        Ok(())
    }

    /// Enforce the attached-collection budget: detach the least-recently-used
    /// collection (never `just_attached`). Detach is safe — the bucket is the
    /// source of truth — and local files are deleted only AFTER the global
    /// lock is released (never filesystem I/O under the lock). The evicted
    /// namespace stays registered for future re-attach.
    async fn maybe_evict_lru(&self, just_attached: &str) {
        if self.max_attached == 0 {
            return;
        }
        // Pick the victim under a short read lock.
        let victim: Option<String> = {
            let collections = self.collections.read().await;
            if collections.len() <= self.max_attached {
                None
            } else {
                collections
                    .iter()
                    .filter(|(name, _)| name.as_str() != just_attached)
                    .min_by_key(|(_, l)| l.last_used.load(std::sync::atomic::Ordering::Relaxed))
                    .map(|(name, _)| name.clone())
            }
        };
        let Some(name) = victim else { return };
        // Serialize with attach/re-attach on the same namespace: file deletion
        // must never race a rebuild into the same directory.
        let lock = self.attach_lock(&name).await;
        let _guard = lock.lock().await;
        {
            let mut collections = self.collections.write().await;
            // Re-check under the attach mutex (a racer may have evicted or the
            // budget may have been satisfied meanwhile).
            if collections.len() <= self.max_attached || !collections.contains_key(&name) {
                return;
            }
            collections.remove(&name);
        } // global lock released before any filesystem work.
        self.registered.write().await.insert(name.clone());
        if let Err(e) = store::delete_collection_data(&self.data_dir, &name) {
            tracing::warn!("detach '{}': local cleanup failed: {}", name, e);
        }
        tracing::info!("Detached '{}' (LRU, budget {})", name, self.max_attached);
    }

    /// Converge this node's local indexes with the bucket manifest: apply
    /// fragments this node hasn't seen (a remote writer's, or another serving
    /// node's), in seq order, idempotently. Returns the manifest's `next_seq`.
    ///
    /// Two-branch compaction rule: if the compaction watermark has passed our
    /// contiguous frontier, fragments we NEVER applied were folded into the
    /// segment — the only correct recovery is a full re-attach. Otherwise the
    /// folded fragments are ones we already applied, and only the live tail
    /// needs replay.
    pub async fn refresh_collection(
        &self,
        collection_name: &str,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        if !self.cloud_mode {
            return Ok(0);
        }
        // Deleted-collection detection (no namespace generations yet): a
        // manifest that has VANISHED means the collection was deleted on
        // another node — detach instead of warning forever while serving dead
        // data.
        let (manifest, _) = match crate::storage::lsm::read_manifest(
            self.storage.as_ref(),
            collection_name,
        )
        .await
        {
            Ok(m) => m,
            Err(e) => {
                let manifest_gone = !self
                    .storage
                    .exists(&format!("{collection_name}/manifest"))
                    .await
                    .unwrap_or(true);
                if manifest_gone {
                    self.detach_deleted(collection_name).await;
                    return Err(format!(
                        "collection '{collection_name}' was deleted in object storage"
                    )
                    .into());
                }
                return Err(e.into());
            }
        };
        let next_seq = manifest.next_seq;

        // Config convergence: vector-space adds/removes/default switches are
        // CAS'd into {ns}/collection.json, NOT written as fragments — sync them
        // here so already-attached nodes learn about them. A recreate (config
        // created_at differs from ours) forces a full re-attach.
        let bucket_cfg = cloud::read_bucket_config(self.storage.as_ref(), collection_name).await?;
        let mut force_reattach = false;
        if let Some(cfg) = &bucket_cfg {
            let mut collections = self.collections.write().await;
            if let Some(loaded) = collections.get_mut(collection_name) {
                if loaded.metadata.created_at != cfg.created_at {
                    // Same name, different collection: it was deleted and
                    // recreated while we were attached.
                    force_reattach = true;
                } else if loaded.metadata.vector_spaces != cfg.vector_spaces
                    || loaded.metadata.default_vector_space != cfg.default_vector_space
                {
                    for (name, spec) in &cfg.vector_spaces {
                        if !loaded.vector_spaces.contains_key(name) {
                            loaded.vector_spaces.insert(
                                name.clone(),
                                Arc::new(VectorState {
                                    index: None,
                                    key_to_chunk_id: Vec::new(),
                                    mmap_vectors: None,
                                    vectors: Vec::new(),
                                    dims: spec.dims,
                                }),
                            );
                        }
                    }
                    loaded
                        .vector_spaces
                        .retain(|name, _| cfg.vector_spaces.contains_key(name));
                    loaded.metadata.vector_spaces = cfg.vector_spaces.clone();
                    loaded.metadata.default_vector_space = cfg.default_vector_space.clone();
                    loaded.metadata.config = cfg.config.clone();
                    store::save_metadata(&self.data_dir, &loaded.metadata)?;
                    tracing::info!(
                        "refresh '{}': synced vector-space config from bucket",
                        collection_name
                    );
                }
            }
        }

        let contiguous = {
            let collections = self.collections.read().await;
            let loaded = collections
                .get(collection_name)
                .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
            loaded.applied.contiguous
        };

        let needs_reattach = force_reattach
            || manifest
                .compaction_watermark
                .map(|wm| wm + 1 > contiguous)
                .unwrap_or(false);
        if needs_reattach {
            // Full re-attach, SERIALIZED on the per-namespace attach mutex so
            // concurrent refresh ticks / min_seq waiters can't run destructive
            // rebuilds into the same live directory.
            let lock = self.attach_lock(collection_name).await;
            let _guard = lock.lock().await;
            // Re-check under the mutex: a racer may have already re-attached.
            let still_needed = {
                let collections = self.collections.read().await;
                match collections.get(collection_name) {
                    Some(loaded) => {
                        force_reattach
                            && loaded.metadata.created_at
                                != bucket_cfg
                                    .as_ref()
                                    .map(|c| c.created_at)
                                    .unwrap_or(loaded.metadata.created_at)
                            || manifest
                                .compaction_watermark
                                .map(|wm| wm + 1 > loaded.applied.contiguous)
                                .unwrap_or(false)
                    }
                    None => true,
                }
            };
            if still_needed {
                tracing::info!(
                    "refresh '{}': full re-attach (compaction passed local frontier or recreate)",
                    collection_name
                );
                crate::metrics::inc(&crate::metrics::REFRESH_REATTACHES_TOTAL);
                self.rebuild_collection_from_storage(collection_name)
                    .await?;
            }
            return Ok(next_seq);
        }

        // Filter fragment REFS first, fetch only what we need (a caught-up
        // node fetches nothing), then apply per-fragment with the lock
        // RELEASED between fragments so a large backlog can't cause a
        // node-wide read outage.
        let pending_refs: Vec<crate::storage::lsm::FragmentRef> = manifest
            .uncompacted()
            .filter(|r| r.seq >= contiguous)
            .cloned()
            .collect();
        if pending_refs.is_empty() {
            return Ok(next_seq);
        }
        let mut applied_any = false;
        for fref in pending_refs {
            let bytes = crate::storage::lsm::read_fragment(
                self.storage.as_ref(),
                collection_name,
                &fref.id,
            )
            .await?;
            let mut collections = self.collections.write().await;
            let loaded = collections
                .get_mut(collection_name)
                .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
            if loaded.applied.covers(fref.seq) {
                continue;
            }
            Self::apply_fragment_locally(
                &self.data_dir,
                collection_name,
                loaded,
                fref.kind,
                &bytes,
            )?;
            loaded.applied.mark(fref.seq);
            crate::metrics::inc(&crate::metrics::REFRESH_FRAGMENTS_APPLIED_TOTAL);
            applied_any = true;
        }
        if applied_any {
            let mut collections = self.collections.write().await;
            if let Some(loaded) = collections.get_mut(collection_name) {
                loaded.metadata.applied_seq = loaded.applied.contiguous;
                store::save_metadata(&self.data_dir, &loaded.metadata)?;
            }
        }
        Ok(next_seq)
    }

    /// Detach a collection whose bucket namespace disappeared (deleted by
    /// another node): drop it from the map + registry + caches and remove
    /// local files, serialized on the attach mutex.
    async fn detach_deleted(&self, ns: &str) {
        let lock = self.attach_lock(ns).await;
        let _guard = lock.lock().await;
        let removed = {
            let mut collections = self.collections.write().await;
            collections.remove(ns).is_some()
        };
        self.registered.write().await.remove(ns);
        self.bucket_configs.write().await.remove(ns);
        self.writer_pools.lock().await.remove(ns);
        if removed {
            let _ = store::delete_collection_data(&self.data_dir, ns);
            tracing::info!("Detached '{}': deleted in object storage", ns);
        }
    }

    /// Refresh every attached collection    /// Refresh every attached collection (the background refresher's tick).
    pub async fn refresh_all(&self) {
        let names: Vec<String> = {
            let collections = self.collections.read().await;
            collections.keys().cloned().collect()
        };
        for name in names {
            if let Err(e) = self.refresh_collection(&name).await {
                tracing::warn!("refresh of '{}' failed: {}", name, e);
            }
        }
    }

    pub async fn delete_chunks(
        &self,
        collection_name: &str,
        ids: &[u64],
    ) -> Result<(usize, Option<u64>), Box<dyn std::error::Error + Send + Sync>> {
        crate::metrics::inc(&crate::metrics::DELETE_REQUESTS_TOTAL);
        // Writer role: durable tombstone only. Without local indexes we can't
        // filter to ids-that-exist; a tombstone for an absent id is an
        // idempotent no-op on replay, so append the deduped set as-is.
        if self.role == NodeRole::Writer {
            validate_name_segment(collection_name, "Collection")?;
            // Existence check: without it, a tombstone for a bogus namespace
            // would CREATE that namespace in the bucket (phantom collection).
            self.bucket_config(collection_name, false).await?;
            let mut seen = std::collections::HashSet::new();
            let newly: Vec<u64> = ids.iter().copied().filter(|id| seen.insert(*id)).collect();
            if newly.is_empty() {
                return Ok((0, None));
            }
            // Ids can never legitimately reach the allocator frontier; a bogus
            // huge id would otherwise poison max_id forever (rebuilds compute
            // next_id = max_id + 1 → overflow / id reuse).
            let frontier =
                crate::storage::id_alloc::frontier(self.storage.as_ref(), collection_name).await?;
            if let Some(bad) = newly.iter().find(|id| **id >= frontier) {
                return Err(format!(
                    "chunk id {bad} was never allocated in '{collection_name}' \
                     (allocator frontier {frontier})"
                )
                .into());
            }
            let seq = crate::storage::lsm::append_tombstone(
                self.storage.as_ref(),
                collection_name,
                &newly,
            )
            .await
            .map_err(|e| format!("LSM tombstone append failed: {e}"))?;
            maybe_auto_compact(
                self.storage.clone(),
                collection_name.to_string(),
                self.compacting.clone(),
            );
            return Ok((newly.len(), Some(seq)));
        }
        self.ensure_attached(collection_name).await?;

        // Phase 1 (read lock): determine which ids are actually deletable.
        // DEDUP the input — `{"ids":[5,5,5]}` must count (and decrement
        // chunk_count by) ONE delete, not three.
        let newly: Vec<u64> = {
            let collections = self.collections.read().await;
            let loaded = collections
                .get(collection_name)
                .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
            let mut seen = std::collections::HashSet::new();
            ids.iter()
                .copied()
                .filter(|id| seen.insert(*id) && loaded.filter_index.contains(*id))
                .collect()
        }; // read lock released here.
        if newly.is_empty() {
            return Ok((0, None));
        }

        // Phase 2 (NO lock held): DURABLE S3 tombstone FIRST. This is the S3
        // network round-trip; doing it without the collections lock means a slow
        // S3 call no longer stalls every other collection's reads/writes (#2).
        // S3-first also fixes the F5 split-brain: on failure nothing local is
        // committed, so the caller retries cleanly.
        let mut appended_seq: Option<u64> = None;
        if self.cloud_mode {
            let seq = crate::storage::lsm::append_tombstone(
                self.storage.as_ref(),
                collection_name,
                &newly,
            )
            .await
            .map_err(|e| format!("LSM tombstone append failed (delete not applied): {e}"))?;
            appended_seq = Some(seq);
            maybe_auto_compact(
                self.storage.clone(),
                collection_name.to_string(),
                self.compacting.clone(),
            );
        }

        // Phase 3 (write lock): apply local state. Re-check membership under the
        // lock (a concurrent delete could have tombstoned some ids meanwhile);
        // a redundant S3 tombstone for an already-deleted id is a harmless
        // idempotent no-op on replay.
        let mut collections = self.collections.write().await;
        let loaded = collections
            .get_mut(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
        // Double-apply guard: the refresher may have applied OUR tombstone
        // fragment between the append and this reacquire.
        if let Some(seq) = appended_seq {
            if loaded.applied.covers(seq) {
                return Ok((newly.len(), appended_seq));
            }
        }
        let apply: Vec<u64> = newly
            .iter()
            .copied()
            .filter(|id| loaded.filter_index.contains(*id))
            .collect();
        if apply.is_empty() {
            return Ok((0, appended_seq));
        }

        Self::apply_tombstones_locally(&self.data_dir, loaded, &apply)?;
        // Mark ONLY after the apply succeeded: marking first would make a
        // failed apply invisible to the refresher forever (the node would keep
        // serving deleted data).
        if let Some(seq) = appended_seq {
            loaded.applied.mark(seq);
            loaded.metadata.applied_seq = loaded.applied.contiguous;
            store::save_metadata(&self.data_dir, &loaded.metadata)?;
        }

        tracing::info!(
            "Deleted {} chunk(s) from '{}' (tombstoned{})",
            apply.len(),
            collection_name,
            if self.cloud_mode {
                " + WAL tombstone"
            } else {
                ""
            }
        );
        Ok((apply.len(), appended_seq))
    }

    /// Soft-delete every chunk matching a metadata filter (e.g. all chunks of a
    /// file_id, or a metadata predicate). Resolves matching ids, then delegates
    /// to `delete_chunks`.
    pub async fn delete_by_filter(
        &self,
        collection_name: &str,
        filters: &HashMap<String, FilterValue>,
    ) -> Result<(usize, Option<u64>), Box<dyn std::error::Error + Send + Sync>> {
        if self.role == NodeRole::Writer {
            return Err(
                "delete-by-filter needs a serving node's indexes; this node runs in writer role \
                 (delete by explicit ids instead)"
                    .into(),
            );
        }
        self.ensure_attached(collection_name).await?;
        // Collect matching, not-yet-deleted ids under a read lock first.
        let ids: Vec<u64> = {
            let collections = self.collections.read().await;
            let loaded = collections
                .get(collection_name)
                .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
            let mut ids: Vec<u64> = Vec::new();
            loaded.chunk_store.for_each(|id, c| {
                if !loaded.tombstones.contains(&id) && crate::filter::matches_filters(&c, filters) {
                    ids.push(id);
                }
            })?;
            ids
        };
        if ids.is_empty() {
            return Ok((0, None));
        }
        self.delete_chunks(collection_name, &ids).await
    }

    // ── Cloud compaction (object-storage mode) ───────────────────────────

    /// Compact a collection's S3 LSM: fold all segments + WAL fragments into a
    /// single new segment (applying deletes), then rewrite the manifest to
    /// reference only it. Reclaims space for tombstoned data. No-op in local
    /// mode. Returns the number of live records in the resulting segment.
    ///
    /// This CAS-retries against concurrent appends: if the manifest changed
    /// under us, we re-materialize the fresh state and try again.
    pub async fn compact_collection(
        &self,
        collection_name: &str,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        if !self.cloud_mode {
            return Ok(0);
        }
        self.ensure_attached(collection_name).await?;
        // Verify the collection exists (under a short read lock).
        {
            let collections = self.collections.read().await;
            if !collections.contains_key(collection_name) {
                return Err(format!("Collection '{}' not found", collection_name).into());
            }
        }

        Ok(compact_storage(self.storage.as_ref(), collection_name).await?)
    }

    /// Rebuild a collection's LOCAL indexes from its object-storage manifest
    /// (materialize segments + WAL fragments → chunks → local redb/Tantivy/HNSW/
    /// filter index). This is cold-start recovery: an ephemeral node with an
    /// empty local disk reconstructs the collection entirely from S3. Returns the
    /// number of live chunks recovered. Cloud mode only.
    pub async fn rebuild_collection_from_storage(
        &self,
        collection_name: &str,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        validate_name_segment(collection_name, "Collection")?;
        let (manifest, _) =
            crate::storage::lsm::read_manifest(self.storage.as_ref(), collection_name).await?;
        let materialized =
            cloud::materialize(self.storage.as_ref(), collection_name, &manifest).await?;
        let chunks: Vec<DocumentChunk> = materialized.chunks.values().cloned().collect();
        let live_count = chunks.len();

        // Collection config: the bucket `{ns}/collection.json` is authoritative
        // (carries the user's real vector-space specs, created_at, and
        // CollectionConfig — the old inference fabricated all three and lost
        // `embed_model`). Fall back to inference only for pre-v0.4 namespaces,
        // and back-fill the bucket config so the fallback runs at most once.
        let bucket_cfg = cloud::read_bucket_config(self.storage.as_ref(), collection_name).await?;
        let (vector_spaces, default_space, dims, created_at, coll_config) = match &bucket_cfg {
            Some(cfg) => (
                cfg.vector_spaces.clone(),
                cfg.default_vector_space.clone(),
                cfg.embedding_dims,
                cfg.created_at,
                cfg.config.clone(),
            ),
            None => {
                // Legacy inference from recovered embeddings.
                let mut vector_spaces: HashMap<String, VectorSpaceConfig> = HashMap::new();
                for c in &chunks {
                    for (space, emb) in &c.embeddings {
                        vector_spaces
                            .entry(space.clone())
                            .or_insert(VectorSpaceConfig {
                                dims: emb.len(),
                                model: "recovered".to_string(),
                                status: "active".to_string(),
                            });
                    }
                }
                if vector_spaces.is_empty() {
                    vector_spaces.insert(
                        "default".to_string(),
                        VectorSpaceConfig {
                            dims: 384,
                            model: "recovered".to_string(),
                            status: "active".to_string(),
                        },
                    );
                }
                let default_space = vector_spaces.keys().next().cloned();
                let dims = vector_spaces.values().next().map(|s| s.dims).unwrap_or(384);
                (
                    vector_spaces,
                    default_space,
                    dims,
                    Utc::now(),
                    CollectionConfig::default(),
                )
            }
        };
        // next_id must never regress or reuse an id: one past the high-water
        // mark whenever ANY id was ever assigned (max_id covers tombstoned ids
        // via the WAL and the segment's stored max_id). The old
        // `+ if live_count > 0` form reused the highest id when every chunk was
        // deleted.
        let next_id = if materialized.max_id > 0 || live_count > 0 {
            materialized.max_id + 1
        } else {
            0
        };

        let metadata = Collection {
            name: collection_name.to_string(),
            created_at,
            vector_spaces: vector_spaces.clone(),
            default_vector_space: default_space.clone(),
            embedding_dims: dims,
            chunk_count: live_count as u64,
            next_id,
            config: coll_config,
            applied_seq: manifest.next_seq,
        };
        store::save_metadata(&self.data_dir, &metadata)?;
        // Organic migration: back-fill the bucket config for pre-v0.4
        // namespaces (create-only, race-safe; best-effort).
        if bucket_cfg.is_none() {
            let backfill = cloud::BucketConfig::from_collection(&metadata);
            if let Err(e) = cloud::write_bucket_config_if_absent(
                self.storage.as_ref(),
                collection_name,
                &backfill,
            )
            .await
            {
                if !matches!(e, crate::storage::StorageError::AlreadyExists(_)) {
                    tracing::warn!(
                        "bucket config back-fill for '{}' failed: {}",
                        collection_name,
                        e
                    );
                }
            }
        }

        // Build local stores from the materialized chunks. Start from a CLEAN
        // chunk store: S3 is the source of truth on recovery, so any pre-existing
        // local redb (e.g. a stale chunk left by a partially-committed ingest that
        // was later compensated with a tombstone) must not survive. Wipe first.
        let chunks_db = store::chunks_db_path(&self.data_dir, collection_name);
        if let Some(parent) = chunks_db.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let _ = std::fs::remove_file(&chunks_db);
        let chunk_store = ChunkCache::new(ChunkStore::open(&chunks_db)?);
        let to_persist: Vec<(u64, DocumentChunk)> =
            chunks.iter().map(|c| (c.id, c.clone())).collect();
        chunk_store.insert_batch(&to_persist)?;

        // FTS index.
        let tantivy_dir = store::tantivy_dir(&self.data_dir, collection_name);
        let fts = tantivy_fts::build_index(&tantivy_dir, &chunks, 0)?;

        // Vector spaces (HNSW) from embeddings.
        let vectors_dir = store::vectors_dir(&self.data_dir, collection_name);
        let mut vs_map: HashMap<String, Arc<VectorState>> = HashMap::new();
        for (space, cfg) in &vector_spaces {
            let mut ids = Vec::new();
            let mut vecs = Vec::new();
            for c in &chunks {
                if let Some(emb) = c.embeddings.get(space) {
                    ids.push(c.id);
                    vecs.push(emb.clone());
                }
            }
            let index_path = vectors_dir.join(format!("{}.index", space));
            let vecs_path = vectors_dir.join(format!("{}.bin", space));
            let vs = vector::build_vector_index(&index_path, &vecs_path, &ids, &vecs, cfg.dims)?;
            vs_map.insert(space.clone(), Arc::new(vs));
        }

        // Relationships + filter index from the recovered chunks.
        let mut relationships = RelationshipStore::new();
        for c in &chunks {
            relationships.add(c.id, c.parent_id, c.group_id.clone());
        }
        // Materialized state is already live-only (tombstones applied on
        // replay); build the index streaming, no full map in RAM.
        let mut filter_index = FilterIndex::new();
        for c in &chunks {
            filter_index.insert(c.id, &filter_meta(c));
        }
        filter_index.finalize();

        // Reconstruct the typed-relation store from the materialized relations
        // (recovered from the S3 WAL/segments) — so relations survive a cold
        // restart, not just chunks.
        let relations_db = store::relations_db_path(&self.data_dir, collection_name);
        let _ = std::fs::remove_file(&relations_db); // start clean, then repopulate
        let relation_store = RelationStore::open(&relations_db)?;
        let recovered_relations: Vec<ChunkRelation> =
            materialized.relations.values().cloned().collect();
        if !recovered_relations.is_empty() {
            relation_store.insert_batch(&recovered_relations)?;
        }

        let loaded = LoadedCollection {
            id_pool: Default::default(),
            hnsw_unsaved: HashMap::new(),
            last_used: std::sync::atomic::AtomicU64::new(next_lru_tick()),
            // A rebuild materialized EVERYTHING in the manifest it read.
            applied: SeqTracker::starting_at(manifest.next_seq),
            metadata,
            fts,
            vector_spaces: vs_map,
            relationships,
            chunk_store,
            relation_store,
            tombstones: std::collections::HashSet::new(),
            next_id,
            filter_index,
        };
        let mut collections = self.collections.write().await;
        collections.insert(collection_name.to_string(), loaded);
        Ok(live_count)
    }

    /// Get facet counts for a collection.
    pub async fn get_facets(
        &self,
        collection_name: &str,
        query: &str,
        fields: &[String],
    ) -> Result<
        (HashMap<String, HashMap<String, u64>>, u64),
        Box<dyn std::error::Error + Send + Sync>,
    > {
        if self.role == NodeRole::Writer {
            return Err("this node runs in writer role and does not serve queries".into());
        }
        self.ensure_attached(collection_name).await?;
        let collections = self.collections.read().await;
        let loaded = collections
            .get(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
        loaded
            .last_used
            .store(next_lru_tick(), std::sync::atomic::Ordering::Relaxed);
        tantivy_fts::get_facets(&loaded.fts, query, fields)
    }

    /// Get all chunk texts and IDs for rebuild jobs.
    pub async fn get_all_chunk_data(
        &self,
        collection_name: &str,
    ) -> Result<(Vec<String>, Vec<u64>), Box<dyn std::error::Error + Send + Sync>> {
        let collections = self.collections.read().await;
        let loaded = collections
            .get(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;

        let mut texts = Vec::new();
        let mut ids = Vec::new();
        loaded.chunk_store.for_each(|id, chunk| {
            if !loaded.tombstones.contains(&id) {
                ids.push(id);
                texts.push(chunk.text);
            }
        })?;
        Ok((texts, ids))
    }

    /// Temporal point/range lookup for TAMS-style segments.
    ///
    /// Returns every chunk where `doc_type == "segment"`, `group_id == Some(asset)`,
    /// and the time window matches the requested query. If no time params are
    /// provided, returns all segments for the asset (enumeration mode).
    ///
    /// Time unit: all parameters and the `timerange_start_ms` / `timerange_end_ms`
    /// metadata fields are in integer milliseconds. Results are sorted ascending
    /// by `timerange_start_ms` for stable ordering.
    pub async fn segments_at(
        &self,
        collection_name: &str,
        asset: &str,
        time_ms: Option<f64>,
        time_start_ms: Option<f64>,
        time_end_ms: Option<f64>,
    ) -> Result<Vec<DocumentChunk>, Box<dyn std::error::Error + Send + Sync>> {
        let collections = self.collections.read().await;
        let loaded = collections
            .get(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;

        let mut collected: Vec<DocumentChunk> = Vec::new();
        loaded.chunk_store.for_each(|id, c| {
            if !loaded.tombstones.contains(&id)
                && c.doc_type == "segment"
                && c.group_id.as_deref() == Some(asset)
                && segment_in_time_window(&c, time_ms, time_start_ms, time_end_ms)
            {
                collected.push(c);
            }
        })?;
        let mut results: Vec<DocumentChunk> = collected.into_iter().collect();

        // Sort ascending by timerange_start_ms. Segments missing the metadata
        // sort to the end (f64::INFINITY) instead of position 0, so callers
        // don't see malformed data masquerading as the earliest segment.
        // `total_cmp` is NaN-safe and deterministic (NaN sorts after Infinity).
        results.sort_by(|a, b| {
            let ta = a
                .metadata
                .get("timerange_start_ms")
                .and_then(MetadataValue::as_f64)
                .unwrap_or(f64::INFINITY);
            let tb = b
                .metadata
                .get("timerange_start_ms")
                .and_then(MetadataValue::as_f64)
                .unwrap_or(f64::INFINITY);
            ta.total_cmp(&tb)
        });

        Ok(results)
    }
}

/// Whether a segment chunk's [timerange_start_ms, timerange_end_ms] window
/// matches the requested time query. All values are in integer milliseconds.
///
/// - If no time params are provided, returns true (enumeration mode).
/// - If `time_ms` is set, returns true when
///   `timerange_start_ms <= time_ms <= timerange_end_ms`.
/// - If `time_start_ms` and/or `time_end_ms` are set, returns true when the
///   segment's window overlaps the query range. Missing bounds default to
///   ±infinity.
/// - Segments missing `timerange_start_ms` or `timerange_end_ms` are excluded
///   when any time filter is set.
/// - Instants (zero-duration events) are stored as segments where
///   `timerange_start_ms == timerange_end_ms`. A point query at that exact
///   millisecond matches the instant; range queries that overlap that
///   millisecond also match.
pub(crate) fn segment_in_time_window(
    chunk: &DocumentChunk,
    time_ms: Option<f64>,
    time_start_ms: Option<f64>,
    time_end_ms: Option<f64>,
) -> bool {
    let no_filter = time_ms.is_none() && time_start_ms.is_none() && time_end_ms.is_none();
    if no_filter {
        return true;
    }
    let ts = chunk
        .metadata
        .get("timerange_start_ms")
        .and_then(MetadataValue::as_f64);
    let te = chunk
        .metadata
        .get("timerange_end_ms")
        .and_then(MetadataValue::as_f64);
    let (s, e) = match (ts, te) {
        (Some(s), Some(e)) => (s, e),
        _ => return false,
    };
    if let Some(t) = time_ms {
        return s <= t && t <= e;
    }
    let lo = time_start_ms.unwrap_or(f64::NEG_INFINITY);
    let hi = time_end_ms.unwrap_or(f64::INFINITY);
    s <= hi && e >= lo
}

#[cfg(test)]
mod segments_at_tests {
    use super::*;

    fn make_segment(group_id: &str, ts_ms: f64, te_ms: f64) -> DocumentChunk {
        let mut metadata = HashMap::new();
        metadata.insert(
            "timerange_start_ms".to_string(),
            MetadataValue::Float(ts_ms),
        );
        metadata.insert("timerange_end_ms".to_string(), MetadataValue::Float(te_ms));
        DocumentChunk {
            id: 1,
            collection: "test".to_string(),
            file_id: "f1".to_string(),
            chunk_index: 0,
            page: None,
            text: String::new(),
            metadata,
            doc_type: "segment".to_string(),
            parent_id: None,
            group_id: Some(group_id.to_string()),
            embeddings: HashMap::new(),
            embedding: None,
        }
    }

    /// Make a zero-duration "instant" segment, the convention for sidecar
    /// events that have a single timestamp (e.g. standout_timestamps).
    fn make_instant(group_id: &str, t_ms: f64) -> DocumentChunk {
        make_segment(group_id, t_ms, t_ms)
    }

    #[test]
    fn point_inside_window() {
        let c = make_segment("a", 100.0, 200.0);
        assert!(segment_in_time_window(&c, Some(150.0), None, None));
    }

    #[test]
    fn point_outside_window() {
        let c = make_segment("a", 100.0, 200.0);
        assert!(!segment_in_time_window(&c, Some(250.0), None, None));
    }

    #[test]
    fn point_boundaries_inclusive() {
        let c = make_segment("a", 100.0, 200.0);
        assert!(segment_in_time_window(&c, Some(100.0), None, None));
        assert!(segment_in_time_window(&c, Some(200.0), None, None));
    }

    #[test]
    fn range_overlap_matches() {
        let c = make_segment("a", 100.0, 200.0);
        assert!(segment_in_time_window(&c, None, Some(180.0), Some(300.0)));
    }

    #[test]
    fn range_no_overlap() {
        let c = make_segment("a", 100.0, 200.0);
        assert!(!segment_in_time_window(&c, None, Some(250.0), Some(400.0)));
    }

    #[test]
    fn range_open_lower_bound() {
        let c = make_segment("a", 100.0, 200.0);
        assert!(segment_in_time_window(&c, None, None, Some(150.0)));
        assert!(!segment_in_time_window(&c, None, None, Some(50.0)));
    }

    #[test]
    fn range_open_upper_bound() {
        let c = make_segment("a", 100.0, 200.0);
        assert!(segment_in_time_window(&c, None, Some(150.0), None));
        assert!(!segment_in_time_window(&c, None, Some(300.0), None));
    }

    #[test]
    fn missing_metadata_with_filter_excludes() {
        let c = DocumentChunk {
            id: 2,
            collection: "test".to_string(),
            file_id: "f2".to_string(),
            chunk_index: 0,
            page: None,
            text: String::new(),
            metadata: HashMap::new(),
            doc_type: "segment".to_string(),
            parent_id: None,
            group_id: Some("a".to_string()),
            embeddings: HashMap::new(),
            embedding: None,
        };
        assert!(!segment_in_time_window(&c, Some(100.0), None, None));
        assert!(!segment_in_time_window(&c, None, Some(0.0), Some(1000.0)));
    }

    #[test]
    fn no_filter_matches_all() {
        let with_meta = make_segment("a", 100.0, 200.0);
        assert!(segment_in_time_window(&with_meta, None, None, None));

        let without_meta = DocumentChunk {
            id: 3,
            collection: "test".to_string(),
            file_id: "f3".to_string(),
            chunk_index: 0,
            page: None,
            text: String::new(),
            metadata: HashMap::new(),
            doc_type: "segment".to_string(),
            parent_id: None,
            group_id: Some("a".to_string()),
            embeddings: HashMap::new(),
            embedding: None,
        };
        assert!(segment_in_time_window(&without_meta, None, None, None));
    }

    // When both `time_ms` and `time_start_ms`/`time_end_ms` are provided,
    // `time_ms` wins. Documented in the segments.rs handler comment; this
    // test asserts it.
    #[test]
    fn point_lookup_takes_precedence_over_range() {
        let c = make_segment("a", 100.0, 200.0);
        // Point=150 is inside [100, 200], but the range [300, 400] is outside.
        // If `time_ms` correctly takes precedence, this must return true.
        assert!(segment_in_time_window(
            &c,
            Some(150.0),
            Some(300.0),
            Some(400.0)
        ));
        // Point=250 is outside, but the range [100, 300] would match.
        // If `time_ms` correctly takes precedence, this must return false.
        assert!(!segment_in_time_window(
            &c,
            Some(250.0),
            Some(100.0),
            Some(300.0)
        ));
    }

    // Instants (zero-duration events like a standout_timestamp) match a
    // point query at their exact timestamp and any range that overlaps it.
    // Critical for ingesting sidecar fields like
    // `gemini.response.standout_timestamps[]` which only carry a single ms.
    #[test]
    fn instant_matches_exact_point_query() {
        let c = make_instant("a", 5200.0);
        assert!(segment_in_time_window(&c, Some(5200.0), None, None));
        assert!(!segment_in_time_window(&c, Some(5199.0), None, None));
        assert!(!segment_in_time_window(&c, Some(5201.0), None, None));
    }

    #[test]
    fn instant_matches_overlapping_range_query() {
        let c = make_instant("a", 5200.0);
        assert!(segment_in_time_window(&c, None, Some(5000.0), Some(6000.0)));
        assert!(segment_in_time_window(&c, None, Some(5200.0), Some(5200.0)));
        assert!(!segment_in_time_window(
            &c,
            None,
            Some(5201.0),
            Some(6000.0)
        ));
    }
}

/// Build a roaring-bitmap FilterIndex over a chunk map. Synthesizes a
/// `doc_type` metadata entry from the struct field so filter expressions
/// can target it without requiring callers to duplicate `doc_type` into
/// `chunk.metadata`. Matches the semantics of `filter::matches_filters`.
///
/// Called on collection load (over rehydrated chunks) and after every
/// ingest batch (alongside FTS/HNSW rebuild). The index lives in-memory
/// only for now; persistence lands when chunk metadata migrates off redb.
/// Uncompacted-fragment count above which a cloud collection is auto-compacted.
/// Keeps the WAL bounded and reclaims tombstoned data without operator action.
pub(crate) const AUTO_COMPACT_FRAGMENT_THRESHOLD: usize = 32;

/// Storage-only compaction (no local manager state touched): materialize the
/// full live set from S3, write it as one new segment, and CAS-rewrite the
/// manifest to reference only it. Runs the CAS-retry loop so it converges
/// against concurrent appends. Safe to spawn detached — compaction never mutates
/// the local indexes (they already hold the data).
pub(crate) async fn compact_storage(
    storage: &dyn Storage,
    ns: &str,
) -> Result<u64, crate::storage::StorageError> {
    /// Segments tolerated before a full merge. Tail folds are O(batch); only
    /// the merge is O(live set), and it runs 1/K as often.
    const MERGE_SEGMENTS: usize = 8;
    const MAX_RETRIES: u32 = 10;

    // Phase 1: fold the WAL tail into an APPENDED segment (bounded work).
    for _ in 0..MAX_RETRIES {
        let (manifest, version) = crate::storage::lsm::read_manifest(storage, ns).await?;
        let tail: Vec<_> = manifest.uncompacted().cloned().collect();
        if tail.is_empty() {
            break;
        }
        let folded_through = tail.iter().map(|f| f.seq).max().unwrap();
        let frags = crate::storage::lsm::read_uncompacted_fragments(storage, ns, &manifest).await?;
        let segment = cloud::fold_tail(&frags)?;
        let records = segment.chunks.len() as u64;
        let bytes = cloud::encode_segment_v2(&segment)?;
        match crate::storage::lsm::append_segment(
            storage,
            ns,
            &version,
            &manifest,
            bytes::Bytes::from(bytes),
            records,
            folded_through,
        )
        .await
        {
            Ok(()) => {
                crate::metrics::inc(&crate::metrics::COMPACTIONS_TOTAL);
                tracing::info!(
                    "Compacted '{}': folded WAL tail through seq {} ({} live records)",
                    ns,
                    folded_through,
                    records
                );
                break;
            }
            Err(crate::storage::StorageError::VersionConflict { .. }) => continue,
            Err(e) => return Err(e),
        }
    }

    // Phase 2: merge segments when they pile up (the only O(live-set) step).
    for _ in 0..MAX_RETRIES {
        let (manifest, version) = crate::storage::lsm::read_manifest(storage, ns).await?;
        if manifest.segments.len() <= MERGE_SEGMENTS {
            return Ok(manifest.segments.iter().map(|s| s.records).sum());
        }
        let materialized = cloud::materialize(storage, ns, &manifest).await?;
        let chunks: Vec<DocumentChunk> = materialized.chunks.values().cloned().collect();
        let relations: Vec<ChunkRelation> = materialized.relations.values().cloned().collect();
        let records = chunks.len() as u64;
        let segment_bytes = cloud::encode_segment(&chunks, &relations, materialized.max_id)?;
        match crate::storage::lsm::replace_with_single_segment(
            storage,
            ns,
            &version,
            &manifest,
            bytes::Bytes::from(segment_bytes),
            records,
        )
        .await
        {
            Ok(()) => {
                tracing::info!("Merged '{}' segments: {} live records", ns, records);
                return Ok(records);
            }
            Err(crate::storage::StorageError::VersionConflict { .. }) => continue,
            Err(e) => return Err(e),
        }
    }
    Err(crate::storage::StorageError::Io(
        "compaction failed after max CAS retries (persistent contention)".into(),
    ))
}

/// Spawn a detached background compaction if the cloud collection's uncompacted
/// fragment count is over the threshold. Best-effort: logs and moves on. Called
/// after cloud ingest/delete so deletes actually reclaim space over time.
fn maybe_auto_compact(
    storage: Arc<dyn Storage>,
    ns: String,
    inflight: Arc<std::sync::Mutex<std::collections::HashSet<String>>>,
) {
    // Single-flight: if a compaction for this ns is already running, skip. This
    // stops concurrent triggers from each writing (and, on CAS loss, leaking) a
    // full segment.
    {
        let mut set = inflight.lock().unwrap_or_else(|e| e.into_inner());
        if !set.insert(ns.clone()) {
            return; // already compacting this ns
        }
    }
    tokio::spawn(async move {
        let result = async {
            let (manifest, _) = crate::storage::lsm::read_manifest(storage.as_ref(), &ns).await?;
            let uncompacted = manifest.uncompacted().count();
            if uncompacted >= AUTO_COMPACT_FRAGMENT_THRESHOLD {
                tracing::info!("Auto-compacting '{}' ({} fragments)", ns, uncompacted);
                compact_storage(storage.as_ref(), &ns).await?;
            }
            Ok::<(), crate::storage::StorageError>(())
        }
        .await;
        if let Err(e) = result {
            tracing::warn!("auto-compaction of '{}' failed: {}", ns, e);
        }
        // Clear the in-flight flag so a later trigger can run.
        inflight
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&ns);
    });
}

/// Build the filter index from the chunk map, EXCLUDING tombstoned ids — so
/// `eligible`/selectivity agree with what search may actually return, on every
/// load path (local load, ingest rebuild, cloud recovery). Freshly-deleted ids
/// are additionally masked post-retrieval until the next rebuild.
/// Process-monotonic LRU tick (no wall clock — avoids Date-based flakiness).
fn next_lru_tick() -> u64 {
    static TICK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    TICK.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// The metadata view the filter index sees: chunk metadata plus the mirrored
/// doc_type field (the filter language treats it as metadata).
fn filter_meta(chunk: &DocumentChunk) -> HashMap<String, MetadataValue> {
    let mut m = chunk.metadata.clone();
    m.insert(
        "doc_type".to_string(),
        MetadataValue::String(chunk.doc_type.clone()),
    );
    m
}

pub(crate) fn build_filter_index_from_chunks(
    chunks: &HashMap<u64, DocumentChunk>,
    tombstones: &std::collections::HashSet<u64>,
) -> FilterIndex {
    let mut idx = FilterIndex::new();
    for (&chunk_id, chunk) in chunks {
        if tombstones.contains(&chunk_id) {
            continue;
        }
        let mut effective = chunk.metadata.clone();
        // doc_type is a struct field, not a metadata key, but the filter
        // language treats it as one. Mirror it here so the bitmap covers it.
        effective.insert(
            "doc_type".to_string(),
            MetadataValue::String(chunk.doc_type.clone()),
        );
        // chunk_id is the full u64; the treemap-backed FilterIndex indexes the
        // whole id space, so no chunk is dropped regardless of id magnitude.
        idx.insert(chunk_id, &effective);
    }
    idx.finalize();
    idx
}

/// Build a deduplicated cache of parent chunk metadata for a set of candidate
/// chunk ids. Used by `search()` to enrich segment hits with their parent's
/// top-level metadata without paying for repeated lookups when multiple
/// segments share the same parent.
///
/// N segments pointing at the same parent trigger exactly one HashMap lookup.
/// Candidates that are not segments, or are segments without a `parent_id`,
/// contribute nothing to the cache.
///
/// Orphan parents (segment has a `parent_id` but the parent chunk is not in
/// `chunks`) are NOT inserted into the cache. This means `parent_metadata_for`
/// returns `None` for them, which lets callers distinguish "no parent at all"
/// from "parent exists with empty metadata."
pub(crate) fn build_parent_metadata_cache(
    candidate_chunk_ids: &[u64],
    chunks: &ChunkCache,
) -> HashMap<u64, HashMap<String, MetadataValue>> {
    let mut cache: HashMap<u64, HashMap<String, MetadataValue>> = HashMap::new();
    for cid in candidate_chunk_ids {
        let Ok(Some(chunk)) = chunks.get(*cid) else {
            continue;
        };
        if chunk.doc_type != "segment" {
            continue;
        }
        let Some(pid) = chunk.parent_id else {
            continue;
        };
        if cache.contains_key(&pid) {
            continue;
        }
        // Only cache parents that actually exist. Missing parents stay out
        // of the cache so `parent_metadata_for` returns None for them.
        if let Ok(Some(parent)) = chunks.get(pid) {
            cache.insert(pid, parent.metadata.clone());
        }
    }
    cache
}

/// Look up parent metadata for a given chunk from a pre-built cache.
///
/// Returns `None` when the chunk is not a segment or has no `parent_id`.
/// Returns `Some(metadata)` (possibly empty) when the chunk is a segment
/// whose `parent_id` was included in the cache.
pub(crate) fn parent_metadata_for(
    chunk: &DocumentChunk,
    cache: &HashMap<u64, HashMap<String, MetadataValue>>,
) -> Option<HashMap<String, MetadataValue>> {
    if chunk.doc_type != "segment" {
        return None;
    }
    chunk.parent_id.and_then(|pid| cache.get(&pid).cloned())
}

#[cfg(test)]
mod parent_metadata_tests {
    use super::*;

    fn segment(id: u64, parent_id: Option<u64>) -> DocumentChunk {
        DocumentChunk {
            id,
            collection: "test".to_string(),
            file_id: format!("f{}", id),
            chunk_index: 0,
            page: None,
            text: String::new(),
            metadata: HashMap::new(),
            doc_type: "segment".to_string(),
            parent_id,
            group_id: None,
            embeddings: HashMap::new(),
            embedding: None,
        }
    }

    fn source_with_meta(id: u64, key: &str, val: &str) -> DocumentChunk {
        let mut metadata = HashMap::new();
        metadata.insert(key.to_string(), MetadataValue::String(val.to_string()));
        DocumentChunk {
            id,
            collection: "test".to_string(),
            file_id: format!("f{}", id),
            chunk_index: 0,
            page: None,
            text: String::new(),
            metadata,
            doc_type: "source".to_string(),
            parent_id: None,
            group_id: None,
            embeddings: HashMap::new(),
            embedding: None,
        }
    }

    fn into_map(chunks: Vec<DocumentChunk>) -> ChunkCache {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "compass_pmc_{}_{}.redb",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_file(&p);
        let cache = ChunkCache::new(ChunkStore::open(&p).unwrap());
        let batch: Vec<(u64, DocumentChunk)> = chunks.into_iter().map(|c| (c.id, c)).collect();
        cache.insert_batch(&batch).unwrap();
        cache
    }

    #[test]
    fn segment_with_parent_gets_metadata() {
        let chunks = into_map(vec![
            source_with_meta(1, "title", "Keynote"),
            segment(2, Some(1)),
        ]);
        let cache = build_parent_metadata_cache(&[2], &chunks);
        let meta = parent_metadata_for(&chunks.get(2).unwrap().unwrap(), &cache);
        assert_eq!(
            meta.unwrap().get("title"),
            Some(&MetadataValue::String("Keynote".to_string()))
        );
    }

    #[test]
    fn source_hit_gets_none() {
        let chunks = into_map(vec![source_with_meta(1, "title", "Keynote")]);
        let cache = build_parent_metadata_cache(&[1], &chunks);
        let meta = parent_metadata_for(&chunks.get(1).unwrap().unwrap(), &cache);
        assert!(meta.is_none());
    }

    #[test]
    fn segment_without_parent_gets_none() {
        let chunks = into_map(vec![segment(2, None)]);
        let cache = build_parent_metadata_cache(&[2], &chunks);
        let meta = parent_metadata_for(&chunks.get(2).unwrap().unwrap(), &cache);
        assert!(meta.is_none());
    }

    #[test]
    fn dedup_one_lookup_per_unique_parent() {
        // Three segments, all pointing at parent_id=10. The cache should
        // contain exactly one entry (for pid=10), proving the dedup.
        let chunks = into_map(vec![
            source_with_meta(10, "source_id", "src-001"),
            segment(11, Some(10)),
            segment(12, Some(10)),
            segment(13, Some(10)),
        ]);
        let cache = build_parent_metadata_cache(&[11, 12, 13], &chunks);
        assert_eq!(
            cache.len(),
            1,
            "expected one cache entry for the shared parent"
        );
        assert!(cache.contains_key(&10));
        // All three segments resolve to the same parent metadata.
        for cid in [11, 12, 13] {
            let meta = parent_metadata_for(&chunks.get(cid).unwrap().unwrap(), &cache);
            assert_eq!(
                meta.unwrap().get("source_id"),
                Some(&MetadataValue::String("src-001".to_string()))
            );
        }
    }

    #[test]
    fn orphan_segment_yields_none() {
        // parent_id=99 not in chunks. The cache must NOT contain pid=99,
        // and parent_metadata_for must return None. This distinguishes
        // "parent exists with empty metadata" (Some({})) from "parent
        // doesn't exist" (None).
        let chunks = into_map(vec![segment(5, Some(99))]);
        let cache = build_parent_metadata_cache(&[5], &chunks);
        assert!(!cache.contains_key(&99), "orphan parent must not be cached");
        let meta = parent_metadata_for(&chunks.get(5).unwrap().unwrap(), &cache);
        assert!(meta.is_none(), "orphan segment must yield None");
    }

    #[test]
    fn parent_exists_with_empty_metadata_yields_some_empty() {
        // Parent chunk exists but has no metadata fields. Must return Some({})
        // so callers can distinguish from the orphan case (None).
        let parent_no_meta = DocumentChunk {
            id: 20,
            collection: "test".to_string(),
            file_id: "f20".to_string(),
            chunk_index: 0,
            page: None,
            text: String::new(),
            metadata: HashMap::new(),
            doc_type: "source".to_string(),
            parent_id: None,
            group_id: None,
            embeddings: HashMap::new(),
            embedding: None,
        };
        let chunks = into_map(vec![parent_no_meta, segment(21, Some(20))]);
        let cache = build_parent_metadata_cache(&[21], &chunks);
        let meta = parent_metadata_for(&chunks.get(21).unwrap().unwrap(), &cache);
        assert!(meta.is_some());
        assert!(meta.unwrap().is_empty());
    }

    #[test]
    fn parent_metadata_for_cache_miss_returns_none() {
        // Defensive: if the cache was built with a different set of IDs than
        // the one we're looking up, the function must return None (not panic,
        // not return stale data). Catches regressions where someone "optimizes"
        // parent_metadata_for to assume the cache is always complete.
        let parent = source_with_meta(1, "title", "Keynote");
        let seg = segment(2, Some(1));
        let chunks = into_map(vec![parent, seg]);
        // Build cache against an empty candidate list, then look up segment 2.
        let cache = build_parent_metadata_cache(&[], &chunks);
        assert!(cache.is_empty());
        let meta = parent_metadata_for(&chunks.get(2).unwrap().unwrap(), &cache);
        assert!(meta.is_none());
    }
}

#[cfg(test)]
mod persistence_tests {
    //! End-to-end durability test. Builds a CollectionManager in a temp dir,
    //! ingests chunks, drops the manager (closing the chunk store), creates a
    //! new manager pointing at the same dir, and asserts the chunks come back.
    //!
    //! This is the test that proves Compass survives process restarts. Without
    //! the disk-backed ChunkStore wiring, this test would fail because
    //! `loaded.chunks` would be empty after the manager restart.

    use super::*;
    use crate::embed::EmbedState;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn unique_data_dir() -> std::path::PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "compass-persist-test-{}-{}-{}",
            std::process::id(),
            nanos,
            N.fetch_add(1, Ordering::SeqCst)
        ))
    }

    fn empty_embed_state() -> EmbedState {
        // No embedding models loaded. Safe for the persistence test because
        // we provide chunks without text-only embedding requirements. Any
        // call to embed_query returns Err and the ingest path tolerates that.
        EmbedState {
            bge: None,
            distilled: None,
        }
    }

    fn make_ingest_chunk(file_id: &str, text: &str) -> IngestChunk {
        IngestChunk {
            client_id: None,
            file_id: file_id.to_string(),
            chunk_index: 0,
            page: None,
            text: text.to_string(),
            metadata: HashMap::new(),
            doc_type: "chunk".to_string(),
            parent_id: None,
            parent_ref: None,
            group_id: None,
            embeddings: HashMap::new(),
            embedding: None,
        }
    }

    #[tokio::test]
    async fn chunks_persist_across_manager_restart() {
        let data_dir = unique_data_dir();
        std::fs::create_dir_all(&data_dir).unwrap();
        let embed = empty_embed_state();

        // First manager lifetime: create collection, ingest three chunks,
        // then drop the manager to close all file handles (including redb).
        {
            let manager = CollectionManager::new(&data_dir).await.unwrap();
            manager
                .create_collection("persist-test", None, None, None)
                .await
                .unwrap();
            let to_ingest = vec![
                make_ingest_chunk("f1", "first chunk"),
                make_ingest_chunk("f2", "second chunk"),
                make_ingest_chunk("f3", "third chunk"),
            ];
            let (ingested, _, _) = manager
                .ingest("persist-test", to_ingest, &embed)
                .await
                .unwrap();
            assert_eq!(ingested, 3, "ingest call reports 3 chunks written");
            // manager dropped here
        }

        // Second manager: same data dir, must rehydrate chunks from disk.
        let manager2 = CollectionManager::new(&data_dir).await.unwrap();
        let (texts, ids) = manager2.get_all_chunk_data("persist-test").await.unwrap();

        assert_eq!(
            ids.len(),
            3,
            "expected 3 chunks rehydrated from disk after manager restart, got {}",
            ids.len()
        );
        let mut sorted_texts = texts.clone();
        sorted_texts.sort();
        assert_eq!(
            sorted_texts,
            vec![
                "first chunk".to_string(),
                "second chunk".to_string(),
                "third chunk".to_string(),
            ],
            "chunk texts should match what was ingested before the restart"
        );

        // Cleanup
        let _ = std::fs::remove_dir_all(&data_dir);
    }

    #[tokio::test]
    async fn next_id_advances_correctly_after_rehydration() {
        // After rehydration, next_id should be max(seen) + 1 so new ingests
        // don't collide with persisted IDs. Verify by ingesting again after
        // restart and checking the new chunk got a fresh ID.
        let data_dir = unique_data_dir();
        std::fs::create_dir_all(&data_dir).unwrap();
        let embed = empty_embed_state();

        // Round 1: ingest two chunks (IDs 0, 1)
        {
            let manager = CollectionManager::new(&data_dir).await.unwrap();
            manager
                .create_collection("next-id-test", None, None, None)
                .await
                .unwrap();
            manager
                .ingest(
                    "next-id-test",
                    vec![
                        make_ingest_chunk("f0", "round-one-a"),
                        make_ingest_chunk("f1", "round-one-b"),
                    ],
                    &embed,
                )
                .await
                .unwrap();
        }

        // Round 2: restart and ingest one more chunk. The new chunk's ID
        // should be 2, not 0.
        let manager2 = CollectionManager::new(&data_dir).await.unwrap();
        manager2
            .ingest(
                "next-id-test",
                vec![make_ingest_chunk("f2", "round-two")],
                &embed,
            )
            .await
            .unwrap();
        let (_, ids) = manager2.get_all_chunk_data("next-id-test").await.unwrap();
        let mut sorted_ids = ids.clone();
        sorted_ids.sort();
        assert_eq!(
            sorted_ids,
            vec![0, 1, 2],
            "next_id must advance past max persisted id, got ids: {:?}",
            sorted_ids
        );

        let _ = std::fs::remove_dir_all(&data_dir);
    }
}

#[cfg(test)]
mod validate_name_segment_tests {
    use super::validate_name_segment;

    #[test]
    fn accepts_simple_names() {
        assert!(validate_name_segment("my-collection", "Collection").is_ok());
        assert!(validate_name_segment("harrier", "Vector space").is_ok());
        assert!(validate_name_segment("qwen3-vl", "Vector space").is_ok());
        assert!(validate_name_segment("a", "Collection").is_ok());
    }

    #[test]
    fn rejects_empty() {
        let err = validate_name_segment("", "Vector space").expect_err("empty name should error");
        assert!(err.to_string().contains("Vector space"));
    }

    #[test]
    fn rejects_path_traversal() {
        // The whole reason this validator exists: a vector space name flows
        // into on-disk paths like `<vectors_dir>/<name>.bin`. A `../` segment
        // must never be accepted.
        for bad in [
            "../etc/passwd",
            "..",
            "foo/bar",
            "foo\\bar",
            "/abs",
            "name with space",
            "name.with.dot",
            "name_with_underscore", // hyphens only, no underscores
            "tab\there",
            "name\nwith\nnewline",
        ] {
            assert!(
                validate_name_segment(bad, "Vector space").is_err(),
                "validator must reject {bad:?}"
            );
        }
    }

    #[test]
    fn rejects_unicode_lookalikes() {
        // Cyrillic 'а' (U+0430) looks like 'a' but is not ASCII.
        assert!(validate_name_segment("\u{0430}bc", "Collection").is_err());
        assert!(validate_name_segment("emoji-🚀", "Collection").is_err());
    }
}

#[cfg(test)]
mod filter_aware_search_tests {
    //! End-to-end test of filter-aware /search + /explain (a follow-up).
    //!
    //! Builds a real CollectionManager, ingests chunks with caller-provided
    //! embeddings (skipping the in-process BGE model), runs filtered hybrid
    //! search, and asserts:
    //!   1. All hits respect the filter (filter-aware path, not post-filter).
    //!   2. The /explain field is populated when requested and absent when not.
    //!   3. Filter selectivity is reported correctly.

    use super::*;
    use crate::embed::EmbedState;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn unique_data_dir() -> std::path::PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "compass-filter-search-test-{}-{}-{}",
            std::process::id(),
            nanos,
            N.fetch_add(1, Ordering::SeqCst)
        ))
    }

    fn embed_state() -> EmbedState {
        EmbedState {
            bge: None,
            distilled: None,
        }
    }

    /// Deterministic 4-dim unit vector seeded from an integer.
    fn pseudo_vec(seed: u64) -> Vec<f32> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut v = Vec::with_capacity(4);
        for _ in 0..4 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let f = (state >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0;
            v.push(f);
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    fn ingest_with(org: &str, idx: u32) -> IngestChunk {
        let mut metadata = HashMap::new();
        metadata.insert("org_id".to_string(), MetadataValue::String(org.to_string()));
        metadata.insert(
            "created_at".to_string(),
            MetadataValue::Int(1_700_000_000 + idx as i64),
        );
        let mut embeddings = HashMap::new();
        embeddings.insert("default".to_string(), pseudo_vec(idx as u64 + 1));
        IngestChunk {
            client_id: None,
            file_id: format!("f{idx}"),
            chunk_index: 0,
            page: None,
            text: format!("chunk-{idx}"),
            metadata,
            doc_type: "chunk".to_string(),
            parent_id: None,
            parent_ref: None,
            group_id: None,
            embeddings,
            embedding: None,
        }
    }

    #[tokio::test]
    async fn filter_aware_search_returns_only_matching_chunks() {
        let data_dir = unique_data_dir();
        std::fs::create_dir_all(&data_dir).unwrap();
        let embed = embed_state();
        let manager = CollectionManager::new(&data_dir).await.unwrap();
        manager
            .create_collection("filter-search", None, Some(4), None)
            .await
            .unwrap();

        // 100 chunks: 20 from "acme", 80 from "widgets".
        let mut chunks = Vec::new();
        for i in 0..100u32 {
            let org = if i % 5 == 0 { "acme" } else { "widgets" };
            chunks.push(ingest_with(org, i));
        }
        manager
            .ingest("filter-search", chunks, &embed)
            .await
            .unwrap();

        // Search with filter org_id=acme. ALL hits must come from acme.
        let mut filters = HashMap::new();
        filters.insert(
            "org_id".to_string(),
            FilterValue::Exact(MetadataValue::String("acme".into())),
        );
        let req = SearchRequest {
            query: "chunk".to_string(),
            mode: "hybrid".to_string(),
            vector_space: None,
            top_k: 10,
            query_vector: Some(pseudo_vec(99_999)),
            filters,
            score_weights: None,
            recency: None,
            recency_preset: None,
            recency_field: None,
            boosts: Vec::new(),
            relationship_boost: None,
            explain: true,
            include_relations: false,
            relation_types: None,
            relation_direction: RelationDirection::Outgoing,
            min_seq: None,
        };
        let (hits, total, _took_us, explain) =
            manager.search("filter-search", &req, &embed).await.unwrap();

        assert!(!hits.is_empty(), "search returned no hits");
        for (chunk, _, _, _, _) in &hits {
            assert_eq!(
                chunk.metadata.get("org_id"),
                Some(&MetadataValue::String("acme".into())),
                "all hits must satisfy the filter; got chunk {} with org_id {:?}",
                chunk.id,
                chunk.metadata.get("org_id")
            );
        }
        assert!(
            total <= 20,
            "no more than 20 hits possible at 20% selectivity"
        );

        // /explain should be populated.
        let explain = explain.expect("explain plan requested but not returned");
        assert_eq!(explain.filter.eligible_count, 20);
        assert_eq!(explain.filter.universe_count, 100);
        assert!((explain.filter.selectivity - 0.20).abs() < 1e-9);
        assert!(
            matches!(explain.ann.engine.as_str(), "hnsw" | "brute_force"),
            "ann engine reported as {}",
            explain.ann.engine
        );

        let _ = std::fs::remove_dir_all(&data_dir);
    }

    #[tokio::test]
    async fn explain_absent_when_not_requested() {
        let data_dir = unique_data_dir();
        std::fs::create_dir_all(&data_dir).unwrap();
        let embed = embed_state();
        let manager = CollectionManager::new(&data_dir).await.unwrap();
        manager
            .create_collection("no-explain", None, Some(4), None)
            .await
            .unwrap();
        manager
            .ingest("no-explain", vec![ingest_with("acme", 0)], &embed)
            .await
            .unwrap();

        let req = SearchRequest {
            query: "chunk".to_string(),
            mode: "semantic".to_string(),
            vector_space: None,
            top_k: 1,
            query_vector: Some(pseudo_vec(42)),
            filters: HashMap::new(),
            score_weights: None,
            recency: None,
            recency_preset: None,
            recency_field: None,
            boosts: Vec::new(),
            relationship_boost: None,
            explain: false,
            include_relations: false,
            relation_types: None,
            relation_direction: RelationDirection::Outgoing,
            min_seq: None,
        };
        let (_hits, _total, _took, explain) =
            manager.search("no-explain", &req, &embed).await.unwrap();
        assert!(explain.is_none(), "explain must be None when not requested");

        let _ = std::fs::remove_dir_all(&data_dir);
    }

    #[tokio::test]
    async fn relations_crud_and_search_enrichment() {
        let data_dir = unique_data_dir();
        std::fs::create_dir_all(&data_dir).unwrap();
        let embed = embed_state();
        let manager = CollectionManager::new(&data_dir).await.unwrap();
        manager
            .create_collection("rel-search", None, Some(4), None)
            .await
            .unwrap();

        // Ingest 5 chunks -> ids 0..5 in order.
        let chunks: Vec<_> = (0..5u32).map(|i| ingest_with("acme", i)).collect();
        manager.ingest("rel-search", chunks, &embed).await.unwrap();

        // Create relations: 0 --cites--> 1, 0 --cites--> 2, 3 --supersedes--> 0.
        let created = manager
            .create_relations(
                "rel-search",
                vec![
                    CreateRelation {
                        source_chunk_id: 0,
                        target_chunk_id: 1,
                        target_document_id: None,
                        relation_type: "cites".into(),
                        metadata: HashMap::new(),
                    },
                    CreateRelation {
                        source_chunk_id: 0,
                        target_chunk_id: 2,
                        target_document_id: None,
                        relation_type: "cites".into(),
                        metadata: HashMap::new(),
                    },
                    CreateRelation {
                        source_chunk_id: 3,
                        target_chunk_id: 0,
                        target_document_id: None,
                        relation_type: "supersedes".into(),
                        metadata: HashMap::new(),
                    },
                ],
            )
            .await
            .unwrap();
        assert_eq!(created.len(), 3);
        assert!(created.iter().all(|r| !r.relation_id.is_empty()));
        assert!(created.iter().all(|r| r.target_status == "found"));

        // Self-relation is rejected.
        let bad = manager
            .create_relations(
                "rel-search",
                vec![CreateRelation {
                    source_chunk_id: 1,
                    target_chunk_id: 1,
                    target_document_id: None,
                    relation_type: "cites".into(),
                    metadata: HashMap::new(),
                }],
            )
            .await;
        assert!(bad.is_err());

        // Direction filters.
        let out = manager
            .get_chunk_relations("rel-search", 0, RelationDirection::Outgoing, None)
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
        let inc = manager
            .get_chunk_relations("rel-search", 0, RelationDirection::Incoming, None)
            .await
            .unwrap();
        assert_eq!(inc.len(), 1);
        assert_eq!(inc[0].relation_type, "supersedes");

        // Type filter.
        let cites = manager
            .get_chunk_relations(
                "rel-search",
                0,
                RelationDirection::Both,
                Some(&["cites".to_string()]),
            )
            .await
            .unwrap();
        assert_eq!(cites.len(), 2);

        let base_req = |include: bool| SearchRequest {
            query: "chunk".to_string(),
            mode: "semantic".to_string(),
            vector_space: None,
            top_k: 10,
            query_vector: Some(pseudo_vec(7)),
            filters: HashMap::new(),
            score_weights: None,
            recency: None,
            recency_preset: None,
            recency_field: None,
            boosts: Vec::new(),
            relationship_boost: None,
            explain: false,
            include_relations: include,
            relation_types: None,
            relation_direction: RelationDirection::Outgoing,
            min_seq: None,
        };

        // Without include_relations -> hits carry None.
        let (hits_off, _, _, _) = manager
            .search("rel-search", &base_req(false), &embed)
            .await
            .unwrap();
        assert!(hits_off.iter().all(|(_, _, _, _, rels)| rels.is_none()));

        // With include_relations -> chunk 0's hit carries its 2 outgoing cites.
        let (hits_on, _, _, _) = manager
            .search("rel-search", &base_req(true), &embed)
            .await
            .unwrap();
        let chunk0 = hits_on
            .iter()
            .find(|(c, _, _, _, _)| c.id == 0)
            .expect("chunk 0 in results");
        let rels = chunk0.4.as_ref().expect("Some(relations) when requested");
        assert_eq!(rels.len(), 2, "chunk 0 has 2 outgoing cites");
        assert!(rels.iter().all(|r| r.relation_type == "cites"));
        assert!(rels.iter().all(|r| r.target_status == "found"));

        // Delete one relation; outgoing from 0 drops to 1.
        let rid = &created[0].relation_id;
        assert!(manager.delete_relation("rel-search", rid).await.unwrap());
        let out2 = manager
            .get_chunk_relations("rel-search", 0, RelationDirection::Outgoing, None)
            .await
            .unwrap();
        assert_eq!(out2.len(), 1);

        let _ = std::fs::remove_dir_all(&data_dir);
    }

    #[tokio::test]
    async fn delete_removes_from_search_and_survives_restart() {
        let data_dir = unique_data_dir();
        std::fs::create_dir_all(&data_dir).unwrap();
        let embed = embed_state();

        {
            let manager = CollectionManager::new(&data_dir).await.unwrap();
            manager
                .create_collection("del", None, Some(4), None)
                .await
                .unwrap();
            // Ingest 10 chunks (ids 0..10), org=acme.
            let chunks: Vec<_> = (0..10u32).map(|i| ingest_with("acme", i)).collect();
            manager.ingest("del", chunks, &embed).await.unwrap();

            // Delete chunk id 3 by id.
            let (n, _) = manager.delete_chunks("del", &[3]).await.unwrap();
            assert_eq!(n, 1);
            // Re-deleting is a no-op.
            assert_eq!(manager.delete_chunks("del", &[3]).await.unwrap().0, 0);

            // A search must never return the deleted id.
            let req = SearchRequest {
                query: "chunk".to_string(),
                mode: "semantic".to_string(),
                vector_space: None,
                top_k: 20,
                query_vector: Some(pseudo_vec(4)),
                filters: HashMap::new(),
                score_weights: None,
                recency: None,
                recency_preset: None,
                recency_field: None,
                boosts: Vec::new(),
                relationship_boost: None,
                explain: false,
                include_relations: false,
                relation_types: None,
                relation_direction: RelationDirection::Outgoing,
                min_seq: None,
            };
            let (hits, _, _, _) = manager.search("del", &req, &embed).await.unwrap();
            assert!(
                hits.iter().all(|(c, _, _, _, _)| c.id != 3),
                "deleted chunk must not appear in results"
            );

            // Delete-by-filter: delete everything with file_id f5 (chunk 5).
            let mut filters = HashMap::new();
            filters.insert(
                "file_id".to_string(),
                FilterValue::Exact(MetadataValue::String("f5".into())),
            );
            // ingest_with doesn't set file_id in metadata, so use a metadata field.
            // org_id=acme matches all remaining -> delete the rest via a scan.
            let mut org_filter = HashMap::new();
            org_filter.insert(
                "org_id".to_string(),
                FilterValue::Exact(MetadataValue::String("acme".into())),
            );
            let deleted = manager.delete_by_filter("del", &org_filter).await.unwrap();
            // 10 ingested - 1 already deleted (id 3) = 9 remaining deleted now.
            assert_eq!(deleted.0, 9);
            let _ = filters;
        }

        // Restart: tombstones must persist. Reopen the manager over the same dir.
        {
            let manager = CollectionManager::new(&data_dir).await.unwrap();
            let req = SearchRequest {
                query: "chunk".to_string(),
                mode: "semantic".to_string(),
                vector_space: None,
                top_k: 20,
                query_vector: Some(pseudo_vec(4)),
                filters: HashMap::new(),
                score_weights: None,
                recency: None,
                recency_preset: None,
                recency_field: None,
                boosts: Vec::new(),
                relationship_boost: None,
                explain: false,
                include_relations: false,
                relation_types: None,
                relation_direction: RelationDirection::Outgoing,
                min_seq: None,
            };
            let (hits, _, _, _) = manager.search("del", &req, &embed).await.unwrap();
            assert!(
                hits.is_empty(),
                "all chunks deleted; none should survive restart, got {}",
                hits.len()
            );
        }

        let _ = std::fs::remove_dir_all(&data_dir);
    }

    // F6 coverage: deleting a chunk that participates in relations must prune
    // those edges (both endpoints), not leave dangling references.
    #[tokio::test]
    async fn delete_chunk_prunes_its_relations() {
        let data_dir = unique_data_dir();
        std::fs::create_dir_all(&data_dir).unwrap();
        let embed = embed_state();
        let manager = CollectionManager::new(&data_dir).await.unwrap();
        manager
            .create_collection("delrel", None, Some(4), None)
            .await
            .unwrap();
        let chunks: Vec<_> = (0..3u32).map(|i| ingest_with("acme", i)).collect();
        manager.ingest("delrel", chunks, &embed).await.unwrap();

        // 0 -> 1, 2 -> 0 (chunk 0 is both a source and a target).
        manager
            .create_relations(
                "delrel",
                vec![
                    CreateRelation {
                        source_chunk_id: 0,
                        target_chunk_id: 1,
                        target_document_id: None,
                        relation_type: "cites".into(),
                        metadata: HashMap::new(),
                    },
                    CreateRelation {
                        source_chunk_id: 2,
                        target_chunk_id: 0,
                        target_document_id: None,
                        relation_type: "cites".into(),
                        metadata: HashMap::new(),
                    },
                ],
            )
            .await
            .unwrap();

        // Delete chunk 0 — both edges (as source and as target) must be pruned.
        manager.delete_chunks("delrel", &[0]).await.unwrap();

        assert!(
            manager
                .get_chunk_relations("delrel", 0, RelationDirection::Both, None)
                .await
                .unwrap()
                .is_empty(),
            "deleted chunk's own edges gone"
        );
        // Chunk 2's outgoing edge (to deleted 0) must also be gone.
        assert!(
            manager
                .get_chunk_relations("delrel", 2, RelationDirection::Outgoing, None)
                .await
                .unwrap()
                .is_empty(),
            "edge pointing AT the deleted chunk must be pruned"
        );
        let _ = std::fs::remove_dir_all(&data_dir);
    }

    // A stale tombstone must not suppress a NEWLY-ingested chunk. Since next_id
    // is a monotonic high-water mark, re-ingest gets a fresh id that was never
    // tombstoned, so it's fully searchable.
    #[tokio::test]
    async fn delete_then_reingest_new_chunk_is_searchable() {
        let data_dir = unique_data_dir();
        std::fs::create_dir_all(&data_dir).unwrap();
        let embed = embed_state();
        let manager = CollectionManager::new(&data_dir).await.unwrap();
        manager
            .create_collection("reing", None, Some(4), None)
            .await
            .unwrap();
        manager
            .ingest("reing", vec![ingest_with("acme", 0)], &embed)
            .await
            .unwrap();
        manager.delete_chunks("reing", &[0]).await.unwrap();

        // Re-ingest: gets id 1 (next_id advanced), NOT the tombstoned id 0.
        manager
            .ingest("reing", vec![ingest_with("acme", 9)], &embed)
            .await
            .unwrap();

        let req = SearchRequest {
            query: "chunk".to_string(),
            mode: "semantic".to_string(),
            vector_space: None,
            top_k: 10,
            query_vector: Some(pseudo_vec(10)),
            filters: HashMap::new(),
            score_weights: None,
            recency: None,
            recency_preset: None,
            recency_field: None,
            boosts: Vec::new(),
            relationship_boost: None,
            explain: false,
            include_relations: false,
            relation_types: None,
            relation_direction: RelationDirection::Outgoing,
            min_seq: None,
        };
        let (hits, _, _, _) = manager.search("reing", &req, &embed).await.unwrap();
        assert_eq!(hits.len(), 1, "the re-ingested chunk must be searchable");
        assert_eq!(hits[0].0.id, 1, "re-ingest got a fresh (untombstoned) id");
        let _ = std::fs::remove_dir_all(&data_dir);
    }
}

#[cfg(all(test, feature = "object-storage"))]
mod cloud_ingest_tests {
    //! Verifies that in object-storage (cloud) mode, ingest mirrors the batch
    //! into the LSM as a WAL fragment + CAS-committed manifest — the S3-native
    //! path. Uses the in-memory object_store backend, which
    //! exercises the identical `Storage`/`ObjectStoreBackend` code an S3 bucket
    //! would, without needing real credentials.

    use super::*;
    use crate::embed::EmbedState;
    use crate::storage::object_store_backend::ObjectStoreBackend;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn unique_data_dir() -> std::path::PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "compass-cloud-ingest-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn embed_state() -> EmbedState {
        // No models needed: chunks carry precomputed embeddings.
        EmbedState {
            bge: None,
            distilled: None,
        }
    }

    fn ingest_chunk(idx: u32) -> IngestChunk {
        let mut metadata = HashMap::new();
        metadata.insert(
            "org_id".to_string(),
            MetadataValue::String("acme".to_string()),
        );
        let mut embeddings = HashMap::new();
        embeddings.insert("default".to_string(), vec![0.1, 0.2, 0.3, 0.4]);
        IngestChunk {
            client_id: None,
            file_id: format!("f{idx}"),
            chunk_index: 0,
            page: None,
            text: format!("chunk-{idx}"),
            metadata,
            doc_type: "chunk".to_string(),
            parent_id: None,
            parent_ref: None,
            group_id: None,
            embeddings,
            embedding: None,
        }
    }

    #[tokio::test]
    async fn ingest_writes_wal_fragment_and_manifest_to_object_storage() {
        let data_dir = unique_data_dir();
        std::fs::create_dir_all(&data_dir).unwrap();
        let embed = embed_state();

        // In-memory object storage backend (same code path as s3://).
        let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            std::sync::Arc::new(object_store::memory::InMemory::new()),
            "object-store:memory",
        ));
        let manager = CollectionManager::new_with_storage(&data_dir, storage.clone())
            .await
            .unwrap();
        manager
            .create_collection("cloudcoll", None, Some(4), None)
            .await
            .unwrap();

        // Ingest two batches.
        manager
            .ingest("cloudcoll", vec![ingest_chunk(0), ingest_chunk(1)], &embed)
            .await
            .unwrap();
        manager
            .ingest("cloudcoll", vec![ingest_chunk(2)], &embed)
            .await
            .unwrap();

        // The manifest exists and records two WAL fragments.
        let (manifest, version) = crate::storage::lsm::read_manifest(storage.as_ref(), "cloudcoll")
            .await
            .unwrap();
        assert!(version.is_some(), "manifest must exist in object storage");
        assert_eq!(manifest.fragments.len(), 2, "one fragment per ingest batch");
        assert_eq!(manifest.next_seq, 2);

        // The WAL fragment objects exist and decode back to the ingested chunks.
        let frags = crate::storage::lsm::read_uncompacted_fragments(
            storage.as_ref(),
            "cloudcoll",
            &manifest,
        )
        .await
        .unwrap();
        assert_eq!(frags.len(), 2);

        let batch0: Vec<DocumentChunk> = serde_json::from_slice(&frags[0].1).unwrap();
        assert_eq!(batch0.len(), 2);
        assert_eq!(batch0[0].text, "chunk-0");
        let batch1: Vec<DocumentChunk> = serde_json::from_slice(&frags[1].1).unwrap();
        assert_eq!(batch1.len(), 1);
        assert_eq!(batch1[0].text, "chunk-2");

        // Total records across fragments == total chunks ingested.
        let total: u64 = manifest.fragments.iter().map(|f| f.records).sum();
        assert_eq!(total, 3);

        let _ = std::fs::remove_dir_all(&data_dir);
    }

    #[tokio::test]
    async fn local_mode_writes_no_wal() {
        // Sanity: a local-disk manager must NOT create any WAL/manifest objects.
        let data_dir = unique_data_dir();
        std::fs::create_dir_all(&data_dir).unwrap();
        let embed = embed_state();
        let manager = CollectionManager::new(&data_dir).await.unwrap();
        manager
            .create_collection("localcoll", None, Some(4), None)
            .await
            .unwrap();
        manager
            .ingest("localcoll", vec![ingest_chunk(0)], &embed)
            .await
            .unwrap();

        // No manifest object should exist under the collection prefix.
        let manifest_path = data_dir.join("localcoll").join("manifest");
        assert!(
            !manifest_path.exists(),
            "local mode must not write an LSM manifest"
        );
        let _ = std::fs::remove_dir_all(&data_dir);
    }

    #[tokio::test]
    async fn delete_writes_tombstone_wal_fragment() {
        use crate::storage::lsm::{read_manifest, read_uncompacted_fragments, FragmentKind};

        let data_dir = unique_data_dir();
        std::fs::create_dir_all(&data_dir).unwrap();
        let embed = embed_state();
        let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            std::sync::Arc::new(object_store::memory::InMemory::new()),
            "object-store:memory",
        ));
        let manager = CollectionManager::new_with_storage(&data_dir, storage.clone())
            .await
            .unwrap();
        manager
            .create_collection("delcloud", None, Some(4), None)
            .await
            .unwrap();
        manager
            .ingest(
                "delcloud",
                vec![ingest_chunk(0), ingest_chunk(1), ingest_chunk(2)],
                &embed,
            )
            .await
            .unwrap();

        // Delete chunk 1 -> a tombstone WAL fragment lands in object storage.
        let (n, _) = manager.delete_chunks("delcloud", &[1]).await.unwrap();
        assert_eq!(n, 1);

        let (manifest, _) = read_manifest(storage.as_ref(), "delcloud").await.unwrap();
        // seq 0 = data fragment (the ingest), seq 1 = tombstone fragment.
        assert_eq!(manifest.fragments.len(), 2);
        assert_eq!(manifest.fragments[0].kind, FragmentKind::Data);
        assert_eq!(manifest.fragments[1].kind, FragmentKind::Tombstone);

        // The tombstone fragment decodes to the deleted id [1].
        let frags = read_uncompacted_fragments(storage.as_ref(), "delcloud", &manifest)
            .await
            .unwrap();
        let deleted_ids: Vec<u64> = serde_json::from_slice(&frags[1].1).unwrap();
        assert_eq!(deleted_ids, vec![1]);

        let _ = std::fs::remove_dir_all(&data_dir);
    }

    // THE structural fix: a cloud collection must survive a restart on a FRESH
    // local disk by rebuilding from S3. Ingest, delete one, then drop the manager
    // AND wipe the local data dir, then reload from the SAME object store — the
    // data (minus the deleted chunk) must come back.
    #[tokio::test]
    async fn cloud_restart_rehydrates_from_object_storage() {
        let embed = embed_state();
        // Shared object store persists across the "restart".
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());

        let data_dir_a = unique_data_dir();
        std::fs::create_dir_all(&data_dir_a).unwrap();
        {
            let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
                store.clone(),
                "object-store:memory",
            ));
            let m = CollectionManager::new_with_storage(&data_dir_a, storage)
                .await
                .unwrap();
            m.create_collection("survive", None, Some(4), None)
                .await
                .unwrap();
            m.ingest(
                "survive",
                vec![ingest_chunk(0), ingest_chunk(1), ingest_chunk(2)],
                &embed,
            )
            .await
            .unwrap();
            m.delete_chunks("survive", &[1]).await.unwrap();
        }
        // Simulate node loss: wipe the local disk entirely.
        std::fs::remove_dir_all(&data_dir_a).unwrap();

        // Restart on a BRAND-NEW empty local dir, same object store.
        let data_dir_b = unique_data_dir();
        std::fs::create_dir_all(&data_dir_b).unwrap();
        let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m2 = CollectionManager::new_with_storage(&data_dir_b, storage)
            .await
            .unwrap();

        // The collection is back, recovered from S3.
        let info = m2.get_collection("survive").await;
        assert!(info.is_some(), "collection must be recovered from S3");

        // Search finds the surviving chunks (0 and 2), not the deleted one (1).
        let req = SearchRequest {
            query: "chunk".to_string(),
            mode: "semantic".to_string(),
            vector_space: None,
            top_k: 10,
            query_vector: Some(vec![0.1, 0.2, 0.3, 0.4]),
            filters: HashMap::new(),
            score_weights: None,
            recency: None,
            recency_preset: None,
            recency_field: None,
            boosts: Vec::new(),
            relationship_boost: None,
            explain: false,
            include_relations: false,
            relation_types: None,
            relation_direction: RelationDirection::Outgoing,
            min_seq: None,
        };
        let (hits, _, _, _) = m2.search("survive", &req, &embed).await.unwrap();
        let ids: std::collections::HashSet<u64> = hits.iter().map(|(c, _, _, _, _)| c.id).collect();
        assert!(ids.contains(&0), "chunk 0 recovered");
        assert!(ids.contains(&2), "chunk 2 recovered");
        assert!(!ids.contains(&1), "deleted chunk 1 must NOT reappear");

        let _ = std::fs::remove_dir_all(&data_dir_b);
    }

    // Compaction folds segments+fragments into one segment, dropping tombstoned
    // records so they can never resurrect.
    #[tokio::test]
    async fn compaction_reclaims_tombstoned_data() {
        use crate::storage::lsm::read_manifest;
        let embed = embed_state();
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let data_dir = unique_data_dir();
        std::fs::create_dir_all(&data_dir).unwrap();
        let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m = CollectionManager::new_with_storage(&data_dir, storage.clone())
            .await
            .unwrap();
        m.create_collection("comp", None, Some(4), None)
            .await
            .unwrap();
        m.ingest(
            "comp",
            vec![ingest_chunk(0), ingest_chunk(1), ingest_chunk(2)],
            &embed,
        )
        .await
        .unwrap();
        m.delete_chunks("comp", &[1]).await.unwrap();

        // Before: manifest has data + tombstone fragments, no segment.
        let (before, _) = read_manifest(storage.as_ref(), "comp").await.unwrap();
        assert!(before.segments.is_empty());
        assert_eq!(before.fragments.len(), 2);

        // Compact.
        let live = m.compact_collection("comp").await.unwrap();
        assert_eq!(live, 2, "2 live records (0 and 2) after dropping deleted 1");

        // After: the WAL tail folded into an appended segment, no live fragments.
        let (after, _) = read_manifest(storage.as_ref(), "comp").await.unwrap();
        assert_eq!(after.segments.len(), 1);
        assert!(after.uncompacted().count() == 0);

        // Durable truth via materialize (exercises the v2 binary codec):
        // live chunks 0 and 2 survive, deleted 1 is gone.
        let mat = cloud::materialize(storage.as_ref(), "comp", &after)
            .await
            .unwrap();
        let ids: std::collections::HashSet<u64> = mat.chunks.keys().copied().collect();
        assert!(ids.contains(&0) && ids.contains(&2));
        assert!(
            !ids.contains(&1),
            "compaction must drop the tombstoned chunk"
        );

        let _ = std::fs::remove_dir_all(&data_dir);
    }

    // #1 regression: typed RELATIONS must survive a cold restart from S3 (the bug
    // where relation_store was local-redb-only and vanished on rebuild). Create
    // relations, wipe the local disk, restart on a fresh dir, relations return.
    #[tokio::test]
    async fn cloud_restart_recovers_relations() {
        let embed = embed_state();
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());

        let data_dir_a = unique_data_dir();
        std::fs::create_dir_all(&data_dir_a).unwrap();
        {
            let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
                store.clone(),
                "object-store:memory",
            ));
            let m = CollectionManager::new_with_storage(&data_dir_a, storage)
                .await
                .unwrap();
            m.create_collection("relsurv", None, Some(4), None)
                .await
                .unwrap();
            m.ingest(
                "relsurv",
                vec![ingest_chunk(0), ingest_chunk(1), ingest_chunk(2)],
                &embed,
            )
            .await
            .unwrap();
            // Create two relations, then delete one — only the survivor should
            // come back.
            let created = m
                .create_relations(
                    "relsurv",
                    vec![
                        CreateRelation {
                            source_chunk_id: 0,
                            target_chunk_id: 1,
                            target_document_id: None,
                            relation_type: "cites".into(),
                            metadata: HashMap::new(),
                        },
                        CreateRelation {
                            source_chunk_id: 0,
                            target_chunk_id: 2,
                            target_document_id: None,
                            relation_type: "supersedes".into(),
                            metadata: HashMap::new(),
                        },
                    ],
                )
                .await
                .unwrap();
            m.delete_relation("relsurv", &created[1].relation_id)
                .await
                .unwrap();
        }
        // Node loss: wipe local disk.
        std::fs::remove_dir_all(&data_dir_a).unwrap();

        // Restart on a fresh local dir, same object store.
        let data_dir_b = unique_data_dir();
        std::fs::create_dir_all(&data_dir_b).unwrap();
        let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m2 = CollectionManager::new_with_storage(&data_dir_b, storage)
            .await
            .unwrap();

        // The surviving relation (0 --cites--> 1) must be recovered from S3;
        // the deleted one (0 --supersedes--> 2) must NOT reappear.
        let out = m2
            .get_chunk_relations("relsurv", 0, RelationDirection::Outgoing, None)
            .await
            .unwrap();
        assert_eq!(out.len(), 1, "exactly one relation should survive restart");
        assert_eq!(out[0].relation_type, "cites");
        assert_eq!(out[0].target_chunk_id, 1);

        let _ = std::fs::remove_dir_all(&data_dir_b);
    }

    // #3: auto-compaction. Ingest enough batches to cross the fragment threshold;
    // the background trigger should fold them into a segment. We poll briefly for
    // the detached task to run, then assert the WAL is bounded.
    #[tokio::test]
    async fn auto_compaction_bounds_the_wal() {
        let embed = embed_state();
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let data_dir = unique_data_dir();
        std::fs::create_dir_all(&data_dir).unwrap();
        let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m = CollectionManager::new_with_storage(&data_dir, storage.clone())
            .await
            .unwrap();
        m.create_collection("auto", None, Some(4), None)
            .await
            .unwrap();

        // One chunk per ingest = one fragment per ingest. Cross the threshold.
        let batches = AUTO_COMPACT_FRAGMENT_THRESHOLD + 2;
        for i in 0..batches {
            m.ingest("auto", vec![ingest_chunk(i as u32)], &embed)
                .await
                .unwrap();
        }

        // Poll up to ~3s for the detached auto-compaction to land a segment and
        // shrink the uncompacted fragment set.
        let mut compacted = false;
        for _ in 0..30 {
            let (man, _) = crate::storage::lsm::read_manifest(storage.as_ref(), "auto")
                .await
                .unwrap();
            if !man.segments.is_empty()
                && man.uncompacted().count() < AUTO_COMPACT_FRAGMENT_THRESHOLD
            {
                compacted = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        assert!(
            compacted,
            "auto-compaction should have folded the WAL into a segment"
        );

        // All data still present after auto-compaction (via materialize).
        let (man, _) = crate::storage::lsm::read_manifest(storage.as_ref(), "auto")
            .await
            .unwrap();
        let mat = cloud::materialize(storage.as_ref(), "auto", &man)
            .await
            .unwrap();
        assert_eq!(mat.chunks.len(), batches, "no data lost in auto-compaction");

        let _ = std::fs::remove_dir_all(&data_dir);
    }

    // Negative: auto-compaction must NOT fire below the fragment threshold (a
    // regression dropping the threshold to ~0 would compact on every ingest).
    #[tokio::test]
    async fn auto_compaction_does_not_fire_below_threshold() {
        let embed = embed_state();
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let data_dir = unique_data_dir();
        std::fs::create_dir_all(&data_dir).unwrap();
        let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m = CollectionManager::new_with_storage(&data_dir, storage.clone())
            .await
            .unwrap();
        m.create_collection("below", None, Some(4), None)
            .await
            .unwrap();

        // Well under the threshold: a handful of single-chunk ingests.
        for i in 0..5u32 {
            m.ingest("below", vec![ingest_chunk(i)], &embed)
                .await
                .unwrap();
        }
        // Give any (wrongly) spawned compaction ample time to land a segment.
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let (man, _) = crate::storage::lsm::read_manifest(storage.as_ref(), "below")
            .await
            .unwrap();
        assert!(
            man.segments.is_empty(),
            "auto-compaction must not fire below the threshold"
        );
        assert_eq!(man.fragments.len(), 5, "all fragments still in the WAL");

        let _ = std::fs::remove_dir_all(&data_dir);
    }

    // F1 regression: compaction must physically GC old objects (deferred one
    // cycle), not leak them forever. Ingest, compact twice, assert the first
    // segment's object is deleted and the object count stays bounded.
    #[tokio::test]
    async fn compaction_gcs_old_objects() {
        let embed = embed_state();
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let data_dir = unique_data_dir();
        std::fs::create_dir_all(&data_dir).unwrap();
        let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m = CollectionManager::new_with_storage(&data_dir, storage.clone())
            .await
            .unwrap();
        m.create_collection("gc", None, Some(4), None)
            .await
            .unwrap();
        m.ingest("gc", vec![ingest_chunk(0), ingest_chunk(1)], &embed)
            .await
            .unwrap();

        // First compaction → segment S1, stages the 1 fragment for next-cycle GC.
        m.compact_collection("gc").await.unwrap();
        let (man1, _) = crate::storage::lsm::read_manifest(storage.as_ref(), "gc")
            .await
            .unwrap();
        let seg1_id = man1.segments[0].id.clone();
        // The old WAL fragment object is staged (still present this cycle).
        assert_eq!(man1.pending_deletes.len(), 1);

        // Drive enough tail-fold cycles to cross the merge threshold (8
        // segments) so a full merge runs; the merge (plus deferred GC) must
        // physically delete S1 — the key point is it's GC'd, not leaked.
        for i in 2..14u32 {
            m.ingest("gc", vec![ingest_chunk(i)], &embed).await.unwrap();
            m.compact_collection("gc").await.unwrap();
        }
        // One more cycle so the merge's staged deletes are GC'd (deferred one
        // cycle for in-flight readers).
        m.ingest("gc", vec![ingest_chunk(99)], &embed)
            .await
            .unwrap();
        m.compact_collection("gc").await.unwrap();
        m.ingest("gc", vec![ingest_chunk(100)], &embed)
            .await
            .unwrap();
        m.compact_collection("gc").await.unwrap();
        let (man2, _) = crate::storage::lsm::read_manifest(storage.as_ref(), "gc")
            .await
            .unwrap();

        // Segment S1 must be physically deleted (GC'd after being folded away).
        let s1_key = format!("gc/segments/{seg1_id}");
        assert!(
            !storage.exists(&s1_key).await.unwrap(),
            "old segment must be GC'd, not leaked"
        );
        // Object count stays BOUNDED across many compaction cycles — proving no
        // unbounded leak (the F1 bug would grow this without limit).
        let all = storage.list("gc/").await.unwrap();
        // Fixed per-namespace objects (manifest, collection.json, id-alloc)
        // plus up to MERGE_SEGMENTS(8) tail segments and this-cycle staged
        // objects — bounded, never growing with cycle count.
        assert!(
            all.len() <= 16,
            "object count must stay bounded across cycles, got {}",
            all.len()
        );
        // Data intact.
        let mat = cloud::materialize(storage.as_ref(), "gc", &man2)
            .await
            .unwrap();
        assert_eq!(mat.chunks.len(), 16);

        let _ = std::fs::remove_dir_all(&data_dir);
    }

    // Relations must survive COMPACTION-then-restart (segment-relations path),
    // not just the fragment-replay path.
    #[tokio::test]
    async fn relations_survive_compaction_then_restart() {
        let embed = embed_state();
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());

        let data_dir_a = unique_data_dir();
        std::fs::create_dir_all(&data_dir_a).unwrap();
        {
            let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
                store.clone(),
                "object-store:memory",
            ));
            let m = CollectionManager::new_with_storage(&data_dir_a, storage)
                .await
                .unwrap();
            m.create_collection("rc", None, Some(4), None)
                .await
                .unwrap();
            m.ingest("rc", vec![ingest_chunk(0), ingest_chunk(1)], &embed)
                .await
                .unwrap();
            m.create_relations(
                "rc",
                vec![CreateRelation {
                    source_chunk_id: 0,
                    target_chunk_id: 1,
                    target_document_id: None,
                    relation_type: "cites".into(),
                    metadata: HashMap::new(),
                }],
            )
            .await
            .unwrap();
            // Compact so the relation lives in the SEGMENT, not a WAL fragment.
            m.compact_collection("rc").await.unwrap();
        }
        std::fs::remove_dir_all(&data_dir_a).unwrap();

        let data_dir_b = unique_data_dir();
        std::fs::create_dir_all(&data_dir_b).unwrap();
        let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m2 = CollectionManager::new_with_storage(&data_dir_b, storage)
            .await
            .unwrap();
        let out = m2
            .get_chunk_relations("rc", 0, RelationDirection::Outgoing, None)
            .await
            .unwrap();
        assert_eq!(out.len(), 1, "relation must survive compaction+restart");
        assert_eq!(out[0].relation_type, "cites");

        let _ = std::fs::remove_dir_all(&data_dir_b);
    }

    // Concurrent ingests into the same collection: all chunks visible, all ids
    // unique, no lost writes (stresses the lock drop/reacquire window).
    #[tokio::test]
    async fn concurrent_ingests_same_collection() {
        let embed = std::sync::Arc::new(embed_state());
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let data_dir = unique_data_dir();
        std::fs::create_dir_all(&data_dir).unwrap();
        let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m = CollectionManager::new_with_storage(&data_dir, storage.clone())
            .await
            .unwrap();
        m.create_collection("conc", None, Some(4), None)
            .await
            .unwrap();

        let n = 12usize;
        let mut handles = Vec::new();
        for i in 0..n {
            let m2 = m.clone();
            let e2 = embed.clone();
            handles.push(tokio::spawn(async move {
                m2.ingest("conc", vec![ingest_chunk(i as u32)], &e2).await
            }));
        }
        for h in handles {
            h.await.unwrap().unwrap();
        }

        // All N chunks present, ids 0..N unique (no collision from the lock gap).
        let (man, _) = crate::storage::lsm::read_manifest(storage.as_ref(), "conc")
            .await
            .unwrap();
        let mat = cloud::materialize(storage.as_ref(), "conc", &man)
            .await
            .unwrap();
        assert_eq!(mat.chunks.len(), n, "all concurrent ingests durable");
        let ids: std::collections::HashSet<u64> = mat.chunks.keys().copied().collect();
        assert_eq!(ids.len(), n, "no duplicate/lost ids");
        assert_eq!(ids, (0..n as u64).collect());

        let _ = std::fs::remove_dir_all(&data_dir);
    }

    // next_id must NEVER regress across compaction + cold restart. Compaction
    // physically drops tombstoned chunks; without the segment's stored max_id
    // high-water mark, a fresh-disk rebuild would recompute next_id from the
    // live set only and REUSE the deleted ids for new chunks.
    #[tokio::test]
    async fn no_id_reuse_after_compaction_and_cold_restart() {
        let embed = embed_state();
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let data_dir_a = unique_data_dir();
        std::fs::create_dir_all(&data_dir_a).unwrap();
        {
            let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
                store.clone(),
                "object-store:memory",
            ));
            let m = CollectionManager::new_with_storage(&data_dir_a, storage)
                .await
                .unwrap();
            m.create_collection("idreuse", None, Some(4), None)
                .await
                .unwrap();
            // ids 0..3; delete the two HIGHEST, then compact them away.
            m.ingest("idreuse", (0..4u32).map(ingest_chunk).collect(), &embed)
                .await
                .unwrap();
            m.delete_chunks("idreuse", &[2, 3]).await.unwrap();
            m.compact_collection("idreuse").await.unwrap();
        }
        // Node loss: wipe local disk, cold-rebuild from S3 (max live id is 1).
        std::fs::remove_dir_all(&data_dir_a).unwrap();
        let data_dir_b = unique_data_dir();
        std::fs::create_dir_all(&data_dir_b).unwrap();
        let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m2 = CollectionManager::new_with_storage(&data_dir_b, storage.clone())
            .await
            .unwrap();

        // A new ingest must get a FRESH id (4), not reuse deleted id 2.
        m2.ingest("idreuse", vec![ingest_chunk(9)], &embed)
            .await
            .unwrap();
        let (man, _) = crate::storage::lsm::read_manifest(storage.as_ref(), "idreuse")
            .await
            .unwrap();
        let mat = cloud::materialize(storage.as_ref(), "idreuse", &man)
            .await
            .unwrap();
        // Under block allocation the exact new id is an allocator detail (a
        // fresh node claims a fresh block); the INVARIANT is that no previously
        // assigned id — live or deleted — is ever reused.
        let new_ids: Vec<u64> = mat.chunks.keys().copied().filter(|id| *id > 3).collect();
        assert_eq!(
            new_ids.len(),
            1,
            "exactly one new chunk with a never-before-assigned id, got {:?}",
            mat.chunks.keys().collect::<Vec<_>>()
        );
        assert!(
            !mat.chunks.contains_key(&2) && !mat.chunks.contains_key(&3),
            "deleted ids must not be reused"
        );

        let _ = std::fs::remove_dir_all(&data_dir_b);
    }

    // PERSISTENT-DISK restart path (the one the adversarial review flagged):
    // in cloud mode, a node restarting with its local disk intact runs
    // `load_collection` (rehydrate from redb) and SKIPS rebuild-from-S3 for
    // already-loaded collections. A chunk tombstoned locally (redb) — which is
    // exactly what delete AND the ingest-compensation path write — must stay
    // masked after that restart, even though it's still physically in redb.
    #[tokio::test]
    async fn persistent_disk_restart_honors_local_tombstones() {
        let embed = embed_state();
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let data_dir = unique_data_dir();
        std::fs::create_dir_all(&data_dir).unwrap();
        {
            let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
                store.clone(),
                "object-store:memory",
            ));
            let m = CollectionManager::new_with_storage(&data_dir, storage)
                .await
                .unwrap();
            m.create_collection("pdisk", None, Some(4), None)
                .await
                .unwrap();
            m.ingest(
                "pdisk",
                vec![ingest_chunk(0), ingest_chunk(1), ingest_chunk(2)],
                &embed,
            )
            .await
            .unwrap();
            // Writes the redb tombstone + RAM tombstone + S3 tombstone — the
            // same three places the ingest-compensation path writes.
            assert_eq!(m.delete_chunks("pdisk", &[1]).await.unwrap().0, 1);
        }

        // Restart with the SAME data_dir (persistent disk — NOT wiped). This
        // takes the load_collection-first, skip-cloud-rebuild path.
        let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m2 = CollectionManager::new_with_storage(&data_dir, storage)
            .await
            .unwrap();

        let req = SearchRequest {
            query: "chunk".to_string(),
            mode: "semantic".to_string(),
            vector_space: None,
            top_k: 20,
            query_vector: Some(vec![0.1, 0.2, 0.3, 0.4]),
            filters: HashMap::new(),
            score_weights: None,
            recency: None,
            recency_preset: None,
            recency_field: None,
            boosts: Vec::new(),
            relationship_boost: None,
            explain: false,
            include_relations: false,
            relation_types: None,
            relation_direction: RelationDirection::Outgoing,
            min_seq: None,
        };
        let (hits, _, _, _) = m2.search("pdisk", &req, &embed).await.unwrap();
        let hit_ids: std::collections::HashSet<u64> =
            hits.iter().map(|(c, _, _, _, _)| c.id).collect();
        assert!(
            !hit_ids.contains(&1),
            "tombstoned chunk must stay masked after persistent-disk restart"
        );
        assert!(
            hit_ids.contains(&0) && hit_ids.contains(&2),
            "live chunks must survive, got {:?}",
            hit_ids
        );

        let _ = std::fs::remove_dir_all(&data_dir);
    }

    // ── Warm-serverless: bucket config + id allocator + writer role ──────

    // The bucket collection.json is the source of truth on recovery: specs,
    // created_at, and CollectionConfig must survive a cold rebuild instead of
    // being re-inferred as model:"recovered" / defaults.
    #[tokio::test]
    async fn cold_rebuild_recovers_real_collection_config() {
        let embed = embed_state();
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let data_dir_a = unique_data_dir();
        std::fs::create_dir_all(&data_dir_a).unwrap();
        let created;
        {
            let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
                store.clone(),
                "object-store:memory",
            ));
            let m = CollectionManager::new_with_storage(&data_dir_a, storage)
                .await
                .unwrap();
            let mut spaces = HashMap::new();
            spaces.insert(
                "custom".to_string(),
                VectorSpaceConfig {
                    dims: 4,
                    model: "my-real-model".to_string(),
                    status: "active".to_string(),
                },
            );
            created = m
                .create_collection("cfg", Some(spaces), None, None)
                .await
                .unwrap();
            m.ingest("cfg", vec![ingest_chunk(0)], &embed)
                .await
                .unwrap();
        }
        std::fs::remove_dir_all(&data_dir_a).unwrap();

        let data_dir_b = unique_data_dir();
        std::fs::create_dir_all(&data_dir_b).unwrap();
        let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m2 = CollectionManager::new_with_storage(&data_dir_b, storage)
            .await
            .unwrap();
        let recovered = m2.get_collection("cfg").await.unwrap();
        let space = recovered.vector_spaces.get("custom").unwrap();
        assert_eq!(
            space.model, "my-real-model",
            "specs must not be re-inferred"
        );
        assert_eq!(recovered.created_at, created.created_at);
        let _ = std::fs::remove_dir_all(&data_dir_b);
    }

    // A zero-ingest collection must be discoverable from a fresh disk (the
    // create-only empty manifest + bucket config make the namespace exist).
    #[tokio::test]
    async fn empty_collection_survives_node_loss() {
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let data_dir_a = unique_data_dir();
        std::fs::create_dir_all(&data_dir_a).unwrap();
        {
            let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
                store.clone(),
                "object-store:memory",
            ));
            let m = CollectionManager::new_with_storage(&data_dir_a, storage)
                .await
                .unwrap();
            m.create_collection("emptyns", None, Some(4), None)
                .await
                .unwrap();
        }
        std::fs::remove_dir_all(&data_dir_a).unwrap();

        let data_dir_b = unique_data_dir();
        std::fs::create_dir_all(&data_dir_b).unwrap();
        let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m2 = CollectionManager::new_with_storage(&data_dir_b, storage)
            .await
            .unwrap();
        assert!(
            m2.get_collection("emptyns").await.is_some(),
            "zero-ingest collection must be rediscovered from the bucket"
        );
        let _ = std::fs::remove_dir_all(&data_dir_b);
    }

    // Writer role end-to-end: a node with NO local collection state ingests;
    // a fresh serving node sees the data. Ids from writer and attached node
    // never collide (both allocate from {ns}/id-alloc).
    #[tokio::test]
    async fn writer_role_ingest_is_stateless_and_ids_disjoint() {
        let embed = embed_state();
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());

        // Full node creates the collection and ingests two chunks.
        let dir_full = unique_data_dir();
        std::fs::create_dir_all(&dir_full).unwrap();
        let storage_full: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m_full =
            CollectionManager::new_with_storage_role(&dir_full, storage_full, NodeRole::Full)
                .await
                .unwrap();
        m_full
            .create_collection("wns", None, Some(4), None)
            .await
            .unwrap();
        m_full
            .ingest("wns", vec![ingest_chunk(0), ingest_chunk(1)], &embed)
            .await
            .unwrap();

        // Writer node: EMPTY data dir, writer role. Ingest must succeed with
        // zero local collection state and never create local index files.
        let dir_writer = unique_data_dir();
        std::fs::create_dir_all(&dir_writer).unwrap();
        let storage_writer: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m_writer =
            CollectionManager::new_with_storage_role(&dir_writer, storage_writer, NodeRole::Writer)
                .await
                .unwrap();
        let (n, _, _) = m_writer
            .ingest("wns", vec![ingest_chunk(2), ingest_chunk(3)], &embed)
            .await
            .unwrap();
        assert_eq!(n, 2);
        assert!(
            !dir_writer.join("wns").exists(),
            "writer role must not create local collection state"
        );
        // Reads are refused on the writer.
        assert!(m_writer.get_facets("wns", "", &[]).await.is_err());

        // A fresh serving node materializes ALL four chunks with unique ids.
        let dir_read = unique_data_dir();
        std::fs::create_dir_all(&dir_read).unwrap();
        let storage_read: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m_read = CollectionManager::new_with_storage(&dir_read, storage_read.clone())
            .await
            .unwrap();
        let (man, _) = crate::storage::lsm::read_manifest(storage_read.as_ref(), "wns")
            .await
            .unwrap();
        let mat = cloud::materialize(storage_read.as_ref(), "wns", &man)
            .await
            .unwrap();
        assert_eq!(mat.chunks.len(), 4, "all chunks durable");
        let ids: std::collections::HashSet<u64> = mat.chunks.keys().copied().collect();
        assert_eq!(
            ids.len(),
            4,
            "no id collisions between writer and full node"
        );
        assert!(m_read.get_collection("wns").await.is_some());

        let _ = std::fs::remove_dir_all(&dir_full);
        let _ = std::fs::remove_dir_all(&dir_writer);
        let _ = std::fs::remove_dir_all(&dir_read);
    }

    // Pre-v0.4 migration: a namespace with data but NO id-alloc object seeds
    // the allocator from the bucket-derived high-water mark — new ids never
    // collide with existing ones.
    #[tokio::test]
    async fn id_alloc_migration_seeds_past_existing_ids() {
        let embed = embed_state();
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let data_dir = unique_data_dir();
        std::fs::create_dir_all(&data_dir).unwrap();
        let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m = CollectionManager::new_with_storage(&data_dir, storage.clone())
            .await
            .unwrap();
        m.create_collection("mig", None, Some(4), None)
            .await
            .unwrap();
        m.ingest(
            "mig",
            vec![ingest_chunk(0), ingest_chunk(1), ingest_chunk(2)],
            &embed,
        )
        .await
        .unwrap();
        // Simulate a pre-v0.4 namespace: remove the allocator object.
        storage.delete("mig/id-alloc").await.unwrap();
        // Drain the local pool by restarting the manager (pool is in-RAM).
        drop(m);
        let m2 = CollectionManager::new_with_storage(&data_dir, storage.clone())
            .await
            .unwrap();
        m2.ingest("mig", vec![ingest_chunk(9)], &embed)
            .await
            .unwrap();

        let (man, _) = crate::storage::lsm::read_manifest(storage.as_ref(), "mig")
            .await
            .unwrap();
        let mat = cloud::materialize(storage.as_ref(), "mig", &man)
            .await
            .unwrap();
        assert_eq!(mat.chunks.len(), 4);
        let ids: std::collections::HashSet<u64> = mat.chunks.keys().copied().collect();
        assert_eq!(ids.len(), 4, "migrated allocator must not reuse ids 0-2");
        assert!(
            ids.contains(&3),
            "first migrated id is one past the high-water"
        );
        let _ = std::fs::remove_dir_all(&data_dir);
    }

    // ── Warm-serverless: manifest refresh + read-your-writes ─────────────

    fn cloud_search_req(min_seq: Option<u64>) -> SearchRequest {
        SearchRequest {
            query: "chunk".to_string(),
            mode: "semantic".to_string(),
            vector_space: None,
            top_k: 20,
            query_vector: Some(vec![0.1, 0.2, 0.3, 0.4]),
            filters: HashMap::new(),
            score_weights: None,
            recency: None,
            recency_preset: None,
            recency_field: None,
            boosts: Vec::new(),
            relationship_boost: None,
            explain: false,
            include_relations: false,
            relation_types: None,
            relation_direction: RelationDirection::Outgoing,
            min_seq,
        }
    }

    // Two serving nodes on one bucket: writes on A become visible on B via
    // refresh_collection — chunks, deletes, and relations all converge.
    #[tokio::test]
    async fn two_nodes_converge_via_refresh() {
        let embed = embed_state();
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let dir_a = unique_data_dir();
        let dir_b = unique_data_dir();
        std::fs::create_dir_all(&dir_a).unwrap();
        std::fs::create_dir_all(&dir_b).unwrap();
        let sa: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let sb: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let a = CollectionManager::new_with_storage(&dir_a, sa)
            .await
            .unwrap();
        a.create_collection("conv", None, Some(4), None)
            .await
            .unwrap();
        a.ingest("conv", vec![ingest_chunk(0)], &embed)
            .await
            .unwrap();

        // B boots AFTER the first write (rebuilds to seq frontier).
        let b = CollectionManager::new_with_storage(&dir_b, sb)
            .await
            .unwrap();
        let (hits, _, _, _) = b
            .search("conv", &cloud_search_req(None), &embed)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1, "B rebuilt A's first write at boot");

        // A writes more: a new chunk, a relation, and a delete of chunk id 0.
        a.ingest("conv", vec![ingest_chunk(1), ingest_chunk(2)], &embed)
            .await
            .unwrap();
        let a_ids: Vec<u64> = {
            let (hits, _, _, _) = a
                .search("conv", &cloud_search_req(None), &embed)
                .await
                .unwrap();
            hits.iter().map(|(c, _, _, _, _)| c.id).collect()
        };
        assert_eq!(a_ids.len(), 3);
        let first_id = *a_ids.iter().min().unwrap();
        let others: Vec<u64> = a_ids.iter().copied().filter(|i| *i != first_id).collect();
        a.create_relations(
            "conv",
            vec![CreateRelation {
                source_chunk_id: others[0],
                target_chunk_id: others[1],
                target_document_id: None,
                relation_type: "cites".into(),
                metadata: HashMap::new(),
            }],
        )
        .await
        .unwrap();
        a.delete_chunks("conv", &[first_id]).await.unwrap();

        // B converges via refresh (no restart, no rebuild).
        b.refresh_collection("conv").await.unwrap();
        let (hits, _, _, _) = b
            .search("conv", &cloud_search_req(None), &embed)
            .await
            .unwrap();
        let b_ids: std::collections::HashSet<u64> =
            hits.iter().map(|(c, _, _, _, _)| c.id).collect();
        assert!(!b_ids.contains(&first_id), "A's delete visible on B");
        assert_eq!(b_ids.len(), 2, "A's later chunks visible on B");
        let rels = b
            .get_chunk_relations("conv", others[0], RelationDirection::Outgoing, None)
            .await
            .unwrap();
        assert_eq!(rels.len(), 1, "A's relation visible on B");
        assert_eq!(rels[0].relation_type, "cites");

        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);
    }

    // The refresher must never double-apply a node's OWN fragments (the seq
    // tracker covers them out-of-band).
    #[tokio::test]
    async fn refresh_never_double_applies_own_writes() {
        let embed = embed_state();
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let dir = unique_data_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let st: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m = CollectionManager::new_with_storage(&dir, st).await.unwrap();
        m.create_collection("own", None, Some(4), None)
            .await
            .unwrap();
        m.ingest("own", vec![ingest_chunk(0), ingest_chunk(1)], &embed)
            .await
            .unwrap();
        m.delete_chunks("own", &[0]).await.unwrap();

        // Refresh repeatedly: state (incl. chunk_count) must not change.
        let before = m.get_collection("own").await.unwrap().chunk_count;
        for _ in 0..3 {
            m.refresh_collection("own").await.unwrap();
        }
        let after = m.get_collection("own").await.unwrap().chunk_count;
        assert_eq!(before, after, "replay of own fragments must be a no-op");
        assert_eq!(after, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Compaction two-branch rule: a node that saw everything skips segments;
    // a node whose frontier is BEHIND the watermark re-attaches fully.
    #[tokio::test]
    async fn refresh_survives_remote_compaction() {
        let embed = embed_state();
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let dir_a = unique_data_dir();
        let dir_b = unique_data_dir();
        std::fs::create_dir_all(&dir_a).unwrap();
        std::fs::create_dir_all(&dir_b).unwrap();
        let sa: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let sb: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let a = CollectionManager::new_with_storage(&dir_a, sa)
            .await
            .unwrap();
        a.create_collection("rc2", None, Some(4), None)
            .await
            .unwrap();
        a.ingest("rc2", vec![ingest_chunk(0)], &embed)
            .await
            .unwrap();

        // B attaches at frontier 1 (one fragment applied).
        let b = CollectionManager::new_with_storage(&dir_b, sb)
            .await
            .unwrap();

        // Branch 1: A ingests + compacts; B's frontier is BEHIND the watermark
        // (never saw seq 1) → refresh must full re-attach, not skip.
        a.ingest("rc2", vec![ingest_chunk(1)], &embed)
            .await
            .unwrap();
        a.compact_collection("rc2").await.unwrap();
        b.refresh_collection("rc2").await.unwrap();
        let (hits, _, _, _) = b
            .search("rc2", &cloud_search_req(None), &embed)
            .await
            .unwrap();
        assert_eq!(hits.len(), 2, "stale node re-attaches across compaction");

        // Branch 2: B now has everything; another compaction (A side) must be
        // a cheap no-op on refresh (no re-attach needed) and lose nothing.
        a.ingest("rc2", vec![ingest_chunk(2)], &embed)
            .await
            .unwrap();
        b.refresh_collection("rc2").await.unwrap(); // B applies seq tail first
        a.compact_collection("rc2").await.unwrap();
        b.refresh_collection("rc2").await.unwrap(); // wm <= frontier → skip
        let (hits, _, _, _) = b
            .search("rc2", &cloud_search_req(None), &embed)
            .await
            .unwrap();
        assert_eq!(hits.len(), 3);

        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);
    }

    // Read-your-writes across nodes: a write on A returns a seq; a search on B
    // with min_seq=seq refreshes and serves the write.
    #[tokio::test]
    async fn min_seq_gives_read_your_writes_across_nodes() {
        let embed = embed_state();
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let dir_a = unique_data_dir();
        let dir_b = unique_data_dir();
        std::fs::create_dir_all(&dir_a).unwrap();
        std::fs::create_dir_all(&dir_b).unwrap();
        let sa: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let sb: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let a = CollectionManager::new_with_storage(&dir_a, sa)
            .await
            .unwrap();
        a.create_collection("ryw", None, Some(4), None)
            .await
            .unwrap();
        a.ingest("ryw", vec![ingest_chunk(0)], &embed)
            .await
            .unwrap();
        let b = CollectionManager::new_with_storage(&dir_b, sb)
            .await
            .unwrap();

        // A writes; B searches with min_seq — must see it without manual refresh.
        let (_, _, seq) = a
            .ingest("ryw", vec![ingest_chunk(1)], &embed)
            .await
            .unwrap();
        let seq = seq.expect("cloud ingest returns a seq");
        let (hits, _, _, _) = b
            .search("ryw", &cloud_search_req(Some(seq)), &embed)
            .await
            .unwrap();
        assert_eq!(hits.len(), 2, "min_seq forces convergence before serving");

        // A min_seq beyond the write history is rejected, not waited on.
        assert!(b
            .search("ryw", &cloud_search_req(Some(9_999)), &embed)
            .await
            .is_err());

        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);
    }

    // ── Warm-serverless: lazy attach + LRU detach ─────────────────────────

    // Lazy boot registers namespaces without rebuilding; the first request
    // attaches; a concurrent stampede attaches exactly once.
    #[tokio::test]
    async fn lazy_attach_on_first_request() {
        let embed = embed_state();
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        // Seed the bucket with a collection via an eager node.
        let dir_seed = unique_data_dir();
        std::fs::create_dir_all(&dir_seed).unwrap();
        {
            let st: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
                store.clone(),
                "object-store:memory",
            ));
            let m = CollectionManager::new_with_storage(&dir_seed, st)
                .await
                .unwrap();
            m.create_collection("lazy", None, Some(4), None)
                .await
                .unwrap();
            m.ingest("lazy", vec![ingest_chunk(0), ingest_chunk(1)], &embed)
                .await
                .unwrap();
        }
        std::fs::remove_dir_all(&dir_seed).unwrap();

        // Lazy node: boot must NOT rebuild (no local dir for the collection).
        let dir = unique_data_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let st: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m = CollectionManager::new_with_storage_opts(&dir, st, NodeRole::Full, true, 0, 0)
            .await
            .unwrap();
        assert!(
            !dir.join("lazy").join("chunks.redb").exists(),
            "lazy boot must not rebuild collections"
        );

        // Stampede: 8 concurrent first-requests; all succeed, attach happens once.
        let mut handles = Vec::new();
        for _ in 0..8 {
            let m2 = m.clone();
            let e2 = embed_state();
            handles.push(tokio::spawn(async move {
                let (hits, _, _, _) = m2
                    .search("lazy", &cloud_search_req(None), &e2)
                    .await
                    .unwrap();
                hits.len()
            }));
        }
        for h in handles {
            assert_eq!(h.await.unwrap(), 2);
        }
        assert!(dir.join("lazy").join("chunks.redb").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // LRU detach: with a budget of 1, attaching a second collection evicts the
    // least-recently-used one; the evicted collection re-attaches on demand
    // with all its data (bucket is the source of truth).
    #[tokio::test]
    async fn lru_detach_and_reattach_roundtrip() {
        let embed = embed_state();
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let dir_seed = unique_data_dir();
        std::fs::create_dir_all(&dir_seed).unwrap();
        {
            let st: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
                store.clone(),
                "object-store:memory",
            ));
            let m = CollectionManager::new_with_storage(&dir_seed, st)
                .await
                .unwrap();
            for name in ["one", "two"] {
                m.create_collection(name, None, Some(4), None)
                    .await
                    .unwrap();
                m.ingest(name, vec![ingest_chunk(0)], &embed).await.unwrap();
            }
        }
        std::fs::remove_dir_all(&dir_seed).unwrap();

        let dir = unique_data_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let st: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m = CollectionManager::new_with_storage_opts(&dir, st, NodeRole::Full, true, 1, 0)
            .await
            .unwrap();

        // Attach "one", then "two" — budget 1 evicts "one".
        let (hits, _, _, _) = m
            .search("one", &cloud_search_req(None), &embed)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        let (hits, _, _, _) = m
            .search("two", &cloud_search_req(None), &embed)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        {
            let attached = m.collections.read().await;
            assert_eq!(attached.len(), 1, "LRU budget enforced");
            assert!(attached.contains_key("two"));
        }
        assert!(!dir.join("one").join("chunks.redb").exists());

        // Evicted collection re-attaches on demand, data intact.
        let (hits, _, _, _) = m
            .search("one", &cloud_search_req(None), &embed)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1, "re-attach after eviction serves all data");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Lazy mode keeps metadata correct: list/get see registered collections;
    // a collection created on ANOTHER node after boot attaches on demand.
    #[tokio::test]
    async fn lazy_attach_discovers_foreign_creates() {
        let embed = embed_state();
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let dir_a = unique_data_dir();
        let dir_b = unique_data_dir();
        std::fs::create_dir_all(&dir_a).unwrap();
        std::fs::create_dir_all(&dir_b).unwrap();
        let sa: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let sb: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        // Lazy node boots FIRST (empty bucket).
        let b = CollectionManager::new_with_storage_opts(&dir_b, sb, NodeRole::Full, true, 0, 0)
            .await
            .unwrap();
        // Another node creates + writes afterwards.
        let a = CollectionManager::new_with_storage(&dir_a, sa)
            .await
            .unwrap();
        a.create_collection("late", None, Some(4), None)
            .await
            .unwrap();
        a.ingest("late", vec![ingest_chunk(0)], &embed)
            .await
            .unwrap();

        // B never saw "late" at boot; first request attaches it anyway.
        let (hits, _, _, _) = b
            .search("late", &cloud_search_req(None), &embed)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1, "foreign create attaches on demand");

        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);
    }

    // ── Review-driven regression tests (adversarial round) ───────────────

    #[test]
    fn seq_tracker_semantics() {
        let mut t = SeqTracker::default();
        assert!(!t.covers(0));
        t.mark(0);
        assert_eq!(t.contiguous, 1);
        // Out-of-band mark ahead of the frontier; contiguous holds.
        t.mark(2);
        assert!(t.covers(2) && !t.covers(1));
        assert_eq!(t.contiguous, 1);
        // Filling the gap drains the whole out-of-band run.
        t.mark(1);
        assert_eq!(t.contiguous, 3);
        assert!(t.out_of_band.is_empty());
        // Duplicate + below-frontier marks are no-ops (no unbounded growth).
        t.mark(1);
        t.mark(2);
        assert_eq!(t.contiguous, 3);
        assert!(t.out_of_band.is_empty());
        // starting_at seeds the frontier.
        let t2 = SeqTracker::starting_at(7);
        assert!(t2.covers(6) && !t2.covers(7));
    }

    // H4 regression: eviction must be least-recently-USED, not least-recently-
    // attached. 3 collections, budget 2: attach a, attach b, USE a, attach c
    // → b (not a) is evicted.
    #[tokio::test]
    async fn lru_evicts_least_recently_used() {
        let embed = embed_state();
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let dir_seed = unique_data_dir();
        std::fs::create_dir_all(&dir_seed).unwrap();
        {
            let st: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
                store.clone(),
                "object-store:memory",
            ));
            let m = CollectionManager::new_with_storage(&dir_seed, st)
                .await
                .unwrap();
            for name in ["a", "b", "c"] {
                m.create_collection(name, None, Some(4), None)
                    .await
                    .unwrap();
                m.ingest(name, vec![ingest_chunk(0)], &embed).await.unwrap();
            }
        }
        std::fs::remove_dir_all(&dir_seed).unwrap();
        let dir = unique_data_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let st: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m = CollectionManager::new_with_storage_opts(&dir, st, NodeRole::Full, true, 2, 0)
            .await
            .unwrap();
        m.search("a", &cloud_search_req(None), &embed)
            .await
            .unwrap();
        m.search("b", &cloud_search_req(None), &embed)
            .await
            .unwrap();
        // USE a again — it is now hotter than b.
        m.search("a", &cloud_search_req(None), &embed)
            .await
            .unwrap();
        m.search("c", &cloud_search_req(None), &embed)
            .await
            .unwrap();
        let attached = m.collections.read().await;
        assert!(attached.contains_key("a"), "hot collection must survive");
        assert!(!attached.contains_key("b"), "cold collection is the victim");
        assert!(attached.contains_key("c"));
    }

    // C1 regression: a vector space added on node A becomes visible on an
    // already-attached node B via refresh (config is synced, not just
    // fragments), so B never quarantines chunks carrying the new space.
    #[tokio::test]
    async fn vector_space_add_propagates_via_refresh() {
        let embed = embed_state();
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let dir_a = unique_data_dir();
        let dir_b = unique_data_dir();
        std::fs::create_dir_all(&dir_a).unwrap();
        std::fs::create_dir_all(&dir_b).unwrap();
        let sa: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let sb: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let a = CollectionManager::new_with_storage(&dir_a, sa)
            .await
            .unwrap();
        a.create_collection("vsprop", None, Some(4), None)
            .await
            .unwrap();
        a.ingest("vsprop", vec![ingest_chunk(0)], &embed)
            .await
            .unwrap();
        let b = CollectionManager::new_with_storage(&dir_b, sb)
            .await
            .unwrap();

        // A adds an 8-dim space, then ingests a chunk carrying it.
        a.add_vector_space("vsprop", "wide", 8, "test-model")
            .await
            .unwrap();
        let mut ic = ingest_chunk(1);
        ic.embeddings.insert("wide".to_string(), vec![0.1; 8]);
        a.ingest("vsprop", vec![ic], &embed).await.unwrap();

        // B refreshes: must learn the space AND apply the chunk (no quarantine).
        b.refresh_collection("vsprop").await.unwrap();
        let bc = b.get_collection("vsprop").await.unwrap();
        assert!(bc.vector_spaces.contains_key("wide"), "config converged");
        let (hits, _, _, _) = b
            .search("vsprop", &cloud_search_req(None), &embed)
            .await
            .unwrap();
        assert_eq!(
            hits.len(),
            2,
            "chunk with the new space applied, not quarantined"
        );

        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);
    }

    // Rank-1 regression: ingest racing a refresher loop never double-applies
    // (chunk_count exact, no duplicate hits).
    #[tokio::test]
    async fn ingest_races_refresher_no_double_apply() {
        let embed = std::sync::Arc::new(embed_state());
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let dir = unique_data_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let st: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m = CollectionManager::new_with_storage(&dir, st).await.unwrap();
        m.create_collection("race", None, Some(4), None)
            .await
            .unwrap();

        let n = 10usize;
        let refresher = {
            let m2 = m.clone();
            tokio::spawn(async move {
                for _ in 0..200 {
                    let _ = m2.refresh_collection("race").await;
                    tokio::task::yield_now().await;
                }
            })
        };
        let mut handles = Vec::new();
        for i in 0..n {
            let m2 = m.clone();
            let e2 = embed.clone();
            handles.push(tokio::spawn(async move {
                m2.ingest("race", vec![ingest_chunk(i as u32)], &e2).await
            }));
        }
        for h in handles {
            h.await.unwrap().unwrap();
        }
        refresher.await.unwrap();
        let _ = m.refresh_collection("race").await;

        let c = m.get_collection("race").await.unwrap();
        assert_eq!(c.chunk_count as usize, n, "no double-count under the race");
        let (hits, _, _, _) = m
            .search("race", &cloud_search_req(None), &embed)
            .await
            .unwrap();
        assert_eq!(hits.len(), n, "no duplicate/lost chunks under the race");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Rank-6: persistent-disk restart catches up the REMOTE delta via refresh
    // instead of serving stale data (applied_seq persistence path).
    #[tokio::test]
    async fn persistent_restart_catches_up_remote_delta() {
        let embed = embed_state();
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let dir_a = unique_data_dir();
        let dir_w = unique_data_dir();
        std::fs::create_dir_all(&dir_a).unwrap();
        std::fs::create_dir_all(&dir_w).unwrap();
        {
            let sa: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
                store.clone(),
                "object-store:memory",
            ));
            let a = CollectionManager::new_with_storage(&dir_a, sa)
                .await
                .unwrap();
            a.create_collection("pd", None, Some(4), None)
                .await
                .unwrap();
            a.ingest("pd", vec![ingest_chunk(0)], &embed).await.unwrap();
        } // node A down; its disk PERSISTS.
        {
            let sw: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
                store.clone(),
                "object-store:memory",
            ));
            let w = CollectionManager::new_with_storage_role(&dir_w, sw, NodeRole::Writer)
                .await
                .unwrap();
            w.ingest("pd", vec![ingest_chunk(1)], &embed).await.unwrap();
        }
        // A restarts on the SAME dir (load_collection path, not rebuild).
        let sa: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let a = CollectionManager::new_with_storage(&dir_a, sa)
            .await
            .unwrap();
        a.refresh_collection("pd").await.unwrap();
        let (hits, _, _, _) = a
            .search("pd", &cloud_search_req(None), &embed)
            .await
            .unwrap();
        assert_eq!(
            hits.len(),
            2,
            "restart + refresh catches up the writer's delta"
        );
        let c = a.get_collection("pd").await.unwrap();
        assert_eq!(c.chunk_count, 2, "delta applied exactly once");
        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_w);
    }

    // Rank-8: a wrong-dims chunk inside a fragment is quarantined on replay
    // without corrupting anything else.
    #[tokio::test]
    async fn refresh_quarantines_wrong_dims_without_corruption() {
        let embed = embed_state();
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let dir = unique_data_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let st: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m = CollectionManager::new_with_storage(&dir, st.clone())
            .await
            .unwrap();
        m.create_collection("quar", None, Some(4), None)
            .await
            .unwrap();
        m.ingest("quar", vec![ingest_chunk(0)], &embed)
            .await
            .unwrap();

        // Hand-craft a fragment with one bad (3-dim) and one good chunk,
        // simulating a poisoned foreign writer.
        let mut bad = DocumentChunk {
            id: 500_000,
            collection: "quar".into(),
            file_id: "bad".into(),
            chunk_index: 0,
            page: None,
            text: "bad chunk".into(),
            metadata: HashMap::new(),
            doc_type: "chunk".into(),
            parent_id: None,
            group_id: None,
            embeddings: HashMap::new(),
            embedding: None,
        };
        bad.embeddings.insert("default".into(), vec![0.1, 0.2, 0.3]);
        let mut good = bad.clone();
        good.id = 500_001;
        good.file_id = "good".into();
        good.text = "good chunk".into();
        good.embeddings
            .insert("default".into(), vec![0.1, 0.2, 0.3, 0.4]);
        let payload = serde_json::to_vec(&vec![bad, good]).unwrap();
        crate::storage::lsm::append_fragment(st.as_ref(), "quar", bytes::Bytes::from(payload), 2)
            .await
            .unwrap();

        m.refresh_collection("quar").await.unwrap();
        let (hits, _, _, _) = m
            .search("quar", &cloud_search_req(None), &embed)
            .await
            .unwrap();
        let ids: std::collections::HashSet<u64> = hits.iter().map(|(c, _, _, _, _)| c.id).collect();
        assert!(ids.contains(&500_001), "good chunk applied");
        assert!(!ids.contains(&500_000), "bad chunk quarantined");
        // Post-quarantine ingest still works and searches correctly (mmap not shifted).
        m.ingest("quar", vec![ingest_chunk(9)], &embed)
            .await
            .unwrap();
        let (hits, _, _, _) = m
            .search("quar", &cloud_search_req(None), &embed)
            .await
            .unwrap();
        assert_eq!(hits.len(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Rank-9: min_seq is ignored in local mode; exact boundary at next_seq.
    #[tokio::test]
    async fn min_seq_local_mode_and_boundary() {
        let embed = embed_state();
        // Local mode: min_seq must be ignored, not error.
        let dir = unique_data_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let m = CollectionManager::new(&dir).await.unwrap();
        m.create_collection("loc", None, Some(4), None)
            .await
            .unwrap();
        m.ingest("loc", vec![ingest_chunk(0)], &embed)
            .await
            .unwrap();
        let (hits, _, _, _) = m
            .search("loc", &cloud_search_req(Some(999)), &embed)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1, "local mode ignores min_seq");
        let _ = std::fs::remove_dir_all(&dir);

        // Cloud: last valid seq (next_seq-1) succeeds; next_seq is rejected.
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let dir2 = unique_data_dir();
        std::fs::create_dir_all(&dir2).unwrap();
        let st: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m2 = CollectionManager::new_with_storage(&dir2, st)
            .await
            .unwrap();
        m2.create_collection("bnd", None, Some(4), None)
            .await
            .unwrap();
        let (_, _, seq) = m2
            .ingest("bnd", vec![ingest_chunk(0)], &embed)
            .await
            .unwrap();
        let seq = seq.unwrap();
        assert!(m2
            .search("bnd", &cloud_search_req(Some(seq)), &embed)
            .await
            .is_ok());
        assert!(m2
            .search("bnd", &cloud_search_req(Some(seq + 1)), &embed)
            .await
            .is_err());
        let _ = std::fs::remove_dir_all(&dir2);
    }

    // Rank-4/H3: a writer delete against a bogus namespace must NOT create a
    // phantom collection, and absurd ids are rejected by the allocator frontier.
    #[tokio::test]
    async fn writer_delete_validates_namespace_and_ids() {
        let embed = embed_state();
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let dir = unique_data_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let st: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let w = CollectionManager::new_with_storage_role(&dir, st.clone(), NodeRole::Writer)
            .await
            .unwrap();
        // Bogus namespace: error + nothing created in the bucket.
        assert!(w.delete_chunks("ghost", &[1]).await.is_err());
        assert!(
            !st.exists("ghost/manifest").await.unwrap(),
            "no phantom namespace"
        );

        // Real collection: absurd id rejected (would poison max_id forever).
        let dir_f = unique_data_dir();
        std::fs::create_dir_all(&dir_f).unwrap();
        let sf: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let f = CollectionManager::new_with_storage(&dir_f, sf)
            .await
            .unwrap();
        f.create_collection("real", None, Some(4), None)
            .await
            .unwrap();
        f.ingest("real", vec![ingest_chunk(0)], &embed)
            .await
            .unwrap();
        assert!(w.delete_chunks("real", &[u64::MAX]).await.is_err());
        // In-range delete works.
        assert!(w.delete_chunks("real", &[0]).await.is_ok());
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&dir_f);
    }

    // H1-lite: delete+recreate on another node is detected via created_at and
    // the stale node re-attaches to the NEW collection.
    #[tokio::test]
    async fn delete_recreate_detected_by_refresh() {
        let embed = embed_state();
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let dir_a = unique_data_dir();
        let dir_b = unique_data_dir();
        std::fs::create_dir_all(&dir_a).unwrap();
        std::fs::create_dir_all(&dir_b).unwrap();
        let sa: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let sb: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let a = CollectionManager::new_with_storage(&dir_a, sa)
            .await
            .unwrap();
        a.create_collection("cycle", None, Some(4), None)
            .await
            .unwrap();
        a.ingest(
            "cycle",
            vec![ingest_chunk(0), ingest_chunk(1), ingest_chunk(2)],
            &embed,
        )
        .await
        .unwrap();
        let b = CollectionManager::new_with_storage(&dir_b, sb)
            .await
            .unwrap();

        // A deletes and recreates with different content.
        a.delete_collection("cycle").await.unwrap();
        a.create_collection("cycle", None, Some(4), None)
            .await
            .unwrap();
        a.ingest("cycle", vec![ingest_chunk(9)], &embed)
            .await
            .unwrap();

        // B refreshes: must serve the NEW collection (1 chunk), not the old 3.
        b.refresh_collection("cycle").await.unwrap();
        let (hits, _, _, _) = b
            .search("cycle", &cloud_search_req(None), &embed)
            .await
            .unwrap();
        assert_eq!(
            hits.len(),
            1,
            "stale node re-attached to the recreated collection"
        );
        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);
    }

    // ── Scale harness (env-gated) ─────────────────────────────────────────
    // COMPASS_SCALE_N=<chunks> [COMPASS_SCALE_DIMS=<dims>] cargo test
    //   --features object-storage --release scale_envelope -- --nocapture
    // Measures ingest throughput, attach (cold rebuild) time, and search
    // latency against a local-disk Storage backend (same code paths as S3,
    // disk-bound). Skips (passes) when COMPASS_SCALE_N is unset.
    #[tokio::test]
    async fn scale_envelope() {
        let Ok(n) = std::env::var("COMPASS_SCALE_N") else {
            eprintln!("skipped: COMPASS_SCALE_N not set");
            return;
        };
        let n: usize = n.parse().unwrap();
        let dims: usize = std::env::var("COMPASS_SCALE_DIMS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(128);
        let batch = 2_000usize;
        let embed = embed_state();

        let bucket_dir = unique_data_dir();
        std::fs::create_dir_all(&bucket_dir).unwrap();
        let storage: Arc<dyn Storage> =
            Arc::new(crate::storage::local::LocalDiskStorage::new(&bucket_dir).unwrap());
        // local-disk backend reports "local-disk" => cloud_mode false. Wrap it
        // to report as a cloud backend so the full S3-native path runs.
        struct CloudyDisk(Arc<dyn Storage>);
        #[async_trait::async_trait]
        impl Storage for CloudyDisk {
            async fn get(&self, k: &str) -> Result<bytes::Bytes, crate::storage::StorageError> {
                self.0.get(k).await
            }
            async fn get_range(
                &self,
                k: &str,
                r: std::ops::Range<u64>,
            ) -> Result<bytes::Bytes, crate::storage::StorageError> {
                self.0.get_range(k, r).await
            }
            async fn get_versioned(
                &self,
                k: &str,
            ) -> Result<(bytes::Bytes, crate::storage::Version), crate::storage::StorageError>
            {
                self.0.get_versioned(k).await
            }
            async fn put(
                &self,
                k: &str,
                b: bytes::Bytes,
            ) -> Result<crate::storage::Version, crate::storage::StorageError> {
                self.0.put(k, b).await
            }
            async fn put_if_match(
                &self,
                k: &str,
                b: bytes::Bytes,
                e: &crate::storage::Version,
            ) -> Result<crate::storage::Version, crate::storage::StorageError> {
                self.0.put_if_match(k, b, e).await
            }
            async fn put_if_not_exists(
                &self,
                k: &str,
                b: bytes::Bytes,
            ) -> Result<crate::storage::Version, crate::storage::StorageError> {
                self.0.put_if_not_exists(k, b).await
            }
            async fn delete(&self, k: &str) -> Result<(), crate::storage::StorageError> {
                self.0.delete(k).await
            }
            async fn list(
                &self,
                p: &str,
            ) -> Result<Vec<crate::storage::ObjectMeta>, crate::storage::StorageError> {
                self.0.list(p).await
            }
            async fn list_dirs(
                &self,
                p: &str,
            ) -> Result<Vec<String>, crate::storage::StorageError> {
                self.0.list_dirs(p).await
            }
            fn backend_name(&self) -> &'static str {
                "scale-disk"
            }
        }
        let storage: Arc<dyn Storage> = Arc::new(CloudyDisk(storage));

        let node_dir = unique_data_dir();
        std::fs::create_dir_all(&node_dir).unwrap();
        let m = CollectionManager::new_with_storage_opts(
            &node_dir,
            storage.clone(),
            NodeRole::Full,
            false,
            0,
            0,
        )
        .await
        .unwrap();
        let mut spaces = HashMap::new();
        spaces.insert(
            "default".to_string(),
            VectorSpaceConfig {
                dims,
                model: "scale".into(),
                status: "active".into(),
            },
        );
        m.create_collection("scale", Some(spaces), None, None)
            .await
            .unwrap();

        // Deterministic pseudo-random embeddings (no Math.random / clock).
        let mk_vec = |seed: usize| -> Vec<f32> {
            let mut x = seed as u64 * 6364136223846793005 + 1442695040888963407;
            (0..dims)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    ((x % 2000) as f32 / 1000.0) - 1.0
                })
                .collect()
        };
        let t0 = std::time::Instant::now();
        for b0 in (0..n).step_by(batch) {
            let chunks: Vec<IngestChunk> = (b0..(b0 + batch).min(n))
                .map(|i| {
                    let mut embeddings = HashMap::new();
                    embeddings.insert("default".to_string(), mk_vec(i));
                    IngestChunk {
                        client_id: None,
                        file_id: format!("f{i}"),
                        chunk_index: 0,
                        page: None,
                        text: format!("scale test chunk number {i} lorem ipsum"),
                        metadata: HashMap::new(),
                        doc_type: "chunk".to_string(),
                        parent_id: None,
                        parent_ref: None,
                        group_id: None,
                        embeddings,
                        embedding: None,
                    }
                })
                .collect();
            m.ingest("scale", chunks, &embed).await.unwrap();
        }
        let ingest_s = t0.elapsed().as_secs_f64();

        // Cold attach: fresh node dir, same bucket.
        drop(m);
        let node2 = unique_data_dir();
        std::fs::create_dir_all(&node2).unwrap();
        let t1 = std::time::Instant::now();
        let m2 = CollectionManager::new_with_storage_opts(
            &node2,
            storage.clone(),
            NodeRole::Full,
            false,
            0,
            0,
        )
        .await
        .unwrap();
        let attach_s = t1.elapsed().as_secs_f64();

        // Search latency (semantic, 50 queries).
        let mut req = cloud_search_req(None);
        let t2 = std::time::Instant::now();
        let mut hits_total = 0usize;
        for q in 0..50 {
            req.query_vector = Some(mk_vec(q * 7919));
            let (hits, _, _, _) = m2.search("scale", &req, &embed).await.unwrap();
            hits_total += hits.len();
        }
        let search_ms = t2.elapsed().as_secs_f64() * 1000.0 / 50.0;
        assert!(hits_total > 0);

        eprintln!(
            "SCALE n={n} dims={dims}: ingest {:.1}s ({:.0} chunks/s) | cold attach {:.1}s | search avg {:.1}ms",
            ingest_s, n as f64 / ingest_s, attach_s, search_ms
        );
        let _ = std::fs::remove_dir_all(&bucket_dir);
        let _ = std::fs::remove_dir_all(&node_dir);
        let _ = std::fs::remove_dir_all(&node2);
    }
}
