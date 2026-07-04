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
#[cfg(all(test, feature = "object-storage"))]
mod partition_cloud_tests;
#[cfg(test)]
mod partition_tests;
pub mod partitions;
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
    /// USearch's `filtered_search`. Also the live-id universe for facets and
    /// delete-by-filter.
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
    /// Test-only convenience; `main.rs` goes through `new_with_storage_opts`.
    #[cfg(test)]
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
        let mut fts = if tantivy_dir.join("meta.json").exists() {
            tantivy_fts::open_index(&tantivy_dir)?
        } else {
            tantivy_fts::build_index(&tantivy_dir, &[])?
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
        let mut facet_rebuild = tantivy_fts::FacetBitsets::default();
        chunk_store.for_each(|id, chunk| {
            if id >= max_seen_id {
                max_seen_id = id;
            }
            rehydrated_count += 1;
            if !tombstones.contains(&id) {
                filter_index.insert(id, &filter_meta(&chunk));
                // Facets were EMPTY after every restart (open_index returns
                // none and nothing rebuilt them) — rebuild here, same pass.
                facet_rebuild.insert_chunk(&chunk);
            }
        })?;
        fts.facet_bitsets = facet_rebuild;
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
        // The partition separator is reserved: user collections must not
        // squat on internal partition namespaces.
        if partitions::is_partition_ns(name) {
            return Err(format!(
                "Collection name '{name}' contains the reserved partition separator '{}'",
                partitions::PART_SEP
            )
            .into());
        }
        if let Some(cfg) = &config {
            if let Some(field) = &cfg.partition_by {
                if field.is_empty() {
                    return Err("partition_by must name a metadata field".into());
                }
            }
        }
        self.create_collection_inner(name, vector_spaces, embedding_dims, config)
            .await
    }

    /// Shared create path. Partition namespaces (containing [`partitions::PART_SEP`])
    /// may only be created internally by the ingest router.
    async fn create_collection_inner(
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
            let fts = tantivy_fts::build_index(&tantivy_dir, &[])?;

            // Create empty vector spaces
            let mut vs_map = HashMap::new();
            for sname in collection.vector_spaces.keys() {
                vs_map.insert(
                    sname.clone(),
                    Arc::new(VectorState {
                        index: None,
                        key_to_chunk_id: Vec::new(),
                        mmap_vectors: None,
                        vectors: Vec::new(),
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
                    // Partition namespaces mint from the PARENT's allocator
                    // (collection-unique ids) and are never seeded themselves.
                    if let Err(e) = if partitions::is_partition_ns(name) {
                        Ok(())
                    } else {
                        crate::storage::id_alloc::seed(self.storage.as_ref(), name, 0).await
                    } {
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

        // Partitioned parents allocate chunk ids from a shared CAS allocator
        // in LOCAL mode too (partitions must never mint colliding ids). This
        // is a single JSON file under the collection dir — no WAL/manifest.
        if !self.cloud_mode
            && collection.config.partition_by.is_some()
            && !partitions::is_partition_ns(name)
        {
            crate::storage::id_alloc::seed(self.storage.as_ref(), name, 0).await?;
        }

        tracing::info!("Created collection '{}'", name);
        Ok(collection)
    }

    /// Metadata of ATTACHED collections only — no bucket round-trips (used
    /// by /metrics; lazy-registered namespaces are intentionally excluded).
    pub async fn attached_collections(&self) -> Vec<Collection> {
        let collections = self.collections.read().await;
        collections.values().map(|c| c.metadata.clone()).collect()
    }

    pub async fn list_collections(&self) -> Vec<Collection> {
        let mut out: Vec<Collection> = {
            let collections = self.collections.read().await;
            collections.values().map(|c| c.metadata.clone()).collect()
        };
        // Partition namespaces are internal — the parent represents them.
        out.retain(|c| !partitions::is_partition_ns(&c.name));
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
                if partitions::is_partition_ns(&name) {
                    continue;
                }
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
        // Partitioned parent: cascade over every partition namespace FIRST,
        // so a failure mid-cascade leaves the parent (and the retry path)
        // intact. Partitions are discovered from all sources — attached map,
        // lazy registry, local dirs, and the bucket (a writer node may have
        // created partitions this node never saw).
        if !partitions::is_partition_ns(name) {
            let prefix = format!("{name}{}", partitions::PART_SEP);
            let mut parts: std::collections::HashSet<String> = std::collections::HashSet::new();
            {
                let collections = self.collections.read().await;
                parts.extend(
                    collections
                        .keys()
                        .filter(|k| k.starts_with(&prefix))
                        .cloned(),
                );
            }
            parts.extend(
                self.registered
                    .read()
                    .await
                    .iter()
                    .filter(|k| k.starts_with(&prefix))
                    .cloned(),
            );
            if let Ok(entries) = std::fs::read_dir(&self.data_dir) {
                for e in entries.flatten() {
                    if let Some(n) = e.file_name().to_str() {
                        if n.starts_with(&prefix) {
                            parts.insert(n.to_string());
                        }
                    }
                }
            }
            if self.cloud_mode {
                if let Ok(all) = crate::storage::lsm::list_namespaces(self.storage.as_ref()).await {
                    parts.extend(all.into_iter().filter(|n| n.starts_with(&prefix)));
                }
            }
            for part in parts {
                Box::pin(self.delete_collection(&part)).await?;
            }
        }
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
                return Err(not_found(format_args!("Collection \'{}\' not found", name)));
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
        self.reject_if_partitioned(collection_name, "vector-space changes")
            .await?;
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
            let loaded = collections.get(collection_name).ok_or_else(|| {
                not_found(format_args!("Collection \'{}\' not found", collection_name))
            })?;
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
        let loaded = collections.get_mut(collection_name).ok_or_else(|| {
            not_found(format_args!("Collection \'{}\' not found", collection_name))
        })?;
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
        self.reject_if_partitioned(collection_name, "vector-space changes")
            .await?;
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
            let loaded = collections.get(collection_name).ok_or_else(|| {
                not_found(format_args!("Collection \'{}\' not found", collection_name))
            })?;
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
        let loaded = collections.get_mut(collection_name).ok_or_else(|| {
            not_found(format_args!("Collection \'{}\' not found", collection_name))
        })?;
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
        self.reject_if_partitioned(collection_name, "vector-space changes")
            .await?;
        if self.role == NodeRole::Writer {
            return Err(
                "this node runs in writer role; manage vector spaces via a serving node".into(),
            );
        }
        self.ensure_attached(collection_name).await?;
        // Phase 1 (short read lock): preconditions.
        {
            let collections = self.collections.read().await;
            let loaded = collections.get(collection_name).ok_or_else(|| {
                not_found(format_args!("Collection \'{}\' not found", collection_name))
            })?;
            if !loaded.metadata.vector_spaces.contains_key(space_name) {
                return Err(not_found(format_args!(
                    "Vector space \'{}\' not found",
                    space_name
                )));
            }
        }

        // Phase 2 (NO lock): bucket-first CAS.
        if self.cloud_mode {
            cloud::cas_update_bucket_config(self.storage.as_ref(), collection_name, |cfg| {
                if !cfg.vector_spaces.contains_key(space_name) {
                    return Err(not_found(format_args!(
                        "Vector space \'{}\' not found",
                        space_name
                    )));
                }
                cfg.default_vector_space = Some(space_name.to_string());
                Ok(())
            })
            .await?;
        }

        // Phase 3 (write lock): apply locally.
        let mut collections = self.collections.write().await;
        let loaded = collections.get_mut(collection_name).ok_or_else(|| {
            not_found(format_args!("Collection \'{}\' not found", collection_name))
        })?;
        loaded.metadata.default_vector_space = Some(space_name.to_string());
        store::save_metadata(&self.data_dir, &loaded.metadata)?;
        Ok(())
    }

    /// Mark a vector space as active: flip the persisted status (bucket-first
    /// CAS in cloud mode) and hot-load the rebuilt index into the serving
    /// collection. Called by the rebuild job on completion.
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
        let loaded = collections.get_mut(collection_name).ok_or_else(|| {
            not_found(format_args!("Collection \'{}\' not found", collection_name))
        })?;

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

    // ── Tenant partitions (Phase 6) ──────────────────────────────────────

    /// The partition field of a collection, or None for normal collections
    /// and partition namespaces themselves. Attaches the parent if needed
    /// (cheap: a partitioned parent holds config only, no chunk data).
    async fn partition_field(
        &self,
        name: &str,
    ) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        if partitions::is_partition_ns(name) {
            return Ok(None);
        }
        self.ensure_attached(name).await?;
        let collections = self.collections.read().await;
        Ok(collections
            .get(name)
            .and_then(|l| l.metadata.config.partition_by.clone()))
    }

    /// Make sure a partition namespace exists and is servable, creating it on
    /// first sight (inheriting the parent's vector spaces + embed model).
    /// Racing creators and partitions created by writer nodes resolve via the
    /// create path's own already-exists handling.
    async fn ensure_partition(
        &self,
        parent: &str,
        pval: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let ns = partitions::partition_ns(parent, pval);
        if self.collections.read().await.contains_key(&ns) {
            return Ok(ns);
        }
        if self.cloud_mode && self.registered.read().await.contains(&ns) {
            return Ok(ns); // lazy attach loads it at the entry point
        }
        let (spaces, config) = {
            let collections = self.collections.read().await;
            let parent_meta = collections
                .get(parent)
                .ok_or_else(|| not_found(format_args!("Collection '{}' not found", parent)))?;
            (
                parent_meta.metadata.vector_spaces.clone(),
                CollectionConfig {
                    embed_model: parent_meta.metadata.config.embed_model.clone(),
                    partition_by: None,
                },
            )
        };
        match self
            .create_collection_inner(&ns, Some(spaces), None, Some(config))
            .await
        {
            Ok(_) => Ok(ns),
            // Lost a create race (local map, bucket config, or bucket data) —
            // the partition exists; ensure_attached at the entry point loads it.
            Err(e) if e.to_string().contains("already") => Ok(ns),
            Err(e) => Err(e),
        }
    }

    /// Ingest into a partitioned collection: one delegated ingest per touched
    /// partition. `seq` is passed through when exactly one partition was
    /// touched; multi-partition batches return None (each partition has its
    /// own manifest and thus its own seq domain).
    async fn ingest_partitioned(
        &self,
        parent: &str,
        field: &str,
        ingest_chunks: Vec<IngestChunk>,
        embed_state: &EmbedState,
    ) -> Result<(usize, HashMap<String, u64>, Option<u64>), Box<dyn std::error::Error + Send + Sync>>
    {
        let groups = partitions::group_by_partition(field, ingest_chunks)?;
        let multi = groups.len() > 1;
        let mut total = 0usize;
        let mut id_map = HashMap::new();
        let mut last_seq = None;
        for (pval, group) in groups {
            let ns = self.ensure_partition(parent, &pval).await?;
            let (n, ids, seq) = Box::pin(self.ingest(&ns, group, embed_state)).await?;
            total += n;
            id_map.extend(ids);
            last_seq = seq;
        }
        Ok((total, id_map, if multi { None } else { last_seq }))
    }

    /// Search a partitioned collection: route to the partitions named by the
    /// partition-field filter, merge by score, truncate to top_k. Partitions
    /// that do not exist yet contribute zero results (a tenant with no data
    /// is empty, not an error).
    #[allow(clippy::type_complexity)]
    async fn search_partitioned(
        &self,
        parent: &str,
        field: &str,
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
        let pvals = partitions::partition_values_from_filters(field, &req.filters)?;
        if req.min_seq.is_some() && pvals.len() > 1 {
            return Err("min_seq applies to a single partition's write history; \
                 filter to one partition value when using it"
                .into());
        }
        let mut merged = Vec::new();
        let mut total = 0usize;
        let mut took = 0u64;
        let mut explain = None;
        for pval in pvals {
            let ns = partitions::partition_ns(parent, &pval);
            match Box::pin(self.search(&ns, req, embed_state)).await {
                Ok((results, t, us, ex)) => {
                    merged.extend(results);
                    total += t;
                    took += us;
                    if explain.is_none() {
                        explain = ex;
                    }
                }
                Err(e) if e.downcast_ref::<NotFound>().is_some() => continue,
                Err(e) => return Err(e),
            }
        }
        merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        merged.truncate(req.top_k);
        Ok((merged, total, took, explain))
    }

    /// Writer-side partition bootstrap: create-only bucket objects for a
    /// partition namespace (config copy + empty manifest). Idempotent — racing
    /// writers and serving nodes all converge on the first writer's objects.
    /// No per-partition id allocator is seeded (ids mint from the parent's).
    async fn ensure_partition_ns_cloud(
        &self,
        parent_cfg: &cloud::BucketConfig,
        pval: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let ns = partitions::partition_ns(&parent_cfg.name, pval);
        if self.bucket_configs.read().await.contains_key(&ns) {
            return Ok(ns);
        }
        let mut part_cfg = parent_cfg.clone();
        part_cfg.name = ns.clone();
        part_cfg.config.partition_by = None;
        match cloud::write_bucket_config_if_absent(self.storage.as_ref(), &ns, &part_cfg).await {
            Ok(()) | Err(crate::storage::StorageError::AlreadyExists(_)) => {}
            Err(e) => return Err(format!("partition config write failed: {e}").into()),
        }
        match crate::storage::lsm::init_namespace(self.storage.as_ref(), &ns).await {
            Ok(()) | Err(crate::storage::StorageError::AlreadyExists(_)) => {}
            Err(e) => return Err(format!("partition manifest init failed: {e}").into()),
        }
        Ok(ns)
    }

    /// Typed fence for operations not yet routed on partitioned collections.
    /// Writer nodes consult the bucket config (they hold no local metadata);
    /// without this a writer would durably append e.g. relations into the
    /// parent namespace, which no serving node ever materializes.
    async fn reject_if_partitioned(
        &self,
        name: &str,
        what: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.is_partitioned_any_role(name).await? {
            return Err(format!(
                "{what} is not supported on a partitioned collection yet \
                 (collection '{name}' is partitioned)"
            )
            .into());
        }
        Ok(())
    }

    /// Role-aware "is this collection partitioned?": serving nodes read local
    /// metadata (attaching if needed); writers consult the bucket config.
    async fn is_partitioned_any_role(
        &self,
        name: &str,
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        if partitions::is_partition_ns(name) {
            return Ok(false);
        }
        if self.role == NodeRole::Writer {
            return Ok(self
                .bucket_config(name, false)
                .await
                .map(|c| c.config.partition_by.is_some())
                .unwrap_or(false));
        }
        Ok(self.partition_field(name).await?.is_some())
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
        // Partitions mint from the PARENT's allocator: chunk ids stay unique
        // across the whole partitioned collection.
        let ns = partitions::alloc_ns(ns);
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
                let loaded = collections.get_mut(collection_name).ok_or_else(|| {
                    not_found(format_args!("Collection \'{}\' not found", collection_name))
                })?;
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
                None => {
                    return Err(not_found(format_args!(
                        "Collection \'{}\' not found",
                        collection_name
                    )))
                }
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
            .ok_or_else(|| {
                not_found(format_args!(
                    "Collection \'{}\' not found in object storage",
                    ns
                ))
            })?;
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

        // Partitioned collection: group by the partition field, make each
        // partition's bucket objects exist (idempotent create-only writes —
        // a writer may see a tenant before any serving node does), delegate.
        if let Some(field) = cfg.config.partition_by.clone() {
            let groups = partitions::group_by_partition(&field, ingest_chunks)?;
            let multi = groups.len() > 1;
            let mut total = 0usize;
            let mut id_map = HashMap::new();
            let mut last_seq = None;
            for (pval, group) in groups {
                let ns = self.ensure_partition_ns_cloud(&cfg, &pval).await?;
                let (n, ids, seq) =
                    Box::pin(self.ingest_stateless(&ns, group, embed_state)).await?;
                total += n;
                id_map.extend(ids);
                last_seq = seq;
            }
            return Ok((total, id_map, if multi { None } else { last_seq }));
        }

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

        // Partitioned collection: group by the partition field and delegate
        // each group to its partition namespace (partitions.rs).
        if let Some(field) = self.partition_field(collection_name).await? {
            return self
                .ingest_partitioned(collection_name, &field, ingest_chunks, embed_state)
                .await;
        }

        let count = ingest_chunks.len();
        self.ensure_attached(collection_name).await?;

        // Cloud mode: ids come from CAS-leased blocks (storage/id_alloc.rs) so
        // they can NEVER collide with a stateless writer's ids. This happens
        // BEFORE taking the write lock (its refill path does S3 round-trips).
        // A failed ingest after this point leaks the taken ids — gaps are fine;
        // the invariant is no-reuse, not density.
        //
        // Partition namespaces use the block allocator in LOCAL mode too: all
        // partitions of one collection mint from the PARENT's allocator, so a
        // per-partition next_id counter would collide across siblings.
        let cloud_ids: Option<Vec<u64>> =
            if (self.cloud_mode || partitions::is_partition_ns(collection_name)) && count > 0 {
                Some(self.take_ids_cloud(collection_name, count).await?)
            } else {
                None
            };

        let mut collections = self.collections.write().await;
        let loaded = collections.get_mut(collection_name).ok_or_else(|| {
            not_found(format_args!("Collection \'{}\' not found", collection_name))
        })?;

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
                return Err(not_found(format_args!(
                    "Collection \'{}\' not found",
                    collection_name
                )));
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

        // Phase 4: Update Tantivy FTS index. build_index returns facet state
        // for THIS batch only — absorb the prior batches' facets (replacing
        // them wholesale was the latent since-v0.2 facet bug).
        let tantivy_dir = store::tantivy_dir(data_dir, collection_name);
        let mut new_fts = tantivy_fts::build_index(&tantivy_dir, chunks)?;
        new_fts.facet_bitsets.absorb(&loaded.fts.facet_bitsets);
        loaded.fts = new_fts;

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
                                // A prior batch may have errored after its
                                // in-RAM adds but before a save: the on-disk
                                // file is STALE (missing committed batches
                                // whose vectors live in the mmap). Adding only
                                // the new batch and saving would bake that
                                // hole in permanently — heal from the mmap
                                // first (rows idx.size()..base_key).
                                if (idx.size() as usize) < base_key {
                                    if let Some(m) = &vs.mmap_vectors {
                                        let threads = vector::index_threads();
                                        idx.reserve_capacity_and_threads(total, threads)
                                            .map_err(|e| format!("Reserve failed: {}", e))?;
                                        for i in (idx.size() as usize)..base_key.min(m.len()) {
                                            idx.add(i as u64, m.get(i)).map_err(|e| {
                                                format!("Failed to heal index: {}", e)
                                            })?;
                                        }
                                    }
                                }
                                (idx, true)
                            }
                        };
                        let threads = vector::index_threads();
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

        // Partitioned collection: route to the partitions named by the filter.
        if let Some(field) = self.partition_field(collection_name).await? {
            return self
                .search_partitioned(collection_name, &field, req, embed_state)
                .await;
        }
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
                        let loaded = collections.get(collection_name).ok_or_else(|| {
                            not_found(format_args!("Collection \'{}\' not found", collection_name))
                        })?;
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
        let loaded = collections.get(collection_name).ok_or_else(|| {
            not_found(format_args!("Collection \'{}\' not found", collection_name))
        })?;
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
            let (raw, _, _) = tantivy_fts::search(&loaded.fts, &req.query, rerank_k)?;
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
        self.reject_if_partitioned(collection_name, "creating relations")
            .await?;
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
            let loaded = collections.get(collection_name).ok_or_else(|| {
                not_found(format_args!("Collection \'{}\' not found", collection_name))
            })?;
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
                None => Err(not_found(format_args!(
                    "Collection \'{}\' not found",
                    collection_name
                ))),
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
        self.reject_if_partitioned(collection_name, "deleting relations")
            .await?;
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
                return Err(not_found(format_args!(
                    "Collection \'{}\' not found",
                    collection_name
                )));
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
        let loaded = collections.get_mut(collection_name).ok_or_else(|| {
            not_found(format_args!("Collection \'{}\' not found", collection_name))
        })?;
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
        self.reject_if_partitioned(collection_name, "listing relations")
            .await?;
        if self.role == NodeRole::Writer {
            return Err("this node runs in writer role and does not serve queries".into());
        }
        self.ensure_attached(collection_name).await?;
        let collections = self.collections.read().await;
        let loaded = collections.get(collection_name).ok_or_else(|| {
            not_found(format_args!("Collection \'{}\' not found", collection_name))
        })?;
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
            match loaded.chunk_store.get(*id) {
                Ok(Some(c)) => loaded.filter_index.remove(*id, &filter_meta(&c)),
                other => tracing::error!(
                    "filter-index removal skipped for chunk {id}: {other:?} — universe may \
                     overcount until re-attach (results stay correct via tombstone masking)"
                ),
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
        // Partition namespaces attach on demand EVEN in non-lazy cloud mode:
        // partitions appear dynamically (a writer node can mint one at any
        // time), so "everything attached at boot" can never hold for them.
        let dynamic_partition = self.cloud_mode && partitions::is_partition_ns(ns);
        if !self.lazy_attach && !dynamic_partition {
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
                return Err(not_found(format_args!("Collection \'{}\' not found", ns)));
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
                    for name in cfg.vector_spaces.keys() {
                        if !loaded.vector_spaces.contains_key(name) {
                            loaded.vector_spaces.insert(
                                name.clone(),
                                Arc::new(VectorState {
                                    index: None,
                                    key_to_chunk_id: Vec::new(),
                                    mmap_vectors: None,
                                    vectors: Vec::new(),
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
            let loaded = collections.get(collection_name).ok_or_else(|| {
                not_found(format_args!("Collection \'{}\' not found", collection_name))
            })?;
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
            let loaded = collections.get_mut(collection_name).ok_or_else(|| {
                not_found(format_args!("Collection \'{}\' not found", collection_name))
            })?;
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
        // Ids alone don't say which partition holds them (a fan-out probe of
        // every partition would be unbounded) — partitioned collections
        // delete via POST /delete with the partition filter. This applies to
        // writer nodes too: a tombstone appended to the PARENT namespace
        // would never be materialized by any serving node.
        if self.is_partitioned_any_role(collection_name).await? {
            return Err(format!(
                "collection '{collection_name}' is partitioned: delete via filters \
                 (POST .../delete with the partition field), not bare ids"
            )
            .into());
        }
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
            let frontier = crate::storage::id_alloc::frontier(
                self.storage.as_ref(),
                partitions::alloc_ns(collection_name),
            )
            .await?;
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
            let loaded = collections.get(collection_name).ok_or_else(|| {
                not_found(format_args!("Collection \'{}\' not found", collection_name))
            })?;
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
        let loaded = collections.get_mut(collection_name).ok_or_else(|| {
            not_found(format_args!("Collection \'{}\' not found", collection_name))
        })?;
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
        // Partitioned collection: route to the partitions named by the filter
        // (same routing rule as search; NotFound partitions delete nothing).
        if let Some(field) = self.partition_field(collection_name).await? {
            let pvals = partitions::partition_values_from_filters(&field, filters)?;
            let multi = pvals.len() > 1;
            let mut total = 0usize;
            let mut last_seq = None;
            for pval in pvals {
                let ns = partitions::partition_ns(collection_name, &pval);
                match Box::pin(self.delete_by_filter(&ns, filters)).await {
                    Ok((n, seq)) => {
                        total += n;
                        last_seq = seq;
                    }
                    Err(e) if e.downcast_ref::<NotFound>().is_some() => continue,
                    Err(e) => return Err(e),
                }
            }
            return Ok((total, if multi { None } else { last_seq }));
        }
        self.ensure_attached(collection_name).await?;
        // Resolve matching live ids from the roaring filter index — the same
        // pushdown search uses, so delete-by-filter and search can never
        // disagree about what a filter matches. (This replaced a second,
        // chunk-scanning filter implementation.)
        let ids: Vec<u64> = {
            let collections = self.collections.read().await;
            let loaded = collections.get(collection_name).ok_or_else(|| {
                not_found(format_args!("Collection \'{}\' not found", collection_name))
            })?;
            let expr = crate::search::filter_pushdown::FilterExpr::compile(filters);
            loaded.filter_index.eligible(&expr).iter().collect()
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
                return Err(not_found(format_args!(
                    "Collection \'{}\' not found",
                    collection_name
                )));
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
        let fts = tantivy_fts::build_index(&tantivy_dir, &chunks)?;

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
        self.reject_if_partitioned(collection_name, "facet counting")
            .await?;
        self.ensure_attached(collection_name).await?;
        let collections = self.collections.read().await;
        let loaded = collections.get(collection_name).ok_or_else(|| {
            not_found(format_args!("Collection \'{}\' not found", collection_name))
        })?;
        loaded
            .last_used
            .store(next_lru_tick(), std::sync::atomic::Ordering::Relaxed);
        tantivy_fts::get_facets(&loaded.fts, query, fields, loaded.filter_index.universe())
    }

    /// Get all chunk texts and IDs for rebuild jobs.
    pub async fn get_all_chunk_data(
        &self,
        collection_name: &str,
    ) -> Result<(Vec<String>, Vec<u64>), Box<dyn std::error::Error + Send + Sync>> {
        let collections = self.collections.read().await;
        let loaded = collections.get(collection_name).ok_or_else(|| {
            not_found(format_args!("Collection \'{}\' not found", collection_name))
        })?;

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
        self.reject_if_partitioned(collection_name, "temporal segment lookup")
            .await?;
        let collections = self.collections.read().await;
        let loaded = collections.get(collection_name).ok_or_else(|| {
            not_found(format_args!("Collection \'{}\' not found", collection_name))
        })?;

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
mod segments_at_tests;

/// Typed "does not exist" error. The API layer downcasts to map these to
/// HTTP 404; every other engine error keeps the handler's default status.
/// (Previously a missing collection surfaced as 500 from /search and 400
/// from /ingest — stringly errors carried no classification.)
#[derive(Debug)]
pub struct NotFound(pub String);

impl std::fmt::Display for NotFound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NotFound {}

fn not_found(what: impl std::fmt::Display) -> Box<dyn std::error::Error + Send + Sync> {
    Box::new(NotFound(what.to_string()))
}

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
    let mut folded_this_run = false;
    for _ in 0..MAX_RETRIES {
        let (manifest, version) = crate::storage::lsm::read_manifest(storage, ns).await?;
        let tail: Vec<_> = manifest.uncompacted().cloned().collect();
        if tail.is_empty() {
            break;
        }
        let folded_through = tail.iter().map(|f| f.seq).max().unwrap();
        // STRICT reads: folding advances the watermark past these fragments;
        // a tolerated NotFound here would be silent data loss.
        let frags =
            crate::storage::lsm::read_uncompacted_fragments_strict(storage, ns, &manifest).await?;
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
                folded_this_run = true;
                break;
            }
            Err(crate::storage::StorageError::VersionConflict { .. }) => continue,
            Err(e) => return Err(e),
        }
    }

    // Phase 2: merge segments when they pile up (the only O(live-set) step).
    // NEVER in the same invocation as a fold: phase 1 staged the folded
    // fragments for next-cycle GC, and an immediate merge would GC them out
    // from under readers still holding the pre-fold manifest.
    if folded_this_run {
        let (manifest, _) = crate::storage::lsm::read_manifest(storage, ns).await?;
        return Ok(manifest.segments.iter().map(|s| s.records).sum());
    }
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
mod parent_metadata_tests;

#[cfg(test)]
mod persistence_tests;

#[cfg(test)]
mod validate_name_segment_tests;

#[cfg(test)]
mod filter_aware_search_tests;

#[cfg(all(test, feature = "object-storage"))]
mod cloud_ingest_tests;
