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
struct LoadedCollection {
    metadata: Collection,
    fts: FtsState,
    /// Named vector spaces, each with its own USearch HNSW index.
    /// Arc-wrapped so search can clone cheaply and run in spawn_blocking.
    vector_spaces: HashMap<String, Arc<VectorState>>,
    /// Document relationships (parent-child + sibling groups)
    relationships: RelationshipStore,
    /// All chunks in memory, keyed by chunk ID for O(1) retrieval. This is a
    /// hot cache; the disk source of truth is `chunk_store`. Populated on
    /// startup from `chunk_store.for_each` and kept in sync on every ingest.
    chunks: HashMap<u64, DocumentChunk>,
    /// Disk-backed chunk metadata. Every ingest writes through to this redb
    /// database so chunks survive process restarts and crashes.
    chunk_store: ChunkStore,
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
        std::fs::create_dir_all(data_dir)?;

        // Clean up any stale rebuild directories from crashes
        rebuild::cleanup_stale_rebuilds(data_dir);

        let cloud_mode = storage.backend_name() != "local-disk";
        let manager = Arc::new(Self {
            data_dir: data_dir.to_path_buf(),
            collections: RwLock::new(HashMap::new()),
            rebuild_tracker: rebuild::new_tracker(),
            storage,
            cloud_mode,
            compacting: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        });

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
                }
                Err(e) => tracing::error!("Could not list collections from object storage: {}", e),
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
        let chunk_store = ChunkStore::open(&chunks_db)?;
        let mut chunks: HashMap<u64, DocumentChunk> = HashMap::new();
        let mut max_seen_id: u64 = 0;
        chunk_store.for_each(|id, chunk| {
            if id >= max_seen_id {
                max_seen_id = id;
            }
            chunks.insert(id, chunk);
        })?;
        let rehydrated_count = chunks.len();
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
        let tombstones: std::collections::HashSet<u64> =
            chunk_store.load_tombstones()?.into_iter().collect();
        let filter_index = build_filter_index_from_chunks(&chunks, &tombstones);
        let relations_db = store::relations_db_path(&self.data_dir, name);
        let relation_store = RelationStore::open(&relations_db)?;
        let loaded = LoadedCollection {
            next_id,
            metadata,
            fts,
            vector_spaces,
            relationships,
            chunks,
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
            let chunk_store = ChunkStore::open(&chunks_db)?;
            let relations_db = store::relations_db_path(&self.data_dir, name);
            let relation_store = RelationStore::open(&relations_db)?;

            let loaded = LoadedCollection {
                metadata: collection.clone(),
                fts,
                vector_spaces: vs_map,
                relationships: RelationshipStore::new(),
                chunks: HashMap::new(),
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
                Ok(()) => {}
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
        let collections = self.collections.read().await;
        collections.values().map(|c| c.metadata.clone()).collect()
    }

    pub async fn get_collection(&self, name: &str) -> Option<Collection> {
        let collections = self.collections.read().await;
        collections.get(name).map(|c| c.metadata.clone())
    }

    pub async fn delete_collection(
        &self,
        name: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut collections = self.collections.write().await;
        if collections.remove(name).is_none() {
            return Err(format!("Collection '{}' not found", name).into());
        }
        store::delete_collection_data(&self.data_dir, name)?;
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
    pub async fn ingest(
        &self,
        collection_name: &str,
        ingest_chunks: Vec<IngestChunk>,
        embed_state: &EmbedState,
    ) -> Result<(usize, HashMap<String, u64>), Box<dyn std::error::Error + Send + Sync>> {
        let mut collections = self.collections.write().await;
        let loaded = collections
            .get_mut(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;

        let count = ingest_chunks.len();

        // Phase 1: Assign IDs and build client_id -> chunk_id map
        let mut client_id_map: HashMap<String, u64> = HashMap::new();
        let mut assigned_ids: Vec<u64> = Vec::with_capacity(count);

        for ic in &ingest_chunks {
            let id = loaded.next_id;
            loaded.next_id += 1;
            assigned_ids.push(id);
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
            Some(l) => l,
            None => {
                // Collection was deleted in the lock gap. `delete_collection`
                // purges S3, but our fragment may have landed after that purge —
                // append a tombstone so a re-materialize (which would recreate a
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
                    loaded.chunks.remove(id);
                }
                loaded.filter_index =
                    build_filter_index_from_chunks(&loaded.chunks, &loaded.tombstones);
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

        Ok((count, client_id_map))
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
        for chunk in chunks {
            loaded.chunks.insert(chunk.id, chunk.clone());
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

                    // Add to HNSW index (use load() for mutability, not view())
                    let total = vs.key_to_chunk_id.len();
                    if total >= 1000 && (vs.index.is_none() || index_path.exists()) {
                        let index_path_str = index_path
                            .to_str()
                            .ok_or("USearch index path is not valid UTF-8")?;
                        let index = vector::create_index(dims, total)?;
                        if index_path.exists() {
                            index
                                .load(index_path_str)
                                .map_err(|e| format!("Failed to load USearch index: {}", e))?;
                        }
                        // Reserve for new vectors
                        let threads = 128.max(rayon::current_num_threads());
                        index
                            .reserve_capacity_and_threads(total, threads)
                            .map_err(|e| format!("Reserve failed: {}", e))?;
                        // Add new vectors incrementally
                        for (i, (_, vec)) in new_vecs.iter().enumerate() {
                            index
                                .add((base_key + i) as u64, vec)
                                .map_err(|e| format!("Failed to add vector: {}", e))?;
                        }
                        index
                            .save(index_path_str)
                            .map_err(|e| format!("Failed to save index: {}", e))?;
                        vs.index = Some(index);
                    }
                    Ok(())
                })();
                // Space goes back in whatever happened; a partial update is
                // recoverable (caller compensates the batch), a vanished space
                // is a silent outage.
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
        loaded.filter_index = build_filter_index_from_chunks(&loaded.chunks, &loaded.tombstones);
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
        let start = std::time::Instant::now();
        let collections = self.collections.read().await;
        let loaded = collections
            .get(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;

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
            let chunk_metadata: HashMap<u64, HashMap<String, MetadataValue>> = candidates
                .iter()
                .filter_map(|c| {
                    loaded
                        .chunks
                        .get(&c.chunk_id)
                        .map(|chunk| (c.chunk_id, chunk.metadata.clone()))
                })
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
        let parent_meta_cache = build_parent_metadata_cache(&candidate_chunk_ids, &loaded.chunks);

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
                    edge.target_status = if loaded.chunks.contains_key(&edge.target_chunk_id)
                        && !loaded.tombstones.contains(&edge.target_chunk_id)
                    {
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
                loaded.chunks.get(&c.chunk_id).map(|chunk| {
                    let parent_metadata = parent_metadata_for(chunk, &parent_meta_cache);
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
                    target_status: if loaded.chunks.contains_key(&r.target_chunk_id)
                        && !loaded.tombstones.contains(&r.target_chunk_id)
                    {
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
        if self.cloud_mode && !built.is_empty() {
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

        // Phase 3 (read lock): apply locally (durable S3 record already written).
        // On EITHER failure mode — collection deleted in the lock gap, or a
        // local insert error — compensate with a RelationDelete for the minted
        // ids, so the durable upsert fragment can't resurrect orphan edges on a
        // later materialize (the same discipline ingest applies to chunks).
        let apply_result: Result<(), Box<dyn std::error::Error + Send + Sync>> = {
            let collections = self.collections.read().await;
            match collections.get(collection_name) {
                Some(loaded) => loaded.relation_store.insert_batch(&built),
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
        if self.cloud_mode {
            crate::storage::lsm::append_relation_delete(
                self.storage.as_ref(),
                collection_name,
                std::slice::from_ref(&relation_id.to_string()),
            )
            .await
            .map_err(|e| format!("cloud relation-delete append failed: {e}"))?;
        }

        // Apply locally.
        let collections = self.collections.read().await;
        let loaded = collections
            .get(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
        loaded.relation_store.delete(relation_id)
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
        let collections = self.collections.read().await;
        let loaded = collections
            .get(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
        let mut edges = loaded
            .relation_store
            .for_chunk(chunk_id, direction, types)?;
        for edge in edges.iter_mut() {
            edge.target_status = if loaded.chunks.contains_key(&edge.target_chunk_id)
                && !loaded.tombstones.contains(&edge.target_chunk_id)
            {
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
    pub async fn delete_chunks(
        &self,
        collection_name: &str,
        ids: &[u64],
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
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
                .filter(|id| {
                    seen.insert(*id)
                        && loaded.chunks.contains_key(id)
                        && !loaded.tombstones.contains(id)
                })
                .collect()
        }; // read lock released here.
        if newly.is_empty() {
            return Ok(0);
        }

        // Phase 2 (NO lock held): DURABLE S3 tombstone FIRST. This is the S3
        // network round-trip; doing it without the collections lock means a slow
        // S3 call no longer stalls every other collection's reads/writes (#2).
        // S3-first also fixes the F5 split-brain: on failure nothing local is
        // committed, so the caller retries cleanly.
        if self.cloud_mode {
            crate::storage::lsm::append_tombstone(self.storage.as_ref(), collection_name, &newly)
                .await
                .map_err(|e| format!("LSM tombstone append failed (delete not applied): {e}"))?;
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
        let apply: Vec<u64> = newly
            .iter()
            .copied()
            .filter(|id| loaded.chunks.contains_key(id) && !loaded.tombstones.contains(id))
            .collect();
        if apply.is_empty() {
            return Ok(0);
        }

        loaded.chunk_store.tombstone_batch(&apply)?;
        for id in &apply {
            loaded.tombstones.insert(*id);
        }
        // Persist the corrected live count IMMEDIATELY after the tombstones —
        // before the fallible relation pruning — so a pruning error can't leave
        // chunk_count permanently overstated.
        let removed = apply.len() as u64;
        loaded.metadata.chunk_count = loaded.metadata.chunk_count.saturating_sub(removed);
        store::save_metadata(&self.data_dir, &loaded.metadata)?;
        // Keep the filter index in step with the tombstones so `eligible` /
        // selectivity don't count deleted chunks (which would underfill top-k
        // on deleted-heavy collections).
        loaded.filter_index = build_filter_index_from_chunks(&loaded.chunks, &loaded.tombstones);
        // Prune relations incident on the deleted chunks (F6: propagate errors;
        // on failure the edges are orphaned but target_status reports their
        // endpoints as missing, and cloud replay prunes them independently).
        for &id in &apply {
            let edges = loaded
                .relation_store
                .for_chunk(id, RelationDirection::Both, None)?;
            for e in edges {
                loaded.relation_store.delete(&e.relation_id)?;
            }
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
        Ok(apply.len())
    }

    /// Soft-delete every chunk matching a metadata filter (e.g. all chunks of a
    /// file_id, or a metadata predicate). Resolves matching ids, then delegates
    /// to `delete_chunks`.
    pub async fn delete_by_filter(
        &self,
        collection_name: &str,
        filters: &HashMap<String, FilterValue>,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        // Collect matching, not-yet-deleted ids under a read lock first.
        let ids: Vec<u64> = {
            let collections = self.collections.read().await;
            let loaded = collections
                .get(collection_name)
                .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
            loaded
                .chunks
                .values()
                .filter(|c| !loaded.tombstones.contains(&c.id))
                .filter(|c| crate::filter::matches_filters(c, filters))
                .map(|c| c.id)
                .collect()
        };
        if ids.is_empty() {
            return Ok(0);
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
        let chunk_store = ChunkStore::open(&chunks_db)?;
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
        let chunk_map: HashMap<u64, DocumentChunk> =
            chunks.iter().map(|c| (c.id, c.clone())).collect();
        // Materialized state is already live-only (tombstones applied on replay).
        let filter_index =
            build_filter_index_from_chunks(&chunk_map, &std::collections::HashSet::new());

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
            metadata,
            fts,
            vector_spaces: vs_map,
            relationships,
            chunks: chunk_map,
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
        let collections = self.collections.read().await;
        let loaded = collections
            .get(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
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
        for (&id, chunk) in &loaded.chunks {
            ids.push(id);
            texts.push(chunk.text.clone());
        }
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

        let mut results: Vec<DocumentChunk> = loaded
            .chunks
            .values()
            .filter(|c| !loaded.tombstones.contains(&c.id))
            .filter(|c| c.doc_type == "segment")
            .filter(|c| c.group_id.as_deref() == Some(asset))
            .filter(|c| segment_in_time_window(c, time_ms, time_start_ms, time_end_ms))
            .cloned()
            .collect();

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
    const MAX_RETRIES: u32 = 10;
    for _ in 0..MAX_RETRIES {
        let (manifest, version) = crate::storage::lsm::read_manifest(storage, ns).await?;
        if manifest.segments.is_empty() && manifest.fragments.is_empty() {
            return Ok(0);
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
                tracing::info!(
                    "Compacted '{}': {} live records in one segment",
                    ns,
                    records
                );
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
    chunks: &HashMap<u64, DocumentChunk>,
) -> HashMap<u64, HashMap<String, MetadataValue>> {
    let mut cache: HashMap<u64, HashMap<String, MetadataValue>> = HashMap::new();
    for cid in candidate_chunk_ids {
        let Some(chunk) = chunks.get(cid) else {
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
        if let Some(parent) = chunks.get(&pid) {
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

    fn into_map(chunks: Vec<DocumentChunk>) -> HashMap<u64, DocumentChunk> {
        chunks.into_iter().map(|c| (c.id, c)).collect()
    }

    #[test]
    fn segment_with_parent_gets_metadata() {
        let chunks = into_map(vec![
            source_with_meta(1, "title", "Keynote"),
            segment(2, Some(1)),
        ]);
        let cache = build_parent_metadata_cache(&[2], &chunks);
        let meta = parent_metadata_for(chunks.get(&2).unwrap(), &cache);
        assert_eq!(
            meta.unwrap().get("title"),
            Some(&MetadataValue::String("Keynote".to_string()))
        );
    }

    #[test]
    fn source_hit_gets_none() {
        let chunks = into_map(vec![source_with_meta(1, "title", "Keynote")]);
        let cache = build_parent_metadata_cache(&[1], &chunks);
        let meta = parent_metadata_for(chunks.get(&1).unwrap(), &cache);
        assert!(meta.is_none());
    }

    #[test]
    fn segment_without_parent_gets_none() {
        let chunks = into_map(vec![segment(2, None)]);
        let cache = build_parent_metadata_cache(&[2], &chunks);
        let meta = parent_metadata_for(chunks.get(&2).unwrap(), &cache);
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
            let meta = parent_metadata_for(chunks.get(&cid).unwrap(), &cache);
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
        let meta = parent_metadata_for(chunks.get(&5).unwrap(), &cache);
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
        let meta = parent_metadata_for(chunks.get(&21).unwrap(), &cache);
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
        let meta = parent_metadata_for(chunks.get(&2).unwrap(), &cache);
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
            let (ingested, _) = manager
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
            let n = manager.delete_chunks("del", &[3]).await.unwrap();
            assert_eq!(n, 1);
            // Re-deleting is a no-op.
            assert_eq!(manager.delete_chunks("del", &[3]).await.unwrap(), 0);

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
            assert_eq!(deleted, 9);
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
        let n = manager.delete_chunks("delcloud", &[1]).await.unwrap();
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

        // After: one segment, no fragments.
        let (after, _) = read_manifest(storage.as_ref(), "comp").await.unwrap();
        assert_eq!(after.segments.len(), 1);
        assert!(after.fragments.is_empty());

        // The compacted segment contains only live chunks (0, 2) — deleted 1 gone.
        let seg =
            crate::storage::lsm::read_segment(storage.as_ref(), "comp", &after.segments[0].id)
                .await
                .unwrap();
        let segment: cloud::Segment = serde_json::from_slice(&seg).unwrap();
        let ids: std::collections::HashSet<u64> = segment.chunks.iter().map(|c| c.id).collect();
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

        // Compact twice more (each cycle GCs the PRIOR cycle's staged objects,
        // deferred one cycle for in-flight readers). After enough cycles, S1 is
        // physically gone — the key point is it's GC'd, not leaked forever.
        for i in 2..5u32 {
            m.ingest("gc", vec![ingest_chunk(i)], &embed).await.unwrap();
            m.compact_collection("gc").await.unwrap();
        }
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
        assert!(
            all.len() <= 5,
            "object count must stay bounded across cycles, got {}",
            all.len()
        );
        // Data intact.
        let mat = cloud::materialize(storage.as_ref(), "gc", &man2)
            .await
            .unwrap();
        assert_eq!(mat.chunks.len(), 5);

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
        assert!(
            mat.chunks.contains_key(&4),
            "new chunk must take id 4 (one past the pre-compaction high-water), got ids {:?}",
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
            assert_eq!(m.delete_chunks("pdisk", &[1]).await.unwrap(), 1);
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
}
