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

pub mod rebuild;
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
use chrono::Utc;
use rebuild::RebuildTracker;
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
}

impl CollectionManager {
    /// Create a new manager and load existing collections from disk.
    pub async fn new(
        data_dir: &Path,
    ) -> Result<Arc<Self>, Box<dyn std::error::Error + Send + Sync>> {
        std::fs::create_dir_all(data_dir)?;

        // Clean up any stale rebuild directories from crashes
        rebuild::cleanup_stale_rebuilds(data_dir);

        let manager = Arc::new(Self {
            data_dir: data_dir.to_path_buf(),
            collections: RwLock::new(HashMap::new()),
            rebuild_tracker: rebuild::new_tracker(),
        });

        // Load existing collections from disk
        let names = store::list_collection_names(data_dir)?;
        for name in &names {
            if let Err(e) = manager.load_collection(name).await {
                tracing::error!("Failed to load collection '{}': {}", name, e);
            }
        }

        if !names.is_empty() {
            tracing::info!("Loaded {} collection(s) from disk", names.len());
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
        // next_id is max(seen) + 1 if any chunks exist, otherwise resume from
        // the metadata's chunk_count. The +1 guards against deleted-id gaps
        // (no delete-chunk API today, but cheap insurance).
        let next_id = if rehydrated_count > 0 {
            max_seen_id + 1
        } else {
            metadata.chunk_count
        };

        let chunk_count = metadata.chunk_count;
        let filter_index = build_filter_index_from_chunks(&chunks);
        let loaded = LoadedCollection {
            next_id,
            metadata,
            fts,
            vector_spaces,
            relationships,
            chunks,
            chunk_store,
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

        let loaded = LoadedCollection {
            metadata: collection.clone(),
            fts,
            vector_spaces: vs_map,
            relationships: RelationshipStore::new(),
            chunks: HashMap::new(),
            chunk_store,
            next_id: 0,
            filter_index: FilterIndex::new(),
        };

        collections.insert(name.to_string(), loaded);
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

        let mut collections = self.collections.write().await;
        let loaded = collections
            .get_mut(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;

        if loaded.metadata.vector_spaces.contains_key(space_name) {
            return Err(format!("Vector space '{}' already exists", space_name).into());
        }

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

        let mut collections = self.collections.write().await;
        let loaded = collections
            .get_mut(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;

        // Don't delete the default vector space
        if loaded.metadata.default_vector_space.as_deref() == Some(space_name) {
            return Err("Cannot delete the default vector space. Switch default first.".into());
        }

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
        let mut collections = self.collections.write().await;
        let loaded = collections
            .get_mut(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;

        if !loaded.metadata.vector_spaces.contains_key(space_name) {
            return Err(format!("Vector space '{}' not found", space_name).into());
        }

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
            // If no embeddings provided at all, compute using built-in embedder for default space
            if embeddings.is_empty() {
                if let Ok(emb) = embed_state.embed_query(&ic.text) {
                    embeddings.insert(default_space.clone(), emb);
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

            // Add relationship
            loaded.relationships.add(id, parent_id, group_id);

            loaded.chunks.insert(id, chunk.clone());
            chunks.push(chunk);
        }

        // Phase 3b: Persist chunks to the disk-backed store BEFORE updating
        // FTS/HNSW. If this write fails we error out before any index commits,
        // so we never end up with a Tantivy or HNSW index referencing chunks
        // that don't exist on disk. redb writes are atomic per batch.
        let to_persist: Vec<(u64, DocumentChunk)> =
            chunks.iter().map(|c| (c.id, c.clone())).collect();
        loaded.chunk_store.insert_batch(&to_persist)?;

        // Phase 4: Update Tantivy FTS index
        let tantivy_dir = store::tantivy_dir(&self.data_dir, collection_name);
        loaded.fts = tantivy_fts::build_index(&tantivy_dir, &chunks, loaded.metadata.chunk_count)?;

        // Phase 5: Update each vector space's HNSW index
        let vectors_dir = store::vectors_dir(&self.data_dir, collection_name);
        for (space_name, new_vecs) in space_vectors {
            let dims = loaded
                .metadata
                .vector_spaces
                .get(&space_name)
                .map(|c| c.dims)
                .unwrap_or(384);

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

                // Append new vectors to mmap file
                if let Some(ref mut mmap) = vs.mmap_vectors {
                    mmap.append(&new_vecs)?;
                }

                // Extend key mapping
                let base_key = vs.key_to_chunk_id.len();
                for (cid, _) in &new_vecs {
                    vs.key_to_chunk_id.push(*cid);
                }

                // Add to HNSW index (use load() for mutability, not view())
                let total = vs.key_to_chunk_id.len();
                if total >= 1000 {
                    if vs.index.is_none() || index_path.exists() {
                        let index = vector::create_index(dims, total)?;
                        if index_path.exists() {
                            index
                                .load(index_path.to_str().unwrap())
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
                            .save(index_path.to_str().unwrap())
                            .map_err(|e| format!("Failed to save index: {}", e))?;
                        vs.index = Some(index);
                    }
                }

                // Save updated keymap
                let map_path = index_path.with_extension("keymap");
                vector::save_key_map(&map_path, &vs.key_to_chunk_id)?;

                loaded.vector_spaces.insert(space_name, Arc::new(vs));
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

        // Phase 6: Save metadata + relationships, then rebuild the filter index
        loaded.metadata.chunk_count += count as u64;
        store::save_metadata(&self.data_dir, &loaded.metadata)?;
        let rel_path =
            store::collection_dir(&self.data_dir, collection_name).join("relationships.bin");
        loaded.relationships.save(&rel_path)?;
        loaded.filter_index = build_filter_index_from_chunks(&loaded.chunks);

        tracing::info!(
            "Ingested {} chunks into '{}' ({} relationships tracked)",
            count,
            collection_name,
            loaded.relationships.len()
        );

        Ok((count, client_id_map))
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
                    .filter(|(id, _)| {
                        u32::try_from(*id)
                            .map(|k| eligible.contains(k))
                            .unwrap_or(false)
                    })
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
            !filter_active
                || candidates.iter().all(|c| {
                    u32::try_from(c.chunk_id)
                        .map(|k| eligible.contains(k))
                        .unwrap_or(false)
                }),
            "filter-aware retrieval produced a candidate outside the eligible bitmap"
        );

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

        let hits: Vec<(
            DocumentChunk,
            f32,
            String,
            Option<HashMap<String, MetadataValue>>,
        )> = candidates
            .iter()
            .filter_map(|c| {
                loaded.chunks.get(&c.chunk_id).map(|chunk| {
                    let parent_metadata = parent_metadata_for(chunk, &parent_meta_cache);
                    (
                        chunk.clone(),
                        c.final_score,
                        c.source.clone(),
                        parent_metadata,
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
pub(crate) fn build_filter_index_from_chunks(chunks: &HashMap<u64, DocumentChunk>) -> FilterIndex {
    let mut idx = FilterIndex::new();
    for (&chunk_id, chunk) in chunks {
        let Ok(key) = u32::try_from(chunk_id) else {
            tracing::warn!(
                "Chunk id {} exceeds u32; skipping FilterIndex insert. a follow-up widens this to u64.",
                chunk_id
            );
            continue;
        };
        let mut effective = chunk.metadata.clone();
        // doc_type is a struct field, not a metadata key, but the filter
        // language treats it as one. Mirror it here so the bitmap covers it.
        effective.insert(
            "doc_type".to_string(),
            MetadataValue::String(chunk.doc_type.clone()),
        );
        idx.insert(key, &effective);
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
        };
        let (hits, total, _took_us, explain) =
            manager.search("filter-search", &req, &embed).await.unwrap();

        assert!(!hits.is_empty(), "search returned no hits");
        for (chunk, _, _, _) in &hits {
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
        };
        let (_hits, _total, _took, explain) =
            manager.search("no-explain", &req, &embed).await.unwrap();
        assert!(explain.is_none(), "explain must be None when not requested");

        let _ = std::fs::remove_dir_all(&data_dir);
    }
}
