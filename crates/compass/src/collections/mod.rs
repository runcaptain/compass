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
use crate::filter;
use crate::models::*;
use crate::scoring::{self, ScoredCandidate};
use crate::search::hybrid;
use crate::search::tantivy_fts::{self, FtsState};
use crate::search::vector::{self, VectorState};
use crate::search::SearchMode;
use chrono::Utc;
use rebuild::RebuildTracker;
use relationships::RelationshipStore;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

/// A loaded collection with all its search indices in memory.
struct LoadedCollection {
    metadata: Collection,
    fts: FtsState,
    /// Named vector spaces, each with its own USearch HNSW index.
    /// Arc-wrapped so search can clone cheaply and run in spawn_blocking.
    vector_spaces: HashMap<String, Arc<VectorState>>,
    /// Document relationships (parent-child + sibling groups)
    relationships: RelationshipStore,
    /// All chunks in memory, keyed by chunk ID for O(1) retrieval
    chunks: HashMap<u64, DocumentChunk>,
    /// Next auto-increment ID for new chunks
    next_id: u64,
}

/// Manages all collections. Thread-safe via Arc<RwLock<...>>.
pub struct CollectionManager {
    data_dir: PathBuf,
    collections: RwLock<HashMap<String, LoadedCollection>>,
    pub rebuild_tracker: RebuildTracker,
}

impl CollectionManager {
    /// Create a new manager and load existing collections from disk.
    pub async fn new(data_dir: &Path) -> Result<Arc<Self>, Box<dyn std::error::Error + Send + Sync>> {
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
    async fn load_collection(&self, name: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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
                        tracing::warn!("Failed to load vector space '{}' for '{}': {}", space_name, name, e);
                    }
                }
            } else {
                // Empty vector space (no vectors yet)
                vector_spaces.insert(space_name.clone(), Arc::new(VectorState {
                    index: None,
                    key_to_chunk_id: Vec::new(),
                    mmap_vectors: None,
                    vectors: Vec::new(),
                    dims: space_config.dims,
                }));
            }
        }

        // Load relationship store
        let rel_path = store::collection_dir(&self.data_dir, name).join("relationships.bin");
        let relationships = RelationshipStore::load(&rel_path)?;

        let chunk_count = metadata.chunk_count;
        let loaded = LoadedCollection {
            next_id: metadata.chunk_count,
            metadata,
            fts,
            vector_spaces,
            relationships,
            chunks: HashMap::new(),
        };

        let mut collections = self.collections.write().await;
        collections.insert(name.to_string(), loaded);
        tracing::info!("Loaded collection '{}' ({} chunks, {} vector spaces)",
            name, chunk_count, collections.get(name).map(|c| c.vector_spaces.len()).unwrap_or(0));

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
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
            return Err("Collection name must be kebab-case (a-z, 0-9, hyphens only)".into());
        }

        let mut collections = self.collections.write().await;
        if collections.contains_key(name) {
            return Err(format!("Collection '{}' already exists", name).into());
        }

        // Build vector spaces config: use explicit spaces, or create a "default" space
        let spaces = vector_spaces.unwrap_or_else(|| {
            let dims = embedding_dims.unwrap_or(384);
            let mut m = HashMap::new();
            m.insert("default".to_string(), VectorSpaceConfig {
                dims,
                model: "bge-small-en-v1.5".to_string(),
                status: "active".to_string(),
            });
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
            vs_map.insert(sname.clone(), Arc::new(VectorState {
                index: None,
                key_to_chunk_id: Vec::new(),
                mmap_vectors: None,
                vectors: Vec::new(),
                dims: sconfig.dims,
            }));
        }

        let loaded = LoadedCollection {
            metadata: collection.clone(),
            fts,
            vector_spaces: vs_map,
            relationships: RelationshipStore::new(),
            chunks: HashMap::new(),
            next_id: 0,
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

    pub async fn delete_collection(&self, name: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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
        let mut collections = self.collections.write().await;
        let loaded = collections
            .get_mut(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;

        if loaded.metadata.vector_spaces.contains_key(space_name) {
            return Err(format!("Vector space '{}' already exists", space_name).into());
        }

        loaded.metadata.vector_spaces.insert(space_name.to_string(), VectorSpaceConfig {
            dims,
            model: model.to_string(),
            status: "building".to_string(),
        });

        loaded.vector_spaces.insert(space_name.to_string(), Arc::new(VectorState {
            index: None,
            key_to_chunk_id: Vec::new(),
            mmap_vectors: None,
            vectors: Vec::new(),
            dims,
        }));

        store::save_metadata(&self.data_dir, &loaded.metadata)?;
        Ok(())
    }

    /// Delete a vector space from a collection.
    pub async fn delete_vector_space(
        &self,
        collection_name: &str,
        space_name: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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
        let dims = loaded.metadata.vector_spaces.get(space_name)
            .map(|c| c.dims).unwrap_or(384);

        if vecs_path.exists() {
            let vs = vector::load_vector_index(&index_path, &vecs_path, dims)?;
            loaded.vector_spaces.insert(space_name.to_string(), Arc::new(vs));
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
        let parent_refs: Vec<Option<String>> = ingest_chunks.iter().map(|ic| ic.parent_ref.clone()).collect();
        let group_ids: Vec<Option<String>> = ingest_chunks.iter().map(|ic| ic.group_id.clone()).collect();
        let resolved = RelationshipStore::resolve_batch_refs(
            &client_id_map, &parent_ids, &parent_refs, &group_ids,
        );

        // Phase 3: Build DocumentChunks and collect embeddings per vector space
        let default_space = loaded.metadata.default_vector_space.clone().unwrap_or_else(|| "default".into());
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

        // Phase 4: Update Tantivy FTS index
        let tantivy_dir = store::tantivy_dir(&self.data_dir, collection_name);
        loaded.fts = tantivy_fts::build_index(&tantivy_dir, &chunks, loaded.metadata.chunk_count)?;

        // Phase 5: Update each vector space's HNSW index
        let vectors_dir = store::vectors_dir(&self.data_dir, collection_name);
        for (space_name, new_vecs) in space_vectors {
            let dims = loaded.metadata.vector_spaces.get(&space_name)
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
                    let mut all_ids: Vec<u64> = existing.map(|e| e.key_to_chunk_id.clone()).unwrap_or_default();
                    let mut all_vecs: Vec<Vec<f32>> = existing
                        .and_then(|e| e.mmap_vectors.as_ref())
                        .map(|m| m.to_vecs())
                        .unwrap_or_default();
                    for (cid, vec) in new_vecs { all_ids.push(cid); all_vecs.push(vec); }
                    let vs = vector::build_vector_index(&index_path, &vecs_path, &all_ids, &all_vecs, dims)?;
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
                            index.load(index_path.to_str().unwrap())
                                .map_err(|e| format!("Failed to load USearch index: {}", e))?;
                        }
                        // Reserve for new vectors
                        let threads = 128.max(rayon::current_num_threads());
                        index.reserve_capacity_and_threads(total, threads)
                            .map_err(|e| format!("Reserve failed: {}", e))?;
                        // Add new vectors incrementally
                        for (i, (_, vec)) in new_vecs.iter().enumerate() {
                            index.add((base_key + i) as u64, vec)
                                .map_err(|e| format!("Failed to add vector: {}", e))?;
                        }
                        index.save(index_path.to_str().unwrap())
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
                let mut all_ids: Vec<u64> = existing.map(|e| e.key_to_chunk_id.clone()).unwrap_or_default();
                let mut all_vecs: Vec<Vec<f32>> = existing.map(|e| e.vectors.clone()).unwrap_or_default();

                for (cid, vec) in new_vecs {
                    all_ids.push(cid);
                    all_vecs.push(vec);
                }

                let vs = vector::build_vector_index(&index_path, &vecs_path, &all_ids, &all_vecs, dims)?;
                loaded.vector_spaces.insert(space_name, Arc::new(vs));
            }
        }

        // Phase 6: Save metadata + relationships
        loaded.metadata.chunk_count += count as u64;
        store::save_metadata(&self.data_dir, &loaded.metadata)?;
        let rel_path = store::collection_dir(&self.data_dir, collection_name).join("relationships.bin");
        loaded.relationships.save(&rel_path)?;

        tracing::info!("Ingested {} chunks into '{}' ({} relationships tracked)",
            count, collection_name, loaded.relationships.len());

        Ok((count, client_id_map))
    }

    // ── Search ───────────────────────────────────────────────────────────

    /// Search with full scoring pipeline: retrieve → filter → score → return.
    pub async fn search(
        &self,
        collection_name: &str,
        req: &SearchRequest,
        embed_state: &EmbedState,
    ) -> Result<(Vec<(DocumentChunk, f32, String)>, usize, u64), Box<dyn std::error::Error + Send + Sync>> {
        let start = std::time::Instant::now();
        let collections = self.collections.read().await;
        let loaded = collections
            .get(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;

        let mode = SearchMode::from_str_param(&req.mode);
        let rerank_k = req.top_k * 3; // fetch extra candidates for scoring

        // Determine which vector space to use
        let space_name = req.vector_space.as_deref()
            .or(loaded.metadata.default_vector_space.as_deref())
            .unwrap_or("default");

        // ── Step 1: Retrieve candidates ──────────────────────────────────
        let fts_results = if matches!(mode, SearchMode::Fts | SearchMode::Hybrid) {
            let (results, _, _) = tantivy_fts::search(&loaded.fts, &req.query, &HashMap::new(), rerank_k)?;
            results
        } else {
            Vec::new()
        };

        let semantic_results = if matches!(mode, SearchMode::Semantic | SearchMode::Hybrid) {
            if let Some(vs) = loaded.vector_spaces.get(space_name) {
                let query_vec_opt: Option<Vec<f32>> = req.query_vector.clone()
                    .or_else(|| embed_state.embed_query(&req.query).ok());
                if let Some(query_vec) = query_vec_opt {
                    // Clone the Arc<VectorState> cheaply and run the blocking
                    // USearch FFI call on a dedicated thread pool. This prevents
                    // the search from starving the tokio async runtime.
                    let vs_clone = vs.clone();
                    let vr = tokio::task::spawn_blocking(move || {
                        vector::search_vectors(&query_vec, &vs_clone, rerank_k)
                    }).await.unwrap_or_default();
                    vr.iter().map(|r| (r.chunk_id, r.score)).collect::<Vec<_>>()
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
                    Some(sw) => (sw.rrf_k as f32, sw.fts_weight as f32, sw.semantic_weight as f32),
                    None => (60.0, 1.0, 1.0),
                };
                let merged = hybrid::merge_rrf(&fts_results, &semantic_results, rerank_k, rrf_k, fts_w, sem_w);
                merged.iter().map(|r| ScoredCandidate {
                    chunk_id: r.chunk_id,
                    base_score: r.rrf_score,
                    final_score: r.rrf_score,
                    source: r.source.as_str().to_string(),
                }).collect()
            }
            SearchMode::Fts => {
                fts_results.iter().map(|(id, score)| ScoredCandidate {
                    chunk_id: *id, base_score: *score, final_score: *score,
                    source: "fts".to_string(),
                }).collect()
            }
            SearchMode::Semantic => {
                semantic_results.iter().map(|(id, score)| ScoredCandidate {
                    chunk_id: *id, base_score: *score, final_score: *score,
                    source: "semantic".to_string(),
                }).collect()
            }
            _ => Vec::new(),
        };

        // ── Step 3: Filter by metadata (BEFORE scoring) ─────────────────
        if !req.filters.is_empty() {
            candidates.retain(|c| {
                if let Some(chunk) = loaded.chunks.get(&c.chunk_id) {
                    filter::matches_filters(chunk, &req.filters)
                } else {
                    true
                }
            });
        }

        // ── Step 4: Apply scoring pipeline ──────────────────────────────
        // Resolve recency preset into a full config (explicit `recency` wins)
        let recency_config = req.recency.clone().or_else(|| {
            req.recency_preset.as_deref().and_then(|preset| {
                req.recency_field.as_deref().map(|field| {
                    RecencyConfig::from_preset(preset, field.to_string())
                }).flatten()
            })
        });

        let has_scoring = recency_config.is_some() || !req.boosts.is_empty() || req.relationship_boost.is_some();

        if has_scoring && !candidates.is_empty() {
            let chunk_metadata: HashMap<u64, HashMap<String, MetadataValue>> = candidates
                .iter()
                .filter_map(|c| {
                    loaded.chunks.get(&c.chunk_id).map(|chunk| {
                        (c.chunk_id, chunk.metadata.clone())
                    })
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

        let hits: Vec<(DocumentChunk, f32, String)> = candidates
            .iter()
            .filter_map(|c| {
                loaded.chunks.get(&c.chunk_id).map(|chunk| {
                    (chunk.clone(), c.final_score, c.source.clone())
                })
            })
            .collect();

        let took_us = start.elapsed().as_micros() as u64;
        Ok((hits, total, took_us))
    }

    /// Get facet counts for a collection.
    pub async fn get_facets(
        &self,
        collection_name: &str,
        query: &str,
        fields: &[String],
    ) -> Result<(HashMap<String, HashMap<String, u64>>, u64), Box<dyn std::error::Error + Send + Sync>> {
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
}
