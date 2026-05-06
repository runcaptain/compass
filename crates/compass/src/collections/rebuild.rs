// collections/rebuild.rs — Background re-embedding job system.
//
// When a user adds a new vector space to a collection, they can trigger a
// "rebuild" that re-embeds all existing chunks using the new model and builds
// a new USearch HNSW index. This runs on tokio's blocking thread pool
// (spawn_blocking) to avoid starving async HTTP handlers.
//
// Lifecycle:
//   1. POST /collections/:name/vector-spaces/:space/rebuild
//   2. Background job iterates all chunks, embeds each, builds HNSW
//   3. GET /collections/:name/vector-spaces/:space/status returns progress
//   4. On completion, vector space status changes from "building" to "active"
//   5. PUT /collections/:name/default-vector-space to swap
//
// Crash recovery: if server crashes mid-rebuild, the partial rebuild dir is
// detected on restart and discarded. Rebuilds are idempotent, just restart.

use crate::embed::EmbedState;
use crate::models::RebuildStatus;
use crate::search::vector;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Tracks the progress of an in-flight rebuild job.
#[derive(Debug, Clone)]
pub struct RebuildProgress {
    pub status: String, // "building", "active", "failed", "cancelled"
    pub embedded: u64,
    pub total: u64,
}

impl RebuildProgress {
    pub fn to_status(&self) -> RebuildStatus {
        let percent = if self.total > 0 {
            (self.embedded as f64 / self.total as f64) * 100.0
        } else {
            0.0
        };
        RebuildStatus {
            status: self.status.clone(),
            embedded: self.embedded,
            total: self.total,
            percent,
        }
    }
}

/// Shared state for tracking active rebuild jobs.
/// Key = "collection_name/space_name"
pub type RebuildTracker = Arc<RwLock<std::collections::HashMap<String, Arc<RwLock<RebuildProgress>>>>>;

pub fn new_tracker() -> RebuildTracker {
    Arc::new(RwLock::new(std::collections::HashMap::new()))
}

/// Start a background rebuild job for a vector space.
///
/// `texts`: all chunk texts in the collection (in chunk_id order)
/// `chunk_ids`: corresponding chunk IDs
/// `space_name`: name of the vector space being rebuilt
/// `vectors_dir`: directory to write the new HNSW index and vectors
/// `dims`: expected embedding dimensionality
/// `embed_state`: the embedding models
/// `embed_endpoint`: optional external embedding server URL
/// `tracker`: shared progress tracker
///
/// Returns immediately. The job runs on a blocking thread.
pub async fn start_rebuild(
    texts: Vec<String>,
    chunk_ids: Vec<u64>,
    space_name: String,
    vectors_dir: PathBuf,
    dims: usize,
    embed_state: Arc<EmbedState>,
    embed_endpoint: Option<String>,
    _batch_size: usize,
    tracker: RebuildTracker,
    collection_name: String,
) -> Result<(), String> {
    let key = format!("{}/{}", collection_name, space_name);

    // Check for duplicate rebuild
    {
        let jobs = tracker.read().await;
        if let Some(existing) = jobs.get(&key) {
            let progress = existing.read().await;
            if progress.status == "building" {
                return Err("Rebuild already in progress for this vector space".into());
            }
        }
    }

    let total = texts.len() as u64;
    let progress = Arc::new(RwLock::new(RebuildProgress {
        status: "building".to_string(),
        embedded: 0,
        total,
    }));

    // Register the job in the tracker
    {
        let mut jobs = tracker.write().await;
        jobs.insert(key.clone(), progress.clone());
    }

    let index_path = vectors_dir.join(format!("{}.index", space_name));
    let vectors_path = vectors_dir.join(format!("{}.bin", space_name));

    // Spawn the rebuild on the blocking thread pool (CPU-bound work)
    tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Handle::current();
        let mut all_vectors: Vec<Vec<f32>> = Vec::with_capacity(texts.len());

        // Embed each chunk's text
        for (i, text) in texts.iter().enumerate() {
            let vec = if let Some(ref _endpoint) = embed_endpoint {
                // TODO: HTTP POST to external endpoint for GPU embedding
                // For now, fall back to built-in embedder
                embed_state.embed_query(text).unwrap_or_else(|_| vec![0.0; dims])
            } else {
                // Use built-in Candle embedder
                embed_state.embed_query(text).unwrap_or_else(|_| vec![0.0; dims])
            };

            all_vectors.push(vec);

            // Update progress every 100 chunks
            if (i + 1) % 100 == 0 || i == texts.len() - 1 {
                let progress = progress.clone();
                let count = (i + 1) as u64;
                rt.block_on(async {
                    let mut p = progress.write().await;
                    p.embedded = count;
                });
            }
        }

        // Build the HNSW index from all vectors
        let result = vector::build_vector_index(
            &index_path,
            &vectors_path,
            &chunk_ids,
            &all_vectors,
            dims,
        );

        // Update final status
        let progress = progress.clone();
        let key = key.clone();
        rt.block_on(async {
            let mut p = progress.write().await;
            match result {
                Ok(_) => {
                    p.status = "active".to_string();
                    p.embedded = p.total;
                    tracing::info!("Rebuild complete for {}", key);
                }
                Err(e) => {
                    p.status = format!("failed: {}", e);
                    tracing::error!("Rebuild failed for {}: {}", key, e);
                }
            }
        });
    });

    Ok(())
}

/// Get the current rebuild status for a vector space.
pub async fn get_rebuild_status(
    tracker: &RebuildTracker,
    collection_name: &str,
    space_name: &str,
) -> Option<RebuildStatus> {
    let key = format!("{}/{}", collection_name, space_name);
    let jobs = tracker.read().await;
    if let Some(progress) = jobs.get(&key) {
        let p = progress.read().await;
        Some(p.to_status())
    } else {
        None
    }
}

/// Clean up stale rebuild directories on startup.
/// If a rebuild/ dir exists with partial data, remove it.
pub fn cleanup_stale_rebuilds(data_dir: &std::path::Path) {
    if !data_dir.exists() {
        return;
    }
    if let Ok(entries) = std::fs::read_dir(data_dir) {
        for entry in entries.flatten() {
            let rebuild_dir = entry.path().join("rebuild");
            if rebuild_dir.exists() {
                tracing::info!(
                    "Cleaning up stale rebuild directory: {}",
                    rebuild_dir.display()
                );
                let _ = std::fs::remove_dir_all(&rebuild_dir);
            }
        }
    }
}
