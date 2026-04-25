// embed/mod.rs — Embedding module: BGE-small (full quality) + distilled M2V (fast fallback).
//
// Two embedding strategies, chosen at runtime based on what model files are available:
//   BGE-small (Candle):  Full 6-layer BERT transformer, ~2-3ms/query, best quality
//   Distilled Model2Vec: Static lookup table, ~50-100μs/query, lower quality fallback

pub mod candle_bge;
pub mod distilled;

use candle_bge::ThreadSafeBgeEmbedder;
use distilled::ThreadSafeDistilledEmbedder;
use std::path::Path;

/// Holds whichever embedding models are available on this system.
/// At least one must be loaded for semantic/hybrid search to work.
pub struct EmbedState {
    /// Full BGE-small transformer (preferred, ~2-3ms per query)
    pub bge: Option<ThreadSafeBgeEmbedder>,
    /// Distilled Model2Vec (fallback, ~50-100μs per query)
    pub distilled: Option<ThreadSafeDistilledEmbedder>,
}

impl EmbedState {
    /// Embed a query string using the best available model.
    /// Tries BGE-small first, falls back to distilled M2V.
    /// Returns an error if no embedding model is loaded.
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>, String> {
        // Try BGE-small first (full transformer, best quality)
        if let Some(ref bge) = self.bge {
            match bge.encode(text) {
                Ok(vec) if !vec.iter().all(|&x| x == 0.0) => return Ok(vec),
                _ => {} // fall through to distilled
            }
        }

        // Fall back to distilled M2V (static lookup, faster but lower quality)
        if let Some(ref distilled) = self.distilled {
            match distilled.encode(text) {
                Ok(vec) if !vec.iter().all(|&x| x == 0.0) => return Ok(vec),
                _ => {}
            }
        }

        Err("No embedding model available. Download BGE-small or create a distilled model.".into())
    }
}

/// Initialize the embedding state by loading whatever models are available on disk.
///
/// Looks for models in the `models/` subdirectory of `data_dir`:
///   - `models/bge-small/` for the full BGE-small transformer
///   - `models/distilled/` for the distilled Model2Vec lookup table
pub fn init_embedders(data_dir: &Path) -> EmbedState {
    let bge_dir = data_dir.join("models").join("bge-small");
    let distilled_dir = data_dir.join("models").join("distilled");

    let bge = candle_bge::init_candle_bge(&bge_dir);
    let distilled = distilled::init_distilled(&distilled_dir);

    if bge.is_none() && distilled.is_none() {
        tracing::warn!(
            "No embedding models found. Semantic search will be unavailable. \
             Download BGE-small: huggingface-cli download BAAI/bge-small-en-v1.5 --local-dir {}",
            bge_dir.display()
        );
    }

    EmbedState { bge, distilled }
}
