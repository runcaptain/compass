// search/mod.rs — Search module: FTS + Vector + Hybrid.
//
// Three search modes, all targeting sub-millisecond latency:
//   FTS:      Tantivy inverted index with BM25 scoring
//   Semantic: USearch HNSW approximate nearest neighbor search
//   Hybrid:   Both combined via Reciprocal Rank Fusion (RRF, k=60)

pub mod backend;
pub mod chunk_store;
pub mod hybrid;
pub mod mmap_vectors;
pub mod tantivy_fts;
pub mod vector;

// Re-export the stable trait surface. Internal callers (and the `compass`
// library) bind to this so backends swap without touching call sites.
pub use backend::{
    build_backend, IndexError, IndexParams, LoadableIndex, UsearchHnswIndex,
    VectorIndex, VectorMatch,
};

/// Search mode — determines which search engines are used for a query.
#[derive(Debug, Clone, Copy)]
pub enum SearchMode {
    Fts,
    Semantic,
    Hybrid,
}

impl SearchMode {
    /// Parse a search mode from a string parameter (e.g. from a JSON request).
    /// Defaults to Hybrid if the string doesn't match a known mode.
    pub fn from_str_param(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "fts" => SearchMode::Fts,
            "semantic" => SearchMode::Semantic,
            _ => SearchMode::Hybrid,
        }
    }
}
