// search/mod.rs — Search module: FTS + Vector + Hybrid.
//
// Three search modes, all targeting sub-millisecond latency:
//   FTS:      Tantivy inverted index with BM25 scoring
//   Semantic: USearch HNSW approximate nearest neighbor search
//   Hybrid:   Both combined via Reciprocal Rank Fusion (RRF, k=60)

#[allow(dead_code)]
pub mod backend;
#[allow(dead_code)]
pub mod chunk_store;
// Filter-aware ANN modules . Not yet wired into the API
// surface; `search_vectors_filtered` below is the prototype call site.
#[cfg(test)]
mod filter_bench;
#[allow(dead_code)]
pub mod filter_index;
#[allow(dead_code)]
pub mod filter_pushdown;
pub mod hybrid;
#[allow(dead_code)]
pub mod mmap_vectors;
pub mod tantivy_fts;
pub mod vector;

// Re-export the stable trait surface for external consumers and future use.
#[allow(unused_imports)]
pub use backend::{
    build_backend, IndexError, IndexParams, LoadableIndex, UsearchHnswIndex, VectorIndex,
    VectorMatch,
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
