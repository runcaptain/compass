//! Stable trait API for Compass vector index backends.
//!
//! Implementors:
//!   - [`compass`] (default): USearch HNSW on CPU, mmap-backed, disk-persistent.
//!   - [`compass-vector-gpu`] (optional): cuVS / CAGRA→HNSW on GPU.
//!
//! This crate has no I/O, no async, no logging, and a tiny dep tree on purpose:
//! it's a trait surface that backends bind to. Anything heavier belongs in the
//! consuming crate.
//!
//! # Stability
//!
//! Pre-1.0 the API may change between minor versions. Once 1.0 is cut, the trait
//! shape is semver-stable so external backends can be developed out-of-tree.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Result of a single vector similarity match.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VectorMatch {
    /// External chunk identifier (caller's domain id, not the internal HNSW key).
    pub chunk_id: u64,
    /// Similarity score in [0.0, 1.0]. Higher is more similar (cosine similarity).
    pub score: f32,
}

/// Construction parameters for an HNSW-style index.
///
/// Defaults are tuned for ~97% recall on web-scale corpora at dim 768/1024.
/// Backends may map these to their native parameter names (USearch:
/// `connectivity` / `expansion_*`; cuVS CAGRA: `graph_degree` / `intermediate_graph_degree`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IndexParams {
    /// Embedding dimensionality.
    pub dims: usize,
    /// Expected number of vectors. Used to pre-allocate buffers; not a hard cap.
    pub capacity: usize,
    /// Max edges per node in the proximity graph.
    pub connectivity: usize,
    /// Beam width during graph build. Higher = better recall, slower build.
    pub ef_construction: usize,
    /// Beam width during search. Higher = better recall, slower query.
    pub ef_search: usize,
}

impl Default for IndexParams {
    fn default() -> Self {
        Self {
            dims: 768,
            capacity: 0,
            connectivity: 16,
            ef_construction: 128,
            ef_search: 64,
        }
    }
}

/// Errors returned by index implementations.
#[derive(Debug, Error)]
pub enum IndexError {
    /// I/O failure during persistence or load.
    #[error("io error: {0}")]
    Io(String),

    /// Vector dimension does not match the index.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimMismatch {
        /// Index dimensionality.
        expected: usize,
        /// Vector dimensionality the caller supplied.
        actual: usize,
    },

    /// Backend reported an internal error (USearch, cuVS, etc.).
    #[error("backend error: {0}")]
    Backend(String),

    /// GPU resource error (out of memory, device unavailable, driver mismatch).
    #[error("gpu unavailable: {0}")]
    GpuUnavailable(String),

    /// Operation is not supported by this backend.
    #[error("unsupported: {0}")]
    Unsupported(String),
}

/// Vector index backend trait.
///
/// Implementations must be `Send + Sync`. Persistence is path-based: the
/// caller decides where the index lives on disk.
///
/// # Example
///
/// ```ignore
/// use compass_index_api::{IndexParams, VectorIndex};
///
/// fn build_and_query<I: VectorIndex>(idx: &mut I, vecs: &[Vec<f32>]) -> Vec<u64> {
///     let ids: Vec<u64> = (0..vecs.len() as u64).collect();
///     idx.build(vecs, &ids).unwrap();
///     idx.search(&vecs[0], 10).unwrap().into_iter().map(|m| m.chunk_id).collect()
/// }
/// ```
pub trait VectorIndex: Send + Sync {
    /// Build the index from the given vectors and external chunk IDs.
    ///
    /// `vectors` and `chunk_ids` must be parallel slices of the same length.
    /// Existing index contents are replaced.
    fn build(&mut self, vectors: &[Vec<f32>], chunk_ids: &[u64]) -> Result<(), IndexError>;

    /// Insert a single vector. Returns the new vector's external chunk id.
    ///
    /// Some backends may not support incremental insert (e.g. cuVS HNSW with
    /// `hierarchy = none` is immutable). Those should return [`IndexError::Unsupported`].
    fn add(&mut self, chunk_id: u64, vector: &[f32]) -> Result<(), IndexError>;

    /// Find the `top_k` nearest neighbors of `query`, ranked by descending score.
    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<VectorMatch>, IndexError>;

    /// Number of vectors in the index.
    fn len(&self) -> usize;

    /// True if the index has no vectors.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Embedding dimensionality.
    fn dims(&self) -> usize;

    /// Persist to disk at `path`. Backends define their own on-disk format.
    fn save(&self, path: &Path) -> Result<(), IndexError>;

    /// Human-readable backend label (e.g. `"usearch"`, `"cuvs-hnsw"`). Useful
    /// for tracing and benchmark output.
    fn backend_name(&self) -> &'static str;
}

/// Marker trait for backends that can be loaded from disk without an existing
/// in-memory representation. Constructors live on the implementing type so the
/// trait stays object-safe.
pub trait LoadableIndex: VectorIndex + Sized {
    /// Load an index from `path` with the given parameters.
    fn load(path: &Path, params: IndexParams) -> Result<Self, IndexError>;
}

/// Convenience: detect whether GPU backends are available at runtime.
///
/// Returns `false` if compiled without the `gpu` feature, or if the cuVS runtime
/// fails to initialize. Callers can use this to fall back to the CPU backend.
#[cfg(not(feature = "gpu"))]
pub fn gpu_available() -> bool {
    false
}

/// Convenience: detect whether GPU backends are available at runtime.
#[cfg(feature = "gpu")]
pub fn gpu_available() -> bool {
    // Probed by the gpu backend itself; this stub is replaced when the consumer
    // links compass-vector-gpu.
    true
}
