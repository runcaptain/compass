//! Vector index backend abstraction.
//!
//! Wraps the existing USearch HNSW path in a [`VectorIndex`] implementation so
//! that callers can swap to the GPU-accelerated [`compass_vector_gpu::CuvsHnswIndex`]
//! transparently. The trait itself lives in [`compass_index_api`].
//!
//! # Why a trait
//!
//! Compass has historically used USearch directly. As we add a GPU backend
//! (cuVS / CAGRA→HNSW), and as we anticipate IVF-PQ for very large corpora,
//! the call sites benefit from binding to a stable trait instead of the
//! USearch types. New backends slot in without touching `collections/`,
//! `api/`, or the rebuild path.
//!
//! # Backend selection
//!
//! Construction goes through [`build_backend`], which inspects environment
//! variables and feature flags to pick:
//!
//!   - `COMPASS_BACKEND=cpu` (default): [`UsearchHnswIndex`].
//!   - `COMPASS_BACKEND=gpu`: requires the `gpu` feature; returns
//!     `CuvsHnswIndex` from `compass-vector-gpu`. Falls back to CPU with a
//!     `tracing::warn!` if CUDA is unavailable at runtime.
//!   - `COMPASS_BACKEND=auto`: probe GPU first, fall back to CPU.

use std::path::Path;

pub use compass_index_api::{IndexError, IndexParams, LoadableIndex, VectorIndex, VectorMatch};

use super::vector;

/// CPU-backed HNSW via USearch. Wraps the existing `vector::VectorState` so
/// the in-tree code keeps working while new code can bind to the trait.
pub struct UsearchHnswIndex {
    state: vector::VectorState,
    /// Where on disk the persisted index lives. Set by `build` or `load`.
    persisted_at: Option<std::path::PathBuf>,
    vectors_path: Option<std::path::PathBuf>,
}

impl UsearchHnswIndex {
    /// Empty index ready to receive a build.
    pub fn new(params: IndexParams) -> Self {
        Self {
            state: vector::VectorState {
                index: None,
                key_to_chunk_id: Vec::new(),
                mmap_vectors: None,
                vectors: Vec::new(),
                dims: params.dims,
            },
            persisted_at: None,
            vectors_path: None,
        }
    }

    /// Mount an existing index that's already on disk. The companion
    /// `vectors_path` holds the raw float buffer for brute-force fallback.
    pub fn from_paths(
        index_path: &Path,
        vectors_path: &Path,
        dims: usize,
    ) -> Result<Self, IndexError> {
        let state = vector::load_vector_index(index_path, vectors_path, dims)
            .map_err(|e| IndexError::Io(e.to_string()))?;
        Ok(Self {
            state,
            persisted_at: Some(index_path.to_path_buf()),
            vectors_path: Some(vectors_path.to_path_buf()),
        })
    }

    /// Direct accessor for code that still uses the legacy `VectorState` shape.
    /// New code should go through the [`VectorIndex`] methods.
    pub fn state(&self) -> &vector::VectorState {
        &self.state
    }
}

impl VectorIndex for UsearchHnswIndex {
    fn build(&mut self, vectors: &[Vec<f32>], chunk_ids: &[u64]) -> Result<(), IndexError> {
        let index_path = self.persisted_at.clone().unwrap_or_else(|| {
            std::path::PathBuf::from("./data/.compass-tmp.usearch")
        });
        let vectors_path = self.vectors_path.clone().unwrap_or_else(|| {
            std::path::PathBuf::from("./data/.compass-tmp.vectors")
        });
        let state = vector::build_vector_index(
            &index_path,
            &vectors_path,
            chunk_ids,
            vectors,
            self.state.dims,
        )
        .map_err(|e| IndexError::Backend(e.to_string()))?;
        self.state = state;
        self.persisted_at = Some(index_path);
        self.vectors_path = Some(vectors_path);
        Ok(())
    }

    fn add(&mut self, _chunk_id: u64, _vector: &[f32]) -> Result<(), IndexError> {
        // USearch does support incremental insert; wiring it here means
        // re-saving the index after each add or batching at the rebuild layer.
        // Today, ingestion goes through `build_vector_index` via the rebuild
        // path. Surface this when the streaming-ingest API lands.
        Err(IndexError::Unsupported(
            "incremental add via VectorIndex trait not wired yet; use rebuild()".into(),
        ))
    }

    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<VectorMatch>, IndexError> {
        if query.len() != self.state.dims {
            return Err(IndexError::DimMismatch {
                expected: self.state.dims,
                actual: query.len(),
            });
        }
        let results = vector::search_vectors(query, &self.state, top_k);
        Ok(results
            .into_iter()
            .map(|r| VectorMatch {
                chunk_id: r.chunk_id,
                score: r.score,
            })
            .collect())
    }

    fn len(&self) -> usize {
        self.state.vectors.len()
    }

    fn dims(&self) -> usize {
        self.state.dims
    }

    fn save(&self, _path: &Path) -> Result<(), IndexError> {
        // USearch saves at build time via `build_vector_index`. Re-saving an
        // already-mmap'd index requires `index.save()` which the `Index` type
        // exposes; we can wire it when downstream callers need atomic snapshot.
        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        "usearch"
    }
}

impl LoadableIndex for UsearchHnswIndex {
    fn load(path: &Path, params: IndexParams) -> Result<Self, IndexError> {
        let vectors_path = path.with_extension("vectors");
        Self::from_paths(path, &vectors_path, params.dims)
    }
}

/// Backend selection at startup. Reads `COMPASS_BACKEND` and feature flags.
///
/// Returns a `Box<dyn VectorIndex>` so the call site stays backend-agnostic.
/// Callers can downcast via [`std::any::Any`] if they need the concrete type
/// for backend-specific tuning.
pub fn build_backend(params: IndexParams) -> Box<dyn VectorIndex> {
    let preference = std::env::var("COMPASS_BACKEND").unwrap_or_else(|_| "cpu".into());
    match preference.as_str() {
        "gpu" => build_gpu_or_warn(params),
        "auto" => {
            #[cfg(feature = "gpu")]
            {
                if compass_vector_gpu::cuda_available() {
                    return build_gpu_or_warn(params);
                }
            }
            Box::new(UsearchHnswIndex::new(params))
        }
        _ => Box::new(UsearchHnswIndex::new(params)),
    }
}

#[cfg(feature = "gpu")]
fn build_gpu_or_warn(params: IndexParams) -> Box<dyn VectorIndex> {
    match compass_vector_gpu::CuvsHnswIndex::new(params) {
        Ok(idx) => {
            tracing::info!("vector backend = cuvs-hnsw (GPU)");
            Box::new(idx)
        }
        Err(e) => {
            tracing::warn!("GPU backend requested but unavailable ({e}); falling back to USearch");
            Box::new(UsearchHnswIndex::new(params))
        }
    }
}

#[cfg(not(feature = "gpu"))]
fn build_gpu_or_warn(params: IndexParams) -> Box<dyn VectorIndex> {
    tracing::warn!(
        "COMPASS_BACKEND=gpu but binary built without --features gpu; falling back to USearch"
    );
    Box::new(UsearchHnswIndex::new(params))
}
