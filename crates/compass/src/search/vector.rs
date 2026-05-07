// search/vector.rs — Vector (semantic) search via USearch HNSW index.
//
// Architecture:
//   - Pre-computed embeddings are stored alongside the HNSW index on disk
//   - The HNSW index is mmap-backed: loads in <1 second regardless of size
//   - Query embedding is done in-process via Candle BGE-small (~2-3ms) or
//     distilled Model2Vec fallback (~100μs)
//   - For datasets under 1000 docs, we skip HNSW and use brute-force cosine similarity

use std::path::Path;
use usearch::Index;
use usearch::IndexOptions;
use usearch::MetricKind;
use usearch::ScalarKind;

// ── HNSW tuning parameters ──────────────────────────────────────────────────
// These control the accuracy/speed tradeoff of the approximate nearest neighbor search.
// Higher values = more accurate but slower. These are USearch defaults (good for ~97% recall).
const HNSW_CONNECTIVITY: usize = 16; // max edges per node in the graph
const HNSW_EF_CONSTRUCTION: usize = 128; // search width during index build (higher = better graph)
const HNSW_EF_SEARCH: usize = 64; // search width during queries (higher = more accurate)
const HNSW_THRESHOLD: usize = 1000; // below this count, brute-force beats HNSW

// ── VectorState ──────────────────────────────────────────────────────────────
// Holds the HNSW index and the mapping from index keys back to chunk IDs.

pub struct VectorState {
    /// USearch HNSW index (mmap-backed, disk-persistent)
    pub index: Option<Index>,
    /// Maps HNSW key -> chunk ID. HNSW keys are sequential (0, 1, 2, ...),
    /// chunk IDs may not be (especially after deletions or multi-batch ingests).
    pub key_to_chunk_id: Vec<u64>,
    /// Memory-mapped vector storage. Replaces the old Vec<Vec<f32>> to avoid
    /// loading all vectors into RAM. Zero-copy reads via mmap.
    pub mmap_vectors: Option<super::mmap_vectors::MmapVectors>,
    /// Legacy in-memory vectors for datasets without an mmap file (e.g. first build).
    pub vectors: Vec<Vec<f32>>,
    /// Embedding dimensionality (e.g. 384 for BGE-small)
    pub dims: usize,
}

unsafe impl Send for VectorState {}
unsafe impl Sync for VectorState {}

/// Result of a vector similarity search.
#[derive(Debug, Clone)]
pub struct VectorResult {
    pub chunk_id: u64,
    /// Cosine similarity score (0.0 to 1.0, higher = more similar)
    pub score: f32,
}

/// Create a new USearch HNSW index with the given dimensions and capacity.
pub fn create_index(
    dims: usize,
    capacity: usize,
) -> Result<Index, Box<dyn std::error::Error + Send + Sync>> {
    let opts = IndexOptions {
        dimensions: dims,
        metric: MetricKind::Cos,       // cosine similarity
        quantization: ScalarKind::F32, // store vectors as 32-bit floats
        connectivity: HNSW_CONNECTIVITY,
        expansion_add: HNSW_EF_CONSTRUCTION,
        expansion_search: HNSW_EF_SEARCH,
        multi: false, // one vector per key
    };
    let index = Index::new(&opts).map_err(|e| format!("Failed to create USearch index: {}", e))?;
    if capacity > 0 {
        // Reserve enough concurrent search slots for the spawn_blocking pool.
        // Default rayon threads (=CPU count) is too low when search runs on
        // tokio's blocking pool. 128 slots costs ~256KB and avoids the
        // "No available threads to lock" fallback to brute-force.
        let threads = 128.max(rayon::current_num_threads());
        index
            .reserve_capacity_and_threads(capacity, threads)
            .map_err(|e| format!("Failed to reserve USearch capacity: {}", e))?;
    }
    Ok(index)
}

/// Build a new VectorState by inserting vectors into an HNSW index.
/// Vectors are written to disk at `index_path` via USearch's save method.
/// `vectors_path` stores the raw vectors for brute-force fallback and rebuilds.
pub fn build_vector_index(
    index_path: &Path,
    vectors_path: &Path,
    chunk_ids: &[u64],
    vectors: &[Vec<f32>],
    dims: usize,
) -> Result<VectorState, Box<dyn std::error::Error + Send + Sync>> {
    if vectors.is_empty() {
        return Ok(VectorState {
            index: None,
            key_to_chunk_id: Vec::new(),
            mmap_vectors: None,
            vectors: Vec::new(),
            dims,
        });
    }

    // Save raw vectors to mmap-backed file (replaces in-memory Vec<Vec<f32>>)
    if let Some(parent) = vectors_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mmap = super::mmap_vectors::MmapVectors::create(vectors_path, dims, vectors)?;

    // For small datasets, skip HNSW and use brute-force search
    if vectors.len() < HNSW_THRESHOLD {
        return Ok(VectorState {
            index: None,
            key_to_chunk_id: chunk_ids.to_vec(),
            mmap_vectors: Some(mmap),
            vectors: Vec::new(),
            dims,
        });
    }

    // Build the HNSW index
    let index = create_index(dims, vectors.len())?;

    // Insert vectors using parallel threads via rayon
    for (key, vec) in vectors.iter().enumerate() {
        index
            .add(key as u64, vec)
            .map_err(|e| format!("Failed to add vector {}: {}", key, e))?;
    }

    // Persist the HNSW index to disk (mmap-backed, survives restarts)
    if let Some(parent) = index_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    index
        .save(index_path.to_str().unwrap())
        .map_err(|e| format!("Failed to save USearch index: {}", e))?;

    // Save the key-to-chunk-id mapping alongside the index
    let map_path = index_path.with_extension("keymap");
    save_key_map(&map_path, chunk_ids)?;

    Ok(VectorState {
        index: Some(index),
        key_to_chunk_id: chunk_ids.to_vec(),
        mmap_vectors: Some(mmap),
        vectors: Vec::new(),
        dims,
    })
}

/// Load an existing VectorState from disk (used on server restart).
pub fn load_vector_index(
    index_path: &Path,
    vectors_path: &Path,
    dims: usize,
) -> Result<VectorState, Box<dyn std::error::Error + Send + Sync>> {
    // Load vectors via mmap (zero-copy, no RAM allocation for vector data)
    let mmap = if vectors_path.exists() {
        Some(super::mmap_vectors::MmapVectors::open(vectors_path)?)
    } else {
        // Fall back to legacy binary format
        let vecs = load_vectors(vectors_path, dims)?;
        if !vecs.is_empty() {
            // Migrate: create mmap file from legacy data
            let m = super::mmap_vectors::MmapVectors::create(vectors_path, dims, &vecs)?;
            Some(m)
        } else {
            None
        }
    };

    let count = mmap.as_ref().map(|m| m.len()).unwrap_or(0);

    // Load the key-to-chunk-id mapping
    let map_path = index_path.with_extension("keymap");
    let key_to_chunk_id = load_key_map(&map_path)?;

    // For small datasets, skip HNSW
    if count < HNSW_THRESHOLD {
        return Ok(VectorState {
            index: None,
            key_to_chunk_id,
            mmap_vectors: mmap,
            vectors: Vec::new(),
            dims,
        });
    }

    // Load the HNSW index via mmap (near-instant regardless of index size)
    if index_path.exists() {
        let index = create_index(dims, 0)?;
        index
            .view(index_path.to_str().unwrap())
            .map_err(|e| format!("Failed to mmap USearch index: {}", e))?;

        Ok(VectorState {
            index: Some(index),
            key_to_chunk_id,
            mmap_vectors: mmap,
            vectors: Vec::new(),
            dims,
        })
    } else {
        Ok(VectorState {
            index: None,
            key_to_chunk_id,
            mmap_vectors: mmap,
            vectors: Vec::new(),
            dims,
        })
    }
}

/// Search for the most similar vectors to a query vector.
/// Uses HNSW for large datasets (sub-ms), brute-force cosine for small ones.
pub fn search_vectors(query_vec: &[f32], state: &VectorState, top_k: usize) -> Vec<VectorResult> {
    // Try HNSW index first (fast approximate search)
    if let Some(ref index) = state.index {
        match index.search(query_vec, top_k) {
            Ok(matches) => {
                return matches
                    .keys
                    .iter()
                    .zip(matches.distances.iter())
                    .map(|(&key, &distance)| {
                        let chunk_id = state
                            .key_to_chunk_id
                            .get(key as usize)
                            .copied()
                            .unwrap_or(key);
                        VectorResult {
                            chunk_id,
                            // USearch cosine distance is 1 - similarity, so we invert it
                            score: 1.0 - distance,
                        }
                    })
                    .collect();
            }
            Err(e) => {
                tracing::warn!("USearch search failed: {}, falling back to brute-force", e);
            }
        }
    }

    // Brute-force fallback: compute cosine similarity against all vectors
    let mut scores: Vec<(usize, f32)> = if let Some(ref mmap) = state.mmap_vectors {
        mmap.iter()
            .enumerate()
            .map(|(i, v)| {
                let score: f32 = query_vec.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                (i, score)
            })
            .collect()
    } else {
        state
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let score: f32 = query_vec.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                (i, score)
            })
            .collect()
    };

    // Sort by score descending (highest similarity first)
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    scores
        .into_iter()
        .take(top_k)
        .map(|(i, score)| {
            let chunk_id = state.key_to_chunk_id.get(i).copied().unwrap_or(i as u64);
            VectorResult { chunk_id, score }
        })
        .collect()
}

// ── Persistence helpers ──────────────────────────────────────────────────────
// Simple binary formats for saving/loading vectors and key maps to disk.

/// Save vectors to a binary file.
/// Format: [u32 count] [u32 dims] [count * dims * f32 values]
fn save_vectors(
    path: &Path,
    vectors: &[Vec<f32>],
    dims: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let count = vectors.len();
    let mut buf: Vec<u8> = Vec::with_capacity(8 + count * dims * 4);
    buf.extend_from_slice(&(count as u32).to_le_bytes());
    buf.extend_from_slice(&(dims as u32).to_le_bytes());
    for vec in vectors {
        for &val in vec {
            buf.extend_from_slice(&val.to_le_bytes());
        }
    }
    std::fs::write(path, buf)?;
    Ok(())
}

/// Load vectors from a binary file.
fn load_vectors(
    path: &Path,
    expected_dims: usize,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let buf = std::fs::read(path)?;
    if buf.len() < 8 {
        return Err("vectors file too small".into());
    }
    let count = u32::from_le_bytes(buf[0..4].try_into()?) as usize;
    let dims = u32::from_le_bytes(buf[4..8].try_into()?) as usize;
    if dims != expected_dims {
        return Err(format!(
            "dimension mismatch: file has {} but expected {}",
            dims, expected_dims
        )
        .into());
    }

    let mut vectors = Vec::with_capacity(count);
    let mut pos = 8;
    for _ in 0..count {
        let mut vec = Vec::with_capacity(dims);
        for _ in 0..dims {
            let val = f32::from_le_bytes(buf[pos..pos + 4].try_into()?);
            vec.push(val);
            pos += 4;
        }
        vectors.push(vec);
    }
    Ok(vectors)
}

/// Save key-to-chunk-id mapping. Format: [u32 count] [count * u64 chunk_ids]
pub fn save_key_map(
    path: &Path,
    chunk_ids: &[u64],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut buf: Vec<u8> = Vec::with_capacity(4 + chunk_ids.len() * 8);
    buf.extend_from_slice(&(chunk_ids.len() as u32).to_le_bytes());
    for &id in chunk_ids {
        buf.extend_from_slice(&id.to_le_bytes());
    }
    std::fs::write(path, buf)?;
    Ok(())
}

/// Load key-to-chunk-id mapping from disk.
fn load_key_map(path: &Path) -> Result<Vec<u64>, Box<dyn std::error::Error + Send + Sync>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let buf = std::fs::read(path)?;
    if buf.len() < 4 {
        return Err("keymap file too small".into());
    }
    let count = u32::from_le_bytes(buf[0..4].try_into()?) as usize;
    let mut ids = Vec::with_capacity(count);
    let mut pos = 4;
    for _ in 0..count {
        let id = u64::from_le_bytes(buf[pos..pos + 8].try_into()?);
        ids.push(id);
        pos += 8;
    }
    Ok(ids)
}
