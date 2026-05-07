// embed/distilled.rs — Distilled Model2Vec embedder for sub-100μs query embedding.
//
// This is the fast fallback when the full BGE-small model isn't available.
// It uses a "distilled" model: just a static embedding lookup table.
//
// How it works:
//   1. Tokenize the query into WordPiece token IDs
//   2. Look up each token's pre-computed embedding row (from model.safetensors)
//   3. Average all the rows (mean pooling)
//   4. L2 normalize the result
//
// The lookup table is ~5MB and loads in <100ms. Query embedding takes ~50-100μs.
// Quality is lower than the full transformer, but good enough for most queries.

use half::f16;
use safetensors::SafeTensors;
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// Distilled Model2Vec embedder: tokenize -> lookup -> average -> normalize.
struct DistilledEmbedder {
    tokenizer: Tokenizer,
    /// Pre-computed embedding matrix: embeddings[token_id] = Vec<f32> of length `dims`.
    /// Loaded from safetensors at startup, converted from FP16 to FP32.
    embeddings: Vec<Vec<f32>>,
    dims: usize,
}

impl DistilledEmbedder {
    /// Encode a query string into a normalized embedding vector.
    fn encode(&self, text: &str) -> Vec<f32> {
        let encoding = self.tokenizer.encode(text, false).unwrap();
        let ids = encoding.get_ids();

        if ids.is_empty() {
            return vec![0.0; self.dims];
        }

        // Mean pool: sum the embedding rows for each token, then divide by count
        let mut sum = vec![0.0f32; self.dims];
        let mut count = 0usize;
        for &id in ids {
            if let Some(row) = self.embeddings.get(id as usize) {
                for (s, &v) in sum.iter_mut().zip(row.iter()) {
                    *s += v;
                }
                count += 1;
            }
        }

        // Average
        if count > 0 {
            let inv = 1.0 / count as f32;
            for s in sum.iter_mut() {
                *s *= inv;
            }
        }

        // L2 normalize so cosine similarity = dot product
        let norm: f32 = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for s in sum.iter_mut() {
                *s /= norm;
            }
        }

        sum
    }
}

/// Thread-safe wrapper around the distilled embedder.
pub struct ThreadSafeDistilledEmbedder {
    inner: Mutex<DistilledEmbedder>,
}

impl ThreadSafeDistilledEmbedder {
    /// Encode a query string into a normalized embedding vector (~50-100μs).
    pub fn encode(&self, text: &str) -> Result<Vec<f32>, String> {
        let embedder = self
            .inner
            .lock()
            .map_err(|e| format!("Lock poisoned: {}", e))?;
        Ok(embedder.encode(text))
    }
}

/// Load the distilled Model2Vec model from a directory on disk.
/// Expected files: tokenizer.json, model.safetensors (with an "embeddings" tensor).
///
/// Returns None if the model files don't exist.
pub fn init_distilled(model_dir: &Path) -> Option<ThreadSafeDistilledEmbedder> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    let safetensors_path = model_dir.join("model.safetensors");

    if !safetensors_path.exists() {
        tracing::info!(
            "Distilled model not found at {}. Run distill_bge.py to create it.",
            safetensors_path.display()
        );
        return None;
    }

    tracing::info!(
        "Loading distilled query embedder from {}...",
        model_dir.display()
    );
    let load_start = std::time::Instant::now();

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_path.to_str().unwrap()).ok()?;

    // Load embedding matrix from safetensors (stored as FP16, we convert to FP32)
    let safetensors_data = std::fs::read(&safetensors_path).ok()?;
    let tensors = SafeTensors::deserialize(&safetensors_data).ok()?;
    let tensor = tensors.tensor("embeddings").ok()?;

    let shape = tensor.shape();
    let vocab_size = shape[0];
    let dims = shape[1];

    // Convert FP16 bytes to FP32 embedding matrix
    let fp16_data = tensor.data();
    let fp16_slice: &[f16] = unsafe {
        std::slice::from_raw_parts(fp16_data.as_ptr() as *const f16, fp16_data.len() / 2)
    };

    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(vocab_size);
    for row_idx in 0..vocab_size {
        let start = row_idx * dims;
        let row: Vec<f32> = fp16_slice[start..start + dims]
            .iter()
            .map(|&v| v.to_f32())
            .collect();
        embeddings.push(row);
    }

    tracing::info!(
        "Loaded {} token embeddings ({} dims) in {:.3}s",
        vocab_size,
        dims,
        load_start.elapsed().as_secs_f64()
    );

    Some(ThreadSafeDistilledEmbedder {
        inner: Mutex::new(DistilledEmbedder {
            tokenizer,
            embeddings,
            dims,
        }),
    })
}
