// embed/candle_bge.rs — BGE-small-en-v1.5 query embedder via Candle.
//
// Runs the full 6-layer BERT transformer natively in Rust. No Python, no ONNX, no sidecar.
// ~2-3ms per query on CPU, sub-1ms on GPU.
//
// The model produces 384-dimensional normalized vectors suitable for cosine similarity.
// It's loaded from HuggingFace safetensors format on disk.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// Candle-based BGE-small query embedder.
/// Wraps the BERT model, tokenizer, and device (CPU or CUDA GPU).
pub struct CandleBgeEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl CandleBgeEmbedder {
    /// Encode a text query into a 384-dim normalized embedding vector.
    /// Steps: tokenize -> run through BERT -> mean pool over token positions -> L2 normalize
    pub fn encode(&self, text: &str) -> Result<Vec<f32>, String> {
        // Tokenize the input text (adds [CLS] and [SEP] tokens automatically)
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| format!("tokenizer encode failed: {e}"))?;
        let ids = encoding.get_ids();
        let type_ids = encoding.get_type_ids();
        let seq_len = ids.len();

        let tensor_err = |e: candle_core::Error| format!("tensor op failed: {e}");

        // Convert token IDs to tensors with batch dimension of 1
        let input_ids = Tensor::new(
            ids.iter().map(|&x| x as u32).collect::<Vec<_>>().as_slice(),
            &self.device,
        )
        .map_err(tensor_err)?
        .unsqueeze(0)
        .map_err(tensor_err)?;

        let token_type_ids = Tensor::new(
            type_ids
                .iter()
                .map(|&x| x as u32)
                .collect::<Vec<_>>()
                .as_slice(),
            &self.device,
        )
        .map_err(tensor_err)?
        .unsqueeze(0)
        .map_err(tensor_err)?;

        // Run the BERT model forward pass -> output shape is [1, seq_len, 384]
        let output = self
            .model
            .forward(&input_ids, &token_type_ids, None)
            .map_err(tensor_err)?;

        let output = output.to_dtype(DType::F32).map_err(tensor_err)?;

        // Mean pooling across token positions, then L2 normalize.
        let sum = output
            .sum(1)
            .map_err(tensor_err)?
            .squeeze(0)
            .map_err(tensor_err)?;
        let count = Tensor::new(&[seq_len as f32], &self.device).map_err(tensor_err)?;
        let mean = sum.broadcast_div(&count).map_err(tensor_err)?;

        let norm = mean
            .sqr()
            .map_err(tensor_err)?
            .sum_all()
            .map_err(tensor_err)?
            .sqrt()
            .map_err(tensor_err)?;
        let normalized = mean.broadcast_div(&norm).map_err(tensor_err)?;

        normalized.to_vec1::<f32>().map_err(tensor_err)
    }
}

/// Thread-safe wrapper around the Candle embedder.
/// Uses a Mutex because the BERT model's forward pass is not thread-safe.
pub struct ThreadSafeBgeEmbedder {
    inner: Mutex<CandleBgeEmbedder>,
}

impl ThreadSafeBgeEmbedder {
    /// Encode a query string into a normalized embedding vector.
    pub fn encode(&self, text: &str) -> Result<Vec<f32>, String> {
        let embedder = self
            .inner
            .lock()
            .map_err(|e| format!("Lock poisoned: {}", e))?;
        embedder.encode(text)
    }
}

/// Try to load the BGE-small model from a directory on disk.
/// Expected files: config.json, model.safetensors, tokenizer.json
///
/// Returns None if the model files don't exist (the server will fall back to distilled M2V).
pub fn init_candle_bge(model_dir: &Path) -> Option<ThreadSafeBgeEmbedder> {
    let weights_path = model_dir.join("model.safetensors");
    let config_path = model_dir.join("config.json");
    let tokenizer_path = model_dir.join("tokenizer.json");

    if !weights_path.exists() {
        tracing::info!(
            "BGE-small model not found at {}. Download with: \
             huggingface-cli download BAAI/bge-small-en-v1.5 --local-dir {}",
            weights_path.display(),
            model_dir.display()
        );
        return None;
    }

    tracing::info!(
        "Loading BGE-small via Candle from {}...",
        model_dir.display()
    );
    let load_start = std::time::Instant::now();

    // Try CUDA GPU first, fall back to CPU
    let device = match Device::new_cuda(0) {
        Ok(d) => {
            tracing::info!("Using CUDA GPU 0 for query embedding");
            d
        }
        Err(_) => {
            tracing::info!("CUDA not available, using CPU for query embedding");
            Device::Cpu
        }
    };

    // Load model config
    let config_str = std::fs::read_to_string(&config_path).ok()?;
    let config: BertConfig = serde_json::from_str(&config_str).ok()?;

    // Load model weights from safetensors (always FP32 to avoid dtype mismatches in layer norms)
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path.to_str().unwrap()], DType::F32, &device)
            .ok()?
    };

    let model = BertModel::load(vb, &config).ok()?;

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_path.to_str().unwrap()).ok()?;

    tracing::info!(
        "BGE-small loaded in {:.3}s",
        load_start.elapsed().as_secs_f64()
    );

    let embedder = CandleBgeEmbedder {
        model,
        tokenizer,
        device,
    };

    // Warmup: run one dummy query to trigger any lazy initialization
    let _ = embedder.encode("warmup");
    tracing::info!("BGE-small warmup complete");

    Some(ThreadSafeBgeEmbedder {
        inner: Mutex::new(embedder),
    })
}
