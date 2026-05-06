//! Smoke test for the GPU vector backend.
//!
//! Builds a tiny CAGRA→HNSW index, runs a query, prints latency. Useful for
//! validating that CUDA + cuVS are wired up correctly on a fresh GPU box
//! before running the full stress test.
//!
//! Run with:
//!     cargo run -p compass-vector-gpu --bin compass-gpu-smoke --release

use std::time::Instant;

use compass_index_api::{IndexParams, VectorIndex};
use compass_vector_gpu::{cuda_available, CuvsHnswIndex};

fn main() {
    if !cuda_available() {
        eprintln!("no CUDA-capable device detected");
        std::process::exit(2);
    }

    let dims = 1024;
    let n = 10_000;
    let mut rng_state = 0x9E3779B97F4A7C15u64;
    let mut next = || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state as f32 / u64::MAX as f32) * 2.0 - 1.0
    };

    // Generate random unit vectors.
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..dims).map(|_| next()).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.into_iter().map(|x| x / norm).collect()
        })
        .collect();
    let chunk_ids: Vec<u64> = (0..n as u64).collect();

    println!("Building cuVS HNSW: n={n}, dims={dims}");
    let t0 = Instant::now();
    let mut index = CuvsHnswIndex::new(IndexParams {
        dims,
        capacity: n,
        connectivity: 32,
        ef_construction: 128,
        ef_search: 64,
    })
    .expect("CuvsHnswIndex::new");

    if let Err(e) = index.build(&vectors, &chunk_ids) {
        eprintln!("build failed: {e}");
        std::process::exit(1);
    }
    println!("Built in {:.2?}", t0.elapsed());

    println!("Running 1000 queries...");
    let t0 = Instant::now();
    for i in 0..1000 {
        let q = &vectors[i % vectors.len()];
        let _ = index.search(q, 10).expect("search");
    }
    let total = t0.elapsed();
    println!(
        "1000 queries in {:.2?} ({:.0} qps single-thread)",
        total,
        1000.0 / total.as_secs_f64()
    );
}
