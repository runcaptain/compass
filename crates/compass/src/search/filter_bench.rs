// search/filter_bench.rs — Recall + latency benchmark for filter-aware ANN.
//
// Marked #[ignore] so it doesn't slow down regular `cargo test`. Run with:
//   cargo test -p compass --bin compass --release filter_aware_ann_recall_100k -- --ignored --nocapture
//
// The benchmark answers the design question: does USearch's native
// filtered_search hit ≥ 0.95 recall@10 at 1% selectivity over 100k vectors?

use std::collections::HashMap;
use std::time::Instant;

use crate::models::{FilterValue, MetadataValue};
use crate::search::filter_index::{selectivity, FilterIndex};
use crate::search::filter_pushdown::FilterExpr;
use crate::search::vector::{create_index, search_vectors_filtered, VectorResult, VectorState};

const DIMS: usize = 64;

/// Deterministic pseudo-random unit vector. Seeds + LCG so the benchmark is
/// reproducible without pulling in `rand` for one test file.
fn pseudo_vec(seed: u64, dims: usize) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut v = Vec::with_capacity(dims);
    for _ in 0..dims {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map u64 -> [-1.0, 1.0).
        let bits = (state >> 11) as f32;
        v.push((bits / (1u64 << 53) as f32) * 2.0 - 1.0);
    }
    // L2 normalize so cosine distances behave.
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Build a synthetic VectorState + FilterIndex pair with `n` vectors. The
/// `bucket` metadata field is set so that exactly 10% / 1% / 0.1% of chunks
/// hit each respective bucket.
fn build_corpus(n: u32) -> (VectorState, FilterIndex) {
    let index = create_index(DIMS, n as usize).expect("create_index");
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n as usize);
    let mut chunk_ids: Vec<u64> = Vec::with_capacity(n as usize);
    let mut filter_index = FilterIndex::new();
    for i in 0..n {
        let v = pseudo_vec(i as u64 + 1, DIMS);
        index.add(i as u64, &v).expect("add vector");
        vectors.push(v);
        chunk_ids.push(i as u64);
        let bucket = if i % 10 == 0 {
            "ten" // 10%
        } else {
            "rest"
        };
        let bucket1 = if i % 100 == 0 { "one" } else { "rest" }; // 1%
        let bucket01 = if i % 1000 == 0 { "tenth" } else { "rest" }; // 0.1%
        let mut metadata = HashMap::new();
        metadata.insert(
            "bucket10".to_string(),
            MetadataValue::String(bucket.to_string()),
        );
        metadata.insert(
            "bucket1".to_string(),
            MetadataValue::String(bucket1.to_string()),
        );
        metadata.insert(
            "bucket01".to_string(),
            MetadataValue::String(bucket01.to_string()),
        );
        filter_index.insert(i, &metadata);
    }
    filter_index.finalize();
    let state = VectorState {
        index: Some(index),
        key_to_chunk_id: chunk_ids,
        mmap_vectors: None,
        vectors,
        dims: DIMS,
    };
    (state, filter_index)
}

/// Brute-force ground truth over the eligible set.
fn ground_truth(
    query: &[f32],
    state: &VectorState,
    eligible_keys: &[u32],
    top_k: usize,
) -> Vec<u64> {
    let mut scored: Vec<(u64, f32)> = eligible_keys
        .iter()
        .map(|&k| {
            let v = &state.vectors[k as usize];
            let score: f32 = query.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            (k as u64, score)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(top_k).map(|(k, _)| k).collect()
}

fn recall_at_k(got: &[VectorResult], gt: &[u64]) -> f64 {
    if gt.is_empty() {
        return 1.0;
    }
    let got_set: std::collections::HashSet<u64> = got.iter().map(|r| r.chunk_id).collect();
    let hit = gt.iter().filter(|id| got_set.contains(id)).count();
    hit as f64 / gt.len() as f64
}

fn run_case(
    state: &VectorState,
    idx: &FilterIndex,
    bucket_field: &str,
    bucket_value: &str,
    label: &str,
) {
    let top_k = 10;
    let queries = 20;
    let mut total_recall = 0.0;
    let mut latencies = Vec::with_capacity(queries);
    let mut inspected_total = 0u64;

    let mut filters = HashMap::new();
    filters.insert(
        bucket_field.to_string(),
        FilterValue::Exact(MetadataValue::String(bucket_value.to_string())),
    );
    let expr = FilterExpr::compile(&filters);
    let eligible = idx.eligible(&expr);
    let eligible_count = eligible.len();
    let selectivity_val = selectivity(&eligible, idx.len());
    let eligible_keys: Vec<u32> = eligible.iter().collect();

    for q_seed in 0..queries {
        let query = pseudo_vec(900_000 + q_seed as u64, DIMS);

        let start = Instant::now();
        let (hits, explain) = search_vectors_filtered(&query, state, top_k, &eligible);
        latencies.push(start.elapsed());
        inspected_total += explain.candidates_inspected;

        let gt = ground_truth(&query, state, &eligible_keys, top_k);
        total_recall += recall_at_k(&hits, &gt);
    }

    latencies.sort();
    let p50 = latencies[queries / 2];
    let p95 = latencies[(queries as f64 * 0.95) as usize];
    let avg_recall = total_recall / queries as f64;
    let avg_inspected = inspected_total / queries as u64;

    println!(
        "  {label:18} eligible={eligible_count:>6} selectivity={selectivity_val:>6.3} \
         recall@10={avg_recall:>5.3} p50={p50:?} p95={p95:?} avg_candidates={avg_inspected}"
    );
}

#[test]
#[ignore]
fn filter_aware_ann_recall_100k() {
    println!();
    println!("Building 100K vector corpus ({DIMS}-dim)...");
    let build_start = Instant::now();
    let (state, idx) = build_corpus(100_000);
    println!("  corpus built in {:?}", build_start.elapsed());
    println!("Filter-aware ANN recall + latency:");
    run_case(&state, &idx, "bucket10", "ten", "10% selectivity");
    run_case(&state, &idx, "bucket1", "one", "1% selectivity");
    run_case(&state, &idx, "bucket01", "tenth", "0.1% selectivity");
}

#[test]
fn filter_aware_ann_recall_10k_smoke() {
    // Smaller corpus that runs as part of the regular test sweep. Confirms
    // wiring works end-to-end (filter pushdown -> bitmap -> usearch -> recall).
    let (state, idx) = build_corpus(10_000);
    let mut filters = HashMap::new();
    filters.insert(
        "bucket1".to_string(),
        FilterValue::Exact(MetadataValue::String("one".into())),
    );
    let expr = FilterExpr::compile(&filters);
    let eligible = idx.eligible(&expr);
    assert_eq!(eligible.len(), 100, "1% of 10k -> 100 eligible chunks");
    let eligible_keys: Vec<u32> = eligible.iter().collect();
    let query = pseudo_vec(424242, DIMS);
    let (hits, explain) = search_vectors_filtered(&query, &state, 10, &eligible);
    assert_eq!(hits.len(), 10);
    for hit in &hits {
        assert!(
            eligible.contains(hit.chunk_id as u32),
            "hit {} must be eligible",
            hit.chunk_id
        );
    }
    let gt = ground_truth(&query, &state, &eligible_keys, 10);
    let recall = recall_at_k(&hits, &gt);
    assert!(
        recall >= 0.9,
        "recall@10 was {recall}, expected >= 0.9 at 1% selectivity"
    );
    assert!(explain.used_hnsw, "expected HNSW path on 10k vectors");
}
