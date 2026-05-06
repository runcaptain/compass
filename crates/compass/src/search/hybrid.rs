// search/hybrid.rs — Reciprocal Rank Fusion (RRF) for combining FTS + semantic results.
//
// RRF merges two ranked result lists by computing a combined score:
//   score(doc) = 1/(k + rank_fts) + 1/(k + rank_semantic)
// where k = 60 (standard constant from the original RRF paper).
//
// A document that appears at rank 0 in both lists gets:
//   1/(60+0) + 1/(60+0) = 0.0333
// A document that only appears at rank 0 in FTS:
//   1/(60+0) + 0 = 0.0167
//
// This naturally rewards documents found by BOTH search methods.

use std::collections::HashMap;

/// A merged search result with source tracking.
#[derive(Debug, Clone)]
pub struct HybridResult {
    /// Chunk ID from the document store
    pub chunk_id: u64,
    /// Combined RRF score (higher = more relevant)
    pub rrf_score: f32,
    /// Which search method(s) contributed this result
    pub source: ResultSource,
}

/// Tracks whether a result came from full-text, semantic, or both search methods.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResultSource {
    Fts,
    Semantic,
    Both,
}

impl ResultSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            ResultSource::Fts => "fts",
            ResultSource::Semantic => "semantic",
            ResultSource::Both => "both",
        }
    }
}

/// Merge FTS and semantic results using Reciprocal Rank Fusion.
///
/// Both input lists should be ordered by relevance (position 0 = best match).
/// Returns merged results sorted by combined RRF score, truncated to `limit`.
///
/// `rrf_k`: RRF constant (default 60.0). Lower values amplify top-rank differences.
/// `fts_weight` / `semantic_weight`: relative contribution weights (default 1.0 each).
pub fn merge_rrf(
    fts_results: &[(u64, f32)],
    semantic_results: &[(u64, f32)],
    limit: usize,
    rrf_k: f32,
    fts_weight: f32,
    semantic_weight: f32,
) -> Vec<HybridResult> {
    let mut scores: HashMap<u64, (f32, bool, bool)> = HashMap::new();

    for (rank, &(chunk_id, _bm25)) in fts_results.iter().enumerate() {
        let entry = scores.entry(chunk_id).or_insert((0.0, false, false));
        entry.0 += fts_weight * (1.0 / (rrf_k + rank as f32));
        entry.1 = true;
    }

    for (rank, &(chunk_id, _cosine)) in semantic_results.iter().enumerate() {
        let entry = scores.entry(chunk_id).or_insert((0.0, false, false));
        entry.0 += semantic_weight * (1.0 / (rrf_k + rank as f32));
        entry.2 = true;
    }

    // Convert to HybridResults and sort by combined score
    let mut results: Vec<HybridResult> = scores
        .into_iter()
        .map(|(chunk_id, (rrf_score, from_fts, from_semantic))| {
            let source = match (from_fts, from_semantic) {
                (true, true) => ResultSource::Both,
                (true, false) => ResultSource::Fts,
                (false, true) => ResultSource::Semantic,
                (false, false) => unreachable!(),
            };
            HybridResult { chunk_id, rrf_score, source }
        })
        .collect();

    // Sort by RRF score descending (best matches first)
    results.sort_by(|a, b| b.rrf_score.partial_cmp(&a.rrf_score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);
    results
}
