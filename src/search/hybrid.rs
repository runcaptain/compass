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

/// RRF constant. k=60 is the standard value from the original paper by Cormack et al.
const K: f32 = 60.0;

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
pub fn merge_rrf(
    fts_results: &[(u64, f32)],      // (chunk_id, bm25_score), ranked by relevance
    semantic_results: &[(u64, f32)],  // (chunk_id, cosine_score), ranked by relevance
    limit: usize,
) -> Vec<HybridResult> {
    // Accumulator: chunk_id -> (rrf_score, found_in_fts, found_in_semantic)
    let mut scores: HashMap<u64, (f32, bool, bool)> = HashMap::new();

    // Add FTS contributions: each result gets 1/(k + its rank position)
    for (rank, &(chunk_id, _bm25)) in fts_results.iter().enumerate() {
        let entry = scores.entry(chunk_id).or_insert((0.0, false, false));
        entry.0 += 1.0 / (K + rank as f32);
        entry.1 = true;
    }

    // Add semantic contributions: same formula
    for (rank, &(chunk_id, _cosine)) in semantic_results.iter().enumerate() {
        let entry = scores.entry(chunk_id).or_insert((0.0, false, false));
        entry.0 += 1.0 / (K + rank as f32);
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
