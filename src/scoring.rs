// scoring.rs — Query-time scoring pipeline.
//
// Applies AFTER retrieval and filtering, on the top_k * 3 candidates.
// Three stages, all multiplicative:
//
//   final_score = base_score * recency_decay * metadata_boost * relationship_factor
//
// Each stage returns a multiplier (1.0 = no effect). This keeps the pipeline
// composable: skip any stage by omitting its config from the search request.

use crate::models::{BoostConfig, MetadataValue, RecencyConfig, RelationshipBoostConfig};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// A candidate result flowing through the scoring pipeline.
#[derive(Debug, Clone)]
pub struct ScoredCandidate {
    pub chunk_id: u64,
    /// Base score from retrieval (BM25, cosine, or RRF)
    pub base_score: f32,
    /// Final score after all pipeline stages
    pub final_score: f32,
    /// Source label: "fts", "semantic", or "both"
    pub source: String,
}

// ── Recency Decay ────────────────────────────────────────────────────────
// Formula: decay = max(min_score, 2^(-age_days / half_life_days))
//
// A 30-day half-life means:
//   - Today's doc:     1.0
//   - 30 days old:     0.5
//   - 60 days old:     0.25
//   - 90 days old:     0.125
//   - min_score floor:  0.1 (old docs never vanish completely)

/// Compute the recency decay multiplier for a single candidate.
/// Returns 1.0 if the chunk has no timestamp in the specified field.
pub fn recency_decay(
    metadata: &HashMap<String, MetadataValue>,
    config: &RecencyConfig,
    now: DateTime<Utc>,
) -> f64 {
    // Look up the timestamp field in metadata
    let timestamp_str = match metadata.get(&config.field) {
        Some(MetadataValue::String(s)) => s.clone(),
        _ => return 1.0, // No timestamp field, no decay
    };

    // Parse the timestamp string
    let doc_time = match timestamp_str.parse::<DateTime<Utc>>() {
        Ok(t) => t,
        Err(_) => return 1.0, // Unparseable timestamp, skip decay
    };

    // Calculate age in days (fractional)
    let age_seconds = (now - doc_time).num_seconds() as f64;
    let age_days = age_seconds / 86400.0;

    // Future timestamps get clamped to 1.0 (no boost for future docs)
    if age_days <= 0.0 {
        return 1.0;
    }

    // Exponential decay: score halves every half_life_days
    let decay = 2.0_f64.powf(-age_days / config.half_life_days);

    // Apply floor so old docs never go to zero
    decay.max(config.min_score)
}

// ── Metadata Boost ───────────────────────────────────────────────────────
// Multiplicative boost when a metadata field matches a condition.
// Multiple boosts stack multiplicatively: boost1 * boost2 * ...

/// Compute the combined metadata boost multiplier for a single candidate.
/// Returns 1.0 if no boosts match.
pub fn metadata_boost(
    metadata: &HashMap<String, MetadataValue>,
    boosts: &[BoostConfig],
) -> f64 {
    let mut factor = 1.0;

    for boost in boosts {
        let matched = match metadata.get(&boost.field) {
            None => false,
            Some(val) => {
                // Check exact string match
                if let Some(ref target) = boost.value {
                    if let Some(s) = val.as_str() {
                        s == target
                    } else {
                        false
                    }
                }
                // Check numeric range (gte and/or lte)
                else if boost.gte.is_some() || boost.lte.is_some() {
                    if let Some(num) = val.as_f64() {
                        let gte_ok = boost.gte.map_or(true, |g| num >= g);
                        let lte_ok = boost.lte.map_or(true, |l| num <= l);
                        gte_ok && lte_ok
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
        };

        if matched {
            factor *= boost.weight;
        }
    }

    factor
}

// ── Relationship Boost ───────────────────────────────────────────────────
// Boosts a candidate's score if its parent or siblings also appear in the
// result set. This rewards clusters of related relevant documents.
//
// Formula:
//   factor = 1.0
//          + parent_weight * I(parent_in_results)
//          + sibling_weight * agg(normalized_sibling_scores)
//
// I(parent_in_results) = 1.0 if the parent's chunk_id is in the candidate set
// agg = max | avg | sum of siblings' base scores (normalized to [0,1])

/// Compute the relationship boost for a single candidate.
///
/// `parent_id`: this candidate's parent chunk ID (if any)
/// `sibling_ids`: all chunk IDs that share this candidate's group_id
/// `result_scores`: map of chunk_id -> normalized base score for all candidates
pub fn relationship_boost(
    parent_id: Option<u64>,
    sibling_ids: &[u64],
    result_scores: &HashMap<u64, f32>,
    config: &RelationshipBoostConfig,
) -> f64 {
    let mut factor = 1.0;

    // Parent boost: +parent_weight if parent appears in result set
    if let Some(pid) = parent_id {
        if result_scores.contains_key(&pid) {
            factor += config.parent_weight;
        }
    }

    // Sibling boost: aggregate sibling scores from the result set
    let sibling_scores: Vec<f32> = sibling_ids
        .iter()
        .filter_map(|sid| result_scores.get(sid).copied())
        .collect();

    if !sibling_scores.is_empty() {
        let agg_score = match config.mode.as_str() {
            "avg" => {
                let sum: f32 = sibling_scores.iter().sum();
                sum / sibling_scores.len() as f32
            }
            "sum" => sibling_scores.iter().sum(),
            _ => {
                // "max" is the default
                sibling_scores.iter().cloned().fold(0.0f32, f32::max)
            }
        };
        factor += config.sibling_weight * agg_score as f64;
    }

    factor
}

// ── Full Pipeline ────────────────────────────────────────────────────────

/// Apply the full scoring pipeline to a set of candidates.
/// Candidates are modified in place (final_score updated).
///
/// `chunk_metadata`: map of chunk_id -> metadata for each candidate
/// `parent_ids`: map of chunk_id -> parent_id
/// `sibling_map`: map of chunk_id -> list of sibling chunk_ids
pub fn apply_scoring_pipeline(
    candidates: &mut Vec<ScoredCandidate>,
    chunk_metadata: &HashMap<u64, HashMap<String, MetadataValue>>,
    parent_ids: &HashMap<u64, u64>,
    sibling_map: &HashMap<u64, Vec<u64>>,
    recency: &Option<RecencyConfig>,
    boosts: &[BoostConfig],
    relationship: &Option<RelationshipBoostConfig>,
) {
    let now = Utc::now();

    // Build a map of chunk_id -> normalized base score for relationship lookups
    let max_score = candidates
        .iter()
        .map(|c| c.base_score)
        .fold(0.0f32, f32::max)
        .max(1e-10); // avoid division by zero

    let result_scores: HashMap<u64, f32> = candidates
        .iter()
        .map(|c| (c.chunk_id, c.base_score / max_score))
        .collect();

    // Apply each scoring stage to each candidate
    for candidate in candidates.iter_mut() {
        let mut multiplier = 1.0_f64;

        let meta = chunk_metadata.get(&candidate.chunk_id);

        // Stage 1: Recency decay
        if let (Some(ref rc), Some(m)) = (recency, meta) {
            multiplier *= recency_decay(m, rc, now);
        }

        // Stage 2: Metadata boost
        if !boosts.is_empty() {
            if let Some(m) = meta {
                multiplier *= metadata_boost(m, boosts);
            }
        }

        // Stage 3: Relationship boost
        if let Some(ref rc) = relationship {
            let parent = parent_ids.get(&candidate.chunk_id).copied();
            let siblings = sibling_map
                .get(&candidate.chunk_id)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            multiplier *= relationship_boost(parent, siblings, &result_scores, rc);
        }

        candidate.final_score = candidate.base_score * multiplier as f32;
    }

    // Re-sort by final score (descending)
    candidates.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}
