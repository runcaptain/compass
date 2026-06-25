// search/filter_index.rs — Per-collection filter index (roaring bitmaps + sorted ranges).
//
// Goal: turn an AND of equality/range/contains/in predicates into ONE roaring-
// bitmap intersection over chunk-id sets. The output is the eligible candidate
// set passed into USearch's filter callback, so the HNSW walk only visits
// eligible nodes.
//
// Storage shape (v0):
//   - equality:   (field, canonical_string) -> RoaringBitmap of chunk_ids
//   - numeric:    field -> Vec<(value: f64, chunk_id)> sorted by value
//   - string_list:(field, element) -> RoaringBitmap (for `contains`)
//   - present:    field -> RoaringBitmap of chunk_ids that have any value
//
// Inserts are batch: build once, query many times. The structures rebuild
// alongside the HNSW index after each ingest batch — same lifecycle, same
// disk-flush cadence as the rest of the search state. Persistence wiring is
// out of scope for the initial implementation.

use std::collections::HashMap;

use roaring::RoaringBitmap;

use crate::models::MetadataValue;
use crate::search::filter_pushdown::{FilterExpr, Predicate};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum MetadataKey {
    Bool(bool),
    Int(i64),
    Float(u64),
    String(String),
    StringList(Vec<String>),
}

impl MetadataKey {
    fn from_metadata(value: &MetadataValue) -> Self {
        match value {
            MetadataValue::Bool(b) => Self::Bool(*b),
            MetadataValue::Int(i) => Self::Int(*i),
            MetadataValue::Float(f) => Self::Float(f.to_bits()),
            MetadataValue::String(s) => Self::String(s.clone()),
            MetadataValue::StringList(xs) => Self::StringList(xs.clone()),
        }
    }
}

#[derive(Default)]
pub struct FilterIndex {
    /// field -> typed value -> chunk_ids that match exactly.
    equality: HashMap<String, HashMap<MetadataKey, RoaringBitmap>>,
    /// field -> string value -> chunk_ids (for `in` semantics on strings).
    equality_strings: HashMap<String, HashMap<String, RoaringBitmap>>,
    /// field -> sorted (value, chunk_id) for range predicates.
    numeric: HashMap<String, Vec<(f64, u32)>>,
    /// field -> element -> chunk_ids whose StringList contains the element.
    string_list_contains: HashMap<String, HashMap<String, RoaringBitmap>>,
    /// field -> chunk_ids that have any value for this field.
    present: HashMap<String, RoaringBitmap>,
    /// Universe of all known chunk ids. Used as the starting set for empty
    /// expressions and as a fallback when a predicate spans the whole field.
    universe: RoaringBitmap,
}

impl FilterIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> u64 {
        self.universe.len()
    }

    pub fn is_empty(&self) -> bool {
        self.universe.is_empty()
    }

    /// Insert a single chunk with its metadata. Caller is responsible for
    /// passing chunk_ids that fit in u32 (USearch keys are u64 but real
    /// collections won't exceed 4B chunks per partition in the foreseeable
    /// future; a follow-up tracks widening to u64).
    pub fn insert(&mut self, chunk_id: u32, metadata: &HashMap<String, MetadataValue>) {
        self.universe.insert(chunk_id);
        for (field, value) in metadata {
            self.present
                .entry(field.clone())
                .or_default()
                .insert(chunk_id);
            self.equality
                .entry(field.clone())
                .or_default()
                .entry(MetadataKey::from_metadata(value))
                .or_default()
                .insert(chunk_id);
            if let MetadataValue::String(s) = value {
                self.equality_strings
                    .entry(field.clone())
                    .or_default()
                    .entry(s.clone())
                    .or_default()
                    .insert(chunk_id);
            }
            if let Some(n) = value.as_f64() {
                self.numeric
                    .entry(field.clone())
                    .or_default()
                    .push((n, chunk_id));
            }
            if let MetadataValue::StringList(xs) = value {
                for x in xs {
                    self.string_list_contains
                        .entry(field.clone())
                        .or_default()
                        .entry(x.clone())
                        .or_default()
                        .insert(chunk_id);
                }
            }
        }
    }

    /// Call after all inserts so range scans are O(log N) per bound.
    pub fn finalize(&mut self) {
        for v in self.numeric.values_mut() {
            v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        }
    }

    /// Resolve a filter expression to the eligible chunk-id set.
    /// Returns the universe when the expression is empty.
    pub fn eligible(&self, expr: &FilterExpr) -> RoaringBitmap {
        if expr.is_empty() {
            return self.universe.clone();
        }
        let mut acc: Option<RoaringBitmap> = None;
        for predicate in &expr.predicates {
            let bm = self.resolve(predicate);
            acc = Some(match acc {
                Some(prev) => prev & bm,
                None => bm,
            });
            if let Some(a) = &acc {
                if a.is_empty() {
                    return RoaringBitmap::new();
                }
            }
        }
        acc.unwrap_or_else(|| self.universe.clone())
    }

    fn resolve(&self, p: &Predicate) -> RoaringBitmap {
        match p {
            Predicate::Eq { field, value } => self
                .equality
                .get(field)
                .and_then(|by_value| by_value.get(&MetadataKey::from_metadata(value)))
                .cloned()
                .unwrap_or_default(),
            Predicate::Range { field, gte, lte } => self.range(field, *gte, *lte),
            Predicate::Contains { field, value } => self
                .string_list_contains
                .get(field)
                .and_then(|by_value| by_value.get(value))
                .cloned()
                .unwrap_or_default(),
            Predicate::In { field, values } => {
                let mut out = RoaringBitmap::new();
                if let Some(by_value) = self.equality_strings.get(field) {
                    for v in values {
                        if let Some(bm) = by_value.get(v) {
                            out |= bm;
                        }
                    }
                }
                out
            }
        }
    }

    fn range(&self, field: &str, gte: Option<f64>, lte: Option<f64>) -> RoaringBitmap {
        let Some(sorted) = self.numeric.get(field) else {
            return RoaringBitmap::new();
        };
        let lo = gte.unwrap_or(f64::NEG_INFINITY);
        let hi = lte.unwrap_or(f64::INFINITY);
        // sorted is by value; binary-search the bounds.
        let start = sorted.partition_point(|(v, _)| *v < lo);
        let end = sorted.partition_point(|(v, _)| *v <= hi);
        let mut out = RoaringBitmap::new();
        for (_, id) in &sorted[start..end] {
            out.insert(*id);
        }
        out
    }
}

/// Estimate selectivity = |eligible| / |universe|. Used by the search planner
/// to scale USearch's `ef_search` parameter.
pub fn selectivity(eligible: &RoaringBitmap, universe_len: u64) -> f64 {
    if universe_len == 0 {
        return 1.0;
    }
    eligible.len() as f64 / universe_len as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{FilterCondition, FilterValue};

    fn meta(pairs: &[(&str, MetadataValue)]) -> HashMap<String, MetadataValue> {
        pairs
            .iter()
            .cloned()
            .map(|(k, v)| (k.to_string(), v))
            .collect()
    }

    fn build_index() -> FilterIndex {
        let mut idx = FilterIndex::new();
        for i in 0u32..1000 {
            let org = if i % 100 == 0 { "acme" } else { "widgets" };
            let ts = (i as i64) * 10;
            idx.insert(
                i,
                &meta(&[
                    ("org_id", MetadataValue::String(org.into())),
                    ("created_at", MetadataValue::Int(ts)),
                    (
                        "tags",
                        MetadataValue::StringList(if i % 2 == 0 {
                            vec!["even".into()]
                        } else {
                            vec!["odd".into()]
                        }),
                    ),
                ]),
            );
        }
        idx.finalize();
        idx
    }

    #[test]
    fn eq_selectivity_one_percent() {
        let idx = build_index();
        let mut f = HashMap::new();
        f.insert(
            "org_id".into(),
            FilterValue::Exact(MetadataValue::String("acme".into())),
        );
        let expr = FilterExpr::compile(&f);
        let eligible = idx.eligible(&expr);
        assert_eq!(eligible.len(), 10);
        let s = selectivity(&eligible, idx.len());
        assert!((s - 0.01).abs() < 1e-9);
    }

    #[test]
    fn range_inclusive() {
        let idx = build_index();
        let mut f = HashMap::new();
        f.insert(
            "created_at".into(),
            FilterValue::Condition(FilterCondition {
                gte: Some(100.0),
                lte: Some(200.0),
                contains: None,
                in_values: None,
            }),
        );
        let expr = FilterExpr::compile(&f);
        let eligible = idx.eligible(&expr);
        // ts = i*10 in [100, 200] -> i in [10, 20] -> 11 chunks.
        assert_eq!(eligible.len(), 11);
    }

    #[test]
    fn and_of_eq_and_range() {
        let idx = build_index();
        let mut f = HashMap::new();
        f.insert(
            "org_id".into(),
            FilterValue::Exact(MetadataValue::String("acme".into())),
        );
        f.insert(
            "created_at".into(),
            FilterValue::Condition(FilterCondition {
                gte: Some(0.0),
                lte: Some(5000.0),
                contains: None,
                in_values: None,
            }),
        );
        let expr = FilterExpr::compile(&f);
        let eligible = idx.eligible(&expr);
        // acme at i in {0, 100, 200, 300, 400, 500} -> ts in [0, 5000] -> 6.
        assert_eq!(eligible.len(), 6);
    }

    #[test]
    fn contains_on_string_list() {
        let idx = build_index();
        let mut f = HashMap::new();
        f.insert(
            "tags".into(),
            FilterValue::Condition(FilterCondition {
                gte: None,
                lte: None,
                contains: Some("even".into()),
                in_values: None,
            }),
        );
        let expr = FilterExpr::compile(&f);
        let eligible = idx.eligible(&expr);
        assert_eq!(eligible.len(), 500);
    }

    #[test]
    fn empty_filter_returns_universe() {
        let idx = build_index();
        let eligible = idx.eligible(&FilterExpr::default());
        assert_eq!(eligible.len(), 1000);
    }
}
