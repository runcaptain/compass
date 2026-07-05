// search/filter_index.rs — Per-collection filter index (roaring treemaps + sorted ranges).
//
// Goal: turn an AND of equality/range/contains/in predicates into ONE roaring-
// treemap intersection over chunk-id sets. The output is the eligible candidate
// set passed into USearch's filter callback, so the HNSW walk only visits
// eligible nodes.
//
// Chunk ids are u64 (matching DocumentChunk::id and USearch keys). We use
// `RoaringTreemap` — the roaring crate's 64-bit variant — rather than the
// 32-bit `RoaringBitmap`, so the index covers the full id space with no cast,
// no overflow guard, and no silently-dropped chunks past u32::MAX.
//
// Storage shape (v0):
//   - equality:   (field, canonical_string) -> RoaringTreemap of chunk_ids
//   - numeric:    field -> BTreeMap<ordered f64 bits, ids> (O(log N) ops)
//   - string_list:(field, element) -> RoaringTreemap (for `contains`)
//   - present:    field -> RoaringTreemap of chunk_ids that have any value
//
// Inserts are batch: build once, query many times. The structures rebuild
// alongside the HNSW index after each ingest batch — same lifecycle, same
// disk-flush cadence as the rest of the search state. Persistence wiring is
// out of scope for the initial implementation.

use std::collections::HashMap;

use roaring::RoaringTreemap;

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
    equality: HashMap<String, HashMap<MetadataKey, RoaringTreemap>>,
    /// field -> string value -> chunk_ids (for `in` semantics on strings).
    equality_strings: HashMap<String, HashMap<String, RoaringTreemap>>,
    /// field -> total-order-encoded f64 -> ids, for range predicates.
    /// BTreeMap keys let inserts/removes stay O(log N) (a sorted Vec made
    /// every incremental update O(N) — disqualifying at scale).
    numeric: HashMap<String, std::collections::BTreeMap<u64, RoaringTreemap>>,
    /// field -> element -> chunk_ids whose StringList contains the element.
    string_list_contains: HashMap<String, HashMap<String, RoaringTreemap>>,
    /// field -> chunk_ids that have any value for this field.
    present: HashMap<String, RoaringTreemap>,
    /// Universe of all known chunk ids. Used as the starting set for empty
    /// expressions and as a fallback when a predicate spans the whole field.
    universe: RoaringTreemap,
}

/// Map f64 to a u64 preserving total order (IEEE-754 bit trick; NaNs are
/// filtered before insertion by `as_f64`).
fn f64_ord_key(x: f64) -> u64 {
    let b = x.to_bits();
    if b >> 63 == 1 {
        !b
    } else {
        b | (1 << 63)
    }
}

impl FilterIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> u64 {
        self.universe.len()
    }

    /// The live-id universe (treemap) — shared with facet counting so deleted
    /// chunks never inflate counts.
    pub fn universe(&self) -> &RoaringTreemap {
        &self.universe
    }

    /// Is this id live (inserted and not removed)? The universe excludes
    /// tombstoned ids on every maintenance path, so this doubles as the
    /// existence check now that chunks are not held in RAM.
    pub fn contains(&self, id: u64) -> bool {
        self.universe.contains(id)
    }

    /// Insert a single chunk with its metadata. `chunk_id` is the full u64
    /// `DocumentChunk::id`; the treemap covers the entire id space, so there is
    /// no cap and no chunk is ever dropped for having a large id.
    pub fn insert(&mut self, chunk_id: u64, metadata: &HashMap<String, MetadataValue>) {
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
                    .entry(f64_ord_key(n))
                    .or_default()
                    .insert(chunk_id);
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

    /// Remove one chunk (reverse of `insert`). O(log N) per field value —
    /// deletes no longer trigger an O(collection) index rebuild.
    pub fn remove(&mut self, chunk_id: u64, metadata: &HashMap<String, MetadataValue>) {
        self.universe.remove(chunk_id);
        for (field, value) in metadata {
            if let Some(tm) = self.present.get_mut(field) {
                tm.remove(chunk_id);
            }
            if let Some(vals) = self.equality.get_mut(field) {
                let key = MetadataKey::from_metadata(value);
                if let Some(tm) = vals.get_mut(&key) {
                    tm.remove(chunk_id);
                    if tm.is_empty() {
                        vals.remove(&key);
                    }
                }
            }
            if let MetadataValue::String(sv) = value {
                if let Some(vals) = self.equality_strings.get_mut(field) {
                    if let Some(tm) = vals.get_mut(sv) {
                        tm.remove(chunk_id);
                        if tm.is_empty() {
                            vals.remove(sv);
                        }
                    }
                }
            }
            if let Some(n) = value.as_f64() {
                if let Some(vals) = self.numeric.get_mut(field) {
                    let key = f64_ord_key(n);
                    if let Some(tm) = vals.get_mut(&key) {
                        tm.remove(chunk_id);
                        if tm.is_empty() {
                            vals.remove(&key);
                        }
                    }
                }
            }
            if let MetadataValue::StringList(xs) = value {
                if let Some(vals) = self.string_list_contains.get_mut(field) {
                    for x in xs {
                        if let Some(tm) = vals.get_mut(x) {
                            tm.remove(chunk_id);
                            if tm.is_empty() {
                                vals.remove(x);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Resolve a filter expression to the eligible chunk-id set.
    /// Returns the universe when the expression is empty.
    pub fn eligible(&self, expr: &FilterExpr) -> RoaringTreemap {
        if expr.is_empty() {
            return self.universe.clone();
        }
        let mut acc: Option<RoaringTreemap> = None;
        for predicate in &expr.predicates {
            let bm = self.resolve(predicate);
            acc = Some(match acc {
                Some(prev) => prev & bm,
                None => bm,
            });
            if let Some(a) = &acc {
                if a.is_empty() {
                    return RoaringTreemap::new();
                }
            }
        }
        acc.unwrap_or_else(|| self.universe.clone())
    }

    fn resolve(&self, p: &Predicate) -> RoaringTreemap {
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
                let mut out = RoaringTreemap::new();
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

    fn range(&self, field: &str, gte: Option<f64>, lte: Option<f64>) -> RoaringTreemap {
        let Some(vals) = self.numeric.get(field) else {
            return RoaringTreemap::new();
        };
        let lo = f64_ord_key(gte.unwrap_or(f64::NEG_INFINITY));
        let hi = f64_ord_key(lte.unwrap_or(f64::INFINITY));
        let mut out = RoaringTreemap::new();
        for (_, tm) in vals.range(lo..=hi) {
            out |= tm;
        }
        out
    }
}

/// Estimate selectivity = |eligible| / |universe|. Used by the search planner
/// to scale USearch's `ef_search` parameter.
pub fn selectivity(eligible: &RoaringTreemap, universe_len: u64) -> f64 {
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
        for i in 0u64..1000 {
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

    // Regression: chunk ids beyond u32::MAX must be fully filterable. The old
    // RoaringBitmap implementation keyed on u32 and silently dropped these
    // chunks at insert (and treated them as ineligible at query). With
    // RoaringTreemap they index and resolve normally across every operator.
    #[test]
    fn ids_above_u32_max_are_filterable() {
        let big = (u32::MAX as u64) + 1; // 4_294_967_296 — overflows the old u32 key
        let mut idx = FilterIndex::new();
        idx.insert(
            big,
            &meta(&[
                ("org_id", MetadataValue::String("acme".into())),
                ("score", MetadataValue::Int(42)),
                ("tags", MetadataValue::StringList(vec!["even".into()])),
            ]),
        );

        assert_eq!(idx.len(), 1);

        // Equality resolves to the big id.
        let mut eq = HashMap::new();
        eq.insert(
            "org_id".into(),
            FilterValue::Exact(MetadataValue::String("acme".into())),
        );
        let eligible = idx.eligible(&FilterExpr::compile(&eq));
        assert!(eligible.contains(big), "equality must surface the >u32 id");
        assert_eq!(eligible.len(), 1);

        // Range resolves to the big id.
        let mut rng = HashMap::new();
        rng.insert(
            "score".into(),
            FilterValue::Condition(FilterCondition {
                gte: Some(40.0),
                lte: Some(50.0),
                contains: None,
                in_values: None,
            }),
        );
        assert!(idx.eligible(&FilterExpr::compile(&rng)).contains(big));

        // Contains resolves to the big id.
        let mut con = HashMap::new();
        con.insert(
            "tags".into(),
            FilterValue::Condition(FilterCondition {
                gte: None,
                lte: None,
                contains: Some("even".into()),
                in_values: None,
            }),
        );
        assert!(idx.eligible(&FilterExpr::compile(&con)).contains(big));
    }
}
