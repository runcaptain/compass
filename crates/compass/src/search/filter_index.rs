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

    pub fn is_empty(&self) -> bool {
        self.universe.is_empty()
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

    /// No-op since the numeric index moved to a BTreeMap (kept so existing
    /// build sites don't churn).
    pub fn finalize(&mut self) {}

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

// ── Persistence ────────────────────────────────────────────────────────────
// Serialize the whole index to bytes. NOT YET WIRED: every load path currently
// rebuilds the index from the chunk map; persisting/reloading it through the
// Storage trait (skipping the O(N) rebuild on startup) is future work.
// RoaringTreemaps use their native portable format; the container framing is a
// small length-prefixed encoding. Note the length prefixes are u32 — per-field
// entry counts are bounded by that (fine in practice; the u64 work in this
// module is about CHUNK IDS, not per-field entry counts).

impl MetadataKey {
    // Type-tagged encoding. Tag byte + payload.
    fn encode(&self, buf: &mut Vec<u8>) {
        match self {
            MetadataKey::Bool(b) => {
                buf.push(0);
                buf.push(*b as u8);
            }
            MetadataKey::Int(i) => {
                buf.push(1);
                buf.extend_from_slice(&i.to_le_bytes());
            }
            MetadataKey::Float(bits) => {
                buf.push(2);
                buf.extend_from_slice(&bits.to_le_bytes());
            }
            MetadataKey::String(s) => {
                buf.push(3);
                write_str(buf, s);
            }
            MetadataKey::StringList(xs) => {
                buf.push(4);
                buf.extend_from_slice(&(xs.len() as u32).to_le_bytes());
                for x in xs {
                    write_str(buf, x);
                }
            }
        }
    }

    fn decode(buf: &[u8], pos: &mut usize) -> Option<Self> {
        let tag = *buf.get(*pos)?;
        *pos += 1;
        Some(match tag {
            0 => {
                let b = *buf.get(*pos)? != 0;
                *pos += 1;
                MetadataKey::Bool(b)
            }
            1 => MetadataKey::Int(read_i64(buf, pos)?),
            2 => MetadataKey::Float(read_u64(buf, pos)?),
            3 => MetadataKey::String(read_str(buf, pos)?),
            4 => {
                let n = read_u32(buf, pos)? as usize;
                let mut xs = Vec::with_capacity(n);
                for _ in 0..n {
                    xs.push(read_str(buf, pos)?);
                }
                MetadataKey::StringList(xs)
            }
            _ => return None,
        })
    }
}

fn write_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn read_u32(buf: &[u8], pos: &mut usize) -> Option<u32> {
    let end = *pos + 4;
    let v = u32::from_le_bytes(buf.get(*pos..end)?.try_into().ok()?);
    *pos = end;
    Some(v)
}

fn read_u64(buf: &[u8], pos: &mut usize) -> Option<u64> {
    let end = *pos + 8;
    let v = u64::from_le_bytes(buf.get(*pos..end)?.try_into().ok()?);
    *pos = end;
    Some(v)
}

fn read_i64(buf: &[u8], pos: &mut usize) -> Option<i64> {
    Some(read_u64(buf, pos)? as i64)
}

fn read_str(buf: &[u8], pos: &mut usize) -> Option<String> {
    let len = read_u32(buf, pos)? as usize;
    let end = *pos + len;
    let s = String::from_utf8(buf.get(*pos..end)?.to_vec()).ok()?;
    *pos = end;
    Some(s)
}

fn write_treemap(buf: &mut Vec<u8>, t: &RoaringTreemap) {
    let mut tmp = Vec::new();
    // RoaringTreemap::serialize_into writes the portable format.
    t.serialize_into(&mut tmp).expect("treemap serialize");
    buf.extend_from_slice(&(tmp.len() as u32).to_le_bytes());
    buf.extend_from_slice(&tmp);
}

fn read_treemap(buf: &[u8], pos: &mut usize) -> Option<RoaringTreemap> {
    let len = read_u32(buf, pos)? as usize;
    let end = *pos + len;
    let slice = buf.get(*pos..end)?;
    let t = RoaringTreemap::deserialize_from(slice).ok()?;
    *pos = end;
    Some(t)
}

// field -> RoaringTreemap
fn write_map_tm(buf: &mut Vec<u8>, m: &HashMap<String, RoaringTreemap>) {
    buf.extend_from_slice(&(m.len() as u32).to_le_bytes());
    for (k, v) in m {
        write_str(buf, k);
        write_treemap(buf, v);
    }
}

fn read_map_tm(buf: &[u8], pos: &mut usize) -> Option<HashMap<String, RoaringTreemap>> {
    let n = read_u32(buf, pos)? as usize;
    let mut m = HashMap::with_capacity(n);
    for _ in 0..n {
        let k = read_str(buf, pos)?;
        let v = read_treemap(buf, pos)?;
        m.insert(k, v);
    }
    Some(m)
}

// field -> (string -> RoaringTreemap)
fn write_map_str_tm(buf: &mut Vec<u8>, m: &HashMap<String, HashMap<String, RoaringTreemap>>) {
    buf.extend_from_slice(&(m.len() as u32).to_le_bytes());
    for (k, inner) in m {
        write_str(buf, k);
        write_map_tm(buf, inner);
    }
}

fn read_map_str_tm(
    buf: &[u8],
    pos: &mut usize,
) -> Option<HashMap<String, HashMap<String, RoaringTreemap>>> {
    let n = read_u32(buf, pos)? as usize;
    let mut m = HashMap::with_capacity(n);
    for _ in 0..n {
        let k = read_str(buf, pos)?;
        let inner = read_map_tm(buf, pos)?;
        m.insert(k, inner);
    }
    Some(m)
}

impl FilterIndex {
    /// Format version for the serialized index (bump on any framing change).
    const FORMAT_VERSION: u8 = 1;

    /// Serialize the whole index to a byte buffer.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(Self::FORMAT_VERSION);

        // equality: field -> (MetadataKey -> treemap)
        buf.extend_from_slice(&(self.equality.len() as u32).to_le_bytes());
        for (field, inner) in &self.equality {
            write_str(&mut buf, field);
            buf.extend_from_slice(&(inner.len() as u32).to_le_bytes());
            for (key, tm) in inner {
                key.encode(&mut buf);
                write_treemap(&mut buf, tm);
            }
        }

        write_map_str_tm(&mut buf, &self.equality_strings);

        // numeric: field -> flattened (ordered-bits, id) pairs.
        buf.extend_from_slice(&(self.numeric.len() as u32).to_le_bytes());
        for (field, vals) in &self.numeric {
            write_str(&mut buf, field);
            let n: u64 = vals.values().map(|tm| tm.len()).sum();
            buf.extend_from_slice(&(n as u32).to_le_bytes());
            for (key, tm) in vals {
                for id in tm {
                    buf.extend_from_slice(&key.to_le_bytes());
                    buf.extend_from_slice(&id.to_le_bytes());
                }
            }
        }

        write_map_str_tm(&mut buf, &self.string_list_contains);
        write_map_tm(&mut buf, &self.present);
        write_treemap(&mut buf, &self.universe);
        buf
    }

    /// Reconstruct an index from bytes produced by [`FilterIndex::serialize`].
    pub fn deserialize(buf: &[u8]) -> Option<Self> {
        let mut pos = 0usize;
        let version = *buf.get(pos)?;
        pos += 1;
        if version != Self::FORMAT_VERSION {
            return None;
        }

        let mut idx = FilterIndex::new();

        let n_eq = read_u32(buf, &mut pos)? as usize;
        for _ in 0..n_eq {
            let field = read_str(buf, &mut pos)?;
            let n_inner = read_u32(buf, &mut pos)? as usize;
            let mut inner = HashMap::with_capacity(n_inner);
            for _ in 0..n_inner {
                let key = MetadataKey::decode(buf, &mut pos)?;
                let tm = read_treemap(buf, &mut pos)?;
                inner.insert(key, tm);
            }
            idx.equality.insert(field, inner);
        }

        idx.equality_strings = read_map_str_tm(buf, &mut pos)?;

        let n_num = read_u32(buf, &mut pos)? as usize;
        for _ in 0..n_num {
            let field = read_str(buf, &mut pos)?;
            let n_vals = read_u32(buf, &mut pos)? as usize;
            let mut vals: std::collections::BTreeMap<u64, RoaringTreemap> = Default::default();
            for _ in 0..n_vals {
                let key = read_u64(buf, &mut pos)?;
                let id = read_u64(buf, &mut pos)?;
                vals.entry(key).or_default().insert(id);
            }
            idx.numeric.insert(field, vals);
        }

        idx.string_list_contains = read_map_str_tm(buf, &mut pos)?;
        idx.present = read_map_tm(buf, &mut pos)?;
        idx.universe = read_treemap(buf, &mut pos)?;

        Some(idx)
    }
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
        idx.finalize();

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

    #[test]
    fn serialize_roundtrip_preserves_queries() {
        let idx = build_index();
        let bytes = idx.serialize();
        let restored = FilterIndex::deserialize(&bytes).expect("deserialize ok");

        assert_eq!(restored.len(), idx.len());

        // Equality query matches identically.
        let mut eq = HashMap::new();
        eq.insert(
            "org_id".into(),
            FilterValue::Exact(MetadataValue::String("acme".into())),
        );
        let expr = FilterExpr::compile(&eq);
        assert_eq!(idx.eligible(&expr).len(), restored.eligible(&expr).len());
        assert_eq!(restored.eligible(&expr).len(), 10);

        // Range query matches identically after restore.
        let mut rng = HashMap::new();
        rng.insert(
            "created_at".into(),
            FilterValue::Condition(FilterCondition {
                gte: Some(100.0),
                lte: Some(200.0),
                contains: None,
                in_values: None,
            }),
        );
        let rexpr = FilterExpr::compile(&rng);
        assert_eq!(idx.eligible(&rexpr).len(), restored.eligible(&rexpr).len());

        // Contains query matches identically.
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
        let cexpr = FilterExpr::compile(&con);
        assert_eq!(restored.eligible(&cexpr).len(), 500);
    }

    #[test]
    fn deserialize_rejects_bad_version() {
        assert!(FilterIndex::deserialize(&[]).is_none());
        assert!(FilterIndex::deserialize(&[99]).is_none()); // bad version byte
    }
}
