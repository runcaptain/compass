// search/cold.rs — serve-from-storage: answer semantic queries on a
// collection that is NOT attached, with a handful of object-storage range
// reads instead of a full index rebuild.
//
// Query flow (per namespace):
//   1. GET manifest (freshness anchor: everything committed is visible, so
//      cold reads satisfy read-your-writes by construction)
//   2. per segment (immutable → artifacts cached by segment id):
//      header+TOC, `cent:<space>` centroids, `tombs`, `metaidx` — all small
//   3. rank clusters by centroid dot-product, range-GET the top `nprobe`
//      clusters, score their (unit-norm) vectors against the query
//   4. brute-force the uncompacted WAL tail (bounded by the auto-compact
//      threshold) and apply tombstones; newest version of an id wins
//   5. hydrate the top candidates' chunk JSON by byte range via `metaidx`,
//      apply metadata filters, return
//
// RAM cost per cold namespace: centroids + directories + tombstones + the
// metadata index — megabytes, independent of collection size.

use crate::models::{DocumentChunk, FilterValue, MetadataValue};
use crate::search::filter_pushdown::{FilterExpr, Predicate};
use crate::search::ivf;
use crate::storage::{lsm, Storage, StorageError};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Clusters probed per segment per query (env-tunable via the manager).
pub const DEFAULT_NPROBE: usize = 8;
/// Overfetch factor before filtering/deduping down to top_k.
const OVERFETCH: usize = 4;
/// Header bytes fetched optimistically (magic + max_id + toc_len + TOC).
const HEADER_PROBE: u64 = 16 * 1024;
/// metaidx at or below this size is fetched whole; larger ones page in
/// blocks on demand.
const METAIDX_FULL_MAX: u64 = 8 * 1024 * 1024;
const METAIDX_BLOCK_ROWS: usize = 2048;

const MAGIC_V2: &[u8; 8] = b"CSEG0002";
const MAGIC_V3: &[u8; 8] = b"CSEG0003";

type BoxErr = Box<dyn std::error::Error + Send + Sync>;

/// Cached, immutable cold-read artifacts for ONE segment object.
pub struct ColdSegment {
    ns: String,
    segment_id: String,
    /// section name -> (absolute byte offset in the object, length)
    sections: HashMap<String, (u64, u64)>,
    /// space -> parsed centroids + cluster directory
    cents: HashMap<String, ivf::Centroids>,
    /// ids tombstoned BY this segment (apply to this and all older segments)
    pub tombstones: HashSet<u64>,
    metaidx: MetaIdx,
}

enum MetaIdx {
    /// Whole index resident: sorted (id, off, len) rows.
    Full(Vec<(u64, u64, u32)>),
    /// Sparse anchors (first id of each block) + block byte range info; blocks
    /// are fetched on demand per query (not cached — queries touch few).
    Paged {
        anchors: Vec<u64>, // first id of block i
        n_rows: u64,
        idx_offset: u64, // absolute offset of the first row (after the count)
    },
}

fn seg_key(ns: &str, id: &str) -> String {
    format!("{ns}/segments/{id}")
}

async fn get_range(
    storage: &dyn Storage,
    key: &str,
    start: u64,
    len: u64,
) -> Result<bytes::Bytes, StorageError> {
    storage.get_range(key, start..start + len).await
}

impl ColdSegment {
    /// Build the cached artifacts with a few small reads. Total fetched:
    /// TOC + centroids + tombstones + (metaidx or its anchors).
    pub async fn open(storage: &dyn Storage, ns: &str, segment_id: &str) -> Result<Self, BoxErr> {
        let key = seg_key(ns, segment_id);
        // Header + TOC (optimistic single read; re-read if the TOC is huge).
        let head = storage.get_range(&key, 0..HEADER_PROBE).await?;
        if head.len() < 20 {
            return Err(format!("segment {segment_id}: truncated header").into());
        }
        if &head[0..8] != MAGIC_V3 && &head[0..8] != MAGIC_V2 {
            return Err(format!(
                "segment {segment_id} is not cold-servable (pre-v2 JSON format); \
                 run POST /collections/:name/compact once to upgrade it"
            )
            .into());
        }
        let toc_len = u32::from_le_bytes(head[16..20].try_into().unwrap()) as u64;
        let toc_bytes = if 20 + toc_len <= head.len() as u64 {
            head.slice(20..(20 + toc_len) as usize)
        } else {
            get_range(storage, &key, 20, toc_len).await?
        };
        let toc: Vec<(String, u64)> = serde_json::from_slice(&toc_bytes)
            .map_err(|e| format!("segment {segment_id}: bad TOC: {e}"))?;
        let mut sections = HashMap::new();
        let mut pos = 20 + toc_len;
        for (name, len) in toc {
            sections.insert(name, (pos, len));
            pos += len;
        }

        // Centroids for every clustered space (small — cache them all).
        let mut cents = HashMap::new();
        for (name, &(off, len)) in &sections {
            if let Some(space) = name.strip_prefix("cent:") {
                let body = get_range(storage, &key, off, len).await?;
                let c = ivf::parse_cent(&body)
                    .ok_or_else(|| format!("segment {segment_id}: bad {name}"))?;
                cents.insert(space.to_string(), c);
            }
        }

        // Tombstones (u64 LE array; bounded by deletes-per-fold).
        let mut tombstones = HashSet::new();
        if let Some(&(off, len)) = sections.get("tombs") {
            if len > 0 {
                let body = get_range(storage, &key, off, len).await?;
                for c in body.chunks_exact(8) {
                    tombstones.insert(u64::from_le_bytes(c.try_into().unwrap()));
                }
            }
        }

        // Metadata index: whole if small, paged anchors otherwise.
        let metaidx = match sections.get("metaidx") {
            Some(&(off, len)) if len > 8 => {
                if len <= METAIDX_FULL_MAX {
                    let body = get_range(storage, &key, off, len).await?;
                    let n = u64::from_le_bytes(body[0..8].try_into().unwrap()) as usize;
                    let mut rows = Vec::with_capacity(n);
                    for i in 0..n {
                        let p = 8 + i * 20;
                        rows.push((
                            u64::from_le_bytes(body[p..p + 8].try_into().unwrap()),
                            u64::from_le_bytes(body[p + 8..p + 16].try_into().unwrap()),
                            u32::from_le_bytes(body[p + 16..p + 20].try_into().unwrap()),
                        ));
                    }
                    MetaIdx::Full(rows)
                } else {
                    // Anchor row (the id) of every block: one strided read per
                    // block start — batched into a single ranged read of the
                    // first 8 bytes of each block would still be N requests;
                    // instead read the count, then fetch anchor ids in one
                    // pass over block-leading rows via a coalesced read of
                    // just the id columns is not possible over HTTP — so
                    // fetch the whole index ONCE here (paged builds accept a
                    // one-time cost bounded by index size / 50MB at 2.5M
                    // rows) and keep only anchors resident.
                    let body = get_range(storage, &key, off, len).await?;
                    let n = u64::from_le_bytes(body[0..8].try_into().unwrap());
                    let mut anchors = Vec::new();
                    let mut i = 0u64;
                    while i < n {
                        let p = (8 + i * 20) as usize;
                        anchors.push(u64::from_le_bytes(body[p..p + 8].try_into().unwrap()));
                        i += METAIDX_BLOCK_ROWS as u64;
                    }
                    MetaIdx::Paged {
                        anchors,
                        n_rows: n,
                        idx_offset: off + 8,
                    }
                }
            }
            _ => MetaIdx::Full(Vec::new()),
        };

        Ok(Self {
            ns: ns.to_string(),
            segment_id: segment_id.to_string(),
            sections,
            cents,
            tombstones,
            metaidx,
        })
    }

    /// Rank this segment's clusters for `q` (unit-norm) and return the top
    /// `nprobe` cluster byte ranges to fetch.
    fn probe_plan(&self, space: &str, q: &[f32], nprobe: usize) -> Vec<(u64, u64)> {
        let Some(cent) = self.cents.get(space) else {
            return Vec::new();
        };
        let Some(&(clu_off, _)) = self.sections.get(&format!("clu:{space}")) else {
            return Vec::new();
        };
        let mut ranked: Vec<(usize, f32)> = cent
            .centroids
            .iter()
            .enumerate()
            .filter(|(i, _)| cent.dir[*i].count > 0)
            .map(|(i, c)| (i, ivf::dot(c, q)))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
            .into_iter()
            .take(nprobe)
            .map(|(i, _)| {
                let d = cent.dir[i];
                (clu_off + d.offset, d.len)
            })
            .collect()
    }

    fn dims_for(&self, space: &str) -> Option<usize> {
        self.cents.get(space).map(|c| c.dims)
    }

    /// The flat `emb:<space>` byte range (segments below the clustering
    /// threshold) — brute-forced whole.
    fn flat_range(&self, space: &str) -> Option<(u64, u64)> {
        self.sections.get(&format!("emb:{space}")).copied()
    }

    /// Look up the meta2 byte ranges for a set of ids.
    async fn meta_ranges(
        &self,
        storage: &dyn Storage,
        ids: &[u64],
    ) -> Result<Vec<(u64, u64, u32)>, BoxErr> {
        let Some(&(meta_off, _)) = self.sections.get("meta2") else {
            return Ok(Vec::new());
        };
        let key = seg_key(&self.ns, &self.segment_id);
        let mut out = Vec::new();
        match &self.metaidx {
            MetaIdx::Full(rows) => {
                for &id in ids {
                    if let Ok(i) = rows.binary_search_by_key(&id, |r| r.0) {
                        let (rid, off, len) = rows[i];
                        out.push((rid, meta_off + off, len));
                    }
                }
            }
            MetaIdx::Paged {
                anchors,
                n_rows,
                idx_offset,
            } => {
                // Group wanted ids by block, fetch each needed block once.
                let mut by_block: HashMap<usize, Vec<u64>> = HashMap::new();
                for &id in ids {
                    let block = match anchors.binary_search(&id) {
                        Ok(i) => i,
                        Err(0) => continue, // below the first anchor: absent
                        Err(i) => i - 1,
                    };
                    by_block.entry(block).or_default().push(id);
                }
                for (block, wanted) in by_block {
                    let start_row = (block * METAIDX_BLOCK_ROWS) as u64;
                    let rows_here = (*n_rows - start_row).min(METAIDX_BLOCK_ROWS as u64);
                    let body =
                        get_range(storage, &key, idx_offset + start_row * 20, rows_here * 20)
                            .await?;
                    for r in body.chunks_exact(20) {
                        let rid = u64::from_le_bytes(r[0..8].try_into().unwrap());
                        if wanted.contains(&rid) {
                            out.push((
                                rid,
                                meta_off + u64::from_le_bytes(r[8..16].try_into().unwrap()),
                                u32::from_le_bytes(r[16..20].try_into().unwrap()),
                            ));
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

/// One scored candidate before hydration. `generation` orders duplicates of
/// the same id: higher wins (segments in manifest order, tail above all).
struct Candidate {
    id: u64,
    score: f32,
    generation: usize,
    /// Tail candidates already carry their chunk.
    chunk: Option<DocumentChunk>,
    segment: Option<usize>, // index into segments, for hydration
}

/// A cold semantic search over one namespace. `segments` are the cached
/// artifacts in manifest order; the WAL tail is read fresh per query.
#[allow(clippy::too_many_arguments)]
pub async fn search(
    storage: &dyn Storage,
    ns: &str,
    segments: &[Arc<ColdSegment>],
    manifest: &lsm::Manifest,
    space: &str,
    query: &[f32],
    top_k: usize,
    nprobe: usize,
    filters: &HashMap<String, FilterValue>,
) -> Result<Vec<(DocumentChunk, f32)>, BoxErr> {
    let mut q = query.to_vec();
    ivf::normalize(&mut q);

    // Tombstones: every segment's carried deletes + the live WAL tail's.
    let mut dead: HashSet<u64> = HashSet::new();
    for s in segments {
        dead.extend(s.tombstones.iter().copied());
    }

    // WAL tail: bounded by the auto-compaction threshold in healthy
    // operation. A pathologically long tail (compaction disabled/failing)
    // would make this a full-dataset materialization per query — refuse
    // loudly instead of degrading into that silently.
    let tail_len = manifest.uncompacted().count();
    if tail_len > 2 * crate::collections::AUTO_COMPACT_FRAGMENT_THRESHOLD {
        return Err(format!(
            "namespace '{ns}' has {tail_len} uncompacted WAL fragments — too many to \
             cold-serve. Run POST /collections/:name/compact (or check why \
             auto-compaction is not running), then retry."
        )
        .into());
    }
    let tail = lsm::read_uncompacted_fragments(storage, ns, manifest).await?;
    let mut tail_chunks: HashMap<u64, DocumentChunk> = HashMap::new();
    for (fref, payload) in &tail {
        match fref.kind {
            lsm::FragmentKind::Data => {
                let chunks: Vec<DocumentChunk> = serde_json::from_slice(payload)
                    .map_err(|e| format!("tail fragment decode: {e}"))?;
                for c in chunks {
                    dead.remove(&c.id); // re-ingest after delete resurrects
                    tail_chunks.insert(c.id, c);
                }
            }
            lsm::FragmentKind::Tombstone => {
                let ids: Vec<u64> = serde_json::from_slice(payload)
                    .map_err(|e| format!("tail tombstone decode: {e}"))?;
                for id in ids {
                    dead.insert(id);
                    tail_chunks.remove(&id);
                }
            }
            _ => {}
        }
    }

    let want = (top_k * OVERFETCH).max(top_k);
    let mut candidates: Vec<Candidate> = Vec::new();

    // Segment candidates: probe clusters (or brute-force flat sections).
    for (gen, seg) in segments.iter().enumerate() {
        let key = seg_key(ns, &seg.segment_id);
        let mut ranges = seg.probe_plan(space, &q, nprobe);
        let dims = match seg.dims_for(space) {
            Some(d) => d,
            None => match seg.flat_range(space) {
                Some((off, len)) if len >= 12 => {
                    // Flat section: [u32 dims][u64 n][rows] — brute force it.
                    let head = get_range(storage, &key, off, 12).await?;
                    let dims = u32::from_le_bytes(head[0..4].try_into().unwrap()) as usize;
                    ranges = vec![(off + 12, len - 12)];
                    dims
                }
                _ => continue, // space absent in this segment
            },
        };
        if dims != q.len() {
            return Err(format!(
                "query has {} dims but segment space '{space}' has {dims}",
                q.len()
            )
            .into());
        }
        // Fetch probed ranges concurrently.
        let bodies = futures::future::try_join_all(
            ranges
                .iter()
                .map(|&(off, len)| get_range(storage, &key, off, len)),
        )
        .await?;
        for body in bodies {
            for (id, v) in ivf::parse_cluster_rows(&body, dims) {
                if dead.contains(&id) || tail_chunks.contains_key(&id) {
                    continue;
                }
                // Flat sections store raw vectors; clustered store unit-norm.
                // Normalizing again is idempotent for the latter.
                let mut v = v;
                ivf::normalize(&mut v);
                candidates.push(Candidate {
                    id,
                    score: ivf::dot(&v, &q),
                    generation: gen,
                    chunk: None,
                    segment: Some(gen),
                });
            }
        }
    }

    // Tail candidates: brute-force the fresh writes.
    let tail_gen = segments.len();
    for (id, c) in &tail_chunks {
        if let Some(emb) = c.embeddings.get(space) {
            let mut v = emb.clone();
            ivf::normalize(&mut v);
            candidates.push(Candidate {
                id: *id,
                score: ivf::dot(&v, &q),
                generation: tail_gen,
                chunk: Some(c.clone()),
                segment: None,
            });
        }
    }

    // Dedupe by id, newest generation wins; then keep the global top `want`.
    candidates.sort_by(|a, b| a.id.cmp(&b.id).then(b.generation.cmp(&a.generation)));
    candidates.dedup_by_key(|c| c.id);
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.truncate(want);

    // Hydrate: group segment candidates per segment, batch the meta lookups.
    let expr = FilterExpr::compile(filters);
    let mut hydrated: Vec<(DocumentChunk, f32)> = Vec::new();
    let mut by_seg: HashMap<usize, Vec<u64>> = HashMap::new();
    let mut scores: HashMap<u64, f32> = HashMap::new();
    for c in &candidates {
        scores.insert(c.id, c.score);
        match (&c.chunk, c.segment) {
            (Some(ch), _) => {
                if eval_filters(&expr, ch) {
                    hydrated.push((ch.clone(), c.score));
                }
            }
            (None, Some(seg_i)) => by_seg.entry(seg_i).or_default().push(c.id),
            _ => {}
        }
    }
    for (seg_i, ids) in by_seg {
        let seg = &segments[seg_i];
        let key = seg_key(ns, &seg.segment_id);
        let mut ranges = seg.meta_ranges(storage, &ids).await?;
        // Coalesce adjacent-ish rows into fewer GETs.
        ranges.sort_by_key(|r| r.1);
        let mut batches: Vec<(u64, u64, Vec<(u64, u64, u32)>)> = Vec::new();
        for r in ranges {
            match batches.last_mut() {
                Some((_start, end, rows)) if r.1 <= *end + 64 * 1024 => {
                    *end = (*end).max(r.1 + r.2 as u64);
                    rows.push(r);
                }
                _ => batches.push((r.1, r.1 + r.2 as u64, vec![r])),
            }
        }
        for (start, end, rows) in batches {
            let body = get_range(storage, &key, start, end - start).await?;
            for (id, off, len) in rows {
                let lo = (off - start) as usize;
                let chunk: DocumentChunk = serde_json::from_slice(&body[lo..lo + len as usize])
                    .map_err(|e| format!("meta2 row decode (id {id}): {e}"))?;
                if eval_filters(&expr, &chunk) {
                    hydrated.push((chunk, scores.get(&id).copied().unwrap_or(0.0)));
                }
            }
        }
    }

    hydrated.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    hydrated.truncate(top_k);
    Ok(hydrated)
}

/// Metadata filter evaluation for cold hits (the roaring FilterIndex only
/// exists for attached collections). Mirrors FilterIndex::eligible semantics,
/// including the doc_type-as-metadata rule.
fn eval_filters(expr: &FilterExpr, chunk: &DocumentChunk) -> bool {
    if expr.is_empty() {
        return true;
    }
    let get = |field: &str| -> Option<MetadataValue> {
        if field == "doc_type" {
            Some(MetadataValue::String(chunk.doc_type.clone()))
        } else {
            chunk.metadata.get(field).cloned()
        }
    };
    expr.predicates.iter().all(|p| match p {
        Predicate::Eq { field, value } => get(field).as_ref() == Some(value),
        Predicate::Range { field, gte, lte } => match get(field).and_then(|m| m.as_f64()) {
            Some(n) => gte.map(|g| n >= g).unwrap_or(true) && lte.map(|l| n <= l).unwrap_or(true),
            None => false,
        },
        Predicate::Contains { field, value } => match get(field) {
            Some(MetadataValue::StringList(xs)) => xs.iter().any(|x| x == value),
            Some(MetadataValue::String(s)) => &s == value,
            _ => false,
        },
        Predicate::In { field, values } => match get(field) {
            Some(MetadataValue::String(s)) => values.contains(&s),
            _ => false,
        },
    })
}
