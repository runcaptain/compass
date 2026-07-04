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
/// Overfetch factor before deduping down to top_k.
const OVERFETCH: usize = 4;
/// Additional overfetch multiplier when metadata filters are present (cold
/// filtering is post-selection; see the recall-contract note in search()).
const FILTER_OVERFETCH: usize = 8;
/// Header bytes fetched optimistically (magic + max_id + toc_len + TOC).
const HEADER_PROBE: u64 = 16 * 1024;
/// metaidx at or below this size is fetched whole; larger ones page in
/// blocks on demand.
const METAIDX_FULL_MAX: u64 = 8 * 1024 * 1024;
const METAIDX_BLOCK_ROWS: usize = 2048;

type BoxErr = Box<dyn std::error::Error + Send + Sync>;

/// Cached, immutable cold-read artifacts for ONE segment object.
pub struct ColdSegment {
    ns: String,
    segment_id: String,
    /// section name -> (absolute byte offset in the object, length)
    sections: HashMap<String, (u64, u64)>,
    /// space -> parsed centroids + cluster directory
    cents: HashMap<String, ivf::Centroids>,
    /// ids tombstoned BY this segment (they apply to OLDER segments only —
    /// see the generation rule in `search`)
    tombstones: HashSet<u64>,
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
        Self::open_with_limits(storage, ns, segment_id, METAIDX_FULL_MAX).await
    }

    /// `open` with an explicit full-fetch threshold — lets tests exercise the
    /// paged metadata-index path without a 400k-chunk segment.
    async fn open_with_limits(
        storage: &dyn Storage,
        ns: &str,
        segment_id: &str,
        metaidx_full_max: u64,
    ) -> Result<Self, BoxErr> {
        let key = seg_key(ns, segment_id);
        // Header + TOC (optimistic single read; re-read if the TOC is huge).
        let head = storage.get_range(&key, 0..HEADER_PROBE).await?;
        if head.len() < 20 {
            return Err(format!("segment {segment_id}: truncated header").into());
        }
        // Only v3 segments carry the cold-read sections (metaidx/meta2 and
        // clusters). v2 would brute-force its whole flat section and then
        // drop every hit at hydration — reject loudly instead.
        if head[0..8] != crate::collections::cloud::SEG_MAGIC_V3 {
            return Err(format!(
                "segment {segment_id} predates the cold-servable format; \
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
                if len <= metaidx_full_max {
                    let body = get_range(storage, &key, off, len).await?;
                    let n = u64::from_le_bytes(body[0..8].try_into().unwrap()) as usize;
                    if body.len() < 8 + n * 20 {
                        return Err(format!(
                            "segment {segment_id}: truncated metaidx ({n} rows declared)"
                        )
                        .into());
                    }
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
                    // Big index: fetch it whole ONCE at open (bounded by
                    // index size, ~20B/row) but keep only per-block anchor
                    // ids resident; lookups page 20B×2048 blocks on demand.
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

    // Tombstone semantics must match materialize(): a segment's carried
    // tombstones apply only to OLDER segments (its own chunks are written
    // after them, and a NEWER segment's re-ingest of the same id must
    // survive). So a candidate from generation g dies only to a tombstone
    // from generation > g. A single flat union would permanently suppress
    // re-ingested chunks that warm search serves.
    let tomb_of = |generation: usize| -> &HashSet<u64> { &segments[generation].tombstones };
    let killed_by_newer = |id: u64, generation: usize| -> bool {
        ((generation + 1)..segments.len()).any(|j| tomb_of(j).contains(&id))
    };
    // Tail tombstones (replayed in seq order below) are the newest
    // generation of all: they kill any segment candidate.
    let mut tail_dead: HashSet<u64> = HashSet::new();

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
                    tail_dead.remove(&c.id); // re-ingest after delete resurrects
                    tail_chunks.insert(c.id, c);
                }
            }
            lsm::FragmentKind::Tombstone => {
                let ids: Vec<u64> = serde_json::from_slice(payload)
                    .map_err(|e| format!("tail tombstone decode: {e}"))?;
                for id in ids {
                    tail_dead.insert(id);
                    tail_chunks.remove(&id);
                }
            }
            _ => {}
        }
    }

    // Filters are applied POST-candidate-selection on the cold path (there
    // is no roaring index to push down without attaching), so a selective
    // filter needs a deeper candidate pool. Recall contract: cold filtered
    // queries can under-return when matches are rarer than ~1/FILTER_OVERFETCH
    // of the probed neighborhoods; warm search has no such limit.
    let overfetch = if filters.is_empty() {
        OVERFETCH
    } else {
        OVERFETCH * FILTER_OVERFETCH
    };
    let want = (top_k * overfetch).max(top_k).min(512);
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
                if tail_dead.contains(&id)
                    || tail_chunks.contains_key(&id)
                    || killed_by_newer(id, gen)
                {
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
        // Parity with FilterIndex: `contains` matches STRING LISTS only (the
        // warm index populates string_list_contains from StringList values) —
        // matching bare strings here would make cold return hits warm never
        // would.
        Predicate::Contains { field, value } => match get(field) {
            Some(MetadataValue::StringList(xs)) => xs.iter().any(|x| x == value),
            _ => false,
        },
        Predicate::In { field, values } => match get(field) {
            Some(MetadataValue::String(s)) => values.contains(&s),
            _ => false,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::local::LocalDiskStorage;

    fn storage(name: &str) -> (std::path::PathBuf, Arc<dyn Storage>) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "compass_cold_unit_{}_{}_{}",
            name,
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        let s: Arc<dyn Storage> = Arc::new(LocalDiskStorage::new(&root).unwrap());
        (root, s)
    }

    fn seg_with_chunks(ids: &[u64], tombstones: &[u64]) -> Vec<u8> {
        use crate::models::DocumentChunk;
        let chunks: Vec<DocumentChunk> = ids
            .iter()
            .map(|&i| {
                let mut c = DocumentChunk {
                    id: i,
                    collection: "ns".into(),
                    file_id: format!("f{i}"),
                    chunk_index: 0,
                    page: None,
                    text: format!("t{i}"),
                    metadata: Default::default(),
                    doc_type: "chunk".into(),
                    parent_id: None,
                    group_id: None,
                    embeddings: Default::default(),
                    embedding: None,
                };
                c.embeddings
                    .insert("default".into(), vec![i as f32, 1.0, 0.0, 0.0]);
                c
            })
            .collect();
        crate::collections::cloud::encode_segment_v3(&crate::collections::cloud::Segment {
            version: 2,
            chunks,
            relations: vec![],
            max_id: ids.iter().copied().max().unwrap_or(0),
            tombstones: tombstones.to_vec(),
            relation_tombstones: vec![],
        })
        .unwrap()
    }

    // C1 regression: an OLDER segment's carried tombstone must not suppress
    // the same id re-ingested into a NEWER segment (materialize parity).
    #[tokio::test]
    async fn newer_segment_survives_older_tombstone() {
        let (root, s) = storage("gen");
        // seg A (gen 0): chunk 1 live, carries tombstone for id 5.
        s.put(
            "ns/segments/a",
            bytes::Bytes::from(seg_with_chunks(&[1], &[5])),
        )
        .await
        .unwrap();
        // seg B (gen 1): id 5 re-ingested.
        s.put(
            "ns/segments/b",
            bytes::Bytes::from(seg_with_chunks(&[5], &[])),
        )
        .await
        .unwrap();
        let manifest = lsm::Manifest {
            segments: vec![
                lsm::SegmentRef {
                    id: "a".into(),
                    records: 1,
                },
                lsm::SegmentRef {
                    id: "b".into(),
                    records: 1,
                },
            ],
            ..Default::default()
        };
        let segs = vec![
            Arc::new(ColdSegment::open(s.as_ref(), "ns", "a").await.unwrap()),
            Arc::new(ColdSegment::open(s.as_ref(), "ns", "b").await.unwrap()),
        ];
        let hits = search(
            s.as_ref(),
            "ns",
            &segs,
            &manifest,
            "default",
            &[5.0, 1.0, 0.0, 0.0],
            10,
            DEFAULT_NPROBE,
            &Default::default(),
        )
        .await
        .unwrap();
        assert!(
            hits.iter().any(|(c, _)| c.id == 5),
            "id 5 lives in the NEWER segment; the older tombstone must not kill it"
        );
        // And the reverse still holds: a NEWER segment's tombstone kills an
        // OLDER segment's chunk.
        s.put(
            "ns/segments/c",
            bytes::Bytes::from(seg_with_chunks(&[9], &[1])),
        )
        .await
        .unwrap();
        let manifest2 = lsm::Manifest {
            segments: vec![
                lsm::SegmentRef {
                    id: "a".into(),
                    records: 1,
                },
                lsm::SegmentRef {
                    id: "c".into(),
                    records: 1,
                },
            ],
            ..Default::default()
        };
        let segs2 = vec![
            segs[0].clone(),
            Arc::new(ColdSegment::open(s.as_ref(), "ns", "c").await.unwrap()),
        ];
        let hits = search(
            s.as_ref(),
            "ns",
            &segs2,
            &manifest2,
            "default",
            &[1.0, 1.0, 0.0, 0.0],
            10,
            DEFAULT_NPROBE,
            &Default::default(),
        )
        .await
        .unwrap();
        assert!(
            hits.iter().all(|(c, _)| c.id != 1),
            "newer segment's tombstone must kill the older chunk"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    // Paged metadata-index path: force it with a tiny full-fetch threshold
    // and verify hydration still resolves every candidate.
    #[tokio::test]
    async fn paged_metaidx_hydrates() {
        let (root, s) = storage("paged");
        let ids: Vec<u64> = (0..50).collect();
        s.put(
            "ns/segments/p",
            bytes::Bytes::from(seg_with_chunks(&ids, &[])),
        )
        .await
        .unwrap();
        let seg = ColdSegment::open_with_limits(s.as_ref(), "ns", "p", 16)
            .await
            .unwrap();
        assert!(
            matches!(seg.metaidx, MetaIdx::Paged { .. }),
            "tiny threshold must force the paged variant"
        );
        let ranges = seg
            .meta_ranges(s.as_ref(), &[0, 7, 49, 999_999])
            .await
            .unwrap();
        let found: std::collections::HashSet<u64> = ranges.iter().map(|r| r.0).collect();
        assert_eq!(
            found,
            [0u64, 7, 49].into_iter().collect(),
            "paged lookups must resolve present ids and skip absent ones"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    // Pre-v3 segments are rejected loudly — v2 has no metaidx, so cold
    // serving it would read the whole flat section and then drop every hit.
    #[tokio::test]
    async fn v2_segment_rejected_with_upgrade_hint() {
        let (root, s) = storage("v2");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&crate::collections::cloud::SEG_MAGIC_V2);
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(b"[]");
        s.put("ns/segments/old", bytes::Bytes::from(bytes))
            .await
            .unwrap();
        let err = match ColdSegment::open(s.as_ref(), "ns", "old").await {
            Err(e) => e,
            Ok(_) => panic!("v2 segment must be rejected"),
        };
        assert!(err.to_string().contains("compact"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
