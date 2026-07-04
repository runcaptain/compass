//! Cloud (object-storage) materialization: turn the S3 LSM (segments + WAL
//! fragments) back into the live set of chunks AND relations.
//!
//! This is the piece that makes object storage the actual SOURCE OF TRUTH:
//!   - **Compaction** uses `materialize` to fold everything into one segment.
//!   - **Cold-start recovery** uses `materialize` to rebuild the local indexes
//!     from S3, so an ephemeral node with an empty local disk comes back with
//!     all its data — chunks, hierarchy, AND typed relations.
//!
//! Semantics (standard LSM): segments are applied first (oldest state), then
//! uncompacted WAL fragments in `seq` order, latest-wins per id, deletes applied.
//! Fragment payloads by `kind`:
//!   - `Data`           → JSON array of `DocumentChunk` (upserts)
//!   - `Tombstone`      → JSON array of `u64` chunk ids (deletes)
//!   - `RelationUpsert` → JSON array of `ChunkRelation` (created edges)
//!   - `RelationDelete` → JSON array of `String` relation ids (removed edges)
//!
//! A segment payload is a versioned JSON object `{chunks, relations}` — the full
//! live set at compaction time.

use crate::models::{
    ChunkRelation, Collection, CollectionConfig, DocumentChunk, VectorSpaceConfig,
};
use crate::storage::lsm::{self, FragmentKind, Manifest};
use crate::storage::{Storage, StorageError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Bucket-resident collection config — `{ns}/collection.json` in object
/// storage. The durable source of truth for everything in [`Collection`]
/// EXCEPT the node-local counters (`chunk_count`, `next_id`). Without it, a
/// cold rebuild has to fabricate metadata (inferring vector-space specs from
/// recovered embeddings and silently losing `CollectionConfig.embed_model`).
///
/// Distinct from the LOCAL file `data/{ns}/collection.json` (node cache);
/// bucket writes are strictly gated on cloud mode so a local-disk Storage
/// backend can never clobber the real local metadata file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketConfig {
    #[serde(default)]
    pub version: u8,
    pub name: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub vector_spaces: HashMap<String, VectorSpaceConfig>,
    pub default_vector_space: Option<String>,
    pub embedding_dims: usize,
    #[serde(default)]
    pub config: CollectionConfig,
}

const BUCKET_CONFIG_VERSION: u8 = 1;

impl BucketConfig {
    pub fn from_collection(c: &Collection) -> Self {
        Self {
            version: BUCKET_CONFIG_VERSION,
            name: c.name.clone(),
            created_at: c.created_at,
            vector_spaces: c.vector_spaces.clone(),
            default_vector_space: c.default_vector_space.clone(),
            embedding_dims: c.embedding_dims,
            config: c.config.clone(),
        }
    }
}

pub fn config_key(ns: &str) -> String {
    format!("{ns}/collection.json")
}

/// Read the bucket config, or None when absent (pre-v0.4 collection).
pub async fn read_bucket_config(
    storage: &dyn Storage,
    ns: &str,
) -> Result<Option<BucketConfig>, StorageError> {
    match storage.get(&config_key(ns)).await {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes).map_err(|e| {
            StorageError::Io(format!("bucket config decode for '{ns}': {e}"))
        })?)),
        Err(StorageError::NotFound(_)) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Create-only write of the bucket config. `AlreadyExists` bubbles up so the
/// caller can distinguish "fresh create" from "collection already in bucket".
pub async fn write_bucket_config_if_absent(
    storage: &dyn Storage,
    ns: &str,
    cfg: &BucketConfig,
) -> Result<(), StorageError> {
    let bytes = serde_json::to_vec(cfg)
        .map_err(|e| StorageError::Io(format!("bucket config encode: {e}")))?;
    storage
        .put_if_not_exists(&config_key(ns), bytes::Bytes::from(bytes))
        .await
        .map(|_| ())
}

/// CAS read-modify-write on the bucket config. `mutate` sees the LATEST doc
/// each attempt and may fail validation (e.g. "space already exists") — that
/// error aborts the loop. Retries only on version conflicts.
pub async fn cas_update_bucket_config<F>(
    storage: &dyn Storage,
    ns: &str,
    mut mutate: F,
) -> Result<BucketConfig, Box<dyn std::error::Error + Send + Sync>>
where
    F: FnMut(&mut BucketConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
{
    const MAX_RETRIES: u32 = 10;
    for _ in 0..MAX_RETRIES {
        let (bytes, version) = storage.get_versioned(&config_key(ns)).await?;
        let mut cfg: BucketConfig = serde_json::from_slice(&bytes)
            .map_err(|e| StorageError::Io(format!("bucket config decode for '{ns}': {e}")))?;
        mutate(&mut cfg)?;
        let encoded = serde_json::to_vec(&cfg)
            .map_err(|e| StorageError::Io(format!("bucket config encode: {e}")))?;
        match storage
            .put_if_match(&config_key(ns), bytes::Bytes::from(encoded), &version)
            .await
        {
            Ok(_) => return Ok(cfg),
            Err(StorageError::VersionConflict { .. }) => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Err("bucket config CAS failed after max retries (persistent contention)".into())
}

/// The live materialized state of a collection reconstructed from object storage.
pub struct Materialized {
    /// Live chunks, keyed by id (deletes already applied).
    pub chunks: HashMap<u64, DocumentChunk>,
    /// Live relations, keyed by relation_id (deletes applied).
    pub relations: HashMap<String, ChunkRelation>,
    /// Highest chunk id ever observed (for the next_id high-water mark).
    pub max_id: u64,
}

/// On-disk segment: the full live set at compaction time. Versioned so the
/// format can evolve.
#[derive(Serialize, Deserialize, Default)]
pub struct Segment {
    #[serde(default)]
    pub version: u8,
    pub chunks: Vec<DocumentChunk>,
    #[serde(default)]
    pub relations: Vec<ChunkRelation>,
    /// Highest chunk id EVER assigned as of this compaction — including ids
    /// whose chunks were tombstoned and physically dropped here. Without it, a
    /// cold rebuild after compaction would compute next_id from live chunks
    /// only and REUSE deleted ids, breaking the monotonic-id invariant.
    #[serde(default)]
    pub max_id: u64,
    /// Chunk ids deleted in the folded range that may still exist in OLDER
    /// segments (partitioned compaction folds only the WAL tail, so deletes
    /// must carry across segment boundaries until a full merge drops them).
    #[serde(default)]
    pub tombstones: Vec<u64>,
    /// Relation ids deleted in the folded range (same cross-segment rule).
    #[serde(default)]
    pub relation_tombstones: Vec<String>,
}

/// v2 binary segment magic. v1 segments are JSON (decoded via fallback).
const SEG_MAGIC_V2: [u8; 8] = *b"CSEG0002";
/// v3 adds serve-from-storage sections: row-addressable chunk metadata
/// (`meta2` + `metaidx`) and IVF-clustered vectors (`cent:`/`clu:` replace
/// `emb:` for spaces past the clustering threshold). v3 readers decode v2;
/// v2 readers FAIL LOUDLY on v3 (magic mismatch) rather than silently
/// dropping sections — do not mix pre-v0.5 readers with v0.5 writers.
const SEG_MAGIC_V3: [u8; 8] = *b"CSEG0003";

/// Encode a segment in the v3 sectioned binary layout:
/// `[magic][u64 max_id][u32 toc_len][toc JSON][sections...]`
/// Sections:
///   `meta2`   — concatenated per-chunk JSON rows (embeddings stripped);
///                each row independently parseable so cold reads can fetch
///                single chunks by byte range
///   `metaidx` — `[u64 n][n × (u64 id, u64 off, u32 len)]` sorted by id,
///                offsets into `meta2`
///   `emb:<space>`  — flat `[u32 dims][u64 n][n × (u64 id + dims×f32)]`,
///                only for spaces below the clustering threshold
///   `cent:<space>` / `clu:<space>` — IVF centroids + clustered vectors
///                (see search/ivf.rs) for spaces at/above the threshold;
///                vectors are stored L2-normalized
///   `rels` (JSON), `tombs` (u64 LE array), `rtombs` (JSON ids)
pub fn encode_segment_v2(seg: &Segment) -> Result<Vec<u8>, StorageError> {
    let err = |e: String| StorageError::Io(format!("segment v3 encode: {e}"));
    let mut sections: Vec<(String, Vec<u8>)> = Vec::new();

    // Row-addressable metadata + index (sorted by id for range lookups).
    let mut sorted: Vec<&DocumentChunk> = seg.chunks.iter().collect();
    sorted.sort_by_key(|c| c.id);
    let mut meta2 = Vec::new();
    let mut metaidx = Vec::with_capacity(8 + sorted.len() * 20);
    metaidx.extend_from_slice(&(sorted.len() as u64).to_le_bytes());
    let mut by_space: std::collections::BTreeMap<String, Vec<(u64, Vec<f32>)>> =
        std::collections::BTreeMap::new();
    for c in sorted {
        let mut m = c.clone();
        for (space, emb) in std::mem::take(&mut m.embeddings) {
            by_space.entry(space).or_default().push((c.id, emb));
        }
        let row = serde_json::to_vec(&m).map_err(|e| err(e.to_string()))?;
        metaidx.extend_from_slice(&c.id.to_le_bytes());
        metaidx.extend_from_slice(&(meta2.len() as u64).to_le_bytes());
        metaidx.extend_from_slice(&(row.len() as u32).to_le_bytes());
        meta2.extend_from_slice(&row);
    }
    sections.push(("meta2".into(), meta2));
    sections.push(("metaidx".into(), metaidx));

    for (space, rows) in by_space {
        let dims = rows.first().map(|(_, v)| v.len()).unwrap_or(0);
        for (_, v) in &rows {
            if v.len() != dims {
                return Err(err(format!("ragged dims in space '{space}'")));
            }
        }
        if rows.len() >= crate::search::ivf::CLUSTER_MIN_ROWS && dims > 0 {
            let (cent, clu) = crate::search::ivf::build_sections(rows, dims);
            sections.push((format!("cent:{space}"), cent));
            sections.push((format!("clu:{space}"), clu));
        } else {
            let mut buf = Vec::with_capacity(12 + rows.len() * (8 + dims * 4));
            buf.extend_from_slice(&(dims as u32).to_le_bytes());
            buf.extend_from_slice(&(rows.len() as u64).to_le_bytes());
            for (id, v) in &rows {
                buf.extend_from_slice(&id.to_le_bytes());
                for x in v {
                    buf.extend_from_slice(&x.to_le_bytes());
                }
            }
            sections.push((format!("emb:{space}"), buf));
        }
    }
    sections.push((
        "rels".into(),
        serde_json::to_vec(&seg.relations).map_err(|e| err(e.to_string()))?,
    ));
    let mut tombs = Vec::with_capacity(seg.tombstones.len() * 8);
    for id in &seg.tombstones {
        tombs.extend_from_slice(&id.to_le_bytes());
    }
    sections.push(("tombs".into(), tombs));
    sections.push((
        "rtombs".into(),
        serde_json::to_vec(&seg.relation_tombstones).map_err(|e| err(e.to_string()))?,
    ));

    let toc: Vec<(String, u64)> = sections
        .iter()
        .map(|(n, b)| (n.clone(), b.len() as u64))
        .collect();
    let toc_bytes = serde_json::to_vec(&toc).map_err(|e| err(e.to_string()))?;
    let mut out = Vec::new();
    out.extend_from_slice(&SEG_MAGIC_V3);
    out.extend_from_slice(&seg.max_id.to_le_bytes());
    out.extend_from_slice(&(toc_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&toc_bytes);
    for (_, b) in sections {
        out.extend_from_slice(&b);
    }
    Ok(out)
}

fn decode_segment_v2(bytes: &[u8]) -> Result<Segment, StorageError> {
    let err = |e: String| StorageError::Io(format!("segment v2 decode: {e}"));
    let need = |n: usize, have: usize| -> Result<(), StorageError> {
        if have < n {
            Err(err("truncated".into()))
        } else {
            Ok(())
        }
    };
    need(20, bytes.len())?;
    let max_id = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    let toc_len = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
    need(20 + toc_len, bytes.len())?;
    let toc: Vec<(String, u64)> =
        serde_json::from_slice(&bytes[20..20 + toc_len]).map_err(|e| err(e.to_string()))?;
    let mut pos = 20 + toc_len;
    let mut seg = Segment {
        version: 2,
        max_id,
        ..Default::default()
    };
    let mut embs: HashMap<u64, HashMap<String, Vec<f32>>> = HashMap::new();
    let mut cent_dims: HashMap<String, usize> = HashMap::new();
    for (name, len) in toc {
        let len = len as usize;
        need(pos + len, bytes.len())?;
        let body = &bytes[pos..pos + len];
        pos += len;
        if name == "meta" {
            seg.chunks = serde_json::from_slice(body).map_err(|e| err(e.to_string()))?;
        } else if name == "meta2" {
            // v3 row-addressable metadata: concatenated standalone JSON rows.
            let mut de = serde_json::Deserializer::from_slice(body).into_iter::<DocumentChunk>();
            for c in de.by_ref() {
                seg.chunks.push(c.map_err(|e| err(e.to_string()))?);
            }
        } else if name == "metaidx" {
            // Full decode doesn't need the index (meta2 rows stream in order);
            // it exists for cold range reads.
        } else if name.starts_with("cent:") {
            let c = crate::search::ivf::parse_cent(body)
                .ok_or_else(|| err(format!("bad {name} section")))?;
            cent_dims.insert(name.clone(), c.dims);
        } else if let Some(space) = name.strip_prefix("clu:") {
            let cent_key = format!("cent:{space}");
            let dims = cent_dims
                .get(&cent_key)
                .copied()
                .ok_or_else(|| err(format!("clu:{space} without preceding cent section")))?;
            for (id, v) in crate::search::ivf::parse_cluster_rows(body, dims) {
                embs.entry(id).or_default().insert(space.to_string(), v);
            }
        } else if let Some(space) = name.strip_prefix("emb:") {
            need(12, body.len())?;
            let dims = u32::from_le_bytes(body[0..4].try_into().unwrap()) as usize;
            let n = u64::from_le_bytes(body[4..12].try_into().unwrap()) as usize;
            let row = 8 + dims * 4;
            need(12 + n * row, body.len())?;
            for i in 0..n {
                let off = 12 + i * row;
                let id = u64::from_le_bytes(body[off..off + 8].try_into().unwrap());
                let mut v = Vec::with_capacity(dims);
                for d in 0..dims {
                    let o = off + 8 + d * 4;
                    v.push(f32::from_le_bytes(body[o..o + 4].try_into().unwrap()));
                }
                embs.entry(id).or_default().insert(space.to_string(), v);
            }
        } else if name == "rels" {
            seg.relations = serde_json::from_slice(body).map_err(|e| err(e.to_string()))?;
        } else if name == "tombs" {
            seg.tombstones = body
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                .collect();
        } else if name == "rtombs" {
            seg.relation_tombstones =
                serde_json::from_slice(body).map_err(|e| err(e.to_string()))?;
        }
        // Unknown sections are skipped (forward compat).
    }
    for c in &mut seg.chunks {
        if let Some(e) = embs.remove(&c.id) {
            c.embeddings = e;
        }
    }
    Ok(seg)
}

/// Serialize a live set as a segment payload (v2 binary). `max_id` must be
/// the id high-water mark INCLUDING tombstoned ids.
pub fn encode_segment(
    chunks: &[DocumentChunk],
    relations: &[ChunkRelation],
    max_id: u64,
) -> Result<Vec<u8>, StorageError> {
    encode_segment_v2(&Segment {
        version: 2,
        chunks: chunks.to_vec(),
        relations: relations.to_vec(),
        max_id,
        tombstones: Vec::new(),
        relation_tombstones: Vec::new(),
    })
}

fn decode_segment(bytes: &[u8]) -> Result<Segment, StorageError> {
    // v2 binary (magic-tagged) first; then v1 JSON object; then the oldest
    // bare-JSON-array form.
    if bytes.len() >= 8 && (bytes[0..8] == SEG_MAGIC_V2 || bytes[0..8] == SEG_MAGIC_V3) {
        return decode_segment_v2(bytes);
    }
    if let Ok(seg) = serde_json::from_slice::<Segment>(bytes) {
        return Ok(seg);
    }
    let chunks: Vec<DocumentChunk> = serde_json::from_slice(bytes)
        .map_err(|e| StorageError::Io(format!("segment decode: {e}")))?;
    Ok(Segment {
        version: 0,
        chunks,
        ..Default::default()
    })
}

fn decode_chunks(bytes: &[u8]) -> Result<Vec<DocumentChunk>, StorageError> {
    serde_json::from_slice(bytes).map_err(|e| StorageError::Io(format!("chunk decode: {e}")))
}

fn decode_ids(bytes: &[u8]) -> Result<Vec<u64>, StorageError> {
    serde_json::from_slice(bytes).map_err(|e| StorageError::Io(format!("tombstone decode: {e}")))
}

fn decode_relations(bytes: &[u8]) -> Result<Vec<ChunkRelation>, StorageError> {
    serde_json::from_slice(bytes).map_err(|e| StorageError::Io(format!("relation decode: {e}")))
}

fn decode_relation_ids(bytes: &[u8]) -> Result<Vec<String>, StorageError> {
    serde_json::from_slice(bytes)
        .map_err(|e| StorageError::Io(format!("relation-delete decode: {e}")))
}

/// Fold ONLY a WAL tail (uncompacted fragments, in seq order) into a Segment
/// — the bounded-work unit of partitioned compaction. Deletes that don't hit
/// a chunk/relation within the tail are carried as segment tombstones so they
/// still apply to OLDER segments at materialize time.
pub fn fold_tail(frags: &[(lsm::FragmentRef, bytes::Bytes)]) -> Result<Segment, StorageError> {
    let mut chunks: HashMap<u64, DocumentChunk> = HashMap::new();
    let mut relations: HashMap<String, ChunkRelation> = HashMap::new();
    let mut tombs: std::collections::BTreeSet<u64> = Default::default();
    let mut rtombs: std::collections::BTreeSet<String> = Default::default();
    let mut max_id = 0u64;
    for (fref, bytes) in frags {
        match fref.kind {
            FragmentKind::Data => {
                for chunk in decode_chunks(bytes)? {
                    max_id = max_id.max(chunk.id);
                    tombs.remove(&chunk.id); // re-created after an earlier delete
                    chunks.insert(chunk.id, chunk);
                }
            }
            FragmentKind::Tombstone => {
                for id in decode_ids(bytes)? {
                    max_id = max_id.max(id);
                    chunks.remove(&id);
                    relations.retain(|_, r| r.source_chunk_id != id && r.target_chunk_id != id);
                    tombs.insert(id); // must ALSO apply to older segments
                }
            }
            FragmentKind::RelationUpsert => {
                for rel in decode_relations(bytes)? {
                    rtombs.remove(&rel.relation_id);
                    relations.insert(rel.relation_id.clone(), rel);
                }
            }
            FragmentKind::RelationDelete => {
                for rid in decode_relation_ids(bytes)? {
                    relations.remove(&rid);
                    rtombs.insert(rid);
                }
            }
        }
    }
    Ok(Segment {
        version: 2,
        chunks: chunks.into_values().collect(),
        relations: relations.into_values().collect(),
        max_id,
        tombstones: tombs.into_iter().collect(),
        relation_tombstones: rtombs.into_iter().collect(),
    })
}

/// Materialize the full live state (chunks + relations) from a manifest: read
/// all segments, then replay uncompacted fragments in seq order (latest-wins,
/// deletes applied). A chunk delete (tombstone) also drops any relation incident
/// on that chunk, matching the local delete-prunes-relations behavior.
pub async fn materialize(
    storage: &dyn Storage,
    ns: &str,
    manifest: &Manifest,
) -> Result<Materialized, StorageError> {
    let mut chunks: HashMap<u64, DocumentChunk> = HashMap::new();
    let mut relations: HashMap<String, ChunkRelation> = HashMap::new();
    let mut max_id: u64 = 0;

    // 1. Apply compacted segments first (oldest durable state).
    for seg in &manifest.segments {
        let bytes = lsm::read_segment(storage, ns, &seg.id).await?;
        let segment = decode_segment(&bytes)?;
        // The stored high-water mark covers tombstoned ids that compaction
        // physically dropped — required so next_id never regresses/reuses.
        max_id = max_id.max(segment.max_id);
        // Cross-segment deletes first: a tail-fold segment's tombstones apply
        // to everything OLDER than it (already accumulated), never to its own
        // surviving chunks (compaction removed those before encoding).
        for id in &segment.tombstones {
            max_id = max_id.max(*id);
            chunks.remove(id);
            relations.retain(|_, r| r.source_chunk_id != *id && r.target_chunk_id != *id);
        }
        for rid in &segment.relation_tombstones {
            relations.remove(rid);
        }
        for chunk in segment.chunks {
            max_id = max_id.max(chunk.id);
            chunks.insert(chunk.id, chunk);
        }
        for rel in segment.relations {
            relations.insert(rel.relation_id.clone(), rel);
        }
    }

    // 2. Replay uncompacted WAL fragments in seq order.
    let frags = lsm::read_uncompacted_fragments(storage, ns, manifest).await?;
    for (fref, bytes) in &frags {
        match fref.kind {
            FragmentKind::Data => {
                for chunk in decode_chunks(bytes)? {
                    max_id = max_id.max(chunk.id);
                    chunks.insert(chunk.id, chunk);
                }
            }
            FragmentKind::Tombstone => {
                for id in decode_ids(bytes)? {
                    max_id = max_id.max(id);
                    chunks.remove(&id);
                    // Drop relations incident on the deleted chunk.
                    relations.retain(|_, r| r.source_chunk_id != id && r.target_chunk_id != id);
                }
            }
            FragmentKind::RelationUpsert => {
                for rel in decode_relations(bytes)? {
                    relations.insert(rel.relation_id.clone(), rel);
                }
            }
            FragmentKind::RelationDelete => {
                for rid in decode_relation_ids(bytes)? {
                    relations.remove(&rid);
                }
            }
        }
    }

    Ok(Materialized {
        chunks,
        relations,
        max_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::local::LocalDiskStorage;
    use crate::storage::lsm;
    use bytes::Bytes;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    fn store(name: &str) -> Arc<dyn Storage> {
        static N: AtomicU64 = AtomicU64::new(0);
        let mut root = std::env::temp_dir();
        root.push(format!(
            "compass_cloud_test_{}_{}_{}",
            name,
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        Arc::new(LocalDiskStorage::new(root).unwrap())
    }

    fn chunk(id: u64, text: &str) -> DocumentChunk {
        DocumentChunk {
            id,
            collection: "t".into(),
            file_id: format!("f{id}"),
            chunk_index: 0,
            page: None,
            text: text.into(),
            metadata: HashMap::new(),
            doc_type: "chunk".into(),
            parent_id: None,
            group_id: None,
            embeddings: HashMap::new(),
            embedding: None,
        }
    }

    fn relation(id: &str, src: u64, tgt: u64) -> ChunkRelation {
        ChunkRelation {
            relation_id: id.into(),
            source_chunk_id: src,
            target_chunk_id: tgt,
            target_document_id: None,
            relation_type: "cites".into(),
            target_status: "found".into(),
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
        }
    }

    async fn mat(s: &dyn Storage, ns: &str) -> Materialized {
        let (m, _) = lsm::read_manifest(s, ns).await.unwrap();
        materialize(s, ns, &m).await.unwrap()
    }

    #[tokio::test]
    async fn empty_manifest_materializes_empty() {
        let s = store("empty");
        let r = mat(s.as_ref(), "ns").await;
        assert!(r.chunks.is_empty() && r.relations.is_empty());
        assert_eq!(r.max_id, 0);
    }

    #[tokio::test]
    async fn latest_wins_across_data_fragments() {
        let s = store("latest");
        // Two fragments upserting the SAME id; the later one must win.
        let f1 = serde_json::to_vec(&vec![chunk(5, "old")]).unwrap();
        lsm::append_fragment(s.as_ref(), "ns", Bytes::from(f1), 1)
            .await
            .unwrap();
        let f2 = serde_json::to_vec(&vec![chunk(5, "new")]).unwrap();
        lsm::append_fragment(s.as_ref(), "ns", Bytes::from(f2), 1)
            .await
            .unwrap();
        let r = mat(s.as_ref(), "ns").await;
        assert_eq!(r.chunks.len(), 1);
        assert_eq!(r.chunks[&5].text, "new", "latest fragment wins");
        assert_eq!(r.max_id, 5);
    }

    #[tokio::test]
    async fn tombstone_removes_and_prunes_relations() {
        let s = store("tomb");
        let data = serde_json::to_vec(&vec![chunk(1, "a"), chunk(2, "b")]).unwrap();
        lsm::append_fragment(s.as_ref(), "ns", Bytes::from(data), 2)
            .await
            .unwrap();
        // Relation 1->2, then delete chunk 2 → relation must be pruned too.
        let rels = serde_json::to_vec(&vec![relation("r1", 1, 2)]).unwrap();
        lsm::append_relation_upsert(s.as_ref(), "ns", Bytes::from(rels), 1)
            .await
            .unwrap();
        lsm::append_tombstone(s.as_ref(), "ns", &[2]).await.unwrap();
        let r = mat(s.as_ref(), "ns").await;
        assert_eq!(r.chunks.len(), 1);
        assert!(r.chunks.contains_key(&1) && !r.chunks.contains_key(&2));
        assert!(r.relations.is_empty(), "relation on deleted chunk 2 pruned");
    }

    #[tokio::test]
    async fn relation_delete_is_idempotent_noop() {
        let s = store("reldel");
        // Delete a relation that never existed — must not error, no relations.
        lsm::append_relation_delete(s.as_ref(), "ns", &["ghost".into()])
            .await
            .unwrap();
        let r = mat(s.as_ref(), "ns").await;
        assert!(r.relations.is_empty());
    }

    #[tokio::test]
    async fn back_compat_bare_array_segment_decodes() {
        // An OLD segment was a bare JSON array of chunks (no {chunks,relations}).
        let bare = serde_json::to_vec(&vec![chunk(7, "legacy")]).unwrap();
        let seg = decode_segment(&bare).unwrap();
        assert_eq!(seg.chunks.len(), 1);
        assert_eq!(seg.chunks[0].id, 7);
        assert!(seg.relations.is_empty());
    }

    #[tokio::test]
    async fn segment_and_fragment_layer_correctly() {
        let s = store("layer");
        // Build a data fragment, compact-style: write a segment via encode_segment
        // referencing id 1, then a fragment upserting id 2 and re-upserting id 1.
        let seg_bytes = encode_segment(&[chunk(1, "seg-v1")], &[relation("r1", 1, 2)], 1).unwrap();
        // Manually place the segment + manifest by using replace_with_single_segment.
        let (m0, v0) = lsm::read_manifest(s.as_ref(), "ns").await.unwrap();
        lsm::replace_with_single_segment(s.as_ref(), "ns", &v0, &m0, Bytes::from(seg_bytes), 1)
            .await
            .unwrap();
        // Now a WAL fragment re-upserts id 1 (should win over the segment) + adds 2.
        let frag = serde_json::to_vec(&vec![chunk(1, "frag-v2"), chunk(2, "new")]).unwrap();
        lsm::append_fragment(s.as_ref(), "ns", Bytes::from(frag), 2)
            .await
            .unwrap();

        let r = mat(s.as_ref(), "ns").await;
        assert_eq!(r.chunks.len(), 2);
        assert_eq!(r.chunks[&1].text, "frag-v2", "fragment overrides segment");
        assert_eq!(r.chunks[&2].text, "new");
        // The relation from the segment survives.
        assert_eq!(r.relations.len(), 1);
        assert!(r.relations.contains_key("r1"));
    }

    // F2 residual: if an ingest's S3 fragment is durable but the local commit
    // failed, the manager writes a COMPENSATING tombstone for those ids. This
    // proves the mechanism: a data fragment + a tombstone for the same ids
    // materializes to nothing — the batch does not resurrect on cold rebuild.
    #[tokio::test]
    async fn compensating_tombstone_prevents_resurrection() {
        let s = store("compensate");
        // Durable fragment for ids 0,1 (the "S3 append succeeded" part).
        let data = serde_json::to_vec(&vec![chunk(0, "a"), chunk(1, "b")]).unwrap();
        lsm::append_fragment(s.as_ref(), "ns", Bytes::from(data), 2)
            .await
            .unwrap();
        // Local commit "failed" → manager compensates with a tombstone for 0,1.
        lsm::append_tombstone(s.as_ref(), "ns", &[0, 1])
            .await
            .unwrap();

        // Materialize (== cold rebuild): the batch must NOT come back.
        let r = mat(s.as_ref(), "ns").await;
        assert!(
            r.chunks.is_empty(),
            "compensated ingest must not resurrect, got {:?}",
            r.chunks.keys().collect::<Vec<_>>()
        );
    }

    // ── Segment v2 / partitioned compaction ───────────────────────────────

    #[test]
    fn segment_v2_roundtrip_with_embeddings_and_tombstones() {
        let mut c1 = chunk(1, "one");
        c1.embeddings
            .insert("default".into(), vec![0.1, 0.2, 0.3, 0.4]);
        let mut c2 = chunk(2, "two");
        c2.embeddings
            .insert("default".into(), vec![0.5, 0.6, 0.7, 0.8]);
        c2.embeddings.insert("wide".into(), vec![1.0; 8]);
        let seg = Segment {
            version: 2,
            chunks: vec![c1, c2],
            relations: vec![relation("r1", 1, 2)],
            max_id: 42,
            tombstones: vec![7, 9],
            relation_tombstones: vec!["dead".into()],
        };
        let bytes = encode_segment_v2(&seg).unwrap();
        assert_eq!(&bytes[0..8], b"CSEG0003");
        let back = decode_segment(&bytes).unwrap();
        assert_eq!(back.max_id, 42);
        assert_eq!(back.tombstones, vec![7, 9]);
        assert_eq!(back.relation_tombstones, vec!["dead".to_string()]);
        assert_eq!(back.chunks.len(), 2);
        let c2b = back.chunks.iter().find(|c| c.id == 2).unwrap();
        assert_eq!(c2b.embeddings["default"], vec![0.5, 0.6, 0.7, 0.8]);
        assert_eq!(c2b.embeddings["wide"].len(), 8);
        assert_eq!(back.relations.len(), 1);
    }

    // Past the clustering threshold the encoder emits cent:/clu: instead of
    // emb:; the full decode must reconstruct every chunk's (normalized)
    // embedding from the clustered layout.
    #[test]
    fn segment_v3_clustered_roundtrip() {
        let n = crate::search::ivf::CLUSTER_MIN_ROWS + 100;
        let chunks: Vec<DocumentChunk> = (0..n as u64)
            .map(|i| {
                let mut c = chunk(i, &format!("t{i}"));
                let v: Vec<f32> = (0..8).map(|d| ((i + d) % 13) as f32 + 1.0).collect();
                c.embeddings.insert("default".into(), v);
                c
            })
            .collect();
        let seg = Segment {
            version: 2,
            chunks,
            relations: vec![],
            max_id: n as u64,
            tombstones: vec![],
            relation_tombstones: vec![],
        };
        let bytes = encode_segment_v2(&seg).unwrap();
        let toc_len = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
        let toc = std::str::from_utf8(&bytes[20..20 + toc_len]).unwrap();
        assert!(toc.contains("cent:default"), "toc: {toc}");
        assert!(toc.contains("clu:default"), "toc: {toc}");
        assert!(!toc.contains("emb:default"), "toc: {toc}");

        let back = decode_segment(&bytes).unwrap();
        assert_eq!(back.chunks.len(), n);
        for c in &back.chunks {
            let v = &c.embeddings["default"];
            assert_eq!(v.len(), 8);
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "clu vectors are unit-norm");
        }
    }

    // A delete folded into a NEWER tail segment must erase a chunk living in
    // an OLDER segment at materialize time (cross-segment tombstones).
    #[tokio::test]
    async fn tail_segment_tombstones_apply_to_older_segments() {
        let s = store("xseg");
        // Older state via a REAL fold: data + relation fragments -> segment A.
        let mut c1 = chunk(1, "old");
        c1.embeddings
            .insert("default".into(), vec![0.1, 0.2, 0.3, 0.4]);
        let data = serde_json::to_vec(&vec![c1, chunk(2, "keep")]).unwrap();
        lsm::append_fragment(s.as_ref(), "ns", Bytes::from(data), 2)
            .await
            .unwrap();
        let rels = serde_json::to_vec(&vec![relation("r1", 1, 2)]).unwrap();
        lsm::append_relation_upsert(s.as_ref(), "ns", Bytes::from(rels), 1)
            .await
            .unwrap();
        let fold_once = |sref: Arc<dyn Storage>| async move {
            let (m1, v1) = lsm::read_manifest(sref.as_ref(), "ns").await.unwrap();
            let frags = lsm::read_uncompacted_fragments(sref.as_ref(), "ns", &m1)
                .await
                .unwrap();
            let tail = fold_tail(&frags).unwrap();
            let folded_through = m1.uncompacted().map(|f| f.seq).max().unwrap();
            let records = tail.chunks.len() as u64;
            lsm::append_segment(
                sref.as_ref(),
                "ns",
                &v1,
                &m1,
                Bytes::from(encode_segment_v2(&tail).unwrap()),
                records,
                folded_through,
            )
            .await
            .unwrap();
            tail
        };
        let seg_a = fold_once(s.clone()).await;
        assert!(seg_a.tombstones.is_empty());

        // Newer tail: delete chunk 1; the delete finds nothing IN the tail so
        // it must be carried as a cross-segment tombstone.
        lsm::append_tombstone(s.as_ref(), "ns", &[1]).await.unwrap();
        let seg_b = fold_once(s.clone()).await;
        assert_eq!(
            seg_b.tombstones,
            vec![1],
            "unmatched delete carried forward"
        );

        let r = mat(s.as_ref(), "ns").await;
        assert!(!r.chunks.contains_key(&1), "older-segment chunk deleted");
        assert!(r.chunks.contains_key(&2));
        assert!(r.relations.is_empty(), "incident relation pruned");
        assert_eq!(r.max_id, 2);
    }
}
