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
}

const SEGMENT_VERSION: u8 = 1;

/// Serialize a live set as a segment payload. `max_id` must be the id
/// high-water mark INCLUDING tombstoned ids (pass `Materialized::max_id`, not
/// the max of the live set).
pub fn encode_segment(
    chunks: &[DocumentChunk],
    relations: &[ChunkRelation],
    max_id: u64,
) -> Result<Vec<u8>, StorageError> {
    let seg = Segment {
        version: SEGMENT_VERSION,
        chunks: chunks.to_vec(),
        relations: relations.to_vec(),
        max_id,
    };
    serde_json::to_vec(&seg).map_err(|e| StorageError::Io(format!("segment encode: {e}")))
}

fn decode_segment(bytes: &[u8]) -> Result<Segment, StorageError> {
    // Back-compat: an older segment was a bare JSON array of chunks. Try the
    // versioned object first, then fall back to a plain chunk array.
    if let Ok(seg) = serde_json::from_slice::<Segment>(bytes) {
        return Ok(seg);
    }
    let chunks: Vec<DocumentChunk> = serde_json::from_slice(bytes)
        .map_err(|e| StorageError::Io(format!("segment decode: {e}")))?;
    Ok(Segment {
        version: 0,
        chunks,
        relations: Vec::new(),
        max_id: 0,
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
}
