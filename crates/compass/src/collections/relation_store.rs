//! Disk-backed chunk-relation store using redb.
//!
//! Directed, typed, many-to-many edges between chunks: each row is one edge
//! `source_chunk_id --relation_type--> target_chunk_id`. Modeled on
//! [`crate::search::chunk_store::ChunkStore`] — same redb open/lock-retry +
//! repair-callback discipline, same batched-write transactions.
//!
//! Three tables:
//!   - `relations`      : relation_id (&str)     -> serialized ChunkRelation
//!   - `by_source_idx`  : source_chunk_id (u64)  ->> relation_ids (multimap)
//!   - `by_target_idx`  : target_chunk_id (u64)  ->> relation_ids (multimap)
//!
//! The two endpoint indexes make "relations of chunk X" a keyed lookup, not a
//! scan. They are redb MULTIMAP tables: inserting or removing one edge touches
//! one entry — no read-append-rewrite of a per-chunk id blob, so a hub chunk
//! with N edges costs O(N log N) to build, not O(N²). (The `_idx` names are
//! deliberate: pre-release dev files used plain tables named `by_source` /
//! `by_target`; distinct names keep those files openable.)
//!
//! Source of truth is on disk; nothing is held resident in RAM. Search reads
//! relations on demand via [`RelationStore::for_chunks`] — the disk-served
//! pattern, never a rehydrated graph. This keeps the store ready to move
//! behind the future `Storage` trait (object storage) without a redesign.

use crate::models::{ChunkRelation, RelationDirection};
use redb::{Database, DatabaseError, MultimapTableDefinition, ReadableTable, TableDefinition};
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

const RELATIONS_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("relations");
const BY_SOURCE_TABLE: MultimapTableDefinition<u64, &str> =
    MultimapTableDefinition::new("by_source_idx");
const BY_TARGET_TABLE: MultimapTableDefinition<u64, &str> =
    MultimapTableDefinition::new("by_target_idx");

/// Matches ChunkStore's lock-retry budget (~30s) for networked-filesystem
/// flock-release latency.
const OPEN_MAX_ATTEMPTS: u32 = 6;
const OPEN_RETRY_BACKOFF: Duration = Duration::from_secs(5);

type BoxErr = Box<dyn std::error::Error + Send + Sync>;

pub struct RelationStore {
    db: Database,
}

impl RelationStore {
    /// Open or create the relation store at `path`. Tolerates the transient
    /// `DatabaseAlreadyOpen` flock error on networked filesystems, and enables
    /// redb's repair callback so a dirty file auto-recovers.
    pub fn open(path: &Path) -> Result<Self, BoxErr> {
        Self::open_with_retries(path, OPEN_MAX_ATTEMPTS, OPEN_RETRY_BACKOFF)
    }

    pub(crate) fn open_with_retries(
        path: &Path,
        max_attempts: u32,
        backoff: Duration,
    ) -> Result<Self, BoxErr> {
        let mut attempt = 0u32;
        let db = loop {
            attempt += 1;
            match Database::builder().set_repair_callback(|_| {}).create(path) {
                Ok(db) => break db,
                Err(DatabaseError::DatabaseAlreadyOpen) if attempt < max_attempts => {
                    tracing::warn!(
                        "relations redb file {:?} is locked (attempt {}/{}). Retrying in {:?}.",
                        path,
                        attempt,
                        max_attempts,
                        backoff
                    );
                    std::thread::sleep(backoff);
                }
                Err(e) => return Err(e.into()),
            }
        };
        // Create all three tables up front so reads never hit a missing table.
        {
            let txn = db.begin_write()?;
            let _ = txn.open_table(RELATIONS_TABLE)?;
            let _ = txn.open_multimap_table(BY_SOURCE_TABLE)?;
            let _ = txn.open_multimap_table(BY_TARGET_TABLE)?;
            txn.commit()?;
        }
        Ok(Self { db })
    }

    /// Insert a batch of edges in one transaction. Updates the primary table and
    /// both endpoint indexes. Caller is responsible for assigning `relation_id`
    /// and rejecting self-relations (see the API handler / manager). Re-inserting
    /// an existing relation_id with different endpoints prunes its old index
    /// entries first, so the endpoint indexes never go stale.
    pub fn insert_batch(&self, relations: &[ChunkRelation]) -> Result<(), BoxErr> {
        let txn = self.db.begin_write()?;
        {
            let mut rels = txn.open_table(RELATIONS_TABLE)?;
            let mut by_source = txn.open_multimap_table(BY_SOURCE_TABLE)?;
            let mut by_target = txn.open_multimap_table(BY_TARGET_TABLE)?;

            for r in relations {
                let bytes = serde_json::to_vec(r)?;
                // Upsert semantics: if this relation_id already exists with
                // different endpoints, drop the old index entries.
                let prior: Option<ChunkRelation> = {
                    match rels.get(r.relation_id.as_str())? {
                        Some(v) => Some(serde_json::from_slice(v.value())?),
                        None => None,
                    }
                };
                if let Some(old) = prior {
                    if old.source_chunk_id != r.source_chunk_id {
                        by_source.remove(old.source_chunk_id, old.relation_id.as_str())?;
                    }
                    if old.target_chunk_id != r.target_chunk_id {
                        by_target.remove(old.target_chunk_id, old.relation_id.as_str())?;
                    }
                }
                rels.insert(r.relation_id.as_str(), bytes.as_slice())?;
                // Multimap insert: one entry per edge, deduped by redb — no
                // read-append-rewrite of a per-chunk blob.
                by_source.insert(r.source_chunk_id, r.relation_id.as_str())?;
                by_target.insert(r.target_chunk_id, r.relation_id.as_str())?;
            }
        }
        txn.commit()?;
        Ok(())
    }

    /// Delete an edge by id. Removes it from the primary table and both endpoint
    /// indexes. Returns true if the relation existed.
    pub fn delete(&self, relation_id: &str) -> Result<bool, BoxErr> {
        // Read the edge first so we know which endpoint index entries to prune.
        let existing: Option<ChunkRelation> = {
            let txn = self.db.begin_read()?;
            let rels = txn.open_table(RELATIONS_TABLE)?;
            match rels.get(relation_id)? {
                Some(v) => Some(serde_json::from_slice(v.value())?),
                None => None,
            }
        };
        let Some(rel) = existing else {
            return Ok(false);
        };

        let txn = self.db.begin_write()?;
        {
            let mut rels = txn.open_table(RELATIONS_TABLE)?;
            let mut by_source = txn.open_multimap_table(BY_SOURCE_TABLE)?;
            let mut by_target = txn.open_multimap_table(BY_TARGET_TABLE)?;
            rels.remove(relation_id)?;
            by_source.remove(rel.source_chunk_id, relation_id)?;
            by_target.remove(rel.target_chunk_id, relation_id)?;
        }
        txn.commit()?;
        Ok(true)
    }

    /// All relations incident on a single chunk, filtered by direction and
    /// (optionally) relation_type. `target_status` is left as-stored ("unknown")
    /// here; the caller resolves it against the chunk set.
    pub fn for_chunk(
        &self,
        chunk_id: u64,
        dir: RelationDirection,
        types: Option<&[String]>,
    ) -> Result<Vec<ChunkRelation>, BoxErr> {
        let map = self.for_chunks(&[chunk_id], dir, types)?;
        Ok(map.into_values().next().unwrap_or_default())
    }

    /// Batched hot path: relations for a page of chunk ids, keyed by chunk id.
    /// One read transaction for the whole page. An id with no relations is
    /// simply absent from the returned map.
    pub fn for_chunks(
        &self,
        chunk_ids: &[u64],
        dir: RelationDirection,
        types: Option<&[String]>,
    ) -> Result<HashMap<u64, Vec<ChunkRelation>>, BoxErr> {
        let txn = self.db.begin_read()?;
        let rels = txn.open_table(RELATIONS_TABLE)?;
        let by_source = txn.open_multimap_table(BY_SOURCE_TABLE)?;
        let by_target = txn.open_multimap_table(BY_TARGET_TABLE)?;

        let mut out: HashMap<u64, Vec<ChunkRelation>> = HashMap::new();

        for &cid in chunk_ids {
            // Collect candidate relation ids from the requested endpoint(s),
            // deduplicated (an edge can match both endpoints only if src==target,
            // which we reject at insert, but dedup keeps Both robust regardless).
            let mut ids: Vec<String> = Vec::new();
            if matches!(dir, RelationDirection::Outgoing | RelationDirection::Both) {
                for v in by_source.get(cid)? {
                    ids.push(v?.value().to_string());
                }
            }
            if matches!(dir, RelationDirection::Incoming | RelationDirection::Both) {
                for v in by_target.get(cid)? {
                    let id = v?.value().to_string();
                    if !ids.contains(&id) {
                        ids.push(id);
                    }
                }
            }
            if ids.is_empty() {
                continue;
            }

            let mut edges: Vec<ChunkRelation> = Vec::with_capacity(ids.len());
            for id in ids {
                if let Some(v) = rels.get(id.as_str())? {
                    let rel: ChunkRelation = serde_json::from_slice(v.value())?;
                    // Defensive: trust-but-verify the endpoint index. A reused
                    // relation_id re-inserted with different endpoints leaves a
                    // stale index entry; never return an edge that doesn't
                    // actually touch this chunk.
                    if rel.source_chunk_id != cid && rel.target_chunk_id != cid {
                        continue;
                    }
                    if let Some(filter) = types {
                        if !filter.iter().any(|t| t == &rel.relation_type) {
                            continue;
                        }
                    }
                    edges.push(rel);
                }
            }
            if !edges.is_empty() {
                out.insert(cid, edges);
            }
        }
        Ok(out)
    }

    /// Total number of stored edges. Used by tests + diagnostics.
    pub fn count(&self) -> Result<u64, BoxErr> {
        use redb::ReadableTableMetadata;
        let txn = self.db.begin_read()?;
        let rels = txn.open_table(RELATIONS_TABLE)?;
        Ok(rels.len()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn rel(id: &str, src: u64, tgt: u64, rtype: &str) -> ChunkRelation {
        ChunkRelation {
            relation_id: id.to_string(),
            source_chunk_id: src,
            target_chunk_id: tgt,
            target_document_id: None,
            relation_type: rtype.to_string(),
            target_status: "unknown".to_string(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
        }
    }

    fn store(name: &str) -> RelationStore {
        let mut p = std::env::temp_dir();
        p.push(format!("compass_relstore_{}.redb", name));
        let _ = std::fs::remove_file(&p);
        RelationStore::open(&p).unwrap()
    }

    #[test]
    fn insert_then_read_outgoing_and_incoming() {
        let s = store("dir");
        s.insert_batch(&[rel("r1", 1, 2, "cites"), rel("r2", 3, 1, "supersedes")])
            .unwrap();
        assert_eq!(s.count().unwrap(), 2);

        // Outgoing from chunk 1: only r1 (1 is the source).
        let out = s.for_chunk(1, RelationDirection::Outgoing, None).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].relation_id, "r1");

        // Incoming to chunk 1: only r2 (1 is the target).
        let inc = s.for_chunk(1, RelationDirection::Incoming, None).unwrap();
        assert_eq!(inc.len(), 1);
        assert_eq!(inc[0].relation_id, "r2");

        // Both: r1 and r2.
        let both = s.for_chunk(1, RelationDirection::Both, None).unwrap();
        assert_eq!(both.len(), 2);
    }

    #[test]
    fn type_filter() {
        let s = store("types");
        s.insert_batch(&[
            rel("r1", 1, 2, "cites"),
            rel("r2", 1, 3, "supersedes"),
            rel("r3", 1, 4, "cites"),
        ])
        .unwrap();
        let cites = s
            .for_chunk(1, RelationDirection::Outgoing, Some(&["cites".to_string()]))
            .unwrap();
        assert_eq!(cites.len(), 2);
        assert!(cites.iter().all(|r| r.relation_type == "cites"));
    }

    #[test]
    fn delete_prunes_both_indexes() {
        let s = store("delete");
        s.insert_batch(&[rel("r1", 1, 2, "cites")]).unwrap();
        assert!(s.delete("r1").unwrap());
        assert_eq!(s.count().unwrap(), 0);
        assert!(s
            .for_chunk(1, RelationDirection::Outgoing, None)
            .unwrap()
            .is_empty());
        assert!(s
            .for_chunk(2, RelationDirection::Incoming, None)
            .unwrap()
            .is_empty());
        // Deleting a missing id is a no-op false.
        assert!(!s.delete("nope").unwrap());
    }

    #[test]
    fn for_chunks_batched() {
        let s = store("batch");
        s.insert_batch(&[rel("r1", 1, 2, "cites"), rel("r2", 5, 6, "cites")])
            .unwrap();
        let map = s
            .for_chunks(&[1, 5, 99], RelationDirection::Outgoing, None)
            .unwrap();
        assert_eq!(map.len(), 2); // chunk 99 has none, absent from map
        assert_eq!(map.get(&1).unwrap()[0].relation_id, "r1");
        assert_eq!(map.get(&5).unwrap()[0].relation_id, "r2");
    }

    // Upserting an existing relation_id with DIFFERENT endpoints must prune the
    // old endpoint-index entries — a stale entry would surface the edge for a
    // chunk it no longer touches (defensively filtered on read, but the index
    // itself must not rot).
    #[test]
    fn upsert_with_changed_endpoints_prunes_stale_index() {
        let s = store("upsert");
        s.insert_batch(&[rel("r1", 1, 2, "cites")]).unwrap();
        // Same id, new endpoints 3 -> 4.
        s.insert_batch(&[rel("r1", 3, 4, "cites")]).unwrap();
        assert_eq!(s.count().unwrap(), 1);

        // Old endpoints no longer index the edge…
        assert!(s
            .for_chunk(1, RelationDirection::Both, None)
            .unwrap()
            .is_empty());
        assert!(s
            .for_chunk(2, RelationDirection::Both, None)
            .unwrap()
            .is_empty());
        // …the new ones do.
        assert_eq!(
            s.for_chunk(3, RelationDirection::Outgoing, None)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            s.for_chunk(4, RelationDirection::Incoming, None)
                .unwrap()
                .len(),
            1
        );
    }

    // Hub chunk: many edges on one source. With multimap indexes this is
    // one entry per edge (no per-chunk blob rewrite); verify correctness at a
    // size that would have been visibly quadratic before.
    #[test]
    fn hub_chunk_many_edges() {
        let s = store("hub");
        let edges: Vec<ChunkRelation> = (0..2_000u64)
            .map(|i| rel(&format!("r{i}"), 1, i + 2, "cites"))
            .collect();
        s.insert_batch(&edges).unwrap();
        assert_eq!(s.count().unwrap(), 2_000);
        let out = s.for_chunk(1, RelationDirection::Outgoing, None).unwrap();
        assert_eq!(out.len(), 2_000);
        // Delete one from the middle; the rest stay intact.
        assert!(s.delete("r1000").unwrap());
        let out = s.for_chunk(1, RelationDirection::Outgoing, None).unwrap();
        assert_eq!(out.len(), 1_999);
    }

    #[test]
    fn persists_across_reopen() {
        let mut p = std::env::temp_dir();
        p.push("compass_relstore_persist.redb");
        let _ = std::fs::remove_file(&p);
        {
            let s = RelationStore::open(&p).unwrap();
            s.insert_batch(&[rel("r1", 1, 2, "cites")]).unwrap();
        }
        let s2 = RelationStore::open(&p).unwrap();
        assert_eq!(s2.count().unwrap(), 1);
        let _ = std::fs::remove_file(&p);
    }
}
