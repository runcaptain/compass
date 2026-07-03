//! LSM persistence over the [`Storage`] trait — WAL fragments + manifest + compaction.
//!
//! This is the object-storage answer to "do everything redb does, but in object
//! storage." redb can't run on object storage (mmap'd local file), so durable
//! state becomes an LSM — the model proven by object-storage-native databases:
//!
//!   - **WAL fragments** — each write batch becomes an immutable object
//!     `{ns}/wal/{uuid}.frag`, written first (durability). Keys are UUIDs so
//!     concurrent appends never collide; replay order comes from the manifest.
//!   - **Manifest** — `{ns}/manifest` lists the live fragments + segments and a
//!     compaction watermark. Committed by **CAS** (`put_if_match`) so concurrent
//!     writers serialize without a lock; on conflict the writer re-reads + retries.
//!   - **Compaction** — folds WAL fragments into a compacted segment object and
//!     advances the watermark, then defers fragment deletion.
//!
//! It is generic over any [`Storage`] backend, so the identical code path runs
//! on local disk or S3/GCS/Azure. This module provides the substrate; wiring the
//! engine's chunks/vectors/relations onto it is a later step. The payloads here
//! are opaque `Bytes` — the LSM does not care what a fragment contains.

use super::{Storage, StorageError, Version};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Max CAS attempts when committing the manifest before giving up.
const MAX_CAS_RETRIES: u32 = 10;

/// What a WAL fragment contains.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum FragmentKind {
    /// Upserted chunk records (the default; older manifests without this field).
    #[default]
    Data,
    /// A tombstone batch: a list of chunk ids to treat as deleted.
    Tombstone,
    /// A batch of created chunk relations (payload = JSON array of ChunkRelation).
    RelationUpsert,
    /// A batch of deleted relation ids (payload = JSON array of relation_id strings).
    RelationDelete,
}

/// A reference to a WAL fragment object recorded in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FragmentRef {
    /// Globally-unique fragment id (UUIDv4). This is the object-key stem, so two
    /// concurrent appends NEVER collide on the same key (fixes the seq-clobber
    /// bug: the old scheme keyed by `next_seq`, which racing writers could both
    /// pick, overwriting each other's fragment object).
    pub id: String,
    /// Monotonic ordering sequence, assigned at manifest-commit time (NOT the
    /// object key). Determines fragment replay order for latest-wins semantics.
    pub seq: u64,
    /// Number of logical records in the fragment (for planning/metrics).
    pub records: u64,
    /// Whether this fragment holds data upserts or deletion tombstones.
    #[serde(default)]
    pub kind: FragmentKind,
}

/// A reference to a compacted segment object recorded in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SegmentRef {
    pub id: String,
    pub records: u64,
}

/// The manifest: the single source of truth for what is live in a namespace.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct Manifest {
    /// Uncompacted WAL fragments, in seq order.
    pub fragments: Vec<FragmentRef>,
    /// Compacted segments.
    pub segments: Vec<SegmentRef>,
    /// Next sequence number to assign to a fragment.
    pub next_seq: u64,
    /// Fragments with seq <= this have been folded into a segment. `None` means
    /// nothing has been compacted yet (distinct from `Some(0)`, which means the
    /// seq-0 fragment IS compacted). Using Option avoids the off-by-one where a
    /// fresh manifest (watermark 0) would wrongly treat fragment seq=0 as compacted.
    pub compaction_watermark: Option<u64>,
    /// Object keys staged for deferred deletion (old fragments/segments). GC'd
    /// on the next compaction so in-flight readers holding an older manifest
    /// snapshot don't hit a missing object.
    pub pending_deletes: Vec<String>,
}

impl Manifest {
    /// Fragments strictly after the compaction watermark (the live tail).
    /// With no watermark yet (`None`), every fragment is uncompacted.
    pub fn uncompacted(&self) -> impl Iterator<Item = &FragmentRef> {
        let wm = self.compaction_watermark;
        self.fragments
            .iter()
            .filter(move |f| wm.map(|w| f.seq > w).unwrap_or(true))
    }
}

fn manifest_key(ns: &str) -> String {
    format!("{ns}/manifest")
}

fn fragment_key(ns: &str, id: &str) -> String {
    // Keyed by the fragment's unique id (UUID), so concurrent appends can't
    // collide on the object key. Replay ORDER comes from the manifest `seq`,
    // not the key.
    format!("{ns}/wal/{id}.frag")
}

fn segment_key(ns: &str, id: &str) -> String {
    format!("{ns}/segments/{id}")
}

/// Read the manifest for a namespace together with its version (for CAS).
/// Returns a fresh default manifest (version None) if none exists yet.
pub async fn read_manifest(
    storage: &dyn Storage,
    ns: &str,
) -> Result<(Manifest, Option<Version>), StorageError> {
    match storage.get_versioned(&manifest_key(ns)).await {
        Ok((bytes, version)) => {
            let m: Manifest = serde_json::from_slice(&bytes)
                .map_err(|e| StorageError::Io(format!("manifest decode: {e}")))?;
            Ok((m, Some(version)))
        }
        Err(StorageError::NotFound(_)) => Ok((Manifest::default(), None)),
        Err(e) => Err(e),
    }
}

/// Commit a manifest via CAS. `expected` is the version read alongside it
/// (None = the manifest must not yet exist → create-only). Returns the new
/// version. The caller is expected to retry the whole read-modify-write cycle
/// on [`StorageError::VersionConflict`].
async fn commit_manifest(
    storage: &dyn Storage,
    ns: &str,
    manifest: &Manifest,
    expected: &Option<Version>,
) -> Result<Version, StorageError> {
    let bytes = Bytes::from(
        serde_json::to_vec(manifest)
            .map_err(|e| StorageError::Io(format!("manifest encode: {e}")))?,
    );
    let key = manifest_key(ns);
    match expected {
        Some(v) => storage.put_if_match(&key, bytes, v).await,
        None => storage.put_if_not_exists(&key, bytes).await,
    }
}

/// Fetch one WAL fragment's payload by id.
pub async fn read_fragment(
    storage: &dyn Storage,
    ns: &str,
    id: &str,
) -> Result<Bytes, StorageError> {
    storage.get(&fragment_key(ns, id)).await
}

/// Create-only commit of an EMPTY manifest for a new namespace, making a
/// zero-ingest collection discoverable (`list_namespaces` keys off
/// `{ns}/manifest`). `AlreadyExists` bubbles up — it means the namespace
/// already has data in the bucket (e.g. a pre-existing collection).
pub async fn init_namespace(storage: &dyn Storage, ns: &str) -> Result<(), StorageError> {
    commit_manifest(storage, ns, &Manifest::default(), &None)
        .await
        .map(|_| ())
}

/// Append a data WAL fragment. Returns the assigned sequence number.
pub async fn append_fragment(
    storage: &dyn Storage,
    ns: &str,
    payload: Bytes,
    records: u64,
) -> Result<u64, StorageError> {
    append_fragment_kind(storage, ns, payload, records, FragmentKind::Data).await
}

/// Append a tombstone WAL fragment marking `deleted_ids` as deleted. The payload
/// is a JSON array of the deleted chunk ids. Replayed latest-wins on read and
/// dropped-with-their-records at compaction.
pub async fn append_tombstone(
    storage: &dyn Storage,
    ns: &str,
    deleted_ids: &[u64],
) -> Result<u64, StorageError> {
    let payload = Bytes::from(
        serde_json::to_vec(deleted_ids)
            .map_err(|e| StorageError::Io(format!("tombstone encode: {e}")))?,
    );
    let records = deleted_ids.len() as u64;
    append_fragment_kind(storage, ns, payload, records, FragmentKind::Tombstone).await
}

/// Append a relation-upsert fragment. `payload` is a pre-serialized JSON array of
/// the created relations (the caller owns the concrete relation type).
pub async fn append_relation_upsert(
    storage: &dyn Storage,
    ns: &str,
    payload: Bytes,
    records: u64,
) -> Result<u64, StorageError> {
    append_fragment_kind(storage, ns, payload, records, FragmentKind::RelationUpsert).await
}

/// Append a relation-delete fragment marking `relation_ids` as removed.
pub async fn append_relation_delete(
    storage: &dyn Storage,
    ns: &str,
    relation_ids: &[String],
) -> Result<u64, StorageError> {
    let payload = Bytes::from(
        serde_json::to_vec(relation_ids)
            .map_err(|e| StorageError::Io(format!("relation-delete encode: {e}")))?,
    );
    let records = relation_ids.len() as u64;
    append_fragment_kind(storage, ns, payload, records, FragmentKind::RelationDelete).await
}

/// Shared append: mint a UNIQUE fragment id, write the immutable
/// fragment object at its own key (so concurrent appends never collide), then
/// CAS-commit the manifest with a ref whose `seq` is assigned at commit time.
/// Retries the manifest commit on conflict up to [`MAX_CAS_RETRIES`].
async fn append_fragment_kind(
    storage: &dyn Storage,
    ns: &str,
    payload: Bytes,
    records: u64,
    kind: FragmentKind,
) -> Result<u64, StorageError> {
    // Unique per write. Fixes F1: two racing appends get different keys, so
    // neither can overwrite the other's fragment object.
    let frag_id = uuid::Uuid::new_v4().to_string();

    // Write the fragment object first (durability), at its unique key. Because
    // the key is unique, this write is never contended — no CAS needed here.
    storage.put(&fragment_key(ns, &frag_id), payload).await?;

    // CAS-commit the manifest ref, assigning `seq` from the freshest manifest on
    // each attempt so concurrent commits produce a strict, gap-free ordering.
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        let (mut manifest, version) = read_manifest(storage, ns).await?;
        let seq = manifest.next_seq;
        manifest.next_seq += 1;
        manifest.fragments.push(FragmentRef {
            id: frag_id.clone(),
            seq,
            records,
            kind,
        });

        match commit_manifest(storage, ns, &manifest, &version).await {
            Ok(_) => return Ok(seq),
            Err(StorageError::VersionConflict { .. }) | Err(StorageError::AlreadyExists(_))
                if attempt < MAX_CAS_RETRIES =>
            {
                // Someone committed concurrently; re-read and retry with a fresh
                // seq. The fragment object is already durably written and its key
                // is unique, so retrying only re-does the (cheap) manifest CAS.
                continue;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Read every live (uncompacted) WAL fragment, in seq order, with its ref (so
/// callers see `kind`). Tolerates a fragment object that has already been GC'd
/// (skips it) so a *reader* holding a slightly stale manifest doesn't fail.
/// Do NOT use this for compaction — see `read_uncompacted_fragments_strict`.
pub async fn read_uncompacted_fragments(
    storage: &dyn Storage,
    ns: &str,
    manifest: &Manifest,
) -> Result<Vec<(FragmentRef, Bytes)>, StorageError> {
    let mut out = Vec::new();
    for fref in manifest.uncompacted() {
        match storage.get(&fragment_key(ns, &fref.id)).await {
            Ok(bytes) => out.push((fref.clone(), bytes)),
            Err(StorageError::NotFound(_)) => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(out)
}

/// Strict read for compaction: every uncompacted fragment the manifest lists
/// MUST be readable. A missing fragment here is corruption — folding a subset
/// and then advancing the watermark past the missing one would silently drop
/// that batch. So we fail loudly instead of skipping.
async fn read_uncompacted_fragments_strict(
    storage: &dyn Storage,
    ns: &str,
    manifest: &Manifest,
) -> Result<Vec<(FragmentRef, Bytes)>, StorageError> {
    let mut out = Vec::new();
    for fref in manifest.uncompacted() {
        match storage.get(&fragment_key(ns, &fref.id)).await {
            Ok(bytes) => out.push((fref.clone(), bytes)),
            Err(StorageError::NotFound(_)) => {
                return Err(StorageError::Io(format!(
                    "compaction aborted: WAL fragment seq={} id={} for '{}' is missing; \
                     advancing the watermark past it would lose data",
                    fref.seq, fref.id, ns
                )));
            }
            Err(e) => return Err(e),
        }
    }
    Ok(out)
}

/// Compact: fold all uncompacted WAL fragments into ONE new segment object via
/// `merge`, advance the watermark, and CAS-commit. `merge` receives the ordered
/// `(FragmentRef, payload)` list — so it can see each fragment's `kind` and
/// apply latest-wins + drop tombstoned records — and returns the segment bytes +
/// record count.
///
/// Fixes vs. the earlier version:
/// - **F3**: the segment gets a UNIQUE id (UUID), so two concurrent compactions
///   never overwrite each other's segment object.
/// - **F2/F4**: deferred-delete GC runs only AFTER the manifest CAS succeeds, so
///   a losing retry never deletes objects the committed manifest still needs.
/// - Strict read: a missing fragment aborts (no silent data loss).
pub async fn compact<F>(storage: &dyn Storage, ns: &str, merge: F) -> Result<bool, StorageError>
where
    // `Fn` (not `FnOnce`) because the CAS retry loop may call it more than once.
    F: Fn(&[(FragmentRef, Bytes)]) -> Result<(Bytes, u64), StorageError>,
{
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        let (mut manifest, version) = read_manifest(storage, ns).await?;

        // Snapshot the previous cycle's deferred deletes; we only physically
        // delete these AFTER our CAS commit succeeds (below).
        let carried_deletes = manifest.pending_deletes.clone();

        // Strict read: every listed uncompacted fragment must be present, so we
        // only ever fold a complete set (never a subset with a hole).
        let frags = read_uncompacted_fragments_strict(storage, ns, &manifest).await?;
        if frags.is_empty() {
            // Nothing to compact. Still commit if we have deletes to drain.
            if carried_deletes.is_empty() {
                return Ok(false);
            }
            manifest.pending_deletes.clear();
            match commit_manifest(storage, ns, &manifest, &version).await {
                Ok(_) => {
                    gc_keys(storage, &carried_deletes).await;
                    return Ok(false);
                }
                Err(StorageError::VersionConflict { .. }) if attempt < MAX_CAS_RETRIES => continue,
                Err(e) => return Err(e),
            }
        }

        let (segment_bytes, records) = merge(&frags)?;
        // Unique segment id (F3): concurrent compactions can't clobber.
        let segment_id = uuid::Uuid::new_v4().to_string();
        storage
            .put(&segment_key(ns, &segment_id), segment_bytes)
            .await?;

        // Safe: `frags` is the COMPLETE uncompacted set, so max seq covers
        // exactly what we merged.
        let new_watermark = frags.iter().map(|(fref, _)| fref.seq).max().unwrap_or(0);

        // The fragment objects we just compacted away — stage them for deletion
        // NEXT cycle (keyed by unique id), so any in-flight reader still on the
        // old manifest can read them for one more cycle.
        let newly_staged: Vec<String> = manifest
            .fragments
            .iter()
            .filter(|f| f.seq <= new_watermark)
            .map(|f| fragment_key(ns, &f.id))
            .collect();

        manifest.compaction_watermark = Some(new_watermark);
        manifest.fragments.retain(|f| f.seq > new_watermark);
        manifest.segments.push(SegmentRef {
            id: segment_id,
            records,
        });
        manifest.pending_deletes = newly_staged;

        match commit_manifest(storage, ns, &manifest, &version).await {
            Ok(_) => {
                // Commit succeeded: NOW physically delete the carried (previous
                // cycle's) objects. The just-compacted fragments stay one cycle
                // in pending_deletes so any in-flight reader on the old manifest
                // can still read them.
                gc_keys(storage, &carried_deletes).await;
                return Ok(true);
            }
            Err(StorageError::VersionConflict { .. }) if attempt < MAX_CAS_RETRIES => continue,
            Err(e) => return Err(e),
        }
    }
}

/// Physically delete a set of object keys, best-effort (a transient failure is
/// logged; the key stays referenced only if it was still in pending_deletes).
async fn gc_keys(storage: &dyn Storage, keys: &[String]) {
    for key in keys {
        if let Err(e) = storage.delete(key).await {
            tracing::warn!("deferred GC failed to delete {key}: {e}");
        }
    }
}

/// Read a compacted segment's bytes by id.
pub async fn read_segment(
    storage: &dyn Storage,
    ns: &str,
    id: &str,
) -> Result<Bytes, StorageError> {
    storage.get(&segment_key(ns, id)).await
}

/// Delete ALL objects for a namespace (manifest, WAL fragments, segments,
/// pending deletes). Used when a collection is deleted, so its data can't be
/// resurrected from object storage on a later cold start.
pub async fn delete_namespace(storage: &dyn Storage, ns: &str) -> Result<(), StorageError> {
    let prefix = format!("{ns}/");
    for obj in storage.list(&prefix).await? {
        // Guard against a prefix that is a superstring of another ns name.
        if obj.key == format!("{ns}/manifest")
            || obj.key.starts_with(&format!("{ns}/wal/"))
            || obj.key.starts_with(&format!("{ns}/segments/"))
            || obj.key.starts_with(&prefix)
        {
            let _ = storage.delete(&obj.key).await;
        }
    }
    Ok(())
}

/// List every namespace (collection) that has a manifest in storage. Used for
/// cloud-mode cold-start recovery: discover collections living in S3.
pub async fn list_namespaces(storage: &dyn Storage) -> Result<Vec<String>, StorageError> {
    // Delimiter listing: one call for the top-level prefixes (candidate
    // namespaces), then one existence check each for `{ns}/manifest`. Boot cost
    // is O(namespaces), not O(every WAL fragment/segment in the bucket).
    let mut names = Vec::new();
    for ns in storage.list_dirs("").await? {
        if storage.exists(&manifest_key(&ns)).await? {
            names.push(ns);
        }
    }
    Ok(names)
}

/// Partitioned compaction commit: APPEND a segment folding only the WAL tail
/// (fragments with seq <= `folded_through`) and advance the watermark. Old
/// segments stay; the folded fragments are staged for next-cycle GC and the
/// PRIOR cycle's staged keys are deleted now. On CAS conflict the just-written
/// orphan segment is removed. Bounded work: O(tail), never O(collection).
pub async fn append_segment(
    storage: &dyn Storage,
    ns: &str,
    expected: &Option<Version>,
    prior: &Manifest,
    segment_bytes: Bytes,
    records: u64,
    folded_through: u64,
) -> Result<(), StorageError> {
    let segment_id = uuid::Uuid::new_v4().to_string();
    let new_segment_key = segment_key(ns, &segment_id);
    storage.put_large(&new_segment_key, segment_bytes).await?;

    let folded: Vec<String> = prior
        .fragments
        .iter()
        .filter(|f| f.seq <= folded_through)
        .map(|f| fragment_key(ns, &f.id))
        .collect();
    let mut segments = prior.segments.clone();
    segments.push(SegmentRef {
        id: segment_id,
        records,
    });
    let new_manifest = Manifest {
        fragments: prior
            .fragments
            .iter()
            .filter(|f| f.seq > folded_through)
            .cloned()
            .collect(),
        segments,
        next_seq: prior.next_seq,
        compaction_watermark: Some(
            prior
                .compaction_watermark
                .map(|w| w.max(folded_through))
                .unwrap_or(folded_through),
        ),
        pending_deletes: folded,
    };
    match commit_manifest(storage, ns, &new_manifest, expected).await {
        Ok(_) => {
            gc_keys(storage, &prior.pending_deletes).await;
            Ok(())
        }
        Err(e) => {
            let _ = storage.delete(&new_segment_key).await;
            Err(e)
        }
    }
}

/// Full compaction: replace the ENTIRE manifest state (all segments + all
/// uncompacted fragments) with a single new segment containing `segment_bytes`
/// (the fully-materialized live set, deletes already applied). This is the
/// correct LSM compaction — folding old segments in too, so a tombstone can't
/// resurrect a chunk that lived in a prior segment.
///
/// The caller materializes the live set against a manifest read at `expected`
/// version; this function writes the new segment, then CAS-commits a manifest
/// that references only it.
///
/// Deferred-delete GC (fixes the "never GCs" leak): after a successful commit it
/// physically deletes the PRIOR cycle's `pending_deletes` (safe now — no live
/// reader references them anymore), while staging THIS cycle's folded objects
/// (old segments + fragments) into the new manifest's `pending_deletes` for the
/// next cycle (so an in-flight reader on the old manifest can still read them one
/// more cycle).
///
/// On CAS conflict it deletes the just-written orphan segment (nothing references
/// it) and returns `VersionConflict` so the caller can re-materialize and retry —
/// this prevents concurrent/lost compactions from leaking full segments.
pub async fn replace_with_single_segment(
    storage: &dyn Storage,
    ns: &str,
    expected: &Option<Version>,
    prior: &Manifest,
    segment_bytes: Bytes,
    records: u64,
) -> Result<(), StorageError> {
    let segment_id = uuid::Uuid::new_v4().to_string();
    let new_segment_key = segment_key(ns, &segment_id);
    storage.put_large(&new_segment_key, segment_bytes).await?;

    // Objects we're folding away THIS cycle (old segments + all fragments) — stage
    // for deletion NEXT cycle.
    let mut newly_staged: Vec<String> = prior
        .fragments
        .iter()
        .map(|f| fragment_key(ns, &f.id))
        .collect();
    newly_staged.extend(prior.segments.iter().map(|s| segment_key(ns, &s.id)));

    let new_manifest = Manifest {
        fragments: Vec::new(),
        segments: vec![SegmentRef {
            id: segment_id,
            records,
        }],
        next_seq: prior.next_seq, // preserve monotonic seq
        compaction_watermark: prior.next_seq.checked_sub(1),
        pending_deletes: newly_staged,
    };

    match commit_manifest(storage, ns, &new_manifest, expected).await {
        Ok(_) => {
            // Commit landed: physically GC the PRIOR cycle's staged objects now.
            gc_keys(storage, &prior.pending_deletes).await;
            Ok(())
        }
        Err(e) => {
            // Lost the CAS (or other error): the segment we just wrote is
            // referenced by no manifest → delete it so it doesn't leak.
            let _ = storage.delete(&new_segment_key).await;
            Err(e)
        }
    }
}

/// Convenience to share a storage handle into the async helpers.
pub type SharedStorage = Arc<dyn Storage>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::local::LocalDiskStorage;

    fn storage(name: &str) -> Arc<dyn Storage> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let mut root = std::env::temp_dir();
        // Unique per invocation so no two test runs share on-disk state.
        root.push(format!(
            "compass_lsm_test_{}_{}_{}",
            name,
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        Arc::new(LocalDiskStorage::new(root).unwrap())
    }

    /// A Storage wrapper that fails every `put` to a given key substring after
    /// `fail_after` successful puts — for failure-injection tests.
    struct FaultyStorage {
        inner: Arc<dyn Storage>,
        fail_substr: String,
    }
    #[async_trait::async_trait]
    impl Storage for FaultyStorage {
        async fn get(&self, key: &str) -> Result<Bytes, StorageError> {
            self.inner.get(key).await
        }
        async fn get_range(
            &self,
            key: &str,
            range: std::ops::Range<u64>,
        ) -> Result<Bytes, StorageError> {
            self.inner.get_range(key, range).await
        }
        async fn get_versioned(&self, key: &str) -> Result<(Bytes, Version), StorageError> {
            self.inner.get_versioned(key).await
        }
        async fn put(&self, key: &str, bytes: Bytes) -> Result<Version, StorageError> {
            if key.contains(&self.fail_substr) {
                return Err(StorageError::Io("injected put failure".into()));
            }
            self.inner.put(key, bytes).await
        }
        async fn put_if_match(
            &self,
            key: &str,
            bytes: Bytes,
            expected: &Version,
        ) -> Result<Version, StorageError> {
            self.inner.put_if_match(key, bytes, expected).await
        }
        async fn put_if_not_exists(
            &self,
            key: &str,
            bytes: Bytes,
        ) -> Result<Version, StorageError> {
            self.inner.put_if_not_exists(key, bytes).await
        }
        async fn delete(&self, key: &str) -> Result<(), StorageError> {
            self.inner.delete(key).await
        }
        async fn list(
            &self,
            prefix: &str,
        ) -> Result<Vec<crate::storage::ObjectMeta>, StorageError> {
            self.inner.list(prefix).await
        }
        fn backend_name(&self) -> &'static str {
            "faulty"
        }
    }

    // Failure injection: if the fragment PUT fails, append returns Err and the
    // manifest is NOT mutated (no orphan ref, no partial commit).
    #[tokio::test]
    async fn append_fails_cleanly_when_fragment_put_fails() {
        let inner = storage("faulty_put");
        let faulty = FaultyStorage {
            inner: inner.clone(),
            fail_substr: "/wal/".to_string(),
        };
        let r = append_fragment(&faulty, "ns", Bytes::from_static(b"x"), 1).await;
        assert!(r.is_err(), "append must fail when the fragment put fails");

        // The manifest must not reference a fragment that was never written.
        let (m, _) = read_manifest(inner.as_ref(), "ns").await.unwrap();
        assert!(
            m.fragments.is_empty(),
            "no manifest ref for a failed fragment write"
        );
    }

    #[tokio::test]
    async fn append_and_read_fragments() {
        let s = storage("append");
        let seq0 = append_fragment(s.as_ref(), "ns", Bytes::from_static(b"a"), 1)
            .await
            .unwrap();
        let seq1 = append_fragment(s.as_ref(), "ns", Bytes::from_static(b"b"), 1)
            .await
            .unwrap();
        assert_eq!(seq0, 0);
        assert_eq!(seq1, 1);

        let (manifest, _) = read_manifest(s.as_ref(), "ns").await.unwrap();
        assert_eq!(manifest.fragments.len(), 2);
        assert_eq!(manifest.next_seq, 2);

        let frags = read_uncompacted_fragments(s.as_ref(), "ns", &manifest)
            .await
            .unwrap();
        assert_eq!(frags.len(), 2);
        assert_eq!(&frags[0].1[..], b"a");
        assert_eq!(&frags[1].1[..], b"b");
    }

    #[tokio::test]
    async fn manifest_persists_and_defaults() {
        let s = storage("persist");
        // No manifest yet -> default, version None.
        let (m, v) = read_manifest(s.as_ref(), "ns").await.unwrap();
        assert_eq!(m, Manifest::default());
        assert!(v.is_none());

        append_fragment(s.as_ref(), "ns", Bytes::from_static(b"x"), 1)
            .await
            .unwrap();
        let (m2, v2) = read_manifest(s.as_ref(), "ns").await.unwrap();
        assert_eq!(m2.fragments.len(), 1);
        assert!(v2.is_some());
    }

    #[tokio::test]
    async fn compaction_folds_fragments_into_segment() {
        let s = storage("compact");
        for i in 0..3u8 {
            append_fragment(s.as_ref(), "ns", Bytes::from(vec![b'0' + i]), 1)
                .await
                .unwrap();
        }
        // Merge concatenates fragment payloads.
        let did = compact(s.as_ref(), "ns", |frags| {
            let mut out = Vec::new();
            for (_, b) in frags {
                out.extend_from_slice(b);
            }
            let records = frags.len() as u64;
            Ok((Bytes::from(out), records))
        })
        .await
        .unwrap();
        assert!(did);

        let (m, _) = read_manifest(s.as_ref(), "ns").await.unwrap();
        assert_eq!(m.segments.len(), 1);
        assert_eq!(m.segments[0].records, 3);
        assert_eq!(m.compaction_watermark, Some(2));
        assert!(m.uncompacted().next().is_none());

        // Segment id is server-assigned (UUID); read it back from the manifest.
        let seg_id = m.segments[0].id.clone();
        let seg = read_segment(s.as_ref(), "ns", &seg_id).await.unwrap();
        assert_eq!(&seg[..], b"012");

        // A subsequent compaction with nothing new is a no-op.
        let did2 = compact(s.as_ref(), "ns", |frags| {
            assert!(frags.is_empty());
            Ok((Bytes::new(), 0))
        })
        .await
        .unwrap();
        assert!(!did2);
    }

    #[tokio::test]
    async fn new_writes_after_compaction_are_uncompacted() {
        let s = storage("after_compact");
        append_fragment(s.as_ref(), "ns", Bytes::from_static(b"old"), 1)
            .await
            .unwrap();
        compact(s.as_ref(), "ns", |frags| {
            Ok((Bytes::from_static(b"seg"), frags.len() as u64))
        })
        .await
        .unwrap();

        let seq = append_fragment(s.as_ref(), "ns", Bytes::from_static(b"new"), 1)
            .await
            .unwrap();
        assert_eq!(seq, 1, "seq continues past the compacted fragment");
        let (m, _) = read_manifest(s.as_ref(), "ns").await.unwrap();
        let live: Vec<_> = m.uncompacted().collect();
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].seq, 1);
    }

    #[tokio::test]
    async fn tombstone_fragment_is_recorded() {
        let s = storage("tombstone");
        append_fragment(s.as_ref(), "ns", Bytes::from_static(b"data"), 2)
            .await
            .unwrap();
        let seq = append_tombstone(s.as_ref(), "ns", &[7, 8, 9])
            .await
            .unwrap();
        assert_eq!(seq, 1);

        let (m, _) = read_manifest(s.as_ref(), "ns").await.unwrap();
        assert_eq!(m.fragments.len(), 2);
        assert_eq!(m.fragments[0].kind, FragmentKind::Data);
        assert_eq!(m.fragments[1].kind, FragmentKind::Tombstone);

        // The tombstone fragment decodes back to the deleted ids.
        let frags = read_uncompacted_fragments(s.as_ref(), "ns", &m)
            .await
            .unwrap();
        let ids: Vec<u64> = serde_json::from_slice(&frags[1].1).unwrap();
        assert_eq!(ids, vec![7, 8, 9]);
    }

    // A missing fragment during compaction must ABORT (not silently skip and
    // advance the watermark past lost data). This is the data-loss guard.
    #[tokio::test]
    async fn compaction_aborts_on_missing_fragment() {
        let s = storage("missing_frag");
        append_fragment(s.as_ref(), "ns", Bytes::from_static(b"a"), 1)
            .await
            .unwrap();
        append_fragment(s.as_ref(), "ns", Bytes::from_static(b"b"), 1)
            .await
            .unwrap();

        // Simulate corruption: delete the first fragment's object out from under
        // the manifest (keyed by its unique id, read from the manifest).
        let (m0, _) = read_manifest(s.as_ref(), "ns").await.unwrap();
        let victim = format!("ns/wal/{}.frag", m0.fragments[0].id);
        s.delete(&victim).await.unwrap();

        let result = compact(s.as_ref(), "ns", |frags| {
            Ok((Bytes::from_static(b"seg"), frags.len() as u64))
        })
        .await;
        assert!(
            result.is_err(),
            "compaction must abort on a missing fragment"
        );

        // Manifest is untouched: watermark did NOT advance, fragment 1 still live.
        let (m, _) = read_manifest(s.as_ref(), "ns").await.unwrap();
        assert_eq!(m.compaction_watermark, None);
        assert_eq!(m.fragments.len(), 2);
    }

    // F1 regression: N concurrent appends must all survive — distinct fragment
    // ids, distinct gap-free seqs, and every payload readable. The old scheme
    // keyed fragments by `next_seq`, so racing writers overwrote each other's
    // fragment object. UUID keys + CAS-assigned seq fix it.
    #[tokio::test]
    async fn concurrent_appends_do_not_lose_writes() {
        let s = storage("concurrent");
        let n = 16u64;

        let mut handles = Vec::new();
        for i in 0..n {
            let s2 = s.clone();
            handles.push(tokio::spawn(async move {
                // Distinct payload per writer so a clobber would be detectable.
                let payload = Bytes::from(format!("writer-{i}").into_bytes());
                append_fragment(s2.as_ref(), "ns", payload, 1).await
            }));
        }
        for h in handles {
            h.await.unwrap().unwrap();
        }

        let (m, _) = read_manifest(s.as_ref(), "ns").await.unwrap();
        assert_eq!(m.fragments.len() as u64, n, "all N appends recorded");
        assert_eq!(m.next_seq, n, "seqs are gap-free 0..N");

        // Seqs are exactly 0..N (strict ordering, no dupes).
        let mut seqs: Vec<u64> = m.fragments.iter().map(|f| f.seq).collect();
        seqs.sort_unstable();
        assert_eq!(seqs, (0..n).collect::<Vec<_>>());

        // Fragment ids are all distinct (no key collision).
        let mut ids: Vec<String> = m.fragments.iter().map(|f| f.id.clone()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len() as u64, n, "all fragment ids distinct");

        // Every distinct writer payload is present exactly once (no lost write).
        let frags = read_uncompacted_fragments(s.as_ref(), "ns", &m)
            .await
            .unwrap();
        let mut payloads: Vec<String> = frags
            .iter()
            .map(|(_, b)| String::from_utf8(b.to_vec()).unwrap())
            .collect();
        payloads.sort();
        let mut expected: Vec<String> = (0..n).map(|i| format!("writer-{i}")).collect();
        expected.sort();
        assert_eq!(payloads, expected, "no writer's payload was clobbered");
    }

    // Relation fragment kinds round-trip through the WAL with the right kind tag
    // and payload (the lsm-level analog of the tombstone test).
    #[tokio::test]
    async fn relation_fragments_recorded_with_kinds() {
        let s = storage("relfrag");
        append_relation_upsert(s.as_ref(), "ns", Bytes::from_static(b"[{\"fake\":1}]"), 1)
            .await
            .unwrap();
        append_relation_delete(s.as_ref(), "ns", &["r-1".into(), "r-2".into()])
            .await
            .unwrap();

        let (m, _) = read_manifest(s.as_ref(), "ns").await.unwrap();
        assert_eq!(m.fragments.len(), 2);
        assert_eq!(m.fragments[0].kind, FragmentKind::RelationUpsert);
        assert_eq!(m.fragments[1].kind, FragmentKind::RelationDelete);
        assert_eq!(m.fragments[1].records, 2);

        let frags = read_uncompacted_fragments(s.as_ref(), "ns", &m)
            .await
            .unwrap();
        let ids: Vec<String> = serde_json::from_slice(&frags[1].1).unwrap();
        assert_eq!(ids, vec!["r-1".to_string(), "r-2".to_string()]);
    }

    // Failure injection: if the compacted SEGMENT put fails, the manifest is
    // untouched (still references the original fragments) and nothing is lost.
    #[tokio::test]
    async fn compaction_segment_put_failure_leaves_manifest_intact() {
        let inner = storage("segfail");
        append_fragment(inner.as_ref(), "ns", Bytes::from_static(b"data"), 1)
            .await
            .unwrap();
        let (before, version) = read_manifest(inner.as_ref(), "ns").await.unwrap();

        let faulty = FaultyStorage {
            inner: inner.clone(),
            fail_substr: "/segments/".to_string(),
        };
        let r = replace_with_single_segment(
            &faulty,
            "ns",
            &version,
            &before,
            Bytes::from_static(b"seg"),
            1,
        )
        .await;
        assert!(
            r.is_err(),
            "compaction must fail when the segment put fails"
        );

        let (after, _) = read_manifest(inner.as_ref(), "ns").await.unwrap();
        assert_eq!(after.fragments.len(), 1, "manifest untouched on failure");
        assert!(after.segments.is_empty());
    }

    // F3 regression: a compaction that LOSES the manifest CAS must delete the
    // segment object it already wrote (no unreachable orphan).
    #[tokio::test]
    async fn cas_losing_compaction_cleans_up_its_orphan_segment() {
        let s = storage("casloss");
        append_fragment(s.as_ref(), "ns", Bytes::from_static(b"data"), 1)
            .await
            .unwrap();
        let (manifest, version) = read_manifest(s.as_ref(), "ns").await.unwrap();

        // Simulate a concurrent append landing AFTER our manifest read — the
        // upcoming CAS (against the stale `version`) must conflict.
        append_fragment(s.as_ref(), "ns", Bytes::from_static(b"racer"), 1)
            .await
            .unwrap();

        let r = replace_with_single_segment(
            s.as_ref(),
            "ns",
            &version,
            &manifest,
            Bytes::from_static(b"seg"),
            1,
        )
        .await;
        assert!(
            matches!(r, Err(StorageError::VersionConflict { .. })),
            "stale-version compaction must lose the CAS, got {r:?}"
        );

        // The losing compaction's segment object must NOT linger: no objects
        // under ns/segments/ at all (the winner never wrote one).
        let leaked = s.list("ns/segments/").await.unwrap();
        assert!(
            leaked.is_empty(),
            "CAS-losing compaction leaked segment objects: {:?}",
            leaked.iter().map(|o| &o.key).collect::<Vec<_>>()
        );
    }
}
