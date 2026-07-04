// collections/cloud_ingest_tests.rs — extracted test module (was inline in mod.rs).
// Child module of `collections`: `super::*` sees the parent's private items.

//! Verifies that in object-storage (cloud) mode, ingest mirrors the batch
//! into the LSM as a WAL fragment + CAS-committed manifest — the S3-native
//! path. Uses the in-memory object_store backend, which
//! exercises the identical `Storage`/`ObjectStoreBackend` code an S3 bucket
//! would, without needing real credentials.

use super::*;
use crate::embed::EmbedState;
use crate::storage::object_store_backend::ObjectStoreBackend;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

fn unique_data_dir() -> std::path::PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "compass-cloud-ingest-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ))
}

fn embed_state() -> EmbedState {
    // No models needed: chunks carry precomputed embeddings.
    EmbedState {
        bge: None,
        distilled: None,
    }
}

fn ingest_chunk(idx: u32) -> IngestChunk {
    let mut metadata = HashMap::new();
    metadata.insert(
        "org_id".to_string(),
        MetadataValue::String("acme".to_string()),
    );
    let mut embeddings = HashMap::new();
    embeddings.insert("default".to_string(), vec![0.1, 0.2, 0.3, 0.4]);
    IngestChunk {
        client_id: None,
        file_id: format!("f{idx}"),
        chunk_index: 0,
        page: None,
        text: format!("chunk-{idx}"),
        metadata,
        doc_type: "chunk".to_string(),
        parent_id: None,
        parent_ref: None,
        group_id: None,
        embeddings,
        embedding: None,
    }
}

#[tokio::test]
async fn ingest_writes_wal_fragment_and_manifest_to_object_storage() {
    let data_dir = unique_data_dir();
    std::fs::create_dir_all(&data_dir).unwrap();
    let embed = embed_state();

    // In-memory object storage backend (same code path as s3://).
    let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        std::sync::Arc::new(object_store::memory::InMemory::new()),
        "object-store:memory",
    ));
    let manager = CollectionManager::new_with_storage(&data_dir, storage.clone())
        .await
        .unwrap();
    manager
        .create_collection("cloudcoll", None, Some(4), None)
        .await
        .unwrap();

    // Ingest two batches.
    manager
        .ingest("cloudcoll", vec![ingest_chunk(0), ingest_chunk(1)], &embed)
        .await
        .unwrap();
    manager
        .ingest("cloudcoll", vec![ingest_chunk(2)], &embed)
        .await
        .unwrap();

    // The manifest exists and records two WAL fragments.
    let (manifest, version) = crate::storage::lsm::read_manifest(storage.as_ref(), "cloudcoll")
        .await
        .unwrap();
    assert!(version.is_some(), "manifest must exist in object storage");
    assert_eq!(manifest.fragments.len(), 2, "one fragment per ingest batch");
    assert_eq!(manifest.next_seq, 2);

    // The WAL fragment objects exist and decode back to the ingested chunks.
    let frags =
        crate::storage::lsm::read_uncompacted_fragments(storage.as_ref(), "cloudcoll", &manifest)
            .await
            .unwrap();
    assert_eq!(frags.len(), 2);

    let batch0: Vec<DocumentChunk> = serde_json::from_slice(&frags[0].1).unwrap();
    assert_eq!(batch0.len(), 2);
    assert_eq!(batch0[0].text, "chunk-0");
    let batch1: Vec<DocumentChunk> = serde_json::from_slice(&frags[1].1).unwrap();
    assert_eq!(batch1.len(), 1);
    assert_eq!(batch1[0].text, "chunk-2");

    // Total records across fragments == total chunks ingested.
    let total: u64 = manifest.fragments.iter().map(|f| f.records).sum();
    assert_eq!(total, 3);

    let _ = std::fs::remove_dir_all(&data_dir);
}

#[tokio::test]
async fn local_mode_writes_no_wal() {
    // Sanity: a local-disk manager must NOT create any WAL/manifest objects.
    let data_dir = unique_data_dir();
    std::fs::create_dir_all(&data_dir).unwrap();
    let embed = embed_state();
    let manager = CollectionManager::new(&data_dir).await.unwrap();
    manager
        .create_collection("localcoll", None, Some(4), None)
        .await
        .unwrap();
    manager
        .ingest("localcoll", vec![ingest_chunk(0)], &embed)
        .await
        .unwrap();

    // No manifest object should exist under the collection prefix.
    let manifest_path = data_dir.join("localcoll").join("manifest");
    assert!(
        !manifest_path.exists(),
        "local mode must not write an LSM manifest"
    );
    // Nor an id-block allocator: local mode allocates from next_id.
    assert!(
        !data_dir.join("localcoll").join("id-alloc").exists(),
        "local mode must not seed the id-block allocator"
    );
    // And ids stay dense from 0 (block allocation would start at 0 too,
    // but a second ingest would jump; assert both batches are contiguous).
    manager
        .ingest("localcoll", vec![ingest_chunk(1)], &embed)
        .await
        .unwrap();
    let (_, mut ids) = manager.get_all_chunk_data("localcoll").await.unwrap();
    ids.sort_unstable();
    assert_eq!(ids, vec![0, 1], "local ids must be dense next_id values");
    let _ = std::fs::remove_dir_all(&data_dir);
}

// A stray COMPASS_ROLE=writer on a local-disk deployment must be
// neutralized: cloud_mode is false, so the constructor forces Full and
// the node keeps serving reads and creating collections normally.
#[tokio::test]
async fn writer_role_is_neutralized_in_local_mode() {
    let data_dir = unique_data_dir();
    std::fs::create_dir_all(&data_dir).unwrap();
    let embed = embed_state();
    let storage: Arc<dyn Storage> =
        Arc::new(crate::storage::local::LocalDiskStorage::new(&data_dir).unwrap());
    let manager = CollectionManager::new_with_storage_opts(
        &data_dir,
        storage,
        NodeRole::Writer,
        false,
        usize::MAX,
        0,
    )
    .await
    .unwrap();
    manager
        .create_collection("localwriter", None, Some(4), None)
        .await
        .expect("local node must create collections despite COMPASS_ROLE=writer");
    manager
        .ingest("localwriter", vec![ingest_chunk(0)], &embed)
        .await
        .unwrap();
    let (_, ids) = manager
        .get_all_chunk_data("localwriter")
        .await
        .expect("local node must serve reads despite COMPASS_ROLE=writer");
    assert_eq!(ids.len(), 1);
    let _ = std::fs::remove_dir_all(&data_dir);
}

#[tokio::test]
async fn delete_writes_tombstone_wal_fragment() {
    use crate::storage::lsm::{read_manifest, read_uncompacted_fragments, FragmentKind};

    let data_dir = unique_data_dir();
    std::fs::create_dir_all(&data_dir).unwrap();
    let embed = embed_state();
    let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        std::sync::Arc::new(object_store::memory::InMemory::new()),
        "object-store:memory",
    ));
    let manager = CollectionManager::new_with_storage(&data_dir, storage.clone())
        .await
        .unwrap();
    manager
        .create_collection("delcloud", None, Some(4), None)
        .await
        .unwrap();
    manager
        .ingest(
            "delcloud",
            vec![ingest_chunk(0), ingest_chunk(1), ingest_chunk(2)],
            &embed,
        )
        .await
        .unwrap();

    // Delete chunk 1 -> a tombstone WAL fragment lands in object storage.
    let (n, _) = manager.delete_chunks("delcloud", &[1]).await.unwrap();
    assert_eq!(n, 1);

    let (manifest, _) = read_manifest(storage.as_ref(), "delcloud").await.unwrap();
    // seq 0 = data fragment (the ingest), seq 1 = tombstone fragment.
    assert_eq!(manifest.fragments.len(), 2);
    assert_eq!(manifest.fragments[0].kind, FragmentKind::Data);
    assert_eq!(manifest.fragments[1].kind, FragmentKind::Tombstone);

    // The tombstone fragment decodes to the deleted id [1].
    let frags = read_uncompacted_fragments(storage.as_ref(), "delcloud", &manifest)
        .await
        .unwrap();
    let deleted_ids: Vec<u64> = serde_json::from_slice(&frags[1].1).unwrap();
    assert_eq!(deleted_ids, vec![1]);

    let _ = std::fs::remove_dir_all(&data_dir);
}

// THE structural fix: a cloud collection must survive a restart on a FRESH
// local disk by rebuilding from S3. Ingest, delete one, then drop the manager
// AND wipe the local data dir, then reload from the SAME object store — the
// data (minus the deleted chunk) must come back.
#[tokio::test]
async fn cloud_restart_rehydrates_from_object_storage() {
    let embed = embed_state();
    // Shared object store persists across the "restart".
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());

    let data_dir_a = unique_data_dir();
    std::fs::create_dir_all(&data_dir_a).unwrap();
    {
        let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m = CollectionManager::new_with_storage(&data_dir_a, storage)
            .await
            .unwrap();
        m.create_collection("survive", None, Some(4), None)
            .await
            .unwrap();
        m.ingest(
            "survive",
            vec![ingest_chunk(0), ingest_chunk(1), ingest_chunk(2)],
            &embed,
        )
        .await
        .unwrap();
        m.delete_chunks("survive", &[1]).await.unwrap();
    }
    // Simulate node loss: wipe the local disk entirely.
    std::fs::remove_dir_all(&data_dir_a).unwrap();

    // Restart on a BRAND-NEW empty local dir, same object store.
    let data_dir_b = unique_data_dir();
    std::fs::create_dir_all(&data_dir_b).unwrap();
    let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let m2 = CollectionManager::new_with_storage(&data_dir_b, storage)
        .await
        .unwrap();

    // The collection is back, recovered from S3.
    let info = m2.get_collection("survive").await;
    assert!(info.is_some(), "collection must be recovered from S3");

    // Search finds the surviving chunks (0 and 2), not the deleted one (1).
    let req = SearchRequest {
        query: "chunk".to_string(),
        mode: "semantic".to_string(),
        vector_space: None,
        top_k: 10,
        query_vector: Some(vec![0.1, 0.2, 0.3, 0.4]),
        filters: HashMap::new(),
        score_weights: None,
        recency: None,
        recency_preset: None,
        recency_field: None,
        boosts: Vec::new(),
        relationship_boost: None,
        explain: false,
        include_relations: false,
        relation_types: None,
        relation_direction: RelationDirection::Outgoing,
        min_seq: None,
    };
    let (hits, _, _, _) = m2.search("survive", &req, &embed).await.unwrap();
    let ids: std::collections::HashSet<u64> = hits.iter().map(|(c, _, _, _, _)| c.id).collect();
    assert!(ids.contains(&0), "chunk 0 recovered");
    assert!(ids.contains(&2), "chunk 2 recovered");
    assert!(!ids.contains(&1), "deleted chunk 1 must NOT reappear");

    let _ = std::fs::remove_dir_all(&data_dir_b);
}

// Compaction folds segments+fragments into one segment, dropping tombstoned
// records so they can never resurrect.
#[tokio::test]
async fn compaction_reclaims_tombstoned_data() {
    use crate::storage::lsm::read_manifest;
    let embed = embed_state();
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    let data_dir = unique_data_dir();
    std::fs::create_dir_all(&data_dir).unwrap();
    let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let m = CollectionManager::new_with_storage(&data_dir, storage.clone())
        .await
        .unwrap();
    m.create_collection("comp", None, Some(4), None)
        .await
        .unwrap();
    m.ingest(
        "comp",
        vec![ingest_chunk(0), ingest_chunk(1), ingest_chunk(2)],
        &embed,
    )
    .await
    .unwrap();
    m.delete_chunks("comp", &[1]).await.unwrap();

    // Before: manifest has data + tombstone fragments, no segment.
    let (before, _) = read_manifest(storage.as_ref(), "comp").await.unwrap();
    assert!(before.segments.is_empty());
    assert_eq!(before.fragments.len(), 2);

    // Compact.
    let live = m.compact_collection("comp").await.unwrap();
    assert_eq!(live, 2, "2 live records (0 and 2) after dropping deleted 1");

    // After: the WAL tail folded into an appended segment, no live fragments.
    let (after, _) = read_manifest(storage.as_ref(), "comp").await.unwrap();
    assert_eq!(after.segments.len(), 1);
    assert!(after.uncompacted().count() == 0);

    // Durable truth via materialize (exercises the v2 binary codec):
    // live chunks 0 and 2 survive, deleted 1 is gone.
    let mat = cloud::materialize(storage.as_ref(), "comp", &after)
        .await
        .unwrap();
    let ids: std::collections::HashSet<u64> = mat.chunks.keys().copied().collect();
    assert!(ids.contains(&0) && ids.contains(&2));
    assert!(
        !ids.contains(&1),
        "compaction must drop the tombstoned chunk"
    );

    let _ = std::fs::remove_dir_all(&data_dir);
}

// #1 regression: typed RELATIONS must survive a cold restart from S3 (the bug
// where relation_store was local-redb-only and vanished on rebuild). Create
// relations, wipe the local disk, restart on a fresh dir, relations return.
#[tokio::test]
async fn cloud_restart_recovers_relations() {
    let embed = embed_state();
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());

    let data_dir_a = unique_data_dir();
    std::fs::create_dir_all(&data_dir_a).unwrap();
    {
        let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m = CollectionManager::new_with_storage(&data_dir_a, storage)
            .await
            .unwrap();
        m.create_collection("relsurv", None, Some(4), None)
            .await
            .unwrap();
        m.ingest(
            "relsurv",
            vec![ingest_chunk(0), ingest_chunk(1), ingest_chunk(2)],
            &embed,
        )
        .await
        .unwrap();
        // Create two relations, then delete one — only the survivor should
        // come back.
        let created = m
            .create_relations(
                "relsurv",
                vec![
                    CreateRelation {
                        source_chunk_id: 0,
                        target_chunk_id: 1,
                        target_document_id: None,
                        relation_type: "cites".into(),
                        metadata: HashMap::new(),
                    },
                    CreateRelation {
                        source_chunk_id: 0,
                        target_chunk_id: 2,
                        target_document_id: None,
                        relation_type: "supersedes".into(),
                        metadata: HashMap::new(),
                    },
                ],
            )
            .await
            .unwrap();
        m.delete_relation("relsurv", &created[1].relation_id)
            .await
            .unwrap();
    }
    // Node loss: wipe local disk.
    std::fs::remove_dir_all(&data_dir_a).unwrap();

    // Restart on a fresh local dir, same object store.
    let data_dir_b = unique_data_dir();
    std::fs::create_dir_all(&data_dir_b).unwrap();
    let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let m2 = CollectionManager::new_with_storage(&data_dir_b, storage)
        .await
        .unwrap();

    // The surviving relation (0 --cites--> 1) must be recovered from S3;
    // the deleted one (0 --supersedes--> 2) must NOT reappear.
    let out = m2
        .get_chunk_relations("relsurv", 0, RelationDirection::Outgoing, None)
        .await
        .unwrap();
    assert_eq!(out.len(), 1, "exactly one relation should survive restart");
    assert_eq!(out[0].relation_type, "cites");
    assert_eq!(out[0].target_chunk_id, 1);

    let _ = std::fs::remove_dir_all(&data_dir_b);
}

// #3: auto-compaction. Ingest enough batches to cross the fragment threshold;
// the background trigger should fold them into a segment. We poll briefly for
// the detached task to run, then assert the WAL is bounded.
#[tokio::test]
async fn auto_compaction_bounds_the_wal() {
    let embed = embed_state();
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    let data_dir = unique_data_dir();
    std::fs::create_dir_all(&data_dir).unwrap();
    let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let m = CollectionManager::new_with_storage(&data_dir, storage.clone())
        .await
        .unwrap();
    m.create_collection("auto", None, Some(4), None)
        .await
        .unwrap();

    // One chunk per ingest = one fragment per ingest. Cross the threshold.
    let batches = AUTO_COMPACT_FRAGMENT_THRESHOLD + 2;
    for i in 0..batches {
        m.ingest("auto", vec![ingest_chunk(i as u32)], &embed)
            .await
            .unwrap();
    }

    // Poll up to ~3s for the detached auto-compaction to land a segment and
    // shrink the uncompacted fragment set.
    let mut compacted = false;
    for _ in 0..30 {
        let (man, _) = crate::storage::lsm::read_manifest(storage.as_ref(), "auto")
            .await
            .unwrap();
        if !man.segments.is_empty() && man.uncompacted().count() < AUTO_COMPACT_FRAGMENT_THRESHOLD {
            compacted = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert!(
        compacted,
        "auto-compaction should have folded the WAL into a segment"
    );

    // All data still present after auto-compaction (via materialize).
    let (man, _) = crate::storage::lsm::read_manifest(storage.as_ref(), "auto")
        .await
        .unwrap();
    let mat = cloud::materialize(storage.as_ref(), "auto", &man)
        .await
        .unwrap();
    assert_eq!(mat.chunks.len(), batches, "no data lost in auto-compaction");

    let _ = std::fs::remove_dir_all(&data_dir);
}

// Negative: auto-compaction must NOT fire below the fragment threshold (a
// regression dropping the threshold to ~0 would compact on every ingest).
#[tokio::test]
async fn auto_compaction_does_not_fire_below_threshold() {
    let embed = embed_state();
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    let data_dir = unique_data_dir();
    std::fs::create_dir_all(&data_dir).unwrap();
    let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let m = CollectionManager::new_with_storage(&data_dir, storage.clone())
        .await
        .unwrap();
    m.create_collection("below", None, Some(4), None)
        .await
        .unwrap();

    // Well under the threshold: a handful of single-chunk ingests.
    for i in 0..5u32 {
        m.ingest("below", vec![ingest_chunk(i)], &embed)
            .await
            .unwrap();
    }
    // Give any (wrongly) spawned compaction ample time to land a segment.
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let (man, _) = crate::storage::lsm::read_manifest(storage.as_ref(), "below")
        .await
        .unwrap();
    assert!(
        man.segments.is_empty(),
        "auto-compaction must not fire below the threshold"
    );
    assert_eq!(man.fragments.len(), 5, "all fragments still in the WAL");

    let _ = std::fs::remove_dir_all(&data_dir);
}

// F1 regression: compaction must physically GC old objects (deferred one
// cycle), not leak them forever. Ingest, compact twice, assert the first
// segment's object is deleted and the object count stays bounded.
#[tokio::test]
async fn compaction_gcs_old_objects() {
    let embed = embed_state();
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    let data_dir = unique_data_dir();
    std::fs::create_dir_all(&data_dir).unwrap();
    let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let m = CollectionManager::new_with_storage(&data_dir, storage.clone())
        .await
        .unwrap();
    m.create_collection("gc", None, Some(4), None)
        .await
        .unwrap();
    m.ingest("gc", vec![ingest_chunk(0), ingest_chunk(1)], &embed)
        .await
        .unwrap();

    // First compaction → segment S1, stages the 1 fragment for next-cycle GC.
    m.compact_collection("gc").await.unwrap();
    let (man1, _) = crate::storage::lsm::read_manifest(storage.as_ref(), "gc")
        .await
        .unwrap();
    let seg1_id = man1.segments[0].id.clone();
    // The old WAL fragment object is staged (still present this cycle).
    assert_eq!(man1.pending_deletes.len(), 1);

    // Drive enough tail-fold cycles to cross the merge threshold (8
    // segments) so a full merge runs; the merge (plus deferred GC) must
    // physically delete S1 — the key point is it's GC'd, not leaked.
    for i in 2..14u32 {
        m.ingest("gc", vec![ingest_chunk(i)], &embed).await.unwrap();
        m.compact_collection("gc").await.unwrap();
    }
    // Fold and merge are deliberately SEPARATE invocations (the merge
    // never runs in the same call as a fold, preserving the one-cycle GC
    // grace) — drive bare compactions so the merge and its deferred GC run.
    m.compact_collection("gc").await.unwrap(); // merge (no tail)
    m.ingest("gc", vec![ingest_chunk(99)], &embed)
        .await
        .unwrap();
    m.compact_collection("gc").await.unwrap(); // fold
    m.compact_collection("gc").await.unwrap(); // merge + GC prior staged
    m.ingest("gc", vec![ingest_chunk(100)], &embed)
        .await
        .unwrap();
    m.compact_collection("gc").await.unwrap(); // fold
    m.compact_collection("gc").await.unwrap(); // merge + GC
    let (man2, _) = crate::storage::lsm::read_manifest(storage.as_ref(), "gc")
        .await
        .unwrap();

    // Segment S1 must be physically deleted (GC'd after being folded away).
    let s1_key = format!("gc/segments/{seg1_id}");
    assert!(
        !storage.exists(&s1_key).await.unwrap(),
        "old segment must be GC'd, not leaked"
    );
    // Object count stays BOUNDED across many compaction cycles — proving no
    // unbounded leak (the F1 bug would grow this without limit).
    let all = storage.list("gc/").await.unwrap();
    // Fixed per-namespace objects (manifest, collection.json, id-alloc)
    // plus up to MERGE_SEGMENTS(8) tail segments and this-cycle staged
    // objects — bounded, never growing with cycle count.
    assert!(
        all.len() <= 16,
        "object count must stay bounded across cycles, got {}",
        all.len()
    );
    // Data intact.
    let mat = cloud::materialize(storage.as_ref(), "gc", &man2)
        .await
        .unwrap();
    assert_eq!(mat.chunks.len(), 16);

    let _ = std::fs::remove_dir_all(&data_dir);
}

// Relations must survive COMPACTION-then-restart (segment-relations path),
// not just the fragment-replay path.
#[tokio::test]
async fn relations_survive_compaction_then_restart() {
    let embed = embed_state();
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());

    let data_dir_a = unique_data_dir();
    std::fs::create_dir_all(&data_dir_a).unwrap();
    {
        let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m = CollectionManager::new_with_storage(&data_dir_a, storage)
            .await
            .unwrap();
        m.create_collection("rc", None, Some(4), None)
            .await
            .unwrap();
        m.ingest("rc", vec![ingest_chunk(0), ingest_chunk(1)], &embed)
            .await
            .unwrap();
        m.create_relations(
            "rc",
            vec![CreateRelation {
                source_chunk_id: 0,
                target_chunk_id: 1,
                target_document_id: None,
                relation_type: "cites".into(),
                metadata: HashMap::new(),
            }],
        )
        .await
        .unwrap();
        // Compact so the relation lives in the SEGMENT, not a WAL fragment.
        m.compact_collection("rc").await.unwrap();
    }
    std::fs::remove_dir_all(&data_dir_a).unwrap();

    let data_dir_b = unique_data_dir();
    std::fs::create_dir_all(&data_dir_b).unwrap();
    let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let m2 = CollectionManager::new_with_storage(&data_dir_b, storage)
        .await
        .unwrap();
    let out = m2
        .get_chunk_relations("rc", 0, RelationDirection::Outgoing, None)
        .await
        .unwrap();
    assert_eq!(out.len(), 1, "relation must survive compaction+restart");
    assert_eq!(out[0].relation_type, "cites");

    let _ = std::fs::remove_dir_all(&data_dir_b);
}

// Concurrent ingests into the same collection: all chunks visible, all ids
// unique, no lost writes (stresses the lock drop/reacquire window).
#[tokio::test]
async fn concurrent_ingests_same_collection() {
    let embed = std::sync::Arc::new(embed_state());
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    let data_dir = unique_data_dir();
    std::fs::create_dir_all(&data_dir).unwrap();
    let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let m = CollectionManager::new_with_storage(&data_dir, storage.clone())
        .await
        .unwrap();
    m.create_collection("conc", None, Some(4), None)
        .await
        .unwrap();

    let n = 12usize;
    let mut handles = Vec::new();
    for i in 0..n {
        let m2 = m.clone();
        let e2 = embed.clone();
        handles.push(tokio::spawn(async move {
            m2.ingest("conc", vec![ingest_chunk(i as u32)], &e2).await
        }));
    }
    for h in handles {
        h.await.unwrap().unwrap();
    }

    // All N chunks present, ids 0..N unique (no collision from the lock gap).
    let (man, _) = crate::storage::lsm::read_manifest(storage.as_ref(), "conc")
        .await
        .unwrap();
    let mat = cloud::materialize(storage.as_ref(), "conc", &man)
        .await
        .unwrap();
    assert_eq!(mat.chunks.len(), n, "all concurrent ingests durable");
    let ids: std::collections::HashSet<u64> = mat.chunks.keys().copied().collect();
    assert_eq!(ids.len(), n, "no duplicate/lost ids");
    assert_eq!(ids, (0..n as u64).collect());

    let _ = std::fs::remove_dir_all(&data_dir);
}

// next_id must NEVER regress across compaction + cold restart. Compaction
// physically drops tombstoned chunks; without the segment's stored max_id
// high-water mark, a fresh-disk rebuild would recompute next_id from the
// live set only and REUSE the deleted ids for new chunks.
#[tokio::test]
async fn no_id_reuse_after_compaction_and_cold_restart() {
    let embed = embed_state();
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    let data_dir_a = unique_data_dir();
    std::fs::create_dir_all(&data_dir_a).unwrap();
    {
        let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m = CollectionManager::new_with_storage(&data_dir_a, storage)
            .await
            .unwrap();
        m.create_collection("idreuse", None, Some(4), None)
            .await
            .unwrap();
        // ids 0..3; delete the two HIGHEST, then compact them away.
        m.ingest("idreuse", (0..4u32).map(ingest_chunk).collect(), &embed)
            .await
            .unwrap();
        m.delete_chunks("idreuse", &[2, 3]).await.unwrap();
        m.compact_collection("idreuse").await.unwrap();
    }
    // Node loss: wipe local disk, cold-rebuild from S3 (max live id is 1).
    std::fs::remove_dir_all(&data_dir_a).unwrap();
    let data_dir_b = unique_data_dir();
    std::fs::create_dir_all(&data_dir_b).unwrap();
    let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let m2 = CollectionManager::new_with_storage(&data_dir_b, storage.clone())
        .await
        .unwrap();

    // A new ingest must get a FRESH id (4), not reuse deleted id 2.
    m2.ingest("idreuse", vec![ingest_chunk(9)], &embed)
        .await
        .unwrap();
    let (man, _) = crate::storage::lsm::read_manifest(storage.as_ref(), "idreuse")
        .await
        .unwrap();
    let mat = cloud::materialize(storage.as_ref(), "idreuse", &man)
        .await
        .unwrap();
    // Under block allocation the exact new id is an allocator detail (a
    // fresh node claims a fresh block); the INVARIANT is that no previously
    // assigned id — live or deleted — is ever reused.
    let new_ids: Vec<u64> = mat.chunks.keys().copied().filter(|id| *id > 3).collect();
    assert_eq!(
        new_ids.len(),
        1,
        "exactly one new chunk with a never-before-assigned id, got {:?}",
        mat.chunks.keys().collect::<Vec<_>>()
    );
    assert!(
        !mat.chunks.contains_key(&2) && !mat.chunks.contains_key(&3),
        "deleted ids must not be reused"
    );

    let _ = std::fs::remove_dir_all(&data_dir_b);
}

// PERSISTENT-DISK restart path (the one the adversarial review flagged):
// in cloud mode, a node restarting with its local disk intact runs
// `load_collection` (rehydrate from redb) and SKIPS rebuild-from-S3 for
// already-loaded collections. A chunk tombstoned locally (redb) — which is
// exactly what delete AND the ingest-compensation path write — must stay
// masked after that restart, even though it's still physically in redb.
#[tokio::test]
async fn persistent_disk_restart_honors_local_tombstones() {
    let embed = embed_state();
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    let data_dir = unique_data_dir();
    std::fs::create_dir_all(&data_dir).unwrap();
    {
        let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m = CollectionManager::new_with_storage(&data_dir, storage)
            .await
            .unwrap();
        m.create_collection("pdisk", None, Some(4), None)
            .await
            .unwrap();
        m.ingest(
            "pdisk",
            vec![ingest_chunk(0), ingest_chunk(1), ingest_chunk(2)],
            &embed,
        )
        .await
        .unwrap();
        // Writes the redb tombstone + RAM tombstone + S3 tombstone — the
        // same three places the ingest-compensation path writes.
        assert_eq!(m.delete_chunks("pdisk", &[1]).await.unwrap().0, 1);
    }

    // Restart with the SAME data_dir (persistent disk — NOT wiped). This
    // takes the load_collection-first, skip-cloud-rebuild path.
    let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let m2 = CollectionManager::new_with_storage(&data_dir, storage)
        .await
        .unwrap();

    let req = SearchRequest {
        query: "chunk".to_string(),
        mode: "semantic".to_string(),
        vector_space: None,
        top_k: 20,
        query_vector: Some(vec![0.1, 0.2, 0.3, 0.4]),
        filters: HashMap::new(),
        score_weights: None,
        recency: None,
        recency_preset: None,
        recency_field: None,
        boosts: Vec::new(),
        relationship_boost: None,
        explain: false,
        include_relations: false,
        relation_types: None,
        relation_direction: RelationDirection::Outgoing,
        min_seq: None,
    };
    let (hits, _, _, _) = m2.search("pdisk", &req, &embed).await.unwrap();
    let hit_ids: std::collections::HashSet<u64> = hits.iter().map(|(c, _, _, _, _)| c.id).collect();
    assert!(
        !hit_ids.contains(&1),
        "tombstoned chunk must stay masked after persistent-disk restart"
    );
    assert!(
        hit_ids.contains(&0) && hit_ids.contains(&2),
        "live chunks must survive, got {:?}",
        hit_ids
    );

    let _ = std::fs::remove_dir_all(&data_dir);
}

// ── Warm-serverless: bucket config + id allocator + writer role ──────

// The bucket collection.json is the source of truth on recovery: specs,
// created_at, and CollectionConfig must survive a cold rebuild instead of
// being re-inferred as model:"recovered" / defaults.
#[tokio::test]
async fn cold_rebuild_recovers_real_collection_config() {
    let embed = embed_state();
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    let data_dir_a = unique_data_dir();
    std::fs::create_dir_all(&data_dir_a).unwrap();
    let created;
    {
        let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m = CollectionManager::new_with_storage(&data_dir_a, storage)
            .await
            .unwrap();
        let mut spaces = HashMap::new();
        spaces.insert(
            "custom".to_string(),
            VectorSpaceConfig {
                dims: 4,
                model: "my-real-model".to_string(),
                status: "active".to_string(),
            },
        );
        created = m
            .create_collection("cfg", Some(spaces), None, None)
            .await
            .unwrap();
        m.ingest("cfg", vec![ingest_chunk(0)], &embed)
            .await
            .unwrap();
    }
    std::fs::remove_dir_all(&data_dir_a).unwrap();

    let data_dir_b = unique_data_dir();
    std::fs::create_dir_all(&data_dir_b).unwrap();
    let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let m2 = CollectionManager::new_with_storage(&data_dir_b, storage)
        .await
        .unwrap();
    let recovered = m2.get_collection("cfg").await.unwrap();
    let space = recovered.vector_spaces.get("custom").unwrap();
    assert_eq!(
        space.model, "my-real-model",
        "specs must not be re-inferred"
    );
    assert_eq!(recovered.created_at, created.created_at);
    let _ = std::fs::remove_dir_all(&data_dir_b);
}

// A zero-ingest collection must be discoverable from a fresh disk (the
// create-only empty manifest + bucket config make the namespace exist).
#[tokio::test]
async fn empty_collection_survives_node_loss() {
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    let data_dir_a = unique_data_dir();
    std::fs::create_dir_all(&data_dir_a).unwrap();
    {
        let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m = CollectionManager::new_with_storage(&data_dir_a, storage)
            .await
            .unwrap();
        m.create_collection("emptyns", None, Some(4), None)
            .await
            .unwrap();
    }
    std::fs::remove_dir_all(&data_dir_a).unwrap();

    let data_dir_b = unique_data_dir();
    std::fs::create_dir_all(&data_dir_b).unwrap();
    let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let m2 = CollectionManager::new_with_storage(&data_dir_b, storage)
        .await
        .unwrap();
    assert!(
        m2.get_collection("emptyns").await.is_some(),
        "zero-ingest collection must be rediscovered from the bucket"
    );
    let _ = std::fs::remove_dir_all(&data_dir_b);
}

// Writer role end-to-end: a node with NO local collection state ingests;
// a fresh serving node sees the data. Ids from writer and attached node
// never collide (both allocate from {ns}/id-alloc).
#[tokio::test]
async fn writer_role_ingest_is_stateless_and_ids_disjoint() {
    let embed = embed_state();
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());

    // Full node creates the collection and ingests two chunks.
    let dir_full = unique_data_dir();
    std::fs::create_dir_all(&dir_full).unwrap();
    let storage_full: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let m_full = CollectionManager::new_with_storage_role(&dir_full, storage_full, NodeRole::Full)
        .await
        .unwrap();
    m_full
        .create_collection("wns", None, Some(4), None)
        .await
        .unwrap();
    m_full
        .ingest("wns", vec![ingest_chunk(0), ingest_chunk(1)], &embed)
        .await
        .unwrap();

    // Writer node: EMPTY data dir, writer role. Ingest must succeed with
    // zero local collection state and never create local index files.
    let dir_writer = unique_data_dir();
    std::fs::create_dir_all(&dir_writer).unwrap();
    let storage_writer: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let m_writer =
        CollectionManager::new_with_storage_role(&dir_writer, storage_writer, NodeRole::Writer)
            .await
            .unwrap();
    let (n, _, _) = m_writer
        .ingest("wns", vec![ingest_chunk(2), ingest_chunk(3)], &embed)
        .await
        .unwrap();
    assert_eq!(n, 2);
    assert!(
        !dir_writer.join("wns").exists(),
        "writer role must not create local collection state"
    );
    // Reads are refused on the writer.
    assert!(m_writer.get_facets("wns", "", &[]).await.is_err());

    // A fresh serving node materializes ALL four chunks with unique ids.
    let dir_read = unique_data_dir();
    std::fs::create_dir_all(&dir_read).unwrap();
    let storage_read: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let m_read = CollectionManager::new_with_storage(&dir_read, storage_read.clone())
        .await
        .unwrap();
    let (man, _) = crate::storage::lsm::read_manifest(storage_read.as_ref(), "wns")
        .await
        .unwrap();
    let mat = cloud::materialize(storage_read.as_ref(), "wns", &man)
        .await
        .unwrap();
    assert_eq!(mat.chunks.len(), 4, "all chunks durable");
    let ids: std::collections::HashSet<u64> = mat.chunks.keys().copied().collect();
    assert_eq!(
        ids.len(),
        4,
        "no id collisions between writer and full node"
    );
    assert!(m_read.get_collection("wns").await.is_some());

    let _ = std::fs::remove_dir_all(&dir_full);
    let _ = std::fs::remove_dir_all(&dir_writer);
    let _ = std::fs::remove_dir_all(&dir_read);
}

// Pre-v0.4 migration: a namespace with data but NO id-alloc object seeds
// the allocator from the bucket-derived high-water mark — new ids never
// collide with existing ones.
#[tokio::test]
async fn id_alloc_migration_seeds_past_existing_ids() {
    let embed = embed_state();
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    let data_dir = unique_data_dir();
    std::fs::create_dir_all(&data_dir).unwrap();
    let storage: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let m = CollectionManager::new_with_storage(&data_dir, storage.clone())
        .await
        .unwrap();
    m.create_collection("mig", None, Some(4), None)
        .await
        .unwrap();
    m.ingest(
        "mig",
        vec![ingest_chunk(0), ingest_chunk(1), ingest_chunk(2)],
        &embed,
    )
    .await
    .unwrap();
    // Simulate a pre-v0.4 namespace: remove the allocator object.
    storage.delete("mig/id-alloc").await.unwrap();
    // Drain the local pool by restarting the manager (pool is in-RAM).
    drop(m);
    let m2 = CollectionManager::new_with_storage(&data_dir, storage.clone())
        .await
        .unwrap();
    m2.ingest("mig", vec![ingest_chunk(9)], &embed)
        .await
        .unwrap();

    let (man, _) = crate::storage::lsm::read_manifest(storage.as_ref(), "mig")
        .await
        .unwrap();
    let mat = cloud::materialize(storage.as_ref(), "mig", &man)
        .await
        .unwrap();
    assert_eq!(mat.chunks.len(), 4);
    let ids: std::collections::HashSet<u64> = mat.chunks.keys().copied().collect();
    assert_eq!(ids.len(), 4, "migrated allocator must not reuse ids 0-2");
    assert!(
        ids.contains(&3),
        "first migrated id is one past the high-water"
    );
    let _ = std::fs::remove_dir_all(&data_dir);
}

// ── Warm-serverless: manifest refresh + read-your-writes ─────────────

fn cloud_search_req(min_seq: Option<u64>) -> SearchRequest {
    SearchRequest {
        query: "chunk".to_string(),
        mode: "semantic".to_string(),
        vector_space: None,
        top_k: 20,
        query_vector: Some(vec![0.1, 0.2, 0.3, 0.4]),
        filters: HashMap::new(),
        score_weights: None,
        recency: None,
        recency_preset: None,
        recency_field: None,
        boosts: Vec::new(),
        relationship_boost: None,
        explain: false,
        include_relations: false,
        relation_types: None,
        relation_direction: RelationDirection::Outgoing,
        min_seq,
    }
}

// Two serving nodes on one bucket: writes on A become visible on B via
// refresh_collection — chunks, deletes, and relations all converge.
#[tokio::test]
async fn two_nodes_converge_via_refresh() {
    let embed = embed_state();
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    let dir_a = unique_data_dir();
    let dir_b = unique_data_dir();
    std::fs::create_dir_all(&dir_a).unwrap();
    std::fs::create_dir_all(&dir_b).unwrap();
    let sa: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let sb: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let a = CollectionManager::new_with_storage(&dir_a, sa)
        .await
        .unwrap();
    a.create_collection("conv", None, Some(4), None)
        .await
        .unwrap();
    a.ingest("conv", vec![ingest_chunk(0)], &embed)
        .await
        .unwrap();

    // B boots AFTER the first write (rebuilds to seq frontier).
    let b = CollectionManager::new_with_storage(&dir_b, sb)
        .await
        .unwrap();
    let (hits, _, _, _) = b
        .search("conv", &cloud_search_req(None), &embed)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1, "B rebuilt A's first write at boot");

    // A writes more: a new chunk, a relation, and a delete of chunk id 0.
    a.ingest("conv", vec![ingest_chunk(1), ingest_chunk(2)], &embed)
        .await
        .unwrap();
    let a_ids: Vec<u64> = {
        let (hits, _, _, _) = a
            .search("conv", &cloud_search_req(None), &embed)
            .await
            .unwrap();
        hits.iter().map(|(c, _, _, _, _)| c.id).collect()
    };
    assert_eq!(a_ids.len(), 3);
    let first_id = *a_ids.iter().min().unwrap();
    let others: Vec<u64> = a_ids.iter().copied().filter(|i| *i != first_id).collect();
    a.create_relations(
        "conv",
        vec![CreateRelation {
            source_chunk_id: others[0],
            target_chunk_id: others[1],
            target_document_id: None,
            relation_type: "cites".into(),
            metadata: HashMap::new(),
        }],
    )
    .await
    .unwrap();
    a.delete_chunks("conv", &[first_id]).await.unwrap();

    // B converges via refresh (no restart, no rebuild).
    b.refresh_collection("conv").await.unwrap();
    let (hits, _, _, _) = b
        .search("conv", &cloud_search_req(None), &embed)
        .await
        .unwrap();
    let b_ids: std::collections::HashSet<u64> = hits.iter().map(|(c, _, _, _, _)| c.id).collect();
    assert!(!b_ids.contains(&first_id), "A's delete visible on B");
    assert_eq!(b_ids.len(), 2, "A's later chunks visible on B");
    let rels = b
        .get_chunk_relations("conv", others[0], RelationDirection::Outgoing, None)
        .await
        .unwrap();
    assert_eq!(rels.len(), 1, "A's relation visible on B");
    assert_eq!(rels[0].relation_type, "cites");

    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

// The refresher must never double-apply a node's OWN fragments (the seq
// tracker covers them out-of-band).
#[tokio::test]
async fn refresh_never_double_applies_own_writes() {
    let embed = embed_state();
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    let dir = unique_data_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let st: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let m = CollectionManager::new_with_storage(&dir, st).await.unwrap();
    m.create_collection("own", None, Some(4), None)
        .await
        .unwrap();
    m.ingest("own", vec![ingest_chunk(0), ingest_chunk(1)], &embed)
        .await
        .unwrap();
    m.delete_chunks("own", &[0]).await.unwrap();

    // Refresh repeatedly: state (incl. chunk_count) must not change.
    let before = m.get_collection("own").await.unwrap().chunk_count;
    for _ in 0..3 {
        m.refresh_collection("own").await.unwrap();
    }
    let after = m.get_collection("own").await.unwrap().chunk_count;
    assert_eq!(before, after, "replay of own fragments must be a no-op");
    assert_eq!(after, 1);
    let _ = std::fs::remove_dir_all(&dir);
}

// Compaction two-branch rule: a node that saw everything skips segments;
// a node whose frontier is BEHIND the watermark re-attaches fully.
#[tokio::test]
async fn refresh_survives_remote_compaction() {
    let embed = embed_state();
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    let dir_a = unique_data_dir();
    let dir_b = unique_data_dir();
    std::fs::create_dir_all(&dir_a).unwrap();
    std::fs::create_dir_all(&dir_b).unwrap();
    let sa: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let sb: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let a = CollectionManager::new_with_storage(&dir_a, sa)
        .await
        .unwrap();
    a.create_collection("rc2", None, Some(4), None)
        .await
        .unwrap();
    a.ingest("rc2", vec![ingest_chunk(0)], &embed)
        .await
        .unwrap();

    // B attaches at frontier 1 (one fragment applied).
    let b = CollectionManager::new_with_storage(&dir_b, sb)
        .await
        .unwrap();

    // Branch 1: A ingests + compacts; B's frontier is BEHIND the watermark
    // (never saw seq 1) → refresh must full re-attach, not skip.
    a.ingest("rc2", vec![ingest_chunk(1)], &embed)
        .await
        .unwrap();
    a.compact_collection("rc2").await.unwrap();
    b.refresh_collection("rc2").await.unwrap();
    let (hits, _, _, _) = b
        .search("rc2", &cloud_search_req(None), &embed)
        .await
        .unwrap();
    assert_eq!(hits.len(), 2, "stale node re-attaches across compaction");

    // Branch 2: B now has everything; another compaction (A side) must be
    // a cheap no-op on refresh (no re-attach needed) and lose nothing.
    a.ingest("rc2", vec![ingest_chunk(2)], &embed)
        .await
        .unwrap();
    b.refresh_collection("rc2").await.unwrap(); // B applies seq tail first
    a.compact_collection("rc2").await.unwrap();
    b.refresh_collection("rc2").await.unwrap(); // wm <= frontier → skip
    let (hits, _, _, _) = b
        .search("rc2", &cloud_search_req(None), &embed)
        .await
        .unwrap();
    assert_eq!(hits.len(), 3);

    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

// Read-your-writes across nodes: a write on A returns a seq; a search on B
// with min_seq=seq refreshes and serves the write.
#[tokio::test]
async fn min_seq_gives_read_your_writes_across_nodes() {
    let embed = embed_state();
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    let dir_a = unique_data_dir();
    let dir_b = unique_data_dir();
    std::fs::create_dir_all(&dir_a).unwrap();
    std::fs::create_dir_all(&dir_b).unwrap();
    let sa: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let sb: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let a = CollectionManager::new_with_storage(&dir_a, sa)
        .await
        .unwrap();
    a.create_collection("ryw", None, Some(4), None)
        .await
        .unwrap();
    a.ingest("ryw", vec![ingest_chunk(0)], &embed)
        .await
        .unwrap();
    let b = CollectionManager::new_with_storage(&dir_b, sb)
        .await
        .unwrap();

    // A writes; B searches with min_seq — must see it without manual refresh.
    let (_, _, seq) = a
        .ingest("ryw", vec![ingest_chunk(1)], &embed)
        .await
        .unwrap();
    let seq = seq.expect("cloud ingest returns a seq");
    let (hits, _, _, _) = b
        .search("ryw", &cloud_search_req(Some(seq)), &embed)
        .await
        .unwrap();
    assert_eq!(hits.len(), 2, "min_seq forces convergence before serving");

    // A min_seq beyond the write history is rejected, not waited on.
    assert!(b
        .search("ryw", &cloud_search_req(Some(9_999)), &embed)
        .await
        .is_err());

    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

// ── Warm-serverless: lazy attach + LRU detach ─────────────────────────

// Lazy boot registers namespaces without rebuilding; the first request
// attaches; a concurrent stampede attaches exactly once.
#[tokio::test]
async fn lazy_attach_on_first_request() {
    let embed = embed_state();
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    // Seed the bucket with a collection via an eager node.
    let dir_seed = unique_data_dir();
    std::fs::create_dir_all(&dir_seed).unwrap();
    {
        let st: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m = CollectionManager::new_with_storage(&dir_seed, st)
            .await
            .unwrap();
        m.create_collection("lazy", None, Some(4), None)
            .await
            .unwrap();
        m.ingest("lazy", vec![ingest_chunk(0), ingest_chunk(1)], &embed)
            .await
            .unwrap();
    }
    std::fs::remove_dir_all(&dir_seed).unwrap();

    // Lazy node: boot must NOT rebuild (no local dir for the collection).
    let dir = unique_data_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let st: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let m = CollectionManager::new_with_storage_opts(&dir, st, NodeRole::Full, true, 0, 0)
        .await
        .unwrap();
    assert!(
        !dir.join("lazy").join("chunks.redb").exists(),
        "lazy boot must not rebuild collections"
    );

    // Stampede: 8 concurrent first-requests; all succeed, attach happens once.
    let mut handles = Vec::new();
    for _ in 0..8 {
        let m2 = m.clone();
        let e2 = embed_state();
        handles.push(tokio::spawn(async move {
            let (hits, _, _, _) = m2
                .search("lazy", &cloud_search_req(None), &e2)
                .await
                .unwrap();
            hits.len()
        }));
    }
    for h in handles {
        assert_eq!(h.await.unwrap(), 2);
    }
    assert!(dir.join("lazy").join("chunks.redb").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

// LRU detach: with a budget of 1, attaching a second collection evicts the
// least-recently-used one; the evicted collection re-attaches on demand
// with all its data (bucket is the source of truth).
#[tokio::test]
async fn lru_detach_and_reattach_roundtrip() {
    let embed = embed_state();
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    let dir_seed = unique_data_dir();
    std::fs::create_dir_all(&dir_seed).unwrap();
    {
        let st: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m = CollectionManager::new_with_storage(&dir_seed, st)
            .await
            .unwrap();
        for name in ["one", "two"] {
            m.create_collection(name, None, Some(4), None)
                .await
                .unwrap();
            m.ingest(name, vec![ingest_chunk(0)], &embed).await.unwrap();
        }
    }
    std::fs::remove_dir_all(&dir_seed).unwrap();

    let dir = unique_data_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let st: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let m = CollectionManager::new_with_storage_opts(&dir, st, NodeRole::Full, true, 1, 0)
        .await
        .unwrap();

    // Attach "one", then "two" — budget 1 evicts "one".
    let (hits, _, _, _) = m
        .search("one", &cloud_search_req(None), &embed)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    let (hits, _, _, _) = m
        .search("two", &cloud_search_req(None), &embed)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    {
        let attached = m.collections.read().await;
        assert_eq!(attached.len(), 1, "LRU budget enforced");
        assert!(attached.contains_key("two"));
    }
    assert!(!dir.join("one").join("chunks.redb").exists());

    // Evicted collection re-attaches on demand, data intact.
    let (hits, _, _, _) = m
        .search("one", &cloud_search_req(None), &embed)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1, "re-attach after eviction serves all data");
    let _ = std::fs::remove_dir_all(&dir);
}

// Lazy mode keeps metadata correct: list/get see registered collections;
// a collection created on ANOTHER node after boot attaches on demand.
#[tokio::test]
async fn lazy_attach_discovers_foreign_creates() {
    let embed = embed_state();
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    let dir_a = unique_data_dir();
    let dir_b = unique_data_dir();
    std::fs::create_dir_all(&dir_a).unwrap();
    std::fs::create_dir_all(&dir_b).unwrap();
    let sa: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let sb: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    // Lazy node boots FIRST (empty bucket).
    let b = CollectionManager::new_with_storage_opts(&dir_b, sb, NodeRole::Full, true, 0, 0)
        .await
        .unwrap();
    // Another node creates + writes afterwards.
    let a = CollectionManager::new_with_storage(&dir_a, sa)
        .await
        .unwrap();
    a.create_collection("late", None, Some(4), None)
        .await
        .unwrap();
    a.ingest("late", vec![ingest_chunk(0)], &embed)
        .await
        .unwrap();

    // B never saw "late" at boot; first request attaches it anyway.
    let (hits, _, _, _) = b
        .search("late", &cloud_search_req(None), &embed)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1, "foreign create attaches on demand");

    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

// ── Review-driven regression tests (adversarial round) ───────────────

#[test]
fn seq_tracker_semantics() {
    let mut t = SeqTracker::default();
    assert!(!t.covers(0));
    t.mark(0);
    assert_eq!(t.contiguous, 1);
    // Out-of-band mark ahead of the frontier; contiguous holds.
    t.mark(2);
    assert!(t.covers(2) && !t.covers(1));
    assert_eq!(t.contiguous, 1);
    // Filling the gap drains the whole out-of-band run.
    t.mark(1);
    assert_eq!(t.contiguous, 3);
    assert!(t.out_of_band.is_empty());
    // Duplicate + below-frontier marks are no-ops (no unbounded growth).
    t.mark(1);
    t.mark(2);
    assert_eq!(t.contiguous, 3);
    assert!(t.out_of_band.is_empty());
    // starting_at seeds the frontier.
    let t2 = SeqTracker::starting_at(7);
    assert!(t2.covers(6) && !t2.covers(7));
}

// H4 regression: eviction must be least-recently-USED, not least-recently-
// attached. 3 collections, budget 2: attach a, attach b, USE a, attach c
// → b (not a) is evicted.
#[tokio::test]
async fn lru_evicts_least_recently_used() {
    let embed = embed_state();
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    let dir_seed = unique_data_dir();
    std::fs::create_dir_all(&dir_seed).unwrap();
    {
        let st: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let m = CollectionManager::new_with_storage(&dir_seed, st)
            .await
            .unwrap();
        for name in ["a", "b", "c"] {
            m.create_collection(name, None, Some(4), None)
                .await
                .unwrap();
            m.ingest(name, vec![ingest_chunk(0)], &embed).await.unwrap();
        }
    }
    std::fs::remove_dir_all(&dir_seed).unwrap();
    let dir = unique_data_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let st: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let m = CollectionManager::new_with_storage_opts(&dir, st, NodeRole::Full, true, 2, 0)
        .await
        .unwrap();
    m.search("a", &cloud_search_req(None), &embed)
        .await
        .unwrap();
    m.search("b", &cloud_search_req(None), &embed)
        .await
        .unwrap();
    // USE a again — it is now hotter than b.
    m.search("a", &cloud_search_req(None), &embed)
        .await
        .unwrap();
    m.search("c", &cloud_search_req(None), &embed)
        .await
        .unwrap();
    let attached = m.collections.read().await;
    assert!(attached.contains_key("a"), "hot collection must survive");
    assert!(!attached.contains_key("b"), "cold collection is the victim");
    assert!(attached.contains_key("c"));
}

// C1 regression: a vector space added on node A becomes visible on an
// already-attached node B via refresh (config is synced, not just
// fragments), so B never quarantines chunks carrying the new space.
#[tokio::test]
async fn vector_space_add_propagates_via_refresh() {
    let embed = embed_state();
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    let dir_a = unique_data_dir();
    let dir_b = unique_data_dir();
    std::fs::create_dir_all(&dir_a).unwrap();
    std::fs::create_dir_all(&dir_b).unwrap();
    let sa: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let sb: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let a = CollectionManager::new_with_storage(&dir_a, sa)
        .await
        .unwrap();
    a.create_collection("vsprop", None, Some(4), None)
        .await
        .unwrap();
    a.ingest("vsprop", vec![ingest_chunk(0)], &embed)
        .await
        .unwrap();
    let b = CollectionManager::new_with_storage(&dir_b, sb)
        .await
        .unwrap();

    // A adds an 8-dim space, then ingests a chunk carrying it.
    a.add_vector_space("vsprop", "wide", 8, "test-model")
        .await
        .unwrap();
    let mut ic = ingest_chunk(1);
    ic.embeddings.insert("wide".to_string(), vec![0.1; 8]);
    a.ingest("vsprop", vec![ic], &embed).await.unwrap();

    // B refreshes: must learn the space AND apply the chunk (no quarantine).
    b.refresh_collection("vsprop").await.unwrap();
    let bc = b.get_collection("vsprop").await.unwrap();
    assert!(bc.vector_spaces.contains_key("wide"), "config converged");
    let (hits, _, _, _) = b
        .search("vsprop", &cloud_search_req(None), &embed)
        .await
        .unwrap();
    assert_eq!(
        hits.len(),
        2,
        "chunk with the new space applied, not quarantined"
    );

    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

// Rank-1 regression: ingest racing a refresher loop never double-applies
// (chunk_count exact, no duplicate hits).
#[tokio::test]
async fn ingest_races_refresher_no_double_apply() {
    let embed = std::sync::Arc::new(embed_state());
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    let dir = unique_data_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let st: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let m = CollectionManager::new_with_storage(&dir, st).await.unwrap();
    m.create_collection("race", None, Some(4), None)
        .await
        .unwrap();

    let n = 10usize;
    let refresher = {
        let m2 = m.clone();
        tokio::spawn(async move {
            for _ in 0..200 {
                let _ = m2.refresh_collection("race").await;
                tokio::task::yield_now().await;
            }
        })
    };
    let mut handles = Vec::new();
    for i in 0..n {
        let m2 = m.clone();
        let e2 = embed.clone();
        handles.push(tokio::spawn(async move {
            m2.ingest("race", vec![ingest_chunk(i as u32)], &e2).await
        }));
    }
    for h in handles {
        h.await.unwrap().unwrap();
    }
    refresher.await.unwrap();
    let _ = m.refresh_collection("race").await;

    let c = m.get_collection("race").await.unwrap();
    assert_eq!(c.chunk_count as usize, n, "no double-count under the race");
    let (hits, _, _, _) = m
        .search("race", &cloud_search_req(None), &embed)
        .await
        .unwrap();
    assert_eq!(hits.len(), n, "no duplicate/lost chunks under the race");
    let _ = std::fs::remove_dir_all(&dir);
}

// Rank-6: persistent-disk restart catches up the REMOTE delta via refresh
// instead of serving stale data (applied_seq persistence path).
#[tokio::test]
async fn persistent_restart_catches_up_remote_delta() {
    let embed = embed_state();
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    let dir_a = unique_data_dir();
    let dir_w = unique_data_dir();
    std::fs::create_dir_all(&dir_a).unwrap();
    std::fs::create_dir_all(&dir_w).unwrap();
    {
        let sa: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let a = CollectionManager::new_with_storage(&dir_a, sa)
            .await
            .unwrap();
        a.create_collection("pd", None, Some(4), None)
            .await
            .unwrap();
        a.ingest("pd", vec![ingest_chunk(0)], &embed).await.unwrap();
    } // node A down; its disk PERSISTS.
    {
        let sw: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
            store.clone(),
            "object-store:memory",
        ));
        let w = CollectionManager::new_with_storage_role(&dir_w, sw, NodeRole::Writer)
            .await
            .unwrap();
        w.ingest("pd", vec![ingest_chunk(1)], &embed).await.unwrap();
    }
    // A restarts on the SAME dir (load_collection path, not rebuild).
    let sa: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let a = CollectionManager::new_with_storage(&dir_a, sa)
        .await
        .unwrap();
    a.refresh_collection("pd").await.unwrap();
    let (hits, _, _, _) = a
        .search("pd", &cloud_search_req(None), &embed)
        .await
        .unwrap();
    assert_eq!(
        hits.len(),
        2,
        "restart + refresh catches up the writer's delta"
    );
    let c = a.get_collection("pd").await.unwrap();
    assert_eq!(c.chunk_count, 2, "delta applied exactly once");
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_w);
}

// Rank-8: a wrong-dims chunk inside a fragment is quarantined on replay
// without corrupting anything else.
#[tokio::test]
async fn refresh_quarantines_wrong_dims_without_corruption() {
    let embed = embed_state();
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    let dir = unique_data_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let st: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let m = CollectionManager::new_with_storage(&dir, st.clone())
        .await
        .unwrap();
    m.create_collection("quar", None, Some(4), None)
        .await
        .unwrap();
    m.ingest("quar", vec![ingest_chunk(0)], &embed)
        .await
        .unwrap();

    // Hand-craft a fragment with one bad (3-dim) and one good chunk,
    // simulating a poisoned foreign writer.
    let mut bad = DocumentChunk {
        id: 500_000,
        collection: "quar".into(),
        file_id: "bad".into(),
        chunk_index: 0,
        page: None,
        text: "bad chunk".into(),
        metadata: HashMap::new(),
        doc_type: "chunk".into(),
        parent_id: None,
        group_id: None,
        embeddings: HashMap::new(),
        embedding: None,
    };
    bad.embeddings.insert("default".into(), vec![0.1, 0.2, 0.3]);
    let mut good = bad.clone();
    good.id = 500_001;
    good.file_id = "good".into();
    good.text = "good chunk".into();
    good.embeddings
        .insert("default".into(), vec![0.1, 0.2, 0.3, 0.4]);
    let payload = serde_json::to_vec(&vec![bad, good]).unwrap();
    crate::storage::lsm::append_fragment(st.as_ref(), "quar", bytes::Bytes::from(payload), 2)
        .await
        .unwrap();

    m.refresh_collection("quar").await.unwrap();
    let (hits, _, _, _) = m
        .search("quar", &cloud_search_req(None), &embed)
        .await
        .unwrap();
    let ids: std::collections::HashSet<u64> = hits.iter().map(|(c, _, _, _, _)| c.id).collect();
    assert!(ids.contains(&500_001), "good chunk applied");
    assert!(!ids.contains(&500_000), "bad chunk quarantined");
    // Post-quarantine ingest still works and searches correctly (mmap not shifted).
    m.ingest("quar", vec![ingest_chunk(9)], &embed)
        .await
        .unwrap();
    let (hits, _, _, _) = m
        .search("quar", &cloud_search_req(None), &embed)
        .await
        .unwrap();
    assert_eq!(hits.len(), 3);
    let _ = std::fs::remove_dir_all(&dir);
}

// Rank-9: min_seq is ignored in local mode; exact boundary at next_seq.
#[tokio::test]
async fn min_seq_local_mode_and_boundary() {
    let embed = embed_state();
    // Local mode: min_seq must be ignored, not error.
    let dir = unique_data_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let m = CollectionManager::new(&dir).await.unwrap();
    m.create_collection("loc", None, Some(4), None)
        .await
        .unwrap();
    m.ingest("loc", vec![ingest_chunk(0)], &embed)
        .await
        .unwrap();
    let (hits, _, _, _) = m
        .search("loc", &cloud_search_req(Some(999)), &embed)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1, "local mode ignores min_seq");
    let _ = std::fs::remove_dir_all(&dir);

    // Cloud: last valid seq (next_seq-1) succeeds; next_seq is rejected.
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    let dir2 = unique_data_dir();
    std::fs::create_dir_all(&dir2).unwrap();
    let st: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let m2 = CollectionManager::new_with_storage(&dir2, st)
        .await
        .unwrap();
    m2.create_collection("bnd", None, Some(4), None)
        .await
        .unwrap();
    let (_, _, seq) = m2
        .ingest("bnd", vec![ingest_chunk(0)], &embed)
        .await
        .unwrap();
    let seq = seq.unwrap();
    assert!(m2
        .search("bnd", &cloud_search_req(Some(seq)), &embed)
        .await
        .is_ok());
    assert!(m2
        .search("bnd", &cloud_search_req(Some(seq + 1)), &embed)
        .await
        .is_err());
    let _ = std::fs::remove_dir_all(&dir2);
}

// Rank-4/H3: a writer delete against a bogus namespace must NOT create a
// phantom collection, and absurd ids are rejected by the allocator frontier.
#[tokio::test]
async fn writer_delete_validates_namespace_and_ids() {
    let embed = embed_state();
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    let dir = unique_data_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let st: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let w = CollectionManager::new_with_storage_role(&dir, st.clone(), NodeRole::Writer)
        .await
        .unwrap();
    // Bogus namespace: error + nothing created in the bucket.
    assert!(w.delete_chunks("ghost", &[1]).await.is_err());
    assert!(
        !st.exists("ghost/manifest").await.unwrap(),
        "no phantom namespace"
    );

    // Real collection: absurd id rejected (would poison max_id forever).
    let dir_f = unique_data_dir();
    std::fs::create_dir_all(&dir_f).unwrap();
    let sf: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let f = CollectionManager::new_with_storage(&dir_f, sf)
        .await
        .unwrap();
    f.create_collection("real", None, Some(4), None)
        .await
        .unwrap();
    f.ingest("real", vec![ingest_chunk(0)], &embed)
        .await
        .unwrap();
    assert!(w.delete_chunks("real", &[u64::MAX]).await.is_err());
    // In-range delete works.
    assert!(w.delete_chunks("real", &[0]).await.is_ok());
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&dir_f);
}

// H1-lite: delete+recreate on another node is detected via created_at and
// the stale node re-attaches to the NEW collection.
#[tokio::test]
async fn delete_recreate_detected_by_refresh() {
    let embed = embed_state();
    let store = std::sync::Arc::new(object_store::memory::InMemory::new());
    let dir_a = unique_data_dir();
    let dir_b = unique_data_dir();
    std::fs::create_dir_all(&dir_a).unwrap();
    std::fs::create_dir_all(&dir_b).unwrap();
    let sa: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let sb: Arc<dyn Storage> = Arc::new(ObjectStoreBackend::from_store(
        store.clone(),
        "object-store:memory",
    ));
    let a = CollectionManager::new_with_storage(&dir_a, sa)
        .await
        .unwrap();
    a.create_collection("cycle", None, Some(4), None)
        .await
        .unwrap();
    a.ingest(
        "cycle",
        vec![ingest_chunk(0), ingest_chunk(1), ingest_chunk(2)],
        &embed,
    )
    .await
    .unwrap();
    let b = CollectionManager::new_with_storage(&dir_b, sb)
        .await
        .unwrap();

    // A deletes and recreates with different content.
    a.delete_collection("cycle").await.unwrap();
    a.create_collection("cycle", None, Some(4), None)
        .await
        .unwrap();
    a.ingest("cycle", vec![ingest_chunk(9)], &embed)
        .await
        .unwrap();

    // B refreshes: must serve the NEW collection (1 chunk), not the old 3.
    b.refresh_collection("cycle").await.unwrap();
    let (hits, _, _, _) = b
        .search("cycle", &cloud_search_req(None), &embed)
        .await
        .unwrap();
    assert_eq!(
        hits.len(),
        1,
        "stale node re-attached to the recreated collection"
    );
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

// ── Scale harness (env-gated) ─────────────────────────────────────────
// COMPASS_SCALE_N=<chunks> [COMPASS_SCALE_DIMS=<dims>] cargo test
//   --features object-storage --release scale_envelope -- --nocapture
// Measures ingest throughput, attach (cold rebuild) time, and search
// latency against a local-disk Storage backend (same code paths as S3,
// disk-bound). Skips (passes) when COMPASS_SCALE_N is unset.
#[tokio::test]
async fn scale_envelope() {
    let Ok(n) = std::env::var("COMPASS_SCALE_N") else {
        eprintln!("skipped: COMPASS_SCALE_N not set");
        return;
    };
    let n: usize = n.parse().unwrap();
    let dims: usize = std::env::var("COMPASS_SCALE_DIMS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let batch = 2_000usize;
    let embed = embed_state();

    let bucket_dir = unique_data_dir();
    std::fs::create_dir_all(&bucket_dir).unwrap();
    let storage: Arc<dyn Storage> =
        Arc::new(crate::storage::local::LocalDiskStorage::new(&bucket_dir).unwrap());
    // local-disk backend reports "local-disk" => cloud_mode false. Wrap it
    // to report as a cloud backend so the full S3-native path runs.
    struct CloudyDisk(Arc<dyn Storage>);
    #[async_trait::async_trait]
    impl Storage for CloudyDisk {
        async fn get(&self, k: &str) -> Result<bytes::Bytes, crate::storage::StorageError> {
            self.0.get(k).await
        }
        async fn get_range(
            &self,
            k: &str,
            r: std::ops::Range<u64>,
        ) -> Result<bytes::Bytes, crate::storage::StorageError> {
            self.0.get_range(k, r).await
        }
        async fn get_versioned(
            &self,
            k: &str,
        ) -> Result<(bytes::Bytes, crate::storage::Version), crate::storage::StorageError> {
            self.0.get_versioned(k).await
        }
        async fn put(
            &self,
            k: &str,
            b: bytes::Bytes,
        ) -> Result<crate::storage::Version, crate::storage::StorageError> {
            self.0.put(k, b).await
        }
        async fn put_if_match(
            &self,
            k: &str,
            b: bytes::Bytes,
            e: &crate::storage::Version,
        ) -> Result<crate::storage::Version, crate::storage::StorageError> {
            self.0.put_if_match(k, b, e).await
        }
        async fn put_if_not_exists(
            &self,
            k: &str,
            b: bytes::Bytes,
        ) -> Result<crate::storage::Version, crate::storage::StorageError> {
            self.0.put_if_not_exists(k, b).await
        }
        async fn delete(&self, k: &str) -> Result<(), crate::storage::StorageError> {
            self.0.delete(k).await
        }
        async fn put_large(
            &self,
            k: &str,
            b: bytes::Bytes,
        ) -> Result<crate::storage::Version, crate::storage::StorageError> {
            self.0.put_large(k, b).await
        }
        async fn list(
            &self,
            p: &str,
        ) -> Result<Vec<crate::storage::ObjectMeta>, crate::storage::StorageError> {
            self.0.list(p).await
        }
        async fn list_dirs(&self, p: &str) -> Result<Vec<String>, crate::storage::StorageError> {
            self.0.list_dirs(p).await
        }
        fn backend_name(&self) -> &'static str {
            "scale-disk"
        }
    }
    let storage: Arc<dyn Storage> = Arc::new(CloudyDisk(storage));

    let node_dir = unique_data_dir();
    std::fs::create_dir_all(&node_dir).unwrap();
    let m = CollectionManager::new_with_storage_opts(
        &node_dir,
        storage.clone(),
        NodeRole::Full,
        false,
        0,
        0,
    )
    .await
    .unwrap();
    let mut spaces = HashMap::new();
    spaces.insert(
        "default".to_string(),
        VectorSpaceConfig {
            dims,
            model: "scale".into(),
            status: "active".into(),
        },
    );
    m.create_collection("scale", Some(spaces), None, None)
        .await
        .unwrap();

    // Deterministic pseudo-random embeddings (no Math.random / clock).
    let mk_vec = |seed: usize| -> Vec<f32> {
        let mut x = seed as u64 * 6364136223846793005 + 1442695040888963407;
        (0..dims)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x % 2000) as f32 / 1000.0) - 1.0
            })
            .collect()
    };
    let t0 = std::time::Instant::now();
    for b0 in (0..n).step_by(batch) {
        let chunks: Vec<IngestChunk> = (b0..(b0 + batch).min(n))
            .map(|i| {
                let mut embeddings = HashMap::new();
                embeddings.insert("default".to_string(), mk_vec(i));
                IngestChunk {
                    client_id: None,
                    file_id: format!("f{i}"),
                    chunk_index: 0,
                    page: None,
                    text: format!("scale test chunk number {i} lorem ipsum"),
                    metadata: HashMap::new(),
                    doc_type: "chunk".to_string(),
                    parent_id: None,
                    parent_ref: None,
                    group_id: None,
                    embeddings,
                    embedding: None,
                }
            })
            .collect();
        m.ingest("scale", chunks, &embed).await.unwrap();
    }
    let ingest_s = t0.elapsed().as_secs_f64();

    // Cold attach: fresh node dir, same bucket.
    drop(m);
    let node2 = unique_data_dir();
    std::fs::create_dir_all(&node2).unwrap();
    let t1 = std::time::Instant::now();
    let m2 = CollectionManager::new_with_storage_opts(
        &node2,
        storage.clone(),
        NodeRole::Full,
        false,
        0,
        0,
    )
    .await
    .unwrap();
    let attach_s = t1.elapsed().as_secs_f64();

    // Search latency (semantic, 50 queries).
    let mut req = cloud_search_req(None);
    let t2 = std::time::Instant::now();
    let mut hits_total = 0usize;
    for q in 0..50 {
        req.query_vector = Some(mk_vec(q * 7919));
        let (hits, _, _, _) = m2.search("scale", &req, &embed).await.unwrap();
        hits_total += hits.len();
    }
    let search_ms = t2.elapsed().as_secs_f64() * 1000.0 / 50.0;
    assert!(hits_total > 0);

    eprintln!(
        "SCALE n={n} dims={dims}: ingest {:.1}s ({:.0} chunks/s) | cold attach {:.1}s | search avg {:.1}ms",
        ingest_s, n as f64 / ingest_s, attach_s, search_ms
    );
    let _ = std::fs::remove_dir_all(&bucket_dir);
    let _ = std::fs::remove_dir_all(&node_dir);
    let _ = std::fs::remove_dir_all(&node2);
}
