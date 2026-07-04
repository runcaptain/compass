// collections/persistence_tests.rs — extracted test module (was inline in mod.rs).
// Child module of `collections`: `super::*` sees the parent's private items.

//! End-to-end durability test. Builds a CollectionManager in a temp dir,
//! ingests chunks, drops the manager (closing the chunk store), creates a
//! new manager pointing at the same dir, and asserts the chunks come back.
//!
//! This is the test that proves Compass survives process restarts. Without
//! the disk-backed ChunkStore wiring, this test would fail because
//! `loaded.chunks` would be empty after the manager restart.

use super::*;
use crate::embed::EmbedState;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

fn unique_data_dir() -> std::path::PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "compass-persist-test-{}-{}-{}",
        std::process::id(),
        nanos,
        N.fetch_add(1, Ordering::SeqCst)
    ))
}

fn empty_embed_state() -> EmbedState {
    // No embedding models loaded. Safe for the persistence test because
    // we provide chunks without text-only embedding requirements. Any
    // call to embed_query returns Err and the ingest path tolerates that.
    EmbedState {
        bge: None,
        distilled: None,
    }
}

fn make_ingest_chunk(file_id: &str, text: &str) -> IngestChunk {
    IngestChunk {
        client_id: None,
        file_id: file_id.to_string(),
        chunk_index: 0,
        page: None,
        text: text.to_string(),
        metadata: HashMap::new(),
        doc_type: "chunk".to_string(),
        parent_id: None,
        parent_ref: None,
        group_id: None,
        embeddings: HashMap::new(),
        embedding: None,
    }
}

// Regression for the three facet bugs the live E2E harness caught:
// (1) a second ingest batch replaced facet state instead of accumulating
//     (latent since v0.2 — build_index returned new-batch-only bitsets);
// (2) facets came back empty after a restart (open_index returns empty
//     state and nothing rebuilt it);
// (3) deleted chunks kept inflating counts (facets never saw tombstones).
#[tokio::test]
async fn facets_accumulate_survive_restart_and_exclude_deleted() {
    let data_dir = unique_data_dir();
    std::fs::create_dir_all(&data_dir).unwrap();
    let embed = empty_embed_state();
    let tagged = |file: &str, text: &str, kind: &str| {
        let mut c = make_ingest_chunk(file, text);
        c.metadata.insert(
            "kind".to_string(),
            crate::models::MetadataValue::String(kind.to_string()),
        );
        c
    };
    let field = ["kind".to_string()];

    {
        let manager = CollectionManager::new(&data_dir).await.unwrap();
        manager
            .create_collection("facet-test", None, None, None)
            .await
            .unwrap();
        manager
            .ingest(
                "facet-test",
                vec![
                    tagged("a", "alpha doc", "report"),
                    tagged("b", "beta doc", "memo"),
                ],
                &embed,
            )
            .await
            .unwrap();
        // Bug 1: this second batch must ADD to the first, not replace it.
        manager
            .ingest(
                "facet-test",
                vec![tagged("c", "gamma doc", "report")],
                &embed,
            )
            .await
            .unwrap();
        let (facets, _) = manager.get_facets("facet-test", "", &field).await.unwrap();
        let kind = facets.get("kind").expect("facets survive a second batch");
        assert_eq!(kind.get("report"), Some(&2));
        assert_eq!(kind.get("memo"), Some(&1));
    }

    // Bug 2: facets must be rebuilt from the chunk store on restart.
    let manager2 = CollectionManager::new(&data_dir).await.unwrap();
    let (facets, _) = manager2.get_facets("facet-test", "", &field).await.unwrap();
    let kind = facets.get("kind").expect("facets survive a restart");
    assert_eq!(kind.get("report"), Some(&2));
    assert_eq!(kind.get("memo"), Some(&1));

    // Bug 3: deleting a chunk must drop it from counts immediately.
    let (_, ids) = manager2.get_all_chunk_data("facet-test").await.unwrap();
    let (texts, _) = manager2.get_all_chunk_data("facet-test").await.unwrap();
    let memo_id = ids
        .iter()
        .zip(texts.iter())
        .find(|(_, t)| t.contains("beta"))
        .map(|(id, _)| *id)
        .unwrap();
    manager2
        .delete_chunks("facet-test", &[memo_id])
        .await
        .unwrap();
    let (facets, _) = manager2.get_facets("facet-test", "", &field).await.unwrap();
    let kind = facets.get("kind").unwrap();
    assert_eq!(kind.get("report"), Some(&2));
    assert!(
        kind.get("memo").is_none() || kind.get("memo") == Some(&0),
        "deleted chunk still counted in facets: {kind:?}"
    );

    let _ = std::fs::remove_dir_all(&data_dir);
}

#[tokio::test]
async fn chunks_persist_across_manager_restart() {
    let data_dir = unique_data_dir();
    std::fs::create_dir_all(&data_dir).unwrap();
    let embed = empty_embed_state();

    // First manager lifetime: create collection, ingest three chunks,
    // then drop the manager to close all file handles (including redb).
    {
        let manager = CollectionManager::new(&data_dir).await.unwrap();
        manager
            .create_collection("persist-test", None, None, None)
            .await
            .unwrap();
        let to_ingest = vec![
            make_ingest_chunk("f1", "first chunk"),
            make_ingest_chunk("f2", "second chunk"),
            make_ingest_chunk("f3", "third chunk"),
        ];
        let (ingested, _, _) = manager
            .ingest("persist-test", to_ingest, &embed)
            .await
            .unwrap();
        assert_eq!(ingested, 3, "ingest call reports 3 chunks written");
        // manager dropped here
    }

    // Second manager: same data dir, must rehydrate chunks from disk.
    let manager2 = CollectionManager::new(&data_dir).await.unwrap();
    let (texts, ids) = manager2.get_all_chunk_data("persist-test").await.unwrap();

    assert_eq!(
        ids.len(),
        3,
        "expected 3 chunks rehydrated from disk after manager restart, got {}",
        ids.len()
    );
    let mut sorted_texts = texts.clone();
    sorted_texts.sort();
    assert_eq!(
        sorted_texts,
        vec![
            "first chunk".to_string(),
            "second chunk".to_string(),
            "third chunk".to_string(),
        ],
        "chunk texts should match what was ingested before the restart"
    );

    // Cleanup
    let _ = std::fs::remove_dir_all(&data_dir);
}

#[tokio::test]
async fn next_id_advances_correctly_after_rehydration() {
    // After rehydration, next_id should be max(seen) + 1 so new ingests
    // don't collide with persisted IDs. Verify by ingesting again after
    // restart and checking the new chunk got a fresh ID.
    let data_dir = unique_data_dir();
    std::fs::create_dir_all(&data_dir).unwrap();
    let embed = empty_embed_state();

    // Round 1: ingest two chunks (IDs 0, 1)
    {
        let manager = CollectionManager::new(&data_dir).await.unwrap();
        manager
            .create_collection("next-id-test", None, None, None)
            .await
            .unwrap();
        manager
            .ingest(
                "next-id-test",
                vec![
                    make_ingest_chunk("f0", "round-one-a"),
                    make_ingest_chunk("f1", "round-one-b"),
                ],
                &embed,
            )
            .await
            .unwrap();
    }

    // Round 2: restart and ingest one more chunk. The new chunk's ID
    // should be 2, not 0.
    let manager2 = CollectionManager::new(&data_dir).await.unwrap();
    manager2
        .ingest(
            "next-id-test",
            vec![make_ingest_chunk("f2", "round-two")],
            &embed,
        )
        .await
        .unwrap();
    let (_, ids) = manager2.get_all_chunk_data("next-id-test").await.unwrap();
    let mut sorted_ids = ids.clone();
    sorted_ids.sort();
    assert_eq!(
        sorted_ids,
        vec![0, 1, 2],
        "next_id must advance past max persisted id, got ids: {:?}",
        sorted_ids
    );

    let _ = std::fs::remove_dir_all(&data_dir);
}
