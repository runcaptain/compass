// collections/cold_serve_tests.rs — Phase 5 serve-from-storage coverage.
// A node with cold serving on answers semantic queries on namespaces it has
// NEVER attached, straight from object-storage range reads.

use super::*;
use crate::embed::EmbedState;
use crate::storage::object_store_backend::ObjectStoreBackend;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

fn unique_data_dir() -> std::path::PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "compass-cold-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ))
}

fn embed_state() -> EmbedState {
    EmbedState {
        bge: None,
        distilled: None,
    }
}

fn mem_storage() -> Arc<dyn Storage> {
    Arc::new(ObjectStoreBackend::from_store(
        std::sync::Arc::new(object_store::memory::InMemory::new()),
        "object-store:memory",
    ))
}

const DIMS: usize = 8;

/// Deterministic embedding: direction depends on i % 4, plus a nudge that is
/// UNIQUE per id (ties would make self-recall assertions ambiguous).
fn vec_for(i: u64) -> Vec<f32> {
    let mut v = vec![0.05f32; DIMS];
    v[(i % 4) as usize * 2] = 1.0;
    v[7] = i as f32 * 1e-4;
    v
}

fn mk_chunk(i: u64, kind: &str) -> IngestChunk {
    let mut metadata = HashMap::new();
    metadata.insert("kind".to_string(), MetadataValue::String(kind.to_string()));
    let mut embeddings = HashMap::new();
    embeddings.insert("default".to_string(), vec_for(i));
    IngestChunk {
        client_id: Some(format!("c{i}")),
        file_id: format!("f{i}"),
        chunk_index: 0,
        page: None,
        text: format!("cold document number {i}"),
        metadata,
        doc_type: "chunk".to_string(),
        parent_id: None,
        parent_ref: None,
        group_id: None,
        embeddings,
        embedding: None,
    }
}

fn semantic_req(target: u64, top_k: usize) -> SearchRequest {
    SearchRequest {
        query: String::new(),
        mode: "semantic".to_string(),
        vector_space: None,
        top_k,
        query_vector: Some(vec_for(target)),
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
        relation_direction: RelationDirection::default(),
        min_seq: None,
    }
}

async fn cold_manager(dir: &std::path::Path, storage: Arc<dyn Storage>) -> Arc<CollectionManager> {
    std::fs::create_dir_all(dir).unwrap();
    let m = CollectionManager::new_with_storage_opts(dir, storage, NodeRole::Full, true, 0, 0)
        .await
        .unwrap();
    m.set_cold_serve(true);
    m.set_warm_after(0); // promotion off unless a test opts in
    m
}

// Core promise: a fresh node answers semantic queries on a compacted (v3,
// clustered) namespace WITHOUT attaching it — and sees WAL-tail writes and
// tombstones that landed after compaction (read-your-writes from storage).
#[tokio::test]
async fn cold_search_serves_without_attach() {
    let storage = mem_storage();
    let embed = embed_state();

    // Writer side: ingest past the clustering threshold, then compact.
    let n = (crate::search::ivf::CLUSTER_MIN_ROWS + 200) as u64;
    {
        let dir = unique_data_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let a = CollectionManager::new_with_storage(&dir, storage.clone())
            .await
            .unwrap();
        a.create_collection("frozen", None, Some(DIMS), None)
            .await
            .unwrap();
        let mut batch = Vec::new();
        for i in 0..n {
            batch.push(mk_chunk(i, if i % 2 == 0 { "even" } else { "odd" }));
            if batch.len() == 1000 {
                a.ingest("frozen", std::mem::take(&mut batch), &embed)
                    .await
                    .unwrap();
            }
        }
        if !batch.is_empty() {
            a.ingest("frozen", batch, &embed).await.unwrap();
        }
        let live = a.compact_collection("frozen").await.unwrap();
        assert_eq!(live, n);
        // Post-compaction writes + a delete stay in the WAL tail.
        a.ingest("frozen", vec![mk_chunk(n, "tail")], &embed)
            .await
            .unwrap();
        a.delete_chunks("frozen", &[2]).await.unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Cold node: never attaches, still answers.
    let dir_b = unique_data_dir();
    let b = cold_manager(&dir_b, storage.clone()).await;

    // Exact-vector self-recall: the target chunk must be the top hit.
    let target = 1234u64;
    let (results, _, _, _) = b
        .search("frozen", &semantic_req(target, 5), &embed)
        .await
        .unwrap();
    assert!(!results.is_empty(), "cold search returned nothing");
    assert_eq!(
        results[0].0.file_id,
        format!("f{target}"),
        "self-recall: exact stored vector must rank first"
    );
    assert_eq!(results[0].2, "semantic-cold");

    // The namespace is still NOT attached (that's the whole point).
    assert!(
        !b.collections.read().await.contains_key("frozen"),
        "cold search must not attach"
    );

    // Tail write is visible; tombstoned chunk is not.
    let (results, _, _, _) = b
        .search("frozen", &semantic_req(n, 5), &embed)
        .await
        .unwrap();
    assert!(
        results.iter().any(|r| r.0.file_id == format!("f{n}")),
        "post-compaction tail write must be cold-visible"
    );
    let (results, _, _, _) = b
        .search("frozen", &semantic_req(2, 20), &embed)
        .await
        .unwrap();
    assert!(
        results.iter().all(|r| r.0.file_id != "f2"),
        "tombstoned chunk leaked into cold results"
    );

    // Metadata filters apply cold.
    let mut req = semantic_req(target, 10);
    req.filters.insert(
        "kind".to_string(),
        FilterValue::Exact(MetadataValue::String("even".to_string())),
    );
    let (results, _, _, _) = b.search("frozen", &req, &embed).await.unwrap();
    assert!(!results.is_empty());
    for r in &results {
        assert_eq!(
            r.0.metadata.get("kind"),
            Some(&MetadataValue::String("even".to_string()))
        );
    }

    // FTS stays honest: clear error, not silent emptiness.
    let mut req = semantic_req(0, 5);
    req.mode = "fts".to_string();
    req.query = "cold".to_string();
    req.query_vector = None;
    let err = b.search("frozen", &req, &embed).await.unwrap_err();
    assert!(err.to_string().contains("cold"), "{err}");

    let _ = std::fs::remove_dir_all(&dir_b);
}

// Cold serving composes with tenant partitions: a partition namespace is
// cold-served through the same router, tenant isolation intact.
#[tokio::test]
async fn cold_search_composes_with_partitions() {
    let storage = mem_storage();
    let embed = embed_state();
    {
        let dir = unique_data_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let a = CollectionManager::new_with_storage(&dir, storage.clone())
            .await
            .unwrap();
        a.create_collection(
            "mt",
            None,
            Some(DIMS),
            Some(CollectionConfig {
                embed_model: "bge-small".to_string(),
                partition_by: Some("tenant".to_string()),
            }),
        )
        .await
        .unwrap();
        let mut chunks = Vec::new();
        for i in 0..40u64 {
            let mut c = mk_chunk(i, "x");
            c.metadata.insert(
                "tenant".to_string(),
                MetadataValue::String(if i % 2 == 0 { "acme" } else { "globex" }.to_string()),
            );
            chunks.push(c);
        }
        a.ingest("mt", chunks, &embed).await.unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    let dir_b = unique_data_dir();
    let b = cold_manager(&dir_b, storage.clone()).await;
    let mut req = semantic_req(4, 10); // id 4 is acme (even)
    req.filters.insert(
        "tenant".to_string(),
        FilterValue::Exact(MetadataValue::String("acme".to_string())),
    );
    let (results, _, _, _) = b.search("mt", &req, &embed).await.unwrap();
    assert!(!results.is_empty());
    for r in &results {
        assert_eq!(
            r.0.metadata.get("tenant"),
            Some(&MetadataValue::String("acme".to_string())),
            "tenant isolation must hold on the cold path"
        );
    }
    assert!(
        !b.collections.read().await.contains_key("mt--part--acme"),
        "partition must be cold-served, not attached"
    );
    let _ = std::fs::remove_dir_all(&dir_b);
}

// Repeated cold hits promote a background attach; once attached, queries
// take the hot path.
#[tokio::test]
async fn cold_hits_promote_background_attach() {
    let storage = mem_storage();
    let embed = embed_state();
    {
        let dir = unique_data_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let a = CollectionManager::new_with_storage(&dir, storage.clone())
            .await
            .unwrap();
        a.create_collection("warmup", None, Some(DIMS), None)
            .await
            .unwrap();
        a.ingest(
            "warmup",
            (0..20u64).map(|i| mk_chunk(i, "x")).collect(),
            &embed,
        )
        .await
        .unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    let dir_b = unique_data_dir();
    let b = cold_manager(&dir_b, storage.clone()).await;
    b.set_warm_after(2);
    for _ in 0..2 {
        let (results, _, _, _) = b
            .search("warmup", &semantic_req(3, 3), &embed)
            .await
            .unwrap();
        assert!(!results.is_empty());
    }
    // The promotion attach runs in the background; poll briefly.
    let mut attached = false;
    for _ in 0..100 {
        if b.collections.read().await.contains_key("warmup") {
            attached = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(attached, "warm promotion never attached the namespace");
    // Post-promotion searches run the hot path (engine != semantic-cold).
    let (results, _, _, _) = b
        .search("warmup", &semantic_req(3, 3), &embed)
        .await
        .unwrap();
    assert!(!results.is_empty());
    assert_ne!(results[0].2, "semantic-cold");
    let _ = std::fs::remove_dir_all(&dir_b);
}
