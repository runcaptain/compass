// collections/partition_cloud_tests.rs — Phase 6 cloud-mode coverage: writer
// routing, cross-node partition discovery, and cold rebuild. Uses the
// in-memory object_store backend (same code path as real S3).

use super::*;
use crate::embed::EmbedState;
use crate::storage::object_store_backend::ObjectStoreBackend;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

fn unique_data_dir() -> std::path::PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "compass-partition-cloud-{}-{}",
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

fn tenant_chunk(tenant: &str, file: &str, text: &str, vec: [f32; 4]) -> IngestChunk {
    let mut metadata = HashMap::new();
    metadata.insert(
        "tenant".to_string(),
        MetadataValue::String(tenant.to_string()),
    );
    let mut embeddings = HashMap::new();
    embeddings.insert("default".to_string(), vec.to_vec());
    IngestChunk {
        client_id: None,
        file_id: file.to_string(),
        chunk_index: 0,
        page: None,
        text: text.to_string(),
        metadata,
        doc_type: "chunk".to_string(),
        parent_id: None,
        parent_ref: None,
        group_id: None,
        embeddings,
        embedding: None,
    }
}

fn partitioned_config() -> Option<CollectionConfig> {
    Some(CollectionConfig {
        embed_model: "bge-small".to_string(),
        partition_by: Some("tenant".to_string()),
    })
}

fn search_req(query: &str, tenant: &str) -> SearchRequest {
    let mut filters = HashMap::new();
    filters.insert(
        "tenant".to_string(),
        FilterValue::Exact(MetadataValue::String(tenant.to_string())),
    );
    SearchRequest {
        query: query.to_string(),
        mode: "fts".to_string(),
        vector_space: None,
        top_k: 10,
        query_vector: None,
        filters,
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

// A stateless writer routes partitioned ingest into per-partition namespaces
// it bootstraps itself; a serving node that booted BEFORE those partitions
// existed attaches them on demand and serves the data.
#[tokio::test]
async fn writer_partitioned_ingest_visible_on_serving_node() {
    let storage = mem_storage();
    let embed = embed_state();

    // Serving node boots first and creates the (empty) partitioned parent.
    let serve_dir = unique_data_dir();
    std::fs::create_dir_all(&serve_dir).unwrap();
    let serving = CollectionManager::new_with_storage(&serve_dir, storage.clone())
        .await
        .unwrap();
    serving
        .create_collection("wp", None, Some(4), partitioned_config())
        .await
        .unwrap();

    // Writer node ingests for two tenants the serving node has never seen.
    let writer_dir = unique_data_dir();
    std::fs::create_dir_all(&writer_dir).unwrap();
    let writer = CollectionManager::new_with_storage_opts(
        &writer_dir,
        storage.clone(),
        NodeRole::Writer,
        false,
        usize::MAX,
        0,
    )
    .await
    .unwrap();
    let (n, id_map, _) = writer
        .ingest(
            "wp",
            vec![
                {
                    let mut c =
                        tenant_chunk("acme", "a1", "durable acme fact", [0.9, 0.1, 0.0, 0.0]);
                    c.client_id = Some("a1".to_string());
                    c
                },
                {
                    let mut c =
                        tenant_chunk("globex", "g1", "durable globex fact", [0.1, 0.9, 0.0, 0.0]);
                    c.client_id = Some("g1".to_string());
                    c
                },
            ],
            &embed,
        )
        .await
        .unwrap();
    assert_eq!(n, 2);
    let ids: std::collections::HashSet<u64> = id_map.values().copied().collect();
    assert_eq!(ids.len(), 2, "writer ids must be collection-unique");

    // The serving node attaches the new partitions on first query.
    let (results, _, _, _) = serving
        .search("wp", &search_req("durable", "acme"), &embed)
        .await
        .unwrap();
    assert_eq!(results.len(), 1, "acme partition attaches on demand");
    assert_eq!(results[0].0.file_id, "a1");
    let (results, _, _, _) = serving
        .search("wp", &search_req("durable", "globex"), &embed)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0.file_id, "g1");

    // Writer refuses partitioned operations that would black-hole data.
    let err = writer.delete_chunks("wp", &[0]).await.unwrap_err();
    assert!(
        err.to_string()
            .contains("route the delete through a serving node"),
        "{err}"
    );
    let err = writer
        .create_relations(
            "wp",
            vec![CreateRelation {
                source_chunk_id: 0,
                target_chunk_id: 1,
                target_document_id: None,
                relation_type: "cites".to_string(),
                metadata: HashMap::new(),
            }],
        )
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("not supported on a partitioned"),
        "{err}"
    );

    let _ = std::fs::remove_dir_all(&serve_dir);
    let _ = std::fs::remove_dir_all(&writer_dir);
}

// A brand-new node with an empty disk recovers a partitioned collection —
// parent config, every partition's data, and the shared id allocator — from
// the bucket alone.
#[tokio::test]
async fn partitioned_collection_cold_rebuild_from_bucket() {
    let storage = mem_storage();
    let embed = embed_state();

    {
        let dir_a = unique_data_dir();
        std::fs::create_dir_all(&dir_a).unwrap();
        let a = CollectionManager::new_with_storage(&dir_a, storage.clone())
            .await
            .unwrap();
        a.create_collection("cold", None, Some(4), partitioned_config())
            .await
            .unwrap();
        a.ingest(
            "cold",
            vec![
                tenant_chunk("acme", "a1", "cold acme", [0.9, 0.1, 0.0, 0.0]),
                tenant_chunk("globex", "g1", "cold globex", [0.1, 0.9, 0.0, 0.0]),
            ],
            &embed,
        )
        .await
        .unwrap();
        let _ = std::fs::remove_dir_all(&dir_a);
    } // node A gone, local disk gone; only the bucket remains.

    let dir_b = unique_data_dir();
    std::fs::create_dir_all(&dir_b).unwrap();
    let b = CollectionManager::new_with_storage(&dir_b, storage.clone())
        .await
        .unwrap();

    // Parent is listed (partitions hidden), routing metadata survived.
    let listed = b.list_collections().await;
    let names: Vec<&str> = listed.iter().map(|c| c.name.as_str()).collect();
    assert!(names.contains(&"cold"), "{names:?}");
    assert!(!names.iter().any(|n| n.contains(partitions::PART_SEP)));

    let (results, _, _, _) = b
        .search("cold", &search_req("cold", "acme"), &embed)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0.file_id, "a1");

    // New ingest keeps minting collection-unique ids from the recovered
    // allocator (never reuses the pre-rebuild ids).
    let (_, id_map, _) = b
        .ingest(
            "cold",
            vec![{
                let mut c = tenant_chunk("acme", "a2", "post-rebuild", [0.7, 0.3, 0.0, 0.0]);
                c.client_id = Some("a2".to_string());
                c
            }],
            &embed,
        )
        .await
        .unwrap();
    let new_id = *id_map.values().next().unwrap();
    assert!(
        new_id >= 2,
        "rebuilt node must not reuse ids (got {new_id})"
    );

    // Cascade delete purges parent + partitions from the bucket.
    b.delete_collection("cold").await.unwrap();
    let remaining = crate::storage::lsm::list_namespaces(storage.as_ref())
        .await
        .unwrap();
    assert!(
        remaining.iter().all(|n| !n.starts_with("cold")),
        "bucket still has: {remaining:?}"
    );

    let _ = std::fs::remove_dir_all(&dir_b);
}
