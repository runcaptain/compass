// collections/partition_tests.rs — tenant-partitioned collections (Phase 6).
// Child module of `collections`: `super::*` sees the parent's private items.

use super::*;
use crate::embed::EmbedState;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

fn unique_data_dir() -> std::path::PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "compass-partition-{}-{}",
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

fn tenant_filter(tenant: &str) -> HashMap<String, FilterValue> {
    let mut f = HashMap::new();
    f.insert(
        "tenant".to_string(),
        FilterValue::Exact(MetadataValue::String(tenant.to_string())),
    );
    f
}

fn search_req(query: &str, vec: Option<[f32; 4]>, tenant: &str) -> SearchRequest {
    SearchRequest {
        query: query.to_string(),
        mode: if vec.is_some() {
            "semantic".to_string()
        } else {
            "fts".to_string()
        },
        vector_space: None,
        top_k: 10,
        query_vector: vec.map(|v| v.to_vec()),
        filters: tenant_filter(tenant),
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

async fn local_manager(data_dir: &std::path::Path) -> std::sync::Arc<CollectionManager> {
    std::fs::create_dir_all(data_dir).unwrap();
    CollectionManager::new(data_dir).await.unwrap()
}

// ── Local mode ──────────────────────────────────────────────────────────

#[tokio::test]
async fn partitioned_ingest_routes_and_isolates() {
    let data_dir = unique_data_dir();
    let manager = local_manager(&data_dir).await;
    let embed = embed_state();
    manager
        .create_collection("multi", None, Some(4), partitioned_config())
        .await
        .unwrap();

    let (n, _, _) = manager
        .ingest(
            "multi",
            vec![
                tenant_chunk("acme", "a1", "acme secret report", [0.9, 0.1, 0.0, 0.0]),
                tenant_chunk("acme", "a2", "acme quarterly numbers", [0.8, 0.2, 0.0, 0.0]),
                tenant_chunk("globex", "g1", "globex secret memo", [0.1, 0.9, 0.0, 0.0]),
            ],
            &embed,
        )
        .await
        .unwrap();
    assert_eq!(n, 3);

    // Tenant-scoped search sees ONLY that tenant's chunks — even for a query
    // term both tenants share.
    let (results, _, _, _) = manager
        .search("multi", &search_req("secret", None, "acme"), &embed)
        .await
        .unwrap();
    assert_eq!(
        results.len(),
        1,
        "acme must see exactly its own 'secret' hit"
    );
    assert_eq!(results[0].0.file_id, "a1");

    let (results, _, _, _) = manager
        .search("multi", &search_req("secret", None, "globex"), &embed)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0.file_id, "g1");

    // A tenant that never ingested is empty, not an error.
    let (results, total, _, _) = manager
        .search("multi", &search_req("secret", None, "initech"), &embed)
        .await
        .unwrap();
    assert!(results.is_empty());
    assert_eq!(total, 0);

    // Unfiltered search on a partitioned collection is a clear error.
    let mut req = search_req("secret", None, "acme");
    req.filters.clear();
    let err = manager.search("multi", &req, &embed).await.unwrap_err();
    assert!(err.to_string().contains("partitioned by 'tenant'"), "{err}");

    let _ = std::fs::remove_dir_all(&data_dir);
}

#[tokio::test]
async fn partition_ids_are_collection_unique() {
    let data_dir = unique_data_dir();
    let manager = local_manager(&data_dir).await;
    let embed = embed_state();
    manager
        .create_collection("uniq", None, Some(4), partitioned_config())
        .await
        .unwrap();

    // Interleave ingests across tenants; every assigned id must be distinct
    // (partitions share the parent's id allocator).
    let mut all_ids = Vec::new();
    for round in 0..3 {
        for tenant in ["t-a", "t-b", "t-c"] {
            let (_, id_map, _) = manager
                .ingest(
                    "uniq",
                    vec![{
                        let mut c = tenant_chunk(
                            tenant,
                            &format!("{tenant}-{round}"),
                            "payload",
                            [0.5, 0.5, 0.0, 0.0],
                        );
                        c.client_id = Some(format!("{tenant}-{round}"));
                        c
                    }],
                    &embed,
                )
                .await
                .unwrap();
            all_ids.extend(id_map.values().copied());
        }
    }
    assert_eq!(all_ids.len(), 9);
    let unique: std::collections::HashSet<u64> = all_ids.iter().copied().collect();
    assert_eq!(unique.len(), 9, "ids must never collide across partitions");

    let _ = std::fs::remove_dir_all(&data_dir);
}

#[tokio::test]
async fn partitioned_delete_routes_by_filter_and_rejects_bare_ids() {
    let data_dir = unique_data_dir();
    let manager = local_manager(&data_dir).await;
    let embed = embed_state();
    manager
        .create_collection("deltest", None, Some(4), partitioned_config())
        .await
        .unwrap();
    manager
        .ingest(
            "deltest",
            vec![
                tenant_chunk("acme", "a1", "doomed", [0.9, 0.1, 0.0, 0.0]),
                tenant_chunk("globex", "g1", "doomed", [0.1, 0.9, 0.0, 0.0]),
            ],
            &embed,
        )
        .await
        .unwrap();

    // Bare ids are ambiguous across partitions — rejected with guidance.
    let err = manager.delete_chunks("deltest", &[0]).await.unwrap_err();
    assert!(err.to_string().contains("delete via filters"), "{err}");

    // Filter-scoped delete removes acme's chunk only.
    let (n, _) = manager
        .delete_by_filter("deltest", &tenant_filter("acme"))
        .await
        .unwrap();
    assert_eq!(n, 1);
    let (results, _, _, _) = manager
        .search("deltest", &search_req("doomed", None, "acme"), &embed)
        .await
        .unwrap();
    assert!(results.is_empty(), "acme's chunk is gone");
    let (results, _, _, _) = manager
        .search("deltest", &search_req("doomed", None, "globex"), &embed)
        .await
        .unwrap();
    assert_eq!(results.len(), 1, "globex is untouched");

    let _ = std::fs::remove_dir_all(&data_dir);
}

#[tokio::test]
async fn partitions_hidden_from_listing_and_cascade_deleted() {
    let data_dir = unique_data_dir();
    let manager = local_manager(&data_dir).await;
    let embed = embed_state();
    manager
        .create_collection("casc", None, Some(4), partitioned_config())
        .await
        .unwrap();
    manager
        .ingest(
            "casc",
            vec![
                tenant_chunk("acme", "a1", "x", [0.9, 0.1, 0.0, 0.0]),
                tenant_chunk("globex", "g1", "y", [0.1, 0.9, 0.0, 0.0]),
            ],
            &embed,
        )
        .await
        .unwrap();

    // Listing shows the parent only — partition namespaces are internal.
    let listed = manager.list_collections().await;
    let names: Vec<&str> = listed.iter().map(|c| c.name.as_str()).collect();
    assert!(names.contains(&"casc"));
    assert!(
        !names.iter().any(|n| n.contains(partitions::PART_SEP)),
        "partition namespaces must not be listed: {names:?}"
    );

    // Deleting the parent removes every partition's data on disk.
    manager.delete_collection("casc").await.unwrap();
    let leftovers: Vec<String> = std::fs::read_dir(&data_dir)
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| e.file_name().to_str().map(String::from))
                .filter(|n| n.starts_with("casc"))
                .collect()
        })
        .unwrap_or_default();
    assert!(
        leftovers.is_empty(),
        "cascade left dirs behind: {leftovers:?}"
    );

    let _ = std::fs::remove_dir_all(&data_dir);
}

#[tokio::test]
async fn partitioned_fences_and_validation() {
    let data_dir = unique_data_dir();
    let manager = local_manager(&data_dir).await;
    let embed = embed_state();
    manager
        .create_collection("fenced", None, Some(4), partitioned_config())
        .await
        .unwrap();
    manager
        .ingest(
            "fenced",
            vec![tenant_chunk("acme", "a1", "x", [0.9, 0.1, 0.0, 0.0])],
            &embed,
        )
        .await
        .unwrap();

    // Reserved separator in user collection names.
    let err = manager
        .create_collection("evil--part--x", None, Some(4), None)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("reserved partition separator"));

    // Chunks missing the partition field are rejected with the field name.
    let mut bad = tenant_chunk("acme", "b", "x", [0.1, 0.1, 0.0, 0.0]);
    bad.metadata.clear();
    let err = manager
        .ingest("fenced", vec![bad], &embed)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("missing partition field 'tenant'"));

    // Unrouted operations fail closed with a clear message.
    let err = manager
        .get_facets("fenced", "", &["tenant".to_string()])
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not supported on a partitioned"));
    let err = manager
        .create_relations(
            "fenced",
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
    assert!(err.to_string().contains("not supported on a partitioned"));
    let err = manager
        .add_vector_space("fenced", "extra", 8, "model")
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not supported on a partitioned"));

    let _ = std::fs::remove_dir_all(&data_dir);
}

#[tokio::test]
async fn partitioned_collection_survives_restart() {
    let data_dir = unique_data_dir();
    let embed = embed_state();
    {
        let manager = local_manager(&data_dir).await;
        manager
            .create_collection("persist", None, Some(4), partitioned_config())
            .await
            .unwrap();
        manager
            .ingest(
                "persist",
                vec![
                    tenant_chunk("acme", "a1", "durable acme", [0.9, 0.1, 0.0, 0.0]),
                    tenant_chunk("globex", "g1", "durable globex", [0.1, 0.9, 0.0, 0.0]),
                ],
                &embed,
            )
            .await
            .unwrap();
    }

    let manager = CollectionManager::new(&data_dir).await.unwrap();
    // Routing metadata survives: tenant-scoped search still works, ids keep
    // minting from the shared allocator without collision.
    let (results, _, _, _) = manager
        .search("persist", &search_req("durable", None, "acme"), &embed)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0.file_id, "a1");
    let (_, id_map, _) = manager
        .ingest(
            "persist",
            vec![{
                let mut c = tenant_chunk("acme", "a2", "post-restart", [0.7, 0.3, 0.0, 0.0]);
                c.client_id = Some("a2".to_string());
                c
            }],
            &embed,
        )
        .await
        .unwrap();
    let new_id = *id_map.values().next().unwrap();
    assert!(new_id >= 2, "restart must not reuse ids (got {new_id})");

    let _ = std::fs::remove_dir_all(&data_dir);
}
