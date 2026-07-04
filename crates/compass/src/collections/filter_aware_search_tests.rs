// collections/filter_aware_search_tests.rs — extracted test module (was inline in mod.rs).
// Child module of `collections`: `super::*` sees the parent's private items.

//! End-to-end test of filter-aware /search + /explain (a follow-up).
//!
//! Builds a real CollectionManager, ingests chunks with caller-provided
//! embeddings (skipping the in-process BGE model), runs filtered hybrid
//! search, and asserts:
//!   1. All hits respect the filter (filter-aware path, not post-filter).
//!   2. The /explain field is populated when requested and absent when not.
//!   3. Filter selectivity is reported correctly.

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
        "compass-filter-search-test-{}-{}-{}",
        std::process::id(),
        nanos,
        N.fetch_add(1, Ordering::SeqCst)
    ))
}

fn embed_state() -> EmbedState {
    EmbedState {
        bge: None,
        distilled: None,
    }
}

/// Deterministic 4-dim unit vector seeded from an integer.
fn pseudo_vec(seed: u64) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut v = Vec::with_capacity(4);
    for _ in 0..4 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let f = (state >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0;
        v.push(f);
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn ingest_with(org: &str, idx: u32) -> IngestChunk {
    let mut metadata = HashMap::new();
    metadata.insert("org_id".to_string(), MetadataValue::String(org.to_string()));
    metadata.insert(
        "created_at".to_string(),
        MetadataValue::Int(1_700_000_000 + idx as i64),
    );
    let mut embeddings = HashMap::new();
    embeddings.insert("default".to_string(), pseudo_vec(idx as u64 + 1));
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
async fn filter_aware_search_returns_only_matching_chunks() {
    let data_dir = unique_data_dir();
    std::fs::create_dir_all(&data_dir).unwrap();
    let embed = embed_state();
    let manager = CollectionManager::new(&data_dir).await.unwrap();
    manager
        .create_collection("filter-search", None, Some(4), None)
        .await
        .unwrap();

    // 100 chunks: 20 from "acme", 80 from "widgets".
    let mut chunks = Vec::new();
    for i in 0..100u32 {
        let org = if i % 5 == 0 { "acme" } else { "widgets" };
        chunks.push(ingest_with(org, i));
    }
    manager
        .ingest("filter-search", chunks, &embed)
        .await
        .unwrap();

    // Search with filter org_id=acme. ALL hits must come from acme.
    let mut filters = HashMap::new();
    filters.insert(
        "org_id".to_string(),
        FilterValue::Exact(MetadataValue::String("acme".into())),
    );
    let req = SearchRequest {
        query: "chunk".to_string(),
        mode: "hybrid".to_string(),
        vector_space: None,
        top_k: 10,
        query_vector: Some(pseudo_vec(99_999)),
        filters,
        score_weights: None,
        recency: None,
        recency_preset: None,
        recency_field: None,
        boosts: Vec::new(),
        relationship_boost: None,
        explain: true,
        include_relations: false,
        relation_types: None,
        relation_direction: RelationDirection::Outgoing,
        min_seq: None,
    };
    let (hits, total, _took_us, explain) =
        manager.search("filter-search", &req, &embed).await.unwrap();

    assert!(!hits.is_empty(), "search returned no hits");
    for (chunk, _, _, _, _) in &hits {
        assert_eq!(
            chunk.metadata.get("org_id"),
            Some(&MetadataValue::String("acme".into())),
            "all hits must satisfy the filter; got chunk {} with org_id {:?}",
            chunk.id,
            chunk.metadata.get("org_id")
        );
    }
    assert!(
        total <= 20,
        "no more than 20 hits possible at 20% selectivity"
    );

    // /explain should be populated.
    let explain = explain.expect("explain plan requested but not returned");
    assert_eq!(explain.filter.eligible_count, 20);
    assert_eq!(explain.filter.universe_count, 100);
    assert!((explain.filter.selectivity - 0.20).abs() < 1e-9);
    assert!(
        matches!(explain.ann.engine.as_str(), "hnsw" | "brute_force"),
        "ann engine reported as {}",
        explain.ann.engine
    );

    let _ = std::fs::remove_dir_all(&data_dir);
}

#[tokio::test]
async fn explain_absent_when_not_requested() {
    let data_dir = unique_data_dir();
    std::fs::create_dir_all(&data_dir).unwrap();
    let embed = embed_state();
    let manager = CollectionManager::new(&data_dir).await.unwrap();
    manager
        .create_collection("no-explain", None, Some(4), None)
        .await
        .unwrap();
    manager
        .ingest("no-explain", vec![ingest_with("acme", 0)], &embed)
        .await
        .unwrap();

    let req = SearchRequest {
        query: "chunk".to_string(),
        mode: "semantic".to_string(),
        vector_space: None,
        top_k: 1,
        query_vector: Some(pseudo_vec(42)),
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
    let (_hits, _total, _took, explain) = manager.search("no-explain", &req, &embed).await.unwrap();
    assert!(explain.is_none(), "explain must be None when not requested");

    let _ = std::fs::remove_dir_all(&data_dir);
}

#[tokio::test]
async fn relations_crud_and_search_enrichment() {
    let data_dir = unique_data_dir();
    std::fs::create_dir_all(&data_dir).unwrap();
    let embed = embed_state();
    let manager = CollectionManager::new(&data_dir).await.unwrap();
    manager
        .create_collection("rel-search", None, Some(4), None)
        .await
        .unwrap();

    // Ingest 5 chunks -> ids 0..5 in order.
    let chunks: Vec<_> = (0..5u32).map(|i| ingest_with("acme", i)).collect();
    manager.ingest("rel-search", chunks, &embed).await.unwrap();

    // Create relations: 0 --cites--> 1, 0 --cites--> 2, 3 --supersedes--> 0.
    let created = manager
        .create_relations(
            "rel-search",
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
                    relation_type: "cites".into(),
                    metadata: HashMap::new(),
                },
                CreateRelation {
                    source_chunk_id: 3,
                    target_chunk_id: 0,
                    target_document_id: None,
                    relation_type: "supersedes".into(),
                    metadata: HashMap::new(),
                },
            ],
        )
        .await
        .unwrap();
    assert_eq!(created.len(), 3);
    assert!(created.iter().all(|r| !r.relation_id.is_empty()));
    assert!(created.iter().all(|r| r.target_status == "found"));

    // Self-relation is rejected.
    let bad = manager
        .create_relations(
            "rel-search",
            vec![CreateRelation {
                source_chunk_id: 1,
                target_chunk_id: 1,
                target_document_id: None,
                relation_type: "cites".into(),
                metadata: HashMap::new(),
            }],
        )
        .await;
    assert!(bad.is_err());

    // Direction filters.
    let out = manager
        .get_chunk_relations("rel-search", 0, RelationDirection::Outgoing, None)
        .await
        .unwrap();
    assert_eq!(out.len(), 2);
    let inc = manager
        .get_chunk_relations("rel-search", 0, RelationDirection::Incoming, None)
        .await
        .unwrap();
    assert_eq!(inc.len(), 1);
    assert_eq!(inc[0].relation_type, "supersedes");

    // Type filter.
    let cites = manager
        .get_chunk_relations(
            "rel-search",
            0,
            RelationDirection::Both,
            Some(&["cites".to_string()]),
        )
        .await
        .unwrap();
    assert_eq!(cites.len(), 2);

    let base_req = |include: bool| SearchRequest {
        query: "chunk".to_string(),
        mode: "semantic".to_string(),
        vector_space: None,
        top_k: 10,
        query_vector: Some(pseudo_vec(7)),
        filters: HashMap::new(),
        score_weights: None,
        recency: None,
        recency_preset: None,
        recency_field: None,
        boosts: Vec::new(),
        relationship_boost: None,
        explain: false,
        include_relations: include,
        relation_types: None,
        relation_direction: RelationDirection::Outgoing,
        min_seq: None,
    };

    // Without include_relations -> hits carry None.
    let (hits_off, _, _, _) = manager
        .search("rel-search", &base_req(false), &embed)
        .await
        .unwrap();
    assert!(hits_off.iter().all(|(_, _, _, _, rels)| rels.is_none()));

    // With include_relations -> chunk 0's hit carries its 2 outgoing cites.
    let (hits_on, _, _, _) = manager
        .search("rel-search", &base_req(true), &embed)
        .await
        .unwrap();
    let chunk0 = hits_on
        .iter()
        .find(|(c, _, _, _, _)| c.id == 0)
        .expect("chunk 0 in results");
    let rels = chunk0.4.as_ref().expect("Some(relations) when requested");
    assert_eq!(rels.len(), 2, "chunk 0 has 2 outgoing cites");
    assert!(rels.iter().all(|r| r.relation_type == "cites"));
    assert!(rels.iter().all(|r| r.target_status == "found"));

    // Delete one relation; outgoing from 0 drops to 1.
    let rid = &created[0].relation_id;
    assert!(manager.delete_relation("rel-search", rid).await.unwrap());
    let out2 = manager
        .get_chunk_relations("rel-search", 0, RelationDirection::Outgoing, None)
        .await
        .unwrap();
    assert_eq!(out2.len(), 1);

    let _ = std::fs::remove_dir_all(&data_dir);
}

#[tokio::test]
async fn delete_removes_from_search_and_survives_restart() {
    let data_dir = unique_data_dir();
    std::fs::create_dir_all(&data_dir).unwrap();
    let embed = embed_state();

    {
        let manager = CollectionManager::new(&data_dir).await.unwrap();
        manager
            .create_collection("del", None, Some(4), None)
            .await
            .unwrap();
        // Ingest 10 chunks (ids 0..10), org=acme.
        let chunks: Vec<_> = (0..10u32).map(|i| ingest_with("acme", i)).collect();
        manager.ingest("del", chunks, &embed).await.unwrap();

        // Delete chunk id 3 by id.
        let (n, _) = manager.delete_chunks("del", &[3]).await.unwrap();
        assert_eq!(n, 1);
        // Re-deleting is a no-op.
        assert_eq!(manager.delete_chunks("del", &[3]).await.unwrap().0, 0);

        // A search must never return the deleted id.
        let req = SearchRequest {
            query: "chunk".to_string(),
            mode: "semantic".to_string(),
            vector_space: None,
            top_k: 20,
            query_vector: Some(pseudo_vec(4)),
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
        let (hits, _, _, _) = manager.search("del", &req, &embed).await.unwrap();
        assert!(
            hits.iter().all(|(c, _, _, _, _)| c.id != 3),
            "deleted chunk must not appear in results"
        );

        // Delete-by-filter: delete everything with file_id f5 (chunk 5).
        let mut filters = HashMap::new();
        filters.insert(
            "file_id".to_string(),
            FilterValue::Exact(MetadataValue::String("f5".into())),
        );
        // ingest_with doesn't set file_id in metadata, so use a metadata field.
        // org_id=acme matches all remaining -> delete the rest via a scan.
        let mut org_filter = HashMap::new();
        org_filter.insert(
            "org_id".to_string(),
            FilterValue::Exact(MetadataValue::String("acme".into())),
        );
        let deleted = manager.delete_by_filter("del", &org_filter).await.unwrap();
        // 10 ingested - 1 already deleted (id 3) = 9 remaining deleted now.
        assert_eq!(deleted.0, 9);
        let _ = filters;
    }

    // Restart: tombstones must persist. Reopen the manager over the same dir.
    {
        let manager = CollectionManager::new(&data_dir).await.unwrap();
        let req = SearchRequest {
            query: "chunk".to_string(),
            mode: "semantic".to_string(),
            vector_space: None,
            top_k: 20,
            query_vector: Some(pseudo_vec(4)),
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
        let (hits, _, _, _) = manager.search("del", &req, &embed).await.unwrap();
        assert!(
            hits.is_empty(),
            "all chunks deleted; none should survive restart, got {}",
            hits.len()
        );
    }

    let _ = std::fs::remove_dir_all(&data_dir);
}

// F6 coverage: deleting a chunk that participates in relations must prune
// those edges (both endpoints), not leave dangling references.
#[tokio::test]
async fn delete_chunk_prunes_its_relations() {
    let data_dir = unique_data_dir();
    std::fs::create_dir_all(&data_dir).unwrap();
    let embed = embed_state();
    let manager = CollectionManager::new(&data_dir).await.unwrap();
    manager
        .create_collection("delrel", None, Some(4), None)
        .await
        .unwrap();
    let chunks: Vec<_> = (0..3u32).map(|i| ingest_with("acme", i)).collect();
    manager.ingest("delrel", chunks, &embed).await.unwrap();

    // 0 -> 1, 2 -> 0 (chunk 0 is both a source and a target).
    manager
        .create_relations(
            "delrel",
            vec![
                CreateRelation {
                    source_chunk_id: 0,
                    target_chunk_id: 1,
                    target_document_id: None,
                    relation_type: "cites".into(),
                    metadata: HashMap::new(),
                },
                CreateRelation {
                    source_chunk_id: 2,
                    target_chunk_id: 0,
                    target_document_id: None,
                    relation_type: "cites".into(),
                    metadata: HashMap::new(),
                },
            ],
        )
        .await
        .unwrap();

    // Delete chunk 0 — both edges (as source and as target) must be pruned.
    manager.delete_chunks("delrel", &[0]).await.unwrap();

    assert!(
        manager
            .get_chunk_relations("delrel", 0, RelationDirection::Both, None)
            .await
            .unwrap()
            .is_empty(),
        "deleted chunk's own edges gone"
    );
    // Chunk 2's outgoing edge (to deleted 0) must also be gone.
    assert!(
        manager
            .get_chunk_relations("delrel", 2, RelationDirection::Outgoing, None)
            .await
            .unwrap()
            .is_empty(),
        "edge pointing AT the deleted chunk must be pruned"
    );
    let _ = std::fs::remove_dir_all(&data_dir);
}

// A stale tombstone must not suppress a NEWLY-ingested chunk. Since next_id
// is a monotonic high-water mark, re-ingest gets a fresh id that was never
// tombstoned, so it's fully searchable.
#[tokio::test]
async fn delete_then_reingest_new_chunk_is_searchable() {
    let data_dir = unique_data_dir();
    std::fs::create_dir_all(&data_dir).unwrap();
    let embed = embed_state();
    let manager = CollectionManager::new(&data_dir).await.unwrap();
    manager
        .create_collection("reing", None, Some(4), None)
        .await
        .unwrap();
    manager
        .ingest("reing", vec![ingest_with("acme", 0)], &embed)
        .await
        .unwrap();
    manager.delete_chunks("reing", &[0]).await.unwrap();

    // Re-ingest: gets id 1 (next_id advanced), NOT the tombstoned id 0.
    manager
        .ingest("reing", vec![ingest_with("acme", 9)], &embed)
        .await
        .unwrap();

    let req = SearchRequest {
        query: "chunk".to_string(),
        mode: "semantic".to_string(),
        vector_space: None,
        top_k: 10,
        query_vector: Some(pseudo_vec(10)),
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
    let (hits, _, _, _) = manager.search("reing", &req, &embed).await.unwrap();
    assert_eq!(hits.len(), 1, "the re-ingested chunk must be searchable");
    assert_eq!(hits[0].0.id, 1, "re-ingest got a fresh (untombstoned) id");
    let _ = std::fs::remove_dir_all(&data_dir);
}
