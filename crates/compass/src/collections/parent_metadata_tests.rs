// collections/parent_metadata_tests.rs — extracted test module (was inline in mod.rs).
// Child module of `collections`: `super::*` sees the parent's private items.

use super::*;

fn segment(id: u64, parent_id: Option<u64>) -> DocumentChunk {
    DocumentChunk {
        id,
        collection: "test".to_string(),
        file_id: format!("f{}", id),
        chunk_index: 0,
        page: None,
        text: String::new(),
        metadata: HashMap::new(),
        doc_type: "segment".to_string(),
        parent_id,
        group_id: None,
        embeddings: HashMap::new(),
        embedding: None,
    }
}

fn source_with_meta(id: u64, key: &str, val: &str) -> DocumentChunk {
    let mut metadata = HashMap::new();
    metadata.insert(key.to_string(), MetadataValue::String(val.to_string()));
    DocumentChunk {
        id,
        collection: "test".to_string(),
        file_id: format!("f{}", id),
        chunk_index: 0,
        page: None,
        text: String::new(),
        metadata,
        doc_type: "source".to_string(),
        parent_id: None,
        group_id: None,
        embeddings: HashMap::new(),
        embedding: None,
    }
}

fn into_map(chunks: Vec<DocumentChunk>) -> ChunkCache {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "compass_pmc_{}_{}.redb",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_file(&p);
    let cache = ChunkCache::new(ChunkStore::open(&p).unwrap());
    let batch: Vec<(u64, DocumentChunk)> = chunks.into_iter().map(|c| (c.id, c)).collect();
    cache.insert_batch(&batch).unwrap();
    cache
}

#[test]
fn segment_with_parent_gets_metadata() {
    let chunks = into_map(vec![
        source_with_meta(1, "title", "Keynote"),
        segment(2, Some(1)),
    ]);
    let cache = build_parent_metadata_cache(&[2], &chunks);
    let meta = parent_metadata_for(&chunks.get(2).unwrap().unwrap(), &cache);
    assert_eq!(
        meta.unwrap().get("title"),
        Some(&MetadataValue::String("Keynote".to_string()))
    );
}

#[test]
fn source_hit_gets_none() {
    let chunks = into_map(vec![source_with_meta(1, "title", "Keynote")]);
    let cache = build_parent_metadata_cache(&[1], &chunks);
    let meta = parent_metadata_for(&chunks.get(1).unwrap().unwrap(), &cache);
    assert!(meta.is_none());
}

#[test]
fn segment_without_parent_gets_none() {
    let chunks = into_map(vec![segment(2, None)]);
    let cache = build_parent_metadata_cache(&[2], &chunks);
    let meta = parent_metadata_for(&chunks.get(2).unwrap().unwrap(), &cache);
    assert!(meta.is_none());
}

#[test]
fn dedup_one_lookup_per_unique_parent() {
    // Three segments, all pointing at parent_id=10. The cache should
    // contain exactly one entry (for pid=10), proving the dedup.
    let chunks = into_map(vec![
        source_with_meta(10, "source_id", "src-001"),
        segment(11, Some(10)),
        segment(12, Some(10)),
        segment(13, Some(10)),
    ]);
    let cache = build_parent_metadata_cache(&[11, 12, 13], &chunks);
    assert_eq!(
        cache.len(),
        1,
        "expected one cache entry for the shared parent"
    );
    assert!(cache.contains_key(&10));
    // All three segments resolve to the same parent metadata.
    for cid in [11, 12, 13] {
        let meta = parent_metadata_for(&chunks.get(cid).unwrap().unwrap(), &cache);
        assert_eq!(
            meta.unwrap().get("source_id"),
            Some(&MetadataValue::String("src-001".to_string()))
        );
    }
}

#[test]
fn orphan_segment_yields_none() {
    // parent_id=99 not in chunks. The cache must NOT contain pid=99,
    // and parent_metadata_for must return None. This distinguishes
    // "parent exists with empty metadata" (Some({})) from "parent
    // doesn't exist" (None).
    let chunks = into_map(vec![segment(5, Some(99))]);
    let cache = build_parent_metadata_cache(&[5], &chunks);
    assert!(!cache.contains_key(&99), "orphan parent must not be cached");
    let meta = parent_metadata_for(&chunks.get(5).unwrap().unwrap(), &cache);
    assert!(meta.is_none(), "orphan segment must yield None");
}

#[test]
fn parent_exists_with_empty_metadata_yields_some_empty() {
    // Parent chunk exists but has no metadata fields. Must return Some({})
    // so callers can distinguish from the orphan case (None).
    let parent_no_meta = DocumentChunk {
        id: 20,
        collection: "test".to_string(),
        file_id: "f20".to_string(),
        chunk_index: 0,
        page: None,
        text: String::new(),
        metadata: HashMap::new(),
        doc_type: "source".to_string(),
        parent_id: None,
        group_id: None,
        embeddings: HashMap::new(),
        embedding: None,
    };
    let chunks = into_map(vec![parent_no_meta, segment(21, Some(20))]);
    let cache = build_parent_metadata_cache(&[21], &chunks);
    let meta = parent_metadata_for(&chunks.get(21).unwrap().unwrap(), &cache);
    assert!(meta.is_some());
    assert!(meta.unwrap().is_empty());
}

#[test]
fn parent_metadata_for_cache_miss_returns_none() {
    // Defensive: if the cache was built with a different set of IDs than
    // the one we're looking up, the function must return None (not panic,
    // not return stale data). Catches regressions where someone "optimizes"
    // parent_metadata_for to assume the cache is always complete.
    let parent = source_with_meta(1, "title", "Keynote");
    let seg = segment(2, Some(1));
    let chunks = into_map(vec![parent, seg]);
    // Build cache against an empty candidate list, then look up segment 2.
    let cache = build_parent_metadata_cache(&[], &chunks);
    assert!(cache.is_empty());
    let meta = parent_metadata_for(&chunks.get(2).unwrap().unwrap(), &cache);
    assert!(meta.is_none());
}
