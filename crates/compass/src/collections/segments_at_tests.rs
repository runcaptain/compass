// collections/segments_at_tests.rs — extracted test module (was inline in mod.rs).
// Child module of `collections`: `super::*` sees the parent's private items.

use super::*;

fn make_segment(group_id: &str, ts_ms: f64, te_ms: f64) -> DocumentChunk {
    let mut metadata = HashMap::new();
    metadata.insert(
        "timerange_start_ms".to_string(),
        MetadataValue::Float(ts_ms),
    );
    metadata.insert("timerange_end_ms".to_string(), MetadataValue::Float(te_ms));
    DocumentChunk {
        id: 1,
        collection: "test".to_string(),
        file_id: "f1".to_string(),
        chunk_index: 0,
        page: None,
        text: String::new(),
        metadata,
        doc_type: "segment".to_string(),
        parent_id: None,
        group_id: Some(group_id.to_string()),
        embeddings: HashMap::new(),
        embedding: None,
    }
}

/// Make a zero-duration "instant" segment, the convention for sidecar
/// events that have a single timestamp (e.g. standout_timestamps).
fn make_instant(group_id: &str, t_ms: f64) -> DocumentChunk {
    make_segment(group_id, t_ms, t_ms)
}

#[test]
fn point_inside_window() {
    let c = make_segment("a", 100.0, 200.0);
    assert!(segment_in_time_window(&c, Some(150.0), None, None));
}

#[test]
fn point_outside_window() {
    let c = make_segment("a", 100.0, 200.0);
    assert!(!segment_in_time_window(&c, Some(250.0), None, None));
}

#[test]
fn point_boundaries_inclusive() {
    let c = make_segment("a", 100.0, 200.0);
    assert!(segment_in_time_window(&c, Some(100.0), None, None));
    assert!(segment_in_time_window(&c, Some(200.0), None, None));
}

#[test]
fn range_overlap_matches() {
    let c = make_segment("a", 100.0, 200.0);
    assert!(segment_in_time_window(&c, None, Some(180.0), Some(300.0)));
}

#[test]
fn range_no_overlap() {
    let c = make_segment("a", 100.0, 200.0);
    assert!(!segment_in_time_window(&c, None, Some(250.0), Some(400.0)));
}

#[test]
fn range_open_lower_bound() {
    let c = make_segment("a", 100.0, 200.0);
    assert!(segment_in_time_window(&c, None, None, Some(150.0)));
    assert!(!segment_in_time_window(&c, None, None, Some(50.0)));
}

#[test]
fn range_open_upper_bound() {
    let c = make_segment("a", 100.0, 200.0);
    assert!(segment_in_time_window(&c, None, Some(150.0), None));
    assert!(!segment_in_time_window(&c, None, Some(300.0), None));
}

#[test]
fn missing_metadata_with_filter_excludes() {
    let c = DocumentChunk {
        id: 2,
        collection: "test".to_string(),
        file_id: "f2".to_string(),
        chunk_index: 0,
        page: None,
        text: String::new(),
        metadata: HashMap::new(),
        doc_type: "segment".to_string(),
        parent_id: None,
        group_id: Some("a".to_string()),
        embeddings: HashMap::new(),
        embedding: None,
    };
    assert!(!segment_in_time_window(&c, Some(100.0), None, None));
    assert!(!segment_in_time_window(&c, None, Some(0.0), Some(1000.0)));
}

#[test]
fn no_filter_matches_all() {
    let with_meta = make_segment("a", 100.0, 200.0);
    assert!(segment_in_time_window(&with_meta, None, None, None));

    let without_meta = DocumentChunk {
        id: 3,
        collection: "test".to_string(),
        file_id: "f3".to_string(),
        chunk_index: 0,
        page: None,
        text: String::new(),
        metadata: HashMap::new(),
        doc_type: "segment".to_string(),
        parent_id: None,
        group_id: Some("a".to_string()),
        embeddings: HashMap::new(),
        embedding: None,
    };
    assert!(segment_in_time_window(&without_meta, None, None, None));
}

// When both `time_ms` and `time_start_ms`/`time_end_ms` are provided,
// `time_ms` wins. Documented in the segments.rs handler comment; this
// test asserts it.
#[test]
fn point_lookup_takes_precedence_over_range() {
    let c = make_segment("a", 100.0, 200.0);
    // Point=150 is inside [100, 200], but the range [300, 400] is outside.
    // If `time_ms` correctly takes precedence, this must return true.
    assert!(segment_in_time_window(
        &c,
        Some(150.0),
        Some(300.0),
        Some(400.0)
    ));
    // Point=250 is outside, but the range [100, 300] would match.
    // If `time_ms` correctly takes precedence, this must return false.
    assert!(!segment_in_time_window(
        &c,
        Some(250.0),
        Some(100.0),
        Some(300.0)
    ));
}

// Instants (zero-duration events like a standout_timestamp) match a
// point query at their exact timestamp and any range that overlaps it.
// Critical for ingesting sidecar fields like
// `gemini.response.standout_timestamps[]` which only carry a single ms.
#[test]
fn instant_matches_exact_point_query() {
    let c = make_instant("a", 5200.0);
    assert!(segment_in_time_window(&c, Some(5200.0), None, None));
    assert!(!segment_in_time_window(&c, Some(5199.0), None, None));
    assert!(!segment_in_time_window(&c, Some(5201.0), None, None));
}

#[test]
fn instant_matches_overlapping_range_query() {
    let c = make_instant("a", 5200.0);
    assert!(segment_in_time_window(&c, None, Some(5000.0), Some(6000.0)));
    assert!(segment_in_time_window(&c, None, Some(5200.0), Some(5200.0)));
    assert!(!segment_in_time_window(
        &c,
        None,
        Some(5201.0),
        Some(6000.0)
    ));
}
