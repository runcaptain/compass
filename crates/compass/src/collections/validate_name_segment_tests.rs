// collections/validate_name_segment_tests.rs — extracted test module (was inline in mod.rs).
// Child module of `collections`: `super::*` sees the parent's private items.

use super::validate_name_segment;

#[test]
fn accepts_simple_names() {
    assert!(validate_name_segment("my-collection", "Collection").is_ok());
    assert!(validate_name_segment("harrier", "Vector space").is_ok());
    assert!(validate_name_segment("qwen3-vl", "Vector space").is_ok());
    assert!(validate_name_segment("a", "Collection").is_ok());
}

#[test]
fn rejects_empty() {
    let err = validate_name_segment("", "Vector space").expect_err("empty name should error");
    assert!(err.to_string().contains("Vector space"));
}

#[test]
fn rejects_path_traversal() {
    // The whole reason this validator exists: a vector space name flows
    // into on-disk paths like `<vectors_dir>/<name>.bin`. A `../` segment
    // must never be accepted.
    for bad in [
        "../etc/passwd",
        "..",
        "foo/bar",
        "foo\\bar",
        "/abs",
        "name with space",
        "name.with.dot",
        "name_with_underscore", // hyphens only, no underscores
        "tab\there",
        "name\nwith\nnewline",
    ] {
        assert!(
            validate_name_segment(bad, "Vector space").is_err(),
            "validator must reject {bad:?}"
        );
    }
}

#[test]
fn rejects_unicode_lookalikes() {
    // Cyrillic 'а' (U+0430) looks like 'a' but is not ASCII.
    assert!(validate_name_segment("\u{0430}bc", "Collection").is_err());
    assert!(validate_name_segment("emoji-🚀", "Collection").is_err());
}
