//! Real-S3 integration tests — run against an actual S3-compatible endpoint
//! (MinIO, AWS S3, R2), exercising the behaviors the in-memory backend can't
//! prove: real ETag CAS semantics, conditional puts, delimiter listing.
//!
//! Gated on env so they SKIP (pass trivially) when no endpoint is configured:
//!
//! ```bash
//! docker compose -f docker-compose.minio.yml up -d minio createbucket
//! COMPASS_TEST_S3_BUCKET=compass-data \
//! COMPASS_S3_ENDPOINT=http://localhost:9000 \
//! COMPASS_S3_ALLOW_HTTP=true \
//! AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin \
//! AWS_DEFAULT_REGION=us-east-1 \
//! cargo test --features object-storage s3_integration -- --nocapture
//! ```

use super::object_store_backend::ObjectStoreBackend;
use super::{lsm, Storage, StorageError};
use bytes::Bytes;

/// Returns a backend against the configured real endpoint, or None (→ skip)
/// when COMPASS_TEST_S3_BUCKET isn't set.
fn real_s3() -> Option<ObjectStoreBackend> {
    let bucket = std::env::var("COMPASS_TEST_S3_BUCKET").ok()?;
    if bucket.is_empty() {
        return None;
    }
    Some(
        ObjectStoreBackend::from_url(&format!("s3://{bucket}"))
            .expect("real-S3 backend must build from COMPASS_TEST_S3_BUCKET"),
    )
}

/// Unique namespace per test process so runs never collide on shared buckets.
fn test_ns(tag: &str) -> String {
    format!("it-{tag}-{}", std::process::id())
}

/// Best-effort cleanup of every object under the test namespace.
async fn cleanup(s: &dyn Storage, ns: &str) {
    let _ = lsm::delete_namespace(s, ns).await;
}

// Real ETag CAS: the property the whole manifest-commit protocol rests on.
// The memory backend fakes this; MinIO/S3 enforce it server-side.
#[tokio::test]
async fn s3_integration_cas_semantics() {
    let Some(s) = real_s3() else {
        eprintln!("skipped: COMPASS_TEST_S3_BUCKET not set");
        return;
    };
    let ns = test_ns("cas");
    let key = format!("{ns}/cas-probe");

    let v1 = s.put(&key, Bytes::from_static(b"one")).await.unwrap();
    assert!(!v1.is_empty(), "real S3 must return a usable version token");

    // Fresh CAS succeeds; the returned version differs.
    let v2 = s
        .put_if_match(&key, Bytes::from_static(b"two"), &v1)
        .await
        .unwrap();
    assert_ne!(v1, v2);
    assert_eq!(&s.get(&key).await.unwrap()[..], b"two");

    // Stale CAS must conflict — enforced by the SERVER, not our code.
    let r = s.put_if_match(&key, Bytes::from_static(b"x"), &v1).await;
    assert!(
        matches!(r, Err(StorageError::VersionConflict { .. })),
        "stale ETag must be rejected by the real endpoint, got {r:?}"
    );

    // Conditional create: second create of the same key must fail.
    let ckey = format!("{ns}/create-once");
    s.put_if_not_exists(&ckey, Bytes::from_static(b"a"))
        .await
        .unwrap();
    let r = s.put_if_not_exists(&ckey, Bytes::from_static(b"b")).await;
    assert!(
        matches!(r, Err(StorageError::AlreadyExists(_))),
        "duplicate create must fail on the real endpoint, got {r:?}"
    );

    cleanup(&s, &ns).await;
}

// The full LSM lifecycle against real S3: append → tombstone → compact (with
// GC) → namespace discovery via delimiter listing.
#[tokio::test]
async fn s3_integration_lsm_lifecycle() {
    let Some(s) = real_s3() else {
        eprintln!("skipped: COMPASS_TEST_S3_BUCKET not set");
        return;
    };
    let ns = test_ns("lsm");
    cleanup(&s, &ns).await; // start clean even after a crashed prior run

    // Three appends: data, data, tombstone.
    lsm::append_fragment(&s, &ns, Bytes::from_static(b"[]"), 1)
        .await
        .unwrap();
    lsm::append_fragment(&s, &ns, Bytes::from_static(b"[]"), 1)
        .await
        .unwrap();
    lsm::append_tombstone(&s, &ns, &[0]).await.unwrap();

    let (m, version) = lsm::read_manifest(&s, &ns).await.unwrap();
    assert_eq!(m.fragments.len(), 3);
    assert_eq!(m.next_seq, 3);

    // Full compaction: one segment replaces the WAL; CAS commit on real ETags.
    lsm::replace_with_single_segment(&s, &ns, &version, &m, Bytes::from_static(b"{}"), 1)
        .await
        .unwrap();
    let (m2, _) = lsm::read_manifest(&s, &ns).await.unwrap();
    assert_eq!(m2.segments.len(), 1);
    assert!(m2.uncompacted().count() == 0);
    // The folded fragments are staged for next-cycle GC, not yet deleted.
    assert_eq!(m2.pending_deletes.len(), 3);

    // A second compaction cycle physically GCs the staged fragments.
    let (m2, v2) = lsm::read_manifest(&s, &ns).await.unwrap();
    lsm::replace_with_single_segment(&s, &ns, &v2, &m2, Bytes::from_static(b"{}"), 1)
        .await
        .unwrap();
    let leftovers = s.list(&format!("{ns}/wal/")).await.unwrap();
    assert!(
        leftovers.is_empty(),
        "WAL fragments must be physically GC'd from real S3, found {:?}",
        leftovers.iter().map(|o| &o.key).collect::<Vec<_>>()
    );

    // Namespace discovery via the delimiter listing finds this ns.
    let names = lsm::list_namespaces(&s).await.unwrap();
    assert!(
        names.contains(&ns),
        "delimiter-based discovery must find '{ns}', got {names:?}"
    );

    cleanup(&s, &ns).await;
}

// Concurrent appends against the REAL endpoint: no lost writes under true
// server-side CAS (the memory backend can't prove network-level contention).
#[tokio::test]
async fn s3_integration_concurrent_appends() {
    let Some(s) = real_s3() else {
        eprintln!("skipped: COMPASS_TEST_S3_BUCKET not set");
        return;
    };
    let s = std::sync::Arc::new(s);
    let ns = test_ns("conc");
    cleanup(s.as_ref(), &ns).await;

    let n = 8u64;
    let mut handles = Vec::new();
    for i in 0..n {
        let s2 = s.clone();
        let ns2 = ns.clone();
        handles.push(tokio::spawn(async move {
            let payload = Bytes::from(format!("writer-{i}"));
            lsm::append_fragment(s2.as_ref(), &ns2, payload, 1).await
        }));
    }
    for h in handles {
        h.await.unwrap().unwrap();
    }

    let (m, _) = lsm::read_manifest(s.as_ref(), &ns).await.unwrap();
    assert_eq!(m.fragments.len() as u64, n, "no append may be lost");
    let mut seqs: Vec<u64> = m.fragments.iter().map(|f| f.seq).collect();
    seqs.sort_unstable();
    assert_eq!(seqs, (0..n).collect::<Vec<_>>(), "seqs gap-free");

    cleanup(s.as_ref(), &ns).await;
}
