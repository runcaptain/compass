//! CAS-leased chunk-id block allocator — `{ns}/id-alloc` in object storage.
//!
//! In cloud mode, EVERY ingest path allocates chunk ids from blocks claimed
//! here (attached serving nodes pool a block; stateless writers claim per
//! batch). A local `next_id` counter cannot be the allocation source in cloud
//! mode: recovery computes `max_id + 1`, which can land inside another
//! writer's active, partially-used block — colliding with ids that writer
//! will mint next. The invariant this module defends is *no id is ever handed
//! out twice*; global ordering is irrelevant (replay order comes from the
//! manifest `seq`, not from ids), and gaps from crashed writers are fine.
//!
//! Local mode never touches this module (`next_id` remains the allocator).

use super::{Storage, StorageError};
use serde::{Deserialize, Serialize};
use std::ops::Range;

/// Ids claimed per CAS round-trip. Large enough that an attached node's pool
/// refill is rare; small enough that a crashed writer leaks little.
pub const BLOCK: u64 = 10_000;

const MAX_CAS_RETRIES: u32 = 10;

fn alloc_key(ns: &str) -> String {
    format!("{ns}/id-alloc")
}

#[derive(Debug, Serialize, Deserialize)]
struct AllocState {
    next_block_start: u64,
}

/// Create-only seed of the allocator. `start` must be one past the highest id
/// ever assigned in the namespace (0 for a fresh collection). Losing the
/// create race is fine — the winner's value is equally valid because no new
/// ids can be minted while the allocator is absent (all cloud ingest paths
/// require it), so concurrent seeders compute the same high-water mark.
pub async fn seed(storage: &dyn Storage, ns: &str, start: u64) -> Result<(), StorageError> {
    let state = AllocState {
        next_block_start: start,
    };
    let bytes = serde_json::to_vec(&state)
        .map_err(|e| StorageError::Io(format!("id-alloc encode: {e}")))?;
    match storage
        .put_if_not_exists(&alloc_key(ns), bytes::Bytes::from(bytes))
        .await
    {
        Ok(_) => Ok(()),
        Err(StorageError::AlreadyExists(_)) => Ok(()), // racer seeded it — fine
        Err(e) => Err(e),
    }
}

/// Claim a block of at least `count` ids (min [`BLOCK`]) via CAS. Returns the
/// claimed half-open range. `NotFound` means the allocator was never seeded
/// (pre-v0.4 namespace) — the caller migrates via [`seed`] and retries.
pub async fn claim(
    storage: &dyn Storage,
    ns: &str,
    count: u64,
) -> Result<Range<u64>, StorageError> {
    let want = count.max(BLOCK);
    let key = alloc_key(ns);
    for _ in 0..MAX_CAS_RETRIES {
        let (bytes, version) = storage.get_versioned(&key).await?;
        let state: AllocState = serde_json::from_slice(&bytes)
            .map_err(|e| StorageError::Io(format!("id-alloc decode for '{ns}': {e}")))?;
        let start = state.next_block_start;
        let end = start.checked_add(want).ok_or_else(|| {
            StorageError::Io(format!("id space exhausted for '{ns}' (u64 overflow)"))
        })?;
        let next = AllocState {
            next_block_start: end,
        };
        let encoded = serde_json::to_vec(&next)
            .map_err(|e| StorageError::Io(format!("id-alloc encode: {e}")))?;
        match storage
            .put_if_match(&key, bytes::Bytes::from(encoded), &version)
            .await
        {
            Ok(_) => return Ok(start..end),
            Err(StorageError::VersionConflict { .. }) => continue,
            Err(e) => return Err(e),
        }
    }
    Err(StorageError::Io(format!(
        "id-alloc CAS failed after {MAX_CAS_RETRIES} retries for '{ns}' (persistent contention)"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::local::LocalDiskStorage;
    use std::sync::Arc;

    fn storage(name: &str) -> Arc<dyn Storage> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let mut root = std::env::temp_dir();
        root.push(format!(
            "compass_idalloc_test_{}_{}_{}",
            name,
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        Arc::new(LocalDiskStorage::new(root).unwrap())
    }

    #[tokio::test]
    async fn seed_then_claim_advances() {
        let s = storage("basic");
        seed(s.as_ref(), "ns", 0).await.unwrap();
        let a = claim(s.as_ref(), "ns", 5).await.unwrap();
        assert_eq!(a, 0..BLOCK); // min block size applies
        let b = claim(s.as_ref(), "ns", 25_000).await.unwrap();
        assert_eq!(b, BLOCK..BLOCK + 25_000); // large batches claim exactly enough
    }

    #[tokio::test]
    async fn seed_race_is_idempotent() {
        let s = storage("seedrace");
        seed(s.as_ref(), "ns", 42).await.unwrap();
        // A racing seeder (same computed high-water) loses silently.
        seed(s.as_ref(), "ns", 42).await.unwrap();
        let a = claim(s.as_ref(), "ns", 1).await.unwrap();
        assert_eq!(a.start, 42);
    }

    #[tokio::test]
    async fn claim_before_seed_is_not_found() {
        let s = storage("unseeded");
        assert!(matches!(
            claim(s.as_ref(), "ns", 1).await,
            Err(StorageError::NotFound(_))
        ));
    }

    // The allocator's whole job under concurrency: N racing claimants must
    // receive disjoint ranges (modeled on the LSM's concurrent-appends test).
    #[tokio::test]
    async fn concurrent_claims_are_disjoint() {
        let s = storage("concurrent");
        seed(s.as_ref(), "ns", 0).await.unwrap();
        let n = 16;
        let mut handles = Vec::new();
        for _ in 0..n {
            let s2 = s.clone();
            handles.push(tokio::spawn(
                async move { claim(s2.as_ref(), "ns", 1).await },
            ));
        }
        let mut ranges: Vec<Range<u64>> = Vec::new();
        for h in handles {
            ranges.push(h.await.unwrap().unwrap());
        }
        ranges.sort_by_key(|r| r.start);
        for w in ranges.windows(2) {
            assert!(
                w[0].end <= w[1].start,
                "overlapping claims: {:?} vs {:?}",
                w[0],
                w[1]
            );
        }
        // No holes either: 16 min-size blocks tile exactly.
        assert_eq!(ranges.last().unwrap().end, n as u64 * BLOCK);
    }
}
