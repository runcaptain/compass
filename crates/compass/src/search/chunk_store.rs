//! Disk-backed chunk store using redb.
//!
//! Replaces `HashMap<u64, DocumentChunk>` with a persistent embedded database.
//! Point lookups by u64 ID, batch inserts, full scans for rebuild.

use crate::models::DocumentChunk;
use redb::{Database, DatabaseError, ReadableTable, ReadableTableMetadata, TableDefinition};
use std::path::Path;
use std::time::Duration;

const CHUNKS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("chunks");

/// Max attempts to acquire the redb flock during open. 6 attempts at
/// OPEN_RETRY_BACKOFF gives a total upper bound of about 30 seconds before
/// the open errors out, which is enough for any reasonable networked-filesystem lock
/// release latency while staying under typical startup health probes.
const OPEN_MAX_ATTEMPTS: u32 = 6;
const OPEN_RETRY_BACKOFF: Duration = Duration::from_secs(5);

pub struct ChunkStore {
    db: Database,
}

impl ChunkStore {
    /// Open or create the chunk store at `path`. Tolerates the transient
    /// `DatabaseAlreadyOpen` error that happens on networked filesystems
    /// when a previous process's flock has not released yet after it was killed.
    /// Also enables redb's repair callback so a dirty file from an abrupt
    /// shutdown auto-recovers instead of erroring out.
    pub fn open(path: &Path) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::open_with_retries(path, OPEN_MAX_ATTEMPTS, OPEN_RETRY_BACKOFF)
    }

    /// Same as `open`, with configurable retry parameters. Exposed so tests can
    /// exercise the lock-retry path with a sub-second backoff instead of the
    /// 5-second default.
    pub(crate) fn open_with_retries(
        path: &Path,
        max_attempts: u32,
        backoff: Duration,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut attempt = 0u32;
        let db = loop {
            attempt += 1;
            match Database::builder().set_repair_callback(|_| {}).create(path) {
                Ok(db) => break db,
                Err(DatabaseError::DatabaseAlreadyOpen) if attempt < max_attempts => {
                    tracing::warn!(
                        "redb file {:?} is locked (attempt {}/{}). Retrying in {:?}. \
                         This is usually transient on networked filesystems when a previous \
                         process's flock takes time to release after termination.",
                        path,
                        attempt,
                        max_attempts,
                        backoff
                    );
                    std::thread::sleep(backoff);
                }
                Err(e) => return Err(e.into()),
            }
        };
        {
            let txn = db.begin_write()?;
            let _ = txn.open_table(CHUNKS_TABLE)?;
            txn.commit()?;
        }
        Ok(Self { db })
    }

    pub fn get(
        &self,
        id: u64,
    ) -> Result<Option<DocumentChunk>, Box<dyn std::error::Error + Send + Sync>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(CHUNKS_TABLE)?;
        match table.get(id)? {
            Some(val) => {
                let chunk: DocumentChunk = serde_json::from_slice(val.value())?;
                Ok(Some(chunk))
            }
            None => Ok(None),
        }
    }

    pub fn get_batch(
        &self,
        ids: &[u64],
    ) -> Result<Vec<(u64, DocumentChunk)>, Box<dyn std::error::Error + Send + Sync>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(CHUNKS_TABLE)?;
        let mut results = Vec::with_capacity(ids.len());
        for &id in ids {
            if let Some(val) = table.get(id)? {
                let chunk: DocumentChunk = serde_json::from_slice(val.value())?;
                results.push((id, chunk));
            }
        }
        Ok(results)
    }

    pub fn insert(
        &self,
        id: u64,
        chunk: &DocumentChunk,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let bytes = serde_json::to_vec(chunk)?;
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(CHUNKS_TABLE)?;
            table.insert(id, bytes.as_slice())?;
        }
        txn.commit()?;
        Ok(())
    }

    pub fn insert_batch(
        &self,
        chunks: &[(u64, DocumentChunk)],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(CHUNKS_TABLE)?;
            for (id, chunk) in chunks {
                let bytes = serde_json::to_vec(chunk)?;
                table.insert(*id, bytes.as_slice())?;
            }
        }
        txn.commit()?;
        Ok(())
    }

    pub fn count(&self) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(CHUNKS_TABLE)?;
        Ok(table.len()?)
    }

    /// Iterate all chunks. Used for rebuild (infrequent).
    pub fn for_each<F>(&self, mut f: F) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    where
        F: FnMut(u64, DocumentChunk),
    {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(CHUNKS_TABLE)?;
        for entry in table.iter()? {
            let (key, val) = entry?;
            let chunk: DocumentChunk = serde_json::from_slice(val.value())?;
            f(key.value(), chunk);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::MetadataValue;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Unique temp path per test invocation, even under cargo's parallel runner.
    fn unique_path(label: &str) -> std::path::PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "compass-chunkstore-{}-{}-{}-{}.redb",
            label,
            std::process::id(),
            nanos,
            N.fetch_add(1, Ordering::SeqCst)
        ))
    }

    fn make_chunk(id: u64, text: &str) -> DocumentChunk {
        let mut metadata = HashMap::new();
        metadata.insert("k".to_string(), MetadataValue::String("v".to_string()));
        DocumentChunk {
            id,
            collection: "test".to_string(),
            file_id: format!("f{}", id),
            chunk_index: 0,
            page: None,
            text: text.to_string(),
            metadata,
            doc_type: "chunk".to_string(),
            parent_id: None,
            group_id: None,
            embeddings: HashMap::new(),
            embedding: None,
        }
    }

    #[test]
    fn insert_then_get_roundtrip() {
        let path = unique_path("roundtrip");
        let store = ChunkStore::open(&path).unwrap();
        store.insert(7, &make_chunk(7, "hello")).unwrap();
        let got = store.get(7).unwrap().expect("chunk should be present");
        assert_eq!(got.id, 7);
        assert_eq!(got.text, "hello");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn get_returns_none_for_missing_id() {
        let path = unique_path("missing");
        let store = ChunkStore::open(&path).unwrap();
        store.insert(1, &make_chunk(1, "x")).unwrap();
        assert!(store.get(999).unwrap().is_none());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn batch_insert_count_matches() {
        let path = unique_path("count");
        let store = ChunkStore::open(&path).unwrap();
        store
            .insert_batch(&[
                (1, make_chunk(1, "a")),
                (2, make_chunk(2, "b")),
                (3, make_chunk(3, "c")),
            ])
            .unwrap();
        assert_eq!(store.count().unwrap(), 3);
        let _ = std::fs::remove_file(&path);
    }

    /// The core durability test. Insert into a ChunkStore, drop it so redb
    /// closes the file, reopen at the same path, verify every chunk comes
    /// back via `for_each`. This is the primitive that makes Compass safe
    /// across process restarts.
    #[test]
    fn persists_across_close_and_reopen() {
        let path = unique_path("persist");
        {
            let store = ChunkStore::open(&path).unwrap();
            store
                .insert_batch(&[
                    (1, make_chunk(1, "alpha")),
                    (2, make_chunk(2, "beta")),
                    (3, make_chunk(3, "gamma")),
                ])
                .unwrap();
            // store dropped here, redb closes the file
        }
        let reopened = ChunkStore::open(&path).unwrap();
        assert_eq!(reopened.count().unwrap(), 3);
        let mut collected: Vec<(u64, String)> = Vec::new();
        reopened
            .for_each(|id, chunk| {
                collected.push((id, chunk.text));
            })
            .unwrap();
        collected.sort_by_key(|(id, _)| *id);
        assert_eq!(
            collected,
            vec![
                (1, "alpha".to_string()),
                (2, "beta".to_string()),
                (3, "gamma".to_string()),
            ]
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn insert_overwrites_existing_id() {
        let path = unique_path("overwrite");
        let store = ChunkStore::open(&path).unwrap();
        store.insert(5, &make_chunk(5, "first")).unwrap();
        store.insert(5, &make_chunk(5, "second")).unwrap();
        let got = store.get(5).unwrap().unwrap();
        assert_eq!(got.text, "second");
        assert_eq!(store.count().unwrap(), 1);
        let _ = std::fs::remove_file(&path);
    }

    /// When a redb file is locked by another open instance, opening should
    /// retry the configured number of times then fail with the redb lock error.
    /// Simulates the scenario where a previous process's flock has not
    /// been released yet by the kernel/NFS layer.
    #[test]
    fn open_retries_on_lock_then_fails() {
        let path = unique_path("locked-fail");
        // Hold the file exclusively in this scope. flock(LOCK_EX|LOCK_NB) on the
        // same path from another open() will return EWOULDBLOCK, which redb
        // surfaces as DatabaseError::DatabaseAlreadyOpen.
        let _holder = ChunkStore::open(&path).unwrap();

        // 3 attempts at 50ms backoff = ~100ms total. Fast unit test, but it
        // proves the retry path executes and propagates the final error.
        let started = std::time::Instant::now();
        let result = ChunkStore::open_with_retries(&path, 3, Duration::from_millis(50));
        let elapsed = started.elapsed();

        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("second open against a held lock should have errored after retries"),
        };
        assert!(
            err_msg.contains("already open"),
            "error should be the redb 'Database already open' message, got: {}",
            err_msg
        );
        // Sanity check: at least 2 backoff periods should have elapsed (attempts
        // 1 and 2 failed before the final failed attempt 3).
        assert!(
            elapsed >= Duration::from_millis(100),
            "expected at least 100ms of backoff time, observed {:?}",
            elapsed
        );
        // And not way more than 3 backoffs worth (catches a regression to
        // unbounded retries).
        assert!(
            elapsed < Duration::from_secs(2),
            "retries should bound total wait time, observed {:?}",
            elapsed
        );

        // _holder still alive here; explicit drop after the assertions to make
        // sure the second-open above genuinely contended with a live lock.
        drop(_holder);
        let _ = std::fs::remove_file(&path);
    }

    /// After the holder releases the lock, a subsequent open should succeed
    /// without going through the retry path. Confirms the happy path still
    /// works after the new lock-handling code.
    #[test]
    fn open_succeeds_after_lock_released() {
        let path = unique_path("locked-then-released");
        {
            let store = ChunkStore::open(&path).unwrap();
            store.insert(1, &make_chunk(1, "before-restart")).unwrap();
            // store dropped here, flock released
        }
        // Reopen with the retry-enabled API; should not retry at all since the
        // previous holder is gone.
        let started = std::time::Instant::now();
        let reopened = ChunkStore::open_with_retries(&path, 6, Duration::from_secs(5))
            .expect("reopen should succeed once previous holder is dropped");
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_millis(500),
            "happy-path open should not sleep through retry backoffs, observed {:?}",
            elapsed
        );
        let got = reopened
            .get(1)
            .unwrap()
            .expect("inserted chunk should still be there");
        assert_eq!(got.text, "before-restart");
        let _ = std::fs::remove_file(&path);
    }
}
