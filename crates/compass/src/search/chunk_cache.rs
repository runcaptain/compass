//! Bounded LRU read-through cache over the disk-backed [`ChunkStore`].
//!
//! The out-of-core primitive: the source of truth for chunk metadata is the
//! redb `ChunkStore` on disk (or, later, object storage). This cache holds only
//! a bounded number of hot chunks in RAM, so a collection of any size no longer
//! requires rehydrating every chunk into an unbounded `HashMap` at startup.
//!
//! Reads check the cache, fall through to the store on a miss (populating the
//! cache), and evict least-recently-used entries past the capacity. Writes go
//! through to the store and prime the cache. This is the explicit equivalent of
//! the OS page cache the mmap vector layer already relies on.
//!
//! Introduced as an additive building block — the engine's hot path still uses
//! the in-memory map today; wiring search/scoring to read through this cache
//! (and dropping the full rehydrate) is the follow-on change.

use crate::models::DocumentChunk;
use crate::search::chunk_store::ChunkStore;
use lru::LruCache;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Mutex;

type BoxErr = Box<dyn std::error::Error + Send + Sync>;

/// Default number of chunks kept resident. Chosen so a large collection stays
/// bounded in RAM while the common small-collection case is effectively fully
/// cached. Tunable via `ChunkCache::with_capacity`.
pub const DEFAULT_CAPACITY: usize = 100_000;

pub struct ChunkCache {
    store: ChunkStore,
    cache: Mutex<LruCache<u64, DocumentChunk>>,
}

impl ChunkCache {
    /// Wrap a `ChunkStore` with the default cache capacity.
    pub fn new(store: ChunkStore) -> Self {
        Self::with_capacity(store, DEFAULT_CAPACITY)
    }

    /// Wrap a `ChunkStore` with an explicit cache capacity (min 1).
    pub fn with_capacity(store: ChunkStore, capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).unwrap();
        Self {
            store,
            cache: Mutex::new(LruCache::new(cap)),
        }
    }

    /// Read a chunk: cache first, then the store on a miss (populating cache).
    pub fn get(&self, id: u64) -> Result<Option<DocumentChunk>, BoxErr> {
        {
            let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(chunk) = cache.get(&id) {
                return Ok(Some(chunk.clone()));
            }
        }
        match self.store.get(id)? {
            Some(chunk) => {
                let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                cache.put(id, chunk.clone());
                Ok(Some(chunk))
            }
            None => Ok(None),
        }
    }

    /// Read a page of chunks by id. Misses are fetched from the store in one
    /// batch and cached. Returns chunks in the SAME ORDER as `ids` (skipping ids
    /// that do not exist) — honoring the documented ordering contract.
    pub fn get_batch(&self, ids: &[u64]) -> Result<Vec<DocumentChunk>, BoxErr> {
        // First pass: cache hits + collect misses.
        let mut resolved: HashMap<u64, DocumentChunk> = HashMap::new();
        let mut misses: Vec<u64> = Vec::new();
        {
            let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            for &id in ids {
                if let Some(chunk) = cache.get(&id) {
                    resolved.insert(id, chunk.clone());
                } else {
                    misses.push(id);
                }
            }
        }

        // Fetch misses from the store, cache them, add to the resolved map.
        if !misses.is_empty() {
            let fetched = self.store.get_batch(&misses)?;
            let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            for (id, chunk) in fetched {
                cache.put(id, chunk.clone());
                resolved.insert(id, chunk);
            }
        }

        // Assemble output strictly in the requested id order. `get` (not
        // `remove`) so a duplicated requested id yields the chunk each time,
        // honoring the ordering contract for dupes too.
        Ok(ids
            .iter()
            .filter_map(|id| resolved.get(id).cloned())
            .collect())
    }

    /// Insert a batch: write through to the store, then prime the cache.
    pub fn insert_batch(&self, chunks: &[(u64, DocumentChunk)]) -> Result<(), BoxErr> {
        self.store.insert_batch(chunks)?;
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        for (id, chunk) in chunks {
            cache.put(*id, chunk.clone());
        }
        Ok(())
    }

    /// Tombstone passthrough (evicts tombstoned ids from the cache too).
    pub fn tombstone_batch(&self, ids: &[u64]) -> Result<(), BoxErr> {
        self.store.tombstone_batch(ids)?;
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        for id in ids {
            cache.pop(id);
        }
        Ok(())
    }

    /// Load persisted tombstones (passthrough).
    pub fn load_tombstones(&self) -> Result<Vec<u64>, BoxErr> {
        self.store.load_tombstones()
    }

    /// Number of chunks durably stored (not the cache size).
    pub fn count(&self) -> Result<u64, BoxErr> {
        self.store.count()
    }

    /// Iterate all durably stored chunks (delegates to the store; used for
    /// rebuild-style full scans).
    pub fn for_each<F: FnMut(u64, DocumentChunk)>(&self, f: F) -> Result<(), BoxErr> {
        self.store.for_each(f)
    }

    /// Current number of resident (cached) chunks — for tests/metrics.
    pub fn resident(&self) -> usize {
        self.cache.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Access the underlying store (for code paths that must bypass the cache).
    pub fn store(&self) -> &ChunkStore {
        &self.store
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::MetadataValue;
    use std::collections::HashMap;

    fn chunk(id: u64) -> DocumentChunk {
        let mut metadata = HashMap::new();
        metadata.insert("n".to_string(), MetadataValue::Int(id as i64));
        DocumentChunk {
            id,
            collection: "t".to_string(),
            file_id: format!("f{id}"),
            chunk_index: 0,
            page: None,
            text: format!("chunk {id}"),
            metadata,
            doc_type: "chunk".to_string(),
            parent_id: None,
            group_id: None,
            embeddings: HashMap::new(),
            embedding: None,
        }
    }

    fn store(name: &str) -> ChunkStore {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "compass_chunkcache_{}_{}.redb",
            name,
            std::process::id()
        ));
        let _ = std::fs::remove_file(&p);
        ChunkStore::open(&p).unwrap()
    }

    #[test]
    fn read_through_and_persist() {
        let cache = ChunkCache::with_capacity(store("rt"), 8);
        cache.insert_batch(&[(1, chunk(1)), (2, chunk(2))]).unwrap();
        assert_eq!(cache.count().unwrap(), 2);
        assert_eq!(cache.get(1).unwrap().unwrap().id, 1);
        assert_eq!(cache.get(2).unwrap().unwrap().text, "chunk 2");
        assert!(cache.get(99).unwrap().is_none());
    }

    #[test]
    fn eviction_is_bounded_but_store_retains() {
        // Capacity 2, but insert 5 -> cache holds <=2, store holds all 5.
        let cache = ChunkCache::with_capacity(store("evict"), 2);
        let items: Vec<(u64, DocumentChunk)> = (0..5).map(|i| (i, chunk(i))).collect();
        cache.insert_batch(&items).unwrap();
        assert!(cache.resident() <= 2, "cache must stay bounded");
        assert_eq!(cache.count().unwrap(), 5, "store retains all");

        // Every chunk is still readable (served from the store on cache miss).
        for i in 0..5 {
            assert_eq!(cache.get(i).unwrap().unwrap().id, i);
        }
    }

    #[test]
    fn get_batch_mixes_hits_and_misses() {
        let cache = ChunkCache::with_capacity(store("batch"), 100);
        let items: Vec<(u64, DocumentChunk)> = (0..10).map(|i| (i, chunk(i))).collect();
        // Insert directly to the STORE only, so the cache starts cold.
        cache.store().insert_batch(&items).unwrap();

        // Prime a couple into the cache.
        let _ = cache.get(3).unwrap();
        let _ = cache.get(7).unwrap();

        // Request order mixes cached (3,7) and store-miss (5); 42 doesn't exist.
        // Output must be in the SAME ORDER as the request ids (contract), not
        // hits-then-misses.
        let got = cache.get_batch(&[7, 5, 3, 42]).unwrap();
        let ids: Vec<u64> = got.iter().map(|c| c.id).collect();
        assert_eq!(ids, vec![7, 5, 3], "get_batch must preserve request order");
    }
}
