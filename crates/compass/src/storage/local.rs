//! `LocalDiskStorage` — the default [`Storage`] backend over the local filesystem.
//!
//! This *is* the embedded, local-first behavior. Keys become files under a root
//! directory; `/` in a key is a path separator. Writes are atomic (tmp file +
//! rename). The CAS [`Version`] token is a SHA-256 of the object's bytes, so
//! `put_if_match` succeeds iff the on-disk content still hashes to `expected`.
//!
//! Concurrency: a per-key in-process mutex serializes read-modify-write so CAS
//! is correct within one process. Local mode is single-writer by contract, so
//! cross-process CAS is best-effort (the object-storage backend uses real ETag
//! CAS for the multi-writer story). Path components are validated to prevent a
//! key like `../../etc` from escaping the root.

use super::{ObjectMeta, Storage, StorageError, Version};
use async_trait::async_trait;
use bytes::Bytes;
use sha2::{Digest, Sha256};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Number of lock shards. A fixed pool bounds memory (the old per-key map grew
/// unbounded — one entry per distinct key forever, and WAL fragment keys are all
/// unique). Keys hash to a shard; two keys sharing a shard serialize
/// unnecessarily but that's harmless for correctness and rare with 256 shards.
const LOCK_SHARDS: usize = 256;

pub struct LocalDiskStorage {
    root: PathBuf,
    /// Fixed pool of per-shard locks for read-modify-write CAS. Bounded memory.
    lock_shards: Vec<Mutex<()>>,
}

impl LocalDiskStorage {
    /// Create a backend rooted at `root`. The directory is created if missing.
    pub fn new(root: impl Into<PathBuf>) -> std::io::Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        let lock_shards = (0..LOCK_SHARDS).map(|_| Mutex::new(())).collect();
        Ok(Self { root, lock_shards })
    }

    /// Resolve a logical key to a filesystem path under the root, rejecting any
    /// key whose components would escape the root (`..`, absolute, etc.).
    fn path_for(&self, key: &str) -> Result<PathBuf, StorageError> {
        if key.is_empty() {
            return Err(StorageError::Io("empty key".into()));
        }
        let mut p = self.root.clone();
        for comp in key.split('/') {
            if comp.is_empty() || comp == "." || comp == ".." {
                return Err(StorageError::Io(format!(
                    "invalid key component in '{key}'"
                )));
            }
            p.push(comp);
        }
        Ok(p)
    }

    /// The lock shard for a key (fixed pool, bounded memory).
    fn key_lock(&self, key: &str) -> &Mutex<()> {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut h);
        let shard = (h.finish() as usize) % LOCK_SHARDS;
        &self.lock_shards[shard]
    }

    fn version_of(bytes: &[u8]) -> Version {
        let mut h = Sha256::new();
        h.update(bytes);
        Version::etag(format!("{:x}", h.finalize()))
    }

    /// Atomic write: tmp file in the same dir + rename. Returns the new version.
    fn write_atomic(path: &Path, bytes: &[u8]) -> Result<Version, StorageError> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| StorageError::Io(e.to_string()))?;
        }
        // Globally-unique tmp name in the same directory (pid + monotonic
        // counter), appended as a SUFFIX so it never collides with another
        // key's tmp file and never replaces the real file's extension. The
        // rename is atomic on the same filesystem.
        let file_name = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "obj".to_string());
        let n = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp_name = format!(".{file_name}.tmp-{}-{n}", std::process::id());
        let tmp = path.with_file_name(tmp_name);
        std::fs::write(&tmp, bytes).map_err(|e| StorageError::Io(e.to_string()))?;
        std::fs::rename(&tmp, path).map_err(|e| StorageError::Io(e.to_string()))?;
        Ok(Self::version_of(bytes))
    }

    /// Read a whole object. Errors carry the logical KEY, not the filesystem
    /// path — paths would leak server internals into API error bodies.
    fn read_all(path: &Path, key: &str) -> Result<Vec<u8>, StorageError> {
        std::fs::read(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                StorageError::NotFound(key.to_string())
            } else {
                StorageError::Io(e.to_string())
            }
        })
    }
}

#[async_trait]
impl Storage for LocalDiskStorage {
    async fn get(&self, key: &str) -> Result<Bytes, StorageError> {
        let path = self.path_for(key)?;
        let bytes = Self::read_all(&path, key)?;
        Ok(Bytes::from(bytes))
    }

    async fn get_range(&self, key: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        use std::io::{Read, Seek, SeekFrom};
        let path = self.path_for(key)?;
        let mut file = std::fs::File::open(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                StorageError::NotFound(key.to_string())
            } else {
                StorageError::Io(e.to_string())
            }
        })?;
        let size = file
            .metadata()
            .map_err(|e| StorageError::Io(e.to_string()))?
            .len();
        if range.start > range.end || range.end > size {
            return Err(StorageError::InvalidRange {
                start: range.start,
                end: range.end,
                size,
            });
        }
        // Seek + read ONLY the requested range (bounded memory) — honoring the
        // out-of-core contract instead of loading the whole object.
        file.seek(SeekFrom::Start(range.start))
            .map_err(|e| StorageError::Io(e.to_string()))?;
        let len = (range.end - range.start) as usize;
        let mut buf = vec![0u8; len];
        file.read_exact(&mut buf)
            .map_err(|e| StorageError::Io(e.to_string()))?;
        Ok(Bytes::from(buf))
    }

    async fn get_versioned(&self, key: &str) -> Result<(Bytes, Version), StorageError> {
        let path = self.path_for(key)?;
        let bytes = Self::read_all(&path, key)?;
        let version = Self::version_of(&bytes);
        Ok((Bytes::from(bytes), version))
    }

    async fn put(&self, key: &str, bytes: Bytes) -> Result<Version, StorageError> {
        let path = self.path_for(key)?;
        // Serialize against concurrent CAS on the same key — a plain put that
        // interleaved inside put_if_match's read-check-write would be silently
        // clobbered, an in-process lost update.
        let lock = self.key_lock(key);
        let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        Self::write_atomic(&path, &bytes)
    }

    async fn put_if_match(
        &self,
        key: &str,
        bytes: Bytes,
        expected: &Version,
    ) -> Result<Version, StorageError> {
        let path = self.path_for(key)?;
        let lock = self.key_lock(key);
        // Poison-tolerant: the guarded data is a unit marker, uncorrupted by a
        // panic, so recover the guard rather than cascade panics.
        let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());

        // Current version must match `expected`.
        let current = match Self::read_all(&path, key) {
            Ok(b) => Self::version_of(&b),
            Err(StorageError::NotFound(_)) => {
                // No object yet — CAS against a present version can't match.
                return Err(StorageError::VersionConflict { key: key.into() });
            }
            Err(e) => return Err(e),
        };
        if &current != expected {
            return Err(StorageError::VersionConflict { key: key.into() });
        }
        Self::write_atomic(&path, &bytes)
    }

    async fn put_if_not_exists(&self, key: &str, bytes: Bytes) -> Result<Version, StorageError> {
        let path = self.path_for(key)?;
        // Same-key serialization vs put/put_if_match (see put above).
        let lock = self.key_lock(key);
        let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| StorageError::Io(e.to_string()))?;
        }
        // Atomic create-only via O_EXCL (`create_new`): fails with AlreadyExists
        // if the file exists, and is a true create primitive across processes —
        // unlike the old `exists()`-then-write, which had a TOCTOU race and would
        // overwrite a file created by a concurrent writer.
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut file) => {
                use std::io::Write;
                file.write_all(&bytes)
                    .map_err(|e| StorageError::Io(e.to_string()))?;
                file.sync_all()
                    .map_err(|e| StorageError::Io(e.to_string()))?;
                Ok(Self::version_of(&bytes))
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                Err(StorageError::AlreadyExists(key.into()))
            }
            Err(e) => Err(StorageError::Io(e.to_string())),
        }
    }

    async fn delete(&self, key: &str) -> Result<(), StorageError> {
        let path = self.path_for(key)?;
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(StorageError::Io(e.to_string())),
        }
    }

    async fn list(&self, prefix: &str) -> Result<Vec<ObjectMeta>, StorageError> {
        // Map the prefix to a directory (or a parent dir + filename stem). For
        // simplicity we walk the directory the prefix points at; keys are
        // returned as logical (root-relative, '/'-joined) paths.
        let mut out = Vec::new();
        // Tolerate a trailing-slash prefix ("ns/") the way object stores do —
        // path_for would reject the empty trailing component.
        let prefix = prefix.strip_suffix('/').unwrap_or(prefix);
        let base = if prefix.is_empty() {
            self.root.clone()
        } else {
            self.path_for(prefix)?
        };
        // If `base` is a file, list just it; if a dir, walk it.
        if base.is_file() {
            if let Ok(meta) = std::fs::metadata(&base) {
                out.push(ObjectMeta {
                    key: prefix.to_string(),
                    size: meta.len(),
                    version: None,
                });
            }
            return Ok(out);
        }
        walk_dir(&self.root, &base, &mut out).map_err(|e| StorageError::Io(e.to_string()))?;
        Ok(out)
    }

    async fn list_dirs(&self, prefix: &str) -> Result<Vec<String>, StorageError> {
        // Immediate subdirectories only — one readdir, independent of how many
        // files live below them.
        let prefix = prefix.trim_end_matches('/');
        let base = if prefix.is_empty() {
            self.root.clone()
        } else {
            self.path_for(prefix)?
        };
        let mut dirs = Vec::new();
        let entries = match std::fs::read_dir(&base) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(dirs),
            Err(e) => return Err(StorageError::Io(e.to_string())),
        };
        for entry in entries {
            let entry = entry.map_err(|e| StorageError::Io(e.to_string()))?;
            if entry
                .file_type()
                .map_err(|e| StorageError::Io(e.to_string()))?
                .is_dir()
            {
                dirs.push(entry.file_name().to_string_lossy().into_owned());
            }
        }
        Ok(dirs)
    }

    fn backend_name(&self) -> &'static str {
        "local-disk"
    }
}

/// Recursively collect files under `dir`, emitting keys relative to `root`.
fn walk_dir(root: &Path, dir: &Path, out: &mut Vec<ObjectMeta>) -> std::io::Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let ft = entry.file_type()?;
        if ft.is_dir() {
            walk_dir(root, &path, out)?;
        } else if ft.is_file() {
            // Skip sidecar tmp files from in-flight atomic writes
            // (named `.{key}.tmp-{pid}-{n}`).
            let is_tmp = path
                .file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.starts_with('.') && s.contains(".tmp-"))
                .unwrap_or(false);
            if is_tmp {
                continue;
            }
            if let Ok(rel) = path.strip_prefix(root) {
                let key = rel
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy())
                    .collect::<Vec<_>>()
                    .join("/");
                out.push(ObjectMeta {
                    key,
                    size: entry.metadata()?.len(),
                    version: None,
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(name: &str) -> LocalDiskStorage {
        let mut root = std::env::temp_dir();
        root.push(format!("compass_storage_test_{}", name));
        let _ = std::fs::remove_dir_all(&root);
        LocalDiskStorage::new(root).unwrap()
    }

    #[tokio::test]
    async fn put_get_roundtrip_and_version() {
        let s = store("roundtrip");
        let v = s.put("a/b", Bytes::from_static(b"hello")).await.unwrap();
        let (got, gv) = s.get_versioned("a/b").await.unwrap();
        assert_eq!(&got[..], b"hello");
        assert_eq!(v, gv); // version is content-derived and stable
        assert_eq!(&s.get("a/b").await.unwrap()[..], b"hello");
    }

    #[tokio::test]
    async fn get_range_slices() {
        let s = store("range");
        s.put("k", Bytes::from_static(b"0123456789")).await.unwrap();
        assert_eq!(&s.get_range("k", 2..5).await.unwrap()[..], b"234");
        // Out-of-bounds range errors.
        assert!(matches!(
            s.get_range("k", 5..100).await,
            Err(StorageError::InvalidRange { .. })
        ));
    }

    // Boundary cases for the seek-based get_range (each a plausible off-by-one).
    #[tokio::test]
    async fn get_range_boundaries() {
        let s = store("range_bounds");
        s.put("k", Bytes::from_static(b"0123456789")).await.unwrap(); // len 10
                                                                      // Whole file (end == size).
        assert_eq!(&s.get_range("k", 0..10).await.unwrap()[..], b"0123456789");
        // Last byte.
        assert_eq!(&s.get_range("k", 9..10).await.unwrap()[..], b"9");
        // Mid-file to EOF.
        assert_eq!(&s.get_range("k", 7..10).await.unwrap()[..], b"789");
        // Zero-length range mid-file → empty.
        assert!(s.get_range("k", 5..5).await.unwrap().is_empty());
        // Empty read exactly at EOF → empty, not an error.
        assert!(s.get_range("k", 10..10).await.unwrap().is_empty());
        // start > end → error.
        assert!(matches!(
            s.get_range("k", 6..3).await,
            Err(StorageError::InvalidRange { .. })
        ));
        // end past EOF → error.
        assert!(matches!(
            s.get_range("k", 8..11).await,
            Err(StorageError::InvalidRange { .. })
        ));
    }

    #[tokio::test]
    async fn missing_key_is_not_found() {
        let s = store("missing");
        assert!(matches!(
            s.get("nope").await,
            Err(StorageError::NotFound(_))
        ));
        assert!(!s.exists("nope").await.unwrap());
    }

    #[tokio::test]
    async fn cas_put_if_match() {
        let s = store("cas");
        let v1 = s.put("m", Bytes::from_static(b"one")).await.unwrap();
        // Matching version succeeds and returns the new version.
        let v2 = s
            .put_if_match("m", Bytes::from_static(b"two"), &v1)
            .await
            .unwrap();
        assert_ne!(v1, v2);
        assert_eq!(&s.get("m").await.unwrap()[..], b"two");
        // Stale version conflicts.
        assert!(matches!(
            s.put_if_match("m", Bytes::from_static(b"three"), &v1).await,
            Err(StorageError::VersionConflict { .. })
        ));
        assert_eq!(&s.get("m").await.unwrap()[..], b"two"); // unchanged
    }

    #[tokio::test]
    async fn put_if_not_exists() {
        let s = store("create");
        s.put_if_not_exists("c", Bytes::from_static(b"x"))
            .await
            .unwrap();
        assert!(matches!(
            s.put_if_not_exists("c", Bytes::from_static(b"y")).await,
            Err(StorageError::AlreadyExists(_))
        ));
    }

    #[tokio::test]
    async fn delete_then_list() {
        let s = store("list");
        s.put("d/1", Bytes::from_static(b"a")).await.unwrap();
        s.put("d/2", Bytes::from_static(b"bb")).await.unwrap();
        let mut keys: Vec<String> = s
            .list("d")
            .await
            .unwrap()
            .into_iter()
            .map(|m| m.key)
            .collect();
        keys.sort();
        assert_eq!(keys, vec!["d/1".to_string(), "d/2".to_string()]);

        s.delete("d/1").await.unwrap();
        let keys2: Vec<String> = s
            .list("d")
            .await
            .unwrap()
            .into_iter()
            .map(|m| m.key)
            .collect();
        assert_eq!(keys2, vec!["d/2".to_string()]);
        // Deleting missing is a no-op.
        s.delete("d/1").await.unwrap();
    }

    #[tokio::test]
    async fn rejects_path_traversal() {
        let s = store("traversal");
        assert!(s.put("../escape", Bytes::from_static(b"x")).await.is_err());
        assert!(s.get("a/../../etc/passwd").await.is_err());
    }
}
