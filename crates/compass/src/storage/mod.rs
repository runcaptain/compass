//! Storage — the source-of-truth abstraction.
//!
//! A single object-keyed, range-readable, CAS-capable surface that every
//! persistent component can bind to. Two backends implement it:
//!
//!   - [`local::LocalDiskStorage`] (default): the local filesystem. This *is*
//!     today's behavior — shipping the trait is a pure additive seam, no change
//!     to existing embedded deployments.
//!   - `ObjectStoreBackend` (later step): S3 / GCS / Azure / R2 / MinIO via the
//!     `object_store` crate, opt-in via config.
//!
//! Design principle: source of truth is the `Storage` backend; RAM is a cache
//! in front of it. The conditional-write methods (`put_if_match`,
//! `put_if_not_exists`) are the commit primitive the object-storage LSM
//! (WAL + manifest + compaction) will use to atomically swap the manifest
//! without a lock. `get_range` is the primitive that lets large segments be
//! served without pulling the whole object into RAM.
//!
//! Nothing routes through this trait yet — it is introduced standalone and
//! wired into the engine incrementally in later steps.

pub mod local;
pub mod lsm;
#[cfg(feature = "object-storage")]
pub mod object_store_backend;
#[cfg(all(test, feature = "object-storage"))]
mod s3_integration_tests;

use async_trait::async_trait;
use bytes::Bytes;
use std::ops::Range;
use std::sync::Arc;
use thiserror::Error;

/// Opaque version token for compare-and-swap writes. Carries BOTH the HTTP
/// ETag (S3/Azure/MinIO; content hash on local disk) and the provider-native
/// version (the GCS object *generation* — GCS conditional writes require it and
/// ignore the ETag). Treat as opaque — only equality matters to callers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version {
    pub e_tag: String,
    pub version: Option<String>,
}

impl Version {
    /// A version identified by ETag/content-hash only (local disk, tests).
    pub fn etag(e: impl Into<String>) -> Self {
        Self {
            e_tag: e.into(),
            version: None,
        }
    }

    /// True when the token carries no usable precondition (CAS must refuse it).
    pub fn is_empty(&self) -> bool {
        self.e_tag.is_empty() && self.version.as_deref().is_none_or(str::is_empty)
    }
}

/// Metadata about a stored object, returned by `list`.
#[derive(Debug, Clone)]
pub struct ObjectMeta {
    pub key: String,
    pub size: u64,
    pub version: Option<Version>,
}

/// Errors from a storage backend.
#[derive(Debug, Error)]
pub enum StorageError {
    /// The requested key does not exist.
    #[error("not found: {0}")]
    NotFound(String),

    /// A conditional write failed because the current version did not match the
    /// expected one (someone else wrote concurrently). The caller re-reads and
    /// retries. This is the CAS-conflict signal the LSM manifest commit depends on.
    #[error("version conflict on {key}")]
    VersionConflict { key: String },

    /// A `put_if_not_exists` failed because the key already exists.
    #[error("already exists: {0}")]
    AlreadyExists(String),

    /// Underlying I/O or backend error.
    #[error("storage io error: {0}")]
    Io(String),

    /// The requested byte range is invalid for the object.
    #[error("invalid range {start}..{end} for object of size {size}")]
    InvalidRange { start: u64, end: u64, size: u64 },
}

/// Source-of-truth storage. Object-keyed, range-readable, CAS-capable.
///
/// Keys are `/`-delimited logical paths (e.g. `mycoll/manifest`). Backends map
/// them onto a filesystem layout or an object-store prefix. Implementations
/// must be `Send + Sync`.
#[async_trait]
pub trait Storage: Send + Sync {
    /// Whole-object read.
    async fn get(&self, key: &str) -> Result<Bytes, StorageError>;

    /// Range read — fetch only `range` bytes of the object. The primitive that
    /// makes large segments servable without loading the whole object.
    async fn get_range(&self, key: &str, range: Range<u64>) -> Result<Bytes, StorageError>;

    /// Read the object together with its current version, for a CAS cycle.
    async fn get_versioned(&self, key: &str) -> Result<(Bytes, Version), StorageError>;

    /// Unconditional write. Returns the new version.
    async fn put(&self, key: &str, bytes: Bytes) -> Result<Version, StorageError>;

    /// Compare-and-swap write: succeed only if the current version still matches
    /// `expected`. On mismatch returns [`StorageError::VersionConflict`]. This is
    /// how the LSM manifest is committed atomically without a lock.
    async fn put_if_match(
        &self,
        key: &str,
        bytes: Bytes,
        expected: &Version,
    ) -> Result<Version, StorageError>;

    /// Create-only write: succeed only if the key does not already exist. On
    /// conflict returns [`StorageError::AlreadyExists`]. Used for lease
    /// acquisition and first-write.
    async fn put_if_not_exists(&self, key: &str, bytes: Bytes) -> Result<Version, StorageError>;

    /// Delete an object. Deleting a missing key is a no-op (Ok).
    async fn delete(&self, key: &str) -> Result<(), StorageError>;

    /// List objects under a key prefix.
    async fn list(&self, prefix: &str) -> Result<Vec<ObjectMeta>, StorageError>;

    /// List the immediate "directories" (common prefixes) under `prefix` —
    /// the delimiter-listing primitive. Namespace discovery uses this so boot
    /// cost is O(namespaces), not O(total objects in the bucket).
    ///
    /// Returned names are the bare child-prefix components (no trailing `/`).
    /// The default implementation derives them from a full `list` — correct
    /// but O(objects); real backends override with a native delimiter listing.
    async fn list_dirs(&self, prefix: &str) -> Result<Vec<String>, StorageError> {
        let norm = prefix.trim_end_matches('/');
        let objects = self.list(norm).await?;
        let mut dirs: Vec<String> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for obj in objects {
            let rest = if norm.is_empty() {
                obj.key.as_str()
            } else {
                match obj.key.strip_prefix(norm).and_then(|r| r.strip_prefix('/')) {
                    Some(r) => r,
                    None => continue,
                }
            };
            if let Some((child, _)) = rest.split_once('/') {
                if !child.is_empty() && seen.insert(child.to_string()) {
                    dirs.push(child.to_string());
                }
            }
        }
        Ok(dirs)
    }

    /// Whether an object exists.
    async fn exists(&self, key: &str) -> Result<bool, StorageError> {
        match self.get_versioned(key).await {
            Ok(_) => Ok(true),
            Err(StorageError::NotFound(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Human-readable backend label for tracing (e.g. "local-disk").
    fn backend_name(&self) -> &'static str;
}

/// Select a storage backend from the `COMPASS_STORAGE` env var.
///
/// - unset or `local` -> [`local::LocalDiskStorage`] rooted at `data_dir`
///   (the default; local-first deployments need no config).
/// - `s3://…` / `gs://…` / `az://…` / `memory://` -> the object-storage backend
///   (requires the `object-storage` feature; errors otherwise).
///
/// Hard either/or: a deployment is local OR object-storage, never both.
pub fn from_config(data_dir: &std::path::Path) -> Result<Arc<dyn Storage>, StorageError> {
    let cfg = std::env::var("COMPASS_STORAGE").unwrap_or_else(|_| "local".to_string());

    if cfg.is_empty() || cfg == "local" {
        let backend =
            local::LocalDiskStorage::new(data_dir).map_err(|e| StorageError::Io(e.to_string()))?;
        tracing::info!("storage backend = local-disk ({})", data_dir.display());
        return Ok(Arc::new(backend));
    }

    // Non-local config requests object storage.
    object_storage_from_url(&cfg)
}

#[cfg(feature = "object-storage")]
fn object_storage_from_url(cfg: &str) -> Result<Arc<dyn Storage>, StorageError> {
    let backend = object_store_backend::ObjectStoreBackend::from_url(cfg)?;
    tracing::info!("storage backend = {} ({})", backend.backend_name(), cfg);
    Ok(Arc::new(backend))
}

/// Startup connectivity check: round-trip a probe object (put → get → delete) so
/// mis-configured credentials, wrong bucket, region, or endpoint fail loudly at
/// boot instead of on the first ingest. Verifies read+write+delete permissions.
///
/// The probe key is a single root-level object (no directory component, so the
/// local backend leaves no empty directory behind) and is deleted after.
pub async fn verify(storage: &dyn Storage) -> Result<(), StorageError> {
    let key = ".compass-probe-startup";
    let payload = Bytes::from_static(b"compass-connectivity-check");

    storage.put(key, payload.clone()).await.map_err(|e| {
        StorageError::Io(format!(
            "storage write check failed ({}): {e}",
            storage.backend_name()
        ))
    })?;
    let got = storage.get(key).await.map_err(|e| {
        StorageError::Io(format!(
            "storage read check failed ({}): {e}",
            storage.backend_name()
        ))
    })?;
    if got != payload {
        return Err(StorageError::Io(
            "storage round-trip returned unexpected bytes".into(),
        ));
    }
    // Best-effort cleanup; a failed delete shouldn't fail startup.
    let _ = storage.delete(key).await;
    Ok(())
}

#[cfg(not(feature = "object-storage"))]
fn object_storage_from_url(cfg: &str) -> Result<Arc<dyn Storage>, StorageError> {
    Err(StorageError::Io(format!(
        "COMPASS_STORAGE='{cfg}' requests object storage, but this binary was built \
         without the `object-storage` feature. Rebuild with --features object-storage."
    )))
}
