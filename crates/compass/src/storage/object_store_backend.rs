//! `ObjectStoreBackend` — the opt-in [`Storage`] backend over object storage.
//!
//! Wraps the `object_store` crate, which covers S3, GCS, Azure, R2/MinIO
//! (S3-compatible), local filesystem, and in-memory behind one interface — so
//! "support every provider" is one dependency. Enabled by the `object-storage`
//! cargo feature; off by default so local-first builds stay lean.
//!
//! Mapping onto our [`Storage`] trait:
//!   - `get_range` -> `get_opts` with `GetRange::Bounded` (true byte-range read).
//!   - CAS (`put_if_match`) -> `PutMode::Update` with BOTH the ETag and the
//!     provider version: S3/Azure/MinIO match on the ETag, while GCS requires
//!     the object *generation* (its provider version) and ignores the ETag.
//!   - `put_if_not_exists` -> `PutMode::Create`.
//!   - [`Version`] carries both tokens; see `storage::Version`.
//!
//! Selection is by URL scheme in `COMPASS_STORAGE` (`s3://`, `gs://`, `az://`,
//! or `memory://` for tests). A path after the bucket (`s3://bucket/prefix`)
//! scopes all keys under that prefix via `PrefixStore`, so multiple deployments
//! can share a bucket without clobbering each other. Credentials come from the
//! standard provider env vars that `object_store`'s builders read (AWS_*,
//! GOOGLE_*, AZURE_*), plus optional `COMPASS_S3_ENDPOINT` /
//! `COMPASS_S3_ALLOW_HTTP` for MinIO/R2.

use super::{ObjectMeta, Storage, StorageError, Version};
use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use object_store::path::Path as OsPath;
use object_store::{
    Error as OsError, GetOptions, GetRange, ObjectStore, ObjectStoreExt, PutMode, PutOptions,
    UpdateVersion,
};
use std::ops::Range;
use std::sync::Arc;

pub struct ObjectStoreBackend {
    inner: Arc<dyn ObjectStore>,
    label: &'static str,
}

impl ObjectStoreBackend {
    /// Build a backend from a `COMPASS_STORAGE` URL. Supported schemes:
    ///   - `s3://bucket[/prefix]`      (AWS S3 / MinIO / R2 — S3-compatible)
    ///   - `gs://bucket[/prefix]`      (Google Cloud Storage)
    ///   - `az://container[/prefix]`   (Azure Blob)
    ///   - `memory://`                 (in-memory; tests)
    ///
    /// A `/prefix` after the bucket scopes every key under it (implemented with
    /// `object_store::prefix::PrefixStore`), so two deployments can safely
    /// share one bucket.
    ///
    /// Credentials are read from the standard provider env vars by the
    /// `object_store` builders. For S3-compatible endpoints (MinIO/R2) set
    /// `COMPASS_S3_ENDPOINT` and `COMPASS_S3_ALLOW_HTTP=true` as needed.
    pub fn from_url(url: &str) -> Result<Self, StorageError> {
        let err = |m: String| StorageError::Io(m);

        if url == "memory://" || url == "memory:///" {
            return Ok(Self {
                inner: Arc::new(object_store::memory::InMemory::new()),
                label: "object-store:memory",
            });
        }

        let (scheme, rest) = url
            .split_once("://")
            .ok_or_else(|| err(format!("COMPASS_STORAGE '{url}' is not a URL")))?;
        let (bucket, prefix) = match rest.split_once('/') {
            Some((b, p)) => (b, p.trim_matches('/')),
            None => (rest, ""),
        };
        if bucket.is_empty() {
            return Err(err(format!("COMPASS_STORAGE '{url}' has no bucket")));
        }

        // Honor a bucket-internal prefix: scope every key under it. Without
        // this, `s3://bucket/prod` and `s3://bucket/staging` would silently
        // write to the same root keys and clobber each other. PrefixStore must
        // wrap the CONCRETE store type (Arc<dyn ObjectStore> doesn't implement
        // ObjectStore in object_store 0.11), hence the generic helper.
        fn wrap<S: ObjectStore>(store: S, prefix: &str) -> Arc<dyn ObjectStore> {
            if prefix.is_empty() {
                Arc::new(store)
            } else {
                Arc::new(object_store::prefix::PrefixStore::new(
                    store,
                    OsPath::from(prefix),
                ))
            }
        }

        let (inner, label): (Arc<dyn ObjectStore>, &'static str) = match scheme {
            "s3" => {
                let mut b = object_store::aws::AmazonS3Builder::from_env().with_bucket_name(bucket);
                if let Ok(ep) = std::env::var("COMPASS_S3_ENDPOINT") {
                    if !ep.is_empty() {
                        b = b.with_endpoint(ep);
                    }
                }
                if std::env::var("COMPASS_S3_ALLOW_HTTP")
                    .map(|v| v == "true" || v == "1")
                    .unwrap_or(false)
                {
                    b = b.with_allow_http(true);
                }
                // Conditional put (ETag CAS) — required for the LSM manifest commit.
                b = b.with_conditional_put(object_store::aws::S3ConditionalPut::ETagMatch);
                (
                    wrap(b.build().map_err(|e| err(e.to_string()))?, prefix),
                    "object-store:s3",
                )
            }
            "gs" => {
                let b = object_store::gcp::GoogleCloudStorageBuilder::from_env()
                    .with_bucket_name(bucket);
                (
                    wrap(b.build().map_err(|e| err(e.to_string()))?, prefix),
                    "object-store:gcs",
                )
            }
            "az" => {
                let b = object_store::azure::MicrosoftAzureBuilder::from_env()
                    .with_container_name(bucket);
                (
                    wrap(b.build().map_err(|e| err(e.to_string()))?, prefix),
                    "object-store:azure",
                )
            }
            other => return Err(err(format!("unsupported storage scheme '{other}://'"))),
        };
        Ok(Self { inner, label })
    }

    /// Construct directly from an existing object store (used by tests).
    pub fn from_store(inner: Arc<dyn ObjectStore>, label: &'static str) -> Self {
        Self { inner, label }
    }
}

fn map_os_err(key: &str, e: OsError) -> StorageError {
    match e {
        OsError::NotFound { .. } => StorageError::NotFound(key.to_string()),
        OsError::AlreadyExists { .. } => StorageError::AlreadyExists(key.to_string()),
        OsError::Precondition { .. } => StorageError::VersionConflict {
            key: key.to_string(),
        },
        other => StorageError::Io(other.to_string()),
    }
}

#[async_trait]
impl Storage for ObjectStoreBackend {
    async fn get(&self, key: &str) -> Result<Bytes, StorageError> {
        let path = OsPath::from(key);
        let res = self
            .inner
            .get(&path)
            .await
            .map_err(|e| map_os_err(key, e))?;
        res.bytes().await.map_err(|e| map_os_err(key, e))
    }

    async fn get_range(&self, key: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        let path = OsPath::from(key);
        let opts = GetOptions {
            range: Some(GetRange::Bounded(range)),
            ..Default::default()
        };
        let res = self
            .inner
            .get_opts(&path, opts)
            .await
            .map_err(|e| map_os_err(key, e))?;
        res.bytes().await.map_err(|e| map_os_err(key, e))
    }

    async fn get_versioned(&self, key: &str) -> Result<(Bytes, Version), StorageError> {
        let path = OsPath::from(key);
        let res = self
            .inner
            .get_opts(&path, GetOptions::default())
            .await
            .map_err(|e| map_os_err(key, e))?;
        let e_tag = res.meta.e_tag.clone();
        let provider_version = res.meta.version.clone();
        let bytes = res.bytes().await.map_err(|e| map_os_err(key, e))?;
        Ok((
            bytes,
            Version {
                e_tag: e_tag.unwrap_or_default(),
                version: provider_version,
            },
        ))
    }

    async fn put(&self, key: &str, bytes: Bytes) -> Result<Version, StorageError> {
        let path = OsPath::from(key);
        let res = self
            .inner
            .put(&path, bytes.into())
            .await
            .map_err(|e| map_os_err(key, e))?;
        Ok(Version {
            e_tag: res.e_tag.unwrap_or_default(),
            version: res.version,
        })
    }

    async fn put_if_match(
        &self,
        key: &str,
        bytes: Bytes,
        expected: &Version,
    ) -> Result<Version, StorageError> {
        // Guard: an empty token means the store gave us no real precondition.
        // A CAS with an empty precondition would false-match ANY object —
        // corrupting the manifest-commit guarantee. Refuse loudly instead.
        if expected.is_empty() {
            return Err(StorageError::Io(format!(
                "refusing CAS on '{key}': no ETag/version available from this \
                 backend; conditional writes require ETag or generation support \
                 (S3/GCS/Azure with conditional-put enabled)"
            )));
        }
        let path = OsPath::from(key);
        // Pass BOTH tokens through: S3/Azure/MinIO match on the ETag, while GCS
        // requires the provider version (object generation) and errors if it is
        // missing.
        let opts = PutOptions {
            mode: PutMode::Update(UpdateVersion {
                e_tag: (!expected.e_tag.is_empty()).then(|| expected.e_tag.clone()),
                version: expected.version.clone(),
            }),
            ..Default::default()
        };
        let res = self
            .inner
            .put_opts(&path, bytes.into(), opts)
            .await
            .map_err(|e| map_os_err(key, e))?;
        Ok(Version {
            e_tag: res.e_tag.unwrap_or_default(),
            version: res.version,
        })
    }

    async fn put_if_not_exists(&self, key: &str, bytes: Bytes) -> Result<Version, StorageError> {
        let path = OsPath::from(key);
        let opts = PutOptions {
            mode: PutMode::Create,
            ..Default::default()
        };
        let res = self
            .inner
            .put_opts(&path, bytes.into(), opts)
            .await
            .map_err(|e| map_os_err(key, e))?;
        Ok(Version {
            e_tag: res.e_tag.unwrap_or_default(),
            version: res.version,
        })
    }

    async fn put_large(&self, key: &str, bytes: Bytes) -> Result<Version, StorageError> {
        // Multipart for anything past a conservative threshold; small objects
        // take the single-PUT fast path.
        const PART: usize = 16 * 1024 * 1024;
        if bytes.len() <= PART {
            return self.put(key, bytes).await;
        }
        let path = OsPath::from(key);
        let upload = self
            .inner
            .put_multipart(&path)
            .await
            .map_err(|e| map_os_err(key, e))?;
        let mut w = object_store::WriteMultipart::new(upload);
        for part in bytes.chunks(PART) {
            w.write(part);
        }
        w.finish().await.map_err(|e| map_os_err(key, e))?;
        // Multipart results don't return an ETag through this helper; segments
        // are immutable + UUID-keyed, so no CAS token is needed on them.
        Ok(Version::etag(String::new()))
    }

    async fn delete(&self, key: &str) -> Result<(), StorageError> {
        let path = OsPath::from(key);
        match self.inner.delete(&path).await {
            Ok(()) => Ok(()),
            Err(OsError::NotFound { .. }) => Ok(()),
            Err(e) => Err(map_os_err(key, e)),
        }
    }

    async fn list(&self, prefix: &str) -> Result<Vec<ObjectMeta>, StorageError> {
        let os_prefix = if prefix.is_empty() {
            None
        } else {
            Some(OsPath::from(prefix))
        };
        let mut stream = self.inner.list(os_prefix.as_ref());
        let mut out = Vec::new();
        while let Some(item) = stream.next().await {
            let meta = item.map_err(|e| map_os_err(prefix, e))?;
            out.push(ObjectMeta {
                key: meta.location.to_string(),
                size: meta.size,
                version: meta.e_tag.map(Version::etag),
            });
        }
        Ok(out)
    }

    async fn list_dirs(&self, prefix: &str) -> Result<Vec<String>, StorageError> {
        // Native delimiter listing: one request per page of common prefixes,
        // independent of how many objects live under them.
        let os_prefix = if prefix.is_empty() {
            None
        } else {
            Some(OsPath::from(prefix.trim_end_matches('/')))
        };
        let res = self
            .inner
            .list_with_delimiter(os_prefix.as_ref())
            .await
            .map_err(|e| map_os_err(prefix, e))?;
        Ok(res
            .common_prefixes
            .into_iter()
            .filter_map(|p| p.parts().last().map(|part| part.as_ref().to_string()))
            .collect())
    }

    fn backend_name(&self) -> &'static str {
        self.label
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem() -> ObjectStoreBackend {
        ObjectStoreBackend::from_store(
            Arc::new(object_store::memory::InMemory::new()),
            "object-store:memory",
        )
    }

    #[tokio::test]
    async fn put_get_roundtrip() {
        let s = mem();
        s.put("a/b", Bytes::from_static(b"hello")).await.unwrap();
        assert_eq!(&s.get("a/b").await.unwrap()[..], b"hello");
        assert!(s.exists("a/b").await.unwrap());
    }

    #[tokio::test]
    async fn range_read() {
        let s = mem();
        s.put("k", Bytes::from_static(b"0123456789")).await.unwrap();
        assert_eq!(&s.get_range("k", 2..5).await.unwrap()[..], b"234");
    }

    #[tokio::test]
    async fn missing_is_not_found() {
        let s = mem();
        assert!(matches!(
            s.get("nope").await,
            Err(StorageError::NotFound(_))
        ));
        assert!(!s.exists("nope").await.unwrap());
    }

    #[tokio::test]
    async fn cas_roundtrip() {
        let s = mem();
        let v1 = s.put("m", Bytes::from_static(b"one")).await.unwrap();
        let v2 = s
            .put_if_match("m", Bytes::from_static(b"two"), &v1)
            .await
            .unwrap();
        assert_ne!(v1, v2);
        assert_eq!(&s.get("m").await.unwrap()[..], b"two");
        // Stale CAS conflicts.
        assert!(matches!(
            s.put_if_match("m", Bytes::from_static(b"x"), &v1).await,
            Err(StorageError::VersionConflict { .. })
        ));
    }

    #[tokio::test]
    async fn create_only() {
        let s = mem();
        s.put_if_not_exists("c", Bytes::from_static(b"x"))
            .await
            .unwrap();
        assert!(matches!(
            s.put_if_not_exists("c", Bytes::from_static(b"y")).await,
            Err(StorageError::AlreadyExists(_))
        ));
    }

    // A CAS with an empty version token must be refused, not silently matched.
    #[tokio::test]
    async fn cas_with_empty_version_is_refused() {
        let s = mem();
        s.put("k", Bytes::from_static(b"v")).await.unwrap();
        let empty = Version::etag(String::new());
        let r = s.put_if_match("k", Bytes::from_static(b"x"), &empty).await;
        assert!(
            matches!(r, Err(StorageError::Io(_))),
            "empty-token CAS must be refused, got {r:?}"
        );
    }

    // A provider version (GCS generation) alone is a valid CAS token even with
    // no ETag — the F1 (GCS) regression shape.
    #[test]
    fn version_with_generation_only_is_not_empty() {
        let v = Version {
            e_tag: String::new(),
            version: Some("1234567890".into()),
        };
        assert!(!v.is_empty());
        assert!(Version::etag("abc").e_tag == "abc");
        assert!(Version::etag("").is_empty());
    }

    #[tokio::test]
    async fn list_and_delete() {
        let s = mem();
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
    }

    #[test]
    fn from_url_memory() {
        let s = ObjectStoreBackend::from_url("memory://").unwrap();
        assert_eq!(s.backend_name(), "object-store:memory");
    }

    // B-F2 regression: a bucket-internal prefix must scope all keys, so two
    // deployments sharing a bucket cannot clobber each other. Verified via the
    // PrefixStore wrapping: keys written through a prefixed backend land under
    // the prefix in the underlying store.
    #[tokio::test]
    async fn url_prefix_scopes_keys() {
        // Two LocalFileSystem instances over one dir: a prefixed view and a raw
        // view (Arc<InMemory> can't be shared — Arc<T> isn't ObjectStore in 0.11).
        let mut root = std::env::temp_dir();
        root.push(format!("compass_prefix_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let prefixed = ObjectStoreBackend::from_store(
            Arc::new(object_store::prefix::PrefixStore::new(
                object_store::local::LocalFileSystem::new_with_prefix(&root).unwrap(),
                OsPath::from("tenants/prod"),
            )),
            "object-store:memory",
        );
        prefixed
            .put("ns/manifest", Bytes::from_static(b"m"))
            .await
            .unwrap();

        // The raw view sees the key under the prefix, not at the root.
        let raw = ObjectStoreBackend::from_store(
            Arc::new(object_store::local::LocalFileSystem::new_with_prefix(&root).unwrap()),
            "object-store:memory",
        );
        assert!(raw.exists("tenants/prod/ns/manifest").await.unwrap());
        assert!(!raw.exists("ns/manifest").await.unwrap());
        let _ = std::fs::remove_dir_all(&root);
    }
}
