//! Memory-mapped vector storage.
//!
//! Replaces `Vec<Vec<f32>>` with a flat file of f32 values backed by mmap.
//! Zero-copy reads, append-only writes, survives restarts.
//!
//! On-disk format (v2, current):
//!   [0..4)   magic "CMV2" (0x43 0x4D 0x56 0x32) — identifies the v2 layout
//!   [4..8)   u32 LE  dims   — vector dimensionality
//!   [8..16)  u64 LE  count  — number of vectors
//!   [16..)   count × dims × f32 LE — contiguous vector data
//!
//! Legacy format (v1, read-only for migration — never written anymore):
//!   [0..4)   u32 LE  dims
//!   [4..8)   u32 LE  count   — capped at u32::MAX (the bug v2 fixes)
//!   [8..)    count × dims × f32 LE
//!
//! `open` detects the layout by the 4-byte magic: present → v2, absent → v1.
//! v1 files keep working transparently; the next `create`/`append` rewrites the
//! header in v2 form. Widening `count` to u64 removes the silent truncation that
//! corrupted files past 4.29B vectors (`vectors.len() as u32`).

use memmap2::{Mmap, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Magic for the v2 (u64-count) header. ASCII "CMV2".
const MAGIC_V2: [u8; 4] = *b"CMV2";
/// v2 header: 4 (magic) + 4 (dims u32) + 8 (count u64).
const HEADER_SIZE_V2: usize = 16;
/// v1 header: 4 (dims u32) + 4 (count u32). Read-only legacy.
const HEADER_SIZE_V1: usize = 8;

/// Read-only mmap handle for vector searches.
pub struct MmapVectors {
    _file: File,
    mmap: Mmap,
    dims: usize,
    count: usize,
    /// Byte offset where vector data starts. v2 = 16, legacy v1 = 8. Stored so
    /// reads of a still-on-disk v1 file compute correct offsets without forcing
    /// a rewrite on open.
    header_size: usize,
    path: PathBuf,
}

unsafe impl Send for MmapVectors {}
unsafe impl Sync for MmapVectors {}

impl MmapVectors {
    /// Open an existing vectors file for reading. Detects v2 (magic-prefixed,
    /// u64 count) vs legacy v1 (u32 count) by the leading magic bytes.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let meta = file.metadata()?;
        if meta.len() < HEADER_SIZE_V1 as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "file too small for header",
            ));
        }

        let mmap = unsafe { MmapOptions::new().map(&file)? };

        let is_v2 = mmap.len() >= HEADER_SIZE_V2 && mmap[0..4] == MAGIC_V2;
        let (dims, count, header_size) = if is_v2 {
            let dims = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]) as usize;
            let count = u64::from_le_bytes([
                mmap[8], mmap[9], mmap[10], mmap[11], mmap[12], mmap[13], mmap[14], mmap[15],
            ]) as usize;
            (dims, count, HEADER_SIZE_V2)
        } else {
            // Legacy v1: [u32 dims][u32 count], data at offset 8.
            let dims = u32::from_le_bytes([mmap[0], mmap[1], mmap[2], mmap[3]]) as usize;
            let count = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]) as usize;
            (dims, count, HEADER_SIZE_V1)
        };

        // checked_mul: a corrupt header could make count*dims*4 wrap, letting
        // the size check pass and later reads panic out of bounds.
        let expected = count
            .checked_mul(dims)
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_add(header_size))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("corrupt header: {count} vectors × {dims} dims overflows"),
                )
            })?;
        if mmap.len() < expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "file size {} < expected {} for {} vectors × {} dims",
                    mmap.len(),
                    expected,
                    count,
                    dims
                ),
            ));
        }
        // A LARGER file is tolerated: a crash mid-append can leave orphan bytes
        // past the logical end (header count is authoritative). `append` writes
        // at the computed logical end, overwriting any orphan tail.

        Ok(Self {
            _file: file,
            mmap,
            dims,
            count,
            header_size,
            path: path.to_path_buf(),
        })
    }

    /// Create a new vectors file and write initial data.
    pub fn create(path: &Path, dims: usize, vectors: &[Vec<f32>]) -> io::Result<Self> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;

        // Header (v2): magic + u32 dims + u64 count.
        file.write_all(&MAGIC_V2)?;
        file.write_all(&(dims as u32).to_le_bytes())?;
        file.write_all(&(vectors.len() as u64).to_le_bytes())?;

        // Vector data. Wrong-length vectors are a HARD error (not debug_assert):
        // in release builds a silent mismatch would shift every subsequent
        // vector's offset and permanently corrupt the file.
        for vec in vectors {
            if vec.len() != dims {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("vector has {} dims, file expects {}", vec.len(), dims),
                ));
            }
            for &val in vec {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        file.flush()?;
        file.sync_all()?;
        drop(file);

        Self::open(path)
    }

    /// Append new vectors to the file and remap.
    ///
    /// A legacy v1 file (8-byte header, u32 count) is migrated to v2 in place on
    /// the first append: we rewrite the file with the v2 header so the count
    /// becomes u64 and can never wrap. Migration cost is one rewrite, paid once.
    pub fn append(&mut self, new_vectors: &[(u64, Vec<f32>)]) -> io::Result<()> {
        if new_vectors.is_empty() {
            return Ok(());
        }
        // Validate BEFORE any byte is written: one wrong-length vector would
        // shift all subsequent offsets and corrupt the file permanently. A hard
        // error, not debug_assert — release builds must be protected too.
        for (_, vec) in new_vectors {
            if vec.len() != self.dims {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("vector has {} dims, file expects {}", vec.len(), self.dims),
                ));
            }
        }

        let new_count = self.count + new_vectors.len();

        if self.header_size == HEADER_SIZE_V2 {
            // Fast path: already v2. Append data at the LOGICAL end computed
            // from the trusted header count — NOT SeekFrom::End(0). A prior
            // crash between data-write and count-update leaves orphan bytes
            // past the logical end; appending at physical EOF would misalign
            // every subsequent vector. Writing at the logical end overwrites
            // any orphan tail instead.
            let logical_end = (HEADER_SIZE_V2 + self.count * self.dims * 4) as u64;
            let mut file = OpenOptions::new().write(true).open(&self.path)?;
            file.seek(SeekFrom::Start(logical_end))?;
            for (_, vec) in new_vectors {
                for &val in vec {
                    file.write_all(&val.to_le_bytes())?;
                }
            }
            // Data durable BEFORE the count update: a crash in between leaves
            // orphan bytes (harmless, see above), never a count that promises
            // data that isn't there.
            file.sync_all()?;
            // v2 count lives at [8..16) (after 4-byte magic + 4-byte dims).
            file.seek(SeekFrom::Start(8))?;
            file.write_all(&(new_count as u64).to_le_bytes())?;
            file.flush()?;
            file.sync_all()?;
        } else {
            // Migration path: legacy v1 file. Rewrite as v2 (header shifts data
            // by 8 bytes, so a clean rewrite is simpler and safer than an
            // in-place shift). The v1 payload is the same raw LE f32 layout, so
            // stream it as ONE byte copy from the mmap — no per-vector
            // materialization (a large legacy file would otherwise be cloned
            // wholesale into RAM).
            let tmp = self.path.with_extension("vectors.tmp");
            {
                let mut file = OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(&tmp)?;
                file.write_all(&MAGIC_V2)?;
                file.write_all(&(self.dims as u32).to_le_bytes())?;
                file.write_all(&(new_count as u64).to_le_bytes())?;
                let data_end = HEADER_SIZE_V1 + self.count * self.dims * 4;
                file.write_all(&self.mmap[HEADER_SIZE_V1..data_end])?;
                for (_, vec) in new_vectors {
                    for &val in vec {
                        file.write_all(&val.to_le_bytes())?;
                    }
                }
                file.flush()?;
                file.sync_all()?;
            }
            std::fs::rename(&tmp, &self.path)?;
            self.header_size = HEADER_SIZE_V2;
        }

        // Remap
        let file = File::open(&self.path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        self.mmap = mmap;
        self._file = file;
        self.count = new_count;

        Ok(())
    }

    /// Get vector at index i as a slice. O(1), zero-copy.
    #[inline]
    pub fn get(&self, i: usize) -> &[f32] {
        debug_assert!(i < self.count);
        let byte_offset = self.header_size + i * self.dims * 4;
        let byte_end = byte_offset + self.dims * 4;
        let bytes = &self.mmap[byte_offset..byte_end];
        bytemuck::cast_slice(bytes)
    }

    /// Number of stored vectors.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Vector dimensionality.
    #[inline]
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Iterate all vectors as slices.
    pub fn iter(&self) -> impl Iterator<Item = &[f32]> {
        (0..self.count).map(move |i| self.get(i))
    }

    /// Collect all vectors into owned Vecs (for legacy code paths that need Vec<Vec<f32>>).
    pub fn to_vecs(&self) -> Vec<Vec<f32>> {
        self.iter().map(|s| s.to_vec()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        // Unique-ish per test name; tests here don't run the same name twice.
        p.push(format!("compass_mmap_test_{}.vectors", name));
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn create_writes_v2_and_roundtrips() {
        let path = tmp_path("v2_roundtrip");
        let vecs = vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let m = MmapVectors::create(&path, 3, &vecs).unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m.dims(), 3);
        assert_eq!(m.header_size, HEADER_SIZE_V2);
        assert_eq!(m.get(0), &[1.0, 2.0, 3.0]);
        assert_eq!(m.get(1), &[4.0, 5.0, 6.0]);

        // Header on disk starts with the v2 magic.
        let raw = std::fs::read(&path).unwrap();
        assert_eq!(&raw[0..4], &MAGIC_V2);

        // Reopen reads the same data.
        let reopened = MmapVectors::open(&path).unwrap();
        assert_eq!(reopened.len(), 2);
        assert_eq!(reopened.get(1), &[4.0, 5.0, 6.0]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn append_on_v2_bumps_u64_count() {
        let path = tmp_path("v2_append");
        let mut m = MmapVectors::create(&path, 2, &[vec![1.0f32, 1.0]]).unwrap();
        m.append(&[(10, vec![2.0, 2.0]), (11, vec![3.0, 3.0])])
            .unwrap();
        assert_eq!(m.len(), 3);
        assert_eq!(m.get(0), &[1.0, 1.0]);
        assert_eq!(m.get(2), &[3.0, 3.0]);

        // Count is a u64 at offset 8 in v2.
        let raw = std::fs::read(&path).unwrap();
        let count = u64::from_le_bytes(raw[8..16].try_into().unwrap());
        assert_eq!(count, 3);
        let _ = std::fs::remove_file(&path);
    }

    /// Hand-write a legacy v1 file ([u32 dims][u32 count][data]) and confirm it
    /// reads correctly (no magic) and migrates to v2 on append.
    #[test]
    fn legacy_v1_reads_then_migrates_on_append() {
        let path = tmp_path("v1_migrate");
        let dims = 2usize;
        let v0 = [7.0f32, 8.0];
        let v1 = [9.0f32, 10.0];
        let mut raw: Vec<u8> = Vec::new();
        raw.extend_from_slice(&(dims as u32).to_le_bytes());
        raw.extend_from_slice(&(2u32).to_le_bytes()); // v1 u32 count
        for v in [&v0, &v1] {
            for &x in v {
                raw.extend_from_slice(&x.to_le_bytes());
            }
        }
        std::fs::write(&path, &raw).unwrap();

        // Reads as v1 (8-byte header, no magic).
        let mut m = MmapVectors::open(&path).unwrap();
        assert_eq!(m.header_size, HEADER_SIZE_V1);
        assert_eq!(m.len(), 2);
        assert_eq!(m.get(0), &[7.0, 8.0]);
        assert_eq!(m.get(1), &[9.0, 10.0]);

        // Appending migrates the file to v2, preserving all data.
        m.append(&[(99, vec![11.0, 12.0])]).unwrap();
        assert_eq!(m.header_size, HEADER_SIZE_V2);
        assert_eq!(m.len(), 3);
        assert_eq!(m.get(0), &[7.0, 8.0]); // original data intact
        assert_eq!(m.get(2), &[11.0, 12.0]); // appended

        // On-disk header is now v2 with the magic.
        let after = std::fs::read(&path).unwrap();
        assert_eq!(&after[0..4], &MAGIC_V2);
        let count = u64::from_le_bytes(after[8..16].try_into().unwrap());
        assert_eq!(count, 3);
        let _ = std::fs::remove_file(&path);
    }

    /// Torn-append recovery: a crash after data lands but before the count
    /// update leaves orphan bytes past the logical end. The file must still
    /// open (count from header is authoritative) and the NEXT append must write
    /// at the logical end — overwriting the orphan, not appending after it.
    #[test]
    fn torn_append_orphan_bytes_are_overwritten() {
        let path = tmp_path("torn_append");
        let mut m = MmapVectors::create(&path, 2, &[vec![1.0f32, 2.0]]).unwrap();
        drop(m);

        // Simulate the torn state: raw garbage appended, header count still 1.
        {
            use std::io::Write;
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(&[0xAB; 5]).unwrap(); // partial stride, worst case
        }

        // Opens fine — the larger-than-expected file is tolerated.
        m = MmapVectors::open(&path).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m.get(0), &[1.0, 2.0]);

        // Next append lands at the LOGICAL end, clobbering the orphan bytes.
        m.append(&[(9, vec![3.0, 4.0])]).unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m.get(0), &[1.0, 2.0]);
        assert_eq!(m.get(1), &[3.0, 4.0], "vector must not be misaligned");
        let _ = std::fs::remove_file(&path);
    }

    /// Wrong-length vectors are rejected with an error BEFORE any byte is
    /// written (in release builds too — this was a debug_assert).
    #[test]
    fn wrong_dims_rejected_not_corrupting() {
        let path = tmp_path("wrong_dims");
        let mut m = MmapVectors::create(&path, 3, &[vec![1.0f32, 2.0, 3.0]]).unwrap();
        let err = m.append(&[(1, vec![1.0, 2.0])]).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        // Nothing was written; the file is intact.
        assert_eq!(m.len(), 1);
        let reopened = MmapVectors::open(&path).unwrap();
        assert_eq!(reopened.len(), 1);
        assert_eq!(reopened.get(0), &[1.0, 2.0, 3.0]);

        // create() rejects too.
        let path2 = tmp_path("wrong_dims_create");
        assert!(MmapVectors::create(&path2, 3, &[vec![1.0f32]]).is_err());
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&path2);
    }
}
