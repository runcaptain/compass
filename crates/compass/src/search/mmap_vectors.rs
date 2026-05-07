//! Memory-mapped vector storage.
//!
//! Replaces `Vec<Vec<f32>>` with a flat file of f32 values backed by mmap.
//! Zero-copy reads, append-only writes, survives restarts.
//!
//! File format:
//!   [0..4)  u32 LE  dims   — vector dimensionality
//!   [4..8)  u32 LE  count  — number of vectors
//!   [8..)   count × dims × f32 LE — contiguous vector data

use memmap2::{Mmap, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const HEADER_SIZE: usize = 8;

/// Read-only mmap handle for vector searches.
pub struct MmapVectors {
    _file: File,
    mmap: Mmap,
    dims: usize,
    count: usize,
    path: PathBuf,
}

unsafe impl Send for MmapVectors {}
unsafe impl Sync for MmapVectors {}

impl MmapVectors {
    /// Open an existing vectors file for reading.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let meta = file.metadata()?;
        if meta.len() < HEADER_SIZE as u64 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small for header"));
        }

        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let dims = u32::from_le_bytes([mmap[0], mmap[1], mmap[2], mmap[3]]) as usize;
        let count = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]) as usize;

        let expected = HEADER_SIZE + count * dims * 4;
        if mmap.len() < expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("file size {} < expected {} for {} vectors × {} dims", mmap.len(), expected, count, dims),
            ));
        }

        Ok(Self { _file: file, mmap, dims, count, path: path.to_path_buf() })
    }

    /// Create a new vectors file and write initial data.
    pub fn create(path: &Path, dims: usize, vectors: &[Vec<f32>]) -> io::Result<Self> {
        let mut file = OpenOptions::new().create(true).write(true).truncate(true).open(path)?;

        // Header
        file.write_all(&(dims as u32).to_le_bytes())?;
        file.write_all(&(vectors.len() as u32).to_le_bytes())?;

        // Vector data
        for vec in vectors {
            debug_assert_eq!(vec.len(), dims);
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
    pub fn append(&mut self, new_vectors: &[(u64, Vec<f32>)]) -> io::Result<()> {
        if new_vectors.is_empty() {
            return Ok(());
        }

        let new_count = self.count + new_vectors.len();

        {
            let mut file = OpenOptions::new().write(true).open(&self.path)?;

            // Seek to end and write new vector data
            file.seek(SeekFrom::End(0))?;
            for (_, vec) in new_vectors {
                debug_assert_eq!(vec.len(), self.dims);
                for &val in vec {
                    file.write_all(&val.to_le_bytes())?;
                }
            }

            // Update count in header
            file.seek(SeekFrom::Start(4))?;
            file.write_all(&(new_count as u32).to_le_bytes())?;
            file.flush()?;
            file.sync_all()?;
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
        let byte_offset = HEADER_SIZE + i * self.dims * 4;
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
