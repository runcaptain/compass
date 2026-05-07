// collections/relationships.rs — In-memory relationship store with disk persistence.
//
// Stores parent-child and sibling relationships between chunks.
// Two indexes for fast lookups:
//   - Forward: chunk_id -> (parent_id, group_id)
//   - Reverse: group_id -> Vec<chunk_id>  (for sibling lookups, O(1) not O(n))
//
// Memory usage: ~40 bytes per chunk. 20M chunks = ~800MB. Acceptable for a server.
// Flushed to disk as a simple binary file on each ingest commit.

use std::collections::HashMap;
use std::path::Path;

/// A chunk's relationship data.
#[derive(Debug, Clone)]
struct Relationship {
    parent_id: Option<u64>,
    group_id: Option<String>,
}

/// In-memory store for document relationships.
pub struct RelationshipStore {
    /// Forward index: chunk_id -> relationship data
    forward: HashMap<u64, Relationship>,
    /// Reverse index: group_id -> list of chunk_ids (for O(1) sibling lookup)
    groups: HashMap<String, Vec<u64>>,
}

impl RelationshipStore {
    pub fn new() -> Self {
        Self {
            forward: HashMap::new(),
            groups: HashMap::new(),
        }
    }

    /// Add a relationship for a chunk. Called during ingest.
    pub fn add(&mut self, chunk_id: u64, parent_id: Option<u64>, group_id: Option<String>) {
        // Update forward index
        self.forward.insert(
            chunk_id,
            Relationship {
                parent_id,
                group_id: group_id.clone(),
            },
        );

        // Update reverse index (group_id -> sibling list)
        if let Some(ref gid) = group_id {
            self.groups.entry(gid.clone()).or_default().push(chunk_id);
        }
    }

    /// Look up the parent chunk ID for a given chunk.
    pub fn get_parent(&self, chunk_id: u64) -> Option<u64> {
        self.forward.get(&chunk_id).and_then(|r| r.parent_id)
    }

    /// Look up all sibling chunk IDs (chunks sharing the same group_id).
    /// Returns an empty slice if the chunk has no group_id or no siblings.
    pub fn get_siblings(&self, chunk_id: u64) -> Vec<u64> {
        let group_id = match self.forward.get(&chunk_id) {
            Some(r) => match &r.group_id {
                Some(gid) => gid,
                None => return Vec::new(),
            },
            None => return Vec::new(),
        };

        // Return all chunks in the same group, excluding the chunk itself
        self.groups
            .get(group_id)
            .map(|siblings| {
                siblings
                    .iter()
                    .filter(|&&sid| sid != chunk_id)
                    .copied()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Build lookup maps needed by the scoring pipeline.
    /// Returns (parent_ids, sibling_map) for a set of candidate chunk_ids.
    pub fn build_scoring_maps(
        &self,
        candidate_ids: &[u64],
    ) -> (HashMap<u64, u64>, HashMap<u64, Vec<u64>>) {
        let mut parent_ids = HashMap::new();
        let mut sibling_map = HashMap::new();

        for &cid in candidate_ids {
            if let Some(pid) = self.get_parent(cid) {
                parent_ids.insert(cid, pid);
            }
            let siblings = self.get_siblings(cid);
            if !siblings.is_empty() {
                sibling_map.insert(cid, siblings);
            }
        }

        (parent_ids, sibling_map)
    }

    /// Resolve batch parent references: maps client_id -> Compass chunk_id,
    /// then resolves parent_ref fields to parent_id.
    ///
    /// Returns a list of (parent_id, group_id) tuples parallel to the input,
    /// with parent_ref resolved to actual Compass IDs.
    pub fn resolve_batch_refs(
        client_id_map: &HashMap<String, u64>,
        parent_ids: &[Option<u64>],
        parent_refs: &[Option<String>],
        group_ids: &[Option<String>],
    ) -> Vec<(Option<u64>, Option<String>)> {
        parent_ids
            .iter()
            .zip(parent_refs.iter())
            .zip(group_ids.iter())
            .map(|((pid, pref), gid)| {
                // Resolve parent_ref to parent_id if present
                let resolved_parent = if let Some(ref pr) = pref {
                    client_id_map.get(pr).copied().or(*pid)
                } else {
                    *pid
                };
                (resolved_parent, gid.clone())
            })
            .collect()
    }

    /// Total number of tracked relationships.
    pub fn len(&self) -> usize {
        self.forward.len()
    }

    // ── Disk persistence ─────────────────────────────────────────────────
    // Simple binary format:
    //   [u32 count]
    //   For each entry:
    //     [u64 chunk_id] [u8 has_parent] [u64 parent_id] [u16 group_len] [group_bytes]

    /// Save the relationship store to a binary file.
    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&(self.forward.len() as u32).to_le_bytes());

        for (&chunk_id, rel) in &self.forward {
            buf.extend_from_slice(&chunk_id.to_le_bytes());

            // Parent ID (optional)
            match rel.parent_id {
                Some(pid) => {
                    buf.push(1);
                    buf.extend_from_slice(&pid.to_le_bytes());
                }
                None => {
                    buf.push(0);
                    buf.extend_from_slice(&0u64.to_le_bytes());
                }
            }

            // Group ID (optional, variable length string)
            match &rel.group_id {
                Some(gid) => {
                    let bytes = gid.as_bytes();
                    buf.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
                    buf.extend_from_slice(bytes);
                }
                None => {
                    buf.extend_from_slice(&0u16.to_le_bytes());
                }
            }
        }

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, buf)?;
        Ok(())
    }

    /// Load the relationship store from a binary file.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        if !path.exists() {
            return Ok(Self::new());
        }

        let buf = std::fs::read(path)?;
        if buf.len() < 4 {
            return Ok(Self::new());
        }

        let count = u32::from_le_bytes(buf[0..4].try_into()?) as usize;
        let mut store = Self {
            forward: HashMap::with_capacity(count),
            groups: HashMap::new(),
        };

        let mut pos = 4;
        for _ in 0..count {
            if pos + 17 > buf.len() {
                break;
            }

            let chunk_id = u64::from_le_bytes(buf[pos..pos + 8].try_into()?);
            pos += 8;

            let has_parent = buf[pos];
            pos += 1;

            let parent_raw = u64::from_le_bytes(buf[pos..pos + 8].try_into()?);
            pos += 8;

            let parent_id = if has_parent == 1 {
                Some(parent_raw)
            } else {
                None
            };

            if pos + 2 > buf.len() {
                break;
            }
            let group_len = u16::from_le_bytes(buf[pos..pos + 2].try_into()?) as usize;
            pos += 2;

            let group_id = if group_len > 0 && pos + group_len <= buf.len() {
                let gid = String::from_utf8_lossy(&buf[pos..pos + group_len]).to_string();
                pos += group_len;
                Some(gid)
            } else {
                None
            };

            store.add(chunk_id, parent_id, group_id);
        }

        Ok(store)
    }
}
