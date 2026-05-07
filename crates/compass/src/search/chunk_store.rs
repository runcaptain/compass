//! Disk-backed chunk store using redb.
//!
//! Replaces `HashMap<u64, DocumentChunk>` with a persistent embedded database.
//! Point lookups by u64 ID, batch inserts, full scans for rebuild.

use crate::models::DocumentChunk;
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use std::path::Path;

const CHUNKS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("chunks");

pub struct ChunkStore {
    db: Database,
}

impl ChunkStore {
    pub fn open(path: &Path) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let db = Database::create(path)?;
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
