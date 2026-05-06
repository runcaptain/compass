// collections/store.rs — Disk persistence for collection metadata.
//
// Each collection's metadata (name, created_at, dims, chunk count) is stored as a
// JSON file at: ./data/{collection_name}/collection.json
//
// The actual search indices live alongside it:
//   ./data/{collection_name}/tantivy/     — Tantivy FTS index files
//   ./data/{collection_name}/vectors/     — USearch HNSW index + raw vectors

use crate::models::Collection;
use std::path::{Path, PathBuf};

/// Get the base data directory for a collection.
/// Returns: {data_dir}/{collection_name}/
pub fn collection_dir(data_dir: &Path, name: &str) -> PathBuf {
    data_dir.join(name)
}

/// Get the path to a collection's metadata JSON file.
pub fn metadata_path(data_dir: &Path, name: &str) -> PathBuf {
    collection_dir(data_dir, name).join("collection.json")
}

/// Get the Tantivy index directory for a collection.
pub fn tantivy_dir(data_dir: &Path, name: &str) -> PathBuf {
    collection_dir(data_dir, name).join("tantivy")
}

/// Get the vector index directory for a collection.
pub fn vectors_dir(data_dir: &Path, name: &str) -> PathBuf {
    collection_dir(data_dir, name).join("vectors")
}

/// Save collection metadata to disk as JSON.
pub fn save_metadata(
    data_dir: &Path,
    collection: &Collection,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let dir = collection_dir(data_dir, &collection.name);
    std::fs::create_dir_all(&dir)?;
    let path = metadata_path(data_dir, &collection.name);
    let json = serde_json::to_string_pretty(collection)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Load collection metadata from disk.
pub fn load_metadata(
    data_dir: &Path,
    name: &str,
) -> Result<Collection, Box<dyn std::error::Error + Send + Sync>> {
    let path = metadata_path(data_dir, name);
    let json = std::fs::read_to_string(path)?;
    let collection: Collection = serde_json::from_str(&json)?;
    Ok(collection)
}

/// List all collection names by scanning the data directory for subdirectories
/// that contain a collection.json file.
pub fn list_collection_names(
    data_dir: &Path,
) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
    let mut names = Vec::new();
    if !data_dir.exists() {
        return Ok(names);
    }
    for entry in std::fs::read_dir(data_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            let name = entry.file_name().to_string_lossy().to_string();
            // Only include directories that have a collection.json (skip "models/" etc.)
            if metadata_path(data_dir, &name).exists() {
                names.push(name);
            }
        }
    }
    Ok(names)
}

/// Delete a collection's entire data directory from disk.
pub fn delete_collection_data(
    data_dir: &Path,
    name: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let dir = collection_dir(data_dir, name);
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    Ok(())
}
