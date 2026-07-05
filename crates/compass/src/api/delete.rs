// api/delete.rs — Soft-delete chunks (works in both local and object-storage modes).
//
//   DELETE /collections/:name/chunks/:id        delete one chunk by id
//   POST   /collections/:name/delete            delete by ids and/or metadata filter
//
// Deletes are soft (tombstones): the chunk vanishes from search results
// immediately; physical removal from the HNSW/FTS indexes happens on the next
// rebuild/compaction. In object-storage mode a durable tombstone WAL fragment is
// also written. See CollectionManager::delete_chunks / delete_by_filter.

use crate::api::AppState;
use crate::models::{DeleteRequest, DeleteResponse};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use std::sync::Arc;

fn map_err(e: Box<dyn std::error::Error + Send + Sync>) -> (StatusCode, String) {
    crate::api::error_response(e, StatusCode::INTERNAL_SERVER_ERROR)
}

/// DELETE /collections/:name/chunks/:id
pub async fn delete_chunk(
    State(state): State<Arc<AppState>>,
    Path((name, id)): Path<(String, u64)>,
) -> Result<Json<DeleteResponse>, (StatusCode, String)> {
    let (deleted, seq) = state
        .manager
        .delete_chunks(&name, &[id])
        .await
        .map_err(map_err)?;
    if deleted == 0 {
        // Nothing removed: the id didn't exist (or was already deleted).
        return Err((
            StatusCode::NOT_FOUND,
            format!("chunk {id} not found or already deleted"),
        ));
    }
    Ok(Json(DeleteResponse { deleted, seq }))
}

/// POST /collections/:name/compact — fold S3 segments + WAL into one segment,
/// reclaiming space for deleted (tombstoned) records. No-op in local mode.
pub async fn compact_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let records = state
        .manager
        .compact_collection(&name)
        .await
        .map_err(map_err)?;
    Ok(Json(serde_json::json!({ "compacted_records": records })))
}

/// POST /collections/:name/delete  { ids?: [...], filters?: {...} }
pub async fn delete_by_query(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<DeleteRequest>,
) -> Result<Json<DeleteResponse>, (StatusCode, String)> {
    if req.ids.is_empty() && req.filters.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "provide `ids` and/or `filters` to select chunks to delete".to_string(),
        ));
    }

    let mut deleted = 0usize;
    let mut seq: Option<u64> = None;
    if !req.ids.is_empty() {
        let (n, s) = state
            .manager
            .delete_chunks(&name, &req.ids)
            .await
            .map_err(map_err)?;
        deleted += n;
        seq = s.or(seq);
    }
    if !req.filters.is_empty() {
        // If the filter-delete fails after an ids-delete succeeded, report the
        // partial progress — deletes already applied are not undone.
        match state.manager.delete_by_filter(&name, &req.filters).await {
            Ok((n, s)) => {
                deleted += n;
                seq = s.or(seq);
            }
            Err(e) => {
                let (code, msg) = map_err(e);
                return Err((
                    code,
                    format!(
                        "{msg} ({deleted} chunk(s) were already deleted by `ids` before the error)"
                    ),
                ));
            }
        }
    }
    Ok(Json(DeleteResponse { deleted, seq }))
}
