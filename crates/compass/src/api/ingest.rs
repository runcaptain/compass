// api/ingest.rs — Bulk document ingestion endpoint.
//
// POST /collections/:name/ingest
// Accepts an array of chunks with:
//   - Optional client_id + parent_ref for batch cross-referencing
//   - Named embeddings (multiple vector spaces in one call)
//   - Typed metadata
//   - Document type and relationship fields

use crate::api::AppState;
use crate::models::{IngestRequest, IngestResponse};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use std::sync::Arc;

/// POST /collections/:name/ingest
pub async fn ingest_chunks(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<IngestRequest>,
) -> Result<Json<IngestResponse>, (StatusCode, String)> {
    let start = std::time::Instant::now();

    let (count, id_map) = state
        .manager
        .ingest(&name, req.chunks, &state.embed_state)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let took_ms = start.elapsed().as_millis() as u64;

    Ok(Json(IngestResponse {
        indexed: count,
        id_map,
        took_ms,
    }))
}
