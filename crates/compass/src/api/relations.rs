// api/relations.rs — Typed, many-to-many chunk relations.
//
//   POST   /collections/:name/relations                    create one/many edges
//   DELETE /collections/:name/relations/:relation_id        delete an edge
//   GET    /collections/:name/chunks/:chunk_id/relations    list a chunk's edges
//
// Relations are directed labeled edges between chunks
// (`source --relation_type--> target`), independent of the parent/group
// hierarchy. Storage is the disk-backed RelationStore (relations.redb); reads
// are on-demand, never resident in RAM. See collections/relation_store.rs.

use crate::api::AppState;
use crate::models::{
    ChunkRelationsResponse, CreateRelationsRequest, CreateRelationsResponse, RelationDirection,
};
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::Json;
use serde::Deserialize;
use std::sync::Arc;

/// Max edges accepted in one create call. Bounds both the single S3 fragment
/// and the endpoint-index rewrite work done under one redb transaction.
const MAX_RELATIONS_PER_REQUEST: usize = 10_000;

fn map_err(e: Box<dyn std::error::Error + Send + Sync>) -> (StatusCode, String) {
    let msg = e.to_string();
    if msg.contains("not found") {
        (StatusCode::NOT_FOUND, msg)
    } else if msg.contains("must differ") {
        (StatusCode::BAD_REQUEST, msg)
    } else {
        // Log the detail server-side; internal errors (paths, backends, redb
        // internals) don't belong in response bodies.
        tracing::error!("relations handler error: {msg}");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal error (see server logs)".to_string(),
        )
    }
}

/// POST /collections/:name/relations
pub async fn create_relations(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<CreateRelationsRequest>,
) -> Result<Json<CreateRelationsResponse>, (StatusCode, String)> {
    if req.relations.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "relations must contain at least one edge".to_string(),
        ));
    }
    if req.relations.len() > MAX_RELATIONS_PER_REQUEST {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "too many relations in one request: {} (max {MAX_RELATIONS_PER_REQUEST}); \
                 split into batches",
                req.relations.len()
            ),
        ));
    }
    let created = state
        .manager
        .create_relations(&name, req.relations)
        .await
        .map_err(map_err)?;
    let count = created.len();
    Ok(Json(CreateRelationsResponse {
        relations: created,
        created: count,
    }))
}

/// DELETE /collections/:name/relations/:relation_id
pub async fn delete_relation(
    State(state): State<Arc<AppState>>,
    Path((name, relation_id)): Path<(String, String)>,
) -> Result<StatusCode, (StatusCode, String)> {
    let existed = state
        .manager
        .delete_relation(&name, &relation_id)
        .await
        .map_err(map_err)?;
    if existed {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err((StatusCode::NOT_FOUND, "relation not found".to_string()))
    }
}

#[derive(Debug, Deserialize)]
pub struct ChunkRelationsQuery {
    /// "outgoing" (default), "incoming", or "both".
    #[serde(default)]
    pub direction: RelationDirection,
    /// Comma-separated relation_type filter, e.g. `?types=cites,supersedes`.
    #[serde(default)]
    pub types: Option<String>,
}

/// GET /collections/:name/chunks/:chunk_id/relations
pub async fn get_chunk_relations(
    State(state): State<Arc<AppState>>,
    Path((name, chunk_id)): Path<(String, u64)>,
    Query(q): Query<ChunkRelationsQuery>,
) -> Result<Json<ChunkRelationsResponse>, (StatusCode, String)> {
    // `?types=` / `?types=,,` (all entries empty) means "no filter", not
    // "match nothing" — normalize an empty list to None.
    let types: Option<Vec<String>> = q
        .types
        .as_ref()
        .map(|s| {
            s.split(',')
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty());

    let relations = state
        .manager
        .get_chunk_relations(&name, chunk_id, q.direction, types.as_deref())
        .await
        .map_err(map_err)?;
    let total = relations.len();
    Ok(Json(ChunkRelationsResponse { relations, total }))
}
