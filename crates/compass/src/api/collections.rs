// api/collections.rs — Collection CRUD + vector space management endpoints.

use crate::api::AppState;
use crate::models::*;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use std::sync::Arc;

/// POST /collections — create a new collection
pub async fn create_collection(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateCollectionRequest>,
) -> Result<(StatusCode, Json<CollectionInfo>), (StatusCode, String)> {
    let collection = state
        .manager
        .create_collection(&req.name, req.vector_spaces, req.embedding_dims, req.config)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    Ok((StatusCode::CREATED, Json(collection_to_info(&collection))))
}

/// GET /collections — list all collections
pub async fn list_collections(State(state): State<Arc<AppState>>) -> Json<Vec<CollectionInfo>> {
    let collections = state.manager.list_collections().await;
    Json(collections.iter().map(collection_to_info).collect())
}

/// GET /collections/:name — get info about a collection
pub async fn get_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<CollectionInfo>, (StatusCode, String)> {
    let collection = state.manager.get_collection(&name).await.ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("Collection '{}' not found", name),
        )
    })?;
    Ok(Json(collection_to_info(&collection)))
}

/// DELETE /collections/:name — delete a collection
pub async fn delete_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    state
        .manager
        .delete_collection(&name)
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;
    Ok(StatusCode::NO_CONTENT)
}

// ── Vector Space CRUD ────────────────────────────────────────────────────

/// POST /collections/:name/vector-spaces — add a vector space
pub async fn add_vector_space(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<AddVectorSpaceRequest>,
) -> Result<(StatusCode, Json<VectorSpaceInfo>), (StatusCode, String)> {
    state
        .manager
        .add_vector_space(&name, &req.name, req.dims, &req.model)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    Ok((
        StatusCode::CREATED,
        Json(VectorSpaceInfo {
            name: req.name,
            dims: req.dims,
            model: req.model,
            status: "building".to_string(),
        }),
    ))
}

/// GET /collections/:name/vector-spaces — list vector spaces
pub async fn list_vector_spaces(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<Vec<VectorSpaceInfo>>, (StatusCode, String)> {
    let collection = state.manager.get_collection(&name).await.ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("Collection '{}' not found", name),
        )
    })?;

    let spaces: Vec<VectorSpaceInfo> = collection
        .vector_spaces
        .iter()
        .map(|(sname, config)| VectorSpaceInfo {
            name: sname.clone(),
            dims: config.dims,
            model: config.model.clone(),
            status: config.status.clone(),
        })
        .collect();

    Ok(Json(spaces))
}

/// DELETE /collections/:name/vector-spaces/:space — remove a vector space
pub async fn delete_vector_space(
    State(state): State<Arc<AppState>>,
    Path((name, space)): Path<(String, String)>,
) -> Result<StatusCode, (StatusCode, String)> {
    state
        .manager
        .delete_vector_space(&name, &space)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    Ok(StatusCode::NO_CONTENT)
}

/// PUT /collections/:name/default-vector-space — switch default space
pub async fn set_default_vector_space(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<SetDefaultSpaceRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    state
        .manager
        .set_default_vector_space(&name, &req.name)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    Ok(StatusCode::OK)
}

/// POST /collections/:name/vector-spaces/:space/rebuild — trigger re-embedding
pub async fn trigger_rebuild(
    State(state): State<Arc<AppState>>,
    Path((name, space)): Path<(String, String)>,
    Json(req): Json<RebuildRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    // Get collection metadata to find dims
    let collection = state.manager.get_collection(&name).await.ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("Collection '{}' not found", name),
        )
    })?;

    let dims = collection
        .vector_spaces
        .get(&space)
        .map(|c| c.dims)
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                format!("Vector space '{}' not found", space),
            )
        })?;

    // Get all chunk data for re-embedding
    let (texts, chunk_ids) = state
        .manager
        .get_all_chunk_data(&name)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let vectors_dir = state.manager.vectors_dir(&name);

    crate::collections::rebuild::start_rebuild(
        texts,
        chunk_ids,
        space.clone(),
        vectors_dir,
        dims,
        state.embed_state.clone(),
        req.embed_endpoint,
        req.batch_size,
        state.manager.rebuild_tracker.clone(),
        name,
    )
    .await
    .map_err(|e| (StatusCode::CONFLICT, e))?;

    Ok(StatusCode::ACCEPTED)
}

/// GET /collections/:name/vector-spaces/:space/status — rebuild progress
pub async fn rebuild_status(
    State(state): State<Arc<AppState>>,
    Path((name, space)): Path<(String, String)>,
) -> Result<Json<RebuildStatus>, (StatusCode, String)> {
    let status = crate::collections::rebuild::get_rebuild_status(
        &state.manager.rebuild_tracker,
        &name,
        &space,
    )
    .await;

    match status {
        Some(s) => Ok(Json(s)),
        None => {
            // Check if the space exists and is already active
            let collection = state.manager.get_collection(&name).await.ok_or_else(|| {
                (
                    StatusCode::NOT_FOUND,
                    format!("Collection '{}' not found", name),
                )
            })?;
            match collection.vector_spaces.get(&space) {
                Some(config) => Ok(Json(RebuildStatus {
                    status: config.status.clone(),
                    embedded: collection.chunk_count,
                    total: collection.chunk_count,
                    percent: 100.0,
                })),
                None => Err((
                    StatusCode::NOT_FOUND,
                    format!("Vector space '{}' not found", space),
                )),
            }
        }
    }
}

fn collection_to_info(c: &Collection) -> CollectionInfo {
    CollectionInfo {
        name: c.name.clone(),
        created_at: c.created_at,
        embedding_dims: c.embedding_dims,
        chunk_count: c.chunk_count,
        vector_spaces: c.vector_spaces.clone(),
        default_vector_space: c.default_vector_space.clone(),
    }
}
