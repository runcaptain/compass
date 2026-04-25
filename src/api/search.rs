// api/search.rs — Search and facet endpoints.
//
// POST /collections/:name/search — full scoring pipeline:
//   retrieve (FTS/semantic/hybrid) → filter → recency decay → metadata boost → relationship boost
//
// GET /collections/:name/facets — microsecond bitset facet counts

use crate::api::AppState;
use crate::models::{FacetRequest, FacetResponse, SearchHit, SearchRequest, SearchResponse};
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::Json;
use std::sync::Arc;

/// POST /collections/:name/search
pub async fn search_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    let mode_str = req.mode.clone();

    let (results, total, took_us) = state
        .manager
        .search(&name, &req, &state.embed_state)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let hits: Vec<SearchHit> = results
        .into_iter()
        .map(|(chunk, score, source)| SearchHit { chunk, score, source })
        .collect();

    Ok(Json(SearchResponse {
        results: hits,
        total,
        took_us,
        mode: mode_str,
    }))
}

/// GET /collections/:name/facets
pub async fn get_facets(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(req): Query<FacetRequest>,
) -> Result<Json<FacetResponse>, (StatusCode, String)> {
    let query_str = req.query.as_deref().unwrap_or("");

    let (facets, took_us) = state
        .manager
        .get_facets(&name, query_str, &req.fields)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(FacetResponse { facets, took_us }))
}
