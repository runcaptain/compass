// api/mod.rs — Axum router and shared application state.
//
// All HTTP endpoints wired up here. Includes v2 endpoints for
// vector space CRUD, rebuild triggers, and status checks.

pub mod collections;
pub mod ingest;
pub mod search;

use crate::collections::CollectionManager;
use crate::embed::EmbedState;
use crate::models::HealthResponse;
use axum::extract::State;
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use std::sync::Arc;

/// Shared application state passed to every request handler.
pub struct AppState {
    pub manager: Arc<CollectionManager>,
    pub embed_state: Arc<EmbedState>,
}

/// Build the Axum router with all Compass endpoints.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        // ── Collection CRUD ──────────────────────────────────────────────
        .route("/collections", post(collections::create_collection))
        .route("/collections", get(collections::list_collections))
        .route("/collections/{name}", get(collections::get_collection))
        .route(
            "/collections/{name}",
            delete(collections::delete_collection),
        )
        // ── Vector Space CRUD ────────────────────────────────────────────
        .route(
            "/collections/{name}/vector-spaces",
            post(collections::add_vector_space),
        )
        .route(
            "/collections/{name}/vector-spaces",
            get(collections::list_vector_spaces),
        )
        .route(
            "/collections/{name}/vector-spaces/{space}",
            delete(collections::delete_vector_space),
        )
        .route(
            "/collections/{name}/vector-spaces/{space}/rebuild",
            post(collections::trigger_rebuild),
        )
        .route(
            "/collections/{name}/vector-spaces/{space}/status",
            get(collections::rebuild_status),
        )
        .route(
            "/collections/{name}/default-vector-space",
            put(collections::set_default_vector_space),
        )
        // ── Ingest ───────────────────────────────────────────────────────
        .route("/collections/{name}/ingest", post(ingest::ingest_chunks))
        // ── Search + Facets ──────────────────────────────────────────────
        .route(
            "/collections/{name}/search",
            post(search::search_collection),
        )
        .route("/collections/{name}/facets", get(search::get_facets))
        // ── Health ───────────────────────────────────────────────────────
        .route("/health", get(health_check))
        // 64 MB body limit. Default 2 MB is too small for batched ingest with embeddings.
        .layer(axum::extract::DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(state)
}

/// GET /health
async fn health_check(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let collections = state.manager.list_collections().await;
    Json(HealthResponse {
        status: "ok".to_string(),
        collections: collections.len(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}
