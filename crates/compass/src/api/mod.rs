// api/mod.rs — Axum router and shared application state.
//
// All HTTP endpoints wired up here. Includes v2 endpoints for
// vector space CRUD, rebuild triggers, and status checks.
//
// Bearer-token auth middleware is applied to all routes
// except /health. See `AuthConfig` and `auth_middleware` below.

pub mod collections;
pub mod delete;
pub mod ingest;
pub mod relations;
pub mod search;
pub mod segments;

use crate::collections::CollectionManager;
use crate::embed::EmbedState;
use crate::models::HealthResponse;
use axum::extract::{Request, State};
use axum::http::{header::AUTHORIZATION, StatusCode};
use axum::middleware::{from_fn_with_state, Next};
use axum::response::Response;
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use std::sync::Arc;

/// Shared application state passed to every request handler.
/// Map an engine error to an HTTP response. Typed `NotFound` becomes 404
/// regardless of the handler's default; a 500 default logs the detail and
/// returns a generic body (backend/path internals don't belong in responses).
pub(crate) fn error_response(
    e: Box<dyn std::error::Error + Send + Sync>,
    default: StatusCode,
) -> (StatusCode, String) {
    if e.downcast_ref::<crate::collections::NotFound>().is_some() {
        return (StatusCode::NOT_FOUND, e.to_string());
    }
    if default == StatusCode::INTERNAL_SERVER_ERROR {
        tracing::error!("handler error: {e}");
        return (default, "internal error (see server logs)".to_string());
    }
    (default, e.to_string())
}

pub struct AppState {
    pub manager: Arc<CollectionManager>,
    pub embed_state: Arc<EmbedState>,
}

/// Auth configuration for the protected-route middleware.
/// `None` disables auth (dev mode); a `Some(key)` requires `Authorization: Bearer <key>`.
#[derive(Clone)]
pub struct AuthConfig {
    pub expected_key: Option<String>,
}

/// Constant-time byte comparison to avoid timing attacks on the API key check.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Bearer-token middleware. Applied to every route except `/health`.
async fn auth_middleware(
    State(cfg): State<Arc<AuthConfig>>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let provided = req
        .headers()
        .get(AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "));

    match (cfg.expected_key.as_deref(), provided) {
        (Some(expected), Some(got)) if ct_eq(expected.as_bytes(), got.as_bytes()) => {
            Ok(next.run(req).await)
        }
        (None, _) => Ok(next.run(req).await),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

/// Build the Axum router with all Compass endpoints. `/health` is unauthenticated;
/// every other route requires `Authorization: Bearer <COMPASS_API_KEY>` when the key is set.
pub fn build_router(state: Arc<AppState>, auth: Arc<AuthConfig>) -> Router {
    let protected = Router::new()
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
        // ── Delete (soft-delete chunks) ──────────────────────────────────
        .route(
            "/collections/{name}/chunks/{chunk_id}",
            delete(delete::delete_chunk),
        )
        .route("/collections/{name}/delete", post(delete::delete_by_query))
        .route(
            "/collections/{name}/compact",
            post(delete::compact_collection),
        )
        // ── Search + Facets ──────────────────────────────────────────────
        .route(
            "/collections/{name}/search",
            post(search::search_collection),
        )
        .route("/collections/{name}/facets", get(search::get_facets))
        // ── Temporal segment lookup (TAMS point/range query) ─────────────
        .route(
            "/collections/{name}/segments/at",
            get(segments::segments_at),
        )
        // ── Chunk relations (typed, many-to-many edges) ──────────────────
        .route(
            "/collections/{name}/relations",
            post(relations::create_relations),
        )
        .route(
            "/collections/{name}/relations/{relation_id}",
            delete(relations::delete_relation),
        )
        .route(
            "/collections/{name}/chunks/{chunk_id}/relations",
            get(relations::get_chunk_relations),
        )
        .layer(from_fn_with_state(auth, auth_middleware));

    Router::new()
        // ── Health (unauthenticated) ─────────────────────────────────────
        .route("/health", get(health_check))
        .route("/metrics", get(metrics_endpoint))
        .merge(protected)
        // 64 MB body limit. Default 2 MB is too small for batched ingest with embeddings.
        .layer(axum::extract::DefaultBodyLimit::max(64 * 1024 * 1024))
        // Backpressure: bound in-flight requests instead of queueing without
        // limit (COMPASS_MAX_CONCURRENCY; unset = unlimited). The cap must stay
        // under tokio's Semaphore::MAX_PERMITS (usize::MAX >> 3) — a larger
        // value PANICS at startup.
        .layer(tower::limit::GlobalConcurrencyLimitLayer::new(
            std::env::var("COMPASS_MAX_CONCURRENCY")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|n: &usize| *n > 0)
                .unwrap_or(usize::MAX >> 4)
                .min(usize::MAX >> 4),
        ))
        .with_state(state)
}

/// GET /metrics — Prometheus text. Unauthenticated (like /health); carries
/// operational counters plus per-collection gauges.
async fn metrics_endpoint(State(state): State<Arc<AppState>>) -> String {
    let mut gauges = String::new();
    // Attached collections only: list_collections in lazy mode does one S3
    // GET per registered namespace — an unauthenticated request-amplifier if
    // exposed to a scraper.
    for c in state.manager.attached_collections().await {
        gauges.push_str(&format!(
            "compass_collection_chunks{{collection=\"{}\"}} {}\n",
            c.name, c.chunk_count
        ));
        gauges.push_str(&format!(
            "compass_collection_applied_seq{{collection=\"{}\"}} {}\n",
            c.name, c.applied_seq
        ));
    }
    crate::metrics::render(&gauges)
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
