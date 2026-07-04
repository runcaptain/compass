// Pre-existing clippy lints from newer toolchain — will be cleaned up separately.
#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::collapsible_if,
    clippy::map_flatten,
    clippy::unnecessary_cast,
    clippy::option_map_or_none,
    clippy::manual_div_ceil,
    clippy::ptr_arg,
    clippy::unnecessary_map_or,
    clippy::vec_init_then_push,
    // clippy::uninlined_format_args fires on every `format!("...{}", x)` site
    // in the codebase. Migrating to inline format args (`format!("...{x}")`)
    // is a mechanical cleanup, tracked separately.
    clippy::uninlined_format_args
)]
// Compass — Embedded vector + full-text search engine for Captain.
//
// Single-binary search database with zero external dependencies.
// Designed for on-prem enterprise deployments where data can't leave the VPC.
//
// v2 features:
//   - Named vector spaces (multiple embedding models per collection)
//   - Parent-child document relationships with sibling grouping
//   - Query-time scoring pipeline (recency decay, metadata boost, relationship boost)
//   - Background re-embedding with one-click model upgrade
//   - Typed metadata (string, int, float, bool, timestamp, string list)
//   - TAMS-compatible video search hierarchy (Source → Flow → Segment)

mod api;
mod collections;
mod embed;
mod metrics;
mod models;
mod scoring;
mod search;
// Storage abstraction: Storage trait + LocalDiskStorage + object-storage
// backend + the LSM (WAL fragments, manifest, segments) — the cloud-mode
// persistence layer.
mod storage;
mod telemetry;

use api::{AppState, AuthConfig};
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt::init();

    let port = env::var("PORT").unwrap_or_else(|_| "4001".to_string());
    let data_dir = env::var("DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./data"));

    tracing::info!("Compass v{} starting...", env!("CARGO_PKG_VERSION"));
    tracing::info!("Data directory: {}", data_dir.display());

    // Select the storage backend from COMPASS_STORAGE (local disk by default, or
    // your own cloud object storage — see .env.example) and verify connectivity
    // at boot so mis-configured credentials fail loudly here, not on first write.
    let storage = match storage::from_config(&data_dir) {
        Ok(store) => match storage::verify(store.as_ref()).await {
            Ok(()) => {
                tracing::info!(
                    "Storage backend '{}' verified (read/write/delete OK)",
                    store.backend_name()
                );
                store
            }
            Err(e) => {
                tracing::error!(
                    "Storage backend '{}' failed connectivity check: {e}",
                    store.backend_name()
                );
                return Err(e.into());
            }
        },
        Err(e) => {
            tracing::error!("Storage backend configuration error: {e}");
            return Err(e.into());
        }
    };

    // Initialize embedding models (BGE-small via Candle + distilled M2V fallback)
    let embed_state = Arc::new(embed::init_embedders(&data_dir));

    // Load existing collections. In object-storage mode the manager mirrors
    // ingests into the LSM (WAL + manifest) on the configured backend.
    let manager = collections::CollectionManager::new_with_storage(&data_dir, storage).await?;

    let app_state = Arc::new(AppState {
        manager,
        embed_state,
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Anonymous telemetry — opt out with COMPASS_TELEMETRY=off or DO_NOT_TRACK=1
    telemetry::spawn_telemetry(data_dir.clone(), app_state.manager.clone());

    // Bearer-token auth via COMPASS_API_KEY. When unset, auth is disabled.
    let api_key = env::var("COMPASS_API_KEY").ok().filter(|s| !s.is_empty());
    if api_key.is_some() {
        tracing::info!(
            "API key auth enabled (Authorization: Bearer required on all routes except /health)"
        );
    } else {
        tracing::warn!("COMPASS_API_KEY not set — API is unauthenticated (dev mode)");
    }
    let auth_config = Arc::new(AuthConfig {
        expected_key: api_key,
    });

    let app = api::build_router(app_state, auth_config).layer(cors);

    let addr = format!("0.0.0.0:{}", port);
    tracing::info!("Compass listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
