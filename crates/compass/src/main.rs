// Pre-existing clippy lints from newer toolchain — will be cleaned up separately.
#![allow(
    dead_code,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::collapsible_if,
    clippy::map_flatten,
    clippy::unnecessary_cast,
    clippy::option_map_or_none,
    clippy::manual_div_ceil,
    clippy::ptr_arg,
    clippy::unnecessary_map_or,
    clippy::vec_init_then_push
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
mod filter;
mod models;
mod scoring;
mod search;
mod telemetry;

use api::AppState;
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

    // Initialize embedding models (BGE-small via Candle + distilled M2V fallback)
    let embed_state = Arc::new(embed::init_embedders(&data_dir));

    // Load existing collections from disk (indices, relationships, vector spaces)
    let manager = collections::CollectionManager::new(&data_dir).await?;

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

    let app = api::build_router(app_state).layer(cors);

    let addr = format!("0.0.0.0:{}", port);
    tracing::info!("Compass listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
