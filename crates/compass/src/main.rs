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

    let app = api::build_router(app_state).layer(cors);

    let addr = format!("0.0.0.0:{}", port);
    tracing::info!("Compass listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
