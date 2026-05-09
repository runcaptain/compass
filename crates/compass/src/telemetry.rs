//! Anonymous usage telemetry.
//!
//! Sends a daily heartbeat with non-identifying metrics:
//! instance UUID, Compass version, OS, architecture, collection count, total vectors.
//!
//! No data content, IP addresses, or personally identifiable information is collected.
//!
//! Opt out: set `COMPASS_TELEMETRY=off` or `DO_NOT_TRACK=1`.

use std::path::{Path, PathBuf};
use std::time::Duration;

const TELEMETRY_ENDPOINT: &str = "https://telemetry.runcaptain.com/v1/events";
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60); // 24 hours
const STARTUP_DELAY: Duration = Duration::from_secs(60); // wait for collections to load

/// Returns true if telemetry is enabled (default).
pub fn is_enabled() -> bool {
    if let Ok(v) = std::env::var("COMPASS_TELEMETRY") {
        return !matches!(v.to_lowercase().as_str(), "off" | "false" | "0" | "no");
    }
    if let Ok(v) = std::env::var("DO_NOT_TRACK") {
        return !matches!(v.as_str(), "1" | "true");
    }
    true
}

/// Persistent instance ID — generated once, stored in data_dir/instance_id.
fn get_or_create_instance_id(data_dir: &Path) -> String {
    let path = data_dir.join("instance_id");
    if let Ok(id) = std::fs::read_to_string(&path) {
        let id = id.trim().to_string();
        if !id.is_empty() {
            return id;
        }
    }
    let id = uuid::Uuid::new_v4().to_string();
    let _ = std::fs::create_dir_all(data_dir);
    let _ = std::fs::write(&path, &id);
    id
}

#[derive(serde::Serialize)]
struct HeartbeatEvent {
    instance_id: String,
    version: String,
    os: String,
    arch: String,
    collections: u64,
    total_vectors: u64,
    event: String,
}

/// Spawn a background task that sends a heartbeat on startup and every 24 hours.
/// The task is fire-and-forget — failures are silently ignored.
pub fn spawn_telemetry(
    data_dir: PathBuf,
    manager: std::sync::Arc<crate::collections::CollectionManager>,
) {
    if !is_enabled() {
        tracing::info!("Telemetry disabled (COMPASS_TELEMETRY=off or DO_NOT_TRACK=1)");
        return;
    }

    let instance_id = get_or_create_instance_id(&data_dir);
    tracing::info!(
        "Anonymous telemetry enabled (instance: {}). Set COMPASS_TELEMETRY=off to disable.",
        &instance_id[..8]
    );

    tokio::spawn(async move {
        // Wait for startup to settle
        tokio::time::sleep(STARTUP_DELAY).await;

        loop {
            let collections = manager.list_collections().await;
            let total_vectors: u64 = collections.iter().map(|c| c.chunk_count).sum();

            let event = HeartbeatEvent {
                instance_id: instance_id.clone(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                os: std::env::consts::OS.to_string(),
                arch: std::env::consts::ARCH.to_string(),
                collections: collections.len() as u64,
                total_vectors,
                event: "heartbeat".to_string(),
            };

            // Fire and forget — never block the server
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build();

            if let Ok(client) = client {
                let _ = client.post(TELEMETRY_ENDPOINT).json(&event).send().await;
            }

            tokio::time::sleep(HEARTBEAT_INTERVAL).await;
        }
    });
}
