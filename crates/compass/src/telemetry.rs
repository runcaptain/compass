//! Anonymous usage telemetry via PostHog.
//!
//! Sends a daily heartbeat with non-identifying metrics:
//! instance UUID, Compass version, OS, architecture, collection count, total vectors.
//!
//! No data content, IP addresses, or personally identifiable information is collected.
//!
//! Opt out: set `COMPASS_TELEMETRY=off` or `DO_NOT_TRACK=1`.

use std::path::{Path, PathBuf};
use std::time::Duration;

const POSTHOG_ENDPOINT: &str = "https://us.i.posthog.com/capture/";
const POSTHOG_API_KEY: &str = "phc_BFvsmH5rpe8GqJ8zwfqhH9jGAdZMXcNZhEao8mnDEd3X";
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);
const STARTUP_DELAY: Duration = Duration::from_secs(60);

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
struct PostHogCapture {
    api_key: &'static str,
    event: String,
    properties: PostHogProperties,
    timestamp: String,
}

#[derive(serde::Serialize)]
struct PostHogProperties {
    distinct_id: String,
    version: String,
    os: String,
    arch: String,
    collections: u64,
    total_vectors: u64,
}

fn now_iso8601() -> String {
    chrono::Utc::now().to_rfc3339()
}

/// Send a single event to PostHog. Fire-and-forget.
async fn send_event(client: &reqwest::Client, event_name: &str, instance_id: &str, collections: u64, total_vectors: u64) {
    let payload = PostHogCapture {
        api_key: POSTHOG_API_KEY,
        event: event_name.to_string(),
        properties: PostHogProperties {
            distinct_id: instance_id.to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            collections,
            total_vectors,
        },
        timestamp: now_iso8601(),
    };
    let _ = client.post(POSTHOG_ENDPOINT).json(&payload).send().await;
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
        tokio::time::sleep(STARTUP_DELAY).await;

        let client = match reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
        {
            Ok(c) => c,
            Err(_) => return,
        };

        // Send startup event
        let collections = manager.list_collections().await;
        let total_vectors: u64 = collections.iter().map(|c| c.chunk_count).sum();
        send_event(&client, "compass_started", &instance_id, collections.len() as u64, total_vectors).await;

        loop {
            tokio::time::sleep(HEARTBEAT_INTERVAL).await;

            let collections = manager.list_collections().await;
            let total_vectors: u64 = collections.iter().map(|c| c.chunk_count).sum();
            send_event(&client, "compass_heartbeat", &instance_id, collections.len() as u64, total_vectors).await;
        }
    });
}
