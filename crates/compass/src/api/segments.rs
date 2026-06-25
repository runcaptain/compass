// api/segments.rs — Temporal point/range lookup for TAMS-style segment chunks.
//
// GET /collections/:name/segments/at
//
// Query parameters:
//   asset            (required) the `group_id` value identifying which source asset to scan
//   time_ms          (optional) single point lookup in milliseconds: returns segments where
//                               timerange_start_ms <= time_ms <= timerange_end_ms
//   time_start_ms    (optional) range lookup lower bound (inclusive, milliseconds)
//   time_end_ms      (optional) range lookup upper bound (inclusive, milliseconds)
//
// `time_ms` and `time_start_ms`/`time_end_ms` are mutually exclusive. If
// `time_ms` is set, the range parameters are ignored. If none are set, all
// segments for the asset are returned (useful for enumeration).
//
// Asset matching uses `group_id` on the stored chunk. In the TAMS ingest model
// the segment's `group_id` is the source's client_id, making it the stable
// external key. `parent_id` is an internal u64 auto-increment, not suitable
// for client queries.
//
// Time unit convention: all timestamps in Compass are integer milliseconds.
// Segments store `timerange_start_ms` and `timerange_end_ms` as numeric
// metadata. Instants (zero-duration events like a "standout timestamp") are
// stored as segments where `timerange_start_ms == timerange_end_ms`.

use crate::api::AppState;
use crate::models::DocumentChunk;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Deserialize)]
pub struct SegmentsAtQuery {
    /// group_id of the source asset whose segments to scan.
    pub asset: String,
    /// Single point in time (milliseconds). Takes precedence over time_start_ms/time_end_ms.
    pub time_ms: Option<f64>,
    /// Range query lower bound (inclusive, milliseconds).
    pub time_start_ms: Option<f64>,
    /// Range query upper bound (inclusive, milliseconds).
    pub time_end_ms: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct SegmentsAtResponse {
    pub results: Vec<DocumentChunk>,
    pub took_ms: f64,
}

/// GET /collections/:name/segments/at
pub async fn segments_at(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(params): Query<SegmentsAtQuery>,
) -> Result<Json<SegmentsAtResponse>, (StatusCode, String)> {
    let t0 = Instant::now();
    let results = state
        .manager
        .segments_at(
            &name,
            &params.asset,
            params.time_ms,
            params.time_start_ms,
            params.time_end_ms,
        )
        .await
        .map_err(|e| {
            let msg = e.to_string();
            if msg.contains("not found") {
                (StatusCode::NOT_FOUND, msg)
            } else {
                (StatusCode::INTERNAL_SERVER_ERROR, msg)
            }
        })?;

    let took_ms = t0.elapsed().as_secs_f64() * 1_000.0;
    Ok(Json(SegmentsAtResponse { results, took_ms }))
}
