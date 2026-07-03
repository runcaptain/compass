//! Minimal Prometheus-text metrics — no external deps, atomic counters only.
//!
//! Operating a multi-node deployment blind was a production blocker: you
//! could not see ingest/search rates, refresh convergence, or attach costs.
//! This is deliberately tiny; a full metrics facade can replace it later
//! without touching call sites (they go through these free functions).

use std::sync::atomic::{AtomicU64, Ordering};

macro_rules! counters {
    ($($name:ident),* $(,)?) => {
        $(pub static $name: AtomicU64 = AtomicU64::new(0);)*
        fn render_counters(out: &mut String) {
            $(
                out.push_str(&format!(
                    "compass_{} {}\n",
                    stringify!($name).to_lowercase(),
                    $name.load(Ordering::Relaxed)
                ));
            )*
        }
    };
}

counters!(
    INGEST_REQUESTS_TOTAL,
    INGEST_CHUNKS_TOTAL,
    SEARCH_REQUESTS_TOTAL,
    DELETE_REQUESTS_TOTAL,
    REFRESH_FRAGMENTS_APPLIED_TOTAL,
    REFRESH_REATTACHES_TOTAL,
    ATTACH_TOTAL,
    ATTACH_SECONDS_SUM_MILLIS,
    COMPACTIONS_TOTAL,
    QUARANTINED_CHUNKS_TOTAL,
);

#[inline]
pub fn inc(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub fn add(counter: &AtomicU64, n: u64) {
    counter.fetch_add(n, Ordering::Relaxed);
}

/// Render every counter plus caller-supplied gauge lines (e.g. per-collection
/// state the manager owns).
pub fn render(extra_gauges: &str) -> String {
    let mut out = String::with_capacity(1024);
    render_counters(&mut out);
    out.push_str(extra_gauges);
    out
}
