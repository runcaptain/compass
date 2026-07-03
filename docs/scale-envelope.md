# Scale Envelope (measured, not claimed)

Every number here comes from the env-gated harness in
`collections/mod.rs::scale_envelope`:

```bash
COMPASS_SCALE_N=250000 cargo test -p compass --features object-storage \
  --release scale_envelope -- --nocapture
```

Runs use a local-disk Storage backend (identical code paths to S3, disk-bound)
inside a linux/amd64 container **under ARM emulation** — native x86 hardware
runs meaningfully faster; treat these as conservative floors.

| chunks | dims | ingest | cold attach | search (semantic, avg) |
|---|---|---|---|---|
| 250,000 | 128 | 156s (1,603 chunks/s) | 134.5s | 5.5ms |
| 1,000,000 | 128 | (run in progress — see PR) | | |

## What the envelope means

- **RAM is O(cache budget)** since chunks moved out-of-core (bounded LRU over
  redb); segment format v2 + multipart removed the 5GB object ceiling;
  partitioned compaction is O(batch) per cycle; per-write index costs are
  O(batch). None of the previous hard walls bind below ~100M chunks.
- **The binding constraint is cold-attach time** (HNSW rebuild from the mmap
  file — roughly linear in collection size). Lazy attach + LRU keep this a
  first-request cost per namespace, not a boot cost, but a 100M-chunk
  collection still takes tens of minutes to attach on first use.
- **Billion-vector serving therefore remains out of envelope** until
  serve-from-storage indexes land (roadmap Phase 6: centroid routing over
  range-readable segments — attach becomes "fetch centroids", milliseconds).
  Do not deploy a single collection past ~10–50M chunks and expect
  sub-minute cold attach.

## Operating guidance

- Shard very large corpora across collections (attach cost is per-collection).
- Watch `/metrics`: `compass_attach_seconds_sum_millis / compass_attach_total`
  is your real attach cost; `compass_refresh_reattaches_total` climbing means
  compaction is outrunning refresh (raise `COMPASS_REFRESH_INTERVAL` or lower
  write bursts).
- Set `COMPASS_MAX_ATTACHED` on memory-constrained workers and
  `COMPASS_MAX_CONCURRENCY` in front of bursty clients.
