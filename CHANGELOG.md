# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Memory-mapped vector storage.** New `MmapVectors` module (`search/mmap_vectors.rs`) replaces `Vec<Vec<f32>>` with a flat mmap-backed file. Zero-copy reads via `bytemuck::cast_slice`, append-only writes, near-zero RSS for vector data regardless of dataset size. File format: `[u32 dims][u32 count][f32...]`.
- **Disk-backed chunk metadata.** New `ChunkStore` module (`search/chunk_store.rs`) backed by redb (pure Rust embedded DB). Replaces in-memory `HashMap<u64, DocumentChunk>` with persistent, ACID-compliant storage. Point lookups, batch inserts, full scans for rebuild.
- **Incremental HNSW indexing.** Ingest path now loads the existing USearch index via `.load()`, appends new vectors with `.add()`, and saves — instead of cloning all vectors and rebuilding from scratch. Falls back to full rebuild when the `Arc` cannot be unwrapped.
- **New dependencies:** `memmap2` (mmap), `bytemuck` (zero-copy cast), `redb` (embedded DB).
- **Cargo workspace layout.** Source moved from a single crate to a workspace under `crates/`. Splits: `compass` (umbrella + binary), `compass-index-api` (stable trait surface), `compass-vector-gpu` (optional GPU backend).
- **Vector index trait abstraction.** New `compass_index_api::VectorIndex` trait abstracts over CPU and GPU backends. The existing USearch path is wrapped in `UsearchHnswIndex`; new backends bind to the trait without touching call sites.
- **Optional GPU backend (`--features gpu`).** New `compass-vector-gpu` crate integrates NVIDIA cuVS for CAGRA→HNSW build acceleration. CPU-side search after conversion. Linux + CUDA 12+ only. See `ARCHITECTURE.md` for build prerequisites.
- **`COMPASS_BACKEND` env var.** Selects the vector backend at startup: `cpu` (default), `gpu`, `auto`.
- **`ARCHITECTURE.md`.** Contributor's map: crate layout, module map, storage format, rebuild flow.
- **`CONTRIBUTING.md`.** PR checklist, commit format, scope guidelines.
- **Throughput and scaling section** in README documenting QPS expectations and horizontal scaling pattern.
- **Rich metadata filters.** Filters now support numeric range (`gte`/`lte`), array `contains`, and set membership (`in`) in addition to exact match. Backward compatible: plain values still work as exact match.
- **TAMS time-range queries.** Filter segments by `timerange_start`/`timerange_end` using range operators. Combined with `doc_type` filtering and relationship boost for hierarchy-aware video search.
- **Parent metadata on segment search hits.** Search responses now include `parent_metadata` on each `SearchHit` whose chunk is a segment with a `parent_id`. Saves the agent a second round trip to fetch source-level attributes (asset title, fps, duration, etc.). Parents are deduplicated across hits: N segments sharing the same parent pay for one lookup, not N. Non-segment hits and segments without a `parent_id` carry no `parent_metadata` field in the JSON response.
- **Temporal segment lookup.** New `GET /collections/:name/segments/at` endpoint returns `doc_type=segment` chunks matching a point in time (`time`) or overlapping a time window (`time_start`/`time_end`), filtered by `group_id` (`asset` param). Results sorted ascending by `timerange_start`. Full-scan implementation; indexed lookup by `group_id` is tracked as future work.
- **Configurable hybrid score weights.** New `score_weights` field on search requests controls RRF blending: `rrf_k`, `fts_weight`, `semantic_weight`. Previously hardcoded at k=60 with equal weights.
- **Recency presets.** `recency_preset` field accepts `aggressive` (3d), `recent` (7d), `mild` (30d), or `archive` (90d) instead of manually configuring the decay formula. Pair with `recency_field` to specify which timestamp metadata field to use.
- **Apache 2.0 license.**

### Changed

- **Cargo manifest** is now a workspace root with shared `[workspace.dependencies]` and `[workspace.package]` metadata. Per-crate manifests inherit version, edition, and rust-version.
- **Compiler profile** adds `lto = "thin"` and `codegen-units = 1` to release builds for better optimization. Adds debug symbols to bench builds.
- **MSRV bumped to 1.88** from 1.82. Required by `time@0.3.47` (edition 2024) and `icu_collections@2.2.0`.
- **Dockerfile** upgraded to `rust:latest` + `debian:trixie-slim` for glibc compatibility with newer Rust toolchains.

### Deprecated

- Direct use of `compass::search::vector` from external code. New code should bind to `compass::search::VectorIndex`. The legacy `vector.rs` functions remain for in-tree callers and will be wrapped or migrated incrementally.
- **`timerange_start` / `timerange_end` (in seconds).** Renamed to `timerange_start_ms` / `timerange_end_ms` (in integer milliseconds) so the unit is visible in the field name and matches the convention used by real-world sidecar producers (which commonly emit milliseconds). Existing data ingested against the old field names should be renamed and multiplied by 1000 in the transform.

### Changed (breaking)

- **`GET /collections/:name/segments/at` query parameters renamed.** `time` → `time_ms`, `time_start` → `time_start_ms`, `time_end` → `time_end_ms`. All values are now integer milliseconds rather than seconds. Matches the standard sidecar field convention.
- **Time-range field convention is now milliseconds.** Segment metadata uses `timerange_start_ms` and `timerange_end_ms` as numeric (integer) milliseconds. The `/segments/at` endpoint and `segment_in_time_window` predicate look up the new field names. Instants (zero-duration events such as instantaneous markers from sidecars) are stored as segments where `timerange_start_ms == timerange_end_ms`; the point query at that exact ms matches them, as do range queries that overlap.

### Fixed

- **ChunkStore tolerates stale `flock` state on networked filesystems.** redb acquires an exclusive `flock(LOCK_EX | LOCK_NB)` on the database file at open. On network-backed persistent volumes (NFS and similar), the kernel does not always release a previous process's `flock` immediately after that process exits. A restarted process then sees `DatabaseError::DatabaseAlreadyOpen` and `load_collection` would silently log-and-skip the affected collection, leaving the API reporting it as missing despite the data being durable on disk. `ChunkStore::open` now retries up to 6 times with 5s backoff before giving up, and enables redb's `set_repair_callback` so any dirty-file state from a previous abrupt shutdown auto-recovers.
- **Chunk metadata is now durable across restarts.** Wired the existing `ChunkStore` (redb) into the `CollectionManager` lifecycle. Every ingest writes chunks through to `<data_dir>/<collection>/chunks.redb` in a batched atomic transaction, and `load_collection` rehydrates `loaded.chunks` from that database on startup. `next_id` is recomputed as `max(seen_id) + 1` so post-restart ingests pick up where they left off. Before this fix, chunks lived only in an in-memory `HashMap` that was initialized empty on every restart, which meant process restarts and crashes silently destroyed ingested data even when the persistent volume was correctly mounted.
- **Data directory honors `DATA_DIR`.** Ensure the engine writes indexes to the configured `DATA_DIR` persistent path rather than the container's ephemeral writable layer, so data survives restarts when a persistent volume is mounted.

## [0.1.0] - 2026-04-25

### Added

- Initial release as Captain's embedded search engine.
- Tantivy BM25 full-text search with bitset-faceted metadata.
- USearch HNSW vector search (mmap-backed, disk-persistent).
- Reciprocal Rank Fusion hybrid search.
- Named vector spaces — multiple embedding models per collection.
- One-click model upgrades with background re-embedding and atomic swap.
- Parent-child relationships with TAMS-compatible hierarchies.
- Query-time scoring: recency decay, metadata boost, relationship boost.
- Native query embedding via Candle BGE-small.
- External GPU embedding endpoint support.

[Unreleased]: https://github.com/runcaptain/compass/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/runcaptain/compass/releases/tag/v0.1.0
