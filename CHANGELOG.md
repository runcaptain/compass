# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Cargo workspace layout.** Source moved from a single crate to a workspace under `crates/`. Splits: `compass` (umbrella + binary), `compass-index-api` (stable trait surface), `compass-vector-gpu` (optional GPU backend).
- **Vector index trait abstraction.** New `compass_index_api::VectorIndex` trait abstracts over CPU and GPU backends. The existing USearch path is wrapped in `UsearchHnswIndex`; new backends bind to the trait without touching call sites.
- **Optional GPU backend (`--features gpu`).** New `compass-vector-gpu` crate integrates NVIDIA cuVS for CAGRA→HNSW build acceleration. CPU-side search after conversion. Linux + CUDA 12+ only. See `ARCHITECTURE.md` for build prerequisites.
- **`COMPASS_BACKEND` env var.** Selects the vector backend at startup: `cpu` (default), `gpu`, `auto`.
- **`ARCHITECTURE.md`.** Contributor's map: crate layout, module map, storage format, rebuild flow.
- **`CONTRIBUTING.md`.** PR checklist, commit format, scope guidelines.
- **Throughput and scaling section** in README documenting QPS expectations and horizontal scaling pattern.
- **Rich metadata filters.** Filters now support numeric range (`gte`/`lte`), array `contains`, and set membership (`in`) in addition to exact match. Backward compatible: plain values still work as exact match.
- **TAMS time-range queries.** Filter segments by `timerange_start`/`timerange_end` using range operators. Combined with `doc_type` filtering and relationship boost for hierarchy-aware video search.
- **Configurable hybrid score weights.** New `score_weights` field on search requests controls RRF blending: `rrf_k`, `fts_weight`, `semantic_weight`. Previously hardcoded at k=60 with equal weights.
- **Recency presets.** `recency_preset` field accepts `aggressive` (3d), `recent` (7d), `mild` (30d), or `archive` (90d) instead of manually configuring the decay formula. Pair with `recency_field` to specify which timestamp metadata field to use.
- **Apache 2.0 license.**

### Changed

- **Cargo manifest** is now a workspace root with shared `[workspace.dependencies]` and `[workspace.package]` metadata. Per-crate manifests inherit version, edition, and rust-version.
- **Compiler profile** adds `lto = "thin"` and `codegen-units = 1` to release builds for better optimization. Adds debug symbols to bench builds.

### Deprecated

- Direct use of `compass::search::vector` from external code. New code should bind to `compass::search::VectorIndex`. The legacy `vector.rs` functions remain for in-tree callers and will be wrapped or migrated incrementally.

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
