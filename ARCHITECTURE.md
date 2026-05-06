# Compass Architecture

This document is the contributor's map. If you're trying to figure out where to put new code, or what an existing path is doing, start here.

## Crate layout

Compass is a Cargo workspace with three crates:

```
compass/
  crates/
    compass-index-api/      Trait surface for vector backends (no I/O, no async)
    compass/                Main engine: HTTP API, FTS, vector search, embed
    compass-vector-gpu/     Optional cuVS GPU backend (Linux + CUDA only)
```

The split exists for a reason. `compass-index-api` is the smallest possible crate that downstream backends bind to: it has no I/O, no async runtime, no logging. New backends (CPU, GPU, IVF-PQ, sharded) can be developed against it without pulling in the rest of Compass.

`compass-vector-gpu` is opt-in. Default builds don't compile it. Enable with `--features gpu` from the umbrella crate, or depend on it directly for embedded use.

## Module map (compass crate)

```
crates/compass/src/
  main.rs            Binary entry point. Parses env, builds AppState, starts axum.
  models.rs          Domain types: Chunk, Document, Metadata, VectorSpaceConfig.
  scoring.rs         Query-time score adjustments: recency, metadata, relationships.
  api/               HTTP layer (axum routes).
    mod.rs           Router builder, AppState, shared error type.
    collections.rs   POST/GET/DELETE /collections, vector-space management.
    ingest.rs        Bulk ingest endpoint (writes chunks + vectors + relationships).
    search.rs        Search endpoint (mode = fts | semantic | hybrid).
  collections/       Collection state: persistence, rebuild orchestration.
    mod.rs           CollectionManager — owns the on-disk state for all collections.
    store.rs         Per-collection on-disk format (chunks, metadata, indexes).
    rebuild.rs       Re-embed + rebuild a vector space; status tracking.
    relationships.rs Parent-child + sibling graph for TAMS-style hierarchies.
  embed/             Embedding generation.
    mod.rs           EmbedState — picks BGE-small Candle path or distilled fallback.
    candle_bge.rs    BGE-small via Candle (Rust ML, ~2-3ms per query).
    distilled.rs     Model2Vec-distilled fallback (~100μs, lower accuracy).
  search/            Query-side search engines.
    mod.rs           SearchMode enum + re-exports.
    backend.rs       VectorIndex trait shim. UsearchHnswIndex (CPU) lives here.
    vector.rs        USearch HNSW build + search + persistence (CPU primitives).
    tantivy_fts.rs   Full-text search via Tantivy (BM25).
    hybrid.rs        Reciprocal Rank Fusion (RRF, k=60) over FTS + semantic.
```

## Vector backend abstraction

All vector backends implement `compass_index_api::VectorIndex`. The default backend is `UsearchHnswIndex` (CPU, mmap-backed, disk-persistent). The opt-in GPU backend is `compass_vector_gpu::CuvsHnswIndex` (CAGRA build on GPU, HNSW search on CPU).

Selection happens at startup in `search::backend::build_backend`, driven by the `COMPASS_BACKEND` environment variable:

| Value | Behavior |
|-------|----------|
| `cpu` (default) | USearch on CPU. Always available. |
| `gpu` | cuVS on GPU. Requires the `gpu` feature and a CUDA-capable device. Falls back to CPU with a warning if either is missing. |
| `auto` | Probe for GPU, fall back to CPU silently if unavailable. |

The trait is intentionally narrow: `build`, `add`, `search`, `len`, `dims`, `save`, `backend_name`. New backends should fit through this surface or extend it via a follow-up trait, not by branching on a concrete type.

## Storage layout

Per-collection state lives under `$DATA_DIR/<collection>/`:

```
data/<collection>/
  meta.json                 CollectionMetadata (name, default vector space, vector_spaces map)
  chunks.bin                Append-only log of Chunk records
  metadata.bin              Per-chunk metadata (typed values, bitset-faceted)
  fts/                      Tantivy directory
  vectors/<space>/
    index.usearch           USearch HNSW (CPU) — mmap-backed
    index.cuvs              cuVS HNSW (GPU build) — when COMPASS_BACKEND=gpu
    index.keymap            Internal HNSW key -> external chunk id mapping
    vectors.bin             Raw float buffer (used for brute-force fallback + rebuilds)
  relationships.bin         Parent-child + sibling edges
```

The disk format is the contract. Bumping it requires a migration path documented in CHANGELOG.md.

## Rebuild flow (model upgrades)

The "one-click model upgrade" feature relies on the rebuild path. When a new vector space is added with `POST /collections/<name>/vector-spaces` and a rebuild is triggered with `POST .../rebuild`:

1. The new space's status is set to `building`.
2. `rebuild.rs` walks all chunks in batches, re-embeds them via the configured `embed_endpoint` (or the in-process Candle path), and writes vectors into the new space.
3. Once complete, the new space's status flips to `active`. The default vector space can then be switched atomically.
4. The old space remains on disk for rollback until explicitly deleted.

GPU acceleration applies to the embedding step (via the external endpoint) and, when `--features gpu` is enabled, to the index construction step (CAGRA on GPU is ~12x faster than CPU HNSW build at dim 768/1024).

## GPU backend build prerequisites

Building `compass-vector-gpu` requires a Linux x86_64 host with:

- CUDA 12.0 or newer (12.4+ recommended).
- CMake 3.26+, gcc 11+ or clang 14+.
- NVIDIA GPU with compute capability 7.0+ (Volta or newer).
- 16+ GB VRAM for 1M × 768 vector builds with headroom; 24+ GB comfortable.

The first build of cuVS itself takes 30-60 minutes because it pulls a large C++/CUDA codebase via cmake. CI should cache the build artifact aggressively. The GPU crate is locked to a specific cuVS git tag (currently `v25.10.00`); the crates.io publish lags the source tree by months and is not used.

## Performance notes

USearch HNSW on a 16-core box serves around 15k QPS at p99 < 50ms for top-10 retrieval at dim 1024. Beyond that, scale by sharding collections across multiple Compass instances behind a load balancer.

cuVS CAGRA build on an A10G runs ~12x faster than USearch CPU build at the same parameters. Search after the CAGRA→HNSW conversion runs CPU-side at a profile similar to USearch; a future `cagra-search` feature will move search to the GPU for ~10-20x QPS improvement at high concurrency.

## Adding a new backend

1. Create a new crate `crates/compass-vector-<name>/`.
2. Depend on `compass-index-api` (workspace dep) and your backend library.
3. Implement `VectorIndex` (and `LoadableIndex` if loading from disk makes sense).
4. Add a `#[cfg(feature = "<name>")]`-gated branch in `search::backend::build_backend`.
5. Document the build prerequisites in `ARCHITECTURE.md` (this file).
6. Add a smoke binary under `src/bin/` that builds, queries, and prints latency.

The trait crate (`compass-index-api`) is pre-1.0; its API may shift between minor versions. We aim to stabilize at 1.0 with the GPU backend's GA.
