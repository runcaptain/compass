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
    tantivy_fts.rs   Full-text search via Tantivy (BM25) + facet treemaps.
    hybrid.rs        Reciprocal Rank Fusion (RRF, k=60) over FTS + semantic.
    ivf.rs           IVF clustering built at compaction (cold-read layout).
    cold.rs          Serve-from-storage query path (range reads, no attach).
    filter_index.rs  Roaring-treemap metadata filter index (warm pushdown).
    chunk_store.rs / chunk_cache.rs   redb chunk store + bounded LRU cache.
  collections/
    partitions.rs    Tenant-partition routing helpers.
    cloud.rs         Segment codec (CSEG0003), materialize, bucket config.
  storage/           Storage trait + local disk + object-store backends + LSM.
  metrics.rs         /metrics counters. telemetry.rs: opt-in usage pings.
```

## Vector backends

The engine uses USearch HNSW directly (CPU, mmap-backed, disk-persistent)
for warm serving, plus an IVF layout inside segments (`search/ivf.rs`) for
serve-from-storage cold reads. `compass-index-api` (a narrow `VectorIndex`
trait) and `compass-vector-gpu` (cuVS) exist as standalone crates for a
future GPU integration but are NOT wired into the engine — there is no
`COMPASS_BACKEND` knob and no `gpu` feature on the `compass` crate today.

## Storage layout

Per-collection state lives under `$DATA_DIR/<collection>/`:

```
data/<collection>/
  collection.json           Collection metadata (name, config, vector_spaces map, applied_seq)
  chunks.redb               Chunk bodies + metadata (redb; disk source of truth)
  relations.redb            Typed many-to-many chunk relations (redb)
  relationships.bin         Parent-child + sibling edges
  tantivy/                  Tantivy FTS index directory
  vectors/
    <space>.index           USearch HNSW graph — mmap-backed
    <space>.keymap          Internal HNSW key -> external chunk id mapping
    <space>.bin             CMV2 mmap vector file (torn-append-safe, per-batch durable)
```

In cloud mode the object-storage bucket additionally holds, per collection:
`collection.json` (bucket config), `manifest` (LSM manifest, CAS-committed),
`wal/{uuid}.frag` (WAL fragments), `segments/{uuid}` (CSEG0003 sectioned
segments: row-addressable metadata + IVF-clustered vectors; v2 readable), and `id-alloc` (CAS-leased chunk-id blocks).

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
4. Wire it into the engine (there is currently no runtime backend selector —
   proposing that wiring is part of such a PR; open an issue first).
5. Document the build prerequisites in `ARCHITECTURE.md` (this file).
6. Add a smoke binary under `src/bin/` that builds, queries, and prints latency.

The trait crate (`compass-index-api`) is pre-1.0; its API may shift between minor versions. We aim to stabilize at 1.0 with the GPU backend's GA.
