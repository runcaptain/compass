# Compass

Embedded vector + full-text search engine for Captain. Single binary, zero external dependencies. Apache 2.0.

## Build & Run

```bash
cargo build            # compile (debug mode)
cargo run              # start server on port 4001
cargo build --release  # compile optimized binary
PORT=8080 cargo run    # custom port
cargo check            # type-check only (linking has a known issue on Windows)
cargo test --workspace # run all tests
```

## Docker

```bash
docker build -t compass .
docker run -p 4001:4001 -v ./data:/app/data compass
```

## Workspace Layout

```
crates/
  compass/                Main engine binary (Axum API, search, scoring, embed)
  compass-index-api/      VectorIndex trait (no I/O, no async)
  compass-vector-gpu/     cuVS GPU backend crate (standalone; not yet wired into the engine)
```

## Architecture

- Tantivy for BM25 full-text search with precomputed bitset faceting (microsecond facets)
- USearch HNSW for vector search (mmap-backed, disk-persistent)
- Hybrid search via Reciprocal Rank Fusion (RRF, configurable k and weights)
- Query embedding via Candle BGE-small-en-v1.5 (native Rust, ~2-3ms/query on CPU)
- Fallback: distilled Model2Vec for sub-100us query embedding
- REST API via Axum on configurable port (default 4001)
- Metadata filtering: exact match, numeric range (gte/lte), array contains, set membership
- Recency presets: aggressive (3d), recent (7d), mild (30d), archive (90d)
- Parent-child document hierarchy (TAMS-compatible: source/flow/segment)
- Typed many-to-many chunk relations (redb-backed, disk-served)
- Soft deletes (tombstones) + compaction
- Optional object-storage persistence (`--features object-storage`, `COMPASS_STORAGE=s3://…|gs://…|az://…`): LSM of WAL fragments + CAS manifest; bucket is the source of truth, local indexes rebuild from it on boot

## Key Modules

- `models.rs` — All request/response types, FilterValue, FilterCondition, ScoreWeights, RecencyConfig
- `filter.rs` — Post-retrieval metadata filtering with operator support
- `scoring.rs` — Query-time scoring pipeline (recency, metadata boost, relationship boost)
- `collections/mod.rs` — CollectionManager: ingest, search, vector space CRUD
- `collections/relationships.rs` — Parent-child + sibling graph
- `search/hybrid.rs` — RRF merge with configurable weights
- `search/vector.rs` — USearch HNSW build + search
- `search/tantivy_fts.rs` — Full-text search + bitset faceting

## Data Layout

```
./data/
├── models/
│   ├── bge-small/          # BGE-small-en-v1.5 weights (download from HuggingFace)
│   └── distilled/          # Distilled Model2Vec lookup table
├── {collection-name}/
│   ├── collection.json     # Collection metadata
│   ├── chunks.redb         # Chunk metadata + tombstones (redb)
│   ├── relations.redb      # Typed relations + endpoint indexes (redb)
│   ├── relationships.bin   # Parent-child + sibling edges
│   ├── tantivy/            # Tantivy FTS index files
│   └── vectors/
│       ├── {space}.bin     # Raw embedding vectors per named space
│       └── {space}.index   # HNSW index per named space
```

## API

```
POST   /collections                                    Create collection
GET    /collections                                    List collections
GET    /collections/:name                              Get collection info
DELETE /collections/:name                              Delete collection + data

POST   /collections/:name/ingest                       Bulk ingest chunks
POST   /collections/:name/search                       Search (fts|semantic|hybrid)
GET    /collections/:name/facets                       Facet counts

DELETE /collections/:name/chunks/:id                   Soft-delete one chunk
POST   /collections/:name/delete                       Soft-delete by ids and/or filters
POST   /collections/:name/compact                      Compact (object-storage mode)

POST   /collections/:name/relations                    Create typed chunk relations
DELETE /collections/:name/relations/:relation_id       Delete a relation
GET    /collections/:name/chunks/:id/relations         List a chunk's relations

POST   /collections/:name/vector-spaces                Add a vector space
GET    /collections/:name/vector-spaces                List vector spaces
DELETE /collections/:name/vector-spaces/:space         Remove a vector space
POST   /collections/:name/vector-spaces/:space/rebuild Trigger re-embedding
GET    /collections/:name/vector-spaces/:space/status  Rebuild progress
PUT    /collections/:name/default-vector-space         Switch default space

GET    /collections/:name/segments/at                  Temporal segment lookup (TAMS)

GET    /health                                         Health check
GET    /metrics                                        Prometheus-text metrics
```

## Embedding Models

Download BGE-small for query embedding:
```bash
huggingface-cli download BAAI/bge-small-en-v1.5 --local-dir ./data/models/bge-small/
```

Without BGE-small, semantic/hybrid search won't work. FTS search works without any model.

## Conventions

- Port 4001 (4000 conflicts with logdog, 3000 with runcaptain)
- No external database dependencies — everything is embedded and disk-backed
- Collection names must be kebab-case (a-z, 0-9, hyphens)
- Update CHANGELOG.md every time a feature is pushed to git (PR, branch, or main)
- Known issue: Windows linker mismatch between esaxx-rs and cxx. Use `cargo check` on Windows, build/test in Docker or Linux.
