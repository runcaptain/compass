# Compass

Embedded vector + full-text search engine for Captain. Single binary, zero external dependencies.

## Build & Run

```bash
cargo build            # compile (debug mode)
cargo run              # start server on port 4001
cargo build --release  # compile optimized binary
PORT=8080 cargo run    # custom port
```

## Docker

```bash
docker build -t compass .
docker run -p 4001:4001 -v ./data:/app/data compass
```

## Architecture

- Tantivy for BM25 full-text search with precomputed bitset faceting (microsecond facets)
- USearch HNSW for vector search (mmap-backed, disk-persistent)
- Hybrid search via Reciprocal Rank Fusion (RRF, k=60)
- Query embedding via Candle BGE-small-en-v1.5 (native Rust, ~2-3ms/query on CPU)
- Fallback: distilled Model2Vec for sub-100μs query embedding
- REST API via Axum on configurable port (default 4001)

## Data Layout

```
./data/
├── models/
│   ├── bge-small/          # BGE-small-en-v1.5 weights (download from HuggingFace)
│   └── distilled/          # Distilled Model2Vec lookup table
├── {collection-name}/
│   ├── collection.json     # Collection metadata
│   ├── tantivy/            # Tantivy FTS index files
│   └── vectors/
│       ├── vectors.bin     # Raw embedding vectors
│       └── usearch.index   # HNSW index (mmap-backed)
```

## API

```
POST   /collections                   — Create collection
GET    /collections                   — List collections
GET    /collections/:name             — Get collection info
DELETE /collections/:name             — Delete collection + data

POST   /collections/:name/ingest     — Bulk ingest chunks
POST   /collections/:name/search     — Search (fts|semantic|hybrid)
GET    /collections/:name/facets     — Facet counts

GET    /health                        — Health check
```

## Embedding Models

Download BGE-small for query embedding:
```bash
huggingface-cli download BAAI/bge-small-en-v1.5 --local-dir ./data/models/bge-small/
```

Without BGE-small, semantic/hybrid search won't work. FTS search works without any model.

## Conventions

- Port 4001 (4000 conflicts with logdog, 3000 with runcaptain)
- Comments explain what code does for readers unfamiliar with Rust syntax
- No external database dependencies — everything is embedded and disk-backed
- Collection names must be kebab-case (a-z, 0-9, hyphens)
