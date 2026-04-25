# Compass

Embedded vector + full-text search engine for [Captain](https://runcaptain.com). Single binary, zero external dependencies. Built for on-prem enterprise deployments where customer data cannot leave their VPC.

## What it does

- **Full-text search** via Tantivy (BM25) with precomputed bitset faceting (microsecond facets)
- **Vector search** via USearch HNSW (mmap-backed, disk-persistent)
- **Hybrid search** via Reciprocal Rank Fusion (RRF, k=60)
- **Named vector spaces** so you can run multiple embedding models on the same collection
- **One-click model upgrades** with background re-embedding and atomic swap
- **Parent-child documents** with relationship-aware scoring (TAMS video search compatible)
- **Query-time scoring pipeline**: recency decay, metadata boosting, relationship boosting
- **Typed metadata**: string, int, float, bool, timestamp, string lists
- **Native query embedding** via Candle BGE-small (Rust, no Python). External GPU endpoint support for larger models.

## Quick start

```bash
cargo build --release
./target/release/compass
# Listening on http://localhost:4001
```

```bash
# Create a collection
curl -X POST localhost:4001/collections \
  -H 'Content-Type: application/json' \
  -d '{"name": "docs"}'

# Ingest chunks
curl -X POST localhost:4001/collections/docs/ingest \
  -H 'Content-Type: application/json' \
  -d '{"chunks": [{"file_id": "readme", "chunk_index": 0, "text": "Compass is a search engine"}]}'

# Search
curl -X POST localhost:4001/collections/docs/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "search engine", "mode": "hybrid"}'
```

## Docker

```bash
docker build -t compass .
docker run -p 4001:4001 -v ./data:/app/data compass
```

## Embedding models

Compass ships with a built-in BGE-small embedder (CPU, ~2-3ms/query). For production, use a GPU embedding server:

```bash
# Download the flagship model (Qwen3-Embedding-8B, MTEB #1)
huggingface-cli download Qwen/Qwen3-Embedding-8B --local-dir ./data/models/qwen3

# Or use HuggingFace TEI as an external embedding endpoint
docker run -p 8080:80 ghcr.io/huggingface/text-embeddings-inference \
  --model-id Qwen/Qwen3-Embedding-8B
```

Then trigger a rebuild with the external endpoint:

```bash
curl -X POST localhost:4001/collections/docs/vector-spaces/qwen3/rebuild \
  -H 'Content-Type: application/json' \
  -d '{"embed_endpoint": "http://localhost:8080/embed"}'
```

## API

```
POST   /collections                              Create collection
GET    /collections                              List collections
GET    /collections/:name                        Get collection info
DELETE /collections/:name                        Delete collection + data

POST   /collections/:name/ingest                 Bulk ingest chunks
POST   /collections/:name/search                 Search (fts|semantic|hybrid)
GET    /collections/:name/facets                 Facet counts

POST   /collections/:name/vector-spaces          Add a vector space
GET    /collections/:name/vector-spaces          List vector spaces
DELETE /collections/:name/vector-spaces/:space   Remove a vector space
POST   /collections/:name/vector-spaces/:space/rebuild   Trigger re-embedding
GET    /collections/:name/vector-spaces/:space/status    Rebuild progress
PUT    /collections/:name/default-vector-space   Switch default space

GET    /health                                   Health check
```

## License

Private. Copyright Captain (YC W26).
