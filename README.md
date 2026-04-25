# Compass

Embedded vector + full-text search engine for [Captain](https://runcaptain.com). Single binary, zero external dependencies. Built for on-prem enterprise deployments where customer data cannot leave their VPC.

## What it does

- **Full-text search** via Tantivy (BM25) with precomputed bitset faceting (microsecond facets)
- **Vector search** via USearch HNSW (mmap-backed, disk-persistent)
- **Hybrid search** via Reciprocal Rank Fusion (RRF, k=60)
- **Named vector spaces** ... run multiple embedding models on the same collection
- **One-click model upgrades** with background re-embedding and atomic swap
- **Parent-child documents** with relationship-aware scoring (TAMS video search compatible)
- **Query-time scoring**: recency decay, metadata boosting, relationship boosting
- **Metadata filtering**: typed values (string, int, float, bool, timestamp, string list)
- **Native query embedding** via Candle BGE-small (Rust, no Python). GPU endpoint support for larger models.
- **Fully offline**. No API calls. Model weights on disk. Data never leaves the machine.

## Quick start

```bash
cargo build --release
./target/release/compass
# Listening on http://localhost:4001
```

Environment variables: `PORT` (default 4001), `DATA_DIR` (default ./data).

## Examples

### Basic: create, ingest, search

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

### Named vector spaces

Create a collection with two embedding models (text + multimodal):

```bash
curl -X POST localhost:4001/collections \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "media",
    "vector_spaces": {
      "qwen3": {"dims": 1024, "model": "Qwen/Qwen3-Embedding-8B", "status": "active"},
      "qwen3-vl": {"dims": 896, "model": "Qwen/Qwen3-VL-Embedding-2B", "status": "active"}
    }
  }'

# Search a specific vector space
curl -X POST localhost:4001/collections/media/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "sunset over ocean", "mode": "semantic", "vector_space": "qwen3-vl"}'
```

### Recency bias + metadata boosting

```bash
curl -X POST localhost:4001/collections/docs/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "quarterly report",
    "mode": "hybrid",
    "recency": {"field": "created_at", "half_life_days": 30, "min_score": 0.1},
    "boosts": [
      {"field": "department", "value": "Legal", "weight": 2.0},
      {"field": "priority", "gte": 3, "weight": 1.5}
    ]
  }'
```

Recency decay formula: `score *= max(min_score, 2^(-age_days / half_life_days))`. A 30-day-old doc scores 0.5x. A 60-day-old doc scores 0.25x. The floor prevents old docs from vanishing entirely.

### Metadata filtering

```bash
curl -X POST localhost:4001/collections/docs/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "compliance",
    "filters": {"department": "Legal", "active": true}
  }'
```

Filters are hard constraints applied before scoring. Only matching documents are scored and returned.

### Parent-child documents + relationship boost

Ingest a document hierarchy using `client_id` and `parent_ref` to link chunks within a single batch:

```bash
curl -X POST localhost:4001/collections/media/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "chunks": [
      {
        "client_id": "src-001",
        "file_id": "video-001",
        "chunk_index": 0,
        "doc_type": "source",
        "text": "Premier League: Arsenal vs Chelsea",
        "metadata": {"asset_type": "video", "created_at": "2026-03-15T15:00:00Z"}
      },
      {
        "client_id": "seg-001",
        "file_id": "segment-001",
        "chunk_index": 0,
        "doc_type": "segment",
        "parent_ref": "src-001",
        "group_id": "src-001",
        "text": "Goal celebration, minute 34",
        "metadata": {"timerange_start": 2040.0, "timerange_end": 2055.0, "scene_type": "goal"}
      }
    ]
  }'
```

Then search with relationship boosting. Segments whose parent also matches the query get a score boost:

```bash
curl -X POST localhost:4001/collections/media/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "goal celebration",
    "relationship_boost": {"parent_weight": 0.3, "sibling_weight": 0.1, "mode": "max"}
  }'
```

### Facets

```bash
# Get facet counts for all metadata fields
curl 'localhost:4001/collections/docs/facets'

# Facet counts scoped to a text query
curl 'localhost:4001/collections/docs/facets?query=compliance'
```

### One-click model upgrade

```bash
# 1. Add a new vector space
curl -X POST localhost:4001/collections/docs/vector-spaces \
  -H 'Content-Type: application/json' \
  -d '{"name": "qwen3", "dims": 1024, "model": "Qwen/Qwen3-Embedding-8B"}'

# 2. Trigger re-embedding (uses external GPU endpoint for speed)
curl -X POST localhost:4001/collections/docs/vector-spaces/qwen3/rebuild \
  -H 'Content-Type: application/json' \
  -d '{"embed_endpoint": "http://gpu-server:8080/embed"}'

# 3. Check progress
curl localhost:4001/collections/docs/vector-spaces/qwen3/status

# 4. Swap the default (zero downtime, old space stays for rollback)
curl -X PUT localhost:4001/collections/docs/default-vector-space \
  -H 'Content-Type: application/json' \
  -d '{"name": "qwen3"}'

# 5. (Optional) Delete old space when you're confident
curl -X DELETE localhost:4001/collections/docs/vector-spaces/default
```

## Docker

```bash
docker build -t compass .
docker run -p 4001:4001 -v ./data:/app/data compass
```

## Embedding models

Compass ships with a built-in BGE-small embedder (CPU, ~2-3ms/query). For production, use a GPU embedding server:

```bash
# Run HuggingFace TEI with the flagship model
docker run -p 8080:80 --gpus all ghcr.io/huggingface/text-embeddings-inference \
  --model-id Qwen/Qwen3-Embedding-8B
```

| Model | MTEB | Dims | Modality | License | GPU |
|-------|------|------|----------|---------|-----|
| Qwen3-Embedding-8B | 70.58 | 32-7168 | Text + code | Apache 2.0 | 16GB+ |
| Qwen3-VL-Embedding-2B | 0.945 cross-modal | 896+ | Text + image + video | Apache 2.0 | 8GB+ |
| BGE-small-en-v1.5 | ~63 | 384 | Text | MIT | CPU only |

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
