<div align="center">
  <a href="https://runcaptain.com">
    <img src="https://files.buildwithfern.com/runcaptain.docs.buildwithfern.com/7bb93ea0ee016bc6af0af57fa0f197f331125faa2c217c1af23b63789952e295/docs/assets/Captain-Wordmark.svg" height="40" alt="Captain" />
  </a>
  <br><br>
  <img src="docs/compass-logo.svg" height="80" alt="Compass" />
  <h1>Compass</h1>
  <p>Embedded vector + full-text search engine. Single binary, zero external dependencies.</p>

  [![CI](https://github.com/runcaptain/compass/actions/workflows/ci.yml/badge.svg)](https://github.com/runcaptain/compass/actions/workflows/ci.yml)
  [![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
  [![Rust](https://img.shields.io/badge/rust-1.88%2B-orange.svg)](https://www.rust-lang.org)
</div>

Built by [Captain](https://runcaptain.com) for high-throughput retrieval in on-prem enterprise deployments where customer data cannot leave their VPC.

## What it does

- **Full-text search** via Tantivy (BM25) with precomputed bitset faceting (microsecond facets)
- **Vector search** via USearch HNSW (mmap-backed, disk-persistent)
- **Hybrid search** via Reciprocal Rank Fusion (RRF, k=60)
- **Memory-mapped vector storage**. Raw vectors live on disk, not in RAM. Zero-copy reads via mmap.
- **Disk-backed chunk metadata** via redb (pure Rust embedded DB). Handles millions of documents without loading them all into memory.
- **Incremental HNSW indexing**. Adding vectors appends to the index — no full rebuild required.
- **Named vector spaces** ... run multiple embedding models on the same collection
- **One-click model upgrades** with background re-embedding and atomic swap
- **Parent-child documents** with relationship-aware scoring (TAMS video search compatible)
- **Query-time scoring**: recency decay, metadata boosting, relationship boosting
- **Metadata filtering**: exact match, numeric range (gte/lte), array contains, set membership. Typed values (string, int, float, bool, timestamp, string list)
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

### Multi-space retrieval + reranking

Embed every document into multiple vector spaces at ingest time. At query time, search one space, multiple spaces, or all of them. A cross-encoder reranker re-scores the merged candidates for maximum accuracy.

```
Query
  |
  +-- Tantivy BM25 -----------> FTS candidates
  +-- Harrier HNSW -----------> text semantic candidates
  +-- Qwen3-VL HNSW ----------> multimodal candidates
  |
  v
  RRF merge (all three)
  |
  v
  Reranker (cross-encoder re-scores top candidates)
  |
  v
  Filter -> Score (recency, boost, relationships)
  |
  v
  Return top_k
```

Pick the right retrieval path for the query:

- **Text query, text docs:** search `harrier` space only
- **Text query, find images/video:** search `qwen3-vl` space (cross-modal)
- **Mixed collection, best accuracy:** search both spaces, RRF merge, rerank

```bash
# Search multiple vector spaces at once (merged via RRF, then reranked)
curl -X POST localhost:4001/collections/media/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "goal celebration slow motion",
    "mode": "hybrid",
    "vector_space": ["harrier", "qwen3-vl"],
    "top_k": 10
  }'
```

Three retrievers, one reranker, one scoring pipeline. The reranker doesn't care which retriever found the candidate. It just scores (query, text) relevance from scratch.

### Recency bias + metadata boosting

Pick a preset to favor newer results. Older docs score lower but never disappear:

```bash
curl -X POST localhost:4001/collections/docs/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "quarterly report",
    "recency_preset": "mild",
    "recency_field": "created_at",
    "boosts": [
      {"field": "department", "value": "Legal", "weight": 2.0},
      {"field": "priority", "gte": 3, "weight": 1.5}
    ]
  }'
```

Four presets. How quickly old docs lose ranking:

```
  strong bias ◄───────────────► weak bias

  aggressive    recent     mild       archive
  ├── 3d ──┤  ├── 7d ──┤  ├── 30d ──┤  ├── 90d ──┤
```


| Use case                                     | Preset       | Docs lose half their recency score after... | Old docs bottom out at... |
| -------------------------------------------- | ------------ | ------------------------------------------- | ------------------------- |
| Real-time alerts, live events, TAMS segments | `aggressive` | 3 days                                      | 5%                        |
| News, feeds, support tickets                 | `recent`     | 7 days                                      | 20%                       |
| Docs, reports, meeting notes                 | `mild`       | 30 days                                     | 30%                       |
| Long-lived content, legal docs, compliance   | `archive`    | 90 days                                     | 50%                       |


For full control, use `recency` instead (overrides any preset):

```bash
curl -X POST localhost:4001/collections/docs/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "quarterly report",
    "recency": {"field": "created_at", "half_life_days": 30, "min_score": 0.1}
  }'
```

Recency decay formula: `score *= max(min_score, 2^(-age_days / half_life_days))`. A 30-day-old doc scores 0.5x with the default. The `field` is always user-controlled. Compass never assumes which metadata field represents "time".

### Metadata filtering

Filters are hard constraints applied before scoring. Only matching documents are scored and returned.

```bash
# Exact match (string, bool, number)
curl -X POST localhost:4001/collections/docs/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "compliance",
    "filters": {"department": "Legal", "active": true}
  }'

# Numeric range (gte/lte)
curl -X POST localhost:4001/collections/docs/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "quarterly report",
    "filters": {"priority": {"gte": 3, "lte": 10}}
  }'

# Array contains
curl -X POST localhost:4001/collections/docs/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "highlights",
    "filters": {"tags": {"contains": "sports"}}
  }'

# Set membership (doc_type, category, etc.)
curl -X POST localhost:4001/collections/docs/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "meeting notes",
    "filters": {"doc_type": {"in": ["segment", "flow"]}}
  }'
```

All operators: exact match (backward compatible), `gte`/`lte` (numeric range), `contains` (array membership), `in` (set membership). Operators combine as AND across fields.

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
        "metadata": {"timerange_start_ms": 2040000, "timerange_end_ms": 2055000, "scene_type": "goal"}
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

### TAMS time-range search

[TAMS](https://github.com/bbc/tams) (Time-Addressable Media Store) is BBC R&D's open spec for media archives. Media is addressed by time, not by file. The data model: **Source** (logical content) → **Flow** (specific rendition) → **Segment** (time-bounded chunk with `timerange_start_ms`/`timerange_end_ms`).

Compass models this hierarchy via `doc_type` + `parent_id` + `group_id`. Time is stored as integer milliseconds throughout: `timerange_start_ms` and `timerange_end_ms` are numeric metadata fields. Instants (zero-duration events) are stored as segments where `timerange_start_ms == timerange_end_ms`.

Ingest segments with time range metadata, then query by content and time:

```bash
curl -X POST localhost:4001/collections/media/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "goal celebration",
    "filters": {
      "doc_type": {"in": ["segment"]},
      "timerange_start_ms": {"gte": 2040000},
      "timerange_end_ms": {"lte": 2100000}
    },
    "relationship_boost": {"parent_weight": 0.3, "sibling_weight": 0.1}
  }'
```

This finds segments matching "goal celebration" within the 2040000-2100000 ms window (33:60 → 35:00 in HH:MM:SS). Relationship boosting surfaces sibling segments and the parent flow alongside the match.

### Hybrid score weights

Control how FTS and semantic scores blend in hybrid mode. Useful when one signal matters more for your use case:

```bash
curl -X POST localhost:4001/collections/docs/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "quarterly earnings",
    "mode": "hybrid",
    "score_weights": {"rrf_k": 60.0, "fts_weight": 2.0, "semantic_weight": 0.5}
  }'
```

`rrf_k` is the RRF constant (default 60). Lower values amplify top-rank differences. `fts_weight` and `semantic_weight` control relative contribution (default 1.0 each). Set `fts_weight: 2.0` to favor keyword matches, or `semantic_weight: 2.0` when meaning matters more than exact terms.

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

Compass supports any embedding model via named vector spaces. Pick the right model for your use case:

### Recommended open-weight models (May 2026)

**For text search** (documents, code, multilingual), use Harrier or Qwen3:


| Model               | Score | Benchmark | Dims    | License    | GPU   | When to use                                          |
| ------------------- | ----- | --------- | ------- | ---------- | ----- | ---------------------------------------------------- |
| Harrier-OSS-v1-0.6B | ~68   | MTEB v2   | 768     | MIT        | 8GB   | Default for most deployments. Best quality-per-VRAM. |
| Qwen3-Embedding-8B  | 70.58 | MTEB v2   | 32-7168 | Apache 2.0 | 16GB+ | When you need top-2 accuracy and have an A100/H100.  |
| Harrier-OSS-v1-27B  | 74.3  | MTEB v2   | 1024    | MIT        | 48GB+ | Maximum accuracy. Requires H100.                     |


**For multimodal** (text queries finding images, video frames, PDFs):


| Model                 | Score | Benchmark          | Dims | License    | GPU  | When to use                                                           |
| --------------------- | ----- | ------------------ | ---- | ---------- | ---- | --------------------------------------------------------------------- |
| Qwen3-VL-Embedding-2B | 0.945 | MMEB (cross-modal) | 896+ | Apache 2.0 | 8GB+ | Best cross-modal accuracy. Handles text + image + video in one space. |


**For reranking** (re-scoring top results after retrieval):


| Model                     | Score      | Benchmark | License     | GPU   | When to use                                        |
| ------------------------- | ---------- | --------- | ----------- | ----- | -------------------------------------------------- |
| Qwen3-Reranker-8B         | 69.76      | MTEB-R    | Apache 2.0  | 16GB+ | Best open-source reranker for multilingual + code. |
| Contextual AI Reranker v2 | SOTA on QA | Various   | Open source | 8GB+  | Best for Q&A-style retrieval.                      |


**CPU-only fallback** (no GPU available, degraded mode):


| Model             | Score | Benchmark | Dims | License | When to use                                                                      |
| ----------------- | ----- | --------- | ---- | ------- | -------------------------------------------------------------------------------- |
| BGE-small-en-v1.5 | ~63   | MTEB      | 384  | MIT     | Local dev, CI/CD tests, or hardware with no GPU. Not recommended for production. |


### Typical setup

Most deployments need two vector spaces: one for text, one for multimodal (if applicable).

```bash
# Run HuggingFace TEI with the recommended text model
docker run -p 8080:80 --gpus all ghcr.io/huggingface/text-embeddings-inference \
  --model-id microsoft/harrier-oss-v1-0.6b
```

MTEB and MMEB are different benchmarks on different scales. MTEB scores are 0-100 (text tasks). MMEB scores are 0-1 (cross-modal retrieval). They cannot be compared directly.

## Storage architecture

Compass keeps vector data and chunk metadata on disk, not in RAM.

```
data/{collection}/
├── collection.json                  # Collection metadata (name, dims, spaces)
├── relationships.bin                # Parent-child + sibling graph
├── tantivy/                         # BM25 inverted index (disk-backed)
└── vectors/
    ├── {space}.index                # USearch HNSW graph (mmap on read)
    ├── {space}.bin                  # Raw f32 vectors (mmap via MmapVectors)
    └── {space}.keymap               # HNSW key → chunk ID mapping
```

**Vectors**: Stored in a flat `[u32 dims][u32 count][f32...]` file, memory-mapped at query time. Adding vectors appends to the file and remaps — no full rewrite. At 1M vectors × 768 dims this is ~3GB on disk, near-zero RSS.

**HNSW index**: Built incrementally via USearch `.add()` + `.save()`. Loaded via `.load()` for mutation or `.view()` for read-only mmap. The graph structure is separate from the raw vectors.

**Chunk metadata**: Persisted via redb (pure Rust, ACID, MVCC). Point lookups by chunk ID during search result assembly. Batch inserts during ingestion.

**Ingestion path**: New vectors are appended to the mmap file, inserted into the HNSW graph incrementally, and chunk metadata is written to redb — all without cloning existing data.

## Throughput and scaling

**Query throughput.** USearch HNSW serves around 15k QPS per instance on a 16-core box at p99 < 50ms for top-10 retrieval. For very high QPS workloads, shard collections across multiple Compass instances behind a load balancer.

**Indexing throughput.** Point `embed_endpoint` at a GPU-backed HuggingFace TEI or vLLM cluster. A single A10G handles around 1,500 docs/sec on Qwen3-Embedding-8B. Scale linearly by adding GPU replicas.

```bash
# Spin up TEI with the recommended text model
docker run -p 8080:80 --gpus all ghcr.io/huggingface/text-embeddings-inference \
  --model-id Qwen/Qwen3-Embedding-8B

# Point Compass at it during rebuild or ingestion
curl -X POST localhost:4001/collections/docs/vector-spaces/qwen3/rebuild \
  -d '{"embed_endpoint": "http://localhost:8080/embed"}'
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

POST   /collections/:name/vector-spaces                Add a vector space
GET    /collections/:name/vector-spaces                List vector spaces
DELETE /collections/:name/vector-spaces/:space         Remove a vector space
POST   /collections/:name/vector-spaces/:space/rebuild Trigger re-embedding
GET    /collections/:name/vector-spaces/:space/status  Rebuild progress
PUT    /collections/:name/default-vector-space         Switch default space

GET    /health                                         Health check
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, PR guidelines, and commit conventions.

## Security

To report a vulnerability, email **security@runcaptain.com**. See [SECURITY.md](SECURITY.md) for details.

## License

Apache 2.0. See [LICENSE](LICENSE).
