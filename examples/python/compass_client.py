"""
Compass Python Client — all API interactions in one file.

Start Compass first:
    docker run -d --name compass -p 4001:4001 compass

Then:
    pip install requests
    python compass_client.py

Covers: collections, ingest, search (FTS/semantic/hybrid), vector spaces,
metadata filtering, recency, boosting, parent-child, facets, model upgrades.
"""

import requests
import json
import time

BASE = "http://localhost:4001"


def pp(label: str, data):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(json.dumps(data, indent=2) if isinstance(data, (dict, list)) else data)


# ─── Health ───────────────────────────────────────────────────────────

def health():
    r = requests.get(f"{BASE}/health")
    pp("Health", r.json())
    return r.json()


# ─── Collections ──────────────────────────────────────────────────────

def create_collection(name: str, vector_spaces: dict = None):
    body = {"name": name}
    if vector_spaces:
        body["vector_spaces"] = vector_spaces
    r = requests.post(f"{BASE}/collections", json=body)
    pp(f"Create collection '{name}'", r.json() if r.ok else r.text)
    return r

def list_collections():
    r = requests.get(f"{BASE}/collections")
    pp("List collections", r.json())
    return r.json()

def get_collection(name: str):
    r = requests.get(f"{BASE}/collections/{name}")
    pp(f"Collection '{name}'", r.json() if r.ok else r.text)
    return r.json() if r.ok else None

def delete_collection(name: str):
    r = requests.delete(f"{BASE}/collections/{name}")
    pp(f"Delete '{name}'", r.status_code)
    return r.status_code


# ─── Ingest ───────────────────────────────────────────────────────────

def ingest(collection: str, chunks: list):
    r = requests.post(f"{BASE}/collections/{collection}/ingest", json={"chunks": chunks})
    pp(f"Ingest into '{collection}' ({len(chunks)} chunks)", r.json() if r.ok else r.text)
    return r.json() if r.ok else None

def ingest_with_embeddings(collection: str, chunks: list):
    """Ingest chunks with pre-computed embeddings (named vector spaces)."""
    r = requests.post(f"{BASE}/collections/{collection}/ingest", json={"chunks": chunks})
    pp(f"Ingest with embeddings ({len(chunks)} chunks)", r.json() if r.ok else r.text)
    return r.json() if r.ok else None


# ─── Search ───────────────────────────────────────────────────────────

def search(collection: str, query: str, mode: str = "hybrid", top_k: int = 5, **kwargs):
    body = {"query": query, "mode": mode, "top_k": top_k, **kwargs}
    r = requests.post(f"{BASE}/collections/{collection}/search", json=body)
    data = r.json() if r.ok else r.text
    pp(f"Search '{collection}' [{mode}]: \"{query}\"", data)
    return data

def search_with_filters(collection: str, query: str, filters: dict, **kwargs):
    return search(collection, query, filters=filters, **kwargs)

def search_with_recency(collection: str, query: str, preset: str, field: str, **kwargs):
    return search(collection, query, recency_preset=preset, recency_field=field, **kwargs)

def search_with_boosts(collection: str, query: str, boosts: list, **kwargs):
    return search(collection, query, boosts=boosts, **kwargs)

def search_with_vector(collection: str, query: str, vector: list, space: str = None, **kwargs):
    body = {"query": query, "query_vector": vector, "mode": "semantic", **kwargs}
    if space:
        body["vector_space"] = space
    r = requests.post(f"{BASE}/collections/{collection}/search", json=body)
    data = r.json() if r.ok else r.text
    pp(f"Vector search '{collection}': \"{query}\"", data)
    return data


# ─── Vector Spaces ────────────────────────────────────────────────────

def add_vector_space(collection: str, name: str, dims: int, model: str):
    r = requests.post(f"{BASE}/collections/{collection}/vector-spaces",
                      json={"name": name, "dims": dims, "model": model})
    pp(f"Add vector space '{name}' to '{collection}'", r.json() if r.ok else r.text)
    return r

def list_vector_spaces(collection: str):
    r = requests.get(f"{BASE}/collections/{collection}/vector-spaces")
    pp(f"Vector spaces in '{collection}'", r.json())
    return r.json()

def delete_vector_space(collection: str, space: str):
    r = requests.delete(f"{BASE}/collections/{collection}/vector-spaces/{space}")
    pp(f"Delete vector space '{space}'", r.status_code)

def rebuild_vector_space(collection: str, space: str, embed_endpoint: str = None):
    body = {}
    if embed_endpoint:
        body["embed_endpoint"] = embed_endpoint
    r = requests.post(f"{BASE}/collections/{collection}/vector-spaces/{space}/rebuild", json=body)
    pp(f"Rebuild '{space}'", r.status_code)

def vector_space_status(collection: str, space: str):
    r = requests.get(f"{BASE}/collections/{collection}/vector-spaces/{space}/status")
    pp(f"Rebuild status '{space}'", r.json() if r.ok else r.text)

def set_default_vector_space(collection: str, space: str):
    r = requests.put(f"{BASE}/collections/{collection}/default-vector-space",
                     json={"name": space})
    pp(f"Set default space to '{space}'", r.status_code)


# ─── Facets ───────────────────────────────────────────────────────────

def facets(collection: str, query: str = None):
    url = f"{BASE}/collections/{collection}/facets"
    if query:
        url += f"?query={query}"
    r = requests.get(url)
    pp(f"Facets for '{collection}'" + (f" (query: {query})" if query else ""), r.json())
    return r.json()


# ─── Demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Health check
    health()

    # 2. Clean slate
    delete_collection("demo")

    # 3. Create collection
    create_collection("demo")
    list_collections()

    # 4. Ingest documents
    ingest("demo", [
        {
            "file_id": "doc-1",
            "chunk_index": 0,
            "text": "Compass is a high-performance embedded search engine built in Rust.",
            "metadata": {"category": "engineering", "priority": 5, "tags": ["search", "rust"]}
        },
        {
            "file_id": "doc-2",
            "chunk_index": 0,
            "text": "Captain provides enterprise video intelligence and multimodal search.",
            "metadata": {"category": "product", "priority": 3, "tags": ["video", "ai"]}
        },
        {
            "file_id": "doc-3",
            "chunk_index": 0,
            "text": "The quarterly compliance report is due next Friday for the Legal department.",
            "metadata": {"category": "legal", "priority": 8, "department": "Legal", "tags": ["compliance"]}
        },
        {
            "file_id": "doc-4",
            "chunk_index": 0,
            "text": "USearch HNSW provides approximate nearest neighbor search at 15k QPS.",
            "metadata": {"category": "engineering", "priority": 7, "tags": ["search", "performance"]}
        },
        {
            "file_id": "doc-5",
            "chunk_index": 0,
            "text": "Our on-premises deployment ensures customer data never leaves the VPC.",
            "metadata": {"category": "security", "priority": 9, "department": "Security", "tags": ["enterprise", "compliance"]}
        },
    ])

    # 5. Full-text search
    search("demo", "search engine", mode="fts")

    # 6. Semantic search
    search("demo", "fast vector database", mode="semantic")

    # 7. Hybrid search (FTS + semantic + RRF)
    search("demo", "high performance search", mode="hybrid")

    # 8. Metadata filtering
    search_with_filters("demo", "compliance", filters={"department": "Legal"})
    search_with_filters("demo", "search", filters={"priority": {"gte": 5}})
    search_with_filters("demo", "enterprise", filters={"tags": {"contains": "compliance"}})
    search_with_filters("demo", "search", filters={"category": {"in": ["engineering", "product"]}})

    # 9. Boosting
    search_with_boosts("demo", "report", boosts=[
        {"field": "department", "value": "Legal", "weight": 2.0},
        {"field": "priority", "gte": 7, "weight": 1.5},
    ])

    # 10. Facets
    facets("demo")
    facets("demo", query="search")

    # 11. Parent-child documents
    delete_collection("media")
    create_collection("media")
    ingest("media", [
        {
            "client_id": "video-001",
            "file_id": "video-001",
            "chunk_index": 0,
            "doc_type": "source",
            "text": "Premier League: Arsenal vs Chelsea highlights",
            "metadata": {"asset_type": "video"}
        },
        {
            "client_id": "seg-001",
            "file_id": "seg-001",
            "chunk_index": 0,
            "doc_type": "segment",
            "parent_ref": "video-001",
            "group_id": "video-001",
            "text": "Goal celebration at minute 34, Saka scores from close range",
            "metadata": {"timerange_start": 2040.0, "timerange_end": 2055.0, "scene_type": "goal"}
        },
        {
            "client_id": "seg-002",
            "file_id": "seg-002",
            "chunk_index": 0,
            "doc_type": "segment",
            "parent_ref": "video-001",
            "group_id": "video-001",
            "text": "Yellow card for Palmer after a dangerous tackle in midfield",
            "metadata": {"timerange_start": 2700.0, "timerange_end": 2715.0, "scene_type": "foul"}
        },
    ])

    # Search with relationship boost
    search("media", "goal celebration",
           relationship_boost={"parent_weight": 0.3, "sibling_weight": 0.1, "mode": "max"})

    # TAMS time-range search
    search("media", "goal",
           filters={"doc_type": {"in": ["segment"]}, "timerange_start": {"gte": 2000.0}, "timerange_end": {"lte": 2100.0}})

    # 12. Collection info
    get_collection("demo")
    get_collection("media")

    # 13. Vector spaces
    list_vector_spaces("demo")

    # 14. Cleanup
    # delete_collection("demo")
    # delete_collection("media")

    print("\n" + "="*60)
    print("  All done! Collections 'demo' and 'media' are ready to explore.")
    print("="*60)
