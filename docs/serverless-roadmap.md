# Serverless Roadmap

> Status: PLANNED. Target: evolve Compass from a cloud-durable single-node
> engine (v0.3.0) into a fully serverless database — storage/compute separated,
> stateless workers, bounded cold starts, scale-to-zero — with **every item
> additive and open source** under Apache 2.0. Local-first, zero-config
> operation remains the default at every step; all serverless behavior is
> opt-in via config.

## Where v0.3.0 leaves us

Done and hardened (the foundation):

- Object storage as source of truth: LSM of immutable UUID-keyed WAL fragments,
  CAS-committed manifest, compaction with deferred GC (`storage/lsm.rs`)
- Multi-writer safety at the storage layer, proven against real S3 semantics
- Full ephemeral recovery — chunks, hierarchy, typed relations
  (`collections/cloud.rs::materialize`, `rebuild_collection_from_storage`)
- Prefix scoping, O(namespaces) discovery, compensation discipline on every
  partial-failure path

Not yet serverless:

- Serving is stateful: every node rebuilds full local indexes (Tantivy + HNSW)
- Boot rebuilds ALL collections; `materialize` holds a collection's live set in RAM
- Writes require the stateful node that has the collection attached
- Nodes never re-read the manifest, so multi-node views diverge
- One global API key; no per-namespace scoping or metering

## Design rules (apply to every phase)

1. **Additive or it doesn't ship.** New behavior behind config/env/roles;
   `cargo run` with no configuration behaves exactly as today.
2. **The bucket is the only shared state.** Coordination primitives are the
   `Storage` trait's CAS/create-only operations — no new external dependencies
   (no etcd, no Redis, no Postgres).
3. **Every phase lands with**: unit tests, env-gated real-S3 integration tests
   (MinIO), an adversarial review round, CHANGELOG + docs.
4. **Pluggable seams stay public**: `Storage` (backends), `VectorIndex`
   (`compass-index-api`), serving modes per collection.

---

## Phase 0 — Guardrails (days) → part of v0.4

| Item | Detail |
|---|---|
| 0.1 CI covers the cloud build | Add an `object-storage`-feature test job to CI (today CI tests the default build only) and a MinIO service container running the `s3_integration` tests on every PR. |
| 0.2 DCO | Enforce Developer Certificate of Origin sign-off in CI before external contributors arrive; keeps future licensing options open without a CLA's friction. |
| 0.3 Public tracking | This document + a GitHub milestone per phase, issues per work item. |

## Phase 1 — Stateless write path (~2–3 wks) → v0.4

Writes stop requiring a node that has the collection attached.

| Item | Detail |
|---|---|
| 1.1 Collection config in the bucket | `create_collection` writes `{ns}/collection.json` (dims, vector spaces, default space) via create-only/CAS. Recovery reads it instead of inferring specs from recovered embeddings (also fixes the "model: recovered" inference in `rebuild_collection_from_storage`). |
| 1.2 Id-block allocation | Chunk ids are minted from a local `next_id` today. Stateless writers lease id blocks via CAS on a `{ns}/id-alloc` object (e.g. 10k-id blocks); a crashed writer leaks at most one block (ids are monotonic, gaps are already fine). |
| 1.3 Writer role | `COMPASS_ROLE=writer` (or per-request): validate against the bucket-cached collection config → append WAL fragment → CAS manifest → return `{seq}`. No local index update, no collections lock. Consistency contract: **durable immediately, searchable after reader refresh** (Phase 3) or compaction. |
| 1.4 Tests | A writer node with an empty disk ingests; a reader node serves it after refresh; MinIO integration. |

## Phase 2 — Lazy attach + streaming materialize (~2–3 wks) → v0.4

Cold start stops being O(all data), RAM stops being O(collection).

| Item | Detail |
|---|---|
| 2.1 Attach-on-demand | Boot registers namespaces (already O(ns) via `list_dirs`) without rebuilding. First request to a namespace triggers attach; an LRU with a configurable budget (`COMPASS_MAX_ATTACHED` / memory target) detaches idle collections (safe — the bucket is the source of truth; detach deletes local state). |
| 2.2 Streaming materialize | `materialize` gains a sink-based variant folding segments + WAL directly into redb / Tantivy writer / mmap appends in bounded batches — no full-collection HashMap. The in-RAM variant remains for compaction (which needs the full fold anyway until Phase 5). |
| 2.3 Observability | Attach-duration histograms; `/health` reports attached/registered counts. |

**Acceptance:** boot time independent of collection count; attaching an
N-chunk collection runs at bounded RSS.

## Phase 3 — Manifest watch + read consistency (~2 wks) → v0.4

Multiple readers converge on the same view; writers' output becomes visible.

| Item | Detail |
|---|---|
| 3.1 Refresher | Per-attached-namespace background task re-reads the manifest (compare `Version` tokens; manifests are small). Interval configurable. |
| 3.2 Incremental replay | Apply only fragments with `seq >` last-applied to the local indexes — the existing ingest/delete/relation apply logic refactored into a reusable `apply_fragment` so refresh, attach, and ingest share one code path. |
| 3.3 Read-your-writes | Writes return the manifest `seq`; queries accept optional `min_seq` (fast-path refresh or bounded wait). |
| 3.4 Tests | Two managers on one bucket: write via A, visible via B within the interval; tombstones and relations replay correctly. |

**Phase 1–3 outcome: “warm serverless.”** Any worker attaches any namespace
on demand; writes are stateless; readers converge. Cold start is bounded but
still proportional to index size (fixed in Phase 6).

## Phase 4 — Compactor role + leases (~1–2 wks) → v0.5

| Item | Detail |
|---|---|
| 4.1 Lease primitive | `{ns}/lease/compactor` object via `put_if_not_exists` with a TTL payload; expired leases are stolen via CAS. Clock-skew caveat documented (leases are long relative to plausible skew; compaction is idempotent and CAS-guarded regardless — a double-run wastes work, never corrupts). |
| 4.2 Compactor role | `COMPASS_ROLE=compactor` (same binary): scan namespaces, threshold-check, lease, run the already-storage-only `compact_storage`, release. Serving nodes' inline auto-compaction turns off when an external compactor is configured. |

## Phase 5 — Segment format v2 (~2–3 wks) → v0.5

The JSON segment becomes a binary, sectioned, range-readable format.

| Item | Detail |
|---|---|
| 5.1 Layout | Magic + version + TOC (section → offset/len), sections: chunk metadata, text, embeddings per space (contiguous f32 LE rows), relations, the serialized filter-index treemaps (the persistence code exists, currently unwired), id high-water. Zstd per section. |
| 5.2 Back-compat | `decode_segment` already falls back by version; v1 JSON segments remain readable, compaction rewrites to v2. |
| 5.3 Range reads | Attach and query paths fetch only the sections they need via `get_range` (already a true byte-range read on both backends). |

## Phase 6 — Serve directly from object storage (~2–4 mo) → v0.6 *(the innovation epic)*

| Item | Detail |
|---|---|
| 6.1 Vector: per-segment IVF | Compaction runs k-means per segment; centroids live in the segment TOC (tiny, RAM-cacheable per namespace), posting lists are contiguous row ranges in the embeddings section. Query: route by centroids → range-read `nprobe` cells → exact-score → merge across segments + brute-force the (small by construction) WAL tail. Filters intersect posting row-ids with the segment's treemaps. Implemented as a `VectorIndex` (`compass-index-api`) impl, pluggable next to USearch HNSW. |
| 6.2 FTS over storage | Spike: tantivy custom `Directory` over the `Storage` trait with a local block cache. Decision gate after the spike; fallback design is per-segment mini-indexes built at compaction and fetched on attach. |
| 6.3 Serving modes | Per-collection `serving_mode: attached \| stateless` (default `attached` — today's behavior). Stateless mode never rebuilds local indexes. |

**Acceptance:** recall@10 within an agreed delta of HNSW on standard
benchmarks; p95 latency targets on cold namespaces; RAM ceiling per attached
namespace measured and documented.

## Phase 7 — Tenancy, metering, limits (~2–3 wks, parallel with 6) → v0.6

| Item | Detail |
|---|---|
| 7.1 Scoped keys | Per-collection API-key scopes extending `AuthConfig`; the single global key keeps working. |
| 7.2 Usage events | Per-request metering (namespace, operation, read/write bytes, query units) emitted as structured `tracing` events with an optional export sink. OSS emits; any billing pipeline (open or closed) aggregates. |
| 7.3 Quotas | Per-key/per-namespace rate limits and quotas via tower middleware, config-driven. |

## Phase 8 — Open control plane (~4–6 wks) → v0.7

An OSS reference implementation of the service layer — same repo, new crate.

| Item | Detail |
|---|---|
| 8.1 Router | `crates/compass-router` (or `COMPASS_ROLE=router`): rendezvous-hash namespaces → workers, worker registry via storage-backed heartbeat objects (no new dependencies), request proxying with attach-on-demand, drain/failover. |
| 8.2 Scale-to-zero | Idle detach (Phase 2's LRU) + pluggable worker-lifecycle hooks; ship Kubernetes manifests/HPA examples and a compose profile as reference deployments. |
| 8.3 Ops docs | Capacity planning, S3 request-cost model, tuning guide. |

A hosted commercial offering (billing aggregation, org management,
dashboards) can be built on top of all of this later without forking —
every technical capability above stays in the open engine.

---

## Sequencing

```
v0.4  Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3        (~6-8 wks)  "warm serverless"
v0.5  Phase 4 ──► Phase 5                                (~3-5 wks)
v0.6  Phase 6 (epic) ∥ Phase 7                           (~2-4 mo)   stateless serving
v0.7  Phase 8                                            (~4-6 wks)  open control plane
```

## Top risks

| Risk | Mitigation |
|---|---|
| IVF recall/latency vs HNSW | Benchmark gate in Phase 6 acceptance; HNSW attached mode remains the default until parity data exists |
| tantivy-on-object-storage feasibility | Time-boxed spike with an explicit fallback (per-segment mini-indexes) |
| Id-block allocation contention | Blocks are large (10k) and leased rarely; CAS retry loop already proven on the manifest path |
| Lease correctness under clock skew | Long TTLs, idempotent CAS-guarded compaction — worst case is wasted work |
| S3 request costs in stateless mode | Centroid/block caching, section-level range reads, request-count metrics from day one (7.2) |
| JSON→v2 segment migration | Versioned decode already shipped in v0.3.0; compaction performs the migration organically |
