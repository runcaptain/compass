# Search quality — measured recall & latency

Method: 20k docs × 64 dims, 200 held-out queries (perturbed documents),
ground truth = exact cosine top-10 (numpy). Two datasets: **structured**
(32 topic clusters + mild noise — the shape real text/image embeddings have)
and **adversarial uniform** (noise-dominated, nearly structureless — the
worst case for any ANN index). Latency measured over HTTP against Docker +
MinIO on a laptop; treat relative numbers, not absolutes.

## Warm path (attached: HNSW, ef_search=128)

| dataset | recall@10 | p50 | p95 |
|---|---|---|---|
| structured | 1.000 | 1.0ms | 1.1ms |
| structured, filtered (50% selectivity) | 1.000 | 1.0ms | 1.2ms |
| adversarial uniform | 0.895 | 1.2ms | 1.3ms |
| adversarial, filtered | 0.962 | 1.4ms | 1.7ms |

Full-text (BM25): exact-token top-1 50/50; topical precision@10 = 1.000.

## Cold path (serve-from-storage: IVF over object storage)

`COMPASS_COLD_NPROBE` clusters probed per segment (default 8):

| dataset | nprobe | recall@10 | p50 |
|---|---|---|---|
| structured | 4 | 0.947 | 13ms |
| structured | **8 (default)** | **1.000** | 13ms |
| structured | 16 | 1.000 | 14ms |
| adversarial uniform | 4 | 0.269 | 12ms |
| adversarial uniform | 8 | 0.409 | 13ms |
| adversarial uniform | 16 | 0.590 | 14ms |
| adversarial uniform | k (exhaustive) | 1.000 | 30ms |

## The honest contract

- On **clustered embedding spaces** — which is what real embedding models
  produce — cold recall reaches warm parity at the default nprobe, at
  ~10× warm latency (a handful of object-storage range reads).
- On **unstructured/uniform vector spaces**, IVF recall drops steeply (this
  is inherent to inverted-file indexes, not a Compass bug — the exhaustive
  row proves the pipeline is exact). If your vectors are random-ish
  (hashes, uncalibrated projections), raise `COMPASS_COLD_NPROBE`
  aggressively or rely on warm serving (`COMPASS_WARM_AFTER` promotes hot
  namespaces automatically).
- Cold filtered queries apply filters post-selection with an 8× deeper
  candidate pool; extremely selective filters (≪1% match rate) can
  under-return on the cold path — warm search has no such limit.

Reproduce: `scratchpad` eval scripts live in the PR discussion; the harness
is ~100 lines of numpy + HTTP and pins seeds.
