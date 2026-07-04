# Deployment topologies

Compass runs in three shapes. All of them are the same binary; the shape is
chosen entirely by environment variables.

## 1. Local single node (default)

Zero config. Local disk is the source of truth; everything is embedded.

```bash
./compass                     # or: docker run -p 4001:4001 -v ./data:/app/data compass
```

- No cloud credentials, no telemetry, no network calls.
- Tenant partitions (`partition_by`) work fully in this mode.
- Backup = copy `DATA_DIR`.

## 2. Cloud: serving nodes + stateless writers

Object storage is the source of truth; nodes are disposable.

```bash
# Serving node(s): full local indexes, fast reads, background convergence
COMPASS_STORAGE=s3://bucket AWS_ACCESS_KEY_ID=… AWS_SECRET_ACCESS_KEY=… ./compass

# Writer node(s): stateless append-only ingest, boots in milliseconds
COMPASS_ROLE=writer COMPASS_STORAGE=s3://bucket … ./compass
```

- Writers validate against the bucket config, mint chunk ids from a CAS
  allocator (never collide with anyone), append one WAL fragment, return a
  `seq`. Durable immediately; searchable on serving nodes within
  `COMPASS_REFRESH_INTERVAL` (default 5s).
- Read-your-writes: pass a write's `seq` as `min_seq` on search.
- A serving node that loses its disk rebuilds every collection from the
  bucket on boot. Kill -9 is a supported operation.
- Optional: `COMPASS_LAZY_ATTACH=true` + `COMPASS_MAX_ATTACHED=N` bound RAM
  to the hot collection set (LRU detach; re-attach on demand).

Upgrade caveat: do not run pre-v0.4 and v0.4 writers against one bucket;
old readers fail loudly on the v0.4 segment format rather than mis-reading.

## 3. Cloud: cold serving (serverless reads)

```bash
COMPASS_COLD_SERVE=true COMPASS_STORAGE=s3://bucket … ./compass
```

- Boots in <1s regardless of how much data the bucket holds; RAM starts at
  ~tens of MB.
- Semantic queries on collections the node has NEVER attached are answered
  from object-storage range reads (manifest → cached centroids → a few
  cluster reads → byte-range hydration). Freshness is read-your-writes by
  construction — every cold query reads the live manifest.
- `COMPASS_WARM_AFTER` (default 3) cold hits promote a background attach:
  cold → warm → hot automatically.
- Honest limits: cold is semantic-only (FTS errors until the namespace
  warms); recall on unstructured vector spaces needs a higher
  `COMPASS_COLD_NPROBE` (see [search-quality.md](search-quality.md));
  scoring options (recency/boosts/relations) are rejected cold rather than
  silently ignored.

## Multi-tenant collections (any topology)

```bash
curl -X POST :4001/collections -d '{"name":"app","embedding_dims":384,
  "config":{"partition_by":"tenant_id"}}'
```

Every chunk routes to an internal per-tenant partition by
`metadata.tenant_id`. Searches/deletes must filter on the partition field
(exact, or `{"in":[…]}` for ≤16 tenants). Ids are collection-unique;
partitions auto-create on first ingest (writer nodes included), hide from
listings, cascade-delete with the parent. Serving cost tracks the HOT tenant
set — 50 or 200 tenants boot identically.

## What the fleet does NOT give you (yet)

- Tenant-affinity routing between nodes: put a proxy in front and hash a
  tenant header to a node, or every node will warm every hot tenant.
- Per-tenant auth: `COMPASS_API_KEY` is one key for the whole node; tenant
  scoping is the caller's responsibility today.
- `/metrics` and `/health` are unauthenticated by design; firewall them if
  collection names/counts are sensitive.
