# mosaic-compass

Fork of [runcaptain/compass](https://github.com/runcaptain/compass) tailored for
Captain's Mosaic production deployment at `api.mosaic.runcaptain.com` via Porter.

## What this fork changes

| Concern | Upstream | This fork |
|---|---|---|
| Auth | None — all routes open | Bearer-token middleware on every route except `/health` |
| Healthcheck | `docker-compose.yml` calls `curl`, runtime image has no `curl` (always reports unhealthy) | `curl` installed in runtime stage of both Dockerfiles |
| Embedding models | Expected in `$DATA_DIR/models/bge-small/`, never auto-downloaded | `scripts/download-models.sh` Porter pre-deploy job fetches BGE-small from HuggingFace on each deploy (idempotent) |
| Deploy target | Generic Docker / docker-compose | Porter v2 (`porter.yaml` included) |
| Telemetry | On by default | `COMPASS_TELEMETRY=off` set in `porter.yaml` |

## Required environment variables

| Var | Required? | Notes |
|---|---|---|
| `COMPASS_API_KEY` | **Yes (prod)** | Shared secret. Clients must send `Authorization: Bearer <key>`. If unset, the server starts in unauthenticated dev mode and logs a loud warning. Configure as a Porter secret — do not commit. |
| `PORT` | No | Default `4001`. |
| `DATA_DIR` | No | Default `/app/data`. Must be a persistent volume in Porter. |
| `COMPASS_TELEMETRY` | No | Set to `off` (default in `porter.yaml`) to disable anonymous PostHog telemetry. |
| `COMPASS_BGE_REPO` | No | Override the HuggingFace repo used by the pre-deploy job. Default `BAAI/bge-small-en-v1.5`. |

## Auth

All routes require `Authorization: Bearer <COMPASS_API_KEY>` **except** `GET /health`, which is open so Porter and nginx probes can reach it without a key.

```bash
# Allowed
curl https://api.mosaic.runcaptain.com/health

# Requires Bearer
curl -H "Authorization: Bearer $COMPASS_API_KEY" \
  https://api.mosaic.runcaptain.com/collections
```

Returns `401 Unauthorized` on missing or mismatched key. The key check is constant-time (see `ct_eq` in `crates/compass/src/api/mod.rs`).

## Deploying with Porter

1. Create a Porter application from this repo.
2. Set the `COMPASS_API_KEY` secret in the Porter UI (generate with `openssl rand -hex 32`).
3. Attach a persistent volume (EBS) mounted at `/app/data`.
4. Point the `api.mosaic.runcaptain.com` DNS record at the Porter ingress.
5. Deploy. Each deploy will:
   - Run the pre-deploy job (`scripts/download-models.sh`) which idempotently fetches BGE-small into `$DATA_DIR/models/bge-small/`.
   - Start the `api` web service on port 4001.
   - Mark healthy once `GET /health` returns 200.

## Endpoint surface

Unchanged from upstream. See `README.md` for full API documentation. All paths
require the Bearer header except `/health`.

```
GET    /health                                              (public)
POST   /collections
GET    /collections
GET    /collections/{name}
DELETE /collections/{name}
POST   /collections/{name}/vector-spaces
GET    /collections/{name}/vector-spaces
DELETE /collections/{name}/vector-spaces/{space}
POST   /collections/{name}/vector-spaces/{space}/rebuild
GET    /collections/{name}/vector-spaces/{space}/status
PUT    /collections/{name}/default-vector-space
POST   /collections/{name}/ingest                           (64 MB body limit)
POST   /collections/{name}/search
GET    /collections/{name}/facets
```
