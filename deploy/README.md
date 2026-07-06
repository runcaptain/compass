# Deploying Compass

Three paths, in increasing order of ceremony. All of them are the same
binary; topology background lives in [docs/deployment.md](../docs/deployment.md).

## 1. One machine (docker compose)

```bash
docker compose up          # local-first: data on the local volume, zero config
```

## 2. Kubernetes (any cluster) — the serverless topology

`deploy/kubernetes` is a kustomize tree:

```
base/                serving StatefulSet (PVC per replica) + writer Deployment
                     + optional cold Deployment + services + config
overlays/minio-dev   self-contained dev stack (in-cluster MinIO) — try it on
                     kind/k3s/minikube in one command
overlays/aws         production: real S3, IRSA or access-key auth
```

Try the whole topology on any cluster — no clone needed:

```bash
kubectl apply -k "https://github.com/runcaptain/compass//deploy/kubernetes/overlays/minio-dev?ref=main"
# or, from a checkout:
kubectl apply -k deploy/kubernetes/overlays/minio-dev
kubectl -n compass-dev get pods        # serving-0, writer, cold, minio
kubectl -n compass-dev port-forward svc/compass-read 4001:4001
curl localhost:4001/health
```

Production on AWS:

```bash
# 1. Storage half: bucket + least-privilege IAM
cd deploy/terraform/aws
terraform init && terraform apply -var bucket_name=my-compass-data
#    EKS?    add -var eks_oidc_provider_arn=… -var eks_oidc_provider_url=…
#    no EKS? add -var create_access_key=true and create the Secret it hints at

# 2. Compute half: point the overlay at your bucket + image, then
vi deploy/kubernetes/overlays/aws/kustomization.yaml   # bucket, region, image, IRSA arn
kubectl apply -k deploy/kubernetes/overlays/aws
```

What you get:

| workload | kind | storage | scale advice |
|---|---|---|---|
| `compass-serving` | StatefulSet | PVC per replica (warm restarts = seconds) | scale for read QPS; each replica converges independently |
| `compass-writer` | Deployment | none (stateless) | scale for ingest; safe to kill any time |
| `compass-cold` | Deployment (optional) | none | scale-to-many for bursty semantic reads on rarely-touched collections |

Traffic contract: **reads → `compass-read`, writes → `compass-write`**,
optional cold reads → `compass-cold`. Writers refuse reads by design;
serving nodes accept both but you keep clean scaling curves by splitting.

## 3. Terraform (AWS storage half)

`deploy/terraform/aws` provisions the bucket (private, encrypted,
lifecycle-managed) and least-privilege IAM — IRSA role for EKS, or an IAM
user + access key for anything else. It deliberately does NOT create a
cluster; bring any Kubernetes (or run the containers on VMs).

## Operational notes (read before production)

- **The bucket is the database.** PVCs are a warm cache — losing one costs a
  rebuild, never data. Bucket deletion is data loss; `force_destroy` stays
  false for a reason.
- **Consistency**: writes through writers are durable immediately and
  visible on serving nodes within `COMPASS_REFRESH_INTERVAL` (default 5s).
  Pass a write's `seq` as `min_seq` on search for read-your-writes; cold
  pods have read-your-writes by construction.
- **Version skew**: never run pre-v0.4 and v0.4+ writers against one bucket.
  Roll writers first, then serving nodes.
- **Auth**: the API is unauthenticated until you set `COMPASS_API_KEY`. The
  manifests already envFrom the `compass-aws` secret, so add the key there:
  `kubectl -n compass create secret generic compass-aws --from-literal=COMPASS_API_KEY=… [--from-literal=AWS_…]`.
  `/health` and `/metrics` stay unauthenticated by design; keep them
  cluster-internal (no NetworkPolicy ships here — add one if your cluster
  doesn't default-deny).
- **PVC lifecycle**: `kubectl delete -k …` removes the pods but RETAINS the
  StatefulSet PVCs (Kubernetes default) — re-applying reuses the warm cache.
  Delete PVCs explicitly to reclaim disk; that costs a rebuild, never data.
- **Single-replica updates**: with `replicas: 1`, a rolling update has a
  brief read-downtime window while the pod restarts (writes keep flowing via
  writers). Run ≥2 serving replicas if reads must never blip.
- **Sizing**: serving-node RAM tracks attached collections (bound it with
  `COMPASS_LAZY_ATTACH` + `COMPASS_MAX_ATTACHED`); PVC size tracks the same
  data as the bucket per attached collection. Cold pods run in ~tens of MiB.
- **Upgrades**: StatefulSet updates roll one pod at a time; readiness gating
  keeps traffic off a pod until its indexes serve. Writers roll freely.
