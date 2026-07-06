# Compass on AWS — the storage half of a deployment.
#
# Provisions exactly what Compass needs from AWS and nothing else:
#   - an S3 bucket (the database: source of truth for every collection)
#   - a least-privilege IAM policy scoped to that bucket
#   - EITHER an IRSA role for EKS service accounts (recommended)
#     OR an IAM user + access key (for non-EKS clusters / VMs)
#
# The compute half lives in ../../kubernetes (any cluster) or plain
# containers — see deploy/README.md. Nothing here creates a cluster:
# bring your own EKS/K8s, or run the containers however you like.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

# ── The bucket: this IS the database ─────────────────────────────────────────

resource "aws_s3_bucket" "compass" {
  bucket = var.bucket_name

  # The bucket holds every collection; force_destroy=false means
  # `terraform destroy` refuses while data exists. Flip deliberately.
  force_destroy = var.force_destroy
}

resource "aws_s3_bucket_public_access_block" "compass" {
  bucket                  = aws_s3_bucket.compass.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "compass" {
  bucket = aws_s3_bucket.compass.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
      # null = the account's default aws/s3 KMS key; set to use your own CMK.
      kms_master_key_id = var.kms_key_arn
    }
    bucket_key_enabled = true
  }
}

# Compass manages object lifecycle itself (LSM compaction + deferred GC).
# Versioning would resurrect deleted WAL fragments and double storage cost —
# deliberately OFF. Point-in-time recovery = S3 replication if you need it.

resource "aws_s3_bucket_lifecycle_configuration" "compass" {
  bucket = aws_s3_bucket.compass.id
  rule {
    id     = "abort-incomplete-multipart"
    status = "Enabled"
    filter {}
    # Crashed multipart uploads (large segment writes) otherwise bill forever.
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# Deny any non-TLS access outright — defense in depth for a bucket that
# holds every collection.
resource "aws_s3_bucket_policy" "tls_only" {
  bucket = aws_s3_bucket.compass.id
  policy = data.aws_iam_policy_document.tls_only.json

  # The public-access block must land first or the policy PUT can race it.
  depends_on = [aws_s3_bucket_public_access_block.compass]
}

data "aws_iam_policy_document" "tls_only" {
  statement {
    sid     = "DenyInsecureTransport"
    effect  = "Deny"
    actions = ["s3:*"]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    resources = [aws_s3_bucket.compass.arn, "${aws_s3_bucket.compass.arn}/*"]
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

# ── Least-privilege access policy ────────────────────────────────────────────
# Compass needs: read/write/delete objects, list the bucket, multipart
# uploads. It does NOT need bucket administration, ACLs, or anything
# account-wide — and this policy grants none of that.

data "aws_iam_policy_document" "compass" {
  statement {
    sid       = "CompassObjects"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload", "s3:ListMultipartUploadParts"]
    resources = ["${aws_s3_bucket.compass.arn}/*"]
  }
  statement {
    sid       = "CompassList"
    actions   = ["s3:ListBucket", "s3:ListBucketMultipartUploads", "s3:GetBucketLocation"]
    resources = [aws_s3_bucket.compass.arn]
  }
}

resource "aws_iam_policy" "compass" {
  name   = "${var.name_prefix}-s3"
  policy = data.aws_iam_policy_document.compass.json
}

# ── Mode A (recommended): IRSA role for EKS ─────────────────────────────────
# Pods assume this role via their service account — no long-lived keys
# anywhere. Set eks_oidc_provider_arn + eks_oidc_provider_url to enable.

locals {
  irsa_enabled = var.eks_oidc_provider_arn != null
}

data "aws_iam_policy_document" "irsa_trust" {
  count = local.irsa_enabled ? 1 : 0
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [var.eks_oidc_provider_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${trimprefix(var.eks_oidc_provider_url, "https://")}:sub"
      values   = ["system:serviceaccount:${var.k8s_namespace}:${var.k8s_service_account}"]
    }
    condition {
      test     = "StringEquals"
      variable = "${trimprefix(var.eks_oidc_provider_url, "https://")}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "compass_irsa" {
  count              = local.irsa_enabled ? 1 : 0
  name               = "${var.name_prefix}-irsa"
  assume_role_policy = data.aws_iam_policy_document.irsa_trust[0].json
}

resource "aws_iam_role_policy_attachment" "compass_irsa" {
  count      = local.irsa_enabled ? 1 : 0
  role       = aws_iam_role.compass_irsa[0].name
  policy_arn = aws_iam_policy.compass.arn
}

# ── Mode B: IAM user + access key (non-EKS clusters, VMs, dev) ──────────────
# Long-lived credentials; rotate them, keep them in a K8s Secret, and prefer
# IRSA when you're on EKS. Off by default.

resource "aws_iam_user" "compass" {
  count = var.create_access_key ? 1 : 0
  name  = "${var.name_prefix}-svc"
}

resource "aws_iam_user_policy_attachment" "compass" {
  count      = var.create_access_key ? 1 : 0
  user       = aws_iam_user.compass[0].name
  policy_arn = aws_iam_policy.compass.arn
}

resource "aws_iam_access_key" "compass" {
  count = var.create_access_key ? 1 : 0
  user  = aws_iam_user.compass[0].name
}
