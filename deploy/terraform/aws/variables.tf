variable "bucket_name" {
  description = "Globally-unique S3 bucket name. This bucket IS the database."
  type        = string
}

variable "name_prefix" {
  description = "Prefix for IAM resources (policy, role, user)."
  type        = string
  default     = "compass"
}

variable "force_destroy" {
  description = "Allow `terraform destroy` to delete a NON-EMPTY bucket. Leave false: flipping it and destroying erases every collection."
  type        = bool
  default     = false
}

variable "kms_key_arn" {
  description = "Optional customer-managed KMS key for bucket encryption. null = AWS-managed aws/s3 key."
  type        = string
  default     = null
}

# ── Mode A: IRSA (EKS) ──────────────────────────────────────────────────────

variable "eks_oidc_provider_arn" {
  description = "ARN of the cluster's OIDC provider (aws_iam_openid_connect_provider). Set together with eks_oidc_provider_url to mint an IRSA role."
  type        = string
  default     = null
}

variable "eks_oidc_provider_url" {
  description = "URL of the cluster's OIDC provider (with or without https://)."
  type        = string
  default     = null
}

variable "k8s_namespace" {
  description = "Namespace of the Compass service account (IRSA trust condition)."
  type        = string
  default     = "compass"
}

variable "k8s_service_account" {
  description = "Name of the Compass service account (IRSA trust condition)."
  type        = string
  default     = "compass"
}

# ── Mode B: access key ──────────────────────────────────────────────────────

variable "create_access_key" {
  description = "Create an IAM user + long-lived access key instead of (or in addition to) IRSA. For non-EKS clusters and VMs."
  type        = bool
  default     = false
}
