output "bucket_name" {
  value       = aws_s3_bucket.compass.bucket
  description = "The bucket that holds every collection."
}

output "compass_storage_url" {
  value       = "s3://${aws_s3_bucket.compass.bucket}"
  description = "Value for the COMPASS_STORAGE environment variable."
}

output "irsa_role_arn" {
  value       = local.irsa_enabled ? aws_iam_role.compass_irsa[0].arn : null
  description = "Annotate the Compass service account with this (eks.amazonaws.com/role-arn) when using IRSA."
}

output "access_key_id" {
  value       = var.create_access_key ? aws_iam_access_key.compass[0].id : null
  description = "AWS_ACCESS_KEY_ID for mode B. Store in a Kubernetes Secret."
}

output "secret_access_key" {
  value       = var.create_access_key ? aws_iam_access_key.compass[0].secret : null
  sensitive   = true
  description = "AWS_SECRET_ACCESS_KEY for mode B. `terraform output -raw secret_access_key`."
}

output "kubernetes_secret_hint" {
  value       = var.create_access_key ? "kubectl -n compass create secret generic compass-aws --from-literal=AWS_ACCESS_KEY_ID=$(terraform output -raw access_key_id) --from-literal=AWS_SECRET_ACCESS_KEY=$(terraform output -raw secret_access_key)" : "IRSA mode: no secret needed — annotate the service account with irsa_role_arn."
  description = "Next step after apply."
}
