# Variables de entrada. Los valores reales van en terraform.tfvars
# (ignorado por Git) o en variables de entorno TF_VAR_*.

variable "do_token" {
  description = "Token de API de DigitalOcean"
  type        = string
  sensitive   = true
}

variable "spaces_access_id" {
  description = "Access key de Spaces (S3)"
  type        = string
  sensitive   = true
}

variable "spaces_secret_key" {
  description = "Secret key de Spaces (S3)"
  type        = string
  sensitive   = true
}

variable "region" {
  description = "Región de DigitalOcean"
  type        = string
  default     = "nyc3"
}

variable "spaces_bucket" {
  description = "Nombre único del bucket de Spaces"
  type        = string
}

variable "ssh_key_ids" {
  description = "IDs de las llaves SSH registradas en DigitalOcean"
  type        = list(string)
}
