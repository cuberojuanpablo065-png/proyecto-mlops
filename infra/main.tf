# =====================================================================
# infra/main.tf - Infraestructura como código (Terraform) para DigitalOcean
# ---------------------------------------------------------------------
# Define los mismos recursos que scripts/deploy.sh, pero de forma
# declarativa y versionada. Ciclo:  terraform init -> plan -> apply -> destroy
#
# Las credenciales NO van aquí: se pasan por variables de entorno o un
# archivo terraform.tfvars (ignorado por Git). Ver variables.tf.
# =====================================================================

terraform {
  required_version = ">= 1.5"
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }
}

provider "digitalocean" {
  token             = var.do_token
  spaces_access_id  = var.spaces_access_id
  spaces_secret_key = var.spaces_secret_key
}

# 1) Bucket de objetos (Spaces) para el modelo y los datos.
resource "digitalocean_spaces_bucket" "modelos" {
  name   = var.spaces_bucket
  region = var.region
  acl    = "private"
}

# 2) Droplet (máquina virtual) con Docker preinstalado.
resource "digitalocean_droplet" "app" {
  name     = "crime-mlops"
  region   = var.region
  image    = "docker-20-04"
  size     = "s-1vcpu-1gb"
  ssh_keys = var.ssh_key_ids
}

# 3) Firewall: expone solo los puertos necesarios.
resource "digitalocean_firewall" "app" {
  name        = "crime-mlops-fw"
  droplet_ids = [digitalocean_droplet.app.id]

  dynamic "inbound_rule" {
    for_each = [22, 8000, 8001]
    content {
      protocol         = "tcp"
      port_range       = tostring(inbound_rule.value)
      source_addresses = ["0.0.0.0/0", "::/0"]
    }
  }

  outbound_rule {
    protocol              = "tcp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
  outbound_rule {
    protocol              = "udp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
}
