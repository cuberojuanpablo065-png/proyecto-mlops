# Salidas: datos útiles tras "terraform apply".

output "droplet_ip" {
  description = "IP pública del Droplet (la API queda en http://<ip>:8000)"
  value       = digitalocean_droplet.app.ipv4_address
}

output "bucket_endpoint" {
  description = "Endpoint del bucket de Spaces"
  value       = digitalocean_spaces_bucket.modelos.bucket_domain_name
}
