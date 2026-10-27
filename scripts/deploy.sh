#!/usr/bin/env bash
# =====================================================================
# scripts/deploy.sh - Despliegue en DigitalOcean con la CLI (doctl + aws).
# ---------------------------------------------------------------------
# Crea/actualiza los recursos mínimos del Entregable 2:
#   1) Un bucket de objetos (Spaces, S3-compatible) para el modelo.
#   2) Un Droplet (máquina virtual) con Docker donde corren los contenedores.
#   3) Un firewall que expone solo los puertos necesarios (22, 8000, 8001).
#
# Requisitos (ver README):
#   - doctl instalado y autenticado:  doctl auth init
#   - aws CLI instalado (para Spaces): aws configure  (con las llaves de Spaces)
#   - Variables de entorno (NO credenciales en el código):
#       export DO_REGION=nyc3
#       export SPACES_BUCKET=crime-mlops-tuusuario
#       export SSH_KEY_ID=<id de tu llave SSH en DigitalOcean>
#
# Es idempotente: si un recurso ya existe, no lo vuelve a crear.
# =====================================================================
set -euo pipefail

DO_REGION="${DO_REGION:-nyc3}"
SPACES_BUCKET="${SPACES_BUCKET:?Defina SPACES_BUCKET}"
DROPLET_NAME="${DROPLET_NAME:-crime-mlops}"
FIREWALL_NAME="${FIREWALL_NAME:-crime-mlops-fw}"
SPACES_ENDPOINT="https://${DO_REGION}.digitaloceanspaces.com"

echo "==> 1) Bucket de objetos (Spaces): ${SPACES_BUCKET}"
if aws --endpoint-url "${SPACES_ENDPOINT}" s3 ls "s3://${SPACES_BUCKET}" >/dev/null 2>&1; then
  echo "    ya existe."
else
  aws --endpoint-url "${SPACES_ENDPOINT}" s3 mb "s3://${SPACES_BUCKET}"
fi

echo "==> 2) Subir el modelo al bucket"
aws --endpoint-url "${SPACES_ENDPOINT}" s3 cp models/model.joblib \
    "s3://${SPACES_BUCKET}/modelos/model.joblib"

echo "==> 3) Droplet con Docker: ${DROPLET_NAME}"
if doctl compute droplet list --format Name --no-header | grep -qx "${DROPLET_NAME}"; then
  echo "    ya existe."
else
  doctl compute droplet create "${DROPLET_NAME}" \
    --region "${DO_REGION}" \
    --image docker-20-04 \
    --size s-1vcpu-1gb \
    --ssh-keys "${SSH_KEY_ID:?Defina SSH_KEY_ID}" \
    --wait
fi

IP=$(doctl compute droplet get "${DROPLET_NAME}" --format PublicIPv4 --no-header)
echo "    IP pública: ${IP}"

echo "==> 4) Firewall (abre solo 22, 8000, 8001): ${FIREWALL_NAME}"
if ! doctl compute firewall list --format Name --no-header | grep -qx "${FIREWALL_NAME}"; then
  doctl compute firewall create \
    --name "${FIREWALL_NAME}" \
    --inbound-rules "protocol:tcp,ports:22,address:0.0.0.0/0 protocol:tcp,ports:8000,address:0.0.0.0/0 protocol:tcp,ports:8001,address:0.0.0.0/0" \
    --outbound-rules "protocol:tcp,ports:all,address:0.0.0.0/0 protocol:udp,ports:all,address:0.0.0.0/0"
fi
FW_ID=$(doctl compute firewall list --format ID,Name --no-header | grep "${FIREWALL_NAME}" | awk '{print $1}')
DROPLET_ID=$(doctl compute droplet get "${DROPLET_NAME}" --format ID --no-header)
doctl compute firewall add-droplets "${FW_ID}" --droplet-ids "${DROPLET_ID}" || true

cat <<FIN

==> Listo. Recursos creados:
    - Bucket Spaces : s3://${SPACES_BUCKET}  (modelo en /modelos/model.joblib)
    - Droplet       : ${DROPLET_NAME}  (IP ${IP})
    - Firewall      : ${FIREWALL_NAME}  (22, 8000, 8001)

Siguiente paso (desplegar los contenedores en el Droplet):
    ssh root@${IP}
    git clone <tu-repo> && cd proyecto-mlops
    cp .env.example .env   # y completa OPENAI_API_KEY y las llaves de Spaces
    docker compose up -d --build

La API quedará en:  http://${IP}:8000/health
FIN
