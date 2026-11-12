"""Carga del modelo UNA sola vez (al iniciar el servicio, no por solicitud).

Si MODEL_S3_URI está definido, descarga el modelo del bucket (DigitalOcean
Spaces / MinIO / S3) a la ruta local antes de cargarlo. Así el modelo vive
fuera de la imagen del contenedor y se puede actualizar sin reconstruirla
(A.2 del enunciado).
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import joblib

from .settings import settings


def _descargar_archivo(url: str, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    # La URL la configura el operador (variable de entorno de confianza).
    with urllib.request.urlopen(url, timeout=120) as resp, open(destino, "wb") as f:  # noqa: S310  # nosec B310
        f.write(resp.read())


def _verificar_hash(ruta: Path) -> None:
    """E.3 - Verifica el SHA-256 del modelo. Si no coincide, NO se usa.

    El hash esperado viene de MODEL_SHA256 (variable de entorno) o del
    metadato del modelo. Si no hay hash esperado, solo informa el calculado.
    """
    calculado = hashlib.sha256(ruta.read_bytes()).hexdigest()
    esperado = settings.model_sha256
    if esperado:
        if calculado.lower() != esperado.lower():
            raise RuntimeError(
                "Integridad del modelo FALLÓ: el SHA-256 no coincide. "
                f"esperado={esperado} calculado={calculado}. No se carga el modelo."
            )
        print(f"[model] Integridad OK (SHA-256 verificado): {calculado}")
    else:
        print(f"[model] SHA-256 del modelo: {calculado} (sin hash esperado para comparar)")


def _descargar_de_url() -> bool:
    """Descarga el modelo desde una URL pública (MODEL_URL) si está definida.

    Útil para desplegar en servicios como Render usando el modelo publicado
    como asset de un GitHub Release (almacenamiento de objetos gratuito).
    Devuelve True si descargó.
    """
    url = settings.model_url
    if not url:
        return False
    destino = Path(settings.model_path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"[model] Descargando modelo de {url} -> {destino}")
    # La URL la configura el operador (variable de entorno de confianza).
    with urllib.request.urlopen(url, timeout=120) as resp, open(destino, "wb") as f:  # noqa: S310  # nosec B310
        f.write(resp.read())
    return True


def _descargar_de_bucket() -> None:
    """Descarga el modelo del bucket S3-compatible si está configurado."""
    uri = settings.model_s3_uri
    if not uri:
        return  # se usa el archivo local

    try:
        import boto3  # import perezoso: solo si se usa bucket
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "MODEL_S3_URI está definido pero boto3 no está instalado."
        ) from exc

    assert uri.startswith("s3://"), "MODEL_S3_URI debe empezar con s3://"
    sin_prefijo = uri[len("s3://"):]
    bucket, _, clave = sin_prefijo.partition("/")

    destino = Path(settings.model_path)
    destino.parent.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region,
    )
    print(f"[model] Descargando modelo de {uri} -> {destino}")
    s3.download_file(bucket, clave, str(destino))


class ModeloEnMemoria:
    """Contenedor del modelo y sus metadatos, cargados una sola vez."""

    def __init__(self) -> None:
        self.modelo = None
        self.meta: dict = {}

    def cargar(self) -> None:
        # Prioridad: URL pública -> bucket S3 -> archivo local.
        if not _descargar_de_url():
            _descargar_de_bucket()
        ruta = Path(settings.model_path)
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en {ruta}. Genere el modelo "
                "(Entregable 1) o configure MODEL_S3_URI."
            )

        # Si hay URL de metadatos, descargarla (para que /model-info muestre
        # las métricas también en la nube).
        if settings.model_meta_url:
            try:
                _descargar_archivo(settings.model_meta_url, Path(settings.model_meta_path))
            except Exception as exc:  # pragma: no cover
                print(f"[model] No se pudo descargar el meta: {exc}")

        # E.3 - verificar integridad ANTES de cargar en memoria.
        _verificar_hash(ruta)
        self.modelo = joblib.load(ruta)

        meta = Path(settings.model_meta_path)
        if meta.exists():
            self.meta = json.loads(meta.read_text(encoding="utf-8"))
        print(f"[model] Modelo cargado desde {ruta}")

    @property
    def cargado(self) -> bool:
        return self.modelo is not None


# Instancia única compartida por toda la app.
modelo_global = ModeloEnMemoria()
