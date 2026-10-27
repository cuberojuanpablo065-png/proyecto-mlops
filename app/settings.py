"""Configuración del servicio por variables de entorno (sin credenciales en código).

Usa pydantic-settings: lee variables de entorno (o un archivo .env) y las
valida. Así el mismo contenedor funciona en local y en la nube cambiando
solo las variables, nunca el código.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Modelo ---
    # Ruta local del modelo serializado. Si MODEL_S3_URI está definido, el
    # modelo se descarga del bucket a esta ruta al iniciar (ver model_loader).
    model_path: str = "models/model.joblib"
    model_meta_path: str = "models/model_meta.json"

    # --- Descarga por URL pública (opcional; p.ej. asset de GitHub Release) ---
    model_url: str | None = None

    # --- Bucket de objetos (opcional; para descargar el modelo de la nube) ---
    # Ejemplo: s3://mi-bucket/modelos/model.joblib
    model_s3_uri: str | None = None
    s3_endpoint_url: str | None = None          # MinIO o DigitalOcean Spaces
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str = "us-east-1"

    # --- Base de datos de predicciones (SQLite por defecto) ---
    db_path: str = "data/predicciones.db"

    # --- Límite del batch ---
    max_batch: int = 500


settings = Settings()
