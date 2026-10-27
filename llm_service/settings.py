"""Configuración del servicio LLM por variables de entorno."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Proveedor LLM (OpenAI) ---
    openai_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    max_tokens: int = 300                 # límite de longitud de la respuesta
    temperature: float = 0.3

    # Modo simulado: no llama a la API real (para pruebas sin llave).
    llm_mock: bool = False

    # --- URL del servicio del modelo (para obtener la predicción) ---
    model_service_url: str = "http://localhost:8000"

    # --- Archivo de precios ---
    precios_path: str = "config/precios_llm.yaml"

    # --- Reintentos ---
    max_reintentos: int = 4


settings = LLMSettings()
