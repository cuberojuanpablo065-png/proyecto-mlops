"""Prototipo de servidor MCP (Model Context Protocol) - Parte E.

Expone una herramienta sencilla, `obtener_info_modelo`, que consulta la
información del modelo a través de la API (model-service). Se prueba con
el MCP Inspector:

    npx @modelcontextprotocol/inspector python -m mcp_server.server

Requiere el SDK de MCP para Python:  pip install "mcp[cli]"

Herramientas planeadas para el Entregable 4 (dejadas anotadas):
  - predecir(state_code, scope, crime_category, year, lag1, ...):
        llamaría a POST /predict del model-service.
  - consultar_documentos(pregunta):
        haría RAG sobre los documentos/noticias del dataset (news.parquet,
        documents.csv) para responder preguntas en lenguaje natural.
"""
from __future__ import annotations

import os

import httpx
from mcp.server.fastmcp import FastMCP

MODEL_SERVICE_URL = os.environ.get("MODEL_SERVICE_URL", "http://localhost:8000")

mcp = FastMCP("crime-mlops")


@mcp.tool()
def obtener_info_modelo() -> dict:
    """Devuelve la versión del modelo y sus métricas principales.

    Consulta el endpoint /model-info del servicio del modelo.
    """
    r = httpx.get(f"{MODEL_SERVICE_URL}/model-info", timeout=10)
    r.raise_for_status()
    return r.json()


# --- Herramientas futuras (Entregable 4), dejadas como esquema ---
#
# @mcp.tool()
# def predecir(state_code: str, scope: str, crime_category: str,
#              year: int, lag1: float, lag2: float | None = None) -> dict:
#     """Predice si aumentan los casos (llamaría a POST /predict)."""
#     ...
#
# @mcp.tool()
# def consultar_documentos(pregunta: str) -> str:
#     """Responde con RAG sobre noticias/documentos del dataset."""
#     ...


if __name__ == "__main__":
    # Transporte stdio (el que usa el MCP Inspector por defecto).
    mcp.run()
