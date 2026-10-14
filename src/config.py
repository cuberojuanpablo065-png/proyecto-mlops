"""Carga de configuración del proyecto.

Lee `config/business.yaml` (problema de negocio, A.1) y `params.yaml`
(parámetros del pipeline y espacios de búsqueda) y los expone al resto
del código. Centralizar la configuración evita valores "hardcodeados" y
facilita la reproducibilidad.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Raíz del proyecto = carpeta padre de src/
RAIZ = Path(__file__).resolve().parents[1]

BUSINESS_YAML = RAIZ / "config" / "business.yaml"
PARAMS_YAML = RAIZ / "params.yaml"


def _leer_yaml(ruta: Path) -> dict[str, Any]:
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {ruta}")
    with open(ruta, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cargar_business() -> dict[str, Any]:
    """Configuración del problema de negocio (config/business.yaml)."""
    return _leer_yaml(BUSINESS_YAML)


def cargar_params() -> dict[str, Any]:
    """Parámetros del pipeline (params.yaml)."""
    return _leer_yaml(PARAMS_YAML)


# Rutas de salida usadas por varias etapas
RUTA_DATOS_PROCESADOS = RAIZ / "data" / "processed"
RUTA_MODELOS = RAIZ / "models"
RUTA_REPORTES = RAIZ / "reports"
RUTA_FIGURAS = RUTA_REPORTES / "figures"


def asegurar_directorios() -> None:
    """Crea las carpetas de salida si no existen."""
    for d in (RUTA_DATOS_PROCESADOS, RUTA_MODELOS, RUTA_REPORTES, RUTA_FIGURAS):
        d.mkdir(parents=True, exist_ok=True)
