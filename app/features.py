"""Construcción del vector de características a partir de la entrada de la API.

Reutiliza las columnas definidas en el preprocesamiento del Entregable 1
(src/preprocess.py), de modo que la API alimenta al modelo exactamente con
las mismas columnas con que fue entrenado.
"""
from __future__ import annotations

import pandas as pd

from src.preprocess import COLUMNAS_CATEGORICAS, COLUMNAS_NUMERICAS

COLUMNAS_MODELO = COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS


def construir_fila(registro) -> dict:
    """Convierte un RegistroEntrada en un diccionario con todas las columnas.

    Calcula delta_prev y pct_prev a partir de lag1 y lag2 (como en la ingesta).
    """
    lag1 = float(registro.lag1)
    lag2 = registro.lag2
    if lag2 is not None:
        delta_prev = lag1 - float(lag2)
        pct_prev = (lag1 - float(lag2)) / (float(lag2) + 1.0)
    else:
        delta_prev = None
        pct_prev = None

    return {
        "year": registro.year,
        "lag1": lag1,
        "lag2": lag2,
        "lag3": registro.lag3,
        "delta_prev": delta_prev,
        "pct_prev": pct_prev,
        "state_code": registro.state_code,
        "scope": registro.scope,
        "crime_category": registro.crime_category,
    }


def construir_dataframe(registros) -> pd.DataFrame:
    """Arma un DataFrame con las columnas que el modelo espera, en orden."""
    filas = [construir_fila(r) for r in registros]
    return pd.DataFrame(filas, columns=COLUMNAS_MODELO)
