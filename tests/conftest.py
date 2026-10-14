"""Fixtures y datos sintéticos para las pruebas.

Las pruebas usan un panel sintético con el MISMO esquema que el panel real
(estado × scope × crime_head con rezagos) y una señal clara, para que
`pytest` corra rápido y sin depender de la descarga de datos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validate import SCOPES_VALIDOS

ESTADOS = ["MH", "DL", "KL", "TN", "UP", "KA", "GJ", "WB"]
CATEGORIAS = ["murder", "theft_burglary", "cyber", "fraud_economic", "narcotics", "kidnapping"]


def generar_datos_sinteticos(n: int = 1600, semilla: int = 42) -> pd.DataFrame:
    """Genera un panel con el esquema real y señal aprendible.

    target = (lag1 > lag2) con 5% de ruido -> delta_prev separa las clases,
    de modo que un modelo supera con holgura el umbral de la rúbrica.
    """
    rng = np.random.default_rng(semilla)
    lag2 = rng.uniform(1, 5000, n)
    lag1 = lag2 * rng.uniform(0.5, 1.8, n)          # relación con lag2 -> señal
    lag3 = lag2 * rng.uniform(0.5, 1.5, n)

    sube = lag1 > lag2
    ruido = rng.random(n) < 0.05                      # 5% de etiquetas ruidosas
    target = np.where(ruido, ~sube, sube).astype(int)
    # value: coherente con el target (sube => value > lag1)
    factor = np.where(target == 1, rng.uniform(1.01, 1.5, n), rng.uniform(0.5, 0.99, n))
    value = lag1 * factor

    df = pd.DataFrame(
        {
            "state_code": rng.choice(ESTADOS, n),
            "scope": rng.choice(SCOPES_VALIDOS, n),
            "crime_head": rng.choice([f"head_{i}" for i in range(12)], n),
            "crime_category": rng.choice(CATEGORIAS, n),
            "year": rng.integers(2005, 2023, n),
            "value": value,
            "lag1": lag1,
            "lag2": lag2,
            "lag3": lag3,
            "delta_prev": lag1 - lag2,
            "pct_prev": (lag1 - lag2) / (lag2 + 1.0),
            "target": target,
        }
    )
    return df


@pytest.fixture(scope="session")
def datos_sinteticos() -> pd.DataFrame:
    return generar_datos_sinteticos()
