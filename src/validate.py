"""A.3.2 - Validación básica del panel con Pandera.

Verifica columnas esperadas, tipos y rangos del panel modelable. Si los
datos NO cumplen, el pipeline se detiene (lanza excepción). Evita entrenar
sobre datos corruptos.
"""
from __future__ import annotations

import pandas as pd

# Import compatible con versiones nuevas y viejas de Pandera.
try:  # pandera >= 0.22
    import pandera.pandas as pa
    from pandera.pandas import Check, Column, DataFrameSchema
except ImportError:  # pandera < 0.22
    import pandera as pa  # type: ignore
    from pandera import Check, Column, DataFrameSchema  # type: ignore

# Scopes y categorías válidas (ver DATA_DICTIONARY / ref_*.csv del dataset).
SCOPES_VALIDOS = [
    "ipc", "sll", "women", "children", "scheduled_castes",
    "scheduled_tribes", "senior_citizens", "juveniles", "cyber",
]

# Esquema esperado del panel modelable.
ESQUEMA = DataFrameSchema(
    {
        "state_code": Column(str, nullable=False),
        "scope": Column(str, Check.isin(SCOPES_VALIDOS)),
        "crime_head": Column(str, nullable=False),
        "crime_category": Column(str, nullable=False),
        "year": Column(int, Check.in_range(1950, 2100)),
        "value": Column(float, Check.ge(0)),
        "lag1": Column(float, Check.gt(0)),               # filtrado: > 0
        "lag2": Column(float, Check.ge(0), nullable=True),
        "lag3": Column(float, Check.ge(0), nullable=True),
        "delta_prev": Column(float, nullable=True),       # puede ser negativo
        "pct_prev": Column(float, nullable=True),
        "target": Column(int, Check.isin([0, 1])),
    },
    strict=True,       # rechaza columnas inesperadas
    coerce=True,       # convierte tipos compatibles (int64 -> int, etc.)
)

COLUMNAS_ESPERADAS = list(ESQUEMA.columns.keys())


def validar(df: pd.DataFrame) -> pd.DataFrame:
    """Valida el DataFrame contra el esquema. Lanza excepción si no cumple."""
    faltantes = set(COLUMNAS_ESPERADAS) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas esperadas: {sorted(faltantes)}")

    try:
        validado = ESQUEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:  # type: ignore[attr-defined]
        casos = getattr(exc, "failure_cases", exc)
        raise ValueError(
            f"La validación de datos falló. El pipeline se detiene.\n{casos}"
        ) from exc
    return validado


if __name__ == "__main__":
    from .config import RAIZ, cargar_params

    params = cargar_params()
    ruta = RAIZ / params["datos"]["archivo_crudo"]
    df = pd.read_csv(ruta)
    validar(df)
    print(f"[validate] OK: {len(df):,} filas válidas en {ruta}")
