"""A.3.3 - Preprocesamiento (dentro de un ColumnTransformer).

El preprocesamiento se define como un ColumnTransformer y se integra en el
Pipeline del modelo, por lo que se ajusta (fit) SOLO con los datos de
entrenamiento y evita la fuga de datos (data leakage).

Importante: la columna `value` (valor del año objetivo) NO es feature; se
excluye explícitamente para no filtrar el futuro.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import cargar_business

# --- Grupos de columnas ----------------------------------------------
COLUMNAS_NUMERICAS = ["year", "lag1", "lag2", "lag3", "delta_prev", "pct_prev"]
COLUMNAS_CATEGORICAS = ["state_code", "scope", "crime_category"]

# Columnas que NO son features (se excluyen de X).
# 'value' se excluye para evitar fuga de datos (es de donde sale el target).
COLUMNAS_NO_FEATURE = ["value", "crime_head", "state", "target"]


def separar_x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features (X) y objetivo (y), excluyendo columnas no-feature."""
    business = cargar_business()
    objetivo = business["columna_objetivo"]

    y = df[objetivo].astype(int)
    a_excluir = [c for c in COLUMNAS_NO_FEATURE if c in df.columns]
    X = df.drop(columns=a_excluir)
    return X, y


def _ohe() -> OneHotEncoder:
    """OneHotEncoder compatible con versiones nuevas y viejas de sklearn."""
    try:  # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)  # type: ignore


def construir_preprocesador() -> ColumnTransformer:
    """Construye el ColumnTransformer (sin ajustar).

    - Numéricas: imputación por mediana + estandarización.
    - Categóricas: imputación por valor más frecuente + one-hot.
    - El resto de columnas se descartan (remainder='drop').
    """
    pipe_num = Pipeline(
        steps=[
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]
    )
    pipe_cat = Pipeline(
        steps=[
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("onehot", _ohe()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", pipe_num, COLUMNAS_NUMERICAS),
            ("cat", pipe_cat, COLUMNAS_CATEGORICAS),
        ],
        remainder="drop",
    )
