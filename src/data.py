"""A.3.4 - Partición de datos (etapa "preparar" del pipeline DVC).

Lee los datos crudos, los valida y los divide en entrenamiento/prueba con
semilla fija y estratificación. Guarda los conjuntos en data/processed/.

La división se hace ANTES del preprocesamiento; el escalado/codificación
se ajusta dentro del Pipeline solo con train (evita fuga de datos).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RAIZ, RUTA_DATOS_PROCESADOS, asegurar_directorios, cargar_business, cargar_params
from .validate import validar


def cargar_crudo() -> pd.DataFrame:
    params = cargar_params()
    ruta = RAIZ / params["datos"]["archivo_crudo"]
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecute primero: python -m src.ingest "
            "(o coloque el CSV manualmente; ver README)."
        )
    return pd.read_csv(ruta)


def dividir(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide en train/test de forma estratificada y reproducible."""
    params = cargar_params()
    business = cargar_business()
    objetivo = business["columna_objetivo"]

    estratos = df[objetivo] if params["split"]["estratificar"] else None
    train, test = train_test_split(
        df,
        test_size=params["split"]["test_size"],
        random_state=params["semilla"],
        stratify=estratos,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def preparar() -> tuple[Path, Path]:
    """Pipeline de preparación: ingesta(validación) + split + guardado."""
    asegurar_directorios()
    df = cargar_crudo()
    df = validar(df)                     # detiene el pipeline si los datos no cumplen
    train, test = dividir(df)

    ruta_train = RUTA_DATOS_PROCESADOS / "train.csv"
    ruta_test = RUTA_DATOS_PROCESADOS / "test.csv"
    train.to_csv(ruta_train, index=False)
    test.to_csv(ruta_test, index=False)
    print(
        f"[data] train={len(train):,}  test={len(test):,}  "
        f"-> {RUTA_DATOS_PROCESADOS}"
    )
    return ruta_train, ruta_test


def cargar_procesados() -> tuple[pd.DataFrame, pd.DataFrame]:
    ruta_train = RUTA_DATOS_PROCESADOS / "train.csv"
    ruta_test = RUTA_DATOS_PROCESADOS / "test.csv"
    if not (ruta_train.exists() and ruta_test.exists()):
        raise FileNotFoundError(
            "No existen los datos procesados. Ejecute: python -m src.data"
        )
    return pd.read_csv(ruta_train), pd.read_csv(ruta_test)


if __name__ == "__main__":
    preparar()
    sys.exit(0)
