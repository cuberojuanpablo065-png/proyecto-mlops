"""Pruebas de partición (A.7 - categoría Partición)."""
from __future__ import annotations

from src.data import dividir


def test_particion_sin_fuga(datos_sinteticos):
    """No hay registros repetidos entre entrenamiento y prueba."""
    df = datos_sinteticos.copy()
    df["row_id"] = range(len(df))  # id único para rastrear filas
    train, test = dividir(df)

    ids_train = set(train["row_id"])
    ids_test = set(test["row_id"])

    assert ids_train.isdisjoint(ids_test), "Hay filas compartidas entre train y test"
    assert len(ids_train) + len(ids_test) == len(df), "Se perdieron o duplicaron filas"


def test_particion_es_estratificada(datos_sinteticos):
    """La proporción de la clase positiva se mantiene aprox. en train y test."""
    train, test = dividir(datos_sinteticos)
    p_train = train["target"].mean()
    p_test = test["target"].mean()
    assert abs(p_train - p_test) < 0.05, "La estratificación no se mantuvo"


def test_objetivo_sin_nulos(datos_sinteticos):
    """El objetivo no tiene valores nulos."""
    assert datos_sinteticos["target"].notna().all()
