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


def test_particion_temporal_sin_solape_de_anios(datos_sinteticos):
    """Feedback E1: la división es por tiempo (train años viejos, test recientes)."""
    train, test = dividir(datos_sinteticos)   # tipo=temporal por defecto
    anios_train = set(train["year"])
    anios_test = set(test["year"])
    assert anios_train.isdisjoint(anios_test), "No debe compartirse ningún año"
    assert max(anios_train) < min(anios_test), "El test debe ser de años posteriores al train"


def test_objetivo_sin_nulos(datos_sinteticos):
    """El objetivo no tiene valores nulos."""
    assert datos_sinteticos["target"].notna().all()
