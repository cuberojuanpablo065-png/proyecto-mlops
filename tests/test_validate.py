"""Pruebas de validación de datos (A.7 - categoría Datos)."""
from __future__ import annotations

import pytest

from src.validate import validar


def test_validacion_acepta_datos_correctos(datos_sinteticos):
    """Datos con el esquema correcto pasan la validación."""
    validado = validar(datos_sinteticos)
    assert len(validado) == len(datos_sinteticos)


def test_validacion_rechaza_columnas_faltantes(datos_sinteticos):
    """Si falta una columna esperada, la validación falla (pipeline se detiene)."""
    df_malo = datos_sinteticos.drop(columns=["lag1"])
    with pytest.raises(ValueError):
        validar(df_malo)


def test_validacion_rechaza_valores_fuera_de_rango(datos_sinteticos):
    """Valores inválidos (target fuera de {0,1}) son rechazados."""
    df_malo = datos_sinteticos.copy()
    df_malo.loc[0, "target"] = 7  # target solo admite 0 o 1
    with pytest.raises(ValueError):
        validar(df_malo)
