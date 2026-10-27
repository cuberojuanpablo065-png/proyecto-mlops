"""Esquemas de entrada y salida de la API (validación con Pydantic).

El cliente envía las características históricas de una serie
(estado x scope x categoría) y la API responde si el próximo año aumenta.
Las variables derivadas (delta_prev, pct_prev) las calcula el servidor.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

# Valores válidos (coinciden con el esquema de entrenamiento).
SCOPES = ["ipc", "sll", "women", "children", "scheduled_castes",
          "scheduled_tribes", "senior_citizens", "juveniles", "cyber"]


class RegistroEntrada(BaseModel):
    """Una fila a predecir."""
    state_code: str = Field(..., examples=["MH"], description="Código ISO del estado/UT")
    scope: str = Field(..., examples=["ipc"], description=f"Uno de: {SCOPES}")
    crime_category: str = Field(..., examples=["murder"])
    year: int = Field(..., ge=1990, le=2100, examples=[2022])
    lag1: float = Field(..., gt=0, description="Casos del año anterior (t-1)")
    lag2: float | None = Field(None, ge=0, description="Casos en t-2")
    lag3: float | None = Field(None, ge=0, description="Casos en t-3")


class LoteEntrada(BaseModel):
    registros: list[RegistroEntrada] = Field(..., min_length=1)


class Prediccion(BaseModel):
    prediccion: int = Field(..., description="1 = sube, 0 = no sube")
    etiqueta: str = Field(..., description="'sube' o 'no sube'")
    probabilidad: float = Field(..., description="Probabilidad de la clase 'sube'")


class RespuestaLote(BaseModel):
    predicciones: list[Prediccion]


class InfoModelo(BaseModel):
    modelo: str
    version_datos: str | None = None
    metrica_principal: str | None = None
    metricas: dict | None = None


class Salud(BaseModel):
    estado: str
    modelo_cargado: bool
