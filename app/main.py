"""API del modelo (model-service) con FastAPI - Parte B del Entregable 2.

Endpoints:
  GET  /health         -> estado del servicio y del modelo
  GET  /model-info     -> versión y métricas del modelo
  POST /predict        -> predicción de un registro
  POST /predict/batch  -> predicción de una lista (con límite de tamaño)
  WS   /ws/predict     -> predicciones por WebSocket

El modelo se carga UNA sola vez al iniciar (lifespan), no por solicitud.
La documentación automática queda en /docs.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from . import storage
from .features import construir_dataframe
from .model_loader import modelo_global
from .schemas import (
    InfoModelo,
    LoteEntrada,
    Prediccion,
    RegistroEntrada,
    RespuestaLote,
    Salud,
)
from .settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Al iniciar: cargar el modelo y preparar la base de datos.
    modelo_global.cargar()
    storage.inicializar()
    yield
    # Al apagar: nada que limpiar.


app = FastAPI(
    title="API - Predicción de aumento de criminalidad",
    version="0.2.0",
    description="Expone el modelo del Entregable 1 como servicio web.",
    lifespan=lifespan,
)


def _predecir_df(df):
    """Ejecuta el modelo sobre un DataFrame y arma las predicciones."""
    if not modelo_global.cargado:
        # 503: el modelo no está disponible.
        raise HTTPException(status_code=503, detail="El modelo no está disponible")
    pred = modelo_global.modelo.predict(df)
    proba = modelo_global.modelo.predict_proba(df)[:, 1]
    salidas = []
    for p, pr in zip(pred, proba):
        salidas.append(
            Prediccion(
                prediccion=int(p),
                etiqueta="sube" if int(p) == 1 else "no sube",
                probabilidad=round(float(pr), 4),
            )
        )
    return salidas


@app.get("/health", response_model=Salud)
def health():
    """Indica si el servicio y el modelo están listos."""
    return Salud(estado="ok", modelo_cargado=modelo_global.cargado)


@app.get("/model-info", response_model=InfoModelo)
def model_info():
    """Versión del modelo y sus métricas principales."""
    if not modelo_global.cargado:
        raise HTTPException(status_code=503, detail="El modelo no está disponible")
    meta = modelo_global.meta
    return InfoModelo(
        modelo=meta.get("mejor_modelo", "desconocido"),
        version_datos=meta.get("version_datos"),
        metrica_principal=meta.get("metrica_principal"),
        metricas=meta.get("metricas_test") or meta.get("metricas"),
    )


@app.post("/predict", response_model=Prediccion)
def predict(registro: RegistroEntrada):
    """Predice si el próximo año aumentan los casos para un registro."""
    df = construir_dataframe([registro])
    salida = _predecir_df(df)[0]
    storage.guardar_prediccion(
        registro.model_dump(), salida.prediccion, salida.probabilidad,
        modelo_global.meta.get("version_datos"),
    )
    return salida


@app.post("/predict/batch", response_model=RespuestaLote)
def predict_batch(lote: LoteEntrada):
    """Predice una lista de registros (con límite de tamaño)."""
    if len(lote.registros) > settings.max_batch:
        raise HTTPException(
            status_code=422,
            detail=f"El lote excede el máximo de {settings.max_batch} registros",
        )
    df = construir_dataframe(lote.registros)
    salidas = _predecir_df(df)
    version = modelo_global.meta.get("version_datos")
    for reg, sal in zip(lote.registros, salidas):
        storage.guardar_prediccion(
            reg.model_dump(), sal.prediccion, sal.probabilidad, version
        )
    return RespuestaLote(predicciones=salidas)


@app.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket):
    """Predicciones por WebSocket: recibe un registro JSON y responde la predicción."""
    await websocket.accept()
    try:
        while True:
            datos = await websocket.receive_json()
            try:
                registro = RegistroEntrada(**datos)
            except Exception as exc:  # entrada inválida
                await websocket.send_json({"error": f"entrada inválida: {exc}"})
                continue
            df = construir_dataframe([registro])
            salida = _predecir_df(df)[0]
            storage.guardar_prediccion(
                registro.model_dump(), salida.prediccion, salida.probabilidad,
                modelo_global.meta.get("version_datos"),
            )
            await websocket.send_json(salida.model_dump())
    except WebSocketDisconnect:
        pass
