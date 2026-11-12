"""API del modelo (model-service) con FastAPI.

Endpoints:
  GET  /health         -> estado del servicio y del modelo
  GET  /model-info     -> versión y métricas del modelo
  GET  /metrics        -> métricas operativas (C.1)
  POST /predict        -> predicción de un registro  (requiere API key si está configurada)
  POST /predict/batch  -> predicción de una lista (con límite de tamaño)
  WS   /ws/predict     -> predicciones por WebSocket

Seguridad (E.2): validación con Pydantic (extra=forbid), autenticación por
API key y limitación de tasa por IP. El modelo se verifica por SHA-256 al
cargar (E.3) y se carga UNA sola vez al iniciar (lifespan).
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect

from . import storage
from .features import construir_dataframe
from .metrics import contadores
from .model_loader import modelo_global
from .schemas import (
    InfoModelo,
    LoteEntrada,
    Metricas,
    Prediccion,
    RegistroEntrada,
    RespuestaLote,
    Salud,
)
from .settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Al iniciar: cargar el modelo (con verificación de integridad) y la BD.
    modelo_global.cargar()
    storage.inicializar()
    yield


app = FastAPI(
    title="API - Predicción de aumento de criminalidad",
    version="0.3.0",
    description="Expone el modelo del Entregable 1 como servicio web.",
    lifespan=lifespan,
)


# --- C.1: middleware de métricas (solicitudes, errores, latencia) ----------
@app.middleware("http")
async def medir(request: Request, call_next):
    t0 = time.perf_counter()
    try:
        respuesta = await call_next(request)
    except Exception:
        contadores.registrar_solicitud((time.perf_counter() - t0) * 1000, es_error=True)
        raise
    latencia_ms = (time.perf_counter() - t0) * 1000
    contadores.registrar_solicitud(latencia_ms, es_error=respuesta.status_code >= 500)
    return respuesta


# --- E.2: limitación de tasa por IP (ventana deslizante en memoria) ---------
_historial: dict[str, deque] = defaultdict(deque)


def _limite_por_minuto() -> int:
    try:
        n, _ = settings.rate_limit.split("/")
        return int(n)
    except Exception:
        return 60


def limitar_tasa(request: Request) -> None:
    limite = _limite_por_minuto()
    ip = request.client.host if request.client else "desconocido"
    ahora = time.time()
    cola = _historial[ip]
    while cola and ahora - cola[0] > 60:
        cola.popleft()
    if len(cola) >= limite:
        raise HTTPException(status_code=429, detail="Demasiadas solicitudes (límite de tasa)")
    cola.append(ahora)


# --- E.2: autenticación por API key ----------------------------------------
def verificar_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Si API_KEY está configurada, exige la cabecera X-API-Key correcta."""
    if settings.api_key:
        if not x_api_key:
            raise HTTPException(status_code=401, detail="Falta la cabecera X-API-Key")
        if x_api_key != settings.api_key:
            raise HTTPException(status_code=403, detail="API key inválida")


# Dependencias combinadas para los endpoints protegidos.
PROTEGIDO = [Depends(limitar_tasa), Depends(verificar_api_key)]


def _predecir_df(df):
    """Ejecuta el modelo sobre un DataFrame y arma las predicciones."""
    if not modelo_global.cargado:
        raise HTTPException(status_code=503, detail="El modelo no está disponible")
    pred = modelo_global.modelo.predict(df)
    proba = modelo_global.modelo.predict_proba(df)[:, 1]
    salidas = []
    for p, pr in zip(pred, proba):
        etiqueta = "sube" if int(p) == 1 else "no sube"
        contadores.registrar_prediccion(etiqueta)   # C.1: distribución
        salidas.append(
            Prediccion(prediccion=int(p), etiqueta=etiqueta, probabilidad=round(float(pr), 4))
        )
    return salidas


@app.get("/health", response_model=Salud)
def health():
    return Salud(estado="ok", modelo_cargado=modelo_global.cargado)


@app.get("/model-info", response_model=InfoModelo)
def model_info():
    if not modelo_global.cargado:
        raise HTTPException(status_code=503, detail="El modelo no está disponible")
    meta = modelo_global.meta
    return InfoModelo(
        modelo=meta.get("mejor_modelo", "desconocido"),
        version_datos=meta.get("version_datos"),
        metrica_principal=meta.get("metrica_principal"),
        metricas=meta.get("metricas_test") or meta.get("metricas"),
    )


@app.get("/metrics", response_model=Metricas)
def metrics():
    """C.1 - Métricas operativas: solicitudes, errores, latencia y distribución."""
    return Metricas(**contadores.resumen())


@app.post("/predict", response_model=Prediccion, dependencies=PROTEGIDO)
def predict(registro: RegistroEntrada):
    df = construir_dataframe([registro])
    salida = _predecir_df(df)[0]
    storage.guardar_prediccion(
        registro.model_dump(), salida.prediccion, salida.probabilidad,
        modelo_global.meta.get("version_datos"),
    )
    return salida


@app.post("/predict/batch", response_model=RespuestaLote, dependencies=PROTEGIDO)
def predict_batch(lote: LoteEntrada):
    df = construir_dataframe(lote.registros)
    salidas = _predecir_df(df)
    version = modelo_global.meta.get("version_datos")
    for reg, sal in zip(lote.registros, salidas):
        storage.guardar_prediccion(reg.model_dump(), sal.prediccion, sal.probabilidad, version)
    return RespuestaLote(predicciones=salidas)


@app.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            datos = await websocket.receive_json()
            try:
                registro = RegistroEntrada(**datos)
            except Exception as exc:
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
