"""Servicio LLM: explica en lenguaje natural una predicción del modelo.

Flujo (ver diagrama de arquitectura en el README):
  cliente -> llm-service -> model-service (/predict) -> LLM (OpenAI) -> streaming

Requisitos cubiertos (Parte D):
  - Respuesta en streaming (el texto se envía a medida que se genera).
  - Conteo de tokens de entrada y salida.
  - Cálculo del costo con precios de un archivo de configuración.
  - Límite de tasa (429) con reintentos y espera exponencial (tenacity).
  - Límite de tokens (max_tokens).
  - Registro (log) en formato JSON por solicitud.
"""
from __future__ import annotations

import json
import logging
import sys
import time

import httpx
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .settings import settings

# --- Log en formato JSON a stdout ---
logger = logging.getLogger("llm-service")
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")


def _cargar_precios() -> dict:
    try:
        with open(settings.precios_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {"precio_input_por_millon": 0.0, "precio_output_por_millon": 0.0}


PRECIOS = _cargar_precios()
app = FastAPI(title="Servicio LLM - explicación de predicciones", version="0.2.0")


class ExplicacionEntrada(BaseModel):
    state_code: str = Field(..., examples=["MH"])
    scope: str = Field(..., examples=["ipc"])
    crime_category: str = Field(..., examples=["murder"])
    year: int = Field(..., examples=[2022])
    lag1: float = Field(..., gt=0)
    lag2: float | None = None
    lag3: float | None = None


def _costo(tokens_in: int, tokens_out: int) -> float:
    pi = PRECIOS.get("precio_input_por_millon", 0.0)
    po = PRECIOS.get("precio_output_por_millon", 0.0)
    return round(tokens_in / 1e6 * pi + tokens_out / 1e6 * po, 6)


def _obtener_prediccion(entrada: ExplicacionEntrada) -> dict:
    """Llama al model-service para obtener la predicción."""
    url = f"{settings.model_service_url}/predict"
    try:
        r = httpx.post(url, json=entrada.model_dump(), timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"No se pudo consultar el modelo: {exc}"
        )


def _prompt(entrada: ExplicacionEntrada, pred: dict) -> list[dict]:
    sistema = (
        "Eres un analista que explica, en español claro y breve (2-3 frases), "
        "predicciones de un modelo sobre si los casos de un tipo de crimen en "
        "un estado de la India aumentarán el próximo año. No inventes cifras."
    )
    usuario = (
        f"Estado: {entrada.state_code}, categoría: {entrada.crime_category}, "
        f"ámbito: {entrada.scope}, año: {entrada.year}, casos t-1: {entrada.lag1}, "
        f"casos t-2: {entrada.lag2}. El modelo predice '{pred['etiqueta']}' "
        f"con probabilidad {pred['probabilidad']}. Explica esta predicción."
    )
    return [{"role": "system", "content": sistema},
            {"role": "user", "content": usuario}]


# Reintentos con espera exponencial ante errores de límite de tasa / red.
class ErrorReintentable(Exception):
    pass


def _llamar_api(mensajes: list[dict]):
    """Hace la llamada real al LLM y devuelve el flujo (stream) de eventos."""
    from openai import OpenAI

    cliente = OpenAI(api_key=settings.openai_api_key)
    return cliente.chat.completions.create(
        model=settings.llm_model,
        messages=mensajes,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        stream=True,
        stream_options={"include_usage": True},
    )


@retry(
    retry=retry_if_exception_type(ErrorReintentable),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(settings.max_reintentos),
    reraise=True,
)
def _crear_stream(mensajes: list[dict]):
    """Crea el flujo del LLM con reintentos ante 429/errores de red.

    El reintento envuelve la LLAMADA a la API (donde se lanza el 429), no la
    iteración de los fragmentos, que es como tenacity funciona correctamente.
    """
    try:
        from openai import APIConnectionError, APIStatusError, RateLimitError
    except Exception:  # pragma: no cover - permite importar sin el SDK en pruebas
        return _llamar_api(mensajes)

    try:
        return _llamar_api(mensajes)
    except (RateLimitError, APIConnectionError) as exc:  # 429 / red -> reintentar
        raise ErrorReintentable(str(exc)) from exc
    except APIStatusError as exc:
        if exc.status_code == 429:
            raise ErrorReintentable(str(exc)) from exc
        raise


def _generar(entrada: ExplicacionEntrada, pred: dict):
    """Generador que produce el texto en streaming y al final registra el log."""
    t0 = time.perf_counter()
    mensajes = _prompt(entrada, pred)
    tokens_in = tokens_out = 0
    texto_total = []

    if settings.llm_mock or not settings.openai_api_key:
        # Modo simulado: respuesta fija por fragmentos (sin llamar a la API).
        demo = (
            f"El modelo predice que los casos de {entrada.crime_category} en "
            f"{entrada.state_code} {'aumentarán' if pred['prediccion']==1 else 'no aumentarán'} "
            f"(probabilidad {pred['probabilidad']}). "
            "Esto se basa en la tendencia de los años previos."
        )
        for palabra in demo.split():
            texto_total.append(palabra + " ")
            yield palabra + " "
        tokens_in = len(" ".join(m["content"] for m in mensajes).split())
        tokens_out = len(demo.split())
    else:
        flujo = _crear_stream(mensajes)
        for evento in flujo:
            if evento.choices and evento.choices[0].delta.content:
                frag = evento.choices[0].delta.content
                texto_total.append(frag)
                yield frag
            if getattr(evento, "usage", None):
                tokens_in = evento.usage.prompt_tokens
                tokens_out = evento.usage.completion_tokens

    latencia = round(time.perf_counter() - t0, 3)
    log = {
        "evento": "explicacion",
        "modelo_llm": settings.llm_model,
        "prediccion": pred["etiqueta"],
        "tokens_entrada": tokens_in,
        "tokens_salida": tokens_out,
        "costo_usd": _costo(tokens_in, tokens_out),
        "latencia_s": latencia,
    }
    logger.info(json.dumps(log, ensure_ascii=False))


@app.get("/health")
def health():
    return {"estado": "ok", "modo": "mock" if (settings.llm_mock or not settings.openai_api_key) else "openai"}


@app.post("/explain")
def explain(entrada: ExplicacionEntrada):
    """Explica en lenguaje natural la predicción del modelo (respuesta en streaming)."""
    pred = _obtener_prediccion(entrada)
    return StreamingResponse(_generar(entrada, pred), media_type="text/plain")
