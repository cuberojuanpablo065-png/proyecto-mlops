"""Prueba del servicio LLM: reintentos ante error 429 (Parte D, valor agregado).

Simula que la llamada a la API falla con un error reintentable (429) las
primeras veces y luego responde, y verifica que `_crear_stream` reintenta.
"""
from __future__ import annotations

from llm_service import main as llm


def test_reintenta_ante_429(monkeypatch):
    llamadas = {"n": 0}

    def fake_llamar(mensajes):
        llamadas["n"] += 1
        if llamadas["n"] < 3:
            raise llm.ErrorReintentable("429 Too Many Requests")
        return iter(())  # a la 3a, "responde" (flujo vacío)

    # Evita esperas largas entre reintentos durante la prueba.
    monkeypatch.setattr(llm, "_llamar_api", fake_llamar)
    monkeypatch.setattr(llm._crear_stream.retry, "wait", lambda *a, **k: 0)

    resultado = llm._crear_stream(["mensaje"])
    list(resultado)                       # consumir el flujo vacío
    assert llamadas["n"] == 3             # falló 2 veces y tuvo éxito a la 3a
