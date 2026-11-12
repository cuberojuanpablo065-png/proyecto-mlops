"""Pruebas de seguridad de la API (E.2) y métricas (C.1).

Verifican: rechazo de campos desconocidos (extra=forbid), límite de tamaño
de lote, autenticación por API key (sin credenciales / credenciales malas) y
el endpoint /metrics.
"""
from __future__ import annotations

import joblib
import pytest
from fastapi.testclient import TestClient

from src.modelos import construir_pipeline_random_forest
from src.preprocess import separar_x_y
from tests.conftest import generar_datos_sinteticos

OK = {"state_code": "MH", "scope": "ipc", "crime_category": "murder",
      "year": 2022, "lag1": 300, "lag2": 250}


def _crear_app(tmp_path, api_key=None):
    df = generar_datos_sinteticos(n=1000, semilla=5)
    X, y = separar_x_y(df)
    modelo = construir_pipeline_random_forest()
    modelo.fit(X, y)
    ruta = tmp_path / "model.joblib"
    joblib.dump(modelo, ruta)

    import app.main as m
    from app import model_loader, storage
    model_loader.settings.model_path = str(ruta)
    model_loader.settings.model_meta_path = str(tmp_path / "no.json")
    model_loader.settings.model_sha256 = None
    storage.settings.db_path = str(tmp_path / "p.db")
    m.settings.api_key = api_key
    return m.app


@pytest.fixture
def cliente(tmp_path_factory):
    app = _crear_app(tmp_path_factory.mktemp("sec"), api_key=None)
    with TestClient(app) as c:
        yield c


def test_rechaza_campo_desconocido(cliente):
    r = cliente.post("/predict", json={**OK, "campo_raro": 1})
    assert r.status_code == 422


def test_rechaza_lote_demasiado_grande(cliente):
    lote = {"registros": [OK] * 501}       # supera MAX_LOTE
    r = cliente.post("/predict/batch", json=lote)
    assert r.status_code == 422


def test_metrics_endpoint(cliente):
    cliente.post("/predict", json=OK)
    m = cliente.get("/metrics").json()
    assert "solicitudes" in m and "latencia_promedio_ms" in m
    assert "distribucion_predicciones" in m


def test_auth_rechaza_sin_credenciales(tmp_path_factory):
    app = _crear_app(tmp_path_factory.mktemp("auth"), api_key="secreto")
    with TestClient(app) as c:
        assert c.post("/predict", json=OK).status_code == 401
        assert c.post("/predict", headers={"X-API-Key": "malo"}, json=OK).status_code == 403
        assert c.post("/predict", headers={"X-API-Key": "secreto"}, json=OK).status_code == 200


def test_integridad_sha256_rechaza_modelo_alterado(tmp_path_factory):
    """E.3 - Si el SHA-256 esperado no coincide, el modelo NO se carga."""
    import joblib as _joblib

    from app import model_loader

    tmp = tmp_path_factory.mktemp("integridad")
    df = generar_datos_sinteticos(n=500, semilla=9)
    X, y = separar_x_y(df)
    modelo = construir_pipeline_random_forest().fit(X, y)
    ruta = tmp / "model.joblib"
    _joblib.dump(modelo, ruta)

    model_loader.settings.model_path = str(ruta)
    model_loader.settings.model_url = None
    model_loader.settings.model_sha256 = "0" * 64   # hash esperado incorrecto
    m = model_loader.ModeloEnMemoria()
    try:
        with pytest.raises(RuntimeError):
            m.cargar()
    finally:
        model_loader.settings.model_sha256 = None   # no contaminar otras pruebas
