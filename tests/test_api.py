"""Pruebas de la API del modelo con TestClient (Parte B.3).

Cubren: una predicción exitosa, una entrada inválida y el WebSocket.
El modelo se entrena sobre datos sintéticos y se carga en memoria, para
que las pruebas corran sin depender de archivos externos.
"""
from __future__ import annotations

import joblib
import pytest
from fastapi.testclient import TestClient

from src.modelos import construir_pipeline_random_forest
from src.preprocess import separar_x_y
from tests.conftest import generar_datos_sinteticos


@pytest.fixture(scope="module")
def cliente(tmp_path_factory, ):
    # 1) Entrenar un modelo pequeño y serializarlo a un archivo temporal.
    df = generar_datos_sinteticos(n=1200, semilla=1)
    X, y = separar_x_y(df)
    modelo = construir_pipeline_random_forest()
    modelo.fit(X, y)

    tmp = tmp_path_factory.mktemp("api")
    ruta_modelo = tmp / "model.joblib"
    joblib.dump(modelo, ruta_modelo)

    # 2) Apuntar la configuración al modelo y a una base temporal.
    from app import model_loader, storage
    model_loader.settings.model_path = str(ruta_modelo)
    model_loader.settings.model_meta_path = str(tmp / "no_existe.json")
    storage.settings.db_path = str(tmp / "pred.db")

    from app.main import app
    with TestClient(app) as c:
        yield c


def test_health(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json()["modelo_cargado"] is True


def test_prediccion_exitosa(cliente):
    r = cliente.post("/predict", json={
        "state_code": "MH", "scope": "ipc", "crime_category": "murder",
        "year": 2022, "lag1": 300, "lag2": 250, "lag3": 240,
    })
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["prediccion"] in (0, 1)
    assert 0.0 <= cuerpo["probabilidad"] <= 1.0


def test_entrada_invalida(cliente):
    # lag1 negativo viola la validación (gt=0) -> 422
    r = cliente.post("/predict", json={
        "state_code": "MH", "scope": "ipc", "crime_category": "murder",
        "year": 2022, "lag1": -10,
    })
    assert r.status_code == 422


def test_websocket(cliente):
    with cliente.websocket_connect("/ws/predict") as ws:
        ws.send_json({
            "state_code": "DL", "scope": "cyber", "crime_category": "cyber",
            "year": 2022, "lag1": 100, "lag2": 80,
        })
        respuesta = ws.receive_json()
        assert respuesta["prediccion"] in (0, 1)
