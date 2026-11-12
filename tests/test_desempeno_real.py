"""Prueba de desempeño del MODELO REAL (atiende el feedback del E1).

A diferencia de las pruebas unitarias con datos sintéticos, esta prueba evalúa
el modelo real serializado sobre el conjunto de prueba real (división temporal)
y verifica que supere el umbral mínimo de negocio. Si los artefactos reales no
están presentes (p. ej. no se ha ejecutado el pipeline), la prueba se omite.
"""
from __future__ import annotations

import joblib
import pandas as pd
import pytest

from src.config import RAIZ, cargar_business
from src.metricas import calcular_metricas, proba_positiva
from src.preprocess import separar_x_y

MODELO = RAIZ / "models" / "model.joblib"
TEST = RAIZ / "data" / "processed" / "test.csv"


@pytest.mark.skipif(
    not (MODELO.exists() and TEST.exists()),
    reason="Faltan artefactos reales (ejecute el pipeline: dvc repro / src.train).",
)
def test_modelo_real_supera_umbral():
    business = cargar_business()
    metrica = business["metrica_tecnica"]
    umbral = business["umbral_minimo_metrica"]

    modelo = joblib.load(MODELO)
    df_test = pd.read_csv(TEST)
    X_test, y_test = separar_x_y(df_test)

    met = calcular_metricas(y_test, modelo.predict(X_test), proba_positiva(modelo, X_test))
    assert met[metrica] >= umbral, (
        f"El modelo real tiene {metrica}={met[metrica]:.3f}, por debajo del umbral {umbral}"
    )
