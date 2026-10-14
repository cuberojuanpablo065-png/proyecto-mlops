"""Prueba de serialización (A.7 / A.8.2).

Carga el modelo guardado y verifica que predice igual que el original.
"""
from __future__ import annotations

import joblib
from sklearn.model_selection import train_test_split

from src.modelos import construir_pipeline_random_forest
from src.preprocess import separar_x_y


def test_carga_modelo_predice_igual(datos_sinteticos, tmp_path):
    """El modelo serializado con joblib predice igual tras recargarse."""
    train, test = train_test_split(
        datos_sinteticos, test_size=0.2, random_state=42, stratify=datos_sinteticos["target"]
    )
    X_train, y_train = separar_x_y(train)
    X_test, _ = separar_x_y(test)

    modelo = construir_pipeline_random_forest()
    modelo.fit(X_train, y_train)
    pred_original = modelo.predict(X_test)

    ruta = tmp_path / "modelo_test.joblib"
    joblib.dump(modelo, ruta)
    modelo_cargado = joblib.load(ruta)
    pred_cargado = modelo_cargado.predict(X_test)

    assert (pred_original == pred_cargado).all()
