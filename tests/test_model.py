"""Pruebas de modelo, desempeño y reproducibilidad (A.7)."""
from __future__ import annotations

from sklearn.model_selection import train_test_split

from src.config import cargar_business
from src.metricas import calcular_metricas, proba_positiva
from src.modelos import construir_pipeline_random_forest
from src.preprocess import separar_x_y


def _split_xy(df, semilla=42):
    train, test = train_test_split(
        df, test_size=0.2, random_state=semilla, stratify=df["target"]
    )
    X_train, y_train = separar_x_y(train)
    X_test, y_test = separar_x_y(test)
    return X_train, y_train, X_test, y_test


def test_prediccion_numero_resultados(datos_sinteticos):
    """El modelo devuelve el número esperado de predicciones (Modelo)."""
    X_train, y_train, X_test, y_test = _split_xy(datos_sinteticos)
    modelo = construir_pipeline_random_forest()
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    assert len(pred) == len(X_test)
    assert set(pred).issubset({0, 1})


def test_desempeno_supera_umbral(datos_sinteticos):
    """La métrica principal supera umbral_minimo_metrica de business.yaml."""
    business = cargar_business()
    metrica = business["metrica_tecnica"]
    umbral = business["umbral_minimo_metrica"]

    X_train, y_train, X_test, y_test = _split_xy(datos_sinteticos)
    modelo = construir_pipeline_random_forest()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    met = calcular_metricas(y_test, y_pred, proba_positiva(modelo, X_test))
    assert met[metrica] >= umbral, (
        f"{metrica}={met[metrica]:.3f} no supera el umbral {umbral}"
    )


def test_reproducibilidad_misma_semilla(datos_sinteticos):
    """Entrenar dos veces con la misma semilla da el mismo resultado."""
    X_train, y_train, X_test, y_test = _split_xy(datos_sinteticos)

    m1 = construir_pipeline_random_forest()
    m1.fit(X_train, y_train)
    pred1 = m1.predict(X_test)

    m2 = construir_pipeline_random_forest()
    m2.fit(X_train, y_train)
    pred2 = m2.predict(X_test)

    assert (pred1 == pred2).all(), "Mismo seed debería dar predicciones idénticas"
