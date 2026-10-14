"""Definición de modelos y líneas base.

- Línea base trivial: DummyClassifier (estrategia más frecuente).
- Línea base de negocio: "regla de reversión": si el año pasado no subió
  (lag1 <= lag2), predecir que este año sube.
- Modelos: Regresión Logística y Random Forest, cada uno dentro de un
  Pipeline que incluye el preprocesamiento (evita fuga de datos).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import cargar_params
from .preprocess import construir_preprocesador


class ReglaReversion(BaseEstimator, ClassifierMixin):
    """Línea base de negocio (reversión a la media).

    Predice "sube" (1) cuando el año anterior NO subió respecto al previo,
    es decir cuando lag1 <= lag2. Opera sobre columnas crudas; no entrena.
    """

    def fit(self, X, y=None):  # noqa: D102
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):  # noqa: D102
        lag1 = X["lag1"].to_numpy(dtype=float)
        lag2 = X["lag2"].to_numpy(dtype=float)
        # NaN en lag2 -> comparación False -> predice 0
        pred = np.where(np.isnan(lag2), 0, (lag1 <= lag2).astype(int))
        return pred.astype(int)

    def predict_proba(self, X):  # noqa: D102
        pred = self.predict(X)
        return np.column_stack([1 - pred, pred])


def _pipeline(clf) -> Pipeline:
    """Envuelve un clasificador con el preprocesamiento."""
    return Pipeline(steps=[("prep", construir_preprocesador()), ("clf", clf)])


def construir_modelos() -> dict[str, object]:
    """Devuelve el diccionario {nombre: estimador} a comparar (A.4)."""
    params = cargar_params()
    semilla = params["semilla"]
    cfg = params["modelos"]

    modelos: dict[str, object] = {}
    modelos["baseline_dummy"] = _pipeline(
        DummyClassifier(strategy="most_frequent", random_state=semilla)
    )
    modelos["baseline_regla_reversion"] = ReglaReversion()

    lr = cfg["regresion_logistica"]
    modelos["regresion_logistica"] = _pipeline(
        LogisticRegression(
            C=lr["C"],
            max_iter=lr["max_iter"],
            class_weight=lr["class_weight"],
            random_state=semilla,
        )
    )
    rf = cfg["random_forest"]
    modelos["random_forest"] = _pipeline(
        RandomForestClassifier(
            n_estimators=rf["n_estimators"],
            class_weight=rf["class_weight"],
            n_jobs=rf["n_jobs"],
            random_state=semilla,
        )
    )
    return modelos


def construir_pipeline_random_forest() -> Pipeline:
    """Pipeline RF base para optimización de hiperparámetros (A.5)."""
    params = cargar_params()
    return _pipeline(
        RandomForestClassifier(random_state=params["semilla"], n_jobs=1)
    )
