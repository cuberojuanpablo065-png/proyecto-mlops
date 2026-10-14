"""Cálculo de métricas de clasificación (A.6).

Centraliza el cálculo para que train, evaluate y las pruebas usen la misma
definición. La métrica principal se lee de business.yaml.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calcular_metricas(y_true, y_pred, y_proba=None) -> dict[str, float]:
    """Devuelve un diccionario de métricas para la clase positiva (1)."""
    metricas = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            metricas["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            metricas["pr_auc"] = float(average_precision_score(y_true, y_proba))
        except ValueError:
            # p.ej. si y_true tiene una sola clase en un subconjunto
            metricas["roc_auc"] = float("nan")
            metricas["pr_auc"] = float("nan")
    return metricas


def proba_positiva(modelo, X) -> np.ndarray | None:
    """Obtiene la probabilidad de la clase positiva si el modelo la expone."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return None
