"""A.6 - Evaluación: métricas y gráficos generados por código.

Carga el modelo serializado y el conjunto de prueba, calcula la métrica
principal y secundarias, y genera:
  - reports/metrics.json
  - reports/figures/matriz_confusion.png
  - reports/figures/curva_roc.png
  - reports/figures/curva_pr.png

Opcional (valor agregado): calcula el umbral de decisión que minimiza el
costo esperado usando los costos de business.yaml.

Uso:
    python -m src.evaluate
"""
from __future__ import annotations

import json
import sys

import joblib
import matplotlib

matplotlib.use("Agg")  # backend sin interfaz gráfica
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

from .config import RUTA_FIGURAS, RUTA_MODELOS, RUTA_REPORTES, asegurar_directorios, cargar_business
from .data import cargar_procesados
from .metricas import calcular_metricas, proba_positiva
from .preprocess import separar_x_y


def _cargar_modelo():
    ruta = RUTA_MODELOS / "model.joblib"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecute primero: python -m src.train"
        )
    return joblib.load(ruta)


def umbral_optimo_por_costo(y_true, y_proba, costo_fn: float, costo_fp: float):
    """Umbral que minimiza el costo esperado (FN*c_fn + FP*c_fp)."""
    if y_proba is None:
        return None
    umbrales = np.linspace(0.01, 0.99, 99)
    mejor_u, mejor_costo = 0.5, float("inf")
    for u in umbrales:
        pred = (y_proba >= u).astype(int)
        fn = int(((y_true == 1) & (pred == 0)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        costo = fn * costo_fn + fp * costo_fp
        if costo < mejor_costo:
            mejor_costo, mejor_u = costo, float(u)
    return {"umbral": round(mejor_u, 3), "costo_esperado": mejor_costo}


def main() -> int:
    asegurar_directorios()
    business = cargar_business()
    metrica_principal = business["metrica_tecnica"]

    modelo = _cargar_modelo()
    _, df_test = cargar_procesados()
    X_test, y_test = separar_x_y(df_test)

    y_pred = modelo.predict(X_test)
    y_proba = proba_positiva(modelo, X_test)
    met = calcular_metricas(y_test, y_pred, y_proba)

    # --- Matriz de confusión ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No sube", "Sube"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Matriz de confusión (test)")
    fig.tight_layout()
    fig.savefig(RUTA_FIGURAS / "matriz_confusion.png", dpi=120)
    plt.close(fig)

    # --- Curvas ROC y PR (si hay probabilidades) ---
    if y_proba is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
        ax.set_title("Curva ROC (test)")
        fig.tight_layout()
        fig.savefig(RUTA_FIGURAS / "curva_roc.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax)
        ax.set_title("Curva Precision-Recall (test)")
        fig.tight_layout()
        fig.savefig(RUTA_FIGURAS / "curva_pr.png", dpi=120)
        plt.close(fig)

    # --- Opcional: umbral óptimo por costo ---
    costos = business.get("costos", {})
    umbral = umbral_optimo_por_costo(
        y_test.to_numpy(),
        y_proba,
        costos.get("falso_negativo", 1.0),
        costos.get("falso_positivo", 1.0),
    )

    resultado = {
        "metrica_principal": metrica_principal,
        "valor_metrica_principal": met.get(metrica_principal),
        "metricas": met,
        "umbral_optimo_por_costo": umbral,
        "matriz_confusion": cm.tolist(),
    }
    ruta_metrics = RUTA_REPORTES / "metrics.json"
    ruta_metrics.write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[evaluate] {metrica_principal} = {met.get(metrica_principal):.4f}")
    print(f"[evaluate] Métricas: {json.dumps(met, indent=2)}")
    print(f"[evaluate] Guardado: {ruta_metrics} y figuras en {RUTA_FIGURAS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
