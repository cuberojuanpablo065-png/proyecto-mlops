"""Genera una versión compacta del modelo para desplegar en la nube.

El modelo completo puede pesar cientos de MB. Para la capa gratuita de
Render conviene uno pequeño (menos árboles y profundidad limitada), que
se publica como asset de un GitHub Release y el servicio descarga al iniciar.

Uso:  python -m scripts.modelo_compacto
"""
from __future__ import annotations

import json
import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.data import cargar_procesados
from src.metricas import calcular_metricas, proba_positiva
from src.preprocess import construir_preprocesador, separar_x_y


def main() -> int:
    tr, te = cargar_procesados()
    X_train, y_train = separar_x_y(tr)
    X_test, y_test = separar_x_y(te)

    clf = RandomForestClassifier(
        n_estimators=60, max_depth=16, min_samples_leaf=4,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    modelo = Pipeline([("prep", construir_preprocesador()), ("clf", clf)])
    modelo.fit(X_train, y_train)

    met = calcular_metricas(y_test, modelo.predict(X_test), proba_positiva(modelo, X_test))
    joblib.dump(modelo, "models/model.joblib", compress=3)
    json.dump(
        {"mejor_modelo": "random_forest", "metrica_principal": "f1",
         "version_datos": "v2", "metricas": {k: round(v, 4) for k, v in met.items()}},
        open("models/model_meta.json", "w"), indent=2, ensure_ascii=False,
    )
    print(f"F1={met['f1']:.4f}  tamaño={os.path.getsize('models/model.joblib')/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
