"""B.3 - Registro de dos versiones del modelo con alias en MLflow.

Entrena el modelo con las DOS versiones de los datos y registra cada una
como una versión en el Registro de Modelos de MLflow, asignándole un alias:
  - "baseline"  -> modelo entrenado con los datos v1 (años <= corte)
  - "champion"  -> modelo entrenado con los datos v2 (todos los años)

Usa un backend sqlite (mlflow.db) porque el Registro de Modelos lo requiere.

Uso:
    python -m src.registrar_modelos
    mlflow ui --backend-store-uri sqlite:///mlflow.db   # para verlo
"""
from __future__ import annotations

import sys

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from .config import RAIZ, cargar_params
from .data import dividir
from .ingest import construir_panel
from .metricas import calcular_metricas, proba_positiva
from .modelos import construir_modelos
from .preprocess import separar_x_y
from .validate import validar

NOMBRE_REGISTRO = "modelo-aumento-criminalidad"
EXPERIMENTO = "crime-india-aumento"

# Corte fijo para la versión "baseline" (datos v1), independiente de
# params.yaml, para que baseline y champion sean dos versiones distintas.
CORTE_BASELINE = 2019


def _registrar(modelo):
    """log_model con nombre de registro (compatible MLflow 2.x y 3.x)."""
    try:  # MLflow 3.x
        return mlflow.sklearn.log_model(
            modelo, name="modelo",
            serialization_format="cloudpickle",
            registered_model_name=NOMBRE_REGISTRO,
        )
    except TypeError:  # MLflow 2.x
        return mlflow.sklearn.log_model(
            modelo, artifact_path="modelo",
            serialization_format="cloudpickle",
            registered_model_name=NOMBRE_REGISTRO,
        )


def main() -> int:
    params = cargar_params()
    fuente = RAIZ / params["datos"]["fuente_externa"]
    metrica = params["datos"]["metrica_fuente"]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENTO)
    client = MlflowClient()

    # (etiqueta, año de corte, alias)
    versiones = [
        ("datos_v1", CORTE_BASELINE, "baseline"),
        ("datos_v2", None, "champion"),
    ]

    for etiqueta, corte, alias in versiones:
        panel = validar(construir_panel(fuente, metrica, corte))
        train, test = dividir(panel)
        X_train, y_train = separar_x_y(train)
        X_test, y_test = separar_x_y(test)

        modelo = construir_modelos()["random_forest"]
        with mlflow.start_run(run_name=f"registro_{etiqueta}"):
            modelo.fit(X_train, y_train)
            met = calcular_metricas(
                y_test, modelo.predict(X_test), proba_positiva(modelo, X_test)
            )
            mlflow.set_tag("version_datos", etiqueta)
            mlflow.log_metrics(met)
            _registrar(modelo)

        # La versión recién creada es la de mayor número; le ponemos el alias.
        versiones_reg = client.search_model_versions(f"name='{NOMBRE_REGISTRO}'")
        ultima = max(versiones_reg, key=lambda v: int(v.version))
        client.set_registered_model_alias(NOMBRE_REGISTRO, alias, ultima.version)
        print(
            f"[registro] {etiqueta}: version {ultima.version} "
            f"alias '{alias}'  f1={met['f1']:.4f}"
        )

    print(f"\n[registro] Modelo '{NOMBRE_REGISTRO}' con 2 versiones y alias.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
