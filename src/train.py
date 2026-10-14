"""Entrenamiento, comparación, optimización, serialización y tracking.

Cubre:
  A.4 - Línea base + 2 modelos + tabla de comparación (reports/comparacion.csv)
  A.5 - Optimización de hiperparámetros (RandomizedSearchCV, espacio en params.yaml)
  A.8 - Serialización del mejor modelo (joblib) junto con el preprocesamiento
  A.9 - Seguimiento de experimentos en MLflow (params, métricas, modelo, commit git)

Uso:
    python -m src.train
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .config import (
    RUTA_MODELOS,
    RUTA_REPORTES,
    asegurar_directorios,
    cargar_business,
    cargar_params,
)
from .data import cargar_procesados
from .metricas import calcular_metricas, proba_positiva
from .modelos import construir_modelos, construir_pipeline_random_forest
from .preprocess import separar_x_y

EXPERIMENTO_MLFLOW = "crime-india-aumento"


def _log_model(modelo, nombre: str = "modelo") -> None:
    """Registra el modelo en MLflow (compatible con MLflow 2.x y 3.x)."""
    # serialization_format="cloudpickle" evita el serializador skops
    # (que en MLflow 3.x rechaza tipos como numpy.dtype).
    try:  # MLflow 3.x usa 'name'
        mlflow.sklearn.log_model(modelo, name=nombre, serialization_format="cloudpickle")
    except TypeError:  # MLflow 2.x usa 'artifact_path'
        mlflow.sklearn.log_model(
            modelo, artifact_path=nombre, serialization_format="cloudpickle"
        )


def _commit_git() -> str:
    """Devuelve el hash corto del commit actual (o 'sin-git')."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "sin-git"


def _hash_datos_dvc() -> str:
    """Hash MD5 del .dvc de los datos crudos (versión de datos), si existe."""
    params = cargar_params()
    from .config import RAIZ

    dvc_file = RAIZ / (params["datos"]["archivo_crudo"] + ".dvc")
    if dvc_file.exists():
        import yaml

        with open(dvc_file) as f:
            meta = yaml.safe_load(f)
        try:
            return meta["outs"][0]["md5"]
        except Exception:
            return "sin-hash"
    return "sin-dvc"


def entrenar_y_comparar(X_train, y_train, X_test, y_test, metrica_principal: str):
    """A.4 - Entrena baseline + modelos y arma la tabla de comparación."""
    modelos = construir_modelos()
    filas = []
    entrenados: dict[str, object] = {}

    mlflow.set_experiment(EXPERIMENTO_MLFLOW)
    commit = _commit_git()
    hash_datos = _hash_datos_dvc()

    for nombre, modelo in modelos.items():
        with mlflow.start_run(run_name=nombre):
            t0 = time.perf_counter()
            modelo.fit(X_train, y_train)
            t_entrenamiento = time.perf_counter() - t0

            y_pred = modelo.predict(X_test)
            y_proba = proba_positiva(modelo, X_test)
            met = calcular_metricas(y_test, y_pred, y_proba)

            # A.9 - registrar en MLflow
            mlflow.set_tag("git_commit", commit)
            mlflow.set_tag("dvc_data_md5", hash_datos)
            mlflow.set_tag("modelo", nombre)
            mlflow.log_param("modelo", nombre)
            mlflow.log_metric("tiempo_entrenamiento_s", t_entrenamiento)
            for k, v in met.items():
                mlflow.log_metric(k, v)
            _log_model(modelo)

            fila = {"modelo": nombre, "tiempo_entrenamiento_s": round(t_entrenamiento, 4)}
            fila.update({k: round(v, 4) for k, v in met.items()})
            filas.append(fila)
            entrenados[nombre] = modelo
            print(
                f"[train] {nombre:30s} {metrica_principal}="
                f"{met.get(metrica_principal, float('nan')):.4f}  "
                f"({t_entrenamiento:.2f}s)"
            )

    comparacion = pd.DataFrame(filas).sort_values(
        metrica_principal, ascending=False
    )
    return comparacion, entrenados


def optimizar_random_forest(X_train, y_train, metrica_principal: str):
    """A.5 - Optimiza RF con RandomizedSearchCV/GridSearchCV (CV en train)."""
    params = cargar_params()
    opt = params["optimizacion"]
    pipe = construir_pipeline_random_forest()
    espacio = opt["espacio_busqueda"]

    scoring = metrica_principal  # p.ej. "f1"
    cv = params["cv"]["n_folds"]
    semilla = params["semilla"]

    if opt["metodo"] == "grid_search":
        buscador = GridSearchCV(
            pipe, espacio, scoring=scoring, cv=cv, n_jobs=-1, refit=True
        )
    else:
        buscador = RandomizedSearchCV(
            pipe,
            espacio,
            n_iter=opt["n_iter"],
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            random_state=semilla,
            refit=True,
        )

    mlflow.set_experiment(EXPERIMENTO_MLFLOW)
    with mlflow.start_run(run_name="random_forest_optimizado"):
        t0 = time.perf_counter()
        buscador.fit(X_train, y_train)
        t_opt = time.perf_counter() - t0

        mlflow.set_tag("git_commit", _commit_git())
        mlflow.set_tag("modelo", "random_forest_optimizado")
        mlflow.log_params(buscador.best_params_)
        mlflow.log_metric(f"cv_{scoring}", float(buscador.best_score_))
        mlflow.log_metric("tiempo_optimizacion_s", t_opt)
        _log_model(buscador.best_estimator_)

    print(
        f"[train] RF optimizado: mejor cv_{scoring}={buscador.best_score_:.4f} "
        f"({t_opt:.1f}s)"
    )
    print(f"[train] Mejores hiperparámetros: {buscador.best_params_}")
    return buscador.best_estimator_, buscador.best_params_, float(buscador.best_score_)


def main() -> int:
    asegurar_directorios()
    business = cargar_business()
    metrica_principal = business["metrica_tecnica"]

    df_train, df_test = cargar_procesados()
    X_train, y_train = separar_x_y(df_train)
    X_test, y_test = separar_x_y(df_test)

    # A.4 - comparación
    comparacion, entrenados = entrenar_y_comparar(
        X_train, y_train, X_test, y_test, metrica_principal
    )

    # A.5 - optimización del RF
    rf_opt, mejores_params, cv_score = optimizar_random_forest(
        X_train, y_train, metrica_principal
    )

    # Evaluar el RF optimizado en test y agregarlo a la comparación
    y_pred = rf_opt.predict(X_test)
    y_proba = proba_positiva(rf_opt, X_test)
    met_opt = calcular_metricas(y_test, y_pred, y_proba)
    fila_opt = {"modelo": "random_forest_optimizado", "tiempo_entrenamiento_s": None}
    fila_opt.update({k: round(v, 4) for k, v in met_opt.items()})
    comparacion = pd.concat(
        [comparacion, pd.DataFrame([fila_opt])], ignore_index=True
    ).sort_values(metrica_principal, ascending=False, ignore_index=True)

    # Guardar tabla de comparación (A.4)
    ruta_comp = RUTA_REPORTES / "comparacion.csv"
    comparacion.to_csv(ruta_comp, index=False)
    print(f"\n[train] Comparación guardada en {ruta_comp}:")
    print(comparacion.to_string(index=False))

    # Seleccionar el mejor modelo por la métrica principal
    mejor_nombre = comparacion.iloc[0]["modelo"]
    candidatos = dict(entrenados)
    candidatos["random_forest_optimizado"] = rf_opt
    mejor_modelo = candidatos[mejor_nombre]
    print(f"\n[train] Mejor modelo: {mejor_nombre}")

    # A.8 - Serialización (Pipeline incluye el preprocesamiento)
    ruta_modelo = RUTA_MODELOS / "model.joblib"
    joblib.dump(mejor_modelo, ruta_modelo)
    print(f"[train] Modelo serializado en {ruta_modelo}")

    # Metadatos del modelo seleccionado
    meta = {
        "mejor_modelo": mejor_nombre,
        "metrica_principal": metrica_principal,
        "metricas_test": met_opt if mejor_nombre == "random_forest_optimizado" else None,
        "mejores_hiperparametros": mejores_params,
        "cv_score": cv_score,
        "git_commit": _commit_git(),
    }
    (RUTA_MODELOS / "model_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
