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

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

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
    """A.4 - Compara modelos por VALIDACIÓN CRUZADA en train (no por test).

    Feedback E1: el modelo final NO debe elegirse con el conjunto de prueba.
    Aquí se calcula el score de validación cruzada (cv) de cada candidato sobre
    el TRAIN; esa es la base para elegir. Las métricas de test se calculan solo
    como reporte honesto del desempeño, nunca para seleccionar.
    """
    modelos = construir_modelos()
    params = cargar_params()
    n_folds = params["cv"]["n_folds"]
    filas = []
    entrenados: dict[str, object] = {}
    cv_scores: dict[str, float] = {}

    mlflow.set_experiment(EXPERIMENTO_MLFLOW)
    commit = _commit_git()
    hash_datos = _hash_datos_dvc()

    for nombre, modelo in modelos.items():
        with mlflow.start_run(run_name=nombre):
            # Score de validación cruzada en train (criterio de selección).
            try:
                cv = cross_val_score(
                    clone(modelo), X_train, y_train,
                    scoring=metrica_principal, cv=n_folds, n_jobs=-1,
                ).mean()
            except Exception:
                cv = float("nan")

            # Ajuste final en todo el train y métricas de test (solo reporte).
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
            mlflow.log_metric(f"cv_{metrica_principal}", cv)
            mlflow.log_metric("tiempo_entrenamiento_s", t_entrenamiento)
            for k, v in met.items():
                mlflow.log_metric(f"test_{k}", v)
            _log_model(modelo)

            fila = {"modelo": nombre, f"cv_{metrica_principal}": round(cv, 4)}
            fila.update({f"test_{k}": round(v, 4) for k, v in met.items()})
            filas.append(fila)
            entrenados[nombre] = modelo
            cv_scores[nombre] = cv
            print(
                f"[train] {nombre:30s} cv_{metrica_principal}={cv:.4f}  "
                f"test_{metrica_principal}={met.get(metrica_principal, float('nan')):.4f}"
            )

    comparacion = pd.DataFrame(filas).sort_values(
        f"cv_{metrica_principal}", ascending=False, ignore_index=True
    )
    return comparacion, entrenados, cv_scores


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

    # A.4 - comparación por validación cruzada
    comparacion, entrenados, cv_scores = entrenar_y_comparar(
        X_train, y_train, X_test, y_test, metrica_principal
    )

    # A.5 - optimización del RF (su score es de validación cruzada en train)
    rf_opt, mejores_params, cv_score_rf = optimizar_random_forest(
        X_train, y_train, metrica_principal
    )
    cv_scores["random_forest_optimizado"] = cv_score_rf

    # Agregar el RF optimizado a la comparación (test solo como reporte)
    y_pred = rf_opt.predict(X_test)
    y_proba = proba_positiva(rf_opt, X_test)
    met_opt = calcular_metricas(y_test, y_pred, y_proba)
    fila_opt = {"modelo": "random_forest_optimizado", f"cv_{metrica_principal}": round(cv_score_rf, 4)}
    fila_opt.update({f"test_{k}": round(v, 4) for k, v in met_opt.items()})
    comparacion = pd.concat(
        [comparacion, pd.DataFrame([fila_opt])], ignore_index=True
    ).sort_values(f"cv_{metrica_principal}", ascending=False, ignore_index=True)

    # Guardar tabla de comparación (A.4)
    ruta_comp = RUTA_REPORTES / "comparacion.csv"
    comparacion.to_csv(ruta_comp, index=False)
    print(f"\n[train] Comparación guardada en {ruta_comp}:")
    print(comparacion.to_string(index=False))

    # Seleccionar el mejor modelo por el SCORE DE VALIDACIÓN CRUZADA (no test)
    candidatos = dict(entrenados)
    candidatos["random_forest_optimizado"] = rf_opt
    mejor_nombre = max(cv_scores, key=lambda k: (cv_scores[k] if cv_scores[k] == cv_scores[k] else -1))
    mejor_modelo = candidatos[mejor_nombre]
    print(f"\n[train] Mejor modelo (por cv_{metrica_principal}): {mejor_nombre}")

    # Métricas de test del modelo elegido (reporte honesto final)
    met_mejor = calcular_metricas(
        y_test, mejor_modelo.predict(X_test), proba_positiva(mejor_modelo, X_test)
    )

    # A.8 - Serialización (Pipeline incluye el preprocesamiento)
    ruta_modelo = RUTA_MODELOS / "model.joblib"
    joblib.dump(mejor_modelo, ruta_modelo)
    print(f"[train] Modelo serializado en {ruta_modelo}")

    # E.3 - Hash SHA-256 del modelo (integridad). Se guarda en el meta y en un
    # archivo aparte para verificarlo al cargar en el servicio.
    import hashlib

    sha = hashlib.sha256(ruta_modelo.read_bytes()).hexdigest()
    (RUTA_MODELOS / "model.sha256").write_text(sha, encoding="utf-8")
    print(f"[train] SHA-256: {sha}")

    # Metadatos del modelo seleccionado (siempre con métricas -> arregla /model-info)
    meta = {
        "mejor_modelo": mejor_nombre,
        "metrica_principal": metrica_principal,
        "cv_score": round(cv_scores[mejor_nombre], 4),
        "metricas": {k: round(v, 4) for k, v in met_mejor.items()},
        "metricas_test": {k: round(v, 4) for k, v in met_mejor.items()},
        "version_datos": "v2",
        "seleccion": f"validacion_cruzada_{metrica_principal}",
        "sha256": sha,
        "git_commit": _commit_git(),
    }
    (RUTA_MODELOS / "model_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
