"""C.2 - Reporte de drift.

Compara la distribución de los datos recientes (predicciones registradas en
SQLite) contra la distribución de entrenamiento (referencia) usando el Índice
de Estabilidad de Población (PSI). Genera reports/drift_report.json y un
resumen en Markdown, y marca si hay drift según config/drift.yaml.

Uso:
    python -m scripts.drift_report
Salida: código 2 si se detecta drift (útil para que el workflow dispare alerta).
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import RAIZ, RUTA_REPORTES, cargar_business

DRIFT_YAML = RAIZ / "config" / "drift.yaml"


def _psi(referencia: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Índice de Estabilidad de Población entre dos muestras numéricas."""
    referencia = referencia[~np.isnan(referencia)]
    actual = actual[~np.isnan(actual)]
    if len(referencia) == 0 or len(actual) == 0:
        return 0.0
    # Bordes por cuantiles de la referencia.
    bordes = np.unique(np.quantile(referencia, np.linspace(0, 1, bins + 1)))
    if len(bordes) < 3:
        return 0.0
    ref_hist, _ = np.histogram(referencia, bins=bordes)
    act_hist, _ = np.histogram(actual, bins=bordes)
    ref_pct = np.clip(ref_hist / ref_hist.sum(), 1e-6, None)
    act_pct = np.clip(act_hist / max(act_hist.sum(), 1), 1e-6, None)
    return float(np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct)))


def _cargar_predicciones(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    con = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(
            "SELECT entrada_json, prediccion, probabilidad FROM predicciones", con
        )
    finally:
        con.close()
    if df.empty:
        return df
    entradas = pd.json_normalize(df["entrada_json"].apply(json.loads))
    return pd.concat([entradas, df[["prediccion", "probabilidad"]]], axis=1)


def main() -> int:
    cfg = yaml.safe_load(DRIFT_YAML.read_text(encoding="utf-8"))
    business = cargar_business()

    # Referencia: datos de entrenamiento.
    ref_path = RAIZ / "data" / "processed" / "train.csv"
    if not ref_path.exists():
        print("[drift] No hay datos de referencia (train.csv). Ejecute el pipeline.")
        return 1
    ref = pd.read_csv(ref_path)

    # Actual: predicciones registradas por el servicio.
    from src.config import RAIZ as _R
    db = _R / "data" / "predicciones.db"
    actual = _cargar_predicciones(db)

    resultado: dict = {"drift_detectado": False, "variables": {}, "avisos": []}

    if len(actual) < cfg["minimo_muestras"]:
        resultado["avisos"].append(
            f"Muestras insuficientes ({len(actual)} < {cfg['minimo_muestras']}); "
            "análisis no concluyente."
        )
        _guardar(resultado)
        print(f"[drift] {resultado['avisos'][-1]}")
        return 0

    # PSI por variable.
    for var in cfg["variables"]:
        if var in ref.columns and var in actual.columns:
            psi = _psi(ref[var].to_numpy(float), actual[var].to_numpy(float))
            hay = psi > cfg["umbral_psi"]
            resultado["variables"][var] = {"psi": round(psi, 4), "drift": hay}
            if hay:
                resultado["drift_detectado"] = True

    # Cambio en la tasa de predicciones positivas.
    tasa_ref = float((ref[business["columna_objetivo"]] == 1).mean())
    tasa_act = float((actual["prediccion"] == 1).mean())
    cambio = abs(tasa_act - tasa_ref)
    resultado["tasa_prediccion"] = {
        "referencia": round(tasa_ref, 4), "actual": round(tasa_act, 4),
        "cambio": round(cambio, 4), "drift": cambio > cfg["umbral_tasa_prediccion"],
    }
    if cambio > cfg["umbral_tasa_prediccion"]:
        resultado["drift_detectado"] = True

    _guardar(resultado)
    print(f"[drift] drift_detectado={resultado['drift_detectado']}")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    return 2 if resultado["drift_detectado"] else 0


def _guardar(resultado: dict) -> None:
    RUTA_REPORTES.mkdir(parents=True, exist_ok=True)
    (RUTA_REPORTES / "drift_report.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Resumen Markdown para la alerta.
    lineas = ["# Reporte de drift", "",
              f"**Drift detectado:** {resultado['drift_detectado']}", ""]
    for var, info in resultado.get("variables", {}).items():
        lineas.append(f"- {var}: PSI={info['psi']} (drift={info['drift']})")
    if "tasa_prediccion" in resultado:
        t = resultado["tasa_prediccion"]
        lineas.append(f"- tasa 'sube': ref={t['referencia']} actual={t['actual']} (drift={t['drift']})")
    for aviso in resultado.get("avisos", []):
        lineas.append(f"- aviso: {aviso}")
    (RUTA_REPORTES / "drift_report.md").write_text("\n".join(lineas), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
