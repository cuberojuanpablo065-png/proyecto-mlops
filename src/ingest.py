"""A.3.1 - Ingesta y construcción del panel modelable.

Lee la tabla fuente `crime_state_year.csv` (CrimeBase / NCRB India) desde
`data/external/`, filtra los casos registrados y construye un panel por
serie (estado × scope × crime_head) con rezagos (features estrictamente
históricos) y el objetivo binario: ¿el valor del año AUMENTA respecto al
año anterior?

La ingeniería de rezagos es determinista (no se "ajusta" con datos) y se
hace sobre toda la serie usando SOLO el pasado de cada fila, por lo que no
introduce fuga de datos. El escalado/codificación (que sí se ajusta) va
dentro del Pipeline de sklearn, solo con train.

Uso:
    python -m src.ingest                 # construye la versión 1 (años <= corte)
    python -m src.ingest --simular-v2    # versión 2: incluye años recientes (B.2.3)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAIZ, cargar_params

# Clave que identifica cada serie temporal.
CLAVE_SERIE = ["state_code", "scope", "crime_head"]


def construir_panel(fuente: Path, metrica: str, anio_corte: int | None) -> pd.DataFrame:
    """Construye el panel con rezagos y objetivo desde la tabla fuente."""
    if not fuente.exists():
        raise FileNotFoundError(
            f"No existe la tabla fuente: {fuente}\n"
            "Descargue el dataset CrimeBase (NCRB India) de Kaggle y coloque "
            "crime_state_year.csv en data/external/ (ver README)."
        )
    df = pd.read_csv(fuente)

    # 1) Filtrar casos registrados, con año y estado válidos.
    d = df[
        (df["metric"] == metrica)
        & df["year"].notna()
        & df["state_code"].notna()
    ].copy()
    d["year"] = d["year"].astype(int)

    # 2) Un valor por (serie, crime_category, año). No se suma entre scopes
    #    (ver DATA_DICTIONARY). Dentro de una serie, se consolida por año.
    d = (
        d.groupby(CLAVE_SERIE + ["crime_category", "year"], as_index=False)["value"]
        .sum()
        .sort_values(CLAVE_SERIE + ["year"])
    )

    # 3) Rezagos por serie (solo pasado).
    g = d.groupby(CLAVE_SERIE)
    d["lag1"] = g["value"].shift(1)
    d["lag2"] = g["value"].shift(2)
    d["lag3"] = g["value"].shift(3)
    d["delta_prev"] = d["lag1"] - d["lag2"]
    d["pct_prev"] = (d["lag1"] - d["lag2"]) / (d["lag2"] + 1.0)

    # 4) Objetivo: ¿el valor de ESTE año supera al del año anterior?
    d["target"] = (d["value"] > d["lag1"]).astype(int)

    # 5) Filtrar filas sin historia o sin actividad previa (ceros triviales).
    d = d[d["lag1"].notna() & (d["lag1"] > 0)].copy()

    # 6) Versionamiento de datos: la v1 usa años <= corte.
    if anio_corte is not None:
        d = d[d["year"] <= anio_corte].copy()

    d = d.reset_index(drop=True)
    return d


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingesta y construcción del panel")
    parser.add_argument(
        "--simular-v2",
        action="store_true",
        help="Versión 2 de datos: incluye años recientes (sin corte)",
    )
    args = parser.parse_args(argv)

    params = cargar_params()
    cfg = params["datos"]
    fuente = RAIZ / cfg["fuente_externa"]
    destino = RAIZ / cfg["archivo_crudo"]
    destino.parent.mkdir(parents=True, exist_ok=True)

    anio_corte = None if args.simular_v2 else cfg.get("anio_corte_v1")
    panel = construir_panel(fuente, cfg["metrica_fuente"], anio_corte)
    panel.to_csv(destino, index=False)

    etiqueta = "v2 (todos los años)" if args.simular_v2 else f"v1 (<= {anio_corte})"
    print(
        f"[ingest] Panel {etiqueta}: {len(panel):,} filas, "
        f"años {panel['year'].min()}-{panel['year'].max()}, "
        f"% 'sube' = {panel['target'].mean():.3f} -> {destino}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
