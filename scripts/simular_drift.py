"""C.2 - Simula drift enviando datos MODIFICADOS a la API.

Toma filas reales del conjunto de prueba, les aplica un desplazamiento
(multiplica los rezagos por un factor) para cambiar su distribución, y las
envía a POST /predict. Así las predicciones registradas se "contaminan" y el
reporte de drift (scripts/drift_report.py) debe detectarlo.

Uso:
    python -m scripts.simular_drift --url http://localhost:8000 --n 100 --factor 3
"""
from __future__ import annotations

import argparse
import sys

import httpx
import pandas as pd

from src.config import RAIZ


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Envía datos modificados a la API")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--factor", type=float, default=3.0, help="Factor de desplazamiento de los rezagos")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args(argv)

    test_path = RAIZ / "data" / "processed" / "test.csv"
    if not test_path.exists():
        print("[simular] No hay test.csv. Ejecute el pipeline primero.")
        return 1
    df = pd.read_csv(test_path).sample(min(args.n, 10_000), random_state=1)

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    enviados = 0
    for _, fila in df.iterrows():
        payload = {
            "state_code": str(fila["state_code"]),
            "scope": str(fila["scope"]),
            "crime_category": str(fila["crime_category"]),
            "year": int(fila["year"]),
            # Desplazamiento que induce drift:
            "lag1": max(float(fila["lag1"]) * args.factor, 1.0),
            "lag2": (float(fila["lag2"]) * args.factor if pd.notna(fila.get("lag2")) else None),
        }
        try:
            r = httpx.post(f"{args.url}/predict", json=payload, headers=headers, timeout=15)
            if r.status_code == 200:
                enviados += 1
        except Exception as exc:
            print(f"[simular] error: {exc}")
            break
    print(f"[simular] {enviados} solicitudes con datos modificados (factor={args.factor})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
