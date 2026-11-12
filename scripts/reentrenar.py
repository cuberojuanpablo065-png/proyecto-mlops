"""D - Reentrenamiento con control de calidad y auditoría.

Flujo:
  1. Reentrena el modelo (src.train) -> models/model.joblib + model_meta.json
  2. Compara el candidato contra el champion actual (config/qa_gates.yaml).
  3. Si CUMPLE las condiciones -> promueve el candidato a champion.
     Si NO -> conserva el champion actual.
  4. Genera un JSON de auditoría (fecha, disparador, commit, versión, decisión).

Uso:
    python -m scripts.reentrenar --disparador manual
    python -m scripts.reentrenar --disparador drift
El champion vive en models/champion.joblib (+ champion_meta.json), que el
servicio puede descargar sin reconstruir la imagen.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.qa_gates import evaluar
from src.config import RAIZ, RUTA_MODELOS, RUTA_REPORTES

CHAMPION = RUTA_MODELOS / "champion.joblib"
CHAMPION_META = RUTA_MODELOS / "champion_meta.json"


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "sin-git"


def _leer_json(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reentrenamiento con QA y auditoría")
    parser.add_argument("--disparador", default="manual",
                        choices=["manual", "drift", "schedule"])
    args = parser.parse_args(argv)

    # 1. Reentrenar (genera model.joblib + model_meta.json).
    print("[reentrenar] Entrenando candidato...")
    r = subprocess.run([sys.executable, "-m", "src.train"], cwd=str(RAIZ))
    if r.returncode != 0:
        print("[reentrenar] El entrenamiento falló.")
        return 1

    cand_meta = _leer_json(RUTA_MODELOS / "model_meta.json") or {}
    champ_meta = _leer_json(CHAMPION_META)

    # 2. Evaluar condiciones de QA.
    aprobado, detalle = evaluar(
        cand_meta.get("metricas", {}),
        champ_meta.get("metricas") if champ_meta else None,
    )
    print(f"[reentrenar] QA aprobado={aprobado}")

    # 3. Promover o conservar.
    if aprobado:
        shutil.copy(RUTA_MODELOS / "model.joblib", CHAMPION)
        CHAMPION_META.write_text(json.dumps(cand_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        decision = "promovido_a_champion"
    else:
        decision = "conservado_champion_actual"

    # 4. Auditoría (D.3).
    auditoria = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "disparador": args.disparador,
        "commit": _commit(),
        "version_modelo_candidato": cand_meta.get("sha256", "")[:12],
        "metrica_principal": cand_meta.get("metrica_principal"),
        "metricas_candidato": cand_meta.get("metricas"),
        "qa": detalle,
        "decision": decision,
    }
    RUTA_REPORTES.mkdir(parents=True, exist_ok=True)
    salida = RUTA_REPORTES / "auditoria_reentrenamiento.json"
    salida.write_text(json.dumps(auditoria, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[reentrenar] Decisión: {decision}")
    print(f"[reentrenar] Auditoría guardada en {salida}")
    print(json.dumps(auditoria, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
