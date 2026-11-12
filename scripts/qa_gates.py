"""C.3 - Condiciones de calidad (QA gates) para reemplazar el modelo.

Un modelo candidato (recién reentrenado) solo reemplaza al champion si cumple
TODAS las condiciones de config/qa_gates.yaml.

Uso como módulo:
    from scripts.qa_gates import evaluar
    aprobado, detalle = evaluar(metricas_candidato, metricas_champion)
"""
from __future__ import annotations

import yaml

from src.config import RAIZ, cargar_business

QA_YAML = RAIZ / "config" / "qa_gates.yaml"


def evaluar(metricas_candidato: dict, metricas_champion: dict | None) -> tuple[bool, dict]:
    """Evalúa el candidato contra las condiciones de QA.

    Devuelve (aprobado, detalle) donde detalle explica cada condición.
    """
    cfg = yaml.safe_load(QA_YAML.read_text(encoding="utf-8"))
    business = cargar_business()
    metrica = business["metrica_tecnica"]
    umbral = business["umbral_minimo_metrica"]

    valor_cand = float(metricas_candidato.get(metrica, 0.0))
    detalle: dict = {"metrica": metrica, "valor_candidato": round(valor_cand, 4),
                     "condiciones": {}}
    aprobado = True

    # 1. Superar el umbral mínimo de negocio.
    if cfg.get("superar_umbral_minimo", True):
        ok = valor_cand >= umbral
        detalle["condiciones"]["supera_umbral_minimo"] = {
            "umbral": umbral, "cumple": ok}
        aprobado = aprobado and ok

    # 2. No ser peor que el champion (dentro de la tolerancia).
    tol = cfg.get("tolerancia_vs_champion", 0.0)
    if metricas_champion is not None:
        valor_champ = float(metricas_champion.get(metrica, 0.0))
        ok = valor_cand >= valor_champ - tol
        detalle["condiciones"]["no_peor_que_champion"] = {
            "champion": round(valor_champ, 4), "tolerancia": tol, "cumple": ok}
        aprobado = aprobado and ok
    else:
        detalle["condiciones"]["no_peor_que_champion"] = {
            "champion": None, "cumple": True, "nota": "no hay champion previo"}

    # 3. Recall mínimo (opcional).
    if "recall_minimo" in cfg:
        rec = float(metricas_candidato.get("recall", 0.0))
        ok = rec >= cfg["recall_minimo"]
        detalle["condiciones"]["recall_minimo"] = {
            "minimo": cfg["recall_minimo"], "valor": round(rec, 4), "cumple": ok}
        aprobado = aprobado and ok

    detalle["aprobado"] = aprobado
    return aprobado, detalle
