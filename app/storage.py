"""Registro de predicciones en SQLite (para el monitoreo del Entregable 3).

Cada predicción se guarda con su fecha, la versión del modelo, la entrada y
el resultado. SQLite es un solo archivo, sin servidor aparte.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .settings import settings

_SQL_CREAR = """
CREATE TABLE IF NOT EXISTS predicciones (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha_utc     TEXT NOT NULL,
    modelo_version TEXT,
    entrada_json  TEXT NOT NULL,
    prediccion    INTEGER NOT NULL,
    probabilidad  REAL NOT NULL
);
"""


def _conectar() -> sqlite3.Connection:
    ruta = Path(settings.db_path)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(ruta))
    return con


def inicializar() -> None:
    with _conectar() as con:
        con.execute(_SQL_CREAR)


def guardar_prediccion(entrada: dict, prediccion: int, probabilidad: float,
                       modelo_version: str | None) -> None:
    """Inserta una predicción en la base."""
    with _conectar() as con:
        con.execute(
            "INSERT INTO predicciones "
            "(fecha_utc, modelo_version, entrada_json, prediccion, probabilidad) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                modelo_version,
                json.dumps(entrada, ensure_ascii=False, default=str),
                int(prediccion),
                float(probabilidad),
            ),
        )


def contar() -> int:
    with _conectar() as con:
        cur = con.execute("SELECT COUNT(*) FROM predicciones")
        return int(cur.fetchone()[0])
