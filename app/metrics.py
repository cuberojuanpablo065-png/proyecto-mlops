"""C.1 - Métricas operativas en memoria (solicitudes, errores, latencia).

Contadores simples y seguros entre hilos. Se exponen en GET /metrics.
Para producción real se usaría Prometheus (opcional, valor agregado).
"""
from __future__ import annotations

import threading
from collections import Counter

_lock = threading.Lock()


class Contadores:
    def __init__(self) -> None:
        self.solicitudes = 0
        self.errores = 0
        self.latencia_total_ms = 0.0
        self.distribucion_predicciones: Counter = Counter()  # {"sube": n, "no sube": m}

    def registrar_solicitud(self, latencia_ms: float, es_error: bool) -> None:
        with _lock:
            self.solicitudes += 1
            self.latencia_total_ms += latencia_ms
            if es_error:
                self.errores += 1

    def registrar_prediccion(self, etiqueta: str) -> None:
        with _lock:
            self.distribucion_predicciones[etiqueta] += 1

    def resumen(self) -> dict:
        with _lock:
            prom = (self.latencia_total_ms / self.solicitudes) if self.solicitudes else 0.0
            return {
                "solicitudes": self.solicitudes,
                "errores": self.errores,
                "latencia_promedio_ms": round(prom, 2),
                "distribucion_predicciones": dict(self.distribucion_predicciones),
            }


contadores = Contadores()
