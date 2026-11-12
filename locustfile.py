"""E.4 - Prueba de carga con Locust.

Ejecuta carga contra el servicio del modelo (directo o a través de Nginx).

Dos niveles de usuarios (ejemplos):
    # Nivel 1 (carga ligera):
    locust -f locustfile.py --headless -u 20 -r 5 -t 1m --host http://localhost:8080 \
        --html reports/locust_nivel1.html
    # Nivel 2 (carga mayor):
    locust -f locustfile.py --headless -u 100 -r 10 -t 2m --host http://localhost:8080 \
        --html reports/locust_nivel2.html

Durante la prueba, detener una réplica (docker compose ... stop model-b) y
verificar que el servicio sigue respondiendo (redundancia).
"""
from __future__ import annotations

import random

from locust import HttpUser, between, task

ESTADOS = ["MH", "DL", "KL", "TN", "UP", "KA"]
CATEGORIAS = ["murder", "theft_burglary", "cyber", "fraud_economic"]
SCOPES = ["ipc", "cyber", "women"]


class UsuarioAPI(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(3)
    def predecir(self):
        payload = {
            "state_code": random.choice(ESTADOS),
            "scope": random.choice(SCOPES),
            "crime_category": random.choice(CATEGORIAS),
            "year": 2022,
            "lag1": random.uniform(50, 1000),
            "lag2": random.uniform(50, 1000),
        }
        self.client.post("/predict", json=payload, name="/predict")

    @task(1)
    def salud(self):
        self.client.get("/health", name="/health")
