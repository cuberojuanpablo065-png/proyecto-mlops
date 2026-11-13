# Guía de entrega — Git, DVC, Pull Requests y tag `v0.1.0`

Esta guía cubre la **Parte B (versionamiento)** y los pasos finales que
debés ejecutar en tu máquina y en GitHub. Comandos para **Linux / WSL**.

> Antes de empezar: instalá todo y verificá que el pipeline y las pruebas
> corren (ver `README.md`): `pip install -r requirements.txt`, luego
> `pytest` y `dvc repro`.

---

## 0. Configurar Git (una sola vez)

```bash
git config --global user.name "Tu Nombre"
git config --global user.email "cuberojuanpablo065@gmail.com"
```

---

## B.1 — Código con Git y GitHub

### 1) Inicializar y primeros commits (historial gradual)

El profesor valora un **historial que muestre avance gradual**, no un único
commit con todo. Hacé los commits en orden, uno por etapa. Se recomienda el
formato *Conventional Commits* (`feat:`, `fix:`, `docs:`, `test:`…).

> **Consejo para que se vea natural:** hacé estos commits vos mismo, de a
> poco (podés incluso repartirlos en dos sesiones: hoy el pipeline, mañana
> pruebas y versionamiento). Cambiá los mensajes por tus propias palabras si
> querés. Como configuraste tu nombre y correo en el paso 0, todos los
> commits quedan a tu nombre.

```bash
git init
git branch -M main

# estructura base
git add .gitignore .dvcignore requirements.txt pytest.ini
git commit -m "chore: estructura inicial del proyecto"

# problema de negocio y parámetros
git add config/ params.yaml
git commit -m "feat: definir problema de negocio y parámetros en config"

# ingesta y validación de datos
git add src/__init__.py src/config.py src/ingest.py src/validate.py
git commit -m "feat(datos): construir panel con rezagos y validación con pandera"

# preprocesamiento y partición
git add src/preprocess.py src/data.py
git commit -m "feat(datos): preprocesamiento sin fuga y split train/test"

# modelos base y comparación
git add src/modelos.py src/metricas.py
git commit -m "feat(modelo): baselines, regresion logistica y random forest"

# entrenamiento, optimización y tracking
git add src/train.py
git commit -m "feat(modelo): comparacion, optimizacion de hiperparametros y mlflow"

# evaluación
git add src/evaluate.py
git commit -m "feat(eval): metricas y graficos (matriz de confusion, roc, pr)"

# pruebas
git add tests/
git commit -m "test: agregar pruebas de validacion, split, modelo y serializacion"

# pipeline dvc
git add dvc.yaml
git commit -m "build: pipeline reproducible con dvc"

# documentación (nota: GUIA_ENTREGA.md es tu guía personal; podés NO subirla)
git add README.md notebooks/ data/**/.gitkeep models/.gitkeep reports/figures/.gitkeep
git commit -m "docs: readme y notebook de exploracion"
```

> Más adelante, cuando ajustes algo (por ejemplo el umbral mínimo en
> `business.yaml` tras ver tus resultados), hacé un commit pequeño extra como
> `fix: ajustar umbral minimo de f1 segun resultados` — ese tipo de commits
> pequeños hacen que el historial se vea orgánico.

Creá el repo en GitHub (vacío, sin README) y conectalo:

```bash
git remote add origin https://github.com/<tu-usuario>/<tu-repo>.git
git push -u origin main
```

### 2) Rama `develop` y dos Pull Requests

El profesor pide **al menos una rama además de `main`** y **fusionar al
menos dos Pull Requests**. Hacelo con dos mejoras pequeñas:

```bash
git checkout -b develop
git push -u origin develop
```

**PR #1 — documentación de métricas:**
```bash
git checkout -b feature/doc-metricas
# (ejemplo: agregá una sección de resultados al README con tus números)
echo "

## 9. Resultados obtenidos
Random Forest optimizado: F1 ≈ 0.66, ROC-AUC ≈ 0.76 (ver reports/)." >> README.md
git add README.md
git commit -m "docs: documentar resultados del modelo en el README"
git push -u origin feature/doc-metricas
```
En GitHub: abrí un Pull Request de `feature/doc-metricas` → `develop` y
fusionalo (**Merge pull request**).

**PR #2 — una prueba adicional:**
```bash
git checkout develop && git pull
git checkout -b feature/prueba-extra
# (ejemplo: agregá una prueba que verifique que no hay nulos en el objetivo)
cat >> tests/test_split.py <<'EOF'


def test_objetivo_sin_nulos(datos_sinteticos):
    """El objetivo no tiene valores nulos."""
    assert datos_sinteticos["target"].notna().all()
EOF
git add tests/test_split.py
git commit -m "test: verificar que el objetivo no tiene nulos"
git push -u origin feature/prueba-extra
```
En GitHub: abrí un Pull Request de `feature/prueba-extra` → `develop` y
fusionalo.

Finalmente, llevá `develop` a `main` con un último PR (o merge):
```bash
git checkout main && git pull
git merge develop
git push origin main
```

---

## B.2 — Datos con DVC

### 1) Inicializar DVC y configurar un remoto

```bash
dvc init
git add .dvc .dvcignore
git commit -m "build(dvc): inicializar DVC"

# Remoto: carpeta local (lo más simple). También sirve Google Drive o un bucket.
mkdir -p /tmp/dvc-remote
dvc remote add -d localremote /tmp/dvc-remote
git add .dvc/config
git commit -m "build(dvc): configurar remoto local para datos"
```

### 2) Versionar los datos y el pipeline

```bash
# la fuente cruda se versiona con DVC (no con Git)
dvc add data/external/crime_state_year.csv
git add data/external/crime_state_year.csv.dvc data/external/.gitkeep
git commit -m "feat(datos): versionar fuente cruda con DVC"

# ejecutar el pipeline completo (crea data/raw, processed, models, reports)
dvc repro
git add dvc.lock
git commit -m "feat(pipeline): primera corrida reproducible (datos v1, <=2019)"

dvc push        # subir datos/artefactos al remoto
```

### 3) Dos versiones de los datos (y del modelo)

La **versión 1** usa años ≤ 2019 (`anio_corte_v1` en `params.yaml`).
Para la **versión 2**, incluí los años recientes (2020–2022) y volvé a
reproducir: DVC detecta el cambio y re-entrena.

```bash
# editá params.yaml:  anio_corte_v1: 2019  ->  anio_corte_v1: 2022
dvc repro
git add params.yaml dvc.lock
git commit -m "feat(datos): panel v2 con años 2020-2022 (más registros)"
dvc push
```

Ahora el historial muestra **dos versiones de datos** y **cómo cambió el
modelo** entre ellas (comparando `reports/metrics.json` de cada commit).
Esto prepara además el *drift* del Entregable 3.

---

## B.3 — Cierre: tag `v0.1.0`

```bash
git checkout main && git pull
git merge develop            # si quedó algo en develop
git tag -a v0.1.0 -m "Entregable 1: pipeline reproducible, probado y versionado"
git push origin main --tags
```

---

## Lista de verificación antes de enviar

- [ ] El `README.md` explica cómo instalar y ejecutar el proyecto.
- [ ] `dvc repro` ejecuta el pipeline completo.
- [ ] `pytest` pasa con al menos 5 pruebas (hay 9).
- [ ] No hay datos ni credenciales en Git (`git ls-files | grep -E 'csv|joblib'` solo debe mostrar `.dvc`).
- [ ] Existen al menos 2 Pull Requests fusionados y una rama además de `main`.
- [ ] Existe el tag `v0.1.0`.
- [ ] Enviar al profesor: enlace al repo con el tag `v0.1.0`. Si el remoto de DVC no es público, incluir las instrucciones de acceso a los datos.
