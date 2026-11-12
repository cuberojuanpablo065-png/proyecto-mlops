# Descripción

<!-- ¿Qué cambia este PR y por qué? -->

## Lista de verificación del revisor (A.2)

- [ ] El CI pasa (estilo, pruebas con cobertura ≥ 60%, análisis estático, build).
- [ ] No se suben datos, modelos ni credenciales al repositorio.
- [ ] Las entradas de la API siguen validadas (Pydantic `extra="forbid"`).
- [ ] Si cambia el modelo o los datos, se actualizaron las métricas y el README.
- [ ] Los cambios en configuración (`params.yaml`, `config/*.yaml`) están justificados.
- [ ] No se usa `continue-on-error: true` para ocultar fallos.

## Tipo de cambio

- [ ] Corrección (fix)
- [ ] Nueva funcionalidad (feat)
- [ ] Infraestructura / CI-CD (build/ci)
- [ ] Documentación (docs)
