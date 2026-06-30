# Protocolo TTC — Marco de Tres Expertos

> Mandato 9. Aplicar en TODA tarea spec-crítica: eliminaciones, migraciones, cambios estructurales.

## Los Tres Marcos

**Marco 1 — Senior Software Engineer** (lo que existe en disco vs el spec)
```
grep -rn "nombre_archivo" --include="*.py"
ls -la directorio/
python -m pytest tests/ --no-cov -q  → baseline
```
Pregunta clave: ¿El spec describe lo que existe hoy?

**Marco 2 — Analista de Datos** (linaje y contaminación)
```
pre-2026-05-28: contaminado (surface=0%, match_id="tennis")
post-2026-05-28: datos limpios
JSONs únicos validados → conservar | artefactos regenerables → eliminar
```
Pregunta clave: ¿Este dato, consumido accidentalmente, daña el P&L?

**Marco 3 — Arquitecto de Software** (decisiones tripartitas)

| Decisión | Condición |
|---|---|
| **ELIMINAR** | Sin importadores activos + sin valor futuro único |
| **SUSPENDER** | Desconectado del pipeline, válido para el futuro |
| **MANTENER** | Función activa o valor diagnóstico irreemplazable |

Pregunta clave: ¿Está en un stack paralelo (isla) o acoplado al pipeline S1-S8?

## Checklist
```
[ ] SE-1: grep confirma 0 importadores
[ ] SE-2: ls -la confirma contenido real
[ ] SE-3: pytest baseline guardado
[ ] DA-1: datos pre/post Nodo-03 evaluados
[ ] DA-2: impacto en P&L si se carga accidentalmente
[ ] DA-3: regenerable vs único evaluado
[ ] ARQ-1: clasificado ELIMINAR/SUSPENDER/MANTENER
[ ] ARQ-2: stack paralelo vs pipeline S1-S8 identificado
[ ] ARQ-3: razonamiento documentado en Nodo
[ ] FINAL: pytest confirma baseline mantenido
```

Cuándo NO aplicar: ediciones puntuales de código, lectura para entender, tasks de producción.
