# Constitución — Mandatos No Negociables

> **Wikilinks:** [[Pipeline-Arquitectura]] | [[Grafo-Dependencias-Datos]] | [[Sprint-Pipeline]] | [[Fuentes-Datos]] | [[Nodo-07-Strangler-Fig]] | [[Nodo-08-File-Selection-Bug]] | [[Nodo-12-Inventario-Infraestructura-Legado]]
> 9 mandatos que ninguna sesión puede violar sin aprobación explícita del usuario.

---

## Índice del Vault (.spec/)

```
00_Constitution/
  Mandatos-No-Negociables.md    ← este archivo

01_Nodos/
  Nodo-01-Edge-Calculator.md    Kelly-KL + edge real           [POR CONSTRUIR]
  Nodo-02-Markov-Changepoint.md PELT + estados HOT/COLD        [POR CONSTRUIR]
  Nodo-03-Scraper-Fix.md        3 bugs críticos en URL scraper [ROTO — fix primero]
  Nodo-04-Dataset-Fix.md        KNN shape + SmartLogger.error  [ROTO]
  Nodo-05-Validacion-API.md     dc_1_{event_id} post-partido   [POR CONSTRUIR]
  Nodo-06-Erdos-Graph.md        Grafo transitivo de victorias  [ENHANCEMENT]

02_Sources/
  Fuentes-Datos.md              FlashScore Playwright + API + rankings

03_Atlas/
  Pipeline-Arquitectura.md      Diagrama flujo + health metrics
  Grafo-Dependencias-Datos.md   8 Señales + productores + consumidores

04_Pipeline/
  Sprint-Pipeline.md            Motor de construcción — 7 fases, 31 tareas T-coded
```

---

---

## Mandato 1: Métrica de éxito = P&L, nunca accuracy

El sistema existe para ganar dinero. La accuracy del modelo es un proxy — puede ser 52% con edge real y ganar, o 70% sin edge y perder. Toda decisión de arquitectura se evalúa contra P&L, no contra accuracy.

## Mandato 2: Solo apostar con edge calculado

`Edge = P_modelo - P_implícita_bookmaker`

No apostar si edge < 5%. No apostar si Kelly-KL < 2%. No apostar si n_historial_superficie < 10 partidos.

## Mandato 3: Kelly-KL, nunca Kelly clásico

Kelly clásico asume probabilidades exactas. Con accuracy variable, usar siempre:
```
f*_KL = f*_clásico × exp(-λ × KL(P_modelo || P_histórica))
```
Esto protege contra la ruina cuando el modelo diverge de la historia real.

## Mandato 4: Datos limpios antes de entrenar

No entrenar el modelo hasta que `surface_specialization > 0%` en todos los partidos. Los datos pre-2026-05-28 tienen feature contaminada — no usar para inferir edge.

## Mandato 5: Spec-Driven Development

Ninguna línea de código nueva sin Nodo aprobado en `.spec/01_Nodos/`. El CLAUDE.md es la fuente de verdad. Obsidian es el editor del vault.

## Mandato 6: Tests antes que código

Cualquier bug documentado requiere un test que falle primero. El fix hace pasar el test. `python -m pytest tests/ --no-cov -q` debe dar **767 passed** antes de cualquier commit. (2026-05-29: Nodo-09 +2 tests; Nodo-08 +5 tests; D-09 SmartLogger +15 tests en `test_utils_logger.py`; Nodo-07 prep +51 tests en `test_sequential_h2h_extractor.py`)

## Mandato 7: Fuente única de verdad para predicciones

La predicción vive en `ranking_analysis.prediction.favored_player`. No en `prediccion_ganador` (top-level, siempre None). Todo código que consuma predicciones debe leer el path correcto.

## Mandato 8: Pipeline = API + Playwright (híbrido)

- FlashScore Ninja API (`dc_1_{event_id}`): resultados en tiempo real, labels post-partido
- Playwright: H2H histórico profundo (endpoints H2H de la API dan 404 para tenis)
- No reemplazar Playwright completamente — solo complementar con API donde sea posible

## Mandato 9: TTC + Marco de Tres Expertos para tareas Spec-críticas

Toda tarea de spec que implique eliminación, migración o cambios estructurales **debe** ejecutarse activando Test-time Compute con tres marcos mentales simultáneos antes de tocar disco:

**Senior Software Engineer** — detecta lo que el spec no actualizó:
`grep` antes de borrar. Verificar imports activos. Evaluar blast radius. Comprobar reversibilidad git. Buscar discrepancias entre spec y disco.

**Analista de Datos** — detecta contaminación y linaje:
¿pre-Nodo-03 (surface=0%, match_id="tennis") o post-fix? ¿Qué dato, cargado por error, contamina el modelo ML? ¿Qué es regenerable (log, artifact) vs único (label validada)?

**Arquitecto de Software** — clasifica sin binarios:
Nunca solo ELIMINAR/MANTENER. Siempre ELIMINAR / SUSPENDER / MANTENER con razón y condición. Detectar stacks paralelos desconectados del pipeline S1-S8. Evaluar acoplamiento e impacto en P&L.

**Evidencia (Nodo-12, 2026-05-30):**
- SE detectó que `services/` no estaba vacío — spec decía "eliminar si 0 bytes", había `selenium_config.py` → decisión cambió a SUSPENDER
- ARQ detectó que `testpaths=tests` en `pytest.ini` → `screenshots/` nunca ejecutado → dead code estructural → ELIMINAR con confianza total
- DA detectó que `reports/ginput/` (Ago 2025) era veneno ML pre-Nodo-03 aunque estuviera en `reports/` (directorio "activo")

Ver protocolo completo en `CLAUDE.md §10`.
