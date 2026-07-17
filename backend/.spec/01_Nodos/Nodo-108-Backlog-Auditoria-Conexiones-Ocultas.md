# Nodo-108 — Backlog post-Sprints: correcciones de auditoría restantes + conexiones ocultas

> **Wikilinks:** [[Nodo-90-Auditoria-Fable-Nodo89]] | [[Nodo-92-Sprint1-Implementado]] | [[Nodo-95-Sprint4-PatternRecognition]] | [[Nodo-107-Riesgo-Agregado-Motor-Reconciliacion]] | [[Nodo-67]] (conexiones C1-C5) | [[Nodo-104-Graphify-Faceted-Filter-Wikilinks-Audit]]
> **Fecha:** 2026-07-17 | **Autor:** Fable 5 | **Contexto:** GRAPH_REPORT.md 2026-07-17 (4438 nodos, 5697 edges, commit `9d3a768`) + verificación de estado Sprints 1-4 (Nodo-92→95, tests 1827→1945)
> **Qué es:** lo que QUEDA de la auditoría Nodo-90 tras los 4 sprints implementados, más los hallazgos nuevos de esta sesión. Cada ítem es implementable por Sonnet sin preguntas.

---

## §1. HALLAZGOS NUEVOS (sesión 2026-07-17)

### B108-01 — Colisión de nombre: DOS archivos Nodo-100
`Nodo-100-Taxonomia-Estrategias-Generacion-Combos.md` y `Nodo-100-Triple-Convergencia-Live.md` coexisten. Rompe la unicidad del índice SDD, los wikilinks `[[Nodo-100...]]` y `check_contradictions.py` (compara "últimos 10 nodos" por número).
**Fix:** renombrar `Nodo-100-Triple-Convergencia-Live.md` → `Nodo-100B-Triple-Convergencia-Live.md` (excepción mínima a la inmutabilidad: cambia el nombre de archivo, no el contenido; añadir línea al tope: "Renombrado de Nodo-100 por colisión — ver Nodo-108 B108-01"), actualizar los wikilinks entrantes (`grep -rn "Nodo-100-Triple"`), correr `python3 scripts/rebuild_nodos_index.py` y `graphify update .`. Test: `rebuild_nodos_index` detecta números duplicados y falla ruidosamente (nueva validación, REGLA-T53).

### B108-02 — O-01 sin registro escrito → resuelto vía [[Nodo-107]] D107-01 (no duplicar aquí).

### B108-03 — El grafo confirma la fragmentación de name-matching (C2 de Nodo-67)
GRAPH_REPORT: `_normalize_name` aparece como hub propio; comunidades separadas para `PlayerRegistry`, `RankingManager.normalize_name`, `_normalize_player_name_for_prof`, `shadow_book._parse_apellido`, `kambi _parse_nombre`. **5+ normalizaciones vivas.** Es el patrón estructural del Nodo-86 §4 (degradación en los bordes) aún abierto.
**Fix (C2, ejecutable ya):** todos delegan a `core/player_registry.normalize_player_name()` — empezar por los 2 de mayor riesgo de dinero: `betslip_registrar`/`shadow_book` (settle) y `kambi_tennis`. Un módulo por commit, suite verde entre cada uno. Cerrar el TODO F0-DEUDA de `core/player_registry.py:29-30` (RankingManager delega). Tests: mismos inputs raros (iniciales múltiples, guiones, diacríticos) → misma salida en todos los call-sites migrados.

## §2. RESTANTE DE NODO-90 (Sprint 5 — ahora con precondiciones cumplidas)

### B108-04 — D90-11: N28F2 por tier — YA HAY DATOS ACUMULANDO
H89-01 (CAPA2) y H89-02 (ELO_DOMINANCE) acumulan desde 2026-07-13 (Nodo-92). **Acción:** Sonnet añade al checklist semanal la lectura de ambos segmentos en `shadow_book --report`; cuando n≥30 → decisión de recalibración con SPRT (`hypothesis_tracker.sprt_verdict`). PROHIBIDO tocar el threshold antes.

### B108-05 — D90-08: OddsAggregator staged — fase 0 es UNA sesión de curl
Verificación empírica de endpoints ANTES de codificar (Nodo-90 §3 R3): probar el offering-key Kambi de betcris/luckia/sportium con el mismo `listView/tennis.json` de `betplay_combo_builder.py:113`. Si ≥1 confirma → cliente Kambi parametrizado por offering (refactor de `fetch_kambi_outcomes()` a `fetch_kambi_outcomes(offering='betplay')`); si 0 confirman → se archiva la D89-09 con evidencia. Time-box: 1 sesión.

### B108-06 — D90-06: RealTime Intelligence MVP observacional
Solo cuando B108-03/04 estén cerrados. Weather primero (open-meteo por requests, sin key, venues del día desde el zita file) → campo observacional `weather_flag` en edge_report + hipótesis antes de cualquier ajuste de p_modelo. Injury/news: DIFERIDO (superficie de scraping frágil, R3).

## §3. CONEXIONES OCULTAS (Nodo-67 C1-C5 — estado y orden)

| C | Conexión | Estado | Acción |
|---|---|---|---|
| C2 | Name-matching unificado | ABIERTO | B108-03 (este nodo) — LA prioridad |
| C3 | "Campeón reciente" contada 4× (BONUS/GCS/tier_mismatch/H77-03) | ABIERTO | Diseño: un solo campo `campeon_signal` con dueño en rivalry_analyzer; consumidores leen, no recalculan. Spec corta antes de código (nodo propio) |
| C1/C4/C5 | n8n/dashboard/docker/tamp | Parcial (I3 governor-JSON ya en `combo_governor.py:103-123`) | Después del veto D107-04, conectar exit-code del governor al flujo n8n (notificación, no bloqueo remoto) |

## §4. ORDEN MAESTRO (actualizado 2026-07-17 — cubre los 4 nodos: 107, 108, 109, 110)
1. **B108-01** (colisión Nodo-100 — 15 min, desbloquea el índice)
2. **[[Nodo-107]] completo** (S107-A→E; S107-F YA COMPLETADO — H107-01 registrada y ACUMULANDO en `preregistered_hypotheses.json`, Sonnet solo la lee). Es prerequisito de los dos siguientes: D107-02/03 dan la cobertura agregada y D107-04 el exit-code del governor.
3. **[[Nodo-110-Modo-Operador-Favoritos-Compuestos]]** (dolor #1 del operador; requiere su OK para D110-01 y H110-01 ANTES de codificar; el governor ya con veto lo cuenta como estrategia #13 — extiende la matriz de D107-02 a 13/13)
4. **[[Nodo-109-Live-Trading-Desk-Dashboard]]** (DEPENDE de D107-04: el panel P4 lee el exit-code PASS/WARN/BLOCK del governor — no empezar antes; y ya muestra la estrategia #13 en P6)
5. B108-03 / C2 (name-matching, 2 call-sites de dinero primero)
6. B108-04 (solo lectura semanal de segmentos H89-*)
7. B108-05 (curl session, time-boxed) → C3 spec → B108-06

Cada paso: GIT-FIRST, baseline pytest (≥1945), commit propio, evidencia de ejecución en el nodo de cierre (patrón Nodo-87/92).
