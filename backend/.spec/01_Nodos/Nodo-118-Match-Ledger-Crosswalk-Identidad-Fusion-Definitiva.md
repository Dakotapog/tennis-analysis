---
estado: activo
---
# Nodo-118 — MATCH LEDGER + CROSSWALK: la solución definitiva al problema de datos (no es scraping, es identidad)

> **Wikilinks:** [[Nodo-117-Auditoria-Scraping-Rankings-Cobertura-H2H]] | [[Nodo-80-Kambi-Name-Matching]] | [[Nodo-82-Kambi-Match-ID-Structural]] | [[Nodo-51-Plan-Estrategico-Data-Layer-Torneo]] (player_registry) | [[Nodo-72-Phantom-Identity-Guard]] (phantom identity) | [[Nodo-48-FlashScore-Odds-Scraper-Testing]] | [[Nodo-110-Modo-Operador-Favoritos-Compuestos]] | [[Nodo-89-Sistema-Inteligencia-Integral]]
> **Fecha:** 2026-07-18 | **Autor:** Fable 5 (análisis de raíz + spec) | **Implementa:** Sonnet, por fases, SIN presión de sprint
> **Responde:** la pregunta de D117-03 (opciones 1/2/3) — la respuesta es **Opción 1, pero con la arquitectura correcta**: el merge no se hace adivinando nombres inline; se hace vía ledger + crosswalk persistente.

---

## §0. POR QUÉ 3 AÑOS SIN RESOLVERSE — el diagnóstico que faltaba

**El problema nunca fue scraping. Los scrapers FUNCIONAN** (Playwright trae ~130 partidos con match_ids; Kambi trae ~66 con cuotas; API Ninja trae H2H). El problema es que son **tres espacios de identidad distintos** ("Badosa P." ≠ "P. Badosa" ≠ "Paula Badosa") y durante 3 años se intentó unirlos con **string-matching efímero, por día, sin memoria**. Cada sesión de Sonnet resolvía el síntoma del día (normalizar un nombre, elegir "el mejor archivo", un fallback más) y al día siguiente el join volvía a fallar con jugadores nuevos — porque **ninguna resolución de identidad se persistía**.

**Tres errores estructurales acumulados:**
1. **SELECCIÓN en vez de FUSIÓN:** `select_best_json_file()` (D117-02) elige UN archivo — el sistema literalmente descarta una fuente entera cada día. Playwright ve 130, API ve 66 con cuotas, y el pipeline se queda con uno de los dos.
2. **JOIN por string en caliente:** cada cruce de nombres se recalcula desde cero cada día, con la misma tasa de fallo, para los mismos jugadores. Un grinder de ITF juega cada semana — y cada semana el sistema vuelve a fallar en reconocerlo.
3. **PÉRDIDA SILENCIOSA:** los ~46 partidos con cuotas reales que no cruzan no aparecen en NINGÚN reporte. No hay lista de "no pude unir estos, por esto". Zero-Null se aplicó a los builders pero nunca a la frontera de datos.

**El activo desaprovechado:** 194 archivos zita históricos en `data/` + meses de H2H + edge_reports + `player_registry` (Nodo-51) + 4361 perfiles IRP. Cada día pasado donde un partido SÍ cruzó es una resolución de identidad verificada que se tiró a la basura. Este nodo la recupera toda.

---

## §1. ARQUITECTURA — dos piezas persistentes

### 1a. PLAYER CROSSWALK (`data/player_crosswalk.json`) — la memoria de identidad
Tabla acumulativa: `{canonical_id: {"canonical": "paula badosa", "aliases": ["badosa p.", "p. badosa", "badosa paula"], "kambi_ids": [...], "fs_slugs": [...], "last_seen": "...", "confidence": "VERIFIED|AUTO|MANUAL"}}`.
- **Se escribe UNA vez por identidad, se lee para siempre.** Cada join exitoso (auto o manual) persiste los alias de ambas fuentes.
- Extiende `core/player_registry.py` (NO módulo paralelo — Nodo-51 es la entidad canónica): nuevos métodos `add_alias(canonical, alias, source, confidence)` y `resolve_crosswalk(name) -> canonical_id | None` que consulta aliases ANTES del fuzzy.
- Crece monotónicamente: en 2-3 semanas de operación el join de jugadores recurrentes converge a ~100% determinístico (lookup exacto, cero fuzzy).

### 1b. MATCH LEDGER (`data/match_ledger_YYYY-MM-DD.json`) — la fuente única del día
Un solo artefacto por día, producido por el paso de FUSIÓN (nuevo PASO 1.5). Cada registro:
```json
{"ledger_id": "L20260718-001", "canonical_p1": "...", "canonical_p2": "...",
 "kambi_event_id": 1028401998, "fs_match_id": "abc123|null",
 "cuota1": 1.81, "cuota2": 1.87, "cuota_es_real": true,
 "torneo": "...", "tier": "...", "hora": "...",
 "h2h_status": "OK|PENDIENTE|SIN_MATCH_ID", "join_method": "CROSSWALK|SCORED|QUARANTINE",
 "sources": ["kambi","flashscore"]}
```
- **TODO el downstream (extraer_historh2h, edge_calculator, builders, desk) lee el ledger** — se acabó elegir archivos. `select_best_json_file()` queda como fallback legacy cuando no hay ledger.
- Dedupe por `kambi_event_id` y `fs_match_id` (IDs duros), NUNCA por nombre.
- Un partido visto por una sola fuente ENTRA igual (con sus campos null) — fusión es unión, no intersección. El que solo tiene cuotas va a RANKING_ONLY (D110-06); el que solo tiene match_id obtiene H2H y espera cuotas del ciclo siguiente.

---

## §2. EL JOIN — record linkage real (Fellegi-Sunter simplificado), no string equality

`fusionar_dia(kambi_matches, fs_matches, crosswalk) -> (ledger, quarantine)` — función pura, testeable.

**Paso 1 — BLOCKING (reduce el espacio):** candidatos a par solo si `|hora_kambi - hora_fs| ≤ 12h` Y misma disciplina (singles). NO bloquear por torneo-string (los nombres de torneo también divergen entre fuentes; el torneo SUMA score, no filtra).

**Paso 2 — SCORE por par de partidos (0-100):**
| Componente | Peso | Regla |
|---|---|---|
| Jugador A | 35 | crosswalk hit exacto=35; apellido exacto + inicial compatible ("P. Badosa" vs "Badosa Paula" → apellido "badosa" + inicial "p" coinciden)=30; solo apellido exacto=20; fuzzy token-overlap ≥0.8=12 |
| Jugador B | 35 | ídem |
| Torneo | 15 | token-overlap ≥0.5 entre nombres de torneo=15; país/ciudad coincide=8 |
| Hora | 15 | Δ≤2h=15; Δ≤6h=8; Δ≤12h=3 |

Apellidos compuestos y invertidos: normalizar con `normalize_player_name()` existente + comparar CONJUNTOS de tokens (no orden). Inicial compatible = la inicial de una fuente ∈ iniciales de los tokens de la otra.

**Paso 3 — DECISIÓN de tres zonas (lo que elimina el riesgo de la Opción 1 que preocupaba a Sonnet):**
- `score ≥ 75` → **AUTO-JOIN** + persistir aliases nuevos en crosswalk (confidence=AUTO). Duplicado imposible: cada kambi_event_id y fs_match_id se consume una sola vez (asignación greedy por score desc; empates → cuarentena).
- `55 ≤ score < 75` → **CUARENTENA** (`data/join_quarantine_FECHA.json`): el par, su score, y el componente que falló. El partido entra al ledger DOS veces marcado `join_method=QUARANTINE` — visible, jamás silencioso. Revisión humana opcional promueve a MANUAL (1 comando: `python3 scripts/resolve_quarantine.py --pair N --confirm`).
- `score < 55` → no son el mismo partido; cada uno entra al ledger como single-source.

**Homónimos (Nodo-72):** si dos candidatos del crosswalk comparten apellido+inicial, exigir score de torneo>0 para auto-join; si no, cuarentena. Phantom Guard se mantiene intacto aguas abajo.

---

## §3. BOOTSTRAP RETROACTIVO — los 3 años de datos como set de entrenamiento (la jugada nunca hecha)

`scripts/build_crosswalk_bootstrap.py` (correr UNA vez, luego semanal en cron):
1. Recorre los **194 archivos zita** + todos los H2H (glob del patrón real en `data/`) + edge_reports históricos.
2. Donde una fuente trae AMBAS formas ya unidas (los H2H enhanced tienen el nombre FS del partido Y el resultado del cruce que aquel día funcionó; los edge_reports tienen favorito_predicho ya resuelto), extrae pares (alias_kambi, alias_fs) con join implícito verificado → crosswalk con confidence=VERIFIED.
3. Cruza con rankings ATP/WTA (nombres completos oficiales) y los 4361 perfiles IRP para poblar la forma canónica.
4. Reporte final: `X identidades, Y aliases, cobertura estimada sobre los últimos 7 días: Z%`.
**Resultado esperado:** el crosswalk nace con cientos-miles de identidades el día 1 — el join del día siguiente arranca resolviendo por lookup la mayoría del circuito activo, no desde cero.

---

## §4. INTEGRACIÓN EN EL PIPELINE (resuelve D117-03 sin su riesgo)

```
FASE NOCHE (~22:00, hay TIEMPO — la ventaja del --tomorrow por fin explotada):
  PASO 1a: extraer_partidos_api.py --tomorrow          (cuotas, ~66)
  PASO 1b: extraer_URL_partidos_version2.py --tomorrow (match_ids, ~130)
  PASO 1.5: python3 match_ledger.py --build --tomorrow (FUSIÓN → ledger + quarantine)
  PASO 2: extraer_historh2h.py lee el LEDGER → pide H2H para TODO registro con fs_match_id
FASE MAÑANA (~07:00):
  PASO 1c: re-run API → refresh de cuotas sobre el ledger (update in place por kambi_event_id;
           partidos nuevos de última hora entran como single-source)
  PASO 1.5b: re-fusión incremental (los PENDIENTE de anoche reintentan join con datos frescos)
  PASO 3+: edge_calculator lee el ledger (adapter: el ledger emite el MISMO schema zita que
           edge_calculator ya consume — cero cambios en el motor de predicción)
```
La respuesta a las 3 opciones de Sonnet: **Opción 1, ejecutada como fusión con cuarentena** — el riesgo de "duplicados con cuotas inconsistentes" desaparece porque (a) dedupe por IDs duros, (b) zona ambigua va a cuarentena visible en vez de adivinarse, (c) el crosswalk hace el matching cada vez más determinístico. La Opción 2 (archivos separados) perpetúa la selección; la Opción 3 (esperar evidencia) ya tiene la evidencia: 46 partidos/día con cuotas sin analizar y 8/8 combos manuales del operador.

---

## §5. EL EMBUDO VISIBLE — Zero-Null en la frontera de datos (para siempre)

El ledger serializa `cobertura` y run_daily lo IMPRIME en cada corrida:
```
EMBUDO DE DATOS 2026-07-18:
  Universo FS:        130   Con cuotas (Kambi):  66
  Join auto:           58   Cuarentena:           5   Single-source: 73
  Con H2H:             49   Analizados (edge):   49   → APOSTAR 3 / WATCH 21
  FUGA: 8 con cuotas sin join (lista: ...) | 9 con match_id sin H2H (causa: ...)
```
**Gate:** join de la población con-cuotas < 60% → WARN con la lista nominal de no-unidos y el componente de score que falló en cada uno. KPI persistido en el ledger para trend semanal (¿el crosswalk está convergiendo? debe subir semana a semana).

---

## §6. ORDEN PARA SONNET — fases independientes, cada una con valor propio

| Fase | Entregable | Tests T53 | Gate de avance |
|---|---|---|---|
| F1 | `match_ledger.py`: `fusionar_dia()` pura + score + 3 zonas + CLI --build | ~10: score por componente; apellido invertido; inicial; homónimo→cuarentena; greedy sin duplicados; single-source entra; dedupe por event_id | 6/6 componentes de score verdes |
| F2 | Crosswalk en player_registry (`add_alias`/`resolve_crosswalk`) + persistencia | ~4: alias persiste y resuelve; VERIFIED>AUTO; no sobreescribe MANUAL | F1 verde |
| F3 | Bootstrap retroactivo (194 zita + H2H + edge_reports) | ~3: extrae par verificado de fixture histórico real; reporte de cobertura | correr y PEGAR el reporte en este nodo |
| F4 | Integración run_daily (PASO 1.5, noche/mañana) + adapter schema zita para edge_calculator | ~4: adapter emite schema idéntico; refresh cuotas actualiza in-place; PENDIENTE reintenta | F1-F3 verdes + baseline pytest intacto |
| F5 | Embudo §5 en run_daily + desk :7780 (panel DATA con la fuga nominal) | ~2 | F4 verde |

**PROHIBIDO:** joins por nombre sin pasar por `fusionar_dia()`; borrar `select_best_json_file` (fallback legacy); tocar edge_calculator más allá del adapter de entrada; auto-join bajo score 75 "porque parece obvio"; resolver cuarentenas programáticamente sin el comando explícito.

## §F2-reporte — Bootstrap retroactivo 2026-07-18

```
Fecha bootstrap: 2026-07-18 22:07
Archivos zita analizados: 194 (5 fechas con pares API+Playwright)
Edge_reports analizados: 152 (2673 picks)
Identidades canónicas: 2091 | Aliases totales: 2091 | Por confidence: VERIFIED=2091
Cobertura estimada últimos 7 días: 57.7% (2317/4014 partidos resueltos)

Detalle por fecha:
  2026-07-03: 84 auto-joins / 2 cuarentena (score=73) | 56.0% cobertura
  2026-07-06: 57 auto-joins / 1 cuarentena (score=73) | 77.0% cobertura
  2026-07-08: 73 auto-joins / 232 cuarentena (score=55) | 17.6% WARN — archivos sin match_id cruzado
  2026-07-11: 147 auto-joins / 0 cuarentena | 100.0% cobertura
  2026-07-18: 43 auto-joins / 0 cuarentena | 78.2% cobertura
  Joins totales: 404 | Aliases nuevos: 251

Nota D118-F3-01: match_id compartido = shortcut score=100 (FlashScore ID único). Implementado.
Nota D118-F3-02: hora_partido (campo Playwright) cubierto en _get_hora(). Implementado.
Nota D118-F3-03: 2026-07-08 con 232 cuarentena — ambos archivos son API (sin match_id cruzado);
  cuarentena correcta, no false negatives. Crosswalk seguirá creciendo con días nuevos.
```

**Criterio de éxito definitivo (medible, 7 días):** join de población con-cuotas ≥85% (día 1 post-bootstrap ≥60%); CERO partidos con cuotas invisibles (todos en ledger: unidos, cuarentena o single-source); el embudo impreso en cada run_daily; crosswalk creciendo (identidades día 7 > día 1). Cuando esto se cumpla, el problema de 3 años queda cerrado — no porque el scraping mejore, sino porque la identidad deja de perderse.
