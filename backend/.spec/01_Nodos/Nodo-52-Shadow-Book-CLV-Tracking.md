# Nodo-52: Shadow Book — Libro Sombra con Validación Post-Hoc y CLV Tracking

> **Wikilinks:** [[Nodo-51-Plan-Estrategico-Data-Layer-Torneo]] | [[Nodo-48-FlashScore-Odds-Scraper-Testing]] | [[Nodo-44-Watchlist-Alpha-Signal]] | [[Nodo-46-Markov-Surface-Context-Discount]] | [[Nodo-21-Pesos-Diferenciados-Tier]] | [[Nodo-33-Filtro-Coinflip-Sin-H2H]]
> **Fecha de creación:** 2026-07-01
> **Estado:** 📋 ESPECIFICADO — listo para implementación por Sonnet
> **Dependencia de Nodo-51:** consume F1 (tournament_context) y F2 (provenance) si existen; puede arrancar SIN ellas con campos degradados (ver §8)

**Prioridad:** MÁXIMA — es el único componente que genera aprendizaje a costo de capital $0
**Principio rector:** Aprender y ganar son actividades separadas. El shadow book aprende; el capital solo se despliega donde el shadow book ya aprendió.
**Archivos objetivo:** `shadow_book.py` (nuevo, módulo raíz) + hook en `edge_calculator.py` + `reports/shadow_book/` (datos)

---

## 0. El Problema que Resuelve

El pipeline enfrenta una paradoja operativa documentada en la sesión 2026-07-01:

```
Gates estrictos (p≥0.55, T33-01, FIX-3) → casi 0 picks desplegables en transición de superficie
Sin picks desplegados → sin n para validar D44-03 (WAS n≥30), D46-07 (calibrar constantes), D44-05
Sin n → presión de bajar umbrales para "generar volumen"
Bajar umbrales con capital real → pagar el aprendizaje con el bankroll
```

**La premisa falsa:** que acumular n requiere desplegar capital. No lo requiere. Requiere **registrar predicciones con timestamp inmutable ANTES del partido y validarlas post-hoc**. El Nodo-48 (`--flashscore-only`) ya construyó la mitad de la infraestructura: cuotas y resultados post-match de la jornada completa. El Nodo-52 construye la otra mitad: el registro pre-match y el motor de settlement.

### Qué produce

1. **n masivo y gratis:** ~50-100 observaciones/semana por segmento (vs 0-5 desplegando solo lo que pasa gates)
2. **CLV tracking:** el indicador de edge que converge con n=50 en vez de n=500 (ver §4)
3. **Criterios de graduación objetivos:** qué segmento pasa de shadow → stake fijo → Kelly, con reglas congeladas antes de ver los datos (anti p-hacking, hereda MM-5 de Nodo-51)
4. **Resolución empírica de las preguntas abiertas:** ¿umbral 0.55 vs 0.52 en qualifiers? ¿WAS sostiene >55%? ¿el surface discount mejora Brier score? — todas contestables con el mismo dataset

---

## 1. Arquitectura — Tres Momentos, Un Registro

```
MOMENTO 1 — LOG (pre-match, automático):
  PASO 3 (edge_calculator) termina → hook escribe TODOS los picks al shadow book:
    picks aprobados + watchlist completa + NO_DATA (si F2 existe)
  → reports/shadow_book/sb_YYYY-MM-DD.jsonl  (append-only, un JSON por línea)

MOMENTO 2 — CLOSING SNAPSHOT (opcional, cerca del inicio):
  Re-correr PASO 1 Kambi ~15-30 min antes del primer partido de la sesión
  → captura cuota_cierre_kambi para los picks ya loggeados (match por canonical_id/match_key)
  → si no se corre: FlashScore final odds del settlement actúan como proxy (§4.2)

MOMENTO 3 — SETTLE (post-match, día siguiente):
  python3 shadow_book.py --settle 2026-07-01
  → invoca extract_matches_flashscore_only() (Nodo-48) para la fecha
  → cruza resultados + cuotas finales FlashScore contra los registros abiertos
  → escribe won/lost/void + cuota_cierre + CLV en cada registro
  → python3 shadow_book.py --report  → métricas por segmento
```

**Regla de inmutabilidad:** los registros del Momento 1 NUNCA se editan retroactivamente en sus campos de predicción (p_modelo, cuota_tomada, señales, timestamp). El settlement solo AÑADE campos de resolución. Un shadow book editable a posteriori no vale nada como evidencia — esta regla es la que le da rango de cohorte (MM-5).

---

## 2. Esquema del Registro (JSONL)

```json
{
  "sb_id": "2026-07-01_wimbledon-q_minnen-ferro_ML-minnen",
  "logged_at": "2026-07-01T04:12:33-05:00",

  "match": {
    "match_key": "minnen_ferro_2026-07-01",
    "match_id_flashscore": "abc123",
    "p1": "Greet Minnen", "p2": "Fiona Ferro",
    "torneo_nombre": "Wimbledon Qualifying",
    "tier": "grand_slam", "superficie": "grass",
    "es_qualifying": true,
    "season_transition_flag": true
  },

  "prediccion": {
    "pick": "Greet Minnen",
    "p_modelo": 0.712,
    "cuota_tomada": 1.29,
    "cuota_provenance": "kambi_live",
    "edge": 0.062,
    "kelly_frac": 0.041
  },

  "estado_pipeline": {
    "status": "APROBADO",
    "gate_bloqueante": null,
    "n_h2h": 4, "n_axes": 3,
    "history_provenance": {"p1": "ninja_api", "p2": "thf_cache"},
    "markov": {"pick_estado": "HOT", "pick_conf": 0.66,
               "rival_estado": "COLD", "rival_conf": 0.71,
               "surface_overlap_pick": 0.8, "surface_overlap_rival": 0.2},
    "senales_was": ["PCRS"],
    "games_zona": "DOMINANTE"
  },

  "resolucion": {
    "settled_at": "2026-07-02T09:00:00-05:00",
    "resultado": "WON",
    "cuota_cierre": 1.22,
    "cuota_cierre_provenance": "flashscore_ref",
    "clv_pct": 5.74,
    "pnl_flat_1u": 0.29
  }
}
```

**Valores de `estado_pipeline.status`:** `APROBADO` | `WATCHLIST` (con `gate_bloqueante`: `T33-01`, `FIX-3`, `P_MODELO_MIN`, `REGLA-G6`...) | `NO_DATA` (F2 de Nodo-51). Los tres se registran — la watchlist es donde vive el n que los gates hoy desperdician.

**El `sb_id` es determinista** (fecha + torneo + match + mercado) para que re-correr el PASO 3 el mismo día no duplique registros (upsert por sb_id, conservando el `logged_at` original).

---

## 3. Segmentos Pre-Registrados — Congelados ANTES de Ver Datos

Para evitar p-hacking (MM-5 de Nodo-51), los cortes de análisis se definen HOY y no se modifican hasta que cada celda alcance su n de parada. Cortes:

| Dimensión | Valores |
|---|---|
| tier | grand_slam / atp-wta / challenger / itf |
| es_qualifying | true / false |
| status | APROBADO / WATCHLIST(por gate) |
| banda p_modelo | [0.50-0.55) / [0.55-0.60) / [0.60+ |
| banda cuota | [1.2-1.7) / [1.7-2.5) / [2.5-4.0) / [4.0+ |
| señal | WAS / PCRS / GAMES / ninguna |
| season_transition_flag | true / false |

**Hipótesis pre-registradas (con n de parada y métrica de éxito, congeladas 2026-07-01):**

| ID | Hipótesis | Métrica | n parada | Éxito si |
|---|---|---|---|---|
| H52-01 | WAS (edge≥10%, cuota≥2.0, señal Markov) tiene hit% > breakeven | hit% + IC Wilson 95% | 30 | límite inferior IC > 1/cuota_media |
| H52-02 | Qualifiers GS/WTA con p∈[0.52,0.55) rinden ≥ que [0.55,0.60) en cuadro principal | ROI flat 1u | 50 c/u | ROI_qualy ≥ ROI_main − 2pts |
| H52-03 | Picks con CLV+ mediano tienen ROI positivo a largo plazo | correlación CLV↔PnL | 50 | CLV mediano > 0 en segmento graduable |
| H52-04 | Surface discount (Nodo-46 ON) mejora Brier vs OFF | Brier score A/B | 5 casos atribuibles | Brier_ON < Brier_OFF |
| H52-05 | T33-01 (n_h2h=0, p<0.55) sigue siendo coin-flip con datos Playwright (post Nodo-49) | hit% | 30 | re-evaluar gate solo si IC excluye 50% |

Cerrada una hipótesis, se puede registrar la siguiente. Modificar umbrales a mitad de muestra invalida la hipótesis — se documenta y se reinicia el contador.

---

## 4. CLV — Closing Line Value

### 4.1 Definición y por qué es la métrica principal

```
CLV% = (cuota_tomada / cuota_cierre − 1) × 100

Ejemplo: tomaste Jorge @4.90, la línea cerró @3.80 → CLV = +28.9%
         El mercado terminó dándote la razón ANTES de que se jugara el punto.
```

El resultado de un partido es 1 bit de información ruidosa; la línea de cierre es el consenso de todo el dinero informado del mercado. Si tus picks sistemáticamente cierran a cuota menor que la tomada, tenés edge real **aunque estés en racha perdedora**; si cierran igual o mayor, no lo tenés **aunque estés en racha ganadora**. Por eso CLV separa edge de varianza con n=50 donde el hit% necesita n=500.

**Métrica por segmento:** CLV mediano (no medio — robusto a outliers) + % de picks con CLV>0. Un segmento con CLV mediano > +2% es candidato a graduación aunque su hit% de corto plazo sea mediocre.

### 4.2 Fuentes de cuota de cierre, en orden de preferencia

| Fuente | Provenance | Calidad | Cuándo |
|---|---|---|---|
| Kambi re-snapshot pre-inicio | `kambi_close` | ALTA — mismo bookmaker que cuota_tomada | Momento 2 corrió |
| FlashScore final odds (bookmaker 523) | `flashscore_ref` | MEDIA — bookmaker distinto, sesgo sistemático estimable | Siempre disponible (Nodo-48) |

El CLV con `flashscore_ref` compara bookmakers distintos → tiene un offset. Manejo: calcular el offset mediano Kambi↔FlashScore en los partidos donde existen AMBAS fuentes, y reportar el CLV flashscore_ref ajustado por ese offset. Mientras no haya n para el offset, reportar ambas provenances por separado y NUNCA mezclarlas en la misma métrica.

---

## 5. Reporte — `shadow_book.py --report`

Salida por segmento pre-registrado:

```
════════ SHADOW BOOK — 2026-07-01 → 2026-07-28 (n=214) ════════
SEGMENTO: qualifying=true | tier=grand_slam | status=WATCHLIST(T33-01)
  n=34  hit%=58.8  IC95=[42.2, 73.6]  breakeven=48.1
  ROI flat 1u: +11.2%   CLV mediano: +3.1% (flashscore_ref, ajustado)
  → H52-05: IC aún incluye 50% — CONTINUAR, no tocar gate

GRADUACIÓN:
  ✅ Ningún segmento cumple criterios todavía (mínimo n=30 + IC + CLV)
  ⏳ Más cercano: WAS edge≥10% (n=19/30)
════════════════════════════════════════════════════════════════
```

Métricas: hit% con **IC de Wilson** (no normal — mejor con n pequeño), ROI simulado flat 1 unidad, CLV mediano por provenance, Brier score del p_modelo (calibración), y drawdown máximo simulado del segmento (para dimensionar el stake real futuro).

---

## 6. Criterios de Graduación — Shadow → Capital

Un segmento se gradúa SOLO si cumple los tres simultáneamente:

```
1. n ≥ 30 registros settled en el segmento
2. Límite INFERIOR del IC Wilson 95% del hit% > breakeven del segmento (1/cuota_media)
3. CLV mediano > 0 con la misma provenance en todo el segmento
```

**Escalera de despliegue (sin saltos):**

```
NIVEL 0  Shadow book            capital $0        ← TODO empieza aquí
NIVEL 1  Stake fijo mínimo      1 unidad flat     ← al graduarse; promos WAS viven aquí (REGLA-WAS-1)
NIVEL 2  Kelly fraccional (¼)   tras +30 settled en NIVEL 1 manteniendo criterios
NIVEL 3  Kelly del pipeline     gates actuales intactos — este nodo NO los relaja
```

**Regla de descenso:** si un segmento graduado cae bajo criterios con la muestra acumulada (no con una mala semana — con el IC completo), desciende un nivel. Sin excepciones ni "una sesión más".

**Lo que este nodo NO hace:** no baja p≥0.55, no desactiva T33-01/FIX-3, no autoriza parlays largas. La respuesta a "el pipeline no genera apuestas" es este nodo — no relajar los gates.

---

## 7. Implementación — Instrucciones para Sonnet

### Archivos

**`shadow_book.py` (nuevo, ~300 líneas):**
- `log_picks(edge_report: dict, session_meta: dict)` — Momento 1. Upsert por sb_id a `reports/shadow_book/sb_YYYY-MM-DD.jsonl`
- `settle(fecha: str)` — Momento 3. Llama `extract_matches_flashscore_only()` para la fecha, cruza por match_key (reusar `_build_match_key()` de Nodo-48), resuelve WON/LOST/VOID, calcula CLV, añade bloque `resolucion`
- `report(desde: str, hasta: str)` — métricas §5. Wilson CI: implementación directa, sin dependencia nueva
- CLI: `--settle FECHA`, `--report [--desde --hasta]`, `--close-snapshot` (Momento 2)

**Hook en `edge_calculator.py` (~10 líneas):** al final del cálculo, si `--shadow-log` (default ON), llamar `shadow_book.log_picks()` con picks + watchlist completa. El hook NUNCA puede romper el PASO 3: try/except con warning.

**VOID handling:** partido cancelado/retiro (como Ortenzi en las apuestas del 23-jun) → `resultado=VOID`, excluido de hit% y ROI, contabilizado aparte.

### Tests — `tests/test_nodo52.py`

| Test | Qué prueba |
|---|---|
| T52-01 | log_picks escribe APROBADO + WATCHLIST con gate_bloqueante correcto |
| T52-02 | sb_id determinista: doble corrida del mismo día no duplica (upsert conserva logged_at original) |
| T52-03 | settle cruza match_key FlashScore y marca WON/LOST correcto (fixture con 3 partidos) |
| T52-04 | CLV: cuota_tomada=4.90, cierre=3.80 → clv_pct=28.9 |
| T52-05 | VOID excluido de hit% y ROI |
| T52-06 | Wilson CI: n=34, hits=20 → IC ≈ [42.2, 73.6] |
| T52-07 | Campos de `prediccion` y `estado_pipeline` inmutables en settle (solo se añade `resolucion`) |
| T52-08 | Hook con edge_report malformado → warning, PASO 3 no crashea |

### Orden

```
1. Esquema + log_picks + T52-01/02/07/08
2. Hook en edge_calculator (--shadow-log)
3. settle + CLV + T52-03/04/05
4. report + Wilson + T52-06
5. Correr en producción HOY MISMO — cada día sin shadow book es n perdido
Baseline: 1438 tests siguen pasando.
```

---

## 8. Degradación sin Nodo-51 F1/F2

El shadow book NO espera a las fases de Nodo-51 — arranca hoy con campos degradados y se enriquece cuando existan:

| Campo | Con F1/F2 | Sin F1/F2 (hoy) |
|---|---|---|
| `superficie`, `es_qualifying` | de tournament_context | parse best-effort de `torneo_completo` ("Clasificatorios"/"Qualifying" → true) |
| `history_provenance` | de F2 | `"unknown"` |
| `status=NO_DATA` | de F2 | no existe — n_h2h=0 queda como WATCHLIST(T33-01) |

Cuando F1/F2 lleguen, los registros nuevos llevan los campos ricos; los viejos se segmentan con lo que tienen. No se re-escriben (inmutabilidad §1).

---

## 9. Relación con Otros Nodos

| Nodo | Relación |
|---|---|
| [[Nodo-48-FlashScore-Odds-Scraper-Testing]] | Provee el motor de settlement completo (`--flashscore-only`) — Nodo-52 es su caso de uso principal |
| [[Nodo-51-Plan-Estrategico-Data-Layer-Torneo]] | Nodo-52 implementa la F5 (validación pre-registrada) de forma operativa diaria |
| [[Nodo-44-Watchlist-Alpha-Signal]] | D44-03 (WAS n≥30) se resuelve con H52-01 — la watchlist loggeada ES el dataset |
| [[Nodo-46-Markov-Surface-Context-Discount]] | H52-04 provee el A/B que D46-07 necesita para calibrar constantes |
| [[Nodo-33-Filtro-Coinflip-Sin-H2H]] | H52-05 re-valida el gate con datos post-Nodo-49 — el gate se toca solo con evidencia |
| [[Nodo-21-Pesos-Diferenciados-Tier]] | La segmentación por tier del reporte confirma o refuta dónde está la ventaja informacional |

## 10. Deuda Técnica

| ID | Tarea | Prioridad |
|---|---|---|
| D52-01 | `shadow_book.py` completo + tests T52-01→08 | MÁXIMA |
| D52-02 | Hook `--shadow-log` en edge_calculator | MÁXIMA |
| D52-03 | Momento 2 (`--close-snapshot` Kambi pre-inicio) | MEDIA |
| D52-04 | Offset Kambi↔FlashScore para CLV ajustado (necesita n de partidos con ambas fuentes) | MEDIA |
| D52-05 | Sección shadow book en pipeline_tracker | BAJA |
