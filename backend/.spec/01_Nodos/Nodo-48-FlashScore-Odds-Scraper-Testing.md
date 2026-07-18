# Nodo-48: FlashScore Odds Scraper — Cuotas Independientes de Kambi para Testing

> **Wikilinks:** [[Nodo-45-Temporal-History-Fallback]] | ~~~~[[Nodo-31-Ronda-Futura-H2H]]~~ _(MISSING — [[Nodo-31-Future-Match-Data-Leakage]] es diferente)_~~ _(MISSING — [[Nodo-31-Future-Match-Data-Leakage]] es diferente)_ | [[Nodo-117-Auditoria-Scraping-Rankings-Cobertura-H2H]] (B117-02: cuotas Kambi ausentes en Playwright → H2H ve 0 partidos)
> **Fecha de descubrimiento:** 2026-06-30
> **Estado:** ✅ IMPLEMENTADO 2026-06-30 — D48-BUG-01, D48-01, D48-02, D48-04 completados

**Prioridad:** MEDIA — habilita testing completo del pipeline sin depender de Kambi/Betplay
**Archivo objetivo:** `extraer_cuotas_partidos.py` (recuperado de git) → migrar lógica a `scraping/kambi_tennis.py`
**Dependencias:** Playwright (ya instalado para fallback)

---

## El Problema — Dependencia de Kambi para Testing

El pipeline actual requiere cuotas de Kambi (Betplay) para:
1. `extraer_partidos_api.py` (PASO 1) — filtra partidos con cuotas
2. `ninja_h2h_parser.py` (PASO 2) — `es_singles_cuadro_principal()` requiere `cuota1 is not None`
3. `edge_calculator.py` (PASO 3) — calcula edge = p_modelo - p_implicita

**Consecuencia:** Cuando los partidos terminan, Kambi retira las cuotas. No se puede correr el pipeline post-hoc para validar predicciones, testear nodos nuevos, ni auditar la calidad del modelo en la jornada completa.

### Evidencia — 2026-06-30

- PASO 1 con Kambi: solo 13/495 partidos (el resto ya habia terminado)
- Se generaron cuotas dummy @1.90 para testear → edge calculado no es real
- FlashScore tenia los 495 partidos con historiales completos pero sin cuotas

---

## La Solucion — ZitaScraper (ya existia, recuperado de git)

### Historia del archivo

El usuario construyo `extraer_cuotas_partidos.py` (clase `ZitaScraper`, 566 lineas) ANTES de usar AI.
Fue eliminado en commit `bac389d` (2026-05-30, Strangler Fig Fase 2).
Recuperado de git en 2026-06-30 con: `git show 23d2d91:backend/extraer_cuotas_partidos.py`

**Leccion critica:** La IA declaro este nodo BLOQUEADO por anti-bot porque uso la URL equivocada
(`d.flashscore.co` — endpoint Ninja API) en lugar de la URL del sitio real
(`www.flashscore.com/tennis/`). El codigo del usuario ya tenia la solucion correcta.

### Resultado del test 2026-06-30

```
URL:                  www.flashscore.com/tennis/
Cookie consent:       #onetrust-accept-btn-handler → ACEPTADO
Click "Odds":         Encontrado y clickeado → cuotas visibles
Elementos .event__match encontrados: 437
Partidos con jugadores parseados:    60
Partidos unicos guardados:           1  ← BUG en remove_duplicates()
Cuotas extraidas (ejemplo real):     Kudermetova 2.4 / Samsonova 1.0 ✓
Anti-bot:             NO BLOQUEA con www.flashscore.com/tennis/
```

**El scraper funciona.** El problema es solo un bug de indentacion en `remove_duplicates()`.

### Por que la IA fallo — Error documentado

| Aspecto | Lo correcto (usuario) | Lo que hizo la IA |
|---|---|---|
| URL | `www.flashscore.com/tennis/` | `d.flashscore.co` (Ninja API endpoint) |
| Cookie consent | `#onetrust-accept-btn-handler` | No manejado |
| Scroll para carga | `scrollTo(0, body.scrollHeight)` | No hecho |
| Wait strategy | `domcontentloaded` + `sleep(3)` | `networkidle` |
| Buscar en git antes | SI — el usuario lo habia hecho | NO — la IA invento desde cero |

**Raiz:** `config.py` tiene `FLASHSCORE_BASE = "https://global.flashscore.ninja/202/x/feed"` (Ninja API).
La IA derivo la URL del browser desde la URL de la API — son dos cosas completamente distintas.

---

## FlashScore muestra cuotas en el listado

La estructura HTML es estable:

```html
<div class="event__odds">
  <div class="odds__odd event__odd--odd1" data-bookmaker-id="523">
    <span>2.25</span>
  </div>
  <div class="odds__odd event__odd--odd2" data-bookmaker-id="523">
    <span>1.60</span>
  </div>
</div>
```

### Campos clave

| Selector | Significado |
|---|---|
| `.event__match` | Contenedor de partido |
| `[class*="odds"]`, `[data-odd]`, `.event__odds` | Cuotas (multiples selectores fallback) |
| `event__odd--odd1` | Cuota jugador 1 |
| `event__odd--odd2` | Cuota jugador 2 |
| `data-bookmaker-id="523"` | Bookmaker de referencia |
| `.event__participant` | Nombres de jugadores |

### Ventaja sobre Kambi

| Aspecto | Kambi | FlashScore Odds |
|---|---|---|
| Disponibilidad post-match | NO (retira cuotas) | SI (mantiene cuotas finales) |
| Fuente | Betplay especifico | Bookmaker de referencia (id=523) |
| Velocidad | API ~1s | Playwright ~90s para 60 partidos |
| Cuotas reales para apostar | SI | NO (son de referencia) |
| Util para testing/validacion | Solo partidos activos | Jornada completa |

### Distincion critica

**Kambi = cuotas para APOSTAR.** Son las reales de Betplay donde se despliega capital.

**FlashScore Odds = cuotas para TESTING.** Son de referencia para calcular edge aproximado y
validar el modelo post-hoc. No se usan para sizing ni deploy.

Campo `cuota_es_real` ya existe en el pipeline. Las cuotas FlashScore marcarian `cuota_es_real=False`.

---

## Bug Conocido — remove_duplicates() Indentacion Rota

En `extraer_cuotas_partidos.py` lineas ~260-275, el `return unique_matches` esta
dentro del `for` loop en lugar de fuera. Esto causa que retorne despues del primer
partido procesado, colapsando 60 partidos → 1.

```python
# BUG ACTUAL (return dentro del for):
for match in matches_data:
    ...
    if key not in seen and j1 and j2:
        seen.add(key)
        unique_matches.append(match)

        return unique_matches  # ← MAL: dentro del for

# CORRECTO (return fuera del for):
for match in matches_data:
    ...
    if key not in seen and j1 and j2:
        seen.add(key)
        unique_matches.append(match)

return unique_matches  # ← BIEN: fuera del for
```

**Estado:** Pendiente de fix (D48-BUG-01)

---

## Scope y Limites

### Esto SI es
- Herramienta de **testing y validacion** del pipeline
- Permite correr el pipeline completo post-hoc con cuotas reales de mercado
- Permite auditar calidad del modelo sobre jornadas completas (no solo 13 partidos)

### Esto NO es
- NO reemplaza Kambi para apuestas reales
- NO se usa para sizing ni Kelly deployment
- NO requiere cambios en `edge_calculator.py` ni `trader_ev_tenis.py`
- NO cambia el flujo de produccion (PASO 1-4 siguen usando Kambi)

---

## Investigacion 2026-06-30 — Resultados Completos

### Intento 1 (IA): Ninja API (endpoints de odds) — FALLO
Se probaron multiples patrones de endpoint en `global.flashscore.ninja/202/x/feed/`:
- `df_oo_{bookmaker}_{match_id}` → retorna `0`
- `df_od_{match_id}` → retorna `0`
- `f_2_0_13_es_1_odds` → retorna el mismo feed sin cuotas
- Brute force con 30+ combinaciones → ninguna retorna odds

**Conclusion:** La Ninja API no expone cuotas.

### Intento 2 (IA): Playwright con URL equivocada — FALLO
URL usada: `d.flashscore.co` (derivada de Ninja API URL)
Resultado: pagina de challenge/bloqueo anti-bot (10,925 bytes, solo SVG del logo)

**Conclusion:** BLOQUEADO — pero la conclusion fue erronea. El problema era la URL, no el anti-bot.

### Intento 3 (codigo del usuario): www.flashscore.com/tennis/ — FUNCIONA
Recuperado de git commit `23d2d91`, archivo `extraer_cuotas_partidos.py`:
- Cookie consent manejado: SI
- Click en "Odds": SI
- Elementos encontrados: 437
- Cuotas extraidas: SI (ejemplo: 2.4 / 1.0)
- Anti-bot: NO bloquea

**Conclusion:** El scraper funciona. Solo habia un bug de indentacion en remove_duplicates() — corregido.

---

## Implementacion Completada 2026-06-30

### Como funciona (explicacion simple)

El pipeline de produccion (Kambi) es la maquina original — no fue tocada.
Lo que se agrego es un **boton nuevo** en la misma maquina:

```bash
# Modo normal — Kambi (produccion, cuotas reales Betplay)
python3 extraer_partidos_api.py

# Modo testing — FlashScore-only (sin Kambi, jornada completa post-hoc)
python3 extraer_partidos_api.py --flashscore-only
```

Si no se pasa `--flashscore-only`, el pipeline corre exactamente igual que siempre.

### Resultado del test final 2026-06-30

```
FlashScore feed:    507 partidos (vs 13 de Kambi cuando los partidos ya terminaron)
FlashScore odds:    291 pares con cuotas extraidos de Playwright
Cruce feed+odds:    233/507 con cuotas (46%) — ITF tienen menos cobertura de odds
Match_ids:          507/507 — H2H funciona para todos
Torneos:            37 (grand_slam: 77 | challenger: 59 | itf: 371)
Tiempo total:       77.9s (~4s feed API + ~74s Playwright)
cuota_es_real:      False en todos — guard para no desplegar capital
```

### Arquitectura implementada

**`scraping/kambi_tennis.py` — dos funciones nuevas:**

`fetch_flashscore_odds()` — Playwright scraper:
- URL: `www.flashscore.com/tennis/`
- Cookie consent: `#onetrust-accept-btn-handler`
- Click "Odds" para activar columnas de cuotas
- Scroll para cargar todos los partidos
- Selectores especificos: `.event__odd--odd1` / `.event__odd--odd2`
- Retorna `Dict[match_key -> (cuota1, cuota2)]` indexado por `_build_match_key()`
- Guard: cuotas <= 1.0 descartadas (eran scores de sets, no cuotas)

`extract_matches_flashscore_only(day_offset, tiers)`:
- Llama `fetch_flashscore_feed()` (Ninja API, ~4s)
- Llama `fetch_flashscore_odds()` (Playwright, ~74s)
- Cruza por `_build_match_key()` — mismo sistema de matching que usa Kambi
- Detecta tier con `detectar_tier()` de `config.py`
- Guarda con `save_matches()` — mismo formato que el pipeline normal
- `cuota_es_real=False` en todos los partidos

**`extraer_partidos_api.py` — flag nuevo:**
- `--flashscore-only` invoca `extract_matches_flashscore_only()`
- Sin el flag: corre Kambi exactamente igual que antes

---

## Deuda Tecnica

| ID | Tarea | Prioridad | Estado |
|---|---|---|---|
| D48-BUG-01 | Fix indentacion `remove_duplicates()` en `extraer_cuotas_partidos.py` | ALTA | ✅ COMPLETADO |
| D48-01 | `fetch_flashscore_odds()` en `scraping/kambi_tennis.py` | ALTA | ✅ COMPLETADO |
| D48-02 | `extract_matches_flashscore_only()` — odds integradas en archivo FlashScore-only | ALTA | ✅ COMPLETADO |
| D48-03 | Tests: smoke test de estructura HTML + unit test de parsing | MEDIA | PENDIENTE |
| D48-04 | Flag `--flashscore-only` en `extraer_partidos_api.py` | MEDIA | ✅ COMPLETADO |
| D48-05 | Guard en trader: skip deploy si `cuota_es_real=False` | ALTA | ✅ COMPLETADO |

### Orden completado: D48-BUG-01 → D48-01 → D48-02 → D48-04 → D48-05 ✅
### Pendiente: D48-03 (tests smoke — diferida indefinidamente, valor bajo)

## D48-05 — Implementacion del Guard

Tres archivos modificados en cadena:

1. **`scraping/ninja_h2h_parser.py`** — propaga `cuota_es_real` del partido al `h2h_results_enhanced`
2. **`edge_calculator.py`** — incluye `cuota_es_real` en cada pick del `edge_report`
3. **`trader_ev_tenis.py`** — guard: si cualquier pick tiene `cuota_es_real=False` → imprime advertencia y para

Output del guard cuando se intenta usar cuotas de testing para apostar:

```
======================================================================
  GUARD D48-05 — CUOTAS NO REALES DETECTADAS
======================================================================
  X/Y picks tienen cuota_es_real=False
  Origen: FlashScore odds de referencia (--flashscore-only), NO Betplay.
  Este reporte es SOLO para testing/validacion post-hoc.

  NO desplegar capital real con este reporte.
  Para apuestas reales: correr PASO 1 con Kambi antes del partido.
======================================================================
```

---

## Relacion con Otros Nodos

| Nodo | Relacion |
|---|---|
| [[Nodo-45-Temporal-History-Fallback]] | Ambos resuelven "datos que existen pero el pipeline no accede" |
| ~~~~[[Nodo-31-Ronda-Futura-H2H]]~~ _(MISSING — [[Nodo-31-Future-Match-Data-Leakage]] es diferente)_~~ _(MISSING — [[Nodo-31-Future-Match-Data-Leakage]] es diferente)_ | Patron similar: fallback cuando fuente primaria no tiene datos |
| [[Nodo-21-Pesos-Diferenciados-Por-Tier]] | ITF/Challenger son los mas afectados por falta de cuotas post-match |
