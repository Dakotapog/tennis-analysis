# Nodo-49: Playwright H2H Fallback para n_h2h=0 — El mismo error de Nodo-48

> **Wikilinks:** [[Nodo-48-FlashScore-Odds-Scraper-Testing]] | [[Nodo-45-Temporal-History-Fallback]] | [[Nodo-33-Filtro-Coinflip-Sin-H2H]] | [[Nodo-35-Historial-Vacio-Flag-Pipeline]] | [[Nodo-117-Auditoria-Scraping-Rankings-Cobertura-H2H]] (B117-02/B117-03: Playwright sin cuotas + selector elige archivo incorrecto)
> **Fecha de descubrimiento:** 2026-07-01
> **Estado:** ⏳ EN IMPLEMENTACION

**Prioridad:** ALTA — genera phantom edge estructural en ITF/Challenger
**Archivos objetivo:** `scraping/ninja_h2h_parser.py`
**Dependencias:** Playwright (ya instalado), `scraping/h2h_extractor.py` (selectores reutilizados)

---

## 0. El Problema — Mismo Error Estructural que Nodo-48

Cuando `extraer_historh2h.py --api-mode` (Ninja API) no encuentra historial de un jugador,
devuelve `p1_history=[]` o `p2_history=[]`. La cadena de fallback actual es:

```
Ninja API → vacío → THF (Nodo-45, sesiones anteriores) → si sigue vacío → NADA
```

El pipeline sigue con `n_h2h=0`, el modelo pone `p_modelo=0.500` (coin-flip),
pero la cuota del bookmaker es @9.50 → `edge = 50% - 10.5% = 39.5%` **FANTASMA**.

El guard de Nodo-33 bloquea estos picks del pool MEGA y de APOSTAR individual.
Pero el **trader** los mete en su pool de cobertura sin verificación → Combo1-8 de hoy
(Mario Arce @9.50, Vlajic @4.50, Guajardo @4.80, Cooper @3.70) son todos n_h2h=0.

### Raíz Idéntica a Nodo-48

| Nodo-48 (cuotas) | Nodo-49 (historial jugador) |
|---|---|
| IA usó `global.flashscore.ninja` (API) | IA usa Ninja H2H API que no indexa jugadores ITF |
| Usuario sabía: `www.flashscore.com/tennis/` funciona | Usuario sabía: `www.flashscore.co/partido/tenis/{match_id}/#/h2h` funciona |
| Nodo declarado BLOQUEADO — solución existía en git | n_h2h=0 declarado como "sin datos" — el DOM de FlashScore SÍ los tiene |
| REGLA GIT-FIRST violada | REGLA GIT-FIRST violada — dos archivos del usuario tenían la solución |

### Evidencia 2026-07-01 — Sesión de producción

```
Mario Arce Fernandez @9.50  — n_h2h=0, p_modelo=0.500, edge=60.4%  → FANTASMA
Teodora Vlajic       @4.50  — n_h2h=0, p_modelo=0.500, edge=40.8%  → FANTASMA
Alexander Guajardo   @4.80  — n_h2h=0, p_modelo=0.500, edge=32.8%  → FANTASMA
Stefan Cooper        @3.70  — n_h2h=0, p_modelo=0.500, edge=27.2%  → FANTASMA
```

Los 4 generaron Combo1-8 en el trader ITF. El usuario lo identificó en 2 minutos.
La IA tardó en ver la causa raíz.

---

## 1. La Solución — El Usuario Ya La Tenía (git commit 23d2d91)

Dos archivos del commit inicial del usuario (2025-10-13) contienen la solución:

### `flashs_revisa h2h_inspector.py` — FlashScoreAdvancedInspector
URL probada y validada:
```
https://www.flashscore.co/partido/tenis/{match_id}/#/h2h/general
```
Selectores validados:
- `.h2h__section` → 3 secciones (P1 historial | P2 historial | Enfrentamientos)
- `.h2h__row` → filas de partidos dentro de cada sección
- `[data-testid="wcl-stageTime"]` → fecha
- `[data-testid="wcl-tableScore"]` → resultado/scores
- `[class*="h2h__participant"]` → jugadores
- `.h2h__icon > div` con clase "win" → outcome ganó/perdió
- `.h2h__event` → torneo

### `extraer_historh2h_version2.py` — método _navigate_to_match_url + _extract_h2h_sections
Navega a `match_url` del partido y hace click en H2H tab, extrae las mismas secciones.

### `scraping/h2h_extractor.py` — código de producción existente (ya en pipeline)
La misma lógica de extracción DOM está implementada en `_parse_player_history()` (líneas 565-630).
El JS evaluate con los selectores correctos está probado en producción.

**La solución es añadir Playwright como tercer eslabón de la cadena de fallback en `ninja_h2h_parser.py`.**

---

## 2. Cadena de Fallback Completa (post-Nodo-49)

```
PASO 1: Ninja API (global.flashscore.ninja) → 197 partidos en ~3.45s/partido
   ↓ si p1_history=[] o p2_history=[]
PASO 2: THF (Nodo-45) → busca en sesiones anteriores (JSON cache)
   ↓ si sigue vacío Y match_id disponible
PASO 3: [NUEVO] Playwright FlashScore DOM → www.flashscore.co/partido/tenis/{match_id}/#/h2h
   → extrae historial real del jugador desde el DOM
   → mismos selectores que h2h_extractor.py (ya validados en producción)
```

---

## 3. Implementación

### Archivo: `scraping/ninja_h2h_parser.py`

**Función nueva (módulo-level, sync wrapper sobre async Playwright):**

```python
def _fetch_player_history_playwright(match_id: str, player_name: str,
                                      section_idx: int) -> List[Dict]:
    """
    Nodo-49: Playwright fallback para historial vacío cuando Ninja API falla.
    section_idx: 0=jugador1, 1=jugador2
    Usa ThreadPoolExecutor para correr async Playwright desde contexto sync.
    """
    import concurrent.futures

    def _run():
        return asyncio.run(_playwright_h2h_async(match_id, player_name, section_idx))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_run).result(timeout=90)
    except Exception as e:
        logger.warning(f"   ⚠️ Playwright fallback falló para {player_name}: {e}")
        return []


async def _playwright_h2h_async(match_id: str, player_name: str,
                                  section_idx: int) -> List[Dict]:
    """Async Playwright: navega al H2H de FlashScore y extrae historial de un jugador."""
    from playwright.async_api import async_playwright
    import asyncio

    h2h_url = f"https://www.flashscore.co/partido/tenis/{match_id}/#/h2h/general"

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=[
            '--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu',
            '--disable-software-rasterizer',
            '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        ])
        page = await browser.new_page()
        await page.set_viewport_size({"width": 1920, "height": 1080})

        try:
            await page.goto(h2h_url, wait_until="domcontentloaded", timeout=30000)

            # Cookie consent
            try:
                btn = await page.wait_for_selector("#onetrust-accept-btn-handler", timeout=5000)
                if btn:
                    await btn.click()
                    await asyncio.sleep(2)
            except Exception:
                pass

            await asyncio.sleep(5)

            # Extraer secciones H2H
            sections = await page.locator('.h2h__section').all()
            if len(sections) <= section_idx:
                logger.warning(f"   ⚠️ Solo {len(sections)} secciones H2H para {player_name}")
                return []

            section = sections[section_idx]
            rows = await section.locator('.h2h__row').all()
            logger.info(f"   🌐 Playwright H2H [{player_name}]: {len(rows)} filas en sección {section_idx}")

            matches = []
            for row in rows:
                try:
                    row_data = await row.evaluate('''el => {
                        const date_el = el.querySelector('[data-testid="wcl-stageTime"]');
                        const score_spans = el.querySelectorAll('[data-testid="wcl-tableScore"]');
                        const result = score_spans.length > 0
                            ? Array.from(score_spans).map(s => s.textContent.trim()).join('-')
                            : (el.querySelector('.h2h__result') ? el.querySelector('.h2h__result').textContent.trim() : null);
                        const participants = el.querySelectorAll('[class*="h2h__participant"]:not([class*="participantInner"])');
                        let opponent = null;
                        for (const p of participants) {
                            const nameSpan = p.querySelector('[data-testid="wcl-scores-simple-text-01"]');
                            if (nameSpan && !nameSpan.className.includes('wcl-hasBackground')) {
                                opponent = nameSpan.textContent.trim();
                                break;
                            }
                        }
                        const icon_div = el.querySelector('.h2h__icon > div');
                        const outcome = icon_div && icon_div.className.toLowerCase().includes('win') ? 'Gano' : 'Perdio';
                        const event_el = el.querySelector('.h2h__event');
                        return {
                            date: date_el ? date_el.textContent.trim() : null,
                            result: result,
                            opponent: opponent,
                            outcome: outcome,
                            tournament: event_el ? event_el.textContent.trim() : 'N/A',
                            event_class: event_el ? (event_el.getAttribute('class') || '') : '',
                        };
                    }''')

                    if not row_data.get('date') or not row_data.get('result'):
                        continue

                    ec = row_data.get('event_class', '').lower()
                    surface = 'N/A'
                    if 'hard' in ec:
                        surface = 'Dura'
                    elif 'clay' in ec:
                        surface = 'Arcilla'
                    elif 'grass' in ec:
                        surface = 'Hierba'
                    elif 'indoor' in ec:
                        surface = 'Indoor'

                    matches.append({
                        'fecha': row_data['date'],
                        'oponente': (row_data.get('opponent') or 'N/A').strip(),
                        'resultado': row_data['result'].replace('\n', '-'),
                        'outcome': row_data['outcome'],
                        'torneo': row_data['tournament'].replace('\n', ' '),
                        'ciudad': 'N/A',
                        'pais': 'N/A',
                        'superficie': surface,
                    })
                except Exception:
                    continue

            return matches

        finally:
            await browser.close()
```

**Integración en `_process_match()` — DESPUÉS del THF (Nodo-45):**

```python
# ── Nodo-45 THF (existente) ──
if not p1_history:
    _t = _lookup_player_history_temporal(p1)
    if _t:
        p1_history = _t
if not p2_history:
    _t = _lookup_player_history_temporal(p2)
    if _t:
        p2_history = _t

# ── Nodo-49: Playwright fallback — si THF no pudo suplementar y hay match_id ──
if not p1_history and match_id:
    logger.info(f"   🌐 Playwright fallback P1: {p1} (n_h2h=0, THF vacío)")
    _pw = _fetch_player_history_playwright(match_id, p1, section_idx=0)
    if _pw:
        logger.info(f"   ✅ Playwright recuperó {len(_pw)} partidos para {p1}")
        p1_history = _pw

if not p2_history and match_id:
    logger.info(f"   🌐 Playwright fallback P2: {p2} (n_h2h=0, THF vacío)")
    _pw = _fetch_player_history_playwright(match_id, p2, section_idx=1)
    if _pw:
        logger.info(f"   ✅ Playwright recuperó {len(_pw)} partidos para {p2}")
        p2_history = _pw
```

---

## 4. Comportamiento Esperado

```bash
# ANTES (Nodo-49 ausente):
# ⚙️ [X/197] Mario Arce Fernandez vs Rival
#    ⚠️ Mario Arce Fernandez no en API proxy y sin match_id_j2 — historial vacío
#    📊 Mario Arce Fernandez: 0 partidos | Rival: 15 | H2H: 0
#    → edge=60.4% FANTASMA en report

# DESPUÉS (Nodo-49 activo):
# ⚙️ [X/197] Mario Arce Fernandez vs Rival
#    ⚠️ Mario Arce Fernandez no en API proxy — historial vacío (API)
#    📚 THF: no hay sesiones anteriores para Mario Arce Fernandez
#    🌐 Playwright fallback P1: Mario Arce Fernandez (n_h2h=0, THF vacío)
#    ✅ Playwright recuperó 18 partidos para Mario Arce Fernandez
#    📊 Mario Arce Fernandez: 18 partidos | Rival: 15 | H2H: 0
#    → edge calculado con datos reales → edge real o descartado por modelo
```

---

## 5. Limitaciones y Scope

### Esto SI es
- Fallback selectivo: solo se activa cuando API + THF fallan y match_id existe
- Datos reales del jugador desde FlashScore DOM
- Compatible con formato existente de `_parse_player_history()`

### Esto NO es
- No reemplaza el modo API (que sigue siendo ~3.45s/partido)
- Añade ~15-30s por partido que active el fallback (solo ITF sin historial)
- No extrae H2H directo entre los dos jugadores (solo historial individual)
- No funciona para partidos sin match_id (los 81 sin cruce FlashScore en PASO 1)

### Impacto en tiempo total
- Hoy: 197 partidos × 3.45s = 679.9s
- Con Nodo-49: 197 partidos × 3.45s + N_fallback × 30s
- N_fallback estimado: 10-20 partidos ITF por sesión = +300-600s adicionales
- Total estimado: ~16-20 min (vs 11 min actual)

---

## 6. Tests

**Archivo:** `tests/test_nodo49.py`

| Test | Qué prueba |
|---|---|
| T49-01 | Fallback se activa cuando p1_history=[] y match_id disponible |
| T49-02 | Fallback NO se activa cuando THF ya supplementó (p1_history no vacío) |
| T49-03 | Fallback NO se activa cuando match_id=None |
| T49-04 | Output del fallback tiene campos correctos: fecha, oponente, resultado, outcome, superficie |
| T49-05 | Si Playwright falla (timeout), retorna [] sin romper el pipeline |
| T49-06 | section_idx=0 extrae sección de jugador1, section_idx=1 extrae sección de jugador2 |

---

## 7. Deuda Técnica Relacionada

| ID | Nodo | Estado | Relación |
|---|---|---|---|
| Nodo-33 | Coin-flip guard en combos | ✅ Fase 1 | Bloquea el SÍNTOMA |
| Nodo-35 | Flag historial vacío | ⏳ PENDIENTE | Bloquea predicciones con datos vacíos |
| Nodo-45 | THF cache sesiones anteriores | ✅ | Primer fallback antes de Playwright |
| **Nodo-49** | **Playwright fallback DOM** | **EN CURSO** | **Soluciona la CAUSA RAÍZ** |

Nodo-33 y Nodo-35 bloquean el síntoma.
Nodo-49 elimina la causa raíz — con datos reales el modelo puede calcular edge real.

---

## 8. Lección Aprendida (REGLA GIT-FIRST)

El usuario tenía `flashs_revisa h2h_inspector.py` (694 líneas, git 23d2d91) con:
- URL correcta validada: `www.flashscore.co/partido/tenis/{match_id}/#/h2h/general`
- Selectores `.h2h__section`, `.h2h__row`, `wcl-stageTime`, `wcl-tableScore` confirmados
- Anti-bot: NO bloquea en esta URL

La IA declaró n_h2h=0 como "sin datos disponibles" sin buscar en git.
El usuario resolvió esto en su primera versión del scraper (octubre 2025).
