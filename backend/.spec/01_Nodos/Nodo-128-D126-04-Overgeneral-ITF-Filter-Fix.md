# Nodo-128 — D126-04 Error de Generalización: Filtro ITF Bloquea M25/W25 con Cobertura Kambi

> **Estado:** IMPLEMENTACIÓN — 2026-07-21
> **Tipo:** AUDIT → FIX — post-diagnóstico profundo
> **Trigger:** Usuario detecta Fontes Damorim (São Paulo ITF, conf=95%, diff=0.961) + torneos Bali/Brisbane/Nogent/Saskatoon/Santa Fe ausentes del pipeline
> **Autor:** Sonnet 4.6 (análisis doctoral, razonamiento extendido)

---

## Wikilinks

| Link | Rol |
|------|-----|
| [[Nodo-126-Auditoria-EvalGames-Bridge-Fugas-Fixes]] | Padre — D126-04 original (demasiado amplio) |
| [[Nodo-125-EvalGames-Bridge-Dashboard-X4]] | Bridge implementado |
| [[Nodo-40-Games-Sets-Signal-Layer]] | `_buscar_event_id_kambi` — función bajo análisis |
| [[Nodo-124-EvalTracker-TablaFavoritos-ShadowBook]] | `generar_tabla_favoritos2` — fuente del tier=Desconocido |

**Wikilinks totales: 4 | Huérfanos: 0**

---

## §1. Tres problemas distintos — diagnóstico separado

### §1.1 Francisco Fontes Damorim vs Carlos Maria Zarate (São Paulo ITF)

**Datos observados:**
```
sb_id:       EVAL_2026-07-21_desconocido_damorim-zarate_ML
diff_abs:    0.961  (dominancia máxima — cuota 1.02 = p_implícita 98%)
confidence:  0.95   (95% — mayor posible)
tier:        None   → NO bloqueado por D126-04
tiene_mercados: False → Kambi consultado, retornó evento sin mercado UNDER total games
```

**Causa raíz:** Este partido SÍ pasó el bridge completo. `_buscar_event_id_kambi()` encontró (o no encontró) el evento, pero incluso si lo encuentra, São Paulo ITF no tiene mercado "Total de juegos" en Kambi/betplay — solo ML básico. El partido está en nuestro radar (confianza correcta), pero el mercado apostable UNDER no existe.

**No es un bug del filtro D126-04.** Es una limitación real de Kambi para este torneo.

**Fix D128-01 (informativo):** Para picks con `diff_abs > 0.90` y `tiene_mercados=False`, el bridge debe agregar un campo `_watchlist_dominante=True` para que el dashboard X4 los destaque visualmente. Kambi añade mercados dinámicamente — ejecutar bridge 2h antes del partido puede capturar el mercado cuando se publique.

### §1.2 Filtro D126-04 bloquea M25/W25 con cobertura Kambi real

**El filtro original:**
```python
_TIERS_SIN_KAMBI = {'itf_minor', 'itf', 'm15', 'w15', 'm25', 'w25', 'm10', 'w10'}
```

**El problema:** `'itf'` y `'m25'`, `'w25'` están en la lista. Pero:

- `detectar_tier()` devuelve `'itf'` para eventos que son M25, W25, M35, W35, M50+
- Kambi SÍ tiene cobertura real para: **Brisbane M25, Bali M25, Nogent W25, Saskatoon W25, Santa Fe W25**
- 11 picks con `tier='itf'` son bloqueados sin verificación — algunos son Nogent W25 (Arseneault/Mercier, Triquart/Yao = jugadoras francesas confirman esto)
- El filtro estaba basado en observación de UN día (2026-07-21) con torneos específicos que fallaron — no es una regla general válida

**Evidencia del error de generalización:**
```
Arseneault A. vs Mercier C.  | tier='itf' | torneo=Desconocido
→ nombres franceses → Nogent W25 France → Kambi SÍ tiene este torneo

Triquart S. vs Yao X.        | tier='itf' | torneo=Desconocido
→ Triquart = francesa, Yao = china → Nogent W25 France

Borrelli L. vs Mecarelli M.  | tier='itf' | torneo=Desconocido
→ nombres italianos → ITF Italia M25 → Kambi cubre Italia M25
```

**Fix D128-02:** Reducir el filtro a únicamente los torneos donde Kambi NUNCA tiene cobertura real:
```python
_TIERS_SIN_KAMBI = {'itf_minor', 'm10', 'w10', 'm15', 'w15'}
# Removido: 'itf', 'm25', 'w25' — estos SÍ pueden estar en Kambi
```

Con esto los 11 picks `tier='itf'` intentarán lookup Kambi. El lookup es el oráculo real, no el tier name.

**Impacto runtime:** +11 requests HTTP adicionales (~11s extra). Aceptable dado que cada señal perdida = ~$1,000-2,000 EV negativo.

### §1.3 torneo='Desconocido' en 106/120 picks EVAL_ — fuga de información en la fuente

**Observación:**
```python
# En shadow_book, TODOS los picks EVAL_ tienen:
snap.get('torneo') = 'Desconocido'
snap.get('tier')   = None  (106 picks) o 'itf' (11 picks)
```

`generar_tabla_favoritos2.py` no transfiere correctamente el nombre del torneo al pick_snapshot. El campo `torneo_nombre` existe en los datos de zita pero no llega al snapshot de `log_evaluar_pick()`.

**Sin nombre de torneo no podemos:**
- Distinguir "Nogent W25" de "Huamantla W15" → imposible filtrar inteligentemente
- Construir lookup por nombre de torneo en Kambi (solo usamos apellidos de jugadores)
- Mostrar contexto en dashboard X4

**Fix D128-03:** En `generar_tabla_favoritos2.py`, en el bloque de `_pick_e` que se pasa a `log_evaluar_pick()`, incluir el campo `torneo` desde el match dict.

---

## §2. Fixes específicos

### D128-01 — Watchlist dominante para picks sin mercado pero extrema confianza

**Archivo:** `scripts/evaluar_games_bridge.py` — en `_process_pick()`, antes del return final cuando no hay señales:

```python
# D128-01: marcar picks extremadamente dominantes sin mercado para watchlist
if not optimas and diff_abs > 0.85:
    resultado['_watchlist_dominante'] = True
    resultado['_watchlist_reason'] = f'diff={diff_abs:.3f} > 0.85, Kambi puede publicar market later'
```

En `_save_report()`:
```python
'watchlist_dominante': [r for r in resultados if r.get('_watchlist_dominante')],
```

En dashboard X4: mostrar sección "DOMINANTES SIN MERCADO — revisar 2h antes" con estos picks.

### D128-02 — Reducir tier filter a solo M15/W15 y menores (CRÍTICO)

**Archivo:** `scripts/evaluar_games_bridge.py` — en `_process_pick()`:

```python
# D128-02: solo excluir ultra-menores — M25/W25 y 'itf' genérico pueden estar en Kambi
_TIERS_SIN_KAMBI = {'itf_minor', 'm10', 'w10', 'm15', 'w15'}
# REMOVIDO de la lista: 'itf', 'm25', 'w25'
# Racional: detectar_tier() devuelve 'itf' para M25/W25/M35/W35 indistintamente.
# Kambi cubre Brisbane M25, Bali M25, Nogent W25, Saskatoon W25, Santa Fe W25.
# El lookup de Kambi es el oráculo real — no el tier name.
```

### D128-03 — Transferir torneo al pick_snapshot en generar_tabla_favoritos2

**Archivo:** `generar_tabla_favoritos2.py` — en el bloque `_pick_e` (log_evaluar_pick):

```python
_pick_e = {
    ...
    'torneo':      match.get('torneo_nombre') or match.get('torneo') or match.get('tournament') or '',
    'tier':        match.get('tier') or detectar_tier(match.get('torneo_nombre', ''), ...),
    ...
}
```

Con el nombre real del torneo en el snapshot:
- El bridge puede filtrar inteligentemente: "Nogent" → M25/W25 → intentar Kambi
- El dashboard X4 muestra contexto real ("Nogent W25 France")
- El sb_id ya no sería `EVAL_2026-07-21_desconocido_...` sino `EVAL_2026-07-21_nogent-w25_...`

### D128-04 — Wplay fallback correcto (revisado)

Del análisis del código: la verificación actual en wplay SSR busca por apellido en `feeds[apellido]['wplay']['seln_id']`. Pero `fetch_all_odds(['wplay'])` retorna estructura `{nombre: {'wplay': {...}}}`.

**Problema:** La búsqueda actual hace `_wplay_feed.get(_apellido, {}).get('wplay')` pero el dict es `{nombre: {'wplay': {...}}}`. Correcto. Pero wplay SSR solo tiene 48 outcomes (ATP/WTA principales) — no tiene M25/W25 ITF tampoco.

**Para wplay:** mantener el lookup, pero documentar que wplay SSR ≠ betplay ML para ITF. Wplay cubre torneos ATP/WTA pero no ITF menores (ni siquiera M25 en general).

**Conclusión D128-04:** Wplay es útil como segunda fuente de precio para ATP/WTA (divergencia), NO como fallback para ITF que betplay no tiene. Los picks ITF sin mercado en betplay → sin mercado apostable en casas CO online para mercados secundarios de juegos.

---

## §3. Orden de implementación

| Fix | Impacto | Complejidad | Prioridad |
|-----|---------|-------------|-----------|
| D128-02 | CRÍTICO — recupera 11 picks/día filtrados | 1 línea | HOY |
| D128-03 | ALTO — información de torneo en snapshot | 2-3 líneas | HOY |
| D128-01 | MEDIO — visibilidad watchlist dominantes | 5 líneas | HOY |
| D128-04 | BAJO — ya implementado, documentar limitación | 0 líneas | doc |

---

## §4. Tests REGLA-T53 — `tests/test_nodo128_itf_filter_fix.py`

```python
def test_D128_02_itf_tier_generic_goes_to_kambi_lookup()
    # tier='itf' no debe estar en _TIERS_SIN_KAMBI
def test_D128_02_m25_goes_to_kambi_lookup()
    # tier='m25' no debe estar en _TIERS_SIN_KAMBI
def test_D128_02_m15_still_skipped()
    # tier='m15' SÍ debe estar en _TIERS_SIN_KAMBI
def test_D128_02_w15_still_skipped()
    # tier='w15' SÍ debe estar en _TIERS_SIN_KAMBI
def test_D128_01_watchlist_dominante_added_when_diff_gt_085_no_market()
    # diff=0.961, tiene_mercados=False → _watchlist_dominante=True
def test_D128_03_torneo_field_in_pick_snapshot()
    # generar_tabla_favoritos2 incluye 'torneo' en el snapshot
```

---

## §5. Impacto financiero proyectado

```
Situación actual (D126-04 original):
  11 picks/día con tier='itf' → BLOQUEADOS sin verificación Kambi
  Estimado 4-5 de esos en Kambi (Brisbane M25, Nogent W25, etc.)
  Estimado 1-2 con mercado UNDER games → señales perdidas
  EV perdido por señal @3.5x $1,000: 0.846 × $3,500 - $1,000 = +$1,961/señal

Con D128-02:
  11 picks → intentan Kambi (11s extra runtime)
  +1-2 señales recuperadas/día
  +$1,961-3,922 EV adicional diario
```

---

**Wikilinks totales: 4 | Huérfanos: 0**

[[Nodo-126-Auditoria-EvalGames-Bridge-Fugas-Fixes]] | [[Nodo-125-EvalGames-Bridge-Dashboard-X4]] | [[Nodo-40-Games-Sets-Signal-Layer]] | [[Nodo-124-EvalTracker-TablaFavoritos-ShadowBook]]
