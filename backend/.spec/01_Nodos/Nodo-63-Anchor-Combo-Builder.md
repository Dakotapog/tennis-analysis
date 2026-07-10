# Nodo-63 — Anchor Combo Builder + Insufficient History Guard

> Fecha: 2026-07-06
> Estado: COMPLETO
> Tests: T63-01→T63-12 (12 tests nuevos, todos pasan)

---

## §0 Veredicto Ejecutivo

| Problema | Causa Raíz | Fix | Impacto |
|---|---|---|---|
| Edge falso en Magadan vs Rodriguez | FlashScore retorna 3 partidos ITF qualifying → `days_since=356d` → `form_decay x0.35` | Insufficient History Guard: n<8 → fd=1.0 | Elimina edge fantasma de 31.2% por inactividad mal diagnosticada |
| Sin capa de combos con cuota alta | Picks ancla (@1.65+) con alta priority no tienen vehículo separado | Anchor Combo Builder: 1A+3B / 2A+2B / 3A+2B | Nueva capa con cuotas @4-35x, alpha ortogonal al CORE |

---

## §1 Diagnóstico Bug Rodriguez (Caso hvYf5E1t)

**Partido:** Magadan vs Rodriguez J. A. — Challenger Bogotá

**Cadena de fallo:**
1. FlashScore extrae 3 partidos de Rodriguez (todos 2025 qualifying ITF)
2. `days_since_p2 = 356` (último partido hace ~1 año según esos 3 registros)
3. `form_decay_factor(356) = max(0.35, exp(-0.025 × (356-30))) = 0.35` (floor)
4. Rodriguez aparece como "inactivo" → score penalizado
5. Magadan sube artificialmente a 79.7% confianza, edge=31.2%
6. **Edge es falso** — Rodriguez juega ITF regularmente, FlashScore simplemente no captura qualifying locales

**Realidad:** Rodriguez no está inactivo. FlashScore devuelve pocos partidos en qualifying/ITF local. Son datos incompletos de la fuente, no inactividad del jugador.

---

## §2 Fix — PARTE A: Insufficient History Guard

**Archivo:** `analysis/rivalry_analyzer.py`

**Nueva constante (línea ~25):**
```python
_MIN_HISTORY_FOR_DECAY = 8   # Nodo-63: n mínimo para aplicar form_decay
```

**Guard en `generate_advanced_prediction` (reemplaza bloque form_decay original):**
```python
_n_p1 = len(player1_history) if player1_history else 0
_n_p2 = len(player2_history) if player2_history else 0

_fd_p1 = 1.0 if _n_p1 < _MIN_HISTORY_FOR_DECAY else _form_decay_factor(days_since_p1)
_fd_p2 = 1.0 if _n_p2 < _MIN_HISTORY_FOR_DECAY else _form_decay_factor(days_since_p2)
```

**Log cuando guard activo:**
```
LOG_INSUFFICIENT_HISTORY: {player_name} — solo N partidos,
form_decay omitido (n<8 = datos incompletos, no inactividad real)
```

**Guard en `generar_tabla_favoritos2.py`:**
Cuando `LOG_INSUFFICIENT_HISTORY` está en reasoning para un jugador,
la señal `INACTIVIDAD: {player} -- {N}d sin jugar` NO se muestra.

**Semántica:** n<8 significa "FlashScore no tiene suficientes datos" — no que el jugador esté inactivo.

---

## §3 Feature — PARTE B: Anchor Combo Builder

**Archivo:** `combo_confianza_builder.py`

### Definición de ancla

Un pick es "ancla" si tiene cuota alta Y fiabilidad por señal:
- `combo_priority >= 75.0` AND `cuota >= 1.65`
- OR `conf >= 60.0` AND `cuota >= 1.65`
- OR `edge_pct >= 10.0%` AND `cuota >= 1.65`

Los picks con `cuota < 1.65` son siempre BASE (sin importar priority).

### 3 tiers de combos

| Tier | Estructura | Cuota objetivo | P(win) típico |
|---|---|---|---|
| 1A+3B | 1 ancla + 3 base | @4-7x | 18-25% |
| 2A+2B | 2 anclas + 2 base | @7-15x | 7-14% |
| 3A+2B | 3 anclas + 2 base | @15-35x (moonshot) | 3-6% |

### Guards

- `P(win) >= ANCHOR_PWIN_MIN = 2.5%` — combos con demasiada cuota excluidos
- Max 2 picks del mismo torneo por combo (`MAX_SAME_TOURNAMENT`)
- Budget = 30% del budget de fase, dividido en 3 tiers iguales

### Uso

```bash
python3 combo_confianza_builder.py --bankroll 125000 --anchor
python3 combo_confianza_builder.py --bankroll 125000 --fase 4 --anchor --excluir "Jugador X"
```

Genera `AC*.bat` en el escritorio (prefijo AC = Anchor Combo).

### Constantes (congeladas)

```python
ANCHOR_CUOTA_MIN    = 1.65
ANCHOR_PRIORITY_MIN = 75.0
ANCHOR_CONF_MIN     = 60.0
ANCHOR_EDGE_MIN     = 10.0
ANCHOR_PWIN_MIN     = 0.025
MAX_ANCHOR_COMBOS   = 12
```

---

## §4 Tests T63-01→T63-12

### PARTE A

| Test | Descripción | Resultado |
|---|---|---|
| T63-01 | n=3, days=60 → fd=1.0 (guard activo) | PASS |
| T63-02 | n=10, days=60 → fd<1.0 (decay normal) | PASS |
| T63-03 | n=3, days=356 → fd=1.0 (no x0.35 floor) | PASS |
| T63-04 | n=7 (boundary: <8) → guard activo | PASS |
| T63-05 | n=8 (exactamente =8) → guard NO activo | PASS |
| T63-06 | `_MIN_HISTORY_FOR_DECAY == 8` | PASS |

### PARTE B

| Test | Descripción | Resultado |
|---|---|---|
| T63-07 | priority=85, cuota=2.06 → ANCLA | PASS |
| T63-08 | priority=65, cuota=1.33 → BASE | PASS |
| T63-09 | `_build_anchor_combos` → combos_1a3b no vacío | PASS |
| T63-10 | combo 1A+3B tiene >=1 ancla (cuota>=1.65) | PASS |
| T63-11 | combo 2A+2B tiene >=2 anclas | PASS |
| T63-12 | `ANCHOR_CUOTA_MIN==1.65`, `ANCHOR_PRIORITY_MIN==75.0` | PASS |

### Actualización T57-01

T57-01 usaba n=4 history matches para testear decay con 49d. Con Nodo-63, n=4 < 8 → guard suprime decay. Corrección: n=8 history matches (umbral exacto donde decay empieza a aplicar).

---

## §6 Limitación conocida: playwright_queue.json sin consumidor (auditoría 2026-07-09)

La cola de candidatos Playwright (`data/playwright_queue.json`) tiene mecanismo de
escritura funcional (`_enqueue_playwright_candidate` en `rivalry_analyzer.py:32`),
conectado correctamente al gate n<8 (mismo bloque `if _match_id:` que emite
`LOG_PLAYWRIGHT_CANDIDATE`).

**Ciclo incompleto:** ningún script lee `playwright_queue.json` para disparar
re-scraping Playwright real. El archivo ni siquiera existe en producción (nunca
se ha activado la condición n<8+match_id en modo API desde su creación).

**Decisión 2026-07-09:** no construir consumidor todavía.
- Condición de disparo históricamente: 0 activaciones.
- Playwright es PRIMARIO — el modo API (donde aplica este guard) es fallback.
- Guard mínimo añadido: `pre_game_validator.py` emite `[WARN] PLAYWRIGHT_QUEUE_PENDIENTE`
  si `data/playwright_queue.json` llega a tener contenido.
- Prioridad de implementar consumidor: BAJA mientras Playwright siga siendo primario
  y la condición de disparo no se active en producción.

---

## §5 Checklist de Verificación Post-Implementación

```bash
# 1. Tests nuevos pasan
python -m pytest tests/test_nodo63.py -v --no-cov
# Esperado: 12 passed

# 2. Suite completa sin regresiones
python -m pytest tests/ --no-cov -q | tail -3
# Esperado: 1691 passed (1679 baseline + 12 nuevos)

# 3. Anchor combos se generan con datos reales
python3 combo_confianza_builder.py --bankroll 125000 --fase 4 --anchor
# Esperado: sección "ANCHOR COMBOS (Nodo-63)" con anclas identificadas

# 4. Confirmar constante
python3 -c "from analysis.rivalry_analyzer import _MIN_HISTORY_FOR_DECAY; print(_MIN_HISTORY_FOR_DECAY)"
# Esperado: 8

# 5. Verificar LOG_INSUFFICIENT_HISTORY en próximo pipeline
# (se verá en el siguiente run de extraer_historh2h.py para partidos con n<8)
python3 extraer_historh2h.py --api-mode --all-tournaments
grep "LOG_INSUFFICIENT_HISTORY" reports/h2h_results_enhanced_*.json
```
