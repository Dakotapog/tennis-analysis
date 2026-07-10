# Nodo-74 — Combo Governor: Presupuesto de Sesión Multi-Capa (C63-B)

**Fecha:** 2026-07-09  
**Rama:** main  
**Estado:** BORRADOR — condición de graduación pendiente (ver Decisión C63-B)  
**Cierra:** FABLE_02 §2 C63-B, V-26-2a (gap session_budget)

---

## Contexto

Con la introducción de Anchor Combos (Nodo-63), el sistema ahora tiene **6 capas independientes de combos**:
1. CORE (trader-driven, tier-basado)
2. Satellite (SAT_1, SAT_2, SAT_3)
3. MOONSHOT (oportunidades extremas)
4. COBERTURA (COB_*)
5. Mega + Safe + Games (Betplay)
6. GCS sub-plan (promo, hierba)

**Deuda anterior (V-26-2a):** `session_budget(bankroll)` en `betplay_combo_builder.py` solo cubre megas de Betplay (4% del bankroll). Las otras 5 capas acumulan stakes sin límite de sesión.

**Riesgo:** suma total de stakes en una sesión puede exceder el target M-26-2 (4-7% del bankroll según tier), dejando la cartera exposeda sin protección.

---

## Solución

### D74-A: `combo_governor.py`

Archivo nuevo: supervisor de presupuesto **read-only** que:
- Lee el combo_plan_*.txt del día (si existe, generado por combo_confianza_builder)
- Lee el último trader_plan_*.json (si existe)
- Parsea todos los archivos de combo `.txt` en `reports/` (CORE, Satellite, mega, safe, etc.)
- Suma stakes por capa
- Reporta (i) total acumulado, (ii) límite M-26-2 declarado, (iii) si excede

**Modo:** READ-ONLY (no modifica stakes, no bloquea builders). Reporte pre-emisión de `.bat` files.

**Capas supervisadas:**
```
combo_confianza_builder:  CORE | SAT_* | MOONSHOT | COB_*
betplay_combo_builder:    mega | safe | games | WAS (si hay apuestas_FECHA.txt)
```

**Orden de corte si usuario decide recortar** (de mayor varianza primero):
```
1. ANCHOR_3A2B
2. ANCHOR_2A2B
3. ANCHOR_1A3B
4. MOONSHOT
5. SAT_*
6. COB_*
7. CORE
```

---

## Archivos

| Archivo | Rol |
|---|---|
| `combo_governor.py` | Lectura, parseo, reporte. Salida a stdout + log. |
| `reports/combo_plan_*.txt` | Entrada (CORE + Satellite, generado por combo_confianza_builder) |
| `reports/trader_plan_*.json` | Entrada (tier, fecha, bankroll referencia) |
| `reports/apuestas_*.json` | Entrada (para WAS detection) |
| `logs/combo_governor.log` | Salida append-only (REPORTE sobrescribible diario) |

---

## Uso

```bash
# Detectar bankroll del trader_plan más reciente, día actual
python3 combo_governor.py

# Especificar bankroll
python3 combo_governor.py --bankroll 125000

# Día específico
python3 combo_governor.py --fecha 2026-07-08

# Modo verbose
python3 combo_governor.py --verbose
```

**Salida esperada:**
```
[combo_governor] 2026-07-09 14:30:00
Bankroll: $125,000 | Session Budget (4%): $5,000
─────────────────────────────────────────────────────
CORE            $1,200  (24.0%)
SAT_1           $400    (8.0%)
SAT_2           $300    (6.0%)
MOONSHOT        $150    (3.0%)
mega            $800    (16.0%)
safe            $600    (12.0%)
games           $400    (8.0%)
─────────────────────────────────────────────────────
TOTAL           $3,850  (77.0% of budget)
STATUS          ✅ OK (overhead: $1,150)
```

Si excede:
```
STATUS          ⚠️  OVER BUDGET (+$150 / 3% overage)
RECOMMENDATION  recorta MOONSHOT ($150) o SAT_2 ($300) según riesgo
```

---

## Decisiones de diseño

- **D74-01:** Read-only. No modifica; solo advierte. La decisión de recorte la toma el operador en `combo_confianza_builder --phase X --max-stake` o `betplay_combo_builder --safe-mode`.
- **D74-02:** Parsing robusto. Si no encuentra archivo, asume $0 para esa capa (no crashea).
- **D74-03:** Bankroll inferencia: (1) argumento CLI, (2) campo en combo_plan, (3) últimos trader_plan, (4) último betslip registrado. Fallback a 125000 con warning.
- **D74-04:** Log append-only (nunca sobrescribe), similar a `logs/n8n_snapshots.log`. Permite auditoría de cambios de presupuesto.
- **D74-05:** Versión 1.0 es reporte. Versión 1.1+ puede integrarse en `run_daily.py` con auto-recorte (futuro).

---

## Tests

| Test | Caso | Resultado esperado |
|---|---|---|
| T74-01 | Archivo combo_plan_*.txt parsea CORE + SAT | stakes correctos ±1% |
| T74-02 | Archivo betplay_combo_builder genera megas | suma de megas correcta |
| T74-03 | Total < 4% bankroll | STATUS "OK" |
| T74-04 | Total > 4% bankroll | STATUS "OVER BUDGET" + recomendación |
| T74-05 | Sin archivos encontrados | asume $0, no crashea |
| T74-06 | Bankroll ambiguo (múltiples fuentes) | prioridad correcta (CLI > combo_plan > trader_plan) |
| T74-07 | Bankroll=0 o inválido | error explicativo, no traceback |

---

## Integración M-26-2

**Restricción:** M-26-2 session_budget = 4% del bankroll (default).  
**Cálculo:** `session_budget(125000) = 5000`  
**Governor:** suma 6+ capas, compara contra $5000.

**Cadena de decisión:**
1. `combo_confianza_builder` genera combo_plan.txt con stakes parciales
2. `betplay_combo_builder` añade megas/safe a reports/
3. **AQUÍ:** operador corre `python3 combo_governor.py` (o es automático en daily_brief)
4. Si "STATUS OK" → emitir .bat files
5. Si "STATUS OVER BUDGET" → recortar manualmente y re-generar, o desactivar `--fase` más agresivo

---

## Prohibido

- NO modificar stakes automáticamente (no es su trabajo)
- NO exponer botón de "recorte automático" en dashboard sin pre-registro de H74-01
- NO cambiar las constantes de M-26-2 (4%) en este archivo — solo `betplay_combo_builder.py` las define
- NO usar combo_governor.py para bloquear apuestas (es reporte, no gate)

---

## Limitaciones conocidas

- **Modo READ-ONLY activo desde 2026-07-09.** El governor imprime WARNING pero no bloquea ni modifica stakes. Esta es la limitación más importante: el riesgo de exceder M-26-2 existe pero no es bloqueado automáticamente.
- **0 ejecuciones pre-integración.** Antes de ser integrado en `run_daily.py` (2026-07-09), el governor nunca fue ejecutado en producción. No existe historial real de excedencias. El riesgo de portfolio multi-capa es hasta ahora teórico.
- **Criterio de graduación a bloqueo:** Se requieren ≥10 sesiones reales registradas en `logs/combo_governor.log` con evidencia de cuántas veces (si alguna) se excedió M-26-2. Sin ese n mínimo, cambiar a bloqueo automático sería desplegar a ciegas — lo que la Constitución §2 REGLA-HF-5 prohíbe explícitamente.
- **Parsing basado en texto.** El governor parsea `combo_plan_*.txt` con regex. Si el formato del archivo cambia (ej. nueva capa, nuevo prefijo), el parser puede subestimar el total. Verificar con `--verbose` si se agregan capas nuevas.
- **C63-A corrección (auditoría 2026-07-09):** La cola Playwright para n<8 (`_enqueue_playwright_candidate`) ya existía como cola JSON completa en `data/playwright_queue.json` con deduplicación por `match_id::nombre`. El diagnóstico inicial de "log string" era incorrecto. C63-A está más implementado de lo reportado.

---

## Trabajo futuro (post-Fase 2)

| Item | Descripción | Prioridad |
|---|---|---|
| Auto-recorte en run_daily | Si se invoca con `--auto-trim`, recorta automáticamente por orden D74-04 | BAJA |
| Gate de datos → decisión | n≥50 sesiones: medir si el overage real (suma sin truncar) se correlaciona con drawdown | BAJA |
| H74-01 prospectiva | Pre-registrar: "sesiones con governor-OK tienen P&L±5pp mejor que sesiones OVER" | FUTURA |

---

## Auditoría de cierre

- [ ] `combo_governor.py` existe y es ejecutable
- [ ] `python3 combo_governor.py` sin args → bankroll detectado correctamente
- [ ] T74-01 a T74-07 pasan
- [ ] Salida reporta formato claro (tabla con porcentajes)
- [ ] Logs en `logs/combo_governor.log` append (no trunca)
- [ ] SDD compliance: este Nodo documenta el archivo (retroactivo)
