# Nodo-103 — Auditoría Combo Builder: Gates n_h2h=0 + Watchlist Filtration + Correlation Cap

> **Wikilinks:** [[Nodo-37-Combo-Confianza-Builder]] | [[Nodo-63-Anchor-Combo-Builder]] | [[Nodo-33-Filtro-Coinflip-Sin-H2H]] | [[Nodo-28-Conditional-Decomposition-Metamodel]] | [[Nodo-74-Combo-Governor]] | [[Nodo-86-Auditoria-Fable5]] | [[Nodo-99-Auditoria-Fable-N97-N98]] | [[Nodo-65-Convergencia-Multi-Senal-Patron-Combos]] | [[Nodo-49-Playwright-H2H-Fallback-n-h2h-0]] | [[Nodo-100B-Triple-Convergencia-Live]] | [[Nodo-101-Shadow-Book-Live-CLV]] | [[Nodo-117-Auditoria-Scraping-Rankings-Cobertura-H2H]] (gates n_h2h afectan mismo universo restringido por B117-02/B117-03)
> **Fecha:** 2026-07-16 | **Severidad:** CRÍTICA — pérdida real materializada 2026-07-15, combos con legs apostar=False
> **Autor:** Auditoría Sonnet 4.6 con evidencia de edge_report + shadow_book
> **NO DUPLICA:** Nodo-33 (gate cuota<2.10 individual), Nodo-63 (insuficient history guard individual). Este nodo es específico para el builder de combos.

---

## 1. CONTEXTO — Pérdidas reales 2026-07-15

Seis combos generados el 15 de julio 2026 con cuotas totales 7.64–79.23 (apuestas $500–$529 cada uno).
Resultado: 5 combos perdidos, 1 ganado. Pérdida neta estimada > $2,000.

**Picks recurrentes en los combos perdidos:**

| Pick | Apariciones | Resultado | `apostar` | `n_h2h` |
|---|---|---|---|---|
| Karla Bartel @ 3.50 | 4 combos | PERDIDA | **False** | **0** |
| Ustiniya Lekomtseva @ 2.65 | 4 combos | PERDIDA | **False** | **0** |
| Ava Schueller @ 3.35 | 5 combos | PERDIDA | **False** | **0** |
| Amahee Charrier @ 2.43 | 2 combos | PERDIDA | True | 0 |
| Eva Oxford @ 4.10 | 1 combo | PERDIDA | True | 0 |
| Gio Jang @ 2.55 | 4 combos | GANADA | True | 0 |

---

## 2. DIAGNÓSTICO — 4 causas raíz con evidencia de campo

### D103-RC1: Picks con `apostar: False` entraron en combos (BLOQUEANTE)

Bartel, Lekomtseva y Schueller tienen `apostar: False` en el edge_report — el trader las rechazó para apuestas individuales. El combo builder las tomó igual. Esta es una brecha de seguridad directa: picks que no pasan el gate individual no deben ser patas de combos.

**Evidencia directa (edge_report_20260715_092046.json):**
```
Bartel K.:     apostar: False  confidence_flag: LOW   n_h2h: 0
Lekomtseva U.: apostar: False  confidence_flag: STRONG n_h2h: 0
Schueller A.:  apostar: False  confidence_flag: LOW   n_h2h: 0
```

### D103-RC2: `n_h2h = 0` — cero historial directo en todos los perdedores

Ningún pick perdedor tiene un solo partido previo entre esas jugadoras. El modelo opera con cero datos empíricos del matchup. El "edge" calculado es ruido de modelo, no señal real:
- ELO + BBI sin H2H = especulación calibrada, no predicción
- El bookmaker tiene información de matchup que el modelo no tiene

**Nodo-33** ya documenta el problema para picks individuales. **Nodo-49** lo documenta para el fallback H2H. Este nodo es la instancia combo.

### D103-RC3: `n_axes_active ≤ 1` — BBI sola no predice (N28F2 ya lo sabía)

Los tres picks problemáticos tienen `n_axes_active: 1`. La regla N28F2 del **Nodo-28** establece explícitamente: "BBI sola no predice". Lekomtseva incluso tiene en el edge_report:
```
motivo_reclasificacion: N28F2: n_axes_active < 2 (BBI sola no predice)
```
El combo builder ignoró una reclasificación N28F2 explícita.

Adicionalmente, Bartel tiene `net_alignment: -0.286` — **alineamiento negativo**, señales alineadas EN CONTRA del pick — y aun así entró en 4 combos.

### D103-RC4: Correlación catastrófica — 3 legs idénticos en 5 combos simultáneos

Bartel + Lekomtseva + Schueller comparten las mismas fallas estructurales (n_h2h=0, n_axes=1, apostar=False). Al perder las tres, todos los combos que las contienen fallan en cascada. Correlación ρ=1.0 en la pérdida.

**Nodo-74** (Combo Governor) y el parámetro ρ de Kelly existen exactamente para prevenir esto. El governor no estaba aplicando límite de repetición por pick.

---

## 3. MORFOLOGÍA DEL EDGE FANTASMA

El mecanismo exacto por el que el modelo "ve edge" sin tenerlo:

```
p_historica_usada = 0.3366  (bucket global ITF, n=99 registros del bucket)
n_calibracion = 99           ← este es el n del BUCKET, no del par de jugadoras
shrinkage = 99/(99+20) = 0.832  ← casi peso completo al prior del bucket
calibration_confidence = 0.8319  ← inflada artificialmente por n del bucket

p_elo_base ≈ 0.45–0.46       (jugadoras con ELO similar → coin flip)
BBI (bbi: 0.72–0.73)         (bookmaker aparentemente subvalora underdog)
p_blend ≈ 0.49–0.50          (mezcla ELO + BBI → apenas sobre 50%)
p_modelo ≈ 0.508–0.527       (agrega H2H vacío → prior pesa casi todo)

edge = p_modelo - p_implicita ≈ 23–25%  ← edge FANTASMA
```

El `n_calibracion=99` no refleja confianza en el matchup — refleja cuántos registros tiene el bucket global ITF. La shrinkage da 83% de peso a un prior genérico de "underdog ITF en hard gana el 33.66% de las veces". Con ese prior, cualquier underdog con ELO comparable parece tener edge.

---

## 4. GATES REQUERIDOS EN `combo_confianza_builder.py`

### G1 — `apostar: True` obligatorio (Gate de seguridad primario)

```python
# combo_confianza_builder.py — _filtrar_picks_para_combos()
if not pick.get('apostar', False):
    log_blocked(pick, 'G1: apostar=False — trader ya rechazó este pick')
    continue
```

**Por qué**: Si el trader rechazó el pick para apuesta individual, no hay base para incluirlo en combos. El combo no tiene un filtro de Kelly propio — hereda el del trader.

### G2 — `n_h2h ≥ 1` mínimo (Gate H2H)

```python
if pick.get('n_h2h', 0) < 1:
    log_blocked(pick, 'G2: n_h2h=0 — sin historial directo, edge no verificable')
    continue
```

**Por qué**: Sin H2H, el edge es ruido de modelo. El **Nodo-33** ya bloquea picks individuales con coin-flip; los combos necesitan el mismo gate. Ver también **Nodo-49**.

**Excepción**: picks con `confidence_flag=STRONG` AND `n_axes_active ≥ 3` AND `score_directo ≥ 3` pueden entrar con n_h2h=0 (triple convergencia confirma sin H2H). Documentar como override explícito.

### G3 — `n_axes_active ≥ 2` (Gate multi-señal, N28F2)

```python
if pick.get('n_axes_active', 0) < 2:
    log_blocked(pick, 'G3: n_axes_active<2 — N28F2: BBI sola no predice')
    continue
```

**Por qué**: Regla N28F2 del **Nodo-28** ya establece que BBI sola no predice. El combo builder debe respetar esta regla que el clasificador individual ya aplica.

### G4 — `alignment_flag != NO_ALIGNMENT` (Gate convergencia)

```python
if pick.get('alignment_flag') == 'NO_ALIGNMENT':
    log_blocked(pick, 'G4: NO_ALIGNMENT — señales internas no convergen')
    continue
```

**Por qué**: `alignment_flag=NO_ALIGNMENT` con `net_alignment ≤ 0` significa que las señales activas están en contra del pick. Bartel tenía `net_alignment: -0.286`. Ver **Nodo-65**.

### G5 — Límite de repetición por pick: máximo 2 combos (Gate correlación)

```python
# Contar cuántos combos ya incluyen este pick
apariciones = sum(1 for c in combos_generados if partido in c['legs'])
if apariciones >= 2:
    log_blocked(pick, 'G5: pick ya en 2 combos — cap correlación')
    continue
```

**Por qué**: Un pick que aparece en N combos crea correlación ρ=1.0 en esa dimensión. Schueller apareció en 5 combos el 15-jul. Cuando perdió, 5 combos cayeron. Ver **Nodo-74** (governor) y **Nodo-38** (portfolio isolation).

---

## 5. DECISIONES (D103-01 → D103-08)

| ID | Decisión | Fecha | Estado |
|---|---|---|---|
| D103-01 | Gate G1 (`apostar: True`) — `_apply_combo_gates()` en `combo_confianza_builder.py` | 2026-07-16 | ✅ IMPLEMENTADO |
| D103-02 | Gate G2 (`n_h2h ≥ 1`) con excepción triple-convergencia (STRONG+axes≥3+sd≥3) | 2026-07-16 | ✅ IMPLEMENTADO |
| D103-03 | Gate G3 (`n_axes_active ≥ 2`) respetando N28F2 | 2026-07-16 | ✅ IMPLEMENTADO |
| D103-04 | ~~Gate G4 (`alignment_flag != NO_ALIGNMENT`)~~ → RECALIBRADO por D103-08 | 2026-07-16 | SUPERSEDED |
| D103-05 | Gate G5 (máx 2 combos por pick) — post-proceso en `main()` | 2026-07-16 | ✅ IMPLEMENTADO |
| D103-06 | Log de picks bloqueados → `reports/combo_gate_log_FECHA.json` | 2026-07-16 | ✅ IMPLEMENTADO |
| D103-07 | Retroalimentar `scripts/signal_audit.py` con no-bets bloqueados (trazabilidad) | 2026-07-16 | ✅ IMPLEMENTADO |
| D103-08 | G4 recalibrado: `NO_ALIGNMENT AND net_alignment < -0.10` — evidencia de campo Jul-16 | 2026-07-17 | ✅ IMPLEMENTADO |

**Verificación retroactiva 2026-07-15**: Bartel BLOQUEADO G1, Lekomtseva BLOQUEADO G1, Schueller BLOQUEADO G1, Charrier BLOQUEADO G2, Jang BLOQUEADO G2. Los 5 combos perdedores (~$2,500) hubieran sido eliminados o reducidos.

---

## 5b. EVIDENCIA DE CAMPO — D103-08 (2026-07-17)

### Contexto: combos ganadores 2026-07-16

Combos del 16-jul ganaron. Análisis post-facto reveló que el gate G4 original (`alignment_flag == 'NO_ALIGNMENT'`) era demasiado agresivo: habría bloqueado **Gaines Jr** (el pick ancla de 7/8 combos ganadores).

### Por qué G4 original era incorrecto

`alignment_flag = NO_ALIGNMENT` no implica que las señales estén **en contra** del pick — solo que no convergen positivamente. El campo relevante es `net_alignment`:

| Pick | Fecha | `alignment_flag` | `net_alignment` | Resultado |
|---|---|---|---|---|
| Gaines Jr | Jul-16 | NO_ALIGNMENT | **0.0** (neutral) | GANO |
| Zantedeschi | Jul-16 | PARTIAL_ALIGNMENT | +0.282 | GANO |
| Bernard | Jul-16 | NO_ALIGNMENT | **0.0** (neutral) | GANO |
| Forbes | Jul-16 | NO_ALIGNMENT | +0.020 | GANO |
| Bartel | Jul-15 | NO_ALIGNMENT | **-0.286** (señales en contra) | PERDIO |

### Discriminador correcto

El riesgo real no es la **ausencia** de alineamiento (net=0.0) sino la **contradicción activa** (net < 0). El umbral -0.10 captura picks donde señales activas votan contra la dirección del modelo.

```python
# G4 recalibrado (D103-08) — discriminador: net_alignment < -0.10
if alignment == 'NO_ALIGNMENT':
    net = float(edge_data.get('net_alignment') or 0.0)
    if net < -0.10:
        return True, f'G4: NO_ALIGNMENT net={net:.3f}<-0.10 — señales activamente en contra'
    # net >= -0.10 → señales neutrales o ausentes, NO bloquear
```

### Verificación retroactiva

- **Jul-15 Bartel**: net=-0.286 < -0.10 → BLOQUEADO (correcto)
- **Jul-16 Gaines Jr**: net=0.0 ≥ -0.10 → PERMITIDO (correcto, ganó)
- **Jul-16 Bernard**: net=0.0 ≥ -0.10 → PERMITIDO (correcto, ganó)

---

## 6. HIPÓTESIS PRE-REGISTRADA

**H103-01**: Aplicar G1+G2+G3 simultáneamente reduce hit% de picks que entran en combos de <40% a ≥55% (n_stop=50 combos settled post-gates).

Pre-registrada en `validation/preregistered_hypotheses.json` como H103-01 — 2026-07-16. ✅

---

## 7. CONEXIONES OCULTAS (no documentadas antes)

1. **N28F2 → Combo Builder**: La regla N28F2 del **Nodo-28** clasifica picks individuales pero no tiene hook en el combo builder. Gap: el builder no consulta `motivo_reclasificacion`. Fix: G3 cierra este gap.

2. **Nodo-33 gate lateral → Combo Builder**: El gate cuota<2.10 de Nodo-33 aplica a picks individuales. Los combos aceptan cuota 3.35–4.10 sin gate equivalente. Fix: G2 + G3 son el equivalente para combos.

3. **Nodo-74 Governor → repetición por pick**: El governor controla presupuesto total de sesión pero no límite de repetición por pick individual dentro del presupuesto. Fix: G5.

4. **signal_audit.py → no registra no-bets**: La trazabilidad actual solo registra picks que llegaron a shadow book. Los picks bloqueados por gates son invisibles — imposible aprender de ellos. Fix: D103-07.

---

## 8. ARCHIVOS A MODIFICAR

| Archivo | Cambio | Gate |
|---|---|---|
| `combo_confianza_builder.py` | `_filtrar_picks_para_combos()` con G1–G5 | G1–G5 |
| `combo_confianza_builder.py` | Log de bloqueados → `combo_gate_log_FECHA.json` | D103-06 |
| `scripts/signal_audit.py` | Registrar no-bets bloqueados por gate | D103-07 |
| `validation/preregistered_hypotheses.json` | H103-01 | — |

**No modificar:** `edge_calculator.py`, `trader_ev_tenis.py`, `rivalry_analyzer.py` — el problema está en el builder, no en el modelo de predicción.

---

## 9. REGLAS INMUTABLES QUE ESTE NODO NO TOCA

- **REGLA-HF-1**: cuota < 1.50 nunca en pool — ya existe, no cambiar
- **REGLA-HF-5**: KGR < 0 → NO DESPLEGAR — ya existe, no cambiar  
- **REGLA-T53**: tests invocan función real — aplicar a tests de G1–G5
- **SDD**: ningún código sin nodo — este nodo es el prerequisito

---

## 10. CONTEXTO DE NO-REPETICIÓN

Este nodo NO duplica:
- **Nodo-33**: gate individual cuota<2.10 coin-flip — diferente (individual vs combo)
- **Nodo-49**: fallback Playwright cuando n_h2h=0 — diferente (data fetch vs gate de combo)
- **Nodo-63**: insuficient history guard en anchor builder — diferente (anchor individual vs pool de legs)
- **Nodo-74**: governor de presupuesto de sesión — diferente (presupuesto total vs repetición por pick)
- **Nodo-86/87**: auditoría de 15 hallazgos del pipeline general — diferente (pipeline general vs combo builder específico)

Evidencia de pérdida: shadow_book/sb_2026-07-15.jsonl (36 registros, n_settled pendiente), combos reportados por usuario 2026-07-16 con cuotas 79.23 / 75.50 / 55.01 / 29.90 / 31.07 / 7.64.
