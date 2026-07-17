# Nodo-38 — Portfolio con Aislamiento de Riesgo: CORE / Satellite / Moonshot

> **Fecha inicio:** 2026-06-26
> **Severidad:** MEJORA ARQUITECTÓNICA — Nodo-37 implementó combos progresivos C5→C20 sin aislamiento de riesgo. El 26-jun Da Silva @3.60 estuvo en C7, C8 y C11 simultáneamente: cuando ganó (C7) = jackpot, cuando perdió (C8/C11) = destruyó combos con 7-9 piernas ganadoras. El problema no fue incluirlo — fue incluirlo en TODOS los combos sin aislamiento.
> **Prerequisitos:** Nodo-37 (combo_confianza_builder.py base), Nodo-15 (Cobertura por Exclusión)
> **Archivos modificados:** `combo_confianza_builder.py`
> **Archivos NO modificados:** `edge_calculator.py`, `trader_ev_tenis.py`, `betplay_combo_builder.py`
> **Tests:** `tests/test_nodo38.py`
> **Implementa:** Sonnet
>
> **Estado:** 🔄 EN CURSO

---

## 0. RESUMEN EJECUTIVO

Nodo-37 construía combos con una progresión plana C5→C20 donde picks de cuota 1.20 y picks de cuota 3.60 podían coexistir en el mismo combo. Esto genera riesgo de contaminación: un pick de alto valor que falla destruye un combo que tenía 7-9 piernas ganadoras.

Nodo-38 reemplaza esa progresión con una arquitectura de **aislamiento por tipo de riesgo**:

```
CORE:      solo Cat-A (1.15-1.59) + Cat-B (1.60-2.20) → NUNCA Cat-C
SATELLITE: Cat-A/B base + exactamente 1 Cat-C1 (2.20-3.50, conf≥60%)
MOONSHOT:  Cat-A base + 2-3 Cat-C (conf≥57%)
```

Si un pick Cat-C falla, solo su satellite muere. El CORE sobrevive intacto.

---

## 1. HALLAZGO QUE MOTIVÓ ESTE NODO

### 1.1 Evidencia empírica del 26-jun-2026

| Apuesta | Composición | Resultado | Lección |
|---|---|---|---|
| C4 Nodo-37 puro | Cat-C + Cat-B mezclados | GANÓ ~24x | Alpha real en underdogs |
| C3 Pipeline puro | 3 Cat-C pipeline | GANÓ 13.2x → $6,610 | Pipeline funciona |
| C7 Mixto | Da Silva @3.60 + 6 Cat-A/B | GANÓ ~63x | Cat-C + base = explosivo |
| C8 con Da Silva | Da Silva @3.60 + 7 Cat-A/B | PERDIÓ | 7/8 ganaron, Da Silva destruyó |
| C11 mezcla amplia | Da Silva @3.60 (Cat-C2) + Cardozo @2.55 (Cat-C1) + 9 Cat-A/B | PERDIÓ | 2 Cat-C en mismo combo — viola REGLA-ISO-2; ambos fallaron |

**Da Silva @3.60 estuvo en C7, C8 y C11. Cardozo @2.55 también estuvo en C11** — dos Cat-C en el mismo combo mezclado. Si hubieran estado cada uno SOLO en un satellite aislado, el CORE (los 9 picks ganadores) habría sobrevivido intacto.

### 1.2 Tres categorías empíricas de picks

```
CAT-A — MULTIPLICADORES (cuota 1.15-1.59):
  P(win) ≈ 63-87% (implícita del mercado)
  Función: multiplicar odds del combo sin agregar riesgo significativo
  Ejemplo: Kicker @1.16, Feistel @1.23, Wallin @1.43, Rehberg @1.52
  4 juntos = 3.1x de multiplicación con riesgo mínimo

CAT-B — VALOR (cuota 1.60-2.20):
  P(win) ≈ 45-63% (implícita) pero modelo los ve favoritos (conf≥53%)
  Función: corazón del combo, dan el retorno real
  Ejemplo: Weightman @1.85, Wazny @2.02, Kopp @1.62

CAT-C — ALTO VALOR (cuota >2.20):
  P(win) ≈ 28-45% (implícita) pero modelo los ve favoritos
  Función: alpha del modelo donde bookmaker tiene menos datos
  Subdivisión:
    CAT-C1 (satellite): cuota 2.20-3.50 AND confianza ≥60%
    CAT-C2 (moonshot):  cuota >3.50 OR confianza <60%
  Ejemplo: Yamalapalli @2.75 (C1), Da Silva @3.60 (C2)
```

---

## 2. DISEÑO DEL SISTEMA

### 2.1 Categorización de picks

```python
def _categorizar_pick(cuota, confianza, pipeline_picks, nombre):
    # Exclusiones absolutas
    if confianza < 53.0: return None
    if cuota < 1.15: return None
    if confianza < 55.0 and 1.55 <= cuota <= 1.70: return None  # parejo

    # Categorías
    if cuota <= 1.59: return 'CAT_A'
    if cuota <= 2.20: return 'CAT_B'

    # Subdivisión Cat-C
    # Señal doble (pipeline + Nodo-37) promueve Cat-C2 → Cat-C1
    if pipeline_flag and cuota <= 4.50 and confianza >= 57: return 'CAT_C1'
    if cuota <= 3.50 and confianza >= 60: return 'CAT_C1'
    return 'CAT_C2'
```

### 2.2 Arquitectura de combos

```
               ┌─────────────────────┐
               │       CORE          │
               │  Cat-A + Cat-B      │
               │  C4-C7, 45% budget  │
               │  NUNCA Cat-C        │
               └─────────────────────┘

  ┌──────────┐   ┌──────────┐   ┌──────────┐
  │  SAT-1   │   │  SAT-2   │   │  SAT-3   │
  │ 4×A/B    │   │ 4×A/B    │   │ 4×A/B    │
  │ +1×C1    │   │ +1×C1    │   │ +1×C1    │
  │ 15% ea   │   │ 15% ea   │   │ 15% ea   │
  └──────────┘   └──────────┘   └──────────┘

               ┌─────────────────────┐
               │     MOONSHOT        │
               │  3×Cat-A + 2-3×C   │
               │  conf≥57%, 5% bud  │
               └─────────────────────┘
```

### 2.3 Reglas de aislamiento

```
REGLA-ISO-1: Pick Cat-C NUNCA entra al CORE.
REGLA-ISO-2: Máximo 1 Cat-C por satellite.
REGLA-ISO-3: Un Cat-C puede estar en máximo 2 combos: su satellite + moonshot.
REGLA-ISO-4: Si ≥2 Cat-C del mismo torneo, solo 1 en moonshot.
REGLA-ISO-5: Cuota >4.00 solo moonshot (nunca satellite).
```

### 2.4 Guards reutilizados

```
TOURNAMENT-GUARD: Max 2 picks del mismo torneo en cualquier combo
                  (reutiliza Guard 2 de betplay_combo_builder.py)
PAREJO-GUARD:     confianza <55% AND cuota 1.55-1.70 → EXCLUIR
CORE-SIZE-GUARD:  CORE max 7 piernas (P(C8)≈27% es demasiado bajo)
P-CORE-GUARD:     Si P(CORE wins) < 25% → reducir tamaño hasta P>25%
```

### 2.5 VaR Guard

```python
MAX_DAILY_PCT = 0.12  # 12% bankroll (deja espacio para pipeline 10%)
# Si Nodo-37 + pipeline > 25% bankroll, Nodo-37 escala down
```

### 2.6 Integración con pipeline (Protocolo D)

Cuando un pick aparece en edge_report.apostar/watchlist Y en el pool Nodo-37:
- No se cambia el stake
- Se PROMUEVE la categoría: Cat-C2 con pipeline edge → Cat-C1 (si cuota≤4.50, conf≥57%)
- Guard: exposición total a un pick (pipeline + Nodo-37) ≤ 5% bankroll

---

## 3. PROTOCOLO DE ESCALADO

```
FASE 1 (días 1-7):  solo CORE,              2% bankroll/día
FASE 2 (días 8-14): CORE + 1 satellite,     4% bankroll/día
FASE 3 (días 15-21): CORE + 3 SAT + MOON,   7% bankroll/día
FASE 4 (día 22+):   todo + cobertura,       12% bankroll/día
```

Gates de salida por fase:
- 1→2: accuracy Cat-AB ≥80% (n≥30), 0 circuit breakers en 5 días
- 2→3: accuracy Cat-C1 ≥55% (n≥8), ROI acumulado >0, n_total≥70
- 3→4: ROI últimas 2 semanas >0, n_total≥100, accuracy global ≥75%

Circuit breakers:
- CB-1: accuracy Cat-AB rolling-20 < 70% → pausar 2 días
- CB-2: 3 días consecutivos pérdida neta → bajar 1 fase
- CB-3: accuracy Cat-C rolling-10 < 40% → pausar moonshot
- CB-4: bankroll cayó >15% desde inicio de fase → bajar 1 fase

---

## 4. MANEJO DE NULOS

- Picks nulos (partido cancelado/suspendido): NO cuentan para accuracy
- Betplay recalcula combo dividiendo cuota total por cuota del pick nulo
- Satellite con Cat-C nulo: sobrevive como combo reducido
- Max 2 picks del mismo torneo por combo (guard de concentración) limita daño por nulos

---

## 5. REFINAMIENTO CAT-C SUBDIVISIÓN

**Hipótesis elegida: B modificada** (cuota + confianza, sin filtro por tier)

| Cuota | Confianza | Pipeline | → Categoría | Combos |
|---|---|---|---|---|
| 2.21-3.50 | ≥60% | — | CAT_C1 | SATELLITE, MOONSHOT |
| 2.21-3.50 | 53-59% | No | CAT_C2 | MOONSHOT |
| 2.21-3.50 | 57-59% | Sí | CAT_C1 | SATELLITE, MOONSHOT |
| 2.21-4.50 | ≥57% | Sí | CAT_C1 | SATELLITE, MOONSHOT |
| 3.51-4.50 | ≥60% | No | CAT_C2 | MOONSHOT |
| >4.50 | cualquiera | — | CAT_C2 | MOONSHOT |

**Razón:** el tier afecta la eficiencia del bookmaker, no la calidad de la predicción
una vez filtrada por confianza. Filtrar por tier sería doble-contar información ya
capturada en cuota y confianza.

---

## 6. PENDIENTES

| # | Acción | Estado |
|---|---|---|
| 1 | Refactor `_build_portfolio()` → `_build_portfolio_v2()` con CORE/SAT/MOON | 🔄 En curso |
| 2 | `_categorizar_pick()` con todos los guards | 🔄 En curso |
| 3 | VaR guard automático (MAX_DAILY_PCT=0.12) | 🔄 En curso |
| 4 | Cross-reference con edge_report para pipeline_picks y cuotas Kambi | 🔄 En curso |
| 5 | Tests `tests/test_nodo38.py` — 15+ tests | ⏳ Pendiente |
| 6 | Acumular accuracy en `calibracion_edge.json` sección `combo_confianza` | ⏳ Pendiente |
| 7 | Actualizar Nodo-37 como SUPERSEDED por Nodo-38 | ⏳ Pendiente |

---

## 7. WIKILINKS

- [[Nodo-37-Combo-Confianza-Builder]] — Nodo original que Nodo-38 refactoriza
- [[Nodo-15-Portfolio-HedgeFund]] — Cobertura por Exclusión reutilizada
- [[Nodo-32-Calibracion-Pipeline-Señales-Rotas]] — Phantom edge gate
- [[MOC-Principal]] — índice de specs
