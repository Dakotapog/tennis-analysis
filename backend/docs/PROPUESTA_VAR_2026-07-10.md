# PROPUESTA VAR — Diagnóstico APROBADO vs WATCHLIST
**Fecha:** 2026-07-10  
**Período analizado:** 2026-07-02 → 2026-07-10 (9 días, 146 picks settled)  
**Hipótesis de referencia:** H54-01 (pre-registrada 2026-07-03)  
**Estado:** SOLO DIAGNOSTICO — sin cambios a umbrales

---

## 1. Estado de H54-01

```
nombre:    "APOSTAR con stake_real=0 tienen hit% y CLV iguales o mejores que APOSTAR financiados"
medicion:  var_flattened=true AND apostar=true
n_stop:    30
estado:    ACUMULANDO
n_actual:  0   ← NUNCA ha recibido un dato
```

**Hallazgo inmediato:** H54-01 tiene n=0 a pesar de que 8 picks en el JSONL tienen
`var_flattened=True`. La hipótesis mide el segmento correcto pero el hypothesis_tracker
no está leyendo automáticamente del shadow_book — requiere actualización manual vía
`llr_update()` o equivalente. Los datos YA EXISTEN en el JSONL pero no se transfirieron.

---

## 2. Tabla completa — APROBADO vs WATCHLIST (picks settled con resultado)

### shadow_book --report (fuente de verdad agregada)

| Segmento    | n  | hit%  | IC95           | breakeven | ROI flat |
|-------------|-----|-------|----------------|-----------|----------|
| APROBADO    | 15  | 33.3% | [15.2%, 58.3%] | 35.8%     | -20.1%   |
| WATCHLIST   | 82  | 39.0% | [29.2%, 49.8%] | 34.7%     | +7.9%    |
| NO_DATA     | 49  | 46.9% | [33.7%, 60.6%] | 36.0%     | -0.8%    |

**WATCHLIST supera a APROBADO en hit% Y en ROI.**  
IC95 de WATCHLIST [29.2, 49.8] cruza breakeven=34.7% → evidencia positiva pero
estadísticamente no concluyente todavía.

### Picks WATCHLIST-VAR (var_flattened=True en JSONL — el núcleo de H54-01)

| Fecha      | Jugador                | Tier | Edge   | Cuota | Conf     | kelly_kl | Resultado |
|------------|------------------------|------|--------|-------|----------|----------|-----------|
| 2026-07-03 | Aziz Ouakaa            | itf  | 7.0%   | 1.74  | STRONG   | 0.1012   | WON       |
| 2026-07-03 | Maria Sara Popa        | itf  | 6.4%   | 2.04  | MODERATE | 0.0843   | WON       |
| 2026-07-08 | Pieter de Lange        | itf  | 20.1%  | 2.55  | MODERATE | 0.3210   | LOST      |
| 2026-07-08 | Xiang Liu              | itf  | 25.9%  | 2.95  | MODERATE | 0.3152   | LOST      |
| 2026-07-08 | Marilouise Van Zyl     | itf  | 10.0%  | 2.04  | MODERATE | 0.1449   | LOST      |
| 2026-07-08 | Morris A.              | itf  | 53.2%  | 3.45  | STRONG   | 0.3700   | LOST      |
| 2026-07-09 | Leyton Rivera          | itf  | 39.7%  | 4.35  | STRONG   | 0.4857   | WON       |
| 2026-07-09 | Maria Luisa Oliveira   | itf  | 47.0%  | 4.80  | STRONG   | 0.4525   | LOST      |

**WATCHLIST-VAR: n=8, hits=3, hit%=37.5%**  
APROBADO real (stake_real>0): **n=1** (Van Der Meerschen, challenger, $6,000, LOST)

---

## 3. Patrón en los datos

**¿Los picks WATCHLIST-VAR comparten algo?**
- Tier: **100% ITF** (8/8)  
- Cuota: 1.74 → 4.80 (rango amplio, no es el filtro que discrimina)  
- Edge: 6.4% → 53.2% (también amplio — no es el edge lo que los mata)  
- Conf: MODERATE y STRONG (no es confianza baja)  
- **Lo que los une: bankroll pequeño (ITF = $10,000) + kelly_kl bajo (0.08→0.48)**

**¿Los picks APROBADO que perdieron comparten algo?**  
Solo hay 1 pick APROBADO real (n=1 es demasiado pequeño para patrón).  
Los 15 "APROBADO" del report incluyen picks pre-trader-deploy (stake_real=None ≠ stake_real=0).  
El sistema confunde "pasó los gates de kelly" con "efectivamente apostado" — son distintos.

**¿Es un tier específico?**  
Sí. El problema es **casi exclusivamente ITF**. Challenger y GS tienen bankrolls
suficientemente grandes para sobrevivir el waterfall.

---

## 4. Traza del waterfall — por qué ITF siempre cae en MIN_BET_CLIFF

### Fórmula real (trader_ev_tenis.py:1123-1126)

```
stake_final = round( kelly_kl × bankroll × portfolio_factor
                     × var_factor                          (VaR constraint)
                     × cppi_factor                         (CPPI floor)
                     / MIN_BET ) × MIN_BET

si stake_final == 0 y stake_pre > 0 → MIN_BET_CLIFF (var_flattened=True)
```

### Valores fijos actuales

| Parámetro       | Valor  | Fuente                        |
|-----------------|--------|-------------------------------|
| MIN_BET         | $1,000 | trader_ev_tenis.py:43         |
| MAX_VAR_PCT     | 25%    | trader_ev_tenis.py:157        |
| _CPPI_FLOOR_PCT | 70%    | trader_ev_tenis.py:55 PROVISIONAL |
| _CPPI_MULTIPLIER| 2.0    | trader_ev_tenis.py:56 PROVISIONAL |
| cppi_factor     | **0.60** | bankroll=peak → cushion=0.30 → 2×0.30=0.60 |
| var_factor      | **0.25** | cuando VaR excede MAX_VAR_PCT |

### Waterfall combinado cuando VaR está excedido

```
multiplicador_efectivo = var_factor × cppi_factor = 0.25 × 0.60 = 0.15

Para que un pick sobreviva MIN_BET_CLIFF:
  stake_pre_var × 0.15 ≥ $1,000
  stake_pre_var ≥ $6,667
```

### Umbral de kelly_kl necesario por tier

| Tier       | Bankroll | kelly_kl mínimo para sobrevivir | ¿Posible? |
|------------|----------|----------------------------------|-----------|
| ITF        | $10,000  | 66.7%  (6,667/10,000)           | **NO** — Kelly nunca va tan alto con λ=4.5 |
| Challenger | $20,000  | 33.3%  (6,667/20,000)           | Marginal — solo picks extremos |
| ATP500     | (var)    | depende del bankroll configurado | Posible |
| Grand Slam | $125,000 | 5.3%   (6,667/125,000)          | **SÍ** — umbral realista |

### Traza concreta: Leyton Rivera (2026-07-09, ITF, WON)

```
edge=39.7%  cuota=4.35  kelly_kl=0.4857  bankroll_ITF=$10,000

stake_pre_var = 0.4857 × 10,000 × portfolio_factor ≈ $4,857  (asumiendo factor≈1)
stake_post_var  = $4,857 × 0.25 = $1,214
stake_post_cppi = $1,214 × 0.60 = $728

$728 < MIN_BET=$1,000 → CLIFF → stake_final = $0 → var_flattened=True

Pick ganó. Ganancia perdida: $728 × (4.35-1) = ~$2,437 netos (estimado)
```

### Traza concreta: Maria Sara Popa (2026-07-03, ITF, WON)

```
edge=6.4%  cuota=2.04  kelly_kl=0.0843  bankroll_ITF=$10,000

stake_pre_var = 0.0843 × 10,000 = $843
→ YA está por debajo de $6,667 antes de cualquier var_factor
→ $843 × 0.25 × 0.60 = $126 < $1,000 → CLIFF
```

---

## 5. Diagnóstico — causa raíz

### Conclusión: **(b) — El problema es ANTERIOR al VaR**

El VaR (MAX_VAR_PCT=25%) es el mecanismo correcto. El problema es que MIN_BET=$1,000
es un umbral absoluto que no está calibrado por tier.

**Causa raíz específica: MIN_BET no escala con el bankroll por tier.**

- Para GS ($125k): MIN_BET = 0.8% del bankroll → umbral benigno
- Para ITF ($10k): MIN_BET = 10% del bankroll → umbral prohibitivo

El CPPI (factor=0.60 PROVISIONAL) amplifica el problema pero no lo causa. Incluso sin
CPPI (factor=1.0), ITF necesitaría kelly_kl ≥ 25% para sobrevivir, que sigue siendo
alto para picks marginales de edge 5-15%.

### Por qué WATCHLIST supera a APROBADO

WATCHLIST (n=82) incluye picks con `apostar=False` — rechazados por baja confianza o
edge insuficiente. El hecho de que su hit% (39%) supere a APROBADO (33.3%) sugiere
que el filtro pre-stake está eliminando picks de calidad real junto con los de baja
calidad. El clasificador de APROBADO/WATCHLIST en el shadow_book usa `apostar` como
flag binario, pero lo que termina apostado (stake_real>0) es casi cero pick en 9 días.

**En síntesis:** el sistema está en modo "observación pagada" — los 82 WATCHLIST son
datos gratis, pero el P&L real viene del 1 pick que llegó a stake>0. Eso es riesgo
de muestra, no señal de mala calibración del VaR.

---

## 6. ¿Cuántas apuestas más se necesitan?

Para concluir con solidez (poder estadístico 80%, α=0.05, diferencia detectada 10pp):

- APROBADO vs WATCHLIST: n≥65 por segmento (ahora: n=15 vs 82 — APROBADO insuficiente)
- H54-01 específicamente (var_flattened): n≥30 (definido en la hipótesis)

**Con el ritmo actual (8 picks WATCHLIST-VAR en 9 días) → ~11 días más para n=30.**

---

## 7. Propuesta escrita (sin aplicar)

### Opción A — MIN_BET proporcional por tier (cambio mínimo, mayor impacto)

```python
# En trader_ev_tenis.py — reemplazar MIN_BET fijo por dict por tier
_MIN_BET_BY_TIER = {
    'itf':        100,   # 1% de bankroll $10k
    'challenger': 200,   # 1% de bankroll $20k
    'atp500':     500,
    'atp1000':    750,
    'grand_slam': 1000,
}
```

Riesgo: reduce el piso de stake en ITF. Picks de edge<5% que hoy se eliminan podrían
colarse. Requiere Nodo nuevo + H54-01 graduada antes de aplicar.

### Opción B — Reducir CPPI floor de 70% a 50% (cambia cppi_factor de 0.60 a 1.0)

```python
# En trader_ev_tenis.py:55
_CPPI_FLOOR_PCT = 0.50  # cushion=(1-0.50)=0.50 → factor=min(1,2×0.50)=1.0
```

Efecto: cppi_factor sube de 0.60 a 1.0. Waterfall combinado: 0.25×1.0=0.25 (vs 0.15
actual). ITF threshold baja de kelly≥66.7% a kelly≥25%. Más picks ITF sobrevivirían.
Riesgo: menos protección de drawdown (CPPI marcado PROVISIONAL — Nodo-70).

### Opción C — Esperar H54-01 (n=30) antes de cualquier cambio

No hacer nada hasta que H54-01 acumule n=30. Para eso, habilitar la lectura automática
del campo `var_flattened` desde el shadow_book JSONL al hypothesis_tracker. Con 8 picks
ya en el JSONL, faltarían ~22 más (~30 días al ritmo actual de ITF settled).

---

## 8. Acción inmediata recomendada — sin tocar umbrales

**Corregir H54-01 para que acumule los datos que ya existen:**

```bash
# Los 8 picks var_flattened=True ya están en el JSONL
# El hypothesis_tracker necesita leerlos — revisar si hypothesis_tracker.py
# tiene un modo de backfill desde shadow_book, o si requiere llamada manual:
python3 validation/hypothesis_tracker.py --update H54-01 2>/dev/null
```

Esto no cambia ningún umbral. Solo sincroniza el contador de H54-01 con la realidad.

---

**ESPERANDO CONFIRMACIÓN — NO EJECUTO NADA MÁS EN ESTE TURNO.**
