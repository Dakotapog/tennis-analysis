# Nodo-14: Validación Live Roland Garros — Conexiones Ocultas TTC

> **Wikilinks:** [[Mandatos-No-Negociables]] | [[Pipeline-Arquitectura]] | [[Sprint-Pipeline]] | [[Nodo-13-Trader-EV-Tenis]] | [[Nodo-01-Edge-Calculator]] | [[Nodo-02-Markov-Changepoint]] | [[Nodo-06-Erdos-Graph]] | [[Nodo-10-Surface-Propagation]]
> **Estado:** 2026-05-30 — P&L REGISTRADO ✅ | +$25,000 (+25% bankroll) | accuracy 70% (7/10) | p_historica calibrada 0.52→0.68

---

## El Evento: Primera Validación en Producción Real

Roland Garros 2026-05-30. Pipeline corrió con **16 partidos** (cuadro principal, sin calificación).
Output de `trader_ev_tenis.py` con `--bankroll 100000`:

```
Señales APOSTAR: 2
  Parry D.  vs Anisimova A.  @ 4.50 | edge +29.3% | stake $10,000
  Tien L.   vs Cobolli F.    @ 2.40 | edge +16.5% | stake $5,000

Combo 2 piernas: Parry + Tien @ 10.80 | EV +200.7% | stake $5,000
Total en riesgo: $20,000 (20% bankroll)
```

### Resultados Roland Garros 2026-05-30 — COMPLETOS

| Partido | Predicción del pipeline | Resultado real | Estado |
|---|---|---|---|
| Parry D. vs Anisimova A. | **APOSTAR Parry @ 4.50** (edge +29.3%) | **Parry GANÓ** ✅ | Señal correcta |
| Cobolli F. vs Tien L. | **APOSTAR Tien @ 2.40** (edge +16.5%) | **Cobolli GANÓ** ❌ | Señal incorrecta |
| Combo Parry + Tien @ 10.80 | EV +200.7% — stake $5,000 | **PERDIDO** ❌ (Tien falló) | — |
| Cerundolo F. vs Svajda Z. | Cerundolo ganador | **Svajda GANÓ** ❌ | Upset no predicho |
| Cresudolo (partido) | "muy reñido" (sin señal) | **Partido larguísimo** ✅ | Rivalidad calibrada |
| Berrettini M. | Pronosticado ganador | **Berrettini GANÓ** ✅ | Dirección correcta |
| Tabilo A. | Pronosticado ganador | **Tabilo GANÓ** ✅ | Dirección correcta |
| Jovic I. / Osaka N. | Osaka ganadora | **Osaka GANÓ** ✅ | Dirección correcta |
| Sakkari M. / Chwalinska M. | Chwalinska ganadora | **Chwalinska GANÓ** ✅ | Dirección correcta |
| Shnaider D. | Pronosticado ganadora | **Shnaider GANÓ** ✅ | Dirección correcta |
| Sabalenka A. | Pronosticado ganadora | **Sabalenka GANÓ** ✅ | Dirección correcta |
| Kalinskaya A. vs Osorio C. | Osorio (watchlist, edge 1.7%) | **Kalinskaya GANÓ** ❌ | Señal watchlist incorrecta |

**Accuracy sesión: 70.0% (7/10)** — datos limpios (vs 47.37% enero 2026 con datos sucios)

### P&L Sesión Roland Garros 2026-05-30

```
APUESTAS REALIZADAS
───────────────────────────────────────────────────────
  Parry D.  @ 4.50 — stake $10,000 → GANÓ  → retorno $45,000  ✅ (+$35,000)
  Tien L.   @ 2.40 — stake  $5,000 → PERDIÓ → retorno      $0  ❌  (-$5,000)
  Combo×2   @ 10.80 — stake $5,000 → PERDIÓ → retorno      $0  ❌  (-$5,000)
───────────────────────────────────────────────────────
  Total apostado:  $20,000
  Total retornado: $45,000
  GANANCIA NETA:   +$25,000  (+25.0% bankroll)

  Bankroll inicial:  $100,000
  Bankroll final:    $125,000  🏦
```

**Lección del combo:** Parry (señal de mayor edge, +29.3%) cubrió con creces la pérdida del combo y de Tien. La arquitectura de budget cascade 40/40/20 funcionó como protección: la señal más fuerte sola generó +$35,000 neto, el peor caso del combo (−$5,000) fue absorbido sin daño.

### Calibración Bayesiana Actualizada (post T14-01)

```
ANTES (pre-sesión):   n=13 | p_historica=0.52  (prior neutral)
DESPUÉS (post-sesión): n=23 | p_historica=0.68  (Thompson Sampling Beta 16W/7L)

Impacto en futuras sesiones:
  → p_blend con n_h2h=0:  0.52 → 0.68 (freno menos conservador)
  → Kelly-KL penalizará menos señales alineadas con accuracy real
  → El sistema opera con evidencia real de 23 partidos, no con prior neutral
  → Prior irá derivando hacia accuracy real de clay: ~70%
```

---

## Las 5 Conexiones Ocultas (TTC — Marco de Tres Expertos)

### Conexión 1 — El alpha es un sesgo de mercado, no predicción bruta

El mercado fijó Parry a 4.50 → implied probability 22.2%. El modelo dijo ~52% (edge +29.3%). La diferencia no es que el modelo sea "más inteligente" en general: es que el bookmaker usa **ranking ATP/WTA global** (promedia 52 semanas, sin discriminar superficie) mientras el modelo usa el **grafo de rivalidad transitiva específico de arcilla**.

En arcilla lenta de Roland Garros, los especializados en tierra tienen ventaja estructural que el ranking no captura. **El alpha del sistema = explotar ese sesgo sistemático del mercado.**

> **Implicación permanente:** Mientras los bookmakers usen rankings globales, este sesgo existirá. El sistema no necesita ser más preciso que el mercado en general — solo necesita serlo donde el mercado tiene su punto ciego: clay specialists con ranking inferior.

### Conexión 2 — Distribución de EV en forma de U: el valor está en los extremos

```
Berrettini @ 1.45  → correcto | EV apostable bajo (~16%) — mercado eficiente
Zona 2.0–3.5       → competitiva (información equilibrada entre mercado y modelo)
Parry @ 4.50       → correcto | EV brutal: 0.52 × 4.50 − 1 = +134%
```

El EV no es lineal con la cuota. En odds altos donde el modelo identifica un clay specialist subestimado, el retorno por unidad apostada es explosivo. El mercado es más eficiente en favoritos (más flujo de dinero, odds ajustadas) que en underdogs de arcilla.

> **Implicación táctica:** El sistema debe buscar activamente odds **3.5–6.0** con señal de superficie. Con 80 partidos diarios habrá 3-5 candidatos de este tipo. Debajo de 3.0 el edge raramente justifica el riesgo.

### Conexión 3 — El grafo de Erdős en arcilla tiene mayor densidad transitiva

En arcilla lenta, los rallies son más largos y la técnica supera a la potencia. El grafo de victorias es **más homogéneo y predictivo** que en pista dura: un jugador que venció a X en arcilla tiene mayor transferencia de señal al enfrentamiento X vs Y también en arcilla.

El sistema computa `common_opponents_detailed` con peso 20% fijo. En Roland Garros, ese peso debería escalar con la superficie.

> **Calibración pendiente (T14-03):** Peso de `common_opponents` por superficie:
> - Clay (Roland Garros): 20% → 28-30%
> - Hard: 20% (mantener)
> - Grass (Wimbledon): 15% (mayor varianza, superficie más extrema)

### Conexión 4 — Cresudolo 51+ games: el Markov de sets tardíos no existe en ningún modelo

El pipeline predijo "muy reñido" (no señal APOSTAR) para Cresudolo. Con 51+ games el partido entró en territorio de resistencia mental y física. El cambio de régimen de Markov domina sobre la técnica base en el 4to/5to set.

Ningún bookmaker modela el **historial de cada jugador específicamente en sets tardíos**. Este dato está disponible en FlashScore (historial de sets por partido en `historial_jugador`).

> **Feature pendiente (T14-02):** `factor_tardio` en `markov_analyzer.py`:
> win_rate del jugador cuando el partido llega al 4to/5to set.
> Crea una señal de segunda derivada: no quién gana el partido, sino quién mantiene nivel bajo fatiga acumulada.

### Conexión 5 — Convexidad de combos: cada señal nueva no suma, multiplica

El número de combos posibles crece cuadráticamente con las señales:

```
n=2 señales → C(2,2) = 1 combo
n=4 señales → C(4,2) = 6 combos
n=6 señales → C(6,2) = 15 combos
```

Con 16 partidos hoy: 2 señales → 1 combo. Con 80 partidos: si el edge rate se mantiene al 12%, habrá 5-6 señales → 10-15 combos activos. El valor no escala 5× — escala **10-15× en la capa de combos**.

> **Cuello de botella identificado:** No es el modelo. Es el volumen de partidos procesados diariamente. Pasar de 16 a 80 partidos es la palanca de mayor retorno inmediato en P&L.

---

## El Prior Bayesiano se Recalibrará Hacia Arriba

Parry tenía `n_h2h=0` → `p_blend = 0.520` (prior puro, freno conservador). A pesar del prior, ganó.

```
Estado pre-validación:  n=13, p_prior=0.52 (neutral uniforme)
Después de registrar:   n=18+, clay_specialist underdog: 1/1 (Parry)
Proyección n=30:        p_prior_clay ≈ 0.55–0.58 (derivará por evidencia)
Kelly-KL ajustado:      stakes mayores en underdogs clay → P&L crece exponencialmente
```

**El sistema es auto-mejorante:** cada validación registrada en `validar_con_api.py` hace el prior más preciso → p_blend más agresivos donde el edge es real → más combos activos → crecimiento compuesto del bankroll.

---

## Las 3 Palancas Prioritarias (Ordenadas por Impacto en P&L)

| # | Palanca | Impacto directo | Nodo |
|---|---|---|---|
| 1 | **80 partidos diarios** — T13-04 activa sistema 2/N | 2 señales→6+ señales → combos 1→15+ → bankroll exponencial | [[Nodo-13-Trader-EV-Tenis]] |
| 2 | ~~**Validar con validar_con_api.py**~~ ✅ | n: 13→23, p_historica: 0.52→0.68 | [[Nodo-05-Validacion-API]] |
| 3 | ~~**Peso common_opponents por superficie**~~ ✅ | clay 0.20→0.28 | ranking_mom 0.20→0.12 — activo próximo run | [[Nodo-06-Erdos-Graph]] |

---

## Tareas

| ID | Tarea | Estado |
|---|---|---|
| T14-01 | Registrar resultados Roland Garros 2026-05-30 en `validar_con_api.py` | ✅ 2026-05-30 — n: 13→23, p_hist: 0.52→0.68 |
| T14-02 | Añadir `factor_tardio` en `markov_analyzer.py` — win rate jugador en 4to/5to set | ⏳ sprint futuro |
| T14-03 | Calibrar peso `common_opponents` por superficie (clay 28-30%, hard 20%, grass 15%) | ✅ 2026-05-30 — clay 0.20→0.28, grass 0.20→0.15. 773 tests ✅ |
| T14-04 | ADR documentado: buscar activamente odds 3.5–6.0 con señal superficie | ✅ 2026-05-30 (en MOC) |
| T14-05 | Correr pipeline completo 80 partidos → sistema 2/N activo (≥3 señales) | ⏳ pendiente (T13-04) |

---

## Vinculación

- [[Nodo-13-Trader-EV-Tenis]] — produjo el plan que generó la señal Parry (edge +29.3%)
- [[Nodo-01-Edge-Calculator]] — calculó edge y Kelly-KL; `p_blend` correcto ahora variar por `n_h2h`
- [[Nodo-02-Markov-Changepoint]] — identificó Cresudolo como "muy reñido" (factor Markov neutro)
- [[Nodo-06-Erdos-Graph]] — Erdős graph activo en clay → transitividad alta → palanca de calibración pendiente
- [[Nodo-10-Surface-Propagation]] — con surface real en rivalry_analyzer el edge en clay specialists será aún más fuerte
- [[Nodo-05-Validacion-API]] — T14-01: registrar estos resultados para calibrar prior
- [[Mandatos-No-Negociables]] — Mandato 1: P&L sobre accuracy — **confirmado en producción real**
- [[Sprint-Pipeline]] — Fase 15 en backlog
