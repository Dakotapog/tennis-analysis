# Nodo-139 — Kambi-First Combo Builder: Inversión del Flujo de Señal

**Fecha:** 2026-07-22
**Estado:** ESPECIFICADO — pendiente implementación
**Autor-análisis:** Nivel doctoral (razonamiento extendido, sin presión de sprint)
**Wikilinks:** [[Nodo-40]] [[Nodo-100]] [[Nodo-103]] [[Nodo-110]] [[Nodo-118]] [[Nodo-136]] [[Nodo-138]]

---

## §1. Diagnóstico raíz — por qué el sistema actual no sirve

### 1.1 El flujo invertido (el error arquitectural)

Todos los builders actuales operan en la misma dirección equivocada:

```
FLUJO ACTUAL (ROTO):
  Nuestros picks (H2H scrapeado) → buscar en Kambi → falla si Kambi no los tiene
  Resultado: 0-4 picks encontrados de 22 picks, combos sin .bat

FLUJO CORRECTO:
  Kambi (138 favoritos disponibles AHORA) → aplicar nuestro modelo → combos desplegables
  Resultado: universo real de 138 picks, combos con outcome_id ya conocido
```

El error es de orden de operaciones. Kambi es la fuente de verdad del mercado apostable.
Nuestro modelo es el filtro de valor. No al revés.

### 1.2 El universo real que estamos ignorando

Evidencia empírica extraída 2026-07-22:
- **195 eventos Kambi hoy** (154 NOT_STARTED, 41 STARTED)
- **138 favoritos claros** en rango cuota [1.10, 1.80] todavía disponibles
- Distribuidos en 18 ventanas temporales (00:00Z → 23:00Z)
- Pico: 18:00Z = 23 picks | 01:00Z = 14 picks | 19:00Z = 14 picks

Sistema actual usó: **4 picks** de ese universo de 138. Eficiencia: 2.9%.

### 1.3 El error del tope de cuota combinada (COMBO_MAX_CUOTA=7.0)

**Raíz matemática:** Si cada pierna tiene edge positivo (p_modelo_i > 1/cuota_i), entonces
el combo tiene EV positivo. La demostración:

```
Para cada pierna i: p_i × q_i > 1  (condición de edge positivo)

EV_combo = ∏(p_i) × ∏(q_i) - 1
         = ∏(p_i × q_i) - 1
         > 1 - 1 = 0  ✓ (producto de factores > 1 es > 1)
```

**Conclusión:** El tope COMBO_MAX_CUOTA=7.0 es matemáticamente injustificado cuando
cada pierna tiene edge > 0. Eliminar el cap es la decisión correcta. Reemplazarlo
por gate de EV mínimo por combo.

### 1.4 El error de staking fijo

Los builders actuales usan $1,000-$5,000 por combo (arbitrario). El correcto es Kelly:

```
kelly_combo = EV_combo / (cuota_combo - 1)  [fracción óptima de bankroll]
stake_combo = bankroll × kelly_combo × 0.5   [half-Kelly para control de varianza]
MAX_STAKE   = min(stake_combo, bankroll × 0.03)  [cap absoluto 3% por combo]
```

### 1.5 El error de nombre-matching (raíz del "Sin outcome")

Nuestro formato: `"McFadzean L."` (Apellido Inicial.)
Kambi formato:   `"Lachlan Mcfadzean"` (Nombre Apellido)

La función `_find_outcome` tomaba el ÚLTIMO token como apellido → extraía `"l"` (inicial).
**Fix parcial implementado en D138-01 (combo_confianza_builder).** Nodo-139 formaliza
y centraliza el matching con un algoritmo robusto para todos los builders.

---

## §2. Arquitectura de la solución

### 2.1 Diagrama de flujo D139

```
Kambi listView (NOT_STARTED, 154 eventos)
    │
    ├─ D139-01: _fetch_kambi_betting_universe()
    │    → Lista de KambiLeg: {event_id, partido, player_fav, player_dog,
    │                           cuota_fav, cuota_dog, outcome_id_fav,
    │                           outcome_id_dog, start_utc, group_path}
    │    → Filtro: state=NOT_STARTED, cuota_fav ∈ [1.10, 1.80]
    │    → 138 picks disponibles
    │
    ├─ D139-02: _match_to_predictions(kambi_legs, edge_report, h2h_data)
    │    → Para cada KambiLeg: buscar en edge_report y H2H por apellido
    │    → TIER_A: match edge_report  → señales completas (edge, kelly, conf, axes)
    │    → TIER_B: match H2H data    → h2h_win_rate + surface + ELO proxy
    │    → TIER_C: sin match         → EXCLUIR (no apostar sin señal propia)
    │
    ├─ D139-03: _compute_leg_signal(matched_leg) → ScoredLeg
    │    → TIER_A score = edge (×3) + kelly_kl (×2) + conf_numeric (×1)
    │    → TIER_B score = (h2h_wr - 1/cuota_kambi) × 5  [relative edge]
    │    → Gate mínimo: edge_efectivo > 0 AND p_efectivo ≥ 0.55
    │
    ├─ D139-04: _build_kambi_combos(scored_legs, bankroll)
    │    → Agrupa por ventana temporal ±3h
    │    → Para cada grupo ≥3 legs: genera combinaciones de 3 a MAX_LEGS=7
    │    → Filtra por EV_combo > 0.05 (5% EV mínimo)
    │    → Ordena por EV × kelly (retorno esperado ponderado por confianza)
    │    → Top-10 combos, solape máximo 2 piernas entre cualquier par
    │
    ├─ D139-05: _kelly_stake(combo, bankroll, n_simultaneous)
    │    → kelly_combo = EV_combo / (cuota_combo - 1)
    │    → portfolio_factor = 1 / (1 + 0.15 × (n-1))
    │    → stake = bankroll × kelly_combo × 0.5 × portfolio_factor
    │    → Bounded: [500, bankroll × 0.03]  [$500 mínimo, 3% máximo]
    │
    └─ D139-06: _generate_kambi_first_bat(combos)
         → Cada combo tiene outcome_ids ya conocidos desde D139-01
         → .bat en Desktop: KB_1.bat, KB_2.bat, ... KB_N.bat
         → .html: KB_report_FECHA.html con tabla resumen
```

### 2.2 Tres niveles de señal (TIER_A / TIER_B / TIER_C)

| Tier | Fuente | Señal disponible | Gate adicional |
|------|--------|-----------------|----------------|
| **TIER_A** | edge_report match | edge, kelly_kl, conf, n_axes, score_directo | edge > 0 AND kelly_kl > 0 |
| **TIER_B** | H2H data match | h2h_win_rate, surface_wr, n_matches | h2h_win_rate > 1/cuota_kambi AND n_matches ≥ 5 |
| **TIER_C** | Sin match | — | **EXCLUIDO siempre** |

Ratio esperado hoy: 22 picks edge_report → ~15-20 TIER_A matches en Kambi.
H2H data (302 matches) → ~30-50 TIER_B adicionales.
Total pool efectivo: ~45-70 picks con señal.

---

## §3. Implementación detallada

### D139-01 — `_fetch_kambi_betting_universe()`

**Archivo:** `betplay_combo_builder.py` (nueva función, ~50 líneas)

```python
def _fetch_kambi_betting_universe(
    min_cuota: float = 1.10,
    max_cuota: float = 1.80,
) -> list[dict]:
    """
    Devuelve lista de KambiLeg para todos los eventos NOT_STARTED con
    cuota favorito en [min_cuota, max_cuota].

    Cada KambiLeg:
    {
        'event_id':       int,
        'partido':        str,  # "Player A - Player B"
        'player_fav':     str,  # Kambi label del favorito
        'player_dog':     str,  # Kambi label del rival
        'cuota_fav':      float,
        'cuota_dog':      float,
        'outcome_id_fav': str,   # ID para el .bat
        'outcome_id_dog': str,
        'start_utc':      str,   # ISO 8601
        'group_path':     str,   # ej: "ITF / Men Singles"
        'p_implied_fav':  float, # 1/cuota_fav (sin vig)
    }
    """
    ...
    # Criterio de label para match-winner:
    LABELS_MATCH = ('Match', 'Cuotas del partido', 'Match Betting', '1X2')
    ...
    # IMPORTANTE: state == 'NOT_STARTED' únicamente
    # Partidos STARTED excluidos — el usuario no puede apostar retroactivamente
```

**Thresholds justificados:**
- `min_cuota=1.10`: por debajo de 1.10 el vig (5-7%) elimina prácticamente todo el edge
- `max_cuota=1.80`: más allá de 1.80 el favorito deja de ser "claro" — aumenta ruido

### D139-02 — `_match_to_predictions(kambi_legs, edge_report, h2h_data)`

**Archivo:** `betplay_combo_builder.py` (nueva función, ~80 líneas)

```python
def _apellido_kambi(label: str) -> str:
    """
    Kambi: "Lachlan Mcfadzean" → "mcfadzean" (último token significativo).
    Excluye tokens ≤2 chars (iniciales o partículas cortas).
    Para apellidos compuestos como "Van De Zandschulp": tomar últimos 2 tokens.
    """
    norm = _normalize_name(label)
    parts = [p for p in norm.split() if len(p) > 2]
    if not parts:
        return norm
    # Si el último token parece partícula (len<=3 AND no es el único): tomar 2 finales
    if len(parts) >= 2 and len(parts[-1]) <= 3:
        return ' '.join(parts[-2:])
    return parts[-1]


def _apellido_pick(nombre: str) -> str:
    """
    Nuestro formato: "McFadzean L." → "mcfadzean"
                     "van Loben Sels E." → "loben sels" (o "van loben sels")
    Algoritmo: quitar tokens ≤2 chars del FINAL, tomar el primer token restante.
    """
    norm = _normalize_name(nombre)
    parts = norm.split()
    while parts and len(parts[-1]) <= 2:
        parts.pop()
    if not parts:
        return norm
    # Para apellidos compuestos (≥2 tokens restantes): tomar todos
    return ' '.join(parts)


def _match_score_names(kambi_label: str, pick_nombre: str) -> float:
    """
    0.0 = sin coincidencia
    0.9 = coincidencia parcial (apellido en apellido)
    1.0 = coincidencia exacta de apellido

    Para apellidos compuestos: match si el apellido_pick está CONTENIDO
    en apellido_kambi o viceversa (ej: "loben sels" in "van loben sels").
    """
    ak = _apellido_kambi(kambi_label)
    ap = _apellido_pick(pick_nombre)
    if ak == ap:
        return 1.0
    if ap in ak or ak in ap:
        return 0.9
    # Fuzzy: Jaccard de bigramas
    def bigrams(s):
        return set(s[i:i+2] for i in range(len(s)-1))
    bg_ak = bigrams(ak)
    bg_ap = bigrams(ap)
    if not bg_ak or not bg_ap:
        return 0.0
    jacc = len(bg_ak & bg_ap) / len(bg_ak | bg_ap)
    return jacc if jacc >= 0.7 else 0.0


def _match_to_predictions(kambi_legs, edge_report_picks, h2h_players):
    """
    Para cada KambiLeg, buscar predicción en:
    1. edge_report_picks: lista de picks de reports/edge_report_*.json (apostar+watchlist)
    2. h2h_players: dict apellido→{win_rate, surface_wr, n_matches, elo}

    Retorna lista de ScoredLeg con 'tier', 'p_efectivo', 'edge_efectivo', etc.
    """
```

**Precedencia de matching:**
1. Si pick de edge_report tiene `apostar=True` y match_score ≥ 0.85 → TIER_A
2. Si pick de edge_report tiene `apostar=False` (watchlist) y match ≥ 0.85 → TIER_A (solo señal, sin restricción de apostar)
3. Si player en h2h_data con n_matches ≥ 5 y match ≥ 0.85 → TIER_B
4. Sin match → TIER_C (excluir)

**Nota crítica sobre `apostar=False`:** El gate `apostar=True` de G1 en combo_confianza_builder
se aplica cuando el trader rechazó el pick INDIVIDUAL. Para Kambi-first combos, la lógica
es diferente — la señal individual puede ser insuficiente para apuesta individual pero
válida como pierna de combo cuando el combo compensa con diversificación. Por tanto,
en TIER_A de Nodo-139 incluimos también picks watchlist (no solo apostar=True).

### D139-03 — `_compute_leg_signal(matched_leg)` → ScoredLeg

```python
def _compute_leg_signal(matched_leg: dict) -> dict | None:
    """
    Retorna ScoredLeg o None si no pasa gates.

    Gates (en orden):
    G_CUOTA:  cuota_kambi ≥ 1.10 (ya filtrado en D139-01)
    G_EDGE:   edge_efectivo > 0  (p_efectivo > p_implied)
    G_CONF:   p_efectivo ≥ 0.55 (no coinflip)
    G_TIER_B: TIER_B requiere n_matches ≥ 5 Y h2h_win_rate ≥ 0.55

    Score compuesto (0-10 escala):
    TIER_A: score = edge_efectivo×3 + kelly_kl×20 + (1 si conf=STRONG else 0.5)
    TIER_B: score = (h2h_win_rate - p_implied)×5

    Campos de retorno:
    {
        ...campos de KambiLeg...,
        'tier':          'A' | 'B',
        'p_efectivo':    float,   # p_modelo (TIER_A) o h2h_win_rate (TIER_B)
        'edge_efectivo': float,   # p_efectivo - 1/cuota_kambi
        'kelly_kl':      float,   # de edge_report (TIER_A) o calculado simple (TIER_B)
        'score':         float,   # para ordenar
        'conf_flag':     str,     # de edge_report o 'H2H' para TIER_B
        'n_legs_ok':     bool,    # True = pasa todos los gates
    }
    """
```

**Cálculo de kelly para TIER_B** (sin Kelly-KL completo, usar Kelly clásico con shrinkage):
```python
f_raw  = edge_efectivo / (cuota_kambi - 1)      # Kelly clásico
n_obs  = h2h_data['n_matches']
shrink = n_obs / (n_obs + 20)                    # Shrinkage Nodo-63
kelly_b = f_raw × shrink                          # Kelly-TIER_B
```

### D139-04 — `_build_kambi_combos(scored_legs, bankroll)`

**Sin tope de cuota combinada.** Gate reemplazante:

```python
# Gate correcto: EV mínimo del combo (no tope de cuota)
EV_combo = p_combo × cuota_combo - 1
assert EV_combo >= EV_MIN_COMBO  # EV_MIN_COMBO = 0.02 (2%)

# Gate de probabilidad mínima: con ≥7 piernas, p_combo puede ser muy bajo
MIN_P_COMBO = {3: 0.10, 4: 0.08, 5: 0.06, 6: 0.04, 7: 0.03}
assert p_combo >= MIN_P_COMBO[n_legs]
```

**Algoritmo de construcción:**

```python
def _build_kambi_combos(scored_legs, bankroll, n_legs_min=3, n_legs_max=7):
    # 1. Ordenar legs por score descending
    legs = sorted(scored_legs, key=lambda x: x['score'], reverse=True)

    # 2. Agrupar por ventana temporal ±3 horas
    groups = _group_by_time_window(legs, window_hours=3)

    all_combos = []
    for group in groups:
        if len(group) < n_legs_min:
            continue  # grupo muy pequeño

        # 3. Generar todas las combinaciones en el grupo
        for n in range(n_legs_min, min(n_legs_max, len(group)) + 1):
            for combo_legs in combinations(group, n):
                # Diversificación: máx 1 pierna por jugador
                jugadores = [_apellido_pick(l['player_fav']) for l in combo_legs]
                if len(set(jugadores)) < n:
                    continue  # jugador repetido

                p_combo    = math.prod(l['p_efectivo'] for l in combo_legs)
                cuota_combo = math.prod(l['cuota_fav'] for l in combo_legs)
                EV_combo   = p_combo * cuota_combo - 1

                # Gates sin tope de cuota
                if EV_combo < EV_MIN_COMBO:
                    continue
                if p_combo < MIN_P_COMBO.get(n, 0.03):
                    continue

                all_combos.append({
                    'legs':       list(combo_legs),
                    'p_combo':    round(p_combo, 4),
                    'cuota_combo': round(cuota_combo, 2),
                    'EV_combo':   round(EV_combo, 4),
                    'n_legs':     n,
                    'tiers':      [l['tier'] for l in combo_legs],
                })

    # 4. Ordenar por EV_combo × p_combo (retorno esperado neto)
    all_combos.sort(key=lambda c: c['EV_combo'] * c['p_combo'], reverse=True)

    # 5. Top-10 con solape ≤2 piernas entre cualquier par seleccionado
    selected = _select_with_overlap_constraint(all_combos, max_overlap=2, top_n=10)
    return selected
```

**Agrupación temporal:**
```python
def _group_by_time_window(legs, window_hours=3):
    """
    Agrupa picks en ventanas de ±window_hours horas.
    Algoritmo greedy: cada pick se asigna al primer grupo donde
    max(hora_grupo) - min(hora_grupo) + hora_pick <= window_hours.
    """
    ...
    # Resultado: lista de grupos, cada grupo es lista de legs
    # Un leg puede pertenecer a UN solo grupo (primera asignación gana)
```

### D139-05 — `_kelly_stake(combo, bankroll, n_simultaneous)`

```python
EV_MIN_COMBO    = 0.02    # 2% EV mínimo por combo
MIN_STAKE       = 500     # $500 mínimo (evitar apuestas triviales)
MAX_STAKE_PCT   = 0.03    # 3% bankroll máximo por combo
HALF_KELLY      = 0.5     # Factor de prudencia estándar (Kelly fraccionario)
RHO_PARLAY      = 0.15    # Correlación entre combos (misma sesión)

def _kelly_stake(combo, bankroll, n_simultaneous):
    EV        = combo['EV_combo']
    q_combo   = combo['cuota_combo']
    f_raw     = EV / (q_combo - 1)        # Kelly clásico para parlay
    f_half    = f_raw * HALF_KELLY
    pf        = 1 / (1 + RHO_PARLAY * (n_simultaneous - 1))
    stake     = bankroll * f_half * pf
    stake     = max(MIN_STAKE, min(stake, bankroll * MAX_STAKE_PCT))
    return round(stake / 100) * 100        # Redondear a $100
```

### D139-06 — `_generate_kambi_first_bat(combos, bankroll)`

```python
def _generate_kambi_first_bat(combos, bankroll):
    """
    Genera KB_1.bat, KB_2.bat ... en Desktop.
    Formato Kambi: outcome_id_1,outcome_id_2,...||replace
    Cada .bat abre Betplay con el combo prellenado.

    También genera KB_report_FECHA.html con:
    - Tabla de combos: piernas, cuota_combo, EV, stake, retorno_esperado
    - Badge tier: A vs B piernas marcadas
    - Agrupación por ventana horaria
    """
```

### D139-07 — Flag `--kambi-first` en `betplay_combo_builder.py`

```python
# En main():
if args.kambi_first:
    legs = _fetch_kambi_betting_universe()
    edge_report = _load_latest_edge_report()
    h2h_data    = _load_h2h_players()
    matched     = _match_to_predictions(legs, edge_report, h2h_data)
    scored      = [_compute_leg_signal(m) for m in matched if m]
    scored      = [s for s in scored if s and s['n_legs_ok']]
    combos      = _build_kambi_combos(scored, args.bankroll)
    _generate_kambi_first_bat(combos, args.bankroll)
    _print_kambi_first_report(combos, args.bankroll)
```

**CLAUDE.md §4 PASO:**
```bash
# PASO 4.5 — Kambi-First Combo Builder (Nodo-139)
python3 betplay_combo_builder.py --kambi-first --bankroll 125000
```

---

## §4. Análisis matemático de impacto esperado

### 4.1 Con los datos de HOY (2026-07-22)

Pool disponible: 138 Kambi favorites en [1.10, 1.80]
Edge_report: 22 picks (10 atp500, 5 itf, 7 challenger)
H2H data: 302 matches (apellidos únicos ~200)

Proyección de matching:
- TIER_A (edge_report match): ~15-20 picks (apellido matching mejorado)
- TIER_B (H2H match): ~25-40 picks adicionales
- Total scored pool: ~40-60 picks con edge > 0

De esos 40-60, picks que pasan gates (edge > 0 AND p ≥ 0.55):
- Estimado conservador: 20-30 picks
- En grupos de ≥3 por ventana: fácil formar 5-8 combos de 3-5 piernas

Retorno esperado por combo (asumiendo EV_combo = 0.08, cuota = 5x, stake $1,500):
- EV monetario = $1,500 × 0.08 = +$120 por combo
- Con 7 combos/sesión: +$840 EV esperado por sesión

### 4.2 Por qué quitar el tope de cuota genera más ganancias

Ejemplo concreto con datos de hoy:
```
Leg 1: McFadzean @1.93, p_modelo=0.70, edge=+0.181
Leg 2: Lewis @1.55, p_modelo=0.68, edge=+0.035
Leg 3: Angel @3.50, p_modelo=0.75, edge=+0.464

cuota_combo = 1.93 × 1.55 × 3.50 = 10.47x  ← BLOQUEADO por cap actual (>7.0)
p_combo     = 0.70 × 0.68 × 0.75 = 0.357
EV_combo    = 0.357 × 10.47 - 1  = 2.74 = +274%  ← ENORME

Con stake Kelly: f = 2.74/9.47 = 0.289 → half-Kelly = 14.5% → cap 3% = $3,750
Retorno esperado: $3,750 × 2.74 = $10,275
```

El tope de 7.0 estaba bloqueando el mejor combo del día con +274% EV.

### 4.3 Calibración conservadora del MIN_P_COMBO

El MIN_P_COMBO por n_legs previene "lottery ticket combos" sin probabilidad real:

| n_legs | MIN_P_COMBO | Implies each leg p ≥ |
|--------|-------------|----------------------|
| 3      | 0.10        | 0.46                 |
| 4      | 0.08        | 0.53                 |
| 5      | 0.06        | 0.55                 |
| 6      | 0.04        | 0.53                 |
| 7      | 0.03        | 0.54                 |

Con p_efectivo ≥ 0.55 por gate individual, los combos de 3-4 piernas son los más
frecuentes. 7 piernas requiere picks con p muy alto (≥0.63 cada uno).

---

## §5. Hipótesis pre-registrada

**H139-01:** "Kambi-first combos con ≥2 piernas TIER_A tienen hit% > breakeven"
- `n_stop = 50`
- Breakeven calculado por cuota_combo promedio al momento del registro
- Archivo: `validation/preregistered_hypotheses.json`

**H139-02:** "TIER_B combos (solo H2H match) tienen hit% ≤ breakeven"
- Hipótesis nula de control: si H2H solo no aporta alpha, excluirlos
- `n_stop = 30`

---

## §6. Constraints NO negociables (guards de ruina)

1. **TIER_C siempre excluido** — no apostar sin señal propia
2. **edge_efectivo > 0 por pierna** — si p_efectivo ≤ 1/cuota_kambi, excluir
3. **p_efectivo ≥ 0.55** — no combinaciones de coinflips
4. **cuota_fav ≥ 1.10** — por debajo el vig destruye el edge matemáticamente
5. **MAX_STAKE = 3% bankroll por combo** — protección de drawdown
6. **No apostar STARTED events** — filtro state=NOT_STARTED en D139-01
7. **combo_governor.py** — verificar exposición total después de generar stakes

---

## §7. Tests REGLA-T53 — `tests/test_nodo139_kambi_first.py`

```python
def test_D139_01_fetch_returns_not_started_only()
    # Todos los eventos retornados tienen state=NOT_STARTED

def test_D139_01_cuota_filter_bounds()
    # Ningún leg retornado tiene cuota_fav < 1.10 o > 1.80

def test_D139_02_apellido_kambi_extracts_surname_from_firstname_last()
    # "Lachlan Mcfadzean" → "mcfadzean"
    # "Van De Zandschulp B." → "zandschulp" (o "de zandschulp")

def test_D139_02_apellido_pick_extracts_surname_from_surname_first()
    # "McFadzean L." → "mcfadzean"
    # "van Loben Sels E." → "loben sels"

def test_D139_02_match_score_handles_mcfadzean_case()
    # _match_score_names("Lachlan Mcfadzean", "McFadzean L.") ≥ 0.85

def test_D139_03_gate_excludes_negative_edge_legs()
    # pick con p_efectivo=0.60, cuota_kambi=1.90 → p_implied=0.526
    # edge = 0.60 - 0.526 = +0.074 → pasa
    # pick con p_efectivo=0.50, cuota_kambi=1.90 → edge = -0.026 → excluir

def test_D139_04_no_cuota_cap_allows_high_product_combo()
    # 3 legs: cuota 1.93 × 1.55 × 3.50 = 10.47 → debe generar combo (no bloqueado)
    # EV = 0.357 × 10.47 - 1 = 2.74 > EV_MIN = 0.02 → pasa

def test_D139_04_ev_gate_blocks_low_ev_combo()
    # Combo con EV_combo = 0.01 < EV_MIN_COMBO = 0.02 → bloqueado

def test_D139_05_kelly_stake_bounded_by_3pct_bankroll()
    # Bankroll=125000 → max_stake = 3750
    # Combo con f_kelly=0.20 → stake sin cap = 125000×0.20×0.5 = 12500
    # Después del cap: stake = 3750

def test_D139_05_kelly_stake_minimum_500()
    # Combo con f_kelly muy bajo → stake mínimo = 500
```

---

## §8. Orden de implementación (Sonnet)

1. **D139-01**: `_fetch_kambi_betting_universe()` — función nueva en betplay_combo_builder.py
2. **D139-02**: `_apellido_kambi()`, `_apellido_pick()`, `_match_score_names()`, `_match_to_predictions()` — funciones puras
3. **D139-03**: `_compute_leg_signal()` — leer edge_report y H2H, retornar ScoredLeg
4. **D139-04**: `_build_kambi_combos()` con `_group_by_time_window()` y `_select_with_overlap_constraint()`
5. **D139-05**: `_kelly_stake()` — función pura, testeable aislado
6. **D139-06**: `_generate_kambi_first_bat()` y `_print_kambi_first_report()`
7. **D139-07**: Flag `--kambi-first` en `main()` + PASO 4.5 en run_daily.py
8. **Tests**: 10 tests REGLA-T53 en `tests/test_nodo139_kambi_first.py`
9. **H139-01/02**: Pre-registrar hipótesis en `validation/preregistered_hypotheses.json`
10. **nodos_index**: rebuild

---

## §9. Dependencias y riesgos

| Dependencia | Estado | Riesgo |
|-------------|--------|--------|
| `_fetch_kambi_betting_universe` usa KAMBI_BASE/PARAMS/HEADERS | Existentes en kambi_tennis.py | Bajo |
| Matching a edge_report | edge_report se genera en PASO 3 | Bajo — ya existe |
| Matching a H2H | h2h_results_enhanced_*.json | Bajo — ya existe |
| `_apellido_pick` para compound surnames | Requiere test extensivo | Medio |
| combo_governor verifica resultado | D137-01 ya excluye MOTOR | Bajo |
| `outcome_id_fav` en .bat | Extraído en D139-01 del mismo fetch | Bajo |

**Riesgo principal:** El matching TIER_B (H2H data) puede generar falsos positivos
si dos jugadores tienen el mismo apellido. Mitigación: requerir al menos los 2
primeros tokens del apellido para compound surnames, y mínimo n_matches ≥ 5.

---

## §10. Apertura futura (NO implementar en Nodo-139)

- **TIER_B enriquecido con IRP** (Nodo-96): Individual Return Profile para TIER_B picks
- **Cuota live en el momento del .bat** vs cuota al momento de análisis: usar D139-01 live
- **Rushbet como segunda casa**: si Kambi no tiene el partido, buscar en Rushbet (Nodo-121)
- **D136-02** (pendiente): propagar torneo_nombre en extraer_historh2h.py — mejoraría
  matching al incluir metadato de circuito en apellido disambiguation
