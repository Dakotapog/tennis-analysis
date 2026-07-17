# Nodo-64 — RFI: Return From Inactivity Signal

> **Wikilinks:** [[Nodo-57-Penalizacion-Inactividad-Campeon-Validacion]] | ~~~~[[Nodo-32-Fase3-Markov-Postnorm]]~~ _(MISSING)_~~ _(MISSING)_ | [[Nodo-21-Pesos-Diferenciados-Por-Tier]] | [[Nodo-44-Watchlist-Alpha-Signal]]
> **Fecha descubrimiento:** 2026-07-09
> **Estado:** BORRADOR — hipótesis pre-registrada H76-01, implementación pendiente (n_stop=30)
> **Severidad:** ALTA — alpha estructural explotable, bookmaker blind spot sistemático
> **Descubierto en:** Pick Rivera @4.35 vs Michnev (M15 Lodz, arcilla) — 2026-07-09
> **Evidencia inicial:** 5 casos resueltos hoy, activo ganó 4/5 (80%)

---

## 0. Descubrimiento

### Caso semilla — Michnev vs Rivera (2026-07-09)

```
Partido:    Petr Michnev vs Leyton Rivera | M15 Lodz (ITF clay)
Cuotas:     Michnev 1.17 (bookie fav) | Rivera 4.35 (underdog)
Resultado:  Rivera ganó 6-0 6-0 (dominancia total)

Señales en el pick_snapshot:
  p_modelo:          0.627   ← modelo predice 62.7% Rivera
  p_implicita:       0.230   ← bookmaker asigna solo 23% a Rivera
  edge:              39.7%   ← la mayor discrepancia del día
  confidence_flag:   STRONG
  BBI:               0.7701  ← bookmaker blind
  surface_signal:    0.709   ← Rivera mejor en arcilla (relativo)
  circuit_asymmetry: 4.24    ← Michnev viene de mundo hard-court
  alpha_vs_elo:     +0.177   ← Rivera 17.7% sobre ELO por factores superficie
  
  LOG_FORM_DECAY: p1_days=273  fd_p1=0.350  ← Michnev: forma decaída al 35%
                  p2_days=8    fd_p2=1.000  ← Rivera: 100% fresco (jugó hace 8 días)
```

**La señal raíz:** Nodo-57 ya aplicaba `form_decay` (fd=0.35 para 273 días). Lo que falta es formalizar el patrón como señal named y pre-registrar la hipótesis para acumulación sistemática.

---

## 1. Anatomía del Patrón RFI

### Las 5 capas convergentes

| Capa | Michnev (inactivo) | Rivera (activo) | Delta |
|---|---|---|---|
| **Form Decay (Nodo-57)** | fd = 0.35 (273d) | fd = 1.00 (8d) | ×2.86 a favor activo |
| **Form score normalizado** | 1.49 | 4.26 | ×2.85 |
| **Ranking ATP** | 1746 | 1143 | −603 posiciones peor |
| **Clay Alpha** | −11.1% vs ELO | −4.2% vs ELO | Michnev mucho peor en arcilla |
| **Circuit Asymmetry (CTI)** | 0.674 (hard world) | 0.159 (clay circuit) | ratio = 4.24 |

### Por qué el bookmaker se equivoca sistemáticamente

El bookmaker usa datos históricos de rendimiento sin aplicar decay por inactividad larga.
Para un jugador inactivo 9 meses que vuelve a una superficie secundaria:

```
Bookmaker ve:  ELO histórico 1483 > Rivera 1448 → Michnev favorito @1.17
Modelo ve:     ELO × form_decay(273d) × clay_penalty → Rivera 62.7%
```

**El error NO es de datos — es de temporal discounting.** El bookmaker conoce la historia
de Michnev pero no penaliza suficientemente los 9 meses de parón en arcilla.

---

## 2. Definición del Patrón RFI

### Condición de activación

```
RFI activo cuando:
  1. Un jugador del partido tiene days_since_last_match > THRESHOLD_DIAS
  2. El jugador inactivo es favorito del bookmaker (cuota_inactivo < cuota_activo)
  3. El modelo predice al jugador ACTIVO como ganador (p_modelo_activo > 0.50)
```

### Tiers de señal

```
RFI-0:    0-89 días     → sin señal (normal entre torneos)
RFI-1:   90-179 días    → señal débil  | WATCH
RFI-2:  180-269 días    → señal fuerte | potencial boost
RFI-3:  270d+           → señal crítica | auto-STRONG si edge≥15%

RFI-ULTRA = RFI-2 o RFI-3
           + inactivo es favorito bookmaker (cuota < 2.0)
           + inactivo tiene peor ranking ATP que el activo
           → error sistemático de mercado confirmado
```

### Campos propuestos en pick_snapshot

```python
"rfi_signal":          "RFI-ULTRA" | "RFI-3" | "RFI-2" | "RFI-1" | null
"rfi_tier":            3           # 0-3
"rfi_days_inactive":   273         # días del jugador inactivo
"rfi_is_bookie_fav":   true        # el inactivo es fav del bookmaker
"rfi_inactivo":        "Petr Michnev"
"rfi_activo":          "Leyton Rivera"
```

---

## 3. Validación — 6 casos HOY (2026-07-09)

| Partido | Inactivo | Días | Fav.Book | Superficie | Activo ganó |
|---|---|---|---|---|---|
| Kubka vs Hays | Hays | 351d | Kubka (activo) | hard | **SI** |
| Mazzola vs Murray | Murray | 343d | Mazzola (activo) | clay | **SI** |
| Filip vs Jedrzejczak | Jedrzejczak | 323d | Filip (activo) | clay | NO* |
| **Michnev vs Rivera** | Michnev | 273d | **Michnev (inactivo!)** | clay | **SI** |
| Orlov vs Chlodnicki | Chlodnicki | 167d | Orlov (activo) | clay | **SI** |
| (Melero vs Torcq) | Melero | 260d | Torcq | clay | PENDIENTE |

**4/5 resueltos: activo ganó. Modelo correcto en los 5 casos.**

*Jedrzejczak (323d inactivo) ganó pero el MODELO lo predijo correctamente — otros
factores (ranking, clay history) compensaron la inactividad. Esto es consistente:
RFI no es absoluto, es un prior que el resto del modelo puede sobrescribir.

**El caso Michnev es el más valioso:** el bookmaker lo tenía como FAVORITO PESADO @1.17
a pesar de 9 meses de parón en arcilla (su superficie débil). Ahí está el alpha máximo.

---

## 4. Marco de Expertos

### Marco 1 — Estadístico
El form_decay de Nodo-57 (fd=0.35 para 273d) captura el decay individual.
RFI añade una capa de orden superior: cuando el MERCADO no aplica ese mismo decay,
existe una divergencia explotable P_modelo >> P_implícita con dirección estructural.
La señal RFI-ULTRA es un indicador de ineficiencia de mercado, no solo de forma.

### Marco 2 — Domain Expert (circuito ATP)
Jugadores inactivos 6-9 meses regresan:
- Sin ritmo de partido (los primeros sets son adaptativos)
- Sin la dinámica física específica de la superficie
- Con incertidumbre en la condición física real
Los bookmakers no tienen acceso completo a reportes de entrenamiento de jugadores
ITF/Challenger — su información es más limitada que en ATP Top 100.

### Marco 3 — Mercado / Bookmaker
Para rangos de ranking 1000-2000 (ITF/Challenger bajo), los bookmakers usan modelos
simples basados en ELO histórico + H2H reciente. No aplican decay exponencial a la
forma reciente por inactividad larga. El BBI alto (>0.75) confirma que el bookmaker
tiene incertidumbre. En ese régimen, la divergencia modelo-mercado es más explotable.

### Marco 4 — Bayesiano
RFI-ULTRA = prior fuerte hacia el activo. La magnitud del prior es proporcional a:
  prior_strength = f(days_inactive) × f(surface_mismatch) × f(ranking_gap)
Cuando los 3 factores convergen (días altos + superficie débil + peor ranking),
el prior es suficientemente fuerte para justificar apostar contra el favorito del mercado
incluso con edge < 20% en condiciones normales.

---

## 5. Deudas de Implementación

| ID | Descripción | Archivo | Prioridad | Gate |
|---|---|---|---|---|
| D64-01 | Calcular `rfi_signal`, `rfi_tier`, `rfi_days_inactive`, `rfi_is_bookie_fav`, `rfi_inactivo`, `rfi_activo` y añadir a pick_snapshot | `edge_calculator.py` | ALTA | n_stop H76-01 ≥ 10 |
| D64-02 | Señal especial RFI en tabla de favoritos: "RFI-ULTRA: {inactivo} {X}d sin jugar + favorito bookmaker" | `generar_tabla_favoritos2.py` | MEDIA | post D64-01 |
| D64-03 | En shadow_book.py `--report`: segmento RFI-ULTRA (hit%, ROI, CLV) | `shadow_book.py` | MEDIA | n≥10 |
| D64-04 | Escalado de confidence_flag: si rfi_tier≥2 AND rfi_is_bookie_fav AND edge≥15% → confidence LOW→MOD, MOD→STRONG | `edge_calculator.py` | ALTA | post n_stop H76-01 |

**GATED:** D64-04 (escalado de confianza) NO se implementa hasta que H76-01 tenga
n≥30 con IC Wilson inferior > breakeven. D64-01/02/03 son observacionales (READ-ONLY
para calibración).

---

## 6. Tests de Validación (REGLA-T53)

Los tests son READ-ONLY / cálculo hasta que D64-04 sea aprobado post H76-01.

| Test | Descripción | Resultado esperado |
|---|---|---|
| T64-01 | Michnev (273d) vs Rivera (8d), cuota Michnev 1.17 | rfi_tier=3, rfi_is_bookie_fav=True, rfi_signal="RFI-ULTRA" |
| T64-02 | Jugador inactive 50d → por debajo del umbral | rfi_tier=0, rfi_signal=null |
| T64-03 | Jugador inactivo 200d, cuota inactivo=3.5 (no es fav) | rfi_tier=2, rfi_is_bookie_fav=False, rfi_signal="RFI-2" |
| T64-04 | Jugador inactivo 350d, cuota inactivo=1.25 | rfi_tier=3, rfi_is_bookie_fav=True, rfi_signal="RFI-ULTRA" |
| T64-05 | Inactivo tiene peor ranking Y es bookie fav → RFI-ULTRA con ranking_gap | rfi_signal="RFI-ULTRA", campos completos |
| T64-06 | rfi_signal en pick_snapshot se propaga a shadow book JSONL | field presente en sb_YYYY-MM-DD.jsonl |

---

## 7. Hipótesis Pre-registrada

**H76-01** — registrada en `validation/preregistered_hypotheses.json`

```
Condición: rfi_tier >= 2 AND rfi_is_bookie_fav = True
           (inactivo >180d + bookmaker lo tiene como favorito)
Métrica:   hit% del jugador activo en estos picks > breakeven (1/cuota_activo_media)
Éxito:     IC Wilson 95% inferior > breakeven con n ≥ 30
n_stop:    30
Gate D64-04: SOLO se activa escalado de confianza si H76-01 gradúa
```

---

## 8. Registro

**Descubierto:** 2026-07-09 — análisis post-partido Rivera @4.35 vs Michnev @1.17
**Caso semilla:** Rivera ganó 6-0 6-0. Edge modelo 39.7% (mayor del día). BBI=0.77.
**Validación mismo día:** 5 casos >90d inactividad — activo ganó 4/5. Modelo acertó 5/5.
**Señal base existente:** Nodo-57 form_decay fd=0.35 ya capturaba el efecto.
**Novedad de Nodo-64:** formaliza el patrón como señal named, añade la dimensión
"bookmaker error sistemático" (cuando el inactivo ES el favorito), y pre-registra
la hipótesis para acumulación controlada antes de activar ningún gate.

**PROHIBIDO antes de H76-01 n≥30:**
- Cambiar confidence_flag basado en rfi_signal (D64-04 está gated)
- Modificar kelly_kl, VaR, shrinkage por el hallazgo de este nodo
- Usar este patrón como gate de exclusión (solo como señal informativa)
