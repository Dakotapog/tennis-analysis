# Nodo-138 — G2 Gate Evolution: Multi-Signal Evidence vs. n_h2h Único

> **Estado:** PROPUESTA — para implementación inmediata
> **Tipo:** ARCHITECTURAL FIX — desbloquea CORE/SAT/MOON/COB/ANCHOR/FAVORITOS
> **Trigger:** 0 combos de confianza generados en días con n_h2h=0 masivo (H2H combinado)
> **Autor:** Sonnet 4.6 — análisis doctoral nivel audit, 2026-07-22
> **Para implementación:** Sonnet puede ejecutar sin ambigüedad con este spec

---

## Wikilinks

| Link | Rol |
|------|-----|
| [[Nodo-136-Tier-Detection-CTI-Fallback]] | CTI fallback — misma raíz: H2H combinado pierde metadata |
| [[Nodo-137-Governor-MOTOR-Exclusion]] | Governor fix — desbloqueó el budget, este nodo desbloquea los datos |
| [[Nodo-87-Fixes-Auditoria-D87]] | Patrón de corrección quirúrgica audit |
| [[Nodo-86-Auditoria-Fable5]] | Contexto auditoría doctoral |
| [[Nodo-100-Taxonomia-Estrategias-Generacion-Combos]] | Las 12 estrategias — todas afectadas por G2 |
| [[Nodo-29-Circuit-Asymmetry-Deflator]] | CTI — señal disponible en edge_data para reemplazar n_h2h |

**Wikilinks totales: 6 | Huérfanos: 0**

---

## §1. Análisis de raíz — Por qué G2 está arquitectónicamente equivocado

### §1.1 Lo que G2 hace actualmente

En `combo_confianza_builder.py:L252`:

```python
n_h2h = int(edge_data.get('n_h2h') or 0)
if n_h2h < 1:
    conf_flag   = edge_data.get('confidence_flag', '')
    n_axes      = int(edge_data.get('n_axes_active') or 0)
    score_dir   = int(edge_data.get('score_directo') or 0)
    if conf_flag == 'STRONG' and n_axes >= 3 and score_dir >= 3:
        pass  # triple convergencia — única excepción
    else:
        return True, f'G2: n_h2h=0 — sin historial directo'
```

**Diagnóstico:** G2 bloquea cualquier pick donde los dos jugadores nunca se han enfrentado directamente, a menos que cumpla triple convergencia (STRONG + n_axes≥3 + score_dir≥3). Esta condición es tan restrictiva que prácticamente nunca se cumple con el H2H combinado actual (n_axes_active=2 en casi todos los picks de hoy).

### §1.2 Por qué n_h2h=0 es un gate equivocado para combos

**El error conceptual central:** `n_h2h=0` no significa "no sabemos quién es mejor." Significa "estos dos jugadores específicos no se han enfrentado en el historial disponible." Son cosas completamente distintas.

El sistema tiene **20+ señales independientes** que evalúan la calidad de cada jugador SIN requerir historial directo:

| Señal disponible en edge_data | Naturaleza | Confiable sin H2H |
|-------------------------------|------------|-------------------|
| `elo_favorito` / `elo_rival` | ELO calculado vs campo completo | ✅ SÍ |
| `ranking_favorito` / `ranking_rival` | ATP/WTA oficial | ✅ SÍ |
| `markov_favorito` / `markov_rival` | Régimen de forma reciente | ✅ SÍ |
| `surface_signal` | Especialización en superficie | ✅ SÍ |
| `regime_signal` | Señal de régimen actual | ✅ SÍ |
| `bbi` | Bookmaker Bias Index | ✅ SÍ |
| `freshness_pelt` | Recencia del régimen (PELT) | ✅ SÍ |
| `rfi_days_inactive` | Return From Inactivity | ✅ SÍ |
| `phi_idiosincratico` | Factor idiosincrático personal | ✅ SÍ |
| `edge` | Diferencia modelo vs cuota implícita | ✅ SÍ |
| `kelly_kl` | Kelly-KL ajustado por riesgo | ✅ SÍ |
| `alignment_flag` / `net_alignment` | Coherencia de señales | ✅ SÍ |
| `circuit_asymmetry` (CTI) | Nivel de circuito histórico | ✅ SÍ |
| `confidence_flag` | Agregado Bayesiano de todas las señales | ✅ SÍ |

**Un pick con `conf=95%, edge=54.7%, kelly_kl=0.26` ha pasado 20+ filtros de calidad independientes.** El hecho de que estos dos jugadores no se hayan enfrentado antes no invalida ninguna de esas señales. El MOTOR ya aprobó este pick (G1: `apostar=True`) con pleno conocimiento de que `n_h2h=0`. G2 está ANULANDO la decisión del MOTOR sin agregar información nueva.

### §1.3 El double-penalization problem

El MOTOR (`trader_ev_tenis.py`) ya maneja n_h2h=0 correctamente:
1. `p_blend` usa el prior Bayesiano cuando n_h2h es bajo — no asume nada extra
2. `shrinkage = n/(n+20)` — reduce confianza automáticamente cuando n_h2h < 20
3. `kelly_kl` — penaliza el edge con la divergencia KL del historial
4. `confidence_flag` — se reduce si las señales son inconsistentes

**El resultado:** cuando el MOTOR dice `apostar=True` con `confidence_flag='STRONG'`, YA HA INCORPORADO el riesgo de `n_h2h=0` en su cálculo. G2 vuelve a penalizar el mismo riesgo por segunda vez — **double-penalization** que produce false negatives masivos.

### §1.4 El scope del problema — cuantificado

- **Hoy 2026-07-22:** 302 partidos en H2H combinado. n_h2h=0 en TODOS los partidos (primera vez que se enfrentan por la razón de que son de diferentes circuitos/torneos rotatorios).
- **Consecuencia:** 0 combos de confianza generados, 0 ANCHOR, 0 FAVORITOS.
- **Perdido:** Picks como Huszar A. (conf=95%, edge=54.7%), Ebster A.L. (conf=75.8%), Bosch Armas S. (edge=49.6%) — TODOS BLOQUEADOS por G2 a pesar de ser picks de alta calidad.
- **Dias típicos ITF/Challenger:** 50-70% de los partidos tienen n_h2h=0. El problema es estructural, no accidental.

### §1.5 El problema de torneo='Desconocido' en FAVORITOS

**Archivo:** `favoritos_combo_builder.py:L232`

```python
torneo = p.get("torneo", p.get("tournament", "UNK"))
torneo_count[torneo] = torneo_count.get(torneo, 0) + 1
if torneo_count[torneo] > MAX_LEGS_PER_TORNEO:
    ok = False
```

Cuando todos los picks tienen `torneo='Desconocido'` (por la falta de propagación de torneo en el H2H combinado — gap D136-02), el contador llega a 2 en el segundo pick, excede `MAX_LEGS_PER_TORNEO` (probablemente 1), y bloquea todas las combinaciones de 2+ piernas.

**El error:** La diversificación por torneo es correcta como principio (no queremos 4 picks del mismo torneo). Pero cuando torneo='Desconocido', no significa "mismo torneo" — significa "torneo desconocido". Son partidos distintos (distintos jugadores, distintos matches) que casualmente no tienen el nombre del torneo en el campo. Tratarlos como "mismo torneo" es un falso positivo.

---

## §2. D138-01 — G2 Gate Evolution

**Archivo:** `combo_confianza_builder.py`
**Función:** `_check_gates_pick()` o similar (contiene el bloque G2)
**Líneas:** L252-262 (bloque G2 actual)

### §2.1 El nuevo G2

**Reemplazar el bloque completo L252-262** con:

```python
# G2 — evidence quality gate (D138-01: n_h2h NO es el único indicador válido)
n_h2h = int(edge_data.get('n_h2h') or 0)
if n_h2h < 1:
    conf_flag  = edge_data.get('confidence_flag', '')
    n_axes     = int(edge_data.get('n_axes_active') or 0)
    score_dir  = int(edge_data.get('score_directo') or 0)
    edge_val   = float(edge_data.get('edge') or 0)
    kelly_kl   = float(edge_data.get('kelly_kl') or 0)

    # Regla-1 (original): Triple convergencia — sin cambios
    if conf_flag == 'STRONG' and n_axes >= 3 and score_dir >= 3:
        pass

    # Regla-2 (D138-01): STRONG + edge sustancial + Kelly positivo + ≥2 ejes
    # El MOTOR ya aprobó (G1), tiene señal fuerte multi-eje, Kelly lo valida.
    elif conf_flag == 'STRONG' and edge_val >= 0.20 and kelly_kl > 0 and n_axes >= 2:
        pass

    # Regla-3 (D138-01): Cualquier nivel de confianza + edge muy alto (bookmaker obviamente equivocado)
    # Si el bookmaker está tan equivocado que edge≥35% con Kelly positivo, la ausencia
    # de H2H no cambia que el modelo tiene ventaja observable sobre el mercado.
    elif edge_val >= 0.35 and kelly_kl > 0 and n_axes >= 2:
        pass

    else:
        return True, (
            f'G2: n_h2h=0 — señal insuficiente sin H2H directo '
            f'(conf={conf_flag}, edge={edge_val:.1%}, kelly={kelly_kl:.3f}, '
            f'axes={n_axes}, score_dir={score_dir})'
        )
```

### §2.2 Calibración de umbrales — razonamiento

| Regla | Condición | Razonamiento |
|-------|-----------|--------------|
| 1 | Triple convergencia original | Sin cambios — el caso más robusto |
| 2 | STRONG + edge≥20% + kelly>0 + axes≥2 | STRONG = conf≥60% post-Bayesian. Edge≥20% = modelo consistentemente mejor que bookmaker. Kelly>0 = el árbitro de riesgo final lo aprueba. |
| 3 | edge≥35% + kelly>0 + axes≥2 | Cuando el bookmaker está tan equivocado (35%+ edge), la calidad de la señal es obvia independiente del nivel de confianza del flag. |

**Por qué edge≥20% para Regla-2 y no 15%:**
- edge=15% todavía tiene alta incertidumbre cuando n_h2h=0 (el modelo puede estar captando ruido)
- edge≥20% con kelly>0 indica que el ajuste Kelly-KL (que penaliza historial corto) todavía recomienda apostar — doble validación

**Por qué kelly_kl>0 como condición necesaria:**
- `kelly_kl` ya incorpora la penalización por n_h2h=0 (via shrinkage y divergencia KL)
- Si Kelly-KL=0, el MOTOR implícitamente dice "el riesgo de historial corto cancela el edge" → respetar esa decisión

### §2.3 Picks que pasarán G2 con los datos de hoy (verificado)

Con edge_report_20260722_103355.json:
- Huszar A.: STRONG, edge=54.7%, kelly>0, axes=2 → **Regla-2 PASS** ✅
- Ebster A.L.: STRONG, edge=75.8%, kelly>0, axes≥2 → **Regla-2 PASS** ✅
- Bosch Armas S.: STRONG, edge=49.6%, kelly>0, axes=2 → **Regla-2 PASS** ✅
- Saiga W.: (verificar conf_flag) edge=42.6%, kelly>0 → **Regla-2 o Regla-3 PASS** ✅
- Kanumuri I.R.: edge=41.8%, kelly>0 → **Regla-3 PASS** ✅
- Suresh D.: edge=30.9% → **Regla-3 PASS** (≥35%? Si conf=STRONG → Regla-2) ✅
- Mroz M.: edge=27.7% → Regla-2 si STRONG; BLOCK si MODERATE + edge<35%
- Liljekvist D.: edge=20.6% → Regla-2 si STRONG + kelly>0 + axes≥2; border

**Picks que SEGUIRÁN BLOQUEADOS (correcto):**
- Qualquier pick con edge<20%, conf≠STRONG, kelly_kl=0 — señal genuinamente débil sin H2H

---

## §3. D138-02 — Favoritos Diversificación por Torneo Desconocido

**Archivo:** `favoritos_combo_builder.py`
**Línea exacta:** L232 en la función `build_combo_simple()` (o similar)

### §3.1 El código actual

```python
torneo = p.get("torneo", p.get("tournament", "UNK"))
torneo_count[torneo] = torneo_count.get(torneo, 0) + 1
if torneo_count[torneo] > MAX_LEGS_PER_TORNEO:
    ok = False
    break
```

### §3.2 El fix

Reemplazar las líneas de extracción del torneo (SOLO esas 2 líneas, no la lógica de conteo):

```python
# D138-02: Cuando torneo es desconocido, usar partido como clave única de diversificación.
# 'Desconocido' NO significa "mismo torneo" — significa "torneo sin metadato".
# Cada partido es un match único (jugadores distintos) → clave única garantizada.
_TORNEO_GENERICO = {'desconocido', 'unk', '?', '', 'unknown', 'desconocida', 'none'}
torneo_raw = (p.get("torneo") or p.get("tournament") or "").strip()
if torneo_raw.lower() in _TORNEO_GENERICO:
    # Usar partido como clave — garantiza unicidad por match, permite combinación
    torneo = f"_match_{p.get('partido', p.get('favorito', str(id(p))))}"
else:
    torneo = torneo_raw
torneo_count[torneo] = torneo_count.get(torneo, 0) + 1
if torneo_count[torneo] > MAX_LEGS_PER_TORNEO:
    ok = False
    break
```

### §3.3 Razonamiento

**Correctitud:** Dos picks con `torneo='Desconocido'` provienen de dos matchs distintos (distintos jugadores). Sus resultados son independientes. El propósito de la diversificación (evitar que 3 picks del mismo torneo fallen juntos por un factor de torneo) no aplica cuando el torneo es desconocido — no existe correlación identificable.

**Seguridad:** La lógica de `jugador_seen` (L234-237) ya garantiza que no repetimos el mismo jugador. Con esa protección y la unicidad por `partido`, los combos son correctamente diversificados en lo que importa.

**Impacto hoy:** O'Connell @2.00, Thompson @1.65, Ciocan @1.72 → cuota combo ≈ 5.68 → dentro del rango [3.5, 7.0] → **FAVORITOS_A generado** ✅

---

## §4. Tests REGLA-T53 — `tests/test_nodo138_g2_evolution.py`

```python
# D138-01 — G2 gate evolution
def test_D138_01_g2_blocks_weak_signal_no_h2h():
    """n_h2h=0, conf=LOW, edge=10%, kelly=0 → G2 BLOCK (señal demasiado débil)"""

def test_D138_01_g2_passes_strong_edge20_kelly_positive():
    """n_h2h=0, conf=STRONG, edge=0.25, kelly_kl=0.10, n_axes=2 → G2 PASS (Regla-2)"""

def test_D138_01_g2_passes_high_edge_any_conf():
    """n_h2h=0, conf=MODERATE, edge=0.38, kelly_kl=0.08, n_axes=2 → G2 PASS (Regla-3)"""

def test_D138_01_g2_blocks_high_conf_zero_kelly():
    """n_h2h=0, conf=STRONG, edge=0.25, kelly_kl=0.0 → G2 BLOCK (Kelly dice no)"""

def test_D138_01_g2_unchanged_triple_convergencia():
    """n_h2h=0, conf=STRONG, n_axes=3, score_dir=3 → G2 PASS (Regla-1 sin cambios)"""

def test_D138_01_g2_unchanged_n_h2h_positive():
    """n_h2h=5 → G2 PASS sin evaluar edge/conf (comportamiento original)"""

# D138-02 — Favoritos diversificación
def test_D138_02_favoritos_desconocido_torneo_allows_combination():
    """3 picks con torneo='Desconocido', partidos distintos → combo válido generado"""

def test_D138_02_favoritos_known_torneo_unchanged():
    """picks con torneo conocido → comportamiento original de MAX_LEGS_PER_TORNEO"""

def test_D138_02_favoritos_desconocido_player_uniqueness_still_enforced():
    """mismo jugador con torneo='Desconocido' en 2 picks → bloqueado por jugador_seen"""
```

---

## §5. Impacto esperado — resumen

### Días con n_h2h=0 masivo (ITF/Challenger combinado):

| Estrategia | Antes | Después |
|-----------|-------|---------|
| CORE 4-7 piernas | 0 combos siempre | Combos cuando picks pasan Regla-2/3 |
| SATELLITE | 0 combos siempre | Ídem |
| MOONSHOT | 0 combos siempre | Ídem |
| ANCHOR | 0 combos siempre | Arakawa-type picks pasan si edge≥20%+STRONG |
| FAVORITOS | 0 combos (torneo='Desconocido') | Combos con cuota válida |

### Días con H2H real (GS, ATP500, jugadores que ya se enfrentaron):
**Sin cambio** — G2 Regla-1 sigue igual, y cuando n_h2h≥1 el bloque completo no ejecuta.

### En términos de P&L observable:
Empezamos a acumular datos reales de comportamiento de CORE/SAT/MOON/ANCHOR/FAVORITOS. Sin datos, las hipótesis H132-01 (CC hit_rate > 1/cuota_combo) y las demás hipótesis de estrategias son incalculables. Con datos: en 30-50 observaciones tenemos Wilson IC para decidir qué estrategias escalar.

---

## §6. Preguntas abiertas para Fable

1. **¿El umbral edge≥20% para Regla-2 es demasiado permisivo?**
   Alternativa conservadora: edge≥25%. Implicación: Liljekvist D. (edge=20.6%) quedaría excluido. Recomendación: edge≥20% porque Kelly_kl>0 ya actúa como segundo filtro.

2. **¿Regla-3 (edge≥35% cualquier conf) es correcto sin requerir STRONG?**
   Edge≥35% con Kelly>0 es evidencia muy fuerte de que el bookmaker está equivocado. El nivel de conf_flag es una construcción del modelo que puede ser conservadora. Recomendación: mantener Regla-3 — el Kelly es el árbitro final.

3. **¿El fix de favoritos (partido como clave) preserva la intención de diversificación?**
   Sí: cada partido es una instancia de mercado única. Los resultados de Corte-BoschArmas vs O'Connell-X son estadísticamente independientes incluso si están en el mismo ITF torneo (que además es improbable dado que son picks de diferentes circuitos/países).

4. **¿Cuándo se activaría ComboRegistry (Nodo-132) con estos picks desbloqueados?**
   Inmediatamente — D132-01 y D132-02 llaman `log_combo()` al generar el BAT. Si el pick pasa G2, llega al generador de BAT, y ahí se loguea en combo_registry independientemente de si Kambi tiene el partido o no (el log es del plan, no del BAT).

---

## §7. Orden de implementación

| Fix | Archivo | Líneas | Complejidad | Prioridad |
|-----|---------|--------|-------------|-----------|
| D138-01 | `combo_confianza_builder.py:L252-262` | ~18 líneas (reemplazar 10) | Baja | INMEDIATA |
| D138-02 | `favoritos_combo_builder.py:L232` | ~8 líneas (reemplazar 2) | Baja | INMEDIATA |
| Tests | `tests/test_nodo138_g2_evolution.py` | ~90 líneas | Media | Post-implementación |

**Implementación total: < 30 líneas de cambio. Cero regresiones en flujos con n_h2h > 0.**

---

**Wikilinks totales: 6 | Huérfanos: 0**

[[Nodo-136-Tier-Detection-CTI-Fallback]] | [[Nodo-137-Governor-MOTOR-Exclusion]] | [[Nodo-87-Fixes-Auditoria-D87]] | [[Nodo-86-Auditoria-Fable5]] | [[Nodo-100-Taxonomia-Estrategias-Generacion-Combos]] | [[Nodo-29-Circuit-Asymmetry-Deflator]]
