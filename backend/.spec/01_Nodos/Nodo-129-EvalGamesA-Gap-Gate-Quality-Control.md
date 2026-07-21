# Nodo-129 — EvalGamesA Gap Gate: Control de Calidad de Señales UNDER

> **Estado:** PROPUESTA — pendiente auditoría Fable
> **Tipo:** FIX — calidad de combo, no corrección de bug estructural
> **Trigger:** Evidencia real 2026-07-21: combo 3p @7.79x muerto en vivo (Frey sets en 3er set, Bonding en 3er set)
> **Autor:** Sonnet 4.6 (análisis doctoral — Nodo-128 context)
> **Para auditoría:** Fable debe verificar (1) hipótesis estadística, (2) umbral propuesto, (3) coherencia con Nodo-40 REGLA-G6

---

## Wikilinks

| Link | Rol |
|------|-----|
| [[Nodo-126-Auditoria-EvalGames-Bridge-Fugas-Fixes]] | D126-01 same-match gate — precedente de calidad en combo |
| [[Nodo-125-EvalGames-Bridge-Dashboard-X4]] | `build_evaluar_games_combos()` — función a modificar |
| [[Nodo-40-Games-Sets-Signal-Layer]] | `_seleccionar_señal_optima()` — define confianza_señal y gap_juegos |
| [[Nodo-128-D126-04-Overgeneral-ITF-Filter-Fix]] | Contexto: D128-02 recuperó picks ITF que fueron a este combo |
| [[Nodo-100-Taxonomia-Estrategias-Generacion-Combos]] | Estrategia #11 GAMES — referencia de gate cuota_combo ≥ 2.50 |

**Wikilinks totales: 5 | Huérfanos: 0**

---

## §1. Evidencia que origina esta propuesta

### §1.1 El combo del 2026-07-21 y su resultado en vivo

Combo generado por `build_evaluar_games_combos()` hoy:

```
EvalGamesA [3p] @7.79x
  Bonding O. vs Bynoe E.   — UNDER 29.5 juegos @2.02  gap=+10.5j  ALTA
  Holmgren A. vs Johns G.  — UNDER 21.5 juegos @1.88  gap=+2.5j   MEDIA
  Frey A. vs Lu J.         — UNDER 2.5 sets    @2.05  gap=None     MEDIA
```

**Estado en vivo (verificado live-tennis.cn, 2026-07-21 ~13:15h):**

| Pierna | Apuesta | Estado real | Veredicto |
|--------|---------|-------------|-----------|
| Bonding vs Bynoe | UNDER 29.5 juegos | 3er set en curso | EN PELIGRO |
| Holmgren vs Johns | UNDER 21.5 juegos | Sin iniciar (Winnipeg CH75) | Pendiente |
| Frey vs Lu | UNDER 2.5 sets | 3er set en curso (7-6, 1-6, ...) | **PERDIDO** |

El combo muere por la pierna Frey — exactamente la pierna con `gap=None` y `conf=MEDIA`.

### §1.2 Contraste con X3 GAMES SIGNAL del mismo día (todos acertaron)

Del `games_signal_report_20260721_102625.json`, señales ALTA que el usuario confirma acertadas:

```
Van De Zandschulp UNDER 25.5  gap=+6.5j  diff=0.450  source=model_real
Gorzny UNDER 23.5             gap=+4.5j  diff=0.590  source=model_real
Johnson UNDER 24.5            gap=+5.5j  diff=0.570  source=model_real
Suresh UNDER 23.5             gap=+4.5j  diff=0.440  source=model_real
Moro Canas UNDER 23.5         gap=+4.5j  diff=0.380  source=model_real
Cassone UNDER 23.5            gap=+4.5j  diff=0.430  source=model_real
Matsuoka UNDER 23.5           gap=+4.5j  diff=0.440  source=model_real
Carballes Baena UNDER 23.5    gap=+4.5j  diff=0.380  source=model_real
```

**Gap mínimo X3 ALTA:** +4.5j. **Gap promedio:** +5.0j.
**Gap combo EvalGamesA MEDIA:** +2.5j (Holmgren), **None** (Frey sets).

---

## §2. Diagnóstico — 3 diferencias estructurales

### §2.1 Diferencia de fuente de diff_abs (la más crítica)

```
X3 GAMES SIGNAL (games_signal_calculator.py):
  diff_abs = p1_final_weight - p2_final_weight
           = función(H2H + rankings + superficie + ELO + Markov)
  Captura: velocidad real del partido en esa cancha, con ese rival,
           en ese momento de forma, en esa superficie

EvalGamesA (evaluar_games_bridge.py):
  diff_abs = (1/cuota_ml - 0.5) × 2   ← proxy de UNA dimensión
  Captura: probabilidad de victoria implícita del bookmaker solamente
```

**Consecuencia directa:** El modelo X3 puede clasificar a un jugador como "diff=0.380" y aun así el gap sea +4.5j porque el modelo SABE que en arcilla este jugador juega largo incluso ganando. El bridge EvalGamesA ve "diff=0.600" (cuota @1.25) y concluye "dominante → juego corto", sin saber si es un grinder de clay o un servidor de hierba. La cuota ML captura "quién gana" — no "cuántos juegos".

### §2.2 Diferencia de margen de seguridad (gap)

```
gap_juegos = linea_kambi - predicted_games_max

X3 ALTA mínimo:    gap = +4.5j  (el partido puede extenderse 4 juegos sobre predicción y sigue ganando)
EvalGamesA MEDIA:  gap = +2.5j  (solo 2 juegos de margen — cualquier game de cortesía lo rompe)
EvalGamesA sets:   gap = None   (predicted_sets = línea_kambi exactamente — sin margen)
```

El gap no es decorativo. Es la diferencia entre una apuesta con margen operativo y una apuesta en el filo.

**Por qué gap=None en sets es inviable en combo:**
- `UNDER 2.5 sets` significa "el favorito gana en 2 sets rectos"
- Cuota @2.05 → el bookmaker asigna ~49% de probabilidad a sets rectos
- El modelo EvalGamesA usa `total_score=0.5` (neutral hardcodeado) → no tiene información de H2H sobre si este jugador específico tiende a ganar en 2 o 3 sets
- Resultado: apuesta con probabilidad ~49% incluida en un combo — destruye el EV del combo entero

### §2.3 Cross-validación ausente

Las señales X3 ALTA que acertaron tienen una propiedad implícita: **Kambi puso una línea alta** (ej. UNDER 23.5 cuando el modelo predice 16-19 games). Esto significa Kambi también estima que el partido podría llegar a 23+ juegos — discrepan con nuestro modelo en cantidad pero no en dirección. El gap grande es consecuencia de esa discrepancia.

En EvalGamesA, la línea Kambi para UNDER 21.5 (Holmgren) y UNDER 2.5 sets (Frey) sugiere que Kambi tiene mucha más incertidumbre sobre la duración del partido que sobre el ganador. Esa incertidumbre de Kambi debería ser señal de precaución, no de confianza.

---

## §3. Propuesta: Gate mínimo gap ≥ 4.0j en build_evaluar_games_combos()

### §3.1 La regla propuesta (D129-01)

**Archivo:** `betplay_combo_builder.py` — en `build_evaluar_games_combos()`, en el bloque de aplanado de señales:

```python
# D129-01: gate mínimo gap ≥ 4.0j para señales EvalGamesA en combo
# Racional: X3 ALTA históricamente aciertan con gap ≥ 4.5j.
# EvalGamesA usa proxy diff_abs → requiere margen extra de seguridad.
# Señales con gap < 4.0j o gap=None (sets sin datos) = fuera del combo.
_GAP_MIN_EVALUAR = 4.0
for s in p.get("señales_optimas", []):
    if not (s.get("apostar") and s.get("direccion") == "UNDER"):
        continue
    gap = s.get("gap_juegos") or 0
    if gap < _GAP_MIN_EVALUAR:
        continue  # D129-01: descartar señales con margen insuficiente
    all_signals.append({...})
```

### §3.2 Impacto en el combo de hoy (validación retroactiva)

Con D129-01 aplicado, el combo del 2026-07-21 habría sido:

```
Señales que PASAN gap ≥ 4.0j:
  Bonding vs Bynoe   UNDER 29.5  gap=+10.5j  ALTA  ✓
  Frey vs Lu         UNDER 27.5  gap=+8.5j   ALTA  ✓  ← juegos, NO sets
  Nishimura vs Marton UNDER 24.5 gap=+5.5j   ALTA  ✓

Señales EXCLUIDAS:
  Holmgren vs Johns  UNDER 21.5  gap=+2.5j   MEDIA ✗
  Frey vs Lu         UNDER 2.5sets gap=None  MEDIA ✗  ← la pierna que mató el combo

Combo resultante: Bonding + Frey_juegos + Nishimura = @(2.02 × 1.88 × 2.28) = @8.66x
```

**El combo alternativo @8.66x excluye la pierna que falló y tiene mayor cuota.**

### §3.3 Por qué 4.0j y no otro umbral

- **< 3.0j**: Demasiado permisivo — permite gap=2.5j (Holmgren), que es insuficiente dado el proxy model
- **4.0j**: Alinea con el mínimo histórico de X3 ALTA que aciertan (4.5j) con un margen de 0.5j de tolerancia por el diferente origen de diff_abs
- **≥ 5.0j**: Demasiado restrictivo — hoy habría excluido Nishimura (5.5j) reduciendo el combo
- **Señales sets (gap=None)**: Excluidas siempre en combo — pueden usarse solo si gap_sets ≥ 0.5 (futuro)

**Nota para Fable:** El umbral 4.0j es una propuesta empírica basada en UN día de evidencia (n=1). Fable debe validar si el umbral es estadísticamente justificable o si se necesita más data antes de hardcodearlo.

---

## §4. Hipótesis pre-registrada (H129-01)

```json
{
  "id": "H129-01",
  "descripcion": "EvalGamesA combos con gap_min ≥ 4.0j tienen hit_rate > breakeven",
  "formula": "hit_rate_combo_alta_gap > 1 / cuota_combo_promedio",
  "n_stop": 20,
  "baseline": "hit_rate_combo_media_gap (gap < 4.0j)",
  "estado": "PENDIENTE_IMPLEMENTACION",
  "fecha_registro": "2026-07-21"
}
```

---

## §5. Preguntas abiertas para auditoría Fable

1. **¿Es válido el umbral 4.0j con n=1 día de evidencia?**
   El análisis se basa en un solo día con resultados claros (X3 ALTA acertaron, EvalGamesA MEDIA falló). ¿Es suficiente evidencia para implementar un gate permanente, o se necesita validación retroactiva sobre más días?

2. **¿El gap_juegos de EvalGamesA es comparable al gap_juegos de X3?**
   Ambos usan `linea_kambi - predicted_games_max` pero `predicted_games_max` viene de modelos distintos (proxy cuota vs modelo real). ¿Son comparables o necesitamos un umbral diferente para EvalGamesA específicamente?

3. **¿Deberían las señales de SETS tener su propio gate o excluirse siempre del combo?**
   `gap=None` para sets porque el predictor no calcula margen en número de sets. ¿La solución correcta es (a) excluir sets siempre del combo, (b) implementar gap_sets con umbral distinto, o (c) permitir sets solo como pierna única no en combo?

4. **¿El total_score=0.5 hardcodeado en el bridge es el origen real del problema?**
   Si `total_score` fuera real (desde H2H o edge_report), el predicted_games sería más preciso y el gap más confiable. ¿Es este el fix correcto a largo plazo vs. el gate de gap?

5. **Coherencia con REGLA-G6 (Nodo-40):** REGLA-G6 limita stake máximo en games combos a $2,000. ¿El gate de calidad D129-01 debe ir acompañado de un ajuste de stake para combos que pasan el gate ALTA?

---

## §6. Tests REGLA-T53 — `tests/test_nodo129_evaluar_games_gap_gate.py`

```python
def test_D129_01_media_gap_25_excluded_from_combo()
    # señal gap=2.5j MEDIA no debe entrar al all_signals de EvalGamesA

def test_D129_01_alta_gap_45_included_in_combo()
    # señal gap=4.5j ALTA sí debe entrar

def test_D129_01_sets_gap_none_excluded()
    # señal UNDER 2.5 sets con gap=None nunca entra al combo

def test_D129_01_combo_uses_only_alta_gap_signals()
    # combo resultante solo contiene señales con gap >= 4.0j

def test_D129_01_retroactive_2026_07_21()
    # con D129-01: combo del día sería Bonding+Frey_juegos+Nishimura, no Bonding+Holmgren+FreySets

def test_H129_01_hypothesis_registered()
    # H129-01 en preregistered_hypotheses.json
```

---

## §7. Resumen ejecutivo para Fable

**Qué pasó:** Combo EvalGamesA 3p @7.79x generado el 2026-07-21 muere en vivo porque una de sus tres piernas (Frey vs Lu UNDER 2.5 sets) estaba en el 3er set cuando debería haber terminado en 2.

**Por qué pasó:** La pierna tenía gap=None (sin margen de seguridad) y cuota @2.05 (~49% implícita). La señal X3 ALTA del mismo día exige mínimo gap=+4.5j y acertó en todos los casos. La diferencia es que X3 usa modelo completo (H2H+rankings+superficie) mientras EvalGamesA usa solo el ML cuota como proxy de dominancia.

**Qué se propone:** Gate mínimo gap ≥ 4.0j en `build_evaluar_games_combos()`. Habría producido un combo alternativo @8.66x con 3 piernas ALTA (todas gap ≥ 5.5j) excluyendo las 2 piernas débiles.

**Qué necesita validación:** El umbral 4.0j se basa en evidencia de un día. Fable debe determinar si es estadísticamente defendible o si se necesita retroalimentación de más sesiones antes de implementarlo como gate permanente.

---

**Wikilinks totales: 5 | Huérfanos: 0**

[[Nodo-126-Auditoria-EvalGames-Bridge-Fugas-Fixes]] | [[Nodo-125-EvalGames-Bridge-Dashboard-X4]] | [[Nodo-40-Games-Sets-Signal-Layer]] | [[Nodo-128-D126-04-Overgeneral-ITF-Filter-Fix]] | [[Nodo-100-Taxonomia-Estrategias-Generacion-Combos]]
