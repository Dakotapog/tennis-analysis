# Nodo-98 — Meta-Señal: Convergencia IRP × RFI × Markov (Triángulo de Alpha)

> **Wikilinks:** [[Nodo-64-RFI-Return-From-Inactivity]] | [[Nodo-96-IRP-Individual-Return-Profile]] | [[Nodo-65-Convergencia-Multi-Senal]] | [[Nodo-95-Sprint4-PatternRecognition]] | [[Nodo-68]] | [[Nodo-97-Live-Edge-Monitor]]
> **Fecha:** 2026-07-14 | **Autor:** Sonnet 4.6 (conexión oculta — meta-modelo del pipeline)
> **Principio:** El bookmaker precio poblaciones. El sistema precio individuos. La intersección de 3 señales ortogonales sobre el mismo partido es el alpha más grande sin explotar del pipeline.

---

## 1. EL META-MODELO DESCUBIERTO

Recorriendo el pipeline completo, emerge una estructura de **5 niveles de ineficiencia anidadas**:

```
Nivel 1 — Ineficiencia de DATOS
  Bookmaker: no tiene H2H limpio, usa homónimos, rankings desactualizados
  Sistema:   Playwright primario, entity resolution canónica, PlayerDB 4650j

Nivel 2 — Ineficiencia de MODELO
  Bookmaker: ELO global, sin superficie, sin momentum
  Sistema:   ELO × superficie × Markov momentum (PELT)

Nivel 3 — Ineficiencia de CALIBRACIÓN
  Bookmaker: prior plano uniforme
  Sistema:   Thompson Beta por tier+superficie, shrinkage n/(n+20)

Nivel 4 — Ineficiencia de ATENCIÓN INDIVIDUAL
  Bookmaker: precio poblacional (todos los jugadores ITF = mismo descuento)
  Sistema:   IRP por jugador (Djokovic vuelve bien, jugador X vuelve mal)

Nivel 5 — Ineficiencia de CORRELACIÓN DE ERRORES
  Bookmaker: no sabe que cuando su favorito está mal priceado, el rival tiene valor
  Sistema:   H88-01 Rival Value Flip (3/3, combinada 41.25x)
```

**El nivel 6 (Nodo-97):** ineficiencia TEMPORAL — el mercado live tarda 90s en reaccionar.

---

## 2. EL TRIÁNGULO DE ALPHA (conexión oculta principal)

Cuando tres señales ortogonales apuntan al mismo partido, el bookmaker
tiene tres capas de información que no ve simultáneamente:

```
        RFI
       (¿cuánto tiempo lleva inactivo?)
            ↓
    IRP ←——— PARTIDO ———→ Markov
(¿cómo rinde ÉL al volver?)    (¿su rival está en HOT?)
```

**Ejemplo de convergencia máxima:**
- Jugador A: 200 días inactivo (RFI-2 ≥180d)
- IRP de A: delta_return = -0.18 (rinde 18% peor al volver)
- Rival B: estado Markov HOT (últimas 5 semanas positivas)
- Resultado: bookmaker favorece a A por ranking → modelo favorece a B por las 3 señales

El bookmaker ve solo: "A tiene mejor ranking".
El sistema ve: "A vuelve de 200 días, históricamente pierde 18% más al volver, y su rival está en racha".

**Este triángulo no tiene hipótesis pre-registrada todavía.** Es el alpha más grande sin explotar.

---

## 3. DECISIONES

| ID | Decisión |
|---|---|
| D98-01 | Definir `meta_signal_score` = suma ponderada de señales activas sobre el mismo partido |
| D98-02 | Señales incluidas: RFI tier (0-3), IRP delta_return (<-0.10 activa), Markov rival HOT, STRONG confidence, ELO dominance, Rival Value (edge_fav <-0.10) |
| D98-03 | `meta_signal_score` se serializa en edge_report como campo observacional (REPORTE_SOLO) |
| D98-04 | Umbral de convergencia alta: ≥3 señales activas simultáneas — hipótesis H98-01 pre-registrada |
| D98-05 | Dashboard: nueva sección "Convergencia Meta-Señal" — tabla de picks con score ≥3 |
| D98-06 | CLV como métrica primaria del sistema (no hit%) — medir si vencemos closing line sistemáticamente |
| D98-07 | `scripts/meta_signal_scorer.py` — REPORTE_SOLO, lee edge_report del día, emite `meta_signal_FECHA.json` |

---

## 4. `meta_signal_score` — Componentes

| Señal | Condición activa | Peso |
|---|---|---|
| RFI tier | rfi_tier ≥ 1 del rival (inactivo ≥90d) | +1 |
| IRP delta | irp_rival.delta_return < -0.10 | +1 |
| Markov HOT | markov_favorito = HOT | +1 |
| STRONG | confidence_flag = STRONG | +1 |
| ELO dom. | elo_dominance_axis = True | +1 |
| Rival Value | edge < -0.10 (rival tiene valor) | +1 (señal contraria) |

Score máximo: 5 (dirección favorable) + 1 (Rival Value, dirección opuesta)

---

## 5. HIPÓTESIS PRE-REGISTRADA

**H98-01:** "Picks con meta_signal_score ≥ 3 tienen hit% significativamente superior al breakeven de su cuota media"
- `n_stop`: 30 picks con score ≥ 3
- `exito`: IC Wilson 95% inferior > 1/cuota_media
- `estado`: ACUMULANDO
- `preregistrado`: 2026-07-14

---

## 6. CLV COMO MÉTRICA PRIMARIA (D98-06)

La brecha hit%_real vs hit%_shadow (-23.5pp) confunde a operadores novatos.
El CLV es la métrica correcta del proceso:

> "Un sistema que vence el closing line el 55%+ del tiempo gana dinero
>  a largo plazo, sin importar los resultados del mes."

El shadow book ya trackea CLV. Lo que falta: reportarlo como **KPI principal**
en el dashboard — por encima del hit%.

---

## 7. IMPLEMENTACIÓN

### 7.1 `scripts/meta_signal_scorer.py`

Lee `edge_report_FECHA.json` (todas las secciones: apostar + watchlist + sin_edge).
Para cada pick calcula `meta_signal_score` y emite `reports/meta_signal_FECHA.json`.

### 7.2 `edge_calculator.py` — campo `meta_signal_score`

Después del bloque IRP (Nodo-96), añadir cálculo del score (4 señales disponibles en ese punto):
`confidence_flag`, `markov_favorito`, `elo_dominance_axis`, `rfi_tier`, `irp_rival.delta_return`.

### 7.3 Dashboard — sección "Convergencia"

Tabla ordenada por `meta_signal_score` desc. Picks con score ≥ 3 resaltados.

---

## 8. OUTPUT EVIDENCIA (obligatorio)

`reports/meta_signal_YYYYMMDD_HHMMSS.json`:
```json
{
  "fecha": "2026-07-14",
  "n_picks_analizados": 47,
  "picks_score_3plus": [
    {
      "partido": "Boogaard vs Onclin",
      "meta_signal_score": 3,
      "senales_activas": ["HOT", "STRONG", "RFI_tier1"],
      "edge": 0.220,
      "cuota": 3.55,
      "tier": "challenger"
    }
  ],
  "h98_01_n_actual": 0
}
```

---

## 9. TESTS (REGLA-T53)

`tests/test_nodo98_meta_signal.py` — mínimo 8 tests:
1. `test_score_0_cuando_ninguna_senal` — pick sin señales → score=0
2. `test_score_3_cuando_hot_strong_rfi` — 3 señales → score=3
3. `test_irp_delta_activa_cuando_menor_umbral` — delta<-0.10 → cuenta
4. `test_rival_value_cuenta_como_senal_contraria` — edge<-0.10 → score=1 Rival Value
5. `test_solo_score_3plus_en_output_destacados` — filtro correcto
6. `test_output_json_escrito` — archivo generado
7. `test_scorer_no_modifica_edge_ni_kelly` — REPORTE_SOLO invariante
8. `test_h98_01_counter_incrementa` — n_actual sube con cada pick score≥3

---

## 9. GAPS CERRADOS PRE-AUDITORÍA (2026-07-14)

### D98-08 — CLV threshold operacional
"Vencer el closing line" = cuota_log > cuota_cierre_kambi en valor esperado.
CLV% = (cuota_log - cuota_cierre) / cuota_cierre × 100.
Threshold: sistema sano si CLV_mediana > 0 con n ≥ 50 picks.
El shadow book ya serializa `cierre_kambi.cuota` y `resolucion.cuota_cierre`.
`pipeline_tracker.py --section shadow` ya lo muestra — falta prominencia en dashboard.

### D98-09 — Orden correcto de señales en `meta_signal_score`
Las señales deben calcularse DESPUÉS de que todos los campos estén disponibles en el pick:
`confidence_flag` (edge_calculator L890) → `markov_favorito` (L874) → `elo_dominance_axis` (Sprint1)
→ `rfi_tier` (L1103) → `irp_rival.delta_return` (Nodo-96, L1163).
El score se añade AL FINAL del pick dict, después del bloque IRP. No antes.

### D98-10 — Rival Value señal CONTRARIA no suma al combo direccional
En D98-02 se incluyó Rival Value (edge<-0.10) como señal. Aclaración:
- Rival Value activo → el RIVAL tiene valor, no el favorito del modelo
- En meta_signal_score el Rival Value suma en dirección OPUESTA
- En combo live: si Rival Value = True, el pick va al RIVAL a su cuota, no al favorito
- Separar en output: `score_directo` (señales pro-favorito) vs `score_rival_value` (señal inversa)

---

## 10. PRECONDICIONES

- `edge_report_FECHA.json` del día con campos: `confidence_flag`, `markov_favorito`,
  `elo_dominance_axis`, `rfi_tier`, `irp_fav`, `irp_rival`, `edge`
- `data/irp_profiles.json` existe (Nodo-96)
- H98-01 pre-registrada en `preregistered_hypotheses.json`
