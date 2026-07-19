# Nodo-102 — Hypothesis Tracking: H98-01 (Meta-Señal) + H100-01 (Triple Convergencia)

> **Wikilinks:** [[Nodo-98-Meta-Senal-Convergencia]] | [[Nodo-100B-Triple-Convergencia-Live]] | [[Nodo-101-Shadow-Book-Live-CLV]] | [[Nodo-52-Shadow-Book-CLV-Tracking]]
> **Fecha:** 2026-07-14 | **Autor:** Sonnet 4.6
> **Contexto:** H98-01 tenía n_actual=0 en preregistered_hypotheses.json — no se estaba
> acumulando evidencia para la Meta-Señal Convergencia. H100-01 no estaba registrada en el
> JSON a pesar de estar descrita en Nodo-100. Este Nodo cierra ambos gaps de tracking.

---

## 1. GAPS DETECTADOS

### G1 — H98-01 sin segmento en shadow_book --report

`score_directo` se calcula y serializa en edge_report (L851 edge_calculator.py, `_calc_meta_score_directo()`).
El campo aparece en todos los picks del shadow book desde Nodo-98.
Pero `report()` no tenía ningún segmento para `score_directo>=3`.
→ n_actual=0 en el JSON aunque hay picks con score_directo=2,3,4,5 en producción.

### G2 — H100-01 no registrada en preregistered_hypotheses.json

Nodo-100 doc menciona "H100-01 pre-registrada" pero el JSON solo tenía `H100-01: {}`.
Sin registro formal, el SPRT/LLR no puede evaluarse, y el reporte no muestra progreso.

---

## 2. DECISIONES

### D102-01: Añadir segmento `score_directo>=3` a report()

Posición: entre RIVAL VALUE y LIVE PICKS (antes de D99-02).
Predicado: `r.get('pick_snapshot', {}).get('score_directo', 0) >= 3`
Label: `"score_directo>=3 (H98-01: convergencia STRONG+HOT+RFI+IRP+ELO_DOM)"`
→ `_append_segment()` con el mismo formato que CAPA2 y ELO_DOMINANCE.

### D102-02: Añadir H98-01 y H100-01 al bloque HIPÓTESIS de report()

Posición: después de H52-08, antes de Graduación.
Usa `_append_hypothesis()` estándar: CONTINUAR/GRADUABLE/NO GRADUABLE.
- H98-01: n_stop=30, predicado `score_directo>=3`
- H100-01: n_stop=20, predicado `pick_type=='live'`

### D102-03: Registrar H100-01 en preregistered_hypotheses.json

Formato idéntico a H97-01 y H98-01 (campos: nombre, descripcion, origen_deuda, preregistrado,
umbrales_congelados, metrica, exito, n_stop, estado, n_actual, hits).

---

## 3. ARCHIVOS MODIFICADOS

| Archivo | Cambio | Líneas |
|---|---|---|
| `shadow_book.py` | Segmento score_directo>=3 en report() | +7 |
| `shadow_book.py` | H98-01 + H100-01 en HIPÓTESIS section | +12 |
| `validation/preregistered_hypotheses.json` | H100-01 completa | +1 entry |

---

## 4. FLUJO POST-NODO-102

```
shadow_book --report muestra:
  ...
  SEGMENTO: score_directo>=3 (H98-01: convergencia STRONG+HOT+RFI+IRP+ELO_DOM)*
    n=0  hit%=0.0  ...  [sin picks score>=3 settled aún]
  ...
  LIVE PICKS H100-01 (Triple Convergencia — pick_type=live)*:
    n=0  ...  [espera primer break_confirmado real]
  ...
  HIPÓTESIS:
    H98-01 [score_directo>=3 supera breakeven]: CONTINUAR (n=0/30)
    H100-01 [BREAK_CONFIRMADO picks superan breakeven live]: CONTINUAR (n=0/20)
  ...
```

Cuando el sistema acumule `n>=3` picks con `score_directo>=3` settled, el segmento
comenzará a mostrar IC Wilson y CLV.

---

## 5. TESTS (REGLA-T53 — 4 tests)

```
tests/test_nodo102_hypothesis_tracking.py

test_score_directo_3_aparece_en_segmento()      — picks score=3 en segmento, score=2 excluidos
test_h9801_en_hipotesis_continuar()             — H98-01 visible en bloque HIPÓTESIS
test_h10001_en_hipotesis_continuar()            — H100-01 visible en bloque HIPÓTESIS
test_h10001_registrada_en_json()                — H100-01 presente en preregistered_hypotheses.json
```
