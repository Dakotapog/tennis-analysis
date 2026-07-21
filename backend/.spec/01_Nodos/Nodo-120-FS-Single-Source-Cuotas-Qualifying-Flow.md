# Nodo-120 — FS Single-Source Cuotas: Qualifying Flow Completo

> Estado: IMPLEMENTADO 2026-07-19
> Extiende: [[Nodo-118-Match-Ledger-Crosswalk-Identidad-Fusion-Definitiva]] F5→F6
> Tests: 3 REGLA-T53 — `tests/test_nodo120_fs_cuotas.py` (2201 passed total)

---

## 0. Diagnóstico en lenguaje simple

El modelo predecía partidos de qualifying correctamente (51.9% hit en n=77), pero **nunca
se apostaban**. ¿Por qué? Betplay no lista esos partidos en Kambi. El sistema buscaba la cuota
de Betplay → no la encontraba → descartaba el partido silenciosamente.

Lo que no sabía el sistema: **FlashScore SÍ tiene cuotas para esos partidos**, y el scraper
Playwright ya las descargaba. Estaban guardadas en el archivo. Simplemente había una
**compuerta cerrada** que las bloqueaba antes de llegar al motor de análisis.

La compuerta era una línea en `match_ledger.py` que decía *"exporta solo picks con cuota de
Betplay/Kambi"*. Se añadió: *"…Y TAMBIÉN picks con cuota de FlashScore si la tienen"*.

**Resultado**: pipeline pasa de ~67 partidos procesados a ~100 (+33 qualifyings con cuotas
FlashScore). El trader verifica la cuota actual antes de apostar — igual que siempre hace
en PASO 3.5.

---

## 1. Root cause técnico

```
Playwright → cuota1=1.8, cuota2=2.1, sin contraparte Kambi
→ fusionar_dia() L376-379: ss_fs_list  ← cuotas SÍ están en el dict
→ exportar_para_edge_calculator() L479: EXCLUYE single_source_fs  ← ÚNICO BLOQUEADOR
→ ninja_h2h_parser load_matches() L960: habría PASADO (cuota1 is not None)
→ edge_calculator L902: habría PASADO (not 1.8 = False)
RESULTADO: picks válidos → invisibles → dinero perdido
```

**Evidencia producción 2026-07-19**: 131 Playwright → 100 con cuotas → 67 joins →
**33 ss_fs CON cuotas** bloqueados. Accuracy del día: 71.2%. Apuestas qualifying: 0.

---

## 2. Decisiones

### D120-01 — Incluir ss_fs con cuota>0 en exportar_para_edge_calculator
**Qué**: Loop en L479 extiende a `joins + single_source_kambi + ss_fs_con_cuotas`
donde `ss_fs_con_cuotas` filtra: `cuota1 is not None, cuota2 is not None, cuota1>0, cuota2>0`.

**Por qué**: La lógica downstream (ninja_h2h_parser, edge_calculator) ya maneja
cuota1/cuota2 correctamente. El único bloqueador era la exclusión en el adapter.

**Límite**: `fusionar_dia()` no se toca — cambia solo el adapter de exportación.

### D120-02 — _cuota_source='flashscore' para trazabilidad
Campo informacional en el registro exportado. Permite al shadow_book y al trader
saber que la cuota viene de FlashScore, no de Kambi/Betplay directamente.

- `_cuota_source='kambi'` → joins y single_source_kambi
- `_cuota_source='flashscore'` → single_source_fs incluidos

### D120-03 — Filtro double-check float(cuota)>0
Aunque cuota1=0 es raro, el Playwright extractor puede retornar 0 explícito cuando
el selector falla parcialmente. El filtro `float(cuota1)>0` excluye estos casos.

### D120-04 — Nota operativa para el trader
Las cuotas FlashScore para qualifying ≠ cuotas Betplay (Betplay no las lista en Kambi).
La apuesta se hace verificando cuota actual en FlashScore/Betplay. Igual que PASO 3.5.
Edge_calculator genera la recomendación; el trader ejecuta con cuota real disponible.

---

## 3. Hipótesis pre-registrada

### H120-01 — ss_fs_con_cuotas picks (qualifying FlashScore) superan breakeven
- **Predicado**: picks `_cuota_source='flashscore'` tienen hit% > breakeven de su cuota media
  (estimado ~45% para cuotas ~2.0–2.5)
- **Gate**: n_stop=20 settled picks de tipo `_cuota_source='flashscore'`
- **Kill-switch**: hit% < 35% con n ≥ 15 → revisar confiabilidad de cuotas FlashScore qualifying
- **Estado**: ACUMULANDO (0/20 al 2026-07-19)
- **Evidencia previa**: NO_DATA n=77 hit=51.9% — mezcla fuentes; H120-01 purifica el segmento

---

## 4. Cambio en código

**Archivo**: `scraping/match_ledger.py`
**Función**: `exportar_para_edge_calculator()` ~L479

```python
# ANTES:
for p in ledger.get("joins", []) + ledger.get("single_source_kambi", []):

# DESPUÉS (D120-01):
ss_fs_con_cuotas = [
    p for p in ledger.get("single_source_fs", [])
    if p.get("cuota1") and p.get("cuota2")
    and float(p.get("cuota1", 0)) > 0 and float(p.get("cuota2", 0)) > 0
]
joins = ledger.get("joins", [])
ssk = ledger.get("single_source_kambi", [])
logger.info(f"   Exportados: {len(joins)} joins + {len(ssk)} kambi + "
            f"{len(ss_fs_con_cuotas)} ss_fs_con_cuotas (Nodo-120)")

for p in joins + ssk + ss_fs_con_cuotas:
    # ... mismo body, añadido:
    "_cuota_source": "flashscore" if p.get("join_method") == "SINGLE_SOURCE_FS" else "kambi",
```

---

## 5. Tests REGLA-T53

**Archivo**: `tests/test_nodo120_fs_cuotas.py` (3 tests, todos GREEN)

| Test | Verifica |
|------|----------|
| `test_ss_fs_con_cuotas_incluido_en_export` | D120-01: ss_fs cuota>0 → 3 registros exportados |
| `test_ss_fs_sin_cuotas_excluido_del_export` | D120-03: ss_fs cuota=None → excluido |
| `test_ss_fs_cuota_source_es_flashscore` | D120-02: trazabilidad _cuota_source correcta |

---

## 6. Impacto esperado

| Métrica | Antes | Después |
|---------|-------|---------|
| Partidos exportados al H2H | ~67 | ~100 |
| Picks qualifying en edge_report | 0 | ~30/día |
| NO_DATA sin apuesta | 77 picks perdidos | qualifying picks con recomendación |
| Segmento trackeable H120-01 | n/a | acumulando desde 2026-07-20 |

---

## §WIKILINKS COMPLETOS

### Forward links (este nodo depende de)
- [[Nodo-118-Match-Ledger-Crosswalk-Identidad-Fusion-Definitiva]] — F1-F5 base; exportar_para_edge_calculator es F4
- [[Nodo-117-Auditoria-Scraping-Rankings-Cobertura-H2H]] — contexto scraping qualifying
- [[Nodo-87-Fixes-Auditoria-D87]] — NO_DATA tracking (segmento base n=77)
- [[Nodo-96-Sprint5-IRP]] — ninja_h2h_parser consume el mismo exported JSON
- [[Nodo-91-Sprint1-CAPA2-ELO]] — edge_calculator downstream no modificado
- [[Nodo-64-RFI-Return-From-Inactivity]] — rfi_tier serializado junto con _cuota_source

### Back links (nodos que deben conocer este)
- [[Nodo-118-Match-Ledger-Crosswalk-Identidad-Fusion-Definitiva]] ← F6 addendum aquí
- [[Nodo-119-Auditoria-Desk-v3-21-Gaps-11-Fixes]] ← NO_DATA origen diagnosticado aquí
- [[Nodo-121-OddsAggregator-Cuota-Enrichment-ss-fs]] ← complementario: enriquece ss_fs cuota=None via betplay/rushbet (2026-07-20)

### Huérfanos operacionales
- `validation/preregistered_hypotheses.json` — H120-01 registrada 2026-07-19
- `nodos_index.json` — reindexado 2026-07-19 (117 nodos)
