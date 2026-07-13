# RETROSPECTIVO — Backtest Rival Value Flip (H88-01)

> **ETIQUETA: RETROSPECTIVO** — según lección C-05, este análisis NO actualiza
> calibración ni activa ningún gate. Solo orienta si vale la pena acumular prospectivamente.
> Fecha: 2026-07-12 | Nodo-68 D68-04

## Filtro aplicado (umbrales congelados H88-01)

- `edge_fav <= -0.10` (muy negativo, supera el vig típico 3-8%)
- `2.50 <= cuota_rival <= 8.00`
- `status != NO_DATA`, `phantom_data = false`
- Base: 177 picks settled en shadow book (total 277 logged)

## Resultados

| Métrica | Valor |
|---|---|
| n (no-VOID en segmento) | **0** |
| n_void excluidos | 0 |
| hit%_rival (% LOST fav) | — |
| IC Wilson 95% | — |
| ROI rival flat 1u | — |
| cuota_rival media | — |
| breakeven (1/cuota_media) | — |
| n_stop H88-01 | 30 |
| **Veredicto** | **CONTINUAR — n=0/30, sin picks retrospectivos en segmento** |

## Sub-segmento sin rfi_ultra

Sin picks en el segmento base — sub-segmento también vacío.

## Detalle de picks en el segmento

*(ninguno)*

## Interpretacion — hallazgo estructural

**El n=0 retrospectivo es esperado y correcto, NO es un problema.**

El shadow book trackea picks de los pools `apostar` y `watchlist` — ambos requieren
`edge > 0` para entrar al pipeline. Los picks con `edge_fav <= -0.10` nunca se
loggearon porque el sistema los excluia antes de llegar a `shadow_book.log_picks()`.

Esto significa:
1. Sin sesgo de supervivencia en datos retrospectivos — simplemente no existen.
2. La acumulacion de H88-01 empieza desde cero prospectivamente (2026-07-12).
3. El mecanismo funciona: D68-01 serializa `rival_value_flag=True` en picks con
   `edge_fav <= -0.10`, aunque el favorito no se apueste. D68-02 los trackeara
   en `shadow_book --report` cuando sean settled.

**Vale la pena seguir acumulando?** Si. El caso semilla (Obradovic -17.2% → Fabre
@5.20 gano) y el precedente Rivera/Michnev (Nodo-64) son motivadores aunque
sean retrospectivos. El costo de acumular es cero (no se apuesta nada hasta
graduacion H88-01, n_stop=30).

---
*Generado por D68-04 (Nodo-68). REPORTE_SOLO — no modifica calibracion ni activa gates.*
