# MOC — GCS Grass Court Signal

> **Tipo:** Map of Content | **Creado:** 2026-07-16 (D105-02)
> **Wikilinks:** [[Nodo-60-GCS-Grass-Surface-Champion-Signal]] | [[Nodo-61-GCS-Season-Window-Fix]]

---

## Nodos en este cluster

| Nodo | Tema | Estado |
|---|---|---|
| [[Nodo-60-GCS-Grass-Surface-Champion-Signal]] | Spec original GCS — prior bayesiano hierba, `_GCS_BOOST_ENABLED`, tiers ATP500+ | completo |
| [[Nodo-61-GCS-Season-Window-Fix]] | Implementación `rivalry_analyzer.py` — boost activo solo en hierba con condiciones estrictas | completo |

## Hipótesis registrada

**H60-01** — GRADUADA 2026-07-10 (n=54, 64.8% hit rate)
- Gate original: n≥30 — alcanzado y superado
- `_GCS_GATE_ENABLED = True` en `config.py`
- Mejor estrategia del sistema por hit% formal (supera ANCHOR, Challenger, ITF)

## Relación con el sistema

- Activo solo en hierba (`superficie == 'grass'`)
- Contribuye automáticamente al combo builder cuando `gcs_active = True` en el edge report
- Prior A60-01 documentado en `calibracion_edge.json`
