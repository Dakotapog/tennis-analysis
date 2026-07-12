# Nodo-86 — Auditoría Fable 5 del Proyecto Completo

> **Wikilinks:** [[Nodo-51-Plan-Estrategico-Data-Layer-Torneo]] | [[Nodo-62-Signal-Bridge]] | [[Nodo-64-RFI-Return-From-Inactivity]] | [[Nodo-65-Convergencia-Multi-Senal-Patron-Combos]] | [[Nodo-72-Phantom-Identity-Guard]] | [[Nodo-74-Combo-Governor]] | [[Nodo-78-Protocolo-Auditoria]]
> **Fecha:** 2026-07-11
> **Estado:** DIAGNÓSTICO PURO — cero cambios de código aplicados. Cada fix se decide con el usuario, uno por uno.
> **Documento completo:** `docs/auditorias/AUDITORIA_FABLE5_2026-07-11.md` (6 secciones, evidencia archivo:línea)
> **Método:** análisis estático solo-lectura (sesión PowerShell, sin venv) + verificación empírica contra shadow book, calibracion_edge.json y apuestas_*.json en disco.

---

## Hallazgos principales (resumen — detalle y evidencia en el documento)

### Bugs no documentados previamente (Sección 1)
1. **Loop de calibración real roto:** apuestas reales llegan con `superficie="?"`, `tier="?"`, `match_id=""` → `betslip_registrar.py:459-487` escribe a buckets `?`/`?_?` que `theta_thompson` jamás lee. ~141 resultados de dinero real huérfanos, con **24% hit rate oculto** (vs 50-64% shadow).
2. **H62-01 muerta al nacer:** `shadow_book.update_alpha_flags` (líneas 633-634) busca campos `nombre/jugador/player` que no existen en pick_snapshot (`favorito_predicho`). Confirmado: 0 ocurrencias de `alpha_promoted` en todo el shadow book. D62-05 opera en producción sin el monitoreo que su hipótesis exige.
3. **`max(MIN_BET,…)` fuerza $1,000 con EV negativo o budget agotado** (`trader_ev_tenis.py:490-491`, 569, 767).
4. **`p_blend` con n_h2h=0 = accuracy del tier (ej. 0.758)** ignorando p_modelo → EV ficticio ordena y financia combos (`trader_ev_tenis.py:431-438`). Además el sizing NO usa el kelly_kl de 5 capas — recalcula kelly_quarter sobre p_blend.
5. **Gate GCS revive picks bloqueados** por T33-01/HOT_sin_BBI/n_axes<2 (`edge_calculator.py:996-999` corre después de los bloqueos soft sin verificarlos).
6. **pre_game_validator solo valida el pool `apostar`** — watchlist/sin_edge (que entran al pool del trader por defecto) nunca se validan.

### Patrón estructural nuevo (Sección 4 — el "siguiente nombre-vs-ID")
**"Degradación silenciosa de contexto en los bordes":** 6 fronteras donde un artefacto pierde campos (identidad/tier/superficie) y el receptor acepta el valor degradado como categoría válida. Solución unificadora: extender `core/data_contract.py` a contrato de schema por artefacto. Conexión clave faltante: **puente `outcome_id` (Kambi) ↔ `match_id` (FlashScore)** persistido en betslip_index — cose el hemisferio del modelo con el hemisferio del dinero.

### Señales (Sección 2)
- RFI (Nodo-64): **no existe código** — D64-01 nunca implementado; H76-01 congelada en n=1/30.
- `data_completeness` y `circuit_asymmetry`: observacionales huérfanas sin hipótesis ni fecha de eliminación.
- "Campeón reciente" contado 4 veces (TORNEO_COMPLETO_BONUS/GCS/tier_mismatch/H77-03) sin dueño común — el BONUS interno (anti-patrón Obradovic, Nodo-65 §2) sigue activo en todas las superficies aunque solo la variante hierba graduó.
- `cierre_kambi` solo en 5/10 días (~46% cobertura Momento 2) pese a n8n activo.

### ML (Sección 3)
Suspensión correcta. Vía de bajo riesgo: modelo de **calibración** (isotónica p_modelo→P(win), REPORTE_SOLO), no predictor. Prerequisito: arreglar loop de calibración real y settle de 07-05/07-10. El gate ">78% accuracy held-out" está mal especificado (la métrica del proyecto es P&L/Brier, no accuracy).

### Estrategia (Sección 6 — resumen)
STOP ITF dinero real (hard_itf 32.7% era_v2) → plomería primero (puente betslip, alpha_flags, `?`, MIN_BET floor, orden GCS) → disciplina de settle + governor cada sesión → apostar solo ANCHOR clay/grass GS + clay Challenger, stakes del trader **sin multiplicar** (el incidente 10× invalida VaR/CPPI) → escalar solo cuando el hit real converja con el shadow.

---

## Estado de implementación

| Fix | Decisión |
|---|---|
| Todos | PENDIENTE — requieren decisión del usuario, uno por uno, con nodo propio si aplica |

**Regla heredada:** este nodo es historia inmutable. Correcciones futuras → nodos nuevos referenciando este.
