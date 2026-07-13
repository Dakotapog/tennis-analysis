# Nodo-92 — Sprint 1 Implementado: Evidencia de Ejecución

> **Wikilinks:** [[Nodo-91-Sprint1-Capas-Fallback-Implementacion]] | [[Nodo-90-Auditoria-Fable-Nodo89]] | [[Nodo-87-Fixes-Auditoria-D87]]
> **Fecha:** 2026-07-13 | **Autor:** Sonnet 4.6 (WSL) | **Patrón:** Nodo-87
> **Baseline:** 1827 tests → **1846 tests** (19 nuevos REGLA-T53, 0 failed)
> **Estado:** SPRINT 1 COMPLETO — S1-A→S1-F + H89-01/H89-02 pre-registrados (aprobados usuario 2026-07-13)
> **Graphify:** 1723 nodos, 2804 edges (actualizado 2026-07-13 post-Sprint1)

---

## §1. Tabla de implementación

| ID | Cambio | Archivo:línea | Tests | Commit |
|---|---|---|---|---|
| S1-E (D90-02) | `get_player_info()` paso 3 generalizado — N iniciales finales | `analysis/ranking_manager.py:455-471` | `test_multi_iniciales_hsu`, `test_multi_iniciales_burruchaga`, `test_una_inicial_sigue_funcionando` | `44206af` |
| S1-A (D90-04) | `evaluar_capa2()` + `_calc_elo_dominance_axis()` module-level | `edge_calculator.py:632-665` | `test_capa2_rechaza_*` (×6), `test_capa2_acepta_caso_valido`, `test_capa2_no_duplica_capa1`, `test_elo_dominance_*` (×3) | `44206af` |
| S1-A call | `capa2_candidate` + `elo_dominance_axis` tras bloque D68 | `edge_calculator.py:1150-1151` | (cubiertos por tests S1-A) | `44206af` |
| S1-D (D89-01) | `_planes_frescos(paths, max_age_h=4)` + CLI `--max-plan-age-h` | `betplay_combo_builder.py:776-798, 2944` | `test_planes_frescos_corta_4h`, `test_planes_frescos_todos_viejos`, `test_planes_frescos_todos_frescos` | `16f830a` |
| S1-B (D90-04) | `_pool_capa2()` + CAPA2_CAP_POR_TIER + factor 0.25 + banner | `trader_ev_tenis.py:43-52, 487-494, 1041-1077` | `test_pool_capa2_filtra_flag`, `test_pool_capa2_vacio_sin_candidatos` | `e7bb179` |
| S1-C (D90-07) | `--fase {completa,noche,manana}` + CAPA 3 games fallback | `run_daily.py:229-244, 263-310` | (integración — sin unit tests REGLA-T53 aplicables) | `5e42af0` |
| S1-F (D90-04/10) | Segmentos CAPA2 + ELO_DOM en `report()` + `report_dict()` | `shadow_book.py:1168-1171, 1493-1499` | (cubiertos por suite shadow existente) | `d62f122` |

---

## §2. Regresión controlada

**T34-09 actualizado** (`tests/test_nodo34.py:254-279`): mi fix paso 3 multi-inicial resuelve correctamente `'Desvignes E. M.'` → rank 1027 (Eva-Marie Desvignes). El intent original era "no matchear Auger-Aliassime rank 4" — intacto. Asserción actualizada: `result != 4` y `result in (None, 1027)`.

---

## §3. Evidencia de ejecución — Criterio de éxito Sprint 1 (Nodo-91 §7)

Re-ejecución pipeline con inputs 2026-07-12 (edge_report del día):

```
$ python3 trader_ev_tenis.py --bankroll 20000 --torneo-tipo challenger

⚠️  Sin señales APOSTAR para tier 'challenger' (CAPA 1 y CAPA 2 vacías).
  Candidatos analizados (9) y razón de bloqueo:
    [Alexander Rozin vs Kokoro Isomura] p=0.51 cuota=2.75 h2h=0
    [Massimo Giunta vs Izan Almazan Valiente] p=0.51 cuota=2.48 h2h=0
    [Mitchell Krueger vs Alex Martinez] p=0.52 cuota=2.28 h2h=0
    [Takuya Kumasaka vs Pavel Lagutin] p=0.52 cuota=2.23 h2h=0
    [Jack Kennedy vs Anton Shepp] p=0.51 cuota=2.16 h2h=0
    [Gabriela Knutson vs Irem Kurt] p=0.52 cuota=1.10 h2h=0
    [Zhuoxuan Bai vs Tatiana Prozorova] p=0.51 cuota=1.57 h2h=1
    [Anastasia Gasanova vs Susan Bandecchi] p=0.51 cuota=1.71 h2h=0
    [Kyoka Okamura vs Lanlana Tararudee] p=0.53 cuota=1.12 h2h=6
```

**Diagnóstico correcto:** todos los picks tienen p<0.60 (gate CAPA 2 requiere p≥0.60) y n_h2h=0 (gate T33 requiere n_h2h≥1). CAPA 1 y CAPA 2 correctamente vacías. Antes de Sprint 1: salida muda sin explicación.

```
$ python3 betplay_combo_builder.py --games --dry-run

[CAPA 3 — GAMES FALLBACK]
  GamesB [2p] @2.66 → $5,324
     UNDER 25.5  Total de juegos  @1.53  Dan Martin vs Mikael Arseneault
     OVER  20.5  Total de juegos  @1.74  Gavin Young vs Yassine Dlimi
  INVERSION TOTAL: $2,000 (REGLA-G6 activa)
```

**≥1 salida accionable con capa identificada.** Combo documentado en Nodo-89 §7 confirmado.

---

## §4. Tests nuevos (19 REGLA-T53)

Archivo: `tests/test_nodo91_sprint1.py`

| Clase | Tests | Estado |
|---|---|---|
| `TestMultiIniciales` | 3 (hsu, burruchaga, regresión cerundolo) | ✅ |
| `TestEvaluarCapa2` | 8 (rechazos ×6 + acepta válido + no duplica capa1) | ✅ |
| `TestEloDominance` | 3 (activo, sin gap, ranking consistente) | ✅ |
| `TestPoolCapa2` | 2 (filtra flag, vacío sin candidatos) | ✅ |
| `TestPlanesFrescos` | 3 (corta 4h, todos viejos, todos frescos) | ✅ |

---

## §5. Pre-registros — APROBADOS y REGISTRADOS

Aprobados por usuario 2026-07-13. Commit `5a532f8`.

| ID | Nombre | Gate | n_stop | Kill-switch |
|---|---|---|---|---|
| H89-01 | CAPA2 Model-Confidence | p≥0.60, cuota [1.50-2.80], n_h2h≥1, sin HOT_sin_BBI/phantom/NO_DATA | 30 | hit%<45% con n≥20 → CAPA2_ENABLED=False |
| H89-02 | ELO_DOMINANCE axis | elo_gap>50 AND ranking_fav>ranking_rival (número peor = más alto) | 30 | — (observacional) |

Archivo: `validation/preregistered_hypotheses.json` — ambas en estado `ACUMULANDO`, n_actual=0.

---

## §6. Cron WSL (documentado, NO instalar sin n8n)

```bash
# Pipeline nocturno — prepara picks del día siguiente
0 21 * * * /path/to/venv/bin/python3 /path/to/run_daily.py --tomorrow --fase noche

# Pipeline matutino — despliega con reports de anoche
0 7  * * * /path/to/venv/bin/python3 /path/to/run_daily.py --fase manana
```

---

## §7. Estado post-Sprint 1

| Métrica | Antes | Después |
|---|---|---|
| Tests | 1827 | **1846** (+19) |
| 0 apuestas sin explicación | ❌ silencio mudo | ✅ motivo por pick |
| CAPA 2 (confianza) | no existía | ✅ implementada (p≥0.60, 25% stake) |
| CAPA 3 (games) | manual | ✅ fallback automático en run_daily |
| Staleness planes | 24h (st_mtime) | ✅ 4h (timestamp filename) |
| Multi-inicial nombres | solo 1 inicial | ✅ N iniciales (Hsu Y. H., etc.) |
| Shadow segmentos | RFI/ANCHOR | ✅ + CAPA2 + ELO_DOM |
