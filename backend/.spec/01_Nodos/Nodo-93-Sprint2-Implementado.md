# Nodo-93 — Sprint 2 Implementado: PlayerDB + kambi_disponible

> **Wikilinks:** [[Nodo-92-Sprint1-Implementado]] | [[Nodo-90-Auditoria-Fable-Nodo89]] | [[Nodo-91-Sprint1-Capas-Fallback-Implementacion]]
> **Fecha:** 2026-07-13 | **Autor:** Sonnet 4.6 (WSL) | **Patrón:** Nodo-87
> **Baseline:** 1846 tests → **1901 tests** (55 nuevos REGLA-T53, 0 failed)
> **Estado:** SPRINT 2 COMPLETO — D90-03 + D90-01

---

## §1. Tabla de implementación

| ID | Cambio | Archivo:línea | Tests | Commit |
|---|---|---|---|---|
| D90-03 | `scripts/build_player_db.py` batch desde historial_* settled (113 H2H). 4650 jugadores, 328260 filas deduplicadas. Dedupe por (slug, fecha_iso, oponente_raw). Stats: surface/tier/ranking_gap/prs. Subproducto: `player_alias_table.json`. | `scripts/build_player_db.py` completo | 35 tests `TestParseFilenameDate`, `TestFechaToIso`, `TestNormalizeSuperficie`, `TestThreeSetMatch`, `TestRankingBracket`, `TestComputePlayerStats`, `TestProcessFiles`, `TestBuildIndex` | `ac59038` |
| D90-01 | `scripts/fetch_kambi_coverage.py`: fetcha Kambi tennis API → `reports/kambi_coverage_FECHA.json` (players_normalized + event_pairs). Edge_calculator: `_load_kambi_coverage_once()` + `_annotate_kambi()` + `resultado['kambi_disponible']` = True/False/None. Sin HTTP en edge_calculator. | `scripts/fetch_kambi_coverage.py` + `edge_calculator.py:632-668, 1152` | 20 tests `TestNormalizeName`, `TestIsPlayerAvailable`, `TestLoadCoverage`, `TestAnnotateKambi` | `a7691f0` |

---

## §2. Outputs generados

| Archivo | Tamaño | Descripción |
|---|---|---|
| `data/player_db.json` | 186 MB | DB completa con rows crudas (para incremental futuro) |
| `data/player_db_index.json` | 1.8 MB | Resumen por jugador: surface/tier win_rates, PRS — carga en runtime |
| `data/player_alias_table.json` | 5.8 MB | slug → lista de nombres de oponentes vistos |

---

## §3. Diseño D90-03 (PlayerDB)

### Fuente de datos
- 113 archivos `h2h_results_enhanced_*.json`
- Bloques `historial_*` settled (jugador ya conocido, resultado confirmado)
- Descarta: resultado `'-'` o `'0-0'` (walkovers / sin datos)

### Deduplicación
- Clave: `(jugador_slug, fecha_iso, oponente_raw)`
- En colisión: prefiere la fila del archivo más reciente (`ranking_asof` mayor)
- Impacto: 1,262,941 filas brutas → 328,260 deduplicadas

### `resolution_confidence`
- Siempre `"exact"`: el jugador viene de la clave `historial_{slug}` del H2H, fuente ya canónica
- Cumple R2 de Nodo-90 §3: toda fila lleva `resolution_confidence`

### `own_ranking` / `ranking_asof`
- Caveat documentado (Fable D90-03): `own_ranking` = ranking en fecha de extracción del archivo, no del partido
- `ranking_asof` = fecha del archivo fuente
- Brackets de RankGap toleran drift ±meses

### Stats computadas por jugador
- `surface_stats`: n/wins/losses/win_rate por superficie (dura/arcilla/hierba/unknown)
- `tier_stats`: por tier (grand_slam/atp1000/atp500/challenger/itf/unknown)
- `ranking_gap_stats`: 5 brackets (dominant/favored/even/underdog_slight/underdog_big)
- `prs_stats`: three_set / two_set / underdog / favorite win_rates (Dim 4 proxy)

### Índice (runtime)
- `player_db_index.json`: solo win_rates por superficie/tier + PRS, sin rows crudas
- Excluye superficies/tiers con n < 3 (demasiado ruido)

---

## §4. Diseño D90-01 (kambi_disponible)

### Principio
- Cumple C-3 de Nodo-90: campo observacional, NUNCA filtra en edge_calculator ni shadow_book
- Endpoint verificado: `https://us.offering-api.kambicdn.com/offering/v2018/betplay/listView/tennis.json` (en producción desde Nodo-XX, usado en `combo_confianza_builder.py:1407`)

### Flujo
```
fetch_kambi_coverage.py  → reports/kambi_coverage_FECHA.json
                                    ↓
edge_calculator.py (PASO 3)  → resultado['kambi_disponible'] = True/False/None
                                    ↓
shadow_book.py            → serializado en pick_snapshot (solo observacional)
                                    ↓
trader_ev_tenis.py        → puede filtrar (Sprint 3)
betplay_combo_builder.py  → puede filtrar (Sprint 3)
```

### Valores
- `True`: jugador encontrado en coverage (nombre exacto o apellido > 3 chars)
- `False`: jugador NO encontrado en coverage cargada
- `None`: no hay archivo coverage en reports/ (campo sin datos)

### Cache
- `_kambi_coverage_cache` global en edge_calculator: cargado una vez por proceso (lazy)
- No hace HTTP — solo lee JSON pre-fetchado por `fetch_kambi_coverage.py`

---

## §5. Tests (55 REGLA-T53)

| Archivo | Clase | Tests | Estado |
|---|---|---|---|
| `tests/test_nodo92_sprint2.py` | `TestParseFilenameDate` | 3 | ✅ |
| | `TestFechaToIso` | 4 | ✅ |
| | `TestNormalizeSuperficie` | 5 | ✅ |
| | `TestThreeSetMatch` | 4 | ✅ |
| | `TestRankingBracket` | 4 | ✅ |
| | `TestComputePlayerStats` | 7 | ✅ |
| | `TestProcessFiles` | 6 | ✅ |
| | `TestBuildIndex` | 2 | ✅ |
| `tests/test_nodo92_d90_01.py` | `TestNormalizeName` | 4 | ✅ |
| | `TestIsPlayerAvailable` | 7 | ✅ |
| | `TestLoadCoverage` | 5 | ✅ |
| | `TestAnnotateKambi` | 4 | ✅ |

---

## §6. Estado post-Sprint 2

| Métrica | Antes (S1) | Después (S2) |
|---|---|---|
| Tests | 1846 | **1901** (+55) |
| PlayerDB | no existía | ✅ 4650 jugadores, 328260 filas |
| kambi_disponible | no existía | ✅ campo observacional en edge_report |
| Sprint 3 pendiente | — | RankGap+SVI en edge_report; ELO_DOM activación; Kambi filter en trader |

---

## §7. Precondiciones Sprint 3 (Nodo-90 §5)

- PlayerDB con ≥30 días y spot-check manual de 20 jugadores (verificar slugs reales)
- H89-02 (ELO_DOMINANCE): acumulando — activar si n≥30 y hit% sano
- Kambi filter en trader: una vez que kambi_disponible acumule observaciones reales
