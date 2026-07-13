# Nodo-94 — Sprint 3: PlayerIntelligence Dim 1+2 + Kambi en trader

> **Wikilinks:** [[Nodo-93-Sprint2-Implementado]] | [[Nodo-90-Auditoria-Fable-Nodo89]]
> **Fecha:** 2026-07-13 | **Autor:** Sonnet 4.6 (WSL) | **Patrón:** Nodo-87
> **Baseline:** 1901 tests → **1920 tests** (19 nuevos REGLA-T53, 0 failed)
> **Estado:** SPRINT 3 COMPLETO — S3-A (PlayerIntelligence) + S3-B (kambi en trader)

---

## §1. Tabla de implementación

| ID | Cambio | Archivo:línea | Tests | Commit |
|---|---|---|---|---|
| S3-A (D90-05) | `_get_player_intelligence()` + `_load_player_db_index_once()` en edge_calculator. Campos: `pi_rank_gap_bracket`, `pi_rank_gap_win_rate`, `pi_svi_surface`, `pi_svi_n_surface`, `pi_n_total`. Sin cambio en gates/kelly. | `edge_calculator.py:632-720, 1290` | 19 tests `TestRankBracketEc`, `TestSurfaceMap`, `TestGetPlayerIntelligence`, `TestPlayerIntelligenceIntegration` | `94f654b` |
| S3-A index fix | `ranking_gap_win_rates` añadido a `build_index()` en build_player_db.py. player_db_index.json reconstruido. | `scripts/build_player_db.py:341-345` | (cubierto por test_nodo92_sprint2 existente) | `94f654b` |
| S3-B | Línea `PI: RankGap(...) | SVI= | n= | KAMBI:NO-DISPONIBLE` en `_print_individuales()` de trader. KAMBI:NO-DISPONIBLE solo cuando `kambi_disponible=False`. | `trader_ev_tenis.py:547-561` | (integración visual) | `94f654b` |

---

## §2. Validación real (2026-07-13)

162 eventos Kambi | 324 jugadores en cobertura | 12 picks analizados:

```
Gabriela Knutson    bracket=dominant     RankGap=55%  SVI=48%  n=88   kambi=True
Tatiana Prozorova   bracket=dominant     RankGap=78%  SVI=66%  n=106  kambi=True
Anastasia Gasanova  bracket=even         RankGap=75%  SVI=67%  n=112  kambi=True
Lanlana Tararudee   bracket=dominant     RankGap=72%  SVI=73%  n=94   kambi=True
Jesper De Jong      bracket=favored      RankGap=58%  SVI=62%  n=126  kambi=False
Alex Martinez       bracket=favored      RankGap=65%  SVI=65%  n=102  kambi=True
```

---

## §3. Diseño PlayerIntelligence

### Dim 1 — RankGap
- `rank_diff = ranking_favorito - ranking_rival` (número menor = mejor)
- 5 brackets: dominant(<-50) / favored(-50 a -10) / even(-10 a +10) / underdog_slight(+10 a +50) / underdog_big(>+50)
- `pi_rank_gap_win_rate`: win_rate histórico del jugador en ese bracket (de PlayerDB)
- `pi_rank_gap_bracket`: nombre del bracket

### Dim 2 — SVI (Surface Victory Index)
- `superficie` en edge_calculator: 'hard'/'clay'/'grass' → mapea a 'dura'/'arcilla'/'hierba' en PlayerDB
- `pi_svi_surface`: win_rate histórico del favorito_predicho en esa superficie
- `pi_svi_n_surface`: None (solo en player_db.json completo, no en índice)

### Campos observacionales (NO cambian gates)
```python
resultado['pi_rank_gap_bracket']  = str | None
resultado['pi_rank_gap_win_rate'] = float | None
resultado['pi_svi_surface']       = float | None
resultado['pi_svi_n_surface']     = None  # futuro
resultado['pi_n_total']           = int | None
```

### S3-B — Trader output
Línea extra en `_print_individuales()`:
```
PI: RankGap(dominant)=55% | SVI=48% | n=88
PI: RankGap(favored)=65% | SVI=65% | n=102 | KAMBI:NO-DISPONIBLE
```
`KAMBI:NO-DISPONIBLE` solo cuando `kambi_disponible=False` (jugador confirmado ausente de Betplay).

---

## §4. Estado post-Sprint 3

| Métrica | Antes (S2) | Después (S3) |
|---|---|---|
| Tests | 1901 | **1920** (+19) |
| PlayerIntelligence | — | ✅ Dim 1 (RankGap) + Dim 2 (SVI) en edge_report |
| Kambi en trader | solo campo | ✅ aviso visual KAMBI:NO-DISPONIBLE |
| ELO_DOM activación | n=0/30 | sin cambio — seguir acumulando |
| Sprint 4 pendiente | — | PatternRecognition, MQI/PRS/CFS, activar PI en gates si stats sanas |

---

## §5. Precondiciones Sprint 4 (Nodo-90 §5)

- n settled suficiente por instrumento (~150 picks settled en shadow book)
- H89-02 (ELO_DOMINANCE): n≥30 para evaluar activación
- Spot-check manual: 20 jugadores de player_db_index.json verificados contra historial real
- PatternRecognition solo lee picks settled (REPORTE_SOLO, D90-09)
