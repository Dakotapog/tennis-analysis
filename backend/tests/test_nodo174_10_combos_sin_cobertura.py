"""REGLA-T53 — Nodo-174 D174-10: tests para las 3 estrategias sin cobertura.

build_system_combos (Nodo-156-B), build_safe_combos (Nodo-25), build_was_combos
(Nodo-44) nunca tuvieron tests pese a estar en producción. Estos tests invocan
las funciones reales del módulo -- nunca hardcodean la fórmula de cuota_combo/
p_todas/score, solo verifican los 5 contratos mínimos exigidos por el spec:
REGLA-HF-1, REGLA-BAT-1, pool insuficiente -> [] sin crash, filtro kambi_disponible,
no reutilización de outcome_id entre piernas.

Hallazgo aplicado en esta sesión: build_safe_combos() no tenía guard explícito
REGLA-HF-1 (a diferencia de build_system_combos/build_was_combos que sí lo
tienen vía cuota>=1.50/2.0) -- corregido con 2 líneas antes de escribir estos
tests, mismo patrón que build_system_combos.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import betplay_combo_builder as bcb
from edge_calculator import GATE_VERSION


def _fake_find_outcome_factory(names_to_ids: dict):
    """fetch_kambi_outcomes()/find_outcome() mock: cada jugador resuelve a un
    outcome_id único (nunca reutilizado) y misma cuota que en el pool."""
    def _fake_find_outcome(jugador, cuota, outcomes_map, started_map, **kwargs):
        if jugador in names_to_ids:
            return {"outcome_id": names_to_ids[jugador], "odds": cuota}, "ok"
        return None, "no encontrado"
    return _fake_find_outcome


def _assert_no_reused_outcome_ids(combos):
    """Dentro de cada combo, ningún outcome_id debe repetirse entre piernas."""
    for combo in combos:
        ids = combo["outcome_ids"]
        assert len(ids) == len(set(ids)), f"outcome_id repetido dentro del combo: {ids}"


def _assert_coupon_format(combos):
    """REGLA-BAT-1: comma-joined, ||replace, sin |ML/."""
    for combo in combos:
        url = combo["url"]
        assert "|ML" not in url
        assert url.endswith("||replace")
        assert ",".join(combo["outcome_ids"]) in url


# ══════════════════════════════════════════════════════════════════════════
# build_system_combos (Nodo-156-B)
# ══════════════════════════════════════════════════════════════════════════

def _sistema_pick(name, cuota, p_modelo, n_h2h=5, kambi_disponible=True, tier="atp"):
    return {
        "favorito_predicho": name,
        "cuota_favorito": cuota,
        "p_modelo": p_modelo,
        "n_h2h": n_h2h,
        "kambi_disponible": kambi_disponible,
        "tier": tier,
    }


def _write_edge_report(tmp_path, apostar=None, watchlist=None):
    edge_data = {
        "metadata": {"gate_version": GATE_VERSION},
        "apostar": apostar or [],
        "watchlist": watchlist or [],
    }
    p = tmp_path / "edge_report_test.json"
    p.write_text(json.dumps(edge_data), encoding="utf-8")
    return str(p)


def _run_system(tmp_path, picks, **kwargs):
    edge_path = _write_edge_report(tmp_path, watchlist=picks)
    names_to_ids = {p["favorito_predicho"]: f"S{i}" for i, p in enumerate(picks)}
    with patch.object(bcb, "_find_latest_edge_report", return_value=edge_path), \
         patch.object(bcb, "fetch_kambi_outcomes", return_value=({"dummy": {}}, {})), \
         patch.object(bcb, "find_outcome", side_effect=_fake_find_outcome_factory(names_to_ids)):
        return bcb.build_system_combos(**kwargs)


def test_sistema_pool_insuficiente_retorna_vacio_sin_crash(tmp_path):
    """4 picks < n_piernas=6 (default) -> [] sin excepción (fail-loud, patrón D172-01)."""
    picks = [_sistema_pick(f"P{i}", 2.0, 0.60) for i in range(4)]
    combos, meta = _run_system(tmp_path, picks)
    assert combos == []
    assert meta == {}


def test_sistema_regla_hf1_excluye_cuota_bajo_1_50(tmp_path):
    picks = [_sistema_pick(f"P{i}", 2.0, 0.60) for i in range(6)]
    picks.append(_sistema_pick("Barato X.", 1.20, 0.90))  # REGLA-HF-1
    combos, meta = _run_system(tmp_path, picks, n_piernas=6)
    assert len(combos) == 6
    for combo in combos:
        names = [leg["jugador"] for leg in combo["legs"]]
        assert "Barato X." not in names


def test_sistema_filtra_kambi_disponible_false(tmp_path):
    picks = [_sistema_pick(f"P{i}", 2.0, 0.60) for i in range(6)]
    picks.append(_sistema_pick("SinKambi Y.", 3.0, 0.70, kambi_disponible=False))
    combos, meta = _run_system(tmp_path, picks, n_piernas=6)
    assert len(combos) == 6
    for combo in combos:
        names = [leg["jugador"] for leg in combo["legs"]]
        assert "SinKambi Y." not in names


def test_sistema_no_reutiliza_outcome_id_entre_piernas(tmp_path):
    picks = [_sistema_pick(f"P{i}", 2.0 + i * 0.1, 0.55 + i * 0.01) for i in range(7)]
    combos, meta = _run_system(tmp_path, picks, n_piernas=6)
    assert len(combos) == 6
    _assert_no_reused_outcome_ids(combos)


def test_sistema_coupon_format_regla_bat_1(tmp_path):
    picks = [_sistema_pick(f"P{i}", 2.0, 0.60) for i in range(6)]
    combos, meta = _run_system(tmp_path, picks, n_piernas=6)
    assert len(combos) == 6
    _assert_coupon_format(combos)


def test_sistema_leave_one_out_propiedad_matematica(tmp_path):
    """N piernas -> N combos de (N-1), cada uno excluye una pierna distinta."""
    picks = [_sistema_pick(f"P{i}", 2.0, 0.60) for i in range(6)]
    combos, meta = _run_system(tmp_path, picks, n_piernas=6)
    assert len(combos) == 6
    excluidas = {c["excluye"] for c in combos}
    assert excluidas == {p["favorito_predicho"] for p in picks}
    for c in combos:
        assert c["piernas"] == 5


# ══════════════════════════════════════════════════════════════════════════
# build_was_combos (Nodo-44)
# ══════════════════════════════════════════════════════════════════════════

def _was_pick(name, cuota, edge_pct="15%", n_h2h=3, kambi_disponible=True,
              markov_rival="COLD", conf_rival=0.70, tier="atp", torneo="X"):
    return {
        "favorito_predicho": name,
        "cuota_favorito": cuota,
        "edge_pct": edge_pct,
        "p_modelo": 0.60,
        "p_blend": 0.60,
        "n_h2h": n_h2h,
        "tier": tier,
        "torneo": torneo,
        "superficie": "hard",
        "kambi_disponible": kambi_disponible,
        "markov_favorito": None,
        "markov_rival": markov_rival,
        "markov_conf_fav": 0,
        "markov_conf_rival": conf_rival,
        "markov_wr_rec_fav": 0.5,
        "markov_wr_rec_rival": 0.5,
    }


def _run_was(tmp_path, picks, **kwargs):
    edge_path = _write_edge_report(tmp_path, watchlist=picks)
    names_to_ids = {p["favorito_predicho"]: f"W{i}" for i, p in enumerate(picks)}
    with patch.object(bcb, "fetch_kambi_outcomes", return_value=({"dummy": {}}, {})), \
         patch.object(bcb, "find_outcome", side_effect=_fake_find_outcome_factory(names_to_ids)):
        return bcb.build_was_combos(edge_file=edge_path, **kwargs)


def test_was_pool_insuficiente_watchlist_vacia_retorna_vacio_sin_crash(tmp_path):
    combos, meta = _run_was(tmp_path, [])
    assert combos == []
    assert meta == {}


def test_was_regla_hf1_cuota_bajo_2_0_excluida(tmp_path):
    """Gate WAS exige cuota>=2.0 (más estricto que REGLA-HF-1>=1.50) -- ningún
    leg final puede tener cuota<1.50, y de hecho ninguno puede tener cuota<2.0."""
    picks = [
        _was_pick("A.", 2.5, edge_pct="20%"),
        _was_pick("B.", 2.2, edge_pct="18%"),
        _was_pick("Barato C.", 1.30, edge_pct="30%"),  # cuota<2.0 -- excluido
    ]
    combos, meta = _run_was(tmp_path, picks)
    assert len(combos) >= 1
    for combo in combos:
        for leg in combo["legs"]:
            assert leg["cuota_kambi"] >= 1.50
            assert leg["jugador"] != "Barato C."


def test_was_filtra_kambi_disponible_false(tmp_path):
    picks = [
        _was_pick("A.", 2.5, edge_pct="20%"),
        _was_pick("B.", 2.2, edge_pct="18%"),
        _was_pick("SinKambi D.", 3.0, edge_pct="25%", kambi_disponible=False),
    ]
    combos, meta = _run_was(tmp_path, picks)
    assert len(combos) >= 1
    for combo in combos:
        names = [leg["jugador"] for leg in combo["legs"]]
        assert "SinKambi D." not in names


def test_was_no_reutiliza_outcome_id_entre_piernas(tmp_path):
    picks = [
        _was_pick("A.", 2.5, edge_pct="20%"),
        _was_pick("B.", 2.2, edge_pct="18%"),
        _was_pick("C.", 2.8, edge_pct="22%"),
    ]
    combos, meta = _run_was(tmp_path, picks)
    assert len(combos) >= 1
    _assert_no_reused_outcome_ids(combos)


def test_was_coupon_format_regla_bat_1(tmp_path):
    picks = [
        _was_pick("A.", 2.5, edge_pct="20%"),
        _was_pick("B.", 2.2, edge_pct="18%"),
    ]
    combos, meta = _run_was(tmp_path, picks)
    assert len(combos) == 1
    _assert_coupon_format(combos)


def test_was_sin_señal_markov_no_califica(tmp_path):
    """Edge alto + cuota alta pero SIN señal Markov explícita -- no es WAS
    (coin-flip puro, T55-04/05)."""
    picks = [
        _was_pick("A.", 2.5, edge_pct="20%", markov_rival=None, conf_rival=0),
        _was_pick("B.", 2.2, edge_pct="18%", markov_rival=None, conf_rival=0),
    ]
    combos, meta = _run_was(tmp_path, picks)
    assert combos == []


# ══════════════════════════════════════════════════════════════════════════
# build_safe_combos (Nodo-25)
# ══════════════════════════════════════════════════════════════════════════

def _trader_individual(name, cuota, p_blend=0.55, p_modelo=0.55):
    return {"favorito": name, "cuota": cuota, "p_blend": p_blend, "p_modelo": p_modelo,
            "edge_pct": "10%"}


def _write_trader_plan(reports_dir, idx, individuales, torneo_tipo="atp", superficie="hard"):
    plan = {
        "metadata": {"parametros": {"torneo_tipo": torneo_tipo, "superficie": superficie}},
        "individuales": individuales,
    }
    p = reports_dir / f"trader_plan_test{idx}.json"
    p.write_text(json.dumps(plan), encoding="utf-8")
    return p


def _run_safe(tmp_path, monkeypatch, individuales_por_plan, **kwargs):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    for i, individuales in enumerate(individuales_por_plan):
        _write_trader_plan(reports_dir, i, individuales, torneo_tipo=f"tier{i}")
    monkeypatch.chdir(tmp_path)

    all_names = [p["favorito"] for plan in individuales_por_plan for p in plan]
    names_to_ids = {name: f"F{i}" for i, name in enumerate(all_names)}

    # edge_report con torneo distinto por jugador -- sin esto todos caen en el
    # mismo fallback f"{plan_tier}_{plan_sup}" (mismo trader_plan) y Guard 2
    # (torneos distintos) excluye TODOS los pares.
    edge_path = _write_edge_report(
        tmp_path,
        watchlist=[
            {"favorito_predicho": name, "torneo": f"Torneo{i}", "tier": "atp", "n_h2h": 3,
             "p_blend": 0.55, "p_modelo": 0.55}
            for i, name in enumerate(all_names)
        ],
    )

    with patch.object(bcb, "_find_latest_edge_report", return_value=edge_path), \
         patch.object(bcb, "fetch_kambi_outcomes", return_value=({"dummy": {}}, {})), \
         patch.object(bcb, "find_outcome", side_effect=_fake_find_outcome_factory(names_to_ids)):
        return bcb.build_safe_combos(**kwargs)


def test_safe_pool_insuficiente_sin_trader_plans_retorna_vacio_sin_crash(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    combos, meta = bcb.build_safe_combos()
    assert combos == []
    assert meta == {}


def test_safe_regla_hf1_excluye_cuota_bajo_1_50(tmp_path, monkeypatch):
    """Hallazgo D174-10: build_safe_combos no filtraba cuota<1.50 -- corregido
    en esta sesión con el mismo patrón que build_system_combos."""
    individuales = [
        [_trader_individual("A.", 2.5), _trader_individual("Barato B.", 1.20)],
    ]
    combos, meta = _run_safe(tmp_path, monkeypatch, individuales, min_p_both=0.0, max_cuota=999)
    for combo in combos:
        names = [leg["jugador"] for leg in combo["legs"]]
        assert "Barato B." not in names
        for leg in combo["legs"]:
            assert leg["cuota"] >= 1.50


def test_safe_filtra_kambi_disponible_false(tmp_path, monkeypatch):
    edge_path = tmp_path / "edge_report_x.json"
    edge_data = {
        "metadata": {"gate_version": GATE_VERSION},
        "apostar": [],
        "watchlist": [
            {"favorito_predicho": "SinKambi C.", "kambi_disponible": False,
             "torneo": "T1", "tier": "atp", "n_h2h": 3},
        ],
    }
    edge_path.write_text(json.dumps(edge_data), encoding="utf-8")

    individuales = [
        [_trader_individual("A.", 2.5), _trader_individual("B.", 2.2),
         _trader_individual("SinKambi C.", 3.0)],
    ]
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_trader_plan(reports_dir, 0, individuales[0])
    monkeypatch.chdir(tmp_path)
    names_to_ids = {"A.": "F0", "B.": "F1", "SinKambi C.": "F2"}
    with patch.object(bcb, "_find_latest_edge_report", return_value=str(edge_path)), \
         patch.object(bcb, "fetch_kambi_outcomes", return_value=({"dummy": {}}, {})), \
         patch.object(bcb, "find_outcome", side_effect=_fake_find_outcome_factory(names_to_ids)):
        combos, meta = bcb.build_safe_combos(min_p_both=0.0, max_cuota=999)

    for combo in combos:
        names = [leg["jugador"] for leg in combo["legs"]]
        assert "SinKambi C." not in names


def test_safe_no_reutiliza_outcome_id_entre_piernas(tmp_path, monkeypatch):
    individuales = [
        [_trader_individual("A.", 2.5), _trader_individual("B.", 2.2),
         _trader_individual("C.", 2.8)],
    ]
    combos, meta = _run_safe(tmp_path, monkeypatch, individuales, min_p_both=0.0, max_cuota=999)
    assert len(combos) >= 1
    _assert_no_reused_outcome_ids(combos)


def test_safe_coupon_format_regla_bat_1(tmp_path, monkeypatch):
    individuales = [
        [_trader_individual("A.", 2.5), _trader_individual("B.", 2.2)],
    ]
    combos, meta = _run_safe(tmp_path, monkeypatch, individuales, min_p_both=0.0, max_cuota=999)
    assert len(combos) == 1
    _assert_coupon_format(combos)


def test_safe_guard_torneos_distintos(tmp_path, monkeypatch):
    """Guard 2 (Nodo-25): mismo torneo entre 2 picks -- no arma combo entre ellos."""
    edge_path = tmp_path / "edge_report_same_torneo.json"
    edge_data = {
        "metadata": {"gate_version": GATE_VERSION},
        "apostar": [],
        "watchlist": [
            {"favorito_predicho": "A.", "torneo": "MismoTorneo", "tier": "atp", "n_h2h": 3},
            {"favorito_predicho": "B.", "torneo": "MismoTorneo", "tier": "atp", "n_h2h": 3},
        ],
    }
    edge_path.write_text(json.dumps(edge_data), encoding="utf-8")
    individuales = [[_trader_individual("A.", 2.5), _trader_individual("B.", 2.2)]]
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_trader_plan(reports_dir, 0, individuales[0])
    monkeypatch.chdir(tmp_path)
    names_to_ids = {"A.": "F0", "B.": "F1"}
    with patch.object(bcb, "_find_latest_edge_report", return_value=str(edge_path)), \
         patch.object(bcb, "fetch_kambi_outcomes", return_value=({"dummy": {}}, {})), \
         patch.object(bcb, "find_outcome", side_effect=_fake_find_outcome_factory(names_to_ids)):
        combos, meta = bcb.build_safe_combos(min_p_both=0.0, max_cuota=999)

    assert combos == []
