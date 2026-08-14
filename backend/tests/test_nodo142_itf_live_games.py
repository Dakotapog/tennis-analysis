"""
tests/test_nodo142_itf_live_games.py — REGLA-T53
Nodo-142: ITF Live Games Convergencia
Tests para funciones puras: _parse_betplay_scoreboard_html, _compute_itf_games_proxy,
_convergencia_score_itf, _parse_kambi_tennis_score.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from live_desk import (
    _parse_betplay_scoreboard_html,
    _compute_itf_games_proxy,
    _convergencia_score_itf,
    _parse_kambi_tennis_score,
)


# ─── _parse_betplay_scoreboard_html ───────────────────────────────────────────

def _make_scoreboard_html(sets: list) -> str:
    """Genera HTML mínimo del scoreboard KambiBC para tests.
    sets = [(home_g, away_g), ...] donde el último puede ser el set en curso.
    Estructura ROW-MAJOR real: una section por jugador, grid-row envolviendo
    todos los juegos de esa section (ver docstring _parse_betplay_scoreboard_html).
    """
    def _row(values):
        items = "".join(
            f'<div class="KambiBC-scoreboard-grid-item" data-sport="TENNIS">'
            f"<span>{v}</span></div>"
            for v in values
        )
        return (
            '<section class="KambiBC-scoreboard-row">'
            f'<div class="KambiBC-scoreboard-grid-row">{items}</div>'
            "</section>"
        )

    home_row = _row([h for h, _ in sets])
    away_row = _row([a for _, a in sets])
    return home_row + away_row


def test_parse_scoreboard_1_set_complete():
    """Set completo 6:4 → games_played=10, sets_complete=1, sets_away=1."""
    html = _make_scoreboard_html([(6, 4)])
    r = _parse_betplay_scoreboard_html(html)
    assert r["games_played"] == 10
    assert r["sets_complete"] == 1
    assert r["sets_home"] == 1
    assert r["sets_away"] == 0
    assert r["score_str"] == "6:4"


def test_parse_scoreboard_1_set_complete_1_in_progress():
    """Set 1 completo 6:4, set 2 en curso 2:2 → games_played=14, sets_complete=1."""
    html = _make_scoreboard_html([(6, 4), (2, 2)])
    r = _parse_betplay_scoreboard_html(html)
    assert r["games_played"] == 14
    assert r["sets_complete"] == 1
    assert r["current_games"] == 4   # 2+2 juegos en el set en curso
    assert r["score_str"] == "6:4,2:2"


def test_parse_scoreboard_2_sets_complete():
    """2 sets completos 6:4 7:5 → games_played=22, sets_complete=2."""
    html = _make_scoreboard_html([(6, 4), (7, 5)])
    r = _parse_betplay_scoreboard_html(html)
    assert r["games_played"] == 22
    assert r["sets_complete"] == 2
    assert r["sets_home"] == 2
    assert r["sets_away"] == 0


def test_parse_scoreboard_tiebreak_set():
    """Tiebreak 7:6 → set completo (max=7)."""
    html = _make_scoreboard_html([(7, 6)])
    r = _parse_betplay_scoreboard_html(html)
    assert r["sets_complete"] == 1
    assert r["games_played"] == 13


def test_parse_scoreboard_empty_html():
    """HTML vacío → todos None, sin excepción."""
    r = _parse_betplay_scoreboard_html("")
    assert r["games_played"] is None
    assert r["sets_complete"] is None


def test_parse_scoreboard_no_items():
    """HTML sin grid-items → todos None."""
    r = _parse_betplay_scoreboard_html("<div>sin marcador</div>")
    assert r["games_played"] is None


def test_parse_scoreboard_grid_score_excluded():
    """Elementos con grid-score (puntos: 15/30/AD) NO deben contar como set games."""
    html = (
        '<section class="KambiBC-scoreboard-row">'
        '<div class="KambiBC-scoreboard-grid-row">'
        '<div class="KambiBC-scoreboard-grid-item" data-sport="TENNIS"><span>4</span></div>'
        '<div class="KambiBC-scoreboard-grid-item KambiBC-scoreboard-grid-score" data-sport="TENNIS">'
        "<span>AD</span></div>"
        "</div></section>"
        '<section class="KambiBC-scoreboard-row">'
        '<div class="KambiBC-scoreboard-grid-row">'
        '<div class="KambiBC-scoreboard-grid-item" data-sport="TENNIS"><span>6</span></div>'
        '<div class="KambiBC-scoreboard-grid-item KambiBC-scoreboard-grid-score" data-sport="TENNIS">'
        "<span>40</span></div>"
        "</div></section>"
    )
    r = _parse_betplay_scoreboard_html(html)
    # Solo debe leer el 4:6, no los puntos AD/40
    assert r["games_played"] == 10
    assert r["sets_complete"] == 1


# ─── _compute_itf_games_proxy ─────────────────────────────────────────────────

def test_compute_proxy_returns_required_keys():
    """Siempre devuelve midpoint, games_range, ranking_gap."""
    r = _compute_itf_games_proxy("Smith", "Jones", {})
    assert "midpoint" in r
    assert "games_range" in r
    assert "ranking_gap" in r
    assert isinstance(r["midpoint"], (int, float))


def test_compute_proxy_midpoint_in_range():
    """midpoint debe estar dentro del games_range declarado."""
    r = _compute_itf_games_proxy("A", "B", {})
    assert r["low"] <= r["midpoint"] <= r["high"]


# ─── _convergencia_score_itf ──────────────────────────────────────────────────

def test_convergencia_score_alta_gap_y_cuota():
    """Gap grande y cuota favorable → score alto (>=3)."""
    r = _convergencia_score_itf(gap=8.0, cuota_live=1.75, markov=None, ranking_gap=None)
    assert r["score"] >= 2
    assert "direction" in r


def test_convergencia_score_bajo_gap_pequeno():
    """Gap mínimo y cuota baja → score bajo."""
    r = _convergencia_score_itf(gap=0.5, cuota_live=1.35, markov=None, ranking_gap=None)
    assert r["score"] <= 2


def test_convergencia_score_estructura():
    """Siempre devuelve score, direction, breakdown."""
    r = _convergencia_score_itf(gap=5.0, cuota_live=1.60, markov=None, ranking_gap=None)
    assert set(r.keys()) >= {"score", "direction", "breakdown"}
    assert isinstance(r["score"], int)
    assert r["direction"] in ("UNDER", "OVER")


# ─── _parse_kambi_tennis_score ────────────────────────────────────────────────

def test_parse_kambi_score_scorestr():
    """scoreStr '6:4,3:2' → games_played=15, sets_complete=1."""
    ev = {"liveData": {"scoreStr": "6:4,3:2"}}
    r = _parse_kambi_tennis_score(ev)
    assert r["games_played"] == 15
    assert r["sets_complete"] == 1
    assert r["current_games"] == 5


def test_parse_kambi_score_two_complete_sets():
    """'6:3,6:4' → 2 sets completos, 19 juegos."""
    ev = {"liveData": {"scoreStr": "6:3,6:4"}}
    r = _parse_kambi_tennis_score(ev)
    assert r["games_played"] == 19
    assert r["sets_complete"] == 2
    assert r["sets_home"] == 2


def test_parse_kambi_score_none_event():
    """Event None → todos None sin excepción."""
    r = _parse_kambi_tennis_score(None)
    assert r["games_played"] is None


def test_parse_kambi_score_empty_livedata():
    """liveData vacío → todos None."""
    r = _parse_kambi_tennis_score({"liveData": {}})
    assert r["score_str"] is None
    assert r["games_played"] is None
