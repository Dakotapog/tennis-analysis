"""
Tests Nodo-166 — convergencia_score por pierna en alta_signals (D166-01).

Replica en ATP/WTA/Challenger/ATP1000/ATP500 lo que ITF ya tenía: en vez de
disparar el combo D133 por un conteo agregado (en_vivo_count>=2, sin gate de
calidad por señal), cada pierna EN_VIVO calcula su propio convergencia_score
(D142-02: gap/cuota/markov/ranking) + el bonus de certeza D147 (D165-01), y
solo entran al combo las que superan >=3 individualmente — mismo umbral y
mismo patrón que itf_live_signals. Reusa _convergencia_score_itf y
_convergencia_certeza_bonus sin duplicar lógica ("replicar no reinventar",
Nodo-164). Único componente nuevo es _get_ranking_gap_er(), que sustituye el
proxy de rango de juegos que usa ITF (sin mercado pre-partido) por los
campos ranking_favorito/ranking_rival ya serializados en edge_report por
edge_calculator.py (alta_signals SÍ tiene mercado pre-partido real).

REGLA-T53: invoca las funciones reales del módulo — nunca hardcodea la
fórmula del score.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live_desk import _get_ranking_gap_er, _convergencia_score_itf


def test_166_01_ranking_gap_er_matchea_por_apellido_home():
    """Pick en er_picks con 'partido' conteniendo el apellido home →
    devuelve abs(ranking_favorito - ranking_rival)."""
    er_picks = [{"partido": "Alcaraz C. vs Rival X.", "ranking_favorito": 2, "ranking_rival": 450}]
    gap = _get_ranking_gap_er("Alcaraz C.", "Rival X.", er_picks)
    assert gap == 448


def test_166_02_ranking_gap_er_matchea_por_apellido_away():
    """El apellido del jugador 'away' también matchea si aparece en el
    string 'partido' del pick — no depende de home/away del pick mismo."""
    er_picks = [{"partido": "Fulano vs Sinner J.", "ranking_favorito": 5, "ranking_rival": 300}]
    gap = _get_ranking_gap_er("Alguien", "Sinner J.", er_picks)
    assert gap == 295


def test_166_03_sin_match_devuelve_none():
    """Ningún pick de er_picks menciona los apellidos → None (sin proxy,
    a diferencia de ITF que reconstruye un rango games-based)."""
    er_picks = [{"partido": "Otro A. vs Otro B.", "ranking_favorito": 10, "ranking_rival": 50}]
    gap = _get_ranking_gap_er("Alcaraz C.", "Rival X.", er_picks)
    assert gap is None


def test_166_04_match_sin_ranking_fields_devuelve_none():
    """Pick matchea por nombre pero no trae ranking_favorito/ranking_rival
    (p.ej. pick incompleto) → None, no crashea ni inventa un valor."""
    er_picks = [{"partido": "Alcaraz C. vs Rival X."}]
    gap = _get_ranking_gap_er("Alcaraz C.", "Rival X.", er_picks)
    assert gap is None


def test_166_05_er_picks_vacio_devuelve_none():
    """Lista vacía de er_picks (sin edge_report cargado) → None, no crashea."""
    assert _get_ranking_gap_er("Alcaraz C.", "Rival X.", []) is None


def test_166_06_ranking_gap_alimenta_convergencia_score_igual_que_itf():
    """Integración con la función ya existente: ranking_gap>300 en
    dirección UNDER suma +1 al score — mismo comportamiento documentado
    para itf_live_signals (D142-02), ahora alcanzable también desde
    alta_signals vía _get_ranking_gap_er en vez del proxy games-range."""
    er_picks = [{"partido": "Favorito F. vs Outsider O.", "ranking_favorito": 8, "ranking_rival": 550}]
    rank_gap = _get_ranking_gap_er("Favorito F.", "Outsider O.", er_picks)
    assert rank_gap == 542
    r = _convergencia_score_itf(gap=2.5, cuota_live=1.80, markov=None, ranking_gap=rank_gap)
    # gap MOD (+1) + ranking_gap>300 en UNDER (+1) = 2
    assert r["score"] == 2
    assert "rank_gap=542(mismatch)" in r["breakdown"]
