"""
tests/test_nodo138_g2_evolution.py — REGLA-T53: tests invocan función real del módulo.

Cubre Nodo-138 D138-01 (G2 multi-signal gate evolution) y D138-02
(favoritos diversification fix para torneo='Desconocido').
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
_ccb = importlib.import_module('combo_confianza_builder')
_fcb = importlib.import_module('favoritos_combo_builder')

_apply_combo_gates = _ccb._apply_combo_gates
armar_combos       = _fcb.armar_combos


# ── Fixtures ────────────────────────────────────────────────────────────────

def _pick_no_h2h(**overrides):
    """Pick con n_h2h=0, apostar=True, base para probar G2."""
    base = {
        'apostar':          True,
        'n_h2h':            0,
        'confidence_flag':  'LOW',
        'n_axes_active':    1,
        'score_directo':    1,
        'edge':             0.10,
        'kelly_kl':         0.0,
        'alignment_flag':   'PARTIAL_ALIGNMENT',
        'net_alignment':    0.2,
    }
    base.update(overrides)
    return base


def _pick_favorito(partido, torneo, cuota_fav=1.60, p_modelo=0.65):
    """Pick mínimo válido para armar_combos (fuente=edge_report)."""
    return {
        'partido':         partido,
        'favorito':        partido.split(' vs ')[0],
        'cuota_favorito':  cuota_fav,
        'cuota_rival':     3.50,
        'p_modelo':        p_modelo,
        'confidence_flag': 'STRONG',
        'torneo':          torneo,
        'tournament':      torneo,
        'fuente':          'edge_report',
        'ranking_gap':     350,
        'ranking_favorito': 50,
        'ranking_rival':   400,
    }


# ── D138-01: G2 gate — nuevas reglas ───────────────────────────────────────

def test_D138_01_g2_blocks_weak_signal_no_h2h():
    """G2 bloquea cuando n_h2h=0 y señal genuinamente débil (LOW, edge bajo, kelly=0)."""
    pick = _pick_no_h2h(confidence_flag='LOW', edge=0.08, kelly_kl=0.0, n_axes_active=1)
    bloqueado, motivo = _apply_combo_gates(pick, 'Jugador A')
    assert bloqueado, 'G2 debe bloquear señal débil sin H2H'
    assert 'G2' in motivo


def test_D138_01_g2_allows_regla1_triple_convergencia():
    """Regla-1 original: STRONG + axes≥3 + score_dir≥3 → pasa sin H2H."""
    pick = _pick_no_h2h(
        confidence_flag='STRONG',
        n_axes_active=3,
        score_directo=3,
        edge=0.12,
        kelly_kl=0.02,
    )
    bloqueado, _ = _apply_combo_gates(pick, 'Jugador A')
    assert not bloqueado, 'Regla-1: triple convergencia debe pasar G2 sin H2H'


def test_D138_01_g2_allows_regla2_strong_edge20_kelly_positive():
    """D138-01 Regla-2: STRONG + edge≥20% + kelly_kl>0 + axes≥2 → pasa sin H2H."""
    pick = _pick_no_h2h(
        confidence_flag='STRONG',
        edge=0.20,
        kelly_kl=0.025,
        n_axes_active=2,
        score_directo=1,
    )
    bloqueado, _ = _apply_combo_gates(pick, 'Jugador A')
    assert not bloqueado, 'Regla-2: STRONG+edge20%+kelly>0 debe pasar G2 sin H2H'


def test_D138_01_g2_allows_regla3_edge35_kelly_positive():
    """D138-01 Regla-3: edge≥35% + kelly_kl>0 + axes≥2 → pasa sin H2H (cualquier conf)."""
    pick = _pick_no_h2h(
        confidence_flag='MODERATE',
        edge=0.35,
        kelly_kl=0.031,
        n_axes_active=2,
        score_directo=1,
    )
    bloqueado, _ = _apply_combo_gates(pick, 'Jugador A')
    assert not bloqueado, 'Regla-3: edge35%+kelly>0 debe pasar G2 independiente de conf'


def test_D138_01_g2_allows_h2h_positive():
    """Con n_h2h≥1 G2 no se activa — cualquier señal pasa."""
    pick = _pick_no_h2h(n_h2h=2, confidence_flag='LOW', edge=0.05, kelly_kl=0.0)
    # G3 podría bloquear si n_axes<2; dar axes=2 para aislar G2
    pick['n_axes_active'] = 2
    bloqueado, motivo = _apply_combo_gates(pick, 'Jugador A')
    # No debe ser bloqueado por G2 (puede pasar G3+G4 con axes=2)
    assert 'G2' not in motivo, 'Con n_h2h≥1 G2 no debe bloquear'


def test_D138_01_g2_blocks_regla2_boundary_edge_below_20pct():
    """Regla-2 límite inferior: STRONG + edge=19% → no pasa (debajo del umbral 20%)."""
    pick = _pick_no_h2h(
        confidence_flag='STRONG',
        edge=0.19,
        kelly_kl=0.025,
        n_axes_active=2,
        score_directo=1,
    )
    bloqueado, motivo = _apply_combo_gates(pick, 'Jugador A')
    assert bloqueado, 'Regla-2: edge=19%<20% debe ser bloqueado por G2'
    assert 'G2' in motivo


def test_D138_01_g2_blocks_regla3_boundary_edge_below_35pct():
    """Regla-3 límite inferior: MODERATE + edge=34% → no pasa (debajo del umbral 35%)."""
    pick = _pick_no_h2h(
        confidence_flag='MODERATE',
        edge=0.34,
        kelly_kl=0.031,
        n_axes_active=2,
        score_directo=1,
    )
    bloqueado, motivo = _apply_combo_gates(pick, 'Jugador A')
    assert bloqueado, 'Regla-3: edge=34%<35% debe ser bloqueado por G2'
    assert 'G2' in motivo


# ── D138-02: favoritos diversification fix ─────────────────────────────────

def test_D138_02_favoritos_torneo_desconocido_treated_as_unique():
    """D138-02: picks con torneo='Desconocido' son tratados como torneos únicos
    (no como el mismo torneo). Tres picks 'Desconocido' deben poder formar un combo
    si sus cuotas están en rango, sin ser bloqueados por MAX_LEGS_PER_TORNEO."""
    picks = [
        _pick_favorito('Alpha A. vs Beta B.', 'Desconocido', cuota_fav=1.52),
        _pick_favorito('Gamma C. vs Delta D.', 'Desconocido', cuota_fav=1.55),
        _pick_favorito('Epsilon E. vs Zeta F.', 'Desconocido', cuota_fav=1.58),
    ]
    # 1.52 × 1.55 × 1.58 = 3.72 — en rango [3.5, 7.0]
    combos = armar_combos(picks)
    assert len(combos) > 0, (
        'D138-02: tres picks con torneo=Desconocido deben poder combinarse '
        '(torneo desconocido ≠ mismo torneo)'
    )


def test_D138_02_favoritos_real_torneo_limits_per_torneo():
    """Con torneo real conocido, MAX_LEGS_PER_TORNEO=2 limita piernas del mismo torneo.
    Tres picks del mismo torneo real no deben formar combo de 3 piernas del mismo torneo."""
    picks = [
        _pick_favorito('P1 vs P2', 'Roland Garros', cuota_fav=1.52),
        _pick_favorito('P3 vs P4', 'Roland Garros', cuota_fav=1.55),
        _pick_favorito('P5 vs P6', 'Roland Garros', cuota_fav=1.58),
    ]
    # Un combo de 3 piernas del mismo torneo violaría MAX_LEGS_PER_TORNEO=2
    combos = armar_combos(picks)
    for combo in combos:
        torneo_legs = [leg['torneo'] for leg in combo['legs']]
        same = [t for t in torneo_legs if t == 'Roland Garros']
        assert len(same) <= _fcb.MAX_LEGS_PER_TORNEO, (
            f'D138-02: combo con {len(same)} piernas del mismo torneo viola MAX_LEGS_PER_TORNEO={_fcb.MAX_LEGS_PER_TORNEO}'
        )
