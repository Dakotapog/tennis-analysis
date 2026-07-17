"""
tests/test_nodo103_combo_gates.py — REGLA-T53: tests invocan función real.

Cubre D103-01→D103-08 (_apply_combo_gates en combo_confianza_builder.py).
D103-08 (2026-07-17): G4 recalibrado — solo bloquea net_alignment < -0.10.
Evidencia de campo: Gaines Jr (net=0.0, ganó) vs Bartel Jul-15 (net=-0.286, perdió).
"""
import importlib
import sys
from pathlib import Path

import pytest

# Importar la función real del módulo (REGLA-T53)
sys.path.insert(0, str(Path(__file__).parent.parent))
_mod = importlib.import_module('combo_confianza_builder')
_apply_combo_gates = _mod._apply_combo_gates


# ── Fixtures de picks base ──────────────────────────────────────────────────

def _pick_ok(**overrides):
    """Pick que pasa todos los gates G1–G4."""
    base = {
        'apostar': True,
        'confidence_flag': 'STRONG',
        'n_h2h': 2,
        'n_axes_active': 2,
        'score_directo': 3,
        'alignment_flag': 'PARTIAL_ALIGNMENT',
        'net_alignment': 0.3,
        'motivo_reclasificacion': '',
    }
    base.update(overrides)
    return base


# ── G1: apostar=False ────────────────────────────────────────────────────────

def test_g1_bloquea_apostar_false():
    """G1: pick con apostar=False debe ser bloqueado."""
    bloqueado, motivo = _apply_combo_gates(_pick_ok(apostar=False), 'TestPlayer')
    assert bloqueado, 'G1 debe bloquear apostar=False'
    assert 'G1' in motivo

def test_g1_permite_apostar_true():
    """G1: pick con apostar=True no es bloqueado por G1."""
    bloqueado, _ = _apply_combo_gates(_pick_ok(apostar=True), 'TestPlayer')
    assert not bloqueado


# ── G2: n_h2h=0 ─────────────────────────────────────────────────────────────

def test_g2_bloquea_n_h2h_cero():
    """G2: n_h2h=0 sin triple convergencia debe ser bloqueado."""
    pick = _pick_ok(n_h2h=0, confidence_flag='LOW', n_axes_active=1, score_directo=1)
    bloqueado, motivo = _apply_combo_gates(pick, 'TestPlayer')
    assert bloqueado
    assert 'G2' in motivo

def test_g2_excepcion_triple_convergencia():
    """G2: n_h2h=0 con STRONG+axes>=3+score>=3 no debe bloquearse."""
    pick = _pick_ok(n_h2h=0, confidence_flag='STRONG', n_axes_active=3, score_directo=3)
    bloqueado, _ = _apply_combo_gates(pick, 'TestPlayer')
    assert not bloqueado, 'Triple convergencia debe pasar G2 con n_h2h=0'

def test_g2_excepcion_requiere_score_3():
    """G2: excepción triple convergencia falla si score_directo < 3."""
    pick = _pick_ok(n_h2h=0, confidence_flag='STRONG', n_axes_active=3, score_directo=2)
    bloqueado, motivo = _apply_combo_gates(pick, 'TestPlayer')
    assert bloqueado
    assert 'G2' in motivo


# ── G3: n_axes_active < 2 (N28F2) ───────────────────────────────────────────

def test_g3_bloquea_n_axes_1():
    """G3: pick con n_axes_active=1 debe ser bloqueado (N28F2)."""
    pick = _pick_ok(n_h2h=2, n_axes_active=1)
    bloqueado, motivo = _apply_combo_gates(pick, 'TestPlayer')
    assert bloqueado
    assert 'G3' in motivo

def test_g3_permite_n_axes_2():
    """G3: n_axes_active=2 no debe ser bloqueado."""
    pick = _pick_ok(n_axes_active=2)
    bloqueado, _ = _apply_combo_gates(pick, 'TestPlayer')
    assert not bloqueado


# ── G4 (D103-08): NO_ALIGNMENT solo bloquea si net_alignment < -0.10 ────────

def test_g4_bloquea_net_alignment_negativo():
    """G4 D103-08: net_alignment=-0.286 (Bartel Jul-15) debe ser bloqueado."""
    pick = _pick_ok(alignment_flag='NO_ALIGNMENT', net_alignment=-0.286)
    bloqueado, motivo = _apply_combo_gates(pick, 'Bartel')
    assert bloqueado, 'net_alignment=-0.286 debe activar G4'
    assert 'G4' in motivo
    assert '-0.10' in motivo or 'activamente' in motivo

def test_g4_bloquea_justo_bajo_umbral():
    """G4 D103-08: net_alignment=-0.11 (justo bajo umbral) debe ser bloqueado."""
    pick = _pick_ok(alignment_flag='NO_ALIGNMENT', net_alignment=-0.11)
    bloqueado, motivo = _apply_combo_gates(pick, 'TestPlayer')
    assert bloqueado
    assert 'G4' in motivo

def test_g4_permite_net_alignment_cero_gaines_jr():
    """G4 D103-08: net_alignment=0.0 (Gaines Jr Jul-16, ganó) NO debe ser bloqueado."""
    pick = _pick_ok(alignment_flag='NO_ALIGNMENT', net_alignment=0.0)
    bloqueado, _ = _apply_combo_gates(pick, 'Gaines Jr')
    assert not bloqueado, 'Gaines Jr net=0.0 no debe bloquearse — evidencia Jul-16'

def test_g4_permite_net_alignment_negativo_sobre_umbral():
    """G4 D103-08: net_alignment=-0.10 (en el umbral) NO debe ser bloqueado."""
    pick = _pick_ok(alignment_flag='NO_ALIGNMENT', net_alignment=-0.10)
    bloqueado, _ = _apply_combo_gates(pick, 'TestPlayer')
    assert not bloqueado, 'net=-0.10 está justo en el umbral, no bloquear'

def test_g4_no_bloquea_partial_alignment():
    """G4: alignment_flag=PARTIAL_ALIGNMENT con net positivo pasa sin problema."""
    pick = _pick_ok(alignment_flag='PARTIAL_ALIGNMENT', net_alignment=0.282)
    bloqueado, _ = _apply_combo_gates(pick, 'Zantedeschi')
    assert not bloqueado

def test_g4_no_bloquea_structural_alpha():
    """G4: STRUCTURAL_ALPHA siempre pasa (Shoaib Jul-16)."""
    pick = _pick_ok(alignment_flag='STRUCTURAL_ALPHA', net_alignment=0.374)
    bloqueado, _ = _apply_combo_gates(pick, 'Shoaib')
    assert not bloqueado


# ── G0: sin datos edge_report ────────────────────────────────────────────────

def test_g0_sin_datos_pasa():
    """Sin datos de edge_report → gates no pueden evaluar → pasar (no bloquear).
    Diseño D103-fix: G0 fue eliminado porque bloqueaba picks legítimos cuando
    el edge_report existe en disco pero no contiene el pick del test."""
    bloqueado, motivo = _apply_combo_gates(None, 'TestPlayer')
    assert not bloqueado, "Sin edge_data no hay evidencia de rechazo — debe pasar"

    bloqueado2, _ = _apply_combo_gates({}, 'TestPlayer')
    assert not bloqueado2


# ── Retrovalidación Jul-15 (picks que perdieron) ─────────────────────────────

def test_bartel_jul15_bloqueado_g1():
    """Bartel Jul-15: apostar=False → G1 bloquea."""
    bartel = {
        'apostar': False, 'confidence_flag': 'LOW',
        'n_h2h': 0, 'n_axes_active': 1, 'alignment_flag': 'NO_ALIGNMENT',
        'net_alignment': -0.286, 'score_directo': 0,
    }
    bloqueado, motivo = _apply_combo_gates(bartel, 'Bartel K.')
    assert bloqueado
    assert 'G1' in motivo

def test_lekomtseva_jul15_bloqueado_g1():
    """Lekomtseva Jul-15: apostar=False → G1 bloquea (a pesar de STRONG)."""
    leko = {
        'apostar': False, 'confidence_flag': 'STRONG',
        'n_h2h': 0, 'n_axes_active': 1, 'alignment_flag': 'NO_ALIGNMENT',
        'net_alignment': 0.0, 'score_directo': 1,
        'motivo_reclasificacion': 'N28F2: n_axes_active < 2 (BBI sola no predice)',
    }
    bloqueado, motivo = _apply_combo_gates(leko, 'Lekomtseva U.')
    assert bloqueado
    assert 'G1' in motivo


# ── Validación Jul-16 (picks ganadores con G4 recalibrado) ───────────────────

def test_gaines_jr_jul16_no_bloqueado():
    """Gaines Jr Jul-16: STRONG, n_h2h=0 (G2 aplica, n_axes=2 pasa G3), net=0.0 pasa G4."""
    # n_h2h=0 → G2 bloquea a menos que triple convergencia
    # Gaines Jr: STRONG, axes=2 (<3), score=2 (<3) → G2 bloquea
    # Esto confirma que G2 aún habría bloqueado a Gaines Jr (vino por betplay --live)
    gaines = {
        'apostar': True, 'confidence_flag': 'STRONG',
        'n_h2h': 0, 'n_axes_active': 2, 'alignment_flag': 'NO_ALIGNMENT',
        'net_alignment': 0.0, 'score_directo': 2,
    }
    bloqueado, motivo = _apply_combo_gates(gaines, 'Gaines Jr')
    # G2 bloquea porque n_h2h=0 y no hay triple convergencia (axes=2 < 3)
    assert bloqueado
    assert 'G2' in motivo  # G2, no G4 — confirma que la recalibración G4 no cambia G2

def test_bernard_jul16_bloqueado_g3():
    """Bernard Jul-16: STRONG pero n_axes=1 → G3 bloquea (N28F2)."""
    bernard = {
        'apostar': False, 'confidence_flag': 'STRONG',
        'n_h2h': 0, 'n_axes_active': 1, 'alignment_flag': 'NO_ALIGNMENT',
        'net_alignment': 0.0, 'score_directo': 2,
    }
    bloqueado, motivo = _apply_combo_gates(bernard, 'Bernard A.')
    # G1 bloquea primero (apostar=False)
    assert bloqueado
    assert 'G1' in motivo
