"""
Tests Nodo-155 — HCUC Pipeline Integration (H152-01 acumulación diaria automática).
REGLA-T53: cada test invoca la función real del módulo — nunca hardcodea la fórmula.

Funciones cubiertas:
  D155-01: analyze_surface_specialization() → top20_wins / campeonatos_expirados_count
  D155-02: _calc_hcuc_convergence() + asignación hcuc_convergence/hcuc_signals en
           calcular_edge_completo()

Casos semilla reales (auditoría 2026-07-29, validation/preregistered_hypotheses.json H152-01):
  Cocciaretto @2.75 (SCALP_TOP20), Shick @2.75 (CAMPEON_RECIENTE),
  Gea @2.50 (RACHA_HOT + SCALP_TOP20 + CAMPEONATOS_EXPIRADOS) — los 3 GANARON.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge_calculator import _calc_hcuc_convergence, calcular_edge_completo


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resultado(superficie='hard', p_modelo=0.51, cuota_favorito=2.75,
               markov_favorito='neutral', markov_conf_fav=0.0):
    return {
        'superficie': superficie,
        'p_modelo': p_modelo,
        'cuota_favorito': cuota_favorito,
        'markov_favorito': markov_favorito,
        'markov_conf_fav': markov_conf_fav,
    }


def _surf(score=0.0, campeon_days_ago=None, campeon_tier=None,
          top20_wins=0, campeonatos_expirados_count=0):
    return {
        'score': score,
        'campeon_days_ago': campeon_days_ago,
        'campeon_tier': campeon_tier,
        'top20_wins': top20_wins,
        'campeonatos_expirados_count': campeonatos_expirados_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Casos semilla reales — deben dar match=True (3/3 ganaron 2026-07-29)
# ─────────────────────────────────────────────────────────────────────────────

class TestCasosSemillaReales:

    def test_cocciaretto_scalp_top20(self):
        """Cocciaretto @2.75 — quality=16.5, delta=0.18, p=0.51, 2 scalps TOP-20. GANO."""
        r = _resultado(p_modelo=0.51, cuota_favorito=2.75)
        fav = _surf(score=16.5, top20_wins=2)
        dog = _surf(score=16.5 - 0.18)
        out = _calc_hcuc_convergence(r, fav, dog)
        assert out['match'] is True
        assert 'SCALP_TOP20' in out['signals']

    def test_shick_campeon_reciente(self):
        """Shick @2.75 — quality=18.6, delta=0.10, p=0.506, campeón ATP500 hace 5 días. GANO."""
        r = _resultado(p_modelo=0.506, cuota_favorito=2.75)
        fav = _surf(score=18.6, campeon_days_ago=5, campeon_tier='atp500')
        dog = _surf(score=18.6 - 0.10)
        out = _calc_hcuc_convergence(r, fav, dog)
        assert out['match'] is True
        assert 'CAMPEON_RECIENTE' in out['signals']

    def test_gea_triple_senal(self):
        """Gea @2.50 — quality=50.5, delta=0.09, p=0.505, HOT+scalp+4 campeonatos expirados. GANO."""
        r = _resultado(p_modelo=0.505, cuota_favorito=2.50,
                       markov_favorito='HOT', markov_conf_fav=0.70)
        fav = _surf(score=50.5, top20_wins=1, campeonatos_expirados_count=4)
        dog = _surf(score=50.5 - 0.09)
        out = _calc_hcuc_convergence(r, fav, dog)
        assert out['match'] is True
        assert set(out['signals']) == {'RACHA_HOT', 'SCALP_TOP20', 'CAMPEONATOS_EXPIRADOS'}


# ─────────────────────────────────────────────────────────────────────────────
# Rutas de rechazo — cada gate individual debe bloquear
# ─────────────────────────────────────────────────────────────────────────────

class TestRutasDeRechazo:

    def test_superficie_no_hard_rechaza(self):
        """Clay/tierra nunca puede activar HCUC, aunque el resto converja."""
        r = _resultado(superficie='clay', p_modelo=0.51, cuota_favorito=2.75)
        fav = _surf(score=50.0, top20_wins=3)
        dog = _surf(score=40.0)
        out = _calc_hcuc_convergence(r, fav, dog)
        assert out == {'match': False, 'signals': []}

    def test_quality_insuficiente_rechaza(self):
        """quality < 16.5 → rechaza aunque delta y confianza converjan."""
        r = _resultado(p_modelo=0.51, cuota_favorito=2.75)
        fav = _surf(score=10.0, top20_wins=3)
        dog = _surf(score=9.5)
        out = _calc_hcuc_convergence(r, fav, dog)
        assert out['match'] is False

    def test_delta_insuficiente_rechaza(self):
        """delta < 0.08 entre favorito y rival → rechaza."""
        r = _resultado(p_modelo=0.51, cuota_favorito=2.75)
        fav = _surf(score=20.0, top20_wins=3)
        dog = _surf(score=19.95)  # delta=0.05 < 0.08
        out = _calc_hcuc_convergence(r, fav, dog)
        assert out['match'] is False

    def test_confianza_fuera_de_rango_rechaza(self):
        """p_modelo fuera de [0.495, 0.52] (no es coin flip) → rechaza."""
        r = _resultado(p_modelo=0.60, cuota_favorito=2.75)
        fav = _surf(score=50.0, top20_wins=3)
        dog = _surf(score=40.0)
        out = _calc_hcuc_convergence(r, fav, dog)
        assert out['match'] is False

    def test_cuota_fuera_de_rango_rechaza(self):
        """cuota_favorito fuera de [2.3, 3.0] → rechaza."""
        r = _resultado(p_modelo=0.51, cuota_favorito=1.80)
        fav = _surf(score=50.0, top20_wins=3)
        dog = _surf(score=40.0)
        out = _calc_hcuc_convergence(r, fav, dog)
        assert out['match'] is False

    def test_sin_senal_especial_rechaza(self):
        """Todo converge (hard+quality+delta+confianza+cuota) pero 0 señales especiales."""
        r = _resultado(p_modelo=0.51, cuota_favorito=2.75,
                       markov_favorito='neutral', markov_conf_fav=0.0)
        fav = _surf(score=20.0, top20_wins=0, campeonatos_expirados_count=0)
        dog = _surf(score=19.5)
        out = _calc_hcuc_convergence(r, fav, dog)
        assert out == {'match': False, 'signals': []}

    def test_campeon_tier_no_calificado_no_activa_senal(self):
        """campeon_days_ago<=30 pero tier=itf (no en atp500/atp1000/gs/wta500) → no cuenta como señal."""
        r = _resultado(p_modelo=0.51, cuota_favorito=2.75)
        fav = _surf(score=20.0, campeon_days_ago=10, campeon_tier='itf')
        dog = _surf(score=19.5)
        out = _calc_hcuc_convergence(r, fav, dog)
        assert out['match'] is False


# ─────────────────────────────────────────────────────────────────────────────
# Integración: calcular_edge_completo() asigna hcuc_convergence/hcuc_signals
# ─────────────────────────────────────────────────────────────────────────────

def test_hcuc_fields_en_edge_completo():
    """calcular_edge_completo incluye hcuc_convergence/hcuc_signals en el resultado
    sin alterar edge/kelly_kl/apostar (D155-02 es puramente observacional)."""
    partido = {
        'jugador1': 'Player A',
        'jugador2': 'Player B',
        'cuota1': 2.75,
        'cuota2': 1.45,
        'superficie': 'hard',
        'pais': 'USA',
        'cuota_es_real': True,
        'torneo_nombre': 'Washington WTA',
        'torneo': 'Washington WTA',
        'tier': 'atp500',
        'ranking_analysis': {
            'prediction': {
                'favored_player': 'Player B',
                'confidence': 51.0,
                'p_model': 0.51,
                'reasoning': [],
                'surface_specialization_meta': {
                    'player1': {'score': 16.5 - 0.18, 'torneo_completo': False, 'gcs_active': False},
                    'player2': {
                        'score': 16.5, 'torneo_completo': False, 'gcs_active': False,
                        'top20_wins': 2, 'campeonatos_expirados_count': 0,
                    },
                },
                'h2h_stats': {'wins': 3, 'losses': 2, 'total': 5},
                'historial_incompleto': {'p1': False, 'p2': False},
            }
        },
    }

    resultado = calcular_edge_completo(partido, calibracion={})

    assert resultado is not None
    assert 'hcuc_convergence' in resultado
    assert 'hcuc_signals' in resultado
    assert isinstance(resultado['hcuc_signals'], list)
