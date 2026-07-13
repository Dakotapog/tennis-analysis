"""
test_nodo94_sprint3.py — REGLA-T53 tests para S3-A PlayerIntelligence (D90-05)

Invoca funciones reales de edge_calculator — nunca hardcodea fórmulas.
"""

import json
import os
import sys

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

import edge_calculator as ec


# ─────────────────────────────────────────────────────────────────────────────
# §1. _rank_bracket_ec
# ─────────────────────────────────────────────────────────────────────────────

class TestRankBracketEc:
    def test_dominant(self):
        assert ec._rank_bracket_ec(-100) == 'dominant'

    def test_favored(self):
        assert ec._rank_bracket_ec(-30) == 'favored'

    def test_even(self):
        assert ec._rank_bracket_ec(0) == 'even'
        assert ec._rank_bracket_ec(10) == 'even'
        assert ec._rank_bracket_ec(-10) == 'even'

    def test_underdog_slight(self):
        assert ec._rank_bracket_ec(30) == 'underdog_slight'

    def test_underdog_big(self):
        assert ec._rank_bracket_ec(200) == 'underdog_big'


# ─────────────────────────────────────────────────────────────────────────────
# §2. _SURFACE_MAP_EC_TO_DB
# ─────────────────────────────────────────────────────────────────────────────

class TestSurfaceMap:
    def test_hard_mapea_dura(self):
        assert ec._SURFACE_MAP_EC_TO_DB['hard'] == 'dura'

    def test_clay_mapea_arcilla(self):
        assert ec._SURFACE_MAP_EC_TO_DB['clay'] == 'arcilla'

    def test_grass_mapea_hierba(self):
        assert ec._SURFACE_MAP_EC_TO_DB['grass'] == 'hierba'

    def test_unknown_mapea_unknown(self):
        assert ec._SURFACE_MAP_EC_TO_DB['unknown'] == 'unknown'


# ─────────────────────────────────────────────────────────────────────────────
# §3. _get_player_intelligence
# ─────────────────────────────────────────────────────────────────────────────

class TestGetPlayerIntelligence:
    def _inject_db(self, monkeypatch, players_dict):
        monkeypatch.setattr(ec, '_player_db_cache', players_dict)

    def _make_entry(self, surface_wr=None, gap_wr=None, n=50):
        return {
            'n_matches': n,
            'surface_win_rates': surface_wr or {'dura': 0.60, 'arcilla': 0.45},
            'ranking_gap_win_rates': gap_wr or {'dominant': 0.70, 'underdog_big': 0.25},
        }

    def test_retorna_none_si_db_vacia(self, monkeypatch):
        self._inject_db(monkeypatch, {})
        result = ec._get_player_intelligence('Carlos Alcaraz', 'hard', 10, 200)
        assert result['pi_rank_gap_win_rate'] is None
        assert result['pi_svi_surface'] is None
        assert result['pi_n_total'] is None

    def test_retorna_none_si_jugador_no_en_db(self, monkeypatch):
        self._inject_db(monkeypatch, {'Novak_Djokovic': self._make_entry()})
        result = ec._get_player_intelligence('Rafael Nadal', 'clay', 1, 100)
        assert result['pi_rank_gap_win_rate'] is None

    def test_svi_hard_correcto(self, monkeypatch):
        entry = self._make_entry(surface_wr={'dura': 0.62})
        self._inject_db(monkeypatch, {'Carlos_Alcaraz': entry})
        result = ec._get_player_intelligence('Carlos Alcaraz', 'hard', 5, 100)
        assert result['pi_svi_surface'] == pytest.approx(0.62)

    def test_svi_clay_correcto(self, monkeypatch):
        entry = self._make_entry(surface_wr={'arcilla': 0.71})
        self._inject_db(monkeypatch, {'Carlos_Alcaraz': entry})
        result = ec._get_player_intelligence('Carlos Alcaraz', 'clay', 5, 100)
        assert result['pi_svi_surface'] == pytest.approx(0.71)

    def test_rank_gap_bracket_dominant(self, monkeypatch):
        # ranking_fav=10 (mejor), ranking_rival=200 → diff=-190 → dominant
        entry = self._make_entry(gap_wr={'dominant': 0.72})
        self._inject_db(monkeypatch, {'Carlos_Alcaraz': entry})
        result = ec._get_player_intelligence('Carlos Alcaraz', 'hard', 10, 200)
        assert result['pi_rank_gap_bracket'] == 'dominant'
        assert result['pi_rank_gap_win_rate'] == pytest.approx(0.72)

    def test_rank_gap_bracket_underdog_big(self, monkeypatch):
        # ranking_fav=300 (peor), ranking_rival=20 → diff=280 → underdog_big
        entry = self._make_entry(gap_wr={'underdog_big': 0.28})
        self._inject_db(monkeypatch, {'Jugador_X': entry})
        result = ec._get_player_intelligence('Jugador X', 'hard', 300, 20)
        assert result['pi_rank_gap_bracket'] == 'underdog_big'
        assert result['pi_rank_gap_win_rate'] == pytest.approx(0.28)

    def test_pi_n_total_presente(self, monkeypatch):
        entry = self._make_entry(n=127)
        self._inject_db(monkeypatch, {'Test_Player': entry})
        result = ec._get_player_intelligence('Test Player', 'hard', 50, 100)
        assert result['pi_n_total'] == 127

    def test_ranking_none_deja_bracket_none(self, monkeypatch):
        entry = self._make_entry()
        self._inject_db(monkeypatch, {'Test_Player': entry})
        result = ec._get_player_intelligence('Test Player', 'hard', None, None)
        assert result['pi_rank_gap_bracket'] is None
        assert result['pi_rank_gap_win_rate'] is None

    def test_superficie_sin_datos_retorna_none_svi(self, monkeypatch):
        entry = self._make_entry(surface_wr={'dura': 0.55})
        self._inject_db(monkeypatch, {'Test_Player': entry})
        result = ec._get_player_intelligence('Test Player', 'grass', 50, 100)
        assert result['pi_svi_surface'] is None


# ─────────────────────────────────────────────────────────────────────────────
# §4. integración: campos presentes en calcular_edge_completo
# ─────────────────────────────────────────────────────────────────────────────

class TestPlayerIntelligenceIntegration:
    def test_campos_pi_presentes_en_resultado(self, monkeypatch):
        """Los 5 campos PI aparecen en el output de calcular_edge_completo."""
        import glob
        h2h_files = sorted(glob.glob('reports/h2h_results_enhanced_*.json'))
        if not h2h_files:
            pytest.skip('No H2H files')
        data = json.load(open(h2h_files[-1]))
        if not data.get('partidos'):
            pytest.skip('No partidos')

        monkeypatch.setattr(ec, '_player_db_cache', {})
        monkeypatch.setattr(ec, '_kambi_coverage_cache', {})

        cal = ec.cargar_calibracion()
        resultado = ec.calcular_edge_completo(data['partidos'][0], cal)
        if resultado is None:
            pytest.skip('calcular_edge_completo returned None')

        for campo in ('pi_rank_gap_bracket', 'pi_rank_gap_win_rate',
                      'pi_svi_surface', 'pi_svi_n_surface', 'pi_n_total'):
            assert campo in resultado, f'Campo {campo} ausente en resultado'
