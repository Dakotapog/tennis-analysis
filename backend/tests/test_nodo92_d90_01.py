"""
test_nodo92_d90_01.py — REGLA-T53 tests for D90-01 (kambi_disponible)

Invoca funciones reales de fetch_kambi_coverage — nunca hardcodea lógica.
"""

import json
import os
import sys
from pathlib import Path

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BACKEND_DIR, 'scripts')
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, SCRIPTS_DIR)

import fetch_kambi_coverage as fkc


# ─────────────────────────────────────────────────────────────────────────────
# §1. _normalize_name
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeName:
    def test_minusculas(self):
        assert fkc._normalize_name('Carlos Alcaraz') == 'carlos alcaraz'

    def test_elimina_tildes(self):
        assert fkc._normalize_name('Nicolás Jarry') == 'nicolas jarry'

    def test_elimina_puntuacion(self):
        result = fkc._normalize_name('Hsu Y.H.')
        assert '.' not in result

    def test_strip_espacios(self):
        assert fkc._normalize_name('  Djokovic  ') == 'djokovic'


# ─────────────────────────────────────────────────────────────────────────────
# §2. is_player_available
# ─────────────────────────────────────────────────────────────────────────────

class TestIsPlayerAvailable:
    def _coverage(self, players):
        return {'players_normalized': players}

    def test_match_exacto(self):
        cov = self._coverage(['carlos alcaraz', 'novak djokovic'])
        assert fkc.is_player_available('Carlos Alcaraz', cov) is True

    def test_match_por_apellido(self):
        cov = self._coverage(['carlos alcaraz'])
        assert fkc.is_player_available('Alcaraz', cov) is True

    def test_no_disponible(self):
        cov = self._coverage(['carlos alcaraz'])
        assert fkc.is_player_available('Rafael Nadal', cov) is False

    def test_coverage_vacio_retorna_false(self):
        assert fkc.is_player_available('Djokovic', {}) is False

    def test_coverage_none_retorna_false(self):
        assert fkc.is_player_available('Djokovic', None) is False

    def test_apellido_corto_no_hace_fuzzy(self):
        # Apellido <= 3 chars no debe hacer fuzzy (evita falsos positivos)
        cov = self._coverage(['wu yibing'])
        assert fkc.is_player_available('Xu', cov) is False

    def test_nombre_con_tilde_encuentra_match(self):
        cov = self._coverage(['nicolas jarry'])
        assert fkc.is_player_available('Nicolás Jarry', cov) is True


# ─────────────────────────────────────────────────────────────────────────────
# §3. load_coverage / find_latest_coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadCoverage:
    def test_retorna_none_si_no_hay_archivos(self, tmp_path):
        result = fkc.load_coverage(reports_dir=tmp_path)
        assert result is None

    def test_carga_archivo_mas_reciente(self, tmp_path):
        older = tmp_path / 'kambi_coverage_20260710_100000.json'
        newer = tmp_path / 'kambi_coverage_20260712_090000.json'
        older.write_text(json.dumps({'fecha': '2026-07-10', 'players_normalized': ['player a']}))
        newer.write_text(json.dumps({'fecha': '2026-07-12', 'players_normalized': ['player b']}))

        result = fkc.load_coverage(reports_dir=tmp_path)
        assert result is not None
        assert result['fecha'] == '2026-07-12'
        assert 'player b' in result['players_normalized']

    def test_retorna_none_si_json_invalido(self, tmp_path):
        bad = tmp_path / 'kambi_coverage_20260713_100000.json'
        bad.write_text('NOT VALID JSON')
        result = fkc.load_coverage(reports_dir=tmp_path)
        assert result is None

    def test_find_latest_retorna_none_si_no_hay_archivos(self, tmp_path):
        assert fkc.find_latest_coverage(reports_dir=tmp_path) is None

    def test_find_latest_retorna_path_del_mas_reciente(self, tmp_path):
        f1 = tmp_path / 'kambi_coverage_20260710_090000.json'
        f2 = tmp_path / 'kambi_coverage_20260713_090000.json'
        f1.write_text('{}')
        f2.write_text('{}')
        result = fkc.find_latest_coverage(reports_dir=tmp_path)
        assert result == f2


# ─────────────────────────────────────────────────────────────────────────────
# §4. _annotate_kambi en edge_calculator (integración sin HTTP)
# ─────────────────────────────────────────────────────────────────────────────

class TestAnnotateKambi:
    def test_retorna_none_sin_coverage_cargado(self, monkeypatch):
        import edge_calculator as ec
        monkeypatch.setattr(ec, '_kambi_coverage_cache', {})
        result = ec._annotate_kambi('Carlos Alcaraz')
        assert result is None

    def test_retorna_true_cuando_jugador_disponible(self, monkeypatch):
        import edge_calculator as ec
        monkeypatch.setattr(ec, '_kambi_coverage_cache',
                            {'players_normalized': ['carlos alcaraz']})
        result = ec._annotate_kambi('Carlos Alcaraz')
        assert result is True

    def test_retorna_false_cuando_jugador_no_disponible(self, monkeypatch):
        import edge_calculator as ec
        monkeypatch.setattr(ec, '_kambi_coverage_cache',
                            {'players_normalized': ['novak djokovic']})
        result = ec._annotate_kambi('Rafael Nadal')
        assert result is False

    def test_campo_presente_en_calcular_edge_completo(self, monkeypatch):
        """kambi_disponible aparece en el output de calcular_edge_completo."""
        import edge_calculator as ec
        # Sin coverage → None
        monkeypatch.setattr(ec, '_kambi_coverage_cache', {})

        # Partido mínimo válido para que calcular_edge_completo retorne algo
        import json, glob
        h2h_files = sorted(glob.glob('reports/h2h_results_enhanced_*.json'))
        if not h2h_files:
            pytest.skip('No H2H files available')
        data = json.load(open(h2h_files[-1]))
        if not data.get('partidos'):
            pytest.skip('No partidos in H2H file')

        calibracion = ec.cargar_calibracion()
        resultado = ec.calcular_edge_completo(data['partidos'][0], calibracion)
        if resultado is None:
            pytest.skip('calcular_edge_completo returned None for this partido')

        assert 'kambi_disponible' in resultado
