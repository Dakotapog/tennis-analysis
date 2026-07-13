"""
test_nodo92_sprint2.py — REGLA-T53 tests for D90-03 (PlayerDB)

Invoca funciones reales del módulo build_player_db — nunca hardcodea fórmulas.
"""

import json
import os
import sys
import tempfile
import textwrap
from datetime import datetime, timedelta

import pytest

# ── import módulo bajo test ────────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BACKEND_DIR, 'scripts')
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, SCRIPTS_DIR)

import build_player_db as bpd


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_h2h_file(tmpdir, filename, partidos):
    """Helper: escribe un archivo H2H mínimo válido."""
    data = {
        'metadata': {'fecha': '2026-07-01'},
        'partidos': partidos,
        'estadisticas_globales': {},
    }
    path = os.path.join(tmpdir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    return path


def _partido_simple(slug1, slug2, hist1, hist2, ranking1=100, ranking2=200):
    return {
        'match_number': 1,
        'jugador1': slug1.replace('_', ' '),
        'jugador2': slug2.replace('_', ' '),
        'ranking_analysis': {
            f'{slug1}_ranking': ranking1,
            f'{slug2}_ranking': ranking2,
        },
        f'historial_{slug1}': hist1,
        f'historial_{slug2}': hist2,
    }


def _entry(fecha='01.07.2026', oponente='Rival A.', resultado='2-0',
           outcome='Ganó', torneo='M25 Laval', superficie='Dura',
           opponent_ranking=300):
    return {
        'fecha': fecha,
        'oponente': oponente,
        'resultado': resultado,
        'outcome': outcome,
        'torneo': torneo,
        'superficie': superficie,
        'opponent_ranking': opponent_ranking,
        'opponent_weight': 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# §1. Utilidades de parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestParseFilenameDate:
    def test_extrae_fecha_correcta(self):
        result = bpd._parse_filename_date('h2h_results_enhanced_20260712_104518.json')
        assert result == '2026-07-12'

    def test_extrae_fecha_otro_formato(self):
        result = bpd._parse_filename_date('/path/to/h2h_results_enhanced_20261231_235959.json')
        assert result == '2026-12-31'

    def test_filename_invalido_retorna_none(self):
        result = bpd._parse_filename_date('garbage_file.json')
        assert result is None


class TestFechaToIso:
    def test_convierte_formato_dd_mm_yyyy(self):
        assert bpd._fecha_to_iso('08.07.2026') == '2026-07-08'

    def test_convierte_primero_enero(self):
        assert bpd._fecha_to_iso('01.01.2025') == '2025-01-01'

    def test_formato_invalido_retorna_none(self):
        assert bpd._fecha_to_iso('2026-07-08') is None

    def test_none_retorna_none(self):
        assert bpd._fecha_to_iso(None) is None


class TestNormalizeSuperficie:
    def test_dura(self):
        assert bpd._normalize_superficie('Dura') == 'dura'

    def test_arcilla(self):
        assert bpd._normalize_superficie('Arcilla') == 'arcilla'

    def test_hierba(self):
        assert bpd._normalize_superficie('Hierba') == 'hierba'

    def test_na_mapea_unknown(self):
        assert bpd._normalize_superficie('N/A') == 'unknown'

    def test_vacio_mapea_unknown(self):
        assert bpd._normalize_superficie('') == 'unknown'


class TestThreeSetMatch:
    def test_dos_uno_es_tres_sets(self):
        assert bpd._is_three_set_match('2-1') is True

    def test_uno_dos_es_tres_sets(self):
        assert bpd._is_three_set_match('1-2') is True

    def test_dos_cero_no_es_tres_sets(self):
        assert bpd._is_three_set_match('2-0') is False

    def test_cero_dos_no_es_tres_sets(self):
        assert bpd._is_three_set_match('0-2') is False


class TestRankingBracket:
    def test_dominant_cuando_somos_mucho_mejor(self):
        # own=50, opp=200 → diff = 50-200 = -150 → dominant
        assert bpd._ranking_bracket(-150) == 'dominant'

    def test_underdog_big_cuando_somos_mucho_peor(self):
        # own=300, opp=50 → diff = 250 → underdog_big
        assert bpd._ranking_bracket(250) == 'underdog_big'

    def test_even_cuando_rankings_parecidos(self):
        assert bpd._ranking_bracket(5) == 'even'
        assert bpd._ranking_bracket(-5) == 'even'

    def test_none_retorna_none(self):
        assert bpd._ranking_bracket(None) is None


# ─────────────────────────────────────────────────────────────────────────────
# §2. Estadísticas agregadas
# ─────────────────────────────────────────────────────────────────────────────

class TestComputePlayerStats:
    def _make_rows(self, won_list, superficie='dura', tier='itf',
                   bracket='even', is_three_set=False, is_underdog=None):
        return [
            {
                'won': w,
                'superficie': superficie,
                'tier': tier,
                'ranking_bracket': bracket,
                'is_three_set': is_three_set,
                'is_underdog': is_underdog,
            }
            for w in won_list
        ]

    def test_n_total_correcto(self):
        rows = self._make_rows([True, False, True])
        stats = bpd._compute_player_stats(rows)
        assert stats['n_total'] == 3

    def test_surface_win_rate_correcto(self):
        rows = self._make_rows([True, True, False], superficie='arcilla')
        stats = bpd._compute_player_stats(rows)
        assert stats['surface_stats']['arcilla']['wins'] == 2
        assert stats['surface_stats']['arcilla']['losses'] == 1
        assert stats['surface_stats']['arcilla']['win_rate'] == pytest.approx(2/3, rel=1e-3)

    def test_tier_stats_agrega_correctamente(self):
        rows = self._make_rows([True, False], tier='challenger')
        stats = bpd._compute_player_stats(rows)
        assert stats['tier_stats']['challenger']['n'] == 2
        assert stats['tier_stats']['challenger']['win_rate'] == pytest.approx(0.5)

    def test_ranking_gap_stats_agrega_correctamente(self):
        rows = self._make_rows([True, True, False, False, False], bracket='underdog_big')
        stats = bpd._compute_player_stats(rows)
        assert stats['ranking_gap_stats']['underdog_big']['n'] == 5
        assert stats['ranking_gap_stats']['underdog_big']['wins'] == 2

    def test_prs_three_set_win_rate(self):
        rows = self._make_rows([True, False, True], is_three_set=True)
        stats = bpd._compute_player_stats(rows)
        assert stats['prs_stats']['three_set']['win_rate'] == pytest.approx(2/3, rel=1e-3)

    def test_prs_underdog_win_rate(self):
        rows = self._make_rows([True, False], is_underdog=True)
        stats = bpd._compute_player_stats(rows)
        assert stats['prs_stats']['underdog']['win_rate'] == pytest.approx(0.5)

    def test_stats_vacias_con_cero_filas(self):
        stats = bpd._compute_player_stats([])
        assert stats['n_total'] == 0


# ─────────────────────────────────────────────────────────────────────────────
# §3. process_files — integración con archivos reales H2H
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessFiles:
    def test_deduplicacion_prefiere_archivo_mas_reciente(self, tmp_path):
        """Mismo partido en dos archivos: la fila más reciente gana."""
        hist = [_entry(fecha='01.06.2026', oponente='Rival X.', resultado='2-0',
                       outcome='Ganó', opponent_ranking=200)]
        hist_newer = [_entry(fecha='01.06.2026', oponente='Rival X.', resultado='2-0',
                             outcome='Ganó', opponent_ranking=350)]  # ranking actualizado

        p1 = _partido_simple('Jugador_A', 'Jugador_B', hist, [])
        p2 = _partido_simple('Jugador_A', 'Jugador_B', hist_newer, [])

        _make_h2h_file(str(tmp_path), 'h2h_results_enhanced_20260601_100000.json', [p1])
        _make_h2h_file(str(tmp_path), 'h2h_results_enhanced_20260602_100000.json', [p2])

        files = sorted((tmp_path / f).as_posix() for f in tmp_path.iterdir())
        players, n_files, n_raw, n_deduped = bpd.process_files(files, verbose=False)

        assert 'Jugador_A' in players
        # Deduplicado: solo 1 fila
        assert len(players['Jugador_A']['rows']) == 1
        # La más reciente: opponent_ranking=350
        assert players['Jugador_A']['rows'][0]['opponent_ranking'] == 350

    def test_descarta_resultado_invalido(self, tmp_path):
        """Filas con resultado '-' o '0-0' deben descartarse."""
        hist = [
            _entry(resultado='-'),
            _entry(resultado='0-0'),
            _entry(resultado='2-1', fecha='02.07.2026'),
        ]
        p = _partido_simple('Jugador_A', 'Jugador_B', hist, [])
        _make_h2h_file(str(tmp_path), 'h2h_results_enhanced_20260702_120000.json', [p])
        files = [str(tmp_path / 'h2h_results_enhanced_20260702_120000.json')]
        players, _, n_raw, n_deduped = bpd.process_files(files, verbose=False)
        assert n_deduped == 1  # solo la fila válida

    def test_resolution_confidence_es_exact(self, tmp_path):
        hist = [_entry(fecha='10.06.2026', oponente='Alguien B.')]
        p = _partido_simple('Jugador_A', 'Jugador_B', hist, [])
        _make_h2h_file(str(tmp_path), 'h2h_results_enhanced_20260610_090000.json', [p])
        files = [str(tmp_path / 'h2h_results_enhanced_20260610_090000.json')]
        players, _, _, _ = bpd.process_files(files, verbose=False)
        row = players['Jugador_A']['rows'][0]
        assert row['resolution_confidence'] == 'exact'

    def test_own_ranking_tomado_de_ranking_analysis(self, tmp_path):
        hist = [_entry()]
        p = _partido_simple('Jugador_A', 'Jugador_B', hist, [], ranking1=77)
        _make_h2h_file(str(tmp_path), 'h2h_results_enhanced_20260612_100000.json', [p])
        files = [str(tmp_path / 'h2h_results_enhanced_20260612_100000.json')]
        players, _, _, _ = bpd.process_files(files, verbose=False)
        assert players['Jugador_A']['own_ranking'] == 77

    def test_tier_detectado_correctamente(self, tmp_path):
        hist = [_entry(torneo='Granby Challenger')]
        p = _partido_simple('Jugador_A', 'Jugador_B', hist, [])
        _make_h2h_file(str(tmp_path), 'h2h_results_enhanced_20260613_100000.json', [p])
        files = [str(tmp_path / 'h2h_results_enhanced_20260613_100000.json')]
        players, _, _, _ = bpd.process_files(files, verbose=False)
        assert players['Jugador_A']['rows'][0]['tier'] == 'challenger'

    def test_stats_computadas_tras_process(self, tmp_path):
        hist = [
            _entry(fecha='01.07.2026', outcome='Ganó', resultado='2-0'),
            _entry(fecha='02.07.2026', outcome='Ganó', resultado='2-1'),
            _entry(fecha='03.07.2026', outcome='Perdió', resultado='1-2'),
        ]
        p = _partido_simple('Jugador_A', 'Jugador_B', hist, [])
        _make_h2h_file(str(tmp_path), 'h2h_results_enhanced_20260703_100000.json', [p])
        files = [str(tmp_path / 'h2h_results_enhanced_20260703_100000.json')]
        players, _, _, _ = bpd.process_files(files, verbose=False)
        stats = players['Jugador_A']['stats']
        assert stats['n_total'] == 3
        assert stats['surface_stats']['dura']['wins'] == 2
        assert stats['prs_stats']['three_set']['n'] == 2  # 2-1 y 1-2


# ─────────────────────────────────────────────────────────────────────────────
# §4. Índice
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildIndex:
    def test_index_contiene_win_rates_superficie(self):
        rows = [
            {'fecha': '2026-07-01', 'oponente': 'X', 'resultado': '2-0', 'won': True,
             'superficie': 'dura', 'torneo': 'M25 Test', 'tier': 'itf',
             'opponent_ranking': 300, 'opponent_weight': 1, 'is_three_set': False,
             'is_underdog': False, 'ranking_bracket': 'dominant',
             'resolution_confidence': 'exact', 'source_file': 'f.json', 'ranking_asof': '2026-07-01'},
            {'fecha': '2026-07-02', 'oponente': 'Y', 'resultado': '2-0', 'won': False,
             'superficie': 'dura', 'torneo': 'M25 Test', 'tier': 'itf',
             'opponent_ranking': 250, 'opponent_weight': 1, 'is_three_set': False,
             'is_underdog': False, 'ranking_bracket': 'dominant',
             'resolution_confidence': 'exact', 'source_file': 'f.json', 'ranking_asof': '2026-07-01'},
            {'fecha': '2026-07-03', 'oponente': 'Z', 'resultado': '2-1', 'won': True,
             'superficie': 'dura', 'torneo': 'M25 Test', 'tier': 'itf',
             'opponent_ranking': 200, 'opponent_weight': 1, 'is_three_set': True,
             'is_underdog': False, 'ranking_bracket': 'dominant',
             'resolution_confidence': 'exact', 'source_file': 'f.json', 'ranking_asof': '2026-07-01'},
        ]
        players = {
            'Test_Player': {
                'slug': 'Test_Player',
                'own_ranking': 100,
                'ranking_asof': '2026-07-01',
                'rows': rows,
                'stats': bpd._compute_player_stats(rows),
            }
        }
        index = bpd.build_index(players)
        assert 'Test_Player' in index
        entry = index['Test_Player']
        assert entry['n_matches'] == 3
        assert 'dura' in entry['surface_win_rates']
        assert entry['surface_win_rates']['dura'] == pytest.approx(2/3, rel=1e-3)
        assert entry['prs_three_set_win_rate'] == pytest.approx(1.0)

    def test_index_excluye_superficie_con_menos_de_3_partidos(self):
        rows = [
            {'fecha': '2026-07-01', 'oponente': 'X', 'resultado': '2-0', 'won': True,
             'superficie': 'hierba', 'torneo': 'Wimbledon', 'tier': 'grand_slam',
             'opponent_ranking': 300, 'opponent_weight': 1, 'is_three_set': False,
             'is_underdog': False, 'ranking_bracket': 'dominant',
             'resolution_confidence': 'exact', 'source_file': 'f.json', 'ranking_asof': '2026-07-01'},
            {'fecha': '2026-07-02', 'oponente': 'Y', 'resultado': '2-0', 'won': True,
             'superficie': 'hierba', 'torneo': 'Wimbledon', 'tier': 'grand_slam',
             'opponent_ranking': 250, 'opponent_weight': 1, 'is_three_set': False,
             'is_underdog': False, 'ranking_bracket': 'dominant',
             'resolution_confidence': 'exact', 'source_file': 'f.json', 'ranking_asof': '2026-07-01'},
        ]
        players = {
            'Test_Player': {
                'slug': 'Test_Player',
                'own_ranking': 100,
                'ranking_asof': '2026-07-01',
                'rows': rows,
                'stats': bpd._compute_player_stats(rows),
            }
        }
        index = bpd.build_index(players)
        # hierba tiene n=2 < 3 → excluida del índice
        assert 'hierba' not in index['Test_Player']['surface_win_rates']
