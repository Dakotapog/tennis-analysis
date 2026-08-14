"""REGLA-T53 — Nodo-174 D174-11: telemetría del gate Kambi en metadata.

Complementa D90-01 (_annotate_kambi) — el % de exclusión por falta de
cobertura Betplay debe verse en el reporte diario, no descubrirse por
auditoría. `metadata.kambi = {n_disponibles, n_no_disponibles, n_desconocido}`.
"""
import json

import pytest


def _make_partido(favored='p1', cuota1=2.02, cuota2=1.70, confidence=75.3,
                   ranking1=850, ranking2=910):
    jugador1, jugador2 = 'Mario Arce', 'Vlajic Josip'
    favored_player = jugador1 if favored == 'p1' else jugador2
    pred = {
        'favored_player': favored_player,
        'confidence': confidence,
        'historial_incompleto': {'p1': False, 'p2': False},
        'scores': {
            'p1_final_weight': 1.5, 'p2_final_weight': 0.8,
            'score_difference': 0.7,
        },
        'score_breakdown': {
            'player1': {k: {'contribution': 0.0, 'contribution_pct': '0.0%'} for k in (
                'surface_specialization', 'form_recent', 'common_opponents', 'h2h_direct',
                'ranking_momentum', 'elo_rating', 'home_advantage', 'strength_of_schedule')},
            'player2': {k: {'contribution': 0.0, 'contribution_pct': '0.0%'} for k in (
                'surface_specialization', 'form_recent', 'common_opponents', 'h2h_direct',
                'ranking_momentum', 'elo_rating', 'home_advantage', 'strength_of_schedule')},
        },
        'weights_used': {
            'surface_specialization': 0.18, 'form_recent': 0.22,
            'common_opponents': 0.10, 'h2h_direct': 0.10,
            'ranking_momentum': 0.20, 'elo_rating': 0.15,
            'home_advantage': 0.05, 'strength_of_schedule': 0.00,
        },
        'markov_analysis': None,
        'tardio_analysis': None,
        'circuit_asymmetry': {'signal': 'SYMMETRIC', 'ratio': 1.0},
        'surface_specialization_meta': {
            'player1': {'volume_confidence': 0.8},
            'player2': {'volume_confidence': 0.8},
        },
    }
    return {
        'jugador1': jugador1,
        'jugador2': jugador2,
        'cuota1': cuota1,
        'cuota2': cuota2,
        'torneo_nombre': 'M15 Cary (USA)',
        'tipo_cancha': 'hard',
        'torneo_completo': 'ITF - INDIVIDUALES: M15 Cary (USA)',
        'match_url': 'https://www.flashscore.co/match/tennis/test/AAABBBCC/#/h2h',
        'match_id': 'AAABBBCC',
        'ranking1': ranking1,
        'ranking2': ranking2,
        'data_quality': {
            'historial_extraido_p1': True, 'historial_extraido_p2': True,
            'n_partidos_p1': 14, 'n_partidos_p2': 11,
        },
        'ranking_analysis': {
            'Mario_Arce_ranking': ranking1,
            'Vlajic_Josip_ranking': ranking2,
            'common_opponents_count': 0,
            'p1_rivalry_score': 0.55,
            'p2_rivalry_score': 0.45,
            'prediction': pred,
            'Mario_Arce_metrics': None,
            'Vlajic_Josip_metrics': None,
            'Mario_Arce_elo': 1400.0,
            'Vlajic_Josip_elo': 1380.0,
        },
        'form_analysis': {'Mario_Arce_form': None, 'Vlajic_Josip_form': None},
        'enfrentamientos_directos': [],
        'estadisticas': {
            'partidos_Mario_Arce': 14, 'partidos_Vlajic_Josip': 11,
            'enfrentamientos_totales': 0,
        },
    }


def _write_h2h(tmp_path, partidos, name='h2h_test.json'):
    from analysis.rivalry_analyzer import RIVALRY_VERSION
    h2h_file = tmp_path / name
    h2h_file.write_text(json.dumps({
        'metadata': {'rivalry_version': RIVALRY_VERSION},
        'partidos': partidos,
    }))
    return h2h_file


def test_174_11_sin_coverage_todo_desconocido(tmp_path, monkeypatch):
    """Sin kambi_coverage cargado -- todos los picks caen en n_desconocido."""
    import edge_calculator as ec
    monkeypatch.setattr(ec, '_kambi_coverage_cache', {})

    partidos = [_make_partido(favored='p1'), _make_partido(favored='p2')]
    h2h_file = _write_h2h(tmp_path, partidos)
    resultado = ec.procesar_archivo_h2h(str(h2h_file), output_file=None, shadow_log=False)

    kambi = resultado['metadata']['kambi']
    assert kambi == {'n_disponibles': 0, 'n_no_disponibles': 0, 'n_desconocido': 2}


def test_174_11_disponible_y_no_disponible_se_cuentan_por_separado(tmp_path, monkeypatch):
    """Con coverage cargado: favorito en players_normalized -> disponible,
    favorito ausente -> no_disponible."""
    import edge_calculator as ec
    monkeypatch.setattr(ec, '_kambi_coverage_cache',
                         {'players_normalized': ['mario arce']})

    partidos = [_make_partido(favored='p1'), _make_partido(favored='p2')]
    h2h_file = _write_h2h(tmp_path, partidos)
    resultado = ec.procesar_archivo_h2h(str(h2h_file), output_file=None, shadow_log=False)

    kambi = resultado['metadata']['kambi']
    assert kambi['n_disponibles'] == 1     # Mario Arce (favored=p1) -- en la lista
    assert kambi['n_no_disponibles'] == 1  # Vlajic Josip (favored=p2) -- ausente
    assert kambi['n_desconocido'] == 0


def test_174_11_invariante_suma_igual_procesados(tmp_path, monkeypatch):
    """n_disponibles + n_no_disponibles + n_desconocido == n_procesados
    (mismo invariante que D173-01 funnel, test_173_01e)."""
    import edge_calculator as ec
    monkeypatch.setattr(ec, '_kambi_coverage_cache',
                         {'players_normalized': ['mario arce']})

    partidos = [_make_partido(favored='p1'), _make_partido(favored='p2')]
    h2h_file = _write_h2h(tmp_path, partidos)
    resultado = ec.procesar_archivo_h2h(str(h2h_file), output_file=None, shadow_log=False)

    kambi = resultado['metadata']['kambi']
    n_procesados = resultado['metadata']['n_procesados']
    assert (kambi['n_disponibles'] + kambi['n_no_disponibles'] + kambi['n_desconocido']
            == n_procesados)


def test_174_11_campo_kambi_disponible_en_cada_resultado_individual(tmp_path, monkeypatch):
    """Cada pick individual sigue trayendo kambi_disponible (D90-01) -- la
    telemetría es un agregado, no reemplaza el campo por-pick."""
    import edge_calculator as ec
    monkeypatch.setattr(ec, '_kambi_coverage_cache',
                         {'players_normalized': ['mario arce']})

    partidos = [_make_partido(favored='p1')]
    h2h_file = _write_h2h(tmp_path, partidos)
    resultado = ec.procesar_archivo_h2h(str(h2h_file), output_file=None, shadow_log=False)

    picks = resultado['apostar'] + resultado['watchlist'] + resultado['sin_edge']
    assert len(picks) == 1
    assert picks[0]['kambi_disponible'] is True
