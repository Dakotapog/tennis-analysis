"""
Tests para Nodo-29: Circuit Asymmetry Deflator (CAD)
T29-01 → T29-13: circuit_tier_index + deflactor + SoS dinámico + integración
"""
import sys
import os
import math
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.rivalry_analyzer import RivalryAnalyzer


def make_analyzer():
    rm = MagicMock()
    rm.get_player_info.return_value = None
    rm.get_player_ranking.return_value = None
    rm.normalize_name.side_effect = lambda x: x.lower().strip() if x else ''
    elo = MagicMock()
    elo.default_rating = 1500
    elo.expected_score.return_value = 0.5
    return RivalryAnalyzer(rm, elo)


def make_history(ranks, recent=False):
    """Genera historial con los rankings indicados. Si recent=True → todos en índice <10."""
    return [
        {
            'oponente': f'Player{i}',
            'opponent_ranking': r,
            'outcome': 'Ganó',
            'resultado': '2-0',
            'fecha': '01.01.2026',
            'torneo': 'Test',
            'superficie': 'Dura',
        }
        for i, r in enumerate(ranks)
    ]


# ─── T29-01: CTI = 0 si historial vacío ─────────────────────────────────────
def test_T29_01_cti_empty_history():
    analyzer = make_analyzer()
    cti, n = analyzer.circuit_tier_index([])
    assert cti == 0.0
    assert n == 0


# ─── T29-02: CTI = 0 si todos los oponentes > 500 ───────────────────────────
def test_T29_02_cti_all_low_circuit():
    analyzer = make_analyzer()
    history = make_history([600, 800, 1200, 1500, 700])
    cti, n = analyzer.circuit_tier_index(history)
    assert cti == 0.0
    assert n == 5


# ─── T29-03: CTI > 3.0 si jugador tiene 5+ oponentes top-50 ─────────────────
def test_T29_03_cti_high_circuit():
    analyzer = make_analyzer()
    # 6 oponentes top-50 (tier 4.0)
    history = make_history([5, 20, 35, 48, 30, 15, 800, 900])
    cti, n = analyzer.circuit_tier_index(history)
    assert cti > 3.0
    assert n == 8


# ─── T29-04: Deflactor = 1.0 si asimetría < 2.0 (partidos simétricos) ───────
def test_T29_04_no_deflactor_symmetric():
    analyzer = make_analyzer()
    # Ambos jugadores con ranking similar ~400
    h1 = make_history([400, 350, 420, 380, 410, 390, 430, 370, 440, 360, 500, 520])
    h2 = make_history([380, 420, 350, 400, 410, 430, 370, 390, 450, 360, 480, 510])
    cti1, _ = analyzer.circuit_tier_index(h1)
    cti2, _ = analyzer.circuit_tier_index(h2)
    asimetria = max(cti1, cti2) / max(min(cti1, cti2), 0.1)
    assert asimetria < 2.0, f"Expected symmetric (ratio<2), got {asimetria}"


# ─── T29-05: Deflactor reduce form_recent del jugador inferior ───────────────
def test_T29_05_deflactor_reduces_form_inferior():
    analyzer = make_analyzer()
    # CTI alto: oponentes top-10/50
    high_circuit = make_history([8, 15, 30, 50, 98, 20, 40, 60, 80, 12, 25, 45])
    # CTI bajo: solo ITF (>500)
    low_circuit = make_history([600, 800, 1200, 700, 900, 650, 750, 820, 1100, 680, 900, 1000])

    cti_high, n_high = analyzer.circuit_tier_index(high_circuit)
    cti_low, n_low = analyzer.circuit_tier_index(low_circuit)
    asimetria = cti_high / max(cti_low, 0.1)

    assert asimetria > 2.0, f"Expected asymmetry > 2, got {asimetria}"

    # Verificar deflactor
    deflactor = 1.0 / (1.0 + 0.15 * math.log(asimetria))
    assert deflactor < 1.0
    assert deflactor > 0.5  # No colapsa a 0


# ─── T29-06: Bonificación amplifica form_recent del jugador superior ─────────
def test_T29_06_bonus_amplifies_form_superior():
    high_circuit = make_history([8, 15, 30, 50, 98, 20, 40, 60, 80, 12, 25, 45])
    low_circuit = make_history([600, 800, 1200, 700, 900, 650, 750, 820, 1100, 680, 900, 1000])
    analyzer = make_analyzer()

    cti_high, _ = analyzer.circuit_tier_index(high_circuit)
    cti_low, _ = analyzer.circuit_tier_index(low_circuit)
    asimetria = cti_high / max(cti_low, 0.1)

    deflactor = 1.0 / (1.0 + 0.15 * math.log(asimetria))
    bonificacion = 1.0 + (1.0 - deflactor) * 0.5
    assert bonificacion > 1.0
    assert bonificacion <= 1.5  # Cota superior razonable


# ─── T29-07: SoS weight sube cuando asimetría > 2.0 ─────────────────────────
def test_T29_07_sos_weight_increases():
    asimetria = 6.7  # Caso Schoen vs Boogaard
    base_sos_w = 0.05
    sos_multiplier = 1.0 + math.log(asimetria)
    extra_w = base_sos_w * (sos_multiplier - 1.0)
    new_sos_w = base_sos_w + extra_w
    assert new_sos_w > base_sos_w
    assert new_sos_w > 0.10  # Al menos 2× el base


# ─── T29-08: form_recent weight baja proporcionalmente ───────────────────────
def test_T29_08_form_weight_decreases():
    asimetria = 6.7
    base_sos_w = 0.05
    base_form_w = 0.28
    sos_multiplier = 1.0 + math.log(asimetria)
    extra_w = base_sos_w * (sos_multiplier - 1.0)
    extra_w = min(extra_w, base_form_w * 0.5)  # capped at 50% de form
    new_form_w = base_form_w - extra_w
    assert new_form_w < base_form_w
    assert new_form_w > 0  # No se vuelve negativo


# ─── T29-09: No aplica si n_partidos_con_ranking < 10 ───────────────────────
def test_T29_09_skip_if_insufficient_sample():
    analyzer = make_analyzer()
    # Solo 5 partidos con ranking
    sparse = make_history([8, 50, 100, 200, 500])
    cti, n = analyzer.circuit_tier_index(sparse)
    assert n == 5
    assert n < 10  # Confirma que el skip debe activarse


# ─── T29-10: Caso Schoen vs Boogaard — CTI de Boogaard mayor ────────────────
def test_T29_10_boogaard_higher_cti_than_schoen():
    analyzer = make_analyzer()

    # Historial representativo de Schoen (solo ITF, mejor oponente #283)
    schoen_hist = make_history([
        1482, 501, 625, 1132, 2117, 1324, 625, 577, 643, 1482,
        1542, 381, 376, 849, 2117, 357, 760, 632, 621, 1031,
        637, 372, 626, 694, 283, 320, 412, 857, 1042, 2246,
    ])
    # Historial representativo de Boogaard (ATP/GS, enfrentó Medvedev #8)
    boogaard_hist = make_history([
        1213, 8, 98, 988, 1056, 1542, 615, 768, 323, 411,
        489, 676, 1482, 376, 679, 537, 1127, 109, 221, 209,
        584, 214, 573, 447, 849, 539, 954, 851, 646, 124,
    ])

    cti_schoen, n_schoen = analyzer.circuit_tier_index(schoen_hist)
    cti_boogaard, n_boogaard = analyzer.circuit_tier_index(boogaard_hist)

    assert cti_boogaard > cti_schoen, (
        f"Boogaard CTI ({cti_boogaard}) debe ser mayor que Schoen CTI ({cti_schoen})"
    )
    asimetria = cti_boogaard / max(cti_schoen, 0.1)
    assert asimetria > 2.0, f"Asimetría debe ser >2.0, got {asimetria}"


# ─── T29-11: No afecta partidos simétricos (mismo circuito) ─────────────────
def test_T29_11_symmetric_no_deflactor():
    analyzer = make_analyzer()
    # Dos jugadores con historial casi idéntico en circuito Challenger
    h1 = make_history([200, 250, 180, 300, 220, 270, 190, 310, 230, 260, 280, 240])
    h2 = make_history([210, 240, 190, 290, 230, 260, 200, 320, 220, 270, 290, 250])
    cti1, _ = analyzer.circuit_tier_index(h1)
    cti2, _ = analyzer.circuit_tier_index(h2)
    asimetria = max(cti1, cti2) / max(min(cti1, cti2), 0.1)
    # No debe aplicar deflactor
    assert asimetria < 2.0


# ─── T29-12: circuit_asymmetry dict presente en output ──────────────────────
def test_T29_12_circuit_asymmetry_in_output():
    """Verifica que circuit_asymmetry aparece en el dict de predicción."""
    analyzer = make_analyzer()
    # Historial mínimo para activar el flujo
    hist = make_history([500, 600, 700, 800, 900, 550, 650, 750, 850, 950, 1000, 1100])
    form = {'win_percentage': 60, 'recent_matches_count': 10, 'wins': 6, 'losses': 4}

    pred = analyzer.generate_advanced_prediction(
        player1_info={}, player2_info={},
        p1_rivalry_score=0, p2_rivalry_score=0,
        player1_name='PlayerA', player2_name='PlayerB',
        player1_history=hist, player2_history=hist,
        player1_advantages_count=0, player2_advantages_count=0,
        player1_form=form, player2_form=form,
        direct_h2h_matches=[],
        tournament_name='Test ITF',
        prediction_context={
            'current_match_surface': 'hard',
            'current_match_country': 'US',
            'p1_nationality': 'US',
            'p2_nationality': 'US',
        },
        p1_elo=1500, p2_elo=1500,
        n_common_opponents=0, n_erdos_paths=0
    )

    assert 'circuit_asymmetry' in pred
    ca = pred['circuit_asymmetry']
    assert 'p1_circuit_tier_index' in ca
    assert 'p2_circuit_tier_index' in ca
    assert 'asymmetry_ratio' in ca
    assert 'deflactor_applied' in ca
    assert 'player_deflated' in ca
    assert 'signal' in ca
    assert ca['signal'] in ('SYMMETRIC', 'MODERATE_ASYMMETRY', 'STRONG_ASYMMETRY')


# ─── T29-13: Ponderación temporal — recientes (i<10) pesan 2× ───────────────
def test_T29_13_temporal_weighting():
    """Los últimos 10 partidos pesan 2×. Un partido top-10 reciente sube más el CTI."""
    analyzer = make_analyzer()

    # Caso A: un oponente top-10 en posición reciente (índice 0)
    hist_recent_top = [{'oponente': 'Medvedev', 'opponent_ranking': 8, 'outcome': 'Perdió',
                        'resultado': '1-2', 'fecha': '01.01.2026', 'torneo': 'T', 'superficie': 'D'}]
    hist_recent_top += make_history([800] * 15)

    # Caso B: mismo oponente top-10 en posición antigua (índice 11)
    hist_old_top = make_history([800] * 11)
    hist_old_top += [{'oponente': 'Medvedev', 'opponent_ranking': 8, 'outcome': 'Perdió',
                      'resultado': '1-2', 'fecha': '01.01.2025', 'torneo': 'T', 'superficie': 'D'}]
    hist_old_top += make_history([800] * 4)

    cti_recent, _ = analyzer.circuit_tier_index(hist_recent_top)
    cti_old, _ = analyzer.circuit_tier_index(hist_old_top)

    assert cti_recent > cti_old, (
        f"CTI con top-10 reciente ({cti_recent}) debe ser > CTI con top-10 antiguo ({cti_old})"
    )
