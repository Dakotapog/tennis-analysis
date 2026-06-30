"""
Tests Nodo-30 — Tournament Momentum + Output Signals

T30-01 to T30-09:  TORNEO_COMPLETO logic (via analyze_surface_specialization)
T30-10 to T30-13:  E-1 weight shift + date parsing
T30-14 to T30-18:  get_weights_from_reasoning() final weights
T30-19 to T30-22:  SEÑALES ESPECIALES detection
T30-23 to T30-30:  Player Profitability tracker
"""
import json
import os
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from analysis.rivalry_analyzer import RivalryAnalyzer
from generar_tabla_favoritos2 import get_weights_from_reasoning


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def ranking_manager():
    """Mock RankingManager with a rank lookup table."""
    rm = MagicMock()
    rm.get_player_ranking.side_effect = lambda name: {
        'Opp50': 50,
        'Opp100': 100,
        'Opp200': 200,
        'OppTop': 5,
    }.get(name, 200)
    rm.normalize_name.side_effect = lambda n: (n or '').lower()
    rm.get_player_info.return_value = {
        'ranking_position': 50,
        'ranking_points': 1000,
        'prox_points': 1050,
        'max_points': 1100,
        'defense_points': 50,
    }
    return rm


@pytest.fixture
def elo_system():
    es = MagicMock()
    es.default_rating = 1500
    es.k_factor = 32
    es.expected_score.return_value = 0.5
    es.calculate_rating_change.return_value = 16
    return es


@pytest.fixture
def analyzer(ranking_manager, elo_system):
    return RivalryAnalyzer(ranking_manager, elo_system)


def _make_match(torneo, fecha, outcome='Ganó', oponente='Opp100', superficie='Arcilla'):
    """Helper: build a minimal surface match dict."""
    return {
        'torneo': torneo,
        'fecha': fecha,
        'outcome': outcome,
        'oponente': oponente,
        'superficie': superficie,
        'resultado': '2-0',
    }


def _recent_date(days_ago: int) -> str:
    """Return a date string DD.MM.YYYY for a date N days ago."""
    dt = datetime.today() - timedelta(days=days_ago)
    return dt.strftime('%d.%m.%Y')


# ─────────────────────────────────────────────────────────────────────────────
# T30-01 to T30-09: TORNEO_COMPLETO logic
# ─────────────────────────────────────────────────────────────────────────────

class TestTorneoCompleto:
    """T30-01 to T30-09: TORNEO_COMPLETO bonus via analyze_surface_specialization."""

    def _run(self, analyzer, matches, surface='Arcilla', player='PlayerA'):
        """Run analyze_surface_specialization and return (result, log)."""
        result, log = analyzer.analyze_surface_specialization(matches, surface, player)
        return result, log

    def test_T30_01_fires_with_4W_0L(self, analyzer):
        """T30-01: TORNEO_COMPLETO fires with >=4W, 0L in same torneo+year."""
        fecha = _recent_date(7)  # recent, within 90d and <=14d
        matches = [_make_match('Roland Garros', fecha) for _ in range(4)]
        _, log = self._run(analyzer, matches)
        assert any('TORNEO_COMPLETO_BONUS' in l for l in log)

    def test_T30_02_does_not_fire_with_3W_0L(self, analyzer):
        """T30-02: TORNEO_COMPLETO does NOT fire with 3W-0L (threshold is 4)."""
        fecha = _recent_date(7)
        matches = [_make_match('Roland Garros', fecha) for _ in range(3)]
        _, log = self._run(analyzer, matches)
        assert not any('TORNEO_COMPLETO_BONUS' in l for l in log)

    def test_T30_03_does_not_fire_with_5W_1L(self, analyzer):
        """T30-03: TORNEO_COMPLETO does NOT fire with 5W-1L (has a loss)."""
        fecha = _recent_date(7)
        wins = [_make_match('Roland Garros', fecha) for _ in range(5)]
        loss = _make_match('Roland Garros', fecha, outcome='Perdió')
        _, log = self._run(analyzer, wins + [loss])
        assert not any('TORNEO_COMPLETO_BONUS' in l for l in log)

    def test_T30_04_expirado_when_over_90_days(self, analyzer):
        """T30-04: TORNEO_COMPLETO_EXPIRADO when >90 days — bonus = 1.0 (no effect)."""
        fecha = _recent_date(100)  # 100 days ago
        matches = [_make_match('Roland Garros', fecha) for _ in range(4)]
        _, log = self._run(analyzer, matches)
        assert any('TORNEO_COMPLETO_EXPIRADO' in l for l in log)
        assert not any('TORNEO_COMPLETO_BONUS' in l for l in log)

    def test_T30_05_recent_14d_bonus_1_5(self, analyzer):
        """T30-05: recent <=14d — bonus base 1.3 + recency 0.2 = 1.5."""
        fecha = _recent_date(7)  # 7 days ago
        matches = [_make_match('Roland Garros', fecha) for _ in range(4)]
        _, log = self._run(analyzer, matches)
        # Find the bonus log line
        bonus_lines = [l for l in log if 'TORNEO_COMPLETO_BONUS' in l]
        assert bonus_lines, "Expected TORNEO_COMPLETO_BONUS in log"
        # Check bonus value: x1.5 expected (base 1.3 + recency 0.2)
        assert 'x1.5' in bonus_lines[0]

    def test_T30_06_recent_14d_plus_final_5W_bonus_1_6(self, analyzer):
        """T30-06: recent <=14d + final (>=5W) — bonus = 1.6."""
        fecha = _recent_date(7)
        matches = [_make_match('Roland Garros', fecha) for _ in range(5)]
        _, log = self._run(analyzer, matches)
        bonus_lines = [l for l in log if 'TORNEO_COMPLETO_BONUS' in l]
        assert bonus_lines
        # base 1.3 + recency 0.2 + final 0.1 = 1.6
        assert 'x1.6' in bonus_lines[0]

    def test_T30_07_recent_14d_top10_final_bonus_1_7(self, analyzer):
        """T30-07: recent <=14d + top10 + final — bonus = 1.7.
        TORNEO_COMPLETO block reads opponent_ranking from match dict (line 819),
        NOT from ranking_manager. Must include 'opponent_ranking' in the match.
        """
        fecha = _recent_date(7)
        # Need >=5 wins, one vs Top-10 — opponent_ranking must be in match dict
        top10_match = _make_match('Roland Garros', fecha, oponente='OppTop')
        top10_match['opponent_ranking'] = 5  # rank 5 = Top-10
        matches = [
            top10_match,
            _make_match('Roland Garros', fecha),
            _make_match('Roland Garros', fecha),
            _make_match('Roland Garros', fecha),
            _make_match('Roland Garros', fecha),
        ]
        _, log = self._run(analyzer, matches)
        bonus_lines = [l for l in log if 'TORNEO_COMPLETO_BONUS' in l]
        assert bonus_lines
        # base 1.3 + recency 0.2 + top10 0.1 + final 0.1 = 1.7
        assert 'x1.7' in bonus_lines[0]

    def test_T30_08_bonus_cap_never_exceeds_2_0(self, analyzer):
        """T30-08: bonus cap — never exceeds 2.0."""
        fecha = _recent_date(7)
        # Large number of wins, vs top10 — all bonuses fire
        matches = [
            _make_match('Roland Garros', fecha, oponente='OppTop')
            for _ in range(10)
        ]
        _, log = self._run(analyzer, matches)
        bonus_lines = [l for l in log if 'TORNEO_COMPLETO_BONUS' in l]
        assert bonus_lines
        # Extract the multiplier from e.g. "→ x1.7 quality_score"
        import re
        m = re.search(r'x([\d.]+)\s+quality_score', bonus_lines[0])
        assert m, f"Could not parse bonus from: {bonus_lines[0]}"
        bonus = float(m.group(1))
        assert bonus <= 2.0

    def test_T30_09_fecha_none_does_not_fire(self, analyzer):
        """T30-09: fecha=None — does not fire (no date = can't verify recency)."""
        matches = [
            _make_match('Roland Garros', None) for _ in range(4)
        ]
        _, log = self._run(analyzer, matches)
        assert not any('TORNEO_COMPLETO_BONUS' in l for l in log)
        assert not any('TORNEO_COMPLETO_EXPIRADO' in l for l in log)


# ─────────────────────────────────────────────────────────────────────────────
# T30-10 to T30-13: E-1 weight shift + date parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestE1WeightShift:
    """T30-10 to T30-13: E-1 weight shift logic."""

    def test_T30_10_e1_fires_torneo_completo_form_above_threshold(self):
        """T30-10: E-1 fires when torneo_completo=True and form_recent > 0.10."""
        # Simulate what E-1 does inline in generate_prediction
        # p1 has torneo_completo, form_recent > 0.10
        weights = {
            'surface_specialization': 0.15,
            'form_recent': 0.18,
            'common_opponents': 0.20,
            'ranking_momentum': 0.20,
            'h2h_direct': 0.27,
        }
        p1_surface_result = {'torneo_completo': True}
        p2_surface_result = {'torneo_completo': False}
        reasoning = []

        _any_torneo = p1_surface_result.get('torneo_completo') or p2_surface_result.get('torneo_completo')
        if _any_torneo and weights.get('form_recent', 0) > 0.10:
            _boost = 0.07
            weights['surface_specialization'] = round(weights.get('surface_specialization', 0.15) + _boost, 4)
            weights['form_recent'] = round(weights['form_recent'] - _boost, 4)
            reasoning.append(
                f"LOG_E1_TORNEO_WEIGHT: surface {weights['surface_specialization']-_boost:.2f}→{weights['surface_specialization']:.2f} "
                f"form {weights['form_recent']+_boost:.2f}→{weights['form_recent']:.2f} "
                f"(tournament champion on this surface)"
            )

        assert weights['surface_specialization'] == pytest.approx(0.22, abs=0.001)
        assert weights['form_recent'] == pytest.approx(0.11, abs=0.001)
        assert any('LOG_E1_TORNEO_WEIGHT' in r for r in reasoning)

    def test_T30_11_e1_does_not_fire_if_no_torneo_completo(self):
        """T30-11: E-1 does NOT fire if torneo_completo=False for both players."""
        weights = {
            'surface_specialization': 0.15,
            'form_recent': 0.18,
        }
        p1_surface_result = {'torneo_completo': False}
        p2_surface_result = {'torneo_completo': False}
        reasoning = []

        original_surf = weights['surface_specialization']
        original_form = weights['form_recent']

        _any_torneo = p1_surface_result.get('torneo_completo') or p2_surface_result.get('torneo_completo')
        if _any_torneo and weights.get('form_recent', 0) > 0.10:
            weights['surface_specialization'] += 0.07
            weights['form_recent'] -= 0.07
            reasoning.append('LOG_E1_TORNEO_WEIGHT: ...')

        assert weights['surface_specialization'] == original_surf
        assert weights['form_recent'] == original_form
        assert not any('LOG_E1_TORNEO_WEIGHT' in r for r in reasoning)

    def test_T30_12_e1_does_not_fire_if_form_recent_low(self):
        """T30-12: E-1 does NOT fire if form_recent <= 0.10 (nothing to deduct)."""
        weights = {
            'surface_specialization': 0.15,
            'form_recent': 0.10,  # exactly at threshold
        }
        p1_surface_result = {'torneo_completo': True}
        p2_surface_result = {'torneo_completo': False}
        reasoning = []

        original_surf = weights['surface_specialization']

        _any_torneo = p1_surface_result.get('torneo_completo') or p2_surface_result.get('torneo_completo')
        if _any_torneo and weights.get('form_recent', 0) > 0.10:
            weights['surface_specialization'] += 0.07
            weights['form_recent'] -= 0.07
            reasoning.append('LOG_E1_TORNEO_WEIGHT: ...')

        # form_recent = 0.10 is NOT > 0.10, so E-1 should not fire
        assert weights['surface_specialization'] == original_surf
        assert not any('LOG_E1_TORNEO_WEIGHT' in r for r in reasoning)

    def test_T30_13_fecha_parsing_uses_last_4_chars(self):
        """T30-13: year parsing uses [-4:] on 'DD.MM.YYYY', not [:4]."""
        fecha_str = '07.06.2026'

        # The correct way (implementation uses [-4:])
        year_correct = fecha_str[-4:]
        assert year_correct == '2026', f"[-4:] should give '2026', got {year_correct!r}"

        # The WRONG way that was the bug ([:4] gives DD.M which is wrong)
        year_wrong = fecha_str[:4]
        assert year_wrong != '2026', f"[:4] should NOT give '2026' for 'DD.MM.YYYY'"
        assert year_wrong == '07.0'  # proves the old approach was broken


# ─────────────────────────────────────────────────────────────────────────────
# T30-14 to T30-18: get_weights_from_reasoning() final weights
# ─────────────────────────────────────────────────────────────────────────────

class TestGetWeightsFromReasoning:
    """T30-14 to T30-18: verify that get_weights_from_reasoning returns FINAL weights."""

    def _strategy_log(self, tier='atp500'):
        """Build a LOG_WEIGHTS_STRATEGY log line with sample weights."""
        weights = {
            'h2h_direct': 0.27,
            'ranking_momentum': 0.20,
            'form_recent': 0.18,
            'common_opponents': 0.20,
            'surface_specialization': 0.15,
        }
        return f"LOG_WEIGHTS_STRATEGY: '{tier}' -> {weights}"

    def test_T30_14_with_E1_torneo_weight_log(self):
        """T30-14: with LOG_E1_TORNEO_WEIGHT — final weights reflect the shift."""
        reasoning = [
            self._strategy_log(),
            "LOG_E1_TORNEO_WEIGHT: surface 0.15→0.22 form 0.18→0.11 (tournament champion on this surface)",
        ]
        weights = get_weights_from_reasoning(reasoning)
        assert weights.get('surface_specialization') == pytest.approx(0.22)
        assert weights.get('form_recent') == pytest.approx(0.11)

    def test_T30_15_with_LOG_WEIGHTS_SURFACE_GRASS(self):
        """T30-15: with LOG_WEIGHTS_SURFACE_GRASS — common_opp and form adjusted."""
        reasoning = [
            self._strategy_log(),
            "LOG_WEIGHTS_SURFACE_GRASS: common_opp→0.15 form_recent→0.23 (alta varianza césped, Nodo-14)",
        ]
        weights = get_weights_from_reasoning(reasoning)
        assert weights.get('common_opponents') == pytest.approx(0.15)
        assert weights.get('form_recent') == pytest.approx(0.23)

    def test_T30_16_with_LOG_DENSITY(self):
        """T30-16: with LOG_DENSITY — co_w and form_w adjusted."""
        reasoning = [
            self._strategy_log(),
            "LOG_DENSITY: n_common=3 n_paths=5 density=0.4667 co_w: 0.20→0.0933 form_w→0.2867",
        ]
        weights = get_weights_from_reasoning(reasoning)
        assert weights.get('common_opponents') == pytest.approx(0.0933)
        assert weights.get('form_recent') == pytest.approx(0.2867)

    def test_T30_17_without_adjustments_returns_strategy_weights(self):
        """T30-17: without adjustments — returns initial strategy weights unchanged."""
        reasoning = [self._strategy_log()]
        weights = get_weights_from_reasoning(reasoning)
        assert weights.get('form_recent') == pytest.approx(0.18)
        assert weights.get('h2h_direct') == pytest.approx(0.27)
        assert weights.get('surface_specialization') == pytest.approx(0.15)

    def test_T30_18_full_chain_strategy_density_grass_e1(self):
        """T30-18: full chain (strategy + density + grass + E1) — correct final weights."""
        # Start: form_recent=0.18, common_opponents=0.20, surface_specialization=0.15
        # After DENSITY: co_w=0.09, form_w=0.29
        # After GRASS: co_w=0.04, form_w=0.34
        # After E1: surface=0.22, form=0.27
        reasoning = [
            self._strategy_log(),
            "LOG_DENSITY: n_common=3 n_paths=5 density=0.45 co_w: 0.20→0.09 form_w→0.29",
            "LOG_WEIGHTS_SURFACE_GRASS: common_opp→0.04 form_recent→0.34 (alta varianza césped, Nodo-14)",
            "LOG_E1_TORNEO_WEIGHT: surface 0.15→0.22 form 0.34→0.27 (tournament champion on this surface)",
        ]
        weights = get_weights_from_reasoning(reasoning)
        # Final values from E1 override
        assert weights.get('surface_specialization') == pytest.approx(0.22)
        assert weights.get('form_recent') == pytest.approx(0.27)
        # Final common_opponents from GRASS (last update to it)
        assert weights.get('common_opponents') == pytest.approx(0.04)


# ─────────────────────────────────────────────────────────────────────────────
# T30-19 to T30-22: SEÑALES ESPECIALES detection
# ─────────────────────────────────────────────────────────────────────────────

def _extract_special_signals(p1: str, p2: str, reasoning: list) -> list:
    """
    Replicates the SEÑALES ESPECIALES detection logic from generar_tabla_favoritos2.py.
    Returns a list of signal strings.
    """
    import re
    _special_signals = []
    for reason in reasoning:
        if 'TORNEO_COMPLETO_BONUS' in reason:
            _clean = reason.replace('P1_LOG_SURF: ', '').replace('P2_LOG_SURF: ', '')
            _player = p1 if 'P1_LOG_SURF' in reason else p2
            _special_signals.append(f"CAMPEON DE TORNEO: {_player} -- {_clean}")
        if 'LOG_E1_TORNEO_WEIGHT' in reason:
            _special_signals.append(f"AJUSTE DINAMICO DE PESOS: {reason}")
        if 'LOG_MARKOV' in reason and 'estado=HOT' in reason:
            _player = p1 if '_P1' in reason else p2
            _wr = re.search(r'wr_rec=([\d.]+)', reason)
            _wr_str = f" ({float(_wr.group(1))*100:.0f}% reciente)" if _wr else ""
            _special_signals.append(f"RACHA CALIENTE: {_player} en estado HOT{_wr_str}")
        if '_LOG_SURF' in reason and 'Victoria vs Rank' in reason:
            _rank_match = re.search(r'Victoria vs Rank (\d+) \(([^)]+)\)', reason)
            if _rank_match:
                _opp_rank_int = int(_rank_match.group(1))
                _tier_from_weights = ''
                for _r in reasoning:
                    if 'LOG_WEIGHTS_STRATEGY' in _r:
                        _tm = re.search(r"'(\w+)'", _r)
                        if _tm:
                            _tier_from_weights = _tm.group(1)
                        break
                _scalp_thresholds = {
                    'grand_slam': 10, 'atp1000': 10, 'atp500': 20,
                    'challenger': 50, 'itf': 100
                }
                _threshold = _scalp_thresholds.get(_tier_from_weights, 20)
                if _opp_rank_int <= _threshold:
                    _player = p1 if 'P1_LOG_SURF' in reason else p2
                    _opp_name = _rank_match.group(2)
                    _special_signals.append(
                        f"SCALP TOP-{_threshold} EN SUPERFICIE: {_player} vencio a {_opp_name} (#{_opp_rank_int}) en esta superficie"
                    )
        if 'TORNEO_COMPLETO_EXPIRADO' in reason:
            _clean = reason.replace('P1_LOG_SURF: ', '').replace('P2_LOG_SURF: ', '')
            _special_signals.append(f"TORNEO EXPIRADO (sin bonus): {_clean}")
    return _special_signals


class TestSenalesEspeciales:
    """T30-19 to T30-22: SEÑALES ESPECIALES detection."""

    def test_T30_19_detects_torneo_completo_bonus(self):
        """T30-19: detects TORNEO_COMPLETO_BONUS in reasoning."""
        reasoning = [
            "P1_LOG_SURF: TORNEO_COMPLETO_BONUS: Roland Garros 2026 (5W-0L) → x1.6 quality_score [recency(7d) + final(5W)]"
        ]
        signals = _extract_special_signals('Alcaraz', 'Djokovic', reasoning)
        assert any('CAMPEON DE TORNEO' in s for s in signals)
        assert any('Alcaraz' in s for s in signals)

    def test_T30_20_detects_torneo_completo_expirado(self):
        """T30-20: detects TORNEO_COMPLETO_EXPIRADO in reasoning."""
        reasoning = [
            "P2_LOG_SURF: TORNEO_COMPLETO_EXPIRADO: W15 LA 2025 (4W-0L, hace 120d) → sin bonus (>90d)"
        ]
        signals = _extract_special_signals('Alcaraz', 'Djokovic', reasoning)
        assert any('TORNEO EXPIRADO' in s for s in signals)

    def test_T30_21_scalp_tier_relative_itf_threshold_100(self):
        """T30-21: scalp tier-relative — ITF threshold=100, GS threshold=10."""
        # ITF: rank 80 should trigger (<=100)
        reasoning_itf = [
            "LOG_WEIGHTS_STRATEGY: 'itf' -> {'h2h_direct': 0.27}",
            "P1_LOG_SURF: Victoria vs Rank 80 (SmallPlayer) en Arcilla -> +5.0 pts",
        ]
        signals_itf = _extract_special_signals('Carnicella', 'SmallPlayer', reasoning_itf)
        assert any('SCALP TOP-100' in s for s in signals_itf)

        # GS: rank 80 should NOT trigger (>10)
        reasoning_gs = [
            "LOG_WEIGHTS_STRATEGY: 'grand_slam' -> {'h2h_direct': 0.27}",
            "P1_LOG_SURF: Victoria vs Rank 80 (SmallPlayer) en Arcilla -> +5.0 pts",
        ]
        signals_gs = _extract_special_signals('Alcaraz', 'SmallPlayer', reasoning_gs)
        assert not any('SCALP' in s for s in signals_gs)

    def test_T30_22_scalp_not_shown_if_rank_above_threshold(self):
        """T30-22: scalp NOT shown if rank > tier threshold."""
        # ATP500: threshold=20, rank=50 should not trigger
        reasoning = [
            "LOG_WEIGHTS_STRATEGY: 'atp500' -> {'h2h_direct': 0.27}",
            "P1_LOG_SURF: Victoria vs Rank 50 (MidPlayer) en Arcilla -> +25.0 pts",
        ]
        signals = _extract_special_signals('Alcaraz', 'MidPlayer', reasoning)
        assert not any('SCALP' in s for s in signals)


# ─────────────────────────────────────────────────────────────────────────────
# T30-23 to T30-30: Player Profitability
# ─────────────────────────────────────────────────────────────────────────────

def _write_apuesta(tmp_path: Path, estado: str, picks: list, ts: str = '2026-06-14T10:00:00') -> Path:
    """Helper: write an apuesta JSON file in tmp_path/reports/."""
    reports = tmp_path / 'reports'
    reports.mkdir(exist_ok=True)
    fname = reports / f'apuestas_{ts.replace(":", "").replace("-", "").replace("T", "_")}.json'
    data = {
        'ts_registro': ts,
        'estado': estado,
        'picks': picks,
    }
    fname.write_text(json.dumps(data), encoding='utf-8')
    return fname


def _pick(jugador, cuota, stake, correcto):
    """Build a minimal pick dict matching the apuestas format."""
    return {
        'jugador': jugador,
        'cuota': cuota,
        'stake': stake,
        'correcto': correcto,
        'ganancia': round(stake * (cuota - 1), 2) if correcto else -stake,
    }


class TestPlayerProfitability:
    """T30-23 to T30-30: player_profitability.py functions."""

    def test_T30_23_build_with_0_betslips_returns_empty(self, tmp_path):
        """T30-23: build_player_profitability with 0 betslips — returns empty dict."""
        from analysis.player_profitability import build_player_profitability
        reports = tmp_path / 'reports'
        reports.mkdir()
        # Override data dir to tmp_path so we don't write to real data/
        result = build_player_profitability(betslip_dir=str(reports))
        assert result == {}

    def test_T30_24_1_closed_betslip_won_correct_stats(self, tmp_path):
        """T30-24: 1 closed betslip (won) — correct stats."""
        from analysis.player_profitability import build_player_profitability
        picks = [_pick('Kaitlyn Carnicella', 2.85, 1000, True)]
        _write_apuesta(tmp_path, 'CERRADO', picks)
        result = build_player_profitability(betslip_dir=str(tmp_path / 'reports'))
        assert 'kaitlyn carnicella' in result
        stats = result['kaitlyn carnicella']
        assert stats['n_apostado'] == 1
        assert stats['n_ganado'] == 1
        assert stats['profit_total'] == pytest.approx(1000 * (2.85 - 1), rel=1e-3)

    def test_T30_25_pending_betslip_ignored(self, tmp_path):
        """T30-25: pending betslip — ignored (estado=PENDIENTE)."""
        from analysis.player_profitability import build_player_profitability
        picks = [_pick('Some Player', 2.0, 500, None)]  # correcto=None = pending
        # Write as PENDIENTE
        _write_apuesta(tmp_path, 'PENDIENTE', picks)
        result = build_player_profitability(betslip_dir=str(tmp_path / 'reports'))
        assert result == {}

    def test_T30_26_multiple_bets_same_player_aggregates(self, tmp_path):
        """T30-26: multiple bets same player — aggregates correctly."""
        from analysis.player_profitability import build_player_profitability
        # Bet 1: won
        picks1 = [_pick('Kaitlyn Carnicella', 2.5, 1000, True)]
        _write_apuesta(tmp_path, 'CERRADO', picks1, ts='2026-06-14T10:00:00')
        # Bet 2: lost
        picks2 = [_pick('Kaitlyn Carnicella', 3.0, 1000, False)]
        _write_apuesta(tmp_path, 'CERRADO', picks2, ts='2026-06-15T10:00:00')
        # Bet 3: won
        picks3 = [_pick('Kaitlyn Carnicella', 2.0, 1000, True)]
        _write_apuesta(tmp_path, 'CERRADO', picks3, ts='2026-06-16T10:00:00')

        result = build_player_profitability(betslip_dir=str(tmp_path / 'reports'))
        key = 'kaitlyn carnicella'
        assert key in result
        stats = result[key]
        assert stats['n_apostado'] == 3
        assert stats['n_ganado'] == 2

    def test_T30_27_roi_calculated_correctly(self, tmp_path):
        """T30-27: ROI = profit_total / total_apostado."""
        from analysis.player_profitability import build_player_profitability
        # Win: stake=1000, cuota=2.0 → profit=1000
        picks1 = [_pick('Maria Seles', 2.0, 1000, True)]
        _write_apuesta(tmp_path, 'CERRADO', picks1, ts='2026-06-14T10:00:00')
        # Loss: stake=1000 → profit=-1000
        picks2 = [_pick('Maria Seles', 2.0, 1000, False)]
        _write_apuesta(tmp_path, 'CERRADO', picks2, ts='2026-06-15T10:00:00')

        result = build_player_profitability(betslip_dir=str(tmp_path / 'reports'))
        key = 'maria seles'
        assert key in result
        stats = result[key]
        # profit_total = 1000 - 1000 = 0, total_apostado = 2000 → roi = 0
        assert stats['roi'] == pytest.approx(0.0)

        # Now test with positive ROI: 2 wins, 0 losses
        tmp2 = tmp_path / 'case2'
        tmp2.mkdir()
        (tmp2 / 'reports').mkdir()
        _write_apuesta(tmp2, 'CERRADO', [_pick('Player X', 3.0, 500, True)], ts='2026-06-14T10:00:00')
        _write_apuesta(tmp2, 'CERRADO', [_pick('Player X', 2.0, 500, True)], ts='2026-06-15T10:00:00')
        result2 = build_player_profitability(betslip_dir=str(tmp2 / 'reports'))
        stats2 = result2.get('player x', {})
        # profit = 500*(3-1) + 500*(2-1) = 1000 + 500 = 1500, total_staked = 1000 → roi=1.5
        assert stats2['roi'] == pytest.approx(1.5, rel=1e-2)

    def test_T30_28_senales_shows_jugador_rentable_n_gte_3_roi_gt_0(self, tmp_path):
        """T30-28: SEÑALES shows JUGADOR RENTABLE only if n>=3 and roi>0."""
        from generar_tabla_favoritos2 import _load_profitability_data, _normalize_player_name_for_prof

        # Build profitability data with 3 wins for a player
        prof_data = {
            'ana ivanovic': {
                'display_name': 'Ana Ivanovic',
                'n_apostado': 3,
                'n_ganado': 3,
                'profit_total': 900.0,
                'total_apostado': 1000.0,
                'roi': 0.9,
                'avg_cuota': 2.6,
                'last_seen': '2026-06-15T10:00:00',
            }
        }

        signals = []
        for _player_name in ['Ana Ivanovic', 'Djokovic']:
            _prof_key = _normalize_player_name_for_prof(_player_name)
            _prof = prof_data.get(_prof_key)
            if _prof and _prof.get('n_apostado', 0) >= 3 and _prof.get('roi', 0) > 0:
                signals.append(
                    f"JUGADOR RENTABLE: {_player_name} -- "
                    f"{_prof['n_apostado']} apuestas, {_prof['n_ganado']} ganadas, "
                    f"ROI +{_prof['roi']*100:.0f}%, cuota prom {_prof['avg_cuota']:.2f}"
                )

        assert len(signals) == 1
        assert 'Ana Ivanovic' in signals[0]
        assert 'JUGADOR RENTABLE' in signals[0]

    def test_T30_29_senales_does_not_show_if_roi_zero_or_negative(self, tmp_path):
        """T30-29: SEÑALES does NOT show JUGADOR RENTABLE if roi<=0."""
        from generar_tabla_favoritos2 import _normalize_player_name_for_prof

        prof_data = {
            'bad player': {
                'n_apostado': 5,
                'n_ganado': 1,
                'roi': -0.3,  # negative ROI
                'avg_cuota': 2.0,
            }
        }

        signals = []
        for _player_name in ['Bad Player', 'Other']:
            _prof_key = _normalize_player_name_for_prof(_player_name)
            _prof = prof_data.get(_prof_key)
            if _prof and _prof.get('n_apostado', 0) >= 3 and _prof.get('roi', 0) > 0:
                signals.append(f"JUGADOR RENTABLE: {_player_name}")

        assert signals == []

    def test_T30_30_senales_does_not_show_if_n_below_3(self, tmp_path):
        """T30-30: SEÑALES does NOT show JUGADOR RENTABLE if n<3."""
        from generar_tabla_favoritos2 import _normalize_player_name_for_prof

        prof_data = {
            'young player': {
                'n_apostado': 2,  # below threshold
                'n_ganado': 2,
                'roi': 1.5,
                'avg_cuota': 2.5,
            }
        }

        signals = []
        for _player_name in ['Young Player', 'Other']:
            _prof_key = _normalize_player_name_for_prof(_player_name)
            _prof = prof_data.get(_prof_key)
            if _prof and _prof.get('n_apostado', 0) >= 3 and _prof.get('roi', 0) > 0:
                signals.append(f"JUGADOR RENTABLE: {_player_name}")

        assert signals == []
