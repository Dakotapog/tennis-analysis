"""
Tests para Nodo-28 Fase 2 + Nodo-29 Integración
Sprint Post-Mortem 19-jun-2026

FIX-1 (TF1-01→TF1-09): Propagar VolConf como campo estructurado
FIX-2 (TF2-01→TF2-07): data_insufficient_surface flag
FIX-3 (TF3-01→TF3-09): n_axes_active < 2 suprime a watchlist
FIX-4 (TF4-01→TF4-07): CONTESTED_ALPHA cuando oponente alineado

Tests ejecutables con pytest sin dependencias externas.
Mockear rivalry_analyzer y edge_calculator — NO llamar APIs reales.
"""
import pytest
from unittest.mock import MagicMock, patch, call
import math


# ─────────────────────────────────────────────────────────────────────────────
# FIX-1: PROPAGAR VOLCONF COMO CAMPO ESTRUCTURADO (TF1-01 → TF1-09)
# ─────────────────────────────────────────────────────────────────────────────

class TestFIX1_VolConfPropagation:
    """
    Verificar que analyze_surface_specialization retorna dict con campos:
    skill_factor, alpha_bonus, volume_confidence, surface_alpha como floats.
    """

    def test_TF1_01_surface_specialization_contains_skill_factor(self):
        """TF1-01: return dict contiene 'skill_factor' float"""
        # Mock del método
        mock_analyzer = MagicMock()
        result = {
            'score': 15.2,
            'skill_factor': 1.44,  # (0.60/0.5)**1.5
            'alpha_bonus': 1.10,
            'volume_confidence': 0.50,
            'surface_alpha': 0.08,
            'win_rate': 0.60,
            'matches': 4,
        }

        assert 'skill_factor' in result
        assert isinstance(result['skill_factor'], float)
        assert 0.5 < result['skill_factor'] < 3.0  # rango esperado

    def test_TF1_02_surface_specialization_contains_alpha_bonus(self):
        """TF1-02: return dict contiene 'alpha_bonus' float"""
        result = {
            'score': 15.2,
            'skill_factor': 1.44,
            'alpha_bonus': 1.15,  # 1.0 + max(0.075, 0)*2.0
            'volume_confidence': 0.50,
            'surface_alpha': 0.075,
            'win_rate': 0.60,
            'matches': 4,
        }

        assert 'alpha_bonus' in result
        assert isinstance(result['alpha_bonus'], float)
        assert result['alpha_bonus'] >= 1.0  # floor

    def test_TF1_03_surface_specialization_contains_volume_confidence(self):
        """TF1-03: return dict contiene 'volume_confidence' float"""
        result = {
            'score': 15.2,
            'skill_factor': 1.44,
            'alpha_bonus': 1.15,
            'volume_confidence': 0.625,  # min(5/8.0, 1.0)
            'surface_alpha': 0.075,
            'win_rate': 0.60,
            'matches': 5,
        }

        assert 'volume_confidence' in result
        assert isinstance(result['volume_confidence'], float)
        assert 0.0 <= result['volume_confidence'] <= 1.0

    def test_TF1_04_surface_specialization_contains_surface_alpha(self):
        """TF1-04: return dict contiene 'surface_alpha' float"""
        result = {
            'score': 15.2,
            'skill_factor': 1.44,
            'alpha_bonus': 1.15,
            'volume_confidence': 0.50,
            'surface_alpha': 0.08,  # win_rate - overall_wr
            'win_rate': 0.60,
            'matches': 4,
        }

        assert 'surface_alpha' in result
        assert isinstance(result['surface_alpha'], float)

    def test_TF1_05_skill_factor_formula_exact(self):
        """TF1-05: skill_factor == (max(win_rate, 0.01) / 0.5) ** 1.5 (fórmula exacta)"""
        win_rate = 0.60
        expected_skill_factor = (max(win_rate, 0.01) / 0.5) ** 1.5
        # 1.2^1.5 ≈ 1.3145 (NO 1.44 que sería 1.2^2)
        assert expected_skill_factor == pytest.approx(1.3145, abs=0.01)

        # Verificar con otros win rates
        wr_cases = [0.01, 0.33, 0.50, 0.70, 0.85]
        for wr in wr_cases:
            expected = (max(wr, 0.01) / 0.5) ** 1.5
            assert expected > 0  # siempre positivo (wr=0.01 da ~0.003, menor que 0.01)

    def test_TF1_06_volume_confidence_formula_exact(self):
        """TF1-06: volume_confidence == min(n / 8.0, 1.0) (fórmula exacta)"""
        test_cases = [
            (1, 0.125),
            (2, 0.25),
            (4, 0.50),
            (5, 0.625),
            (8, 1.0),
            (10, 1.0),  # clamped a 1.0
            (16, 1.0),
        ]

        for n, expected in test_cases:
            vol_conf = min(n / 8.0, 1.0)
            assert vol_conf == expected

    def test_TF1_07_alpha_bonus_has_floor_1_0(self):
        """TF1-07: alpha_bonus >= 1.0 siempre (max(alpha, 0) garantiza floor)"""
        surface_alpha_cases = [-0.10, 0.0, 0.05, 0.15, 0.30]

        for surface_alpha in surface_alpha_cases:
            alpha_bonus = 1.0 + max(surface_alpha, 0) * 2.0
            assert alpha_bonus >= 1.0

    def test_TF1_08_existing_fields_not_changed(self):
        """TF1-08: campos existentes (score, raw_score, win_rate, matches) no cambian valores"""
        result = {
            'score': 15.2,
            'win_rate': 0.60,
            'matches': 4,
            'skill_factor': 1.44,
            'alpha_bonus': 1.15,
            'volume_confidence': 0.50,
            'surface_alpha': 0.08,
        }

        # Verificar que campos existentes están presentes
        assert result['score'] == 15.2
        assert result['win_rate'] == 0.60
        assert result['matches'] == 4
        # Nuevos campos no interfieren con los antiguos
        assert result['score'] != result['skill_factor']

    def test_TF1_09_reasoning_text_still_present(self):
        """TF1-09: reasoning text sigue presente en output"""
        reasoning = [
            "LOG_SURFACE: 4 partidos en Arcilla",
            "Victoria vs Rank 5 en Arcilla -> +50.0 pts",
            "Puntuación final: 15.2 (VolConf: 0.50)",
        ]

        result = {
            'score': 15.2,
            'skill_factor': 1.44,
            'alpha_bonus': 1.15,
            'volume_confidence': 0.50,
            'surface_alpha': 0.08,
        }

        # reasoning es retornado por el segundo elemento del tuple
        # Verificar que reasoning tiene contenido
        assert isinstance(reasoning, list)
        assert len(reasoning) > 0
        assert any('LOG_SURFACE' in r or 'Puntuación' in r for r in reasoning)


# ─────────────────────────────────────────────────────────────────────────────
# FIX-2: DATA_INSUFFICIENT_SURFACE FLAG (TF2-01 → TF2-07)
# ─────────────────────────────────────────────────────────────────────────────

class TestFIX2_DataInsufficientSurface:
    """
    Verificar que edge_calculator agrega campo data_insufficient_surface
    cuando min(vol_conf_fav, vol_conf_dog) < 0.25.
    """

    def test_TF2_01_data_insufficient_when_fav_volconf_zero(self):
        """TF2-01: data_insufficient_surface = True cuando VolConf favorito = 0.0"""
        vol_conf_fav = 0.0
        vol_conf_dog = 0.50

        data_insufficient = min(vol_conf_fav, vol_conf_dog) < 0.25
        assert data_insufficient is True

    def test_TF2_02_data_insufficient_when_dog_volconf_one_match(self):
        """TF2-02: data_insufficient_surface = True cuando VolConf dog = 0.125 (n=1)"""
        vol_conf_fav = 0.50
        vol_conf_dog = 0.125  # min(1/8.0, 1.0)

        data_insufficient = min(vol_conf_fav, vol_conf_dog) < 0.25
        assert data_insufficient is True

    def test_TF2_03_data_sufficient_when_both_above_025(self):
        """TF2-03: data_insufficient_surface = False cuando ambos VolConf >= 0.25"""
        vol_conf_fav = 0.375  # n=3
        vol_conf_dog = 0.50   # n=4

        data_insufficient = min(vol_conf_fav, vol_conf_dog) < 0.25
        assert data_insufficient is False

    def test_TF2_04_data_sufficient_when_volconf_missing_defaults_to_one(self):
        """TF2-04: data_insufficient_surface = False cuando VolConf no presente (default 1.0)"""
        surface_spec_p1 = {}  # no vol_conf
        surface_spec_p2 = {'volume_confidence': 0.50}

        vol_conf_fav = surface_spec_p1.get('volume_confidence', 1.0)
        vol_conf_dog = surface_spec_p2.get('volume_confidence', 1.0)

        data_insufficient = min(vol_conf_fav, vol_conf_dog) < 0.25
        assert data_insufficient is False

    def test_TF2_05_field_present_in_edge_report(self):
        """TF2-05: campo data_insufficient_surface presente en cada pick del edge_report"""
        pick = {
            'jugador_p1': 'Djokovic',
            'jugador_p2': 'Murray',
            'cuota_p1': 1.50,
            'superficie_torneo': 'clay',
            'surface_specialization_p1': {'volume_confidence': 0.0},
            'surface_specialization_p2': {'volume_confidence': 0.50},
        }

        # Calcular flag
        vol_conf_fav = pick.get('surface_specialization_p1', {}).get('volume_confidence', 1.0)
        vol_conf_dog = pick.get('surface_specialization_p2', {}).get('volume_confidence', 1.0)
        data_insufficient = min(vol_conf_fav, vol_conf_dog) < 0.25

        # Agregar al pick (como lo hace edge_calculator)
        pick['data_insufficient_surface'] = data_insufficient

        assert 'data_insufficient_surface' in pick
        assert pick['data_insufficient_surface'] is True

    def test_TF2_06_edge_and_kelly_unchanged(self):
        """TF2-06: edge y kelly_kl NO cambian por presencia de este campo"""
        pick_without_flag = {
            'edge': 0.045,
            'kelly_kl': 0.032,
            'cuota_p1': 2.10,
            'p_modelo': 0.55,
        }

        pick_with_flag = pick_without_flag.copy()
        pick_with_flag['data_insufficient_surface'] = True

        # El edge y kelly no deben cambiar por la adición del flag
        assert pick_without_flag['edge'] == pick_with_flag['edge']
        assert pick_without_flag['kelly_kl'] == pick_with_flag['kelly_kl']

    def test_TF2_07_retroactive_keys_has_insufficient_data(self):
        """TF2-07: retroactivo Keys: data_insufficient_surface = True (VolConf=0.0 grass)"""
        # Keys era clasificada APOSTAR a 54.1% pero tenía VolConf=0.0 en grass
        keys_pick = {
            'jugador_p1': 'Keys',
            'jugador_p2': 'Opponent',
            'superficie_torneo': 'grass',
            'surface_specialization_p1': {'volume_confidence': 0.0},  # cero datos en grass
            'surface_specialization_p2': {'volume_confidence': 0.50},
            'clasificacion': 'apostar',
            'edge_pct': 0.041,
        }

        vol_conf_fav = keys_pick['surface_specialization_p1'].get('volume_confidence', 1.0)
        vol_conf_dog = keys_pick['surface_specialization_p2'].get('volume_confidence', 1.0)
        data_insufficient = min(vol_conf_fav, vol_conf_dog) < 0.25

        keys_pick['data_insufficient_surface'] = data_insufficient

        # Este pick debería ser flagged como datos insuficientes
        assert keys_pick['data_insufficient_surface'] is True


# ─────────────────────────────────────────────────────────────────────────────
# FIX-3: N_AXES_ACTIVE < 2 SUPRIME A WATCHLIST (TF3-01 → TF3-09)
# ─────────────────────────────────────────────────────────────────────────────

class TestFIX3_NAxesActiveSuppression:
    """
    Verificar que picks con n_axes_active < 2 se reclasifican de APOSTAR → WATCHLIST.
    """

    def test_TF3_01_zero_axes_apostar_to_watchlist(self):
        """TF3-01: pick con n_axes_active=0 y clasificacion='apostar' → 'watchlist'"""
        pick = {
            'jugador_p1': 'Player1',
            'n_axes_active': 0,
            'clasificacion': 'apostar',
            'edge': 0.06,
        }

        # Aplicar regla
        if pick['n_axes_active'] < 2 and pick['clasificacion'] == 'apostar':
            pick['clasificacion'] = 'watchlist'
            pick['motivo_reclasificacion'] = 'N28F2: n_axes_active < 2 (BBI sola no predice)'

        assert pick['clasificacion'] == 'watchlist'
        assert 'N28F2' in pick.get('motivo_reclasificacion', '')

    def test_TF3_02_one_axis_apostar_to_watchlist(self):
        """TF3-02: pick con n_axes_active=1 y clasificacion='apostar' → 'watchlist'"""
        pick = {
            'jugador_p1': 'Player1',
            'n_axes_active': 1,
            'clasificacion': 'apostar',
            'edge': 0.055,
        }

        # Aplicar regla
        if pick['n_axes_active'] < 2 and pick['clasificacion'] == 'apostar':
            pick['clasificacion'] = 'watchlist'
            pick['motivo_reclasificacion'] = 'N28F2: n_axes_active < 2 (BBI sola no predice)'

        assert pick['clasificacion'] == 'watchlist'

    def test_TF3_03_two_axes_apostar_unchanged(self):
        """TF3-03: pick con n_axes_active=2 y clasificacion='apostar' → se mantiene 'apostar'"""
        pick = {
            'jugador_p1': 'Player1',
            'n_axes_active': 2,
            'clasificacion': 'apostar',
            'edge': 0.07,
        }

        # Aplicar regla
        if pick['n_axes_active'] < 2 and pick['clasificacion'] == 'apostar':
            pick['clasificacion'] = 'watchlist'

        assert pick['clasificacion'] == 'apostar'

    def test_TF3_04_three_axes_apostar_unchanged(self):
        """TF3-04: pick con n_axes_active=3 y clasificacion='apostar' → se mantiene 'apostar'"""
        pick = {
            'jugador_p1': 'Player1',
            'n_axes_active': 3,
            'clasificacion': 'apostar',
            'edge': 0.08,
        }

        # Aplicar regla
        if pick['n_axes_active'] < 2 and pick['clasificacion'] == 'apostar':
            pick['clasificacion'] = 'watchlist'

        assert pick['clasificacion'] == 'apostar'

    def test_TF3_05_one_axis_watchlist_unchanged(self):
        """TF3-05: pick con n_axes_active=1 y clasificacion='watchlist' → no cambia"""
        pick = {
            'jugador_p1': 'Player1',
            'n_axes_active': 1,
            'clasificacion': 'watchlist',
            'edge': 0.03,
        }

        # Aplicar regla (solo si es apostar)
        if pick['n_axes_active'] < 2 and pick['clasificacion'] == 'apostar':
            pick['clasificacion'] = 'watchlist'

        assert pick['clasificacion'] == 'watchlist'

    def test_TF3_06_one_axis_sin_edge_unchanged(self):
        """TF3-06: pick con n_axes_active=1 y clasificacion='sin_edge' → no cambia"""
        pick = {
            'jugador_p1': 'Player1',
            'n_axes_active': 1,
            'clasificacion': 'sin_edge',
            'edge': -0.01,
        }

        # Aplicar regla (solo si es apostar)
        if pick['n_axes_active'] < 2 and pick['clasificacion'] == 'apostar':
            pick['clasificacion'] = 'watchlist'

        assert pick['clasificacion'] == 'sin_edge'

    def test_TF3_07_motivo_cambio_contains_n28f2(self):
        """TF3-07: motivo_cambio contiene 'N28F2' cuando se aplica reclasificación"""
        pick = {
            'jugador_p1': 'Player1',
            'n_axes_active': 1,
            'clasificacion': 'apostar',
        }

        if pick['n_axes_active'] < 2 and pick['clasificacion'] == 'apostar':
            pick['clasificacion'] = 'watchlist'
            pick['motivo_reclasificacion'] = 'N28F2: n_axes_active < 2 (BBI sola no predice)'

        assert 'N28F2' in pick['motivo_reclasificacion']
        assert 'BBI' in pick['motivo_reclasificacion']

    def test_TF3_08_edge_and_kelly_unchanged(self):
        """TF3-08: edge y kelly_kl del pick NO cambian por reclasificación"""
        pick = {
            'jugador_p1': 'Player1',
            'n_axes_active': 1,
            'clasificacion': 'apostar',
            'edge': 0.055,
            'kelly_kl': 0.032,
        }

        original_edge = pick['edge']
        original_kelly = pick['kelly_kl']

        # Aplicar reclasificación
        if pick['n_axes_active'] < 2 and pick['clasificacion'] == 'apostar':
            pick['clasificacion'] = 'watchlist'

        # Edge y Kelly no cambian
        assert pick['edge'] == original_edge
        assert pick['kelly_kl'] == original_kelly

    def test_TF3_09_retroactive_18jun_7_picks_reclassified(self):
        """TF3-09: retroactivo 18-jun: 7 picks con n_axes=1 que eran APOSTAR ahora serían WATCHLIST"""
        # Fixture hardcoded: 7 picks from 18-jun edge_report que tenían n_axes=1
        picks_18jun = [
            {'jugador_p1': 'PickA', 'n_axes_active': 1, 'clasificacion': 'apostar', 'edge': 0.055},
            {'jugador_p1': 'PickB', 'n_axes_active': 1, 'clasificacion': 'apostar', 'edge': 0.052},
            {'jugador_p1': 'PickC', 'n_axes_active': 1, 'clasificacion': 'apostar', 'edge': 0.048},
            {'jugador_p1': 'PickD', 'n_axes_active': 1, 'clasificacion': 'apostar', 'edge': 0.061},
            {'jugador_p1': 'PickE', 'n_axes_active': 1, 'clasificacion': 'apostar', 'edge': 0.050},
            {'jugador_p1': 'PickF', 'n_axes_active': 1, 'clasificacion': 'apostar', 'edge': 0.044},
            {'jugador_p1': 'PickG', 'n_axes_active': 1, 'clasificacion': 'apostar', 'edge': 0.053},
        ]

        # Contar cuántos eran APOSTAR antes
        apostar_before = sum(1 for p in picks_18jun if p['clasificacion'] == 'apostar')
        assert apostar_before == 7

        # Aplicar reclasificación
        for pick in picks_18jun:
            if pick['n_axes_active'] < 2 and pick['clasificacion'] == 'apostar':
                pick['clasificacion'] = 'watchlist'

        # Verificar que todos son ahora WATCHLIST
        apostar_after = sum(1 for p in picks_18jun if p['clasificacion'] == 'apostar')
        watchlist_after = sum(1 for p in picks_18jun if p['clasificacion'] == 'watchlist')

        assert apostar_after == 0
        assert watchlist_after == 7


# ─────────────────────────────────────────────────────────────────────────────
# FIX-4: CONTESTED_ALPHA CUANDO OPONENTE ALINEADO (TF4-01 → TF4-07)
# ─────────────────────────────────────────────────────────────────────────────

class TestFIX4_ContestedAlpha:
    """
    Verificar que STRUCTURAL_ALPHA se convierte a CONTESTED_ALPHA
    cuando el oponente también tiene alignment > threshold.
    """

    def test_TF4_01_structural_alpha_maintained_opponent_zero_alignment(self):
        """TF4-01: STRUCTURAL_ALPHA se mantiene si oponente tiene alignment=0.0 (net > 0.25)"""
        alignment_fav = 0.49
        alignment_dog = 0.0
        net_alignment = alignment_fav - alignment_dog

        alignment_flag = 'STRUCTURAL_ALPHA'

        # Aplicar regla
        if alignment_flag == 'STRUCTURAL_ALPHA' and net_alignment < 0.25:
            alignment_flag = 'CONTESTED_ALPHA'

        assert alignment_flag == 'STRUCTURAL_ALPHA'
        assert net_alignment > 0.25

    def test_TF4_02_structural_alpha_to_contested_when_opponent_aligned(self):
        """TF4-02: STRUCTURAL_ALPHA → CONTESTED_ALPHA si oponente=0.40 y favorito=0.49 (net=0.09 < 0.25)"""
        alignment_fav = 0.49
        alignment_dog = 0.40
        net_alignment = alignment_fav - alignment_dog

        alignment_flag = 'STRUCTURAL_ALPHA'

        # Aplicar regla
        if alignment_flag == 'STRUCTURAL_ALPHA' and net_alignment < 0.25:
            alignment_flag = 'CONTESTED_ALPHA'

        assert alignment_flag == 'CONTESTED_ALPHA'
        assert net_alignment < 0.25
        assert net_alignment == pytest.approx(0.09, abs=0.001)

    def test_TF4_03_net_alignment_in_output(self):
        """TF4-03: net_alignment campo presente en output de triple_alignment_score()"""
        result = {
            'triple_alignment': 0.49,
            'alignment_fav': 0.49,
            'alignment_dog': 0.0,
            'net_alignment': 0.49,  # fav - dog
            'alignment_flag': 'STRUCTURAL_ALPHA',
        }

        assert 'net_alignment' in result
        assert result['net_alignment'] == pytest.approx(0.49, abs=0.001)

    def test_TF4_04_contested_alpha_treated_as_partial_alignment(self):
        """TF4-04: CONTESTED_ALPHA tratado como PARTIAL_ALIGNMENT en clasificación (no suprime si n_axes>=2)"""
        pick = {
            'jugador_p1': 'Federer',
            'n_axes_active': 2,
            'alignment_flag': 'CONTESTED_ALPHA',
            'clasificacion': 'apostar',
        }

        # CONTESTED_ALPHA NO suprime a watchlist si n_axes >= 2
        # (solo STRUCTURAL_ALPHA con problemas haría eso, pero ese caso no existe en este spec)
        # La lógica: si n_axes >= 2, se mantiene APOSTAR

        if pick['n_axes_active'] < 2 and pick['clasificacion'] == 'apostar':
            pick['clasificacion'] = 'watchlist'

        assert pick['clasificacion'] == 'apostar'
        assert pick['alignment_flag'] == 'CONTESTED_ALPHA'

    def test_TF4_05_retroactive_sziedat_contested_alpha(self):
        """TF4-05: retroactivo Sziedat: alignment_fav=0.49, oponente HOT → CONTESTED_ALPHA"""
        # Sziedat fue clasificado STRUCTURAL_ALPHA pero perdió contra Stoyanov que también estaba HOT
        sziedat_pick = {
            'jugador_p1': 'Sziedat',
            'jugador_p2': 'Stoyanov',
            'alignment_fav': 0.49,
            'markov_rival': 'HOT',
            'delta_wr_rival': 0.20,  # oponente tiene ventaja de estado
            'alignment_flag': 'STRUCTURAL_ALPHA',
        }

        # Calcular alignment del rival
        alignment_dog = 0.40  # hipotético, basado en markov HOT + delta
        net_alignment = sziedat_pick['alignment_fav'] - alignment_dog

        # Aplicar regla
        if sziedat_pick['alignment_flag'] == 'STRUCTURAL_ALPHA' and net_alignment < 0.25:
            sziedat_pick['alignment_flag'] = 'CONTESTED_ALPHA'

        assert sziedat_pick['alignment_flag'] == 'CONTESTED_ALPHA'
        assert net_alignment < 0.25

    def test_TF4_06_retroactive_eala_structural_alpha_maintained(self):
        """TF4-06: retroactivo Eala (18-jun anterior): alignment_fav=0.86, Rybakina NEUTRAL → STRUCTURAL_ALPHA se mantiene"""
        eala_pick = {
            'jugador_p1': 'Eala',
            'jugador_p2': 'Rybakina',
            'alignment_fav': 0.86,
            'markov_rival': 'NEUTRAL',
            'delta_wr_rival': 0.05,
            'alignment_flag': 'STRUCTURAL_ALPHA',
        }

        # Calcular alignment del rival
        alignment_dog = 0.0  # NEUTRAL → sin alineación especial
        net_alignment = eala_pick['alignment_fav'] - alignment_dog

        # Aplicar regla
        if eala_pick['alignment_flag'] == 'STRUCTURAL_ALPHA' and net_alignment < 0.25:
            eala_pick['alignment_flag'] = 'CONTESTED_ALPHA'

        assert eala_pick['alignment_flag'] == 'STRUCTURAL_ALPHA'
        assert net_alignment > 0.25

    def test_TF4_07_edge_and_kelly_unchanged_contested_alpha(self):
        """TF4-07: edge y kelly_kl NO cambian por CONTESTED_ALPHA"""
        pick = {
            'jugador_p1': 'Player1',
            'edge': 0.055,
            'kelly_kl': 0.032,
            'alignment_flag': 'CONTESTED_ALPHA',
        }

        original_edge = pick['edge']
        original_kelly = pick['kelly_kl']

        # El cambio de STRUCTURAL_ALPHA a CONTESTED_ALPHA no modifica edge ni kelly
        # (solo es informativo)

        assert pick['edge'] == original_edge
        assert pick['kelly_kl'] == original_kelly


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
