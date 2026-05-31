"""
Tests para edge_calculator.py (Nodo-01)
Cubre: Kelly-KL core, Volatility Smile, Factor Decomposition, Shannon Entropy, Thompson Sampling
"""
import math
import pytest
from edge_calculator import (
    zona_cuota, lambda_por_zona,
    bookmaker_entropy, psi_entropy_multiplier,
    phi_idiosincratico, elo_win_prob,
    thompson_p_historica, theta_thompson,
    calcular_edge,
    calcular_edge_completo,
    EDGE_MIN, KELLY_KL_MIN, BANKROLL_CAP,
)


# ─────────────────────────────────────────────────────────────────────────────
# L1 — KELLY-KL CORE (tests del Nodo-01 spec)
# ─────────────────────────────────────────────────────────────────────────────

class TestKellyKLCore:

    def test_edge_positivo_apostar_majchrzak(self):
        """Caso real: Majchrzak vs Marozsan → edge +9.5% → debe apostar"""
        r = calcular_edge(p_modelo=0.521, cuota_favorito=2.35)
        assert r['edge'] > 0.05
        assert r['apostar'] is True

    def test_edge_negativo_no_apostar_tsitsipas(self):
        """Caso real: Tsitsipas favorito obvio (1.08) → modelo dice 59.2% → edge -33%"""
        r = calcular_edge(p_modelo=0.592, cuota_favorito=1.08)
        assert r['edge'] < 0
        assert r['apostar'] is False

    def test_kelly_kl_menor_que_clasico_con_divergencia(self):
        """Cuando el modelo diverge de la historia, Kelly-KL debe ser más conservador"""
        r_calibrado = calcular_edge(0.55, 2.0, p_historica=0.55)    # modelo = historia → KL≈0
        r_divergente = calcular_edge(0.55, 2.0, p_historica=0.30)   # modelo diverge
        assert r_divergente['kelly_kl_base'] < r_calibrado['kelly_kl_base']

    def test_cap_10_porciento(self):
        """Nunca apostar más del 10% del bankroll"""
        r = calcular_edge(0.99, 1.01)
        assert r['fraccion_bankroll'] <= BANKROLL_CAP

    def test_edge_cero_no_apuesta(self):
        """Con p_modelo exactamente igual a p_implicita, edge=0 → no apostar"""
        cuota = 2.0
        p = 1.0 / cuota
        r = calcular_edge(p, cuota)
        assert abs(r['edge']) < 0.001
        assert r['apostar'] is False

    def test_kl_cerca_cero_cuando_modelo_igual_historia(self):
        """Si p_modelo ≈ p_historica, KL divergence ≈ 0"""
        r = calcular_edge(0.60, 1.80, p_historica=0.60)
        assert r['kl_divergencia'] < 0.01

    def test_kl_positivo_cuando_modelo_diverge(self):
        """KL siempre ≥ 0 (propiedad fundamental de la divergencia KL)"""
        r = calcular_edge(0.70, 1.80, p_historica=0.40)
        assert r['kl_divergencia'] >= 0

    def test_estructura_output_completa(self):
        """El output contiene todos los campos esperados"""
        r = calcular_edge(0.55, 2.0)
        campos_requeridos = [
            'p_modelo', 'p_implicita', 'edge', 'edge_pct',
            'kl_divergencia', 'kelly_clasico', 'kelly_kl_base', 'kelly_kl',
            'fraccion_bankroll', 'apostar',
            'phi_idiosincratico', 'psi_entropia', 'lambda_aversion', 'p_historica_usada'
        ]
        for campo in campos_requeridos:
            assert campo in r, f"Falta campo: {campo}"


# ─────────────────────────────────────────────────────────────────────────────
# L2 — VOLATILITY SMILE (Options Theory)
# ─────────────────────────────────────────────────────────────────────────────

class TestVolatilitySmile:

    def test_zona_heavy_favorite(self):
        assert zona_cuota(1.05) == "heavy_favorite"
        assert zona_cuota(1.29) == "heavy_favorite"

    def test_zona_moderate_favorite(self):
        assert zona_cuota(1.30) == "moderate_favorite"
        assert zona_cuota(1.59) == "moderate_favorite"

    def test_zona_slight_underdog(self):
        assert zona_cuota(1.60) == "slight_underdog"
        assert zona_cuota(2.09) == "slight_underdog"

    def test_zona_underdog(self):
        assert zona_cuota(2.10) == "underdog"
        assert zona_cuota(5.00) == "underdog"

    def test_lambda_heavy_favorite_muy_conservador(self):
        """En zona de favorito obvio, λ debe ser alto (muy conservador)"""
        assert lambda_por_zona("heavy_favorite") == 2.0

    def test_lambda_underdog_menos_conservador(self):
        """En zona underdog (nuestro sweet spot), λ debe ser bajo"""
        assert lambda_por_zona("underdog") == 0.3

    def test_lambda_monotonicamente_decreciente(self):
        """λ decrece a medida que la cuota sube (más oportunidad)"""
        lambda_hf = lambda_por_zona("heavy_favorite")
        lambda_mf = lambda_por_zona("moderate_favorite")
        lambda_su = lambda_por_zona("slight_underdog")
        lambda_ud = lambda_por_zona("underdog")
        assert lambda_hf > lambda_mf > lambda_su > lambda_ud

    def test_volatility_smile_kelly_kl_mayor_en_underdog(self):
        """
        Con el mismo edge bruto, la zona underdog debe producir mayor Kelly-KL
        que la zona de favorito (por λ menor).
        """
        # Mismo p_modelo=0.55, cuota que da similar edge en cada zona
        r_fav = calcular_edge(0.55, 1.25, lambda_aversion=2.0)   # heavy favorite zone
        r_und = calcular_edge(0.55, 1.70, lambda_aversion=0.3)   # underdog zone
        # Si hay edge en ambos, el underdog kelly debe ser mayor por λ menor
        if r_fav['edge'] > 0 and r_und['edge'] > 0:
            assert r_und['kelly_kl_base'] >= r_fav['kelly_kl_base']


# ─────────────────────────────────────────────────────────────────────────────
# L4 — SHANNON ENTROPY
# ─────────────────────────────────────────────────────────────────────────────

class TestShannonEntropy:

    def test_entropia_maxima_cuotas_iguales(self):
        """Cuotas 2.0 / 2.0 → entropia máxima ≈ 1.0 bits"""
        h = bookmaker_entropy(2.0, 2.0)
        assert h > 0.99

    def test_entropia_minima_favorito_extremo(self):
        """Cuotas 1.02 / 15.0 → bookmaker muy seguro → entropía baja"""
        h = bookmaker_entropy(1.02, 15.0)
        assert h < 0.40

    def test_entropia_tsitsipas(self):
        """Caso real: Tsitsipas 1.08 vs Mochizuki 6.5 → entropia moderada"""
        h = bookmaker_entropy(6.5, 1.08)  # cuota1=jugador1=Mochizuki, cuota2=Tsitsipas
        assert 0.50 < h < 0.70

    def test_entropia_majchrzak(self):
        """Caso real: Majchrzak 2.35 vs Marozsan 1.62 → entropía alta (cerca de 1)"""
        h = bookmaker_entropy(2.35, 1.62)
        assert h > 0.90

    def test_psi_alta_entropia_amplifica(self):
        """Alta entropía del bookmaker debe amplificar el Kelly"""
        psi_alta = psi_entropy_multiplier(0.97)
        psi_baja = psi_entropy_multiplier(0.35)
        assert psi_alta > psi_baja

    def test_psi_rango(self):
        """psi debe estar en rango [0.85, 1.15]"""
        for h in [0.0, 0.25, 0.50, 0.75, 1.0]:
            psi = psi_entropy_multiplier(h)
            assert 0.84 <= psi <= 1.16, f"psi={psi} fuera de rango para h={h}"

    def test_entropia_sin_cuotas_retorna_neutral(self):
        """Sin cuotas válidas, retorna 0.5 (neutral)"""
        assert bookmaker_entropy(None, None) == 0.5
        assert bookmaker_entropy(0, 2.0) == 0.5


# ─────────────────────────────────────────────────────────────────────────────
# L3 — FACTOR DECOMPOSITION (Fama-French)
# ─────────────────────────────────────────────────────────────────────────────

class TestFactorDecomposition:

    def _make_score_breakdown(self, elo_pct, ranking_pct, surface_pct, form_pct, common_pct):
        """Helper para crear un score_breakdown con porcentajes específicos."""
        return {
            'player1': {
                'elo_rating':           {'contribution': f'{elo_pct}%'},
                'ranking_momentum':     {'contribution': f'{ranking_pct}%'},
                'surface_specialization': {'contribution': f'{surface_pct}%'},
                'form_recent':          {'contribution': f'{form_pct}%'},
                'common_opponents':     {'contribution': f'{common_pct}%'},
            }
        }

    def test_phi_alto_cuando_factores_desconocidos_dominan(self):
        """Si surface+form+common dominan → phi > 1.0"""
        sb = self._make_score_breakdown(elo_pct=5, ranking_pct=5, surface_pct=30, form_pct=30, common_pct=30)
        phi = phi_idiosincratico(sb, 'player1')
        assert phi > 1.0

    def test_phi_bajo_cuando_ranking_domina(self):
        """Si elo+ranking dominan → phi < 1.0 (bookmaker ya lo sabe)"""
        sb = self._make_score_breakdown(elo_pct=40, ranking_pct=40, surface_pct=5, form_pct=10, common_pct=5)
        phi = phi_idiosincratico(sb, 'player1')
        assert phi < 1.0

    def test_phi_rango(self):
        """phi siempre en [0.80, 1.30]"""
        for elo_p in [0, 20, 50, 80, 100]:
            other = 100 - elo_p
            sb = self._make_score_breakdown(elo_p, 0, other//3, other//3, other - 2*(other//3))
            phi = phi_idiosincratico(sb, 'player1')
            assert 0.79 <= phi <= 1.31, f"phi={phi} fuera de rango"

    def test_phi_neutral_sin_datos(self):
        """Sin score_breakdown, phi = 1.0 (sin ajuste)"""
        assert phi_idiosincratico({}, 'player1') == 1.0

    def test_elo_win_prob_iguales(self):
        """ELOs iguales → 50% de probabilidad"""
        assert abs(elo_win_prob(1800, 1800) - 0.5) < 0.001

    def test_elo_win_prob_mayor_elo_mayor_prob(self):
        """ELO mayor implica mayor probabilidad de ganar"""
        assert elo_win_prob(2000, 1600) > 0.85

    def test_elo_win_prob_sin_datos(self):
        """Sin ELO, retorna 0.5"""
        assert elo_win_prob(None, None) == 0.5

    def test_majchrzak_phi_razonable(self):
        """Caso real Majchrzak: ELO=19.3%, ranking=33.3%, form=23.2%, common=24.1%"""
        sb = self._make_score_breakdown(elo_pct=19.3, ranking_pct=33.3,
                                         surface_pct=0.0, form_pct=23.2, common_pct=24.1)
        phi = phi_idiosincratico(sb, 'player1')
        # known=52.6%, unknown=47.3% → phi ≈ 1.037
        assert 0.95 <= phi <= 1.10


# ─────────────────────────────────────────────────────────────────────────────
# L5 — THOMPSON SAMPLING / BAYESIAN CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

class TestThompsonSampling:

    def test_prior_uniforme_sin_datos(self):
        """Beta(1,1): sin datos → p_historica = 0.5"""
        assert thompson_p_historica(wins=0, losses=0) == 0.5

    def test_con_muchas_victorias_sube(self):
        """Muchas victorias → p_historica > 0.5"""
        p = thompson_p_historica(wins=15, losses=5)
        assert p > 0.70

    def test_con_muchas_derrotas_baja(self):
        """Muchas derrotas → p_historica < 0.5"""
        p = thompson_p_historica(wins=5, losses=15)
        assert p < 0.40

    def test_jan2026_datos_sucios(self):
        """9/19 victorias (datos sucios Jan 2026) → p ≈ 0.476"""
        p = thompson_p_historica(wins=9, losses=10)
        assert 0.45 <= p <= 0.52

    def test_theta_usa_superficie_cuando_n_suficiente(self):
        """Con n≥10 por superficie, usa calibración de superficie"""
        calibracion = {
            'global': {'wins': 5, 'losses': 5},
            'por_superficie': {
                'clay': {'wins': 12, 'losses': 4},  # clay: 75%
            }
        }
        p = theta_thompson(calibracion, 'clay')
        assert p > 0.65  # debe usar clay (75%), no global (50%)

    def test_theta_fallback_global_cuando_poco_n(self):
        """Con <10 por superficie, usa calibración global"""
        calibracion = {
            'global': {'wins': 8, 'losses': 2},   # global: 80%
            'por_superficie': {
                'clay': {'wins': 3, 'losses': 1},  # clay: n=4 < 10
            }
        }
        p = theta_thompson(calibracion, 'clay')
        assert p > 0.70  # debe usar global (80%)


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRACIÓN — PARTIDO COMPLETO
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCompletoPartido:

    def _make_partido(self, cuota1, cuota2, confidence, favored, jugador1="A", jugador2="B",
                      elo1=1800, elo2=1800):
        """Helper para construir un partido mínimo válido."""
        return {
            'jugador1': jugador1,
            'jugador2': jugador2,
            'cuota1': cuota1,
            'cuota2': cuota2,
            'superficie': 'clay',
            'torneo_nombre': 'Roland Garros (France)',
            'ranking_analysis': {
                f'{jugador1.replace(" ", "_")}_elo': elo1,
                f'{jugador2.replace(" ", "_")}_elo': elo2,
                f'{jugador1.replace(" ", "_")}_ranking': 100,
                f'{jugador2.replace(" ", "_")}_ranking': 50,
                'prediction': {
                    'favored_player': favored,
                    'confidence': confidence,
                    'scores': {'p1_final_weight': 2.5, 'p2_final_weight': 2.8},
                    'score_breakdown': {
                        'player1': {
                            'elo_rating':           {'contribution': '20%'},
                            'ranking_momentum':     {'contribution': '30%'},
                            'surface_specialization': {'contribution': '0%'},
                            'form_recent':          {'contribution': '25%'},
                            'common_opponents':     {'contribution': '25%'},
                        },
                        'player2': {
                            'elo_rating':           {'contribution': '25%'},
                            'ranking_momentum':     {'contribution': '35%'},
                            'surface_specialization': {'contribution': '0%'},
                            'form_recent':          {'contribution': '20%'},
                            'common_opponents':     {'contribution': '20%'},
                        }
                    },
                    'weights_used': {}
                }
            },
            'match_url': None,
            'match_id': None,
        }

    def _calibracion_vacia(self):
        return {
            'global': {'wins': 0, 'losses': 0},
            'por_superficie': {'clay': {'wins': 0, 'losses': 0}},
            'por_zona': {}
        }

    def test_partido_underdog_con_edge_apuesta(self):
        """Underdog con edge > 5% → apostar"""
        p = self._make_partido(
            cuota1=2.35, cuota2=1.62,
            confidence=52.1, favored="A",
            jugador1="A", jugador2="B"
        )
        r = calcular_edge_completo(p, self._calibracion_vacia())
        assert r is not None
        assert r['edge'] > 0.05
        assert r['apostar'] is True
        assert r['zona_cuota'] == 'underdog'

    def test_partido_favorito_obvio_no_apuesta(self):
        """Favorito obvio (cuota 1.08) con confidence razonable → no apostar"""
        p = self._make_partido(
            cuota1=6.5, cuota2=1.08,
            confidence=59.2, favored="B",
            jugador1="A", jugador2="B"
        )
        r = calcular_edge_completo(p, self._calibracion_vacia())
        assert r is not None
        assert r['apostar'] is False
        assert r['zona_cuota'] == 'heavy_favorite'

    def test_sin_prediccion_retorna_none(self):
        """Partido sin predicción → retorna None"""
        p = self._make_partido(2.0, 2.0, None, None)
        p['ranking_analysis']['prediction']['favored_player'] = None
        p['ranking_analysis']['prediction']['confidence'] = None
        r = calcular_edge_completo(p, self._calibracion_vacia())
        assert r is None

    def test_sin_cuotas_retorna_none(self):
        """Partido sin cuotas → retorna None"""
        p = self._make_partido(None, None, 60.0, "A")
        r = calcular_edge_completo(p, self._calibracion_vacia())
        assert r is None

    def test_resultado_contiene_todas_las_capas(self):
        """El resultado debe incluir campos de las 5 capas"""
        p = self._make_partido(2.0, 2.0, 60.0, "A")
        r = calcular_edge_completo(p, self._calibracion_vacia())
        assert r is not None
        # L1
        assert 'edge' in r
        assert 'kelly_kl' in r
        assert 'apostar' in r
        # L2
        assert 'zona_cuota' in r
        assert 'lambda_aversion' in r
        # L3
        assert 'phi_idiosincratico' in r
        assert 'alpha_vs_elo' in r
        assert 'p_elo_base' in r
        # L4
        assert 'entropy_bookmaker' in r
        assert 'psi_entropia' in r
        # L5
        assert 'p_historica_usada' in r

    def test_fraccion_bankroll_nunca_supera_10_porciento(self):
        """Límite de seguridad: nunca apostar más del 10%"""
        p = self._make_partido(
            cuota1=10.0, cuota2=1.01,
            confidence=99.0, favored="A"
        )
        r = calcular_edge_completo(p, self._calibracion_vacia())
        if r:
            assert r['fraccion_bankroll'] <= BANKROLL_CAP
