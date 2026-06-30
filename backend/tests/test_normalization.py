"""
Tests para el módulo de normalización — normalization.py

COBERTURA:
  - MAX_RAW_SCORES (constantes)
  - DEFAULT_WEIGHTS (constantes)
  - normalize_min_max
  - normalize_with_log_scale
  - normalize_elo
  - normalize_percentage
  - WeightManager.validate_weights
  - WeightManager.calculate_adjusted_weights
  - calculate_confidence
  - validate_score_calculation
  - normalize_and_weight_scores (interfaz simplificada)

PRINCIPIO:
  La normalización es la base matemática de todas las predicciones.
  Un error aquí se propaga a todos los puntajes finales.
  Cobertura objetivo: 90%+
"""

import pytest
import math
from normalization import (
    MAX_RAW_SCORES,
    DEFAULT_WEIGHTS,
    normalize_min_max,
    normalize_with_log_scale,
    normalize_elo,
    normalize_percentage,
    WeightManager,
    calculate_confidence,
    validate_score_calculation,
    normalize_and_weight_scores,
)


# ─────────────────────────────────────────────────────────────────────────────
# MAX_RAW_SCORES — Integridad de constantes
# ─────────────────────────────────────────────────────────────────────────────

class TestMaxRawScores:
    """MAX_RAW_SCORES define los límites del sistema. Si cambian sin tests, el modelo cambia."""

    def test_contiene_todos_los_componentes_requeridos(self):
        componentes = {
            'home_advantage', 'surface_specialization', 'ranking_momentum',
            'form_recent', 'common_opponents', 'h2h_direct',
            'elo_rating', 'strength_of_schedule'
        }
        assert componentes == set(MAX_RAW_SCORES.keys())

    def test_todos_los_valores_son_positivos(self):
        for componente, valor in MAX_RAW_SCORES.items():
            assert valor > 0, f"{componente} debe ser positivo, got {valor}"

    def test_valores_especificos_no_cambian_sin_revision(self):
        """Snapshot de los valores actuales. Si cambian → falla → revisión consciente."""
        assert MAX_RAW_SCORES['ranking_momentum'] == 450
        assert MAX_RAW_SCORES['common_opponents'] == 400
        assert MAX_RAW_SCORES['surface_specialization'] == 350
        assert MAX_RAW_SCORES['h2h_direct'] == 350
        assert MAX_RAW_SCORES['form_recent'] == 300
        assert MAX_RAW_SCORES['elo_rating'] == 250
        assert MAX_RAW_SCORES['strength_of_schedule'] == 200
        assert MAX_RAW_SCORES['home_advantage'] == 100

    def test_elo_rating_floor_cap_acoplamiento(self):
        """B-11 prevención: MAX_RAW_SCORES['elo_rating'] debe cubrir el rango
        producido por rivalry_analyzer (floor=1500, ELO_MAX_ESPERADO≈1750).
        Si alguien cambia el floor sin actualizar MAX_RAW, log1p comprime
        las diferencias y el modelo pierde discriminación (sesión Jun-15: 46.9%).
        """
        ELO_FLOOR = 1500       # rivalry_analyzer.py:1312 — floor de raw_scores['elo_rating']
        ELO_MAX_ESPERADO = 1750  # techo realista para ELO calculado en Challenger/ATP
        max_raw_producido = ELO_MAX_ESPERADO - ELO_FLOOR
        assert MAX_RAW_SCORES['elo_rating'] >= max_raw_producido, (
            f"MAX_RAW_SCORES['elo_rating']={MAX_RAW_SCORES['elo_rating']} < "
            f"max_raw_producido={max_raw_producido} (ELO_MAX={ELO_MAX_ESPERADO} - FLOOR={ELO_FLOOR}). "
            f"Actualizar MAX_RAW_SCORES o el floor en rivalry_analyzer.py para evitar B-11."
        )


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT_WEIGHTS — Integridad de pesos
# ─────────────────────────────────────────────────────────────────────────────

class TestDefaultWeights:
    """Todos los tipos de torneo deben tener pesos que sumen 1.0."""

    @pytest.mark.parametrize("tipo_torneo", ['atp_wta', 'challenger', 'itf', 'default'])
    def test_pesos_suman_uno(self, tipo_torneo):
        pesos = DEFAULT_WEIGHTS[tipo_torneo]
        total = sum(pesos.values())
        assert abs(total - 1.0) < 0.001, (
            f"Pesos de '{tipo_torneo}' suman {total:.4f}, esperado 1.0"
        )

    @pytest.mark.parametrize("tipo_torneo", ['atp_wta', 'challenger', 'itf', 'default'])
    def test_todos_los_pesos_son_no_negativos(self, tipo_torneo):
        pesos = DEFAULT_WEIGHTS[tipo_torneo]
        for componente, peso in pesos.items():
            assert peso >= 0, f"Peso negativo en {tipo_torneo}.{componente}: {peso}"

    @pytest.mark.parametrize("tipo_torneo", ['atp_wta', 'challenger', 'itf', 'default'])
    def test_contiene_todos_los_componentes(self, tipo_torneo):
        componentes_requeridos = set(MAX_RAW_SCORES.keys())
        componentes_presentes = set(DEFAULT_WEIGHTS[tipo_torneo].keys())
        assert componentes_requeridos == componentes_presentes, (
            f"Componentes faltantes en {tipo_torneo}: "
            f"{componentes_requeridos - componentes_presentes}"
        )

    def test_atp_wta_pesos_especificos(self):
        """Snapshot de pesos ATP/WTA. Cambiarlos sin test = riesgo silencioso."""
        pesos = DEFAULT_WEIGHTS['atp_wta']
        assert pesos['common_opponents'] == 0.20
        assert pesos['ranking_momentum'] == 0.20
        assert pesos['surface_specialization'] == 0.15
        assert pesos['form_recent'] == 0.15
        assert pesos['h2h_direct'] == 0.15
        assert pesos['elo_rating'] == 0.10
        assert pesos['home_advantage'] == 0.05
        assert pesos['strength_of_schedule'] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# normalize_min_max
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeMinMax:
    """Tests para la normalización lineal min-max."""

    def test_valor_mitad_da_05(self):
        """225 / 450 (ranking_momentum) = 0.5."""
        score, meta = normalize_min_max(225, 'ranking_momentum')
        assert abs(score - 0.5) < 0.001

    def test_valor_cero_da_cero(self):
        score, _ = normalize_min_max(0, 'ranking_momentum')
        assert score == 0.0

    def test_valor_maximo_da_uno(self):
        """El valor máximo del componente → 1.0."""
        max_val = MAX_RAW_SCORES['ranking_momentum']
        score, _ = normalize_min_max(max_val, 'ranking_momentum')
        assert abs(score - 1.0) < 0.001

    def test_valor_sobre_maximo_es_clipeado_a_uno(self):
        """Valores sobre el máximo son clipeados a 1.0."""
        score, _ = normalize_min_max(9999, 'ranking_momentum')
        assert score == 1.0

    def test_valor_negativo_es_clipeado_a_cero(self):
        """Valores negativos son clipeados a 0.0."""
        score, _ = normalize_min_max(-100, 'ranking_momentum')
        assert score == 0.0

    def test_componente_desconocido_usa_100_como_default(self):
        """Componente no en MAX_RAW_SCORES usa 100 como denominador."""
        score, meta = normalize_min_max(50, 'componente_inexistente')
        assert abs(score - 0.5) < 0.001
        assert meta['max_expected'] == 100

    def test_max_expected_personalizado(self):
        """Se puede pasar max_expected personalizado."""
        score, _ = normalize_min_max(25, 'ranking_momentum', max_expected=50)
        assert abs(score - 0.5) < 0.001

    def test_metadata_contiene_campos_requeridos(self):
        """La metadata siempre contiene raw_score, max_expected, normalization_factor, component."""
        _, meta = normalize_min_max(100, 'form_recent')
        assert 'raw_score' in meta
        assert 'max_expected' in meta
        assert 'normalization_factor' in meta
        assert 'component' in meta

    def test_resultado_siempre_entre_0_y_1(self):
        """El resultado normalizado siempre está en [0, 1]."""
        for valor in [-100, 0, 1, 50, 300, 450, 1000]:
            score, _ = normalize_min_max(valor, 'ranking_momentum')
            assert 0.0 <= score <= 1.0, f"score={score} fuera de [0,1] para valor={valor}"

    @pytest.mark.parametrize("componente", list(MAX_RAW_SCORES.keys()))
    def test_todos_los_componentes_normalizan_correctamente(self, componente):
        """Cada componente de MAX_RAW_SCORES normaliza sin errores."""
        max_val = MAX_RAW_SCORES[componente]
        score, _ = normalize_min_max(max_val / 2, componente)
        assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# normalize_with_log_scale
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeWithLogScale:
    """Tests para la normalización logarítmica."""

    def test_valor_cero_da_cero(self):
        score, _ = normalize_with_log_scale(0, 'ranking_momentum')
        assert score == 0.0

    def test_valor_maximo_da_scale_factor(self):
        """El valor máximo del componente → scale_factor (default 10)."""
        max_val = MAX_RAW_SCORES['ranking_momentum']
        score, _ = normalize_with_log_scale(max_val, 'ranking_momentum')
        assert abs(score - 10.0) < 0.001

    def test_escala_es_logaritmica(self):
        """Valores pequeños tienen mayor discriminación que en min-max."""
        score_10, _ = normalize_with_log_scale(10, 'ranking_momentum')
        score_100, _ = normalize_with_log_scale(100, 'ranking_momentum')
        score_400, _ = normalize_with_log_scale(400, 'ranking_momentum')
        # Diferencia entre 10 y 100 > diferencia entre 100 y 400 (compresión logarítmica)
        assert (score_100 - score_10) > (score_400 - score_100)

    def test_scale_factor_personalizado(self):
        """Scale factor personalizado ajusta el rango de salida."""
        max_val = MAX_RAW_SCORES['ranking_momentum']
        score, _ = normalize_with_log_scale(max_val, 'ranking_momentum', scale_factor=5.0)
        assert abs(score - 5.0) < 0.001

    def test_resultado_clipeado_a_scale_factor(self):
        """Valores sobre el máximo son clipeados al scale_factor."""
        score, _ = normalize_with_log_scale(9999, 'ranking_momentum')
        assert score <= 10.0

    def test_resultado_nunca_negativo(self):
        score, _ = normalize_with_log_scale(0, 'ranking_momentum')
        assert score >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# normalize_elo
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeElo:
    """Tests para la normalización del rating ELO a escala 0-10."""

    def test_elo_minimo_da_cero(self):
        """ELO 1200 (mínimo) → 0.0."""
        assert normalize_elo(1200) == 0.0

    def test_elo_maximo_da_diez(self):
        """ELO 2400 (máximo) → 10.0."""
        assert normalize_elo(2400) == 10.0

    def test_elo_medio_da_cinco(self):
        """ELO 1800 (mitad del rango 1200-2400) → 5.0."""
        assert abs(normalize_elo(1800) - 5.0) < 0.01

    def test_elo_default_1500_da_2_5(self):
        """ELO 1500 (default del sistema) → 2.5."""
        assert abs(normalize_elo(1500) - 2.5) < 0.01

    def test_elo_sobre_maximo_es_clipeado_a_diez(self):
        assert normalize_elo(3000) == 10.0

    def test_elo_bajo_minimo_es_clipeado_a_cero(self):
        assert normalize_elo(500) == 0.0

    def test_resultado_siempre_entre_0_y_10(self):
        for elo in [800, 1200, 1500, 1800, 2100, 2400, 2800]:
            result = normalize_elo(elo)
            assert 0.0 <= result <= 10.0, f"ELO {elo} → {result} fuera de [0, 10]"


# ─────────────────────────────────────────────────────────────────────────────
# normalize_percentage
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizePercentage:
    """Tests para la normalización de porcentajes a escala 0-10."""

    def test_70_porciento_da_7(self):
        assert normalize_percentage(70, 100) == 7.0

    def test_100_porciento_da_10(self):
        assert normalize_percentage(100, 100) == 10.0

    def test_0_porciento_da_0(self):
        assert normalize_percentage(0, 100) == 0.0

    def test_sobre_100_clipeado_a_10(self):
        assert normalize_percentage(150, 100) == 10.0

    def test_max_value_cero_retorna_0(self):
        """max_value=0 no lanza ZeroDivisionError, retorna 0."""
        assert normalize_percentage(50, 0) == 0.0

    def test_max_value_personalizado(self):
        """max_value diferente de 100."""
        assert normalize_percentage(5, 10) == 5.0


# ─────────────────────────────────────────────────────────────────────────────
# WeightManager
# ─────────────────────────────────────────────────────────────────────────────

class TestWeightManager:
    """Tests para el gestor de pesos dinámicos."""

    @pytest.fixture
    def pesos_atp(self):
        return DEFAULT_WEIGHTS['atp_wta'].copy()

    @pytest.fixture
    def manager(self, pesos_atp):
        return WeightManager(pesos_atp)

    # ── validate_weights ─────────────────────────────────────────────────────

    def test_validate_weights_pesos_validos(self, manager, pesos_atp):
        valido, mensaje = manager.validate_weights(pesos_atp)
        assert valido is True
        assert "válidos" in mensaje.lower()

    def test_validate_weights_suma_incorrecta(self, manager):
        pesos_mal = {'a': 0.5, 'b': 0.6}  # suma = 1.1
        valido, mensaje = manager.validate_weights(pesos_mal)
        assert valido is False
        assert "1.1" in mensaje or "suma" in mensaje.lower()

    def test_validate_weights_peso_negativo(self, manager):
        pesos_mal = {'a': -0.1, 'b': 1.1}
        valido, mensaje = manager.validate_weights(pesos_mal)
        assert valido is False

    def test_validate_weights_peso_mayor_que_uno(self, manager):
        pesos_mal = {'a': 1.5, 'b': -0.5}
        valido, mensaje = manager.validate_weights(pesos_mal)
        assert valido is False

    # ── calculate_adjusted_weights ───────────────────────────────────────────

    def test_todos_disponibles_mantiene_pesos(self, manager, pesos_atp):
        """Si todos los componentes están disponibles, los pesos no cambian."""
        disponibles = {k: True for k in pesos_atp}
        ajustados = manager.calculate_adjusted_weights(disponibles, pesos_atp)
        for componente, peso in pesos_atp.items():
            assert abs(ajustados[componente] - peso) < 0.001, (
                f"{componente}: esperado {peso}, got {ajustados[componente]}"
            )

    def test_un_componente_no_disponible_redistribuye(self, manager, pesos_atp):
        """Con un componente no disponible, los pesos se redistribuyen y suman 1.0."""
        disponibles = {k: True for k in pesos_atp}
        disponibles['surface_specialization'] = False
        ajustados = manager.calculate_adjusted_weights(disponibles, pesos_atp)
        total = sum(ajustados.values())
        assert abs(total - 1.0) < 0.001

    def test_componente_no_disponible_tiene_peso_cero(self, manager, pesos_atp):
        """El componente no disponible recibe peso 0 en los ajustados."""
        disponibles = {k: True for k in pesos_atp}
        disponibles['surface_specialization'] = False
        ajustados = manager.calculate_adjusted_weights(disponibles, pesos_atp)
        assert ajustados['surface_specialization'] == 0.0

    def test_varios_componentes_no_disponibles(self, manager, pesos_atp):
        """Múltiples componentes no disponibles → pesos redistribuidos suman 1.0."""
        disponibles = {k: True for k in pesos_atp}
        disponibles['surface_specialization'] = False
        disponibles['home_advantage'] = False
        ajustados = manager.calculate_adjusted_weights(disponibles, pesos_atp)
        total = sum(ajustados.values())
        assert abs(total - 1.0) < 0.001

    def test_sin_componentes_disponibles_retorna_vacio(self, manager, pesos_atp):
        """Si ningún componente está disponible → retorna dict vacío."""
        disponibles = {k: False for k in pesos_atp}
        ajustados = manager.calculate_adjusted_weights(disponibles, pesos_atp)
        assert ajustados == {}

    def test_sin_pesos_base_y_sin_config_lanza_error(self):
        """Sin pesos base Y sin config → ValueError."""
        manager = WeightManager(None)
        with pytest.raises(ValueError):
            manager.calculate_adjusted_weights({'a': True})


# ─────────────────────────────────────────────────────────────────────────────
# calculate_confidence
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateConfidence:
    """Tests para el cálculo de confianza (50-95%)."""

    def test_sin_diferencia_es_50_porciento(self):
        """Sin diferencia entre jugadores → 50% (incertidumbre máxima)."""
        assert calculate_confidence(0.0, 5.0) == 50.0

    def test_diferencia_maxima_se_acerca_a_95(self):
        """Diferencia = total_score → 100% de diferencia relativa → 95% (tope)."""
        result = calculate_confidence(10.0, 10.0)
        assert abs(result - 95.0) < 0.001

    def test_resultado_clipeado_a_95(self):
        """Nunca supera 95%."""
        result = calculate_confidence(1000.0, 1.0)
        assert result <= 95.0

    def test_resultado_nunca_bajo_50(self):
        """Nunca baja de 50%."""
        result = calculate_confidence(0.0, 100.0)
        assert result >= 50.0

    def test_total_score_cero_retorna_50(self):
        """total_score=0 no lanza ZeroDivisionError, retorna 50%."""
        result = calculate_confidence(5.0, 0.0)
        assert result == 50.0

    def test_diferencia_negativa_usa_valor_absoluto(self):
        """La diferencia negativa produce el mismo resultado que positiva."""
        pos = calculate_confidence(1.0, 5.0)
        neg = calculate_confidence(-1.0, 5.0)
        assert pos == neg

    def test_confianza_moderada(self):
        """score_diff=0.5, total=5.0 → 50 + (0.5/5.0)*50 = 55%."""
        result = calculate_confidence(0.5, 5.0)
        assert abs(result - 55.0) < 0.001


# ─────────────────────────────────────────────────────────────────────────────
# validate_score_calculation
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateScoreCalculation:
    """Tests para la validación de coherencia de un cálculo de puntuación."""

    @pytest.fixture
    def calculo_valido(self):
        raw = {'ranking': 225.0, 'form': 150.0}
        norm = {'ranking': 0.5, 'form': 0.5}
        weights = {'ranking': 0.5, 'form': 0.5}
        weighted = {'ranking': 0.25, 'form': 0.25}
        final = 0.5
        return raw, norm, weighted, final, weights

    def test_calculo_coherente_retorna_true(self, calculo_valido):
        raw, norm, weighted, final, weights = calculo_valido
        es_coherente, advertencias = validate_score_calculation(
            raw, norm, weighted, final, weights
        )
        assert es_coherente is True

    def test_pesos_incorrectos_genera_advertencia(self, calculo_valido):
        raw, norm, weighted, final, _ = calculo_valido
        pesos_mal = {'ranking': 0.6, 'form': 0.6}  # suma = 1.2
        _, advertencias = validate_score_calculation(raw, norm, weighted, final, pesos_mal)
        assert any("peso" in w.lower() or "weight" in w.lower() or "suma" in w.lower()
                   for w in advertencias)

    def test_score_normalizado_negativo_genera_error(self):
        raw = {'ranking': 100.0}
        norm = {'ranking': -0.1}  # inválido
        weights = {'ranking': 1.0}
        weighted = {'ranking': -0.1}
        final = -0.1
        es_coherente, advertencias = validate_score_calculation(
            raw, norm, weighted, final, weights
        )
        assert es_coherente is False
        assert any("ERROR" in w for w in advertencias)

    def test_score_final_negativo_genera_error(self):
        raw = {'ranking': 100.0}
        norm = {'ranking': 0.5}
        weights = {'ranking': 1.0}
        weighted = {'ranking': 0.5}
        es_coherente, advertencias = validate_score_calculation(
            raw, norm, weighted, -1.0, weights
        )
        assert es_coherente is False


# ─────────────────────────────────────────────────────────────────────────────
# normalize_and_weight_scores (interfaz simplificada)
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeAndWeightScores:
    """Tests para la interfaz simplificada de normalización y ponderación."""

    def test_retorna_keys_requeridas(self):
        raw = {'ranking_momentum': 225.0, 'form_recent': 150.0}
        weights = {'ranking_momentum': 0.5, 'form_recent': 0.5}
        result = normalize_and_weight_scores(raw, weights)
        assert 'raw_scores' in result
        assert 'normalized_scores' in result
        assert 'weighted_scores' in result
        assert 'final_score' in result
        assert 'is_coherent' in result

    def test_score_final_entre_0_y_1(self):
        raw = {'ranking_momentum': 225.0, 'form_recent': 150.0}
        weights = {'ranking_momentum': 0.5, 'form_recent': 0.5}
        result = normalize_and_weight_scores(raw, weights)
        assert 0.0 <= result['final_score'] <= 1.0

    def test_componente_no_disponible_excluido_del_final(self):
        """Si surface_specialization no está disponible, su peso se redistribuye."""
        raw = {'ranking_momentum': 225.0, 'form_recent': 150.0, 'surface_specialization': 0.0}
        weights = {'ranking_momentum': 0.40, 'form_recent': 0.40, 'surface_specialization': 0.20}
        disponibles = {
            'ranking_momentum': True,
            'form_recent': True,
            'surface_specialization': False  # no disponible
        }
        result_sin = normalize_and_weight_scores(raw, weights, available_components=disponibles)
        result_con = normalize_and_weight_scores(raw, weights)
        # El score sin surface debe ser diferente (redistribución de pesos)
        assert result_sin['adjusted_weights']['surface_specialization'] == 0.0

    def test_metodo_log_scale(self):
        """Acepta normalization_method='log_scale'."""
        raw = {'ranking_momentum': 225.0}
        weights = {'ranking_momentum': 1.0}
        result = normalize_and_weight_scores(raw, weights, normalization_method='log_scale')
        assert 0.0 <= result['final_score'] <= 10.0

    def test_calculo_es_coherente_para_datos_validos(self):
        raw = {'ranking_momentum': 225.0, 'form_recent': 150.0}
        weights = {'ranking_momentum': 0.5, 'form_recent': 0.5}
        result = normalize_and_weight_scores(raw, weights)
        assert result['is_coherent'] is True
