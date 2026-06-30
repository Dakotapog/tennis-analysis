"""
Tests para Intelligent_ml_enhancer.py — NIVEL 3 cobertura de producción.

Cubre funciones puras y utilitarias sin dependencias de disco ni entrenamiento real.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# AIConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestAIConfig:
    """AIConfig expone constantes con valores válidos para el dominio."""

    def setup_method(self):
        from Intelligent_ml_enhancer import AIConfig
        self.cfg = AIConfig()

    def test_knn_neighbors_positivo(self):
        assert self.cfg.KNN_NEIGHBORS > 0

    def test_contamination_en_rango(self):
        assert 0 < self.cfg.ISOLATION_CONTAMINATION < 1

    def test_min_features_menor_que_max(self):
        assert self.cfg.MIN_FEATURES < self.cfg.MAX_FEATURES

    def test_correlation_threshold_en_rango(self):
        assert 0 < self.cfg.CORRELATION_THRESHOLD <= 1

    def test_cv_folds_positivo(self):
        assert self.cfg.CV_FOLDS >= 2


# ─────────────────────────────────────────────────────────────────────────────
# IntelligentMLEnhancer — inicialización
# ─────────────────────────────────────────────────────────────────────────────

class TestEnhancerInit:
    """El enhancer se inicializa correctamente sin parámetros."""

    def setup_method(self):
        from Intelligent_ml_enhancer import IntelligentMLEnhancer
        self.enhancer = IntelligentMLEnhancer()

    def test_is_fitted_false_al_inicio(self):
        assert self.enhancer.is_fitted is False

    def test_feature_names_vacio_al_inicio(self):
        assert self.enhancer.feature_names == []

    def test_enhancement_history_vacio_al_inicio(self):
        assert self.enhancer.enhancement_history == []

    def test_target_column_por_defecto(self):
        assert self.enhancer.target_column == 'ganador_real'

    def test_config_asignado(self):
        from Intelligent_ml_enhancer import AIConfig
        assert isinstance(self.enhancer.config, AIConfig)


# ─────────────────────────────────────────────────────────────────────────────
# _analyze_missing_patterns
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeMissingPatterns:
    """_analyze_missing_patterns detecta y clasifica columnas con NaN."""

    def setup_method(self):
        from Intelligent_ml_enhancer import IntelligentMLEnhancer
        self.enhancer = IntelligentMLEnhancer()

    def test_sin_nans_total_cero(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = self.enhancer._analyze_missing_patterns(df)
        assert result['total_missing'] == 0
        assert result['columns_affected'] == 0

    def test_con_nans_detecta_columna(self):
        df = pd.DataFrame({'a': [1, None, 3], 'b': [4, 5, 6]})
        result = self.enhancer._analyze_missing_patterns(df)
        assert result['total_missing'] == 1
        assert result['columns_affected'] == 1
        assert 'a' in result['patterns']

    def test_porcentaje_calculado_correctamente(self):
        df = pd.DataFrame({'a': [None, None, 1, 1, 1, 1, 1, 1, 1, 1]})
        result = self.enhancer._analyze_missing_patterns(df)
        assert result['patterns']['a']['percentage'] == pytest.approx(20.0)

    def test_patron_random_cuando_menos_10_pct(self):
        # 1 NaN en 20 filas = 5% → 'random'
        data = [None] + [1] * 19
        df = pd.DataFrame({'a': data})
        result = self.enhancer._analyze_missing_patterns(df)
        assert result['patterns']['a']['pattern'] == 'random'

    def test_patron_systematic_cuando_mas_10_pct(self):
        # 5 NaN en 20 filas = 25% → 'systematic'
        data = [None] * 5 + [1] * 15
        df = pd.DataFrame({'a': data})
        result = self.enhancer._analyze_missing_patterns(df)
        assert result['patterns']['a']['pattern'] == 'systematic'


# ─────────────────────────────────────────────────────────────────────────────
# _detect_tennis_business_anomalies
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectTennisBusinessAnomalies:
    """Reglas de negocio específicas de tenis detectan filas inválidas."""

    def setup_method(self):
        from Intelligent_ml_enhancer import IntelligentMLEnhancer
        self.enhancer = IntelligentMLEnhancer()

    def _df(self, **kwargs):
        base = {'p1_ranking': 50, 'p2_ranking': 100, 'p1_elo': 1800, 'p2_elo': 1700}
        base.update(kwargs)
        return pd.DataFrame([base])

    def test_ranking_valido_no_es_anomalia(self):
        df = self._df(p1_ranking=50)
        anomalies = self.enhancer._detect_tennis_business_anomalies(df)
        assert 0 not in anomalies

    def test_ranking_cero_es_anomalia(self):
        df = self._df(p1_ranking=0)
        anomalies = self.enhancer._detect_tennis_business_anomalies(df)
        assert 0 in anomalies

    def test_ranking_mayor_5000_es_anomalia(self):
        df = self._df(p2_ranking=5001)
        anomalies = self.enhancer._detect_tennis_business_anomalies(df)
        assert 0 in anomalies

    def test_elo_valido_no_es_anomalia(self):
        df = self._df(p1_elo=1800)
        anomalies = self.enhancer._detect_tennis_business_anomalies(df)
        assert 0 not in anomalies

    def test_elo_menor_1000_es_anomalia(self):
        df = self._df(p1_elo=500)
        anomalies = self.enhancer._detect_tennis_business_anomalies(df)
        assert 0 in anomalies

    def test_elo_mayor_3000_es_anomalia(self):
        df = self._df(p2_elo=3500)
        anomalies = self.enhancer._detect_tennis_business_anomalies(df)
        assert 0 in anomalies

    def test_edad_valida_no_es_anomalia(self):
        df = pd.DataFrame([{'p1_edad': 25, 'p2_edad': 30}])
        anomalies = self.enhancer._detect_tennis_business_anomalies(df)
        assert 0 not in anomalies

    def test_edad_menor_15_es_anomalia(self):
        df = pd.DataFrame([{'p1_edad': 12}])
        anomalies = self.enhancer._detect_tennis_business_anomalies(df)
        assert 0 in anomalies

    def test_ranking_diff_extrema_es_anomalia(self):
        df = pd.DataFrame([{'ranking_diff': 4500}])
        anomalies = self.enhancer._detect_tennis_business_anomalies(df)
        assert 0 in anomalies

    def test_df_sin_columnas_relevantes_no_lanza(self):
        df = pd.DataFrame([{'otra_columna': 1}])
        anomalies = self.enhancer._detect_tennis_business_anomalies(df)
        assert anomalies == []


# ─────────────────────────────────────────────────────────────────────────────
# _consolidate_anomaly_detection
# ─────────────────────────────────────────────────────────────────────────────

class TestConsolidateAnomalyDetection:
    """Consolida resultados de 4 métodos de detección de anomalías."""

    def setup_method(self):
        from Intelligent_ml_enhancer import IntelligentMLEnhancer
        self.enhancer = IntelligentMLEnhancer()

    def test_regla_negocio_siempre_incluida(self):
        # índice 5 solo aparece en business → debe incluirse
        result = self.enhancer._consolidate_anomaly_detection([], [], [], [5])
        assert 5 in result

    def test_consenso_dos_metodos_estadisticos(self):
        # índice 3 aparece en isolation + lof → debe incluirse
        result = self.enhancer._consolidate_anomaly_detection([3], [3], [], [])
        assert 3 in result

    def test_un_solo_metodo_estadistico_no_incluido(self):
        # índice 7 solo en isolation → NO debe incluirse
        result = self.enhancer._consolidate_anomaly_detection([7], [], [], [])
        assert 7 not in result

    def test_sin_anomalias_devuelve_vacio(self):
        result = self.enhancer._consolidate_anomaly_detection([], [], [], [])
        assert result == []

    def test_no_duplicados_en_resultado(self):
        # índice 1 en business y en 2 estadísticos
        result = self.enhancer._consolidate_anomaly_detection([1], [1], [], [1])
        assert result.count(1) == 1


# ─────────────────────────────────────────────────────────────────────────────
# _determine_synthetic_strategy
# ─────────────────────────────────────────────────────────────────────────────

class TestDetermineSyntheticStrategy:
    """Selecciona la estrategia SMOTE correcta según características del dataset."""

    def setup_method(self):
        from Intelligent_ml_enhancer import IntelligentMLEnhancer
        self.enhancer = IntelligentMLEnhancer()

    def _make(self, n_class0, n_class1, n_features=5):
        X = pd.DataFrame(np.random.rand(n_class0 + n_class1, n_features))
        y = pd.Series([0] * n_class0 + [1] * n_class1)
        return X, y

    def test_muy_desbalanceado_usa_adasyn(self):
        X, y = self._make(100, 5)   # ratio 20:1
        assert self.enhancer._determine_synthetic_strategy(X, y) == 'ADASYN'

    def test_moderadamente_desbalanceado_usa_borderline(self):
        X, y = self._make(40, 10)   # ratio 4:1
        assert self.enhancer._determine_synthetic_strategy(X, y) == 'BorderlineSMOTE'

    def test_dataset_pequeño_usa_smote_conservative(self):
        X, y = self._make(40, 40)   # ratio 1:1, n<100
        assert self.enhancer._determine_synthetic_strategy(X, y) == 'SMOTE_conservative'

    def test_caso_general_usa_smote(self):
        X, y = self._make(100, 100)  # balanceado, n>=100
        assert self.enhancer._determine_synthetic_strategy(X, y) == 'SMOTE'


# ─────────────────────────────────────────────────────────────────────────────
# get_enhancement_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestGetEnhancementSummary:
    """get_enhancement_summary reporta estado correcto según historial."""

    def setup_method(self):
        from Intelligent_ml_enhancer import IntelligentMLEnhancer
        self.enhancer = IntelligentMLEnhancer()

    def test_sin_historial_devuelve_no_enhancement(self):
        result = self.enhancer.get_enhancement_summary()
        assert result['status'] == 'no_enhancement_performed'

    def test_con_historial_devuelve_summary_completo(self):
        self.enhancer.enhancement_history = [{
            'processing_time': 1.5,
            'original_shape': (100, 10),
            'final_shape': (150, 8),
            'phases_completed': ['imputation', 'anomaly', 'features', 'synthetic'],
            'final_improvements': {'size_change_pct': 50.0}
        }]
        result = self.enhancer.get_enhancement_summary()
        assert 'dataset_transformation' in result
        assert result['dataset_transformation']['original_shape'] == (100, 10)
        assert result['phases_success'] == 4

    def test_cuatro_fases_overall_success_true(self):
        self.enhancer.enhancement_history = [{
            'processing_time': 2.0,
            'original_shape': (100, 10),
            'final_shape': (120, 9),
            'phases_completed': ['a', 'b', 'c', 'd'],
            'final_improvements': {'size_change_pct': 20.0}
        }]
        result = self.enhancer.get_enhancement_summary()
        assert result['overall_success'] is True

    def test_menos_cuatro_fases_overall_success_false(self):
        self.enhancer.enhancement_history = [{
            'processing_time': 1.0,
            'original_shape': (100, 10),
            'final_shape': (100, 9),
            'phases_completed': ['a', 'b', 'c'],
            'final_improvements': {'size_change_pct': 0.0}
        }]
        result = self.enhancer.get_enhancement_summary()
        assert result['overall_success'] is False
