"""
tests/test_aplicar_enhancer.py — Tests para aplicar_enhancer.py

Cubre:
  - OrchestratorConfig: valores por defecto críticos
  - MLDatasetOrchestrator._validate_base_dataset: lógica de validación (sin I/O)
  - AdvancedMLFormatter: routing y transformaciones pandas puras

Pipeline ML bloqueado hasta n≥100 — estos tests cubren lógica testeable
sin ejecutar el pipeline completo.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aplicar_enhancer import OrchestratorConfig, MLDatasetOrchestrator, AdvancedMLFormatter
from utils.logger import SmartLogger


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def logger():
    return SmartLogger(name='test_aplicar_enhancer')


@pytest.fixture
def orchestrator(tmp_path, logger):
    """MLDatasetOrchestrator con directorios en tmp y enhancer mockeado."""
    config = OrchestratorConfig()
    config.OUTPUT_BASE_DIR = str(tmp_path / 'ml_datasets')
    with patch('aplicar_enhancer.IntelligentMLEnhancer') as mock_enhancer:
        mock_enhancer.return_value = MagicMock()
        orch = MLDatasetOrchestrator(config=config)
        orch.logger = logger
    return orch


@pytest.fixture
def df_valido():
    """DataFrame con estructura válida para el pipeline ML."""
    np.random.seed(42)
    n = 150
    return pd.DataFrame({
        'ganador_real':    np.random.randint(0, 2, n),
        'p1_prob_win':     np.random.uniform(0.3, 0.8, n),
        'ranking_diff':    np.random.randint(-200, 200, n),
        'surface_clay':    np.random.randint(0, 2, n),
        'h2h_win_rate':    np.random.uniform(0.0, 1.0, n),
        'elo_diff':        np.random.uniform(-300, 300, n),
    })


@pytest.fixture
def formatter(logger):
    return AdvancedMLFormatter(logger=logger)


# ── TestOrchestratorConfig ─────────────────────────────────────────────────────

class TestOrchestratorConfig:
    def test_min_samples_es_100(self):
        assert OrchestratorConfig.MIN_SAMPLES == 100

    def test_min_features_es_5(self):
        assert OrchestratorConfig.MIN_FEATURES == 5

    def test_classification_target_correcto(self):
        assert OrchestratorConfig.CLASSIFICATION_TARGET == 'ganador_real'

    def test_min_accuracy_threshold(self):
        assert OrchestratorConfig.MIN_ACCURACY_THRESHOLD == 0.65


# ── TestValidateBaseDataset ────────────────────────────────────────────────────

class TestValidateBaseDataset:
    def test_dataset_valido_retorna_true(self, orchestrator, df_valido):
        assert orchestrator._validate_base_dataset(df_valido) is True

    def test_dataset_pequeno_retorna_false(self, orchestrator):
        df_pequeno = pd.DataFrame({
            'ganador_real': [0, 1, 0],
            'feature_a':    [1.0, 2.0, 3.0],
            'feature_b':    [0.5, 1.5, 2.5],
            'feature_c':    [3.0, 2.0, 1.0],
            'feature_d':    [1.1, 2.1, 3.1],
            'feature_e':    [0.1, 0.2, 0.3],
        })
        assert orchestrator._validate_base_dataset(df_pequeno) is False

    def test_dataset_pocas_features_retorna_false(self, orchestrator):
        df = pd.DataFrame({
            'ganador_real': np.random.randint(0, 2, 150),
            'solo_una':     np.random.uniform(0, 1, 150),
        })
        assert orchestrator._validate_base_dataset(df) is False

    def test_target_detectado_si_presente(self, orchestrator, df_valido):
        orchestrator._validate_base_dataset(df_valido)
        report = orchestrator.orchestration_report.get('base_validation', {})
        assert report.get('target_available') is True

    def test_target_no_detectado_si_ausente(self, orchestrator):
        df = pd.DataFrame({f'feat_{i}': np.random.uniform(0, 1, 150) for i in range(6)})
        orchestrator._validate_base_dataset(df)
        report = orchestrator.orchestration_report.get('base_validation', {})
        assert report.get('target_available') is False


# ── TestAdvancedMLFormatter ────────────────────────────────────────────────────

class TestAdvancedMLFormatter:
    def test_tipo_desconocido_devuelve_df_sin_cambios(self, formatter, df_valido):
        result = formatter.format_for_algorithm(df_valido, 'tipo_inexistente')
        assert result.shape == df_valido.shape

    def test_classification_escala_features_numericas(self, formatter, df_valido):
        result = formatter.format_for_algorithm(
            df_valido, 'classification', target_column='ganador_real'
        )
        # Después de StandardScaler las features deben tener media ≈ 0
        numeric_feats = [c for c in result.select_dtypes(include=np.number).columns
                         if c != 'ganador_real']
        for col in numeric_feats:
            assert abs(result[col].mean()) < 0.1, f"{col} no está centrado"

    def test_regression_elimina_outliers_extremos(self, formatter):
        df = pd.DataFrame({
            'feat': [1.0] * 100 + [1e6],  # outlier extremo
            'target': np.random.uniform(0, 1, 101),
        })
        result = formatter.format_for_algorithm(df, 'regression')
        assert result['feat'].max() < 1e5

    def test_classification_devuelve_mismo_numero_columnas(self, formatter, df_valido):
        result = formatter.format_for_algorithm(
            df_valido, 'classification', target_column='ganador_real'
        )
        assert result.shape[1] == df_valido.shape[1]
