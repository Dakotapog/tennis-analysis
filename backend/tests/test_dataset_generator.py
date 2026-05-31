"""
Tests para generar_dataset_plus.py (Nodo-04)
Cubre: Bug 1 KNN shape mismatch + Bug 2 SmartLogger.error/warning.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import logging


# ─────────────────────────────────────────────────────────────────────────────
# SmartLogger — Bug 2: métodos .error() y .warning() deben existir
# ─────────────────────────────────────────────────────────────────────────────

class TestSmartLogger:

    def setup_method(self):
        """Importa SmartLogger sin side-effects de archivos."""
        from generar_dataset_plus import SmartLogger
        self.SmartLogger = SmartLogger

    def test_error_method_exists(self):
        """SmartLogger debe tener .error() — Bug 2 fix."""
        logger = self.SmartLogger()
        assert hasattr(logger, 'error'), "SmartLogger carece del método .error()"

    def test_warning_method_exists(self):
        """SmartLogger debe tener .warning() — Bug 2 fix."""
        logger = self.SmartLogger()
        assert hasattr(logger, 'warning'), "SmartLogger carece del método .warning()"

    def test_error_is_callable(self):
        """SmartLogger.error() debe ser callable sin excepción."""
        logger = self.SmartLogger()
        logger.error("test error message")  # no debe lanzar AttributeError

    def test_warning_is_callable(self):
        """SmartLogger.warning() debe ser callable sin excepción."""
        logger = self.SmartLogger()
        logger.warning("test warning message")  # no debe lanzar AttributeError

    def test_error_delegates_to_python_logger(self, caplog):
        """SmartLogger.error() debe delegar a logging.Logger.error."""
        logger = self.SmartLogger()
        with caplog.at_level(logging.ERROR):
            logger.error("mensaje de error")
        assert any("ERROR" in r.levelname or "error" in r.message.lower()
                   for r in caplog.records)

    def test_warning_delegates_to_python_logger(self, caplog):
        """SmartLogger.warning() debe delegar a logging.Logger.warning."""
        logger = self.SmartLogger()
        with caplog.at_level(logging.WARNING):
            logger.warning("mensaje de warning")
        assert any("WARN" in r.levelname or "warn" in r.message.lower()
                   for r in caplog.records)

    def test_existing_methods_intact(self):
        """Los 4 métodos originales no fueron tocados."""
        logger = self.SmartLogger()
        assert hasattr(logger, 'critical_alert')
        assert hasattr(logger, 'ml_warning')
        assert hasattr(logger, 'success')
        assert hasattr(logger, 'progress')


# ─────────────────────────────────────────────────────────────────────────────
# _intelligent_imputation — Bug 1: KNN no debe crashear por shape mismatch
# ─────────────────────────────────────────────────────────────────────────────

class TestIntelligentImputation:
    """Prueba _intelligent_imputation directamente usando un objeto mínimo."""

    def _make_generator(self):
        """Crea instancia mínima de IntelligentDatasetGenerator para tests."""
        from generar_dataset_plus import IntelligentDatasetGenerator, SmartLogger
        gen = object.__new__(IntelligentDatasetGenerator)
        gen.logger = SmartLogger()
        gen.target_column = 'actual_winner'
        return gen

    def test_no_nans_retorna_df_intacto(self):
        """Sin NaN → retorna el mismo df sin ejecutar KNN."""
        gen = self._make_generator()
        df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0], 'c': ['x', 'y']})
        result = gen._intelligent_imputation(df)
        pd.testing.assert_frame_equal(result, df)

    def test_con_nans_imputa_sin_crash(self):
        """Con NaN en columnas numéricas → KNN imputa sin ValueError de shape."""
        gen = self._make_generator()
        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, 5.0],
            'b': [2.0, 3.0, np.nan, 5.0, 6.0],
            'c': ['x', 'y', 'z', 'w', 'v'],  # columna no-numérica
        })
        result = gen._intelligent_imputation(df)
        assert result[['a', 'b']].isnull().sum().sum() == 0

    def test_columnas_no_numericas_preservadas(self):
        """Columnas string/object no deben ser alteradas por la imputación."""
        gen = self._make_generator()
        df = pd.DataFrame({
            'num1': [1.0, np.nan, 3.0, 4.0, 5.0],
            'num2': [2.0, 3.0, 4.0, 5.0, 6.0],
            'label': ['a', 'b', 'c', 'd', 'e'],
        })
        result = gen._intelligent_imputation(df)
        assert list(result['label']) == list(df['label'])

    def test_columnas_numericas_en_resultado(self):
        """Resultado debe contener todas las columnas del df original."""
        gen = self._make_generator()
        df = pd.DataFrame({
            'x': [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
            'y': [np.nan, 2.0, 3.0, 4.0, 5.0, 6.0],
            'cat': ['a', 'b', 'c', 'd', 'e', 'f'],
        })
        result = gen._intelligent_imputation(df)
        assert set(result.columns) == set(df.columns)

    def test_shape_preservada(self):
        """El número de filas y columnas no cambia tras la imputación."""
        gen = self._make_generator()
        n_rows, n_cols = 20, 10
        data = np.random.rand(n_rows, n_cols)
        data[0, 0] = np.nan
        df = pd.DataFrame(data, columns=[f'f{i}' for i in range(n_cols)])
        result = gen._intelligent_imputation(df)
        assert result.shape == (n_rows, n_cols)

    def test_knn_no_shape_mismatch_con_columnas_mixtas(self):
        """Reproduce el escenario Bug 1: df con columnas numéricas y no-numéricas."""
        gen = self._make_generator()
        # 71 columnas numéricas + 8 columnas no-numéricas = 79 columnas totales
        num_data = {f'num_{i}': np.random.rand(10) for i in range(71)}
        num_data['num_0'][0] = np.nan  # introducir al menos un NaN
        cat_data = {f'cat_{i}': ['val'] * 10 for i in range(8)}
        df = pd.DataFrame({**num_data, **cat_data})
        # No debe lanzar ValueError: Shape (10,71) vs (10,79)
        result = gen._intelligent_imputation(df)
        assert result.shape == (10, 79)
        assert result.isnull().sum().sum() == 0

    def test_indice_preservado(self):
        """El índice del DataFrame original se mantiene tras la imputación."""
        gen = self._make_generator()
        idx = [100, 200, 300, 400, 500]
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0, 4.0, 5.0]}, index=idx)
        result = gen._intelligent_imputation(df)
        assert list(result.index) == idx
