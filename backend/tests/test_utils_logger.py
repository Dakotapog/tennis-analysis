"""Tests para utils/logger.py — SmartLogger (D-09 resuelto)."""
import logging
import os
import pytest
from unittest.mock import patch
from utils.logger import SmartLogger


class TestSmartLoggerInit:
    """Tests de inicialización."""

    def test_default_instantiation(self):
        logger = SmartLogger('test_default_init')
        assert logger is not None
        assert logger.logger.name == 'test_default_init'

    def test_default_level_is_info(self):
        logger = SmartLogger('test_level')
        assert logger.logger.level == logging.INFO

    def test_custom_level(self):
        logger = SmartLogger('test_debug_level', level=logging.DEBUG)
        assert logger.logger.level == logging.DEBUG

    def test_handlers_created(self):
        logger = SmartLogger('test_handlers_unique_xyz')
        assert len(logger.logger.handlers) >= 1

    def test_log_to_file_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        SmartLogger('test_file_logger_abc', log_to_file=True)
        log_files = list((tmp_path / 'logs').glob('*.log'))
        assert len(log_files) == 1


class TestSmartLoggerMethods:
    """Tests de todos los métodos públicos."""

    @pytest.fixture
    def logger(self):
        return SmartLogger('test_methods_suite')

    def test_has_info(self, logger):
        assert callable(logger.info)

    def test_has_warning(self, logger):
        assert callable(logger.warning)

    def test_has_error(self, logger):
        assert callable(logger.error)

    def test_has_critical(self, logger):
        assert callable(logger.critical)

    def test_has_debug(self, logger):
        assert callable(logger.debug)

    def test_has_success(self, logger):
        assert callable(logger.success)

    def test_has_section(self, logger):
        assert callable(logger.section)

    def test_has_progress(self, logger):
        assert callable(logger.progress)

    def test_has_critical_alert(self, logger):
        assert callable(logger.critical_alert)

    def test_has_ml_warning(self, logger):
        assert callable(logger.ml_warning)

    def test_success_delegates_to_info(self, logger):
        with patch.object(logger.logger, 'info') as mock_info:
            logger.success('done')
            mock_info.assert_called_once()
            assert '✅' in mock_info.call_args[0][0]

    def test_ml_warning_includes_ml_prefix(self, logger):
        with patch.object(logger.logger, 'warning') as mock_warn:
            logger.ml_warning('shape mismatch')
            mock_warn.assert_called_once()
            assert '🤖 ML' in mock_warn.call_args[0][0]

    def test_critical_alert_delegates_to_critical(self, logger):
        with patch.object(logger.logger, 'critical') as mock_crit:
            logger.critical_alert('pipeline down')
            mock_crit.assert_called_once()
            assert 'CRÍTICO' in mock_crit.call_args[0][0]

    def test_different_names_are_independent(self):
        a = SmartLogger('logger_a_unique')
        b = SmartLogger('logger_b_unique')
        assert a.logger is not b.logger
        assert a.logger.name != b.logger.name
