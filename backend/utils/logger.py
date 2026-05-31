"""
utils/logger.py — Fuente única de verdad para SmartLogger.

Unifica las dos implementaciones inline que existían en:
  - generar_dataset_plus.py  (sin parámetros, con file log)
  - Intelligent_ml_enhancer.py  (nombre explícito, sin file log)

Uso:
    from utils.logger import SmartLogger

    logger = SmartLogger()                              # básico
    logger = SmartLogger('IntelligentMLEnhancer')       # nombre explícito
    logger = SmartLogger(log_to_file=True)              # con archivo en logs/
"""

import logging
import os
import sys
from datetime import datetime


class SmartLogger:
    """Logger con niveles semánticos y formatos visuales para el pipeline de tenis."""

    def __init__(self, name: str = 'SmartLogger', level: int = logging.INFO,
                 log_to_file: bool = False):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if not self.logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(message)s', datefmt='%H:%M:%S'
            )

            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

            if log_to_file:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                os.makedirs('logs', exist_ok=True)
                file_handler = logging.FileHandler(
                    f'logs/intelligent_dataset_{timestamp}.log', encoding='utf-8'
                )
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

    # ── Métodos estándar ──────────────────────────────────────────────────────

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(f"⚠️  {msg}", *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(f"❌ {msg}", *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(f"🚨 {msg}", *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    # ── Métodos semánticos ────────────────────────────────────────────────────

    def success(self, msg):
        """Operación completada exitosamente."""
        self.info(f"✅ {msg}")

    def section(self, msg):
        """Encabezado visual de sección."""
        self.info("\n" + "=" * 60)
        self.info(f"🚀 {msg.upper()}")
        self.info("=" * 60)

    def progress(self, msg):
        """Progreso de una operación en curso."""
        self.info(f"⏳ {msg}...")

    def critical_alert(self, msg):
        """Alerta crítica que requiere atención inmediata (compatible con generar_dataset_plus)."""
        self.critical(f"CRÍTICO: {msg}")

    def ml_warning(self, msg):
        """Advertencia específica del pipeline de ML."""
        self.warning(f"🤖 ML: {msg}")
