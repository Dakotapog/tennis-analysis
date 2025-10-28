"""
🎾 Scraping Package - Módulos para extracción de datos H2H
Estructura modular con separación de responsabilidades
"""

from .browser_manager import BrowserManager
from .data_parser import DataParser
from .h2h_extractor import H2HExtractor

__all__ = [
    'BrowserManager',
    'DataParser',
    'H2HExtractor'
]