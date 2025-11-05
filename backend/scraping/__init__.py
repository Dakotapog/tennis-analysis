"""
🎾 Scraping Package - Módulos para extracción de datos H2H
Estructura modular con separación de responsabilidades
"""

from .browser_manager import BrowserManager
from .data_parser import DataParser
# H2HExtractor se importa directamente desde su módulo para evitar dependencias circulares
from .file_utils import (
    find_all_json_files,
    analyze_json_structure,
    select_best_json_file,
    validate_json_file
)

__all__ = [
    'BrowserManager',
    'DataParser',
    'find_all_json_files',
    'analyze_json_structure',
    'select_best_json_file',
    'validate_json_file'
]
