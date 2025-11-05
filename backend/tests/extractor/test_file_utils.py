"""
🧪 Tests para file_utils
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from scraping.file_utils import (
    find_all_json_files,
    analyze_json_structure,
    validate_json_file,
    select_best_json_file,
    get_json_summary
)


class TestFindAllJsonFiles:
    """Tests para find_all_json_files."""
    
    def test_find_json_files_in_directory(self, tmp_path):
        """Test: Encontrar archivos JSON en directorio."""
        # Crear archivos de prueba
        (tmp_path / "test1.json").write_text("{}")
        (tmp_path / "test2.json").write_text("{}")
        (tmp_path / "test.txt").write_text("not json")
        
        result = find_all_json_files(str(tmp_path))
        
        assert len(result) == 2
        assert all(f.suffix == '.json' for f in result)
    
    def test_find_json_files_empty_directory(self, tmp_path):
        """Test: Directorio vacío retorna lista vacía."""
        result = find_all_json_files(str(tmp_path))
        
        assert result == []
    
    def test_find_json_files_nonexistent_directory(self):
        """Test: Directorio inexistente retorna lista vacía."""
        result = find_all_json_files("/nonexistent/path/that/does/not/exist")
        
        assert result == []
    
    def test_find_json_files_with_pattern(self, tmp_path):
        """Test: Buscar con patrón específico."""
        (tmp_path / "matches_2024.json").write_text("{}")
        (tmp_path / "other.json").write_text("{}")
        
        result = find_all_json_files(str(tmp_path), pattern="matches_*.json")
        
        assert len(result) == 1
        assert result[0].name == "matches_2024.json"


class TestAnalyzeJsonStructure:
    """Tests para analyze_json_structure."""
    
    def test_analyze_dict_structure_with_matches(self, tmp_path):
        """Test: Analizar estructura de diccionario con partidos."""
        json_file = tmp_path / "test.json"
        data = {
            'ATP: Test Tournament (Spain), clay': [
                {
                    'jugador1': 'Player A',
                    'jugador2': 'Player B',
                    'match_url': 'https://test.com/match1'
                },
                {
                    'jugador1': 'Player C',
                    'jugador2': 'Player D',
                    'match_url': 'https://test.com/match2'
                }
            ]
        }
        json_file.write_text(json.dumps(data))
        
        result = analyze_json_structure(json_file)
        
        assert result['is_valid'] is True
        assert result['type'] == 'dict'
        assert result['match_count'] == 2
        assert result['has_match_urls'] is True
        assert result['matches_with_urls'] == 2
        assert len(result['tournaments']) == 1
    
    def test_analyze_list_structure(self, tmp_path):
        """Test: Analizar estructura de lista."""
        json_file = tmp_path / "test.json"
        data = [
            {'jugador1': 'A', 'match_url': 'http://test1.com'},
            {'jugador1': 'B', 'match_url': 'http://test2.com'}
        ]
        json_file.write_text(json.dumps(data))
        
        result = analyze_json_structure(json_file)
        
        assert result['is_valid'] is True
        assert result['type'] == 'list'
        assert result['match_count'] == 2
        assert result['has_match_urls'] is True
    
    def test_analyze_matches_without_urls(self, tmp_path):
        """Test: Partidos sin URLs."""
        json_file = tmp_path / "test.json"
        data = {
            'Tournament': [
                {'jugador1': 'A', 'jugador2': 'B'}
            ]
        }
        json_file.write_text(json.dumps(data))
        
        result = analyze_json_structure(json_file)
        
        assert result['is_valid'] is True
        assert result['match_count'] == 1
        assert result['has_match_urls'] is False
        assert result['matches_with_urls'] == 0
    
    def test_analyze_invalid_json(self, tmp_path):
        """Test: JSON inválido."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("{ invalid json }")
        
        result = analyze_json_structure(json_file)
        
        assert result['is_valid'] is False
        assert 'error' in result
    
    def test_analyze_empty_dict(self, tmp_path):
        """Test: Diccionario vacío."""
        json_file = tmp_path / "empty.json"
        json_file.write_text("{}")
        
        result = analyze_json_structure(json_file)
        
        assert result['is_valid'] is True
        assert result['match_count'] == 0
        assert result['tournaments'] == []
    
    def test_analyze_empty_list(self, tmp_path):
        """Test: Lista vacía."""
        json_file = tmp_path / "empty.json"
        json_file.write_text("[]")
        
        result = analyze_json_structure(json_file)
        
        assert result['is_valid'] is True
        assert result['type'] == 'list'
        assert result['match_count'] == 0


class TestValidateJsonFile:
    """Tests para validate_json_file."""
    
    def test_validate_valid_file(self, tmp_path):
        """Test: Validar archivo válido."""
        json_file = tmp_path / "valid.json"
        data = {
            'Tournament': [
                {'jugador1': 'A', 'match_url': 'http://test.com'}
            ]
        }
        json_file.write_text(json.dumps(data))
        
        is_valid, message = validate_json_file(json_file)
        
        assert is_valid is True
        assert "Válido" in message
    
    def test_validate_nonexistent_file(self, tmp_path):
        """Test: Archivo inexistente."""
        json_file = tmp_path / "nonexistent.json"
        
        is_valid, message = validate_json_file(json_file)
        
        assert is_valid is False
        assert "no existe" in message
    
    def test_validate_invalid_json(self, tmp_path):
        """Test: JSON inválido."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("{ bad json ")
        
        is_valid, message = validate_json_file(json_file)
        
        assert is_valid is False
        assert "inválido" in message
    
    def test_validate_insufficient_matches(self, tmp_path):
        """Test: Partidos insuficientes."""
        json_file = tmp_path / "few.json"
        data = {'Tournament': []}
        json_file.write_text(json.dumps(data))
        
        is_valid, message = validate_json_file(json_file, min_matches=1)
        
        assert is_valid is False
        assert "insuficientes" in message
    
    def test_validate_no_match_urls(self, tmp_path):
        """Test: Sin URLs de partidos."""
        json_file = tmp_path / "no_urls.json"
        data = {
            'Tournament': [
                {'jugador1': 'A', 'jugador2': 'B'}
            ]
        }
        json_file.write_text(json.dumps(data))
        
        is_valid, message = validate_json_file(json_file)
        
        assert is_valid is False
        assert "URLs" in message
    
    def test_validate_with_custom_min_matches(self, tmp_path):
        """Test: Validación con mínimo personalizado."""
        json_file = tmp_path / "test.json"
        data = {
            'Tournament': [
                {'match_url': 'http://test1.com'},
                {'match_url': 'http://test2.com'}
            ]
        }
        json_file.write_text(json.dumps(data))
        
        is_valid, message = validate_json_file(json_file, min_matches=5)
        
        assert is_valid is False
        assert "2 < 5" in message


class TestSelectBestJsonFile:
    """Tests para select_best_json_file."""
    
    def test_select_best_auto_mode(self, tmp_path):
        """Test: Selección automática del mejor archivo."""
        # Crear archivos con diferentes cantidades de partidos
        file1 = tmp_path / "small.json"
        file1.write_text(json.dumps({
            'T1': [{'match_url': 'http://test.com'}]
        }))
        
        file2 = tmp_path / "large.json"
        file2.write_text(json.dumps({
            'T1': [
                {'match_url': 'http://test1.com'},
                {'match_url': 'http://test2.com'},
                {'match_url': 'http://test3.com'}
            ]
        }))
        
        result = select_best_json_file(str(tmp_path), auto_select=True)
        
        assert result is not None
        assert 'large.json' in result
    
    def test_select_best_no_valid_files(self, tmp_path):
        """Test: No hay archivos válidos."""
        # Crear archivo inválido
        invalid = tmp_path / "invalid.json"
        invalid.write_text("{ bad json ")
        
        result = select_best_json_file(str(tmp_path), auto_select=True)
        
        assert result is None
    
    def test_select_best_empty_directory(self, tmp_path):
        """Test: Directorio vacío."""
        result = select_best_json_file(str(tmp_path), auto_select=True)
        
        assert result is None
    
    def test_select_best_filters_by_pattern(self, tmp_path):
        """Test: Filtrar por patrón."""
        (tmp_path / "matches_2024.json").write_text(json.dumps({
            'T1': [{'match_url': 'http://test.com'}]
        }))
        (tmp_path / "other.json").write_text(json.dumps({
            'T1': [{'match_url': 'http://test.com'}]
        }))
        
        result = select_best_json_file(
            str(tmp_path), 
            pattern="matches_*.json",
            auto_select=True
        )
        
        assert result is not None
        assert 'matches_2024.json' in result
    
    def test_select_best_prioritizes_more_urls(self, tmp_path):
        """Test: Prioriza archivos con más URLs."""
        file1 = tmp_path / "file1.json"
        file1.write_text(json.dumps({
            'T1': [
                {'match_url': 'http://1.com'},
                {'other': 'data'}  # Sin URL
            ]
        }))
        
        file2 = tmp_path / "file2.json"
        file2.write_text(json.dumps({
            'T1': [
                {'match_url': 'http://1.com'},
                {'match_url': 'http://2.com'},
                {'match_url': 'http://3.com'}
            ]
        }))
        
        result = select_best_json_file(str(tmp_path), auto_select=True)
        
        assert 'file2.json' in result
    
    @patch('builtins.input', return_value='1')
    def test_select_best_interactive_mode(self, mock_input, tmp_path):
        """Test: Modo interactivo."""
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps({
            'T1': [{'match_url': 'http://test.com'}]
        }))
        
        result = select_best_json_file(str(tmp_path), auto_select=False)
        
        assert result is not None
        assert 'test.json' in result
    
    @patch('builtins.input', return_value='')
    def test_select_best_interactive_default(self, mock_input, tmp_path):
        """Test: Modo interactivo con selección por defecto."""
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps({
            'T1': [{'match_url': 'http://test.com'}]
        }))
        
        result = select_best_json_file(str(tmp_path), auto_select=False)
        
        assert result is not None
    
    @patch('builtins.input', side_effect=KeyboardInterrupt)
    def test_select_best_interactive_cancelled(self, mock_input, tmp_path):
        """Test: Modo interactivo cancelado."""
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps({
            'T1': [{'match_url': 'http://test.com'}]
        }))
        
        result = select_best_json_file(str(tmp_path), auto_select=False)
        
        assert result is None


class TestGetJsonSummary:
    """Tests para get_json_summary."""
    
    def test_summary_valid_file(self, tmp_path):
        """Test: Resumen de archivo válido."""
        json_file = tmp_path / "test.json"
        data = {
            'Tournament A': [
                {'match_url': 'http://test1.com'},
                {'match_url': 'http://test2.com'}
            ],
            'Tournament B': [
                {'match_url': 'http://test3.com'}
            ]
        }
        json_file.write_text(json.dumps(data))
        
        summary = get_json_summary(json_file)
        
        assert 'test.json' in summary
        assert 'dict' in summary
        assert 'Partidos: 3' in summary
        assert 'Con URLs: 3' in summary
        assert 'Torneos: 2' in summary
    
    def test_summary_invalid_file(self, tmp_path):
        """Test: Resumen de archivo inválido."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("{ bad json ")
        
        summary = get_json_summary(json_file)
        
        assert 'inválido' in summary.lower()
        assert 'error' in summary.lower() or 'expecting' in summary.lower()
    
    def test_summary_list_structure(self, tmp_path):
        """Test: Resumen de estructura de lista."""
        json_file = tmp_path / "test.json"
        data = [
            {'match_url': 'http://test.com'}
        ]
        json_file.write_text(json.dumps(data))
        
        summary = get_json_summary(json_file)
        
        assert 'list' in summary
        assert 'Partidos: 1' in summary
    
    def test_summary_includes_file_size(self, tmp_path):
        """Test: Resumen incluye tamaño de archivo."""
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps({'data': 'test'}))
        
        summary = get_json_summary(json_file)
        
        assert 'KB' in summary or 'Tamaño' in summary


class TestEdgeCases:
    """Tests para casos especiales y edge cases."""
    
    def test_handle_nested_tournament_structure(self, tmp_path):
        """Test: Manejar estructura anidada compleja."""
        json_file = tmp_path / "nested.json"
        data = {
            'Tournament': {
                'matches': [
                    {'match_url': 'http://test.com'}
                ]
            }
        }
        json_file.write_text(json.dumps(data))
        
        result = analyze_json_structure(json_file)
        
        # Debería manejar estructura inesperada sin crash
        assert result['is_valid'] is True
    
    def test_handle_unicode_characters(self, tmp_path):
        """Test: Manejar caracteres Unicode."""
        json_file = tmp_path / "unicode.json"
        data = {
            'ATP: París (França), terre battue': [
                {
                    'jugador1': 'Nadál',
                    'match_url': 'http://test.com'
                }
            ]
        }
        json_file.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        
        result = analyze_json_structure(json_file)
        
        assert result['is_valid'] is True
        assert len(result['tournaments']) == 1
    
    def test_handle_very_large_file(self, tmp_path):
        """Test: Manejar archivo grande."""
        json_file = tmp_path / "large.json"
        data = {
            'Tournament': [
                {'match_url': f'http://test{i}.com'}
                for i in range(1000)
            ]
        }
        json_file.write_text(json.dumps(data))
        
        result = analyze_json_structure(json_file)
        
        assert result['is_valid'] is True
        assert result['match_count'] == 1000
    
    def test_handle_special_characters_in_filename(self, tmp_path):
        """Test: Manejar caracteres especiales en nombre de archivo."""
        json_file = tmp_path / "matches_2024-01-15_[final].json"
        data = {'T1': [{'match_url': 'http://test.com'}]}
        json_file.write_text(json.dumps(data))
        
        result = validate_json_file(json_file)
        
        assert result[0] is True