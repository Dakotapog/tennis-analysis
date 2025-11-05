"""
🗂️ File Utils - Utilidades para manejo de archivos JSON
Funciones para buscar, analizar y seleccionar archivos JSON
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def find_all_json_files(directory: str = ".", pattern: str = "*.json") -> List[Path]:
    """
    🔍 Buscar todos los archivos JSON en un directorio.
    
    Args:
        directory: Directorio donde buscar
        pattern: Patrón de búsqueda (default: *.json)
    
    Returns:
        list: Lista de rutas a archivos JSON encontrados
    """
    search_dir = Path(directory)
    
    if not search_dir.exists():
        logger.warning(f"⚠️ Directorio no existe: {directory}")
        return []
    
    json_files = list(search_dir.glob(pattern))
    logger.info(f"🔍 Encontrados {len(json_files)} archivos JSON en {directory}")
    
    return json_files


def analyze_json_structure(json_file: Path) -> Dict:
    """
    🔬 Analizar la estructura de un archivo JSON.
    
    Args:
        json_file: Ruta al archivo JSON
    
    Returns:
        dict: Análisis de la estructura del archivo
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        analysis = {
            'file_path': str(json_file),
            'file_name': json_file.name,
            'file_size': json_file.stat().st_size,
            'is_valid': True,
            'type': type(data).__name__,
            'match_count': 0,
            'has_match_urls': False,
            'tournaments': []
        }
        
        # Analizar estructura según tipo
        if isinstance(data, dict):
            analysis['tournaments'] = list(data.keys())
            
            # Contar partidos totales
            total_matches = 0
            matches_with_urls = 0
            
            for tournament_name, matches in data.items():
                if isinstance(matches, list):
                    total_matches += len(matches)
                    matches_with_urls += sum(1 for m in matches if isinstance(m, dict) and m.get('match_url'))
            
            analysis['match_count'] = total_matches
            analysis['has_match_urls'] = matches_with_urls > 0
            analysis['matches_with_urls'] = matches_with_urls
            
        elif isinstance(data, list):
            analysis['match_count'] = len(data)
            analysis['has_match_urls'] = any(
                isinstance(m, dict) and m.get('match_url') for m in data
            )
            analysis['matches_with_urls'] = sum(
                1 for m in data if isinstance(m, dict) and m.get('match_url')
            )
        
        return analysis
    
    except json.JSONDecodeError as e:
        logger.error(f"❌ Error decodificando JSON {json_file}: {e}")
        return {
            'file_path': str(json_file),
            'file_name': json_file.name,
            'is_valid': False,
            'error': str(e)
        }
    except Exception as e:
        logger.error(f"❌ Error analizando {json_file}: {e}")
        return {
            'file_path': str(json_file),
            'file_name': json_file.name,
            'is_valid': False,
            'error': str(e)
        }


def validate_json_file(json_file: Path, min_matches: int = 1) -> Tuple[bool, str]:
    """
    ✅ Validar que un archivo JSON sea adecuado para procesamiento.
    
    Args:
        json_file: Ruta al archivo JSON
        min_matches: Número mínimo de partidos requeridos
    
    Returns:
        tuple: (es_válido, mensaje)
    """
    if not json_file.exists():
        return False, f"Archivo no existe: {json_file}"
    
    analysis = analyze_json_structure(json_file)
    
    if not analysis.get('is_valid'):
        return False, f"JSON inválido: {analysis.get('error', 'Error desconocido')}"
    
    match_count = analysis.get('match_count', 0)
    if match_count < min_matches:
        return False, f"Partidos insuficientes: {match_count} < {min_matches}"
    
    if not analysis.get('has_match_urls'):
        return False, "No se encontraron URLs de partidos"
    
    return True, f"✅ Válido: {match_count} partidos, {analysis.get('matches_with_urls', 0)} con URLs"


def select_best_json_file(directory: str = ".", 
                          pattern: str = "*.json",
                          auto_select: bool = True) -> Optional[str]:
    """
    🎯 Seleccionar el mejor archivo JSON para procesamiento.
    
    Args:
        directory: Directorio donde buscar
        pattern: Patrón de búsqueda
        auto_select: Si True, selecciona automáticamente el mejor
    
    Returns:
        str or None: Ruta al archivo seleccionado
    """
    json_files = find_all_json_files(directory, pattern)
    
    if not json_files:
        logger.error("❌ No se encontraron archivos JSON")
        return None
    
    # Analizar todos los archivos
    valid_files = []
    
    for json_file in json_files:
        is_valid, message = validate_json_file(json_file)
        
        if is_valid:
            analysis = analyze_json_structure(json_file)
            valid_files.append({
                'path': json_file,
                'analysis': analysis,
                'message': message
            })
            logger.info(f"✅ {json_file.name}: {message}")
        else:
            logger.warning(f"⚠️ {json_file.name}: {message}")
    
    if not valid_files:
        logger.error("❌ No se encontraron archivos JSON válidos")
        return None
    
    # Ordenar por número de partidos con URLs (descendente)
    valid_files.sort(
        key=lambda x: x['analysis'].get('matches_with_urls', 0),
        reverse=True
    )
    
    if auto_select:
        selected = valid_files[0]
        logger.info(f"🎯 Seleccionado automáticamente: {selected['path'].name}")
        logger.info(f"   {selected['message']}")
        return str(selected['path'])
    
    # Modo interactivo
    print("\n📋 Archivos JSON disponibles:")
    print("=" * 80)
    
    for i, file_info in enumerate(valid_files, 1):
        analysis = file_info['analysis']
        print(f"\n{i}. {file_info['path'].name}")
        print(f"   Partidos: {analysis.get('match_count', 0)}")
        print(f"   Con URLs: {analysis.get('matches_with_urls', 0)}")
        print(f"   Tamaño: {analysis.get('file_size', 0) / 1024:.1f} KB")
        
        if analysis.get('tournaments'):
            print(f"   Torneos: {len(analysis['tournaments'])}")
    
    print("\n" + "=" * 80)
    
    while True:
        try:
            choice = input(f"\nSelecciona un archivo (1-{len(valid_files)}) o Enter para el mejor: ").strip()
            
            if not choice:
                selected = valid_files[0]
                print(f"✅ Seleccionado: {selected['path'].name}")
                return str(selected['path'])
            
            index = int(choice) - 1
            
            if 0 <= index < len(valid_files):
                selected = valid_files[index]
                print(f"✅ Seleccionado: {selected['path'].name}")
                return str(selected['path'])
            else:
                print(f"⚠️ Número inválido. Elige entre 1 y {len(valid_files)}")
        
        except ValueError:
            print("⚠️ Entrada inválida. Ingresa un número o presiona Enter")
        except KeyboardInterrupt:
            print("\n❌ Selección cancelada")
            return None


def get_json_summary(json_file: Path) -> str:
    """
    📊 Obtener resumen legible de un archivo JSON.
    
    Args:
        json_file: Ruta al archivo JSON
    
    Returns:
        str: Resumen formateado
    """
    analysis = analyze_json_structure(json_file)
    
    if not analysis.get('is_valid'):
        return f"❌ Archivo inválido: {analysis.get('error')}"
    
    summary = [
        f"📄 {analysis['file_name']}",
        f"   Tipo: {analysis['type']}",
        f"   Tamaño: {analysis['file_size'] / 1024:.1f} KB",
        f"   Partidos: {analysis['match_count']}",
        f"   Con URLs: {analysis.get('matches_with_urls', 0)}",
    ]
    
    if analysis.get('tournaments'):
        summary.append(f"   Torneos: {len(analysis['tournaments'])}")
    
    return "\n".join(summary)