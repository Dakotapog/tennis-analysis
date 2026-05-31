"""
🎾 H2H EXTRACTOR SECUENCIAL - PROCESAMIENTO DE 5 PARTIDOS CON ANÁLISIS DE RANKINGS
Procesa los primeros 5 partidos del JSON de forma secuencial
Extrae las 3 secciones H2H: Jugador1, Jugador2, Enfrentamientos Directos
VERSIÓN MEJORADA CON DETECCIÓN DE ARCHIVOS JSON EN CARPETA DATA
NUEVA FUNCIONALIDAD: ANÁLISIS DE RIVALIDADES TRANSITIVAS CON RANKINGS
"""

import json
import asyncio
import logging
import signal
import atexit
import psutil
import os
from datetime import datetime
from playwright.async_api import async_playwright
from contextlib import asynccontextmanager
import re
from pathlib import Path
import math
from extraer_ranking_atp_version2 import CompleteRankingScraper
import time

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from analysis import EloRatingSystem, RankingManager, RivalryAnalyzer

def verificar_sistema():
    """🔍 Verificaciones previas al inicio"""
    logger.info("🔍 Verificando sistema WSL...")
    
    # Verificar memoria disponible
    memory = psutil.virtual_memory()
    logger.info(f"💾 Memoria disponible: {memory.available / (1024**3):.1f}GB")
    
    if memory.available < 1024 * 1024 * 1024:  # 1GB
        logger.warning("⚠️ Poca memoria disponible (<1GB)")
    
    # Limpiar procesos zombie de Chrome
    killed_processes = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] and any(chrome_name in proc.info['name'].lower() 
                                       for chrome_name in ['chrome', 'chromium']):
                proc.kill()
                killed_processes += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if killed_processes > 0:
        logger.info(f"🧹 Eliminados {killed_processes} procesos Chrome/Chromium")
    
    # Verificar si estamos en WSL
    if os.path.exists('/proc/version'):
        try:
            with open('/proc/version', 'r') as f:
                version = f.read()
                if 'Microsoft' in version or 'WSL' in version:
                    logger.info("✅ Ejecutándose en WSL - Aplicando optimizaciones")
                    return True
        except:
            pass
    
    logger.info("✅ Sistema verificado")
    return False

def find_all_json_files():
    """🔍 Encontrar todos los archivos JSON en el directorio actual y carpeta data"""
    logger.info("🔍 Buscando archivos JSON en directorio actual y carpeta 'data'...")
    
    current_dir = Path('.')
    data_dir = Path('data')
    all_json_files = []
    
    # Buscar en directorio actual
    logger.info("   📂 Buscando en directorio actual...")
    for json_file in current_dir.glob('*.json'):
        file_stats = json_file.stat()
        file_info = {
            'filename': str(json_file),
            'size_mb': file_stats.st_size / (1024 * 1024),
            'modified_time': datetime.fromtimestamp(file_stats.st_mtime),
            'location': 'current'
        }
        all_json_files.append(file_info)
    
    # Buscar en carpeta data (si existe)
    if data_dir.exists() and data_dir.is_dir():
        logger.info("   📂 Buscando en carpeta 'data'...")
        for json_file in data_dir.glob('*.json'):
            file_stats = json_file.stat()
            file_info = {
                'filename': str(json_file),
                'size_mb': file_stats.st_size / (1024 * 1024),
                'modified_time': datetime.fromtimestamp(file_stats.st_mtime),
                'location': 'data'
            }
            all_json_files.append(file_info)
    else:
        logger.info("   📂 Carpeta 'data' no encontrada")
    
    # Ordenar por fecha de modificación (más reciente primero)
    all_json_files.sort(key=lambda x: x['modified_time'], reverse=True)
    
    logger.info(f"📂 Encontrados {len(all_json_files)} archivos JSON:")
    for i, file_info in enumerate(all_json_files, 1):
        location_icon = "📁" if file_info['location'] == 'data' else "📄"
        logger.info(f"   {i}. {location_icon} {file_info['filename']} ({file_info['size_mb']:.2f}MB) - {file_info['modified_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_json_files

def analyze_json_structure(filename):
    """🔬 Analizar la estructura de un archivo JSON para verificar si contiene partidos"""
    logger.info(f"🔬 Analizando estructura de {filename}...")
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        analysis = {
            'filename': filename,
            'is_valid': False,
            'total_matches': 0,
            'structure_type': 'unknown',
            'sample_keys': [],
            'has_match_urls': False,
            'sample_match': None
        }
        
        # Analizar estructura
        if isinstance(data, dict):
            analysis['sample_keys'] = list(data.keys())[:5]  # Primeras 5 claves
            
            # Buscar partidos en diferentes estructuras
            total_matches = 0
            sample_match = None
            has_urls = False
            
            for key, value in data.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            total_matches += 1
                            if not sample_match:
                                sample_match = item
                            if item.get('match_url'):
                                has_urls = True
                elif isinstance(value, dict):
                    if value.get('match_url') or value.get('jugador1'):
                        total_matches += 1
                        if not sample_match:
                            sample_match = value
                        if value.get('match_url'):
                            has_urls = True
            
            analysis['structure_type'] = 'dict_with_tournaments'
            analysis['total_matches'] = total_matches
            analysis['has_match_urls'] = has_urls
            analysis['sample_match'] = sample_match
            
        elif isinstance(data, list):
            analysis['total_matches'] = len(data)
            analysis['structure_type'] = 'list_of_matches'
            
            if data:
                analysis['sample_match'] = data[0]
                analysis['has_match_urls'] = any(
                    isinstance(item, dict) and item.get('match_url') 
                    for item in data[:5]  # Revisar primeros 5
                )
        
        # Determinar si es válido para nuestro procesamiento
        analysis['is_valid'] = (
            analysis['total_matches'] > 0 and 
            analysis['has_match_urls'] and
            analysis['sample_match'] is not None
        )
        
        logger.info(f"   📊 Tipo: {analysis['structure_type']}")
        logger.info(f"   🎾 Total partidos: {analysis['total_matches']}")
        logger.info(f"   🔗 Tiene URLs: {'✅' if analysis['has_match_urls'] else '❌'}")
        logger.info(f"   ✅ Válido: {'✅' if analysis['is_valid'] else '❌'}")
        
        if analysis['sample_match']:
            logger.info(f"   👥 Ejemplo: {analysis['sample_match'].get('jugador1', 'N/A')} vs {analysis['sample_match'].get('jugador2', 'N/A')}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"❌ Error analizando {filename}: {e}")
        return {
            'filename': filename,
            'is_valid': False,
            'error': str(e)
        }

def select_best_json_file():
    """🎯 Seleccionar el mejor archivo JSON automáticamente con prioridad en archivo específico"""
    logger.info("🎯 Seleccionando el mejor archivo JSON para procesamiento...")
    
    # Buscar específicamente el archivo zita_tennis_matches_20250725_223524.json en carpeta data
    specific_file = Path('data/zita_tennis_matches_20250725_223524.json')
    if specific_file.exists():
        logger.info(f"🎯 Archivo específico encontrado: {specific_file}")
        analysis = analyze_json_structure(str(specific_file))
        if analysis['is_valid']:
            logger.info(f"🏆 Usando archivo específico: {specific_file}")
            return str(specific_file)
        else:
            logger.warning("⚠️ Archivo específico no es válido para extracción H2H")
    
    # Si no se encuentra el archivo específico, buscar todos los archivos JSON
    json_files = find_all_json_files()
    
    if not json_files:
        logger.error("❌ No se encontraron archivos JSON en el directorio")
        return None
    
    # Analizar cada archivo JSON
    valid_files = []
    for file_info in json_files:
        analysis = analyze_json_structure(file_info['filename'])
        if analysis['is_valid']:
            analysis.update(file_info)  # Añadir info del archivo
            valid_files.append(analysis)
    
    if not valid_files:
        logger.error("❌ No se encontraron archivos JSON válidos con partidos y URLs")
        logger.info("📋 Archivos analizados:")
        for file_info in json_files:
            logger.info(f"   - {file_info['filename']}: No válido para extracción H2H")
        return None
    
    # Seleccionar el mejor archivo (más partidos y más reciente)
    best_file = max(valid_files, key=lambda x: (x['modified_time'], x['total_matches']))

    # T08-06: Advertencia si el archivo seleccionado no es el más reciente
    most_recent = max(valid_files, key=lambda x: x['modified_time'])
    if best_file['filename'] != most_recent['filename']:
        logger.warning(
            f"⚠️ ANOMALÍA: Archivo seleccionado no es el más reciente. "
            f"Seleccionado: {best_file['filename']} | "
            f"Más reciente: {most_recent['filename']} — verificar lógica de selección"
        )

    logger.info(f"🏆 Archivo seleccionado: {best_file['filename']}")
    logger.info(f"   📊 Partidos disponibles: {best_file['total_matches']}")
    logger.info(f"   📅 Modificado: {best_file['modified_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"   📏 Tamaño: {best_file['size_mb']:.2f}MB")
    logger.info(f"   📍 Ubicación: {best_file['location']}")
    
    return best_file['filename']
    
async def main():
    """🎯 Función principal — usa H2HExtractor modular (Nodo-07 Fase 2)"""
    from scraping.h2h_extractor import H2HExtractor

    logger.info("🎾 H2H EXTRACTOR MODULAR CON ANÁLISIS DE RIVALIDAD TRANSITIVA")
    logger.info("=" * 80)
    logger.info("🚀 Versión 5.0 - H2HExtractor modular (Strangler Fig Fase 2)")
    logger.info("=" * 80)

    extractor = H2HExtractor()

    try:
        # 1. Cargar partidos (incluye filtro Roland Garros + superficie)
        if not extractor.load_matches():
            logger.error("❌ No se pudieron cargar los partidos desde JSON")
            return

        # 2. Inicializar navegador y procesar partidos
        await extractor.setup()
        try:
            await extractor.run()
        finally:
            await extractor.cleanup()

        # 3. Guardar resultados
        output_file = extractor.save_results()

        if output_file:
            logger.info("🎉 PROCESAMIENTO COMPLETADO EXITOSAMENTE")
            logger.info(f"📁 Archivo de resultados: {output_file}")
        else:
            logger.error("❌ Error guardando los resultados.")

    except KeyboardInterrupt:
        logger.info("🛑 Procesamiento interrumpido por el usuario")
    except Exception as e:
        logger.error(f"❌ Error crítico en main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("🏁 Programa finalizado")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Programa interrumpido")
    except Exception as e:
        logger.error(f"❌ Error ejecutando programa: {e}")
        import traceback
        traceback.print_exc()
