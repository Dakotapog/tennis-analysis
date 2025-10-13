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

class EloRatingSystem:
    """
    Sistema de Rating ELO para jugadores de tenis.
    - Calcula la probabilidad de victoria esperada.
    - Actualiza los ratings después de un partido.
    """
    def __init__(self, k_factor=32, default_rating=1500):
        self.k_factor = k_factor
        self.default_rating = default_rating
        self.ratings = {}  # Ratings en memoria, no persistentes
        logger.info("ELO System inicializado (en memoria).")

    def get_rating(self, player_name):
        """Obtiene el rating de un jugador, o el rating por defecto si no existe."""
        return self.ratings.get(player_name, self.default_rating)

    def expected_score(self, rating1, rating2):
        """Calcula el puntaje esperado (probabilidad de victoria) para el jugador 1."""
        return 1 / (1 + 10 ** ((rating2 - rating1) / 400))

    def update_ratings(self, winner_name, loser_name):
        """
        Actualiza los ratings de dos jugadores después de un partido.
        NOTA: Esta función debería ser llamada por un proceso que procese resultados
        históricos y confirmados para mantener el sistema ELO actualizado.
        """
        winner_rating = self.get_rating(winner_name)
        loser_rating = self.get_rating(loser_name)

        expected_winner = self.expected_score(winner_rating, loser_rating)
        expected_loser = self.expected_score(loser_rating, winner_rating)

        # Actualizar ratings
        new_winner_rating = winner_rating + self.k_factor * (1 - expected_winner)
        new_loser_rating = loser_rating + self.k_factor * (0 - expected_loser)

        self.ratings[winner_name] = round(new_winner_rating)
        self.ratings[loser_name] = round(new_loser_rating)

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
    best_file = max(valid_files, key=lambda x: (x['total_matches'], x['modified_time']))
    
    logger.info(f"🏆 Archivo seleccionado: {best_file['filename']}")
    logger.info(f"   📊 Partidos disponibles: {best_file['total_matches']}")
    logger.info(f"   📅 Modificado: {best_file['modified_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"   📏 Tamaño: {best_file['size_mb']:.2f}MB")
    logger.info(f"   📍 Ubicación: {best_file['location']}")
    
    return best_file['filename']
    
class RankingManager:
    """Gestor de rankings de jugadores para análisis transitivo - SOLO LECTURA"""
    
    def __init__(self, force_read_only=True, max_age_days=8):
        """
        Args:
            force_read_only: Si True, nunca ejecuta scraping, solo lee archivos existentes
            max_age_days: Máxima antigüedad en días antes de considerar archivo obsoleto
        """
        self.force_read_only = force_read_only
        self.max_age_days = max_age_days
        self.rankings_data = {}
        self.atp_players = {}
        self.wta_players = {}
        self.metadata = {
            'atp_file': None,
            'wta_file': None,
            'atp_count': 0,
            'wta_count': 0,
            'last_updated': None,
            'read_only_mode': force_read_only
        }
        self.load_rankings()
    
    def load_rankings(self):
        """
        Cargar rankings desde los archivos JSON de ATP y WTA más recientes en la carpeta data.
        MODIFICADO: Ampliado a 8 días y modo solo lectura por defecto.
        """
        logger.info("Cargando rankings de jugadores ATP y WTA...")
        data_dir = Path('data')
        max_age_hours = self.max_age_days * 24  # Convertir días a horas

        try:
            # Búsqueda de archivos de ranking COMPLETE más recientes para ATP y WTA
            atp_complete_files = sorted(data_dir.glob('atp_rankings_complete_*.json'), reverse=True)
            wta_complete_files = sorted(data_dir.glob('wta_rankings_complete_*.json'), reverse=True)

            atp_file = None
            wta_file = None
            atp_needs_update = True
            wta_needs_update = True

            # Verificar archivo ATP
            if atp_complete_files:
                atp_file = atp_complete_files[0]
                file_age_seconds = time.time() - atp_file.stat().st_mtime
                file_age_hours = file_age_seconds / 3600
                
                if file_age_hours < max_age_hours:
                    atp_needs_update = False
                    logger.info(f"Archivo ATP es válido ({file_age_hours/24:.1f} días). Usando: {atp_file}")
                else:
                    logger.warning(f"Archivo ATP es antiguo ({file_age_hours/24:.1f} días, máx: {self.max_age_days} días)")
                    if self.force_read_only:
                        logger.info("Modo SOLO LECTURA activado: usando archivo antiguo sin actualizar")
                        atp_needs_update = False
            else:
                logger.warning("No se encontraron archivos de ranking ATP COMPLETOS")
                if self.force_read_only:
                    logger.info("Modo SOLO LECTURA: continuando sin ATP")

            # Verificar archivo WTA
            if wta_complete_files:
                wta_file = wta_complete_files[0]
                file_age_seconds = time.time() - wta_file.stat().st_mtime
                file_age_hours = file_age_seconds / 3600
                
                if file_age_hours < max_age_hours:
                    wta_needs_update = False
                    logger.info(f"Archivo WTA es válido ({file_age_hours/24:.1f} días). Usando: {wta_file}")
                else:
                    logger.warning(f"Archivo WTA es antiguo ({file_age_hours/24:.1f} días, máx: {self.max_age_days} días)")
                    if self.force_read_only:
                        logger.info("Modo SOLO LECTURA activado: usando archivo antiguo sin actualizar")
                        wta_needs_update = False
            else:
                logger.warning("No se encontraron archivos de ranking WTA COMPLETOS")
                if self.force_read_only:
                    logger.info("Modo SOLO LECTURA: continuando sin WTA")

            # MODIFICACIÓN CLAVE: Solo ejecutar scraper si NO está en modo solo lectura
            if (atp_needs_update or wta_needs_update) and not self.force_read_only:
                logger.info("Ejecutando scraper para obtener datos de ranking actualizados...")
                scraper = CompleteRankingScraper()
                
                # El scraper debería generar ambos archivos (ATP y WTA)
                new_files = asyncio.run(scraper.run_complete_extraction())
                
                if new_files:
                    # Actualizar rutas de archivos con los nuevos
                    if isinstance(new_files, dict):
                        if 'atp' in new_files and new_files['atp']:
                            atp_file = Path(new_files['atp'])
                            logger.info(f"Nuevo archivo ATP: {atp_file}")
                        if 'wta' in new_files and new_files['wta']:
                            wta_file = Path(new_files['wta'])
                            logger.info(f"Nuevo archivo WTA: {wta_file}")
                    else:
                        # Si devuelve un solo archivo (compatibilidad con versión anterior)
                        if atp_needs_update:
                            atp_file = Path(new_files)
                            logger.info(f"Nuevo archivo ATP: {atp_file}")
                else:
                    logger.error("El scraper falló. Intentando usar archivos antiguos si existen.")
            elif (atp_needs_update or wta_needs_update) and self.force_read_only:
                logger.info("MODO SOLO LECTURA: Saltando actualización automática de archivos")
                logger.info("Los archivos existentes serán utilizados independientemente de su antigüedad")

            # Cargar archivos ATP y WTA existentes
            atp_loaded = self._load_complete_ranking_file(atp_file, 'ATP') if atp_file else False
            wta_loaded = self._load_complete_ranking_file(wta_file, 'WTA') if wta_file else False

            if not atp_loaded and not wta_loaded:
                logger.error("No se pudieron cargar archivos de ranking COMPLETOS. Usando fallback...")
                self.load_basic_rankings()
            elif not atp_loaded:
                logger.warning("No se pudo cargar ATP COMPLETO. Intentando fallback ATP...")
                self._load_basic_ranking_fallback('ATP')
            elif not wta_loaded:
                logger.warning("No se pudo cargar WTA COMPLETO. Intentando fallback WTA...")
                self._load_basic_ranking_fallback('WTA')

            # Agregar información de modo de operación
            self.metadata['operation_mode'] = 'READ_only' if self.force_read_only else 'auto_update'
            self.metadata['max_age_days'] = self.max_age_days
            
            logger.info(f"Total cargado: {len(self.rankings_data)} jugadores/as con ranking.")
            logger.info(f"   - ATP: {self.metadata['atp_count']} jugadores")
            logger.info(f"   - WTA: {self.metadata['wta_count']} jugadoras")
            logger.info(f"   - Modo operación: {self.metadata['operation_mode']}")

        except Exception as e:
            logger.error(f"Error cargando rankings COMPLETOS: {e}")
            logger.info("Intentando fallback a rankings básicos...")
            self.load_basic_rankings()

    def _load_complete_ranking_file(self, ranking_file, tour_type):
        """Cargar un archivo de ranking COMPLETE específico (ATP o WTA)"""
        if not ranking_file or not ranking_file.exists():
            logger.warning(f"No existe el archivo {tour_type}: {ranking_file}")
            return False

        try:
            with open(ranking_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # CORRECCIÓN: Acceder a la clave "rankings" que es una lista
            players_list = data.get("rankings", [])
            if not isinstance(players_list, list):
                logger.error(f"El archivo {ranking_file} no tiene la estructura esperada (no es una lista de jugadores en 'rankings').")
                return False

            count = 0
            for player_data in players_list:
                name = player_data.get("name")
                if not name:
                    continue

                normalized_name = self.normalize_name(name)
                
                # Agregar identificador de tour y normalizar datos
                player_info = dict(player_data)
                player_info['tour'] = tour_type
                player_info['original_name'] = name  # Preservar nombre original
                
                # Normalizar campos comunes para compatibilidad
                if 'rank' in player_info and 'ranking_position' not in player_info:
                    player_info['ranking_position'] = player_info['rank']
                elif 'ranking_position' in player_info and 'rank' not in player_info:
                    player_info['rank'] = player_info['ranking_position']
                
                # Asegurar que existan campos de puntos
                if 'points' in player_info and 'ranking_points' not in player_info:
                    player_info['ranking_points'] = player_info['points']
                elif 'ranking_points' in player_info and 'points' not in player_info:
                    player_info['points'] = player_info['ranking_points']
                
                # Almacenar en diccionarios específicos por tour
                if tour_type == 'ATP':
                    self.atp_players[normalized_name] = player_info
                    self.metadata['atp_file'] = str(ranking_file)
                else:  # WTA
                    self.wta_players[normalized_name] = player_info
                    self.metadata['wta_file'] = str(ranking_file)
                
                # También mantener en el diccionario general para compatibilidad
                if normalized_name not in self.rankings_data or tour_type == 'ATP':
                    self.rankings_data[normalized_name] = player_info
                
                count += 1
                
            if tour_type == 'ATP':
                self.metadata['atp_count'] = count
            else:
                self.metadata['wta_count'] = count
                
            # Extraer timestamp del archivo para metadata
            file_timestamp = ranking_file.stat().st_mtime
            file_age_days = (time.time() - file_timestamp) / (24 * 3600)
            
            logger.info(f"Cargados {count} jugadores {tour_type} con ranking COMPLETO desde {ranking_file}")
            logger.info(f"   Archivo de {file_age_days:.1f} días de antigüedad")
            return True

        except Exception as e:
            logger.error(f"Error cargando archivo {tour_type} {ranking_file}: {e}")
            return False

    def _load_basic_ranking_fallback(self, tour_type):
        """Cargar ranking básico como fallback para un tour específico"""
        logger.info(f"Cargando ranking básico {tour_type} (fallback)...")
        data_dir = Path('data')
        
        try:
            if tour_type == 'ATP':
                files = sorted(data_dir.glob('atp_rankings_*.json'), reverse=True)
            else:  # WTA
                files = sorted(data_dir.glob('wta_rankings_*.json'), reverse=True)
            
            if files:
                ranking_file = files[0]
                with open(ranking_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    count = 0
                    for player_data in data:
                        name = player_data.get('name', '').strip()
                        rank = player_data.get('rank')
                        if name and rank:
                            normalized_name = self.normalize_name(name)
                            
                            player_info = {
                                'ranking_position': rank,
                                'rank': rank,  # Compatibilidad
                                'name': name,
                                'original_name': name,
                                'ranking_points': player_data.get('points', 0),
                                'points': player_data.get('points', 0),  # Compatibilidad
                                'nationality': player_data.get('nationality', 'N/A'),
                                'tour': tour_type
                            }
                            
                            # Almacenar en diccionarios específicos
                            if tour_type == 'ATP':
                                if normalized_name not in self.atp_players:
                                    self.atp_players[normalized_name] = player_info
                                    count += 1
                            else:  # WTA
                                if normalized_name not in self.wta_players:
                                    self.wta_players[normalized_name] = player_info
                                    count += 1
                            
                            # Mantener compatibilidad con diccionario general
                            if normalized_name not in self.rankings_data or tour_type == 'ATP':
                                self.rankings_data[normalized_name] = player_info
                    
                    if tour_type == 'ATP':
                        self.metadata['atp_count'] = count
                        self.metadata['atp_file'] = str(ranking_file)
                    else:
                        self.metadata['wta_count'] = count
                        self.metadata['wta_file'] = str(ranking_file)
                    
                    logger.info(f"Cargados {count} jugadores {tour_type} básicos desde {ranking_file}")
            else:
                logger.warning(f"No se encontraron archivos básicos {tour_type}")

        except Exception as e:
            logger.error(f"Error cargando ranking básico {tour_type}: {e}")

    def load_basic_rankings(self):
        """Carga rankings básicos como fallback si no se encuentran los enhanced."""
        logger.info("Cargando rankings básicos (fallback completo)...")
        self._load_basic_ranking_fallback('ATP')
        self._load_basic_ranking_fallback('WTA')

    def force_scraping_update(self):
        """
        Método para forzar actualización via scraping (solo cuando sea necesario)
        Útil para casos excepcionales donde se requiere data fresca
        """
        if self.force_read_only:
            logger.warning("Instancia configurada en modo SOLO LECTURA. Para permitir scraping:")
            logger.warning("   1. Cree una nueva instancia con force_read_only=False")
            logger.warning("   2. O use el método enable_scraping() primero")
            return False
            
        logger.info("Forzando actualización via scraping...")
        # Temporalmente desactivar check de antigüedad
        old_max_age = self.max_age_days
        self.max_age_days = 0  # Forzar que todos los archivos sean "antiguos"
        
        try:
            self.load_rankings()
            return True
        finally:
            self.max_age_days = old_max_age

    def enable_scraping(self):
        """Habilitar scraping en una instancia que era solo lectura"""
        logger.info("Habilitando modo scraping en instancia existente")
        self.force_read_only = False
        self.metadata['read_only_mode'] = False
        self.metadata['operation_mode'] = 'auto_update'
        
    def disable_scraping(self):
        """Deshabilitar scraping (volver a modo solo lectura)"""
        logger.info("Deshabilitando scraping - volviendo a modo solo lectura")
        self.force_read_only = True
        self.metadata['read_only_mode'] = True
        self.metadata['operation_mode'] = 'read_only'

    def get_data_freshness_info(self):
        """Obtener información sobre la frescura de los datos cargados"""
        info = {
            'operation_mode': self.metadata.get('operation_mode', 'unknown'),
            'max_age_days': self.max_age_days,
            'files_info': {}
        }
        
        for tour in ['atp', 'wta']:
            file_path = self.metadata.get(f'{tour}_file')
            if file_path and Path(file_path).exists():
                file_stat = Path(file_path).stat()
                age_days = (time.time() - file_stat.st_mtime) / (24 * 3600)
                info['files_info'][tour] = {
                    'file': file_path,
                    'age_days': round(age_days, 1),
                    'is_fresh': age_days < self.max_age_days,
                    'size_kb': round(file_stat.st_size / 1024, 1)
                }
            else:
                info['files_info'][tour] = {
                    'file': None,
                    'age_days': None,
                    'is_fresh': False
                }
        
        return info

    # ... (resto de métodos sin cambios)
    def normalize_name(self, name):
        """Normalizar nombre de jugador para búsqueda"""
        if not name:
            return ""
        
        # Normalizar a minúsculas y quitar espacios
        normalized = name.lower().strip()
        
        # Reemplazar manualmente diacríticos (acentos, etc.)
        replacements = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u',
            'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
            'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',
            'ñ': 'n', 'ç': 'c'
        }
        for char, replacement in replacements.items():
            normalized = normalized.replace(char, replacement)
        
        # Remover sufijos como "+1", "-2", etc.
        normalized = re.sub(r'[+\-]\d+$', '', normalized)
        
        # Remover cualquier caracter que no sea letra, número o espacio
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Normalizar múltiples espacios a uno solo
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def get_player_ranking(self, player_name, tour=None):
        """Obtener ranking de un jugador con manejo robusto de errores."""
        try:
            player_info = self.get_player_info(player_name, tour)
            if player_info:
                # Buscar ranking en múltiples campos posibles
                ranking = (player_info.get('ranking_position') or 
                          player_info.get('rank') or 
                          player_info.get('position'))
                
                # Convertir a entero si es posible
                if ranking is not None:
                    try:
                        return int(ranking)
                    except (ValueError, TypeError):
                        logger.warning(f"Ranking no numérico para {player_name}: {ranking}")
                        return None
                        
            logger.warning(f"No se encontró ranking para '{player_name}' en {tour or 'ATP/WTA'}")
            return None
            
        except Exception as e:
            logger.error(f"Error obteniendo ranking para '{player_name}': {e}")
            return None
    
    def get_player_info(self, player_name, tour=None):
        """Obtener información completa de un jugador con búsqueda mejorada.
        
        Args:
            player_name: Nombre del jugador a buscar
            tour: 'ATP', 'WTA' o None para buscar en ambos
        """
        if not player_name:
            return None

        normalized_name = self.normalize_name(player_name)
        
        # Determinar en qué diccionarios buscar
        search_dicts = []
        if tour == 'ATP':
            search_dicts = [self.atp_players]
        elif tour == 'WTA':
            search_dicts = [self.wta_players]
        else:
            # Buscar en ambos, priorizando ATP para compatibilidad
            search_dicts = [self.atp_players, self.wta_players]
        
        # 1. Búsqueda exacta
        for search_dict in search_dicts:
            if normalized_name in search_dict:
                return search_dict[normalized_name]

        # 2. Búsqueda mejorada por Apellido + Inicial
        name_parts = normalized_name.split()
        if len(name_parts) == 2:
            first_name, last_name = name_parts
            # Buscar formato "Apellido Nombre" coincidente
            for search_dict in search_dicts:
                for ranked_name, data in search_dict.items():
                    ranked_parts = ranked_name.split()
                    if len(ranked_parts) >= 2:
                        # Verificar coincidencia: último apellido + primer nombre
                        if (ranked_parts[-1] == last_name and 
                            len(ranked_parts) > 1 and 
                            ranked_parts[0].startswith(first_name[0])):
                            display_name = data.get('original_name') or data.get('name', 'N/A')
                            tour_info = data.get('tour', '')
                            logger.info(f"Coincidencia Apellido+Inicial para '{player_name}': {display_name} ({tour_info})")
                            return data
        
        # 3. Búsqueda por formato "Apellido(s) + Inicial" (ej: "Cerundolo F.", "Moro Canas A.")
        if len(name_parts) >= 2 and len(name_parts[-1]) == 1:
            last_name_parts = name_parts[:-1]
            initial = name_parts[-1]
            
            for search_dict in search_dicts:
                for ranked_name, data in search_dict.items():
                    ranked_parts = ranked_name.split()
                    
                    # Verificar que todas las partes del apellido estén en el nombre completo
                    if all(part in ranked_parts for part in last_name_parts):
                        # Verificar que una de las partes restantes comience con la inicial
                        remaining_parts = [p for p in ranked_parts if p not in last_name_parts]
                        if any(p.startswith(initial) for p in remaining_parts):
                            display_name = data.get('original_name') or data.get('name', 'N/A')
                            tour_info = data.get('tour', '')
                            logger.info(f"Coincidencia Apellido+Inicial para '{player_name}': {display_name} ({tour_info})")
                            return data
        
        # 4. Búsqueda inteligente con validación de contexto y jugadores retirados

        retired_players = {
        'rafael nadal', 'roger federer', 'serena williams', 'maria sharapova',
        'andy murray', 'petra kvitova', 'venus williams', 'stan wawrinka',
        'david ferrer', 'tommy robredo', 'nicolas almagro'
        }

        # Verificar si es un jugador retirado conocido
        if normalized_name in retired_players:
            logger.warning(f"Jugador retirado detectado: '{player_name}'. No devolviendo coincidencias para evitar falsos positivos.")
            return None
            
        # Búsqueda inteligente con múltiples validaciones
        candidates = []  

        # Solo buscar si tenemos al menos 2 palabras
        if len(name_parts) >= 2:
            for search_dict in search_dicts:
                for ranked_name, data in search_dict.items():
                    ranked_parts = ranked_name.split()
                
                    # VALIDACIÓN 1: Debe haber coincidencia de múltiples partes
                    matches = 0
                    matched_parts = []
                
                    for search_part in name_parts:
                        for ranked_part in ranked_parts:
                            # Coincidencia exacta o inicio de palabra
                            if search_part == ranked_part or search_part.startswith(ranked_part) or ranked_part.startswith(search_part):
                                matches += 1
                                matched_parts.append((search_part, ranked_part))
                                break
                    # VALIDACIÓN 2: Al menos el 60% de las partes deben coincidir
                    match_percentage = matches / len(name_parts)
                    if match_percentage >= 0.6 and matches >= 2: 

                        # VALIDACIÓN 3: Verificar que no sea una coincidencia de solo nombres
                        # (como Rafael -> Rafael, que causó el problema original)
                        is_safe_match = False
                    
                        # Verificar coincidencia de apellidos (últimas palabras)
                        if name_parts[-1] in [part.lower() for part in ranked_parts]: 
                            is_safe_match = True

                        # Verificar coincidencia de nombres completos estructurados
                        if len(name_parts) == 2 and len(ranked_parts) == 2:
                            # Formato "Nombre Apellido" vs "Apellido Nombre"
                            if (name_parts[0] == ranked_parts[1].lower() and 
                                name_parts[1] == ranked_parts[0].lower()):
                                is_safe_match = True
                    
                        if is_safe_match:
                            confidence_score = match_percentage * matches
                            candidates.append((confidence_score, data, ranked_name, matched_parts))       
        
        # Si encontramos candidatos, devolver el de mayor confianza
        if candidates:
            # Ordenar por puntuación de confianza (descendente)
            candidates.sort(reverse=True, key=lambda x: x[0])
        
            best_candidate = candidates[0]
            confidence, data, matched_name, parts = best_candidate
        
            display_name = data.get('original_name') or data.get('name', 'N/A')
            tour_info = data.get('tour', '')
        
            logger.info(f"Coincidencia inteligente para '{player_name}': {display_name} ({tour_info}) - Confianza: {confidence:.2f}")
            return data    
                      
        # 5. Búsqueda de último recurso - coincidencias múltiples
        for search_dict in search_dicts:
            best_match = None
            best_score = 0
            
            for ranked_name, data in search_dict.items():
                ranked_parts = ranked_name.split()
                matches = sum(1 for part in name_parts 
                            if any(part in ranked_part for ranked_part in ranked_parts))
                
                # Requerir al menos 2 coincidencias o todas las partes si es nombre corto
                min_matches = min(2, len(name_parts))
                if matches >= min_matches and matches > best_score:
                    best_match = data
                    best_score = matches
            
            if best_match:
                display_name = best_match.get('original_name') or best_match.get('name', 'N/A')
                tour_info = best_match.get('tour', '')
                logger.info(f"Coincidencia parcial fallback para '{player_name}': {display_name} ({tour_info})")
                return best_match
        
        logger.warning(f"No se encontró ranking para '{player_name}' en {tour or 'ATP/WTA'}.")
        return None
    
    def get_ranking_metrics(self, player_name, tour=None):
        """Obtener métricas avanzadas de ranking (PTS, PROX, defense_points, etc.)
        
        Args:
            player_name: Nombre del jugador
            tour: 'ATP', 'WTA' o None para buscar en ambos
            
        Returns:
            dict: Diccionario con métricas de ranking o None si no se encuentra
        """
        try:    
            player_info = self.get_player_info(player_name, tour)
            if not player_info:
                logger.warning(f"No se encontró información para '{player_name}'")
                return None
            
            # Extraer métricas del archivo JSON completo usando los nombres correctos de los campos
            metrics = {}

            # Campos básicos
            metrics['ranking_position'] = player_info.get('ranking_position')
            metrics['PTS'] = player_info.get('PTS') or player_info.get('ranking_points')
            metrics['PROX'] = player_info.get('PROX') or player_info.get('prox_points')
            metrics['MAX'] = player_info.get('MAX') or player_info.get('max_points')
            metrics['Potencial'] = player_info.get('Potencial') or player_info.get('improvement_potential')

            # CAMPOS CLAVE AGREGADOS: defense_points y otros campos del JSON completo
            metrics['defense_points'] = player_info.get('defense_points')
            metrics['tournaments_played'] = player_info.get('tournaments_played')
            metrics['ranking_change'] = player_info.get('ranking_change')
            metrics['defense_pressure'] = player_info.get('defense_pressure')
            metrics['momentum_secured'] = player_info.get('momentum_secured')
            
            # Información adicional
            metrics['nationality'] = player_info.get('nationality', 'N/A')
            metrics['tour'] = player_info.get('tour', 'N/A')
            metrics['name'] = player_info.get('original_name') or player_info.get('name', player_name)
            metrics['extraction_timestamp'] = player_info.get('extraction_timestamp')
            metrics['data_source'] = player_info.get('data_source')
            
            # Conversiones a enteros/float donde sea apropiado
            numeric_fields = ['ranking_position', 'PTS', 'PROX', 'MAX', 'Potencial', 'defense_points', 
                            'tournaments_played', 'ranking_change', 'momentum_secured']
            
            for key in numeric_fields:
                if metrics.get(key) is not None:
                    try:
                        metrics[key] = int(metrics[key])
                    except (ValueError, TypeError):
                        logger.warning(f"No se pudo convertir {key} a entero para {player_name}: {metrics[key]}")
                        # Mantener el valor original si no se puede convertir
                        pass
            
            # Campos que pueden ser float
            float_fields = ['defense_pressure']
            for key in float_fields:
                if metrics.get(key) is not None:
                    try:
                        metrics[key] = float(metrics[key])
                    except (ValueError, TypeError):
                        logger.warning(f"No se pudo convertir {key} a float para {player_name}: {metrics[key]}")
                        pass
            
            # Log de debug para verificar extracción COMPLETA
            logger.info(f"Métricas extraídas para {player_name}: PTS={metrics.get('PTS')}, "
                       f"PROX={metrics.get('PROX')}, MAX={metrics.get('MAX')}, "
                       f"Potencial={metrics.get('Potencial')}, defense_points={metrics.get('defense_points')}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error obteniendo métricas para '{player_name}': {e}")
            return None
       
    def get_atp_player_info(self, player_name):
        """Búsqueda específica en rankings ATP"""
        return self.get_player_info(player_name, tour='ATP')
    
    def get_wta_player_info(self, player_name):
        """Búsqueda específica en rankings WTA"""
        return self.get_player_info(player_name, tour='WTA')
    
    def get_atp_ranking(self, player_name):
        """Obtener ranking específico ATP"""
        return self.get_player_ranking(player_name, tour='ATP')
    
    def get_wta_ranking(self, player_name):
        """Obtener ranking específico WTA"""
        return self.get_player_ranking(player_name, tour='WTA')
    
    def get_tour_stats(self):
        """Obtener estadísticas de los tours cargados"""
        return {
            'atp_players': len(self.atp_players),
            'wta_players': len(self.wta_players),
            'total_players': len(self.rankings_data),
            'files_loaded': {
                'atp_file': self.metadata['atp_file'],
                'wta_file': self.metadata['wta_file']
            }
        }
    
    def get_player_metrics_summary(self, player_name, tour=None):
        """Obtener resumen de métricas principales en formato legible
        
        Returns:
            dict: Métricas principales (PTS, PROX, MAX, Potencial) o valores por defecto
        """
        try:
            metrics = self.get_ranking_metrics(player_name, tour)
            if not metrics:
                return {
                    'PTS': 'N/A',
                    'PROX': 'N/A', 
                    'MAX': 'N/A',
                    'Potencial': 'N/A'
                }
            
            return {
                'PTS': metrics.get('PTS', 'N/A'),
                'PROX': metrics.get('PROX', 'N/A'),
                'MAX': metrics.get('MAX', 'N/A'), 
                'Potencial': metrics.get('Potencial', 'N/A')
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo resumen de métricas para '{player_name}': {e}")
            return {
                'PTS': 'N/A',
                'PROX': 'N/A',
                'MAX': 'N/A', 
                'Potencial': 'N/A'
            }
    
    def debug_player_data(self, player_name, tour=None):
        """Función de debug para inspeccionar datos de un jugador"""
        player_info = self.get_player_info(player_name, tour)
        if player_info:
            print(f"Debug para '{player_name}':")
            for key, value in player_info.items():
                print(f"  {key}: {value}")
                
            # Mostrar métricas específicas
            print("\nMétricas principales:")
            print(f"  PTS (ranking_points): {player_info.get('ranking_points', 'N/A')}")
            print(f"  PROX (prox_points): {player_info.get('prox_points', 'N/A')}")
            print(f"  MAX (max_points): {player_info.get('max_points', 'N/A')}")
            print(f"  Potencial (improvement_potential): {player_info.get('improvement_potential', 'N/A')}")
        else:
            print(f"No se encontraron datos para '{player_name}'")
        return player_info
    
    def diagnose_data_loading(self):
        """Función de diagnóstico completo para verificar carga de datos"""
        print("DIAGNÓSTICO COMPLETO DE CARGA DE DATOS")
        print("=" * 50)
        
        # 1. Verificar archivos cargados
        print("Archivos cargados:")
        print(f"  ATP: {self.metadata['atp_file']}")
        print(f"  WTA: {self.metadata['wta_file']}")
        
        # 2. Verificar cantidad de jugadores
        print("\nJugadores cargados:")
        print(f"  ATP: {len(self.atp_players)}")
        print(f"  WTA: {len(self.wta_players)}")
        print(f"  Total general: {len(self.rankings_data)}")
        
        # 3. Mostrar ejemplos de jugadores ATP
        if self.atp_players:
            print("\nEjemplos ATP (primeros 3):")
            for i, (name, data) in enumerate(list(self.atp_players.items())[:3]):
                print(f"  {i+1}. Nombre normalizado: '{name}'")
                print(f"     Nombre original: '{data.get('name', 'N/A')}'")
                print(f"     Ranking: {data.get('ranking_position', 'N/A')}")
                print(f"     PTS: {data.get('ranking_points', 'N/A')}")
                print(f"     Campos disponibles: {list(data.keys())}")
                print()
        
        # 4. Verificar estructura de datos típica
        if self.atp_players:
            sample_player = next(iter(self.atp_players.values()))
            print("Estructura de datos típica:")
            for key, value in sample_player.items():
                print(f"  {key}: {type(value).__name__} = {value}")
        
        print("\n" + "=" * 50)
        
    def test_player_search(self, player_names):
        """Función de prueba para múltiples jugadores"""
        print("PRUEBA DE BÚSQUEDA DE JUGADORES")
        print("=" * 50)
        
        for player_name in player_names:
            print(f"\nBuscando: '{player_name}'")
            normalized = self.normalize_name(player_name)
            print(f"   Normalizado: '{normalized}'")
            
            # Buscar en ATP
            atp_found = normalized in self.atp_players
            print(f"   Encontrado en ATP: {atp_found}")
            
            # Buscar en WTA  
            wta_found = normalized in self.wta_players
            print(f"   Encontrado en WTA: {wta_found}")
            
            # Intentar obtener métricas
            metrics = self.get_player_metrics_summary(player_name)
            print(f"   Métricas: {metrics}")
            
        print("\n" + "=" * 50)

class RivalryAnalyzer:
    """⚔️ Analizador de rivalidades transitivas con análisis de rachas y peso de oponentes."""

    def estimate_elo_from_rank(self, rank):
        """Estima un rating ELO basado en el ranking ATP/WTA."""
        if rank is None:
            return 1500  # ELO por defecto para sin ranking
        if rank <= 10:
            return 2400 - (rank - 1) * 20
        elif rank <= 50:
            return 2200 - (rank - 11) * 5
        elif rank <= 100:
            return 2000 - (rank - 51) * 2
        elif rank <= 200:
            return 1800 - (rank - 101) * 1
        else:
            return 1600

    def calculate_elo_from_history(self, player_name, player_history):
        """Calcula el rating ELO de un jugador basado en su historial de partidos extraído."""
        current_elo = self.elo_system.default_rating
        
        if not player_history:
            return current_elo

        # Iterar desde el partido más antiguo al más reciente
        for match in reversed(player_history):
            opponent_rank = match.get('opponent_ranking')
            opponent_elo = self.estimate_elo_from_rank(opponent_rank)
            
            expected_score = self.elo_system.expected_score(current_elo, opponent_elo)
            
            # Usar determine_match_winner para consistencia
            won = self.determine_match_winner(match, player_name)
            actual_score = 1 if won else 0
            
            new_elo = current_elo + self.elo_system.k_factor * (actual_score - expected_score)
            current_elo = new_elo
            
        return round(current_elo)

    def __init__(self, ranking_manager, elo_system):
        self.ranking_manager = ranking_manager
        self.elo_system = elo_system
        self.SURFACE_NORMALIZATION_MAP = {
            'hard': 'Dura', 'dura': 'Dura',
            'clay': 'Arcilla', 'arcilla': 'Arcilla',
            'grass': 'Hierba', 'hierba': 'Hierba',
            'indoor': 'Indoor', 'indoor hard': 'Indoor'
        }
        self.COUNTRY_TO_CONTINENT_MAP = {
            'USA': 'Norteamérica', 'Canada': 'Norteamérica', 'Mexico': 'Norteamérica',
            'Argentina': 'Sudamérica', 'Brazil': 'Sudamérica', 'Chile': 'Sudamérica', 'Colombia': 'Sudamérica', 'Ecuador': 'Sudamérica', 'Uruguay': 'Sudamérica',
            'Spain': 'Europa', 'France': 'Europa', 'Italy': 'Europa', 'Germany': 'Europa', 'Great Britain': 'Europa', 'Russia': 'Europa', 'Serbia': 'Europa', 'Switzerland': 'Europa', 'Sweden': 'Europa', 'Austria': 'Europa', 'Belgium': 'Europa', 'Netherlands': 'Europa', 'Poland': 'Europa', 'Czech Republic': 'Europa', 'Croatia': 'Europa', 'Greece': 'Europa', 'Norway': 'Europa', 'Denmark': 'Europa', 'Finland': 'Europa', 'Portugal': 'Europa', 'Hungary': 'Europa', 'Romania': 'Europa', 'Slovakia': 'Europa', 'Slovenia': 'Europa', 'Ukraine': 'Europa',
            'Australia': 'Oceanía', 'New Zealand': 'Oceanía',
            'Japan': 'Asia', 'China': 'Asia', 'South Korea': 'Asia', 'India': 'Asia', 'Kazakhstan': 'Asia',
            'South Africa': 'África', 'Tunisia': 'África', 'Egypt': 'África'
        }

    def calculate_base_opponent_weight(self, ranking):
        """⚖️ Calcular peso base de oponente basado en ranking para enriquecimiento."""
        if ranking is None:
            return 1  # Peso base para jugadores sin ranking
        if ranking <= 10: return 10
        elif ranking <= 15: return 8
        elif ranking <= 30: return 6
        elif ranking <= 50: return 4
        elif ranking <= 100: return 2
        else: return 1

    def _partidos_recientes(self, player_history, opponent_name, recent_count=20):
        """Verifica si un partido contra opponent_name está en los últimos `recent_count` juegos."""
        normalized_opponent_name = self.ranking_manager.normalize_name(opponent_name)
        for match in player_history[:recent_count]:
            if self.ranking_manager.normalize_name(match.get('oponente', '')) == normalized_opponent_name:
                return True
        return False

    def _count_matches(self, player_history, opponent_name):
        """Cuenta los partidos contra un oponente específico."""
        normalized_opponent_name = self.ranking_manager.normalize_name(opponent_name)
        return sum(1 for match in player_history if self.ranking_manager.normalize_name(match.get('oponente', '')) == normalized_opponent_name)

    def _win_rate_vs_oponente(self, player_history, opponent_name, player_name):
        """Calcula la tasa de victorias contra un oponente específico."""
        normalized_opponent_name = self.ranking_manager.normalize_name(opponent_name)
        matches = [m for m in player_history if self.ranking_manager.normalize_name(m.get('oponente', '')) == normalized_opponent_name]
        if not matches:
            return 0.0
        wins = sum(1 for m in matches if self.determine_match_winner(m, player_name))
        return wins / len(matches)

    def calcular_peso_oponentes_comunes(self, player1_history, player2_history, common_opponent_name, player1_name, player2_name):
        """Calcula el peso de un oponente común basado en ranking y contexto."""
        ranking_oponente = self.ranking_manager.get_player_ranking(common_opponent_name)
        
        # Peso base según ranking del oponente común
        if ranking_oponente is None:
            peso_base = 5
        elif ranking_oponente <= 5: peso_base = 20
        elif ranking_oponente <= 10: peso_base = 15
        elif ranking_oponente <= 20: peso_base = 12
        elif ranking_oponente <= 30: peso_base = 10
        elif ranking_oponente <= 50: peso_base = 8
        else: peso_base = 5
        
        # Multiplicadores por contexto
        multiplicador_contexto = 1.0
        
        # Si ambos jugaron recientemente (últimos 20 partidos)
        if self._partidos_recientes(player1_history, common_opponent_name) and self._partidos_recientes(player2_history, common_opponent_name):
            multiplicador_contexto += 0.3
        
        # Si hay múltiples enfrentamientos con el oponente común
        encuentros_a = self._count_matches(player1_history, common_opponent_name)
        encuentros_b = self._count_matches(player2_history, common_opponent_name)
        if encuentros_a >= 2 and encuentros_b >= 2:
            multiplicador_contexto += 0.4
        
        # El patrón de dominancia se aplica por separado en analyze_rivalry
        return peso_base * multiplicador_contexto

    def analyze_advanced_player_metrics(self, player_history, player_name):
        """
        📊 DIMENSIÓN 3 & 4: Calidad de Historial y Diversidad de Oponentes
        Calcula multiplicadores basados en la calidad y diversidad del historial de un jugador.
        """
        quality_multiplier = 1.0
        diversity_multiplier = 1.0
        analysis_log = []

        # Considerar hasta 45 partidos para el historial
        history_subset = player_history[:45]

        # ============================================
        # 1. CALIDAD DEL HISTORIAL (% victorias vs Top 30 en 45 partidos)
        # ============================================
        matches_vs_top30 = []
        wins_vs_top30 = 0
        for match in history_subset:
            opponent_rank = self.ranking_manager.get_player_ranking(match.get('oponente'))
            if opponent_rank and opponent_rank <= 30:
                matches_vs_top30.append(match)
                if self.determine_match_winner(match, player_name):
                    wins_vs_top30 += 1
        
        if len(matches_vs_top30) > 0:
            win_rate_vs_top30 = (wins_vs_top30 / len(matches_vs_top30)) * 100
            if win_rate_vs_top30 > 60:
                quality_multiplier = 1.5  # AJUSTADO: Máximo 50% de bonificación
                analysis_log.append(f"🏆 Calidad Historial: >60% victorias vs Top 30 ({win_rate_vs_top30:.1f}%) -> x1.5")
            elif 40 <= win_rate_vs_top30 <= 59:
                quality_multiplier = 1.5
                analysis_log.append(f"🏆 Calidad Historial: 40-59% victorias vs Top 30 ({win_rate_vs_top30:.1f}%) -> x1.5")
            elif 20 <= win_rate_vs_top30 <= 39:
                quality_multiplier = 1.2
                analysis_log.append(f"🏆 Calidad Historial: 20-39% victorias vs Top 30 ({win_rate_vs_top30:.1f}%) -> x1.2")
            else: # < 20%
                quality_multiplier = 1.0
                analysis_log.append(f"🏆 Calidad Historial: <20% victorias vs Top 30 ({win_rate_vs_top30:.1f}%) -> x1.0")

        # ============================================
        # 2. DIVERSIDAD DE OPONENTES DE CALIDAD (en todo el historial disponible)
        # ============================================
        faced_top50 = set()
        beaten_top30 = set()

        for match in player_history:
            opponent_name = match.get('oponente')
            opponent_rank = self.ranking_manager.get_player_ranking(opponent_name)
            
            if opponent_rank and opponent_rank <= 50:
                faced_top50.add(opponent_name)
            
            if opponent_rank and opponent_rank <= 30:
                if self.determine_match_winner(match, player_name):
                    beaten_top30.add(opponent_name)

        # Bonus granular por enfrentar jugadores Top 50 (máx 20 jugadores para el bono)
        faced_bonus = (min(len(faced_top50), 20) / 20) * 0.25
        if faced_bonus > 0:
            diversity_multiplier += faced_bonus
            analysis_log.append(f"🌍 Diversidad (Enfrentados): {len(faced_top50)} Top 50 -> +{faced_bonus*100:.1f}%")

        # Bonus granular por vencer jugadores Top 30 (máx 10 jugadores para el bono)
        beaten_bonus = (min(len(beaten_top30), 10) / 10) * 0.40
        if beaten_bonus > 0:
            diversity_multiplier += beaten_bonus
            analysis_log.append(f"💪 Diversidad (Vencidos): {len(beaten_top30)} Top 30 -> +{beaten_bonus*100:.1f}%")

        return quality_multiplier, diversity_multiplier, analysis_log

    def analyze_strength_of_schedule(self, player_history, player_name):
        """
        💪 DIMENSIÓN 5: Analiza la "Fuerza del Calendario" (Strength of Schedule).
        Recompensa a jugadores por enfrentar (y especialmente vencer) a oponentes de alto ranking.
        """
        schedule_score = 0
        analysis_log = []
        
        # Considerar los últimos 45 partidos
        history_subset = player_history[:45]

        for match in history_subset:
            opponent_rank = self.ranking_manager.get_player_ranking(match.get('oponente'))
            if not opponent_rank:
                continue

            points = 0
            # Puntos base por ranking del oponente
            if opponent_rank <= 10: points = 25
            elif opponent_rank <= 20: points = 20
            elif opponent_rank <= 50: points = 15
            elif opponent_rank <= 100: points = 10
            elif opponent_rank <= 200: points = 5
            
            if points == 0: continue

            # Multiplicador por resultado
            if self.determine_match_winner(match, player_name):
                # Bonificación por victoria
                points *= 2.5
                analysis_log.append(f"💪 Victoria vs Rank {opponent_rank} ({match.get('oponente')}) -> +{points:.1f} pts")
            else:
                # Puntos por derrota competitiva (se mantiene el punto base)
                analysis_log.append(f"🛡️ Derrota vs Rank {opponent_rank} ({match.get('oponente')}) -> +{points:.1f} pts")

            schedule_score += points
        
        if analysis_log:
            analysis_log.insert(0, f"📊 Puntuación total de 'Strength of Schedule': {schedule_score:.1f}")

        return schedule_score, analysis_log

    def analyze_streaks_and_consistency(self, player_history, player_name):
        """
        🎯 DIMENSIÓN 2: Factor de Racha/Consistencia MEJORADO
        Analiza rachas, consistencia y momentum según criterios específicos
        """
        streak_multiplier = 1.0
        analysis_log = []

        # ============================================
        # 1. ANÁLISIS DE RACHAS DE VICTORIAS CONSECUTIVAS
        # ============================================
        current_win_streak = 0
        streak_opponents_ranks = []
        
        # Contar racha actual de victorias y obtener rankings de oponentes
        for match in player_history:
            if self.determine_match_winner(match, player_name):
                current_win_streak += 1
                opponent_rank = self.ranking_manager.get_player_ranking(match.get('oponente'))
                if opponent_rank:
                    streak_opponents_ranks.append(opponent_rank)
            else:
                break

        # Aplicar multiplicadores según criterios específicos (condiciones relajadas)
        if current_win_streak >= 5:
            # Racha 5+ victorias con al menos 2 vs Top 50: +50%
            top50_wins_in_streak = sum(1 for rank in streak_opponents_ranks if rank <= 50)
            if top50_wins_in_streak >= 2:
                streak_multiplier *= 1.50
                analysis_log.append(f"🔥 Racha de {current_win_streak} (con {top50_wins_in_streak} vs Top 50) -> +50%")
        
        if current_win_streak >= 3:
            # Racha 3+ victorias con al menos 2 vs Top 30: +30%
            top30_wins_in_streak = sum(1 for rank in streak_opponents_ranks if rank <= 30)
            if top30_wins_in_streak >= 2:
                streak_multiplier *= 1.30
                analysis_log.append(f"⚡ Racha de {current_win_streak} (con {top30_wins_in_streak} vs Top 30) -> +30%")

        # ============================================
        # 2. ANÁLISIS DE CONSISTENCIA VS NIVEL SIMILAR
        # ============================================
        player_info = self.ranking_manager.get_player_info(player_name)
        player_rank = player_info.get('ranking_position') if player_info else None
        if player_rank:
            # Consistencia >75% vs Rank 31-50: +25% multiplicador
            if 31 <= player_rank <= 50:
                matches_vs_similar = []
                for match in player_history[:45]:  # Últimos 45 partidos
                    opponent_rank = self.ranking_manager.get_player_ranking(match.get('oponente'))
                    if opponent_rank and 31 <= opponent_rank <= 50:
                        matches_vs_similar.append(match)
                
                if len(matches_vs_similar) >= 4:  # Mínimo 4 partidos para ser significativo
                    wins_vs_similar = sum(1 for match in matches_vs_similar 
                                        if self.determine_match_winner(match, player_name))
                    consistency_rate = wins_vs_similar / len(matches_vs_similar)
                    
                    if consistency_rate > 0.75:
                        streak_multiplier *= 1.25
                        analysis_log.append(f"🎯 Consistencia {int(consistency_rate*100)}% vs Rank 31-50 ({wins_vs_similar}/{len(matches_vs_similar)}) (+25%)")
            
            # Consistencia general vs mismo tier
            else:
                def get_tier_range(rank):
                    if rank <= 10: return (1, 10)
                    elif rank <= 20: return (11, 20)  
                    elif rank <= 30: return (21, 30)
                    elif rank <= 50: return (31, 50)
                    elif rank <= 100: return (51, 100)
                    else: return (101, 200)

                tier_min, tier_max = get_tier_range(player_rank)
                matches_vs_tier = []
                
                for match in player_history[:45]:
                    opponent_rank = self.ranking_manager.get_player_ranking(match.get('oponente'))
                    if opponent_rank and tier_min <= opponent_rank <= tier_max:
                        matches_vs_tier.append(match)
                
                if len(matches_vs_tier) >= 4:
                    wins_vs_tier = sum(1 for match in matches_vs_tier 
                                     if self.determine_match_winner(match, player_name))
                    consistency_rate = wins_vs_tier / len(matches_vs_tier)
                    
                    if consistency_rate > 0.80:
                        streak_multiplier *= 1.20
                        analysis_log.append(f"💪 Consistencia {int(consistency_rate*100)}% vs Rank {tier_min}-{tier_max} (+20%)")

        # ============================================
        # 3. ANÁLISIS DE MOMENTUM RECIENTE
        # ============================================
        # Momentum últimos 10 partidos >70%: +20% multiplicador
        recent_matches = player_history[:10]
        if len(recent_matches) >= 8:  # Al menos 8 partidos para ser significativo
            recent_wins = sum(1 for match in recent_matches 
                            if self.determine_match_winner(match, player_name))
            momentum_rate = recent_wins / len(recent_matches)
            
            if momentum_rate > 0.70:
                streak_multiplier *= 1.20
                analysis_log.append(f"🚀 Momentum reciente {int(momentum_rate*100)}% en últimos {len(recent_matches)} partidos (+20%)")

        # ============================================
        # 4. ANÁLISIS ADICIONAL: VICTORIAS VS TOP TIERS
        # ============================================
        # Bonificación por victorias recientes contra top players
        recent_top_wins = 0
        for match in player_history[:20]:
            if self.determine_match_winner(match, player_name):
                opponent_rank = self.ranking_manager.get_player_ranking(match.get('oponente'))
                if opponent_rank and opponent_rank <= 20:
                    recent_top_wins += 1
        
        if recent_top_wins >= 3:
            streak_multiplier *= 1.15
            analysis_log.append(f"🏆 {recent_top_wins} victorias recientes vs Top 20 (+15%)")

        # AJUSTADO: Limitar el multiplicador máximo a 1.5
        final_multiplier = min(streak_multiplier, 1.5)
        if final_multiplier < streak_multiplier:
            analysis_log.append(f"⚠️ Multiplicador de racha limitado a x1.5 (calculado: {streak_multiplier:.2f})")

        return final_multiplier, analysis_log
    
    def determine_match_winner(self, match_data, target_player):
        """🏆 Determinar si el jugador objetivo ganó el partido de forma robusta."""
        outcome = match_data.get('outcome', '').lower()
        
        # Prioridad 1: Usar el campo 'outcome' si es claro
        if 'ganó' in outcome or 'win' in outcome:
            return True
        if 'perdió' in outcome or 'loss' in outcome:
            return False
        
        # Prioridad 2: Inferir del resultado si 'outcome' no es claro
        resultado = match_data.get('resultado', '')
        oponente = match_data.get('oponente', '')
        
        # Si el oponente se retiró (RET), el jugador objetivo ganó
        if 'RET' in resultado.upper():
            return True
        
        # Si el oponente no se presentó (WO), el jugador objetivo ganó
        if 'WO' in resultado.upper():
             return True

        # Lógica de fallback: si no hay indicación clara de derrota, asumir victoria
        # Esto es arriesgado pero mantiene el comportamiento anterior como último recurso.
        return 'perdió' not in outcome and 'loss' not in outcome

    def analizar_contundencia(self, resultado):
        """
        Analiza la contundencia de una victoria según las reglas especificadas.
        - Victoria por 2-0 sets contundentes: +1.5x
        - Victoria por 2-0 sets ajustados: +1.2x
        - Victoria por 2-1 sets: +1.0x
        """
        try:
            # Extraer sets (ej: '2-0' de '2-0 (6-2, 6-3)')
            sets_match = re.search(r'(\d+-\d+)', resultado)
            if not sets_match:
                return 1.0

            sets_str = sets_match.group(1)
            sets = [int(s) for s in sets_str.split('-')]
            
            # Victoria en 3 sets
            if min(sets) == 1:
                return 1.0

            # Victoria en 2 sets (2-0)
            if min(sets) == 0:
                # Extraer juegos para diferenciar contundente de ajustado
                games_match = re.findall(r'(\d+-\d+)', resultado)
                if len(games_match) >= 2: # Best of 3
                    total_game_diff = sum(abs(int(g.split('-')[0]) - int(g.split('-')[1])) for g in games_match)
                    
                    # Umbral para contundencia: diferencia de 7 o más juegos
                    if total_game_diff >= 7:
                        return 1.5  # Contundente
                    else:
                        return 1.2  # Ajustado
                return 1.2 # Default para 2-0 si no se pueden parsear los juegos

        except Exception:
            return 1.0 # Default en caso de error
        
        return 1.0

    def analizar_resistencia(self, resultado):
        """
        Analiza la resistencia en una derrota según las reglas especificadas.
        - Perdió pero ganó 1 set: +0.5x
        - Perdió ajustado en 2 sets: +0.3x
        - Perdió contundente: +0.0x
        """
        try:
            # Extraer sets
            sets_match = re.search(r'(\d+-\d+)', resultado)
            if not sets_match: return 0.0

            sets_str = sets_match.group(1)
            sets = [int(s) for s in sets_str.split('-')]

            # Perdió pero ganó 1 set
            if min(sets) == 1:
                return 0.5

            # Perdió en 2 sets (0-2)
            if min(sets) == 0:
                games_match = re.findall(r'(\d+-\d+)', resultado)
                if len(games_match) >= 2:
                    total_game_diff = sum(abs(int(g.split('-')[0]) - int(g.split('-')[1])) for g in games_match)
                    
                    # Si hubo un tie-break o la diferencia de juegos es <= 5, es ajustado
                    if any('7-6' in g or '6-7' in g for g in games_match) or total_game_diff <= 5:
                        return 0.3 # Ajustado
                    else:
                        return 0.0 # Contundente
                return 0.0 # Default para 0-2 si no se pueden parsear juegos

        except Exception:
            return 0.0 # Default en caso de error
            
        return 0.0
    
    def find_common_opponents(self, player1_history, player2_history):
        """🤝 Encontrar oponentes comunes entre dos jugadores"""
        opponents1 = {self.ranking_manager.normalize_name(m.get('oponente', '')) for m in player1_history if m.get('oponente')}
        opponents2 = {self.ranking_manager.normalize_name(m.get('oponente', '')) for m in player2_history if m.get('oponente')}
        return list(opponents1.intersection(opponents2))

    def analyze_direct_h2h(self, h2h_matches, player1_name, player2_name):
        """
        Analiza los enfrentamientos directos (H2H) entre dos jugadores.
        NUEVO: Se introduce una ponderación basada en la antigüedad del partido.
        - Partidos con más de 250 días de antigüedad tienen una ponderación de 0.
        - Partidos recientes tienen una ponderación base de 1.0.
        """
        p1_score = 0.0
        p2_score = 0.0
        log = []
        today = datetime.now()
        H2H_RECENT_DAYS_THRESHOLD = 250

        for match in h2h_matches:
            try:
                match_date_str = match.get('fecha')
                if not match_date_str:
                    continue

                match_date = datetime.strptime(match_date_str, '%d.%m.%y')
                days_since_match = (today - match_date).days
                
                ponderacion = 0.0
                if days_since_match <= H2H_RECENT_DAYS_THRESHOLD:
                    ponderacion = 1.0  # Ponderación base para partidos recientes
                
                winner = match.get("ganador")
                if winner == player1_name:
                    p1_score += ponderacion
                    if ponderacion > 0:
                        log.append(f"Victoria RECIENTE de {player1_name} en {match_date_str} (Ponderación: {ponderacion})")
                    else:
                        log.append(f"Victoria ANTIGUA de {player1_name} en {match_date_str} (Ponderación: 0)")

                elif winner == player2_name:
                    p2_score += ponderacion
                    if ponderacion > 0:
                        log.append(f"Victoria RECIENTE de {player2_name} en {match_date_str} (Ponderación: {ponderacion})")
                    else:
                        log.append(f"Victoria ANTIGUA de {player2_name} en {match_date_str} (Ponderación: 0)")

            except (ValueError, TypeError) as e:
                log.append(f"Error procesando fecha de H2H: {match.get('fecha')} - {e}")
                continue

        return p1_score, p2_score, log

    def get_ranking_metrics(self, player_name):
        """
        Calcula las métricas avanzadas de ranking para un jugador de forma robusta.
        """
        player_info = self.ranking_manager.get_player_info(player_name)
        if not player_info:
            return {}

        def _safe_get_numeric(key, default=0):
            val = player_info.get(key)
            if isinstance(val, (int, float)):
                return val
            return default

        pts = _safe_get_numeric('ranking_points', 0)
        
        # Lógica de fallback mejorada
        prox_pts = _safe_get_numeric('prox_points')
        if prox_pts == 0:
            prox_pts = pts

        pts_max = _safe_get_numeric('max_points')
        if pts_max == 0:
            pts_max = prox_pts
            
        defense_points = _safe_get_numeric('defense_points', 0)

        # Asegurarse de que los cálculos no den negativo
        already_secured = max(0, prox_pts - pts)
        improvement_potential = max(0, pts_max - pts)
        pressure_index = defense_points - already_secured

        return {
            'pts': pts,
            'prox_pts': prox_pts,
            'pts_max': pts_max,
            'defense_points': defense_points,
            'already_secured': already_secured,
            'improvement_potential': improvement_potential,
            'pressure_index': pressure_index
        }

    def analyze_surface_specialization(self, player_history, surface, player_name):
        """
        🔬 Analiza la especialización y calidad de un jugador en una superficie específica.
        Recompensa victorias de calidad y considera la tasa de victorias.
        """
        if not surface or surface == 'Desconocida':
            return 0, []

        normalized_surface = self.SURFACE_NORMALIZATION_MAP.get(surface.lower())
        if not normalized_surface:
            return 0, [f"LOG_SURFACE: Superficie del partido actual '{surface}' no se pudo normalizar."]

        surface_matches = [m for m in player_history if m.get('superficie', '').lower() == normalized_surface]
        
        if len(surface_matches) < 5: # No es significativo si hay menos de 5 partidos
            return 0, [f"LOG_SURFACE: No hay suficientes partidos en {surface} ({len(surface_matches)}) para un análisis profundo."]

        quality_score = 0
        analysis_log = []

        for match in surface_matches:
            opponent_rank = self.ranking_manager.get_player_ranking(match.get('oponente'))
            if not opponent_rank:
                continue

            points = 0
            # Puntos base por victoria según ranking del oponente
            if self.determine_match_winner(match, player_name):
                if opponent_rank <= 10: points = 50
                elif opponent_rank <= 20: points = 40
                elif opponent_rank <= 50: points = 25
                elif opponent_rank <= 100: points = 15
                else: points = 5
                
                # Multiplicador por contundencia
                contundencia = self.analizar_contundencia(match.get('resultado', ''))
                points *= contundencia
                analysis_log.append(f"Victoria vs Rank {opponent_rank} ({match.get('oponente')}) en {surface} -> +{points:.1f} pts")
            else:
                # Puntos por derrota competitiva
                resistencia = self.analizar_resistencia(match.get('resultado', ''))
                if resistencia > 0:
                    points = 10 * resistencia # Max 5 puntos por derrota ajustada
                    analysis_log.append(f"Derrota competitiva vs Rank {opponent_rank} ({match.get('oponente')}) en {surface} -> +{points:.1f} pts")

            quality_score += points

        # Normalizar por número de partidos para no favorecer a quien jugó más
        normalized_quality_score = (quality_score / len(surface_matches)) * 2.5
        
        # Factor de tasa de victorias
        wins = sum(1 for m in surface_matches if self.determine_match_winner(m, player_name))
        win_rate = (wins / len(surface_matches))
        
        final_score = normalized_quality_score * (1 + win_rate) # Ponderar por win_rate
        
        analysis_log.insert(0, f"Puntuación de Calidad en {surface}: {final_score:.1f} (Base: {quality_score:.1f}, Partidos: {len(surface_matches)}, Tasa Vic: {win_rate:.1%})")

        return final_score, analysis_log

    def analyze_surface_performance(self, player_history, player_name):
        """📊 Analiza el rendimiento de un jugador por superficie."""
        surface_stats = {
            'Dura': {'wins': 0, 'losses': 0, 'matches': 0},
            'Arcilla': {'wins': 0, 'losses': 0, 'matches': 0},
            'Hierba': {'wins': 0, 'losses': 0, 'matches': 0},
            'Indoor': {'wins': 0, 'losses': 0, 'matches': 0},
            'Desconocida': {'wins': 0, 'losses': 0, 'matches': 0}
        }

        for match in player_history:
            surface = match.get('superficie', 'Desconocida')
            if surface not in surface_stats:
                surface = 'Desconocida'
            
            surface_stats[surface]['matches'] += 1
            if self.determine_match_winner(match, player_name):
                surface_stats[surface]['wins'] += 1
            else:
                surface_stats[surface]['losses'] += 1
        
        for surface, stats in surface_stats.items():
            if stats['matches'] > 0:
                stats['win_rate'] = round((stats['wins'] / stats['matches']) * 100, 1)
            else:
                stats['win_rate'] = 0.0
        
        return surface_stats

    def analyze_location_factors(self, player_history, player_nationality):
        """🌍 Analiza el factor "Home Advantage" y el rendimiento por región."""
        if not player_nationality or player_nationality == 'N/A':
            return {'home_advantage': None, 'regional_comfort': {}}

        home_stats = {'wins': 0, 'losses': 0, 'matches': 0}
        regional_stats = {}

        for match in player_history:
            match_country = match.get('pais')
            if not match_country:
                continue

            # 1. Home Advantage
            if match_country == player_nationality:
                home_stats['matches'] += 1
                # 1. Home Advantage
            if match_country == player_nationality:
                home_stats['matches'] += 1
                if self.determine_match_winner(match, player_nationality):
                    home_stats['wins'] += 1
                else:
                    home_stats['losses'] += 1
            
            # 2. Regional Comfort
            continent = self.COUNTRY_TO_CONTINENT_MAP.get(match_country, 'Otros')
            if continent not in regional_stats:
                regional_stats[continent] = {'wins': 0, 'losses': 0, 'matches': 0}
            
            regional_stats[continent]['matches'] += 1
            if self.determine_match_winner(match, player_nationality):
                regional_stats[continent]['wins'] += 1
            else:
                regional_stats[continent]['losses'] += 1

        # Calcular tasas de victoria
        if home_stats['matches'] > 0:
            home_stats['win_rate'] = round((home_stats['wins'] / home_stats['matches']) * 100, 1)
        else:
            home_stats['win_rate'] = 0.0

        for region, stats in regional_stats.items():
            if stats['matches'] > 0:
                stats['win_rate'] = round((stats['wins'] / stats['matches']) * 100, 1)
            else:
                stats['win_rate'] = 0.0
        
        return {'home_advantage': home_stats, 'regional_comfort': regional_stats}

    def analyze_rivalry(self, player1_history, player2_history, player1_name, player2_name, player1_form, player2_form, direct_h2h_matches, current_match_context, p1_elo, p2_elo, tournament_name='', optimized_weights=None):
        """⚔️ Realizar análisis completo de rivalidad transitiva"""
        logger.info(f"⚔️ Analizando rivalidad transitiva: {player1_name} vs {player2_name}")
        
        # MEJORADO: Usar get_player_info para una búsqueda de ranking más robusta
        player1_info = self.ranking_manager.get_player_info(player1_name)
        player2_info = self.ranking_manager.get_player_info(player2_name)
        
        player1_rank = player1_info.get('ranking_position') if player1_info else None
        player2_rank = player2_info.get('ranking_position') if player2_info else None
        player1_nationality = player1_info.get('nationality') if player1_info else None
        player2_nationality = player2_info.get('nationality') if player2_info else None

        # Fallback para obtener nacionalidad del historial del oponente
        if not player1_nationality or player1_nationality == 'N/A':
            for match in player2_history:
                if self.ranking_manager.normalize_name(match.get('oponente', '')) == self.ranking_manager.normalize_name(player1_name):
                    if match.get('opponent_nationality') and match.get('opponent_nationality') != 'N/A':
                        player1_nationality = match['opponent_nationality']
                        logger.info(f"   ℹ️ Nacionalidad de {player1_name} inferida del historial de {player2_name}: {player1_nationality}")
                        break
        
        if not player2_nationality or player2_nationality == 'N/A':
            for match in player1_history:
                if self.ranking_manager.normalize_name(match.get('oponente', '')) == self.ranking_manager.normalize_name(player2_name):
                    if match.get('opponent_nationality') and match.get('opponent_nationality') != 'N/A':
                        player2_nationality = match['opponent_nationality']
                        logger.info(f"   ℹ️ Nacionalidad de {player2_name} inferida del historial de {player1_name}: {player2_nationality}")
                        break
        
        logger.info(f"   📊 Rankings: {player1_name} ({player1_rank}, {player1_nationality}) vs {player2_name} ({player2_rank}, {player2_nationality})")
        logger.info(f"   📍 Contexto Partido Actual: Torneo en {current_match_context.get('country')}, Superficie: {current_match_context.get('surface')}")
        
        # Realizar análisis de superficie y ubicación
        p1_surface_stats = self.analyze_surface_performance(player1_history, player1_name)
        p2_surface_stats = self.analyze_surface_performance(player2_history, player2_name)
        p1_location_stats = self.analyze_location_factors(player1_history, player1_nationality)
        p2_location_stats = self.analyze_location_factors(player2_history, player2_nationality)
        
        common_opponents = self.find_common_opponents(player1_history, player2_history)
        logger.info(f"   🤝 Oponentes comunes encontrados: {len(common_opponents)}")
        if common_opponents:
            logger.info("      Oponentes: " + ", ".join(opp for opp in common_opponents[:10]))
        
        prediction_context = {
            'p1_surface_stats': p1_surface_stats,
            'p2_surface_stats': p2_surface_stats,
            'p1_nationality': player1_nationality,
            'p2_nationality': player2_nationality,
            'current_match_surface': current_match_context.get('surface'),
            'current_match_country': current_match_context.get('country')
        }

        if not common_opponents:
            return {
                'player1_rank': player1_rank,
                'player2_rank': player2_rank,
            'common_opponents_count': 0,
            'p1_rivalry_score': 0,
            'p2_rivalry_score': 0,
            'player1_advantages': [],
            'player2_advantages': [],
            'p1_surface_stats': p1_surface_stats,
            'p2_surface_stats': p2_surface_stats,
            'p1_location_stats': p1_location_stats,
            'p2_location_stats': p2_location_stats,
            'prediction': self.generate_advanced_prediction(player1_info, player2_info, 0, 0, player1_name, player2_name, player1_history, player2_history, 0, 0, player1_form, player2_form, direct_h2h_matches, tournament_name, prediction_context, p1_elo, p2_elo, optimized_weights=optimized_weights)
        }
        
        player1_advantages = []
        player2_advantages = []
        p1_common_opponent_score = 0
        p2_common_opponent_score = 0
        
        for common_opponent in common_opponents:
            p1_matches = [m for m in player1_history if self.ranking_manager.normalize_name(m.get('oponente', '')) == common_opponent]
            p2_matches = [m for m in player2_history if self.ranking_manager.normalize_name(m.get('oponente', '')) == common_opponent]
            
            if p1_matches and p2_matches:
                p1_recent = p1_matches[0]
                p2_recent = p2_matches[0]
                
                p1_won = self.determine_match_winner(p1_recent, player1_name)
                p2_won = self.determine_match_winner(p2_recent, player2_name)
                
                opponent_rank = self.ranking_manager.get_player_ranking(p1_recent.get('oponente', ''))
                
                weight = self.calcular_peso_oponentes_comunes(player1_history, player2_history, common_opponent, player1_name, player2_name)
                opponent_display_rank = f"(rank {opponent_rank})" if opponent_rank is not None else "(rank N/A)"
                common_opponent_name = p1_recent.get('oponente', '')

                # Escenario 1: P1 ganó, P2 perdió
                if p1_won and not p2_won:
                    advantage_points = weight
                    if opponent_rank:
                        if opponent_rank <= 15: advantage_points *= 1.5  # AJUSTADO
                        elif 16 <= opponent_rank <= 30: advantage_points *= 1.25 # AJUSTADO
                        elif 31 <= opponent_rank <= 50: advantage_points *= 1.1  # AJUSTADO
                    
                    reason = f"{player1_name} venció a {common_opponent_name} {opponent_display_rank}, mientras que {player2_name} perdió"
                    advantage = {'opponent': common_opponent_name, 'opponent_rank': opponent_rank, 'weight': advantage_points, 'reason': reason, 'player1_result': p1_recent.get('resultado', ''), 'player2_result': p2_recent.get('resultado', '')}
                    player1_advantages.append(advantage)
                    p1_common_opponent_score += advantage_points

                # Escenario 2: P2 ganó, P1 perdió
                elif p2_won and not p1_won:
                    advantage_points = weight
                    if opponent_rank:
                        if opponent_rank <= 15: advantage_points *= 1.5  # AJUSTADO
                        elif 16 <= opponent_rank <= 30: advantage_points *= 1.25 # AJUSTADO
                        elif 31 <= opponent_rank <= 50: advantage_points *= 1.1  # AJUSTADO

                    reason = f"{player2_name} venció a {common_opponent_name} {opponent_display_rank}, mientras que {player1_name} perdió"
                    advantage = {'opponent': common_opponent_name, 'opponent_rank': opponent_rank, 'weight': advantage_points, 'reason': reason, 'player1_result': p1_recent.get('resultado', ''), 'player2_result': p2_recent.get('resultado', '')}
                    player2_advantages.append(advantage)
                    p2_common_opponent_score += advantage_points

                # Escenario 3: Ambos ganaron
                elif p1_won and p2_won:
                    contundencia_p1 = self.analizar_contundencia(p1_recent.get('resultado', ''))
                    contundencia_p2 = self.analizar_contundencia(p2_recent.get('resultado', ''))
                    
                    advantage_points = (contundencia_p1 - contundencia_p2) * weight
                    
                    if advantage_points > 0:
                        reason = f"Ambos vencieron a {common_opponent_name} {opponent_display_rank}, pero {player1_name} fue más contundente (Factor: {contundencia_p1:.1f} vs {contundencia_p2:.1f})"
                        advantage = {'opponent': common_opponent_name, 'opponent_rank': opponent_rank, 'weight': advantage_points, 'reason': reason, 'player1_result': p1_recent.get('resultado', ''), 'player2_result': p2_recent.get('resultado', '')}
                        player1_advantages.append(advantage)
                        p1_common_opponent_score += advantage_points
                    elif advantage_points < 0:
                        reason = f"Ambos vencieron a {common_opponent_name} {opponent_display_rank}, pero {player2_name} fue más contundente (Factor: {contundencia_p2:.1f} vs {contundencia_p1:.1f})"
                        advantage = {'opponent': common_opponent_name, 'opponent_rank': opponent_rank, 'weight': abs(advantage_points), 'reason': reason, 'player1_result': p1_recent.get('resultado', ''), 'player2_result': p2_recent.get('resultado', '')}
                        player2_advantages.append(advantage)
                        p2_common_opponent_score += abs(advantage_points)

                # Escenario 4: Ambos perdieron
                elif not p1_won and not p2_won:
                    resistencia_p1 = self.analizar_resistencia(p1_recent.get('resultado', ''))
                    resistencia_p2 = self.analizar_resistencia(p2_recent.get('resultado', ''))

                    advantage_points = (resistencia_p1 - resistencia_p2) * weight
                    
                    if advantage_points > 0:
                        reason = f"Ambos perdieron con {common_opponent_name} {opponent_display_rank}, pero {player1_name} mostró más resistencia (Factor: {resistencia_p1:.1f} vs {resistencia_p2:.1f})"
                        advantage = {'opponent': common_opponent_name, 'opponent_rank': opponent_rank, 'weight': advantage_points, 'reason': reason, 'player1_result': p1_recent.get('resultado', ''), 'player2_result': p2_recent.get('resultado', '')}
                        player1_advantages.append(advantage)
                        p1_common_opponent_score += advantage_points
                    elif advantage_points < 0:
                        reason = f"Ambos perdieron con {common_opponent_name} {opponent_display_rank}, pero {player2_name} mostró más resistencia (Factor: {resistencia_p2:.1f} vs {resistencia_p1:.1f})"
                        advantage = {'opponent': common_opponent_name, 'opponent_rank': opponent_rank, 'weight': abs(advantage_points), 'reason': reason, 'player1_result': p1_recent.get('resultado', ''), 'player2_result': p2_recent.get('resultado', '')}
                        player2_advantages.append(advantage)
                        p2_common_opponent_score += abs(advantage_points)
        
        if player1_advantages:
            logger.info(f"   ⚡ Ventajas de {player1_name}:")
            for adv in player1_advantages[:3]: logger.info(f"      • {adv['reason']} (Peso: {adv['weight']:.2f})")
        
        if player2_advantages:
            logger.info(f"   ⚡ Ventajas de {player2_name}:")
            for adv in player2_advantages[:3]: logger.info(f"      • {adv['reason']} (Peso: {adv['weight']:.2f})")
        
        logger.info(f"   ⚖️ Peso de rivalidad P1: {p1_common_opponent_score:.2f}, P2: {p2_common_opponent_score:.2f}")
        
        return {
            'player1_rank': player1_rank,
            'player2_rank': player2_rank,
            'player1_nationality': player1_nationality,
            'player2_nationality': player2_nationality,
            'common_opponents_count': len(common_opponents),
            'common_opponents': common_opponents,
            'p1_rivalry_score': p1_common_opponent_score,
            'p2_rivalry_score': p2_common_opponent_score,
            'player1_advantages': player1_advantages,
            'player2_advantages': player2_advantages,
            'p1_surface_stats': p1_surface_stats,
            'p2_surface_stats': p2_surface_stats,
            'p1_location_stats': p1_location_stats,
            'p2_location_stats': p2_location_stats,
            'prediction': self.generate_advanced_prediction(player1_info, player2_info, p1_common_opponent_score, p2_common_opponent_score, player1_name, player2_name, player1_history, player2_history, len(player1_advantages), len(player2_advantages), player1_form, player2_form, direct_h2h_matches, tournament_name, prediction_context, p1_elo, p2_elo, optimized_weights=optimized_weights)
        }
    
    def generate_basic_prediction(self, rank1, rank2, player1_name, player2_name):
        """🔮 Generar predicción básica solo con rankings"""
        if rank1 is None and rank2 is None: return {'favored_player': 'Empate', 'confidence': 0, 'reasoning': ['Ambos jugadores sin ranking disponible']}
        if rank1 is None: return {'favored_player': player2_name, 'confidence': 60, 'reasoning': [f'{player2_name} tiene ranking ({rank2}), {player1_name} sin ranking']}
        if rank2 is None: return {'favored_player': player1_name, 'confidence': 60, 'reasoning': [f'{player1_name} tiene ranking ({rank1}), {player2_name} sin ranking']}
        
        rank_diff = abs(rank1 - rank2)
        if rank1 < rank2:
            return {'favored_player': player1_name, 'confidence': min(50 + (rank_diff * 2), 90), 'reasoning': [f'{player1_name} tiene mejor ranking ({rank1} vs {rank2})']}
        elif rank2 < rank1:
            return {'favored_player': player2_name, 'confidence': min(50 + (rank_diff * 2), 90), 'reasoning': [f'{player2_name} tiene mejor ranking ({rank2} vs {rank1})']}
        else:
            return {'favored_player': 'Empate', 'confidence': 0, 'reasoning': ['Rankings idénticos']}
    
    def classify_tournament(self, tournament_name):
        """Categoriza el torneo para ajustar los pesos del análisis."""
        if not tournament_name:
            return 'default'
        
        name_lower = tournament_name.lower()
        
        # Palabras clave para torneos de alto nivel (ATP/WTA)
        if any(keyword in name_lower for keyword in ['atp', 'wta', 'grand slam', 'masters', 'olympic', 'united cup']):
            return 'atp_wta'
            
        # Palabras clave para Challengers
        if 'challenger' in name_lower:
            return 'challenger'
            
        # Palabras clave para ITFs
        if any(keyword in name_lower for keyword in ['itf', 'm15', 'm25', 'w15', 'w25', 'w35', 'w50', 'w75', 'w100']):
            return 'itf'
            
        # Si no coincide, usar un set de pesos por defecto (similar a ATP)
        return 'default'

    def generate_advanced_prediction(self, player1_info, player2_info, p1_rivalry_score, p2_rivalry_score, player1_name, player2_name, player1_history, player2_history, player1_advantages_count, player2_advantages_count, player1_form, player2_form, direct_h2h_matches, tournament_name, prediction_context, p1_elo, p2_elo, optimized_weights=None):
        """🎯 Generar predicción avanzada aplicando la fórmula de Peso Final - VERSIÓN REBALANCEADA Y NORMALIZADA."""
        
        reasoning = []

        # Extraer ranks y ELOs
        rank1 = player1_info.get('ranking_position') if player1_info else None
        rank2 = player2_info.get('ranking_position') if player2_info else None
        elo1 = p1_elo
        elo2 = p2_elo
        reasoning.append(f"LOG_ELO_RATINGS: {player1_name}={elo1}, {player2_name}={elo2}")

        # 1. AJUSTE DE PESOS
        if optimized_weights:
            weights = optimized_weights
            reasoning.append(f"LOG_WEIGHTS_OPTIMIZED: {weights}")
        else:
            tournament_category = self.classify_tournament(tournament_name)
            
            # PESOS REBALANCEADOS CON ELO Y MOMENTUM
            weights_config = {
                'atp_wta': {
                    'surface_specialization': 0.15, 'form_recent': 0.15, 'common_opponents': 0.20, 
                    'h2h_direct': 0.15, 'ranking_momentum': 0.20, 'elo_rating': 0.10, 'home_advantage': 0.05, 'strength_of_schedule': 0.0
                },
                'challenger': {
                    'surface_specialization': 0.20, 'form_recent': 0.20, 'common_opponents': 0.15, 
                    'ranking_momentum': 0.20, 'elo_rating': 0.15, 'h2h_direct': 0.05, 'home_advantage': 0.05, 'strength_of_schedule': 0.0
                },
                'itf': {
                    'surface_specialization': 0.15, 'form_recent': 0.25, 'common_opponents': 0.10, 
                    'ranking_momentum': 0.20, 'elo_rating': 0.15, 'h2h_direct': 0.05, 'home_advantage': 0.05, 'strength_of_schedule': 0.05
                },
                'default': {
                    'surface_specialization': 0.15, 'form_recent': 0.15, 'common_opponents': 0.20, 
                    'h2h_direct': 0.15, 'ranking_momentum': 0.20, 'elo_rating': 0.10, 'home_advantage': 0.05, 'strength_of_schedule': 0.0
                }
            }
            weights = weights_config.get(tournament_category, weights_config['default'])
            reasoning.append(f"LOG_WEIGHTS_STRATEGY: '{tournament_category}' -> {weights}")

        # --- OBTENER MULTIPLICADORES Y SCORES BASE ---

        # NUEVO: ANÁLISIS DE ESPECIALIZACIÓN POR SUPERFICIE
        current_surface = prediction_context.get('current_match_surface')
        p1_surface_score, p1_surface_log = self.analyze_surface_specialization(player1_history, current_surface, player1_name)
        p2_surface_score, p2_surface_log = self.analyze_surface_specialization(player2_history, current_surface, player2_name)
        if p1_surface_log: reasoning.extend([f"P1_LOG_SURF: {log}" for log in p1_surface_log])
        if p2_surface_log: reasoning.extend([f"P2_LOG_SURF: {log}" for log in p2_surface_log])

        p1_streak_mult, p1_streak_log = self.analyze_streaks_and_consistency(player1_history, player1_name)
        p2_streak_mult, p2_streak_log = self.analyze_streaks_and_consistency(player2_history, player2_name)
        if p1_streak_log: reasoning.extend([f"P1_LOG_MOM: {log}" for log in p1_streak_log])
        if p2_streak_log: reasoning.extend([f"P2_LOG_MOM: {log}" for log in p2_streak_log])

        p1_quality_mult, p1_diversity_mult, p1_adv_log = self.analyze_advanced_player_metrics(player1_history, player1_name)
        p2_quality_mult, p2_diversity_mult, p2_adv_log = self.analyze_advanced_player_metrics(player2_history, player2_name)
        if p1_adv_log: reasoning.extend([f"P1_LOG_DIV: {log}" for log in p1_adv_log])
        if p2_adv_log: reasoning.extend([f"P2_LOG_DIV: {log}" for log in p2_adv_log])

        p1_schedule_score, p1_schedule_log = self.analyze_strength_of_schedule(player1_history, player1_name)
        p2_schedule_score, p2_schedule_log = self.analyze_strength_of_schedule(player2_history, player2_name)
        if p1_schedule_log: reasoning.extend([f"P1_LOG_SoS: {log}" for log in p1_schedule_log])
        if p2_schedule_log: reasoning.extend([f"P2_LOG_SoS: {log}" for log in p2_schedule_log])

        dominance_multiplier_p1 = 1.5 if player1_advantages_count >= 3 else 1.0
        dominance_multiplier_p2 = 1.5 if player2_advantages_count >= 3 else 1.0

        p1_h2h_score, p2_h2h_score, h2h_log = self.analyze_direct_h2h(direct_h2h_matches, player1_name, player2_name)
        if h2h_log: reasoning.extend(h2h_log)

        # --- CÁLCULO DE COMPONENTES RAW (CON LÍMITES) ---
        
        def calculate_raw_scores(player_info, elo, streak_mult, quality_mult, diversity_mult, rivalry_score, dominance_mult, form, h2h_score, schedule_score, player_context):
            raw_scores = {}
            rank = player_info.get('ranking_position') if player_info else None

            # 0a. VENTAJA DE LOCAL (Límite: 100)
            home_advantage_score = 0
            player_nat = player_context.get('nationality', 'N/A')
            match_country = player_context.get('current_match_country', 'N/A')
            if player_nat and player_nat != 'N/A' and match_country and match_country != 'N/A':
                if player_nat.strip().lower() in match_country.strip().lower() or match_country.strip().lower() in player_nat.strip().lower():
                    home_advantage_score += 50
                    reasoning.append(f"LOG_CONTEXT_{player_context['player_name']}: 🏠 Home Advantage Bonus (+50 pts)")
            raw_scores['home_advantage'] = min(home_advantage_score, 100)

            # 0b. ESPECIALIZACIÓN EN SUPERFICIE (Límite: 350)
            surface_quality_score = player_context.get('surface_quality_score', 0)
            raw_scores['surface_specialization'] = min(surface_quality_score, 350)

            # 1. Ranking & Momentum (Límite: 450) - LÓGICA MEJORADA CON DATOS REALES DE FLASHSCORE
            ranking_momentum_score = 0
            player_metrics = self.get_ranking_metrics(player_context['full_name'])

            if player_metrics:
                # A. Puntuación Base (Pts Totales): Calidad y consistencia histórica.
                base_score = math.log1p(player_metrics['pts']) * 20

                # B. Momentum Asegurado (Already Secured): Rendimiento inmediato en el torneo.
                momentum_bonus = math.log1p(player_metrics['already_secured']) * 15

                # C. Potencial y Motivación (Improvement Potential): Techo y motivación.
                potential_bonus = math.log1p(player_metrics['improvement_potential']) * 10

                # D. Índice de Presión (Pressure Index): Factor psicológico.
                # Un índice negativo (ya defendió más de lo necesario) es bueno.
                # Un índice positivo alto (mucha presión) es malo.
                pressure_factor = 0
                if player_metrics['pressure_index'] < 0:
                    pressure_factor = math.log1p(abs(player_metrics['pressure_index'])) * 5 # Bonus por jugar suelto
                else:
                    pressure_factor = -math.log1p(player_metrics['pressure_index']) * 5 # Penalización por presión

                reasoning.append(   
                    f"LOG_RANKING_{player_context['player_name']}: "
                    f"Base({player_metrics['pts']}): {base_score:.2f}, "
                    f"Momentum({player_metrics['already_secured']}): {momentum_bonus:.2f}, "
                    f"Potencial({player_metrics['improvement_potential']}): {potential_bonus:.2f}, "
                    f"Presión({player_metrics['pressure_index']}): {pressure_factor:.2f}"
                )
                
                ranking_momentum_score = base_score + momentum_bonus + potential_bonus + pressure_factor
            
            raw_scores['ranking_momentum'] = min(ranking_momentum_score, 450)

            # 2. Forma Reciente (Límite: 300)
            form_score = 0
            if form and form.get('win_percentage') is not None:
                win_pct = form['win_percentage']
                form_score = (win_pct / 100) * 200  # Puntuación base
                
                if form.get('current_streak_type') == 'victorias':
                    form_score += form.get('current_streak_count', 0) * 15
            raw_scores['form_recent'] = min(form_score * streak_mult, 300)

            # 3. Rivales Comunes (Límite: 400)
            common_opp_score = rivalry_score * quality_mult * dominance_mult
            raw_scores['common_opponents'] = min(common_opp_score, 400)

            # 4. H2H Directo (Límite: 350)
            raw_scores['h2h_direct'] = min(h2h_score * 100, 350)

            # 5. Rating ELO (Límite: 250)
            # ELO ya es una puntuación, solo se necesita limitar. 
            # Se resta el ELO por defecto para normalizar.
            raw_scores['elo_rating'] = max(0, elo - 1500) 

            # 6. Strength of Schedule (Límite: 200)
            raw_scores['strength_of_schedule'] = min(schedule_score * diversity_mult, 200)

            # Calcular días desde el último partido
            days_since = -1  # Valor por defecto si no hay datos
            if form and form.get('last_match_date') and form['last_match_date'] != 'N/A':
                try:
                    last_match_date = datetime.strptime(form['last_match_date'], '%d.%m.%y')
                    days_since = (datetime.now() - last_match_date).days
                except ValueError:
                    days_since = -1

            return raw_scores, days_since

        p1_context = {
            'player_name': 'P1',
            'full_name': player1_name,
            'nationality': prediction_context['p1_nationality'],
            'current_match_country': prediction_context['current_match_country'],
            'surface_quality_score': p1_surface_score
        }
        p2_context = {
            'player_name': 'P2',
            'full_name': player2_name,
            'nationality': prediction_context['p2_nationality'],
            'current_match_country': prediction_context['current_match_country'],
            'surface_quality_score': p2_surface_score
        }

        raw_p1, days_since_p1 = calculate_raw_scores(player1_info, elo1, p1_streak_mult, p1_quality_mult, p1_diversity_mult, p1_rivalry_score, dominance_multiplier_p1, player1_form, p1_h2h_score, p1_schedule_score, p1_context)
        raw_p2, days_since_p2 = calculate_raw_scores(player2_info, elo2, p2_streak_mult, p2_quality_mult, p2_diversity_mult, p2_rivalry_score, dominance_multiplier_p2, player2_form, p2_h2h_score, p2_schedule_score, p2_context)

        # --- LÓGICA DE PONDERACIÓN DINÁMICA (H2H Antiguo vs. Rivales Comunes) ---
        try:
            if direct_h2h_matches:
                last_match_date = datetime.strptime(direct_h2h_matches[0]['fecha'], '%d.%m.%y')
                last_match_days_ago = (datetime.now() - last_match_date).days
                H2H_ANTIQUITY_THRESHOLD = 730 # 2 años

                if last_match_days_ago > H2H_ANTIQUITY_THRESHOLD:
                    h2h_winner_is_p1 = raw_p1['h2h_direct'] > raw_p2['h2h_direct']
                    common_opp_winner_is_p1 = raw_p1['common_opponents'] > raw_p2['common_opponents']

                    # Comprobar si hay contradicción
                    if h2h_winner_is_p1 != common_opp_winner_is_p1:
                        # Determinar quién tiene la ventaja en rivales comunes y por cuánto
                        if common_opp_winner_is_p1:
                            advantage_player_raw_common = raw_p1['common_opponents']
                            disadvantage_player_raw_common = raw_p2['common_opponents']
                        else:
                            advantage_player_raw_common = raw_p2['common_opponents']
                            disadvantage_player_raw_common = raw_p1['common_opponents']

                        # Solo aplicar si la ventaja en rivales comunes es significativa
                        if advantage_player_raw_common > disadvantage_player_raw_common * 1.25: # 25% de ventaja
                            reasoning.append(f"LOG_DYNAMIC_WEIGHTING: H2H antiguo ({last_match_days_ago} días) contradicho por una ventaja significativa en Rivales Comunes.")
                            
                            if common_opp_winner_is_p1:
                                # P1 tiene ventaja en rivales, P2 en H2H antiguo
                                reasoning.append("   -> Bonificando Rivales Comunes de P1 y reduciendo H2H de P2.")
                                raw_p1['common_opponents'] *= 1.30 # Bonificación del 30%
                                raw_p2['h2h_direct'] *= 0.15       # Reducción al 15%
                            else:
                                # P2 tiene ventaja en rivales, P1 en H2H antiguo
                                reasoning.append("   -> Bonificando Rivales Comunes de P2 y reduciendo H2H de P1.")
                                raw_p2['common_opponents'] *= 1.30 # Bonificación del 30%
                                raw_p1['h2h_direct'] *= 0.15       # Reducción al 15%
        except Exception as e:
            reasoning.append(f"LOG_DYNAMIC_WEIGHTING_ERROR: {e}")

        # 2. NORMALIZACIÓN DE SCORES
        def normalize_scores(p1_scores, p2_scores):
            normalized_p1 = {}
            normalized_p2 = {}
            for key in p1_scores:
                p1_val = p1_scores[key]
                p2_val = p2_scores[key]
                
                # Normalización Logarítmica: log(1 + x) para manejar scores de 0
                norm_p1 = math.log1p(p1_val)
                norm_p2 = math.log1p(p2_val)
                
                normalized_p1[key] = norm_p1
                normalized_p2[key] = norm_p2
            return normalized_p1, normalized_p2

        norm_p1, norm_p2 = normalize_scores(raw_p1, raw_p2)
        reasoning.append(f"LOG_RAW_SCORES_P1: { {k: round(v, 2) for k, v in raw_p1.items()} }")
        reasoning.append(f"LOG_RAW_SCORES_P2: { {k: round(v, 2) for k, v in raw_p2.items()} }")
        reasoning.append(f"LOG_NORM_SCORES_P1: { {k: round(v, 2) for k, v in norm_p1.items()} }")
        reasoning.append(f"LOG_NORM_SCORES_P2: { {k: round(v, 2) for k, v in norm_p2.items()} }")

        # --- APLICAR PESOS Y CALCULAR PUNTAJE FINAL ---
        def apply_weights_and_penalties(normalized_scores, weights, days_since):
            weighted_scores = {k: normalized_scores[k] * weights[k] for k in weights}
            
            penalty = 0
            if days_since == -1: penalty = sum(weighted_scores.values()) * 0.3
            elif days_since > 30: penalty = min(sum(weighted_scores.values()) * 0.5, (days_since - 30) * 0.1)

            final_score = sum(weighted_scores.values()) - penalty
            return final_score, weighted_scores, penalty

        final_p1, weighted_p1, penalty_p1 = apply_weights_and_penalties(norm_p1, weights, days_since_p1)
        final_p2, weighted_p2, penalty_p2 = apply_weights_and_penalties(norm_p2, weights, days_since_p2)

        # --- GENERAR DESGLOSE DETALLADO PARA LOGS ---
        def get_breakdown(raw, norm, weighted, penalty, final_score):
            breakdown = {}
            total_weighted = sum(weighted.values())
            if total_weighted > 0:
                for k, v in weighted.items():
                    breakdown[k] = {
                        'raw_score': f"{raw[k]:.1f}",
                        'normalized_score': f"{norm[k]:.2f}",
                        'weight': f"{weights[k]*100:.0f}%",
                        'weighted_score': f"{v:.2f}",
                        'contribution': f"{(v / total_weighted) * 100:.1f}%" if total_weighted > 0 else "0.0%"
                    }
            breakdown['Penalizacion_Inactividad'] = f"{-penalty:.2f} pts"
            breakdown['Puntaje_Final'] = f"{final_score:.2f}"
            return breakdown

        breakdown_p1 = get_breakdown(raw_p1, norm_p1, weighted_p1, penalty_p1, final_p1)
        breakdown_p2 = get_breakdown(raw_p2, norm_p2, weighted_p2, penalty_p2, final_p2)

        # --- DECISIÓN FINAL Y CONFIANZA ---
        score_diff = final_p1 - final_p2
        total_score = final_p1 + final_p2
        
        confidence = 50
        if total_score > 0:
            confidence = 50 + (abs(score_diff) / total_score * 50)

        if score_diff > 0.01: # Umbral mínimo para decidir un ganador
            favored = player1_name
        elif score_diff < -0.01:
            favored = player2_name
        else:
            if rank1 is not None and rank2 is not None:
                favored = player1_name if rank1 < rank2 else player2_name
                confidence = 51
            else:
                favored = 'Empate'
                confidence = 50

        return {
            'favored_player': favored,
            'confidence': round(min(confidence, 95.0), 1),
            'reasoning': reasoning,
            'scores': {
                'p1_final_weight': round(final_p1, 2),
                'p2_final_weight': round(final_p2, 2),
                'score_difference': round(score_diff, 2)
            },
            'score_breakdown': {
                'player1': breakdown_p1,
                'player2': breakdown_p2
            },
            'weights_used': weights
        }
    
class SequentialH2HExtractor:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.is_wsl = verificar_sistema()
        
        # Cola de partidos a procesar
        self.matches_queue = []
        self.current_match_index = 0
        self.total_matches_to_process = 80
                
        # Resultados consolidados
        self.all_results = []
        
        # NUEVA FUNCIONALIDAD: Gestores de ranking, ELO y rivalidad
        self.ranking_manager = RankingManager()
        self.elo_system = EloRatingSystem()
        self.rivalry_analyzer = RivalryAnalyzer(self.ranking_manager, self.elo_system)
        
        # Registrar manejadores de señales
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._cleanup_on_exit)
        
    def _signal_handler(self, signum, frame):
        """🛑 Manejador de señales de interrupción"""
        logger.info("🛑 Señal de interrupción recibida - Iniciando limpieza...")
        asyncio.create_task(self.cleanup())
        
    def _cleanup_on_exit(self):
        """🧹 Limpieza al salir"""
        try:
            if self.browser and hasattr(self.browser, 'close'):
                asyncio.run(self.cleanup())
        except:
            pass
        
    @asynccontextmanager
    async def navegador_seguro(self):
        """🔒 Context manager para manejo seguro del navegador"""
        try:
            await self.setup_browser()
            yield self.playwright, self.browser, self.context, self.page
        except Exception as e:
            logger.error(f"❌ Error en navegador: {e}")
            raise
        finally:
            await self.cleanup()
        
    async def setup_browser(self):
        """🚀 Configurar navegador optimizado para WSL"""
        logger.info("🚀 Configurando navegador para WSL...")
        
        try:
            self.playwright = await async_playwright().start()
            
            # Configuración específica para WSL
            browser_args = [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--disable-software-rasterizer',
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding',
                '--disable-features=VizDisplayCompositor,TranslateUI',
                '--disable-ipc-flooding-protection',
                '--disable-blink-features=AutomationControlled',
                '--no-first-run',
                '--password-store=basic',
                '--use-mock-keychain'
            ]
            
            logger.info("🔧 Usando configuración de navegador simplificada para mayor estabilidad.")
            self.browser = await self.playwright.chromium.launch(
                headless=True,  # Volver a True para mayor estabilidad
                args=browser_args,
                timeout=60000,
                slow_mo=250,    # Reducir un poco la lentitud
            )
            
            self.context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                ignore_https_errors=True,
                java_script_enabled=True,
            )
            
            # Timeouts más generosos para WSL
            self.context.set_default_timeout(60000)
            self.context.set_default_navigation_timeout(60000)
            
            self.page = await self.context.new_page()
            
            # Configurar eventos de error
            self.page.on("crash", lambda: logger.error("💥 Página crasheada"))
            self.page.on("pageerror", lambda err: logger.error(f"🚨 Error JS: {err}"))
            
            await self.page.set_extra_http_headers({
                'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
            })
            
            logger.info("✅ Navegador configurado exitosamente")
            
        except Exception as e:
            logger.error(f"❌ Error configurando navegador: {e}")
            await self.cleanup()
            raise
    
    def load_matches_from_json(self):
        """📂 Cargar los primeros 5 partidos del JSON y crear cola de procesamiento"""
        logger.info("📂 Cargando partidos desde JSON...")
        
        try:
            # Seleccionar el mejor archivo JSON automáticamente
            json_file = select_best_json_file()
            if not json_file:
                logger.error("❌ No se pudo encontrar un archivo JSON válido")
                return False
                
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"📊 Archivo cargado: {json_file}")
            
            # Extraer partidos de la estructura JSON, conservando el nombre del torneo
            all_matches = []
            
            if isinstance(data, dict):
                logger.info("   🔄 Procesando estructura de torneos (diccionario)...")
                for tournament_name, matches_in_tournament in data.items():
                    if isinstance(matches_in_tournament, list):
                        logger.info(f"      - Cargando {len(matches_in_tournament)} partidos de '{tournament_name}'")
                        for match in matches_in_tournament:
                            if isinstance(match, dict):
                                # Lógica corregida para extraer torneo, país y tipo de cancha
                                tournament_string = tournament_name
                                tipo_cancha = "Desconocida"
                                pais = "N/A"
                                
                                # 1. Extraer país del paréntesis
                                country_match = re.search(r'\((.*?)\)', tournament_string)
                                if country_match:
                                    pais = country_match.group(1).strip()

                                # 2. Extraer tipo de cancha
                                parts = tournament_string.split(',')
                                nombre_torneo_base = tournament_string
                                possible_court_types = ['hard', 'clay', 'grass', 'indoor']
                                if len(parts) > 1:
                                    last_part = parts[-1].strip().lower()
                                    is_court_type = any(court in last_part for court in possible_court_types)
                                    if is_court_type:
                                        tipo_cancha = parts[-1].strip()
                                        nombre_torneo_base = ','.join(parts[:-1])

                                # 3. Limpiar el nombre del torneo
                                if ':' in nombre_torneo_base:
                                    nombre_torneo_final = nombre_torneo_base.split(':', 1)[1].strip()
                                else:
                                    nombre_torneo_final = nombre_torneo_base.strip()
                                
                                # Re-añadir el país para el formato deseado
                                if pais != "N/A":
                                    nombre_torneo_final = f"{nombre_torneo_final} ({pais})"

                                match['torneo_nombre'] = nombre_torneo_final
                                match['tipo_cancha'] = tipo_cancha
                                match['pais'] = pais
                                match['torneo_completo'] = tournament_name
                                all_matches.append(match)
            elif isinstance(data, list):
                logger.info("   🔄 Procesando estructura de lista plana de partidos...")
                all_matches = data
            
            if not all_matches:
                logger.error("❌ No se encontraron partidos en el archivo JSON")
                return False
            
            # Filtrar partidos que tengan match_url
            valid_matches = []
            for match in all_matches:
                if isinstance(match, dict) and match.get('match_url'):
                    valid_matches.append(match)
            
            if not valid_matches:
                logger.error("❌ No se encontraron partidos con match_url válidas")
                return False
            
            # Tomar los primeros N partidos según la configuración
            self.matches_queue = valid_matches[:self.total_matches_to_process]
            logger.info(f"✅ Cola de procesamiento creada con {len(self.matches_queue)} partidos:")
            for i, match in enumerate(self.matches_queue, 1):
                player1 = match.get('jugador1', 'N/A')
                player2 = match.get('jugador2', 'N/A')
                cuota1 = match.get('cuota1', 'N/A')
                cuota2 = match.get('cuota2', 'N/A')
                match_url = match.get('match_url', 'N/A')
                logger.info(f"   {i}. {player1} vs {player2} (Cuotas: {cuota1} - {cuota2})")
                logger.info(f"      URL: {match_url}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cargando JSON: {e}")
            return False
    
    async def navigate_to_match_url(self, match_data):
        """🎯 Navegar a la URL específica del partido"""
        match_url = match_data.get('match_url', '')
        player1 = match_data.get('jugador1', 'N/A')
        player2 = match_data.get('jugador2', 'N/A')
        
        logger.info(f"🎯 Navegando al partido: {player1} vs {player2}")
        logger.info(f"   URL: {match_url}")
        
        try:
            # Navegar a la URL del partido
            await self.page.goto(match_url, wait_until='domcontentloaded', timeout=45000)
            
            # Esperar un momento para que cargue el contenido dinámico
            logger.info("⏳ Esperando carga del contenido...")
            await self.page.wait_for_timeout(8000)
            
            # Verificar que la página cargó correctamente
            title = await self.page.title()
            logger.info(f"📄 Título de la página: {title}")
            
            # Tomar screenshot para debugging
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            screenshot_path = f"match_{self.current_match_index + 1}_{timestamp}.png"
            await self.page.screenshot(path=screenshot_path)
            logger.info(f"📸 Screenshot guardado: {screenshot_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error navegando a URL del partido: {e}")
            return False
    
    async def navigate_to_h2h_section(self):
        """🔄 Navegar a la sección H2H dentro de la página del partido"""
        logger.info("🔄 Navegando a sección H2H...")
        
        try:
            # Posibles selectores para botón/enlace H2H
            h2h_selectors = [
                'a[href*="h2h"]',
                'button:has-text("H2H")',
                'a:has-text("H2H")',
                'a:has-text("Cara a cara")',
                'a:has-text("Head to Head")',
                'a:has-text("ENFRENTAMIENTOS")',
                '[data-testid*="h2h"]',
                '.tab-h2h',
                '#h2h-tab',
                '.tabs__tab:has-text("H2H")'
            ]
            
            for selector in h2h_selectors:
                try:
                    h2h_button = self.page.locator(selector).first
                    if await h2h_button.count() > 0 and await h2h_button.is_visible():
                        logger.info(f"   🎯 Encontrado botón H2H con selector: {selector}")
                        
                        await h2h_button.click()
                        await self.page.wait_for_timeout(5000)
                        
                        logger.info("   ✅ Click en H2H realizado")
                        return True
                        
                except Exception:
                    continue
            
            # Si no encontramos botón H2H, intentar modificar la URL
            logger.info("   🔄 Intentando navegación directa a H2H...")
            current_url = self.page.url
            
            # Diferentes patrones de URL para H2H
            h2h_url_patterns = [
                current_url.replace('#/match-summary', '#/h2h'),
                current_url.replace('#/resumen-del-partido/resumen-del-partido', '#/h2h'),
                current_url + '#/h2h' if '#' not in current_url else current_url.split('#')[0] + '#/h2h'
            ]
            
            for h2h_url in h2h_url_patterns:
                try:
                    await self.page.goto(h2h_url, wait_until='domcontentloaded', timeout=30000)
                    await self.page.wait_for_timeout(5000)
                    
                    logger.info(f"   🎯 Navegado a: {h2h_url}")
                    return True
                except:
                    continue
            
            logger.warning("⚠️ No se pudo acceder a la sección H2H")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error navegando a H2H: {e}")
            return False
    
    async def click_show_more_buttons(self, section_locator, section_name, max_clicks=8):
        """🔄 Hacer click en 'Mostrar más' hasta 8 veces en una sección específica"""
        logger.info(f"   🔄 Expandiendo sección {section_name} (máx {max_clicks} clicks)...")
        
        clicks_realizados = 0
        
        for i in range(max_clicks):
            try:
                # Buscar botón "Mostrar más" dentro de la sección
                show_more_selectors = [
                    'button:has-text("Mostrar más")',
                    'button:has-text("Show more")',
                    'a:has-text("Mostrar más")',
                    'a:has-text("Show more")',
                    '.show-more',
                    '[data-testid*="show-more"]',
                    'button[class*="show-more"]'
                ]
                
                button_found = False
                for selector in show_more_selectors:
                    try:
                        show_more_button = section_locator.locator(selector).first
                        if await show_more_button.count() > 0 and await show_more_button.is_visible():
                            await show_more_button.click()
                            await self.page.wait_for_timeout(3000)  # Esperar que carguen más partidos
                            clicks_realizados += 1
                            logger.info(f"      ✅ Click {clicks_realizados} en 'Mostrar más' realizado")
                            button_found = True
                            break
                    except:
                        continue
                
                if not button_found:
                    logger.info(f"      ℹ️ No se encontró más botones 'Mostrar más' en {section_name}")
                    break
                    
            except Exception as e:
                logger.warning(f"      ⚠️ Error en click {i+1}: {e}")
                break
        
        logger.info(f"   ✅ Sección {section_name} expandida con {clicks_realizados} clicks")
        return clicks_realizados
    
    def enrich_match_data_with_rankings(self, match_data, player_name):
        """🏆 NUEVA FUNCIONALIDAD: Enriquecer datos de partido con rankings"""
        for match in match_data:
            if isinstance(match, dict):
                # Obtener ranking del oponente
                opponent = match.get('oponente', '')
                if opponent and opponent != 'N/A':
                    opponent_rank = self.ranking_manager.get_player_ranking(opponent)
                    opponent_info = self.ranking_manager.get_player_info(opponent)
                    
                    match['opponent_ranking'] = opponent_rank
                    if opponent_info:
                        match['opponent_points'] = opponent_info.get('points', 0)
                        match['opponent_nationality'] = opponent_info.get('nationality', 'N/A')
                
                # Calcular peso del oponente
                match['opponent_weight'] = self.rivalry_analyzer.calculate_base_opponent_weight(match.get('opponent_ranking'))
        
        return match_data
    
    def determine_winner_from_result(self, resultado, jugador1, jugador2):
        """🏆 Determinar ganador basado en el resultado (ej: '2-0', '1-2')"""
        if not resultado or resultado == 'N/A':
            return 'N/A'
        
        try:
            # Buscar patrón de sets (ej: "2-0", "1-2", "3-1")
            parts = resultado.split('-')
            if len(parts) == 2:
                sets1 = int(parts[0])
                sets2 = int(parts[1])
                
                if sets1 > sets2:
                    return jugador1
                elif sets2 > sets1:
                    return jugador2
        except:
            pass
        
        return 'N/A'

    def extract_winner_sets(self, resultado):
        """🔢 Extraer los sets del ganador del resultado (ej: '2-0' -> 2)"""
        if not resultado or resultado == 'N/A':
            return 'N/A'
        try:
            parts = resultado.split('-')
            if len(parts) == 2:
                sets1 = int(parts[0])
                sets2 = int(parts[1])
                return max(sets1, sets2)
        except:
            pass
        return 'N/A'

    def analyze_common_opponents_in_extractor(self, player1_history, player2_history, player1_name, player2_name):
        """🤝 Análizar oponentes comunes durante la extracción"""
        
        # Crear mapas de oponentes
        p1_opponents = {}
        p2_opponents = {}
        
        for match in player1_history:
            opponent = match.get('oponente', '')
            if opponent and opponent != 'N/A':
                normalized = self.ranking_manager.normalize_name(opponent)
                if normalized not in p1_opponents:
                    p1_opponents[normalized] = []
                p1_opponents[normalized].append(match)
        
        for match in player2_history:
            opponent = match.get('oponente', '')
            if opponent and opponent != 'N/A':
                normalized = self.ranking_manager.normalize_name(opponent)
                if normalized not in p2_opponents:
                    p2_opponents[normalized] = []
                p2_opponents[normalized].append(match)
        
        # Encontrar y analizar oponentes comunes
        common_opponents_detailed = []
        for opponent in p1_opponents:
            if opponent in p2_opponents:
                p1_matches = p1_opponents[opponent]
                p2_matches = p2_opponents[opponent]
                
                # Tomar el partido más reciente de cada uno
                p1_recent = p1_matches[0]
                p2_recent = p2_matches[0]
                
                # Análisis de resultados
                p1_won = self.rivalry_analyzer.determine_match_winner(p1_recent, player1_name)
                p2_won = self.rivalry_analyzer.determine_match_winner(p2_recent, player2_name)
                
                common_opponent_data = {
                    'opponent_name': p1_recent.get('oponente', ''),
                    'opponent_ranking': self.ranking_manager.get_player_ranking(p1_recent.get('oponente', '')),
                    'player1_result': {
                        'outcome': 'Ganó' if p1_won else 'Perdió',
                        'score': p1_recent.get('resultado', ''),
                        'date': p1_recent.get('fecha', ''),
                        'tournament': p1_recent.get('torneo', '')
                    },
                    'player2_result': {
                        'outcome': 'Ganó' if p2_won else 'Perdió',
                        'score': p2_recent.get('resultado', ''),
                        'date': p2_recent.get('fecha', ''),
                        'tournament': p2_recent.get('torneo', '')
                    },
                    'advantage_for': player1_name if (p1_won and not p2_won) else (player2_name if (p2_won and not p1_won) else 'Ninguno'),
                    'total_encounters_p1': len(p1_matches),
                    'total_encounters_p2': len(p2_matches)
                }
                
                common_opponents_detailed.append(common_opponent_data)
        
        return common_opponents_detailed

    def analyze_recent_form_in_extractor(self, player_history, player_name, recent_count=20):
        """📈 Análizar forma reciente durante la extracción"""
        if not player_history:
            return None
        
        recent = player_history[:recent_count]
        wins = sum(1 for match in recent if match.get('outcome', '').lower() in ['ganó', 'win'])
        total = len(recent)
        
        # Racha actual
        current_streak = 0
        streak_type = None
        
        for match in recent:
            outcome = match.get('outcome', '').lower()
            won = outcome in ['ganó', 'win']
            
            if streak_type is None:
                streak_type = 'win' if won else 'loss'
                current_streak = 1
            elif (streak_type == 'win' and won) or (streak_type == 'loss' and not won):
                current_streak += 1
            else:
                break
        
        return {
            'player_name': player_name,
            'recent_matches_count': total,
            'wins': wins,
            'losses': total - wins,
            'win_percentage': round((wins / total * 100), 1) if total > 0 else 0,
            'current_streak_count': current_streak,
            'current_streak_type': 'victorias' if streak_type == 'win' else 'derrotas',
            'form_status': self.classify_form(wins, total),
            'last_match_date': recent[0].get('fecha', 'N/A') if recent else 'N/A'
        }

    def classify_form(self, wins, total):
        """📊 Clasificar forma del jugador"""
        if total == 0:
            return 'Sin datos'
        
        percentage = wins / total * 100
        if percentage >= 75:
            return 'Excelente'
        elif percentage >= 60:
            return 'Buena'
        elif percentage >= 40:
            return 'Regular'
        else:
            return 'Mala'
    
    async def extract_h2h_sections(self):
        """📊 Extraer las 3 secciones H2H usando selectores específicos con expansión"""
        logger.info("📊 Extrayendo secciones H2H con expansión automática...")
        
        try:
            # Esperar a que cargue el contenido H2H
            await self.page.wait_for_timeout(8000)
            
            # Tomar screenshot del H2H para debugging
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            h2h_screenshot = f"h2h_match_{self.current_match_index + 1}_{timestamp}.png"
            await self.page.screenshot(path=h2h_screenshot)
            logger.info(f"📸 Screenshot H2H: {h2h_screenshot}")
            
            # Extraer el HTML completo para análisis
            page_content = await self.page.content()
            
            # Guardar HTML para análisis offline
            html_file = f"h2h_match_{self.current_match_index + 1}_{timestamp}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(page_content)
            logger.info(f"💾 Contenido HTML guardado: {html_file}")
            
            # Localizar las secciones H2H
            h2h_sections = await self.page.locator('.h2h__section').all()
            logger.info(f"📊 Encontradas {len(h2h_sections)} secciones H2H")
            
            # SECCIÓN 1: Historial Jugador 1 (con expansión)
            logger.info("📊 Extrayendo historial Jugador 1 con expansión...")
            if len(h2h_sections) >= 1:
                await self.click_show_more_buttons(h2h_sections[0], "Jugador 1", max_clicks=8)
            player1_data = await self.extract_player_section(0)
            
            # SECCIÓN 2: Historial Jugador 2 (con expansión)
            logger.info("📊 Extrayendo historial Jugador 2 con expansión...")
            if len(h2h_sections) >= 2:
                await self.click_show_more_buttons(h2h_sections[1], "Jugador 2", max_clicks=8)
            player2_data = await self.extract_player_section(1)
            
            # SECCIÓN 3: Enfrentamientos Directos
            logger.info("📊 Extrayendo enfrentamientos directos...")
            h2h_data = await self.extract_direct_confrontations()
            
            return {
                'player1_history': player1_data,
                'player2_history': player2_data,
                'head_to_head_matches': h2h_data
            }
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo secciones H2H: {e}")
            return {
                'player1_history': [],
                'player2_history': [],
                'head_to_head_matches': []
            }
    
    async def extract_player_section(self, player_section_index: int):
        """👤 Extraer historial de un jugador específico usando selectores detallados.
        player_section_index: 0 para jugador 1, 1 para jugador 2.
        """
        logger.info(f"   👤 Extrayendo datos para sección de jugador {player_section_index + 1}...")
        
        matches_data = []
        try:
            h2h_sections = await self.page.locator('.h2h__section').all()
            
            if len(h2h_sections) < 2:
                logger.error("❌ No se encontraron las dos secciones de historial de jugadores.")
                return []

            player_section = h2h_sections[player_section_index]
            
            match_rows = await player_section.locator('.h2h__row').all()
            logger.info(f"      🎯 Encontrados {len(match_rows)} partidos para el jugador {player_section_index + 1}")

            for i, row in enumerate(match_rows):
                try:
                    date = await row.locator('.h2h__date').inner_text()
                    result = await row.locator('.h2h__result').inner_text()
                    
                    # Identificar al oponente (el que no está 'highlighted')
                    participants = await row.locator('.h2h__participant').all()
                    opponent_name = "N/A"
                    for p in participants:
                        class_attr = await p.get_attribute('class') or ''
                        if 'highlighted' not in class_attr:
                            opponent_element = p.locator('.h2h__participantInner')
                            if await opponent_element.count() > 0:
                                opponent_name = await opponent_element.inner_text()
                            break
                    
                    outcome_element = row.locator('.h2h__icon > div')
                    outcome = "N/A"
                    if await outcome_element.count() > 0:
                        outcome_class = await outcome_element.get_attribute('class') or ''
                        if 'win' in outcome_class.lower():
                            outcome = "Ganó"
                        elif 'loss' in outcome_class.lower():
                            outcome = "Perdió"
                        elif 'draw' in outcome_class.lower():
                            outcome = "Empate"
                    
                    if outcome == "N/A":
                        outcome = "Perdió"

                    # NUEVA LÓGICA MEJORADA para extraer torneo, ciudad, país y superficie
                    tournament_name = "N/A"
                    city = "N/A"
                    country = "N/A"
                    surface = "N/A"

                    # El elemento principal que contiene la información del evento
                    event_element = row.locator('.h2h__event').first
                    
                    if await event_element.count() > 0:
                        # Extraer el nombre corto del torneo del texto del elemento
                        try:
                            tournament_name = await event_element.inner_text()
                        except Exception:
                            tournament_name = "N/A"
                        
                        # Extraer el título que contiene ciudad, país y superficie
                        title_attr = await event_element.get_attribute('title')
                        if title_attr:
                            # Ejemplo: "Toronto (Canadá), dura"
                            parts = title_attr.split(',')
                            if len(parts) > 0:
                                location_part = parts[0].strip()
                                # Extraer ciudad y país
                                match = re.search(r'^(.*)\s\((.*)\)$', location_part)
                                if match:
                                    city = match.group(1).strip()
                                    country = match.group(2).strip()
                                else:
                                    city = location_part  # Si no hay país
                            
                            if len(parts) > 1:
                                surface_from_title = parts[-1].strip()
                                if surface_from_title:
                                    surface = surface_from_title

                        # Extraer superficie también de la clase como fuente principal
                        class_attr = await event_element.get_attribute('class') or ''
                        if 'hard' in class_attr or 'dura' in class_attr:
                            surface = 'Dura'
                        elif 'clay' in class_attr or 'arcilla' in class_attr:
                            surface = 'Arcilla'
                        elif 'grass' in class_attr or 'hierba' in class_attr:
                            surface = 'Hierba'
                        elif 'indoor' in class_attr:
                            surface = 'Indoor'

                    # Fallback a la lógica anterior si la nueva falla
                    if tournament_name == "N/A" or tournament_name == "":
                        tournament_element = row.locator('.h2h__tournament')
                        if await tournament_element.count() > 0:
                            tournament_name = await tournament_element.inner_text()

                    if surface == "N/A" or surface == "":
                        surface_element = row.locator('.h2h__surface')
                        if await surface_element.count() > 0:
                            surface = await surface_element.inner_text()

                    match_info = {
                        'fecha': date.strip(),
                        'oponente': opponent_name.strip(),
                        'resultado': result.strip().replace('\n', '-'),
                        'outcome': outcome,
                        'torneo': tournament_name.strip().replace('\n', ' '),
                        'ciudad': city.strip(),
                        'pais': country.strip(),
                        'superficie': surface.strip().capitalize()
                    }
                    
                    matches_data.append(match_info)
                    
                    if i < 3:  # Solo mostrar los primeros 3 para no saturar el log
                        logger.info(f"         {i+1}. {match_info['fecha']} vs {match_info['oponente']} -> {match_info['outcome']}")
                
                except Exception as e:
                    logger.warning(f"      ⚠️ Error extrayendo partido {i+1}: {e}")
                    continue
            
            logger.info(f"   ✅ Extraídos {len(matches_data)} partidos para jugador {player_section_index + 1}")
            return matches_data
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo sección jugador {player_section_index + 1}: {e}")
            return []
    
    async def extract_direct_confrontations(self):
        """🥊 Extraer enfrentamientos directos con ganador correcto"""
        logger.info("   🥊 Extrayendo enfrentamientos directos...")
        
        confrontations_data = []
        try:
            # Buscar la sección de enfrentamientos directos
            h2h_sections = await self.page.locator('.h2h__section').all()
            
            if len(h2h_sections) >= 3:
                direct_section = h2h_sections[2]  # Tercera sección
            else:
                # Buscar por otros selectores posibles
                direct_section = self.page.locator('.h2h__head-to-head, .direct-matches, .confrontations').first
                if await direct_section.count() == 0:
                    logger.warning("⚠️ No se encontró sección de enfrentamientos directos")
                    return []
            
            # Expandir la sección de enfrentamientos directos (máximo 2 clicks)
            await self.click_show_more_buttons(direct_section, "Enfrentamientos Directos", max_clicks=2)
            
            match_rows = await direct_section.locator('.h2h__row').all()
            logger.info(f"      🎯 Encontrados {len(match_rows)} enfrentamientos directos")
            
            for i, row in enumerate(match_rows):
                try:
                    date = await row.locator('.h2h__date').inner_text()
                    result = await row.locator('.h2h__result').inner_text()
                    
                    # Extraer nombres de ambos jugadores
                    participants = await row.locator('.h2h__participant').all()
                    player1_name = "N/A"
                    player2_name = "N/A"
                    
                    if len(participants) >= 2:
                        player1_element = participants[0].locator('.h2h__participantInner')
                        player2_element = participants[1].locator('.h2h__participantInner')
                        
                        if await player1_element.count() > 0:
                            player1_name = await player1_element.inner_text()
                        if await player2_element.count() > 0:
                            player2_name = await player2_element.inner_text()

                    # NUEVA LÓGICA MEJORADA para extraer torneo, ciudad, país y superficie
                    tournament_name = "N/A"
                    city = "N/A"
                    country = "N/A"
                    surface = "N/A"

                    event_element = row.locator('.h2h__event').first
                    if await event_element.count() > 0:
                        try:
                            tournament_name = await event_element.inner_text()
                        except Exception:
                            tournament_name = "N/A"
                        
                        title_attr = await event_element.get_attribute('title')
                        if title_attr:
                            parts = title_attr.split(',')
                            if len(parts) > 0:
                                location_part = parts[0].strip()
                                match = re.search(r'^(.*)\s\((.*)\)$', location_part)
                                if match:
                                    city = match.group(1).strip()
                                    country = match.group(2).strip()
                                else:
                                    city = location_part
                            
                            if len(parts) > 1:
                                surface_from_title = parts[-1].strip()
                                if surface_from_title:
                                    surface = surface_from_title

                        class_attr = await event_element.get_attribute('class') or ''
                        if 'hard' in class_attr or 'dura' in class_attr:
                            surface = 'Dura'
                        elif 'clay' in class_attr or 'arcilla' in class_attr:
                            surface = 'Arcilla'
                        elif 'grass' in class_attr or 'hierba' in class_attr:
                            surface = 'Hierba'
                        elif 'indoor' in class_attr:
                            surface = 'Indoor'

                    if tournament_name == "N/A" or tournament_name == "":
                        tournament_element = row.locator('.h2h__tournament')
                        if await tournament_element.count() > 0:
                            tournament_name = await tournament_element.inner_text()

                    if surface == "N/A" or surface == "":
                        surface_element = row.locator('.h2h__surface')
                        if await surface_element.count() > 0:
                            surface = await surface_element.inner_text()

                    # LÓGICA DE GANADOR
                    winner = self.determine_winner_from_result(
                        result.strip().replace('\n', '-'),
                        player1_name.strip(),
                        player2_name.strip()
                    )
                    if winner == 'N/A':
                        for j, p in enumerate(participants):
                            class_attr = await p.get_attribute('class') or ''
                            if 'highlighted' in class_attr or 'winner' in class_attr:
                                winner = player1_name if j == 0 else player2_name
                                break
                    
                    confrontation_info = {
                        'fecha': date.strip(),
                        'jugador1': player1_name.strip(),
                        'jugador2': player2_name.strip(),
                        'resultado': result.strip().replace('\n', '-'),
                        'ganador': winner,
                        'torneo': tournament_name.strip().replace('\n', ' '),
                        'ciudad': city.strip(),
                        'pais': country.strip(),
                        'superficie': surface.strip().capitalize(),
                        'ganador_sets': self.extract_winner_sets(result.strip())
                    }
                    
                    confrontations_data.append(confrontation_info)
                    
                    if i < 3:  # Solo mostrar los primeros 3 para no saturar el log
                        logger.info(f"         {i+1}. {confrontation_info['fecha']} - {confrontation_info['ganador']} ganó")
                
                except Exception as e:
                    logger.warning(f"      ⚠️ Error extrayendo enfrentamiento {i+1}: {e}")
                    continue
            
            logger.info(f"   ✅ Extraídos {len(confrontations_data)} enfrentamientos directos")
            return confrontations_data
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo enfrentamientos directos: {e}")
            return []

    async def extract_current_match_info(self):
        """ℹ️ Extraer información del partido actual (torneo, cancha, cuotas) desde la página."""
        logger.info("ℹ️ Extrayendo información del partido actual desde la página...")
        info = {
            "torneo_nombre": None,
            "tipo_cancha": None,
            "torneo_completo": None,
            "cuota1": None,
            "cuota2": None
        }
        try:
            # Extraer información del torneo y cancha
            tournament_element = self.page.locator('.tournamentHeader__country a').first
            if await tournament_element.count() > 0:
                full_tournament_text = await tournament_element.inner_text()
                
                # Comprobación de seguridad para evitar datos corruptos
                if len(full_tournament_text) > 150:
                    logger.warning("   ⚠️ El texto del torneo extraído es demasiado largo, se descarta.")
                else:
                    info["torneo_completo"] = full_tournament_text.strip()
                    
                    parts = full_tournament_text.split(',')
                    torneo_nombre = parts[0]
                    
                    if len(parts) > 1:
                        info["tipo_cancha"] = parts[-1].strip()
                    
                    if ':' in torneo_nombre:
                        torneo_nombre = torneo_nombre.split(':', 1)[1].strip()
                    
                    info["torneo_nombre"] = torneo_nombre.strip()
                    logger.info(f"   ✅ Torneo: {info['torneo_nombre']}, Cancha: {info['tipo_cancha']}")
            else:
                logger.warning("   ⚠️ No se encontró el elemento de cabecera del torneo.")

            # Extraer cuotas
            odds_elements = await self.page.locator('.participant__odd').all()
            if len(odds_elements) >= 2:
                info["cuota1"] = await odds_elements[0].inner_text()
                info["cuota2"] = await odds_elements[1].inner_text()
                logger.info(f"   ✅ Cuotas extraídas: {info['cuota1']} - {info['cuota2']}")
            else:
                logger.warning("   ⚠️ No se encontraron las cuotas del partido.")

        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo información del partido actual: {e}")
        
        return info

    async def process_single_match(self, match_data, optimized_weights=None):
        """⚙️ Procesar un único partido: navegación, extracción y análisis"""
        self.current_match_index += 1
        match_number = self.current_match_index
        
        player1 = match_data.get('jugador1', 'N/A')
        player2 = match_data.get('jugador2', 'N/A')
        
        logger.info("=" * 60)
        logger.info(f"⚙️ PROCESANDO PARTIDO {match_number}/{self.total_matches_to_process}: {player1} vs {player2}")
        
        try:
            # 1. Navegar a la URL del partido
            if not await self.navigate_to_match_url(match_data):
                return False

            # 1.5. Extraer información actualizada del partido desde la página
            current_match_info = await self.extract_current_match_info()
            
            # 2. Navegar a la sección H2H
            if not await self.navigate_to_h2h_section():
                return False
            
            # 3. Extraer las 3 secciones H2H
            h2h_data = await self.extract_h2h_sections()
            
            # 4. NUEVA FUNCIONALIDAD: Enriquecer datos con rankings
            player1_history_enriched = self.enrich_match_data_with_rankings(h2h_data['player1_history'], player1)
            player2_history_enriched = self.enrich_match_data_with_rankings(h2h_data['player2_history'], player2)
            
            # 5. ANÁLISIS DE FORMA (movido antes para alimentar la predicción)
            player1_form = self.analyze_recent_form_in_extractor(player1_history_enriched, player1)
            player2_form = self.analyze_recent_form_in_extractor(player2_history_enriched, player2)

            # NUEVO: Calcular ELO a partir del historial extraído
            p1_calculated_elo = self.rivalry_analyzer.calculate_elo_from_history(player1, player1_history_enriched)
            p2_calculated_elo = self.rivalry_analyzer.calculate_elo_from_history(player2, player2_history_enriched)
            logger.info(f"   ⚡ ELO Calculado: {player1} ({p1_calculated_elo}), {player2} ({p2_calculated_elo})")

            # 6. ANÁLISIS DE RIVALIDAD TRANSITIVA (ahora con datos de forma)
            # Extraer contexto del partido actual (país y superficie)
            # Esta información se obtiene del primer partido del historial directo si existe, o se infiere.
            current_match_context = {
                'country': match_data.get('pais', 'N/A'),
                'surface': match_data.get('tipo_cancha', 'N/A')
            }

            rivalry_analysis = self.rivalry_analyzer.analyze_rivalry(
                player1_history_enriched,
                player2_history_enriched,
                player1,
                player2,
                player1_form,
                player2_form,
                h2h_data['head_to_head_matches'],
                current_match_context,
                p1_calculated_elo,
                p2_calculated_elo,
                tournament_name=match_data.get('torneo_completo', ''),
                optimized_weights=optimized_weights
            )

            # 7. ANÁLISIS DE OPONENTES COMUNES (para el reporte final)
            common_opponents_analysis = self.analyze_common_opponents_in_extractor(
                player1_history_enriched, player2_history_enriched, player1, player2
            )
            
            # 8. Consolidar resultados
            player1_key = player1.replace(' ', '_').replace('.', '')
            player2_key = player2.replace(' ', '_').replace('.', '')

            # 8. NUEVO: Obtener métricas de ranking para el reporte
            p1_ranking_metrics = self.rivalry_analyzer.get_ranking_metrics(player1)
            p2_ranking_metrics = self.rivalry_analyzer.get_ranking_metrics(player2)
            
            result = {
                'match_number': match_number,
                'jugador1': player1,
                'jugador1_nacionalidad': rivalry_analysis.get('player1_nationality', 'N/A'),
                'jugador2': player2,
                'jugador2_nacionalidad': rivalry_analysis.get('player2_nationality', 'N/A'),
                'torneo_nombre': current_match_info.get('torneo_nombre') or match_data.get('torneo_nombre', 'N/A'),
                'tipo_cancha': current_match_info.get('tipo_cancha') or match_data.get('tipo_cancha', 'N/A'),
                'torneo_completo': current_match_info.get('torneo_completo') or match_data.get('torneo_completo', 'N/A'),
                'cuota1': current_match_info.get('cuota1') or match_data.get('cuota1', 'N/A'),
                'cuota2': current_match_info.get('cuota2') or match_data.get('cuota2', 'N/A'),
                'match_url': match_data.get('match_url', ''),
                f'historial_{player1_key}': player1_history_enriched,
                f'historial_{player2_key}': player2_history_enriched,
                'enfrentamientos_directos': h2h_data['head_to_head_matches'],
                'estadisticas': {
                    f'partidos_{player1_key}': len(player1_history_enriched),
                    f'partidos_{player2_key}': len(player2_history_enriched),
                    'enfrentamientos_totales': len(h2h_data['head_to_head_matches'])
                },
                'ranking_analysis': {
                    f'{player1_key}_ranking': rivalry_analysis['player1_rank'],
                    f'{player2_key}_ranking': rivalry_analysis['player2_rank'],
                    'common_opponents_count': rivalry_analysis['common_opponents_count'],
                    'p1_rivalry_score': rivalry_analysis['p1_rivalry_score'],
                    'p2_rivalry_score': rivalry_analysis['p2_rivalry_score'],
                    'prediction': rivalry_analysis['prediction'],
                    f'{player1_key}_metrics': p1_ranking_metrics,
                    f'{player2_key}_metrics': p2_ranking_metrics,
                    f'{player1_key}_elo': p1_calculated_elo,
                    f'{player2_key}_elo': p2_calculated_elo
                },
                'form_analysis': {
                    f'{player1_key}_form': player1_form,
                    f'{player2_key}_form': player2_form,
                },
                'surface_analysis': {
                    f'{player1_key}_surface_stats': rivalry_analysis.get('p1_surface_stats'),
                    f'{player2_key}_surface_stats': rivalry_analysis.get('p2_surface_stats'),
                },
                'location_analysis': {
                    f'{player1_key}_location_stats': rivalry_analysis.get('p1_location_stats'),
                    f'{player2_key}_location_stats': rivalry_analysis.get('p2_location_stats'),
                },
                'common_opponents_detailed': common_opponents_analysis
            }
            
            self.all_results.append(result)
            
            logger.info(f"✅ Partido {match_number} procesado exitosamente")
            logger.info(f"   📊 Historial {player1}: {len(player1_history_enriched)} partidos")
            logger.info(f"   📊 Historial {player2}: {len(player2_history_enriched)} partidos")
            logger.info(f"   🥊 Enfrentamientos directos: {len(h2h_data['head_to_head_matches'])}")
            logger.info(f"   🏆 Ranking {player1}: {rivalry_analysis['player1_rank']}")
            logger.info(f"   🏆 Ranking {player2}: {rivalry_analysis['player2_rank']}")
            logger.info(f"   ⚔️ Oponentes comunes: {rivalry_analysis['common_opponents_count']}")
            logger.info(f"   🎯 Predicción: {rivalry_analysis['prediction']['favored_player']} ({rivalry_analysis['prediction']['confidence']}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error procesando partido {match_number}: {e}")
            return False
    
    async def process_all_matches(self, optimized_weights=None):
        """🔄 Procesar todos los partidos en la cola de forma secuencial"""
        if optimized_weights:
            logger.info(f"🚀 RE-PROCESANDO {len(self.matches_queue)} PARTIDOS CON PESOS OPTIMIZADOS")
            self.all_results = [] # Limpiar resultados para el re-procesamiento
            self.current_match_index = 0
        else:
            logger.info(f"🚀 INICIANDO PROCESAMIENTO SECUENCIAL DE {len(self.matches_queue)} PARTIDOS")
        logger.info("=" * 80)
        
        successful_matches = 0
        failed_matches = 0
        
        for match_data in self.matches_queue:
            try:
                if await self.process_single_match(match_data, optimized_weights=optimized_weights):
                    successful_matches += 1
                else:
                    failed_matches += 1
                
                # Pausa entre partidos para evitar sobrecargar el servidor
                if self.matches_queue.index(match_data) < len(self.matches_queue) - 1:
                    logger.info("⏳ Pausa de 8 segundos antes del siguiente partido...")
                    await self.page.wait_for_timeout(8000)
                
            except Exception as e:
                logger.error(f"❌ Error crítico procesando partido: {e}")
                failed_matches += 1
                continue
        
        logger.info("=" * 80)
        logger.info("🏁 PROCESAMIENTO COMPLETADO")
        logger.info(f"   ✅ Partidos exitosos: {successful_matches}")
        logger.info(f"   ❌ Partidos fallidos: {failed_matches}")
        
        total_processed = successful_matches + failed_matches
        if total_processed > 0:
            success_rate = (successful_matches / total_processed) * 100
            logger.info(f"   📊 Tasa de éxito: {success_rate:.1f}%")
        else:
            logger.info("   📊 Tasa de éxito: 0.0% (No se procesaron partidos)")
    
    def save_results_to_json(self, is_optimized=False):
        """💾 Guardar todos los resultados en un archivo JSON consolidado"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)
        
        if is_optimized:
            filename = reports_dir / f"h2h_results_OPTIMIZED_{timestamp}.json"
        else:
            filename = reports_dir / f"h2h_results_enhanced_{timestamp}.json"
        
        try:
            # Preparar datos para guardar
            output_data = {
                'metadata': {
                    'fecha_extraccion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'total_partidos_procesados': len(self.all_results),
                    'version': '3.0_con_rivalidad_transitiva',
                    'funcionalidades': [
                        'Extracción H2H básica',
                        'Análisis de rankings',
                        'Rivalidad transitiva',
                        'Predicciones avanzadas'
                    ]
                },
                'partidos': self.all_results,
                'estadisticas_globales': self.generate_global_statistics()
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Resultados guardados en: {filename}")
            logger.info(f"   📊 Total partidos: {len(self.all_results)}")
            
            # Generar resumen estadístico
            self.print_results_summary()
            
            return filename
            
        except Exception as e:
            logger.error(f"❌ Error guardando resultados: {e}")
            return None
    
    def generate_global_statistics(self):
        """📈 Generar estadísticas globales de todos los partidos procesados"""
        if not self.all_results:
            return {}
        
        stats = {
            'total_partidos': len(self.all_results),
            'jugadores_unicos': set(),
            'total_historiales_extraidos': 0,
            'total_enfrentamientos_directos': 0,
            'rankings_encontrados': 0,
            'predicciones_generadas': 0,
            'oponentes_comunes_total': 0,
            'analisis_rivalidad_exitosos': 0
        }
        
        for match_result in self.all_results:
            # Contar jugadores únicos
            stats['jugadores_unicos'].add(match_result.get('jugador1', ''))
            stats['jugadores_unicos'].add(match_result.get('jugador2', ''))
            
            # Contar historiales
            player1_key = match_result['jugador1'].replace(' ', '_').replace('.', '')
            player2_key = match_result['jugador2'].replace(' ', '_').replace('.', '')
            
            stats['total_historiales_extraidos'] += len(match_result.get(f'historial_{player1_key}', []))
            stats['total_historiales_extraidos'] += len(match_result.get(f'historial_{player2_key}', []))
            
            # Contar enfrentamientos directos
            stats['total_enfrentamientos_directos'] += len(match_result.get('enfrentamientos_directos', []))
            
            # Contar análisis de ranking
            ranking_analysis = match_result.get('ranking_analysis', {})
            if ranking_analysis.get(f'{player1_key}_ranking') is not None:
                stats['rankings_encontrados'] += 1
            if ranking_analysis.get(f'{player2_key}_ranking') is not None:
                stats['rankings_encontrados'] += 1
            
            if ranking_analysis.get('prediction'):
                stats['predicciones_generadas'] += 1
            
            if ranking_analysis.get('common_opponents_count', 0) > 0:
                stats['oponentes_comunes_total'] += ranking_analysis['common_opponents_count']
                stats['analisis_rivalidad_exitosos'] += 1
        
        # Convertir set a count
        stats['jugadores_unicos'] = len(stats['jugadores_unicos'])
        
        return stats
    
    def print_results_summary(self):
        """📋 Imprimir resumen detallado de los resultados"""
        logger.info("📋 RESUMEN DETALLADO DE RESULTADOS")
        logger.info("=" * 60)
        
        if not self.all_results:
            logger.info("❌ No hay resultados para mostrar")
            return
        
        for i, match_result in enumerate(self.all_results, 1):
            player1 = match_result.get('jugador1', 'N/A')
            player2 = match_result.get('jugador2', 'N/A')
            cuota1 = match_result.get('cuota1', 'N/A')
            cuota2 = match_result.get('cuota2', 'N/A')
            
            ranking_analysis = match_result.get('ranking_analysis', {})
            player1_key = player1.replace(' ', '_').replace('.', '')
            player2_key = player2.replace(' ', '_').replace('.', '')
            
            rank1 = ranking_analysis.get(f'{player1_key}_ranking', 'N/A')
            rank2 = ranking_analysis.get(f'{player2_key}_ranking', 'N/A')
            
            p1_metrics = ranking_analysis.get(f'{player1_key}_metrics', {})
            p2_metrics = ranking_analysis.get(f'{player2_key}_metrics', {})

            player1_info = self.ranking_manager.get_player_info(player1)
            player2_info = self.ranking_manager.get_player_info(player2)
            p1_nationality = player1_info.get('nationality', 'N/A') if player1_info else 'N/A'
            p2_nationality = player2_info.get('nationality', 'N/A') if player2_info else 'N/A'

            surface_analysis = match_result.get('surface_analysis', {})
            p1_surface_stats = surface_analysis.get(f'{player1_key}_surface_stats', {})
            p2_surface_stats = surface_analysis.get(f'{player2_key}_surface_stats', {})
            
            prediction = ranking_analysis.get('prediction', {})
            favored = prediction.get('favored_player', 'N/A')
            confidence = prediction.get('confidence', 0)
            scores = prediction.get('scores', {})
            score_breakdown = prediction.get('score_breakdown', {})
            weights = prediction.get('weights_used', {})

            logger.info(f"🎾 PARTIDO {i}: {player1} (rank {rank1}) vs {player2} (rank {rank2})")
            logger.info(f"   💰 Cuotas: {cuota1} - {cuota2}")

            logger.info(f"   📈 Métricas P1: PTS: {p1_metrics.get('pts', 'N/A')}, "
                        f"PROX: {p1_metrics.get('prox_pts', 'N/A')}, "
                        f"MAX: {p1_metrics.get('pts_max', 'N/A')}, "
                        f"Potencial: {p1_metrics.get('improvement_potential', 'N/A')}")
            logger.info(f"   📈 Métricas P2: PTS: {p2_metrics.get('pts', 'N/A')}, "
                        f"PROX: {p2_metrics.get('prox_pts', 'N/A')}, "
                        f"MAX: {p2_metrics.get('pts_max', 'N/A')}, "
                        f"Potencial: {p2_metrics.get('improvement_potential', 'N/A')}")
            
            direct_matches = match_result.get('enfrentamientos_directos', [])
            logger.info(f"   🥊 H2H ({len(direct_matches)} encontrados):")
            if not direct_matches:
                logger.info("      No se encontraron enfrentamientos directos.")
            else:
                for h2h_match in direct_matches[:3]:
                    logger.info(f"      - {h2h_match['fecha']}: {h2h_match['ganador']} ganó {h2h_match['resultado']}")
            
            # NUEVO: Resumen de Superficie y Ubicación
            current_surface = match_result.get('tipo_cancha', 'N/A')
            logger.info(f"   🏞️ Análisis de Superficie ({current_surface}):")
            if p1_surface_stats and p2_surface_stats and current_surface in p1_surface_stats:
                p1_surf = p1_surface_stats[current_surface]
                p2_surf = p2_surface_stats[current_surface]
                logger.info(f"      - {player1}: {p1_surf.get('win_rate', 0)}% ({p1_surf.get('wins',0)}W-{p1_surf.get('losses',0)}L)")
                logger.info(f"      - {player2}: {p2_surf.get('win_rate', 0)}% ({p2_surf.get('wins',0)}W-{p2_surf.get('losses',0)}L)")
            
            current_country = match_result.get('torneo_nombre', 'N/A').split(',')[-1].strip() # Inferencia simple
            logger.info(f"   🌍 Ventaja Local (Jugando en {current_country}):")
            if p1_nationality == current_country:
                 logger.info(f"      - {player1} juega en casa! ({p1_nationality})")
            if p2_nationality == current_country:
                 logger.info(f"      - {player2} juega en casa! ({p2_nationality})")

            common_opponents_detailed = match_result.get('common_opponents_detailed', [])
            logger.info(f"   ⚔️ Oponentes Comunes ({len(common_opponents_detailed)} encontrados):")
            if not common_opponents_detailed:
                logger.info("      No se encontraron oponentes comunes.")
            else:
                for common_data in common_opponents_detailed[:3]:
                    opponent_rank_str = f"(rank {common_data['opponent_ranking']})" if common_data['opponent_ranking'] else ''
                    logger.info(f"      - vs {common_data['opponent_name']} {opponent_rank_str} -> Ventaja para: {common_data['advantage_for']}")

            logger.info(f"   🎯 Predicción: {favored} ({confidence}%)")
            logger.info(f"      ⚖️ Peso Final: {player1} ({scores.get('p1_final_weight', 0):.2f}) vs {player2} ({scores.get('p2_final_weight', 0):.2f})")
            
            # Mostrar logs de predicción
            if prediction.get('reasoning'):
                logger.info("      --- Logs de Predicción ---")
                for log_line in prediction['reasoning']:
                    if 'LOG_' in log_line:
                        logger.info(f"         - {log_line}")
            
            # MEJORA: Desglose de Puntuación más claro
            if score_breakdown and weights:
                p1_breakdown = score_breakdown.get('player1', {})
                p2_breakdown = score_breakdown.get('player2', {})
                
                logger.info("      --- Desglose de Puntuación ---")
                header = f"      {'Componente':<25} | {'P1 Pts':<10} | {'P2 Pts':<10} | {'P1 %':<7} | {'P2 %':<7}"
                logger.info(header)
                logger.info("      " + "-"*70)
                
                # Orden de importancia para el desglose
                component_order = [
                    'surface_specialization', 'form_recent', 'common_opponents', 
                    'h2h_direct', 'ranking_momentum', 'elo_rating', 'home_advantage', 'strength_of_schedule'
                ]
                
                # Lógica para ELO por defecto
                default_elo = self.elo_system.default_rating
                p1_elo = ranking_analysis.get(f'{player1_key}_elo', default_elo)
                p2_elo = ranking_analysis.get(f'{player2_key}_elo', default_elo)

                for comp in component_order:
                    if weights.get(comp, 0) == 0: continue # No mostrar si el peso es 0
                    
                    # Manejo especial para ELO por defecto
                    if comp == 'elo_rating' and p1_elo == default_elo and p2_elo == default_elo:
                        logger.info(f"      {'Rating ELO'.title():<22} | {'(Default)':<10} | {'(Default)':<10} | {'N/A':<7} | {'N/A':<7}")
                        continue

                    p1_comp_info = p1_breakdown.get(comp, {})
                    p2_comp_info = p2_breakdown.get(comp, {})
                    
                    p1_w_score = p1_comp_info.get('weighted_score', '0.0')
                    p2_w_score = p2_comp_info.get('weighted_score', '0.0')
                    p1_contrib = p1_comp_info.get('contribution', '0.0%')
                    p2_contrib = p2_comp_info.get('contribution', '0.0%')
                    
                    comp_name = comp.replace('_', ' ').title()
                    
                    line = f"      {comp_name:<22} | {p1_w_score:<10} | {p2_w_score:<10} | {p1_contrib:<7} | {p2_contrib:<7}"
                    logger.info(line)
                
                logger.info("      " + "-"*67)
                p1_penalty = p1_breakdown.get('Penalizacion_Inactividad', '0.0 pts')
                p2_penalty = p2_breakdown.get('Penalizacion_Inactividad', '0.0 pts')
                logger.info(f"      {'Penalización':<22} | {p1_penalty:<10} | {p2_penalty:<10} |")
                
                p1_final = p1_breakdown.get('Puntaje_Final', '0.0')
                p2_final = p2_breakdown.get('Puntaje_Final', '0.0')
                logger.info(f"      {'TOTAL':<22} | {p1_final:<10} | {p2_final:<10} |")

            logger.info("-" * 60)

    async def recalculate_and_save_optimized_results(self, initial_file, optimized_weights):
        """🧠 Recalcula las predicciones con pesos optimizados y guarda los resultados."""
        try:
            with open(initial_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            recalculated_matches = []
            for match in data.get('partidos', []):
                player1 = match['jugador1']
                player2 = match['jugador2']
                player1_key = player1.replace(' ', '_').replace('.', '')
                player2_key = player2.replace(' ', '_').replace('.', '')

                # Extraer los datos necesarios que ya fueron scrapeados
                player1_history = match.get(f'historial_{player1_key}', [])
                player2_history = match.get(f'historial_{player2_key}', [])
                player1_form = match.get('form_analysis', {}).get(f'{player1_key}_form')
                player2_form = match.get('form_analysis', {}).get(f'{player2_key}_form')
                direct_h2h = match.get('enfrentamientos_directos', [])

                # Recalcular solo el análisis de rivalidad con los nuevos pesos
                new_rivalry_analysis = self.rivalry_analyzer.analyze_rivalry(
                    player1_history,
                    player2_history,
                    player1,
                    player2,
                    player1_form,
                    player2_form,
                    direct_h2h,
                    optimized_weights=optimized_weights
                )
                
                # Actualizar la sección de análisis del partido
                match['ranking_analysis'] = {
                    **match.get('ranking_analysis', {}),
                    'prediction': new_rivalry_analysis['prediction']
                }
                recalculated_matches.append(match)

            # Reemplazar la lista de partidos con la recalculada
            self.all_results = recalculated_matches
            
            # Guardar en un nuevo archivo optimizado
            return self.save_results_to_json(is_optimized=True)

        except Exception as e:
            logger.error(f"❌ Error recalculando predicciones: {e}")
            return None
    
    async def cleanup(self):
        """🧹 Limpieza de recursos"""
        logger.info("🧹 Iniciando limpieza de recursos...")
        
        try:
            if self.page:
                await self.page.close()
                logger.info("   ✅ Página cerrada")
                
            if self.context:
                await self.context.close()
                logger.info("   ✅ Contexto cerrado")
                
            if self.browser:
                await self.browser.close()
                logger.info("   ✅ Navegador cerrado")
                
            if self.playwright:
                await self.playwright.stop()
                logger.info("   ✅ Playwright detenido")
                
        except Exception as e:
            logger.warning(f"⚠️ Error durante limpieza: {e}")
        
        logger.info("✅ Limpieza completada")

async def main():
    """🎯 Función principal con manejo de errores mejorado"""
    logger.info("🎾 H2H EXTRACTOR SECUENCIAL CON ANÁLISIS DE RIVALIDAD TRANSITIVA")
    logger.info("=" * 80)
    logger.info("🚀 Versión 4.0 - Con Optimización Automática de Pesos")
    logger.info("=" * 80)
    
    extractor = SequentialH2HExtractor()
    
    try:
        # 1. Cargar partidos desde JSON
        if not extractor.load_matches_from_json():
            logger.error("❌ No se pudieron cargar los partidos desde JSON")
            return
        
        # 2. Configurar navegador y procesar partidos con pesos iniciales
        async with extractor.navegador_seguro():
            await extractor.process_all_matches()
        
        # 3. Guardar resultados
        output_file = extractor.save_results_to_json()

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
