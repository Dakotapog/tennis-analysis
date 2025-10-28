import json
import asyncio
import logging
import time
import re
from pathlib import Path
from extraer_ranking_atp_version2 import CompleteRankingScraper

logger = logging.getLogger(__name__)

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

        # Reemplazar guiones por espacios para no juntar nombres
        normalized = normalized.replace('-', ' ')
        
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
