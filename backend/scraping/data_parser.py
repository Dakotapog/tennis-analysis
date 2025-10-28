"""
📊 Data Parser - Parsing y normalización de datos HTML
Extrae y limpia datos de partidos desde HTML de Flashscore
"""

import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DataParser:
    """Parser de datos HTML para partidos de tenis"""
    
    @staticmethod
    def extract_tournament_info(tournament_string):
        """
        Extraer torneo, país y tipo de cancha del string del torneo
        
        Args:
            tournament_string: Ej. "ATP: Toronto (Canada), hard"
        
        Returns:
            dict: {'nombre': str, 'pais': str, 'superficie': str}
        """
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

        # Si el país ya está en el nombre, no hacer nada. Si no, añadirlo.
        if pais != "N/A" and f"({pais})" not in nombre_torneo_final:
            nombre_torneo_final = f"{nombre_torneo_final} ({pais})"
        
        return {
            'nombre': nombre_torneo_final,
            'pais': pais,
            'superficie': tipo_cancha,
            'completo': tournament_string
        }
    
    @staticmethod
    def normalize_surface(surface_text):
        """
        Normalizar nombre de superficie
        
        Args:
            surface_text: Texto de superficie (ej: "hard", "clay", "dura")
        
        Returns:
            str: Superficie normalizada
        """
        if not surface_text:
            return "Desconocida"
        
        surface_map = {
            'hard': 'Dura', 'dura': 'Dura',
            'clay': 'Arcilla', 'arcilla': 'Arcilla',
            'grass': 'Hierba', 'hierba': 'Hierba',
            'indoor': 'Indoor'
        }
        
        surface_lower = surface_text.lower().strip()
        return surface_map.get(surface_lower, surface_text.capitalize())
    
    @staticmethod
    def determine_winner_from_result(resultado, jugador1, jugador2):
        """
        Determinar ganador basado en el resultado
        
        Args:
            resultado: Score del partido (ej: "2-0", "1-2")
            jugador1: Nombre del jugador 1
            jugador2: Nombre del jugador 2
        
        Returns:
            str: Nombre del ganador o 'N/A'
        """
        if not resultado or resultado == 'N/A':
            return 'N/A'
        
        try:
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
    
    @staticmethod
    def extract_winner_sets(resultado):
        """
        Extraer sets del ganador
        
        Args:
            resultado: Score (ej: "2-0")
        
        Returns:
            int or str: Sets ganados o 'N/A'
        """
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
    
    @staticmethod
    def parse_match_date(date_string, date_format='%d.%m.%y'):
        """
        Parsear fecha de partido
        
        Args:
            date_string: Fecha como string
            date_format: Formato esperado
        
        Returns:
            datetime or None
        """
        if not date_string:
            return None
        
        try:
            return datetime.strptime(date_string, date_format)
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parseando fecha '{date_string}': {e}")
            return None
    
    @staticmethod
    def clean_player_name(name):
        """
        Limpiar y normalizar nombre de jugador
        
        Args:
            name: Nombre del jugador
        
        Returns:
            str: Nombre limpio
        """
        if not name:
            return "N/A"
        
        # Quitar espacios extras y caracteres especiales
        cleaned = ' '.join(name.strip().split())
        return cleaned
    
    @staticmethod
    def extract_location_from_title(title_attr):
        """
        Extraer ciudad, país y superficie desde el atributo 'title'
        
        Args:
            title_attr: Ej. "Toronto (Canada), hard"
        
        Returns:
            dict: {'ciudad': str, 'pais': str, 'superficie': str}
        """
        city = "N/A"
        country = "N/A"
        surface = "N/A"
        
        if not title_attr:
            return {'ciudad': city, 'pais': country, 'superficie': surface}
        
        parts = title_attr.split(',')
        
        # Extraer ciudad y país
        if len(parts) > 0:
            location_part = parts[0].strip()
            match = re.search(r'^(.*)\s\((.*)\)$', location_part)
            if match:
                city = match.group(1).strip()
                country = match.group(2).strip()
            else:
                city = location_part
        
        # Extraer superficie
        if len(parts) > 1:
            surface_from_title = parts[-1].strip()
            if surface_from_title:
                surface = DataParser.normalize_surface(surface_from_title)
        
        return {'ciudad': city, 'pais': country, 'superficie': surface}
