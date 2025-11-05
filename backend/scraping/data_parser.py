"""
📊 Data Parser - Parsing y normalización de datos HTML
Extrae y limpia datos de partidos desde HTML de Flashscore
"""

import re
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DataParser:
    """Parser de datos HTML para partidos de tenis"""
    
    @staticmethod
    def extract_tournament_info(tournament_string: str) -> Dict[str, str]:
        """
        Extraer torneo, país y tipo de cancha del string del torneo
        
        Args:
            tournament_string: Ej. "ATP: Toronto (Canada), hard"
        
        Returns:
            dict: {'nombre': str, 'pais': str, 'superficie': str, 'completo': str}
        
        Examples:
            >>> DataParser.extract_tournament_info("ATP: Toronto (Canada), hard")
            {'nombre': 'Toronto (Canada)', 'pais': 'Canada', 'superficie': 'hard', 'completo': '...'}
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
        possible_court_types = ['hard', 'clay', 'grass', 'indoor', 'dura', 'arcilla', 'hierba']
        
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
    def normalize_surface(surface_text: str) -> str:
        """
        Normalizar nombre de superficie
        
        Args:
            surface_text: Texto de superficie (ej: "hard", "clay", "dura")
        
        Returns:
            str: Superficie normalizada
        
        Examples:
            >>> DataParser.normalize_surface("hard")
            'Dura'
            >>> DataParser.normalize_surface("clay")
            'Arcilla'
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
    def determine_winner_from_result(resultado: str, jugador1: str, jugador2: str) -> str:
        """
        Determinar ganador basado en el resultado
        
        Args:
            resultado: Score del partido (ej: "2-0", "1-2")
            jugador1: Nombre del jugador 1
            jugador2: Nombre del jugador 2
        
        Returns:
            str: Nombre del ganador o 'N/A'
        
        Examples:
            >>> DataParser.determine_winner_from_result("2-0", "Nadal", "Federer")
            'Nadal'
            >>> DataParser.determine_winner_from_result("1-2", "Nadal", "Federer")
            'Federer'
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
        except (ValueError, AttributeError) as e:
            logger.warning(f"Error parseando resultado '{resultado}': {e}")
        
        return 'N/A'
    
    @staticmethod
    def extract_winner_sets(resultado: str) -> int:
        """
        Extraer sets del ganador
        
        Args:
            resultado: Score (ej: "2-0")
        
        Returns:
            int or str: Sets ganados o 'N/A'
        
        Examples:
            >>> DataParser.extract_winner_sets("2-0")
            2
            >>> DataParser.extract_winner_sets("1-2")
            2
        """
        if not resultado or resultado == 'N/A':
            return 'N/A'
        
        try:
            parts = resultado.split('-')
            if len(parts) == 2:
                sets1 = int(parts[0])
                sets2 = int(parts[1])
                return max(sets1, sets2)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Error extrayendo sets de '{resultado}': {e}")
        
        return 'N/A'
    
    @staticmethod
    def parse_match_date(date_string: str, date_format: str = '%d.%m.%y') -> Optional[datetime]:
        """
        Parsear fecha de partido
        
        Args:
            date_string: Fecha como string
            date_format: Formato esperado
        
        Returns:
            datetime or None
        
        Examples:
            >>> DataParser.parse_match_date("15.10.23")
            datetime.datetime(2023, 10, 15, 0, 0)
        """
        if not date_string:
            return None
        
        try:
            return datetime.strptime(date_string, date_format)
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parseando fecha '{date_string}': {e}")
            return None
    
    @staticmethod
    def clean_player_name(name: str) -> str:
        """
        Limpiar y normalizar nombre de jugador
        
        Args:
            name: Nombre del jugador
        
        Returns:
            str: Nombre limpio
        
        Examples:
            >>> DataParser.clean_player_name("  Nadal R.  ")
            'Nadal R.'
        """
        if not name:
            return "N/A"
        
        # Quitar espacios extras y caracteres especiales raros
        cleaned = ' '.join(name.strip().split())
        return cleaned
    
    @staticmethod
    def extract_location_from_title(title_attr: str) -> Dict[str, str]:
        """
        Extraer ciudad, país y superficie desde el atributo 'title'
        
        Args:
            title_attr: Ej. "Toronto (Canada), hard"
        
        Returns:
            dict: {'ciudad': str, 'pais': str, 'superficie': str}
        
        Examples:
            >>> DataParser.extract_location_from_title("Toronto (Canada), hard")
            {'ciudad': 'Toronto', 'pais': 'Canada', 'superficie': 'Dura'}
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