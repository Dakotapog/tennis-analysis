"""
🎾 H2H Extractor - Coordinador principal de extracción H2H
Orquesta la navegación, extracción y enriquecimiento de datos
"""

import json
import logging
import signal
import atexit
from datetime import datetime
from pathlib import Path
import asyncio
import re

from analysis import EloRatingSystem, RankingManager, RivalryAnalyzer
from .browser_manager import BrowserManager
from .data_parser import DataParser

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 🛠️ FUNCIONES UTILITARIAS
# ═══════════════════════════════════════════════════════════════

def find_all_json_files():
    """🔍 Buscar archivos JSON en directorio actual y carpeta 'data'."""
    logger.info("🔍 Buscando archivos JSON...")
    data_dir = Path('data')
    all_json_files = []
    
    if data_dir.exists() and data_dir.is_dir():
        for json_file in data_dir.glob('*.json'):
            file_stats = json_file.stat()
            all_json_files.append({
                'filename': str(json_file),
                'size_mb': file_stats.st_size / (1024 * 1024),
                'modified_time': datetime.fromtimestamp(file_stats.st_mtime),
                'location': 'data'
            })
    
    all_json_files.sort(key=lambda x: x['modified_time'], reverse=True)
    return all_json_files


def analyze_json_structure(filename):
    """📊 Analizar estructura de un archivo JSON."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_matches = 0
        has_match_urls = False
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            total_matches += 1
                            if item.get('match_url'):
                                has_match_urls = True
        elif isinstance(data, list):
            total_matches = len(data)
            has_match_urls = any(isinstance(item, dict) and item.get('match_url') for item in data)
            
        return {'is_valid': total_matches > 0 and has_match_urls, 'total_matches': total_matches}
    except Exception as e:
        logger.error(f"❌ Error analizando {filename}: {e}")
        return {'is_valid': False}


def select_best_json_file():
    """🎯 Seleccionar el mejor archivo JSON automáticamente."""
    logger.info("🎯 Seleccionando el mejor archivo JSON...")
    json_files = find_all_json_files()
    if not json_files:
        logger.error("❌ No se encontraron archivos JSON.")
        return None
    
    for file_info in json_files:
        analysis = analyze_json_structure(file_info['filename'])
        if analysis['is_valid']:
            logger.info(f"🏆 Archivo seleccionado: {file_info['filename']}")
            return file_info['filename']
            
    logger.error("❌ No se encontraron archivos JSON válidos.")
    return None


# ═══════════════════════════════════════════════════════════════
# 🎾 CLASE PRINCIPAL
# ═══════════════════════════════════════════════════════════════

class H2HExtractor:
    """
    Extractor principal de datos H2H.
    
    Responsabilidades:
    - Navegación web con Playwright
    - Extracción de datos del DOM
    - Orquestación del flujo completo
    - Delegación de análisis a módulos especializados
    """
    
    def __init__(self):
        # Componentes de scraping
        self.browser_manager = BrowserManager()
        self.data_parser = DataParser()
        
        # Componentes de análisis (delegados)
        self.ranking_manager = RankingManager()
        self.elo_system = EloRatingSystem()
        self.rivalry_analyzer = RivalryAnalyzer(self.ranking_manager, self.elo_system)
        
        # Estado interno
        self.matches_queue = []
        self.current_match_index = 0
        self.total_matches_to_process = 80
        self.all_results = []
        self.page = None
        
        # Manejadores de señales
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._cleanup_on_exit)
    
    # ═══════════════════════════════════════════════════════════════
    # 🔧 LIFECYCLE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════
    
    def _signal_handler(self, signum, frame):
        """🛑 Manejador de señales de interrupción."""
        logger.info("🛑 Señal de interrupción recibida - Iniciando limpieza...")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
            else:
                loop.run_until_complete(self.cleanup())
        except Exception as e:
            logger.warning(f"⚠️ Error en signal handler: {e}")
    
    def _cleanup_on_exit(self):
        """🧹 Limpieza al salir."""
        logger.info("🧹 Ejecutando limpieza final...")
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.run_until_complete(self.cleanup())
        except:
            pass

    async def setup(self):
        """🚀 Configurar navegador."""
        self.page = await self.browser_manager.setup()

    async def cleanup(self):
        """🧹 Limpieza de recursos."""
        await self.browser_manager.cleanup()

    # ═══════════════════════════════════════════════════════════════
    # 📂 CARGA DE DATOS
    # ═══════════════════════════════════════════════════════════════

    def load_matches(self):
        """📂 Cargar partidos desde JSON."""
        json_file = select_best_json_file()
        if not json_file:
            return False
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_matches = []
        if isinstance(data, dict):
            for tournament_name, matches_in_tournament in data.items():
                if isinstance(matches_in_tournament, list):
                    for match in matches_in_tournament:
                        if isinstance(match, dict):
                            info = self.data_parser.extract_tournament_info(tournament_name)
                            match.update({
                                'torneo_nombre': info['nombre'],
                                'tipo_cancha': info['superficie'],
                                'pais': info['pais'],
                                'torneo_completo': info['completo']
                            })
                            all_matches.append(match)
        elif isinstance(data, list):
            all_matches = data
        
        valid_matches = [m for m in all_matches if m.get('match_url')]
        self.matches_queue = valid_matches[:self.total_matches_to_process]
        logger.info(f"✅ Cola de procesamiento creada con {len(self.matches_queue)} partidos.")
        return True

    # ═══════════════════════════════════════════════════════════════
    # 🔄 PROCESAMIENTO PRINCIPAL
    # ═══════════════════════════════════════════════════════════════

    async def run(self, optimized_weights=None):
        """🔄 Procesar todos los partidos en la cola."""
        if optimized_weights:
            logger.info(f"🚀 RE-PROCESANDO CON PESOS OPTIMIZADOS")
            self.all_results = []
            self.current_match_index = 0
        else:
            logger.info(f"🚀 INICIANDO PROCESAMIENTO DE {len(self.matches_queue)} PARTIDOS")

        successful_matches = 0
        failed_matches = 0
        
        for match_data in self.matches_queue:
            try:
                if await self._process_single_match(match_data, optimized_weights=optimized_weights):
                    successful_matches += 1
                else:
                    failed_matches += 1
                
                # Pausa entre partidos
                if self.matches_queue.index(match_data) < len(self.matches_queue) - 1:
                    logger.info("⏳ Pausa de 8 segundos...")
                    await asyncio.sleep(8)
            except Exception as e:
                logger.error(f"❌ Error crítico: {e}")
                failed_matches += 1
                continue
        
        logger.info("=" * 80)
        logger.info("🏁 PROCESAMIENTO COMPLETADO")
        logger.info(f"   ✅ Exitosos: {successful_matches}")
        logger.info(f"   ❌ Fallidos: {failed_matches}")
        
        if successful_matches + failed_matches > 0:
            success_rate = (successful_matches / (successful_matches + failed_matches)) * 100
            logger.info(f"   📊 Tasa de éxito: {success_rate:.1f}%")

    async def _process_single_match(self, match_data, optimized_weights=None):
        """⚙️ Procesar un único partido."""
        self.current_match_index += 1
        match_number = self.current_match_index
        
        p1 = match_data.get('jugador1', 'N/A')
        p2 = match_data.get('jugador2', 'N/A')
        
        logger.info("=" * 60)
        logger.info(f"⚙️ PARTIDO {match_number}/{self.total_matches_to_process}: {p1} vs {p2}")
        
        try:
            # 1. Navegar a la URL
            await self.page.goto(match_data['match_url'], wait_until='domcontentloaded', timeout=45000)
            await asyncio.sleep(8)
            
            # 2. Extraer información actual del partido
            current_info = await self._extract_current_match_info()
            match_data.update(current_info)

            # 3. Navegar a H2H
            await self._navigate_to_h2h_section()
            
            # 4. Extraer secciones H2H
            h2h_data = await self._extract_h2h_sections()
            
            # 5. Enriquecer con rankings (delegar a RankingManager)
            p1_hist = self._enrich_history(h2h_data['player1_history'], p1)
            p2_hist = self._enrich_history(h2h_data['player2_history'], p2)
            
            # 6. Análisis de forma (lógica simple, puede quedarse aquí)
            p1_form = self._analyze_recent_form(p1_hist, p1)
            p2_form = self._analyze_recent_form(p2_hist, p2)

            # 7. Calcular ELO (delegar a EloSystem vía RivalryAnalyzer)
            p1_elo = self.rivalry_analyzer.calculate_elo_from_history(p1, p1_hist)
            p2_elo = self.rivalry_analyzer.calculate_elo_from_history(p2, p2_hist)
            logger.info(f"   ⚡ ELO: {p1} ({p1_elo}), {p2} ({p2_elo})")

            # 8. Análisis de rivalidad (delegar a RivalryAnalyzer)
            current_context = {
                'country': match_data.get('pais', 'N/A'),
                'surface': match_data.get('tipo_cancha', 'N/A')
            }

            rivalry_analysis = self.rivalry_analyzer.analyze_rivalry(
                p1_hist, p2_hist, p1, p2, p1_form, p2_form, 
                h2h_data['head_to_head_matches'],
                current_context,
                p1_elo, p2_elo, 
                match_data.get('torneo_completo', ''), 
                optimized_weights
            )
            
            # 9. Consolidar y guardar
            result = self._consolidate_result(
                match_data, h2h_data, rivalry_analysis, 
                p1_hist, p2_hist, p1_form, p2_form, 
                p1_elo, p2_elo
            )
            self.all_results.append(result)
            
            self._log_match_summary(match_number, p1, p2, p1_hist, p2_hist, 
                                   h2h_data, rivalry_analysis)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error procesando partido {match_number}: {e}", exc_info=True)
            return False

    def _log_match_summary(self, match_number, p1, p2, p1_hist, p2_hist, h2h_data, rivalry):
        """📊 Registrar resumen del partido procesado."""
        logger.info(f"✅ Partido {match_number} procesado exitosamente")
        logger.info(f"   📊 Historial {p1}: {len(p1_hist)} partidos")
        logger.info(f"   📊 Historial {p2}: {len(p2_hist)} partidos")
        logger.info(f"   🥊 H2H directos: {len(h2h_data['head_to_head_matches'])}")
        logger.info(f"   🏆 Rankings: {p1} ({rivalry['player1_rank']}), {p2} ({rivalry['player2_rank']})")
        logger.info(f"   ⚔️ Oponentes comunes: {rivalry['common_opponents_count']}")
        logger.info(f"   🎯 Predicción: {rivalry['prediction']['favored_player']} "
                   f"({rivalry['prediction']['confidence']}%)")

    # ═══════════════════════════════════════════════════════════════
    # 🌐 NAVEGACIÓN Y EXTRACCIÓN
    # ═══════════════════════════════════════════════════════════════

    async def _extract_current_match_info(self):
        """ℹ️ Extraer información del partido actual desde la página."""
        logger.info("ℹ️ Extrayendo información del partido actual...")
        info = {"torneo_nombre": None, "tipo_cancha": None, "cuota1": None, "cuota2": None}
        
        try:
            # Extraer torneo
            tournament_element = self.page.locator('.tournamentHeader__country a').first
            if await tournament_element.count() > 0:
                full_text = await tournament_element.inner_text()
                
                if len(full_text) <= 150:
                    parsed = self.data_parser.extract_tournament_info(full_text)
                    info["torneo_nombre"] = parsed['nombre']
                    info["tipo_cancha"] = parsed['superficie']
                    info["torneo_completo"] = parsed['completo']
                    logger.info(f"   ✅ Torneo: {info['torneo_nombre']}, Cancha: {info['tipo_cancha']}")

            # Extraer cuotas
            odds_elements = await self.page.locator('.participant__odd').all()
            if len(odds_elements) >= 2:
                info["cuota1"] = await odds_elements[0].inner_text()
                info["cuota2"] = await odds_elements[1].inner_text()
                logger.info(f"   ✅ Cuotas: {info['cuota1']} - {info['cuota2']}")

        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo info del partido: {e}")
        
        return info

    async def _navigate_to_h2h_section(self):
        """🔄 Navegar a la sección H2H."""
        logger.info("🔄 Navegando a sección H2H...")
        
        selectors = [
            'a[href*="h2h"]', 'button:has-text("H2H")', 'a:has-text("H2H")',
            'a:has-text("Cara a cara")', 'a:has-text("Head to Head")',
            '[data-testid*="h2h"]', '.tab-h2h', '#h2h-tab'
        ]
        
        for selector in selectors:
            try:
                element = self.page.locator(selector).first
                if await element.count() > 0 and await element.is_visible():
                    await element.click()
                    await asyncio.sleep(5)
                    logger.info(f"   ✅ Click en H2H con selector: {selector}")
                    return True
            except:
                continue
        
        # Fallback: modificar URL
        logger.info("   🔄 Navegación directa a H2H...")
        h2h_url = self.page.url.split('#')[0] + '#/h2h'
        await self.page.goto(h2h_url, wait_until='domcontentloaded', timeout=30000)
        await asyncio.sleep(5)
        return True

    async def _extract_h2h_sections(self):
        """📊 Extraer las 3 secciones H2H."""
        logger.info("📊 Extrayendo secciones H2H...")
        
        try:
            await asyncio.sleep(8)
            
            # Screenshot para debugging
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            await self.page.screenshot(path=f"h2h_match_{self.current_match_index}_{timestamp}.png")
            
            h2h_sections = await self.page.locator('.h2h__section').all()
            logger.info(f"📊 Encontradas {len(h2h_sections)} secciones H2H")
            
            if len(h2h_sections) < 3:
                logger.warning("⚠️ No se encontraron las 3 secciones H2H.")
                return {'player1_history': [], 'player2_history': [], 'head_to_head_matches': []}

            # Extraer secciones con expansión
            await self._click_show_more(h2h_sections[0], max_clicks=8)
            p1_history = await self._parse_player_history(h2h_sections[0])
            
            await self._click_show_more(h2h_sections[1], max_clicks=8)
            p2_history = await self._parse_player_history(h2h_sections[1])
            
            await self._click_show_more(h2h_sections[2], max_clicks=2)
            h2h_matches = await self._parse_direct_h2h(h2h_sections[2])
            
            return {
                'player1_history': p1_history,
                'player2_history': p2_history,
                'head_to_head_matches': h2h_matches
            }
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo H2H: {e}")
            return {'player1_history': [], 'player2_history': [], 'head_to_head_matches': []}

    async def _click_show_more(self, section_locator, max_clicks=8):
        """🔄 Expandir sección haciendo click en 'Mostrar más'."""
        clicks = 0
        
        for _ in range(max_clicks):
            try:
                button = section_locator.locator('button:has-text("Mostrar más")').first
                if await button.count() > 0 and await button.is_visible():
                    await button.click()
                    await asyncio.sleep(3)
                    clicks += 1
                else:
                    break
            except:
                break
        
        if clicks > 0:
            logger.info(f"      ✅ {clicks} clicks en 'Mostrar más'")

    async def _parse_player_history(self, section):
        """👤 Extraer historial de un jugador."""
        matches = []
        rows = await section.locator('.h2h__row').all()
        
        for row in rows:
            try:
                date = await row.locator('.h2h__date').inner_text()
                result = await row.locator('.h2h__result').inner_text()
                
                # Oponente (no highlighted)
                participants = await row.locator('.h2h__participant').all()
                opponent = "N/A"
                for p in participants:
                    class_attr = await p.get_attribute('class') or ''
                    if 'highlighted' not in class_attr:
                        opp_elem = p.locator('.h2h__participantInner')
                        if await opp_elem.count() > 0:
                            opponent = await opp_elem.inner_text()
                        break
                
                # Outcome
                outcome_elem = row.locator('.h2h__icon > div')
                outcome = "Perdió"
                if await outcome_elem.count() > 0:
                    outcome_class = await outcome_elem.get_attribute('class') or ''
                    if 'win' in outcome_class.lower():
                        outcome = "Ganó"

                # Torneo y ubicación
                tournament = "N/A"
                city = "N/A"
                country = "N/A"
                surface = "N/A"

                event_elem = row.locator('.h2h__event').first
                if await event_elem.count() > 0:
                    try:
                        tournament = await event_elem.inner_text()
                    except:
                        pass
                    
                    title_attr = await event_elem.get_attribute('title')
                    if title_attr:
                        location_info = self.data_parser.extract_location_from_title(title_attr)
                        city = location_info['ciudad']
                        country = location_info['pais']
                        surface = location_info['superficie']

                matches.append({
                    'fecha': date.strip(),
                    'oponente': self.data_parser.clean_player_name(opponent),
                    'resultado': result.strip().replace('\n', '-'),
                    'outcome': outcome,
                    'torneo': tournament.strip().replace('\n', ' '),
                    'ciudad': city,
                    'pais': country,
                    'superficie': surface
                })
            
            except Exception as e:
                logger.warning(f"      ⚠️ Error parseando fila: {e}")
                continue
        
        logger.info(f"   ✅ Extraídos {len(matches)} partidos")
        return matches

    async def _parse_direct_h2h(self, section):
        """🥊 Extraer enfrentamientos directos."""
        matches = []
        rows = await section.locator('.h2h__row').all()
        
        for row in rows:
            try:
                date = await row.locator('.h2h__date').inner_text()
                result = await row.locator('.h2h__result').inner_text()
                
                participants = await row.locator('.h2h__participant').all()
                p1_name = "N/A"
                p2_name = "N/A"
                
                if len(participants) >= 2:
                    p1_elem = participants[0].locator('.h2h__participantInner')
                    p2_elem = participants[1].locator('.h2h__participantInner')
                    
                    if await p1_elem.count() > 0:
                        p1_name = await p1_elem.inner_text()
                    if await p2_elem.count() > 0:
                        p2_name = await p2_elem.inner_text()

                # Torneo y ubicación
                tournament = "N/A"
                city = "N/A"
                country = "N/A"
                surface = "N/A"

                event_elem = row.locator('.h2h__event').first
                if await event_elem.count() > 0:
                    try:
                        tournament = await event_elem.inner_text()
                    except:
                        pass
                    
                    title_attr = await event_elem.get_attribute('title')
                    if title_attr:
                        location_info = self.data_parser.extract_location_from_title(title_attr)
                        city = location_info['ciudad']
                        country = location_info['pais']
                        surface = location_info['superficie']

                # Ganador
                winner = self.data_parser.determine_winner_from_result(
                    result.strip().replace('\n', '-'),
                    p1_name, p2_name
                )
                
                # Fallback: buscar highlighted
                if winner == 'N/A':
                    for j, p in enumerate(participants):
                        class_attr = await p.get_attribute('class') or ''
                        if 'highlighted' in class_attr:
                            winner = p1_name if j == 0 else p2_name
                            break
                
                matches.append({
                    'fecha': date.strip(),
                    'jugador1': self.data_parser.clean_player_name(p1_name),
                    'jugador2': self.data_parser.clean_player_name(p2_name),
                    'resultado': result.strip().replace('\n', '-'),
                    'ganador': winner,
                    'torneo': tournament.strip().replace('\n', ' '),
                    'ciudad': city,
                    'pais': country,
                    'superficie': surface,
                    'ganador_sets': self.data_parser.extract_winner_sets(result.strip())
                })
            
            except Exception as e:
                logger.warning(f"      ⚠️ Error parseando H2H: {e}")
                continue
        
        logger.info(f"   ✅ Extraídos {len(matches)} enfrentamientos directos")
        return matches

    # ═══════════════════════════════════════════════════════════════
    # 📊 ENRIQUECIMIENTO Y ANÁLISIS SIMPLE
    # ═══════════════════════════════════════════════════════════════

    def _enrich_history(self, history, player_name):
        """🏆 Enriquecer historial con rankings (delegar a RankingManager)."""
        for match in history:
            if isinstance(match, dict):
                opponent = match.get('oponente', '')
                if opponent and opponent != 'N/A':
                    # Delegar a RankingManager
                    opponent_rank = self.ranking_manager.get_player_ranking(opponent)
                    opponent_info = self.ranking_manager.get_player_info(opponent)
                    
                    match['opponent_ranking'] = opponent_rank
                    if opponent_info:
                        match['opponent_points'] = opponent_info.get('points', 0)
                        match['opponent_nationality'] = opponent_info.get('nationality', 'N/A')
                
                # Delegar peso a RivalryAnalyzer
                match['opponent_weight'] = self.rivalry_analyzer.calculate_base_opponent_weight(
                    match.get('opponent_ranking')
                )
        return history

    def _analyze_recent_form(self, history, player_name, recent_count=20):
        """📈 Análisis simple de forma reciente (puede quedarse aquí)."""
        if not history:
            return None
        
        recent = history[:recent_count]
        wins = sum(1 for m in recent if m.get('outcome', '').lower() in ['ganó', 'win'])
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
        
        win_pct = round((wins / total * 100), 1) if total > 0 else 0
        
        return {
            'player_name': player_name,
            'recent_matches_count': total,
            'wins': wins,
            'losses': total - wins,
            'win_percentage': win_pct,
            'current_streak_count': current_streak,
            'current_streak_type': 'victorias' if streak_type == 'win' else 'derrotas',
            'form_status': self._classify_form(win_pct),
            'last_match_date': recent[0].get('fecha', 'N/A') if recent else 'N/A'
        }

    def _classify_form(self, win_percentage):
        """📊 Clasificar forma del jugador."""
        if win_percentage >= 75:
            return 'Excelente'
        elif win_percentage >= 60:
            return 'Buena'
        elif win_percentage >= 40:
            return 'Regular'
        else:
            return 'Mala'

    # ═══════════════════════════════════════════════════════════════
    # 📦 CONSOLIDACIÓN DE RESULTADOS
    # ═══════════════════════════════════════════════════════════════

    def _consolidate_result(self, match_data, h2h_data, rivalry_analysis, 
                           p1_hist, p2_hist, p1_form, p2_form, p1_elo, p2_elo):
        """📦 Consolidar resultados del partido procesado."""
        p1 = match_data['jugador1']
        p2 = match_data['jugador2']
        p1_key = p1.replace(' ', '_').replace('.', '')
        p2_key = p2.replace(' ', '_').replace('.', '')

        # Obtener métricas de ranking (delegar a RivalryAnalyzer)
        p1_metrics = self.rivalry_analyzer.get_ranking_metrics(p1)
        p2_metrics = self.rivalry_analyzer.get_ranking_metrics(p2)

        return {
            'match_number': self.current_match_index,
            'jugador1': p1,
            'jugador1_nacionalidad': rivalry_analysis.get('player1_nationality', 'N/A'),
            'jugador2': p2,
            'jugador2_nacionalidad': rivalry_analysis.get('player2_nationality', 'N/A'),
            'torneo_nombre': match_data.get('torneo_nombre', 'N/A'),
            'tipo_cancha': match_data.get('tipo_cancha', 'N/A'),
            'torneo_completo': match_data.get('torneo_completo', 'N/A'),
            'cuota1': match_data.get('cuota1', 'N/A'),
            'cuota2': match_data.get('cuota2', 'N/A'),
            'match_url': match_data.get('match_url', ''),
            f'historial_{p1_key}': p1_hist,
            f'historial_{p2_key}': p2_hist,
            'enfrentamientos_directos': h2h_data['head_to_head_matches'],
            'estadisticas': {
                f'partidos_{p1_key}': len(p1_hist),
                f'partidos_{p2_key}': len(p2_hist),
                'enfrentamientos_totales': len(h2h_data['head_to_head_matches'])
            },
            'ranking_analysis': {
                f'{p1_key}_ranking': rivalry_analysis['player1_rank'],
                f'{p2_key}_ranking': rivalry_analysis['player2_rank'],
                'common_opponents_count': rivalry_analysis['common_opponents_count'],
                'p1_rivalry_score': rivalry_analysis['p1_rivalry_score'],
                'p2_rivalry_score': rivalry_analysis['p2_rivalry_score'],
                'prediction': rivalry_analysis['prediction'],
                f'{p1_key}_metrics': p1_metrics,
                f'{p2_key}_metrics': p2_metrics,
                f'{p1_key}_elo': p1_elo,
                f'{p2_key}_elo': p2_elo
            },
            'form_analysis': {
                f'{p1_key}_form': p1_form,
                f'{p2_key}_form': p2_form,
            },
            'surface_analysis': {
                f'{p1_key}_surface_stats': rivalry_analysis.get('p1_surface_stats'),
                f'{p2_key}_surface_stats': rivalry_analysis.get('p2_surface_stats'),
            },
            'location_analysis': {
                f'{p1_key}_location_stats': rivalry_analysis.get('p1_location_stats'),
                f'{p2_key}_location_stats': rivalry_analysis.get('p2_location_stats'),
            },
            'common_opponents_detailed': rivalry_analysis.get('common_opponents_detailed', [])
        }

    # ═══════════════════════════════════════════════════════════════
    # 💾 PERSISTENCIA
    # ═══════════════════════════════════════════════════════════════

    def save_results(self, is_optimized=False):
        """💾 Guardar resultados en archivo JSON."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)
        
        suffix = "OPTIMIZED" if is_optimized else "enhanced"
        filename = reports_dir / f"h2h_results_{suffix}_{timestamp}.json"
        
        output_data = {
            'metadata': {
                'fecha_extraccion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_partidos_procesados': len(self.all_results),
                'version': '3.0_refactorizado',
                'funcionalidades': [
                    'Extracción H2H con Playwright',
                    'Análisis de rankings ATP',
                    'Sistema ELO calculado',
                    'Rivalidad transitiva',
                    'Predicciones avanzadas',
                    'Análisis de superficie',
                    'Ventaja de localización'
                ]
            },
            'partidos': self.all_results,
            'estadisticas_globales': self._generate_global_statistics()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Resultados guardados en: {filename}")
        logger.info(f"   📊 Total partidos: {len(self.all_results)}")
        
        # Imprimir resumen
        self._print_results_summary()
        
        return str(filename)

    def _generate_global_statistics(self):
        """📈 Generar estadísticas globales de todos los partidos."""
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
            # Jugadores únicos
            stats['jugadores_unicos'].add(match_result.get('jugador1', ''))
            stats['jugadores_unicos'].add(match_result.get('jugador2', ''))
            
            # Contar historiales
            p1_key = match_result['jugador1'].replace(' ', '_').replace('.', '')
            p2_key = match_result['jugador2'].replace(' ', '_').replace('.', '')
            
            stats['total_historiales_extraidos'] += len(match_result.get(f'historial_{p1_key}', []))
            stats['total_historiales_extraidos'] += len(match_result.get(f'historial_{p2_key}', []))
            
            # Enfrentamientos directos
            stats['total_enfrentamientos_directos'] += len(match_result.get('enfrentamientos_directos', []))
            
            # Rankings
            ranking_analysis = match_result.get('ranking_analysis', {})
            if ranking_analysis.get(f'{p1_key}_ranking') is not None:
                stats['rankings_encontrados'] += 1
            if ranking_analysis.get(f'{p2_key}_ranking') is not None:
                stats['rankings_encontrados'] += 1
            
            # Predicciones
            if ranking_analysis.get('prediction'):
                stats['predicciones_generadas'] += 1
            
            # Oponentes comunes
            common_count = ranking_analysis.get('common_opponents_count', 0)
            if common_count > 0:
                stats['oponentes_comunes_total'] += common_count
                stats['analisis_rivalidad_exitosos'] += 1
        
        # Convertir set a count
        stats['jugadores_unicos'] = len(stats['jugadores_unicos'])
        
        return stats

    def _print_results_summary(self):
        """📋 Imprimir resumen detallado de los resultados."""
        logger.info("=" * 80)
        logger.info("📋 RESUMEN DETALLADO DE RESULTADOS")
        logger.info("=" * 80)
        
        if not self.all_results:
            logger.info("❌ No hay resultados para mostrar")
            return
        
        for i, match_result in enumerate(self.all_results, 1):
            p1 = match_result.get('jugador1', 'N/A')
            p2 = match_result.get('jugador2', 'N/A')
            p1_key = p1.replace(' ', '_').replace('.', '')
            p2_key = p2.replace(' ', '_').replace('.', '')
            
            ranking_analysis = match_result.get('ranking_analysis', {})
            rank1 = ranking_analysis.get(f'{p1_key}_ranking', 'N/A')
            rank2 = ranking_analysis.get(f'{p2_key}_ranking', 'N/A')
            
            cuota1 = match_result.get('cuota1', 'N/A')
            cuota2 = match_result.get('cuota2', 'N/A')
            
            prediction = ranking_analysis.get('prediction', {})
            favored = prediction.get('favored_player', 'N/A')
            confidence = prediction.get('confidence', 0)
            
            logger.info(f"\n🎾 PARTIDO {i}: {p1} (#{rank1}) vs {p2} (#{rank2})")
            logger.info(f"   💰 Cuotas: {cuota1} - {cuota2}")
            logger.info(f"   🏟️ Torneo: {match_result.get('torneo_nombre', 'N/A')}")
            logger.info(f"   🎾 Superficie: {match_result.get('tipo_cancha', 'N/A')}")
            
            # Historial
            p1_hist_len = len(match_result.get(f'historial_{p1_key}', []))
            p2_hist_len = len(match_result.get(f'historial_{p2_key}', []))
            logger.info(f"   📊 Historial: {p1} ({p1_hist_len}), {p2} ({p2_hist_len})")
            
            # H2H
            h2h_len = len(match_result.get('enfrentamientos_directos', []))
            logger.info(f"   🥊 H2H: {h2h_len} enfrentamientos directos")
            
            # Forma
            form_analysis = match_result.get('form_analysis', {})
            p1_form = form_analysis.get(f'{p1_key}_form', {})
            p2_form = form_analysis.get(f'{p2_key}_form', {})
            
            if p1_form:
                logger.info(f"   📈 Forma {p1}: {p1_form.get('win_percentage', 0)}% "
                           f"({p1_form.get('form_status', 'N/A')})")
            if p2_form:
                logger.info(f"   📈 Forma {p2}: {p2_form.get('win_percentage', 0)}% "
                           f"({p2_form.get('form_status', 'N/A')})")
            
            # ELO
            p1_elo = ranking_analysis.get(f'{p1_key}_elo', 'N/A')
            p2_elo = ranking_analysis.get(f'{p2_key}_elo', 'N/A')
            logger.info(f"   ⚡ ELO: {p1} ({p1_elo}), {p2} ({p2_elo})")
            
            # Oponentes comunes
            common_count = ranking_analysis.get('common_opponents_count', 0)
            logger.info(f"   ⚔️ Oponentes comunes: {common_count}")
            
            # Predicción
            logger.info(f"   🎯 PREDICCIÓN: {favored} ({confidence}% confianza)")
            
            # Desglose de puntuación
            scores = prediction.get('scores', {})
            if scores:
                p1_weight = scores.get('p1_final_weight', 0)
                p2_weight = scores.get('p2_final_weight', 0)
                logger.info(f"      ⚖️ Pesos finales: {p1} ({p1_weight:.2f}) vs {p2} ({p2_weight:.2f})")
            
            logger.info("-" * 80)

    # ═══════════════════════════════════════════════════════════════
    # 🔄 RECALCULACIÓN CON PESOS OPTIMIZADOS
    # ═══════════════════════════════════════════════════════════════

    async def recalculate_with_optimized_weights(self, initial_file, optimized_weights):
        """🧠 Recalcular predicciones con pesos optimizados sin re-scrapear."""
        logger.info("🧠 Recalculando predicciones con pesos optimizados...")
        
        try:
            # Cargar resultados previos
            with open(initial_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            recalculated_matches = []
            
            for match in data.get('partidos', []):
                p1 = match['jugador1']
                p2 = match['jugador2']
                p1_key = p1.replace(' ', '_').replace('.', '')
                p2_key = p2.replace(' ', '_').replace('.', '')

                # Extraer datos ya scrapeados
                p1_hist = match.get(f'historial_{p1_key}', [])
                p2_hist = match.get(f'historial_{p2_key}', [])
                p1_form = match.get('form_analysis', {}).get(f'{p1_key}_form')
                p2_form = match.get('form_analysis', {}).get(f'{p2_key}_form')
                h2h_matches = match.get('enfrentamientos_directos', [])
                
                # Extraer ELO y contexto
                p1_elo = match.get('ranking_analysis', {}).get(f'{p1_key}_elo', 1500)
                p2_elo = match.get('ranking_analysis', {}).get(f'{p2_key}_elo', 1500)
                
                current_context = {
                    'country': match.get('pais', 'N/A'),
                    'surface': match.get('tipo_cancha', 'N/A')
                }
                
                # Recalcular SOLO el análisis de rivalidad con nuevos pesos
                new_rivalry_analysis = self.rivalry_analyzer.analyze_rivalry(
                    p1_hist, p2_hist, p1, p2, p1_form, p2_form,
                    h2h_matches, current_context, p1_elo, p2_elo,
                    match.get('torneo_completo', ''),
                    optimized_weights=optimized_weights
                )
                
                # Actualizar solo la predicción
                match['ranking_analysis']['prediction'] = new_rivalry_analysis['prediction']
                match['ranking_analysis']['p1_rivalry_score'] = new_rivalry_analysis['p1_rivalry_score']
                match['ranking_analysis']['p2_rivalry_score'] = new_rivalry_analysis['p2_rivalry_score']
                
                recalculated_matches.append(match)
            
            # Reemplazar resultados
            self.all_results = recalculated_matches
            
            # Guardar archivo optimizado
            return self.save_results(is_optimized=True)

        except Exception as e:
            logger.error(f"❌ Error recalculando predicciones: {e}", exc_info=True)
            return None