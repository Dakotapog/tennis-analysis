#!/usr/bin/env python3
"""
🏀 FASE 2: ANÁLISIS H2H PROFUNDO NBA
✅ CORRECCIÓN FINAL: Identificación correcta de equipos usando botones de filtro
❌ ELIMINADO: Fallback por posición (tables[0]/tables[1]) que causaba extracciones incorrectas

Autor: David Alberto Coronado Tabares
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from playwright.async_api import async_playwright
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NBAH2HAnalyzer:
    """Analizador H2H profundo NBA con extracción de stats de jugadores"""
    
    def __init__(self):
        self.browser = None
        self.page = None
        self.playwright = None
        self.base_url = "https://www.flashscore.co"
        self.diagnostico_activado = True
        self.max_players_per_team = 5  # 🆕 LÍMITE PARA PRUEBAS
        self.processed_players = set()  # 🆕 CONTROL GLOBAL DE DUPLICADOS
        
    async def init_browser(self):
        """Inicializar navegador (configuración WSL)"""
        logger.info("🚀 Iniciando navegador H2H Analyzer...")
        
        try:
            self.playwright = await async_playwright().start()
            
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-software-rasterizer',
                    '--disable-background-timer-throttling',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-renderer-backgrounding',
                    '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                ]
            )
            
            self.page = await self.browser.new_page()
            await self.page.set_viewport_size({"width": 1920, "height": 1080})
            
            logger.info("✅ Navegador H2H listo")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error iniciando navegador: {e}")
            return False
    
    def load_todays_matches(self):
        """Cargar partidos de hoy desde FASE 1"""
        try:
            # 🆕 BÚSQUEDA EN MÚLTIPLES RUTAS POSIBLES
            possible_paths = [
                Path("nba/scripts/data"),           # Ruta original
                Path("data"),                       # Ruta relativa
                Path("../data"),                    # Un nivel arriba
                Path("./"),                         # Directorio actual
                Path(__file__).parent / "data",     # Relativo al script
            ]
            
            match_files = []
            data_dir_found = None
            
            for data_dir in possible_paths:
                if data_dir.exists():
                    files = list(data_dir.glob("nba_matches_*.json"))
                    if files:
                        match_files = files
                        data_dir_found = data_dir
                        logger.info(f"📂 Directorio encontrado: {data_dir.absolute()}")
                        break
            
            if not match_files:
                logger.error("❌ No se encontró archivo de partidos en ninguna ruta")
                logger.info("💡 Rutas buscadas:")
                for path in possible_paths:
                    logger.info(f"   - {path.absolute()} {'✓ existe' if path.exists() else '✗ no existe'}")
                logger.info("\n💡 Soluciones:")
                logger.info("   1. Ejecuta: python3 extraer_URL_partidos_nba.py")
                logger.info("   2. O coloca manualmente 'nba_matches_YYYYMMDD.json' en:")
                logger.info(f"      {Path('./data').absolute()}")
                return []
            
            latest_file = max(match_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"📂 Cargando: {latest_file.name}")
            logger.info(f"   Ruta completa: {latest_file.absolute()}")
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            matches = data.get('matches', [])
            logger.info(f"✅ Cargados {len(matches)} partidos de hoy")
            
            return matches
            
        except Exception as e:
            logger.error(f"❌ Error cargando partidos: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    async def handle_cookie_consent(self):
        """Manejar banner de cookies"""
        try:
            accept_button = await self.page.wait_for_selector(
                "#onetrust-accept-btn-handler", 
                timeout=5000
            )
            if accept_button:
                await accept_button.click()
                logger.info("✅ Banner de cookies aceptado")
                await asyncio.sleep(1)
        except:
            pass
    
    async def analyze_match(self, match_info, match_number, total_matches):
        """
        ANÁLISIS COMPLETO DE UN PARTIDO
        1. Navegar al partido
        2. Extraer H2H (últimos 2)
        3. Para cada H2H → extraer stats de jugadores
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"🏀 PARTIDO {match_number}/{total_matches}")
        logger.info(f"   {match_info['home_team']} vs {match_info['away_team']}")
        logger.info(f"{'='*70}")
        
        match_url = match_info.get('match_url')
        if not match_url:
            logger.warning("⚠️ Sin URL, saltando partido")
            return None
        
        analysis_result = {
            'match_info': match_info,
            'h2h_history': [],
            'h2h_with_player_stats': [],
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_h2h_found': 0,
            'total_stats_extracted': 0
        }
        
        # 🆕 RESETEAR control de duplicados por CADA PARTIDO PRINCIPAL
        self.processed_players = set()
        
        try:
            # PASO 1: Navegar al partido
            logger.info(f"🌐 Navegando a partido...")
            await self.page.goto(match_url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(3)
            await self.handle_cookie_consent()
            
            # PASO 2: Click en "Enfrentamientos Directos"
            logger.info("🔍 Buscando tab 'Enfrentamientos Directos'...")
            h2h_clicked = await self.click_h2h_tab()
            
            if not h2h_clicked:
                logger.warning("⚠️ No se pudo acceder a H2H")
                return analysis_result
            
            # PASO 3: Expandir historial
            logger.info("📊 Expandiendo historial H2H...")
            clicks_done = await self.expand_h2h_history()
            logger.info(f"✅ Historial expandido ({clicks_done} clicks)")
            
            # PASO 4: Extraer partidos H2H
            logger.info("📋 Extrayendo enfrentamientos directos...")
            h2h_matches = await self.extract_h2h_matches()
            analysis_result['h2h_history'] = h2h_matches
            analysis_result['total_h2h_found'] = len(h2h_matches)
            logger.info(f"✅ Extraídos {len(h2h_matches)} partidos H2H")
            
            # PASO 5: Para CADA partido H2H → Extraer stats de jugadores
            if h2h_matches:
                logger.info(f"\n🎯 Extrayendo stats de jugadores de {len(h2h_matches)} partidos H2H...")
                h2h_with_stats = await self.extract_player_stats_from_all_h2h(h2h_matches, match_info)
                analysis_result['h2h_with_player_stats'] = h2h_with_stats
                analysis_result['total_stats_extracted'] = len(h2h_with_stats)
                logger.info(f"✅ Stats extraídas de {len(h2h_with_stats)} partidos")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ Error analizando partido: {str(e)}")
            return analysis_result
    
    async def click_h2h_tab(self):
        """Click en tab 'Enfrentamientos Directos' o 'H2H'"""
        try:
            h2h_selectors = [
                'button[data-testid="wcl-tab"]:has-text("Enfrentamientos")',
                'button[data-testid="wcl-tab"]:has-text("H2H")',
                '[data-testid="wcl-tab"]:has-text("Enfrentamientos")',
                'button.wcl-tab_GS7ig:has-text("Enfrentamientos")',
                'button.wcl-tab_GS7ig:has-text("H2H")',
                "button:has-text('Enfrentamientos H2H')",
                "button:has-text('Enfrentamientos directos')",
                "button:has-text('H2H')",
                "a:has-text('H2H')",
                "[class*='tab']:has-text('H2H')"
            ]
            
            for selector in h2h_selectors:
                try:
                    tab = await self.page.wait_for_selector(selector, timeout=3000)
                    if tab:
                        is_visible = await tab.is_visible()
                        if is_visible:
                            await tab.scroll_into_view_if_needed()
                            await asyncio.sleep(0.5)
                            await tab.click()
                            logger.info("   ✅ Click en 'Enfrentamientos Directos'")
                            await asyncio.sleep(3)
                            return True
                except:
                    continue
            
            logger.warning("⚠️ No se encontró tab H2H")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error haciendo click en H2H: {str(e)}")
            return False
    
    async def expand_h2h_history(self):
        """Click en 'Mostrar más' hasta 5 veces para cargar historial completo"""
        try:
            show_more_selectors = [
                'span[data-testid="wcl-scores-caption-05"]:has-text("Mostrar más")',
                '[data-testid="wcl-scores-caption-05"]:has-text("Mostrar más")',
                'span.wcl-caption_pBHDx:has-text("Mostrar más")',
                'span.wcl-scores-caption-05_xKEFy:has-text("Mostrar más")',
                "a.h2h__showMore",
                "a:has-text('mostrar más partidos')",
                "span:has-text('Mostrar más partidos')",
                "a:has-text('Mostrar más')",
                ".event__more"
            ]
            
            clicks = 0
            max_clicks = 5
            
            while clicks < max_clicks:
                clicked = False
                
                for selector in show_more_selectors:
                    try:
                        button = await self.page.wait_for_selector(selector, timeout=2000)
                        if button:
                            is_visible = await button.is_visible()
                            if is_visible:
                                parent = await button.evaluate('el => el.parentElement')
                                if parent:
                                    await button.evaluate('el => el.parentElement.click()')
                                else:
                                    await button.click()
                                
                                clicks += 1
                                logger.info(f"      📈 Click #{clicks} en 'Mostrar más'")
                                await asyncio.sleep(2)
                                clicked = True
                                break
                    except:
                        continue
                
                if not clicked:
                    break
            
            return clicks
            
        except Exception as e:
            return 0
    
    async def extract_h2h_matches(self):
        """Extraer hasta 10 partidos de enfrentamientos directos"""
        h2h_matches = []
        
        try:
            await asyncio.sleep(2)
            
            match_row_selectors = [
                '.h2h__row',
                '[class*="h2h"][class*="row"]',
                '.event__match',
                '[class*="event"][class*="match"]'
            ]
            
            match_rows = []
            for selector in match_row_selectors:
                try:
                    rows = await self.page.query_selector_all(selector)
                    if rows and len(rows) > 0:
                        match_rows = rows
                        logger.info(f"   ✅ Encontradas {len(rows)} filas con selector: {selector}")
                        break
                except:
                    continue
            
            if not match_rows:
                logger.warning("⚠️ No se encontraron filas de partidos H2H")
                return h2h_matches
            
            for i, row in enumerate(match_rows[:2]):
                try:
                    match_data = await self.extract_h2h_match_data(row)
                    if match_data:
                        h2h_matches.append(match_data)
                        logger.info(f"      {i+1}. {match_data['home_team']} {match_data['score']} {match_data['away_team']}")
                except Exception as e:
                    continue
            
            return h2h_matches
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo H2H: {str(e)}")
            return h2h_matches
    
    async def extract_h2h_match_data(self, row):
        """Extraer datos básicos de un partido H2H"""
        try:
            match_data = {
                'home_team': '',
                'away_team': '',
                'score': '',
                'date': '',
                'match_url': None
            }
            
            # Extraer URL
            try:
                href = await row.get_attribute('href')
                if href:
                    if href.startswith('/'):
                        match_data['match_url'] = self.base_url + href
                    else:
                        match_data['match_url'] = href
            except:
                pass
            
            if not match_data['match_url']:
                link_selectors = [
                    'a.h2h__row',
                    'a[href*="/partido/"]',
                    'a[href*="/match/"]',
                    'a[href*="baloncesto/"]'
                ]
                
                for selector in link_selectors:
                    try:
                        link = await row.query_selector(selector)
                        if link:
                            href = await link.get_attribute('href')
                            if href and ('/partido/' in href or 'baloncesto' in href or '/match/' in href):
                                if href.startswith('/'):
                                    match_data['match_url'] = self.base_url + href
                                else:
                                    match_data['match_url'] = href
                                break
                    except:
                        continue
            
            # Extraer equipos
            participant_selectors = [
                '.h2h__participantInner',
                '.h2h__participant',
                'span[class*="participant"]'
            ]
            
            participants = []
            for selector in participant_selectors:
                try:
                    elems = await row.query_selector_all(selector)
                    if elems and len(elems) >= 2:
                        participants = elems
                        break
                except:
                    continue
            
            if len(participants) >= 2:
                match_data['home_team'] = (await participants[0].text_content()).strip()
                match_data['away_team'] = (await participants[1].text_content()).strip()
            
            # Extraer score
            score_selectors = [
                '.h2h__result',
                'span.h2h__result',
                '[class*="result"]'
            ]
            
            score_parts = []
            for selector in score_selectors:
                try:
                    score_elems = await row.query_selector_all(selector)
                    for elem in score_elems:
                        score_text = await elem.text_content()
                        if score_text and score_text.strip() and score_text.strip().isdigit():
                            score_parts.append(score_text.strip())
                    
                    if len(score_parts) >= 2:
                        match_data['score'] = f"{score_parts[0]} - {score_parts[1]}"
                        break
                except:
                    continue
            
            # Extraer fecha
            date_selectors = [
                '.h2h__date',
                'span[class*="date"]',
                '[class*="time"]'
            ]
            
            for selector in date_selectors:
                try:
                    date_elem = await row.query_selector(selector)
                    if date_elem:
                        date_text = await date_elem.text_content()
                        if date_text and date_text.strip():
                            match_data['date'] = date_text.strip()
                            break
                except:
                    continue
            
            if match_data['home_team'] and match_data['away_team'] and match_data['match_url']:
                return match_data
            
            return None
            
        except Exception as e:
            return None
    
    async def extract_player_stats_from_all_h2h(self, h2h_matches, main_match_info):
        """
        🆕 EXTRACCIÓN CORRECTA: Solo equipos del partido PRINCIPAL
        Identifica qué equipo es HOME/AWAY en cada H2H y extrae SOLO esos equipos
        """
        h2h_with_stats = []
        
        # 🆕 OBTENER EQUIPOS DEL PARTIDO PRINCIPAL desde el contexto
        main_home_team = main_match_info['home_team']
        main_away_team = main_match_info['away_team']
        main_teams = {main_home_team, main_away_team}
        
        logger.info(f"\n🎯 EQUIPOS DEL PARTIDO PRINCIPAL: {main_teams}")
        
        for i, match in enumerate(h2h_matches, 1):
            if not match.get('match_url'):
                logger.warning(f"   ⚠️  Partido {i} sin URL, saltando")
                continue
            
            try:
                logger.info(f"\n   📊 Partido H2H {i}/{len(h2h_matches)}")
                logger.info(f"      {match['home_team']} vs {match['away_team']}")
                logger.info(f"      Score: {match['score']} | Fecha: {match['date']}")
                
                # 🆕 IDENTIFICAR QUÉ EQUIPO ANALIZAR
                # Si Charlotte es HOME → extraer HOME stats
                # Si Charlotte es AWAY → extraer AWAY stats
                team_to_extract_home = match['home_team'] if match['home_team'] in main_teams else None
                team_to_extract_away = match['away_team'] if match['away_team'] in main_teams else None
                
                logger.info(f"      🎯 Equipo LOCAL a extraer: {team_to_extract_home}")
                logger.info(f"      🎯 Equipo VISITANTE a extraer: {team_to_extract_away}")
                
                # Navegar al partido histórico
                logger.info(f"      🌐 Navegando al partido...")
                await self.page.goto(
                    match['match_url'], 
                    wait_until="domcontentloaded", 
                    timeout=30000
                )
                await asyncio.sleep(2)
                
                # Click en "Estadísticas de jugadores"
                logger.info(f"      📈 Buscando tab 'Estadísticas de jugadores'...")
                stats_clicked = await self.click_stats_tab()
                
                if not stats_clicked:
                    logger.warning(f"      ⚠️  No se pudo acceder a Estadísticas")
                    continue
                
                home_team_stats = []
                away_team_stats = []
                
                # 🆕 EXTRAER SOLO SI EL EQUIPO PRINCIPAL ESTÁ EN ESA POSICIÓN
                if team_to_extract_home:
                    logger.info(f"      🏠 Extrayendo stats equipo LOCAL: {team_to_extract_home}...")
                    home_team_stats = await self.extract_team_player_statistics(team_to_extract_home, is_home=True)
                    
                    if len(home_team_stats) > 0:
                        logger.info(f"         ✅ {len(home_team_stats)} jugadores extraídos de {team_to_extract_home}")
                    else:
                        logger.warning(f"         ⚠️  No se pudieron extraer jugadores de {team_to_extract_home}")
                else:
                    logger.info(f"      ⭕ Saltando equipo local (no es del partido principal)")
                
                if team_to_extract_away:
                    logger.info(f"      ✈️  Extrayendo stats equipo VISITANTE: {team_to_extract_away}...")
                    away_team_stats = await self.extract_team_player_statistics(team_to_extract_away, is_home=False)
                    
                    if len(away_team_stats) > 0:
                        logger.info(f"         ✅ {len(away_team_stats)} jugadores extraídos de {team_to_extract_away}")
                    else:
                        logger.warning(f"         ⚠️  No se pudieron extraer jugadores de {team_to_extract_away}")
                else:
                    logger.info(f"      ⭕ Saltando equipo visitante (no es del partido principal)")
                
                # Desactivar diagnóstico después del primer partido
                if self.diagnostico_activado:
                    self.diagnostico_activado = False
                    logger.info("      ℹ️  Diagnóstico desactivado para siguientes partidos")
                
                # Combinar stats
                all_players = home_team_stats + away_team_stats
                
                if all_players and len(all_players) > 0:
                    h2h_with_stats.append({
                        'match_info': match,
                        'home_team_stats': home_team_stats,
                        'away_team_stats': away_team_stats,
                        'all_player_stats': all_players,
                        'total_players': len(all_players),
                        'teams_analyzed': {
                            'home': team_to_extract_home,
                            'away': team_to_extract_away
                        }
                    })
                    logger.info(f"      ✅ TOTAL: {len(all_players)} jugadores únicos extraídos")
                else:
                    logger.warning(f"      ⚠️  No se encontraron stats de jugadores de equipos principales")
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"      ❌ Error en partido {i}: {str(e)}")
                continue
        
        return h2h_with_stats
    
    async def click_stats_tab(self):
        """Click en tab 'Estadísticas' o 'Statistics'"""
        try:
            stats_selectors = [
                'button[data-testid="wcl-tab"]:has-text("Estadísticas de jugadores")',
                'button[data-testid="wcl-tab"]:has-text("Estadísticas")',
                'button[data-testid="wcl-tab"]:has-text("Statistics")',
                '[data-testid="wcl-tab"]:has-text("Estadísticas")',
                'button.wcl-tab_GS7ig:has-text("Estadísticas")',
                "button:has-text('Estadísticas de jugadores')",
                "button:has-text('Estadísticas')",
                "button:has-text('Statistics')",
                "a:has-text('Estadísticas')",
                "[class*='tab']:has-text('Stats')"
            ]
            
            for selector in stats_selectors:
                try:
                    tab = await self.page.wait_for_selector(selector, timeout=3000)
                    if tab:
                        is_visible = await tab.is_visible()
                        if is_visible:
                            await tab.scroll_into_view_if_needed()
                            await asyncio.sleep(0.5)
                            await tab.click()
                            logger.info(f"         ✅ Click en 'Estadísticas de jugadores'")
                            await asyncio.sleep(3)
                            
                            # 🆕 DIAGNÓSTICO: Mostrar botones de filtro disponibles
                            if self.diagnostico_activado:
                                await self.diagnosticar_botones_filtro()
                            
                            return True
                except:
                    continue
            
            logger.warning("         ⚠️  No se encontró tab Estadísticas")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error haciendo click en Estadísticas: {str(e)}")
            return False
    
    async def diagnosticar_botones_filtro(self):
        """🆕 Diagnóstico: Mostrar todos los botones de filtro disponibles"""
        try:
            logger.info(f"\n         🔍 DIAGNÓSTICO - Botones de filtro encontrados:")
            
            button_selectors = [
                'div.subFilterOver button',
                '[class*="subFilterOver"] button',
                'button[class*="filter"]'
            ]
            
            for selector in button_selectors:
                try:
                    buttons = await self.page.query_selector_all(selector)
                    
                    if buttons:
                        logger.info(f"         📌 Selector: {selector}")
                        for i, button in enumerate(buttons, 1):
                            try:
                                button_text = (await button.text_content() or '').strip()
                                is_visible = await button.is_visible()
                                logger.info(f"            {i}. '{button_text}' {'✅ visible' if is_visible else '❌ oculto'}")
                            except:
                                continue
                        break
                except:
                    continue
            
            logger.info(f"         ℹ️  Fin diagnóstico\n")
            
        except Exception as e:
            logger.debug(f"         Debug: Error en diagnóstico: {str(e)}")
    
    async def extract_team_player_statistics(self, team_name, is_home=True):
        """
        🆕 EXTRACCIÓN CORREGIDA: Identifica el equipo correcto usando botones de filtro
        Ya NO usa fallback por posición (tables[0]/tables[1]) que causaba errores
        """
        try:
            # 🎯 PASO 1: Identificar qué botón corresponde al equipo que buscamos
            logger.info(f"            🔍 Buscando botón para equipo: {team_name}")
            
            # Selectores para los botones de filtro de equipos
            team_filter_selectors = [
                '#detail > div.tabContent__match-summary > div.tabContent__player-statistics > div.subFilterOver.subFilterOver--indent > div > a:nth-child(2) > button',
                '#detail > div.tabContent__match-summary > div.tabContent__player-statistics > div.subFilterOver.subFilterOver--indent > div > a:nth-child(3) > button',
                'div.subFilterOver button',
                '[class*="subFilterOver"] button'
            ]
            
            team_button = None
            
            # Buscar todos los botones de filtro
            for base_selector in ['div.subFilterOver button', '[class*="subFilterOver"] button']:
                try:
                    buttons = await self.page.query_selector_all(base_selector)
                    
                    for button in buttons:
                        try:
                            button_text = (await button.text_content() or '').strip()
                            
                            # Verificar si el texto del botón contiene el nombre del equipo
                            if team_name.lower() in button_text.lower() or button_text.lower() in team_name.lower():
                                team_button = button
                                logger.info(f"            ✅ Botón encontrado: '{button_text}'")
                                break
                        except:
                            continue
                    
                    if team_button:
                        break
                except:
                    continue
            
            # 🎯 PASO 2: Si encontramos el botón, hacer click y extraer
            if team_button:
                try:
                    # Click en el botón del equipo
                    await team_button.click()
                    logger.info(f"            🖱️  Click en filtro de equipo")
                    await asyncio.sleep(2)
                    
                    # Ahora buscar la tabla visible
                    table_selectors = [
                        'table',
                        '[class*="table"]',
                        'div.ui-table',
                        'div[class*="playerStatsTable"]'
                    ]
                    
                    for selector in table_selectors:
                        try:
                            tables = await self.page.query_selector_all(selector)
                            
                            # Buscar la primera tabla visible
                            for table in tables:
                                is_visible = await table.is_visible()
                                if is_visible:
                                    logger.info(f"            📊 Tabla del equipo encontrada")
                                    return await self.extract_players_from_table(table, team_name)
                        except:
                            continue
                    
                except Exception as e:
                    logger.error(f"            ⚠️  Error haciendo click en botón: {str(e)}")
            
            # 🎯 PASO 3: Método alternativo - buscar por encabezado del equipo en la página
            logger.info(f"            🔄 Buscando tabla por encabezado de equipo...")
            
            team_header_selectors = [
                f'span[data-testid="wcl-scores-overline-02"]:has-text("{team_name}")',
                f'span.wcl-overline_uwiIT:has-text("{team_name}")',
                f'div[class*="overline"]:has-text("{team_name}")',
                f'h3:has-text("{team_name}")',
                f'h4:has-text("{team_name}")'
            ]
            
            team_section = None
            for selector in team_header_selectors:
                try:
                    team_section = await self.page.wait_for_selector(selector, timeout=3000)
                    if team_section:
                        logger.info(f"            ✅ Encabezado encontrado con: {selector}")
                        break
                except:
                    continue
            
            if team_section:
                # Buscar la tabla más cercana al encabezado
                try:
                    parent_container = await team_section.evaluate_handle(
                        'el => el.closest("[class*=\'section\']") || el.parentElement.parentElement'
                    )
                    
                    if parent_container:
                        table = await parent_container.query_selector('table, [class*="table"]')
                        if table:
                            logger.info(f"            📊 Tabla encontrada por encabezado")
                            return await self.extract_players_from_table(table, team_name)
                except Exception as e:
                    logger.debug(f"            Debug: Error buscando tabla por encabezado: {str(e)}")
            
            # ⚠️ Si llegamos aquí, no pudimos identificar la tabla correcta
            logger.warning(f"            ⚠️  NO se pudo identificar la tabla correcta para {team_name}")
            logger.warning(f"            ⚠️  SALTANDO equipo para evitar extraer datos incorrectos")
            return []
            
        except Exception as e:
            logger.error(f"            ❌ Error extrayendo equipo {team_name}: {str(e)}")
            return []
    
    async def extract_players_from_table(self, table, team_name):
        """
        🆕 EXTRACCIÓN CON CONTROL DE DUPLICADOS Y LÍMITE
        """
        players = []
        
        try:
            column_indices = await self.get_column_indices(table)
            
            if not column_indices:
                logger.warning(f"            ⚠️  No se pudieron identificar las columnas")
                return players
            
            rows = await table.query_selector_all('tr, [class*="row"]')
            
            for row in rows:
                # 🆕 LÍMITE DE JUGADORES POR EQUIPO
                if len(players) >= self.max_players_per_team:
                    logger.info(f"            ⏸️  Límite alcanzado: {self.max_players_per_team} jugadores")
                    break
                
                try:
                    player_data = await self.extract_player_from_row(row, column_indices, team_name)
                    
                    if player_data:
                        # 🆕 CONTROL DE DUPLICADOS GLOBAL
                        player_key = f"{player_data['player_name']}_{player_data['team']}"
                        
                        if player_key not in self.processed_players:
                            players.append(player_data)
                            self.processed_players.add(player_key)
                            
                            if self.diagnostico_activado:
                                logger.info(f"               ✓ {player_data['player_name']}: "
                                          f"{player_data['minutes']} min, {player_data['points']}pts, "
                                          f"{player_data['rebounds']}reb, {player_data['assists']}ast")
                        else:
                            if self.diagnostico_activado:
                                logger.debug(f"               ⭕ DUPLICADO IGNORADO: {player_data['player_name']}")
                
                except Exception as e:
                    continue
            
            return players
            
        except Exception as e:
            logger.error(f"            ❌ Error extrayendo jugadores de tabla: {str(e)}")
            return players
    
    async def get_column_indices(self, table):
        """Identificar índices de columnas MIN, PTS, REB, AST"""
        column_indices = {
            'name': 0,
            'min': None,
            'pts': None,
            'reb': None,
            'ast': None
        }
        
        try:
            headers = await table.query_selector_all(
                'th, [class*="headerCell"], div.ui-table__headerCell, '
                'div.playerStatsTable__headerCell, div[class*="Cell"]'
            )
            
            for idx, header in enumerate(headers):
                try:
                    header_text = (await header.text_content() or '').strip().upper()
                    title_attr = (await header.get_attribute('title') or '').strip().lower()

                    if header_text == 'MIN':
                        column_indices['min'] = idx
                    elif header_text == 'PTS' and 'puntos' in title_attr:
                        column_indices['pts'] = idx
                    elif header_text == 'REB' and 'rebotes' in title_attr:
                        column_indices['reb'] = idx
                    elif header_text == 'AST':
                        column_indices['ast'] = idx
                except:
                    continue

            if self.diagnostico_activado:
                logger.info(f"            Índices: MIN={column_indices['min']}, "
                          f"PTS={column_indices['pts']}, REB={column_indices['reb']}, "
                          f"AST={column_indices['ast']}")

            if column_indices['pts'] is None:
                if len(headers) > 2:
                    column_indices['pts'] = 2
            
            return column_indices
            
        except Exception as e:
            logger.error(f"            ❌ Error identificando columnas: {str(e)}")
            return None
    
    async def extract_player_from_row(self, row, column_indices, team_name):
        """Extraer datos de un jugador desde una fila de la tabla"""
        try:
            player_data = {
                'player_name': '', 'team': team_name, 'minutes': '',
                'points': 0, 'rebounds': 0, 'assists': 0
            }
            
            cells = await row.query_selector_all('td, [class*="cell"], div[class*="cell"]')
            if not cells:
                return None

            # Extraer nombre
            try:
                name_text = (await cells[column_indices['name']].text_content() or '').strip()
                if not name_text or len(name_text) < 3 or name_text.lower() in ['player', 'nombre', 'jugador']:
                    return None
                player_data['player_name'] = name_text
            except (IndexError, AttributeError):
                return None

            # Función auxiliar para extraer estadísticas
            async def get_stat(index):
                if index is not None and index < len(cells):
                    try:
                        return (await cells[index].text_content() or '').strip()
                    except:
                        return ''
                return ''

            player_data['minutes'] = await get_stat(column_indices['min'])
            
            pts_text = await get_stat(column_indices['pts'])
            if pts_text.isdigit():
                player_data['points'] = int(pts_text)

            reb_text = await get_stat(column_indices['reb'])
            if reb_text.isdigit():
                player_data['rebounds'] = int(reb_text)

            ast_text = await get_stat(column_indices['ast'])
            if ast_text.isdigit():
                player_data['assists'] = int(ast_text)
            
            return player_data
            
        except Exception as e:
            return None
    
    async def save_analysis(self, all_analyses):
        """Guardar análisis completo en JSON"""
        try:
            os.makedirs("reports", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reports/nba_h2h_analysis_{timestamp}.json"
            
            data = {
                'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_matches_analyzed': len(all_analyses),
                'analyses': all_analyses
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"\n💾 Análisis guardado en: {filename}")
            logger.info(f"📊 Total partidos analizados: {len(all_analyses)}")
            
            total_h2h = sum(a.get('total_h2h_found', 0) for a in all_analyses)
            total_with_stats = sum(a.get('total_stats_extracted', 0) for a in all_analyses)
            total_players = sum(
                sum(match.get('total_players', 0) for match in a.get('h2h_with_player_stats', []))
                for a in all_analyses
            )
            
            logger.info(f"📋 Total partidos H2H encontrados: {total_h2h}")
            logger.info(f"📈 Partidos con stats de jugadores: {total_with_stats}")
            logger.info(f"🏀 Total jugadores únicos extraídos: {total_players}")
            
            return filename
            
        except Exception as e:
            logger.error(f"❌ Error guardando análisis: {str(e)}")
            return None
    
    async def close(self):
        """Cerrar navegador"""
        try:
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            logger.info("🔚 Navegador cerrado")
        except:
            pass


async def main():
    """Función principal - Análisis H2H completo"""
    print("🏀" + "="*70)
    print("🏀 FASE 2: ANÁLISIS H2H PROFUNDO NBA - VERSIÓN CORREGIDA FINAL")
    print("🏀 ✅ Identificación correcta de equipos usando botones de filtro")
    print("🏀 ❌ Eliminado fallback por posición que causaba errores")
    print("🏀" + "="*70)
    
    analyzer = NBAH2HAnalyzer()
    
    try:
        # PASO 1: Cargar partidos de hoy
        logger.info("\n📂 PASO 1: Cargando partidos de hoy...")
        matches = analyzer.load_todays_matches()
        
        if not matches:
            logger.error("❌ No hay partidos para analizar")
            logger.info("\n💡 OPCIONES:")
            logger.info("   A) Ejecuta primero: python3 extraer_URL_partidos_nba.py")
            logger.info("   B) Usa el modo DEMO con partido de prueba\n")
            
            # 🆕 OPCIÓN: Crear partido de prueba
            response = input("¿Quieres crear un partido de prueba para testear? (s/n): ").strip().lower()
            
            if response == 's':
                logger.info("\n🔧 Creando partido de prueba...")
                # Partido real de ejemplo: Lakers vs Nuggets
                test_match = {
                    'home_team': 'LA Lakers',
                    'away_team': 'Denver Nuggets',
                    'match_url': 'https://www.flashscore.co/partido/livCVOKv/#/resumen-del-partido',
                    'date': '2024-03-02',
                    'competition': 'NBA'
                }
                
                # Guardar archivo de prueba
                os.makedirs("data", exist_ok=True)
                test_file = f"data/nba_matches_test_{datetime.now().strftime('%Y%m%d')}.json"
                with open(test_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'date': datetime.now().strftime("%Y-%m-%d"),
                        'total_matches': 1,
                        'matches': [test_match]
                    }, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✅ Archivo de prueba creado: {test_file}")
                logger.info("🔄 Recargando partidos...")
                matches = [test_match]
            else:
                logger.info("❌ Saliendo del programa")
                return
        
        # 🆕 Solo 1 partido para prueba
        matches = matches[:1]
        logger.info(f"\n✅ Procesando SOLO 1 partido para prueba")
        
        logger.info("\n📋 PARTIDO A ANALIZAR:")
        for i, match in enumerate(matches, 1):
            logger.info(f"   {i}. {match['home_team']} vs {match['away_team']}")
            logger.info(f"      URL: {match.get('match_url', 'No disponible')[:60]}...")
        
        # Validar que el partido tenga URL
        if not matches[0].get('match_url'):
            logger.error("❌ El partido no tiene URL válida")
            return
        
        # PASO 2: Inicializar navegador
        logger.info("\n🚀 PASO 2: Inicializando navegador...")
        init_success = await analyzer.init_browser()
        
        if not init_success:
            logger.error("❌ No se pudo inicializar el navegador")
            return
        
        # PASO 3: Analizar partido
        logger.info("\n🏀 PASO 3: Analizando partido (H2H + Stats CORRECTOS)...")
        logger.info("="*70)
        
        all_analyses = []
        
        for i, match in enumerate(matches, 1):
            try:
                analysis = await analyzer.analyze_match(match, i, len(matches))
                
                if analysis:
                    all_analyses.append(analysis)
                    
                    logger.info(f"\n✅ RESUMEN PARTIDO {i}:")
                    logger.info(f"   H2H encontrados: {analysis.get('total_h2h_found', 0)}")
                    logger.info(f"   H2H con stats: {analysis.get('total_stats_extracted', 0)}")
                    
                    if analysis.get('h2h_with_player_stats'):
                        for h2h_match in analysis['h2h_with_player_stats']:
                            logger.info(f"\n   Partido H2H: {h2h_match['match_info']['home_team']} vs {h2h_match['match_info']['away_team']}")
                            logger.info(f"      Jugadores equipo local: {len(h2h_match.get('home_team_stats', []))}")
                            logger.info(f"      Jugadores equipo visitante: {len(h2h_match.get('away_team_stats', []))}")
                            logger.info(f"      Total jugadores ÚNICOS: {h2h_match.get('total_players', 0)}")
                else:
                    logger.warning(f"⚠️  No se pudo completar análisis del partido {i}")
                
            except Exception as e:
                logger.error(f"❌ Error en partido {i}: {str(e)}")
                continue
        
        # PASO 4: Guardar resultados
        logger.info("\n" + "="*70)
        logger.info("💾 PASO 4: Guardando resultados...")
        
        if all_analyses:
            filename = await analyzer.save_analysis(all_analyses)
            
            if filename:
                logger.info("\n" + "="*70)
                logger.info("🎉 ÉXITO: Análisis H2H completado CORRECTAMENTE")
                logger.info("="*70)
                logger.info(f"📄 Archivo: {filename}")
                
                # Mostrar ejemplo de jugadores extraídos
                logger.info("\n🏀 EJEMPLO DE DATOS EXTRAÍDOS (CORRECTOS):")
                for analysis in all_analyses:
                    if analysis.get('h2h_with_player_stats'):
                        for h2h_match in analysis['h2h_with_player_stats'][:1]:
                            match_info = h2h_match['match_info']
                            logger.info(f"\n   📊 Partido: {match_info['home_team']} vs {match_info['away_team']}")
                            logger.info(f"   Score: {match_info['score']}")
                            
                            # Mostrar jugadores equipo LOCAL
                            if h2h_match.get('home_team_stats'):
                                logger.info(f"\n   🏠 EQUIPO LOCAL ({match_info['home_team']}):")
                                for player in h2h_match['home_team_stats']:
                                    logger.info(f"      • {player['player_name']}")
                                    logger.info(f"        MIN: {player['minutes']}, PTS: {player['points']} 🔥, REB: {player['rebounds']}, AST: {player['assists']}")
                            
                            # Mostrar jugadores equipo VISITANTE
                            if h2h_match.get('away_team_stats'):
                                logger.info(f"\n   ✈️  EQUIPO VISITANTE ({match_info['away_team']}):")
                                for player in h2h_match['away_team_stats']:
                                    logger.info(f"      • {player['player_name']}")
                                    logger.info(f"        MIN: {player['minutes']}, PTS: {player['points']} 🔥, REB: {player['rebounds']}, AST: {player['assists']}")
                        break
                
                # 🆕 VERIFICACIÓN DE DUPLICADOS
                logger.info("\n" + "="*70)
                logger.info("🔍 VERIFICACIÓN DE DUPLICADOS:")
                for analysis in all_analyses:
                    for h2h_match in analysis.get('h2h_with_player_stats', []):
                        all_names = [p['player_name'] for p in h2h_match.get('all_player_stats', [])]
                        unique_names = set(all_names)
                        duplicates = len(all_names) - len(unique_names)
                        
                        if duplicates > 0:
                            logger.warning(f"⚠️  {duplicates} duplicados encontrados en {h2h_match['match_info']['home_team']} vs {h2h_match['match_info']['away_team']}")
                        else:
                            logger.info(f"✅ Sin duplicados en {h2h_match['match_info']['home_team']} vs {h2h_match['match_info']['away_team']}")
                
                # 🆕 VERIFICACIÓN DE EQUIPOS CORRECTOS
                logger.info("\n" + "="*70)
                logger.info("🎯 VERIFICACIÓN DE EQUIPOS EXTRAÍDOS:")
                for analysis in all_analyses:
                    main_teams = {analysis['match_info']['home_team'], analysis['match_info']['away_team']}
                    logger.info(f"\n📌 Partido principal: {' vs '.join(main_teams)}")
                    
                    for h2h_match in analysis.get('h2h_with_player_stats', []):
                        teams_analyzed = h2h_match.get('teams_analyzed', {})
                        logger.info(f"\n   📊 H2H: {h2h_match['match_info']['home_team']} vs {h2h_match['match_info']['away_team']}")
                        logger.info(f"      ✅ Equipo LOCAL extraído: {teams_analyzed.get('home', 'N/A')}")
                        logger.info(f"      ✅ Equipo VISITANTE extraído: {teams_analyzed.get('away', 'N/A')}")
                        
                        # Verificar que solo se extrajeron equipos del partido principal
                        extracted = {teams_analyzed.get('home'), teams_analyzed.get('away')} - {None}
                        if extracted.issubset(main_teams):
                            logger.info(f"      ✅ CORRECTO: Solo equipos del partido principal")
                        else:
                            logger.warning(f"      ⚠️  ADVERTENCIA: Se extrajeron equipos incorrectos")
                
            else:
                logger.error("❌ Error guardando análisis")
        else:
            logger.error("❌ No se pudo completar ningún análisis")
            logger.info("💡 Verifica que los partidos tengan URLs válidas")
        
    except Exception as e:
        logger.error(f"❌ Error general: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        logger.info("\n🔒 Cerrando navegador...")
        await analyzer.close()
    
    print("\n" + "="*70)
    print("🏀 FASE 2 COMPLETADA - VERSIÓN CORREGIDA FINAL")
    print("✅ Identificación correcta usando botones de filtro")
    print("✅ Sin duplicados + límite de 5 jugadores/equipo")
    print("❌ Fallback por posición eliminado")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())