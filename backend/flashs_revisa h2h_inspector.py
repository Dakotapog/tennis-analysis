#!/usr/bin/env python3
"""
🏀 ZITA NBA SCRAPER V9.0 - MÉTODO DEL TENIS ADAPTADO
Basado en el método exitoso usado para tenis en FlashScore
Autor: David Alberto Coronado Tabares + Claude

FILOSOFÍA:
- Playwright con stealth mode
- Navegación paso a paso verificando cada elemento
- Selectores específicos probados manualmente
- Sin prisas: esperas inteligentes
"""

import asyncio
import logging
import json
import re
from datetime import datetime
from pathlib import Path
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("nba_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NBAFlashScoreScraper:
    def __init__(self, headless=False, slow_mo=100):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.headless = headless
        self.slow_mo = slow_mo
        self.base_url = "https://www.flashscore.co"
        
    async def init_browser(self):
        """Inicializar navegador con configuración stealth"""
        logger.info("🚀 Iniciando navegador...")
        
        try:
            self.playwright = await async_playwright().start()
            
            # Lanzar navegador con opciones anti-detección
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                slow_mo=self.slow_mo,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-web-security',
                    '--disable-features=IsolateOrigins,site-per-process',
                ]
            )
            
            # Crear contexto con configuración realista
            self.context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                locale='es-CO',
                timezone_id='America/Bogota',
                permissions=['geolocation']
            )
            
            # Inyectar script anti-detección
            await self.context.add_init_script("""
                // Ocultar webdriver
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                
                // Ocultar chrome
                window.chrome = {
                    runtime: {}
                };
                
                // Permisos
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
            """)
            
            self.page = await self.context.new_page()
            
            # Configurar timeouts
            self.page.set_default_timeout(30000)
            self.page.set_default_navigation_timeout(60000)
            
            logger.info("✅ Navegador iniciado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error al iniciar navegador: {e}")
            return False
    
    async def accept_cookies(self):
        """Aceptar cookies si aparece el banner"""
        try:
            cookie_button = self.page.locator("#onetrust-accept-btn-handler")
            if await cookie_button.count() > 0:
                await cookie_button.click()
                await asyncio.sleep(1)
                logger.info("🍪 Cookies aceptadas")
        except:
            logger.info("ℹ️ No hay banner de cookies")
    
    async def navigate_to_basketball(self):
        """Navegar a la página principal de baloncesto"""
        logger.info("🌐 Navegando a FlashScore Baloncesto...")
        
        try:
            url = f"{self.base_url}/baloncesto/"
            await self.page.goto(url, wait_until='domcontentloaded')
            await asyncio.sleep(3)
            
            await self.accept_cookies()
            
            # Esperar a que carguen los partidos
            await self.page.wait_for_selector(".event__match, [id^='g_1']", timeout=15000)
            
            logger.info("✅ Página de baloncesto cargada")
            
            # Screenshot de diagnóstico
            await self.page.screenshot(path="screenshots/basketball_main.png", full_page=True)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error navegando a baloncesto: {e}")
            return False
    
    async def find_nba_section(self):
        """Encontrar la sección de NBA específicamente"""
        logger.info("🔍 Buscando sección NBA...")
        
        try:
            # Buscar el título "USA: NBA"
            nba_title = self.page.locator("text=/USA.*NBA/i").first
            
            if await nba_title.count() > 0:
                await nba_title.scroll_into_view_if_needed()
                await asyncio.sleep(1)
                logger.info("✅ Sección NBA encontrada")
                return True
            else:
                logger.warning("⚠️ No se encontró sección NBA")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error buscando sección NBA: {e}")
            return False
    
    async def extract_matches_from_page(self):
        """Extraer partidos de la página actual"""
        logger.info("📊 Extrayendo partidos...")
        
        matches = []
        
        try:
            # Scroll suave para cargar contenido
            for i in range(3):
                await self.page.evaluate(f"window.scrollTo(0, {500 * (i + 1)})")
                await asyncio.sleep(1)
            
            # Buscar todos los eventos (partidos)
            events = await self.page.locator(".event__match").all()
            logger.info(f"📋 Encontrados {len(events)} eventos en la página")
            
            for idx, event in enumerate(events, 1):
                try:
                    match_data = await self._extract_match_data(event, idx)
                    
                    if match_data and self._is_nba_match(match_data):
                        matches.append(match_data)
                        logger.info(f"  ✅ {len(matches)}. {match_data['home_team']} vs {match_data['away_team']}")
                        
                except Exception as e:
                    logger.debug(f"Error en evento {idx}: {e}")
                    continue
            
            logger.info(f"🏀 Total partidos NBA extraídos: {len(matches)}")
            return matches
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo partidos: {e}")
            return matches
    
    async def _extract_match_data(self, event, idx):
        """Extraer datos de un evento individual"""
        try:
            match_data = {}
            
            # Extraer enlace del partido
            link = await event.query_selector("a")
            if link:
                href = await link.get_attribute("href")
                if href:
                    match_data['match_url'] = self.base_url + href if href.startswith('/') else href
                    
                    # Extraer ID del partido
                    match_id = re.search(r'/\?mid=([A-Za-z0-9]+)', href)
                    if match_id:
                        match_data['match_id'] = match_id.group(1)
            
            # Extraer nombres de equipos
            home = await event.query_selector(".event__participant--home")
            away = await event.query_selector(".event__participant--away")
            
            if home and away:
                match_data['home_team'] = (await home.text_content()).strip()
                match_data['away_team'] = (await away.text_content()).strip()
            
            # Extraer resultado
            score = await event.query_selector(".event__score")
            if score:
                score_text = (await score.text_content()).strip()
                match_data['score'] = score_text
                
                # Parsear resultado
                scores = re.findall(r'\d+', score_text)
                if len(scores) >= 2:
                    match_data['home_score'] = int(scores[0])
                    match_data['away_score'] = int(scores[1])
            
            # Extraer estado/tiempo
            time_el = await event.query_selector(".event__time")
            if time_el:
                match_data['status'] = (await time_el.text_content()).strip()
            
            # Validar que tengamos datos mínimos
            if 'home_team' in match_data and 'away_team' in match_data:
                return match_data
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extrayendo datos del evento: {e}")
            return None
    
    def _is_nba_match(self, match):
        """Verificar si el partido es de la NBA"""
        nba_teams = [
            'Celtics', 'Lakers', 'Warriors', 'Heat', 'Knicks', 'Bulls', 'Nets',
            '76ers', 'Sixers', 'Mavericks', 'Mavs', 'Clippers', 'Bucks', 'Suns', 
            'Nuggets', 'Raptors', 'Hawks', 'Cavaliers', 'Cavs', 'Pelicans', 'Jazz',
            'Magic', 'Pacers', 'Hornets', 'Pistons', 'Rockets', 'Kings', 'Spurs',
            'Thunder', 'Timberwolves', 'Wolves', 'Trail Blazers', 'Blazers', 
            'Grizzlies', 'Wizards'
        ]
        
        home = match.get('home_team', '').lower()
        away = match.get('away_team', '').lower()
        
        for team in nba_teams:
            if team.lower() in home or team.lower() in away:
                return True
        
        return False
    
    async def navigate_to_match(self, match_url):
        """Navegar a la página de un partido específico"""
        logger.info(f"🔗 Navegando al partido: {match_url}")
        
        try:
            await self.page.goto(match_url, wait_until='domcontentloaded')
            await asyncio.sleep(3)
            
            # Esperar a que cargue el contenido
            await self.page.wait_for_selector(".tabs, [class*='tab']", timeout=10000)
            
            logger.info("✅ Página del partido cargada")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error navegando al partido: {e}")
            return False
    
    async def click_player_stats_tab(self):
        """Hacer clic en la pestaña de estadísticas de jugadores"""
        logger.info("🎯 Buscando pestaña de estadísticas...")
        
        possible_texts = [
            "Estadísticas de jugadores",
            "Estadísticas",
            "Player Stats",
            "Stats",
            "Jugadores"
        ]
        
        for text in possible_texts:
            try:
                # Intentar diferentes selectores
                selectors = [
                    f'a:has-text("{text}")',
                    f'button:has-text("{text}")',
                    f'[role="tab"]:has-text("{text}")',
                    f'.tabs__tab:has-text("{text}")',
                    f'div:has-text("{text}")'
                ]
                
                for selector in selectors:
                    try:
                        element = await self.page.wait_for_selector(selector, timeout=3000)
                        if element:
                            # Verificar que es visible
                            if await element.is_visible():
                                await element.scroll_into_view_if_needed()
                                await asyncio.sleep(0.5)
                                await element.click()
                                await asyncio.sleep(3)
                                
                                logger.info(f"✅ Click en pestaña: {text}")
                                
                                # Verificar que apareció contenido de stats
                                stats_visible = await self.page.locator("table, .stat, [class*='player']").count()
                                if stats_visible > 0:
                                    logger.info("✅ Estadísticas cargadas")
                                    return True
                    except:
                        continue
                        
            except:
                continue
        
        # Si no encontramos nada, listar pestañas disponibles
        logger.warning("⚠️ No se encontró pestaña de estadísticas")
        await self._list_available_tabs()
        return False
    
    async def _list_available_tabs(self):
        """Listar pestañas disponibles para debug"""
        logger.info("📋 Pestañas disponibles:")
        
        try:
            tabs = await self.page.locator("a, button, [role='tab']").all()
            
            for i, tab in enumerate(tabs[:20], 1):
                try:
                    text = await tab.text_content()
                    if text and text.strip():
                        logger.info(f"  {i}. {text.strip()}")
                except:
                    pass
                    
        except Exception as e:
            logger.debug(f"Error listando pestañas: {e}")
    
    async def extract_player_statistics(self, match_info):
        """Extraer estadísticas de jugadores de un partido"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🏀 EXTRAYENDO STATS: {match_info['home_team']} vs {match_info['away_team']}")
        logger.info(f"{'='*60}")
        
        # Navegar al partido
        if not await self.navigate_to_match(match_info['match_url']):
            return None
        
        # Screenshot inicial
        await self.page.screenshot(path=f"screenshots/match_{match_info['match_id']}_main.png")
        
        # Click en estadísticas de jugadores
        if not await self.click_player_stats_tab():
            logger.warning("⚠️ No se pudo acceder a estadísticas")
            return None
        
        # Screenshot de stats
        await self.page.screenshot(path=f"screenshots/match_{match_info['match_id']}_stats.png")
        
        # Extraer estadísticas de ambos equipos
        stats_data = {
            'match_info': match_info,
            'home_team': {
                'name': match_info['home_team'],
                'players': await self._extract_team_players_stats(match_info['home_team'])
            },
            'away_team': {
                'name': match_info['away_team'],
                'players': await self._extract_team_players_stats(match_info['away_team'])
            }
        }
        
        total_players = len(stats_data['home_team']['players']) + len(stats_data['away_team']['players'])
        logger.info(f"✅ Extraídos {total_players} jugadores en total")
        
        return stats_data
    
    async def _extract_team_players_stats(self, team_name):
        """Extraer estadísticas de jugadores de un equipo"""
        logger.info(f"📊 Extrayendo jugadores de: {team_name}")
        
        players = []
        
        try:
            # Intentar click en pestaña del equipo
            team_selectors = [
                f'button:has-text("{team_name}")',
                f'[role="tab"]:has-text("{team_name}")',
                f'a:has-text("{team_name}")'
            ]
            
            for selector in team_selectors:
                try:
                    element = await self.page.wait_for_selector(selector, timeout=3000)
                    if element and await element.is_visible():
                        await element.click()
                        await asyncio.sleep(2)
                        logger.info(f"  ✅ Seleccionado equipo: {team_name}")
                        break
                except:
                    continue
            
            # Buscar tabla de estadísticas
            # FlashScore usa diferentes estructuras, intentar varias
            
            # Método 1: Tabla HTML estándar
            table = await self.page.query_selector("table")
            if table:
                players = await self._parse_html_table(table)
            
            # Método 2: Divs con clases específicas
            if not players:
                players = await self._parse_custom_stats_divs()
            
            logger.info(f"  ✅ Extraídos {len(players)} jugadores")
            return players
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo {team_name}: {e}")
            return []
    
    async def _parse_html_table(self, table):
        """Parsear tabla HTML de estadísticas"""
        players = []
        
        try:
            rows = await table.query_selector_all("tr")
            
            # Identificar headers
            header_row = rows[0] if rows else None
            headers = []
            
            if header_row:
                header_cells = await header_row.query_selector_all("th, td")
                for cell in header_cells:
                    text = await cell.text_content()
                    headers.append(text.strip().lower())
            
            # Procesar filas de datos
            for row in rows[1:]:
                try:
                    cells = await row.query_selector_all("td")
                    
                    if len(cells) >= 2:
                        player_name = (await cells[0].text_content()).strip()
                        
                        # Verificar que no sea header
                        if player_name and player_name.lower() not in ['jugador', 'player', 'nombre', 'name']:
                            # Extraer valores numéricos
                            stats = {}
                            
                            for i, cell in enumerate(cells[1:], 1):
                                text = await cell.text_content()
                                # Buscar números
                                numbers = re.findall(r'\d+', text)
                                if numbers:
                                    if i < len(headers):
                                        stat_name = headers[i]
                                        stats[stat_name] = int(numbers[0])
                            
                            # Construir objeto jugador
                            player_data = {
                                'nombre': player_name,
                                'puntos': stats.get('pts', stats.get('puntos', 0)),
                                'rebotes': stats.get('reb', stats.get('rebotes', 0)),
                                'asistencias': stats.get('ast', stats.get('asistencias', 0)),
                                'minutos': stats.get('min', stats.get('minutos', 0)),
                                'stats_completas': stats
                            }
                            
                            players.append(player_data)
                            
                except Exception as e:
                    logger.debug(f"Error parseando fila: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error parseando tabla: {e}")
        
        return players
    
    async def _parse_custom_stats_divs(self):
        """Parsear estructura custom de FlashScore (divs)"""
        players = []
        
        try:
            # Buscar contenedores de jugadores
            player_rows = await self.page.locator("[class*='player'], [class*='participant']").all()
            
            for row in player_rows:
                try:
                    text = await row.text_content()
                    
                    # Extraer nombre (generalmente al inicio)
                    lines = text.strip().split('\n')
                    if lines:
                        player_name = lines[0].strip()
                        
                        # Extraer números
                        numbers = re.findall(r'\d+', text)
                        
                        if player_name and len(numbers) >= 3:
                            player_data = {
                                'nombre': player_name,
                                'puntos': int(numbers[0]) if len(numbers) > 0 else 0,
                                'rebotes': int(numbers[1]) if len(numbers) > 1 else 0,
                                'asistencias': int(numbers[2]) if len(numbers) > 2 else 0
                            }
                            
                            players.append(player_data)
                            
                except:
                    continue
                    
        except Exception as e:
            logger.error(f"Error parseando divs: {e}")
        
        return players
    
    async def save_data(self, data, filename):
        """Guardar datos en JSON"""
        try:
            output_dir = Path("data")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = output_dir / f"{filename}_{timestamp}.json"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Datos guardados: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"❌ Error guardando datos: {e}")
            return None
    
    async def close(self):
        """Cerrar navegador"""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            logger.info("👋 Navegador cerrado")
        except:
            pass


async def main():
    """Función principal"""
    print("\n" + "="*70)
    print("🏀 NBA FLASHSCORE SCRAPER V9.0 - MÉTODO TENIS")
    print("="*70 + "\n")
    
    # Crear directorio de screenshots
    Path("screenshots").mkdir(exist_ok=True)
    
    # Inicializar scraper
    scraper = NBAFlashScoreScraper(
        headless=False,  # Cambiar a True cuando funcione
        slow_mo=100
    )
    
    try:
        # Paso 1: Inicializar navegador
        if not await scraper.init_browser():
            logger.error("No se pudo inicializar el navegador")
            return
        
        # Paso 2: Navegar a baloncesto
        if not await scraper.navigate_to_basketball():
            logger.error("No se pudo cargar la página de baloncesto")
            return
        
        # Paso 3: Buscar sección NBA
        await scraper.find_nba_section()
        
        # Paso 4: Extraer partidos
        matches = await scraper.extract_matches_from_page()
        
        if not matches:
            logger.warning("⚠️ No se encontraron partidos NBA")
            return
        
        # Guardar lista de partidos
        await scraper.save_data(matches, "nba_matches")
        
        # Mostrar resumen
        print("\n" + "="*70)
        print(f"📊 RESUMEN: {len(matches)} partidos encontrados")
        print("="*70)
        
        for i, match in enumerate(matches[:10], 1):
            score = match.get('score', 'vs')
            print(f"{i}. {match['home_team']} {score} {match['away_team']}")
        
        # Paso 5: Extraer estadísticas (primeros 3 partidos)
        print("\n" + "="*70)
        print("📈 EXTRAYENDO ESTADÍSTICAS (primeros 3 partidos)")
        print("="*70 + "\n")
        
        all_stats = []
        
        for i, match in enumerate(matches[:3], 1):
            logger.info(f"\n[{i}/3] Procesando partido...")
            stats = await scraper.extract_player_statistics(match)
            
            if stats:
                all_stats.append(stats)
                await scraper.save_data(stats, f"match_stats_{match['match_id']}")
            
            # Pausa entre partidos
            await asyncio.sleep(2)
        
        # Guardar todas las estadísticas
        if all_stats:
            await scraper.save_data(all_stats, "nba_all_stats")
            print(f"\n✅ Estadísticas extraídas de {len(all_stats)} partidos")
        
        print("\n" + "="*70)
        print("🎉 SCRAPING COMPLETADO")
        print("📁 Revisa la carpeta 'data/' para los archivos JSON")
        print("📸 Revisa la carpeta 'screenshots/' para las capturas")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Proceso interrumpido por el usuario")
        
    except Exception as e:
        logger.error(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await scraper.close()


if __name__ == "__main__":
    asyncio.run(main())