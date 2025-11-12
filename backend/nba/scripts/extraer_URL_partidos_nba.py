#!/usr/bin/env python3
"""
🏀 FASE 1: EXTRACCIÓN DE LISTA DE PARTIDOS NBA
Basado exactamente en el método del tenis que SÍ funciona en WSL

OBJETIVO: Crear lista de todos los partidos NBA con sus URLs
SALIDA: data/nba_matches_FECHA.json

Autor: David Alberto Coronado Tabares
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from playwright.async_api import async_playwright

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NBAMatchExtractor:
    """Extractor de lista de partidos NBA - Método Tenis"""
    
    def __init__(self):
        self.browser = None
        self.page = None
        self.playwright = None
        
    async def init_browser(self):
        """Inicializar navegador con configuración WSL (EXACTA DEL TENIS)"""
        logger.info("🚀 Iniciando navegador NBA...")
        
        try:
            self.playwright = await async_playwright().start()
            
            # CONFIGURACIÓN EXACTA DEL TENIS QUE FUNCIONA EN WSL
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
                    '--disable-features=TranslateUI',
                    '--disable-extensions',
                    '--no-first-run',
                    '--disable-default-apps',
                    '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                ]
            )
            
            self.page = await self.browser.new_page()
            await self.page.set_viewport_size({"width": 1920, "height": 1080})
            
            logger.info("✅ Navegador NBA listo")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error iniciando navegador: {e}")
            return False
        
    
    async def navigate_to_basketball(self):
        """Navegar DIRECTAMENTE a FlashScore NBA USA"""
        try:
            logger.info("🌐 Navegando a Flashscore NBA (USA)...")
            await self.page.goto(
                "https://www.flashscore.co/baloncesto/usa/nba/",
                wait_until="domcontentloaded",
                timeout=45000
            )
            
            await asyncio.sleep(3)
            logger.info("✅ Página cargada correctamente")
            
            # Manejar cookies (igual que en tenis)
            await self.handle_cookie_consent()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error navegando: {str(e)}")
            return False
    
    async def handle_cookie_consent(self):
        """Manejar banner de cookies (EXACTO DEL TENIS)"""
        try:
            accept_button_selector = "#onetrust-accept-btn-handler"
            button = await self.page.wait_for_selector(accept_button_selector, timeout=7000)
            if button:
                await button.click()
                logger.info("✅ Banner de cookies aceptado")
                await asyncio.sleep(2)
        except Exception:
            logger.info("ℹ️ No se encontró banner de cookies")
    
    async def extract_nba_matches(self):
        """Extraer partidos NBA (MÉTODO DEL TENIS)"""
        logger.info("🏀 Extrayendo partidos NBA...")
        
        matches_data = []
        
        try:
            # Scroll para cargar contenido (igual que tenis)
            await self.page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(2)
            
            logger.info("📄 Realizando scroll para cargar partidos...")
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(3)
            
            # Extraer partidos
            current_matches = await self.extract_matches_from_dom()
            if current_matches:
                matches_data.extend(current_matches)
                logger.info(f"   ✅ Extraídos {len(current_matches)} partidos")
            
            # Eliminar duplicados
            unique_matches = self.remove_duplicates(matches_data)
            logger.info(f"🎯 Total partidos únicos: {len(unique_matches)}")
            
            return unique_matches
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo partidos: {str(e)}")
            return matches_data
    
    async def extract_matches_from_dom(self):
        """Extraer partidos del DOM (ADAPTADO DEL TENIS)"""
        matches = []
        
        try:
            # Usar selector de FlashScore (igual que tenis)
            match_elements = await self.page.query_selector_all('.event__match')
            logger.info(f"🔍 Encontrados {len(match_elements)} elementos")
            
            for element in match_elements:
                try:
                    match_data = await self.extract_single_match(element)
                    if match_data and match_data.get('home_team') and match_data.get('away_team'):
                        matches.append(match_data)
                except:
                    continue
            
            return matches
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo DOM: {str(e)}")
            return matches
    
    async def extract_single_match(self, element):
        """Extraer datos de UN partido (MÉTODO TENIS CON URLs)"""
        try:
            match_data = {
                'home_team': '',
                'away_team': '',
                'score': '',
                'status': '',
                'match_url': None,
                'match_id': None,
                'league': ''
            }
            
            # NUEVO: Extraer URL y Match ID (EXACTO DEL TENIS)
            try:
                match_link_selectors = [
                    'a[href*="/partido/"]',
                    'a[href*="/match/"]',
                    'a[href*="baloncesto/"]',
                    'a'
                ]
                
                match_link = None
                for selector in match_link_selectors:
                    try:
                        match_link = await element.query_selector(selector)
                        if match_link:
                            href = await match_link.get_attribute('href')
                            if href and ('/partido/' in href or '/match/' in href):
                                break
                    except:
                        continue
                
                if match_link:
                    href = await match_link.get_attribute('href')
                    if href:
                        # Normalizar URL (igual que tenis)
                        if href.startswith('/'):
                            href = 'https://www.flashscore.co' + href
                        elif not href.startswith('http'):
                            href = 'https://www.flashscore.co/' + href
                        
                        match_data['match_url'] = href
                        
                        # Extraer Match ID
                        match_id_patterns = [
                            r'/partido/baloncesto/([^/#]+)',
                            r'/match/([^/#]+)',
                            r'mid=([A-Za-z0-9]+)',
                            r'/([A-Za-z0-9]{8,})/?'
                        ]
                        
                        for pattern in match_id_patterns:
                            match_id_search = re.search(pattern, href)
                            if match_id_search:
                                match_data['match_id'] = match_id_search.group(1)
                                break
                        
                        logger.debug(f"🔗 URL: {href}")
                        if match_data['match_id']:
                            logger.debug(f"🆔 ID: {match_data['match_id']}")
                
            except Exception as e:
                logger.debug(f"Error extrayendo URL/ID: {str(e)}")
                pass
            
            # Extraer nombres de equipos desde URL (MÁS CONFIABLE)
            if match_data['match_url']:
                try:
                    # Formato: /partido/baloncesto/boston-celtics-KYD9hVEm/orlando-magic-QZMS36Dn/
                    url_parts = match_data['match_url'].split('/')
                    
                    for i, part in enumerate(url_parts):
                        if part == 'baloncesto' and i + 2 < len(url_parts):
                            team1_raw = url_parts[i + 1]
                            team2_raw = url_parts[i + 2].split('?')[0]
                            
                            match_data['home_team'] = self._clean_team_name(team1_raw)
                            match_data['away_team'] = self._clean_team_name(team2_raw)
                            break
                except:
                    pass
            
            # Fallback: Extraer equipos del DOM
            if not match_data['home_team'] or not match_data['away_team']:
                try:
                    participant_elements = await element.query_selector_all('.event__participant')
                    if len(participant_elements) >= 2:
                        match_data['home_team'] = (await participant_elements[0].text_content()).strip()
                        match_data['away_team'] = (await participant_elements[1].text_content()).strip()
                except:
                    pass
            
            # Extraer resultado/score
            try:
                score_element = await element.query_selector('[class*="event__score"], [class*="score"]')
                if score_element:
                    score_text = await score_element.text_content()
                    if score_text:
                        match_data['score'] = re.sub(r'\s+', ' ', score_text).strip()
            except:
                pass
            
            # Extraer estado/tiempo
            try:
                time_element = await element.query_selector('[class*="event__time"], [class*="time"]')
                if time_element:
                    time_text = await time_element.text_content()
                    if time_text:
                        match_data['status'] = time_text.strip()
            except:
                pass
            
            # Validar que tengamos datos mínimos
            if not match_data['home_team'] or not match_data['away_team']:
                return None
            
            return match_data
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo partido: {str(e)}")
            return None
    
    def _clean_team_name(self, raw_name):
        """Limpiar nombre de equipo desde URL"""
        # Remover ID al final (8 caracteres)
        if '-' in raw_name:
            parts = raw_name.rsplit('-', 1)
            if len(parts) == 2 and len(parts[1]) == 8 and re.match(r'^[A-Za-z0-9]{8}$', parts[1]):
                name = parts[0]
            else:
                name = raw_name
        else:
            name = raw_name
        
        return name.replace('-', ' ').title()
    
    def remove_duplicates(self, matches_data):
        """Eliminar duplicados (IGUAL QUE TENIS)"""
        seen = set()
        unique_matches = []
        
        for match in matches_data:
            key = f"{match.get('home_team', '')}|{match.get('away_team', '')}"
            if key not in seen and match.get('home_team') and match.get('away_team'):
                seen.add(key)
                unique_matches.append(match)
        
        return unique_matches
    
    async def save_matches_data(self, matches_data):
        """Guardar datos en JSON (FORMATO MEJORADO)"""
        try:
            os.makedirs("data", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/nba_matches_{timestamp}.json"
            
            # Filtrar solo partidos NBA
            nba_matches = [m for m in matches_data if self._is_nba_match(m)]
            
            # Estructura de datos
            data = {
                'extraction_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_matches': len(nba_matches),
                'matches': nba_matches
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Datos guardados en: {filename}")
            logger.info(f"📊 Total partidos NBA: {len(nba_matches)}")
            
            # Estadísticas
            with_urls = sum(1 for m in nba_matches if m.get('match_url'))
            with_ids = sum(1 for m in nba_matches if m.get('match_id'))
            
            logger.info(f"🔗 Partidos con URLs: {with_urls}/{len(nba_matches)}")
            logger.info(f"🆔 Partidos con IDs: {with_ids}/{len(nba_matches)}")
            
            # Mostrar ejemplos
            logger.info("📋 EJEMPLOS DE PARTIDOS EXTRAÍDOS:")
            for i, match in enumerate(nba_matches[:5], 1):
                logger.info(f"   {i}. {match['home_team']} vs {match['away_team']}")
                logger.info(f"      📊 Estado: {match.get('status', 'N/A')}")
                if match.get('score'):
                    logger.info(f"      🏀 Score: {match['score']}")
                if match.get('match_id'):
                    logger.info(f"      🆔 ID: {match['match_id']}")
            
            return filename, nba_matches
            
        except Exception as e:
            logger.error(f"❌ Error guardando: {str(e)}")
            return None, None
    
    def _is_nba_match(self, match):
        """Verificar si es partido NBA"""
        nba_teams = [
            'celtics', 'lakers', 'warriors', 'heat', 'knicks', 'bulls', 'nets',
            '76ers', 'sixers', 'mavericks', 'mavs', 'clippers', 'bucks', 'suns',
            'nuggets', 'raptors', 'hawks', 'cavaliers', 'cavs', 'pelicans', 'jazz',
            'magic', 'pacers', 'hornets', 'pistons', 'rockets', 'kings', 'spurs',
            'thunder', 'timberwolves', 'wolves', 'trail blazers', 'blazers',
            'grizzlies', 'wizards'
        ]
        
        home = match.get('home_team', '').lower()
        away = match.get('away_team', '').lower()
        
        for team in nba_teams:
            if team in home or team in away:
                return True
        
        return False
    
    async def close(self):
        """Cerrar navegador"""
        try:
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            logger.info("📚 Navegador cerrado")
        except:
            pass


async def main():
    """Función principal"""
    print("🏀" + "="*60)
    print("🏀 FASE 1: EXTRACCIÓN DE LISTA DE PARTIDOS NBA")
    print("🏀" + "="*60)
    
    scraper = NBAMatchExtractor()
    
    try:
        # Inicializar
        await scraper.init_browser()
        
        # Navegar
        logger.info("🌐 FASE 1: Navegando a FlashScore...")
        navigation_success = await scraper.navigate_to_basketball()
        
        if not navigation_success:
            logger.error("❌ Falló la navegación")
            return
        
        # Extraer
        logger.info("🏀 FASE 2: Extrayendo partidos NBA...")
        matches_data = await scraper.extract_nba_matches()
        
        if matches_data:
            # Guardar
            logger.info("💾 FASE 3: Guardando datos...")
            filename, nba_matches = await scraper.save_matches_data(matches_data)
            
            if filename and nba_matches:
                logger.info(f"🎉 ÉXITO: {len(nba_matches)} partidos NBA guardados")
                
                # Mostrar primeras URLs
                logger.info("\n🔗 PRIMERAS URLs DETECTADAS:")
                for i, match in enumerate(nba_matches[:3], 1):
                    logger.info(f"   {i}. {match['home_team']} vs {match['away_team']}")
                    if match.get('match_id'):
                        logger.info(f"      🆔 ID: {match['match_id']}")
                    if match.get('match_url'):
                        logger.info(f"      🔗 URL: {match['match_url']}")
            else:
                logger.error("❌ Error guardando datos")
        else:
            logger.error("❌ No se extrajeron partidos")
        
    except Exception as e:
        logger.error(f"❌ Error general: {str(e)}")
    
    finally:
        await scraper.close()
    
    print("\n🏀 FASE 1 COMPLETADA")
    print("🚀 Siguiente paso: Extraer estadísticas de jugadores")

if __name__ == "__main__":
    asyncio.run(main())
    