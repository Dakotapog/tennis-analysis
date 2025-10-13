#!/usr/bin/env python3
"""
ZITA SCRAPER V4.1 - LIVE MATCHES - Extracción Definitiva de Partidos EN VIVO
Extrae únicamente las URLs y datos de los partidos de tenis que se están jugando en vivo en FlashScore.
"""

import asyncio
import logging
import os
import re
import json
from datetime import datetime
from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ZitaLiveScraper:
    def __init__(self):
        self.browser = None
        self.page = None
        self.capture_config = { "x": 0, "y": 0, "width": 1920, "height": 800 }
    
    async def init_browser(self):
        """Inicializar navegador con configuración optimizada."""
        logger.info("🚀 Iniciando navegador Zita para partidos EN VIVO...")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu',
                '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ]
        )
        self.page = await self.browser.new_page()
        await self.page.set_viewport_size({"width": 1920, "height": 1080})
        logger.info("✅ Navegador Zita listo")
    
    async def navigate_to_flashscore(self):
        """Navegar a FlashScore y prepararse para la extracción."""
        try:
            logger.info("📍 Navegando a Flashscore Tennis...")
            await self.page.goto("https://www.flashscore.com/tennis/", wait_until="domcontentloaded", timeout=45000)
            await asyncio.sleep(3)
            logger.info("✅ Página cargada correctamente")
            await self.handle_cookie_consent()
            return True
        except Exception as e:
            logger.error(f"❌ Error navegando: {str(e)}")
            return False

    async def handle_cookie_consent(self):
        """Manejar el banner de consentimiento de cookies si aparece."""
        try:
            button = await self.page.wait_for_selector("#onetrust-accept-btn-handler", timeout=7000)
            if button:
                await button.click()
                logger.info("✅ Banner de cookies aceptado.")
                await asyncio.sleep(2)
        except Exception:
            logger.info("ℹ️ No se encontró banner de cookies o ya fue aceptado.")
    
    async def extract_tennis_matches(self):
        """Extraer partidos de tenis EN VIVO con datos estructurados."""
        logger.info("🎾 Extrayendo partidos EN VIVO...")
        try:
            logger.info("🔄 Realizando scroll para cargar todos los partidos...")
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(3)
            
            current_matches = await self.extract_matches_from_dom()
            logger.info(f"   ✅ Extraídos {len(current_matches)} partidos EN VIVO.")
            
            unique_matches = self.remove_duplicates(current_matches)
            logger.info(f"🎯 Total partidos únicos EN VIVO extraídos: {len(unique_matches)}")
            return unique_matches
        except Exception as e:
            logger.error(f"❌ Error extrayendo partidos EN VIVO: {str(e)}")
            return []
    
    def clean_tournament_name(self, tournament_text):
        if not tournament_text: return "Sin Nombre"
        clean_name = tournament_text.strip()
        clean_name = re.sub(r'^(LIVE|EN VIVO|FINISHED|FINALIZADO)\s*-\s*', '', clean_name, flags=re.IGNORECASE).strip()
        clean_name = re.sub(r'\s+', ' ', clean_name).strip()
        if clean_name.isupper(): clean_name = clean_name.title()
        replacements = { r'\bAtp\b': 'ATP', r'\bWta\b': 'WTA', r'\bItf\b': 'ITF', r'\bUtr\b': 'UTR' }
        for pattern, replacement in replacements.items():
            clean_name = re.sub(pattern, replacement, clean_name, flags=re.IGNORECASE)
        return clean_name if clean_name else "Torneo Sin Nombre"

    async def extract_matches_from_dom(self):
        """
        MÉTODO FINAL: Recorre todos los elementos y filtra por la clase 'event__match--live'.
        """
        matches = []
        current_tournament = "Sin Torneo Asignado"
        is_doubles_tournament = False
        try:
            logger.info("🔍 Procesando elementos y filtrando partidos EN VIVO por su clase CSS...")
            all_elements = await self.page.query_selector_all('.event__title, .event__match')
            logger.info(f"   Encontrados {len(all_elements)} elementos totales (torneos y partidos).")

            for element in all_elements:
                element_class = await element.get_attribute('class') or ''
                
                if 'event__title' in element_class:
                    tournament_text = await element.text_content()
                    if tournament_text:
                        current_tournament = self.clean_tournament_name(tournament_text)
                        is_doubles_tournament = 'dobles' in current_tournament.lower() or 'doubles' in current_tournament.lower()
                        if is_doubles_tournament:
                            logger.info(f"🚫 Omitiendo torneo de dobles: {current_tournament}")
                        else:
                            logger.info(f"🏆 Nuevo torneo detectado: {current_tournament}")
                
                elif 'event__match' in element_class:
                    if is_doubles_tournament:
                        continue

                    # Filtro definitivo: verificar si el partido está en vivo por la clase CSS
                    if 'event__match--live' in element_class:
                        match_data = await self.extract_single_match(element)
                        if match_data and match_data.get('jugador1') and match_data.get('jugador2'):
                            match_data['torneo'] = current_tournament
                            matches.append(match_data)
                            logger.info(f"   🎾 Partido EN VIVO añadido a '{current_tournament}': {match_data['jugador1']} vs {match_data['jugador2']}")
            
            logger.info(f"✅ Extracción EN VIVO completada: {len(matches)} partidos procesados.")
            return matches
        except Exception as e:
            logger.error(f"❌ Error extrayendo del DOM: {str(e)}")
            return matches
    
    async def extract_single_match(self, element):
        """Extraer datos de un partido individual EN VIVO."""
        try:
            match_data = {
                'torneo': '', 'jugador1': '', 'jugador2': '', 'resultado': '', 'estado': '',
                'match_url': None, 'match_id': None, 'h2h_url': None
            }
            
            # Extraer URL y Match ID
            match_link = await element.query_selector('a[href*="/partido/"], a[href*="/match/"]')
            if match_link:
                href = await match_link.get_attribute('href')
                if href:
                    if href.startswith('/'): href = 'https://www.flashscore.com' + href
                    match_data['match_url'] = href
                    match_id_search = re.search(r'/(?:partido|match)/([^/#]+)', href)
                    if match_id_search:
                        match_data['match_id'] = match_id_search.group(1)
                        match_data['h2h_url'] = f"https://www.flashscore.co/partido/tenis/{match_data['match_id']}/#/h2h/general"

            # Extraer jugadores
            participant_elements = await element.query_selector_all('.event__participant')
            if len(participant_elements) >= 2:
                match_data['jugador1'] = (await participant_elements[0].text_content()).strip()
                match_data['jugador2'] = (await participant_elements[1].text_content()).strip()

            # Extraer estado/tiempo
            time_element = await element.query_selector('[class*="event__time"]')
            if time_element:
                match_data['estado'] = (await time_element.text_content()).strip()

            if not match_data['jugador1'] or not match_data['jugador2']: return None
            return match_data
        except Exception as e:
            logger.error(f"❌ Error extrayendo partido individual: {str(e)}")
            return None
    
    def remove_duplicates(self, matches_data):
        seen = set()
        unique_matches = []
        for match in matches_data:
            key = f"{match.get('jugador1', '')}|{match.get('jugador2', '')}"
            if key not in seen and match.get('jugador1') and match.get('jugador2'):
                seen.add(key)
                unique_matches.append(match)
        return unique_matches
    
    async def save_matches_data(self, matches_data):
        """Guardar datos de partidos EN VIVO en archivo JSON."""
        try:
            os.makedirs("data", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/zita_live_tennis_matches_{timestamp}.json"
            grouped_data = {}
            for match in matches_data:
                tournament_name = match.pop('torneo', 'Sin Torneo Asignado')
                if tournament_name not in grouped_data: grouped_data[tournament_name] = []
                grouped_data[tournament_name].append(match)
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(grouped_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Datos de partidos EN VIVO guardados en: {filename}")
            logger.info(f"📊 Total partidos EN VIVO guardados: {len(matches_data)}")
            return filename, grouped_data
        except Exception as e:
            logger.error(f"❌ Error guardando datos de partidos EN VIVO: {str(e)}")
            return None, {}
    
    async def close(self):
        try:
            if self.browser: await self.browser.close()
            if hasattr(self, 'playwright'): await self.playwright.stop()
            logger.info("🔄 Navegador cerrado correctamente")
        except Exception as e:
            logger.error(f"❌ Error cerrando navegador: {str(e)}")

async def main():
    scraper = ZitaLiveScraper()
    try:
        logger.info("🎾 === ZITA SCRAPER V4.1 - LIVE MATCHES ===")
        await scraper.init_browser()
        if not await scraper.navigate_to_flashscore():
            logger.error("❌ Falló la navegación inicial.")
            return
        
        matches_data = await scraper.extract_tennis_matches()
        if not matches_data:
            logger.warning("⚠️ No se encontraron partidos EN VIVO en este momento.")
            return

        filename, grouped_data = await scraper.save_matches_data(matches_data)
        if filename:
            logger.info("✅ === EXTRACCIÓN EN VIVO COMPLETADA EXITOSAMENTE ===")
            logger.info(f"📁 Archivo generado: {filename}")
        else:
            logger.error("❌ Error guardando los datos de partidos EN VIVO")
    except Exception as e:
        logger.error(f"❌ Error en ejecución principal: {str(e)}")
    finally:
        await scraper.close()

if __name__ == "__main__":
    asyncio.run(main())
