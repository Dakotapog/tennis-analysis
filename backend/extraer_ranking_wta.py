#!/usr/bin/env python3
"""
WTA RANKING SCRAPER - Adaptado desde ZITA SCRAPER V2
Extrae el ranking WTA de FlashScore con una estructura robusta y asíncrona.
MEJORAS:
- Clase dedicada para el scraping.
- Uso de Asyncio y Playwright asíncrono.
- Selectores robustos para la extracción de datos.
- Configuración de navegador optimizada para WSL/Linux.
- Manejo de consentimiento de cookies.
- Logging detallado.
- Almacenamiento de datos en formato JSON estructurado.
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

class RankingScraper:
    def __init__(self):
        self.browser = None
        self.page = None
        self.playwright = None

    async def init_browser(self):
        """Inicializar navegador con configuración optimizada para WSL"""
        logger.info("🚀 Iniciando navegador para Ranking Scraper...")
        
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
                '--disable-features=TranslateUI',
                '--disable-extensions',
                '--no-first-run',
                '--disable-default-apps',
                '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ]
        )
        
        self.page = await self.browser.new_page()
        await self.page.set_viewport_size({"width": 1920, "height": 1080})
        logger.info("✅ Navegador listo")

    async def navigate_to_rankings(self):
        """Navegar a la página de rankings WTA de FlashScore.co."""
        try:
            logger.info("📍 Navegando a https://www.flashscore.co/tenis/rankings/wta/...")
            await self.page.goto("https://www.flashscore.co/tenis/rankings/wta/", 
                                wait_until="domcontentloaded", 
                                timeout=60000)
            
            logger.info("✅ Página de rankings cargada")
            await self.handle_cookie_consent()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error navegando a la página de rankings: {str(e)}")
            return False

    async def handle_cookie_consent(self):
        """Manejar el banner de consentimiento de cookies si aparece."""
        try:
            accept_button_selector = "#onetrust-accept-btn-handler"
            await self.page.wait_for_selector(accept_button_selector, timeout=7000)
            await self.page.click(accept_button_selector)
            logger.info("✅ Banner de cookies aceptado.")
            await asyncio.sleep(3)
        except Exception:
            logger.info("ℹ️ No se encontró banner de cookies o ya fue aceptado.")

    async def click_show_more(self, max_clicks=8):
        """🔄 Hacer click en 'Mostrar más' hasta 8 veces en una sección específica"""
        logger.info(f"   🔄 Expandiendo sección de ranking (máx {max_clicks} clicks)...")
        
        clicks_realizados = 0
        max_rank_target = 350

        for i in range(max_clicks):
            try:
                # Extraer el último ranking actual para decidir si continuar
                rows = await self.page.query_selector_all(".rankingTable__row")
                if rows:
                    last_row = rows[-1]
                    last_rank_el = await last_row.query_selector(".rankingTable__cell--rank")
                    if last_rank_el:
                        last_rank_str = await last_rank_el.text_content()
                        last_rank = int(re.sub(r'\D', '', last_rank_str))
                        if last_rank >= max_rank_target:
                            logger.info(f"🎯 Se alcanzó el ranking objetivo de {max_rank_target}. Deteniendo la expansión.")
                            break
                
                show_more_button = self.page.locator('button:has-text("Mostrar más")').first
                if await show_more_button.count() > 0 and await show_more_button.is_visible():
                    await show_more_button.click()
                    await self.page.wait_for_timeout(2000)  # Esperar que carguen más partidos
                    clicks_realizados += 1
                    logger.info(f"      ✅ Click {clicks_realizados} en 'Mostrar más' realizado")
                else:
                    logger.info("      ℹ️ No se encontró más botones 'Mostrar más'")
                    break
                    
            except Exception as e:
                logger.warning(f"      ⚠️ Error en click {i+1}: {e}")
                break
        
        logger.info(f"   ✅ Sección de ranking expandida con {clicks_realizados} clicks")
        return clicks_realizados

    async def extract_rankings(self):
        """Extraer la tabla de rankings WTA, haciendo clic en 'Mostrar más'."""
        logger.info("🎾 Extrayendo rankings WTA con selectores precisos...")
        
        try:
            # Esperar a que la tabla inicial cargue
            rows_selector = ".rankingTable__row"
            logger.info(f"⏳ Esperando por las filas de la tabla con el selector: '{rows_selector}'")
            await self.page.wait_for_selector(rows_selector, timeout=20000)

            # Cargar todos los jugadores hasta el ranking 350
            await self.click_show_more()
            
            player_rows = await self.page.query_selector_all(rows_selector)
            # Omitir la primera fila que es la cabecera (headRow)
            player_rows = [row for row in player_rows if "rankingTable__headRow" not in (await row.get_attribute("class"))]
            
            logger.info(f"🔍 Encontradas {len(player_rows)} filas de jugadoras (excluyendo cabecera).")
            
            rankings_data = []
            for row in player_rows:
                player_data = await self.extract_player_data(row)
                if player_data and player_data.get('name'):
                    rankings_data.append(player_data)
            
            logger.info(f"🎯 Total de jugadoras extraídas: {len(rankings_data)}")
            return rankings_data

        except Exception as e:
            logger.error(f"❌ Error extrayendo la tabla de rankings: {str(e)}")
            await self.page.screenshot(path="error_screenshot.png")
            logger.info("📸 Screenshot de error guardado en 'error_screenshot.png'")
            return []

    async def extract_player_data(self, row_element):
        """Extraer datos de una fila de jugadora usando selectores precisos y robustos."""
        try:
            # Usar query_selector y verificar la existencia de cada elemento
            rank_el = await row_element.query_selector(".rankingTable__cell--rank")
            name_el = await row_element.query_selector(".rankingTable__cell--player a")
            # El título de la nacionalidad está en un span con la clase 'flag'
            nationality_el = await row_element.query_selector(".rankingTable__cell--player span.flag")
            points_el = await row_element.query_selector(".rankingTable__cell--points")
            tournaments_el = await row_element.query_selector(".rankingTable__cell--tournaments")

            # Extraer texto solo si el elemento fue encontrado
            rank = await rank_el.text_content() if rank_el else ""
            name = await name_el.text_content() if name_el else ""
            nationality = await nationality_el.get_attribute('title') if nationality_el else "N/A"
            points = await points_el.text_content() if points_el else ""
            tournaments = await tournaments_el.text_content() if tournaments_el else ""

            if not name:
                return None

            # Limpieza de datos más robusta
            rank_str = re.sub(r'\D', '', rank)
            points_str = re.sub(r'\D', '', points)
            tournaments_str = re.sub(r'\D', '', tournaments)

            rank_clean = int(rank_str) if rank_str else None
            points_clean = int(points_str) if points_str else None
            tournaments_clean = int(tournaments_str) if tournaments_str else None

            return {
                'rank': rank_clean,
                'name': name.strip(),
                'nationality': nationality.strip() if nationality else "N/A",
                'points': points_clean,
                'tournaments_played': tournaments_clean
            }
        except Exception as e:
            logger.error(f"❌ Error procesando una fila de jugadora: {e}")
            return None

    async def save_rankings_data(self, rankings_data):
        """Guardar datos de rankings en un archivo JSON."""
        if not rankings_data:
            logger.warning("⚠️ No hay datos de ranking para guardar.")
            return None
        
        try:
            os.makedirs("data", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/wta_rankings_{timestamp}.json"

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(rankings_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Datos de ranking guardados en: {filename}")
            logger.info(f"📊 Total de jugadoras guardadas: {len(rankings_data)}")
            
            return filename
            
        except Exception as e:
            logger.error(f"❌ Error guardando los datos del ranking: {str(e)}")
            return None

    async def close(self):
        """Cerrar el navegador y playwright."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        logger.info("🔚 Navegador cerrado")

async def main():
    """Función principal para ejecutar el Ranking Scraper."""
    print("🎾" + "="*60)
    print("🎾 WTA RANKING SCRAPER - EXTRACCIÓN OPTIMIZADA")
    print("🎾" + "="*60)
    
    scraper = RankingScraper()
    
    try:
        await scraper.init_browser()
        
        navigation_success = await scraper.navigate_to_rankings()
        if not navigation_success:
            logger.error("❌ Falló la navegación inicial. Abortando.")
            return
        
        rankings = await scraper.extract_rankings()
        
        if rankings:
            filename = await scraper.save_rankings_data(rankings)
            if filename:
                logger.info(f"🎉 ÉXITO: {len(rankings)} jugadoras extraídas y guardadas en {filename}")
                print("\n📋 EJEMPLO DE DATOS (TOP 5):")
                for player in rankings[:5]:
                    print(f"  #{player['rank']} {player['name']} ({player['nationality']}) - {player['points']} puntos")
            else:
                logger.error("❌ Error guardando los datos del ranking.")
        else:
            logger.error("❌ No se extrajeron datos del ranking.")
            
    except Exception as e:
        logger.error(f"❌ Error general en Ranking Scraper: {str(e)}")
    
    finally:
        await scraper.close()
    
    print("\n✅ Proceso de scraping de ranking completado.")

if __name__ == "__main__":
    asyncio.run(main())
