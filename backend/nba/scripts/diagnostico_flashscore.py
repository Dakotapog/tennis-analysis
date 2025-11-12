#!/usr/bin/env python3
"""
🔍 HERRAMIENTA DE DIAGNÓSTICO FLASHSCORE NBA
Para identificar selectores exactos del DOM

OBJETIVO: Ayudar a identificar los selectores correctos
SALIDA: Muestra estructura del DOM y selectores útiles

Autor: David Alberto Coronado Tabares
"""

import asyncio
import logging
from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FlashScoreDiagnostic:
    def __init__(self):
        self.browser = None
        self.page = None
        self.playwright = None
    
    async def init_browser(self):
        """Inicializar navegador - CONFIGURACIÓN WSL"""
        try:
            self.playwright = await async_playwright().start()
            
            # HEADLESS=True para WSL (sin interfaz gráfica)
            self.browser = await self.playwright.chromium.launch(
                headless=True,  # ✅ CAMBIADO: True para WSL
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-software-rasterizer',
                    '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                ]
            )
            
            self.page = await self.browser.new_page()
            await self.page.set_viewport_size({"width": 1920, "height": 1080})
            
            logger.info("✅ Navegador inicializado (modo headless)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error iniciando navegador: {e}")
            return False
    
    async def diagnose_nba_page(self):
        """Diagnosticar página NBA"""
        logger.info("🔍 DIAGNÓSTICO PÁGINA NBA")
        
        try:
            # Navegar
            logger.info("🌐 Navegando a NBA...")
            await self.page.goto("https://www.flashscore.co/baloncesto/usa/nba/", timeout=45000)
            await asyncio.sleep(3)
            
            # Cookies
            try:
                btn = await self.page.wait_for_selector("#onetrust-accept-btn-handler", timeout=5000)
                if btn:
                    await btn.click()
                    await asyncio.sleep(1)
            except:
                pass
            
            # DIAGNÓSTICO 1: Estructura de headers
            logger.info("\n📋 DIAGNÓSTICO 1: HEADERS DE TORNEOS")
            headers = await self.page.query_selector_all('[class*="headerLeague"], [class*="event__header"]')
            for i, header in enumerate(headers[:5], 1):
                text = await header.text_content()
                classes = await header.get_attribute('class')
                logger.info(f"   {i}. Texto: {text.strip()[:50]}")
                logger.info(f"      Classes: {classes}")
            
            # DIAGNÓSTICO 2: Tab "Partidos de hoy"
            logger.info("\n📋 DIAGNÓSTICO 2: TAB 'PARTIDOS DE HOY'")
            tabs = await self.page.query_selector_all('[class*="tabs"]')
            for i, tab in enumerate(tabs[:5], 1):
                text = await tab.text_content()
                classes = await tab.get_attribute('class')
                logger.info(f"   {i}. Texto: {text.strip()[:50]}")
                logger.info(f"      Classes: {classes}")
            
            # Buscar específicamente el tab de hoy
            today_tab = await self.page.query_selector('div.tabs__ear')
            if today_tab:
                today_text = await today_tab.text_content()
                today_classes = await today_tab.get_attribute('class')
                logger.info(f"\n   ✅ TAB ENCONTRADO:")
                logger.info(f"      Texto: {today_text}")
                logger.info(f"      Classes: {today_classes}")
            
            # DIAGNÓSTICO 3: Partidos visibles
            logger.info("\n📋 DIAGNÓSTICO 3: PARTIDOS VISIBLES")
            matches = await self.page.query_selector_all('.event__match')
            logger.info(f"   Total partidos visibles: {len(matches)}")
            
            for i, match in enumerate(matches[:3], 1):
                participants = await match.query_selector_all('[class*="participant"]')
                if len(participants) >= 2:
                    home = await participants[0].text_content()
                    away = await participants[1].text_content()
                    logger.info(f"   {i}. {home.strip()} vs {away.strip()}")
            
            # DIAGNÓSTICO 4: Buscar sección NBA específica
            logger.info("\n📋 DIAGNÓSTICO 4: SECCIÓN NBA")
            nba_selectors = [
                'strong:has-text("NBA")',
                '[data-testid="wcl-scores-simple-text-01"]',
                '.headerLeague__title-text'
            ]
            
            for selector in nba_selectors:
                try:
                    element = await self.page.query_selector(selector)
                    if element:
                        text = await element.text_content()
                        logger.info(f"   ✅ Selector '{selector}' encontró: {text}")
                except:
                    logger.info(f"   ❌ Selector '{selector}' no funcionó")
            
            # DIAGNÓSTICO 5: Click en "Partidos de hoy" y ver cambios
            logger.info("\n📋 DIAGNÓSTICO 5: CLICK EN 'PARTIDOS DE HOY'")
            try:
                today_btn = await self.page.query_selector('div.tabs__ear')
                if today_btn:
                    await today_btn.click()
                    logger.info("   ✅ Click exitoso")
                    await asyncio.sleep(3)
                    
                    matches_after = await self.page.query_selector_all('.event__match')
                    logger.info(f"   Partidos después del click: {len(matches_after)}")
                    
                    for i, match in enumerate(matches_after[:3], 1):
                        participants = await match.query_selector_all('[class*="participant"]')
                        if len(participants) >= 2:
                            home = await participants[0].text_content()
                            away = await participants[1].text_content()
                            logger.info(f"   {i}. {home.strip()} vs {away.strip()}")
            except Exception as e:
                logger.error(f"   ❌ Error en click: {e}")
            
            # ESPERAR PARA INSPECCIONAR MANUALMENTE
            logger.info("\n⏸️  PAUSA: Inspecciona la página manualmente")
            logger.info("   Presiona Ctrl+C cuando termines...")
            await asyncio.sleep(300)  # 5 minutos
            
        except Exception as e:
            logger.error(f"Error: {e}")
    
    async def diagnose_h2h_page(self):
        """Diagnosticar página H2H de un partido"""
        logger.info("\n🔍 DIAGNÓSTICO PÁGINA H2H")
        
        try:
            # URL de ejemplo (cambiar por una real)
            example_url = "https://www.flashscore.co/partido/ejemplo"
            
            logger.info(f"🌐 Navegar a: {example_url}")
            logger.info("   (Cambia la URL en el código)")
            
            await self.page.goto(example_url, timeout=30000)
            await asyncio.sleep(3)
            
            # DIAGNÓSTICO 6: Tabs del partido
            logger.info("\n📋 DIAGNÓSTICO 6: TABS DEL PARTIDO")
            tabs = await self.page.query_selector_all('[class*="tabs"] a')
            for i, tab in enumerate(tabs, 1):
                text = await tab.text_content()
                href = await tab.get_attribute('href')
                logger.info(f"   {i}. Tab: {text.strip()}")
                logger.info(f"      Href: {href}")
            
            # DIAGNÓSTICO 7: Botón "Mostrar más"
            logger.info("\n📋 DIAGNÓSTICO 7: BOTÓN 'MOSTRAR MÁS'")
            show_more_selectors = [
                'a.event__more',
                "a:has-text('Mostrar más')",
                "[class*='more']"
            ]
            
            for selector in show_more_selectors:
                try:
                    btn = await self.page.query_selector(selector)
                    if btn:
                        text = await btn.text_content()
                        visible = await btn.is_visible()
                        logger.info(f"   ✅ '{selector}': {text} (visible={visible})")
                except:
                    logger.info(f"   ❌ '{selector}' no encontrado")
            
            logger.info("\n⏸️  PAUSA: Inspecciona página H2H manualmente")
            await asyncio.sleep(300)
            
        except Exception as e:
            logger.error(f"Error: {e}")
    
    async def close(self):
        """Cerrar"""
        try:
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
        except:
            pass

async def main():
    print("🔍" + "="*60)
    print("🔍 HERRAMIENTA DE DIAGNÓSTICO FLASHSCORE")
    print("🔍" + "="*60)
    
    diag = FlashScoreDiagnostic()
    
    try:
        await diag.init_browser()
        
        # OPCIÓN 1: Diagnosticar página NBA
        await diag.diagnose_nba_page()
        
        # OPCIÓN 2: Diagnosticar página H2H (descomentar y poner URL real)
        # await diag.diagnose_h2h_page()
        
    except KeyboardInterrupt:
        logger.info("\n✅ Diagnóstico finalizado por usuario")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await diag.close()

if __name__ == "__main__":
    asyncio.run(main())