#!/usr/bin/env python3
"""
Flashscore ATP Rankings Inspector - Versión para WSL
Inspecciona la página de rankings ATP de Flashscore.
"""

import asyncio
import logging
import sys
from datetime import datetime
from playwright.async_api import async_playwright, TimeoutError
import signal

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FlashscoreRankingsInspector:
    def __init__(self, url):
        self.browser = None
        self.page = None
        self.playwright = None
        self.url = url
        
    async def setup_browser(self):
        """Configurar navegador con opciones optimizadas para WSL"""
        try:
            logger.info("🚀 Iniciando Playwright...")
            self.playwright = await async_playwright().start()
            
            browser_args = [
                '--no-sandbox',
                '--disable-setuid-sandbox', 
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--single-process'
            ]
            
            logger.info("🔧 Lanzando navegador con configuración WSL...")
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=browser_args,
                slow_mo=50,
                timeout=60000
            )
            
            context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                locale='es-ES'
            )
            
            self.page = await context.new_page()
            self.page.set_default_timeout(30000)
            self.page.set_default_navigation_timeout(45000)
            
            logger.info("✅ Navegador configurado correctamente.")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error configurando navegador: {str(e)}")
            await self.cleanup()
            return False
    
    async def handle_cookie_consent(self):
        """Maneja el diálogo de consentimiento de cookies si aparece."""
        try:
            cookie_button_selector = "#onetrust-accept-btn-handler"
            logger.info("🔍 Buscando el botón de consentimiento de cookies...")
            # Esperar un poco a que aparezca el botón, pero no fallar si no lo hace
            await self.page.wait_for_selector(cookie_button_selector, timeout=5000)
            await self.page.click(cookie_button_selector)
            logger.info("✅ Botón de consentimiento de cookies clickeado.")
            await asyncio.sleep(2) # Dar tiempo para que el banner desaparezca
        except TimeoutError:
            logger.info("🤷 No se encontró el banner de cookies, o ya fue aceptado. Continuando...")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo hacer clic en el botón de cookies: {e}")

    async def navigate_to_url(self):
        """Navegar a la URL especificada y manejar cookies."""
        try:
            logger.info(f"🌐 Navegando a {self.url}...")
            await self.page.goto(self.url, wait_until="domcontentloaded", timeout=45000)
            
            # Manejar el banner de cookies
            await self.handle_cookie_consent()

            await self.page.wait_for_selector('body', timeout=10000)
            logger.info("✅ Página cargada y cookies manejadas.")
            return True
                        
        except Exception as e:
            logger.error(f"❌ Error navegando a la URL: {str(e)}")
            return False
    
    async def inspect_rankings_structure(self):
        """Inspeccionar la estructura, iterando a través de todos los iframes."""
        try:
            logger.info("🔍 Inspeccionando estructura de la página, incluyendo todos los iframes...")
            await asyncio.sleep(5)

            frames = self.page.frames
            logger.info(f"🖼️ Número de frames encontrados: {len(frames)}")

            selectors_to_test = [
                ".ui-table__row",
                "div[class*='ui-table__row']",
                "a[class*='participant__participantName']"
            ]
            
            found_elements = False
            
            # Iterar a través de todos los frames (el principal y los iframes)
            for i, frame in enumerate(frames):
                logger.info(f"--- Inspeccionando Frame #{i} (URL: {frame.url}) ---")
                if "google" in frame.url or "ad" in frame.url:
                    logger.info("    ⏭️ Omitiendo frame de publicidad/seguimiento.")
                    continue

                for selector in selectors_to_test:
                    try:
                        # Usar un timeout corto para cada selector en cada frame
                        elements = await frame.query_selector_all(selector)
                        if elements:
                            logger.info(f"   ✅ ¡ÉXITO! Selector '{selector}' encontró {len(elements)} elementos en el Frame #{i}.")
                            found_elements = True
                            first_element_html = await elements[0].inner_html()
                            logger.info(f"      📝 HTML del primer elemento: {first_element_html[:200]}...")
                            # Romper los bucles una vez que encontramos el contenido
                            break 
                    except Exception as e:
                        logger.info(f"   ❌ Selector '{selector}' no encontrado o causó error en Frame #{i}: {str(e)}")
                
                if found_elements:
                    break

            if not found_elements:
                 logger.error("❌ No se encontró ningún elemento de ranking en ningún frame.")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"final_inspection_{timestamp}.png"
            await self.page.screenshot(path=screenshot_path)
            logger.info(f"📸 Screenshot de la inspección final guardado en: {screenshot_path}")
            
            return found_elements
            
        except Exception as e:
            logger.error(f"❌ Error fatal durante la inspección: {str(e)}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"rankings_screenshot_ERROR_{timestamp}.png"
            await self.page.screenshot(path=screenshot_path)
            logger.info(f"📸 Screenshot de error guardado en: {screenshot_path}")
            return False
    
    async def cleanup(self):
        """Limpiar recursos"""
        try:
            if self.page: await self.page.close()
            if self.browser: await self.browser.close()
            if self.playwright: await self.playwright.stop()
            logger.info("🧹 Recursos limpiados.")
        except Exception as e:
            logger.warning(f"⚠️ Error durante la limpieza: {e}")

async def main():
    """Función principal"""
    url_atp = "https://www.flashscore.co/tenis/rankings/atp/"
    inspector = FlashscoreRankingsInspector(url_atp)
    
    def signal_handler(signum, frame):
        logger.info("🛑 Señal de interrupción recibida, limpiando...")
        asyncio.create_task(inspector.cleanup())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        logger.info("🎾 Iniciando Inspector de Rankings ATP")
        
        if not await inspector.setup_browser(): return
        if not await inspector.navigate_to_url(): return
        if not await inspector.inspect_rankings_structure(): return
        
        logger.info("✅ Inspección de rankings completada exitosamente.")
        
    except Exception as e:
        logger.error(f"❌ Error fatal durante la inspección: {str(e)}")
        
    finally:
        await inspector.cleanup()

if __name__ == "__main__":
    print("🎾" + "="*60)
    print("🎾 FLASHSCORE RANKINGS INSPECTOR - VERSIÓN WSL")
    print("🎾" + "="*60)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Ejecución interrumpida.")
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        sys.exit(1)
