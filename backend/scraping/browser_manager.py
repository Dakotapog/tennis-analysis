"""
🌐 Browser Manager - Gestión del navegador Playwright
Maneja configuración, setup y cleanup del navegador para WSL
"""

import logging
import psutil
import os
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)


class BrowserManager:
    """Gestor del navegador Playwright optimizado para WSL"""
    
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.is_wsl = self._check_wsl()
    
    def _check_wsl(self):
        """Verificar si estamos en WSL"""
        if os.path.exists('/proc/version'):
            try:
                with open('/proc/version', 'r') as f:
                    version = f.read()
                    if 'Microsoft' in version or 'WSL' in version:
                        logger.info("✅ Ejecutándose en WSL - Aplicando optimizaciones")
                        return True
            except:
                pass
        return False
    
    def check_system_resources(self):
        """🔍 Verificar recursos del sistema"""
        logger.info("🔍 Verificando recursos del sistema...")
        
        # Verificar memoria
        memory = psutil.virtual_memory()
        logger.info(f"💾 Memoria disponible: {memory.available / (1024**3):.1f}GB")
        
        if memory.available < 1024 * 1024 * 1024:  # 1GB
            logger.warning("⚠️ Poca memoria disponible (<1GB)")
        
        # Limpiar procesos Chrome zombies
        killed = self._kill_zombie_chrome_processes()
        if killed > 0:
            logger.info(f"🧹 Eliminados {killed} procesos Chrome/Chromium zombies")
    
    def _kill_zombie_chrome_processes(self):
        """Limpiar procesos Chrome/Chromium zombies"""
        killed = 0
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] and any(
                    chrome in proc.info['name'].lower() 
                    for chrome in ['chrome', 'chromium']
                ):
                    proc.kill()
                    killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return killed
    
    async def setup(self):
        """🚀 Configurar navegador optimizado para WSL"""
        logger.info("🚀 Configurando navegador...")
        
        try:
            self.playwright = await async_playwright().start()
            
            # Configuración para WSL
            browser_args = [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--disable-software-rasterizer',
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding',
                '--disable-features=VizDisplayCompositor,TranslateUI',
                '--disable-ipc-flooding-protection',
                '--disable-blink-features=AutomationControlled',
                '--no-first-run',
                '--password-store=basic',
                '--use-mock-keychain'
            ]
            
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=browser_args,
                timeout=60000,
                slow_mo=250,
            )
            
            self.context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
                ignore_https_errors=True,
                java_script_enabled=True,
            )
            
            # Timeouts generosos para WSL
            self.context.set_default_timeout(60000)
            self.context.set_default_navigation_timeout(60000)
            
            self.page = await self.context.new_page()
            
            # Configurar eventos de error
            self.page.on("crash", lambda: logger.error("💥 Página crasheada"))
            self.page.on("pageerror", lambda err: logger.error(f"🚨 Error JS: {err}"))
            
            await self.page.set_extra_http_headers({
                'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            })
            
            logger.info("✅ Navegador configurado exitosamente")
            return self.page
            
        except Exception as e:
            logger.error(f"❌ Error configurando navegador: {e}")
            await self.cleanup()
            raise
    
    async def cleanup(self):
        """🧹 Limpieza de recursos del navegador"""
        logger.info("🧹 Limpiando recursos del navegador...")
        
        try:
            if self.page:
                await self.page.close()
                logger.info("   ✅ Página cerrada")
            
            if self.context:
                await self.context.close()
                logger.info("   ✅ Contexto cerrado")
            
            if self.browser:
                await self.browser.close()
                logger.info("   ✅ Navegador cerrado")
            
            if self.playwright:
                await self.playwright.stop()
                logger.info("   ✅ Playwright detenido")
        
        except Exception as e:
            logger.warning(f"⚠️ Error durante limpieza: {e}")
        
        logger.info("✅ Limpieza completada")