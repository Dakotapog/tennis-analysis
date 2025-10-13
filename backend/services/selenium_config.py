# services/selenium_config.py # 🔴 ESTE ARCHIVO ES NUEVO Y CRÍTICO
# Centraliza toda la configuración de Selenium para WSL2.

import os
import logging
import traceback
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Configurar logging para este módulo
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_configured_driver():
    """
    Configura y devuelve una instancia de Chrome WebDriver optimizada para WSL2.
    Esta función es el corazón de la solución.
    """
    logger.info("🔧 Iniciando configuración avanzada de WebDriver para WSL2...")
    driver = None
    try:
        chrome_options = Options()
        # --- Opciones CRÍTICAS para WSL y Docker ---
        chrome_options.add_argument("--headless") # Esencial, no puede abrir una ventana gráfica
        chrome_options.add_argument("--no-sandbox") # Permite ejecutar como root/en un contenedor
        chrome_options.add_argument("--disable-dev-shm-usage") # Evita problemas de memoria compartida
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        logger.info("Intentando configurar con webdriver-manager...")
        
        # 1. Instala o actualiza chromedriver y obtén la ruta del ejecutable
        try:
            driver_path = ChromeDriverManager().install()
            logger.info(f"WebDriver Manager instaló/encontró el driver en: {driver_path}")
        except Exception as e:
            logger.error(f"Falló la descarga/instalación con WebDriver Manager: {e}")
            return None

        # 2. Asegúrate de que la ruta sea al ejecutable correcto
        if not os.path.basename(driver_path) == 'chromedriver':
            logger.error(f"¡Ruta incorrecta de WebDriver Manager! Apunta a '{os.path.basename(driver_path)}' en lugar de 'chromedriver'.")
            # Intenta corregir la ruta si es un error común
            corrected_path = os.path.join(os.path.dirname(driver_path), 'chromedriver')
            if os.path.exists(corrected_path):
                logger.info(f"Se encontró el ejecutable en la misma carpeta. Usando ruta corregida: {corrected_path}")
                driver_path = corrected_path
            else:
                logger.error("No se pudo encontrar el ejecutable 'chromedriver' en el directorio. La caché puede estar corrupta.")
                return None

        # 3. Pasa la ruta explícitamente al servicio
        service = ChromeService(executable_path=driver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Prueba rápida para ver si funciona
        driver.get("https://www.google.com")
        logger.info(f"✅ WebDriver configurado exitosamente. Título de Google: {driver.title}")
        
        return driver
    
    except Exception as e:
        logger.error(f"❌ Falló la configuración de WebDriver. Error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        if driver:
            driver.quit()
        return None

def test_driver_configuration():
    """
    Una función de prueba simple para verificar si el driver se puede crear.
    """
    print("🧪 Ejecutando prueba de configuración del driver...")
    driver = get_configured_driver()
    if driver:
        print("✅ ¡ÉXITO! La configuración del driver de Selenium funciona correctamente.")
        driver.quit()
    else:
        print("❌ ¡FALLO! No se pudo crear el driver. Revisa los logs de error de arriba.")

if __name__ == '__main__':
    test_driver_configuration()