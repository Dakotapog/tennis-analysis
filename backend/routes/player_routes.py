# routes/player_routes.py - Test de conectividad a Flashscore
# 🎯 ENFOQUE: Solo validar acceso a https://www.flashscore.co

import os
import time
from flask import Blueprint, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# Crear Blueprint para las rutas
bp = Blueprint('players', __name__, url_prefix='/api/players')

# Path del chromedriver
CHROMEDRIVER_PATH = os.path.join(os.getcwd(), 'chromedriver')

def setup_test_driver():
    """Configura WebDriver solo para test de conectividad"""
    try:
        if not os.path.exists(CHROMEDRIVER_PATH):
            return None, f"❌ Chromedriver no encontrado en: {CHROMEDRIVER_PATH}"
        
        # Dar permisos de ejecución
        os.chmod(CHROMEDRIVER_PATH, 0o755)
        
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        service = ChromeService(executable_path=CHROMEDRIVER_PATH)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        return driver, "✅ Driver configurado correctamente"
        
    except Exception as e:
        return None, f"❌ Error configurando driver: {str(e)}"

def test_flashscore_access(driver):
    """Test específico para acceder a Flashscore"""
    try:
        print("🌐 Intentando acceder a Flashscore...")
        
        # Ir a la página principal
        driver.get("https://www.flashscore.co")
        
        # Esperar a que la página cargue
        wait = WebDriverWait(driver, 15)
        
        # Verificar que la página cargó correctamente
        try:
            # Buscar elementos típicos de Flashscore
            body_element = wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            page_title = driver.title
            current_url = driver.current_url
            
            # Manejar cookies si aparecen
            try:
                cookie_accept = driver.find_element(By.ID, "onetrust-accept-btn-handler")
                if cookie_accept:
                    cookie_accept.click()
                    time.sleep(2)
                    print("🍪 Cookies aceptadas")
            except:
                print("🍪 No se encontraron cookies o ya fueron manejadas")
            
            # Buscar elementos específicos de Flashscore
            found_elements = {}
            
            # Logo o marca de Flashscore
            try:
                logo = driver.find_element(By.CSS_SELECTOR, "[class*='logo'], [class*='header']")
                found_elements['logo'] = "✅ Logo encontrado"
            except:
                found_elements['logo'] = "❌ Logo no encontrado"
            
            # Menú de deportes
            try:
                sports_menu = driver.find_element(By.CSS_SELECTOR, "[class*='menu'], [class*='nav'], a[href*='tennis']")
                found_elements['menu'] = "✅ Menú encontrado"
            except:
                found_elements['menu'] = "❌ Menú no encontrado"
            
            # Cualquier contenido de partidos o deportes
            try:
                matches = driver.find_elements(By.CSS_SELECTOR, "[class*='match'], [class*='event'], [class*='fixture']")
                found_elements['content'] = f"✅ {len(matches)} elementos de contenido encontrados"
            except:
                found_elements['content'] = "❌ Contenido no encontrado"
            
            return {
                'success': True,
                'title': page_title,
                'url': current_url,
                'elements_found': found_elements,
                'message': "✅ Acceso a Flashscore exitoso"
            }
            
        except TimeoutException:
            return {
                'success': False,
                'error': 'Timeout esperando que cargue la página',
                'message': "❌ La página tardó demasiado en cargar"
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f"❌ Error accediendo a Flashscore: {str(e)}"
        }

@bp.route('/test/connectivity', methods=['POST'])
def test_connectivity():
    """
    POST /api/players/test/connectivity
    Test de conectividad a Flashscore
    """
    try:
        print("🚀 Iniciando test de conectividad a Flashscore...")
        
        # Configurar driver
        driver, setup_message = setup_test_driver()
        
        if not driver:
            return jsonify({
                'success': False,
                'error': setup_message,
                'message': '❌ No se pudo configurar el driver'
            }), 500
        
        # Test de acceso a Flashscore
        result = test_flashscore_access(driver)
        
        # Cerrar driver
        try:
            driver.quit()
            print("🧹 Driver cerrado correctamente")
        except Exception as e:
            print(f"⚠️ Error cerrando driver: {e}")
        
        # Preparar respuesta
        if result['success']:
            return jsonify({
                'success': True,
                'status': 'connected',
                'website': 'https://www.flashscore.co',
                'title': result.get('title', 'N/A'),
                'current_url': result.get('url', 'N/A'),
                'elements_found': result.get('elements_found', {}),
                'message': result['message'],
                'next_steps': [
                    '✅ Conectividad exitosa',
                    '🎾 Puedes proceder con el scraping de tenis',
                    '📊 Siguiente: implementar extracción de rankings ATP'
                ]
            }), 200
        else:
            return jsonify({
                'success': False,
                'status': 'connection_failed',
                'website': 'https://www.flashscore.co',
                'error': result.get('error', 'Error desconocido'),
                'message': result['message'],
                'troubleshooting': [
                    '🔍 Verificar conexión a internet',
                    '🚫 Revisar si Flashscore está bloqueando el acceso',
                    '⏱️ Intentar nuevamente en unos minutos',
                    '🔧 Verificar configuración del WebDriver'
                ]
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'test_error',
            'error': str(e),
            'message': f'❌ Error ejecutando test de conectividad: {str(e)}'
        }), 500

@bp.route('/test/connectivity/info', methods=['GET'])
def connectivity_info():
    """
    GET /api/players/test/connectivity/info
    Información sobre el test de conectividad
    """
    return jsonify({
        'test_name': 'Flashscore Connectivity Test',
        'target_website': 'https://www.flashscore.co',
        'purpose': 'Validar acceso a Flashscore antes de implementar scraping completo',
        'method': 'POST /api/players/test/connectivity',
        'requirements': [
            '🔧 Chromedriver en la carpeta del proyecto',
            '🌐 Conexión a internet activa',
            '📦 Selenium instalado (pip install selenium)'
        ],
        'what_it_tests': [
            '✅ Acceso básico a la página web',
            '🍪 Manejo de cookies',
            '🔍 Detección de elementos básicos',
            '⏱️ Tiempo de respuesta de la página'
        ],
        'success_indicators': [
            'Página carga correctamente',
            'Título de página obtenido',
            'Elementos de navegación encontrados',
            'Contenido de deportes detectado'
        ],
        'usage': {
            'curl': 'curl -X POST http://localhost:5000/api/players/test/connectivity',
            'python': 'requests.post("http://localhost:5000/api/players/test/connectivity")',
            'browser': 'Usar herramienta como Postman o Thunder Client'
        }
    })

@bp.route('/health', methods=['GET'])
def health_check():
    """
    GET /api/players/health
    Health check básico del servicio
    """
    return jsonify({
        'status': 'healthy',
        'service': 'tennis_players_api',
        'chromedriver_available': os.path.exists(CHROMEDRIVER_PATH),
        'chromedriver_path': CHROMEDRIVER_PATH,
        'available_tests': [
            'POST /api/players/test/connectivity - Test de acceso a Flashscore',
            'GET /api/players/test/connectivity/info - Info del test'
        ],
        'message': '🎾 Servicio operativo - listo para test de conectividad'
    })

# Ruta adicional para diagnóstico del sistema
@bp.route('/system/check', methods=['GET'])
def system_check():
    """Diagnóstico del sistema y dependencias"""
    try:
        # Verificar dependencias
        dependencies = {}
        
        # Selenium
        try:
            import selenium
            dependencies['selenium'] = f"✅ v{selenium.__version__}"
        except ImportError:
            dependencies['selenium'] = "❌ No instalado"
        
        # ChromeDriver
        chromedriver_status = "✅ Encontrado" if os.path.exists(CHROMEDRIVER_PATH) else "❌ No encontrado"
        dependencies['chromedriver'] = chromedriver_status
        
        # Permisos
        try:
            if os.path.exists(CHROMEDRIVER_PATH):
                permissions = oct(os.stat(CHROMEDRIVER_PATH).st_mode)[-3:]
                dependencies['chromedriver_permissions'] = f"✅ {permissions}"
            else:
                dependencies['chromedriver_permissions'] = "❌ Archivo no existe"
        except:
            dependencies['chromedriver_permissions'] = "❌ Error verificando permisos"
        
        return jsonify({
            'system_status': 'ready' if all('✅' in v for v in dependencies.values()) else 'needs_attention',
            'dependencies': dependencies,
            'next_step': 'POST /api/players/test/connectivity para probar acceso a Flashscore',
            'chromedriver_path': CHROMEDRIVER_PATH
        })
        
    except Exception as e:
        return jsonify({
            'system_status': 'error',
            'error': str(e),
            'message': 'Error verificando sistema'
        }), 500