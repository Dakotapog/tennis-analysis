# app.py - Punto de entrada de la aplicación Flask - VERSIÓN CORREGIDA Y OPTIMIZADA

import os
import logging
import sys
from flask import Flask, jsonify
from flask_cors import CORS

# --- INICIO: CORRECCIÓN DE RUTAS DE IMPORTACIÓN ---
# Añadir el directorio raíz del proyecto a sys.path para resolver importaciones.
# Esto asegura que módulos como 'services' y 'models' se encuentren siempre.
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- FIN: CORRECCIÓN DE RUTAS DE IMPORTACIÓN ---

# Configurar logging básico
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 1. CREAR Y CONFIGURAR LA APLICACIÓN FLASK
app = Flask(__name__)
CORS(app)  # Habilita CORS para permitir peticiones desde un frontend

# 2. VARIABLES DE ESTADO PARA DIAGNÓSTICO
ROUTES_IMPORTED = False
DATABASE_INITIALIZED = False
DATABASE_PATH = "tennis_analysis.db"

# 3. IMPORTAR Y REGISTRAR RUTAS DE FORMA SEGURA
try:
    # CORRECCIÓN PRINCIPAL: Importar desde routes.players (no desde scraping_routes)
    from routes.players import players_bp
    
    # Registrar el blueprint con prefijo /api
    app.register_blueprint(players_bp, url_prefix='/api')
    ROUTES_IMPORTED = True
    logger.info("✅ Rutas importadas correctamente desde routes.players")
    logger.info("✅ Blueprint registrado: /api/players/*")
    
except ImportError as e:
    logger.error(f"❌ Error importando rutas: {e}")
    logger.warning("⚠️ La aplicación funcionará en modo básico sin rutas específicas")
    
    # Crear un blueprint temporal de emergencia
    from flask import Blueprint
    emergency_bp = Blueprint('emergency', __name__, url_prefix='/api')
    
    @emergency_bp.route('/status')
    def emergency_status():
        return jsonify({
            "status": "emergency_mode",
            "message": "Rutas principales no disponibles",
            "error": str(e),
            "solution": "Verificar que existe routes/players.py con players_bp"
        })
    
    app.register_blueprint(emergency_bp)
    logger.info("⚠️ Rutas de emergencia activadas: /api/status")

# 4. IMPORTAR E INICIALIZAR BASE DE DATOS
try:
    from models.database import init_db, DATABASE_PATH as DB_PATH
    DATABASE_PATH = DB_PATH
    logger.info("✅ Módulo de base de datos importado correctamente")
    
except ImportError as e:
    logger.error(f"❌ Error importando models.database: {e}")
    logger.warning("⚠️ Usando configuración de base de datos básica")
    
    # Función init_db de emergencia
    def init_db():
        """Función básica de inicialización de BD si no existe el módulo."""
        try:
            # Crear archivo de BD si no existe
            if not os.path.exists(DATABASE_PATH):
                open(DATABASE_PATH, 'a').close()
                logger.info(f"✅ Archivo de BD básico creado: {DATABASE_PATH}")
            else:
                logger.info(f"✅ Archivo de BD ya existe: {DATABASE_PATH}")
        except Exception as ex:
            logger.error(f"❌ Error creando archivo de BD: {ex}")

# 5. RUTA RAÍZ CON INFORMACIÓN COMPLETA
@app.route('/')
def index():
    """Ruta de bienvenida con información del sistema."""
    
    # Determinar endpoints disponibles dinámicamente
    available_endpoints = [
        "GET /",
        "GET /health"
    ]
    
    if ROUTES_IMPORTED:
        available_endpoints.extend([
            "GET /api/players/health",
            "POST /api/players/test/connectivity",
            "GET /api/players/test/connectivity/info", 
            "GET /api/players/system/check"
        ])
    else:
        available_endpoints.append("GET /api/status")
    
    return jsonify({
        "message": "🎾 ¡Bienvenido a la API de Tennis Analysis!",
        "status": "online",
        "version": "2.0-corrected",
        "routes_status": "imported" if ROUTES_IMPORTED else "emergency_mode",
        "database_status": "initialized" if DATABASE_INITIALIZED else "basic",
        "database_path": DATABASE_PATH,
        "database_exists": os.path.exists(DATABASE_PATH),
        "available_endpoints": available_endpoints,
        "documentation": "Accede a /health para más información del sistema"
    })

# 6. HEALTH CHECK COMPLETO
@app.route('/health')
def health_check():
    """Health check completo del sistema."""
    
    db_exists = os.path.exists(DATABASE_PATH)
    db_size = os.path.getsize(DATABASE_PATH) if db_exists else 0
    
    return jsonify({
        "status": "healthy",
        "service": "tennis_analysis_api",
        "version": "2.0-corrected",
        "components": {
            "flask_app": "running",
            "cors": "enabled", 
            "routes": "imported" if ROUTES_IMPORTED else "emergency_mode",
            "database": {
                "initialized": DATABASE_INITIALIZED,
                "file_exists": db_exists,
                "path": DATABASE_PATH,
                "size_bytes": db_size
            }
        },
        "message": "🎾 API operativa y saludable",
        "timestamp": "Sistema funcionando correctamente"
    })

# 7. INICIALIZAR BASE DE DATOS AL ARRANCAR LA APLICACIÓN
def initialize_database():
    """Inicializar la base de datos de forma segura."""
    global DATABASE_INITIALIZED
    
    try:
        with app.app_context():
            init_db()
            DATABASE_INITIALIZED = True
            logger.info(f"✅ Base de datos inicializada correctamente: {DATABASE_PATH}")
            
            # Verificar que el archivo fue creado
            if os.path.exists(DATABASE_PATH):
                size = os.path.getsize(DATABASE_PATH)
                logger.info(f"✅ Archivo de BD verificado - Tamaño: {size} bytes")
            else:
                logger.warning("⚠️ El archivo de BD no se encontró después de la inicialización")
                
    except Exception as e:
        logger.error(f"❌ Error inicializando base de datos: {e}")
        logger.warning("⚠️ La aplicación funcionará sin base de datos")

# 8. FUNCIÓN PRINCIPAL
def main():
    """Función principal para iniciar la aplicación."""
    logger.info("\n" + "="*60)
    logger.info("🚀 INICIANDO TENNIS ANALYSIS API")
    logger.info("="*60)
    logger.info("📍 Servidor: http://localhost:5000")
    logger.info(f"🔧 Modo: {'Completo' if ROUTES_IMPORTED else 'Básico/Emergencia'}")
    logger.info(f"💾 Base de datos: {DATABASE_PATH}")
    
    # Inicializar base de datos
    initialize_database()
    
    logger.info("\n📋 ENDPOINTS DISPONIBLES:")
    logger.info("   - GET  /            (Información de bienvenida)")
    logger.info("   - GET  /health      (Estado del sistema)")
    
    if ROUTES_IMPORTED:
        logger.info("   - GET  /api/players/health")
        logger.info("   - POST /api/players/test/connectivity")
        logger.info("   - GET  /api/players/test/connectivity/info")
        logger.info("   - GET  /api/players/system/check")
    else:
        logger.info("   - GET  /api/status  (Modo de emergencia)")
    
    logger.info("="*60 + "\n")
    
    # Iniciar servidor Flask
    app.run(host='0.0.0.0', port=5000, debug=True)

# 9. BLOQUE DE EJECUCIÓN
if __name__ == '__main__':
    main()
