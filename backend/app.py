# app.py - Punto de entrada de la aplicación Flask
# 🔴 ARCHIVO CRÍTICO - Se ejecuta primero

from flask import Flask, jsonify
from flask_cors import CORS
import os
import sqlite3

# 📦 IMPORTACIONES DE NUESTROS MÓDULOS
# (Se crearán en los próximos pasos)
try:
    from models.database import init_db, get_db
    from routes.players import players_bp
    from services.scraper_service import scraper
except ImportError as e:
    print(f"⚠️ Módulo no encontrado: {e}")
    print("📝 Crearemos estos archivos en los próximos pasos")

# 🏗️ CREAR APLICACIÓN FLASK
app = Flask(__name__)

# 🌐 CONFIGURAR CORS (permite conexiones desde React)
CORS(app)

# 🔧 CONFIGURACIÓN DE LA APLICACIÓN
app.config['SECRET_KEY'] = 'tennis-analysis-secret-key-2024'
app.config['DATABASE'] = 'database.db'

# 📊 RUTA DE PRUEBA BÁSICA
@app.route('/')
def home():
    """
    Ruta principal - Verificar que la aplicación funciona
    Equivalente a: app.get('/', (req, res) => res.json(...)) en Node.js
    """
    return jsonify({
        "message": "🎾 Tennis Analysis API está funcionando!",
        "version": "1.0.0",
        "status": "active"
    })

# 📊 RUTA DE ESTADO DE LA BASE DE DATOS
@app.route('/api/status')
def api_status():
    """
    Verificar que la base de datos está funcionando
    """
    try:
        # Intentar conectar a la base de datos
        conn = sqlite3.connect(app.config['DATABASE'])
        cursor = conn.cursor()
        
        # Verificar si las tablas existen
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        conn.close()
        
        return jsonify({
            "database": "connected",
            "tables": [table[0] for table in tables],
            "chrome_path": "/usr/bin/google-chrome"
        })
    except Exception as e:
        return jsonify({
            "database": "error",
            "error": str(e)
        }), 500

# 🔄 INICIALIZAR BASE DE DATOS
def create_app():
    """
    Función de inicialización de la aplicación
    Se ejecuta cuando se inicia el servidor
    """
    try:
        # Crear base de datos si no existe
        init_db()
        print("✅ Base de datos inicializada correctamente")
        
        # Registrar rutas (blueprints)
        app.register_blueprint(players_bp, url_prefix='/api')
        print("✅ Rutas registradas correctamente")
        
    except Exception as e:
        print(f"⚠️ Error en inicialización: {e}")
        print("📝 Algunos módulos aún no están creados")

# 🚀 EJECUTAR APLICACIÓN
if __name__ == '__main__':
    print("🎾 Iniciando Tennis Analysis API...")
    print("📍 Ruta del proyecto:", os.getcwd())
    print("🌐 Chrome Ubuntu:", "/usr/bin/google-chrome")
    
    # Inicializar aplicación
    create_app()
    
    # Iniciar servidor Flask
    app.run(
        host='0.0.0.0',    # Permitir conexiones externas
        port=5000,         # Puerto del servidor
        debug=True         # Modo desarrollo (recarga automática)
    )

# 📝 NOTAS IMPORTANTES:
# 1. Este archivo se ejecuta con: python app.py
# 2. La API estará disponible en: http://localhost:5000
# 3. Las rutas principales serán:
#    - GET /              → Página de inicio
#    - GET /api/status    → Estado de la aplicación
#    - GET /api/players   → Lista de jugadores (próximo archivo)
# 4. Los imports que fallan son normales, los crearemos después