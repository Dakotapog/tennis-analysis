# routes/players.py - Rutas de API para manejar jugadores
# 🔴 ARCHIVO CRÍTICO - API endpoints para jugadores

from flask import Blueprint, jsonify, request
from models.database import get_all_players, get_player_by_name, insert_player
from services.scraper_service import get_scraper, quick_scrape_top_10

# 🏗️ CREAR BLUEPRINT
# Blueprint = módulo de rutas en Flask (equivalente a Router en Express.js)
players_bp = Blueprint('players', __name__)

# 📊 RUTA: OBTENER TODOS LOS JUGADORES
@players_bp.route('/players', methods=['GET'])
def get_players():
    """
    GET /api/players
    Obtener lista de todos los jugadores
    Equivalente a: app.get('/api/players', (req, res) => {...}) en Node.js
    """
    try:
        # Obtener parámetros de consulta
        limit = request.args.get('limit', type=int)
        country = request.args.get('country', type=str)
        
        # Obtener jugadores de la base de datos
        players = get_all_players()
        
        # Filtrar por país si se especifica
        if country:
            players = [p for p in players if p['country'] and p['country'].lower() == country.lower()]
        
        # Limitar resultados si se especifica
        if limit and limit > 0:
            players = players[:limit]
        
        # Preparar respuesta
        response = {
            "success": True,
            "count": len(players),
            "players": players,
            "message": f"✅ {len(players)} jugadores encontrados"
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "❌ Error obteniendo jugadores"
        }), 500

# 🔍 RUTA: BUSCAR JUGADOR POR NOMBRE
@players_bp.route('/players/<string:player_name>', methods=['GET'])
def get_player(player_name):
    """
    GET /api/players/Carlos-Alcaraz
    Obtener información de un jugador específico
    """
    try:
        # Limpiar nombre del jugador
        clean_name = player_name.replace('-', ' ').replace('_', ' ')
        
        # Buscar jugador en la base de datos
        player = get_player_by_name(clean_name)
        
        if player:
            return jsonify({
                "success": True,
                "player": player,
                "message": f"✅ Jugador {clean_name} encontrado"
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": f"❌ Jugador {clean_name} no encontrado"
            }), 404
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "❌ Error buscando jugador"
        }), 500

# 🎾 RUTA: SCRAPING DEL TOP ATP
@players_bp.route('/players/scrape/atp-top', methods=['POST'])
def scrape_atp_top():
    """
    POST /api/players/scrape/atp-top
    Ejecutar scraping del top ATP y guardar en base de datos
    """
    try:
        # Obtener parámetros del cuerpo de la petición
        data = request.get_json() or {}
        limit = data.get('limit', 10)  # Por defecto top 10
        
        # Validar límite
        if limit < 1 or limit > 100:
            return jsonify({
                "success": False,
                "message": "❌ Límite debe estar entre 1 y 100"
            }), 400
        
        print(f"🚀 Iniciando scraping del top {limit} ATP...")
        
        # Crear scraper y ejecutar
        scraper = get_scraper()
        result = scraper.scrape_and_save_atp_top(limit)
        
        if result:
            # Obtener jugadores actualizados
            players = get_all_players()
            
            return jsonify({
                "success": True,
                "message": f"✅ Scraping completado exitosamente",
                "scraped_count": limit,
                "total_players": len(players),
                "players": players[:limit]  # Mostrar los primeros
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": "❌ Error en el scraping"
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "❌ Error ejecutando scraping"
        }), 500

# 🏃 RUTA: SCRAPING RÁPIDO TOP 10
@players_bp.route('/players/scrape/quick', methods=['POST'])
def quick_scrape():
    """
    POST /api/players/scrape/quick
    Scraping rápido del top 10 ATP
    """
    try:
        print("⚡ Iniciando scraping rápido del top 10...")
        
        # Ejecutar scraping rápido
        result = quick_scrape_top_10()
        
        if result:
            players = get_all_players()
            return jsonify({
                "success": True,
                "message": "⚡ Scraping rápido completado",
                "players_count": len(players),
                "top_players": players[:10]
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": "❌ Error en scraping rápido"
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "❌ Error en scraping rápido"
        }), 500

# ➕ RUTA: AGREGAR JUGADOR MANUALMENTE
@players_bp.route('/players', methods=['POST'])
def add_player():
    """
    POST /api/players
    Agregar jugador manualmente
    Body: {"name": "Rafael Nadal", "ranking": 2, "country": "Spain", "age": 37}
    """
    try:
        # Obtener datos del cuerpo de la petición
        data = request.get_json()
        
        if not data or 'name' not in data:
            return jsonify({
                "success": False,
                "message": "❌ Nombre del jugador es requerido"
            }), 400
        
        # Extraer datos
        name = data['name']
        ranking = data.get('ranking')
        country = data.get('country')
        age = data.get('age')
        
        # Insertar jugador
        player_id = insert_player(name, ranking, country, age)
        
        if player_id:
            # Obtener jugador insertado
            player = get_player_by_name(name)
            
            return jsonify({
                "success": True,
                "message": f"✅ Jugador {name} agregado exitosamente",
                "player": player
            }), 201
        else:
            return jsonify({
                "success": False,
                "message": f"❌ Error agregando jugador {name}"
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "❌ Error agregando jugador"
        }), 500

# 📊 RUTA: ESTADÍSTICAS DE JUGADORES
@players_bp.route('/players/stats', methods=['GET'])
def get_players_stats():
    """
    GET /api/players/stats
    Obtener estadísticas generales de jugadores
    """
    try:
        players = get_all_players()
        
        # Calcular estadísticas
        total_players = len(players)
        countries = list(set([p['country'] for p in players if p['country']]))
        avg_age = sum([p['age'] for p in players if p['age']]) / len([p for p in players if p['age']]) if players else 0
        
        # Top 10 por ranking
        top_10 = [p for p in players if p['ranking'] and p['ranking'] <= 10]
        
        stats = {
            "total_players": total_players,
            "total_countries": len(countries),
            "countries": countries,
            "average_age": round(avg_age, 1),
            "top_10_count": len(top_10),
            "top_10_players": top_10
        }
        
        return jsonify({
            "success": True,
            "stats": stats,
            "message": "✅ Estadísticas calculadas correctamente"
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "❌ Error calculando estadísticas"
        }), 500

# 🔄 RUTA: ACTUALIZAR JUGADOR
@players_bp.route('/players/<int:player_id>', methods=['PUT'])
def update_player(player_id):
    """
    PUT /api/players/1
    Actualizar información de un jugador
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "message": "❌ Datos no proporcionados"
            }), 400
        
        # Aquí implementarías la lógica de actualización
        # Por ahora, solo devolvemos un mensaje
        return jsonify({
            "success": True,
            "message": f"✅ Jugador {player_id} actualizado (función en desarrollo)"
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "❌ Error actualizando jugador"
        }), 500

# 🗑️ RUTA: ELIMINAR JUGADOR
@players_bp.route('/players/<int:player_id>', methods=['DELETE'])
def delete_player(player_id):
    """
    DELETE /api/players/1
    Eliminar un jugador
    """
    try:
        # Implementar lógica de eliminación
        return jsonify({
            "success": True,
            "message": f"✅ Jugador {player_id} eliminado (función en desarrollo)"
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "❌ Error eliminando jugador"
        }), 500

# 📝 DOCUMENTACIÓN DE RUTAS
"""
📋 RUTAS DISPONIBLES:

GET    /api/players              - Obtener todos los jugadores
GET    /api/players/stats        - Estadísticas generales
GET    /api/players/<name>       - Buscar jugador por nombre
POST   /api/players              - Agregar jugador manualmente
POST   /api/players/scrape/atp-top - Scraping del top ATP
POST   /api/players/scrape/quick - Scraping rápido top 10
PUT    /api/players/<id>         - Actualizar jugador
DELETE /api/players/<id>         - Eliminar jugador

🔧 EJEMPLOS DE USO:

# Obtener todos los jugadores
curl http://localhost:5000/api/players

# Obtener top 5 jugadores
curl http://localhost:5000/api/players?limit=5

# Buscar jugador específico
curl http://localhost:5000/api/players/Carlos-Alcaraz

# Scraping rápido
curl -X POST http://localhost:5000/api/players/scrape/quick

# Scraping personalizado
curl -X POST http://localhost:5000/api/players/scrape/atp-top \
  -H "Content-Type: application/json" \
  -d '{"limit": 20}'
"""