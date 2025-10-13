# models/database.py - Configuración y manejo de base de datos SQLite
# 🔴 ARCHIVO CRÍTICO - Base para todos los datos - INTEGRADO CON SCRAPER

import sqlite3
import logging
from datetime import datetime, timedelta

# 📍 CONFIGURACIÓN DE LA BASE DE DATOS
DATABASE_PATH = 'database.db'

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def insert_h2h_match(player1_name, player2_name, date, score, tournament, surface, winner):
    # Aquí va la lógica para insertar un partido H2H en la base de datos
    pass  # Reemplaza 'pass' con tu código de inserción

def get_db():
    """
    Obtener conexión a la base de datos
    Equivalente a: const db = sqlite3.connect() en Node.js
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row  # Permite acceso por nombre de columna
        return conn
    except sqlite3.Error as e:
        logger.error(f"❌ Error conectando a la base de datos: {e}")
        return None

def init_db():
    """
    Inicializar base de datos y crear tablas
    Se ejecuta al iniciar la aplicación
    """
    logger.info("🔧 Inicializando base de datos...")
    
    conn = get_db()
    if conn is None:
        logger.error("❌ No se pudo conectar a la base de datos")
        return False
    
    try:
        cursor = conn.cursor()
        
        # 🎾 TABLA DE JUGADORES - ACTUALIZADA PARA SCRAPER
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                ranking INTEGER,
                country TEXT,
                age INTEGER,
                matches_played INTEGER DEFAULT 0,
                matches_won INTEGER DEFAULT 0,
                matches_lost INTEGER DEFAULT 0,
                win_percentage REAL DEFAULT 0.0,
                flashscore_url TEXT,
                last_scraped TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 🏆 TABLA DE PARTIDOS - ACTUALIZADA PARA H2H
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player1_id INTEGER,
                player2_id INTEGER,
                player1_name TEXT NOT NULL,
                player2_name TEXT NOT NULL,
                winner_id INTEGER,
                winner_name TEXT,
                score TEXT,
                tournament TEXT,
                date_played TEXT,
                surface TEXT,
                round TEXT,
                duration INTEGER,
                match_type TEXT DEFAULT 'h2h',
                flashscore_match_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (player1_id) REFERENCES players (id),
                FOREIGN KEY (player2_id) REFERENCES players (id),
                FOREIGN KEY (winner_id) REFERENCES players (id),
                UNIQUE(player1_name, player2_name, date_played, tournament)
            )
        ''')
        
        # 🏟️ TABLA DE TORNEOS (para futuro)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tournaments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                location TEXT,
                surface TEXT,
                start_date DATE,
                end_date DATE,
                category TEXT,
                prize_money INTEGER,
                flashscore_tournament_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 📊 TABLA DE ESTADÍSTICAS H2H
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS h2h_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player1_id INTEGER,
                player2_id INTEGER,
                player1_name TEXT NOT NULL,
                player2_name TEXT NOT NULL,
                total_matches INTEGER DEFAULT 0,
                player1_wins INTEGER DEFAULT 0,
                player2_wins INTEGER DEFAULT 0,
                player1_win_rate REAL DEFAULT 0.0,
                player2_win_rate REAL DEFAULT 0.0,
                last_encounter_date TEXT,
                last_encounter_winner TEXT,
                longest_streak_player TEXT,
                longest_streak_count INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (player1_id) REFERENCES players (id),
                FOREIGN KEY (player2_id) REFERENCES players (id),
                UNIQUE(player1_name, player2_name)
            )
        ''')
        
        # 📊 TABLA DE ESTADÍSTICAS POR SUPERFICIE
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS surface_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player1_name TEXT NOT NULL,
                player2_name TEXT NOT NULL,
                surface TEXT NOT NULL,
                total_matches INTEGER DEFAULT 0,
                player1_wins INTEGER DEFAULT 0,
                player2_wins INTEGER DEFAULT 0,
                player1_win_rate REAL DEFAULT 0.0,
                player2_win_rate REAL DEFAULT 0.0,
                advantage_player TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(player1_name, player2_name, surface)
            )
        ''')
        
        # 💾 CONFIRMAR CAMBIOS
        conn.commit()
        logger.info("✅ Tablas creadas correctamente:")
        logger.info("   - players (jugadores)")
        logger.info("   - matches (partidos H2H)")
        logger.info("   - tournaments (torneos)")
        logger.info("   - h2h_stats (estadísticas H2H)")
        logger.info("   - surface_stats (estadísticas por superficie)")
        
        # 📋 MOSTRAR TABLAS EXISTENTES
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        logger.info(f"📊 Tablas en la base de datos: {[table[0] for table in tables]}")
        
        return True
        
    except sqlite3.Error as e:
        logger.error(f"❌ Error creando tablas: {e}")
        return False
    finally:
        conn.close()

def insert_player(name, ranking=None, country=None, age=None, flashscore_url=None, points=None):
    """
    Insertar un jugador en la base de datos (ahora acepta 'points')
    Función requerida por scraper_service.py
    """
    conn = get_db()
    if conn is None:
        return False

    try:
        cursor = conn.cursor()
        # Verificar si la columna 'points' existe en la tabla players
        cursor.execute("PRAGMA table_info(players)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'points' not in columns:
            # Agregar la columna 'points' si no existe
            cursor.execute("ALTER TABLE players ADD COLUMN points INTEGER")
            conn.commit()

        cursor.execute('''
            INSERT OR REPLACE INTO players 
            (name, ranking, country, age, flashscore_url, last_updated, points) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (name, ranking, country, age, flashscore_url, datetime.now(), points))

        conn.commit()
        player_id = cursor.lastrowid
        logger.info(f"✅ Jugador insertado: {name} (ID: {player_id})")
        return player_id

    except sqlite3.Error as e:
        logger.error(f"❌ Error insertando jugador {name}: {e}")
        return False
    finally:
        conn.close()

def get_or_create_player(name, ranking=None, country=None, age=None):
    """
    Obtener jugador existente o crear uno nuevo
    Función CRÍTICA requerida por scraper_service.py
    """
    conn = get_db()
    if conn is None:
        return None
    
    try:
        cursor = conn.cursor()
        
        # Buscar jugador existente
        cursor.execute('SELECT id FROM players WHERE name = ?', (name,))
        row = cursor.fetchone()
        
        if row:
            player_id = row[0]
            logger.debug(f"🔍 Jugador existente encontrado: {name} (ID: {player_id})")
            return player_id
        
        # Si no existe, crear nuevo jugador
        cursor.execute('''
            INSERT INTO players (name, ranking, country, age, created_at, last_updated) 
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (name, ranking, country, age, datetime.now(), datetime.now()))
        
        conn.commit()
        player_id = cursor.lastrowid
        logger.info(f"✅ Nuevo jugador creado: {name} (ID: {player_id})")
        return player_id
        
    except sqlite3.Error as e:
        logger.error(f"❌ Error obteniendo/creando jugador {name}: {e}")
        return None
    finally:
        conn.close()

def insert_match(player1_id, player2_id, tournament=None, date=None, score=None, 
                surface=None, winner=None, round_info=None, player1_name=None, player2_name=None):
    """
    Insertar un partido en la base de datos
    Función CRÍTICA requerida por scraper_service.py
    """
    conn = get_db()
    if conn is None:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Determinar winner_id si se proporciona winner name
        winner_id = None
        if winner:
            if winner == player1_name:
                winner_id = player1_id
            elif winner == player2_name:
                winner_id = player2_id
            else:
                # Buscar por nombre
                cursor.execute('SELECT id FROM players WHERE name = ?', (winner,))
                row = cursor.fetchone()
                if row:
                    winner_id = row[0]
        
        # Insertar partido (usar INSERT OR IGNORE para evitar duplicados)
        cursor.execute('''
            INSERT OR IGNORE INTO matches 
            (player1_id, player2_id, player1_name, player2_name, winner_id, winner_name,
             score, tournament, date_played, surface, round, created_at) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (player1_id, player2_id, player1_name, player2_name, winner_id, winner,
              score, tournament, date, surface, round_info, datetime.now()))
        
        if cursor.rowcount > 0:
            conn.commit()
            match_id = cursor.lastrowid
            logger.debug(f"✅ Partido insertado: {player1_name} vs {player2_name} (ID: {match_id})")
            return match_id
        else:
            logger.debug(f"ℹ️ Partido ya existe: {player1_name} vs {player2_name}")
            return True
        
    except sqlite3.Error as e:
        logger.error(f"❌ Error insertando partido: {e}")
        return False
    finally:
        conn.close()

def get_h2h_matches(player1_name, player2_name):
    """
    Obtener historial H2H entre dos jugadores
    Función CRÍTICA requerida por scraper_service.py
    """
    conn = get_db()
    if conn is None:
        return []
    
    try:
        cursor = conn.cursor()
        
        # Buscar partidos H2H (en ambas direcciones)
        cursor.execute('''
            SELECT player1_name, player2_name, winner_name, score, tournament, 
                   date_played, surface, round, created_at
            FROM matches 
            WHERE (player1_name = ? AND player2_name = ?) 
               OR (player1_name = ? AND player2_name = ?)
            ORDER BY date_played DESC
        ''', (player1_name, player2_name, player2_name, player1_name))
        
        matches = []
        for row in cursor.fetchall():
            match_data = {
                'player1': row[0],
                'player2': row[1],
                'winner': row[2],
                'score': row[3],
                'tournament': row[4],
                'date': row[5],
                'surface': row[6],
                'round': row[7],
                'created_at': row[8]
            }
            matches.append(match_data)
        
        logger.info(f"📊 Encontrados {len(matches)} partidos H2H para {player1_name} vs {player2_name}")
        return matches
        
    except sqlite3.Error as e:
        logger.error(f"❌ Error obteniendo H2H: {e}")
        return []
    finally:
        conn.close()

def get_all_players():
    """
    Obtener todos los jugadores de la base de datos
    Retorna lista de diccionarios con información de jugadores
    """
    conn = get_db()
    if conn is None:
        return []
    
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, name, ranking, country, age, 
                   matches_played, matches_won, matches_lost, 
                   win_percentage, flashscore_url, last_updated 
            FROM players 
            ORDER BY name ASC
        ''')
        
        players = []
        for row in cursor.fetchall():
            players.append({
                'id': row[0],
                'name': row[1],
                'ranking': row[2],
                'country': row[3],
                'age': row[4],
                'matches_played': row[5],
                'matches_won': row[6],
                'matches_lost': row[7],
                'win_percentage': row[8],
                'flashscore_url': row[9],
                'last_updated': row[10]
            })
        
        return players
        
    except sqlite3.Error as e:
        logger.error(f"❌ Error obteniendo jugadores: {e}")
        return []
    finally:
        conn.close()

def get_player_by_name(name):
    """
    Buscar jugador por nombre
    """
    conn = get_db()
    if conn is None:
        return None
    
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM players WHERE name = ?', (name,))
        row = cursor.fetchone()
        
        if row:
            return {
                'id': row[0],
                'name': row[1],
                'ranking': row[2],
                'country': row[3],
                'age': row[4],
                'matches_played': row[5],
                'matches_won': row[6],
                'matches_lost': row[7],
                'win_percentage': row[8],
                'flashscore_url': row[9],
                'last_updated': row[11]
            }
        return None
        
    except sqlite3.Error as e:
        logger.error(f"❌ Error buscando jugador {name}: {e}")
        return None
    finally:
        conn.close()

def update_h2h_stats(player1_name, player2_name, analysis):
    """
    Actualizar estadísticas H2H en la base de datos
    Función para complementar el scraper
    """
    conn = get_db()
    if conn is None:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Obtener IDs de jugadores
        player1_id = get_or_create_player(player1_name)
        player2_id = get_or_create_player(player2_name)
        
        if not player1_id or not player2_id:
            return False
        
        # Insertar o actualizar estadísticas H2H
        cursor.execute('''
            INSERT OR REPLACE INTO h2h_stats 
            (player1_id, player2_id, player1_name, player2_name, 
             total_matches, player1_wins, player2_wins, 
             player1_win_rate, player2_win_rate,
             last_encounter_date, last_encounter_winner,
             longest_streak_player, longest_streak_count, last_updated) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            player1_id, player2_id, player1_name, player2_name,
            analysis.get('total_matches', 0),
            analysis.get('player1_wins', 0),
            analysis.get('player2_wins', 0),
            analysis.get('player1_win_rate', 0.0),
            analysis.get('player2_win_rate', 0.0),
            analysis.get('last_encounter', {}).get('date'),
            analysis.get('last_encounter', {}).get('winner'),
            analysis.get('longest_winning_streak', {}).get('player'),
            analysis.get('longest_winning_streak', {}).get('streak', 0),
            datetime.now()
        ))
        
        conn.commit()
        logger.info(f"✅ Estadísticas H2H actualizadas: {player1_name} vs {player2_name}")
        
        # Actualizar estadísticas por superficie
        surface_advantage = analysis.get('surface_advantage', {})
        for surface, data in surface_advantage.items():
            cursor.execute('''
                INSERT OR REPLACE INTO surface_stats
                (player1_name, player2_name, surface, total_matches,
                 player1_wins, player2_wins, player1_win_rate, player2_win_rate,
                 advantage_player, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                player1_name, player2_name, surface,
                analysis['matches_by_surface'][surface]['total'],
                analysis['matches_by_surface'][surface]['player1_wins'],
                analysis['matches_by_surface'][surface]['player2_wins'],
                data['player1_rate'],
                data['player2_rate'],
                data['advantage'],
                datetime.now()
            ))
        
        conn.commit()
        logger.info("✅ Estadísticas por superficie actualizadas")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"❌ Error actualizando estadísticas H2H: {e}")
        return False
    finally:
        conn.close()

def get_h2h_summary(player1_name, player2_name):
    """
    Obtener resumen H2H desde la base de datos
    """
    conn = get_db()
    if conn is None:
        return None
    
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM h2h_stats 
            WHERE (player1_name = ? AND player2_name = ?) 
               OR (player1_name = ? AND player2_name = ?)
        ''', (player1_name, player2_name, player2_name, player1_name))
        
        row = cursor.fetchone()
        if row:
            return {
                'player1_name': row[3],
                'player2_name': row[4],
                'total_matches': row[5],
                'player1_wins': row[6],
                'player2_wins': row[7],
                'player1_win_rate': row[8],
                'player2_win_rate': row[9],
                'last_encounter_date': row[10],
                'last_encounter_winner': row[11],
                'longest_streak_player': row[12],
                'longest_streak_count': row[13],
                'last_updated': row[14]
            }
        return None
        
    except sqlite3.Error as e:
        logger.error(f"❌ Error obteniendo resumen H2H: {e}")
        return None
    finally:
        conn.close()

def update_player_stats(player_id, matches_played=None, matches_won=None, matches_lost=None):
    """
    Actualizar estadísticas de un jugador
    """
    conn = get_db()
    if conn is None:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Calcular porcentaje de victorias
        if matches_played and matches_won:
            win_percentage = (matches_won / matches_played) * 100
        else:
            win_percentage = 0.0
        
        cursor.execute('''
            UPDATE players 
            SET matches_played = ?, matches_won = ?, matches_lost = ?, 
                win_percentage = ?, last_updated = ?
            WHERE id = ?
        ''', (matches_played, matches_won, matches_lost, win_percentage, datetime.now(), player_id))
        
        conn.commit()
        return True
        
    except sqlite3.Error as e:
        logger.error(f"❌ Error actualizando estadísticas del jugador {player_id}: {e}")
        return False
    finally:
        conn.close()

def cleanup_old_data(days=30):
    """
    Limpiar datos antiguos (opcional)
    """
    conn = get_db()
    if conn is None:
        return False
    
    try:
        cursor = conn.cursor()
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Ejemplo: eliminar partidos muy antiguos sin estadísticas importantes
        # cursor.execute('DELETE FROM matches WHERE created_at < ?', (cutoff_date,))
        # conn.commit()
        
        logger.info("🧹 Limpieza de datos completada")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"❌ Error en limpieza de datos: {e}")
        return False
    finally:
        conn.close()

# 🧪 FUNCIÓN DE PRUEBA INTEGRADA CON SCRAPER
def test_database():
    """
    Probar que la base de datos funciona correctamente con las funciones del scraper
    """
    logger.info("🧪 Probando base de datos con integración de scraper...")
    
    # Inicializar
    if not init_db():
        logger.error("❌ Error inicializando base de datos")
        return
    
    # Probar funciones críticas del scraper
    player1_name = "Rafael Nadal"
    player2_name = "Novak Djokovic"
    
    # Probar get_or_create_player
    player1_id = get_or_create_player(player1_name, ranking=2, country="Spain")
    player2_id = get_or_create_player(player2_name, ranking=1, country="Serbia")
    
    if player1_id and player2_id:
        logger.info(f"✅ Jugadores creados: {player1_name} (ID: {player1_id}), {player2_name} (ID: {player2_id})")
    
    # Probar insert_match
    match_inserted = insert_match(
        player1_id=player1_id,
        player2_id=player2_id,
        player1_name=player1_name,
        player2_name=player2_name,
        tournament="Roland Garros",
        date="14.06.2022",
        score="6-2 4-6 6-2 7-6",
        surface="Clay",
        winner=player1_name
    )
    
    if match_inserted:
        logger.info("✅ Partido de prueba insertado")
    
    # Probar get_h2h_matches
    h2h_matches = get_h2h_matches(player1_name, player2_name)
    logger.info(f"📊 Partidos H2H encontrados: {len(h2h_matches)}")
    
    # Obtener todos los jugadores
    players = get_all_players()
    logger.info(f"📊 Total de jugadores en la base de datos: {len(players)}")
    for player in players:
        logger.info(f"   - {player['name']} (#{player['ranking']})")

# 🚀 EJECUTAR PRUEBA SI SE EJECUTA DIRECTAMENTE
if __name__ == '__main__':
    test_database()

# 🔧 AGREGAR ESTAS FUNCIONES AL FINAL DE database.py

def get_player_match_history(player_name, limit=None):
    """
    Obtiene el historial completo de partidos de un jugador desde la base de datos
    """
    import sqlite3
    
    try:
        conn = sqlite3.connect('tennis_predictions.db')
        cursor = conn.cursor()
        
        # Consulta para obtener todos los partidos del jugador
        query = """
        SELECT 
            id, player1_name, player2_name, tournament, 
            date, result, surface, winner, created_at
        FROM matches 
        WHERE player1_name = ? OR player2_name = ?
        ORDER BY date DESC
        """
        
        params = [player_name, player_name]
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor.execute(query, params)
        matches = cursor.fetchall()
        
        conn.close()
        
        # Formatear resultados
        formatted_matches = []
        for match in matches:
            formatted_match = {
                'id': match[0],
                'player1': match[1],
                'player2': match[2],
                'opponent': match[2] if match[1] == player_name else match[1],
                'tournament': match[3],
                'date': match[4],
                'score': match[5],
                'surface': match[6],
                'winner': match[7],
                'result': 'W' if match[7] == player_name else 'L' if match[7] else None,
                'created_at': match[8]
            }
            formatted_matches.append(formatted_match)
        
        return formatted_matches
        
    except Exception as e:
        print(f"❌ Error obteniendo historial del jugador {player_name}: {e}")
        return []

def get_common_opponents(player1_name, player2_name):
    """
    Encuentra adversarios comunes entre dos jugadores
    """
    import sqlite3
    
    try:
        conn = sqlite3.connect('tennis_predictions.db')
        cursor = conn.cursor()
        
        # Consulta para encontrar adversarios comunes
        query = """
        WITH player1_opponents AS (
            SELECT DISTINCT 
                CASE 
                    WHEN player1_name = ? THEN player2_name 
                    ELSE player1_name 
                END as opponent
            FROM matches 
            WHERE player1_name = ? OR player2_name = ?
        ),
        player2_opponents AS (
            SELECT DISTINCT 
                CASE 
                    WHEN player1_name = ? THEN player2_name 
                    ELSE player1_name 
                END as opponent
            FROM matches 
            WHERE player1_name = ? OR player2_name = ?
        )
        SELECT p1.opponent
        FROM player1_opponents p1
        INNER JOIN player2_opponents p2 ON p1.opponent = p2.opponent
        WHERE p1.opponent != ? AND p1.opponent != ?
        ORDER BY p1.opponent
        """
        
        cursor.execute(query, [
            player1_name, player1_name, player1_name,
            player2_name, player2_name, player2_name,
            player1_name, player2_name
        ])
        
        common_opponents = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return common_opponents
        
    except Exception as e:
        print(f"❌ Error encontrando adversarios comunes: {e}")
        return []