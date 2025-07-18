# models/database.py - Configuración y manejo de base de datos SQLite
# 🔴 ARCHIVO CRÍTICO - Base para todos los datos

import sqlite3
import os
from datetime import datetime

# 📍 CONFIGURACIÓN DE LA BASE DE DATOS
DATABASE_PATH = 'database.db'

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
        print(f"❌ Error conectando a la base de datos: {e}")
        return None

def init_db():
    """
    Inicializar base de datos y crear tablas
    Se ejecuta al iniciar la aplicación
    """
    print("🔧 Inicializando base de datos...")
    
    conn = get_db()
    if conn is None:
        print("❌ No se pudo conectar a la base de datos")
        return False
    
    try:
        cursor = conn.cursor()
        
        # 🎾 TABLA DE JUGADORES
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
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 🏆 TABLA DE PARTIDOS
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player1_id INTEGER,
                player2_id INTEGER,
                player1_name TEXT,
                player2_name TEXT,
                winner_id INTEGER,
                score TEXT,
                tournament TEXT,
                date_played DATE,
                surface TEXT,
                round TEXT,
                duration INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (player1_id) REFERENCES players (id),
                FOREIGN KEY (player2_id) REFERENCES players (id),
                FOREIGN KEY (winner_id) REFERENCES players (id)
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 📊 TABLA DE ESTADÍSTICAS (para futuro)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER,
                season INTEGER,
                aces INTEGER DEFAULT 0,
                double_faults INTEGER DEFAULT 0,
                first_serve_percentage REAL DEFAULT 0.0,
                break_points_saved REAL DEFAULT 0.0,
                return_points_won REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (player_id) REFERENCES players (id)
            )
        ''')
        
        # 💾 CONFIRMAR CAMBIOS
        conn.commit()
        print("✅ Tablas creadas correctamente:")
        print("   - players (jugadores)")
        print("   - matches (partidos)")
        print("   - tournaments (torneos)")
        print("   - player_stats (estadísticas)")
        
        # 📋 MOSTRAR TABLAS EXISTENTES
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"📊 Tablas en la base de datos: {[table[0] for table in tables]}")
        
        return True
        
    except sqlite3.Error as e:
        print(f"❌ Error creando tablas: {e}")
        return False
    finally:
        conn.close()

def insert_player(name, ranking=None, country=None, age=None):
    """
    Insertar un jugador en la base de datos
    Parámetros:
    - name: Nombre del jugador (obligatorio)
    - ranking: Posición en el ranking (opcional)
    - country: País (opcional)
    - age: Edad (opcional)
    """
    conn = get_db()
    if conn is None:
        return False
    
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO players 
            (name, ranking, country, age, last_updated) 
            VALUES (?, ?, ?, ?, ?)
        ''', (name, ranking, country, age, datetime.now()))
        
        conn.commit()
        player_id = cursor.lastrowid
        print(f"✅ Jugador insertado: {name} (ID: {player_id})")
        return player_id
        
    except sqlite3.Error as e:
        print(f"❌ Error insertando jugador {name}: {e}")
        return False
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
                   win_percentage, last_updated 
            FROM players 
            ORDER BY ranking ASC
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
                'last_updated': row[9]
            })
        
        return players
        
    except sqlite3.Error as e:
        print(f"❌ Error obteniendo jugadores: {e}")
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
                'last_updated': row[9]
            }
        return None
        
    except sqlite3.Error as e:
        print(f"❌ Error buscando jugador {name}: {e}")
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
        print(f"❌ Error actualizando estadísticas del jugador {player_id}: {e}")
        return False
    finally:
        conn.close()

# 🧪 FUNCIÓN DE PRUEBA
def test_database():
    """
    Probar que la base de datos funciona correctamente
    """
    print("🧪 Probando base de datos...")
    
    # Inicializar
    if not init_db():
        print("❌ Error inicializando base de datos")
        return
    
    # Insertar jugador de prueba
    player_id = insert_player("Carlos Alcaraz", ranking=1, country="Spain", age=21)
    if player_id:
        print(f"✅ Jugador de prueba insertado (ID: {player_id})")
    
    # Obtener todos los jugadores
    players = get_all_players()
    print(f"📊 Jugadores en la base de datos: {len(players)}")
    for player in players:
        print(f"   - {player['name']} (#{player['ranking']})")

# 🚀 EJECUTAR PRUEBA SI SE EJECUTA DIRECTAMENTE
if __name__ == '__main__':
    test_database()