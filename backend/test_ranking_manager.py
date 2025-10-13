#!/usr/bin/env python3
"""
🧪 Script de test para validar RankingManager
Ejecuta y prueba la extracción de datos de rankings ATP
"""

import json
from pathlib import Path

# ✅ IMPORT CORREGIDO - Ya no está comentado
from extraer_historh2h import RankingManager

class RankingManagerTester:
    """🔬 Tester para validar RankingManager"""
    
    def __init__(self):
        self.rm = None
    
    def test_initialization(self):
        """Test 1: Inicialización y carga de datos"""
        print("=" * 60)
        print("🧪 TEST 1: INICIALIZACIÓN")
        print("=" * 60)
        
        try:
            # ✅ LÍNEA DESCOMENTADA - Crear RankingManager
            self.rm = RankingManager()
            print("✅ RankingManager inicializado correctamente")
            print(f"📊 Jugadores cargados: {len(self.rm.rankings_data) if self.rm else 'N/A'}")
            return True
        except Exception as e:
            print(f"❌ Error en inicialización: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_sample_players(self):
        """Test 2: Verificar datos de jugadores específicos"""
        print("\n" + "=" * 60)
        print("🧪 TEST 2: DATOS DE JUGADORES ESPECÍFICOS")
        print("=" * 60)
        
        # Lista de jugadores para probar
        test_players = [
            "Diallo",
            "Novak Djokovic", 
            "Rafael Nadal",
            "Cerundolo F",
            "Alcaraz",
            "Sinner"
        ]
        
        for player in test_players:
            print(f"\n🔍 Buscando: {player}")
            player_info = self.rm.get_player_info(player) if self.rm else None
            
            if player_info:
                print(f"  ✅ Encontrado: {player_info.get('name', 'N/A')}")
                print(f"  🏆 Ranking: {player_info.get('ranking_position', player_info.get('rank', 'N/A'))}")
                print(f"  🎾 PTS: {player_info.get('ranking_points', player_info.get('points', 'N/A'))}")
                
                # Verificar si tiene los datos específicos que mencionaste
                if 'prox_pts' in player_info:
                    print(f"  📈 PROX PTS: {player_info['prox_pts']}")
                if 'pts_max' in player_info:
                    print(f"  🚀 PTS MAX: {player_info['pts_max']}")
                
                # Mostrar todos los campos disponibles
                print(f"  📋 Campos disponibles: {list(player_info.keys())}")
            else:
                print("  ❌ No encontrado")
    
    def test_data_structure(self):
        """Test 3: Analizar estructura de datos"""
        print("\n" + "=" * 60)
        print("🧪 TEST 3: ESTRUCTURA DE DATOS")
        print("=" * 60)
        
        if not self.rm or not self.rm.rankings_data:
            print("❌ No hay datos para analizar")
            return
        
        # Tomar una muestra de 5 jugadores
        sample_players = list(self.rm.rankings_data.items())[:5]
        
        print(f"📊 Total de jugadores: {len(self.rm.rankings_data)}")
        print("\n🔍 MUESTRA DE DATOS (primeros 5 jugadores):")
        
        for i, (name, data) in enumerate(sample_players, 1):
            print(f"\n{i}. {name}")
            print(f"   📋 Datos: {json.dumps(data, indent=6, ensure_ascii=False)}")
    
    def test_search_methods(self):
        """Test 4: Probar métodos de búsqueda"""
        print("\n" + "=" * 60)
        print("🧪 TEST 4: MÉTODOS DE BÚSQUEDA")
        print("=" * 60)
        
        test_cases = [
            ("Búsqueda exacta", "novak djokovic"),
            ("Con acentos", "Rafael Nadál"),
            ("Apellido + inicial", "Cerundolo F"),
            ("Solo apellido", "Alcaraz"),
            ("Nombre parcial", "Daniil"),
        ]
        
        for test_name, query in test_cases:
            print(f"\n🔍 {test_name}: '{query}'")
            normalized = self.rm.normalize_name(query) if self.rm else query
            print(f"   🔄 Normalizado: '{normalized}'")
            
            result = self.rm.get_player_info(query) if self.rm else None
            if result:
                print(f"   ✅ Encontrado: {result.get('name', 'N/A')}")
            else:
                print("   ❌ No encontrado")
    
    def test_files_analysis(self):
        """Test 5: Analizar archivos de datos disponibles"""
        print("\n" + "=" * 60)
        print("🧪 TEST 5: ANÁLISIS DE ARCHIVOS")
        print("=" * 60)
        
        data_dir = Path('data')
        
        if not data_dir.exists():
            print("❌ Directorio 'data' no existe")
            return
        
        # Buscar archivos corrected
        corrected_files = list(data_dir.glob('atp_rankings_corrected_*.json'))
        print(f"📁 Archivos CORRECTED encontrados: {len(corrected_files)}")
        for file in corrected_files:
            print(f"   📄 {file.name} (tamaño: {file.stat().st_size / 1024:.1f} KB)")
        
        # Buscar archivos básicos
        basic_files = list(data_dir.glob('atp_rankings_*.json'))
        basic_files = [f for f in basic_files if 'corrected' not in f.name]
        print(f"\n📁 Archivos BÁSICOS encontrados: {len(basic_files)}")
        for file in basic_files:
            print(f"   📄 {file.name} (tamaño: {file.stat().st_size / 1024:.1f} KB)")
    
    def run_all_tests(self):
        """🚀 Ejecutar todos los tests"""
        print("🧪 INICIANDO TESTS DE RANKING MANAGER")
        print("=" * 60)
        
        # Test de archivos (no requiere clase)
        self.test_files_analysis()
        
        # Tests que requieren la clase
        if self.test_initialization():
            self.test_sample_players()
            self.test_data_structure()
            self.test_search_methods()
        
        print("\n" + "=" * 60)
        print("🏁 TESTS COMPLETADOS")
        print("=" * 60)

if __name__ == "__main__":
    # ✅ YA NO HAY MENSAJES DE ADVERTENCIA - TODO ESTÁ CORREGIDO
    tester = RankingManagerTester()
    tester.run_all_tests()