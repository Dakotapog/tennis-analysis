import unittest
from unittest.mock import MagicMock
from extraer_historh2h import RivalryAnalyzer, RankingManager, EloRatingSystem

class TestRivalryAnalyzer(unittest.TestCase):
    def setUp(self):
        # Configurar RankingManager y EloRatingSystem simulados
        self.ranking_manager = RankingManager()
        self.elo_system = EloRatingSystem()
        
        # Crear instancia de RivalryAnalyzer
        self.rivalry_analyzer = RivalryAnalyzer(self.ranking_manager, self.elo_system)
        
        # Mockear métodos de RankingManager
        self.ranking_manager.get_player_ranking = MagicMock()
        self.ranking_manager.get_player_info = MagicMock()

    def test_calculate_base_opponent_weight(self):
        # Probar cálculo de peso base para diferentes rankings
        self.assertEqual(self.rivalry_analyzer.calculate_base_opponent_weight(5), 10)
        self.assertEqual(self.rivalry_analyzer.calculate_base_opponent_weight(15), 8)
        self.assertEqual(self.rivalry_analyzer.calculate_base_opponent_weight(30), 6)
        self.assertEqual(self.rivalry_analyzer.calculate_base_opponent_weight(50), 4)
        self.assertEqual(self.rivalry_analyzer.calculate_base_opponent_weight(100), 2)
        self.assertEqual(self.rivalry_analyzer.calculate_base_opponent_weight(None), 1)

    def test_analyze_direct_h2h(self):
        # Datos de prueba para enfrentamientos directos
        h2h_matches = [
            {"ganador": "Player1", "fecha": "2023-01-01"},
            {"ganador": "Player2", "fecha": "2023-02-01"},
            {"ganador": "Player1", "fecha": "2023-03-01"},
        ]
        p1_score, p2_score, log = self.rivalry_analyzer.analyze_direct_h2h(h2h_matches, "Player1", "Player2")
        
        self.assertEqual(p1_score, 2)
        self.assertEqual(p2_score, 1)
        self.assertIn("Victoria de Player1 en 2023-01-01", log)
        self.assertIn("Victoria de Player2 en 2023-02-01", log)

    def test_analyze_surface_specialization(self):
        # Datos de prueba para historial de partidos
        player_history = [
            {"superficie": "Dura", "oponente": "PlayerA", "resultado": "2-0", "outcome": "Ganó"},
            {"superficie": "Dura", "oponente": "PlayerB", "resultado": "2-1", "outcome": "Ganó"},
            {"superficie": "Arcilla", "oponente": "PlayerC", "resultado": "0-2", "outcome": "Perdió"},
        ]
        self.ranking_manager.get_player_ranking.side_effect = lambda x: {"PlayerA": 10, "PlayerB": 50, "PlayerC": 30}.get(x, None)
        
        score, log = self.rivalry_analyzer.analyze_surface_specialization(player_history, "Dura", "Player1")
        
        self.assertGreater(score, 0)
        self.assertIn("Victoria vs Rank 10", log[1])
        self.assertIn("Victoria vs Rank 50", log[2])

    def test_analyze_strength_of_schedule(self):
        # Datos de prueba para historial de partidos
        player_history = [
            {"oponente": "PlayerA", "resultado": "2-0", "outcome": "Ganó"},
            {"oponente": "PlayerB", "resultado": "1-2", "outcome": "Perdió"},
        ]
        self.ranking_manager.get_player_ranking.side_effect = lambda x: {"PlayerA": 10, "PlayerB": 50}.get(x, None)
        
        score, log = self.rivalry_analyzer.analyze_strength_of_schedule(player_history, "Player1")
        
        self.assertGreater(score, 0)
        self.assertIn("Victoria vs Rank 10", log[0])
        self.assertIn("Derrota vs Rank 50", log[1])

    def test_determine_match_winner(self):
        # Probar determinación de ganador
        match_data = {"outcome": "Ganó", "resultado": "2-0", "oponente": "PlayerB"}
        self.assertTrue(self.rivalry_analyzer.determine_match_winner(match_data, "Player1"))
        
        match_data = {"outcome": "Perdió", "resultado": "0-2", "oponente": "PlayerB"}
        self.assertFalse(self.rivalry_analyzer.determine_match_winner(match_data, "Player1"))

if __name__ == "__main__":
    unittest.main()