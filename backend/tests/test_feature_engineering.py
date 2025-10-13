import unittest
import pandas as pd
import sys
import os

# Añadir el directorio raíz al sys.path para permitir importaciones de módulos del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.feature_engineering import engineer_features

class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        """Crea un DataFrame de prueba con datos simulados."""
        self.sample_data = [
            {
                "jugador1": "Alcaraz C.",
                "jugador2": "Sinner J.",
                "ranking_analysis": {
                    "Alcaraz_C_ranking": 1,
                    "Sinner_J_ranking": 2,
                    "common_opponents_count": 10,
                    "p1_rivalry_score": 50.5,
                    "p2_rivalry_score": 45.2
                },
                "form_analysis": {
                    "Alcaraz_C_form": {"wins": 8, "losses": 2, "win_rate": 0.8},
                    "Sinner_J_form": {"wins": 9, "losses": 1, "win_rate": 0.9}
                },
                "surface_analysis": {
                    "Alcaraz_C_surface_stats": {"Hard": {"wins": 20, "losses": 5, "win_rate": 0.8}},
                    "Sinner_J_surface_stats": {"Hard": {"wins": 25, "losses": 3, "win_rate": 0.89}}
                },
                "estadisticas": {
                    "enfrentamientos_totales": 5
                },
                "tipo_cancha": "Hard"
            }
        ]
        self.df = pd.DataFrame(self.sample_data)

    def test_engineer_features_creates_columns(self):
        """Verifica que la función crea todas las columnas de características esperadas."""
        processed_df = engineer_features(self.df)
        
        expected_features = [
            'p1_ranking', 'p2_ranking', 'common_opponents_count', 'p1_rivalry_score',
            'p2_rivalry_score', 'p1_form_wins', 'p1_form_losses', 'p1_form_win_rate',
            'p2_form_wins', 'p2_form_losses', 'p2_form_win_rate', 'p1_surface_wins',
            'p1_surface_losses', 'p1_surface_win_rate', 'p2_surface_wins',
            'p2_surface_losses', 'p2_surface_win_rate', 'h2h_total'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, processed_df.columns)

    def test_engineer_features_data_types(self):
        """Verifica que las nuevas columnas de características son de tipo numérico."""
        processed_df = engineer_features(self.df)
        
        numeric_features = [
            'p1_ranking', 'p2_ranking', 'common_opponents_count', 'p1_rivalry_score',
            'p2_rivalry_score', 'p1_form_wins', 'p1_form_losses', 'p1_form_win_rate',
            'p2_form_wins', 'p2_form_losses', 'p2_form_win_rate', 'p1_surface_wins',
            'p1_surface_losses', 'p1_surface_win_rate', 'p2_surface_wins',
            'p2_surface_losses', 'p2_surface_win_rate', 'h2h_total'
        ]
        
        for feature in numeric_features:
            self.assertTrue(pd.api.types.is_numeric_dtype(processed_df[feature]))

    def test_engineer_features_correct_values(self):
        """Verifica que los valores de las características se extraen correctamente."""
        processed_df = engineer_features(self.df)
        
        self.assertEqual(processed_df.loc[0, 'p1_ranking'], 1)
        self.assertEqual(processed_df.loc[0, 'p2_ranking'], 2)
        self.assertEqual(processed_df.loc[0, 'p1_form_win_rate'], 0.8)
        self.assertEqual(processed_df.loc[0, 'p2_surface_wins'], 25)
        self.assertEqual(processed_df.loc[0, 'h2h_total'], 5)

if __name__ == '__main__':
    unittest.main()
