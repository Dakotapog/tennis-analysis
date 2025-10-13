#!/usr/bin/env python3
"""
Sistema de Predicciones ML para Tenis - VERSIÓN CORREGIDA Y MEJORADA
UTILIZA DATOS HISTÓRICOS REALES Y GENERA INFORMES DETALLADOS

PROBLEMA CORREGIDO:
- El modelo se entrena con características reales de `extraer_historh2h.py`.
- Esta versión carga y usa los MISMOS datos históricos reales para predicciones.
- Se integra con `informe_detallado.py` para generar reportes explicativos.
- CORREGIDO: Error del LabelEncoder con etiquetas no vistas

Autor: Sistema de Predicciones de Tenis ML (Corregido y Mejorado)
Fecha: Agosto 2025
"""

import json
import pandas as pd
import numpy as np
import os
import joblib
import logging
from pathlib import Path
from datetime import datetime
import warnings
from sklearn.preprocessing import LabelEncoder
from utils.feature_engineering import engineer_features
from informe_detallado import add_detailed_reporting_to_predictor

warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealDataTennisPredictionSystem:
    """
    Sistema de predicciones que usa DATOS HISTÓRICOS REALES,
    eliminando completamente la simulación de datos.
    """
    def __init__(self, model_path=None, data_folder="data"):
        self.model_path = model_path or self._find_latest_model()
        self.data_folder = Path(data_folder)
        self.model_data = None
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = []
        self.matches_df = None
        self.predicciones_df = None
        self.X_processed = None  # Almacenar datos procesados para el modelo
        self.X_unscaled = None   # Almacenar datos sin escalar para el informe

    def _find_latest_model(self):
        """Encuentra el modelo más reciente en la carpeta 'models'."""
        models_dir = Path("models")
        if not models_dir.exists():
            raise FileNotFoundError("❌ No se encontró la carpeta 'models'")
        
        model_files = list(models_dir.glob("tennis_*_model_*.pkl"))
        if not model_files:
            raise FileNotFoundError("❌ No se encontraron modelos entrenados en 'models'")
        
        latest_model = max(model_files, key=os.path.getctime)
        logger.info(f"📦 Modelo más reciente encontrado: {latest_model}")
        return latest_model

    def load_trained_model(self):
        """Carga el modelo entrenado y sus componentes."""
        logger.info(f"🔥 Cargando modelo: {self.model_path}")
        try:
            self.model_data = joblib.load(self.model_path)
            self.model = self.model_data['model']
            self.scaler = self.model_data.get('scaler')
            self.label_encoder = self.model_data.get('label_encoder')
            self.feature_names = self.model_data['feature_names']
            
            # 🔧 CORRECCIÓN: Asegurar que el label_encoder esté correctamente configurado
            if self.label_encoder is None:
                logger.warning("⚠️ LabelEncoder no encontrado en el modelo. Creando uno robusto...")
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit([0, 1])  # Clases estándar para clasificación binaria
            else:
                # Verificar que el encoder tenga las clases correctas
                if not hasattr(self.label_encoder, 'classes_') or len(self.label_encoder.classes_) < 2:
                    logger.warning("⚠️ LabelEncoder incompleto. Recreando...")
                    self.label_encoder = LabelEncoder()
                    self.label_encoder.fit([0, 1])
            
            logger.info("✅ Modelo cargado exitosamente con {} features.".format(len(self.feature_names)))
            return True
        except Exception as e:
            logger.error(f"❌ Error crítico cargando el modelo: {e}")
            raise

    def load_match_data(self, specific_file=None):
        """Carga datos de partidos desde el archivo de features pre-calculadas."""
        logger.info("📋 Cargando datos de partidos con features...")
        
        if specific_file:
            match_file = Path(specific_file)
        else:
            reports_dir = Path("reports")
            match_files = list(reports_dir.glob("h2h_results_enhanced_*.json"))
            if not match_files:
                error_msg = "❌ No se encontraron archivos 'h2h_results_enhanced_*.json' en 'reports'. Ejecute 'extraer_historh2h.py' primero."
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            match_file = max(match_files, key=os.path.getctime)

        logger.info(f"📂 Procesando archivo de datos: {match_file}")
        try:
            with open(match_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'partidos' not in data:
                raise ValueError("El archivo JSON no contiene la clave 'partidos'.")

            self.matches_df = pd.DataFrame(data['partidos'])
            if 'surface' in self.matches_df.columns:
                self.matches_df.rename(columns={'surface': 'tipo_cancha'}, inplace=True)
            
            logger.info(f"✅ Cargados {len(self.matches_df)} partidos.")
        except Exception as e:
            logger.error(f"❌ Error cargando o procesando el archivo de partidos: {e}")
            raise

    def preprocess_for_prediction(self):
        """Preprocesa los datos usando la función de ingeniería de features."""
        logger.info("🔧 Preprocesando datos para predicción...")
        
        df_with_features = engineer_features(self.matches_df.copy())
        
        missing_features = [f for f in self.feature_names if f not in df_with_features.columns]
        if missing_features:
            logger.warning(f"⚠️ Features del modelo no encontradas en los datos: {missing_features}. Se rellenarán con 0.")
            for feature in missing_features:
                df_with_features[feature] = 0
        
        X = df_with_features[self.feature_names].copy()
        X.fillna(0, inplace=True)
        
        self.X_unscaled = X.copy() # Guardar features sin escalar para el informe

        # 🔧 CORRECCIÓN: Manejo mejorado de columnas categóricas
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if not categorical_cols.empty:
            logger.info(f"🏷️ Codificando columnas categóricas: {list(categorical_cols)}")
            for col in categorical_cols:
                X[col] = X[col].astype(str)
                # Crear un encoder específico para esta columna
                col_encoder = LabelEncoder()
                unique_values = X[col].unique()
                col_encoder.fit(unique_values)
                X[col] = col_encoder.transform(X[col])

        if self.scaler:
            logger.info("📏 Aplicando escalado de características.")
            X_scaled = self.scaler.transform(X)
            self.X_processed = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        else:
            self.X_processed = X
            
        logger.info(f"✅ Datos preprocesados listos para predicción: {self.X_processed.shape}")

    def make_predictions(self):
        """Realiza predicciones sobre los datos preprocesados con manejo robusto de etiquetas."""
        logger.info("🎯 Realizando predicciones...")
        
        try:
            predictions = self.model.predict(self.X_processed)
            probabilities = self.model.predict_proba(self.X_processed)
            
            # 🔧 CORRECCIÓN: Manejo robusto del LabelEncoder
            logger.info(f"🔍 Predicciones brutas del modelo: {np.unique(predictions)}")
            logger.info(f"🔍 Clases conocidas por el encoder: {self.label_encoder.classes_}")
            
            # Verificar si hay etiquetas no vistas
            unique_predictions = np.unique(predictions)
            unknown_labels = set(unique_predictions) - set(self.label_encoder.classes_)
            
            if unknown_labels:
                logger.warning(f"⚠️ Etiquetas no vistas detectadas: {unknown_labels}")
                # Recrear el encoder con todas las clases necesarias
                all_classes = sorted(list(set(self.label_encoder.classes_) | set(unique_predictions)))
                logger.info(f"🔧 Recreando encoder con clases: {all_classes}")
                
                new_encoder = LabelEncoder()
                new_encoder.fit(all_classes)
                self.label_encoder = new_encoder
            
            # Ahora sí hacer la transformación inversa
            predicted_classes = self.label_encoder.inverse_transform(predictions)
            
            # 🔧 CORRECCIÓN: Mapeo más robusto de clases a nombres de jugadores
            results = []
            for i, row in self.matches_df.iterrows():
                p1 = str(row.get('jugador1', 'Player 1')).strip()
                p2 = str(row.get('jugador2', 'Player 2')).strip()
                
                # Mapeo robusto: 0 = jugador1, 1 = jugador2 (o viceversa según el modelo)
                predicted_class = predicted_classes[i]
                if predicted_class == 0:
                    winner = p1
                    winner_prob = probabilities[i][0]
                    loser_prob = probabilities[i][1]
                else:
                    winner = p2
                    winner_prob = probabilities[i][1]
                    loser_prob = probabilities[i][0]
                
                confidence = max(probabilities[i])
                
                result = {
                    'partido': f"{p1} vs {p2}",
                    'jugador1': p1,
                    'jugador2': p2,
                    'ranking_j1': row.get('p1_ranking', 'N/R'),
                    'ranking_j2': row.get('p2_ranking', 'N/R'),
                    'forma_j1': f"{row.get('p1_form_wins', 0)}W-{row.get('p1_form_losses', 0)}L",
                    'forma_j2': f"{row.get('p2_form_wins', 0)}W-{row.get('p2_form_losses', 0)}L",
                    'h2h_real': f"{row.get('h2h_total', 0)} matches",
                    'ganador_predicho': winner,
                    'confianza': f"{confidence:.1%}",
                    'prob_j1': f"{probabilities[i][0]:.1%}",
                    'prob_j2': f"{probabilities[i][1]:.1%}",
                    'torneo': row.get('torneo_nombre', 'N/A'),
                    'fecha': row.get('fecha', 'N/A'),
                    'superficie': row.get('tipo_cancha', 'N/A')
                }
                results.append(result)
                
            self.predicciones_df = pd.DataFrame(results)
            logger.info(f"✅ {len(results)} predicciones realizadas exitosamente.")
            return self.predicciones_df
            
        except Exception as e:
            logger.error(f"❌ Error durante las predicciones: {e}")
            logger.error(f"📊 Forma de X_processed: {self.X_processed.shape}")
            logger.error(f"📊 Tipos de datos: {self.X_processed.dtypes.to_dict()}")
            raise

    def save_predictions(self, filename=None):
        """Guarda las predicciones en un archivo JSON."""
        if self.predicciones_df is None:
            logger.error("❌ No hay predicciones para guardar.")
            return None
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions/predicciones_reales_{timestamp}.json"
        
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.predicciones_df.to_json(output_path, orient='records', indent=4, force_ascii=False)
        logger.info(f"💾 Predicciones guardadas en: {output_path}")
        return output_path

    def run_full_prediction_pipeline(self, match_file=None):
        """Ejecuta el pipeline completo de predicciones."""
        logger.info("🚀 INICIANDO PIPELINE DE PREDICCIONES CON DATOS UNIFICADOS")
        self.load_trained_model()
        self.load_match_data(match_file)
        self.preprocess_for_prediction()
        predictions_df = self.make_predictions()
        output_file = self.save_predictions()
        logger.info("✅ PIPELINE DE PREDICCIÓN COMPLETADO EXITOSAMENTE")
        return predictions_df, output_file

def main():
    """Función principal para ejecutar el sistema de predicciones y reportes."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sistema de Predicciones ML para Tenis con Reportes Detallados')
    parser.add_argument('--model', help='Ruta al modelo entrenado .pkl')
    parser.add_argument('--matches', help='Ruta al archivo JSON de partidos con features')
    
    args = parser.parse_args()
    
    try:
        # 1. Añadir capacidad de reporte a la clase del sistema
        add_detailed_reporting_to_predictor(RealDataTennisPredictionSystem)
        
        # 2. Inicializar el sistema de predicción
        predictor = RealDataTennisPredictionSystem(
            model_path=args.model
        )
        
        # 3. Ejecutar el pipeline de predicción
        predictions_df, output_file = predictor.run_full_prediction_pipeline(args.matches)
        
        if predictions_df is not None:
            print("\n" + "="*70)
            print("📊 RESUMEN DE PREDICCIONES")
            print("="*70)
            print(predictor.predicciones_df.head().to_string())
            print(f"\n📁 Predicciones guardadas en: {output_file}")

            # 4. Generar y guardar el informe detallado
            logger.info("✏️  Generando informe detallado...")
            detailed_report_path = predictor.generate_and_save_detailed_report()
            if detailed_report_path:
                print(f"📄 Informe detallado guardado en: {detailed_report_path}")

    except FileNotFoundError as e:
        logger.error(f"❌ Error de archivo: {e}")
        return 1
    except Exception as e:
        logger.error(f"❌ Ocurrió un error inesperado en la ejecución: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    main()

   