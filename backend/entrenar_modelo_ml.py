#!/usr/bin/env python3
"""
Pipeline de Machine Learning Completo para Predicciones de Tenis
VERSIÓN CORREGIDA - Manejo de clases con pocos ejemplos

Autor: Sistema de Predicciones de Tenis ML
Fecha: Agosto 2025
"""

import json
import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
from datetime import datetime
from sklearn.model_selection import (
    train_test_split, 
    StratifiedKFold, 
    cross_val_score,
    RandomizedSearchCV
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    roc_auc_score
)
import lightgbm as lgb
import warnings
from utils.feature_engineering import engineer_features
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TennisMLPipeline:
    def __init__(self, dataset_files=None):
        """
        Pipeline ML para predicciones de tenis
        Args:
            dataset_files: Lista de archivos JSON unificados, o None para auto-detectar
        """
        self.dataset_files = dataset_files or self._auto_detect_datasets()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_names = []
        self.results = {}
        
    def _auto_detect_datasets(self):
        """Auto-detecta archivos de dataset en la carpeta training_data"""
        training_dir = Path("training_data")
        if not training_dir.exists():
            training_dir = Path("reports")
            
        json_files = list(training_dir.glob("training_dataset_*.json"))
        if not json_files:
            logger.warning("⚠️ No se encontraron archivos de training dataset")
            return []
            
        logger.info(f"🔍 Detectados {len(json_files)} archivos de dataset")
        return json_files
    
    def load_and_prepare_data(self):
        """Cargar y preparar datos desde archivos JSON"""
        logger.info("🔄 Cargando y preparando datos...")
        
        all_data = []
        dataset_dir = "reports"
        
        if not os.path.exists(dataset_dir):
            raise ValueError(f"❌ Directorio {dataset_dir} no encontrado")
        
        # Buscar archivos de dataset
        dataset_files = [f for f in os.listdir(dataset_dir) 
                        if f.startswith('training_dataset_') and f.endswith('.json')]
        
        logger.info(f"📁 Detectados {len(dataset_files)} archivos de dataset")
        
        if not dataset_files:
            raise ValueError("❌ No se encontraron archivos de dataset")
        
        successful_loads = 0
        
        for file_name in dataset_files:
            file_path = os.path.join(dataset_dir, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # CORRECCIÓN: Manejar diferentes estructuras de datos
                if isinstance(data, list):
                    # Si es una lista directa de partidos
                    all_data.extend(data)
                    logger.info(f"✅ Cargados {len(data)} partidos desde {file_name}")
                    successful_loads += 1
                    
                elif isinstance(data, dict):
                    # Si es un diccionario, buscar los partidos
                    if 'partidos' in data:
                        partidos = data['partidos']
                        if isinstance(partidos, list):
                            all_data.extend(partidos)
                            logger.info(f"✅ Cargados {len(partidos)} partidos desde {file_name}")
                            successful_loads += 1
                        else:
                            logger.warning(f"⚠️ 'partidos' no es una lista en {file_name}")
                    
                    elif 'matches' in data:
                        matches = data['matches']
                        if isinstance(matches, list):
                            all_data.extend(matches)
                            logger.info(f"✅ Cargados {len(matches)} partidos desde {file_name}")
                            successful_loads += 1
                        else:
                            logger.warning(f"⚠️ 'matches' no es una lista en {file_name}")
                    
                    else:
                        # Si el diccionario no tiene estructura conocida, intentar extraer valores
                        logger.warning(f"⚠️ Estructura desconocida en {file_name}, intentando extraer datos...")
                        # Buscar listas en el diccionario
                        for key, value in data.items():
                            if isinstance(value, list) and len(value) > 0:
                                # Verificar si parece una lista de partidos
                                if isinstance(value[0], dict) and any(k in value[0] for k in ['player1', 'player2', 'winner', 'jugador1', 'jugador2']):
                                    all_data.extend(value)
                                    logger.info(f"✅ Cargados {len(value)} partidos desde clave '{key}' en {file_name}")
                                    successful_loads += 1
                                    break
                
                else:
                    logger.error(f"❌ Formato de datos no reconocido en {file_name}: {type(data)}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"❌ Error JSON en {file_name}: {e}")
            except Exception as e:
                logger.error(f"❌ Error procesando {file_name}: {e}")
        
        if successful_loads == 0:
            raise ValueError("❌ No se pudieron cargar datos de ningún archivo")
        
        logger.info(f"✅ Total partidos cargados: {len(all_data)} desde {successful_loads} archivos")
        
        if len(all_data) == 0:
            raise ValueError("❌ No se encontraron partidos válidos en los datos")
        
        # Convertir a DataFrame
        self.df = pd.DataFrame(all_data)
        logger.info(f"📊 Dataset creado con {len(self.df)} registros")
        
        # Mostrar columnas disponibles para debug
        logger.info(f"🔍 Columnas encontradas: {list(self.df.columns)}")
        
        # Verificar estructura de datos
        if len(self.df) > 0:
            logger.info("📋 Primer registro de ejemplo:")
            for key, value in self.df.iloc[0].items():
                # Truncar valores muy largos para el log
                str_value = str(value)
                if len(str_value) > 100:
                    str_value = str_value[:100] + "..."
                logger.info(f"   {key}: {str_value}")
        
        # **NUEVA ESTRATEGIA: Clasificación Binaria con Ingeniería de Features**
        logger.info("🔧 Transformando a problema de clasificación binaria...")
        
        # Identificar columna de ganador real
        target_columns = ['ganador_real', 'winner', 'resultado', 'ganador', 'target']
        ganador_col = None
        for col in target_columns:
            if col in self.df.columns:
                ganador_col = col
                break
        
        if ganador_col is None:
            raise ValueError("❌ No se encontró la columna del ganador para crear el target binario.")
            
        # Crear target binario: 0 si gana jugador1, 1 si gana jugador2
        if 'jugador1' not in self.df.columns or 'jugador2' not in self.df.columns:
            raise ValueError("❌ Faltan las columnas 'jugador1' o 'jugador2' para la clasificación binaria.")
            
        self.df['target_binario'] = np.where(self.df[ganador_col] == self.df['jugador1'], 0, 1)
        target_col = 'target_binario'
        logger.info(f"✅ Columna objetivo binaria '{target_col}' creada.")

        # **PASO CLAVE: Ingeniería de Características**
        logger.info("🛠️ Realizando ingeniería de características...")
        self.df = engineer_features(self.df)

        # Identificar columnas de features (excluir target y columnas de identificación/complejas)
        exclude_columns = [
            ganador_col, target_col, 'match_id', 'id', 'fecha', 'tournament', 'torneo',
            'match_number', 'jugador1', 'jugador2', 'jugador1_nacionalidad', 
            'jugador2_nacionalidad', 'torneo_nombre', 'torneo_completo', 
            'match_url', 'cuota1', 'cuota2',
            # Columnas complejas originales que ya fueron procesadas
            'enfrentamientos_directos', 'estadisticas', 'ranking_analysis', 
            'form_analysis', 'surface_analysis', 'location_analysis', 
            'common_opponents_detailed'
        ]
        
        # Excluir también todas las columnas de historial de jugadores
        historial_cols = [col for col in self.df.columns if col.startswith('historial_')]
        exclude_columns.extend(historial_cols)
        
        feature_columns = [col for col in self.df.columns if col not in exclude_columns]
        
        if not feature_columns:
            raise ValueError("❌ No se encontraron columnas de features válidas después de la ingeniería.")
        
        logger.info(f"✅ {len(feature_columns)} características numéricas generadas.")
        
        self.feature_names = feature_columns
        self.X = self.df[feature_columns]
        self.y = self.df[target_col]
        
        # **SECCIÓN DE FILTRADO DE CLASES ELIMINADA**
        # Con la clasificación binaria, ya no es necesario filtrar clases.
        
        class_distribution = self.y.value_counts()
        logger.info("📈 Distribución de clases binarias:")
        for clase, count in class_distribution.items():
            clase_label = "Gana Jugador 1" if clase == 0 else "Gana Jugador 2"
            percentage = (count / len(self.y)) * 100
            logger.info(f"   {clase_label} (Clase {clase}): {count} partidos ({percentage:.1f}%)")

        # Verificar que tenemos suficientes datos
        if len(self.y) < 10:
            raise ValueError(f"❌ Dataset muy pequeño: {len(self.y)} registros. Se necesitan al menos 10.")
        
        return self.X, self.y

    def preprocess_features(self):
        """Preprocesa las features para ML"""
        logger.info("🔧 Preprocesando features...")
        
        # Manejar valores faltantes
        self.X = self.X.fillna(0)
        
        # Codificar variables categóricas si las hay
        categorical_columns = self.X.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            logger.info(f"🏷️ Codificando {len(categorical_columns)} columnas categóricas")
            for col in categorical_columns:
                le = LabelEncoder()
                self.X[col] = le.fit_transform(self.X[col].astype(str))
        
        # Codificar target
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
        logger.info(f"✅ Features preprocesadas: {self.X.shape[1]} características")
        return self.X, self.y_encoded
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        División estratificada de datos y escalado de características.
        El escalado se hace DESPUÉS de la división para evitar data leakage.
        """
        logger.info(f"📊 Dividiendo datos: {int((1-test_size)*100)}% entrenamiento, {int(test_size*100)}% prueba")
        
        # Verificar si podemos usar estratificación
        class_counts = np.bincount(self.y_encoded)
        min_class_count = np.min(class_counts)
        
        logger.info("🔍 Verificando distribución para split estratificado...")
        logger.info(f"   Mínimo ejemplos por clase: {min_class_count}")
        
        # Intentar split estratificado
        try:
            X_train_df, X_test_df, self.y_train, self.y_test = train_test_split(
                self.X, self.y_encoded, 
                test_size=test_size, 
                random_state=random_state,
                stratify=self.y_encoded
            )
            logger.info("✅ Split estratificado exitoso")
            
        except ValueError as e:
            logger.warning(f"⚠️  No se puede usar split estratificado: {e}")
            logger.info("🔄 Usando split aleatorio simple...")
            
            # Fallback: split simple sin estratificación
            X_train_df, X_test_df, self.y_train, self.y_test = train_test_split(
                self.X, self.y_encoded, 
                test_size=test_size, 
                random_state=random_state,
                stratify=None  # Sin estratificación
            )
            logger.info("✅ Split aleatorio completado")

        # **CORRECCIÓN CRÍTICA: Ajustar y aplicar el scaler**
        logger.info("🔧 Escalando características (StandardScaler)...")
        
        # Identificar columnas numéricas para escalar
        numeric_cols = X_train_df.select_dtypes(include=np.number).columns
        
        # Ajustar el scaler SOLO con datos de entrenamiento
        self.scaler.fit(X_train_df[numeric_cols])
        
        # Transformar ambos conjuntos de datos y mantenerlos como DataFrames
        X_train_scaled = self.scaler.transform(X_train_df[numeric_cols])
        X_test_scaled = self.scaler.transform(X_test_df[numeric_cols])
        
        self.X_train = pd.DataFrame(X_train_scaled, index=X_train_df.index, columns=numeric_cols)
        self.X_test = pd.DataFrame(X_test_scaled, index=X_test_df.index, columns=numeric_cols)
        
        # Re-incorporar columnas no numéricas si las hubiera (aunque el script actual las codifica)
        non_numeric_cols = X_train_df.select_dtypes(exclude=np.number).columns
        if not non_numeric_cols.empty:
            self.X_train = pd.concat([self.X_train, X_train_df[non_numeric_cols]], axis=1)
            self.X_test = pd.concat([self.X_test, X_test_df[non_numeric_cols]], axis=1)
        
        # Asegurar el orden de las columnas
        self.X_train = self.X_train[self.feature_names]
        self.X_test = self.X_test[self.feature_names]

        logger.info("✅ Escalado de características completado.")
        
        logger.info(f"   Entrenamiento: {len(self.X_train)} partidos")
        logger.info(f"   Prueba: {len(self.X_test)} partidos")
        
        # Verificar distribución en sets de entrenamiento y prueba
        train_dist = np.bincount(self.y_train)
        test_dist = np.bincount(self.y_test)
        
        logger.info("📈 Distribución en conjunto de entrenamiento:")
        for i, count in enumerate(train_dist):
            if count > 0:
                clase_name = self.label_encoder.inverse_transform([i])[0]
                logger.info(f"   {clase_name}: {count} partidos")
        
        logger.info("📈 Distribución en conjunto de prueba:")
        for i, count in enumerate(test_dist):
            if count > 0:
                clase_name = self.label_encoder.inverse_transform([i])[0]
                logger.info(f"   {clase_name}: {count} partidos")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def cross_validation(self, cv_folds=5):
        """Validación cruzada estratificada con manejo robusto"""
        logger.info(f"🔄 Ejecutando validación cruzada ({cv_folds}-fold)...")
        
        # Configurar modelo base
        lgb_model = lgb.LGBMClassifier(
            objective='binary' if len(self.label_encoder.classes_) == 2 else 'multiclass',
            metric='binary_logloss' if len(self.label_encoder.classes_) == 2 else 'multi_logloss',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.1,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            random_state=42,
            min_child_samples=1  # Permitir nodos con muy pocas muestras
        )
        
        # Intentar validación cruzada estratificada
        try:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(lgb_model, self.X, self.y_encoded, cv=skf, scoring='accuracy')
            logger.info("✅ Validación cruzada estratificada completada")
            
        except ValueError as e:
            logger.warning(f"⚠️  No se puede usar CV estratificado: {e}")
            logger.info("🔄 Usando validación cruzada simple...")
            
            # Fallback: CV simple
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=min(cv_folds, len(self.X)), shuffle=True, random_state=42)
            cv_scores = cross_val_score(lgb_model, self.X, self.y_encoded, cv=kf, scoring='accuracy')
            logger.info("✅ Validación cruzada simple completada")
        
        logger.info("📊 Resultados de validación cruzada:")
        logger.info(f"   Accuracy media: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        logger.info(f"   Rango: {cv_scores.min():.4f} - {cv_scores.max():.4f}")
        
        self.results['cv_scores'] = cv_scores
        self.results['cv_mean'] = cv_scores.mean()
        self.results['cv_std'] = cv_scores.std()
        
        return cv_scores
    
    def optimize_hyperparameters(self, n_iter=10):
        """Optimización de hiperparámetros con RandomizedSearchCV"""
        logger.info(f"🎯 Optimizando hiperparámetros ({n_iter} iteraciones)...")
        
        # Espacio de búsqueda (reducido para datasets pequeños)
        param_grid = {
            'num_leaves': [15, 31],
            'learning_rate': [0.1, 0.15],
            'feature_fraction': [0.8, 1.0],
            'bagging_fraction': [0.8, 1.0],
            'min_child_samples': [1, 5, 10]  # Permitir valores muy pequeños
        }
        
        lgb_model = lgb.LGBMClassifier(
            objective='binary' if len(self.label_encoder.classes_) == 2 else 'multiclass',
            metric='binary_logloss' if len(self.label_encoder.classes_) == 2 else 'multi_logloss',
            boosting_type='gbdt',
            verbose=-1,
            random_state=42
        )
        
        # Búsqueda aleatoria con validación cruzada
        try:
            random_search = RandomizedSearchCV(
                lgb_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring='accuracy',
                cv=3,  # CV reducido para datasets pequeños
                verbose=0,
                random_state=42,
                n_jobs=1  # Evitar problemas de paralelización
            )
            
            random_search.fit(self.X_train, self.y_train)
            
            logger.info("🏆 Mejores parámetros encontrados:")
            for param, value in random_search.best_params_.items():
                logger.info(f"   {param}: {value}")
            
            logger.info(f"📈 Mejor score CV: {random_search.best_score_:.4f}")
            
            self.results['best_params'] = random_search.best_params_
            self.results['best_cv_score'] = random_search.best_score_
            
            return random_search.best_estimator_
            
        except Exception as e:
            logger.warning(f"⚠️  Error en optimización de hiperparámetros: {e}")
            logger.info("🔄 Usando modelo con parámetros por defecto...")
            return None
    
    def train_final_model(self, optimized_model=None):
        """Entrena el modelo final"""
        logger.info("🚀 Entrenando modelo final...")
        
        if optimized_model:
            self.model = optimized_model
        else:
            # Modelo con parámetros por defecto adaptados
            self.model = lgb.LGBMClassifier(
                objective='binary' if len(self.label_encoder.classes_) == 2 else 'multiclass',
                metric='binary_logloss' if len(self.label_encoder.classes_) == 2 else 'multi_logloss',
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.1,
                feature_fraction=0.9,
                bagging_fraction=0.8,
                bagging_freq=5,
                min_child_samples=1,  # Permitir nodos con muy pocas muestras
                verbose=-1,
                random_state=42
            )
        
        # Entrenar en datos de entrenamiento
        self.model.fit(self.X_train, self.y_train)
        
        logger.info("✅ Modelo entrenado exitosamente")
        return self.model
    
    def evaluate_model(self):
        """Evaluación completa del modelo"""
        logger.info("📊 Evaluando rendimiento del modelo...")
        
        # Predicciones en conjunto de prueba
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)
        
        # Métricas básicas
        accuracy = accuracy_score(self.y_test, y_pred)
        
        logger.info("🎯 RESULTADOS FINALES:")
        logger.info(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # ROC-AUC solo para clasificación binaria
        if len(self.label_encoder.classes_) == 2:
            roc_auc = roc_auc_score(self.y_test, y_pred_proba[:, 1])
            logger.info(f"   ROC-AUC: {roc_auc:.4f}")
            self.results['test_roc_auc'] = roc_auc
        else:
            logger.info("   ROC-AUC: N/A (clasificación multiclase)")
            roc_auc = None
        
        # Comparación con sistema actual (85.71%)
        sistema_actual = 0.8571
        mejora = ((accuracy - sistema_actual) / sistema_actual) * 100
        
        logger.info("📈 COMPARACIÓN CON SISTEMA ACTUAL:")
        logger.info(f"   Sistema actual: {sistema_actual:.4f} (85.71%)")
        logger.info(f"   Modelo ML: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        if accuracy > sistema_actual:
            logger.info(f"   ✅ MEJORA: +{mejora:.2f}%")
        else:
            logger.info(f"   ⚠️  DIFERENCIA: {mejora:.2f}%")
        
        # Guardar resultados
        self.results.update({
            'test_accuracy': accuracy,
            'sistema_actual': sistema_actual,
            'mejora_porcentual': mejora
        })
        
        # Reporte de clasificación detallado
        try:
            # CORRECCIÓN: Usar nombres de clase binarios para el reporte
            target_names_binarios = ['Gana Jugador 1', 'Gana Jugador 2']
            
            report = classification_report(
                self.y_test, y_pred, 
                target_names=target_names_binarios,
                output_dict=True,
                zero_division=0
            )
            
            logger.info("📋 Reporte de clasificación:")
            for clase, metrics in report.items():
                if isinstance(metrics, dict):
                    precision = metrics.get('precision', 0)
                    recall = metrics.get('recall', 0)
                    f1 = metrics.get('f1-score', 0)
                    logger.info(f"   {clase}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

            self.results['classification_report'] = report
            
        except Exception as e:
            logger.warning(f"⚠️  Error generando reporte detallado: {e}")
        
        return accuracy, roc_auc
    
    def feature_importance(self):
        """Análisis de importancia de features"""
        logger.info("🔍 Analizando importancia de features...")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("🏆 Top 10 features más importantes:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                logger.info(f"   {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
            
            self.results['feature_importance'] = importance_df
            return importance_df
        else:
            logger.warning("⚠️  El modelo no proporciona importancia de features")
            return None
    
    def save_model_and_results(self):
        """Guarda el modelo y resultados"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear directorio de modelos
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Guardar modelo
        model_path = models_dir / f"tennis_lgb_model_{timestamp}.pkl"
        
        try:
            import joblib
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names,
                'results': self.results
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"💾 Modelo guardado en: {model_path}")
            
        except Exception as e:
            logger.warning(f"⚠️  Error guardando modelo: {e}")
            model_path = "No guardado"
        
        # Guardar reporte de resultados
        results_path = models_dir / f"training_report_{timestamp}.json"
        try:
            with open(results_path, 'w', encoding='utf-8') as f:
                # Preparar resultados para JSON (convertir numpy arrays)
                json_results = {}
                for key, value in self.results.items():
                    if isinstance(value, np.ndarray):
                        json_results[key] = value.tolist()
                    elif hasattr(value, 'to_dict'):  # DataFrame
                        json_results[key] = value.to_dict('records')
                    else:
                        json_results[key] = value
                
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 Reporte guardado en: {results_path}")
            
        except Exception as e:
            logger.warning(f"⚠️  Error guardando reporte: {e}")
            results_path = "No guardado"
        
        return model_path, results_path
    
    def run_complete_pipeline(self):
        """Ejecuta el pipeline completo de ML"""
        logger.info("🚀 INICIANDO PIPELINE COMPLETO DE ML")
        logger.info("=" * 60)
        
        try:
            # 1. Cargar y preparar datos
            self.load_and_prepare_data()
            
            # 2. Preprocesar
            self.preprocess_features()
            
            # 3. Dividir datos
            self.split_data()
            
            # 4. Validación cruzada inicial
            self.cross_validation()
            
            # 5. Optimizar hiperparámetros  
            best_model = self.optimize_hyperparameters()
            
            # 6. Entrenar modelo final
            self.train_final_model(best_model)
            
            # 7. Evaluar
            self.evaluate_model()
            
            # 8. Análisis de features
            self.feature_importance()
            
            # 9. Guardar resultados
            model_path, results_path = self.save_model_and_results()
            
            logger.info("=" * 60)
            logger.info("🎉 PIPELINE COMPLETADO EXITOSAMENTE")
            logger.info(f"📊 Accuracy final: {self.results['test_accuracy']:.4f}")
            logger.info(f"💾 Modelo: {model_path}")
            logger.info(f"📋 Reporte: {results_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Función principal"""
    logger.info("🎾 SISTEMA DE ML PARA PREDICCIONES DE TENIS")
    logger.info("Optimizado para dataset de 70 partidos")
    logger.info("-" * 50)
    
    # Crear pipeline
    pipeline = TennisMLPipeline()
    
    # Ejecutar pipeline completo
    success = pipeline.run_complete_pipeline()
    
    if success:
        logger.info("🎉 ¡Pipeline ejecutado con éxito!")
        logger.info("El modelo está listo para hacer predicciones.")
    else:
        logger.error("💥 Pipeline falló. Revisar logs para detalles.")


if __name__ == "__main__":
    main()
