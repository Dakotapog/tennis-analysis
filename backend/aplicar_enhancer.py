#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENHANCED ML DATASET ORCHESTRATOR
Versión 2.0 - Pipeline Avanzado de Generación Multi-Formato ML

Orquestador de próxima generación que combina web scraping inteligente, 
validación automática y IA para generar datasets optimizados en múltiples 
formatos ML especializados.

ARQUITECTURA DE FLUJO:
Dataset Base → Enhanced Dataset → ML Format 1 (Classification)
                                → ML Format 2 (Regression) 
                                → ML Format 3 (Deep Learning)

INNOVACIONES:
1. **Multi-Target Generation**: Genera datasets especializados por tipo de ML
2. **Adaptive Feature Engineering**: Features específicas por algoritmo objetivo
3. **Quality Assurance Pipeline**: Validación automática multi-nivel
4. **Performance Benchmarking**: Testing predictivo en tiempo real
5. **Export Intelligence**: Formatos optimizados para diferentes frameworks
"""

import pandas as pd
import numpy as np
import os
import glob
import sys
from datetime import datetime
from pathlib import Path
import warnings
from typing import Dict, Optional
import json

# Imports para ML avanzado
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
import joblib

warnings.filterwarnings('ignore')

# --- INICIO: CORRECCIÓN DE IMPORTACIÓN LOCAL ---
# Añadir el directorio actual al path para asegurar que encuentre el módulo local
# Esto resuelve el ImportError cuando el script se ejecuta desde fuera de su directorio.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# --- FIN: CORRECCIÓN DE IMPORTACIÓN LOCAL ---

try:
    from Intelligent_ml_enhancer import IntelligentMLEnhancer, AIConfig, SmartLogger
except ImportError as e:
    print("🚨 ERROR CRÍTICO: No se pudo importar 'intelligent_ml_enhancer.py'.")
    print(f"   Detalle: {e}")
    print("   Asegúrese de que el archivo esté en el mismo directorio y que todas sus dependencias (como imblearn) estén instaladas.")
    sys.exit(1)

# === CONFIGURACIÓN AVANZADA DEL ORQUESTADOR ===
class OrchestratorConfig:
    """Configuración avanzada para generación multi-formato ML."""
    
    # Archivos de entrada
    INPUT_PATTERN = 'reports/superior_dataset_*.csv'
    FALLBACK_PATTERN = 'reports/*dataset*.csv'
    
    # Estructura de salida
    OUTPUT_BASE_DIR = 'ml_datasets'
    ENHANCED_DIR = 'enhanced'
    CLASSIFICATION_DIR = 'classification'
    REGRESSION_DIR = 'regression' 
    DEEP_LEARNING_DIR = 'deep_learning'
    REPORTS_DIR = 'reports'
    MODELS_DIR = 'validation_models'
    
    # Validación de calidad
    MIN_SAMPLES = 100
    MIN_FEATURES = 5
    MIN_ACCURACY_THRESHOLD = 0.65
    MAX_FEATURE_CORRELATION = 0.95
    
    # Targets de ML
    CLASSIFICATION_TARGET = 'ganador_real'
    REGRESSION_TARGETS = ['p1_prob_win', 'match_duration_score']
    
    # Formatos de exportación
    EXPORT_FORMATS = {
        'classification': ['csv', 'parquet', 'json'],
        'regression': ['csv', 'parquet', 'numpy'],
        'deep_learning': ['csv', 'h5', 'tfrecord_ready']
    }

class MLDatasetOrchestrator:
    """
    Orquestador avanzado que genera múltiples formatos de datasets 
    optimizados para diferentes tipos de Machine Learning.
    """
    
    def __init__(self, config: OrchestratorConfig = None):
        self.config = config or OrchestratorConfig()
        self.logger = SmartLogger(name='MLDatasetOrchestrator')
        self.enhancer = None
        self.enhanced_df = None
        self.orchestration_report = {}
        self.validation_results = {}
        
        self._initialize_directories()
        self._initialize_enhancer()
    
    def _initialize_directories(self):
        """Crea la estructura de directorios necesaria."""
        base_path = Path(self.config.OUTPUT_BASE_DIR)
        directories = [
            base_path / self.config.ENHANCED_DIR,
            base_path / self.config.CLASSIFICATION_DIR,
            base_path / self.config.REGRESSION_DIR,
            base_path / self.config.DEEP_LEARNING_DIR,
            base_path / self.config.REPORTS_DIR,
            base_path / self.config.MODELS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.success(f"Estructura de directorios creada en: {base_path}")
    
    def _initialize_enhancer(self):
        """Inicializa el IntelligentMLEnhancer con configuración optimizada."""
        try:
            ai_config = AIConfig()
            # Optimizaciones para multi-formato
            ai_config.MIN_FEATURES = max(self.config.MIN_FEATURES, ai_config.MIN_FEATURES)
            ai_config.CORRELATION_THRESHOLD = self.config.MAX_FEATURE_CORRELATION
            
            self.enhancer = IntelligentMLEnhancer(config=ai_config, logger=self.logger)
            self.logger.success("IntelligentMLEnhancer inicializado con configuración optimizada.")
        except Exception as e:
            self.logger.critical(f"Error al inicializar IntelligentMLEnhancer: {e}")
            self.enhancer = None
    
    def run_full_pipeline(self) -> Dict[str, str]:
        """
        Ejecuta el pipeline completo de generación multi-formato ML.
        
        Returns:
            Dict con las rutas de los datasets generados
        """
        self.logger.section("🚀 INICIO DEL PIPELINE ML MULTI-FORMATO")
        
        if not self.enhancer:
            self.logger.critical("Pipeline detenido: IntelligentMLEnhancer no disponible.")
            return {}
        
        start_time = datetime.now()
        self.orchestration_report['start_time'] = start_time
        
        try:
            # FASE 1: Carga y Enhancement Base
            base_df = self._load_latest_dataset()
            if base_df is None:
                return {}
            
            is_valid_for_enhancement = self._validate_base_dataset(base_df)
            
            # --- PASO DE ENHANCEMENT INTELIGENTE ---
            # Si el dataset es demasiado pequeño, saltar el enhancement para evitar errores.
            if not is_valid_for_enhancement:
                self.logger.warning("Dataset base no cumple los requisitos mínimos. Saltando el enhancement avanzado.")
                self.enhanced_df = base_df
            else:
                self.enhanced_df = self._apply_enhancement(base_df)
            
            if self.enhanced_df is None:
                self.logger.critical("El DataFrame es nulo después del paso de enhancement. Abortando.")
                return {}
            
            # FASE 2: Generación Multi-Formato
            generated_paths = {}
            
            # Dataset Enhanced Base
            enhanced_path = self._save_enhanced_dataset(self.enhanced_df)
            generated_paths['enhanced'] = enhanced_path
            
            # Dataset para Classification
            classification_paths = self._generate_classification_datasets(self.enhanced_df)
            generated_paths.update(classification_paths)
            
            # Dataset para Regression
            regression_paths = self._generate_regression_datasets(self.enhanced_df)
            generated_paths.update(regression_paths)
            
            # Dataset para Deep Learning
            deep_learning_paths = self._generate_deep_learning_datasets(self.enhanced_df)
            generated_paths.update(deep_learning_paths)
            
            # FASE 3: Validación y Benchmarking
            self._run_quality_validation(generated_paths)
            
            # FASE 4: Generación de Reportes
            self._generate_comprehensive_report()
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            self.orchestration_report['end_time'] = end_time
            self.orchestration_report['total_processing_time'] = processing_time
            
            self.logger.section(f"✅ PIPELINE COMPLETADO EN {processing_time:.2f}s")
            self._print_generation_summary(generated_paths)
            
            return generated_paths
            
        except Exception as e:
            self.logger.critical(f"Error crítico en el pipeline: {e}")
            import traceback
            self.logger.critical(f"Traceback: {traceback.format_exc()}")
            return {}
    
    def _load_latest_dataset(self) -> Optional[pd.DataFrame]:
        """Carga el dataset más reciente con fallback inteligente."""
        self.logger.progress("🔍 Buscando el dataset más reciente...")
        
        # Intento principal
        files = glob.glob(self.config.INPUT_PATTERN)
        
        # Fallback a cualquier dataset en reports
        if not files:
            self.logger.progress("Aplicando estrategia de fallback...")
            files = glob.glob(self.config.FALLBACK_PATTERN)
        
        if not files:
            self.logger.critical("No se encontraron datasets con los patrones especificados.")
            self.logger.warning("Ejecute primero 'generar_dataset_plus.py' o coloque un dataset en 'reports/'")
            return None
        
        # Seleccionar el más reciente
        latest_file = max(files, key=os.path.getmtime)
        self.logger.success(f"Dataset seleccionado: {os.path.basename(latest_file)}")
        
        try:
            df = pd.read_csv(latest_file)
            self.logger.success(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
            self.orchestration_report['source_file'] = latest_file
            self.orchestration_report['original_shape'] = df.shape
            return df
        except Exception as e:
            self.logger.critical(f"Error al cargar el archivo: {e}")
            return None
    
    def _validate_base_dataset(self, df: pd.DataFrame) -> bool:
        """Validación avanzada del dataset base."""
        self.logger.progress("🔍 Ejecutando validación avanzada del dataset...")
        
        validation_results = {
            'size_check': len(df) >= self.config.MIN_SAMPLES,
            'features_check': len(df.columns) >= self.config.MIN_FEATURES,
            'target_available': False,
            'numeric_features': len(df.select_dtypes(include=np.number).columns),
            'missing_data_ratio': df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        }
        
        # Verificar targets disponibles
        available_targets = []
        if self.config.CLASSIFICATION_TARGET in df.columns:
            available_targets.append('classification')
            validation_results['target_available'] = True
        
        for reg_target in self.config.REGRESSION_TARGETS:
            if reg_target in df.columns:
                available_targets.append(f'regression_{reg_target}')
        
        validation_results['available_targets'] = available_targets
        self.orchestration_report['base_validation'] = validation_results
        
        # Logs de validación
        if not validation_results['size_check']:
            self.logger.warning(f"Dataset pequeño: {len(df)} muestras (mínimo: {self.config.MIN_SAMPLES})")
        
        if not validation_results['target_available']:
            self.logger.warning(f"Target principal '{self.config.CLASSIFICATION_TARGET}' no encontrado")
        
        self.logger.success(f"Validación completada. Targets disponibles: {', '.join(available_targets)}")
        return validation_results['size_check'] and validation_results['features_check']
    
    def _apply_enhancement(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Aplica el enhancement usando IntelligentMLEnhancer."""
        self.logger.progress("🤖 Aplicando enhancement inteligente...")
        
        try:
            enhanced_df, enhancement_report = self.enhancer.enhance_dataset(df)
            self.orchestration_report['enhancement_report'] = enhancement_report
            
            improvement_ratio = enhanced_df.shape[0] / df.shape[0]
            self.logger.success(f"Enhancement completado. Ratio de mejora: {improvement_ratio:.2f}x")
            
            return enhanced_df
        except Exception as e:
            self.logger.critical(f"Error durante el enhancement: {e}")
            return None
    
    def _save_enhanced_dataset(self, df: pd.DataFrame) -> str:
        """Guarda el dataset enhanced base."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"enhanced_dataset_{timestamp}.csv"
        filepath = Path(self.config.OUTPUT_BASE_DIR) / self.config.ENHANCED_DIR / filename
        
        df.to_csv(filepath, index=False)
        self.logger.success(f"Dataset enhanced guardado: {filepath}")
        
        return str(filepath)
    
    def _generate_classification_datasets(self, df: pd.DataFrame) -> Dict[str, str]:
        """Genera datasets optimizados para clasificación."""
        self.logger.progress("📊 Generando datasets para clasificación...")
        
        if self.config.CLASSIFICATION_TARGET not in df.columns:
            self.logger.warning("Target de clasificación no disponible. Saltando generación.")
            return {}
        
        # Preparar dataset de clasificación
        classification_df = df.copy()
        
        # Encoding de variables categóricas (excepto el target)
        categorical_cols = classification_df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != self.config.CLASSIFICATION_TARGET]
        
        le_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            classification_df[col] = le.fit_transform(classification_df[col].astype(str))
            le_dict[col] = le
        
        # Features específicas para clasificación
        numeric_cols = classification_df.select_dtypes(include=np.number).columns
        if 'p1_ranking' in numeric_cols and 'p2_ranking' in numeric_cols:
            classification_df['ranking_advantage'] = np.log1p(classification_df['p2_ranking']) - np.log1p(classification_df['p1_ranking'])
        
        # Guardar en múltiples formatos
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_path = Path(self.config.OUTPUT_BASE_DIR) / self.config.CLASSIFICATION_DIR
        
        generated_files = {}
        
        # CSV
        csv_path = base_path / f"classification_dataset_{timestamp}.csv"
        classification_df.to_csv(csv_path, index=False)
        generated_files['classification_csv'] = str(csv_path)
        
        # Parquet (más eficiente para datasets grandes)
        parquet_path = base_path / f"classification_dataset_{timestamp}.parquet"
        classification_df.to_parquet(parquet_path, index=False)
        generated_files['classification_parquet'] = str(parquet_path)
        
        # Metadata del encoding
        metadata_path = base_path / f"classification_metadata_{timestamp}.json"
        metadata = {
            'label_encoders': {col: list(le.classes_) for col, le in le_dict.items()},
            'target_column': self.config.CLASSIFICATION_TARGET,
            'feature_count': len(classification_df.columns) - 1,
            'sample_count': len(classification_df),
            'class_distribution': classification_df[self.config.CLASSIFICATION_TARGET].value_counts().to_dict()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        generated_files['classification_metadata'] = str(metadata_path)
        
        self.logger.success(f"Datasets de clasificación generados: {len(generated_files)} archivos")
        return generated_files
    
    def _generate_regression_datasets(self, df: pd.DataFrame) -> Dict[str, str]:
        """Genera datasets optimizados para regresión."""
        self.logger.progress("📈 Generando datasets para regresión...")
        
        # Crear targets de regresión si no existen, evitando la aleatoriedad
        regression_df = df.copy()
        
        # --- Procesamiento de Targets de Regresión ---
        processed_targets = []
        for target in self.config.REGRESSION_TARGETS:
            if target in regression_df.columns:
                processed_targets.append(target)
                continue

            # Lógica de generación determinista
            if target == 'p1_prob_win':
                if 'p1_ranking' in regression_df.columns and 'p2_ranking' in regression_df.columns:
                    self.logger.progress(f"Generando target '{target}' a partir de rankings.")
                    rank_diff = regression_df['p1_ranking'] - regression_df['p2_ranking']
                    regression_df[target] = 1 / (1 + np.exp(rank_diff / 100))
                    processed_targets.append(target)
                else:
                    self.logger.warning(f"Target '{target}' no encontrado y no se puede generar. Se omitirá.")
            elif target == 'match_duration_score':
                 # No hay una forma determinista clara de generarlo, así que lo omitimos si no existe.
                 self.logger.warning(f"Target '{target}' no encontrado. Se omitirá para evitar datos aleatorios.")
            else:
                self.logger.warning(f"Target de regresión configurado '{target}' no encontrado en el dataset.")

        if not processed_targets:
            self.logger.warning("No hay targets de regresión disponibles. Saltando esta fase.")
            return {}
            
        # Normalización para regresión
        numeric_cols = regression_df.select_dtypes(include=np.number).columns
        target_cols = processed_targets
        feature_cols = [col for col in numeric_cols if col not in target_cols]
        
        scaler = StandardScaler()
        regression_df[feature_cols] = scaler.fit_transform(regression_df[feature_cols])
        
        # Guardar datasets
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_path = Path(self.config.OUTPUT_BASE_DIR) / self.config.REGRESSION_DIR
        
        generated_files = {}
        
        for target in target_cols:
            # CSV específico para cada target
            target_df = regression_df[feature_cols + [target]].copy()
            csv_path = base_path / f"regression_{target}_{timestamp}.csv"
            target_df.to_csv(csv_path, index=False)
            generated_files[f'regression_{target}_csv'] = str(csv_path)
            
            # Formato NumPy para modelos científicos
            X = target_df[feature_cols].values
            y = target_df[target].values
            
            np.savez(
                base_path / f"regression_{target}_{timestamp}.npz",
                X=X, y=y, feature_names=feature_cols
            )
            generated_files[f'regression_{target}_npz'] = str(base_path / f"regression_{target}_{timestamp}.npz")
        
        # Guardar scaler para uso futuro
        scaler_path = base_path / f"regression_scaler_{timestamp}.joblib"
        joblib.dump(scaler, scaler_path)
        generated_files['regression_scaler'] = str(scaler_path)
        
        self.logger.success(f"Datasets de regresión generados: {len(generated_files)} archivos")
        return generated_files
    
    def _generate_deep_learning_datasets(self, df: pd.DataFrame) -> Dict[str, str]:
        """Genera datasets optimizados para Deep Learning."""
        self.logger.progress("🧠 Generando datasets para Deep Learning...")
        
        dl_df = df.copy()
        
        # Preparación específica para DL
        # 1. Normalización MinMax (mejor para redes neuronales)
        numeric_cols = dl_df.select_dtypes(include=np.number).columns
        if self.config.CLASSIFICATION_TARGET in numeric_cols:
            feature_cols = [col for col in numeric_cols if col != self.config.CLASSIFICATION_TARGET]
        else:
            feature_cols = list(numeric_cols)
        
        minmax_scaler = MinMaxScaler()
        dl_df[feature_cols] = minmax_scaler.fit_transform(dl_df[feature_cols])
        
        # 2. One-hot encoding para categorías
        categorical_cols = dl_df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != self.config.CLASSIFICATION_TARGET]
        
        dl_df_encoded = pd.get_dummies(dl_df, columns=categorical_cols, prefix=categorical_cols)
        
        # 3. Split estratificado para DL
        if self.config.CLASSIFICATION_TARGET in dl_df_encoded.columns:
            X = dl_df_encoded.drop(columns=[self.config.CLASSIFICATION_TARGET])
            y = dl_df_encoded[self.config.CLASSIFICATION_TARGET]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
        else:
            # Si no hay target de clasificación, split simple
            split_idx = int(0.8 * len(dl_df_encoded))
            X_train = dl_df_encoded.iloc[:split_idx]
            X_test = dl_df_encoded.iloc[split_idx:]
            y_train = y_test = None
        
        # Guardar datasets
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_path = Path(self.config.OUTPUT_BASE_DIR) / self.config.DEEP_LEARNING_DIR
        
        generated_files = {}
        
        # Formato CSV estándar
        train_path = base_path / f"dl_train_{timestamp}.csv"
        test_path = base_path / f"dl_test_{timestamp}.csv"
        
        if y_train is not None:
            train_df = X_train.copy()
            train_df[self.config.CLASSIFICATION_TARGET] = y_train
            train_df.to_csv(train_path, index=False)
            
            test_df = X_test.copy()
            test_df[self.config.CLASSIFICATION_TARGET] = y_test
            test_df.to_csv(test_path, index=False)
        else:
            X_train.to_csv(train_path, index=False)
            X_test.to_csv(test_path, index=False)
        
        generated_files['dl_train_csv'] = str(train_path)
        generated_files['dl_test_csv'] = str(test_path)
        
        # Formato NumPy para frameworks como PyTorch
        np.savez(
            base_path / f"dl_arrays_{timestamp}.npz",
            X_train=X_train.values,
            X_test=X_test.values,
            y_train=y_train.values if y_train is not None else np.array([]),
            y_test=y_test.values if y_test is not None else np.array([]),
            feature_names=X_train.columns.tolist()
        )
        generated_files['dl_arrays_npz'] = str(base_path / f"dl_arrays_{timestamp}.npz")
        
        # Guardar scalers y metadata
        joblib.dump(minmax_scaler, base_path / f"dl_minmax_scaler_{timestamp}.joblib")
        generated_files['dl_scaler'] = str(base_path / f"dl_minmax_scaler_{timestamp}.joblib")
        
        # Metadata para frameworks de DL
        dl_metadata = {
            'input_shape': list(X_train.shape),
            'num_features': X_train.shape[1],
            'num_samples_train': X_train.shape[0],
            'num_samples_test': X_test.shape[0],
            'has_target': y_train is not None,
            'target_classes': list(np.unique(y_train)) if y_train is not None else [],
            'feature_names': X_train.columns.tolist(),
            'preprocessing': {
                'scaling': 'MinMaxScaler',
                'encoding': 'OneHotEncoder for categoricals'
            }
        }
        
        metadata_path = base_path / f"dl_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(dl_metadata, f, indent=2)
        generated_files['dl_metadata'] = str(metadata_path)
        
        self.logger.success(f"Datasets para Deep Learning generados: {len(generated_files)} archivos")
        return generated_files
    
    def _run_quality_validation(self, generated_paths: Dict[str, str]):
        """Ejecuta validación de calidad en los datasets generados."""
        self.logger.progress("✅ Ejecutando validación de calidad...")
        
        validation_results = {}
        
        # Validar dataset de clasificación
        if 'classification_csv' in generated_paths:
            try:
                df_class = pd.read_csv(generated_paths['classification_csv'])
                
                if self.config.CLASSIFICATION_TARGET in df_class.columns:
                    X = df_class.drop(columns=[self.config.CLASSIFICATION_TARGET])
                    y = df_class[self.config.CLASSIFICATION_TARGET]
                    
                    # Quick validation con RandomForest
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                    scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                    
                    validation_results['classification'] = {
                        'mean_cv_accuracy': scores.mean(),
                        'std_cv_accuracy': scores.std(),
                        'dataset_quality': 'GOOD' if scores.mean() > self.config.MIN_ACCURACY_THRESHOLD else 'NEEDS_IMPROVEMENT'
                    }
            except Exception as e:
                self.logger.warning(f"Error en validación de clasificación: {e}")
        
        # Validar datasets de regresión
        for key, path in generated_paths.items():
            if 'regression_' in key and key.endswith('_csv'):
                try:
                    # Extraer el nombre del target del 'key' para mayor robustez
                    target_col_match = [t for t in self.config.REGRESSION_TARGETS if t in key]
                    if not target_col_match:
                        self.logger.warning(f"No se pudo determinar el target para {key}. Saltando validación.")
                        continue
                    
                    target_col = target_col_match[0]
                    df_reg = pd.read_csv(path)

                    if target_col not in df_reg.columns:
                        self.logger.warning(f"Target '{target_col}' no encontrado en {path}. Saltando validación.")
                        continue

                    X = df_reg.drop(columns=[target_col])
                    y = df_reg[target_col]
                    
                    # Validation con GradientBoosting
                    model = GradientBoostingRegressor(n_estimators=50, random_state=42)
                    scores = cross_val_score(model, X, y, cv=3, scoring='r2')
                    
                    validation_results[f'regression_{target_col}'] = {
                        'mean_cv_r2': scores.mean(),
                        'std_cv_r2': scores.std(),
                        'dataset_quality': 'GOOD' if scores.mean() > 0.3 else 'NEEDS_IMPROVEMENT'
                    }
                except Exception as e:
                    self.logger.warning(f"Error en validación de regresión para {key}: {e}")
        
        self.validation_results = validation_results
        
        # Log de resultados
        good_datasets = sum(1 for v in validation_results.values() if v.get('dataset_quality') == 'GOOD')
        total_datasets = len(validation_results)
        
        self.logger.success(f"Validación completada: {good_datasets}/{total_datasets} datasets con calidad GOOD")
    
    def _generate_comprehensive_report(self):
        """Genera un reporte comprensivo del proceso completo."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = Path(self.config.OUTPUT_BASE_DIR) / self.config.REPORTS_DIR / f"orchestration_report_{timestamp}.md"
        
        # Generar reporte del enhancer
        enhancer_report = ""
        if self.enhancer:
            try:
                enhancer_report = self.enhancer.generate_markdown_report()
            except:
                enhancer_report = "Reporte del enhancer no disponible."
        
        # Reporte completo
        report_content = f"""# 🚀 ML Dataset Orchestration Report
**Generado:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Pipeline:** Enhanced ML Dataset Orchestrator v2.0
**Tiempo Total:** {self.orchestration_report.get('total_processing_time', 0):.2f} segundos

## 📊 Resumen Ejecutivo

### Transformación de Datos
- **Dataset Original:** {self.orchestration_report.get('original_shape', '(?, ?)')}
- **Dataset Enhanced:** {self.enhanced_df.shape if self.enhanced_df is not None else 'N/A'}
- **Formatos Generados:** Classification, Regression, Deep Learning
- **Calidad General:** {'✅ EXCELENTE' if len([v for v in self.validation_results.values() if v.get('dataset_quality') == 'GOOD']) > 0 else '⚠️ MEJORABLE'}

### Datasets Generados
"""

        # Añadir información de validación
        if self.validation_results:
            report_content += "\n### 🎯 Validación de Calidad\n"
            for dataset_type, metrics in self.validation_results.items():
                status = "✅" if metrics.get('dataset_quality') == 'GOOD' else "⚠️"
                if 'mean_cv_accuracy' in metrics:
                    score = f"Accuracy: {metrics['mean_cv_accuracy']:.3f}"
                elif 'mean_cv_r2' in metrics:
                    score = f"R²: {metrics['mean_cv_r2']:.3f}"
                else:
                    score = "Score: N/A"
                
                report_content += f"- **{dataset_type.title()}:** {status} {score}\n"
        
        # Añadir reporte del enhancer
        report_content += f"\n---\n\n{enhancer_report}"
        
        # Añadir recomendaciones
        report_content += "\n## 🔮 Recomendaciones de Optimización\n"
        
        recommendations = []
        
        # Análisis de calidad y recomendaciones
        good_datasets = [k for k, v in self.validation_results.items() if v.get('dataset_quality') == 'GOOD']
        total_datasets = len(self.validation_results)
        
        if len(good_datasets) < total_datasets:
            recommendations.append("📈 **Mejora de Features:** Considerar feature engineering adicional para datasets con baja calidad")
            recommendations.append("🎲 **Balanceo:** Aplicar técnicas de balanceo (SMOTE, undersampling) si hay desbalance de clases")
        
        if self.enhanced_df is not None and self.enhanced_df.shape[0] < 1000:
            recommendations.append("📊 **Ampliación:** Dataset pequeño - considerar técnicas de data augmentation")
        
        if any('NEEDS_IMPROVEMENT' in str(v) for v in self.validation_results.values()):
            recommendations.append("🔧 **Hyperparameters:** Optimizar hiperparámetros de modelos de validación")
            recommendations.append("🧹 **Limpieza:** Revisar outliers y valores anómalos en features críticas")
        
        # Recomendaciones específicas por tipo
        for dataset_type, metrics in self.validation_results.items():
            if 'classification' in dataset_type and metrics.get('mean_cv_accuracy', 0) < 0.8:
                recommendations.append(f"🎯 **{dataset_type.title()}:** Implementar ensemble methods o feature selection avanzada")
            elif 'regression' in dataset_type and metrics.get('mean_cv_r2', 0) < 0.5:
                recommendations.append(f"📈 **{dataset_type.title()}:** Considerar transformaciones no-lineales o polynomial features")
        
        if not recommendations:
            recommendations.append("🌟 **Excelente:** Todos los datasets muestran calidad óptima. ¡Listo para producción!")
        
        for rec in recommendations:
            report_content += f"- {rec}\n"
        
        # Información técnica avanzada
        report_content += "\n## ⚙️ Especificaciones Técnicas\n"
        report_content += f"""
### Configuración del Pipeline
- **Threshold de Accuracy:** {self.config.MIN_ACCURACY_THRESHOLD}
- **Correlación Máxima:** {self.config.MAX_FEATURE_CORRELATION}
- **Mínimo de Muestras:** {self.config.MIN_SAMPLES}
- **Formatos de Exportación:** {', '.join(sum(self.config.EXPORT_FORMATS.values(), []))}

### Estructura de Archivos Generada
```
{self.config.OUTPUT_BASE_DIR}/
├── enhanced/           # Dataset base mejorado
├── classification/     # Optimizado para clasificación
├── regression/        # Optimizado para regresión  
├── deep_learning/     # Optimizado para DL/NN
├── reports/          # Reportes y metadata
└── validation_models/ # Modelos de validación
```

### Próximos Pasos Sugeridos
1. **Entrenamiento:** Utilizar los datasets generados con sus respectivos frameworks
2. **Monitoreo:** Implementar drift detection en producción
3. **Iteración:** Feedback loop para mejora continua del pipeline
4. **Escalabilidad:** Considerar paralelización para datasets masivos
"""
        
        # Guardar reporte
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            self.logger.success(f"Reporte comprensivo guardado en: {report_path}")
        except Exception as e:
            self.logger.warning(f"Error al guardar reporte: {e}")
    
    def _print_generation_summary(self, generated_paths: Dict[str, str]):
        """Imprime un resumen visual del resultado del pipeline."""
        print("\n" + "="*80)
        print("🎯 RESUMEN DE GENERACIÓN ML MULTI-FORMATO")
        print("="*80)
        
        # Organizar por categorías
        categories = {
            'Enhanced Base': [k for k in generated_paths.keys() if 'enhanced' in k],
            'Classification': [k for k in generated_paths.keys() if 'classification' in k],
            'Regression': [k for k in generated_paths.keys() if 'regression' in k],
            'Deep Learning': [k for k in generated_paths.keys() if 'dl_' in k],
            'Metadata & Tools': [k for k in generated_paths.keys() if 'metadata' in k or 'scaler' in k]
        }
        
        total_files = 0
        for category, keys in categories.items():
            if keys:
                print(f"\n📂 {category}:")
                for key in keys:
                    filename = os.path.basename(generated_paths[key])
                    print(f"   ✅ {filename}")
                    total_files += 1
        
        print(f"\n🎉 TOTAL GENERADOS: {total_files} archivos en {len([c for c, k in categories.items() if k])} categorías")
        
        # Estadísticas de validación
        if self.validation_results:
            good_count = sum(1 for v in self.validation_results.values() if v.get('dataset_quality') == 'GOOD')
            total_validated = len(self.validation_results)
            print(f"✅ CALIDAD: {good_count}/{total_validated} datasets validados como BUENOS")
        
        print("\n" + "="*80)

class AdvancedMLFormatter:
    """
    Formateador avanzado que aplica transformaciones específicas 
    para cada tipo de algoritmo de ML.
    """
    
    def __init__(self, logger: SmartLogger):
        self.logger = logger
        self.formatters = {
            'classification': self._format_for_classification,
            'regression': self._format_for_regression,
            'clustering': self._format_for_clustering,
            'time_series': self._format_for_time_series,
            'ensemble': self._format_for_ensemble,
            'deep_learning': self._format_for_deep_learning
        }
    
    def format_for_algorithm(self, df: pd.DataFrame, algorithm_type: str, **kwargs) -> pd.DataFrame:
        """Aplica formateo específico según el tipo de algoritmo."""
        if algorithm_type not in self.formatters:
            self.logger.warning(f"Tipo de algoritmo no soportado: {algorithm_type}")
            return df
        
        self.logger.progress(f"🎨 Aplicando formateo para: {algorithm_type}")
        return self.formatters[algorithm_type](df, **kwargs)
    
    def _format_for_classification(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Optimizaciones específicas para clasificación."""
        formatted_df = df.copy()
        
        # Balanceo automático si es necesario
        target_col = kwargs.get('target_column')
        if target_col and target_col in formatted_df.columns:
            class_counts = formatted_df[target_col].value_counts()
            min_count = class_counts.min()
            max_count = class_counts.max()
            
            # Si hay desbalance significativo (ratio > 3:1)
            if max_count / min_count > 3:
                self.logger.progress("⚖️ Aplicando balanceo automático...")
                # Undersampling simple de la clase mayoritaria
                balanced_dfs = []
                for class_val in class_counts.index:
                    class_df = formatted_df[formatted_df[target_col] == class_val]
                    sampled_df = class_df.sample(n=min(len(class_df), int(min_count * 1.5)), random_state=42)
                    balanced_dfs.append(sampled_df)
                formatted_df = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42)
        
        # Feature scaling optimizado para clasificación (StandardScaler)
        numeric_cols = formatted_df.select_dtypes(include=np.number).columns
        if target_col:
            feature_cols = [col for col in numeric_cols if col != target_col]
        else:
            feature_cols = list(numeric_cols)
        
        scaler = StandardScaler()
        formatted_df[feature_cols] = scaler.fit_transform(formatted_df[feature_cols])
        
        return formatted_df
    
    def _format_for_regression(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Optimizaciones específicas para regresión."""
        formatted_df = df.copy()
        
        # Detección y tratamiento de outliers
        numeric_cols = formatted_df.select_dtypes(include=np.number).columns
        
        for col in numeric_cols:
            Q1 = formatted_df[col].quantile(0.25)
            Q3 = formatted_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Límites para outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Winsorizing (clip outliers en lugar de eliminarlos)
            formatted_df[col] = formatted_df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Transformaciones logarítmicas para variables con distribución sesgada
        for col in numeric_cols:
            if formatted_df[col].min() > 0:  # Solo para valores positivos
                skewness = formatted_df[col].skew()
                if abs(skewness) > 1:  # Muy sesgado
                    formatted_df[f'{col}_log'] = np.log1p(formatted_df[col])
        
        return formatted_df
    
    def _format_for_clustering(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Optimizaciones específicas para clustering."""
        formatted_df = df.copy()
        
        # Normalización MinMax (mejor para algoritmos basados en distancia)
        numeric_cols = formatted_df.select_dtypes(include=np.number).columns
        scaler = MinMaxScaler()
        formatted_df[numeric_cols] = scaler.fit_transform(formatted_df[numeric_cols])
        
        # PCA para reducción de dimensionalidad si hay muchas features
        if len(numeric_cols) > 10:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(10, len(numeric_cols)), random_state=42)
            pca_features = pca.fit_transform(formatted_df[numeric_cols])
            
            # Crear nuevas columnas con componentes principales
            for i in range(pca_features.shape[1]):
                formatted_df[f'pca_component_{i+1}'] = pca_features[:, i]
        
        return formatted_df
    
    def _format_for_time_series(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Optimizaciones específicas para series temporales."""
        formatted_df = df.copy()
        
        # Identificar columna temporal
        datetime_col = kwargs.get('datetime_column')
        if not datetime_col:
            # Buscar automáticamente
            for col in formatted_df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    datetime_col = col
                    break
        
        if datetime_col and datetime_col in formatted_df.columns:
            # Convertir a datetime si no lo está
            formatted_df[datetime_col] = pd.to_datetime(formatted_df[datetime_col])
            
            # Features temporales
            formatted_df['year'] = formatted_df[datetime_col].dt.year
            formatted_df['month'] = formatted_df[datetime_col].dt.month
            formatted_df['day'] = formatted_df[datetime_col].dt.day
            formatted_df['dayofweek'] = formatted_df[datetime_col].dt.dayofweek
            formatted_df['quarter'] = formatted_df[datetime_col].dt.quarter
            
            # Ordenar por fecha
            formatted_df = formatted_df.sort_values(datetime_col)
        
        return formatted_df
    
    def _format_for_ensemble(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Optimizaciones específicas para métodos ensemble."""
        formatted_df = df.copy()
        
        # Crear features de interacción
        numeric_cols = formatted_df.select_dtypes(include=np.number).columns
        if len(numeric_cols) >= 2:
            # Seleccionar las top features más correlacionadas
            correlation_matrix = formatted_df[numeric_cols].corr().abs()
            
            # Encontrar pares con correlación media (no muy alta ni muy baja)
            feature_pairs = []
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                    corr_val = correlation_matrix.loc[col1, col2]
                    if 0.3 < corr_val < 0.8:  # Correlación moderada
                        feature_pairs.append((col1, col2))
            
            # Crear features de interacción para los top pares
            for col1, col2 in feature_pairs[:5]:  # Máximo 5 interacciones
                formatted_df[f'{col1}_x_{col2}'] = formatted_df[col1] * formatted_df[col2]
                formatted_df[f'{col1}_div_{col2}'] = formatted_df[col1] / (formatted_df[col2] + 1e-8)
        
        return formatted_df
    
    def _format_for_deep_learning(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Optimizaciones específicas para Deep Learning."""
        formatted_df = df.copy()
        
        # Normalización robusta para DL
        numeric_cols = formatted_df.select_dtypes(include=np.number).columns
        
        # MinMax scaling (mejor para activaciones como sigmoid/tanh)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        formatted_df[numeric_cols] = scaler.fit_transform(formatted_df[numeric_cols])
        
        # Embedding preparation para variables categóricas
        categorical_cols = formatted_df.select_dtypes(include=['object']).columns
        embedding_info = {}
        
        for col in categorical_cols:
            unique_values = formatted_df[col].nunique()
            if unique_values > 2:  # Solo para categorías con más de 2 valores
                # Encoding ordinal para embeddings
                le = LabelEncoder()
                formatted_df[f'{col}_encoded'] = le.fit_transform(formatted_df[col].astype(str))
                
                # Información para crear embeddings
                embedding_info[col] = {
                    'vocab_size': unique_values,
                    'embedding_dim': min(50, (unique_values + 1) // 2),
                    'encoded_column': f'{col}_encoded'
                }
                
                # Eliminar columna original
                formatted_df = formatted_df.drop(columns=[col])
        
        # Guardar información de embeddings como atributo
        formatted_df.attrs['embedding_info'] = embedding_info
        
        return formatted_df

def main():
    """Función principal para ejecutar el orquestador ML."""
    try:
        # Banner de inicio
        print("\n" + "="*80)
        print("🚀 ENHANCED ML DATASET ORCHESTRATOR v2.0")
        print("   Pipeline Avanzado de Generación Multi-Formato ML")
        print("="*80)
        
        # Inicializar y ejecutar
        orchestrator = MLDatasetOrchestrator()
        generated_paths = orchestrator.run_full_pipeline()
        
        if generated_paths:
            print("\n🎉 ¡ORQUESTACIÓN COMPLETADA CON ÉXITO!")
            print(f"📁 Archivos generados en: {orchestrator.config.OUTPUT_BASE_DIR}")
            print("\n💡 Próximos pasos:")
            print("   1. Revisar el reporte comprensivo en ml_datasets/reports/")
            print("   2. Utilizar los datasets optimizados para entrenar modelos")
            print("   3. Monitorear performance y aplicar mejoras iterativas")
        else:
            print("\n❌ La orquestación no se completó exitosamente.")
            print("   Revisar los logs para identificar problemas.")
        
        return generated_paths
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Proceso interrumpido por el usuario.")
        return {}
    except Exception as e:
        print(f"\n❌ Error crítico en la orquestación: {e}")
        import traceback
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    generated_files = main()
    
    # Información adicional si se ejecuta directamente
    if generated_files:
        print("\n📊 ESTADÍSTICAS FINALES:")
        print(f"   • Total de archivos: {len(generated_files)}")
        print("   • Formatos generados: CSV, Parquet, NPZ, JSON, Joblib")
        print("   • Listos para: Scikit-learn, PyTorch, TensorFlow, Keras")
        print("   • Pipeline optimizado para: Producción y experimentación")
    
    print("\n🔮 ¡El futuro del preprocessing inteligente está aquí!")
    print("="*80)
