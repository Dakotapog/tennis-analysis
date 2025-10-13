#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SISTEMA INTELIGENTE DE GENERACIÓN DE DATASET ANTI-COLAPSO ML
Versión 5.2 - FEATURE ENGINEERING Y ROBUSTEZ MEJORADA
Enfoque: Dataset balanceado, features discriminativas, patrones complejos
"""

import json
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
import logging
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.impute import KNNImputer
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURACIÓN AVANZADA ===
class MLConfig:
    """Configuración inteligente para prevenir colapso de modelo."""
    
    # Umbrales críticos para detección de problemas
    DESBALANCE_CRITICO_THRESHOLD = 0.75  # Máximo 75% para una clase
    VARIABILIDAD_MINIMA_CV = 0.1  # Coeficiente de variación mínimo
    COMPETITIVIDAD_MINIMA = 0.4  # 40% de enfrentamientos competitivos mínimo
    PATRON_TRIVIAL_THRESHOLD = 0.85  # Máximo 85% de patrón obvio
    CORRELACION_MAXIMA = 0.95  # Máxima correlación entre features
    
    # Objetivos de calidad del dataset
    TARGET_BALANCE_RATIO = 0.6  # Ratio ideal entre clases: 60-40
    TARGET_MIN_SAMPLES = 200  # Mínimo de samples por clase
    TARGET_FEATURE_COUNT = 25  # Número objetivo de features discriminativas
    
    # Nuevas constantes para validación de clases
    MAX_ALLOWED_CLASSES = 2  # Solo player1/player2
    VALID_CLASS_NAMES = ['player1', 'player2']

    # Configuración de features inteligentes
    FEATURES_CORE = [
        'p1_ranking', 'p2_ranking', 'p1_elo', 'p2_elo', 
        'ranking_diff', 'elo_diff'
    ]
    
    FEATURES_AVANZADAS = [
        'p1_win_rate', 'p2_win_rate', 'p1_recent_form', 'p2_recent_form',
        'surface_advantage_p1', 'surface_advantage_p2', 'h2h_advantage',
        'momentum_p1', 'momentum_p2', 'fatigue_factor_p1', 'fatigue_factor_p2',
        'p1_form_win_percentage', 'p1_form_wins', 'p1_form_losses', 'p1_current_streak_count',
        'p2_form_win_percentage', 'p2_form_wins', 'p2_form_losses', 'p2_current_streak_count',
        'h2h_total_matches', 'h2h_jugador1_wins', 'h2h_jugador2_wins', 'h2h_win_rate_jugador1',
        'p1_pts', 'p2_pts', 'p1_prox_pts', 'p2_prox_pts', 'p1_pts_max', 'p2_pts_max',
        'p1_defense_points', 'p2_defense_points',
        'p1_home_win_rate', 'p2_home_win_rate', 'p1_regional_comfort_win_rate', 'p2_regional_comfort_win_rate'
    ]

# === SISTEMA DE LOGGING AVANZADO ===
class SmartLogger:
    def __init__(self):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = f'logs/intelligent_dataset_{self.timestamp}.log'
        os.makedirs('logs', exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def critical_alert(self, message):
        """Alerta crítica que requiere atención inmediata."""
        self.logger.critical(f"🚨 CRÍTICO: {message}")
        
    def ml_warning(self, message):
        """Advertencia específica de ML."""
        self.logger.warning(f"🤖 ML: {message}")
        
    def success(self, message):
        """Mensaje de éxito."""
        self.logger.info(f"✅ {message}")
        
    def progress(self, message):
        """Progreso del proceso."""
        self.logger.info(f"🔄 {message}")

# === ANALIZADOR INTELIGENTE DE DATOS ===
class IntelligentDataAnalyzer:
    """Analizador que detecta y corrige problemas automáticamente."""
    
    def __init__(self, logger):
        self.logger = logger
        self.problems_detected = []
        self.solutions_applied = []
        
    def analyze_class_distribution(self, dataset):
        """Análisis inteligente de distribución de clases."""
        self.logger.progress("Analizando distribución de clases...")
        
        ganadores = [item.get('actual_winner') for item in dataset if item.get('actual_winner')]
        if not ganadores:
            return {'error': 'No hay ganadores definidos', 'severity': 'critical'}
            
        counter = Counter(ganadores)
        total = len(ganadores)

        # Validación crítica: asegurar que el sistema sea binario
        if len(counter) > MLConfig.MAX_ALLOWED_CLASSES:
            self.logger.critical_alert(f"Se detectaron {len(counter)} clases únicas en el target. Se requiere un sistema binario.")
            self.logger.ml_warning(f"Clases detectadas: {list(counter.keys())[:5]}...") # Muestra las primeras 5
            return {
                'error': f'Sistema detectó {len(counter)} clases. Se requieren exactamente {MLConfig.MAX_ALLOWED_CLASSES} ({MLConfig.VALID_CLASS_NAMES})',
                'severity': 'critical',
                'detected_classes': list(counter.keys())
            }
        
        # Calcular métricas de distribución
        distribution_stats = {}
        for ganador, count in counter.items():
            percentage = (count / total) * 100
            distribution_stats[ganador] = {
                'count': count,
                'percentage': percentage,
                'is_problematic': percentage > (MLConfig.DESBALANCE_CRITICO_THRESHOLD * 100)
            }
        
        # Detectar problemas críticos
        max_percentage = max([stats['percentage'] for stats in distribution_stats.values()]) if distribution_stats else 0
        min_percentage = min([stats['percentage'] for stats in distribution_stats.values()]) if distribution_stats else 0
        
        imbalance_ratio = max_percentage / min_percentage if min_percentage > 0 else float('inf')
        is_critical = max_percentage > (MLConfig.DESBALANCE_CRITICO_THRESHOLD * 100)
        
        if is_critical:
            self.problems_detected.append(f"Desbalance crítico: {max_percentage:.1f}% vs {min_percentage:.1f}%")
            
        return {
            'distribution': distribution_stats,
            'imbalance_ratio': imbalance_ratio,
            'is_critical': is_critical,
            'max_percentage': max_percentage,
            'min_percentage': min_percentage,
            'severity': 'critical' if is_critical else 'acceptable'
        }
    
    def analyze_feature_discriminative_power(self, df):
        """Analiza el poder discriminativo de las features para evitar predicciones idénticas."""
        self.logger.progress("Analizando poder discriminativo de features...")
        
        if 'actual_winner' not in df.columns:
            return {'error': 'No hay columna de target', 'severity': 'critical'}
        
        # Encoder para el target
        le = LabelEncoder()
        y_encoded = le.fit_transform(df['actual_winner'].dropna())
        
        discriminative_analysis = {}
        problematic_features = []
        
        features_to_analyze = list(set(MLConfig.FEATURES_CORE + MLConfig.FEATURES_AVANZADAS))

        for feature in features_to_analyze:
            if feature not in df.columns:
                continue
                
            feature_data = df[feature].dropna()
            if len(feature_data) < 10:
                continue
            
            # Calcular poder discriminativo usando correlación con target
            feature_target_df = df[[feature, 'actual_winner']].dropna()
            if len(feature_target_df) < 10:
                continue
                
            # Para features numéricas, usar correlación punto-biserial
            try:
                correlation = abs(stats.pointbiserialr(
                    le.fit_transform(feature_target_df['actual_winner']),
                    feature_target_df[feature]
                )[0])
            except:
                correlation = 0
            
            # Calcular variabilidad
            cv = feature_data.std() / feature_data.mean() if feature_data.mean() != 0 else 0
            unique_ratio = len(feature_data.unique()) / len(feature_data)
            
            discriminative_analysis[feature] = {
                'correlation_with_target': correlation,
                'coefficient_variation': cv,
                'unique_ratio': unique_ratio,
                'is_discriminative': correlation > 0.1 and cv > MLConfig.VARIABILIDAD_MINIMA_CV,
                'sample_size': len(feature_data)
            }
            
            # Detectar features problemáticas
            if correlation < 0.05 or cv < MLConfig.VARIABILIDAD_MINIMA_CV:
                problematic_features.append(f"{feature}: baja discriminación (corr={correlation:.3f}, cv={cv:.3f})")
        
        self.problems_detected.extend(problematic_features)
        
        return {
            'analysis': discriminative_analysis,
            'problematic_features': problematic_features,
            'severity': 'critical' if len(problematic_features) > len(features_to_analyze) // 2 else 'warning'
        }
    
    def detect_trivial_patterns(self, df):
        """Detecta patrones triviales que causan predicciones idénticas."""
        self.logger.progress("Detectando patrones triviales...")
        
        trivial_patterns = []
        
        # Patrón 1: ¿Siempre gana el mejor rankeado?
        if all(col in df.columns for col in ['p1_ranking', 'p2_ranking', 'actual_winner']):
            clean_df = df[['p1_ranking', 'p2_ranking', 'actual_winner']].dropna()
            
            if len(clean_df) > 10:
                clean_df['mejor_rankeado'] = clean_df.apply(
                    lambda row: 'player1' if row['p1_ranking'] < row['p2_ranking'] else 'player2', 
                    axis=1
                )
                
                matches = (clean_df['mejor_rankeado'] == clean_df['actual_winner']).sum()
                percentage = (matches / len(clean_df)) * 100
                
                if percentage > (MLConfig.PATRON_TRIVIAL_THRESHOLD * 100):
                    trivial_patterns.append(f"Patrón ranking trivial: {percentage:.1f}% predictibilidad")
        
        # Patrón 2: ¿Diferencias ELO muy uniformes?
        if all(col in df.columns for col in ['p1_elo', 'p2_elo']):
            elo_df = df[['p1_elo', 'p2_elo']].dropna()
            if len(elo_df) > 10:
                elo_diff = abs(elo_df['p1_elo'] - elo_df['p2_elo'])
                if elo_diff.std() < 50:  # Diferencias muy uniformes
                    trivial_patterns.append(f"Diferencias ELO uniformes: std={elo_diff.std():.1f}")
        
        # Patrón 3: ¿Una superficie domina excesivamente?
        if 'tipo_cancha' in df.columns:
            surface_counts = df['tipo_cancha'].value_counts()
            if len(surface_counts) > 0:
                dominant_surface_pct = (surface_counts.iloc[0] / len(df)) * 100
                if dominant_surface_pct > 80:
                    trivial_patterns.append(f"Superficie dominante: {dominant_surface_pct:.1f}%")
        
        self.problems_detected.extend(trivial_patterns)
        
        return {
            'patterns': trivial_patterns,
            'severity': 'critical' if len(trivial_patterns) > 0 else 'acceptable'
        }

# === GENERADOR INTELIGENTE DE FEATURES ===
class IntelligentFeatureGenerator:
    """Genera features inteligentes que mejoran la discriminación del modelo."""
    
    def __init__(self, logger):
        self.logger = logger
        
    def generate_advanced_features(self, df):
        """Genera features avanzadas para mejorar el poder predictivo."""
        self.logger.progress("Generando features avanzadas...")
        
        enhanced_df = df.copy()
        
        # Features de diferencia (críticas para evitar predicciones idénticas)
        if all(col in df.columns for col in ['p1_ranking', 'p2_ranking']):
            enhanced_df['ranking_diff'] = df['p1_ranking'] - df['p2_ranking']
            enhanced_df['ranking_ratio'] = df['p1_ranking'] / (df['p2_ranking'] + 1)
            enhanced_df['is_upset_potential'] = (enhanced_df['ranking_diff'] > 50).astype(int)
        
        if all(col in df.columns for col in ['p1_elo', 'p2_elo']):
            enhanced_df['elo_diff'] = df['p1_elo'] - df['p2_elo']
            enhanced_df['elo_advantage'] = np.where(enhanced_df['elo_diff'] > 0, 1, 0)
            enhanced_df['elo_competitive'] = (abs(enhanced_df['elo_diff']) < 100).astype(int)
        
        # Features de superficie (si está disponible)
        if 'tipo_cancha' in df.columns:
            surface_dummies = pd.get_dummies(df['tipo_cancha'], prefix='surface')
            enhanced_df = pd.concat([enhanced_df, surface_dummies], axis=1)
        
        # Features de balance competitivo
        enhanced_df['match_competitiveness'] = 0
        if 'ranking_diff' in enhanced_df.columns:
            enhanced_df['match_competitiveness'] = np.where(
                abs(enhanced_df['ranking_diff']) <= 10, 3,  # Muy competitivo
                np.where(abs(enhanced_df['ranking_diff']) <= 50, 2,  # Competitivo
                         np.where(abs(enhanced_df['ranking_diff']) <= 100, 1, 0))  # Poco competitivo
            )
        
        self.logger.success(f"Features generadas: {len(enhanced_df.columns) - len(df.columns)} nuevas")
        return enhanced_df
    
    def balance_dataset_intelligently(self, df):
        """Balancea el dataset de forma inteligente preservando la calidad."""
        self.logger.progress("Balanceando dataset inteligentemente...")
        
        if 'actual_winner' not in df.columns or df['actual_winner'].isna().all():
            self.logger.ml_warning("No hay datos de target para balancear.")
            return df
        
        class_counts = df['actual_winner'].value_counts()
        
        if len(class_counts) < 2:
            self.logger.ml_warning("Solo una clase presente, no se puede balancear.")
            return df

        # --- LÓGICA DE IDENTIFICACIÓN DE CLASES A PRUEBA DE ERRORES ---
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()
        majority_count = class_counts.max()
        minority_count = class_counts.min()

        # --- VALIDACIÓN DE ESTRATEGIA ---
        MIN_SAMPLES_FOR_VALIDATION = 5
        is_critically_imbalanced = minority_count < MIN_SAMPLES_FOR_VALIDATION
        is_balanced_enough = (majority_count / minority_count) < 1.8 if minority_count > 0 else True

        if is_balanced_enough and not is_critically_imbalanced:
            self.logger.success("Dataset ya está suficientemente balanceado.")
            return df
        
        if is_critically_imbalanced:
            self.logger.ml_warning(
                f"Clase minoritaria ('{minority_class}') tiene solo {minority_count} "
                f"ejemplo(s). Se aplicará solo sobremuestreo."
            )
            
        majority_df = df[df['actual_winner'] == majority_class]
        minority_df = df[df['actual_winner'] == minority_class]

        # --- LÓGICA DE BALANCEO REFINADA ---
        if is_critically_imbalanced:
            # Solo sobremuestrear la minoría para no perder datos de la mayoría
            minority_balanced = resample(minority_df,
                                         n_samples=MIN_SAMPLES_FOR_VALIDATION * 2, # ej. 10
                                         random_state=42,
                                         replace=True)
            # Mantener la mayoría intacta
            majority_balanced = majority_df
        else:
            # Estrategia híbrida para casos no críticos
            self.logger.progress("Aplicando estrategia de balanceo híbrida.")
            target_size = int((majority_count + minority_count) / 2)
            
            if 'match_competitiveness' in majority_df.columns:
                majority_balanced = majority_df.nlargest(target_size, 'match_competitiveness')
            else:
                majority_balanced = majority_df.sample(n=target_size, random_state=42)
            
            minority_balanced = resample(minority_df, 
                                       n_samples=target_size,
                                       random_state=42, 
                                       replace=True)
        
        # Combinar y registrar
        balanced_df = pd.concat([majority_balanced, minority_balanced], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Log con nombres de clase para claridad
        final_counts = balanced_df['actual_winner'].value_counts()
        self.logger.success(
            f"Dataset balanceado: {final_counts.get(majority_class, 0)} ('{majority_class}') vs "
            f"{final_counts.get(minority_class, 0)} ('{minority_class}')"
        )
        return balanced_df

# === VALIDADOR INTELIGENTE DE ML ===
class IntelligentMLValidator:
    """Validador que predice y previene fallos del modelo."""
    
    def __init__(self, logger):
        self.logger = logger
        
    def validate_ml_readiness(self, df):
        """Validación completa de preparación para ML."""
        self.logger.progress("Validando preparación para ML...")
        
        validation_results = {
            'is_ready': True,
            'critical_issues': [],
            'warnings': [],
            'ml_metrics': {},
            'recommendations': []
        }
        
        # Test 1: Simulación de entrenamiento rápido
        ml_simulation = self._simulate_ml_training(df)
        validation_results['ml_metrics']['simulation'] = ml_simulation
        
        if ml_simulation['prediction_diversity'] < 0.3:
            validation_results['critical_issues'].append(
                f"Baja diversidad de predicciones: {ml_simulation['prediction_diversity']:.3f}"
            )
            validation_results['is_ready'] = False
        
        # Test 2: Análisis de correlaciones
        correlation_analysis = self._analyze_feature_correlations(df)
        validation_results['ml_metrics']['correlations'] = correlation_analysis
        
        # Test 3: Validación de separabilidad de clases
        separability = self._analyze_class_separability(df)
        validation_results['ml_metrics']['separability'] = separability
        
        if 'error' not in separability and separability.get('overlap_percentage', 0) > 80:
            validation_results['critical_issues'].append(
                f"Clases muy solapadas: {separability.get('overlap_percentage', 0):.1f}%"
            )
            validation_results['is_ready'] = False
        elif 'error' in separability:
            validation_results['warnings'].append(
                f"No se pudo analizar la separabilidad de clases: {separability['error']}"
            )
        
        # Generar recomendaciones
        validation_results['recommendations'] = self._generate_ml_recommendations(validation_results)
        
        return validation_results
    
    def _simulate_ml_training(self, df):
        """Simula entrenamiento para detectar problemas predictivos."""
        try:
            # Preparar datos
            feature_cols = [col for col in df.columns 
                          if col not in ['actual_winner', 'match_url', 'player1', 'player2', 'fecha', 'p1_last_match_date', 'p2_last_match_date', 'p1_form_status', 'p2_form_status', 'p1_current_streak_type', 'p2_current_streak_type', 'tipo_cancha', 'torneo_nombre', 'jugador1', 'jugador2'] 
                          and df[col].dtype in ['int64', 'float64']]
            
            if len(feature_cols) < 3:
                return {'error': 'Insuficientes features numéricas', 'prediction_diversity': 0}
            
            X = df[feature_cols].fillna(df[feature_cols].median())
            y = df['actual_winner']
            
            # Split rápido
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Entrenar modelo simple
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train, y_train)
            
            # Predecir
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)
            
            # Analizar diversidad de predicciones
            pred_diversity = len(set(predictions)) / len(set(y)) if len(set(y)) > 0 else 0
            prob_variance = np.var(probabilities.max(axis=1))
            
            return {
                'prediction_diversity': pred_diversity,
                'probability_variance': prob_variance,
                'accuracy': (predictions == y_test).mean(),
                'feature_count': len(feature_cols)
            }
            
        except Exception as e:
            return {'error': str(e), 'prediction_diversity': 0}
    
    def _analyze_feature_correlations(self, df):
        """Analiza correlaciones entre features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {'error': 'Insuficientes columnas numéricas'}
        
        corr_matrix = df[numeric_cols].corr()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > MLConfig.CORRELACION_MAXIMA:
                    high_corr_pairs.append(( 
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_val
                    ))
        
        return {
            'high_correlations': high_corr_pairs,
            'max_correlation': corr_matrix.abs().max().max() if len(corr_matrix) > 0 else 0
        }
    
    def _analyze_class_separability(self, df):
        """Analiza qué tan separables son las clases."""
        if 'actual_winner' not in df.columns:
            return {'error': 'No target column'}
        
        # Usar features numéricas principales
        feature_cols = [col for col in ['p1_ranking', 'p2_ranking', 'p1_elo', 'p2_elo', 
                                       'ranking_diff', 'elo_diff'] 
                       if col in df.columns]
        
        if len(feature_cols) < 2:
            return {'error': 'Insuficientes features para análisis'}
        
        # Análisis simple de solapamiento usando percentiles
        classes = df['actual_winner'].unique()
        if len(classes) != 2:
            return {'error': 'Se requieren exactamente 2 clases'}
        
        overlap_scores = []
        for feature in feature_cols[:3]:  # Analizar top 3 features
            class1_data = df[df['actual_winner'] == classes[0]][feature].dropna()
            class2_data = df[df['actual_winner'] == classes[1]][feature].dropna()
            
            if len(class1_data) < 5 or len(class2_data) < 5:
                continue
            
            # Calcular solapamiento usando rangos intercuartílicos
            q1_c1, q3_c1 = class1_data.quantile([0.25, 0.75])
            q1_c2, q3_c2 = class2_data.quantile([0.25, 0.75])
            
            overlap = max(0, min(q3_c1, q3_c2) - max(q1_c1, q1_c2))
            total_range = max(q3_c1, q3_c2) - min(q1_c1, q1_c2)
            
            if total_range > 0:
                overlap_pct = (overlap / total_range) * 100
                overlap_scores.append(overlap_pct)
        
        avg_overlap = np.mean(overlap_scores) if overlap_scores else 100
        
        return {
            'overlap_percentage': avg_overlap,
            'features_analyzed': len(overlap_scores),
            'separability_score': 100 - avg_overlap
        }
    
    def _generate_ml_recommendations(self, validation_results):
        """Genera recomendaciones específicas para ML."""
        recommendations = []
        
        # Recomendaciones por problemas críticos
        for issue in validation_results['critical_issues']:
            if 'diversidad' in issue.lower():
                recommendations.extend([
                    "🎯 Añadir más features discriminativas (diferencias de ranking/ELO)",
                    "🎯 Balancear dataset para evitar sesgo hacia una clase",
                    "🎯 Incluir features de interacción (productos y ratios)"
                ])
            
            if 'solapadas' in issue.lower():
                recommendations.extend([
                    "🎯 Filtrar enfrentamientos más competitivos",
                    "🎯 Añadir features contextuales (superficie, forma reciente)",
                    "🎯 Considerar transformaciones no lineales de features"
                ])
        
        # Recomendaciones por métricas
        simulation = validation_results['ml_metrics'].get('simulation', {})
        if simulation.get('feature_count', 0) < 10:
            recommendations.append("🎯 Aumentar número de features (objetivo: >10)")
        
        if not recommendations:
            recommendations.append("✅ Dataset preparado para ML - No se detectaron problemas críticos")
        
        return recommendations

# === ORQUESTADOR PRINCIPAL ===
class IntelligentDatasetGenerator:
    """Orquestador principal del sistema inteligente."""
    
    def __init__(self):
        self.logger = SmartLogger()
        self.analyzer = IntelligentDataAnalyzer(self.logger)
        self.feature_generator = IntelligentFeatureGenerator(self.logger)
        self.ml_validator = IntelligentMLValidator(self.logger)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.target_column = 'actual_winner'
        
    def _intelligent_imputation(self, df):
        """Imputa valores faltantes usando KNN para preservar relaciones."""
        self.logger.progress("Iniciando imputación inteligente con KNN...")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if df[numeric_cols].isnull().sum().sum() == 0:
            self.logger.success("No hay datos numéricos faltantes para imputar.")
            return df
            
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        
        df_imputed_numeric = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols, index=df.index)
        
        df_final = df.copy()
        df_final.update(df_imputed_numeric)

        self.logger.success("Imputación con KNN completada.")
        return df_final

    def _anomaly_detection(self, df):
        """Detecta y elimina outliers usando Isolation Forest."""
        self.logger.progress("Iniciando detección de anomalías con Isolation Forest...")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if not numeric_cols:
            self.logger.warning("No hay columnas numéricas para la detección de anomalías.")
            return df

        df_numeric = df[numeric_cols].fillna(df[numeric_cols].median())

        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        outlier_labels = iso_forest.fit_predict(df_numeric)
        
        initial_rows = len(df)
        df_clean = df[outlier_labels == 1].reset_index(drop=True)
        rows_removed = initial_rows - len(df_clean)
        
        if rows_removed > 0:
            self.logger.success(f"Se eliminaron {rows_removed} anomalías ({rows_removed/initial_rows:.2%}).")
        else:
            self.logger.success("No se detectaron anomalías significativas.")
            
        return df_clean

    def _hybrid_feature_selection(self, df):
        """Selecciona las features más relevantes usando RFE y análisis de correlación."""
        self.logger.progress("Iniciando selección de features híbrida...")
        
        if self.target_column not in df.columns:
            self.logger.critical_alert("Target column no encontrada para la selección de features.")
            return df

        # Codificar target para RFE
        le = LabelEncoder()
        y = le.fit_transform(df[self.target_column])
        
        feature_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
        
        if not feature_cols:
            self.logger.warning("No hay features numéricas para seleccionar.")
            return df

        X = df[feature_cols].fillna(df[feature_cols].median())

        # 1. Eliminar features con alta correlación
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > MLConfig.CORRELACION_MAXIMA)]
        X = X.drop(to_drop, axis=1)
        self.logger.progress(f"Eliminadas {len(to_drop)} features por alta correlación: {', '.join(to_drop)}")

        # 2. RFE para selección final
        num_features = len(X.columns)
        n_features_to_select = min(MLConfig.TARGET_FEATURE_COUNT, num_features)
        if n_features_to_select < 5:
             self.logger.warning(f"Muy pocas features ({num_features}) para una selección robusta. Se mantendrán todas.")
             selected_features = X.columns.tolist()
        else:
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            rfe = RFE(estimator=model, n_features_to_select=n_features_to_select, step=0.1)
            rfe.fit(X, y)
            selected_features = X.columns[rfe.support_].tolist()

        self.logger.success(f"Selección de features completada. {len(selected_features)} features seleccionadas.")
        
        # --- LÓGICA DE PRESERVACIÓN DE FEATURES CRÍTICAS ---
        critical_features_present = [
            'ranking_diff', 'elo_diff', 'match_competitiveness'
        ]
        
        # Asegurar que las features críticas estén en el resultado final
        final_selected_features = list(set(selected_features + critical_features_present))
        
        # Mantener columnas no numéricas
        non_numeric_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        final_cols = final_selected_features + non_numeric_cols
        
        # Eliminar duplicados y preservar el orden
        final_cols = list(dict.fromkeys(final_cols))
        
        # Filtrar solo columnas que realmente existen en el df
        final_cols = [col for col in final_cols if col in df.columns]
        
        self.logger.success(f"Features críticas preservadas. Total final: {len(final_cols)} features.")
        
        return df[final_cols]

    def generate_superior_dataset(self, features_folder='input/features', 
                                labels_folder='input/labels', 
                                output_folder='reports'):
        """Genera un dataset superior con validaciones inteligentes."""
        
        self.logger.logger.info("=" * 80)
        self.logger.logger.info("🧠 SISTEMA INTELIGENTE DE GENERACIÓN DE DATASET V5.2")
        self.logger.logger.info("🎯 OBJETIVO: Dataset superior que evite predicciones idénticas con feature engineering extensivo y robustez mejorada")
        self.logger.logger.info("=" * 80)
        
        try:
            # FASE 1: Carga inteligente de datos
            self.logger.logger.info("--- FASE 1: Carga y Fusión de Datos ---")
            dataset = self._load_and_merge_data(features_folder, labels_folder)
            if not dataset:
                return False
            self._save_staged_dataset(pd.DataFrame(dataset), "1_merged", output_folder)

            # FASE 2: Análisis profundo de problemas
            self.logger.logger.info("--- FASE 2: Análisis Profundo del Dataset ---")
            analysis_results = self._deep_analysis(dataset)
            
            # FASE 3: Generación de features inteligentes
            self.logger.logger.info("--- FASE 3: Ingeniería de Features Inteligentes ---")
            df_enhanced = self.feature_generator.generate_advanced_features(pd.DataFrame(dataset))
            self._save_staged_dataset(df_enhanced, "2_features_enhanced", output_folder)

            # FASE 4: Imputación Inteligente
            self.logger.logger.info("--- FASE 4: Imputación de Datos Faltantes ---")
            df_imputed = self._intelligent_imputation(df_enhanced)
            self._save_staged_dataset(df_imputed, "3_imputed", output_folder)

            # FASE 5: Detección de Anomalías
            self.logger.logger.info("--- FASE 5: Detección y Eliminación de Anomalías ---")
            df_clean = self._anomaly_detection(df_imputed)
            self._save_staged_dataset(df_clean, "4_anomalies_removed", output_folder)

            # FASE 6: Selección de Features Híbrida
            self.logger.logger.info("--- FASE 6: Selección de Features Relevantes ---")
            df_selected = self._hybrid_feature_selection(df_clean)
            self._save_staged_dataset(df_selected, "5_features_selected", output_folder)
            
            # FASE 7: Balanceo inteligente
            self.logger.logger.info("--- FASE 7: Balanceo Inteligente de Clases ---")
            df_balanced = self.feature_generator.balance_dataset_intelligently(df_selected)
            self._save_staged_dataset(df_balanced, "6_balanced", output_folder)
            
            # FASE 8: Validación final de ML
            self.logger.logger.info("--- FASE 8: Validación Crítica de Preparación para ML ---")
            ml_validation = self._final_ml_validation(df_balanced.to_dict('records'))
            
            # FASE 9: Generación de reportes y guardado
            self.logger.logger.info("--- FASE 9: Generación de Salidas y Reportes ---")
            success = self._save_superior_dataset(
                df_balanced.to_dict('records'), analysis_results, ml_validation, output_folder
            )
            
            return success
            
        except Exception as e:
            self.logger.critical_alert(f"Error en generación: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _save_staged_dataset(self, df, stage_name, output_folder):
        """Guarda un dataset intermedio para depuración."""
        try:
            # Crear una subcarpeta 'stages' para no saturar la carpeta principal
            stages_folder = os.path.join(output_folder, 'stages')
            os.makedirs(stages_folder, exist_ok=True)
            
            filename = f"dataset_{self.timestamp}_stage_{stage_name}.csv"
            filepath = os.path.join(stages_folder, filename)
            
            df.to_csv(filepath, index=False, encoding='utf-8')
            self.logger.success(f"Dataset intermedio guardado: {filepath}")
        except Exception as e:
            self.logger.ml_warning(f"No se pudo guardar el dataset intermedio '{stage_name}': {e}")
    
    def _find_latest_file(self, folder, pattern):
        """Encuentra el archivo más reciente que coincide con un patrón."""
        try:
            files = glob.glob(os.path.join(folder, pattern))
            if not files:
                return None
            latest_file = max(files, key=os.path.getctime)
            return latest_file
        except Exception as e:
            self.logger.ml_warning(f"No se pudo encontrar el archivo más reciente en '{folder}': {e}")
            return None

    def _load_and_merge_data(self, features_folder, labels_folder):
        """Carga y combina datos de múltiples archivos de forma inteligente y robusta."""
        self.logger.progress("Cargando y fusionando datos de múltiples archivos...")

        # 1. Cargar TODOS los datos de features (H2H)
        h2h_files = glob.glob(os.path.join(features_folder, "h2h_results_enhanced_*.json"))
        if not h2h_files:
            self.logger.critical_alert(f"No se encontró ningún archivo de features en '{features_folder}'")
            return None
        
        self.logger.progress(f"Encontrados {len(h2h_files)} archivos de features. Procesando...")
        all_h2h_data = []
        for file_path in h2h_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data and isinstance(data, dict):
                        partidos_data = data.get('partidos')
                        if partidos_data and isinstance(partidos_data, list):
                            all_h2h_data.extend(partidos_data)
            except json.JSONDecodeError:
                self.logger.ml_warning(f"Archivo JSON inválido o vacío, omitiendo: {file_path}")

        # 2. Cargar TODOS los datos de labels (resultados)
        labels_files = glob.glob(os.path.join(labels_folder, "resultados_finales_*.json"))
        if not labels_files:
            self.logger.critical_alert(f"No se encontró ningún archivo de labels en '{labels_folder}'")
            return None

        self.logger.progress(f"Encontrados {len(labels_files)} archivos de labels. Procesando...")
        all_labels_data = []
        for file_path in labels_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data and isinstance(data, dict):
                        detailed_results = data.get('detailed_results')
                        if detailed_results and isinstance(detailed_results, list):
                            all_labels_data.extend(detailed_results)
            except json.JSONDecodeError:
                self.logger.ml_warning(f"Archivo JSON inválido o vacío, omitiendo: {file_path}")

        # 3. Procesar y aplanar los datos de H2H combinados
        features_list = []
        for i, partido in enumerate(all_h2h_data):
            try:
                if partido is None:
                    self.logger.ml_warning(f"Se encontró un registro de partido nulo (None) en el índice {i}. Omitiendo.")
                    continue
                flat_feature = self._extract_analysis_metrics({}, partido)
                features_list.append(flat_feature)
            except Exception as e:
                self.logger.critical_alert(f"Error procesando partido en índice {i} con URL {partido.get('match_url', 'URL no encontrada')}: {e}")
                continue

        features_df = pd.DataFrame(features_list)
        if 'match_url' not in features_df.columns:
            self.logger.critical_alert("La clave 'match_url' no se pudo extraer de los datos de features.")
            return None

        self.logger.success(f"Procesados {len(features_df)} registros de features.")

        # 4. Procesar los datos de labels combinados
        labels_list = []
        for result in all_labels_data:
            if result is None:
                self.logger.ml_warning("Se encontró un registro de resultado nulo (None) en los datos de labels. Omitiendo.")
                continue
            if result.get('verification', {}).get('status') == 'Verified':
                match_info = result.get('match_info', {})
                actual_result = result.get('actual_result', {})
                if match_info.get('match_url') and actual_result.get('actual_winner'):
                    labels_list.append({
                        'match_url': match_info['match_url'],
                        'actual_winner': actual_result['actual_winner']
                    })
        
        labels_df = pd.DataFrame(labels_list)
        if 'match_url' not in labels_df.columns:
            self.logger.critical_alert("La clave 'match_url' no se pudo extraer de los datos de labels.")
            return None
            
        self.logger.success(f"Procesados {len(labels_df)} registros de labels verificados.")

        # 5. Fusionar los dataframes
        merged_df = pd.merge(features_df, labels_df, on='match_url', how='inner')
        self.logger.progress(f"Fusionados {len(merged_df)} registros.")

        if merged_df.empty:
            self.logger.critical_alert("El dataset combinado está vacío. Verifica la consistencia de 'match_url' entre los archivos.")
            return None

        # 6. Crear la columna target binaria (con normalización de nombres)
        def normalize_name(name):
            if not isinstance(name, str):
                return ""
            return name.lower().replace('.', '').strip()

        def determine_winner(row):
            winner_norm = normalize_name(row.get('actual_winner'))
            p1_norm = normalize_name(row.get('jugador1'))
            p2_norm = normalize_name(row.get('jugador2'))

            if winner_norm and p1_norm and winner_norm == p1_norm:
                return 'player1'
            elif winner_norm and p2_norm and winner_norm == p2_norm:
                return 'player2'
            else:
                if winner_norm: # Solo loguear si hay un ganador real
                    self.logger.ml_warning(
                        f"No se pudo determinar el ganador para el match {row.get('match_url', '')}. "
                        f"Ganador: '{row.get('actual_winner')}', "
                        f"P1: '{row.get('jugador1')}', P2: '{row.get('jugador2')}'"
                    )
                return None
        
        merged_df[self.target_column] = merged_df.apply(determine_winner, axis=1)
        
        # Eliminar filas donde el ganador no se pudo determinar
        initial_rows = len(merged_df)
        merged_df.dropna(subset=[self.target_column], inplace=True)
        rows_dropped = initial_rows - len(merged_df)
        if rows_dropped > 0:
            self.logger.ml_warning(f"Se eliminaron {rows_dropped} registros por inconsistencias en los nombres de los ganadores.")

        self.logger.success(f"Dataset combinado final con {len(merged_df)} registros válidos.")
        
        # Devolver como lista de diccionarios para mantener la compatibilidad
        return merged_df.to_dict('records')

    def _extract_analysis_metrics(self, record, original_data):
        """
        Extrae un conjunto completo y aplanado de métricas del análisis original,
        aprovechando toda la inteligencia generada por el scraper.
        Versión robusta para manejar datos faltantes o nulos.
        """
        # Datos básicos del partido
        record['match_url'] = original_data.get('match_url')
        record['jugador1'] = original_data.get('jugador1') or original_data.get('player1')
        record['jugador2'] = original_data.get('jugador2') or original_data.get('player2')
        record['torneo_nombre'] = original_data.get('torneo_nombre')
        record['tipo_cancha'] = original_data.get('tipo_cancha', 'Desconocida').capitalize()
        record['cuota1'] = original_data.get('cuota1')
        record['cuota2'] = original_data.get('cuota2')

        player1 = record.get('jugador1', '') or ""
        player2 = record.get('jugador2', '') or ""
        
        # Normalizar nombres para buscar claves dinámicas
        p1_key_name = player1.replace(' ', '_').replace('.', '') if player1 else ""
        p2_key_name = player2.replace(' ', '_').replace('.', '') if player2 else ""

        # --- Análisis de Ranking y Rivalidad (con acceso robusto) ---
        ranking_analysis = original_data.get('ranking_analysis') or {}
        prediction = ranking_analysis.get('prediction') or {}
        scores = prediction.get('scores') or {}
        score_breakdown = prediction.get('score_breakdown') or {}
        p1_breakdown = score_breakdown.get('player1') or {}
        p2_breakdown = score_breakdown.get('player2') or {}

        # Manejo dinámico de claves de ranking
        record['p1_ranking'] = ranking_analysis.get(f'{p1_key_name}_ranking') if p1_key_name else None
        record['p2_ranking'] = ranking_analysis.get(f'{p2_key_name}_ranking') if p2_key_name else None
        if record['p1_ranking'] is None:
            record['p1_ranking'] = ranking_analysis.get('player1_rank')
        if record['p2_ranking'] is None:
            record['p2_ranking'] = ranking_analysis.get('player2_rank')

        record['p1_rivalry_score'] = ranking_analysis.get('p1_rivalry_score')
        record['p2_rivalry_score'] = ranking_analysis.get('p2_rivalry_score')
        record['common_opponents_count'] = ranking_analysis.get('common_opponents_count')

        # --- Scores de Predicción del Scraper ---
        record['prediction_confidence'] = prediction.get('confidence')
        record['p1_predicted_score'] = scores.get('p1_final_weight', 0)
        record['p2_predicted_score'] = scores.get('p2_final_weight', 0)
        record['predicted_score_diff'] = scores.get('score_difference')

        # --- Desglose de Scores Ponderados ---
        components = [
            'surface_specialization', 'form_recent', 'common_opponents', 
            'h2h_direct', 'ranking_momentum', 'home_advantage', 'strength_of_schedule'
        ]
        for component in components:
            p1_component_data = p1_breakdown.get(component) or {}
            p2_component_data = p2_breakdown.get(component) or {}
            record[f'p1_{component}_score'] = float(p1_component_data.get('weighted_score', 0))
            record[f'p2_{component}_score'] = float(p2_component_data.get('weighted_score', 0))

        # --- 1. Forma Reciente (Completo) ---
        form_analysis = original_data.get('form_analysis') or {}
        p1_form = form_analysis.get(f'{p1_key_name}_form') or {} if p1_key_name else {}
        p2_form = form_analysis.get(f'{p2_key_name}_form') or {} if p2_key_name else {}

        record['p1_form_win_percentage'] = p1_form.get('win_percentage')
        record['p1_form_wins'] = p1_form.get('wins')
        record['p1_form_losses'] = p1_form.get('losses')
        record['p1_form_status'] = p1_form.get('form_status')
        record['p1_last_match_date'] = p1_form.get('last_match_date')
        p1_streak_type = 1 if p1_form.get('current_streak_type') == 'victorias' else -1
        record['p1_current_streak_count'] = (p1_form.get('current_streak_count') or 0) * p1_streak_type
        record['p1_current_streak_type'] = p1_form.get('current_streak_type')

        record['p2_form_win_percentage'] = p2_form.get('win_percentage')
        record['p2_form_wins'] = p2_form.get('wins')
        record['p2_form_losses'] = p2_form.get('losses')
        record['p2_form_status'] = p2_form.get('form_status')
        record['p2_last_match_date'] = p2_form.get('last_match_date')
        p2_streak_type = 1 if p2_form.get('current_streak_type') == 'victorias' else -1
        record['p2_current_streak_count'] = (p2_form.get('current_streak_count') or 0) * p2_streak_type
        record['p2_current_streak_type'] = p2_form.get('current_streak_type')

        # --- 2. Análisis por Superficie (Completo) ---
        surface_analysis = original_data.get('surface_analysis') or {}
        current_surface = record['tipo_cancha']
        
        p1_surface_stats = (surface_analysis.get(f'{p1_key_name}_surface_stats') or {}).get(current_surface) or {} if p1_key_name else {}
        p2_surface_stats = (surface_analysis.get(f'{p2_key_name}_surface_stats') or {}).get(current_surface) or {} if p2_key_name else {}
        
        record[f'p1_{current_surface}_win_rate'] = p1_surface_stats.get('win_rate')
        record[f'p1_{current_surface}_wins'] = p1_surface_stats.get('wins')
        record[f'p1_{current_surface}_losses'] = p1_surface_stats.get('losses')
        record[f'p1_{current_surface}_matches'] = p1_surface_stats.get('matches')

        record[f'p2_{current_surface}_win_rate'] = p2_surface_stats.get('win_rate')
        record[f'p2_{current_surface}_wins'] = p2_surface_stats.get('wins')
        record[f'p2_{current_surface}_losses'] = p2_surface_stats.get('losses')
        record[f'p2_{current_surface}_matches'] = p2_surface_stats.get('matches')

        # --- 3. Features de Ubicación/Regional (Completo) ---
        location_analysis = original_data.get('location_analysis') or {}
        p1_loc_stats = location_analysis.get(f'{p1_key_name}_location_stats') or {} if p1_key_name else {}
        p2_loc_stats = location_analysis.get(f'{p2_key_name}_location_stats') or {} if p2_key_name else {}

        record['p1_home_wins'] = (p1_loc_stats.get('home_advantage') or {}).get('wins')
        record['p1_home_losses'] = (p1_loc_stats.get('home_advantage') or {}).get('losses')
        
        record['p2_home_wins'] = (p2_loc_stats.get('home_advantage') or {}).get('wins')
        record['p2_home_losses'] = (p2_loc_stats.get('home_advantage') or {}).get('losses')
        record['p1_home_win_rate'] = (p1_loc_stats.get('home_advantage') or {}).get('win_rate')
        record['p2_home_win_rate'] = (p2_loc_stats.get('home_advantage') or {}).get('win_rate')

        # Lógica para confort regional
        COUNTRY_TO_CONTINENT_MAP = {
            'USA': 'Norteamérica', 'Canada': 'Norteamérica', 'Mexico': 'Norteamérica',
            'Argentina': 'Sudamérica', 'Brazil': 'Sudamérica', 'Chile': 'Sudamérica', 'Colombia': 'Sudamérica', 'Ecuador': 'Sudamérica', 'Uruguay': 'Sudamérica',
            'Spain': 'Europa', 'France': 'Europa', 'Italy': 'Europa', 'Germany': 'Europa', 'Great Britain': 'Europa', 'Russia': 'Europa', 'Serbia': 'Europa', 'Switzerland': 'Europa', 'Sweden': 'Europa', 'Austria': 'Europa', 'Belgium': 'Europa', 'Netherlands': 'Europa', 'Poland': 'Europa', 'Czech Republic': 'Europa', 'Croatia': 'Europa', 'Greece': 'Europa', 'Norway': 'Europa', 'Denmark': 'Europa', 'Finland': 'Europa', 'Portugal': 'Europa', 'Hungary': 'Europa', 'Romania': 'Europa', 'Slovakia': 'Europa', 'Slovenia': 'Europa', 'Ukraine': 'Europa',
            'Australia': 'Oceanía', 'New Zealand': 'Oceanía',
            'Japan': 'Asia', 'China': 'Asia', 'South Korea': 'Asia', 'India': 'Asia', 'Kazakhstan': 'Asia',
            'South Africa': 'África', 'Tunisia': 'África', 'Egypt': 'África'
        }
        
        match_country = (original_data.get('ranking_analysis') or {}).get('prediction_context', {}).get('current_match_country', 'N/A')
        match_continent = COUNTRY_TO_CONTINENT_MAP.get(match_country, 'Otros')

        p1_regional_stats = (p1_loc_stats.get('regional_comfort') or {}).get(match_continent, {})
        p2_regional_stats = (p2_loc_stats.get('regional_comfort') or {}).get(match_continent, {})

        record['p1_regional_comfort_win_rate'] = p1_regional_stats.get('win_rate')
        record['p2_regional_comfort_win_rate'] = p2_regional_stats.get('win_rate')

        # --- 4. Métricas de Puntos Avanzadas (Completo) ---
        record['p1_elo'] = ranking_analysis.get(f'{p1_key_name}_elo') or ranking_analysis.get('player1_elo', 1500)
        record['p2_elo'] = ranking_analysis.get(f'{p2_key_name}_elo') or ranking_analysis.get('player2_elo', 1500)

        p1_metrics = ranking_analysis.get(f'{p1_key_name}_metrics') or {} if p1_key_name else {}
        p2_metrics = ranking_analysis.get(f'{p2_key_name}_metrics') or {} if p2_key_name else {}

        record['p1_pts'] = p1_metrics.get('pts')
        record['p1_prox_pts'] = p1_metrics.get('prox_pts') or p1_metrics.get('PROX')
        record['p1_pts_max'] = p1_metrics.get('pts_max') or p1_metrics.get('MAX')
        record['p1_defense_points'] = p1_metrics.get('defense_points')

        record['p2_pts'] = p2_metrics.get('pts')
        record['p2_prox_pts'] = p2_metrics.get('prox_pts') or p2_metrics.get('PROX')
        record['p2_pts_max'] = p2_metrics.get('pts_max') or p2_metrics.get('MAX')
        record['p2_defense_points'] = p2_metrics.get('defense_points')

        # --- 5. Rivalidad y H2H (Completo) ---
        direct_h2h = original_data.get('enfrentamientos_directos') or []
        p1_h2h_wins = sum(1 for m in direct_h2h if m and m.get('ganador') == player1) if player1 else 0
        p2_h2h_wins = sum(1 for m in direct_h2h if m and m.get('ganador') == player2) if player2 else 0
        
        record['h2h_total_matches'] = len(direct_h2h)
        record['h2h_jugador1_wins'] = p1_h2h_wins
        record['h2h_jugador2_wins'] = p2_h2h_wins
        
        if record['h2h_total_matches'] > 0:
            record['h2h_win_rate_jugador1'] = p1_h2h_wins / record['h2h_total_matches']
        else:
            record['h2h_win_rate_jugador1'] = 0.5 # Neutral value if no H2H

        record['h2h_win_diff'] = p1_h2h_wins - p2_h2h_wins
        
        return record
    
    def _deep_analysis(self, dataset):
        """Análisis profundo de problemas potenciales."""
        self.logger.progress("Realizando análisis profundo...")
        
        df = pd.DataFrame(dataset)
        
        results = {
            'distribution_analysis': self.analyzer.analyze_class_distribution(dataset),
            'discriminative_analysis': self.analyzer.analyze_feature_discriminative_power(df),
            'pattern_analysis': self.analyzer.detect_trivial_patterns(df),
            'dataset_stats': {
                'total_samples': len(df),
                'total_features': len(df.columns),
                'missing_data_percentage': (df.isnull().sum().sum() / df.size) * 100 if df.size > 0 else 0
            }
        }
        
        return results

    def _final_ml_validation(self, dataset):
        """Validación final crítica antes del uso en ML."""
        self.logger.progress("Validación final de preparación ML...")
        
        df = pd.DataFrame(dataset)
        validation_results = self.ml_validator.validate_ml_readiness(df)
        
        # VERIFICACIÓN CRÍTICA: Test de diversidad predictiva
        diversity_test = self._critical_prediction_diversity_test(df)
        validation_results['critical_diversity_test'] = diversity_test
        
        if not diversity_test['passes_test']:
            self.logger.critical_alert(f"FALLO CRÍTICO: {diversity_test['failure_reason']}")
            validation_results['is_ready'] = False
        
        return validation_results
    
    def _critical_prediction_diversity_test(self, df):
        """Test crítico para asegurar diversidad en predicciones."""
        self.logger.progress("Ejecutando test crítico de diversidad predictiva...")
        
        try:
            # --- Lógica de selección de features robusta ---
            self.logger.progress("Seleccionando features robustas para el test de diversidad...")
            
            critical_features = []
            
            # 1. Priorizar features requeridas si tienen variabilidad
            required_features = ['ranking_diff', 'elo_diff', 'match_competitiveness']
            for feature in required_features:
                if feature in df.columns and df[feature].notna().sum() > len(df) * 0.7:
                    if df[feature].std() > 0.01:  # Chequeo de variabilidad
                        critical_features.append(feature)
            
            self.logger.progress(f"Features críticas con variabilidad encontradas: {critical_features}")

            # 2. Si no hay suficientes, buscar features adicionales con alta varianza
            MIN_CRITICAL_FEATURES = 3
            if len(critical_features) < MIN_CRITICAL_FEATURES:
                self.logger.ml_warning(
                    f"No se encontraron suficientes features críticas con variabilidad. "
                    f"Buscando candidatas adicionales..."
                )
                
                # Calcular coeficiente de variación para todas las features numéricas
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                
                # Excluir las ya seleccionadas y las que no son de features
                excluded_cols = critical_features + ['actual_winner', 'match_url'] 
                candidate_features = [col for col in numeric_cols if col not in excluded_cols]
                
                # Calcular CV para las candidatas
                cv_scores = {}
                for feature in candidate_features:
                    if df[feature].notna().sum() > len(df) * 0.5: # Mínimo de datos
                        mean = df[feature].mean()
                        std = df[feature].std()
                        if abs(mean) > 1e-6:
                            cv = std / mean
                            cv_scores[feature] = abs(cv)

                # Ordenar por CV y añadir las mejores
                sorted_candidates = sorted(cv_scores.items(), key=lambda item: item[1], reverse=True)
                
                needed = MIN_CRITICAL_FEATURES - len(critical_features)
                
                for feature, score in sorted_candidates[:needed]:
                    critical_features.append(feature)
                    self.logger.progress(f"Añadiendo feature candidata: {feature} (CV={score:.3f})")

            if len(critical_features) < 3:
                return {
                    'passes_test': False,
                    'failure_reason': f'Insuficientes features con variabilidad encontradas: {len(critical_features)}/{MIN_CRITICAL_FEATURES} mínimas',
                    'features_available': critical_features
                }
            
            # Preparar datos
            X = df[critical_features].fillna(df[critical_features].median())
            y = df['actual_winner']
            
            # Verificar distribución de clases
            class_distribution = y.value_counts(normalize=True)
            if class_distribution.max() > 0.85:
                return {
                    'passes_test': False,
                    'failure_reason': f'Desbalance crítico: {class_distribution.max():.1%} dominancia',
                    'class_distribution': class_distribution.to_dict()
                }
            
            # Test con múltiples modelos para asegurar diversidad
            models_test_results = {}
            
            # Verificar si la estratificación es posible
            min_class_count = y.value_counts().min()
            stratify_option = y if min_class_count >= 2 else None
            
            if stratify_option is None:
                self.logger.ml_warning(
                    f"Una clase tiene solo {min_class_count} miembro(s), "
                    "desactivando estratificación para el test de diversidad."
                )

            # Split estratificado (si es posible)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=stratify_option
            )
            
            # Test 1: Logistic Regression (linear boundary)
            lr_model = LogisticRegression(random_state=42, max_iter=2000, C=1.0)
            lr_model.fit(X_train, y_train)
            lr_predictions = lr_model.predict(X_test)
            lr_probabilities = lr_model.predict_proba(X_test)
            
            models_test_results['logistic_regression'] = {
                'unique_predictions': len(set(lr_predictions)),
                'prediction_diversity': len(set(lr_predictions)) / len(set(y)) if len(set(y)) > 0 else 0,
                'probability_entropy': stats.entropy(np.mean(lr_probabilities, axis=0)),
                'max_class_probability': np.max(np.mean(lr_probabilities, axis=0))
            }
            
            # Test 2: Random Forest (non-linear, ensemble)
            rf_model = RandomForestClassifier(
                n_estimators=50, max_depth=8, random_state=42, 
                min_samples_split=10, min_samples_leaf=5
            )
            rf_model.fit(X_train, y_train)
            rf_predictions = rf_model.predict(X_test)
            rf_probabilities = rf_model.predict_proba(X_test)
            
            models_test_results['random_forest'] = {
                'unique_predictions': len(set(rf_predictions)),
                'prediction_diversity': len(set(rf_predictions)) / len(set(y)) if len(set(y)) > 0 else 0,
                'probability_entropy': stats.entropy(np.mean(rf_probabilities, axis=0)),
                'max_class_probability': np.max(np.mean(rf_probabilities, axis=0)),
                'feature_importance': dict(zip(critical_features, rf_model.feature_importances_))
            }
            
            # CRITERIOS DE APROBACIÓN CRÍTICOS
            diversity_threshold = 0.4  # Al menos 40% de diversidad predictiva
            entropy_threshold = 0.5   # Entropía mínima para evitar colapso
            max_prob_threshold = 0.8  # Máximo 80% de probabilidad promedio para una clase
            
            critical_failures = []
            
            for model_name, results in models_test_results.items():
                if results['prediction_diversity'] < diversity_threshold:
                    critical_failures.append(
                        f"{model_name}: Baja diversidad {results['prediction_diversity']:.3f}"
                    )
                
                if results['probability_entropy'] < entropy_threshold:
                    critical_failures.append(
                        f"{model_name}: Baja entropía {results['probability_entropy']:.3f}"
                    )
                
                if results['max_class_probability'] > max_prob_threshold:
                    critical_failures.append(
                        f"{model_name}: Sesgo excesivo {results['max_class_probability']:.3f}"
                    )
            
            # Test adicional: Verificar separabilidad usando PCA
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            # Calcular separabilidad en espacio PCA
            class_centers = {}
            center_distance = 0
            separability_ratio = 0
            if len(y.unique()) == 2:
                for class_val in y.unique():
                    class_mask = y == class_val
                    class_centers[class_val] = np.mean(X_pca[class_mask], axis=0)
                
                if len(class_centers) == 2:
                    centers = list(class_centers.values())
                    center_distance = np.linalg.norm(centers[0] - centers[1])
                    
                    # Calcular dispersión intra-clase
                    intra_class_variance = 0
                    for class_val, center in class_centers.items():
                        class_mask = y == class_val
                        class_points = X_pca[class_mask]
                        distances = np.linalg.norm(class_points - center, axis=1)
                        intra_class_variance += np.var(distances)
                    
                    separability_ratio = center_distance / (intra_class_variance + 1e-8)
                    
                    if separability_ratio < 0.5:
                        critical_failures.append(f"Clases no separables: ratio {separability_ratio:.3f}")
            
            # RESULTADO FINAL DEL TEST
            passes_test = len(critical_failures) == 0
            
            return {
                'passes_test': passes_test,
                'failure_reason': '; '.join(critical_failures) if critical_failures else None,
                'model_results': models_test_results,
                'features_used': critical_features,
                'separability_analysis': {
                    'center_distance': center_distance,
                    'separability_ratio': separability_ratio
                },
                'recommendations': self._generate_diversity_recommendations(critical_failures, models_test_results)
            }
            
        except Exception as e:
            return {
                'passes_test': False,
                'failure_reason': f'Error en test de diversidad: {str(e)}',
                'error_details': str(e)
            }
    
    def _generate_diversity_recommendations(self, failures, model_results):
        """Genera recomendaciones específicas para mejorar diversidad predictiva."""
        recommendations = []
        
        if not failures:
            recommendations.append("✅ Dataset aprobó todas las validaciones de diversidad")
            return recommendations
        
        # Recomendaciones por tipo de fallo
        for failure in failures:
            if 'diversidad' in failure.lower():
                recommendations.extend([
                    "🔧 CRÍTICO: Aumentar variabilidad en features de diferencia",
                    "🔧 Filtrar solo enfrentamientos competitivos (ranking_diff < 100)",
                    "🔧 Añadir noise controlado a features para evitar patrones rígidos",
                    "🔧 Implementar feature engineering más sofisticado"
                ])
            
            elif 'entropía' in failure.lower():
                recommendations.extend([
                    "🔧 CRÍTICO: Balancear dataset más agresivamente",
                    "🔧 Añadir features contextuales (superficie, forma reciente)",
                    "🔧 Implementar técnicas de oversampling inteligente"
                ])
            
            elif 'sesgo' in failure.lower():
                recommendations.extend([
                    "🔧 CRÍTICO: Revisar etiquetas - posible sesgo sistemático",
                    "🔧 Verificar calidad de datos de entrada",
                    "🔧 Implementar regularización más fuerte en modelos"
                ])
            
            elif 'separables' in failure.lower():
                recommendations.extend([
                    "🔧 CRÍTICO: Añadir features de interacción (productos, ratios)",
                    "🔧 Considerar transformaciones no-lineales",
                    "🔧 Filtrar casos ambiguos o mal etiquetados"
                ])
        
        # Recomendaciones basadas en importancia de features
        if 'random_forest' in model_results:
            rf_importance = model_results['random_forest'].get('feature_importance', {})
            if rf_importance:
                most_important = max(rf_importance.items(), key=lambda x: x[1])
                if most_important[1] > 0.6:  # Una feature domina demasiado
                    recommendations.append(
                        f"⚠️ Feature '{most_important[0]}' domina ({most_important[1]:.2f}) - "
                        "Añadir features complementarias"
                    )
        
        return recommendations
    
    def _save_superior_dataset(self, dataset, analysis_results, ml_validation, output_folder):
        """Guarda el dataset superior con reportes completos."""
        self.logger.progress("Guardando dataset superior y reportes...")
        
        try:
            os.makedirs(output_folder, exist_ok=True)
            
            # Usar el timestamp consistente de la ejecución
            timestamp = self.timestamp
            
            # Guardar dataset principal
            df = pd.DataFrame(dataset)
            dataset_file = os.path.join(output_folder, f'superior_dataset_{timestamp}.csv')
            df.to_csv(dataset_file, index=False, encoding='utf-8')
            
            # Guardar versión JSON para compatibilidad
            json_file = os.path.join(output_folder, f'superior_dataset_{timestamp}.json')
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False, default=str)
            
            # Generar reporte ejecutivo
            executive_report = self._generate_executive_report(
                df, analysis_results, ml_validation
            )
            
            report_file = os.path.join(output_folder, f'executive_report_{timestamp}.json')
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(executive_report, f, indent=2, ensure_ascii=False, default=str)
            
            # Generar reporte técnico detallado
            technical_report = self._generate_technical_report(
                df, analysis_results, ml_validation, timestamp
            )
            
            tech_report_file = os.path.join(output_folder, f'technical_analysis_{timestamp}.md')
            with open(tech_report_file, 'w', encoding='utf-8') as f:
                f.write(technical_report)
            
            # Log de resultados finales
            self.logger.logger.info("=" * 80)
            self.logger.success(f"Dataset superior generado: {len(df)} registros")
            self.logger.success(f"Features totales: {len(df.columns)}")
            self.logger.success(f"Archivos guardados en: {output_folder}")
            
            if ml_validation['is_ready']:
                self.logger.success("✅ VALIDACIÓN ML: Dataset APROBADO para entrenamiento")
            else:
                self.logger.critical_alert("❌ VALIDACIÓN ML: Dataset requiere mejoras antes del entrenamiento")
            
            self.logger.logger.info("=" * 80)
            
            # Devolver el estado real de la validación
            return ml_validation['is_ready']
            
        except Exception as e:
            self.logger.critical_alert(f"Error guardando dataset: {e}")
            return False
    
    def _generate_executive_report(self, df, analysis_results, ml_validation):
        """Genera reporte ejecutivo para stakeholders."""
        
        # Métricas clave
        total_samples = len(df)
        feature_count = len(df.columns)
        class_distribution = df['actual_winner'].value_counts(normalize=True).to_dict()
        
        # Estado de validación ML
        ml_status = "APROBADO ✅" if ml_validation['is_ready'] else "REQUIERE MEJORAS ❌"
        
        # Problemas críticos identificados
        critical_issues = []
        if analysis_results.get('distribution_analysis', {}).get('is_critical'):
            critical_issues.append("Desbalance crítico de clases")
        
        if ml_validation.get('critical_diversity_test', {}).get('passes_test') == False:
            critical_issues.append("Falta de diversidad predictiva")
        
        if len(analysis_results.get('pattern_analysis', {}).get('patterns', [])) > 0:
            critical_issues.append("Patrones triviales detectados")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'dataset_summary': {
                'total_samples': total_samples,
                'total_features': feature_count,
                'class_distribution': class_distribution,
                'missing_data_percentage': round((df.isnull().sum().sum() / df.size) * 100, 2) if df.size > 0 else 0
            },
            'ml_readiness': {
                'status': ml_status,
                'is_ready': ml_validation['is_ready'],
                'critical_issues': critical_issues,
                'confidence_score': self._calculate_confidence_score(analysis_results, ml_validation)
            },
            'key_improvements': {
                'features_engineered': feature_count - len(MLConfig.FEATURES_CORE),
                'balancing_applied': abs(max(class_distribution.values()) - min(class_distribution.values())) < 0.3 if class_distribution else False,
                'diversity_validated': ml_validation.get('critical_diversity_test', {}).get('passes_test', False)
            },
            'recommendations': ml_validation.get('recommendations', []),
            'next_steps': self._generate_next_steps(ml_validation)
        }
    
    def _calculate_confidence_score(self, analysis_results, ml_validation):
        """Calcula score de confianza del dataset (0-100)."""
        score = 100
        
        # Penalizaciones por problemas
        if analysis_results.get('distribution_analysis', {}).get('is_critical'):
            score -= 25
        
        if not ml_validation.get('critical_diversity_test', {}).get('passes_test', True):
            score -= 30
        
        if len(analysis_results.get('pattern_analysis', {}).get('patterns', [])) > 0:
            score -= 15
        
        # Bonificaciones por calidad
        diversity_test = ml_validation.get('critical_diversity_test', {})
        if diversity_test.get('passes_test'):
            score += 10
        
        model_results = diversity_test.get('model_results', {})
        if model_results:
            avg_diversity = np.mean([
                results.get('prediction_diversity', 0) 
                for results in model_results.values()
            ])
            if avg_diversity > 0.6:
                score += 5
        
        return max(0, min(100, score))
    
    def _generate_next_steps(self, ml_validation):
        """Genera pasos siguientes recomendados."""
        if ml_validation['is_ready']:
            return [
                "1. Proceder con entrenamiento de modelos ML",
                "2. Implementar validación cruzada estratificada",
                "3. Monitorear performance en datos de prueba",
                "4. Configurar pipeline de predicción en producción"
            ]
        else:
            next_steps = ["1. Resolver problemas críticos identificados:"]
            critical_issues = ml_validation.get('critical_issues', [])
            for i, issue in enumerate(critical_issues, 2):
                next_steps.append(f"{i}. {issue}")
            
            next_steps.extend([
                f"{len(critical_issues) + 2}. Re-ejecutar validaciones ML",
                f"{len(critical_issues) + 3}. Repetir ciclo hasta aprobación"
            ])
            return next_steps
    
    def _generate_technical_report(self, df, analysis_results, ml_validation, timestamp):
        """Genera reporte técnico detallado en Markdown."""
        
        report = f"""# Reporte Técnico - Sistema Anti-Colapso ML
**Generado:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Versión:** 5.2 - Feature Engineering y Robustez Mejorada

## 🎯 Resumen Ejecutivo
- **Samples Total:** {len(df):,}
- **Features Generadas:** {len(df.columns)}
- **Estado ML:** {'✅ APROBADO' if ml_validation['is_ready'] else '❌ REQUIERE MEJORAS'}

## 📊 Análisis de Distribución de Clases
"""
        
        class_dist = df['actual_winner'].value_counts()
        for clase, count in class_dist.items():
            percentage = (count / len(df)) * 100 if len(df) > 0 else 0
            report += f"- **{clase}:** {count:,} ({percentage:.1f}%)\n"
        
        dist_analysis = analysis_results.get('distribution_analysis', {})
        if dist_analysis.get('is_critical'):
            report += f"\n⚠️ **ALERTA:** Desbalance crítico detectado (ratio: {dist_analysis.get('imbalance_ratio', 0):.2f})\n"
        
        report += f"""
## 🔍 Análisis de Features Discriminativas
**Features Core Analizadas:** {len(MLConfig.FEATURES_CORE)}
**Features Avanzadas:** {len([col for col in df.columns if col not in MLConfig.FEATURES_CORE])}

### Poder Discriminativo por Feature:
"""
        
        discriminative_analysis = analysis_results.get('discriminative_analysis', {}).get('analysis', {})
        for feature, metrics in discriminative_analysis.items():
            status = "✅" if metrics.get('is_discriminative') else "❌"
            report += f"- **{feature}** {status}: Correlación={metrics.get('correlation_with_target', 0):.3f}, CV={metrics.get('coefficient_variation', 0):.3f}\n"
        
        report += f"""
## 🧠 Validación ML Crítica
**Estado Final:** {'APROBADO ✅' if ml_validation['is_ready'] else 'RECHAZADO ❌'}

### Test de Diversidad Predictiva:
"""
        
        diversity_test = ml_validation.get('critical_diversity_test', {})
        if diversity_test.get('passes_test'):
            report += "✅ **APROBADO** - Dataset genera predicciones diversas\n"
        else:
            report += f"❌ **FALLO** - {diversity_test.get('failure_reason', 'Error desconocido')}\n"
        
        model_results = diversity_test.get('model_results', {})
        if model_results:
            report += "\n### Resultados por Modelo:\n"
            for model_name, results in model_results.items():
                report += f"""
**{model_name.replace('_', ' ').title()}:**
- Diversidad Predictiva: {results.get('prediction_diversity', 0):.3f}
- Entropía Probabilidad: {results.get('probability_entropy', 0):.3f}
- Max ProbClase: {results.get('max_class_probability', 0):.3f}
"""
        
        report += f"""
## 🔧 Recomendaciones Críticas
"""
        recommendations = ml_validation.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. {rec}\n"
        else:
            report += "✅ No se requieren mejoras adicionales\n"
        
        report += f"""
## 🚀 Próximos Pasos
"""
        next_steps = self._generate_next_steps(ml_validation)
        for step in next_steps:
            report += f"{step}\n"
        
        report += f"""
## 📋 Configuración Técnica
- **Umbral Desbalance Crítico:** {MLConfig.DESBALANCE_CRITICO_THRESHOLD * 100}%
- **Variabilidad Mínima CV:** {MLConfig.VARIABILIDAD_MINIMA_CV}
- **Competitividad Mínima:** {MLConfig.COMPETITIVIDAD_MINIMA * 100}%
- **Features Objetivo:** {MLConfig.TARGET_FEATURE_COUNT}
- **Samples Mínimos/Clase:** {MLConfig.TARGET_MIN_SAMPLES}

## 🏷️ Metadatos del Dataset
- **Archivo Generado:** superior_dataset_{timestamp}.csv
- **Encoding:** UTF-8
- **Formato:** CSV con headers
- **Separador:** Coma (,)
"""
        
        return report


# === PUNTO DE ENTRADA PRINCIPAL ===
def main():
    """Función principal del sistema inteligente."""
    print("🧠 Iniciando Sistema Inteligente Anti-Colapso ML v5.2")
    print("🎯 Objetivo: Generar dataset superior que evite predicciones idénticas con feature engineering extensivo y robustez mejorada")
    print("=" * 80)
    
    # Crear generador inteligente
    generator = IntelligentDatasetGenerator()
    
    # Ejecutar generación superior
    success = generator.generate_superior_dataset(
        features_folder='input/features',
        labels_folder='input/labels',
        output_folder='reports'
    )
    
    if success:
        print("\n🎉 ¡PROCESO COMPLETADO! El dataset ha sido generado y validado.")
        print("   El resultado de la validación ML fue: APROBADO")
        print("📁 Revisa la carpeta 'reports' para los archivos finales y la subcarpeta 'stages' para los intermedios.")
    else:
        print("\n❌ ¡PROCESO COMPLETADO CON ADVERTENCIAS! El dataset fue generado pero NO PASÓ la validación ML.")
        print("   Revisa los reportes y logs para identificar los problemas críticos antes de entrenar un modelo.")
    
    return success

if __name__ == "__main__":
    main()
