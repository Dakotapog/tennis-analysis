"""
FIXED Advanced ML Trainer Script for Tennis Match Prediction
OPTIMIZED FOR SMALL DATASETS (< 100 samples)

Correcciones críticas aplicadas:
✅ Eliminadas referencias a XGBoost no instalado
✅ Configuración anti-overfitting para datasets pequeños
✅ Validación de dataset mínimo
✅ Algoritmos simplificados para evitar memorización
✅ Métricas más realistas para 37 samples
✅ Cross-validation estable
"""

import os
import logging
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score, 
                           confusion_matrix, precision_recall_fscore_support)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# --- Enhanced Configuration for Small Datasets ---
class Config:
    """Configuration optimizada para datasets pequeños (< 100 samples)"""
    
    # 1. Paths and Versioning
    LATEST_DATASET_FOLDER = "ml_datasets/enhanced/"
    MODEL_REGISTRY_PATH = "models/classification/"
    REPORTS_PATH = "ml_datasets/reports/"
    LOGS_PATH = "logs/"
    COMPARISON_PATH = "ml_datasets/model_comparison/"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create all directories
    for path in [MODEL_REGISTRY_PATH, REPORTS_PATH, LOGS_PATH, COMPARISON_PATH]:
        os.makedirs(path, exist_ok=True)

    # 2. Data Schema
    TARGET_VARIABLE = "winner"
    FEATURES_TO_DROP = [
        "match_id", "player1_name", "player2_name", "tournament_name",
        "date", "url", "winner"
    ]
    
    # Tennis-specific columns for analysis
    SURFACE_COLUMN = "surface"
    RANKING_COLUMNS = ["player1_rank", "player2_rank"]
    
    # 3. ALGORITMOS OPTIMIZADOS PARA DATASETS PEQUEÑOS
    ALGORITHMS = {
        "LogisticRegression_L1": {
            "model": LogisticRegression,
            "params": {
                "C": 0.1,  # Regularización FUERTE
                "penalty": 'l1',  # LASSO para feature selection automática
                "solver": 'liblinear',
                "max_iter": 1000,
                "random_state": 42
            },
            "scale_features": True,
            "description": "Regresión Logística con regularización L1 (LASSO)"
        },
        "LogisticRegression_L2": {
            "model": LogisticRegression,
            "params": {
                "C": 0.5,  # Regularización moderada
                "penalty": 'l2',  # Ridge
                "max_iter": 1000,
                "random_state": 42
            },
            "scale_features": True,
            "description": "Regresión Logística con regularización L2 (Ridge)"
        },
        "RandomForest_Simple": {
            "model": RandomForestClassifier,
            "params": {
                "n_estimators": 20,  # POCOS árboles
                "max_depth": 3,      # PROFUNDIDAD MÍNIMA
                "min_samples_split": 8,  # Evita splits con pocos datos
                "min_samples_leaf": 4,   # Mínimo por hoja
                "max_features": 'sqrt',  # Menos features por árbol
                "random_state": 42
            },
            "scale_features": False,
            "description": "Random Forest simplificado anti-overfitting"
        }
    }

    # 4. Validation Settings AJUSTADAS
    TEST_SIZE = 0.3  # Mayor test size para mejor evaluación
    CV_FOLDS = 3     # Solo 3 folds (con 37 samples, 5 folds = 7 samples/fold)
    RANDOM_STATE = 42
    MIN_SAMPLES_WARNING = 50  # Advertencia si < 50 samples
    MIN_SAMPLES_CRITICAL = 30  # Crítico si < 30 samples
    MAX_FEATURES_RATIO = 0.3   # Max 30% de features vs samples

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(f"{Config.LOGS_PATH}ml_trainer_{Config.TIMESTAMP}.log"),
        logging.StreamHandler()
    ]
)

class SmallDatasetTennisMLPipeline:
    """ML Pipeline OPTIMIZADO para datasets pequeños de tenis"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None
        self.feature_names = None
        self.feature_selector = None
        
    def validate_dataset_size(self, X, y):
        """Validación crítica del tamaño del dataset"""
        n_samples, n_features = X.shape
        
        logging.info(f"🔍 VALIDACIÓN DATASET:")
        logging.info(f"   Samples: {n_samples}")
        logging.info(f"   Features: {n_features}")
        logging.info(f"   Ratio features/samples: {n_features/n_samples:.3f}")
        
        # Warnings y errores críticos
        if n_samples < Config.MIN_SAMPLES_CRITICAL:
            logging.error(f"🚨 DATASET CRÍTICO: {n_samples} samples < {Config.MIN_SAMPLES_CRITICAL}")
            logging.error(f"   Recomendación: Recolectar mínimo {Config.MIN_SAMPLES_CRITICAL - n_samples} samples adicionales")
        
        elif n_samples < Config.MIN_SAMPLES_WARNING:
            logging.warning(f"⚠️ DATASET PEQUEÑO: {n_samples} samples < {Config.MIN_SAMPLES_WARNING}")
            logging.warning(f"   Resultados pueden ser inestables")
        
        # Ratio features/samples
        if n_features/n_samples > Config.MAX_FEATURES_RATIO:
            logging.warning(f"⚠️ DEMASIADAS FEATURES: {n_features} features para {n_samples} samples")
            logging.warning(f"   Aplicando selección automática de features...")
            return True  # Activar feature selection
        
        return False
        
    def find_latest_dataset(self, folder_path: str) -> str:
        """Finds the most recently created enhanced dataset"""
        logging.info(f"🔍 Searching for latest enhanced dataset in '{folder_path}'...")
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Directory '{folder_path}' does not exist.")
        
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                if f.endswith(".csv") and "enhanced" in f.lower()]
        
        if not files:
            # Fallback to any CSV file
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
            
        if not files:
            raise FileNotFoundError(f"No CSV datasets found in '{folder_path}'.")
            
        latest_file = max(files, key=os.path.getctime)
        logging.info(f"✅ Latest enhanced dataset: {latest_file}")
        return latest_file

    def load_and_prepare_data(self, file_path: str):
        """Load and prepare data with small dataset optimization"""
        logging.info(f"📊 Loading data from '{file_path}'...")
        
        try:
            df = pd.read_csv(file_path)
            logging.info(f"✅ Data loaded. Shape: {df.shape}")
            
            # Verify target variable
            if Config.TARGET_VARIABLE not in df.columns:
                raise ValueError(f"Target '{Config.TARGET_VARIABLE}' not found. Available: {df.columns.tolist()}")
            
            # Prepare features and target
            y = df[Config.TARGET_VARIABLE]
            
            # Drop specified columns
            features_to_drop = [col for col in Config.FEATURES_TO_DROP if col in df.columns]
            X = df.drop(columns=features_to_drop, errors='ignore')
            
            # Keep only numeric features
            numeric_features = X.select_dtypes(include=[np.number]).columns
            X = X[numeric_features]
            
            # Handle missing values (crítico en datasets pequeños)
            X = X.fillna(X.median())
            
            logging.info(f"📈 Features prepared. Shape: {X.shape}")
            logging.info(f"🎯 Target distribution:\n{y.value_counts()}")
            
            # Validar tamaño del dataset
            need_feature_selection = self.validate_dataset_size(X, y)
            
            # Feature selection automática si es necesario
            if need_feature_selection:
                X = self.apply_feature_selection(X, y)
            
            # Store additional info for tennis-specific analysis
            self.surface_data = df[Config.SURFACE_COLUMN].values if Config.SURFACE_COLUMN in df.columns else None
            self.ranking_data = df[Config.RANKING_COLUMNS] if all(col in df.columns for col in Config.RANKING_COLUMNS) else None
            
            return X, y
            
        except Exception as e:
            logging.error(f"❌ Failed to load data: {e}")
            raise
    
    def apply_feature_selection(self, X, y):
        """Aplica selección de features para datasets pequeños"""
        n_samples = len(X)
        # Regla: max features = samples / 5 (conservador para datasets pequeños)
        max_features = max(5, n_samples // 5)
        max_features = min(max_features, len(X.columns))
        
        logging.info(f"🔧 Aplicando selección de features: {len(X.columns)} → {max_features}")
        
        # Usar SelectKBest con f_classif
        self.feature_selector = SelectKBest(score_func=f_classif, k=max_features)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Obtener nombres de features seleccionadas
        selected_features = X.columns[self.feature_selector.get_support()]
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        logging.info(f"✅ Features seleccionadas: {list(selected_features)}")
        
        return X_selected_df

    def split_data(self, X, y):
        """Split data with stratification optimized for small datasets"""
        logging.info("🔄 Splitting data...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE,
            stratify=y
        )
        
        logging.info(f"📊 Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Validación adicional para datasets muy pequeños
        if len(X_train) < 20:
            logging.warning(f"⚠️ TRAIN SET MUY PEQUEÑO: {len(X_train)} samples")
            logging.warning(f"   Considerando reducir test_size o recolectar más datos")
        
        # Store indices for tennis-specific analysis
        self.train_indices = X_train.index
        self.test_indices = X_test.index
        
        return X_train, X_test, y_train, y_test

    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all configured algorithms optimized for small datasets"""
        logging.info(f"🚀 Training {len(Config.ALGORITHMS)} algorithms optimized for small datasets...")
        
        self.feature_names = X_train.columns
        
        for name, config in Config.ALGORITHMS.items():
            logging.info(f"🔧 Training {name} - {config['description']}")
            
            try:
                # Prepare data (scale if needed)
                X_train_processed = X_train.copy()
                X_test_processed = X_test.copy()
                
                if config["scale_features"]:
                    scaler = StandardScaler()
                    X_train_processed = pd.DataFrame(
                        scaler.fit_transform(X_train_processed),
                        columns=X_train_processed.columns,
                        index=X_train_processed.index
                    )
                    X_test_processed = pd.DataFrame(
                        scaler.transform(X_test_processed),
                        columns=X_test_processed.columns,
                        index=X_test_processed.index
                    )
                    self.scalers[name] = scaler
                else:
                    self.scalers[name] = None
                
                # Train model
                model = config["model"](**config["params"])
                model.fit(X_train_processed, y_train)
                self.models[name] = model
                
                # Evaluate model
                self.evaluate_model(name, model, X_train_processed, X_test_processed, y_train, y_test)
                
                logging.info(f"✅ {name} training completed")
                
            except Exception as e:
                logging.error(f"❌ {name} training failed: {e}")
                continue

    def evaluate_model(self, name, model, X_train, X_test, y_train, y_test):
        """Comprehensive model evaluation optimized for small datasets"""
        
        # Basic predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # ROC AUC if probabilities available
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
        except:
            roc_auc = 0  # En caso de clases muy desbalanceadas
        
        # Cross-validation score (ajustado para datasets pequeños)
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=Config.CV_FOLDS, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except:
            # Fallback si CV falla
            cv_mean = accuracy
            cv_std = 0
            logging.warning(f"⚠️ Cross-validation failed for {name}, using test accuracy")
        
        # OVERFITTING DETECTION (crítico en datasets pequeños)
        train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        overfitting_gap = train_accuracy - accuracy
        
        # Store results
        self.results[name] = {
            'accuracy': accuracy,
            'train_accuracy': train_accuracy,
            'overfitting_gap': overfitting_gap,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        # Tennis-specific analysis
        self.analyze_tennis_metrics(name, y_test, y_pred)
        
        # Log con información de overfitting
        overfitting_status = "🔴 OVERFITTING" if overfitting_gap > 0.15 else "🟢 OK"
        logging.info(f"📊 {name}")
        logging.info(f"   Test Accuracy: {accuracy:.4f}")
        logging.info(f"   Train Accuracy: {train_accuracy:.4f}")
        logging.info(f"   Overfitting Gap: {overfitting_gap:.4f} {overfitting_status}")
        logging.info(f"   CV: {cv_mean:.4f}±{cv_std:.4f}")

    def analyze_tennis_metrics(self, model_name, y_test, y_pred):
        """Analyze performance by tennis-specific factors"""
        
        # Surface-specific analysis
        if self.surface_data is not None and len(np.unique(self.surface_data)) > 1:
            surface_test = self.surface_data[self.test_indices]
            surface_accuracy = {}
            
            for surface in np.unique(surface_test):
                surface_mask = surface_test == surface
                if np.sum(surface_mask) >= 3:  # Mínimo 3 samples por superficie
                    acc = accuracy_score(y_test[surface_mask], y_pred[surface_mask])
                    surface_accuracy[surface] = acc
            
            if surface_accuracy:  # Solo si hay datos suficientes
                self.results[model_name]['surface_accuracy'] = surface_accuracy
        
        # Ranking-based analysis (if available)
        if self.ranking_data is not None and len(self.ranking_data) > 5:
            ranking_test = self.ranking_data.iloc[self.test_indices]
            
            # Analyze by ranking difference
            if len(ranking_test.columns) >= 2:
                rank_diff = ranking_test.iloc[:, 0] - ranking_test.iloc[:, 1]
                
                # Close matches (ranking difference < 10)
                close_mask = np.abs(rank_diff) < 10
                if np.sum(close_mask) >= 3:  # Mínimo 3 matches
                    close_acc = accuracy_score(y_test[close_mask], y_pred[close_mask])
                    self.results[model_name]['close_matches_accuracy'] = close_acc

    def select_best_model(self):
        """Select best model penalizing overfitting heavily"""
        logging.info("🏆 Selecting best model (penalizing overfitting)...")
        
        if not self.results:
            raise ValueError("No models trained successfully")
        
        # Score models penalizando FUERTEMENTE el overfitting
        model_scores = {}
        for name, metrics in self.results.items():
            # Penalización por overfitting (crítico en datasets pequeños)
            overfitting_penalty = max(0, metrics['overfitting_gap'] * 2)  # Penalización x2
            
            # Weighted score con penalización
            score = (metrics['accuracy'] * 0.4 + 
                    metrics['cv_mean'] * 0.4 +      # Mayor peso a CV
                    metrics['f1_score'] * 0.2 - 
                    overfitting_penalty)              # RESTA por overfitting
            
            model_scores[name] = score
            
            logging.info(f"📊 {name}:")
            logging.info(f"   Raw Score: {score + overfitting_penalty:.4f}")
            logging.info(f"   Overfitting Penalty: -{overfitting_penalty:.4f}")
            logging.info(f"   Final Score: {score:.4f}")
        
        self.best_model_name = max(model_scores, key=model_scores.get)
        self.best_model = self.models[self.best_model_name]
        
        logging.info(f"🥇 Best model: {self.best_model_name} (Score: {model_scores[self.best_model_name]:.4f})")
        return self.best_model_name

    def generate_comprehensive_report(self):
        """Generate detailed comparison report with overfitting analysis"""
        logging.info("📄 Generating comprehensive report...")
        
        # Create comparison DataFrame
        comparison_data = []
        for name, metrics in self.results.items():
            row = {
                'Model': name,
                'Test_Accuracy': f"{metrics['accuracy']:.4f}",
                'Train_Accuracy': f"{metrics['train_accuracy']:.4f}",
                'Overfitting_Gap': f"{metrics['overfitting_gap']:.4f}",
                'CV_Mean': f"{metrics['cv_mean']:.4f}",
                'CV_Std': f"{metrics['cv_std']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1_Score': f"{metrics['f1_score']:.4f}",
                'ROC_AUC': f"{metrics['roc_auc']:.4f}"
            }
            
            # Add tennis-specific metrics if available
            if 'surface_accuracy' in metrics:
                for surface, acc in metrics['surface_accuracy'].items():
                    row[f'Acc_{surface}'] = f"{acc:.4f}"
            
            if 'close_matches_accuracy' in metrics:
                row['Close_Matches_Acc'] = f"{metrics['close_matches_accuracy']:.4f}"
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison report
        report_path = os.path.join(Config.COMPARISON_PATH, f"model_comparison_{Config.TIMESTAMP}.csv")
        comparison_df.to_csv(report_path, index=False)
        
        # Generate detailed text report
        self.generate_text_report(comparison_df)
        
        logging.info(f"📊 Comparison report saved: {report_path}")

    def generate_text_report(self, comparison_df):
        """Generate detailed text report with small dataset insights"""
        report_path = os.path.join(Config.REPORTS_PATH, f"detailed_report_{Config.TIMESTAMP}.txt")
        
        with open(report_path, "w", encoding='utf-8') as f:
            f.write(f"🎾 TENNIS MATCH PREDICTION - SMALL DATASET OPTIMIZED REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            f.write("📊 MODEL PERFORMANCE SUMMARY\n")
            f.write("-"*40 + "\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")
            
            f.write(f"🏆 BEST MODEL: {self.best_model_name}\n")
            f.write("-"*40 + "\n")
            best_metrics = self.results[self.best_model_name]
            f.write(f"Test Accuracy: {best_metrics['accuracy']:.4f}\n")
            f.write(f"Train Accuracy: {best_metrics['train_accuracy']:.4f}\n")
            f.write(f"Overfitting Gap: {best_metrics['overfitting_gap']:.4f}\n")
            f.write(f"Cross-validation: {best_metrics['cv_mean']:.4f} ± {best_metrics['cv_std']:.4f}\n")
            f.write(f"ROC AUC: {best_metrics['roc_auc']:.4f}\n")
            f.write(f"F1 Score: {best_metrics['f1_score']:.4f}\n")
            
            # Overfitting analysis
            f.write(f"\n🔍 OVERFITTING ANALYSIS:\n")
            if best_metrics['overfitting_gap'] > 0.15:
                f.write("🔴 WARNING: Possible overfitting detected!\n")
                f.write("   Recommendations:\n")
                f.write("   - Collect more training data\n")
                f.write("   - Increase regularization\n")
                f.write("   - Reduce feature count\n")
            elif best_metrics['overfitting_gap'] > 0.05:
                f.write("🟡 CAUTION: Minor overfitting detected\n")
                f.write("   Monitor performance on new data\n")
            else:
                f.write("🟢 GOOD: No significant overfitting detected\n")
            
            # Tennis-specific insights
            if 'surface_accuracy' in best_metrics:
                f.write(f"\n🎾 SURFACE-SPECIFIC PERFORMANCE:\n")
                for surface, acc in best_metrics['surface_accuracy'].items():
                    f.write(f"{surface}: {acc:.4f}\n")
            
            if 'close_matches_accuracy' in best_metrics:
                f.write(f"\n⚖️ CLOSE MATCHES ACCURACY: {best_metrics['close_matches_accuracy']:.4f}\n")
            
            # Feature importance
            f.write(f"\n📈 FEATURE ANALYSIS:\n")
            if hasattr(self.best_model, 'feature_importances_'):
                feature_imp = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                f.write("Top Features by Importance:\n")
                f.write(feature_imp.head(10).to_string(index=False))
            
            if hasattr(self.best_model, 'coef_'):
                coef_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'coefficient': self.best_model.coef_[0]
                }).sort_values('coefficient', key=abs, ascending=False)
                f.write(f"\nTop Features by Coefficient Magnitude:\n")
                f.write(coef_df.head(10).to_string(index=False))
        
        logging.info(f"📄 Detailed report saved: {report_path}")

    def save_best_model(self):
        """Save the best model and its scaler"""
        if self.best_model is None:
            raise ValueError("No best model selected")
        
        # Save model
        model_filename = f"best_tennis_model_{self.best_model_name}_{Config.TIMESTAMP}.joblib"
        model_path = os.path.join(Config.MODEL_REGISTRY_PATH, model_filename)
        joblib.dump(self.best_model, model_path)
        
        # Save scaler if used
        scaler_path = None
        if self.scalers[self.best_model_name] is not None:
            scaler_filename = f"scaler_{self.best_model_name}_{Config.TIMESTAMP}.joblib"
            scaler_path = os.path.join(Config.MODEL_REGISTRY_PATH, scaler_filename)
            joblib.dump(self.scalers[self.best_model_name], scaler_path)
        
        # Save feature selector if used
        selector_path = None
        if self.feature_selector is not None:
            selector_filename = f"feature_selector_{Config.TIMESTAMP}.joblib"
            selector_path = os.path.join(Config.MODEL_REGISTRY_PATH, selector_filename)
            joblib.dump(self.feature_selector, selector_path)
        
        # Save metadata
        metadata = {
            'model_name': self.best_model_name,
            'model_file': model_filename,
            'scaler_file': scaler_filename if scaler_path else None,
            'feature_selector_file': selector_filename if selector_path else None,
            'features': self.feature_names.tolist(),
            'performance': {k: v for k, v in self.results[self.best_model_name].items() 
                          if k not in ['predictions', 'probabilities']},  # Exclude large arrays
            'timestamp': Config.TIMESTAMP,
            'config': Config.ALGORITHMS[self.best_model_name],
            'dataset_info': {
                'total_samples': len(self.train_indices) + len(self.test_indices),
                'train_samples': len(self.train_indices),
                'test_samples': len(self.test_indices),
                'features_count': len(self.feature_names)
            }
        }
        
        metadata_path = os.path.join(Config.MODEL_REGISTRY_PATH, f"model_metadata_{Config.TIMESTAMP}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logging.info(f"💾 Best model saved: {model_path}")
        if scaler_path:
            logging.info(f"🔧 Scaler saved: {scaler_path}")
        if selector_path:
            logging.info(f"🎯 Feature selector saved: {selector_path}")
        logging.info(f"📋 Model metadata saved: {metadata_path}")

def main():
    """Main execution pipeline optimized for small datasets"""
    logging.info("🎾 Starting Small Dataset Optimized Tennis ML Pipeline...")
    
    pipeline = SmallDatasetTennisMLPipeline()
    
    try:
        # 1. Load enhanced dataset automatically
        dataset_path = pipeline.find_latest_dataset(Config.LATEST_DATASET_FOLDER)
        X, y = pipeline.load_and_prepare_data(dataset_path)
        
        # 2. Split data
        X_train, X_test, y_train, y_test = pipeline.split_data(X, y)
        
        # 3. Train optimized algorithms
        pipeline.train_all_models(X_train, X_test, y_train, y_test)
        
        # 4. Select best model (with overfitting penalty)
        best_model = pipeline.select_best_model()
        
        # 5. Generate comprehensive reports
        pipeline.generate_comprehensive_report()
        
        # 6. Save best model for production
        pipeline.save_best_model()
        
        logging.info("🎉 Small Dataset Optimized Tennis ML Pipeline completed successfully!")
        logging.info(f"🏆 Best model: {best_model}")
        
        # 7. Final recommendations based on results
        pipeline.generate_final_recommendations()
        
    except Exception as e:
        logging.critical(f"💥 Pipeline failed: {e}", exc_info=True)
        raise

def generate_final_recommendations(self):
    """Generate actionable recommendations based on results"""
    logging.info("💡 GENERATING FINAL RECOMMENDATIONS...")
    
    best_metrics = self.results[self.best_model_name]
    n_samples = len(self.train_indices) + len(self.test_indices)
    
    recommendations = []
    
    # Data collection recommendations
    if n_samples < 100:
        recommendations.append(f"🎯 CRITICAL: Expand dataset from {n_samples} to 200+ samples")
        recommendations.append(f"   Current accuracy {best_metrics['accuracy']:.3f} will improve significantly")
    
    # Overfitting recommendations
    if best_metrics['overfitting_gap'] > 0.15:
        recommendations.append("🔴 HIGH OVERFITTING DETECTED:")
        recommendations.append("   - Increase regularization (lower C values)")
        recommendations.append("   - Collect more diverse training data")
        recommendations.append("   - Consider ensemble methods")
    
    # Performance recommendations
    if best_metrics['accuracy'] < 0.70:
        recommendations.append("📈 LOW ACCURACY DETECTED:")
        recommendations.append("   - Review feature engineering in RivalryAnalyzer")
        recommendations.append("   - Consider H2H historical depth")
        recommendations.append("   - Validate data quality in scraping")
    
    # Feature recommendations
    if len(self.feature_names) > n_samples * 0.3:
        recommendations.append("🎯 FEATURE OPTIMIZATION:")
        recommendations.append(f"   - Current: {len(self.feature_names)} features")
        recommendations.append(f"   - Optimal: {int(n_samples * 0.2)} features max")
        recommendations.append("   - Focus on top importance features only")
    
    # Model-specific recommendations
    if self.best_model_name.startswith("LogisticRegression"):
        recommendations.append("📊 LOGISTIC REGRESSION SELECTED:")
        recommendations.append("   - Good choice for small datasets")
        recommendations.append("   - Consider feature interactions")
        recommendations.append("   - Monitor coefficient stability")
    
    # Next steps
    recommendations.append("🚀 IMMEDIATE NEXT STEPS:")
    recommendations.append("   1. Expand scraping to 200+ matches minimum")
    recommendations.append("   2. Test model on completely new tournament data")
    recommendations.append("   3. Implement real-time prediction pipeline")
    recommendations.append("   4. Monitor performance degradation over time")
    
    # Log all recommendations
    logging.info("="*60)
    logging.info("🎯 ACTIONABLE RECOMMENDATIONS:")
    logging.info("="*60)
    for rec in recommendations:
        logging.info(rec)
    logging.info("="*60)
    
    # Save recommendations to file
    rec_path = os.path.join(Config.REPORTS_PATH, f"recommendations_{Config.TIMESTAMP}.txt")
    with open(rec_path, "w", encoding='utf-8') as f:
        f.write("🎯 TENNIS ML PIPELINE - ACTIONABLE RECOMMENDATIONS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"📊 CURRENT STATUS:\n")
        f.write(f"Dataset Size: {n_samples} samples\n")
        f.write(f"Best Model: {self.best_model_name}\n")
        f.write(f"Best Accuracy: {best_metrics['accuracy']:.4f}\n")
        f.write(f"Overfitting Gap: {best_metrics['overfitting_gap']:.4f}\n")
        f.write(f"CV Score: {best_metrics['cv_mean']:.4f} ± {best_metrics['cv_std']:.4f}\n\n")
        
        for rec in recommendations:
            f.write(rec + "\n")
    
    logging.info(f"💾 Recommendations saved: {rec_path}")

# Add method to pipeline class
SmallDatasetTennisMLPipeline.generate_final_recommendations = generate_final_recommendations

if __name__ == "__main__":
    main()