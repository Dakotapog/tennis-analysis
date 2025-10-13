#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GENERADOR DE DATASET CON VALIDACIÓN ANTI-COLAPSO DE MODELO
Versión 4.0 - PREVENCIÓN DE PREDICCIONES IDÉNTICAS
"""

import json
import os
import glob
import pandas as pd
from datetime import datetime
from collections import Counter
import logging

# --- CONFIGURACIÓN DE LOGGING ---
timestamp_log = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'logs/dataset_generator_{timestamp_log}.log'
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# --- FUNCIONES DE UTILIDAD Y COLORES ---
class Color:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_summary_box(title, data):
    """Imprime un cuadro de resumen con colores."""
    logging.info(f"\n{Color.BOLD}{Color.BLUE}{'='*70}{Color.END}")
    logging.info(f"{Color.BOLD}{Color.BLUE}{title.center(70)}{Color.END}")
    logging.info(f"{Color.BOLD}{Color.BLUE}{'='*70}{Color.END}")
    for key, value in data.items():
        logging.info(f"{key:<50} {value}")
    logging.info(f"{Color.BOLD}{Color.BLUE}{'-'*70}{Color.END}")

def analizar_distribucion_ganadores(dataset):
    """Analiza la distribución de ganadores para detectar desbalance extremo."""
    logging.info("\n🎯 ANÁLISIS DE DISTRIBUCIÓN DE GANADORES")
    
    if not dataset:
        return {"error": "Dataset vacío"}
    
    ganadores = [item.get('ganador_real') for item in dataset if item.get('ganador_real')]
    if not ganadores:
        return {"error": "No hay ganadores definidos"}
    
    counter = Counter(ganadores)
    total = len(ganadores)
    
    distribucion = {}
    for ganador, count in counter.items():
        porcentaje = (count / total) * 100
        distribucion[ganador] = {
            'count': count,
            'porcentaje': porcentaje
        }
    
    # Detectar desbalance crítico
    porcentajes = [info['porcentaje'] for info in distribucion.values()]
    max_porcentaje = max(porcentajes)
    min_porcentaje = min(porcentajes)
    
    desbalance_critico = max_porcentaje > 80 or min_porcentaje < 20
    
    logging.info("📊 Distribución de ganadores:")
    for ganador, info in distribucion.items():
        color = Color.RED if info['porcentaje'] > 75 or info['porcentaje'] < 25 else Color.GREEN
        status = "⚠️ DESBALANCEADO" if info['porcentaje'] > 75 or info['porcentaje'] < 25 else "✅ BALANCEADO"
        logging.info(f"   {ganador}: {info['count']} ({info['porcentaje']:.1f}%) {color}{status}{Color.END}")
    
    return {
        'distribucion': distribucion,
        'desbalance_critico': desbalance_critico,
        'max_porcentaje': max_porcentaje,
        'min_porcentaje': min_porcentaje,
        'ratio_desbalance': max_porcentaje / min_porcentaje if min_porcentaje > 0 else float('inf')
    }

def validar_variabilidad_features(df):
    """Valida que las features numéricas tengan suficiente variabilidad."""
    logging.info("\n🔍 VALIDACIÓN DE VARIABILIDAD DE FEATURES")
    
    # Features numéricas clave para el modelo
    features_numericas = ['p1_ranking', 'p2_ranking', 'p1_elo', 'p2_elo', 'p1_edad', 'p2_edad']
    features_existentes = [col for col in features_numericas if col in df.columns]
    
    if not features_existentes:
        return {"error": "No hay features numéricas disponibles", "problemas_graves": True}
    
    problemas = []
    warnings = []
    
    for feature in features_existentes:
        valores = df[feature].dropna()
        
        if len(valores) < 2:
            problemas.append(f"Feature '{feature}': Insuficientes valores válidos")
            continue
            
        # Calcular estadísticas de variabilidad
        std_dev = valores.std()
        coef_variacion = std_dev / valores.mean() if valores.mean() != 0 else 0
        valores_unicos = len(valores.unique())
        porcentaje_unicos = (valores_unicos / len(valores)) * 100
        
        # Detectar problemas de variabilidad
        if std_dev == 0:
            problemas.append(f"Feature '{feature}': Valores constantes (desviación estándar = 0)")
        elif coef_variacion < 0.05:
            warnings.append(f"Feature '{feature}': Muy poca variabilidad (CV = {coef_variacion:.3f})")
        elif porcentaje_unicos < 10:
            warnings.append(f"Feature '{feature}': Pocos valores únicos ({valores_unicos} valores, {porcentaje_unicos:.1f}%)")
        
        # Log detallado de cada feature
        color = Color.RED if std_dev == 0 or coef_variacion < 0.05 else (Color.YELLOW if porcentaje_unicos < 20 else Color.GREEN)
        status = "❌ CRÍTICO" if std_dev == 0 else ("⚠️ BAJO" if coef_variacion < 0.1 or porcentaje_unicos < 20 else "✅ OK")
        
        logging.info(f"   {feature:<15}: std={std_dev:.2f}, CV={coef_variacion:.3f}, únicos={valores_unicos} {color}{status}{Color.END}")
    
    return {
        'problemas_graves': len(problemas) > 0,
        'problemas': problemas,
        'warnings': warnings,
        'features_analizadas': len(features_existentes)
    }

def analizar_competitividad_enfrentamientos(df):
    """Analiza si los enfrentamientos son competitivos o muy desiguales."""
    logging.info("\n⚔️ ANÁLISIS DE COMPETITIVIDAD DE ENFRENTAMIENTOS")
    
    if 'p1_ranking' not in df.columns or 'p2_ranking' not in df.columns:
        return {"error": "Falta información de ranking para análisis"}
    
    # Calcular diferencias de ranking
    df_clean = df.dropna(subset=['p1_ranking', 'p2_ranking'])
    if len(df_clean) == 0:
        return {"error": "No hay datos de ranking válidos"}
    
    diferencias_ranking = abs(df_clean['p1_ranking'] - df_clean['p2_ranking'])
    
    # Categorizar enfrentamientos
    muy_desiguales = (diferencias_ranking > 100).sum()  # Diferencia > 100 posiciones
    desiguales = ((diferencias_ranking > 50) & (diferencias_ranking <= 100)).sum()
    competitivos = ((diferencias_ranking > 10) & (diferencias_ranking <= 50)).sum()
    muy_competitivos = (diferencias_ranking <= 10).sum()
    
    total = len(df_clean)
    
    # Calcular porcentajes
    stats = {
        'muy_competitivos': {'count': muy_competitivos, 'pct': (muy_competitivos/total)*100},
        'competitivos': {'count': competitivos, 'pct': (competitivos/total)*100},
        'desiguales': {'count': desiguales, 'pct': (desiguales/total)*100},
        'muy_desiguales': {'count': muy_desiguales, 'pct': (muy_desiguales/total)*100}
    }
    
    # Detectar problemas
    pct_muy_desiguales = stats['muy_desiguales']['pct']
    problema_competitividad = pct_muy_desiguales > 60  # Más del 60% muy desiguales es problemático
    
    logging.info("📊 Distribución de competitividad:")
    logging.info(f"   Muy competitivos (≤10):   {stats['muy_competitivos']['count']} ({stats['muy_competitivos']['pct']:.1f}%)")
    logging.info(f"   Competitivos (11-50):     {stats['competitivos']['count']} ({stats['competitivos']['pct']:.1f}%)")
    logging.info(f"   Desiguales (51-100):      {stats['desiguales']['count']} ({stats['desiguales']['pct']:.1f}%)")
    
    color = Color.RED if pct_muy_desiguales > 60 else (Color.YELLOW if pct_muy_desiguales > 40 else Color.GREEN)
    status = "❌ PROBLEMÁTICO" if pct_muy_desiguales > 60 else ("⚠️ MEJORABLE" if pct_muy_desiguales > 40 else "✅ OK")
    logging.info(f"   Muy desiguales (>100):    {stats['muy_desiguales']['count']} ({stats['muy_desiguales']['pct']:.1f}%) {color}{status}{Color.END}")
    
    return {
        'stats': stats,
        'problema_competitividad': problema_competitividad,
        'diferencia_promedio': diferencias_ranking.mean(),
        'diferencia_mediana': diferencias_ranking.median()
    }

def detectar_patrones_obvios(df):
    """Detecta si hay patrones demasiado obvios que harían el modelo trivial."""
    logging.info("\n🔍 DETECCIÓN DE PATRONES OBVIOS")
    
    problemas_obvios = []
    
    # 1. ¿Siempre gana el mejor rankeado?
    if 'p1_ranking' in df.columns and 'p2_ranking' in df.columns and 'ganador_real' in df.columns:
        df_clean = df.dropna(subset=['p1_ranking', 'p2_ranking', 'ganador_real'])
        
        if len(df_clean) > 0:
            # Determinar quién es mejor rankeado (ranking menor = mejor)
            df_clean['mejor_rankeado'] = df_clean.apply(
                lambda row: 'player1' if row['p1_ranking'] < row['p2_ranking'] else 'player2', 
                axis=1
            )
            
            coincidencias = (df_clean['mejor_rankeado'] == df_clean['ganador_real']).sum()
            porcentaje_obvio = (coincidencias / len(df_clean)) * 100
            
            if porcentaje_obvio > 90:
                problemas_obvios.append(f"PATRÓN TRIVIAL: El mejor rankeado gana {porcentaje_obvio:.1f}% de las veces")
            elif porcentaje_obvio > 80:
                logging.warning(f"⚠️ Patrón muy predecible: Mejor rankeado gana {porcentaje_obvio:.1f}%")
            
            logging.info(f"📈 Mejor rankeado gana: {coincidencias}/{len(df_clean)} ({porcentaje_obvio:.1f}%)")
    
    # 2. ¿Hay muy poca variación en las diferencias?
    if 'p1_elo' in df.columns and 'p2_elo' in df.columns:
        df_elo = df.dropna(subset=['p1_elo', 'p2_elo'])
        if len(df_elo) > 0:
            diferencias_elo = abs(df_elo['p1_elo'] - df_elo['p2_elo'])
            if diferencias_elo.std() < 50:  # Muy poca variación en diferencias ELO
                problemas_obvios.append(f"PATRÓN OBVIO: Diferencias ELO muy uniformes (std = {diferencias_elo.std():.1f})")
    
    return {
        'problemas_obvios': problemas_obvios,
        'patron_trivial_detectado': len(problemas_obvios) > 0
    }

def validar_calidad_ml(dataset_entrenamiento):
    """Validación completa de la calidad de los datos para machine learning."""
    logging.info(f"\n{Color.BOLD}{Color.BLUE}🤖 VALIDACIÓN ESPECÍFICA PARA MACHINE LEARNING{Color.END}")
    
    if not dataset_entrenamiento:
        return False, ["Dataset vacío"], {}
    
    df = pd.DataFrame(dataset_entrenamiento)
    problemas_ml = []
    warnings_ml = []
    stats_ml = {}
    
    # 1. Análisis de distribución de ganadores
    dist_analysis = analizar_distribucion_ganadores(dataset_entrenamiento)
    if dist_analysis.get('desbalance_critico'):
        problemas_ml.append(f"DESBALANCE CRÍTICO: Ratio {dist_analysis.get('ratio_desbalance', 0):.1f}:1")
    stats_ml['distribucion'] = dist_analysis
    
    # 2. Validación de variabilidad de features
    var_analysis = validar_variabilidad_features(df)
    if var_analysis.get('problemas_graves'):
        problemas_ml.extend(var_analysis.get('problemas', []))
    if var_analysis.get('warnings'):
        warnings_ml.extend(var_analysis.get('warnings', []))
    stats_ml['variabilidad'] = var_analysis
    
    # 3. Análisis de competitividad
    comp_analysis = analizar_competitividad_enfrentamientos(df)
    if comp_analysis.get('problema_competitividad'):
        problemas_ml.append("ENFRENTAMIENTOS POCO COMPETITIVOS: >60% muy desiguales")
    stats_ml['competitividad'] = comp_analysis
    
    # 4. Detección de patrones obvios
    patron_analysis = detectar_patrones_obvios(df)
    if patron_analysis.get('patron_trivial_detectado'):
        problemas_ml.extend(patron_analysis.get('problemas_obvios', []))
    stats_ml['patrones'] = patron_analysis
    
    # Veredicto final
    dataset_apto_ml = len(problemas_ml) == 0
    
    if problemas_ml:
        logging.error(f"\n{Color.RED}❌ PROBLEMAS DE CALIDAD PARA ML DETECTADOS:{Color.END}")
        for problema in problemas_ml:
            logging.error(f"   🚨 {problema}")
    
    if warnings_ml:
        logging.warning(f"\n{Color.YELLOW}⚠️ ADVERTENCIAS DE CALIDAD ML:{Color.END}")
        for warning in warnings_ml:
            logging.warning(f"   ⚠️ {warning}")
    
    if dataset_apto_ml:
        logging.info(f"\n{Color.GREEN}✅ DATASET APTO PARA MACHINE LEARNING{Color.END}")
    else:
        logging.error(f"\n{Color.RED}❌ DATASET NO APTO - RIESGO DE PREDICCIONES IDÉNTICAS{Color.END}")
    
    return dataset_apto_ml, problemas_ml, stats_ml

def get_folder_summary(folder_path):
    """Obtiene un resumen de una carpeta: existencia, número de archivos y tamaño."""
    summary = {
        "exists": False,
        "num_files": 0,
        "total_size_mb": 0
    }
    if not os.path.exists(folder_path):
        return summary

    summary["exists"] = True
    files = glob.glob(os.path.join(folder_path, '*.json'))
    summary["num_files"] = len(files)
    
    total_size_bytes = sum(os.path.getsize(f) for f in files)
    summary["total_size_mb"] = round(total_size_bytes / (1024 * 1024), 2)
    
    return summary

def definir_schema_esperado():
    """Define el schema esperado del dataset para validación robusta."""
    return {
        'obligatorias': ['match_url', 'player1', 'player2', 'tipo_cancha', 'fecha', 'ganador_real'],
        'importantes': ['p1_ranking', 'p2_ranking', 'p1_elo', 'p2_elo', 'p1_edad', 'p2_edad'],
        'opcionales': [
            'p1_win_rate', 'p2_win_rate', 'p1_recent_form', 'p2_recent_form', 'p1_h2h_wins', 'p2_h2h_wins',
            'p1_service_games_won', 'p2_service_games_won', 'p1_return_games_won', 'p2_return_games_won',
            'p1_break_points_saved', 'p2_break_points_saved', 'p1_break_points_converted', 'p2_break_points_converted',
            'p1_first_serve_pct', 'p2_first_serve_pct', 'p1_first_serve_won', 'p2_first_serve_won',
            'p1_second_serve_won', 'p2_second_serve_won', 'p1_aces_per_match', 'p2_aces_per_match',
            'p1_double_faults_per_match', 'p2_double_faults_per_match'
        ],
        'rangos_numericos': {
            'p1_ranking': (1, 10000), 'p2_ranking': (1, 10000), 'p1_elo': (800, 3000), 'p2_elo': (800, 3000),
            'p1_edad': (15, 45), 'p2_edad': (15, 45), 'p1_win_rate': (0, 1), 'p2_win_rate': (0, 1),
            'p1_recent_form': (0, 1), 'p2_recent_form': (0, 1), 'p1_h2h_wins': (0, 50), 'p2_h2h_wins': (0, 50)
        },
        'valores_categoricos': {
            'tipo_cancha': ['Hard', 'Clay', 'Grass', 'Indoor Hard', 'Carpet'],
            'ganador_real': None
        }
    }

def validar_estructura_dataset(df, schema):
    """Validación robusta de la estructura del dataset."""
    logging.info("🔍 VALIDANDO ESTRUCTURA DEL DATASET")
    problemas_graves = []
    warnings = []
    
    columnas_faltantes = [col for col in schema['obligatorias'] if col not in df.columns]
    if columnas_faltantes:
        msg = f"Columnas obligatorias faltantes: {columnas_faltantes}"
        problemas_graves.append(msg)
        logging.error(msg)
    else:
        logging.info("✅ Todas las columnas obligatorias presentes")

    cols_importantes_faltantes = [col for col in schema['importantes'] if col not in df.columns]
    if cols_importantes_faltantes:
        msg = f"Columnas importantes faltantes: {cols_importantes_faltantes}"
        warnings.append(msg)
        logging.warning(msg)
    else:
        logging.info("✅ Todas las columnas importantes presentes")
        
    return problemas_graves, warnings

def validar_contenido_dataset(df, schema):
    """Validación del contenido del dataset."""
    logging.info("🔍 VALIDANDO CONTENIDO DEL DATASET")
    problemas_graves = []
    warnings = []

    if len(df) < 50:
        msg = f"Muy pocos registros: {len(df)} < 50"
        problemas_graves.append(msg)
        logging.error(msg)

    for col in schema['obligatorias']:
        if col in df.columns:
            nulos = df[col].isnull().sum()
            porcentaje = (nulos / len(df)) * 100 if len(df) > 0 else 0
            if porcentaje > 30:
                msg = f"Columna '{col}': {porcentaje:.1f}% nulos (crítico)"
                problemas_graves.append(msg)
                logging.error(msg)
            elif porcentaje > 10:
                msg = f"Columna '{col}': {porcentaje:.1f}% nulos"
                warnings.append(msg)
                logging.warning(msg)

    return problemas_graves, warnings

def validar_dataset_generado(dataset, modo_estricto=True):
    """Validación completa del dataset generado."""
    logging.info("\n🔍 VALIDACIÓN COMPLETA DEL DATASET")
    if not dataset:
        logging.error("❌ Dataset vacío")
        return False, ["Dataset vacío"], None

    try:
        df = pd.DataFrame(dataset)
        logging.info(f"📊 Dataset shape: {df.shape}")
    except Exception as e:
        logging.error(f"❌ Error al crear DataFrame: {e}")
        return False, [f"Error al crear DataFrame: {e}"], None

    schema = definir_schema_esperado()
    problemas_estructura, warnings_estructura = validar_estructura_dataset(df, schema)
    
    problemas_contenido = []
    warnings_contenido = []
    if not problemas_estructura:
        problemas_contenido, warnings_contenido = validar_contenido_dataset(df, schema)

    todos_problemas = problemas_estructura + problemas_contenido
    todos_warnings = warnings_estructura + warnings_contenido
    dataset_valido = len(todos_problemas) == 0

    if todos_problemas:
        logging.error("❌ PROBLEMAS GRAVES DETECTADOS:")
        for problema in todos_problemas:
            logging.error(f"   - {problema}")
    
    if todos_warnings:
        logging.warning("⚠️ ADVERTENCIAS:")
        for warning in todos_warnings:
            logging.warning(f"   - {warning}")

    if dataset_valido:
        logging.info("✅ DATASET COMPLETAMENTE VÁLIDO")
    else:
        logging.error("❌ DATASET CON PROBLEMAS")
        if modo_estricto:
            logging.error("🚫 MODO ESTRICTO: Proceso detenido")

    return dataset_valido, todos_problemas, df

def cargar_datos_desde_carpeta_seguro(folder_path, data_key):
    """Carga datos de forma robusta desde una carpeta."""
    logging.info(f"\n📁 CARGANDO DATOS DESDE: {folder_path}")
    datos_consolidados = []
    archivos_json = glob.glob(os.path.join(folder_path, '*.json'))

    for archivo_path in archivos_json:
        try:
            with open(archivo_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and data_key in data and isinstance(data[data_key], list):
                datos_consolidados.extend(data[data_key])
            elif isinstance(data, list):
                datos_consolidados.extend(data)
        except Exception as e:
            logging.error(f"Error procesando {archivo_path}: {e}")
            
    logging.info(f"   📊 Total registros cargados: {len(datos_consolidados)}")
    return datos_consolidados

def generar_dataset_entrenamiento_robusto(features_folder, labels_folder, 
                                         output_folder='reports', modo_estricto=True):
    """Generador de dataset robusto con validaciones anti-colapso."""
    logging.info("🚀 GENERADOR DE DATASET ROBUSTO V4.0 - ANTI-COLAPSO")
    
    # --- FASE 0: VALIDACIÓN PREVIA ---
    logging.info("\n--- FASE 0: VALIDACIÓN PREVIA DE CARPETAS ---")
    features_summary = get_folder_summary(features_folder)
    labels_summary = get_folder_summary(labels_folder)

    print_summary_box("RESUMEN DE CARPETAS DE ENTRADA", {
        "Carpeta Features": features_folder,
        "  - Existe (Features)": f"{Color.GREEN}Sí{Color.END}" if features_summary["exists"] else f"{Color.RED}No{Color.END}",
        "  - Archivos JSON (Features)": str(features_summary["num_files"]),
        "  - Tamaño Total (Features)": f"{features_summary['total_size_mb']} MB",
        "Carpeta Labels": labels_folder,
        "  - Existe (Labels)": f"{Color.GREEN}Sí{Color.END}" if labels_summary["exists"] else f"{Color.RED}No{Color.END}",
        "  - Archivos JSON (Labels)": str(labels_summary["num_files"]),
        "  - Tamaño Total (Labels)": f"{labels_summary['total_size_mb']} MB",
    })

    if not features_summary["exists"] or not labels_summary["exists"]:
        logging.critical("CRÍTICO: Una o ambas carpetas de entrada no existen. Proceso detenido.")
        return False
    if features_summary["num_files"] == 0 or labels_summary["num_files"] == 0:
        logging.critical("CRÍTICO: Una o ambas carpetas no contienen archivos JSON. Proceso detenido.")
        return False

    # --- FASE 1: CARGA DE DATOS ---
    logging.info("\n--- FASE 1: CARGA DE DATOS ---")
    data_features = cargar_datos_desde_carpeta_seguro(features_folder, 'partidos')
    data_labels = cargar_datos_desde_carpeta_seguro(labels_folder, 'detailed_results')

    if not data_features or not data_labels:
        logging.critical("CRÍTICO: No se pudieron cargar datos de features o labels. Proceso detenido.")
        return False

    # --- FASE 2: PROCESAMIENTO DE LABELS ---
    logging.info("\n--- FASE 2: PROCESAMIENTO DE LABELS ---")
    mapa_ganadores = {}
    for resultado in data_labels:
        try:
            if (isinstance(resultado, dict) and
                    resultado.get('verification', {}).get('status') == 'Verified'):
                
                match_info = resultado.get('match_info', {})
                match_url = match_info.get('match_url')
                ganador_real = resultado.get('actual_result', {}).get('actual_winner')

                if match_url and ganador_real:
                    mapa_ganadores[match_url] = {
                        'ganador_real': ganador_real,
                        'player1': match_info.get('jugador1'),
                        'player2': match_info.get('jugador2')
                    }
        except Exception as e:
            logging.warning(f"No se pudo procesar un registro de label: {e}")
            continue
    logging.info(f"✅ Labels válidos procesados: {len(mapa_ganadores)}")

    # --- FASE 3: COMBINACIÓN Y VALIDACIÓN ---
    logging.info("\n--- FASE 3: COMBINACIÓN Y VALIDACIÓN ---")
    urls_features = {p.get('match_url') for p in data_features if isinstance(p, dict) and 'match_url' in p}
    urls_labels = set(mapa_ganadores.keys())
    interseccion = urls_features.intersection(urls_labels)
    
    tasa_matching = (len(interseccion) / len(urls_labels)) * 100 if urls_labels else 0
    logging.info(f"URLs en features: {len(urls_features)}, en labels: {len(urls_labels)}, compatibles: {len(interseccion)}")
    logging.info(f"Tasa de matching: {tasa_matching:.1f}%")

    dataset_entrenamiento = []
    for partido in data_features:
        if isinstance(partido, dict) and (match_url := partido.get('match_url')) in mapa_ganadores:
            partido_actualizado = partido.copy()
            
            partido_actualizado['player1'] = partido_actualizado.pop('jugador1', None)
            partido_actualizado['player2'] = partido_actualizado.pop('jugador2', None)
            partido_actualizado['ganador_real'] = mapa_ganadores[match_url]['ganador_real']
            
            player1_name_label = mapa_ganadores[match_url]['player1']
            player2_name_label = mapa_ganadores[match_url]['player2']
            
            p1_name_key = player1_name_label.replace(' ', '_') if player1_name_label else 'unknown'
            p2_name_key = player2_name_label.replace(' ', '_') if player2_name_label else 'unknown'

            historial_key = f"historial_{p1_name_key}"
            historial_p1 = partido_actualizado.get(historial_key, [])

            if historial_p1 and isinstance(historial_p1, list) and len(historial_p1) > 0:
                partido_actualizado['fecha'] = historial_p1[0].get('fecha', 'N/A')
            else:
                partido_actualizado['fecha'] = 'N/A'

            # Añadir tipo_cancha si no existe
            if 'tipo_cancha' not in partido_actualizado:
                if historial_p1 and len(historial_p1) > 0:
                    partido_actualizado['tipo_cancha'] = historial_p1[0].get('superficie', 'Hard')
                else:
                    partido_actualizado['tipo_cancha'] = 'Hard'

            ranking_analysis = partido_actualizado.get('ranking_analysis', {})
            partido_actualizado['p1_ranking'] = ranking_analysis.get(f'{p1_name_key}_ranking')
            partido_actualizado['p2_ranking'] = ranking_analysis.get(f'{p2_name_key}_ranking')
            # Continuar desde donde se cortó el código...
            
            # Asignar ELO ratings
            elo_analysis = partido_actualizado.get('elo_analysis', {})
            partido_actualizado['p1_elo'] = elo_analysis.get(f'{p1_name_key}_elo')
            partido_actualizado['p2_elo'] = elo_analysis.get(f'{p2_name_key}_elo')
            
            # Asignar edades
            edad_analysis = partido_actualizado.get('edad_analysis', {})
            partido_actualizado['p1_edad'] = edad_analysis.get(f'{p1_name_key}_edad')
            partido_actualizado['p2_edad'] = edad_analysis.get(f'{p2_name_key}_edad')
            
            # Asignar métricas adicionales si están disponibles
            metrics_analysis = partido_actualizado.get('metrics_analysis', {})
            if metrics_analysis:
                partido_actualizado['p1_win_rate'] = metrics_analysis.get(f'{p1_name_key}_win_rate')
                partido_actualizado['p2_win_rate'] = metrics_analysis.get(f'{p2_name_key}_win_rate')
                partido_actualizado['p1_recent_form'] = metrics_analysis.get(f'{p1_name_key}_recent_form')
                partido_actualizado['p2_recent_form'] = metrics_analysis.get(f'{p2_name_key}_recent_form')
            
            # Asignar estadísticas H2H si están disponibles
            h2h_analysis = partido_actualizado.get('h2h_analysis', {})
            if h2h_analysis:
                partido_actualizado['p1_h2h_wins'] = h2h_analysis.get('p1_wins', 0)
                partido_actualizado['p2_h2h_wins'] = h2h_analysis.get('p2_wins', 0)
            
            # Limpiar campos innecesarios para el dataset final
            campos_a_eliminar = [
                'historial_' + p1_name_key, 'historial_' + p2_name_key,
                'ranking_analysis', 'elo_analysis', 'edad_analysis', 
                'metrics_analysis', 'h2h_analysis'
            ]
            for campo in campos_a_eliminar:
                partido_actualizado.pop(campo, None)
            
            dataset_entrenamiento.append(partido_actualizado)

    logging.info(f"✅ Dataset de entrenamiento generado: {len(dataset_entrenamiento)} registros")

    # --- FASE 4: VALIDACIÓN DE ESTRUCTURA Y CONTENIDO ---
    logging.info("\n--- FASE 4: VALIDACIÓN DE ESTRUCTURA Y CONTENIDO ---")
    dataset_valido, problemas_estructura, df = validar_dataset_generado(
        dataset_entrenamiento, modo_estricto=modo_estricto
    )
    
    if not dataset_valido and modo_estricto:
        logging.critical("CRÍTICO: Dataset no válido en modo estricto. Proceso detenido.")
        return False

    # --- FASE 5: VALIDACIÓN ESPECÍFICA PARA MACHINE LEARNING ---
    logging.info("\n--- FASE 5: VALIDACIÓN PARA MACHINE LEARNING ---")
    dataset_apto_ml, problemas_ml, stats_ml = validar_calidad_ml(dataset_entrenamiento)
    
    if not dataset_apto_ml and modo_estricto:
        logging.critical("CRÍTICO: Dataset no apto para ML en modo estricto. Riesgo de colapso del modelo.")
        return False
    elif not dataset_apto_ml:
        logging.warning("⚠️ ADVERTENCIA: Dataset con problemas para ML pero continuando...")

    # --- FASE 6: GENERACIÓN DE REPORTES ---
    logging.info("\n--- FASE 6: GENERACIÓN DE REPORTES ---")
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Guardar dataset principal
    dataset_filename = f"dataset_tennis_training_{timestamp}.json"
    dataset_path = os.path.join(output_folder, dataset_filename)
    
    try:
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "fecha_generacion": datetime.now().isoformat(),
                    "version": "4.0",
                    "total_registros": len(dataset_entrenamiento),
                    "tasa_matching": tasa_matching,
                    "dataset_valido": dataset_valido,
                    "apto_para_ml": dataset_apto_ml,
                    "problemas_detectados": problemas_estructura + problemas_ml,
                    "carpeta_features": features_folder,
                    "carpeta_labels": labels_folder
                },
                "dataset": dataset_entrenamiento,
                "validacion_ml": stats_ml
            }, f, indent=2, ensure_ascii=False)
        
        logging.info(f"✅ Dataset guardado: {dataset_path}")
    except Exception as e:
        logging.error(f"❌ Error guardando dataset: {e}")
        return False

    # Guardar reporte de validación detallado
    reporte_filename = f"validation_report_{timestamp}.json"
    reporte_path = os.path.join(output_folder, reporte_filename)
    
    try:
        reporte_validacion = {
            "timestamp": datetime.now().isoformat(),
            "resumen_ejecutivo": {
                "dataset_valido": dataset_valido,
                "apto_para_ml": dataset_apto_ml,
                "total_registros": len(dataset_entrenamiento),
                "tasa_matching": tasa_matching,
                "problemas_criticos": len(problemas_estructura + problemas_ml),
                "riesgo_colapso_modelo": not dataset_apto_ml
            },
            "validacion_estructura": {
                "problemas": problemas_estructura,
                "dataset_shape": df.shape if df is not None else None
            },
            "validacion_ml": stats_ml,
            "recomendaciones": generar_recomendaciones(dataset_apto_ml, problemas_ml, stats_ml)
        }
        
        with open(reporte_path, 'w', encoding='utf-8') as f:
            json.dump(reporte_validacion, f, indent=2, ensure_ascii=False)
        
        logging.info(f"✅ Reporte de validación guardado: {reporte_path}")
    except Exception as e:
        logging.error(f"❌ Error guardando reporte: {e}")

    # --- FASE 7: RESUMEN FINAL ---
    logging.info("\n--- FASE 7: RESUMEN FINAL ---")
    
    status_dataset = f"{Color.GREEN}✅ VÁLIDO{Color.END}" if dataset_valido else f"{Color.RED}❌ INVÁLIDO{Color.END}"
    status_ml = f"{Color.GREEN}✅ APTO{Color.END}" if dataset_apto_ml else f"{Color.RED}❌ RIESGO COLAPSO{Color.END}"
    
    print_summary_box("RESUMEN FINAL DE GENERACIÓN", {
        "Archivo Dataset": dataset_filename,
        "Ruta Completa": dataset_path,
        "Total Registros": str(len(dataset_entrenamiento)),
        "Tasa Matching": f"{tasa_matching:.1f}%",
        "Estado Dataset": status_dataset,
        "Estado ML": status_ml,
        "Problemas Críticos": str(len(problemas_estructura + problemas_ml)),
        "Modo Estricto": "Activado" if modo_estricto else "Desactivado",
        "Timestamp": timestamp
    })
    
    if dataset_apto_ml:
        logging.info(f"\n{Color.GREEN}{Color.BOLD}🎉 DATASET GENERADO EXITOSAMENTE Y APTO PARA ML{Color.END}")
        logging.info(f"{Color.GREEN}✅ Riesgo de predicciones idénticas: BAJO{Color.END}")
        logging.info(f"{Color.GREEN}✅ El modelo debería generar predicciones variadas{Color.END}")
    else:
        logging.warning(f"\n{Color.YELLOW}{Color.BOLD}⚠️ DATASET GENERADO CON ADVERTENCIAS{Color.END}")
        logging.warning(f"{Color.YELLOW}⚠️ Riesgo de predicciones idénticas: ALTO{Color.END}")
        logging.warning(f"{Color.YELLOW}⚠️ Revisar problemas antes de entrenar modelo{Color.END}")

    return True

def generar_recomendaciones(dataset_apto_ml, problemas_ml, stats_ml):
    """Genera recomendaciones específicas basadas en los problemas detectados."""
    recomendaciones = []
    
    if not dataset_apto_ml:
        recomendaciones.append("CRÍTICO: Dataset presenta alto riesgo de generar predicciones idénticas")
        
        # Recomendaciones específicas por tipo de problema
        for problema in problemas_ml:
            if "DESBALANCE CRÍTICO" in problema:
                recomendaciones.append("- Balancear dataset: añadir más ejemplos de la clase minoritaria")
                recomendaciones.append("- Considerar técnicas de oversampling (SMOTE) o undersampling")
            
            if "PATRÓN TRIVIAL" in problema:
                recomendaciones.append("- Añadir más features discriminativas")
                recomendaciones.append("- Incluir enfrentamientos más competitivos")
                recomendaciones.append("- Revisar si las features capturan la complejidad del dominio")
            
            if "ENFRENTAMIENTOS POCO COMPETITIVOS" in problema:
                recomendaciones.append("- Filtrar enfrentamientos muy desiguales (diferencia ranking >100)")
                recomendaciones.append("- Priorizar partidos entre jugadores de nivel similar")
            
            if "Valores constantes" in problema:
                recomendaciones.append("- Remover features constantes del dataset")
                recomendaciones.append("- Verificar proceso de extracción de features")
    
    # Recomendaciones de variabilidad
    if stats_ml.get('variabilidad', {}).get('warnings'):
        recomendaciones.append("- Aumentar variabilidad de features numéricas")
        recomendaciones.append("- Verificar rangos de valores esperados")
    
    # Recomendaciones generales para ML
    recomendaciones.extend([
        "- Validar dataset con validación cruzada antes del entrenamiento final",
        "- Monitorear métricas de diversidad de predicciones durante entrenamiento",
        "- Implementar early stopping si las predicciones convergen a un solo valor"
    ])
    
    return recomendaciones

# --- FUNCIÓN PRINCIPAL ---
def main():
    """Función principal del generador de dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generador de Dataset Anti-Colapso v4.0')
    parser.add_argument('--features', required=True, help='Carpeta con archivos de features')
    parser.add_argument('--labels', required=True, help='Carpeta con archivos de labels')
    parser.add_argument('--output', default='reports', help='Carpeta de salida (default: reports)')
    parser.add_argument('--no-strict', action='store_true', help='Desactivar modo estricto')
    
    args = parser.parse_args()
    
    modo_estricto = not args.no_strict
    
    logging.info("🚀 Iniciando generador de dataset...")
    logging.info(f"   Features: {args.features}")
    logging.info(f"   Labels: {args.labels}")
    logging.info(f"   Output: {args.output}")
    logging.info(f"   Modo estricto: {'Sí' if modo_estricto else 'No'}")
    
    exito = generar_dataset_entrenamiento_robusto(
        features_folder=args.features,
        labels_folder=args.labels,
        output_folder=args.output,
        modo_estricto=modo_estricto
    )
    
    if exito:
        logging.info("🎉 Proceso completado exitosamente")
        exit(0)
    else:
        logging.error("💥 Proceso falló")
        exit(1)

if __name__ == "__main__":
    main()