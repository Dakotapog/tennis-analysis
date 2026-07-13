import json
import pandas as pd
import sys
import os
import glob
import re
import argparse
import shutil
from datetime import datetime

# Importar módulo de normalización para transparencia en los cálculos
try:
    from normalization import (
        MAX_RAW_SCORES,
        DEFAULT_WEIGHTS,
        validate_weights,
        WeightManager,
        debug_normalization
    )
    NORMALIZATION_AVAILABLE = True
except ImportError:
    NORMALIZATION_AVAILABLE = False
    print("⚠️ Advertencia: No se pudo importar analysis.normalization")

try:
    from config import detectar_tier
    TIER_AVAILABLE = True
except ImportError:
    TIER_AVAILABLE = False
    print("⚠️ Advertencia: No se pudo importar detectar_tier desde config")

try:
    from analysis.player_profitability import _normalize_name as _prof_normalize_name
    PROFITABILITY_AVAILABLE = True
except ImportError:
    PROFITABILITY_AVAILABLE = False


def _load_profitability_data() -> dict:
    """Loads data/player_profitability.json if it exists. Graceful degradation."""
    import json
    from pathlib import Path
    prof_path = Path('data') / 'player_profitability.json'
    if not prof_path.exists():
        return {}
    try:
        return json.loads(prof_path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _load_edge_report() -> dict:
    """Carga el edge_report más reciente. Retorna dict partido→pick. Graceful degradation."""
    from pathlib import Path
    files = sorted(glob.glob('reports/edge_report_*.json'), reverse=True)
    if not files:
        return {}
    try:
        data = json.loads(Path(files[0]).read_text(encoding='utf-8'))
    except Exception:
        return {}
    lookup = {}
    for section in ('apostar', 'watchlist', 'sin_edge', 'sin_datos', 'no_data'):
        for pick in data.get(section, []):
            partido = pick.get('partido', '')
            if partido:
                lookup[partido] = pick
    return lookup


def _normalize_player_name_for_prof(name: str) -> str:
    """Normalize player name for profitability lookup."""
    if PROFITABILITY_AVAILABLE:
        return _prof_normalize_name(name)
    import unicodedata
    import re
    if not name:
        return ''
    nfkd = unicodedata.normalize('NFKD', name)
    ascii_str = nfkd.encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'\s+', ' ', ascii_str.lower().strip())


def show_normalization_transparency(f):
    """
    Muestra información sobre los parámetros de normalización utilizados.
    Esto proporciona transparencia sobre cómo se calculan las puntuaciones.
    """
    if not NORMALIZATION_AVAILABLE:
        f.write("--- INFORMACIÓN DE NORMALIZACIÓN ---\n")
        f.write("⚠️ Módulo de normalización no disponible.\n\n")
        return
    
    f.write("=" * 80 + "\n")
    f.write("🎾 INFORMACIÓN DE NORMALIZACIÓN Y TRANSPARENCIA DE CÁLCULOS\n")
    f.write("=" * 80 + "\n\n")
    
    # Mostrar máximos esperados para cada componente
    f.write("--- VALORES MÁXIMOS ESPERADOS (para normalización 0-1) ---\n")
    f.write("Estos valores definen el '100%' para cada componente del análisis:\n\n")
    
    for component, max_value in MAX_RAW_SCORES.items():
        display_name = component.replace('_', ' ').title()
        f.write(f"  • {display_name}: {max_value} puntos\n")
    
    f.write("\n")
    
    # Mostrar pesos por defecto
    f.write("--- PESOS POR DEFECTO (Torneos ATP/WTA) ---\n")
    f.write("Distribución de pesos cuando todos los componentes tienen datos:\n\n")
    
    for component, weight in DEFAULT_WEIGHTS.items():
        display_name = component.replace('_', ' ').title()
        pct = weight * 100
        f.write(f"  • {display_name}: {pct:.0f}%\n")
    
    total = sum(DEFAULT_WEIGHTS.values())
    f.write(f"\n  Total: {total * 100:.0f}%\n\n")
    
    # Explicación del proceso
    f.write("--- EXPLICACIÓN DEL PROCESO DE NORMALIZACIÓN ---\n")
    f.write("""
  1. Cada componente genera una 'puntuación bruta' basada en los datos del partido.
  2. La puntuación bruta se normaliza dividiendo por el 'Valor Máximo Esperado'.
     Ejemplo: Si ranking_momentum = 225 y el máximo es 450, normalizado = 0.50
  3. Los pesos se aplican a las puntuaciones normalizadas.
  4. Si un componente no tiene datos (ej: H2H vacío), su peso se redistribuye
     proporcionalmente entre los componentes activos.
  5. La puntuación final es la suma de todas las puntuaciones ponderadas.
  
""")
    
    # Validación de pesos
    is_valid, message = validate_weights(DEFAULT_WEIGHTS)
    f.write("--- VALIDACIÓN DE PESOS ---\n")
    f.write(f"Estado: {'✅ Válido' if is_valid else '❌ Error'}\n")
    f.write(f"Mensaje: {message}\n\n")


def get_weights_from_reasoning(reasoning):
    """Extracts analysis weights from the reasoning string.

    Parses LOG_WEIGHTS_STRATEGY for initial weights, then applies
    dynamic adjustments from LOG_WEIGHTS_SURFACE_*, LOG_DENSITY,
    and LOG_E1_TORNEO_WEIGHT to return the FINAL weights used.
    """
    weights = {}
    for reason in reasoning:
        if 'LOG_WEIGHTS' in reason and 'SURFACE' not in reason and 'E1' not in reason:
            match = re.search(r"\{.*\}", reason)
            if match:
                try:
                    weights_str = match.group(0).replace("'", '"')
                    weights_data = json.loads(weights_str)
                    for key, value in weights_data.items():
                        weights[key] = value
                except (json.JSONDecodeError, ValueError):
                    continue
    if not weights:
        return weights

    # Apply dynamic adjustments in the same order as rivalry_analyzer.py
    for reason in reasoning:
        # Density adjustment (common_opponents → form_recent)
        if 'LOG_DENSITY' in reason:
            co_match = re.search(r'co_w:\s*[\d.]+→([\d.]+)', reason)
            form_match = re.search(r'form_w→([\d.]+)', reason)
            if co_match:
                weights['common_opponents'] = float(co_match.group(1))
            if form_match:
                weights['form_recent'] = float(form_match.group(1))
        # Surface adjustment (clay/grass)
        if 'LOG_WEIGHTS_SURFACE_GRASS' in reason:
            co_match = re.search(r'common_opp→([\d.]+)', reason)
            form_match = re.search(r'form_recent→([\d.]+)', reason)
            if co_match:
                weights['common_opponents'] = float(co_match.group(1))
            if form_match:
                weights['form_recent'] = float(form_match.group(1))
        if 'LOG_WEIGHTS_SURFACE_CLAY' in reason:
            co_match = re.search(r'common_opp→([\d.]+)', reason)
            rm_match = re.search(r'ranking_mom→([\d.]+)', reason)
            if co_match:
                weights['common_opponents'] = float(co_match.group(1))
            if rm_match:
                weights['ranking_momentum'] = float(rm_match.group(1))
        # E-1: Tournament champion weight boost
        if 'LOG_E1_TORNEO_WEIGHT' in reason:
            surf_match = re.search(r'surface\s+[\d.]+→([\d.]+)', reason)
            form_match = re.search(r'form\s+[\d.]+→([\d.]+)', reason)
            if surf_match:
                weights['surface_specialization'] = float(surf_match.group(1))
            if form_match:
                weights['form_recent'] = float(form_match.group(1))

    return weights


def format_score_breakdown(f, p1_name, p2_name, score_breakdown, category, category_name, weights):
    """Formats and writes the score breakdown for a specific category."""
    score_breakdown = score_breakdown or {}
    player1_breakdown = score_breakdown.get('player1') or {}
    p1_data = player1_breakdown.get(category) or {}

    player2_breakdown = score_breakdown.get('player2') or {}
    p2_data = player2_breakdown.get(category) or {}
    weight_value = weights.get(category)
    
    weight_str = f"{weight_value*100:.0f}%" if isinstance(weight_value, float) else "N/A"
    f.write(f"\n--- {category_name} (Peso: {weight_str}) ---\n")
    
    df_data = {
        'Jugador': [p1_name, p2_name],
        'Puntaje Bruto': [p1_data.get('raw_score', 'N/A'), p2_data.get('raw_score', 'N/A')],
        'Puntaje Normalizado': [p1_data.get('normalized_score', 'N/A'), p2_data.get('normalized_score', 'N/A')],
        'Puntaje Ponderado': [p1_data.get('weighted_score', 'N/A'), p2_data.get('weighted_score', 'N/A')],
        'Contribución': [p1_data.get('contribution', 'N/A'), p2_data.get('contribution', 'N/A')]
    }
    df = pd.DataFrame(df_data)
    f.write("\n")
    f.write(df.to_markdown(index=False))
    f.write("\n\n")


def format_ranking_momentum_details(f, p1_name, p2_name, reasoning, ranking_analysis):
    """Parses the detailed ranking momentum logs and formats them into a table."""
    p1_log = next((r for r in reasoning if f"LOG_RANKING_{p1_name}" in r or "LOG_RANKING_P1" in r), None)
    p2_log = next((r for r in reasoning if f"LOG_RANKING_{p2_name}" in r or "LOG_RANKING_P2" in r), None)

    if not p1_log and not p2_log:
        return

    def parse_log_scores(log_string):
        """Extrae los valores numéricos de la cadena de log de ranking."""
        if not log_string:
            return {}
        
        details = re.findall(r'(\w+)\s*\((?:.*?)\)\s*=\s*([\d.\-]+)|(\w+)\s*=\s*([\d.\-]+)', log_string)
        parsed_data = {}
        for match in details:
            key = match[0] or match[2]
            value = match[1] or match[3]
            try:
                parsed_data[key.title()] = float(value)
            except (ValueError, TypeError):
                parsed_data[key.title()] = value
        return parsed_data

    def get_player_ranking_data(player_name, analysis_dict):
        player_lastname = player_name.split(' ')[0]
        for key, value in analysis_dict.items():
            if player_lastname in key and isinstance(value, dict) and 'prediction' not in key:
                return value
        return None

    p1_ranking_data = get_player_ranking_data(p1_name, ranking_analysis)
    p2_ranking_data = get_player_ranking_data(p2_name, ranking_analysis)

    p1_points = {}
    if p1_ranking_data:
        p1_points['Pts Totales'] = p1_ranking_data.get('pts')
        p1_points['Próx. Pts'] = p1_ranking_data.get('prox_pts')
        p1_points['Pts Máx.'] = p1_ranking_data.get('pts_max')

    p2_points = {}
    if p2_ranking_data:
        p2_points['Pts Totales'] = p2_ranking_data.get('pts')
        p2_points['Próx. Pts'] = p2_ranking_data.get('prox_pts')
        p2_points['Pts Máx.'] = p2_ranking_data.get('pts_max')

    def parse_log_advanced_metrics(log_string):
        """Extrae las métricas avanzadas de la cadena de log de ranking."""
        if not log_string:
            return {}
        
        metrics = {}
        momentum_match = re.search(r'Momentum\(([\d.\-]+)\)', log_string)
        if momentum_match:
            metrics['Puntos Asegurados'] = float(momentum_match.group(1))

        potential_match = re.search(r'Potencial\(([\d.\-]+)\)', log_string)
        if potential_match:
            metrics['Potencial de Mejora'] = float(potential_match.group(1))

        pressure_match = re.search(r'Presión\(([\d.\-]+)\)', log_string)
        if pressure_match:
            metrics['Índice de Presión'] = float(pressure_match.group(1))
            
        return metrics

    p1_scores = parse_log_scores(p1_log)
    p2_scores = parse_log_scores(p2_log)
    
    p1_advanced = parse_log_advanced_metrics(p1_log)
    p2_advanced = parse_log_advanced_metrics(p2_log)

    # Tabla de Puntos de Ranking
    if p1_points or p2_points:
        f.write(f"\n--- Puntos de Ranking (FlashScore) ---\n")

        points_keys = ['Pts Totales', 'Próx. Pts', 'Pts Máx.']
        df_points_data = {
            'Componente': points_keys,
            p1_name: [p1_points.get(key, 'N/A') for key in points_keys],
            p2_name: [p2_points.get(key, 'N/A') for key in points_keys]
        }
        df_points = pd.DataFrame(df_points_data)
        f.write("\n")
        f.write(df_points.to_markdown(index=False))
        f.write("\n\n")

    # Tabla de Métricas Avanzadas
    if p1_advanced or p2_advanced:
        f.write(f"\n--- Métricas Avanzadas de Ranking ---\n")

        advanced_keys = ['Potencial de Mejora', 'Puntos Asegurados', 'Índice de Presión']
        df_advanced_data = {
            'Métrica': advanced_keys,
            p1_name: [p1_advanced.get(key, 'N/A') for key in advanced_keys],
            p2_name: [p2_advanced.get(key, 'N/A') for key in advanced_keys]
        }
        df_advanced = pd.DataFrame(df_advanced_data)
        f.write("\n")
        f.write(df_advanced.to_markdown(index=False))
        f.write("\n\n")

    # Ordenar las claves para una presentación consistente
    all_keys = sorted(list(set(p1_scores.keys()) | set(p2_scores.keys())))

    df_data = {
        'Componente Momentum': all_keys,
        p1_name: [f"{p1_scores.get(key, 0):.4f}" for key in all_keys],
        p2_name: [f"{p2_scores.get(key, 0):.4f}" for key in all_keys]
    }

    df = pd.DataFrame(df_data)
    f.write(f"\n--- Desglose Detallado de Ranking & Momentum ---\n")
    f.write("\n")
    f.write(df.to_markdown(index=False))
    f.write("\n\n")


def parse_score(score_str):
    """Parsea un resultado como '2-1' en una tupla de enteros (sets_ganados, sets_perdidos)."""
    try:
        parts = score_str.split('-')
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        pass
    return 0, 0


def explicar_ventaja_rival_comun(opponent_data, p1_name, p2_name):
    """
    Genera una explicación textual de por qué se asigna la ventaja a un jugador
    basado en sus resultados contra un oponente común.
    """
    p1_res = opponent_data.get('player1_result') or {}
    p2_res = opponent_data.get('player2_result') or {}
    advantage = opponent_data.get('advantage_for', 'Ninguno')
    opponent_name = opponent_data.get('opponent_name', 'N/A')
    
    p1_outcome = p1_res.get('outcome', 'N/A')
    p2_outcome = p2_res.get('outcome', 'N/A')
    p1_score_str = p1_res.get('score', '0-0')
    p2_score_str = p2_res.get('score', '0-0')

    # Ventaja directa por ganar vs perder
    if p1_outcome == 'Ganó' and p2_outcome == 'Perdió':
        return f"{p1_name} venció a {opponent_name}, mientras que {p2_name} perdió."
    if p2_outcome == 'Ganó' and p1_outcome == 'Perdió':
        return f"{p2_name} venció a {opponent_name}, mientras que {p1_name} perdió."

    # Ambos ganadora o ambos perdieron
    if p1_outcome == p2_outcome:
        p1_sets = parse_score(p1_score_str)
        p2_sets = parse_score(p2_score_str)
        
        if p1_outcome == 'Ganó':
            # Gana el más contundente (menos sets cedidos)
            if p1_sets[1] < p2_sets[1]:
                return f"Ambos ganaron, pero {p1_name} fue más contundente ({p1_score_str} vs {p2_score_str} de {p2_name})."
            elif p2_sets[1] < p1_sets[1]:
                return f"Ambos gana, pero {p2_name} fue más contundente ({p2_score_str} vs {p1_score_str} de {p1_name})."
            else:
                return f"Ambos vencieron a {opponent_name} con resultados similares ({p1_score_str} y {p2_score_str})."
        elif p1_outcome == 'Perdió':
            # Gana el que mostró más esfuerzo (más sets ganados)
            if p1_sets[0] > p2_sets[0]:
                return f"Ambos perdieron, pero {p1_name} mostró mayor resistencia ({p1_score_str} vs {p2_score_str} de {p2_name})."
            elif p2_sets[0] > p1_sets[0]:
                return f"Ambos perdieron, pero {p2_name} mostró mayor resistencia ({p2_score_str} vs {p1_score_str} de {p1_name})."
            else:
                # Si ganaron los mismos sets, el que perdió por menos diferencia de juegos (no implementado aquí, pero es una mejora futura)
                return f"Ambos cayeron ante {opponent_name} de forma similar ({p1_score_str} y {p2_score_str})."

    return f"Ventaja para {advantage} contra {opponent_name} basada en el análisis de resultados."


def analizar_patrones_historial(historial_df, player_ranking):
    """
    Analiza el historial de partidos de un jugador para identificar patrones.
    """
    if historial_df.empty:
        return ["- No hay datos históricos para analizar."]

    # Convertir columnas a tipos numéricos, manejando errores
    historial_df['opponent_ranking'] = pd.to_numeric(historial_df['opponent_ranking'], errors='coerce')
    
    # Extraer sets ganados y perdidos del resultado
    sets_split = historial_df['resultado'].str.split('-', expand=True)
    historial_df['sets_ganados'] = pd.to_numeric(sets_split[0], errors='coerce').fillna(0)
    historial_df['sets_perdidos'] = pd.to_numeric(sets_split[1], errors='coerce').fillna(0) if 1 in sets_split.columns else 0

    # Rellenar NaNs que puedan surgir de la conversión de ranking
    historial_df['opponent_ranking'] = historial_df['opponent_ranking'].fillna(9999)

    # 1. Rendimiento contra diferentes niveles de oponentes
    top_10_wins = historial_df[(historial_df['opponent_ranking'] <= 10) & (historial_df['outcome'] == 'Ganó')].shape[0]
    top_10_losses = historial_df[(historial_df['opponent_ranking'] <= 10) & (historial_df['outcome'] == 'Perdió')].shape[0]
    top_50_wins = historial_df[(historial_df['opponent_ranking'] > 10) & (historial_df['opponent_ranking'] <= 50) & (historial_df['outcome'] == 'Ganó')].shape[0]
    top_50_losses = historial_df[(historial_df['opponent_ranking'] > 10) & (historial_df['opponent_ranking'] <= 50) & (historial_df['outcome'] == 'Perdió')].shape[0]

    # 2. Upsets (Victorias como no favorito / Derrotas como favorito)
    upsets_ganados = 0
    upsets_perdidos = 0
    if player_ranking:
        # Asumimos que un ranking > 30 puestos de diferencia es significativo
        upsets_ganados = historial_df[(historial_df['opponent_ranking'] < player_ranking - 30) & (historial_df['outcome'] == 'Ganó')].shape[0]
        upsets_perdidos = historial_df[(historial_df['opponent_ranking'] > player_ranking + 30) & (historial_df['outcome'] == 'Perdió')].shape[0]

    # 3. Patrones de sets
    victorias_2_sets = historial_df[(historial_df['outcome'] == 'Ganó') & (historial_df['sets_perdidos'] == 0)].shape[0]
    victorias_3_sets = historial_df[(historial_df['outcome'] == 'Ganó') & (historial_df['sets_ganados'] > historial_df['sets_perdidos']) & (historial_df['sets_perdidos'] > 0)].shape[0]
    derrotas_3_sets = historial_df[(historial_df['outcome'] == 'Perdió') & (historial_df['sets_ganados'] > 0)].shape[0]

    # 4. Rachas (analizando los últimos 10 partidos)
    ultimos_10 = historial_df.head(10)
    racha_victorias = 0
    racha_actual = 0
    if not ultimos_10.empty:
        for _, row in ultimos_10.iterrows():
            if row['outcome'] == 'Ganó':
                racha_actual += 1
            else:
                racha_victorias = max(racha_victorias, racha_actual)
                racha_actual = 0
        racha_victorias = max(racha_victorias, racha_actual)

    # Construir el resumen de patrones
    patrones = []
    patrones.append(f"Rendimiento vs Top 10: {top_10_wins}V - {top_10_losses}D")
    patrones.append(f"Rendimiento vs Top 11-50: {top_50_wins}V - {top_50_losses}D")
    if upsets_ganados > 0:
        patrones.append(f"Victorias Sorpresivas (Upset): {upsets_ganados} veces")
    if upsets_perdidos > 0:
        patrones.append(f"Derrotas Inesperadas: {upsets_perdidos} veces")
    patrones.append(f"Victorias en sets corridos: {victorias_2_sets}")
    patrones.append(f"Partidos reñidos: {victorias_3_sets} victorias y {derrotas_3_sets} derrotas en 3+ sets.")
    if racha_victorias >= 3:
        patrones.append(f"Mejor racha de victorias (últimos 10 partidos): {racha_victorias}")

    return [f"- {p}" for p in patrones]


def analizar_probabilidad_overs(hist_p1_df, hist_p2_df):
    """
    Analiza la probabilidad de que un partido se extienda en sets y juegos.
    """
    if hist_p1_df.empty or hist_p2_df.empty:
        return {"prob_over_2_5_sets": "N/A", "prob_over_18_5_games_proxy": "N/A"}

    # Extraer sets para ambos historiales
    for df in [hist_p1_df, hist_p2_df]:
        sets_split = df['resultado'].str.split('-', expand=True)
        df['sets_ganados'] = pd.to_numeric(sets_split[0], errors='coerce').fillna(0)
        df['sets_perdidos'] = pd.to_numeric(sets_split[1], errors='coerce').fillna(0) if 1 in sets_split.columns else 0

    # Probabilidad de > 2.5 sets
    p1_over_2_5 = hist_p1_df[(hist_p1_df['sets_ganados'] > 0) & (hist_p1_df['sets_perdidos'] > 0)].shape[0]
    p1_total = hist_p1_df.shape[0]
    p1_prob = (p1_over_2_5 / p1_total) * 100 if p1_total > 0 else 0

    p2_over_2_5 = hist_p2_df[(hist_p2_df['sets_ganados'] > 0) & (hist_p2_df['sets_perdidos'] > 0)].shape[0]
    p2_total = hist_p2_df.shape[0]
    p2_prob = (p2_over_2_5 / p2_total) * 100 if p2_total > 0 else 0

    combined_prob = (p1_prob + p2_prob) / 2

    return {
        "prob_over_2_5_sets": f"{combined_prob:.1f}%",
        "prob_over_18_5_games_proxy": f"Estimada en ~{combined_prob:.1f}% (basado en prob. de 3 sets)"
    }


def format_weights_distribution(f, weights):
    """Formats and writes the weights distribution table."""
    if not weights:
        return
    
    f.write(f"\n--- DISTRIBUCIÓN DE PESOS DEL ANÁLISIS ---\n")

    
    category_map = {
        'h2h_direct': 'H2H Directo', 'common_opponents': 'Rivales Comunes', 'form_recent': 'Forma Reciente',
        'ranking_momentum': 'Ranking/Momentum', 'strength_of_schedule': 'Fuerza Calendario',
        'surface_advantage': 'Ventaja Superficie', 'home_advantage': 'Ventaja Localía',
        'elo_rating': 'Rating ELO', 'surface_specialization': 'Especialización Superficie'
    }

    weights_data = []
    total_weight = sum(weights.values())
    
    for key, value in weights.items():
        weights_data.append({
            'Componente': category_map.get(key, key.replace('_', ' ').title()),
            'Peso (%)': f"{value * 100:.1f}%"
        })
    
    df = pd.DataFrame(weights_data)
    f.write("\n")
    f.write(df.to_markdown(index=False))
    f.write(f"\n\nSuma Total de Pesos: {total_weight * 100:.1f}%\n")
    # D53-04: suma de pesos debe ser 100% ± 0.5% (assert en producción visible en output)
    if abs(total_weight - 1.0) >= 0.005:
        f.write(f"ALERTA D53-04: pesos suman {total_weight*100:.2f}% (esperado 100% +-0.5%). Bug de reconstruccion de pesos desde logs -- ver Nodo-56. Con _weights_final este alerta deberia desaparecer.\n")
    f.write("\n")
    
    # Validar pesos usando el módulo de normalización
    if NORMALIZATION_AVAILABLE:
        is_valid, message = validate_weights(weights)
        f.write("--- VALIDACIÓN DE PESOS USADOS ---\n")
        if is_valid:
            f.write("✅ Los pesos están correctamente normalizados (suman 100%)\n")
        else:
            f.write(f"⚠️ Problema con los pesos: {message}\n")
        f.write("\n")


def predecir_sets_y_games(score_difference, p1_score, p2_score):
    """
    Predice el número de sets y un rango de juegos basado en la diferencia de puntaje.
    """
    # Normalizar la diferencia de puntaje a un valor absoluto
    diff = abs(score_difference)

    # Lógica de predicción de sets
    if diff > 0.18:  # Diferencia considerable -> Partido a 2 sets
        predicted_sets = "2"
        reason = "La diferencia de puntaje sugiere un claro favorito."
        # Lógica de predicción de juegos para 2 sets
        if diff > 0.35: # Dominio total
            predicted_games = "16-19" # e.g., 6-2, 6-3
        elif diff > 0.25: # Favorito claro
            predicted_games = "18-21" # e.g., 6-4, 6-3
        else: # Ligeramente favorito
            predicted_games = "20-23" # e.g., 7-5, 6-4
    else:  # Diferencia pequeña -> Partido a 3 sets
        predicted_sets = "3"
        reason = "La paridad en los puntajes sugiere un partido reñido."
        # Lógica de predicción de juegos para 3 sets
        # Un partido a 3 sets casi siempre tiene más de 20 juegos.
        # La suma de los puntajes puede dar una idea de la intensidad.
        total_score = p1_score + p2_score
        if total_score > 1.5: # Ambos jugadores con puntajes altos
             predicted_games = "26-32+" # e.g., 6-4, 5-7, 6-4
        else:
             predicted_games = "23-28"   # e.g., 7-5, 4-6, 6-2

    return {
        "predicted_sets": predicted_sets,
        "predicted_games": predicted_games,
        "reason": reason
    }


def generar_resumen_consolidado(f, p1_name, p2_name, score_breakdown, scores):
    """Genera y escribe una tabla consolidada con el resumen de todos los puntos."""
    f.write(f"\n--- CONSOLIDADO DE PUNTUACIÓN ---\n")

    
    p1_breakdown = score_breakdown.get('player1')
    p2_breakdown = score_breakdown.get('player2')

    if not isinstance(p1_breakdown, dict): p1_breakdown = {}
    if not isinstance(p2_breakdown, dict): p2_breakdown = {}

    category_map = {
        'h2h_direct': 'H2H Directo', 'common_opponents': 'Rivales Comunes', 'form_recent': 'Forma Reciente',
        'ranking_momentum': 'Ranking/Momentum', 'strength_of_schedule': 'Fuerza Calendario',
        'surface_advantage': 'Ventaja Superficie', 'home_advantage': 'Ventaja Localía',
        'elo_rating': 'Rating ELO', 'surface_specialization': 'Especialización Superficie'
    }
    
    # Usar una lista predefinida de claves para asegurar el orden y la relevancia
    valid_keys = ['surface_specialization', 'form_recent', 'common_opponents', 'h2h_direct', 'ranking_momentum', 'elo_rating', 'home_advantage', 'strength_of_schedule']
    summary_data = []

    for key in valid_keys:
        p1_category_data = p1_breakdown.get(key)
        p2_category_data = p2_breakdown.get(key)

        # Solo procesar si la clave existe para al menos un jugador
        if p1_category_data is not None or p2_category_data is not None:
            try:
                p1_score = float(p1_category_data.get('weighted_score', 0)) if isinstance(p1_category_data, dict) else 0.0
            except (ValueError, TypeError):
                p1_score = 0.0
            
            try:
                p2_score = float(p2_category_data.get('weighted_score', 0)) if isinstance(p2_category_data, dict) else 0.0
            except (ValueError, TypeError):
                p2_score = 0.0

            summary_data.append({
                'Componente': category_map.get(key, key.replace('_', ' ').title()),
                f'Puntos {p1_name}': f"{p1_score:.4f}",
                f'Puntos {p2_name}': f"{p2_score:.4f}"
            })

    if not summary_data:
        f.write("Desglose de puntuación no disponible.\n\n")
        return

    try:
        p1_final_score = float(scores.get('p1_final_weight', 0))
    except (ValueError, TypeError):
        p1_final_score = 0.0

    try:
        p2_final_score = float(scores.get('p2_final_weight', 0))
    except (ValueError, TypeError):
        p2_final_score = 0.0

    # D56-05: mostrar penalización de inactividad cuando existe (días_desde > 30)
    # sin esto: sum(componentes)=3.77 pero PUNTAJE FINAL=1.89 parece un error de suma
    def _parse_penalty(s):
        try:
            return float(str(s).replace(' pts', ''))
        except (ValueError, TypeError):
            return 0.0

    p1_penalty = _parse_penalty(p1_breakdown.get('Penalizacion_Inactividad', '0.00 pts'))
    p2_penalty = _parse_penalty(p2_breakdown.get('Penalizacion_Inactividad', '0.00 pts'))

    if p1_penalty != 0.0 or p2_penalty != 0.0:
        summary_data.append({
            'Componente': 'Penalizacion Inactividad',
            f'Puntos {p1_name}': f"{p1_penalty:.4f}",
            f'Puntos {p2_name}': f"{p2_penalty:.4f}"
        })

    df = pd.DataFrame(summary_data)
    total_row = {
        'Componente': 'PUNTAJE FINAL TOTAL',
        f'Puntos {p1_name}': f"{p1_final_score:.4f}",
        f'Puntos {p2_name}': f"{p2_final_score:.4f}"
    }
    total_df = pd.DataFrame([total_row])
    final_df = pd.concat([df, total_df], ignore_index=True)
    f.write("\n")
    f.write(final_df.to_markdown(index=False))
    f.write("\n\n")


def analyze_component_status(score_breakdown, weights):
    """
    Analiza qué componentes tienen datos y cuáles están ausentes.
    Returns: (active_components, inactive_components, inactive_reason)
    """
    active = []
    inactive = []
    reason = {}
    
    # Los componentes están activos si tienen datos en score_breakdown
    if score_breakdown:
        p1_breakdown = score_breakdown.get('player1', {})
        p2_breakdown = score_breakdown.get('player2', {})
        
        for component in weights.keys():
            p1_has_data = p1_breakdown.get(component) is not None and p1_breakdown.get(component) != {}
            p2_has_data = p2_breakdown.get(component) is not None and p2_breakdown.get(component) != {}
            
            if p1_has_data or p2_has_data:
                active.append(component)
            else:
                inactive.append(component)
                reason[component] = "Sin datos disponibles"
    else:
        # Si no hay breakdown, asumir que todos están activos
        active = list(weights.keys())
    
    return active, inactive, reason


def format_component_status(f, score_breakdown, weights):
    """
    Muestra el estado de cada componente (con datos vs sin datos).
    Útil para entender la redistribución de pesos.
    """
    if not NORMALIZATION_AVAILABLE or not weights:
        return
    
    active, inactive, reason = analyze_component_status(score_breakdown, weights)
    
    f.write("--- ESTADO DE COMPONENTES DEL ANÁLISIS ---\n\n")
    
    # Mostrar componentes con datos
    f.write("✅ COMPONENTES CON DATOS:\n")
    if active:
        for comp in active:
            display_name = comp.replace('_', ' ').title()
            weight = weights.get(comp, 0) * 100
            f.write(f"  • {display_name}: {weight:.0f}% del peso\n")
    else:
        f.write("  Ninguno\n")
    
    f.write("\n")
    
    # Mostrar componentes sin datos
    f.write("⚠️ COMPONENTES SIN DATOS (peso redistribuido):\n")
    if inactive:
        for comp in inactive:
            display_name = comp.replace('_', ' ').title()
            original_weight = DEFAULT_WEIGHTS.get(comp, 0) * 100
            f.write(f"  • {display_name}: -{original_weight:.0f}% (redistribuido)\n")
    else:
        f.write("  Ninguno\n")
    
    f.write("\n")
    
    # Si hay componentes inactivos, calcular pesos ajustados
    if inactive and active:
        manager = WeightManager(DEFAULT_WEIGHTS, active, reason)
        adjusted = manager.get_adjusted_weights()
        
        f.write("--- PESOS AJUSTADOS (tras redistribución) ---\n")
        f.write("Los pesos originales se ajustaron proporcionalmente:\n\n")
        
        for comp in adjusted:
            display_name = comp.replace('_', ' ').title()
            new_weight = adjusted[comp] * 100
            original = DEFAULT_WEIGHTS.get(comp, 0) * 100
            
            if comp in inactive:
                f.write(f"  • {display_name}: 0% (sin datos)\n")
            else:
                change = new_weight - original
                change_str = f"+{change:.1f}%" if change > 0 else f"{change:.1f}%"
                f.write(f"  • {display_name}: {new_weight:.1f}% ({change_str} por redistribución)\n")
        
        f.write("\n")
    
    # Total
    total_adjusted = sum(weights.values()) if weights else sum(DEFAULT_WEIGHTS.values())
    f.write(f"Peso total utilizado: {total_adjusted * 100:.1f}%\n\n")


def analyze_matches_with_pandas(file_path, output_filename="analisis_partidos_pandas.txt", tier_filter=None):
    """
    Reads match data from a JSON file, analyzes it using pandas,
    and generates a text file with structured tables.
    Optionally filters by tier (grand_slam, atp1000, atp500, challenger, itf).
    """
    try:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1200)
        pd.set_option('display.colheader_justify', 'center')

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        matches = data.get('partidos', [])
        metadata = data.get('metadata', {})

        # Filtrar por tier si se especifica
        if tier_filter and TIER_AVAILABLE:
            matches_filtered = []
            for m in matches:
                torneo = m.get('torneo_nombre', '')
                tier = detectar_tier(torneo)
                if tier == tier_filter:
                    matches_filtered.append(m)
            total_antes = len(matches)
            matches = matches_filtered
            print(f"Filtro tier '{tier_filter}': {len(matches)}/{total_antes} partidos")
            if not matches:
                print(f"⚠️  Sin partidos para tier '{tier_filter}'. Tiers disponibles:")
                all_matches = data.get('partidos', [])
                tiers = {}
                for m in all_matches:
                    t = detectar_tier(m.get('torneo_nombre', ''))
                    tiers[t] = tiers.get(t, 0) + 1
                for t, n in sorted(tiers.items()):
                    print(f"     {t}: {n} partidos")
                return

        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("🎾 ANÁLISIS DE PARTIDOS DE TENIS - SISTEMA DE PREDICCIONES\n")
            f.write("=" * 80 + "\n\n")
            
            # Incluir información de normalización al inicio del reporte
            if NORMALIZATION_AVAILABLE:
                show_normalization_transparency(f)

            
            if metadata:
                f.write("\n--- METADATOS DEL ANÁLISIS ---\n")

                meta_df = pd.DataFrame([metadata])
                f.write("\n")
                f.write(meta_df.to_markdown(index=False))
                f.write("\n\n")

            # Cargar edge_report para edge_vs_mercado (Fase E D53-01/ADDENDUM-3)
            _edge_lookup = _load_edge_report()

            # D65-06: cargar calibracion_edge para WARN superficie discrepante (Nodo-65)
            _calib_por_sup_65 = {}
            _dom_sup_65 = None
            try:
                _calib_path_65 = os.path.join(os.path.dirname(__file__), 'data', 'calibracion_edge.json')
                with open(_calib_path_65) as _cf65:
                    _calib_json_65 = json.load(_cf65)
                _calib_por_sup_65 = _calib_json_65.get('por_superficie', {})
                # Superficie dominante = max total games, excluyendo 'unknown' y '?'
                _dom_sup_games_65 = 0
                for _sup_k65, _sup_v65 in _calib_por_sup_65.items():
                    if _sup_k65 in ('unknown', '?'):
                        continue
                    _sup_total_65 = _sup_v65.get('wins', 0) + _sup_v65.get('losses', 0)
                    if _sup_total_65 > _dom_sup_games_65:
                        _dom_sup_games_65 = _sup_total_65
                        _dom_sup_65 = _sup_k65
            except Exception:
                pass

            for match in matches:
                if not match:
                    continue
                match_number = match.get('match_number', 'N/A')
                p1 = match.get('jugador1', 'N/A')
                p2 = match.get('jugador2', 'N/A')
                
                f.write("="*150 + "\n")
                f.write(f"Partido #{match_number}: {p1} vs {p2}\n")
                torneo = match.get('torneo_nombre', 'N/A')
                cancha = (match.get('tipo_cancha') or 'N/A').capitalize()
                f.write(f"Torneo: {torneo} | Superficie: {cancha}\n")
                match_url = match.get('match_url')
                if match_url:
                    f.write(f"URL del Partido: {match_url}\n")
                f.write("="*150 + "\n\n")

                # Nodo-35: alerta si historial vacío en cualquiera de los dos jugadores
                _dq = match.get('data_quality', {})
                _sin_p1 = not _dq.get('historial_extraido_p1', True)
                _sin_p2 = not _dq.get('historial_extraido_p2', True)
                if _sin_p1 or _sin_p2:
                    _sin_nombres = []
                    if _sin_p1:
                        _sin_nombres.append(p1)
                    if _sin_p2:
                        _sin_nombres.append(p2)
                    f.write("!" * 80 + "\n")
                    f.write(f"SIN HISTORIAL EXTRAIDO PARA: {', '.join(_sin_nombres)}\n")
                    f.write(f"BLOQUEADO EN extraer_historh2h.py. NO APOSTAR.\n")
                    f.write("!" * 80 + "\n\n")

                ranking_analysis = match.get('ranking_analysis') or {}
                prediction = ranking_analysis.get('prediction') or {}
                scores = prediction.get('scores') or {}
                # Bug fix: score_breakdown debe definirse ANTES de usarse en el bloque
                # de reasoning (línea ~744). Si se define después se produce NameError
                # en el primer partido y datos rancios del partido anterior en el resto.
                score_breakdown = prediction.get('score_breakdown', {}) or {}

                # Extraer reasoning ANTES de escribir señales (Fase E: señales al inicio)
                reasoning = prediction.get('reasoning', [])
                # D56-02: usar _weights_final (fuente única de verdad) si está disponible
                weights = prediction.get('_weights_final') or get_weights_from_reasoning(reasoning)

                # --- SEÑALES ESPECIALES al inicio del bloque de partido (Fase E D53 ADDENDUM-3) ---
                _special_signals = []
                for reason in reasoning:
                    if 'TORNEO_COMPLETO_BONUS' in reason:
                        _clean = reason.replace('P1_LOG_SURF: ', '').replace('P2_LOG_SURF: ', '')
                        _player = p1 if 'P1_LOG_SURF' in reason else p2
                        _special_signals.append(f"CAMPEON DE TORNEO: {_player} -- {_clean}")
                    if 'LOG_E1_TORNEO_WEIGHT' in reason:
                        _special_signals.append(f"AJUSTE DINAMICO DE PESOS: {reason}")
                    if 'LOG_MARKOV' in reason and 'estado=HOT' in reason:
                        _player = p1 if '_P1' in reason else p2
                        _wr = re.search(r'wr_rec=([\d.]+)', reason)
                        _wr_str = f" ({float(_wr.group(1))*100:.0f}% reciente)" if _wr else ""
                        _special_signals.append(f"RACHA CALIENTE: {_player} en estado HOT{_wr_str}")
                    if 'motivo_reclasificacion' in reason:
                        _special_signals.append(f"FILTRO MARKOV-BBI: {reason}")
                    if '_LOG_SURF' in reason and 'Victoria vs Rank' in reason:
                        _rank_match = re.search(r'Victoria vs Rank (\d+) \(([^)]+)\)', reason)
                        if _rank_match:
                            _opp_rank_int = int(_rank_match.group(1))
                            _tier_from_weights = ''
                            for _r in reasoning:
                                if 'LOG_WEIGHTS_STRATEGY' in _r:
                                    _tm = re.search(r"'(\w+)'", _r)
                                    if _tm:
                                        _tier_from_weights = _tm.group(1)
                                    break
                            _scalp_thresholds = {
                                'grand_slam': 10, 'atp1000': 10, 'atp500': 20,
                                'challenger': 50, 'itf': 100
                            }
                            _threshold = _scalp_thresholds.get(_tier_from_weights, 20)
                            if _opp_rank_int <= _threshold:
                                _player = p1 if 'P1_LOG_SURF' in reason else p2
                                _opp_name = _rank_match.group(2)
                                _special_signals.append(
                                    f"SCALP TOP-{_threshold} EN SUPERFICIE: {_player} vencio a {_opp_name} (#{_opp_rank_int}) en esta superficie"
                                )
                    if 'TORNEO_COMPLETO_EXPIRADO' in reason:
                        # D57-04: atribuir señal al jugador correcto
                        _player = p1 if 'P1_LOG_SURF' in reason else p2
                        _clean = reason.replace('P1_LOG_SURF: ', '').replace('P2_LOG_SURF: ', '')
                        _special_signals.append(f"CAMPEON ANTERIOR EN SUPERFICIE: {_player} -- {_clean}")
                    if 'LOG_FORM_DECAY' in reason:
                        # D57-05: señal de inactividad visible al usuario
                        # Nodo-63: no mostrar INACTIVIDAD si LOG_INSUFFICIENT_HISTORY activo
                        _insuf_p1 = any('LOG_INSUFFICIENT_HISTORY' in r and p1 in r for r in reasoning)
                        _insuf_p2 = any('LOG_INSUFFICIENT_HISTORY' in r and p2 in r for r in reasoning)
                        _fd_p1m = re.search(r'fd_p1=([\d.]+)', reason)
                        _fd_p2m = re.search(r'fd_p2=([\d.]+)', reason)
                        _dp1m   = re.search(r'p1_days=([-\d]+)', reason)
                        _dp2m   = re.search(r'p2_days=([-\d]+)', reason)
                        for _pn, _fdm, _dm, _insuf in [(p1, _fd_p1m, _dp1m, _insuf_p1), (p2, _fd_p2m, _dp2m, _insuf_p2)]:
                            if _fdm and _dm and not _insuf:
                                _fdv = float(_fdm.group(1))
                                _dv  = int(_dm.group(1))
                                if _fdv < 1.0 and _dv > 30:
                                    _special_signals.append(
                                        f"INACTIVIDAD: {_pn} -- {_dv}d sin jugar -> "
                                        f"form_recent x{_fdv:.2f} (decay exponencial Nodo-57)"
                                    )

                _prof_data = _load_profitability_data()
                for _player_name in [p1, p2]:
                    _prof_key = _normalize_player_name_for_prof(_player_name)
                    _prof = _prof_data.get(_prof_key)
                    if _prof and _prof.get('n_apostado', 0) >= 3 and _prof.get('roi', 0) > 0:
                        _special_signals.append(
                            f"JUGADOR RENTABLE: {_player_name} -- "
                            f"{_prof['n_apostado']} apuestas, {_prof['n_ganado']} ganadas, "
                            f"ROI +{_prof['roi']*100:.0f}%, cuota prom {_prof['avg_cuota']:.2f}"
                        )

                if _special_signals:
                    f.write("--- SENALES ESPECIALES DETECTADAS ---\n")
                    for sig in _special_signals:
                        f.write(f"  >> {sig}\n")
                    f.write("\n")

                # Prediction Summary
                p1_ranking_key = next((k for k in ranking_analysis if k.endswith('_ranking') and p1.split(' ')[0] in k), None)
                p2_ranking_key = next((k for k in ranking_analysis if k.endswith('_ranking') and p2.split(' ')[0] in k), None)

                summary_data = {
                    'Jugador': [p1, p2],
                    'Ranking': [ranking_analysis.get(p1_ranking_key, 'N/A'), ranking_analysis.get(p2_ranking_key, 'N/A')],
                    'Cuota': [match.get('cuota1', 'N/A'), match.get('cuota2', 'N/A')],
                    'Puntaje Final': [scores.get('p1_final_weight', 'N/A'), scores.get('p2_final_weight', 'N/A')]
                }
                summary_df = pd.DataFrame(summary_data)
                f.write("\n--- RESUMEN DE PREDICCIÓN ---\n")
                f.write(summary_df.to_markdown(index=False))
                _confidence = prediction.get('confidence', 0) or 0
                f.write(f"\n\nJugador Favorito: {prediction.get('favored_player', 'N/A')}\n")
                f.write(f"Confianza: {_confidence}%\n")

                # Fase E: banda NO-BET + edge_vs_mercado (ADDENDUM-3)
                if _confidence and float(_confidence) < 54:
                    f.write(f"ACCION: NO-BET — confianza {_confidence}% < 54% (umbral minimo). Coin flip.\n")
                else:
                    f.write(f"ACCION: EVALUAR — confianza {_confidence}%\n")

                # edge_vs_mercado: reutiliza p_implicita de edge_report (no recalcula)
                _partido_key = f"{p1} vs {p2}"
                _partido_key_inv = f"{p2} vs {p1}"
                _edge_pick = _edge_lookup.get(_partido_key) or _edge_lookup.get(_partido_key_inv)
                if _edge_pick:
                    _p_modelo_fav = _edge_pick.get('p_modelo', 0)
                    _p_impl_fav = _edge_pick.get('p_implicita', 0)
                    _favorito_edge = _edge_pick.get('favorito_predicho', prediction.get('favored_player', ''))
                    _edge_fav = _p_modelo_fav - _p_impl_fav
                    _edge_rival = -_edge_fav
                    _p_modelo_rival = 1.0 - _p_modelo_fav
                    _p_impl_rival = 1.0 - _p_impl_fav
                    _cuota_fav = _edge_pick.get('cuota_favorito', 'N/A')
                    _cuota_rival_val = _edge_pick.get('cuota_rival', 'N/A')
                    # D65-02: mostrar edge del jugador favorito SIEMPRE con signo explícito (H77-02 Nodo-65)
                    _edge_sign = '+' if _edge_fav >= 0 else ''
                    _edge_label = 'POSITIVO' if _edge_fav >= 0 else 'NEGATIVO'
                    f.write(
                        f"edge_vs_mercado: {_favorito_edge} {_edge_sign}{_edge_fav*100:.1f}% [{_edge_label}]"
                        f" (modelo {_p_modelo_fav*100:.1f}% vs bookmaker {_p_impl_fav*100:.1f}%, cuota {_cuota_fav})\n"
                    )
                    if _edge_fav < 0:
                        # El rival tiene edge positivo — mostrarlo también para contexto completo
                        _rival_name = p2 if _favorito_edge == p1 else p1
                        f.write(
                            f"edge_vs_mercado_rival: {_rival_name} +{_edge_rival*100:.1f}% [POSITIVO]"
                            f" (modelo {_p_modelo_rival*100:.1f}% vs bookmaker {_p_impl_rival*100:.1f}%, cuota {_cuota_rival_val})\n"
                        )

                    # D65-06: WARN superficie discrepante vs calibración dominante (Nodo-65)
                    _match_sup_65 = (match.get('tipo_cancha') or '').lower()
                    if (_dom_sup_65 and _match_sup_65
                            and _match_sup_65 not in ('unknown', '?', '')
                            and _match_sup_65 != _dom_sup_65):
                        _dom_d65 = _calib_por_sup_65.get(_dom_sup_65, {})
                        _match_d65 = _calib_por_sup_65.get(_match_sup_65, {})
                        _dom_n65 = _dom_d65.get('wins', 0) + _dom_d65.get('losses', 0)
                        _match_n65 = _match_d65.get('wins', 0) + _match_d65.get('losses', 0)
                        _dom_hit65 = round(_dom_d65.get('wins', 0) / _dom_n65 * 100, 1) if _dom_n65 > 0 else 0.0
                        _match_hit65 = round(_match_d65.get('wins', 0) / _match_n65 * 100, 1) if _match_n65 > 0 else 0.0
                        if _dom_hit65 - _match_hit65 >= 5.0:
                            f.write(
                                f"WARN_SUPERFICIE: partido en {_match_sup_65} (hit={_match_hit65}% n={_match_n65})"
                                f" vs superficie dominante {_dom_sup_65} (hit={_dom_hit65}% n={_dom_n65})"
                                f" — menor calibracion historica. Verificar señales adicionales.\n"
                            )

                    # D64-02: señal RFI en tabla favoritos (Nodo-64)
                    _rfi_tier_v = _edge_pick.get('rfi_tier', 0) or 0
                    _rfi_ultra = _edge_pick.get('rfi_ultra', False)
                    _rfi_days = _edge_pick.get('rfi_days_inactive')
                    _rfi_bookie_fav = _edge_pick.get('rfi_is_bookie_fav', False)
                    _rfi_decay = _edge_pick.get('rfi_decay_gap')
                    if _rfi_tier_v >= 2:
                        _rfi_label = 'RFI-ULTRA' if _rfi_ultra else f'RFI-{_rfi_tier_v}'
                        _rfi_days_str = f'{_rfi_days}d' if _rfi_days is not None else '?d'
                        _rfi_fav_str = ' + FAVORITO BOOKMAKER (error sistematico de mercado)' if _rfi_bookie_fav else ''
                        _rfi_decay_str = f' | decay_gap={_rfi_decay:.2f}' if _rfi_decay is not None else ''
                        f.write(
                            f"{_rfi_label}: jugador inactivo {_rfi_days_str}{_rfi_fav_str}{_rfi_decay_str}\n"
                        )

                    # D68-06: señal Rival Value Flip en tabla favoritos (Nodo-68)
                    # OBSERVACIONAL — H88-01 acumula; NO modifica apostar ni kelly.
                    if _edge_pick.get('rival_value_flag', False):
                        _rv_cuota = _edge_pick.get('cuota_rival')
                        _rv_edge  = _edge_pick.get('edge_vs_mercado_rival')
                        _rv_vig   = _edge_pick.get('vig')
                        _rv_cuota_str = f"{_rv_cuota:.2f}" if _rv_cuota is not None else '?'
                        _rv_edge_str  = (f"+{_rv_edge*100:.1f}%" if _rv_edge is not None and _rv_edge >= 0
                                         else f"{_rv_edge*100:.1f}%" if _rv_edge is not None else '?')
                        _rv_vig_str   = f"{_rv_vig*100:.1f}%" if _rv_vig is not None else '?'
                        f.write(
                            f"RIVAL VALUE (H88-01): cuota_rival={_rv_cuota_str}"
                            f" | edge_vs_rival={_rv_edge_str}"
                            f" | vig={_rv_vig_str}"
                            f" — señal observacional, no apuesta automatica\n"
                        )

                f.write(f"Diferencia de Puntaje: {scores.get('score_difference', 'N/A')}\n\n")

                # Predicción de Sets y Games
                score_diff_val = scores.get('score_difference', 0)
                p1_final_score = scores.get('p1_final_weight', 0)
                p2_final_score = scores.get('p2_final_weight', 0)

                set_game_prediction = predecir_sets_y_games(score_diff_val, p1_final_score, p2_final_score)

                f.write("\n--- PREDICCIÓN DE SETS Y GAMES ---\n")

                f.write(f"Sets Pronosticados: {set_game_prediction['predicted_sets']}\n")
                f.write(f"Games Pronosticados: {set_game_prediction['predicted_games']}\n")
                f.write(f"Justificación: {set_game_prediction['reason']}\n\n")

                # Weights distribution (reasoning ya extraído arriba)
                format_weights_distribution(f, weights)

                if reasoning:
                    f.write("\n--- RAZONAMIENTO CLAVE Y LOGS DE PREDICCIÓN ---\n")

                    # Extraer y mostrar la contundencia del H2H si existe
                    h2h_reason = next((r for r in reasoning if 'H2H_Directo' in r and 'contundencia' in r), None)
                    if h2h_reason:
                        f.write(f"Análisis H2H: {h2h_reason}\n\n")

                    f.write("--- Lógica de Ponderación y Scores Detallados ---\n")

                    for reason in reasoning:
                        # Mostrar todos los logs para máxima transparencia
                        if 'H2H_Directo' not in reason: # Evita duplicar la info de H2H
                            f.write(f"- {reason}\n")
                    f.write("\n")
                    
                    # Mostrar estado de componentes si hay breakdown
                    if score_breakdown and NORMALIZATION_AVAILABLE:
                        format_component_status(f, score_breakdown, weights)

                if 'enfrentamientos_directos' in match and match['enfrentamientos_directos']:
                    df_h2h = pd.DataFrame(match['enfrentamientos_directos'])
                    f.write("\n--- Enfrentamientos Directos ---\n")
                    f.write(df_h2h[['fecha', 'ganador', 'resultado', 'superficie']].to_markdown(index=False))
                    f.write("\n\n")
                    if score_breakdown:
                        format_score_breakdown(f, p1, p2, score_breakdown, 'h2h_direct', 'Desglose H2H', weights)

                if 'common_opponents_detailed' in match and match['common_opponents_detailed']:
                    common_opponents_count = ranking_analysis.get('common_opponents_count', len(match['common_opponents_detailed']))
                    f.write(f"\n--- Rivales Comunes (Total: {common_opponents_count}) ---\n")

                    
                    df_common = pd.json_normalize(match['common_opponents_detailed'])
                    # Ordenar por la fecha más antigua entre ambos jugadores (DD.MM.YYYY → cronológico)
                    for _date_col in ['player1_result.date', 'player2_result.date']:
                        if _date_col not in df_common.columns:
                            df_common[_date_col] = 'N/A'
                    def _parse_date(d):
                        try:
                            return pd.to_datetime(d, format='%d.%m.%Y')
                        except Exception:
                            return pd.NaT
                    _p1_dates = df_common['player1_result.date'].apply(_parse_date)
                    _p2_dates = df_common['player2_result.date'].apply(_parse_date)
                    df_common['_min_date'] = pd.DataFrame({'a': _p1_dates, 'b': _p2_dates}).min(axis=1)
                    df_common = df_common.sort_values('_min_date').drop(columns=['_min_date']).reset_index(drop=True)
                    cols_to_show = {
                        'opponent_name': 'Oponente Común', 'opponent_ranking': 'Rank',
                        'player1_result.date': f'Fecha {p1}', 'player1_result.outcome': f'Res. {p1}', 'player1_result.score': 'Score', 'player1_result.surface': f'Sup. {p1}',
                        'player2_result.date': f'Fecha {p2}', 'player2_result.outcome': f'Res. {p2}', 'player2_result.score': 'Score', 'player2_result.surface': f'Sup. {p2}',
                        'advantage_for': 'Ventaja'
                    }
                    for col in cols_to_show.keys():
                        if col not in df_common.columns:
                            df_common[col] = 'N/A'
                    df_common_display = df_common[list(cols_to_show.keys())].rename(columns=cols_to_show)
                    f.write("\n")
                    f.write(df_common_display.to_markdown(index=False))
                    f.write("\n\n")

                    f.write("\n--- Justificación Detallada de Ventaja (Rivales Comunes) ---\n")

                    # Renombrar columnas duplicadas si existen ('Score')
                    cols = pd.Series(df_common_display.columns)
                    for dup in cols[cols.duplicated()].unique(): 
                        cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
                    df_common_display.columns = cols

                    for _, row in df_common_display.iterrows():
                        # Usar .get para evitar errores si una columna no existe
                        opponent_dict = {
                            'opponent_name': row.get('Oponente Común'),
                            'advantage_for': row.get('Ventaja'),
                            'player1_result': {
                                'outcome': row.get(f'Res. {p1}'),
                                'score': row.get('Score')
                            },
                            'player2_result': {
                                'outcome': row.get(f'Res. {p2}'),
                                'score': row.get('Score.1', row.get('Score')) # Handle pandas renaming
                            }
                        }
                        explanation = explicar_ventaja_rival_comun(opponent_dict, p1, p2)
                        f.write(f"- {row.get('Oponente Común')}: {explanation}\n")
                    f.write("\n")

                    if score_breakdown:
                        format_score_breakdown(f, p1, p2, score_breakdown, 'common_opponents', 'Desglose Rivales Comunes', weights)

                if 'form_analysis' in match and match.get('form_analysis'):
                    form_data = match.get('form_analysis')
                    p1_form_key = next((key for key in form_data if p1.split(' ')[0] in key), None)
                    p2_form_key = next((key for key in form_data if p2.split(' ')[0] in key and key != p1_form_key), None)
                    
                    p1_form = form_data.get(p1_form_key) or {} if p1_form_key else {}
                    p2_form = form_data.get(p2_form_key) or {} if p2_form_key else {}
                    
                    df_form_data = {
                        'Jugador': [p1, p2],
                        'Partidos Recientes': [p1_form.get('recent_matches_count', 'N/A'), p2_form.get('recent_matches_count', 'N/A')],
                        'Victorias': [p1_form.get('wins', 'N/A'), p2_form.get('wins', 'N/A')],
                        'Derrotas': [p1_form.get('losses', 'N/A'), p2_form.get('losses', 'N/A')],
                        '% Victorias': [f"{p1_form.get('win_percentage', 0):.1f}%", f"{p2_form.get('win_percentage', 0):.1f}%"],
                        'Racha Actual': [f"{p1_form.get('current_streak_count')} {p1_form.get('current_streak_type', '')}", f"{p2_form.get('current_streak_count')} {p2_form.get('current_streak_type', '')}"],
                        'Estado Forma': [p1_form.get('form_status', 'N/A'), p2_form.get('form_status', 'N/A')]
                    }
                    df_form = pd.DataFrame(df_form_data)
                    f.write("\n--- Análisis de Forma Reciente ---\n")
                    f.write("\n")
                    f.write(df_form.to_markdown(index=False))
                    f.write("\n\n")
                    if score_breakdown:
                        format_score_breakdown(f, p1, p2, score_breakdown, 'form_recent', 'Desglose Forma Reciente', weights)

                if score_breakdown and 'ranking_momentum' in (score_breakdown.get('player1') or {}):
                     format_score_breakdown(f, p1, p2, score_breakdown, 'ranking_momentum', 'Desglose Ranking & Momentum', weights)
                     # Llamada a la nueva función para el desglose detallado
                     if reasoning:
                         format_ranking_momentum_details(f, p1, p2, reasoning, ranking_analysis)

                if score_breakdown and 'elo_rating' in (score_breakdown.get('player1') or {}):
                    format_score_breakdown(f, p1, p2, score_breakdown, 'elo_rating', 'Desglose Rating ELO', weights)

                if score_breakdown and 'strength_of_schedule' in (score_breakdown.get('player1') or {}):
                    format_score_breakdown(f, p1, p2, score_breakdown, 'strength_of_schedule', 'Desglose Fuerza del Calendario (SoS)', weights)

                # Análisis de Superficie
                if 'surface_analysis' in match and match.get('surface_analysis'):
                    f.write("\n--- ANÁLISIS POR SUPERFICIE ---\n")

                    surface_data = match.get('surface_analysis')
                    p1_surface_key = next((k for k in surface_data if p1.split(' ')[0] in k), None)
                    p2_surface_key = next((k for k in surface_data if p2.split(' ')[0] in k), None)

                    if p1_surface_key and p2_surface_key:
                        p1_stats = surface_data.get(p1_surface_key) or {}
                        p2_stats = surface_data.get(p2_surface_key) or {}
                        
                        surfaces = sorted(list(set(p1_stats.keys()) | set(p2_stats.keys())))
                        
                        table_data = []
                        for surface in surfaces:
                            p1_s = p1_stats.get(surface) or {}
                            p2_s = p2_stats.get(surface) or {}
                            table_data.append({
                                'Superficie': surface,
                                f'{p1} V-D': f"{p1_s.get('wins', 0)}-{p1_s.get('losses', 0)}",
                                f'{p1} %': f"{p1_s.get('win_rate', 0):.1f}%",
                                f'{p2} V-D': f"{p2_s.get('wins', 0)}-{p2_s.get('losses', 0)}",
                                f'{p2} %': f"{p2_s.get('win_rate', 0):.1f}%"
                            })
                        
                        df_surface = pd.DataFrame(table_data)
                        f.write("\n")
                        f.write(df_surface.to_markdown(index=False))
                    f.write("\n\n")
                    if score_breakdown:
                        # Try surface_specialization first (new component), fallback to surface_advantage
                        _sb_p1 = (score_breakdown.get('player1') or {})
                        if _sb_p1.get('surface_specialization'):
                            format_score_breakdown(f, p1, p2, score_breakdown, 'surface_specialization', 'Desglose Especialización Superficie', weights)
                        else:
                            format_score_breakdown(f, p1, p2, score_breakdown, 'surface_advantage', 'Desglose Ventaja por Superficie', weights)

                # Análisis de Ubicación (Ventaja de Local y Confort Regional)
                if 'location_analysis' in match and match.get('location_analysis'):
                    f.write("\n--- ANÁLISIS POR UBICACIÓN ---\n")

                    location_data = match.get('location_analysis')
                    p1_loc_key = next((k for k in location_data if p1.split(' ')[0] in k), None)
                    p2_loc_key = next((k for k in location_data if p2.split(' ')[0] in k), None)

                    if p1_loc_key and p2_loc_key:
                        p1_loc = location_data.get(p1_loc_key) or {}
                        p2_loc = location_data.get(p2_loc_key) or {}

                        # Ventaja de Local
                        p1_home = p1_loc.get('home_advantage') or {}
                        p2_home = p2_loc.get('home_advantage') or {}
                        home_data = {
                            'Análisis': ['Ventaja de Local (Jugando en casa)'],
                            f'{p1} (V-D)': [f"{p1_home.get('wins', 0)}-{p1_home.get('losses', 0)}"],
                            f'{p1} %': [f"{p1_home.get('win_rate', 0):.1f}%"],
                            f'{p2} (V-D)': [f"{p2_home.get('wins', 0)}-{p2_home.get('losses', 0)}"],
                            f'{p2} %': [f"{p2_home.get('win_rate', 0):.1f}%"]
                        }
                        df_home = pd.DataFrame(home_data)
                        f.write("\n")
                        f.write(df_home.to_markdown(index=False))
                        f.write("\n\n")


                        # Confort Regional
                        f.write("\n--- Confort Regional (Rendimiento por Continente) ---\n")

                        p1_regional = p1_loc.get('regional_comfort') or {}
                        p2_regional = p1_loc.get('regional_comfort') or {}
                        regions = sorted(list(set(p1_regional.keys()) | set(p2_regional.keys())))
                        
                        regional_table = []
                        for region in regions:
                            p1_r = p1_regional.get(region, {}) or {}
                            p2_r = p2_regional.get(region, {}) or {}
                            regional_table.append({
                                'Región/Continente': region,
                                f'{p1} V-D': f"{p1_r.get('wins', 0)}-{p1_r.get('losses', 0)}",
                                f'{p1} %': f"{p1_r.get('win_rate', 0):.1f}%",
                                f'{p2} V-D': f"{p2_r.get('wins', 0)}-{p2_r.get('losses', 0)}",
                                f'{p2} %': f"{p2_r.get('win_rate', 0):.1f}%"
                            })
                        df_regional = pd.DataFrame(regional_table)
                        f.write("\n")
                        f.write(df_regional.to_markdown(index=False))
                    f.write("\n\n")
                    if score_breakdown:
                        format_score_breakdown(f, p1, p2, score_breakdown, 'home_advantage', 'Desglose Ventaja de Localía', weights)


                # Find history keys dynamically and associate them with players
                hist_keys_in_match = [k for k in match.keys() if k.startswith('historial_')]
                
                p1_lastname = p1.split(' ')[0]
                hist_key1 = next((k for k in hist_keys_in_match if p1_lastname in k), None)

                p2_lastname = p2.split(' ')[0]
                hist_key2 = next((k for k in hist_keys_in_match if p2_lastname in k and k != hist_key1), None)

                if hist_key1:
                    df_hist1 = pd.DataFrame(match[hist_key1])
                    f.write(f"\n--- Historial Detallado de {p1} ---\n")

                    cols_to_show = ['fecha', 'torneo', 'superficie', 'oponente', 'resultado', 'outcome', 'opponent_ranking']
                    rename_cols = {
                        'opponent_ranking': 'Rank Rival',
                        'outcome': 'Resultado',
                        'resultado': 'Score',
                        'oponente': 'Oponente',
                        'superficie': 'Superficie',
                        'torneo': 'Torneo',
                        'fecha': 'Fecha'
                    }
                    existing_cols = [col for col in cols_to_show if col in df_hist1.columns]
                    df_hist1_display = df_hist1[existing_cols].rename(columns=rename_cols)
                    f.write("\n")
                    f.write(df_hist1_display.to_markdown(index=False))
                    f.write("\n")
                    
                    p1_ranking = ranking_analysis.get(p1_ranking_key, None)
                    patrones_p1 = analizar_patrones_historial(df_hist1.copy(), p1_ranking)
                    f.write(f"\n--- Patrones Clave en Historial ---\n")

                    for patron in patrones_p1:
                        f.write(f"{patron}\n")
                    f.write("\n")

                if hist_key2:
                    df_hist2 = pd.DataFrame(match[hist_key2])
                    f.write(f"\n--- Historial Detallado de {p2} ---\n")

                    cols_to_show = ['fecha', 'torneo', 'superficie', 'oponente', 'resultado', 'outcome', 'opponent_ranking']
                    rename_cols = {
                        'opponent_ranking': 'Rank Rival',
                        'outcome': 'Resultado',
                        'resultado': 'Score',
                        'oponente': 'Oponente',
                        'superficie': 'Superficie',
                        'torneo': 'Torneo',
                        'fecha': 'Fecha'
                    }
                    existing_cols = [col for col in cols_to_show if col in df_hist2.columns]
                    df_hist2_display = df_hist2[existing_cols].rename(columns=rename_cols)
                    f.write("\n")
                    f.write(df_hist2_display.to_markdown(index=False))
                    f.write("\n")

                    p2_ranking = ranking_analysis.get(p2_ranking_key, None)
                    patrones_p2 = analizar_patrones_historial(df_hist2.copy(), p2_ranking)
                    f.write("\n--- Patrones Clave en Historial ---\n")

                    for patron in patrones_p2:
                        f.write(f"{patron}\n")
                    f.write("\n")
                
                # Análisis de probabilidad de Overs
                if hist_key1 and hist_key2:
                    prob_overs = analizar_probabilidad_overs(df_hist1.copy(), df_hist2.copy())
                    f.write("\n--- ANÁLISIS DE PROBABILIDAD (OVERS) ---\n")

                    f.write(f"Prob. de > 2.5 Sets: {prob_overs['prob_over_2_5_sets']}\n")
                    f.write(f"Prob. de > 18.5 Juegos: {prob_overs['prob_over_18_5_games_proxy']}\n\n")


                # Resumen consolidado de puntuación al final de cada partido
                if score_breakdown and scores:
                    generar_resumen_consolidado(f, p1, p2, score_breakdown, scores)


        print(f"Análisis completo guardado en: {output_filename}")

    except FileNotFoundError:
        print(f"Error: El archivo '{file_path}' no fue encontrado.")
    except json.JSONDecodeError:
        print(f"Error: El archivo '{file_path}' no es un JSON válido.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")


def find_latest_h2h_file():
    """Finds the most recent h2h_results_enhanced_...json file in the reports directory."""
    try:
        list_of_files = glob.glob('reports/h2h_results_enhanced_*.json')
        if not list_of_files:
            return None
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
    except Exception as e:
        print(f"Error al buscar el último archivo H2H: {e}")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tabla de favoritos — análisis descriptivo de partidos')
    parser.add_argument('file', nargs='?', default=None,
                        help='Archivo H2H JSON a analizar (default: más reciente en reports/)')
    parser.add_argument('--torneo-tipo', type=str, default=None,
                        choices=['grand_slam', 'atp1000', 'atp500', 'challenger', 'itf'],
                        help='Filtrar por tier de torneo (default: todos)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Nombre del archivo de salida (default: reports/analisis_partidos_YYYYMMDD_HHMMSS.txt)')
    args = parser.parse_args()

    if args.file:
        file_to_analyze = args.file
        print(f"Analizando el archivo especificado: {file_to_analyze}\n")
    else:
        file_to_analyze = find_latest_h2h_file()
        if file_to_analyze:
            print(f"Analizando el archivo más reciente encontrado: {file_to_analyze}\n")
        else:
            print("No se encontró ningún archivo de resultados H2H para analizar.")
            file_to_analyze = None

    if file_to_analyze:
        # Build timestamped filename if not specified
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.output:
            timestamped_output = args.output
        else:
            os.makedirs('reports', exist_ok=True)
            timestamped_output = f'reports/analisis_partidos_{ts}.txt'

        analyze_matches_with_pandas(file_to_analyze, output_filename=timestamped_output, tier_filter=args.torneo_tipo)

        # Keep legacy copy for backward compatibility (scripts that read analisis_partidos_pandas.txt)
        legacy_path = 'analisis_partidos_pandas.txt'
        shutil.copy2(timestamped_output, legacy_path)
        print(f"Copia legacy guardada en: {legacy_path}")