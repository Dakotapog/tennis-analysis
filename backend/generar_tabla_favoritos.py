import json
import pandas as pd
import sys
import os
import glob
import re

def get_weights_from_reasoning(reasoning):
    """Extracts analysis weights from the reasoning string."""
    weights = {}
    for reason in reasoning:
        if 'LOG_WEIGHTS' in reason:
            match = re.search(r"\{.*\}", reason)
            if match:
                try:
                    weights_str = match.group(0).replace("'", '"')
                    weights_data = json.loads(weights_str)
                    for key, value in weights_data.items():
                        weights[key] = value
                    return weights
                except (json.JSONDecodeError, ValueError):
                    continue
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
    f.write(f"--- {category_name} (Peso: {weight_str}) ---")
    
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
        f.write(f"--- Puntos de Ranking (FlashScore) ---")

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
        f.write(f"--- Métricas Avanzadas de Ranking ---")

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
    f.write(f"--- Desglose Detallado de Ranking & Momentum ---")
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

    # Ambos ganaron o ambos perdieron
    if p1_outcome == p2_outcome:
        p1_sets = parse_score(p1_score_str)
        p2_sets = parse_score(p2_score_str)
        
        if p1_outcome == 'Ganó':
            # Gana el más contundente (menos sets cedidos)
            if p1_sets[1] < p2_sets[1]:
                return f"Ambos ganaron, pero {p1_name} fue más contundente ({p1_score_str} vs {p2_score_str} de {p2_name})."
            elif p2_sets[1] < p1_sets[1]:
                return f"Ambos ganaron, pero {p2_name} fue más contundente ({p2_score_str} vs {p1_score_str} de {p1_name})."
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
    historial_df['sets_perdidos'] = pd.to_numeric(sets_split[1], errors='coerce').fillna(0)

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
        df['sets_perdidos'] = pd.to_numeric(sets_split[1], errors='coerce').fillna(0)

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
    
    f.write(f"--- DISTRIBUCIÓN DE PESOS DEL ANÁLISIS ---")

    
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
    f.write(f"\n\nSuma Total de Pesos: {total_weight * 100:.1f}%\n\n")

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
    f.write(f"--- CONSOLIDADO DE PUNTUACIÓN ---")

    
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

def analyze_matches_with_pandas(file_path, output_filename="analisis_partidos_pandas.txt"):
    """
    Reads match data from a JSON file, analyzes it using pandas,
    and generates a text file with structured tables.
    """
    try:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1200)
        pd.set_option('display.colheader_justify', 'center')

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        matches = data.get('partidos', [])
        metadata = data.get('metadata', {})

        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("--- ANÁLISIS DE PARTIDOS CON PANDAS ---")

            
            if metadata:
                f.write("--- METADATOS DEL ANÁLISIS ---")

                meta_df = pd.DataFrame([metadata])
                f.write("\n")
                f.write(meta_df.to_markdown(index=False))
                f.write("\n\n")

            for match in matches:
                if not match:
                    continue
                match_number = match.get('match_number', 'N/A')
                p1 = match.get('jugador1', 'N/A')
                p2 = match.get('jugador2', 'N/A')
                
                f.write("="*150 + "\n")
                f.write(f"Partido #{match_number}: {p1} vs {p2}\n")
                torneo = match.get('torneo_nombre', 'N/A')
                cancha = match.get('tipo_cancha', 'N/A').capitalize()
                f.write(f"Torneo: {torneo} | Superficie: {cancha}\n")
                match_url = match.get('match_url')
                if match_url:
                    f.write(f"URL del Partido: {match_url}\n")
                f.write("="*150 + "\n\n")

                ranking_analysis = match.get('ranking_analysis') or {}
                prediction = ranking_analysis.get('prediction') or {}
                scores = prediction.get('scores') or {}
                
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
                f.write("--- RESUMEN DE PREDICCIÓN ---")
                f.write("\n")
                f.write(summary_df.to_markdown(index=False))
                f.write(f"\n\nJugador Favorito: {prediction.get('favored_player', 'N/A')}\n")
                f.write(f"Confianza: {prediction.get('confidence', 'N/A')}%\n")
                f.write(f"Diferencia de Puntaje: {scores.get('score_difference', 'N/A')}\n\n")

                # Predicción de Sets y Games
                score_diff_val = scores.get('score_difference', 0)
                p1_final_score = scores.get('p1_final_weight', 0)
                p2_final_score = scores.get('p2_final_weight', 0)
                
                set_game_prediction = predecir_sets_y_games(score_diff_val, p1_final_score, p2_final_score)
                
                f.write("--- PREDICCIÓN DE SETS Y GAMES ---")

                f.write(f"Sets Pronosticados: {set_game_prediction['predicted_sets']}\n")
                f.write(f"Games Pronosticados: {set_game_prediction['predicted_games']}\n")
                f.write(f"Justificación: {set_game_prediction['reason']}\n\n")

                # Reasoning
                reasoning = prediction.get('reasoning', [])
                weights = get_weights_from_reasoning(reasoning)
                format_weights_distribution(f, weights)

                if reasoning:
                    f.write("--- RAZONAMIENTO CLAVE Y LOGS DE PREDICCIÓN ---")

                    # Extraer y mostrar la contundencia del H2H si existe
                    h2h_reason = next((r for r in reasoning if 'H2H_Directo' in r and 'contundencia' in r), None)
                    if h2h_reason:
                        f.write(f"Análisis H2H: {h2h_reason}\n\n")
                    
                    f.write("--- Lógica de Ponderación y Scores Detallados ---")

                    for reason in reasoning:
                        # Mostrar todos los logs para máxima transparencia
                        if 'H2H_Directo' not in reason: # Evita duplicar la info de H2H
                            f.write(f"- {reason}\n")
                    f.write("\n")

                score_breakdown = prediction.get('score_breakdown', {}) or {}

                if 'enfrentamientos_directos' in match and match['enfrentamientos_directos']:
                    df_h2h = pd.DataFrame(match['enfrentamientos_directos'])
                    f.write("--- Enfrentamientos Directos ---")
                    f.write("\n")
                    f.write(df_h2h[['fecha', 'ganador', 'resultado', 'superficie']].to_markdown(index=False))
                    f.write("\n\n")
                    if score_breakdown:
                        format_score_breakdown(f, p1, p2, score_breakdown, 'h2h_direct', 'Desglose H2H', weights)

                if 'common_opponents_detailed' in match and match['common_opponents_detailed']:
                    common_opponents_count = ranking_analysis.get('common_opponents_count', len(match['common_opponents_detailed']))
                    f.write(f"--- Rivales Comunes (Total: {common_opponents_count}) ---")

                    
                    df_common = pd.json_normalize(match['common_opponents_detailed'])
                    cols_to_show = {
                        'opponent_name': 'Oponente Común', 'opponent_ranking': 'Rank',
                        'player1_result.date': f'Fecha {p1}', 'player1_result.outcome': f'Res. {p1}', 'player1_result.score': 'Score',
                        'player2_result.date': f'Fecha {p2}', 'player2_result.outcome': f'Res. {p2}', 'player2_result.score': 'Score',
                        'advantage_for': 'Ventaja'
                    }
                    for col in cols_to_show.keys():
                        if col not in df_common.columns:
                            df_common[col] = 'N/A'
                    df_common_display = df_common[list(cols_to_show.keys())].rename(columns=cols_to_show)
                    f.write("\n")
                    f.write(df_common_display.to_markdown(index=False))
                    f.write("\n\n")

                    f.write("--- Justificación Detallada de Ventaja (Rivales Comunes) ---")

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
                    f.write("--- Análisis de Forma Reciente ---")
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
                    f.write("--- ANÁLISIS POR SUPERFICIE ---")

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
                        format_score_breakdown(f, p1, p2, score_breakdown, 'surface_advantage', 'Desglose Ventaja por Superficie', weights)

                # Análisis de Ubicación (Ventaja de Local y Confort Regional)
                if 'location_analysis' in match and match.get('location_analysis'):
                    f.write("--- ANÁLISIS POR UBICACIÓN ---")

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
                        f.write("--- Confort Regional (Rendimiento por Continente) ---")

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
                    f.write(f"--- Historial Detallado de {p1} ---")

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
                    f.write(f"--- Patrones Clave en Historial ---")

                    for patron in patrones_p1:
                        f.write(f"{patron}\n")
                    f.write("\n")

                if hist_key2:
                    df_hist2 = pd.DataFrame(match[hist_key2])
                    f.write(f"--- Historial Detallado de {p2} ---")

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
                    f.write("--- Patrones Clave en Historial ---")

                    for patron in patrones_p2:
                        f.write(f"{patron}\n")
                    f.write("\n")
                
                # Análisis de probabilidad de Overs
                if hist_key1 and hist_key2:
                    prob_overs = analizar_probabilidad_overs(df_hist1.copy(), df_hist2.copy())
                    f.write("--- ANÁLISIS DE PROBABILIDAD (OVERS) ---")

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
    if len(sys.argv) > 1:
        file_to_analyze = sys.argv[1]
        print(f"Analizando el archivo especificado por argumento: {file_to_analyze}\n")
    else:
        file_to_analyze = find_latest_h2h_file()
        if file_to_analyze:
            print(f"Analizando el archivo más reciente encontrado: {file_to_analyze}\n")
        else:
            print("No se encontró ningún archivo de resultados H2H para analizar.")
            file_to_analyze = None

    if file_to_analyze:
        analyze_matches_with_pandas(file_to_analyze)
