import json
import pandas as pd
import sys
import os
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
    
    f.write(f"--- {category_name} (Peso: {weight_str}) ---\n")
    
    df_data = {
        'Jugador': [p1_name, p2_name],
        'Puntaje Bruto': [p1_data.get('raw_score', 'N/A'), p2_data.get('raw_score', 'N/A')],
        'Puntaje Normalizado': [p1_data.get('normalized_score', 'N/A'), p2_data.get('normalized_score', 'N/A')],
        'Puntaje Ponderado': [p1_data.get('weighted_score', 'N/A'), p2_data.get('weighted_score', 'N/A')],
        'ContribuciÃ³n': [p1_data.get('contribution', 'N/A'), p2_data.get('contribution', 'N/A')]
    }
    df = pd.DataFrame(df_data)
    f.write(df.to_string(index=False))
    f.write("\n\n")

def format_ranking_momentum_details(f, p1_name, p2_name, reasoning):
    """Parses the detailed ranking momentum logs and formats them into a table."""
    p1_log = next((r for r in reasoning if f"LOG_RANKING_{p1_name}" in r or "LOG_RANKING_P1" in r), None)
    p2_log = next((r for r in reasoning if f"LOG_RANKING_{p2_name}" in r or "LOG_RANKING_P2" in r), None)

    if not p1_log and not p2_log:
        return

    def parse_log_scores(log_string):
        """Extrae los valores numéricos de la cadena de log de ranking."""
        if not log_string:
            return {}
        
        # Expresión regular mejorada y más flexible
        # Busca patrones como: palabra = número o palabra(detalles) = número
        details = re.findall(r'(\w+)(?:\([^)]*\))?\s*=\s*([\d.\-]+)', log_string)
        parsed_data = {}
        for match in details:
            key = match[0]
            value = match[1]
            try:
                # Capitalizar la primera letra para unificar el formato
                parsed_data[key.title()] = float(value)
            except (ValueError, TypeError):
                parsed_data[key.title()] = value
        return parsed_data
    
    def parse_log_points(log_string):
        """Extrae los puntos de ranking de la cadena de log."""
        if not log_string:
            return {}
        
        points_data = {}
        
        # Buscar patrones más flexibles para los puntos
        base_patterns = [
            r'Base\s*\(?\s*([\d.\-]+)\s*\)?',
            r'base\s*[=:]\s*([\d.\-]+)',
            r'puntos_actuales\s*[=:]\s*([\d.\-]+)'
        ]
        
        for pattern in base_patterns:
            base_match = re.search(pattern, log_string, re.IGNORECASE)
            if base_match:
                points_data['Pts Totales'] = int(float(base_match.group(1)))
                break

        momentum_patterns = [
            r'Momentum\s*\(?\s*([\d.\-]+)\s*\)?',
            r'momentum\s*[=:]\s*([\d.\-]+)',
            r'proximos_puntos\s*[=:]\s*([\d.\-]+)'
        ]
        
        for pattern in momentum_patterns:
            momentum_match = re.search(pattern, log_string, re.IGNORECASE)
            if momentum_match:
                points_data['Próx. Pts'] = int(float(momentum_match.group(1)))
                break

        potential_patterns = [
            r'Potencial\s*\(?\s*([\d.\-]+)\s*\)?',
            r'potencial\s*[=:]\s*([\d.\-]+)',
            r'puntos_maximos\s*[=:]\s*([\d.\-]+)'
        ]
        
        for pattern in potential_patterns:
            potential_match = re.search(pattern, log_string, re.IGNORECASE)
            if potential_match:
                points_data['Pts Máx.'] = int(float(potential_match.group(1)))
                break
            
        return points_data

    def parse_log_advanced_metrics(log_string):
        """Extrae las métricas avanzadas de la cadena de log de ranking."""
        if not log_string:
            return {}
        
        metrics = {}
        
        # Buscar momentum/puntos asegurados
        momentum_patterns = [
            r'Momentum\s*\(?\s*([\d.\-]+)\s*\)?',
            r'momentum\s*[=:]\s*([\d.\-]+)',
            r'puntos_asegurados\s*[=:]\s*([\d.\-]+)'
        ]
        
        for pattern in momentum_patterns:
            momentum_match = re.search(pattern, log_string, re.IGNORECASE)
            if momentum_match:
                metrics['Puntos Asegurados'] = float(momentum_match.group(1))
                break

        # Buscar potencial de mejora
        potential_patterns = [
            r'Potencial\s*\(?\s*([\d.\-]+)\s*\)?',
            r'potencial\s*[=:]\s*([\d.\-]+)',
            r'potencial_mejora\s*[=:]\s*([\d.\-]+)'
        ]
        
        for pattern in potential_patterns:
            potential_match = re.search(pattern, log_string, re.IGNORECASE)
            if potential_match:
                metrics['Potencial de Mejora'] = float(potential_match.group(1))
                break

        # Buscar índice de presión
        pressure_patterns = [
            r'Presión\s*\(?\s*([\d.\-]+)\s*\)?',
            r'presion\s*[=:]\s*([\d.\-]+)',
            r'indice_presion\s*[=:]\s*([\d.\-]+)'
        ]
        
        for pattern in pressure_patterns:
            pressure_match = re.search(pattern, log_string, re.IGNORECASE)
            if pressure_match:
                metrics['Índice de Presión'] = float(pressure_match.group(1))
                break
            
        return metrics

    p1_scores = parse_log_scores(p1_log)
    p2_scores = parse_log_scores(p2_log)
    
    p1_points = parse_log_points(p1_log)
    p2_points = parse_log_points(p2_log)
    
    p1_advanced = parse_log_advanced_metrics(p1_log)
    p2_advanced = parse_log_advanced_metrics(p2_log)

    # Tabla de Puntos de Ranking
    if p1_points or p2_points:
        f.write("--- Puntos de Ranking (FlashScore) ---\n")
        points_keys = ['Pts Totales', 'Próx. Pts', 'Pts Máx.']
        df_points_data = {
            'Componente': points_keys,
            p1_name: [p1_points.get(key, 'N/A') for key in points_keys],
            p2_name: [p2_points.get(key, 'N/A') for key in points_keys]
        }
        df_points = pd.DataFrame(df_points_data)
        f.write(df_points.to_string(index=False))
        f.write("\n\n")

    # Tabla de Métricas Avanzadas
    if p1_advanced or p2_advanced:
        f.write("--- Métricas Avanzadas de Ranking ---\n")
        advanced_keys = ['Potencial de Mejora', 'Puntos Asegurados', 'Índice de Presión']
        df_advanced_data = {
            'Métrica': advanced_keys,
            p1_name: [p1_advanced.get(key, 'N/A') for key in advanced_keys],
            p2_name: [p2_advanced.get(key, 'N/A') for key in advanced_keys]
        }
        df_advanced = pd.DataFrame(df_advanced_data)
        f.write(df_advanced.to_string(index=False))
        f.write("\n\n")

    # Si tenemos datos de scores, mostrar el desglose detallado
    if p1_scores or p2_scores:
        # Ordenar las claves para una presentación consistente
        all_keys = sorted(list(set(p1_scores.keys()) | set(p2_scores.keys())))

        df_data = {
            'Componente Momentum': all_keys,
            p1_name: [f"{p1_scores.get(key, 0):.4f}" for key in all_keys],
            p2_name: [f"{p2_scores.get(key, 0):.4f}" for key in all_keys]
        }

        df = pd.DataFrame(df_data)
        f.write("--- Desglose Detallado de Ranking & Momentum ---\n")
        f.write(df.to_string(index=False))
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
    
    f.write("--- DISTRIBUCIÓN DE PESOS DEL ANÁLISIS ---\n")
    
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
    f.write(df.to_string(index=False))
    f.write(f"\nSuma Total de Pesos: {total_weight * 100:.1f}%\n\n")

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
    f.write("--- CONSOLIDADO DE PUNTUACIÓN ---\n")
    
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
    f.write(final_df.to_string(index=False))
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
            f.write("--- ANÁLISIS DE PARTIDOS CON PANDAS ---\n\n")
            
            if metadata:
                f.write("--- METADATOS DEL ANÁLISIS ---\n")
                meta_df = pd.DataFrame([metadata])
                f.write(meta_df.to_string(index=False))
                f.write("\n\n")

            for match in matches:
                if not match:
                    continue
                match_number = match.get('match_number', 'N/A')
                p1 = match.get('jugador1', 'N/A')
                p2 = match.get('jugador2', 'N/A')
                
                f.write(f"=== PARTIDO {match_number}: {p1} vs {p2} ===\n\n")
                
                # Información básica del partido
                f.write("--- INFORMACIÓN DEL PARTIDO ---\n")
                basic_info = {
                    'Campo': [match.get('superficie', 'N/A')],
                    'Torneo': [match.get('torneo', 'N/A')],
                    'Categoría': [match.get('categoria', 'N/A')],
                    'Fecha': [match.get('fecha', 'N/A')]
                }
                basic_df = pd.DataFrame(basic_info)
                f.write(basic_df.to_string(index=False))
                f.write("\n\n")

                # Resultado del análisis y probabilidades
                resultado_ai = match.get('resultado_ai', 'N/A')
                probabilidades = match.get('probabilidades', {})
                scores = match.get('scores', {})
                
                f.write("--- RESULTADO DEL ANÁLISIS ---\n")
                result_data = {
                    'Favorito AI': [resultado_ai],
                    'Prob. P1 (%)': [f"{probabilidades.get('p1_probability', 0)*100:.2f}%" if isinstance(probabilidades.get('p1_probability'), (int, float)) else "N/A"],
                    'Prob. P2 (%)': [f"{probabilidades.get('p2_probability', 0)*100:.2f}%" if isinstance(probabilidades.get('p2_probability'), (int, float)) else "N/A"],
                    'Diferencia': [f"{abs(probabilidades.get('p1_probability', 0) - probabilidades.get('p2_probability', 0))*100:.2f}%" if all(isinstance(probabilidades.get(k), (int, float)) for k in ['p1_probability', 'p2_probability']) else "N/A"]
                }
                result_df = pd.DataFrame(result_data)
                f.write(result_df.to_string(index=False))
                f.write("\n\n")

                # Información de los jugadores
                f.write("--- INFORMACIÓN DE LOS JUGADORES ---\n")
                player_data = {
                    'Jugador': [p1, p2],
                    'Ranking': [match.get('ranking_p1', 'N/A'), match.get('ranking_p2', 'N/A')],
                    'Puntos ATP': [match.get('puntos_p1', 'N/A'), match.get('puntos_p2', 'N/A')],
                    'Edad': [match.get('edad_p1', 'N/A'), match.get('edad_p2', 'N/A')],
                    'País': [match.get('pais_p1', 'N/A'), match.get('pais_p2', 'N/A')]
                }
                players_df = pd.DataFrame(player_data)
                f.write(players_df.to_string(index=False))
                f.write("\n\n")

                # Obtener pesos del reasoning
                reasoning = match.get('reasoning', [])
                weights = get_weights_from_reasoning(reasoning)
                
                # Mostrar distribución de pesos
                format_weights_distribution(f, weights)

                # Desglose detallado por categorías
                score_breakdown = match.get('score_breakdown')
                
                # Solo mostrar categorías que tienen datos
                categories_to_show = [
                    ('surface_specialization', 'Especialización en Superficie'),
                    ('form_recent', 'Forma Reciente'),
                    ('common_opponents', 'Rivales Comunes'),
                    ('h2h_direct', 'Historial Directo (H2H)'),
                    ('ranking_momentum', 'Ranking y Momentum'),
                    ('elo_rating', 'Rating ELO'),
                    ('home_advantage', 'Ventaja de Localía'),
                    ('strength_of_schedule', 'Fuerza del Calendario')
                ]

                if score_breakdown:
                    for category_key, category_name in categories_to_show:
                        format_score_breakdown(f, p1, p2, score_breakdown, category_key, category_name, weights)

                # Resumen consolidado de puntuación
                generar_resumen_consolidado(f, p1, p2, score_breakdown, scores)

                # Análisis detallado de ranking momentum si está disponible
                format_ranking_momentum_details(f, p1, p2, reasoning)

                # Análisis de historiales si están disponibles
                hist_p1_data = match.get('historial_p1', [])
                hist_p2_data = match.get('historial_p2', [])
                
                if hist_p1_data or hist_p2_data:
                    f.write("--- ANÁLISIS DE HISTORIALES ---\n")
                    
                    # Crear DataFrames de los historiales
                    hist_p1_df = pd.DataFrame(hist_p1_data) if hist_p1_data else pd.DataFrame()
                    hist_p2_df = pd.DataFrame(hist_p2_data) if hist_p2_data else pd.DataFrame()
                    
                    # Análisis de patrones para cada jugador
                    if not hist_p1_df.empty:
                        f.write(f"\nPatrones de {p1}:\n")
                        patrones_p1 = analizar_patrones_historial(hist_p1_df, match.get('ranking_p1'))
                        for patron in patrones_p1:
                            f.write(f"{patron}\n")
                    
                    if not hist_p2_df.empty:
                        f.write(f"\nPatrones de {p2}:\n")
                        patrones_p2 = analizar_patrones_historial(hist_p2_df, match.get('ranking_p2'))
                        for patron in patrones_p2:
                            f.write(f"{patron}\n")
                    
                    # Análisis de probabilidad de overs
                    overs_analysis = analizar_probabilidad_overs(hist_p1_df, hist_p2_df)
                    f.write("\n--- ANÁLISIS DE PROBABILIDADES OVER ---\n")
                    f.write(f"Probabilidad Over 2.5 Sets: {overs_analysis['prob_over_2_5_sets']}\n")
                    f.write(f"Probabilidad Over 18.5 Juegos: {overs_analysis['prob_over_18_5_games_proxy']}\n")
                    f.write("\n")

                # Análisis de rivales comunes
                common_opps = match.get('common_opponents_analysis', [])
                if common_opps:
                    f.write("--- ANÁLISIS DE RIVALES COMUNES ---\n")
                    for opponent_data in common_opps:
                        if not isinstance(opponent_data, dict):
                            continue
                        opponent_name = opponent_data.get('opponent_name', 'N/A')
                        advantage_for = opponent_data.get('advantage_for', 'Ninguno')
                        explicacion = explicar_ventaja_rival_comun(opponent_data, p1, p2)
                        
                        f.write(f"Rival: {opponent_name}\n")
                        f.write(f"Ventaja para: {advantage_for}\n")
                        f.write(f"Explicación: {explicacion}\n\n")

                # Predicción de sets y juegos
                if scores and isinstance(scores, dict):
                    p1_final = scores.get('p1_final_weight', 0)
                    p2_final = scores.get('p2_final_weight', 0)
                    if isinstance(p1_final, (int, float)) and isinstance(p2_final, (int, float)):
                        score_diff = p1_final - p2_final
                        prediccion = predecir_sets_y_games(score_diff, p1_final, p2_final)
                        
                        f.write("--- PREDICCIÓN DE SETS Y JUEGOS ---\n")
                        f.write(f"Sets Estimados: {prediccion['predicted_sets']}\n")
                        f.write(f"Juegos Estimados: {prediccion['predicted_games']}\n")
                        f.write(f"Razón: {prediccion['reason']}\n\n")

                # Reasoning detallado
                if reasoning:
                    f.write("--- REASONING DETALLADO ---\n")
                    for i, reason in enumerate(reasoning, 1):
                        if reason and not reason.startswith('LOG_'):
                            f.write(f"{i}. {reason}\n")
                    f.write("\n")

                f.write("=" * 100 + "\n\n")

        print(f"Análisis completado y guardado en: {output_filename}")
        return output_filename

    except Exception as e:
        print(f"Error al procesar el archivo: {e}")
        return None

def main():
    """Función principal para ejecutar el análisis."""
    if len(sys.argv) < 2:
        print("Uso: python script.py <archivo_json>")
        return

    file_path = sys.argv[1]
    
    # Verificar que el archivo existe
    if not os.path.exists(file_path):
        print(f"Error: El archivo {file_path} no existe.")
        return

    # Generar nombre de archivo de salida basado en el archivo de entrada
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_filename = f"analisis_{base_name}_pandas.txt"
    
    # Ejecutar el análisis
    result = analyze_matches_with_pandas(file_path, output_filename)
    
    if result:
        print("Análisis completado exitosamente.")
        print(f"Archivo de salida: {result}")
    else:
        print("Error durante el análisis.")

if __name__ == "__main__":
    main()




                

                        