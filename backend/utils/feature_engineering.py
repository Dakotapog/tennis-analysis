import pandas as pd
import logging

logger = logging.getLogger(__name__)

def normalize_player_name(name):
    """Normaliza el nombre de un jugador para que coincida con las claves del JSON."""
    if not isinstance(name, str):
        return ""
    return name.replace(' ', '_').replace('.', '')

def engineer_features(df):
    """
    Extrae características numéricas de columnas complejas de forma dinámica.
    Esta función es la FUENTE ÚNICA DE VERDAD para la ingeniería de características.
    """
    logger.info("🛠️ Ejecutando ingeniería de características centralizada...")
    new_features_list = []

    for index, row in df.iterrows():
        features = {}
        
        p1_key = normalize_player_name(row.get('jugador1'))
        p2_key = normalize_player_name(row.get('jugador2'))

        # 1. Ranking Analysis
        ra = row.get('ranking_analysis', {})
        if isinstance(ra, dict):
            features['p1_ranking'] = ra.get(f'{p1_key}_ranking')
            features['p2_ranking'] = ra.get(f'{p2_key}_ranking')
            features['common_opponents_count'] = ra.get('common_opponents_count')
            features['p1_rivalry_score'] = ra.get('p1_rivalry_score')
            features['p2_rivalry_score'] = ra.get('p2_rivalry_score')

        # 2. Form Analysis
        fa = row.get('form_analysis', {})
        if isinstance(fa, dict):
            p1_form = fa.get(f'{p1_key}_form', {})
            p2_form = fa.get(f'{p2_key}_form', {})
            features['p1_form_wins'] = p1_form.get('wins') if isinstance(p1_form, dict) else None
            features['p1_form_losses'] = p1_form.get('losses') if isinstance(p1_form, dict) else None
            features['p1_form_win_rate'] = p1_form.get('win_rate') if isinstance(p1_form, dict) else None
            
            # Calculate win rate if missing
            if features['p1_form_win_rate'] is None and features['p1_form_wins'] is not None and features['p1_form_losses'] is not None:
                total = features['p1_form_wins'] + features['p1_form_losses']
                features['p1_form_win_rate'] = features['p1_form_wins'] / total if total > 0 else 0

            features['p2_form_wins'] = p2_form.get('wins') if isinstance(p2_form, dict) else None
            features['p2_form_losses'] = p2_form.get('losses') if isinstance(p2_form, dict) else None
            features['p2_form_win_rate'] = p2_form.get('win_rate') if isinstance(p2_form, dict) else None

            # Calculate win rate if missing
            if features['p2_form_win_rate'] is None and features['p2_form_wins'] is not None and features['p2_form_losses'] is not None:
                total = features['p2_form_wins'] + features['p2_form_losses']
                features['p2_form_win_rate'] = features['p2_form_wins'] / total if total > 0 else 0

        # 3. Surface Analysis
        sa = row.get('surface_analysis', {})
        if isinstance(sa, dict):
            p1_surf = sa.get(f'{p1_key}_surface_stats', {})
            p2_surf = sa.get(f'{p2_key}_surface_stats', {})
            current_surface = row.get('tipo_cancha', 'Dura').capitalize()
            
            p1_surf_stats = p1_surf.get(current_surface, {}) if isinstance(p1_surf, dict) else {}
            p2_surf_stats = p2_surf.get(current_surface, {}) if isinstance(p2_surf, dict) else {}
            features['p1_surface_wins'] = p1_surf_stats.get('wins')
            features['p1_surface_losses'] = p1_surf_stats.get('losses')
            features['p1_surface_win_rate'] = p1_surf_stats.get('win_rate')

            # Calculate surface win rate if missing
            if features['p1_surface_win_rate'] is None and features['p1_surface_wins'] is not None and features['p1_surface_losses'] is not None:
                total = features['p1_surface_wins'] + features['p1_surface_losses']
                features['p1_surface_win_rate'] = features['p1_surface_wins'] / total if total > 0 else 0

            features['p2_surface_wins'] = p2_surf_stats.get('wins')
            features['p2_surface_losses'] = p2_surf_stats.get('losses')
            features['p2_surface_win_rate'] = p2_surf_stats.get('win_rate')

            # Calculate surface win rate if missing
            if features['p2_surface_win_rate'] is None and features['p2_surface_wins'] is not None and features['p2_surface_losses'] is not None:
                total = features['p2_surface_wins'] + features['p2_surface_losses']
                features['p2_surface_win_rate'] = features['p2_surface_wins'] / total if total > 0 else 0

        # 4. H2H Analysis (Enfrentamientos Directos)
        h2h = row.get('enfrentamientos_directos', {})
        if isinstance(h2h, dict):
            p1_h2h_wins = h2h.get(f'victorias_{p1_key}', 0)
            p2_h2h_wins = h2h.get(f'victorias_{p2_key}', 0)
            total_h2h = p1_h2h_wins + p2_h2h_wins
            
            features['h2h_total'] = total_h2h
            features['p1_h2h_win_rate'] = p1_h2h_wins / total_h2h if total_h2h > 0 else 0
            features['p2_h2h_win_rate'] = p2_h2h_wins / total_h2h if total_h2h > 0 else 0

        # 5. Common Opponents Analysis
        cod = row.get('common_opponents_detailed', {})
        if isinstance(cod, dict):
            p1_wins_vs_common = cod.get('player1_wins', 0)
            p2_wins_vs_common = cod.get('player2_wins', 0)
            
            if p1_wins_vs_common + p2_wins_vs_common > 0:
                features['common_opponents_advantage'] = (p1_wins_vs_common - p2_wins_vs_common) / (p1_wins_vs_common + p2_wins_vs_common)
            else:
                features['common_opponents_advantage'] = 0

        # 6. Location Analysis
        la = row.get('location_analysis', {})
        if isinstance(la, dict):
            p1_loc = la.get(f'{p1_key}_location_stats', {})
            p2_loc = la.get(f'{p2_key}_location_stats', {})
            
            # Home advantage
            features['p1_home_advantage'] = 1 if p1_loc.get('is_home') else 0
            features['p2_home_advantage'] = 1 if p2_loc.get('is_home') else 0
            
            # Continent win rate
            continent = la.get('continent', 'Unknown')
            p1_continent_stats = p1_loc.get('continent_performance', {}).get(continent, {})
            p2_continent_stats = p2_loc.get('continent_performance', {}).get(continent, {})
            features['p1_continent_win_rate'] = p1_continent_stats.get('win_rate', 0)
            features['p2_continent_win_rate'] = p2_continent_stats.get('win_rate', 0)

        # 7. Strength of Schedule (SoS)
        sos = row.get('strength_of_schedule', {})
        if isinstance(sos, dict):
            features['p1_sos'] = sos.get(f'avg_opponent_rank_{p1_key}')
            features['p2_sos'] = sos.get(f'avg_opponent_rank_{p2_key}')

        # 8. Player History Patterns
        hist_col_p1 = f'historial_{p1_key}'
        hist_col_p2 = f'historial_{p2_key}'
        
        if hist_col_p1 in row and isinstance(row[hist_col_p1], list):
            p1_hist = row[hist_col_p1]
            # Example: Performance vs Top 10
            # This requires opponent rank in history, which might not be available.
            # For now, let's add a placeholder for a simpler metric.
            features['p1_recent_tiebreaks'] = sum(1 for match in p1_hist[:10] if '7-6' in match.get('resultado', '') or '6-7' in match.get('resultado', ''))
        
        if hist_col_p2 in row and isinstance(row[hist_col_p2], list):
            p2_hist = row[hist_col_p2]
            features['p2_recent_tiebreaks'] = sum(1 for match in p2_hist[:10] if '7-6' in match.get('resultado', '') or '6-7' in match.get('resultado', ''))

        new_features_list.append(features)

    features_df = pd.DataFrame(new_features_list, index=df.index)
    df = pd.concat([df, features_df], axis=1)

    logger.info("✅ Ingeniería de características centralizada completada.")
    return df
