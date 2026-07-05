import re
from datetime import datetime
import logging
import math

from analysis.elo_system import k_factor_efectivo
from analysis.erdos_graph import (
    historial_a_partidos,
    construir_grafo_victorias,
    distancia_erdos,
)

logger = logging.getLogger(__name__)

# Nodo-32 Fase 3: versión del cálculo de predicción serializada en h2h_results_enhanced.
# Incrementar al cambiar el punto de aplicación de factores Markov/tardío (afecta `confidence`).
# Consumidores validan este campo para rechazar archivos generados con Markov PRE-norm.
RIVALRY_VERSION = "nodo32-fase3-markov-postnorm"

# Nodo-57 D57-01: decaimiento exponencial de forma por inactividad
# Solo form_recent decae; ELO, H2H, rivales comunes no tienen fecha de vencimiento.
# Marco bayesiano: encoger hacia prior en vez de penalizar el score total.
_FORM_DECAY_LAMBDA = 0.025   # half-life efectivo ≈ 28d post-gracia
_FORM_GRACE_DAYS   = 30      # sin decay hasta 30 días (transiciones de superficie normales)
_FORM_DECAY_FLOOR  = 0.35    # nunca perder más del 65% de la señal de forma

# Nodo-57 D57-03: victorias mínimas en cuadro principal para ser campeón completo
_MIN_WINS_CHAMPION = {
    'grand_slam': 7,    # R1-R7 cuadro principal
    'atp1000': 6,       # R1-R6 (con bye = 5, sin bye = 6) → conservador
    'atp500': 5,
    'challenger': 5,
    'itf': 4,
}


# ─────────────────────────────────────────────────────────────────────────────
# T21-06 (Nodo-21 Fase 2) — Densidad local del grafo como modulador continuo
# ─────────────────────────────────────────────────────────────────────────────

def density_confidence(n_common_opponents: int, n_erdos_paths: int) -> float:
    """
    Calcula la confianza en señales transitivas según la densidad local del grafo.

    REGLA-T21-3: densidad local > categoría discreta.
    Los pesos por tier asumen densidad uniforme dentro del tier.
    Parry vs Seyboth Wild (RG R1, ambos jóvenes) tiene densidad baja aunque sea Grand Slam.

    Rango: [0.3, 1.0]
      Grand Slam típico: n_common ~15-30, n_paths ~20+ → factor ~1.0
      Challenger típico: n_common ~2-3,  n_paths ~3-5  → factor ~0.4
      Sin data:          n_common=0,     n_paths=0     → factor=0.3
    """
    raw = min(n_common_opponents, 20) / 20.0
    path_boost = min(n_erdos_paths, 30) / 30.0
    return round(0.3 + 0.7 * ((raw + path_boost) / 2), 4)


# ─────────────────────────────────────────────────────────────────────────────
# T21-07 (Nodo-21 Fase 2) — James-Stein shrinkage para pesos por tier
# ─────────────────────────────────────────────────────────────────────────────

def shrink_weights(tier_weights: dict, default_weights: dict, n_tier: int, n_threshold: int = 20) -> dict:
    """
    Empirical Bayes (James-Stein): shrink tier-specific weights toward default
    when there is little empirical evidence for the tier.

    REGLA-T21-4: pesos nunca ciegos — shrinkage obligatorio.
      n=0:   factor=0.00 → 100% default (pesos seguros cuando no hay data)
      n=31:  factor=0.61 → 61% tier, 39% default
      n=100: factor=0.83 → 83% tier

    Args:
        tier_weights:   pesos específicos del tier (e.g. grand_slam)
        default_weights: pesos de fallback conservador (e.g. atp500)
        n_tier:         número de partidos observados para este tier
        n_threshold:    pseudo-count para regularización (default 20)
    """
    factor = n_tier / (n_tier + n_threshold)
    return {
        k: round(factor * tier_weights[k] + (1 - factor) * default_weights[k], 4)
        for k in tier_weights
    }


# ─────────────────────────────────────────────────────────────────────────────
# Nodo-53 — Funciones extraídas a nivel módulo para testabilidad (D53-06, D53-07)
# PRESERVA el bug hasta que los tests lo confirmen: _LINEAR_COMPONENTS intacto, cap=250 intacto.
# ─────────────────────────────────────────────────────────────────────────────

# D53-06 FIX: surface_specialization usa log1p igual que los demás componentes.
# Anterior: lineal con MAX_RAW=350 → escala 0.56 vs form 4.33 → peso efectivo 1.4% (nominal 15%).
# Fix: log1p → escala 3.54 vs form 4.33 → ratio 0.82, peso efectivo proporcional al nominal.
_LINEAR_COMPONENTS = set()


def normalize_scores(p1_scores, p2_scores):
    """Normaliza raw scores. surface_specialization usa escala lineal (D53-06 BUG hasta fix)."""
    from normalization import MAX_RAW_SCORES
    normalized_p1 = {}
    normalized_p2 = {}
    for key in p1_scores:
        p1_val = p1_scores[key]
        p2_val = p2_scores[key]

        if key in _LINEAR_COMPONENTS:
            max_expected = MAX_RAW_SCORES.get(key, 350)
            norm_p1 = min(p1_val / max_expected, 1.0) * math.log1p(max_expected)
            norm_p2 = min(p2_val / max_expected, 1.0) * math.log1p(max_expected)
        else:
            norm_p1 = math.log1p(p1_val)
            norm_p2 = math.log1p(p2_val)

        normalized_p1[key] = norm_p1
        normalized_p2[key] = norm_p2
    return normalized_p1, normalized_p2


def _compute_raw_elo(elo):
    """Convierte ELO a raw score. D53-07 FIX: sin cap → diferencia real entre top-200.
    Deuda D53-12: jugadores ITF con ELO<1500 → raw=0 (floor aceptable, Nodo-21 maneja tier).
    """
    return max(0, elo - 1500)


def _parse_match_date(date_str):
    """Parsea fecha de partido en formato DD.MM.YYYY. D53-01 FIX: año 4 dígitos.

    Antes del fix: '%d.%m.%y' → ValueError 'unconverted data remains: 24' para '09.10.2024'.
    Extraída a nivel módulo para testabilidad (REGLA-T53).
    """
    return datetime.strptime(date_str, '%d.%m.%Y')


class RivalryAnalyzer:
    """⚔️ Analizador de rivalidades transitivas con análisis de rachas y peso de oponentes."""

    def estimate_elo_from_rank(self, rank):
        """Estima un rating ELO basado en el ranking ATP/WTA."""
        if rank is None:
            return 1500  # ELO por defecto para sin ranking
        if rank <= 10:
            return 2200 - (rank - 1) * 20  # CORREGIDO: 2200 - 2020 para rank 1-10
        elif rank <= 50:
            return 2020 - (rank - 11) * 5  # CORREGIDO: 2020 - 1820
        elif rank <= 100:
            return 1820 - (rank - 51) * 2  # CORREGIDO: 1820 - 1720
        elif rank <= 200:
            return 1720 - (rank - 101) * 1  # CORREGIDO: 1720 - 1620
        else:
            return 1600

    def calculate_elo_from_history(self, player_name, player_history):
        """Calcula el rating ELO de un jugador basado en su historial de partidos extraído.
        T21-09 (Nodo-21 Fase 3): K-factor adaptivo por tier de torneo de cada partido.
        """
        current_elo = self.elo_system.default_rating

        if not player_history:
            return current_elo

        # Iterar desde el partido más antiguo al más reciente
        for match in reversed(player_history):
            opponent_rank = match.get('opponent_ranking')
            opponent_elo = self.estimate_elo_from_rank(opponent_rank)

            expected_score = self.elo_system.expected_score(current_elo, opponent_elo)

            won = self.determine_match_winner(match, player_name)
            actual_score = 1 if won else 0

            # T21-09: K-factor por tier del torneo de cada partido
            from config import detectar_tier as _dt
            _torneo = match.get('torneo') or match.get('tournament') or ''
            _tier = _dt(_torneo)
            k = k_factor_efectivo(_tier)
            new_elo = current_elo + k * (actual_score - expected_score)
            current_elo = new_elo

        return round(current_elo)

    def __init__(self, ranking_manager, elo_system):
        self.ranking_manager = ranking_manager
        self.elo_system = elo_system
        self.SURFACE_NORMALIZATION_MAP = {
            'hard': 'Dura', 'dura': 'Dura',
            'clay': 'Arcilla', 'arcilla': 'Arcilla',
            'grass': 'Hierba', 'hierba': 'Hierba',
            'indoor': 'Indoor', 'indoor hard': 'Indoor'
        }
        self.COUNTRY_TO_CONTINENT_MAP = {
            'USA': 'Norteamérica', 'Canada': 'Norteamérica', 'Mexico': 'Norteamérica',
            'Argentina': 'Sudamérica', 'Brazil': 'Sudamérica', 'Chile': 'Sudamérica', 'Colombia': 'Sudamérica', 'Ecuador': 'Sudamérica', 'Uruguay': 'Sudamérica',
            'Spain': 'Europa', 'France': 'Europa', 'Italy': 'Europa', 'Germany': 'Europa', 'Great Britain': 'Europa', 'Russia': 'Europa', 'Serbia': 'Europa', 'Switzerland': 'Europa', 'Sweden': 'Europa', 'Austria': 'Europa', 'Belgium': 'Europa', 'Netherlands': 'Europa', 'Poland': 'Europa', 'Czech Republic': 'Europa', 'Croatia': 'Europa', 'Greece': 'Europa', 'Norway': 'Europa', 'Denmark': 'Europa', 'Finland': 'Europa', 'Portugal': 'Europa', 'Hungary': 'Europa', 'Romania': 'Europa', 'Slovakia': 'Europa', 'Slovenia': 'Europa', 'Ukraine': 'Europa',
            'Australia': 'Oceanía', 'New Zealand': 'Oceanía',
            'Japan': 'Asia', 'China': 'Asia', 'South Korea': 'Asia', 'India': 'Asia', 'Kazakhstan': 'Asia',
            'South Africa': 'África', 'Tunisia': 'África', 'Egypt': 'África'
        }

    def calculate_base_opponent_weight(self, ranking):
        """⚖️ Calcular peso base de oponente basado en ranking para enriquecimiento."""
        if ranking is None:
            return 1  # Peso base para jugadores sin ranking
        if ranking <= 10: return 10
        elif ranking <= 15: return 8
        elif ranking <= 30: return 6
        elif ranking <= 50: return 4
        elif ranking <= 100: return 2
        else: return 1

    def _partidos_recientes(self, player_history, opponent_name, recent_count=20):
        """Verifica si un partido contra opponent_name está en los últimos `recent_count` juegos."""
        normalized_opponent_name = self.ranking_manager.normalize_name(opponent_name)
        for match in player_history[:recent_count]:
            if self.ranking_manager.normalize_name(match.get('oponente', '')) == normalized_opponent_name:
                return True
        return False

    def _count_matches(self, player_history, opponent_name):
        """Cuenta los partidos contra un oponente específico."""
        normalized_opponent_name = self.ranking_manager.normalize_name(opponent_name)
        return sum(1 for match in player_history if self.ranking_manager.normalize_name(match.get('oponente', '')) == normalized_opponent_name)

    def _win_rate_vs_oponente(self, player_history, opponent_name, player_name):
        """Calcula la tasa de victorias contra un oponente específico."""
        normalized_opponent_name = self.ranking_manager.normalize_name(opponent_name)
        matches = [m for m in player_history if self.ranking_manager.normalize_name(m.get('oponente', '')) == normalized_opponent_name]
        if not matches:
            return 0.0
        wins = sum(1 for m in matches if self.determine_match_winner(m, player_name))
        return wins / len(matches)

    def calcular_peso_oponentes_comunes(self, player1_history, player2_history, common_opponent_name, player1_name, player2_name):
        """Calcula el peso de un oponente común basado en ranking y contexto."""
        ranking_oponente = self.ranking_manager.get_player_ranking(common_opponent_name)
        
        # Peso base según ranking del oponente común
        if ranking_oponente is None:
            peso_base = 5
        elif ranking_oponente <= 5: peso_base = 20
        elif ranking_oponente <= 10: peso_base = 15
        elif ranking_oponente <= 20: peso_base = 12
        elif ranking_oponente <= 30: peso_base = 10
        elif ranking_oponente <= 50: peso_base = 8
        else: peso_base = 5
        
        # Multiplicadores por contexto
        multiplicador_contexto = 1.0
        
        # Si ambos jugaron recientemente (últimos 20 partidos)
        if self._partidos_recientes(player1_history, common_opponent_name) and self._partidos_recientes(player2_history, common_opponent_name):
            multiplicador_contexto += 0.3
        
        # Si hay múltiples enfrentamientos con el oponente común
        encuentros_a = self._count_matches(player1_history, common_opponent_name)
        encuentros_b = self._count_matches(player2_history, common_opponent_name)
        if encuentros_a >= 2 and encuentros_b >= 2:
            multiplicador_contexto += 0.4
        
        # El patrón de dominancia se aplica por separado en analyze_rivalry
        return peso_base * multiplicador_contexto

    def analyze_advanced_player_metrics(self, player_history, player_name):
        """
        📊 DIMENSIÓN 3 & 4: Calidad de Historial y Diversidad de Oponentes
        Calcula multiplicadores basados en la calidad y diversidad del historial de un jugador.
        """
        quality_multiplier = 1.0
        diversity_multiplier = 1.0
        analysis_log = []

        # Considerar hasta 45 partidos para el historial
        history_subset = player_history[:45]

        # ============================================
        # 1. CALIDAD DEL HISTORIAL (% victorias vs Top 30 en 45 partidos)
        # ============================================
        matches_vs_top30 = []
        wins_vs_top30 = 0
        for match in history_subset:
            # Intentar obtener el ranking del ranking_manager o del match directamente
            opponent_rank = self.ranking_manager.get_player_ranking(match.get('oponente'))
            # Fallback: usar opponent_ranking del match si existe
            if not opponent_rank:
                opponent_rank = match.get('opponent_ranking')

            # *** CÓDIGO QUE FALTABA - ESTO ES LO CRÍTICO ***
            if opponent_rank and opponent_rank <= 30:
                matches_vs_top30.append(match)
                if self.determine_match_winner(match, player_name):
                    wins_vs_top30 += 1
                   
        if len(matches_vs_top30) > 0:
            win_rate_vs_top30 = (wins_vs_top30 / len(matches_vs_top30)) * 100
            if win_rate_vs_top30 > 60:
                quality_multiplier = 1.5  # AJUSTADO: Máximo 50% de bonificación
                analysis_log.append(f"🏆 Calidad Historial: >60% victorias vs Top 30 ({win_rate_vs_top30:.1f}%) -> x1.5")
            elif 40 <= win_rate_vs_top30 <= 59:
                quality_multiplier = 1.5
                analysis_log.append(f"🏆 Calidad Historial: 40-59% victorias vs Top 30 ({win_rate_vs_top30:.1f}%) -> x1.5")
            elif 20 <= win_rate_vs_top30 <= 39:
                quality_multiplier = 1.2
                analysis_log.append(f"🏆 Calidad Historial: 20-39% victorias vs Top 30 ({win_rate_vs_top30:.1f}%) -> x1.2")
            else: # < 20%
                quality_multiplier = 1.0
                analysis_log.append(f"🏆 Calidad Historial: <20% victorias vs Top 30 ({win_rate_vs_top30:.1f}%) -> x1.0")

        # ============================================
        # 2. DIVERSIDAD DE OPONENTES DE CALIDAD (en todo el historial disponible)
        # ============================================
        faced_top50 = set()
        beaten_top30 = set()

        for match in player_history:
            opponent_name = match.get('oponente')
            opponent_rank = self.ranking_manager.get_player_ranking(opponent_name)

            # *** AGREGAR FALLBACK TAMBIÉN AQUÍ ***
            if not opponent_rank:
                opponent_rank = match.get('opponent_ranking')

            if opponent_rank and opponent_rank <= 50:
                faced_top50.add(opponent_name)
            
            if opponent_rank and opponent_rank <= 30:
                if self.determine_match_winner(match, player_name):
                    beaten_top30.add(opponent_name)

        # Bonus granular por enfrentar jugadores Top 50 (máx 20 jugadores para el bono)
        faced_bonus = (min(len(faced_top50), 20) / 20) * 0.25
        if faced_bonus > 0:
            diversity_multiplier += faced_bonus
            analysis_log.append(f"🌍 Diversidad (Enfrentados): {len(faced_top50)} Top 50 -> +{faced_bonus*100:.1f}%")

        # Bonus granular por vencer jugadores Top 30 (máx 10 jugadores para el bono)
        beaten_bonus = (min(len(beaten_top30), 10) / 10) * 0.40
        if beaten_bonus > 0:
            diversity_multiplier += beaten_bonus
            analysis_log.append(f"💪 Diversidad (Vencidos): {len(beaten_top30)} Top 30 -> +{beaten_bonus*100:.1f}%")

        return quality_multiplier, diversity_multiplier, analysis_log

    def analyze_strength_of_schedule(self, player_history, player_name):
        """
        💪 DIMENSIÓN 5: Analiza la "Fuerza del Calendario" (Strength of Schedule).
        Recompensa a jugadores por enfrentar (y especialmente vencer) a oponentes de alto ranking.
        """

        if not player_history:
            return 0, []

        schedule_score = 0
        analysis_log = []
        
        # Considerar los últimos 45 partidos
        history_subset = player_history[:45]

        for match in history_subset:
            # Intentar obtener el ranking del ranking_manager o del match directamente
            opponent_rank = self.ranking_manager.get_player_ranking(match.get('oponente'))

            # Fallback: usar opponent_ranking del match si existe
            if not opponent_rank:
                opponent_rank = match.get('opponent_ranking')

            if not opponent_rank:
                continue  # Oponente sin ranking, no aporta puntos                

            points = 0
            # Puntos base por ranking del oponente
            if opponent_rank <= 10: points = 25
            elif opponent_rank <= 20: points = 20
            elif opponent_rank <= 50: points = 15
            elif opponent_rank <= 100: points = 10
            elif opponent_rank <= 200: points = 5
            
            if points == 0: continue

            # Verificar si el partido tiene información de resultado
            has_outcome = match.get('outcome') or match.get('resultado')
            if not has_outcome:
                # Si no hay información de resultado, dar puntos neutrales (promedio)
                points *= 1.5  # Punto medio entre victoria (2.5) y derrota (1.0)
                analysis_log.append(f"⚖️ Enfrentamiento vs Rank {opponent_rank} ({match.get('oponente')}) -> +{points:.1f} pts")
                schedule_score += points  # *** LÍNEA CRÍTICA QUE FALTABA ***
            elif self.determine_match_winner(match, player_name):
                # Bonificación por victoria
                points *= 2.5
                analysis_log.append(f"💪 Victoria vs Rank {opponent_rank} ({match.get('oponente')}) -> +{points:.1f} pts")
                schedule_score += points
            else:
                # Puntos por derrota competitiva (se mantiene el punto base)
                analysis_log.append(f"🛡️ Derrota vs Rank {opponent_rank} ({match.get('oponente')}) -> +{points:.1f} pts")
                schedule_score += points
                
        if analysis_log:
            analysis_log.insert(0, f"📊 Puntuación total de 'Strength of Schedule': {schedule_score:.1f}")

        return schedule_score, analysis_log

    # ═══════════════════════════════════════════════════════════════════
    # N29 — CIRCUIT ASYMMETRY DEFLATOR (CAD)
    # ═══════════════════════════════════════════════════════════════════

    def circuit_tier_index(self, player_history):
        """
        Nodo-29: Calcula el Circuit Tier Index (CTI) del jugador.

        Mide el nivel promedio del circuito donde ha competido basándose
        en el ranking de los oponentes enfrentados. Ponderación temporal:
        los últimos 10 partidos pesan 2× para capturar el circuito ACTUAL.

        Criterios por partido:
          ranking ≤ 10  → tier_score = 5.0  (élite absoluta)
          ranking ≤ 50  → tier_score = 4.0  (top ATP)
          ranking ≤ 100 → tier_score = 3.0  (ATP consolidado)
          ranking ≤ 200 → tier_score = 2.0  (ATP/Challenger alto)
          ranking ≤ 500 → tier_score = 1.0  (Challenger/ITF alto)
          ranking > 500 → tier_score = 0.0  (ITF bajo)

        Returns:
            tuple(float, int): (CTI 0.0–5.0, n_partidos_con_ranking)
        """
        if not player_history:
            return 0.0, 0

        def rank_to_tier_score(rank):
            if rank <= 10:  return 5.0
            if rank <= 50:  return 4.0
            if rank <= 100: return 3.0
            if rank <= 200: return 2.0
            if rank <= 500: return 1.0
            return 0.0

        scores = []
        weights_list = []
        n_con_ranking = 0

        for i, match in enumerate(player_history):
            rank = match.get('opponent_ranking')
            if rank is None or rank <= 0:
                continue
            n_con_ranking += 1
            ts = rank_to_tier_score(rank)
            # Últimos 10 (i < 10) pesan 2×, resto 1× (REGLA-N29-3)
            w = 2.0 if i < 10 else 1.0
            scores.append(ts * w)
            weights_list.append(w)

        if not scores:
            return 0.0, 0

        cti = sum(scores) / sum(weights_list)
        return round(cti, 3), n_con_ranking

    def analyze_streaks_and_consistency(self, player_history, player_name):
        """
        🎯 DIMENSIÓN 2: Factor de Racha/Consistencia MEJORADO
        Analiza rachas, consistencia y momentum según criterios específicos
        """
        streak_multiplier = 1.0
        analysis_log = []

        # CORREGIDO: Asegurar que siempre hay logs si hay partidos
        if not player_history:
            return streak_multiplier, analysis_log

        # ============================================
        # 1. ANÁLISIS DE RACHAS DE VICTORIAS CONSECUTIVAS
        # ============================================
        current_win_streak = 0
        streak_opponents_ranks = []
        
        # Contar racha actual de victorias y obtener rankings de oponentes
        for match in player_history:
            if self.determine_match_winner(match, player_name):
                current_win_streak += 1
                opponent_rank = self.ranking_manager.get_player_ranking(match.get('oponente'))
                if opponent_rank:
                    streak_opponents_ranks.append(opponent_rank)
            else:
                break

        # Aplicar multiplicadores según criterios específicos (condiciones relajadas)
        if current_win_streak >= 5:
            # Racha 5+ victorias con al menos 2 vs Top 50: +50%
            top50_wins_in_streak = sum(1 for rank in streak_opponents_ranks if rank <= 50)
            if top50_wins_in_streak >= 2:
                streak_multiplier *= 1.50
                analysis_log.append(f"🔥 Racha de {current_win_streak} (con {top50_wins_in_streak} vs Top 50) -> +50%")
        
        if current_win_streak >= 3:
            # Racha 3+ victorias con al menos 2 vs Top 30: +30%
            top30_wins_in_streak = sum(1 for rank in streak_opponents_ranks if rank <= 30)
            if top30_wins_in_streak >= 2:
                streak_multiplier *= 1.30
                analysis_log.append(f"⚡ Racha de {current_win_streak} (con {top30_wins_in_streak} vs Top 30) -> +30%")

        # ============================================
        # 2. ANÁLISIS DE CONSISTENCIA VS NIVEL SIMILAR
        # ============================================
        player_info = self.ranking_manager.get_player_info(player_name)
        player_rank = player_info.get('ranking_position') if player_info else None
        if player_rank:
            # Consistencia >75% vs Rank 31-50: +25% multiplicador
            if 31 <= player_rank <= 50:
                matches_vs_similar = []
                for match in player_history[:45]:  # Últimos 45 partidos
                    opponent_rank = self.ranking_manager.get_player_ranking(match.get('oponente'))
                    if opponent_rank and 31 <= opponent_rank <= 50:
                        matches_vs_similar.append(match)
                
                if len(matches_vs_similar) >= 4:  # Mínimo 4 partidos para ser significativo
                    wins_vs_similar = sum(1 for match in matches_vs_similar 
                                        if self.determine_match_winner(match, player_name))
                    consistency_rate = wins_vs_similar / len(matches_vs_similar)
                    
                    if consistency_rate > 0.75:
                        streak_multiplier *= 1.25
                        analysis_log.append(f"🎯 Consistencia {int(consistency_rate*100)}% vs Rank 31-50 ({wins_vs_similar}/{len(matches_vs_similar)}) (+25%)")
            
            # Consistencia general vs mismo tier
            else:
                def get_tier_range(rank):
                    if rank <= 10: return (1, 10)
                    elif rank <= 20: return (11, 20)  
                    elif rank <= 30: return (21, 30)
                    elif rank <= 50: return (31, 50)
                    elif rank <= 100: return (51, 100)
                    else: return (101, 200)

                tier_min, tier_max = get_tier_range(player_rank)
                matches_vs_tier = []
                
                for match in player_history[:45]:
                    opponent_rank = self.ranking_manager.get_player_ranking(match.get('oponente'))
                    if opponent_rank and tier_min <= opponent_rank <= tier_max:
                        matches_vs_tier.append(match)
                
                if len(matches_vs_tier) >= 4:
                    wins_vs_tier = sum(1 for match in matches_vs_tier 
                                     if self.determine_match_winner(match, player_name))
                    consistency_rate = wins_vs_tier / len(matches_vs_tier)
                    
                    if consistency_rate > 0.80:
                        streak_multiplier *= 1.20
                        analysis_log.append(f"💪 Consistencia {int(consistency_rate*100)}% vs Rank {tier_min}-{tier_max} (+20%)")

        # ============================================
        # 3. ANÁLISIS DE MOMENTUM RECIENTE
        # ============================================
        # Momentum últimos 10 partidos >70%: +20% multiplicador
        recent_matches = player_history[:10]
        if len(recent_matches) >= 8:  # Al menos 8 partidos para ser significativo
            recent_wins = sum(1 for match in recent_matches 
                            if self.determine_match_winner(match, player_name))
            momentum_rate = recent_wins / len(recent_matches)
            
            if momentum_rate > 0.70:
                streak_multiplier *= 1.20
                analysis_log.append(f"🚀 Momentum reciente {int(momentum_rate*100)}% en últimos {len(recent_matches)} partidos (+20%)")

        # ============================================
        # 4. ANÁLISIS ADICIONAL: VICTORIAS VS TOP TIERS
        # ============================================
        # Bonificación por victorias recientes contra top players
        recent_top_wins = 0
        for match in player_history[:20]:
            if self.determine_match_winner(match, player_name):
                opponent_rank = self.ranking_manager.get_player_ranking(match.get('oponente'))
                if opponent_rank and opponent_rank <= 20:
                    recent_top_wins += 1
        
        if recent_top_wins >= 3:
            streak_multiplier *= 1.15
            analysis_log.append(f"🏆 {recent_top_wins} victorias recientes vs Top 20 (+15%)")

        # CORREGIDO: Si no hay logs específicos, agregar uno básico
        if not analysis_log:
            analysis_log.append(f"📊 Análisis de racha: sin bonificaciones aplicables (multiplicador base: {streak_multiplier:.2f})")

        # AJUSTADO: Limitar el multiplicador máximo a 1.5
        final_multiplier = min(streak_multiplier, 1.5)
        if final_multiplier < streak_multiplier:
            analysis_log.append(f"⚠️ Multiplicador de racha limitado a x1.5 (calculado: {streak_multiplier:.2f})")

        return final_multiplier, analysis_log
    
    def determine_match_winner(self, match_data, target_player):
        """🏆 Determinar si el jugador objetivo ganó el partido de forma robusta."""
        outcome = match_data.get('outcome', '').lower()
        
        # Prioridad 1: Usar el campo 'outcome' si es claro
        if 'ganó' in outcome or 'win' in outcome:
            return True
        if 'perdió' in outcome or 'loss' in outcome:
            return False
        
        # Prioridad 2: Inferir del resultado si 'outcome' no es claro
        resultado = match_data.get('resultado', '')
        oponente = match_data.get('oponente', '')
        
        # Si el oponente se retiró (RET), el jugador objetivo ganó
        if 'RET' in resultado.upper():
            return True
        
        # Si el oponente no se presentó (WO), el jugador objetivo ganó
        if 'WO' in resultado.upper():
             return True

        # Lógica de fallback: si no hay indicación clara de derrota, asumir victoria
        # Esto es arriesgado pero mantiene el comportamiento anterior como último recurso.
        return 'perdió' not in outcome and 'loss' not in outcome

    def analizar_contundencia(self, resultado):
        """
        Analiza la contundencia de una victoria según las reglas especificadas.
        - Victoria por 2-0 sets contundentes: +1.5x
        - Victoria por 2-0 sets ajustados: +1.2x
        - Victoria por 2-1 sets: +1.0x
        """
        try:
            # Extraer sets (ej: '2-0' de '2-0 (6-2, 6-3)')
            sets_match = re.search(r'(\d+-\d+)', resultado)
            if not sets_match:
                return 1.0

            sets_str = sets_match.group(1)
            sets = [int(s) for s in sets_str.split('-')]
            
            # Victoria en 3 sets
            if min(sets) == 1:
                return 1.0

            # Victoria en 2 sets (2-0)
            if min(sets) == 0:
                # Extraer juegos para diferenciar contundente de ajustado
                games_match = re.findall(r'(\d+-\d+)', resultado)
                if len(games_match) >= 2: # Best of 3
                    total_game_diff = sum(abs(int(g.split('-')[0]) - int(g.split('-')[1])) for g in games_match)
                    
                    # Umbral para contundencia: diferencia de 7 o más juegos
                    if total_game_diff >= 7:
                        return 1.5  # Contundente
                    else:
                        return 1.2  # Ajustado
                return 1.2 # Default para 2-0 si no se pueden parsear los juegos

        except Exception:
            return 1.0 # Default en caso de error
        
        return 1.0

    def analizar_resistencia(self, resultado):
        """
        Analiza la resistencia en una derrota según las reglas especificadas.
        - Perdió pero ganó 1 set: +0.5x
        - Perdió ajustado en 2 sets: +0.3x
        - Perdió contundente: +0.0x
        """
        try:
            # Extraer sets
            sets_match = re.search(r'(\d+-\d+)', resultado)
            if not sets_match: return 0.0

            sets_str = sets_match.group(1)
            sets = [int(s) for s in sets_str.split('-')]

            # Perdió pero ganó 1 set
            if min(sets) == 1:
                return 0.5

            # Perdió en 2 sets (0-2)
            if min(sets) == 0:
                games_match = re.findall(r'(\d+-\d+)', resultado)
                if len(games_match) >= 2:
                    total_game_diff = sum(abs(int(g.split('-')[0]) - int(g.split('-')[1])) for g in games_match)
                    
                    # Si hubo un tie-break o la diferencia de juegos es <= 5, es ajustado
                    if any('7-6' in g or '6-7' in g for g in games_match) or total_game_diff <= 5:
                        return 0.3 # Ajustado
                    else:
                        return 0.0 # Contundente
                return 0.0 # Default para 0-2 si no se pueden parsear juegos

        except Exception:
            return 0.0 # Default en caso de error
            
        return 0.0
    
    def find_common_opponents(self, player1_history, player2_history):
        """🤝 Encontrar oponentes comunes entre dos jugadores"""

        if not player1_history or not player2_history:
            return []

        opponents1 = {self.ranking_manager.normalize_name(m.get('oponente', '')) for m in player1_history if m.get('oponente')}
        opponents2 = {self.ranking_manager.normalize_name(m.get('oponente', '')) for m in player2_history if m.get('oponente')}
        return list(opponents1.intersection(opponents2))

    def analyze_direct_h2h(self, h2h_matches, player1_name, player2_name):
        """
        Analiza los enfrentamientos directos (H2H) entre dos jugadores.
        NUEVO: Se introduce una ponderación basada en la antigüedad del partido.
        - Partidos con más de 250 días de antigüedad tienen una ponderación de 0.
        - Partidos recientes tienen una ponderación base de 1.0.
        """
        p1_score = 0.0
        p2_score = 0.0
        log = []
        today = datetime.now()
        H2H_RECENT_DAYS_THRESHOLD = 250

        for match in h2h_matches:
            try:
                match_date_str = match.get('fecha')
                if not match_date_str:
                    continue

                match_date = datetime.strptime(match_date_str, '%d.%m.%Y')
                days_since_match = (today - match_date).days
                
                ponderacion = 0.0
                if days_since_match <= H2H_RECENT_DAYS_THRESHOLD:
                    ponderacion = 1.0  # Ponderación base para partidos recientes
                
                winner = match.get("ganador")
                if winner == player1_name:
                    p1_score += ponderacion
                    if ponderacion > 0:
                        log.append(f"Victoria RECIENTE de {player1_name} en {match_date_str} (Ponderación: {ponderacion})")
                    else:
                        log.append(f"Victoria ANTIGUA de {player1_name} en {match_date_str} (Ponderación: 0)")

                elif winner == player2_name:
                    p2_score += ponderacion
                    if ponderacion > 0:
                        log.append(f"Victoria RECIENTE de {player2_name} en {match_date_str} (Ponderación: {ponderacion})")
                    else:
                        log.append(f"Victoria ANTIGUA de {player2_name} en {match_date_str} (Ponderación: 0)")

            except (ValueError, TypeError) as e:
                log.append(f"Error procesando fecha de H2H: {match.get('fecha')} - {e}")
                continue

        return p1_score, p2_score, log

    def get_ranking_metrics(self, player_name):
        """
        Calcula las métricas avanzadas de ranking para un jugador de forma robusta.
        """
        player_info = self.ranking_manager.get_player_info(player_name)
        if not player_info:
            return {}

        def _safe_get_numeric(key, default=0):
            val = player_info.get(key)
            if isinstance(val, (int, float)):
                return val
            return default

        pts = _safe_get_numeric('ranking_points', 0)
        
        # Lógica de fallback mejorada
        prox_pts = _safe_get_numeric('prox_points')
        if prox_pts == 0:
            prox_pts = pts

        pts_max = _safe_get_numeric('max_points')
        if pts_max == 0:
            pts_max = prox_pts
            
        defense_points = _safe_get_numeric('defense_points', 0)

        # Asegurarse de que los cálculos no den negativo
        already_secured = max(0, prox_pts - pts)
        improvement_potential = max(0, pts_max - pts)
        pressure_index = defense_points - already_secured

        return {
            'pts': pts,
            'prox_pts': prox_pts,
            'pts_max': pts_max,
            'defense_points': defense_points,
            'already_secured': already_secured,
            'improvement_potential': improvement_potential,
            'pressure_index': pressure_index
        }

    def analyze_surface_specialization(self, player_history, surface, player_name):
        """
        🔬 Analiza la especialización y calidad de un jugador en una superficie específica.
        Recompensa victorias de calidad y considera la tasa de victorias.
        """
        _empty_meta = {'score': 0, 'raw_score': 0.0, 'win_rate': 0.0, 'matches': 0, 'skill_factor': 1.0, 'alpha_bonus': 1.0, 'volume_confidence': 0.0, 'surface_alpha': 0.0}
        if not player_history:
            return _empty_meta, []

        if not surface or surface == 'Desconocida':
            return _empty_meta, []

        # Normalizar superficie de entrada (puede venir ya normalizada o en minúsculas)
        surface_lower = surface.lower() if surface else ''
        normalized_surface = self.SURFACE_NORMALIZATION_MAP.get(surface_lower, surface)

        # Si la superficie ya está normalizada (ej: 'Dura'), usarla directamente
        if surface in ['Dura', 'Arcilla', 'Hierba', 'Indoor']:
            normalized_surface = surface
        
        surface_matches = []
        for m in player_history:
            match_surface = m.get('superficie', '')
            # **NUEVO: Si no hay superficie en el match, tratarlo como si coincidiera**
            if not match_surface:
                match_surface_normalized = normalized_surface  # Asumir que coincide
            # Normalizar la superficie del partido del historial
            elif match_surface in ['Dura', 'Arcilla', 'Hierba', 'Indoor']:
                match_surface_normalized = match_surface
            else:
                match_surface_normalized = self.SURFACE_NORMALIZATION_MAP.get(match_surface.lower(), match_surface)

            if match_surface_normalized == normalized_surface:
                surface_matches.append(m)

        if len(surface_matches) < 2: # Mínimo 2 partidos; VolConf (n/8) penaliza muestras pequeñas
            return {'score': 0, 'raw_score': 0.0, 'win_rate': 0.0, 'matches': len(surface_matches), 'skill_factor': 1.0, 'alpha_bonus': 1.0, 'volume_confidence': 0.0, 'surface_alpha': 0.0}, [f"LOG_SURFACE: No hay suficientes partidos en {normalized_surface} ({len(surface_matches)}) para un análisis profundo."]
        
        quality_score = 0
        analysis_log = []

        for match in surface_matches:
            opponent_rank = self.ranking_manager.get_player_ranking(match.get('oponente'))
            if not opponent_rank:
                continue

            points = 0
            # Puntos base por victoria según ranking del oponente
            if self.determine_match_winner(match, player_name):
                if opponent_rank <= 10: points = 50
                elif opponent_rank <= 20: points = 40
                elif opponent_rank <= 50: points = 25
                elif opponent_rank <= 100: points = 15
                else: points = 5
                
                # Multiplicador por contundencia
                contundencia = self.analizar_contundencia(match.get('resultado', ''))
                points *= contundencia
                analysis_log.append(f"Victoria vs Rank {opponent_rank} ({match.get('oponente')}) en {normalized_surface} -> +{points:.1f} pts")
            else:
                # Puntos por derrota competitiva
                resistencia = self.analizar_resistencia(match.get('resultado', ''))
                if resistencia > 0:
                    points = 10 * resistencia # Max 5 puntos por derrota ajustada
                    analysis_log.append(f"Derrota competitiva vs Rank {opponent_rank} ({match.get('oponente')}) en {normalized_surface} -> +{points:.1f} pts")

            quality_score += points

        # FIX-4: Bonus torneo completo — ≥4 victorias sin derrotas en mismo torneo+año
        # Ganar un torneo completo es cualitativamente diferente a partidos dispersos
        # Si torneo='N/A' (Ninja API no devuelve KF), agrupar por semana del año
        import datetime as _dt_mod
        _tour_stats = {}
        _today = _dt_mod.datetime.today()
        for _m in surface_matches:
            _tname = str(_m.get('torneo', '') or '')
            _fecha_str = str(_m.get('fecha', '') or '')
            _tyear = _fecha_str[-4:] if len(_fecha_str) >= 4 else _fecha_str
            if _tname in ('N/A', '', 'Unknown'):
                try:
                    _dt = _dt_mod.datetime.strptime(_fecha_str, '%d.%m.%Y')
                    _tname = f'_semana_{_dt.isocalendar()[1]}'
                except Exception:
                    _tname = f'_chunk_{_tyear}'
            _tk = (_tname, _tyear)
            if _tk not in _tour_stats:
                _tour_stats[_tk] = {'wins': 0, 'losses': 0, 'label': _tname,
                                    'max_fecha': None, 'best_opp_rank': 9999}
            if self.determine_match_winner(_m, player_name):
                _tour_stats[_tk]['wins'] += 1
            else:
                _tour_stats[_tk]['losses'] += 1
            # Track recency and best opponent
            try:
                _mdate = _dt_mod.datetime.strptime(_fecha_str, '%d.%m.%Y')
                if _tour_stats[_tk]['max_fecha'] is None or _mdate > _tour_stats[_tk]['max_fecha']:
                    _tour_stats[_tk]['max_fecha'] = _mdate
            except Exception:
                pass
            _opp_rank = _m.get('opponent_ranking') or _m.get('ranking_oponente') or 9999
            try:
                _opp_rank = int(_opp_rank)
            except (ValueError, TypeError):
                _opp_rank = 9999
            if _opp_rank < _tour_stats[_tk]['best_opp_rank']:
                _tour_stats[_tk]['best_opp_rank'] = _opp_rank

        # Nodo-60 D60-02: tracked for GCS_RECENCY_BOOST
        _gcs_boost_tier = None
        _gcs_boost_days = None
        for (_tname, _tyear), _ts in _tour_stats.items():
            # Nodo-57 D57-03: gate tier-aware — GS necesita 7W, no 4W
            from config import detectar_tier as _dt_tier
            _tier_champ = _dt_tier(_tname)
            _min_wins = _MIN_WINS_CHAMPION.get(_tier_champ, 5)  # fallback conservador
            if _ts['wins'] >= _min_wins and _ts['losses'] == 0:
                # Gate: torneo debe ser reciente (≤90 días). Un torneo de hace 1 año no es señal.
                if _ts['max_fecha'] is not None:
                    _days_ago = (_today - _ts['max_fecha']).days
                    if _days_ago > 90:
                        # Nodo-57 D57-04: compensación reducida para campeones históricos en superficie
                        if _days_ago <= 365:
                            _comp_bonus = 1.15 if _days_ago <= 180 else 1.05
                            quality_score *= _comp_bonus
                            analysis_log.append(
                                f"TORNEO_COMPLETO_EXPIRADO: {_tname} {_tyear} "
                                f"({_ts['wins']}W-0L, hace {_days_ago}d) → sin bonus activo "
                                f"pero +{(_comp_bonus-1)*100:.0f}% historial superficie"
                            )
                        else:
                            analysis_log.append(
                                f"TORNEO_COMPLETO_EXPIRADO: {_tname} {_tyear} "
                                f"({_ts['wins']}W-0L, hace {_days_ago}d) → sin bonus (>365d)"
                            )
                        continue
                else:
                    continue  # sin fecha = no podemos verificar recencia
                # E-2: Bonus escalonado — base 1.3 + recency + top10 + final
                _bonus = 1.3
                _bonus_parts = []
                # +0.2 si torneo terminó hace <=14 días
                if _days_ago <= 14:
                    _bonus += 0.2
                    _bonus_parts.append(f'recency({_days_ago}d)')
                # +0.1 si venció a Top-10
                if _ts['best_opp_rank'] <= 10:
                    _bonus += 0.1
                    _bonus_parts.append(f'top10(#{_ts["best_opp_rank"]})')
                # +0.1 si victorias >= mínimo requerido por tier (implica final ganada)
                if _ts['wins'] >= _min_wins:
                    _bonus += 0.1
                    _bonus_parts.append(f'final({_ts["wins"]}W≥{_min_wins}req)')
                _bonus = min(_bonus, 2.0)  # cap
                quality_score *= _bonus
                _parts_str = ' + '.join(_bonus_parts) if _bonus_parts else 'base'
                analysis_log.append(
                    f"TORNEO_COMPLETO_BONUS: {_tname} {_tyear} "
                    f"({_ts['wins']}W-0L, tier={_tier_champ}, req≥{_min_wins}) "
                    f"→ x{_bonus:.1f} quality_score [{_parts_str}]"
                )
                # Nodo-60 D60-02: track para GCS_RECENCY_BOOST
                _gcs_boost_tier = _tier_champ
                _gcs_boost_days = _days_ago
                break  # un solo bonus aunque haya múltiples torneos completos

        # Normalizar por número de partidos para no favorecer a quien jugó más
        normalized_quality_score = (quality_score / len(surface_matches)) * 2.5
        
        # Factor de tasa de victorias
        wins = sum(1 for m in surface_matches if self.determine_match_winner(m, player_name))
        win_rate = (wins / len(surface_matches))

        # Skill Factor — función convexa anclada en 50% (coin flip = neutral)
        # 85% → 2.22x | 60% → 1.33x | 50% → 1.0x | 33% → 0.54x
        # Reemplaza (1 + win_rate) que solo daba rango [1.0, 2.0] con ratio máximo 1.27x
        skill_factor = (max(win_rate, 0.01) / 0.5) ** 1.5

        # Surface Alpha — win rate en esta superficie vs overall (alpha_vs_elo para superficies)
        # Positivo = especialista real. Negativo = peor en esta superficie que en general.
        overall_wins = sum(1 for m in player_history if self.determine_match_winner(m, player_name))
        overall_wr = overall_wins / max(len(player_history), 1)
        surface_alpha = win_rate - overall_wr
        alpha_bonus = 1.0 + max(surface_alpha, 0) * 2.0  # +10% alpha → 1.20x boost

        final_score = normalized_quality_score * skill_factor * alpha_bonus

        # Confianza por volumen: n=2→0.25, n=5→0.63, n=8+→1.0
        volume_confidence = min(len(surface_matches) / 8.0, 1.0)
        final_score = final_score * volume_confidence

        # Nodo-60 D60-02: GCS_RECENCY_BOOST — campeón pre-torneo en esta misma superficie
        # Aplica DESPUÉS de normalización para evitar dilución por historial histórico.
        # Solo tier≥ATP500 y ≤21 días. Constantes congeladas hasta n≥30 (H60-01).
        _GCS_TIER_MIN = {'grand_slam', 'atp1000', 'atp500'}
        _gcs_active = False
        if (_gcs_boost_tier in _GCS_TIER_MIN and
                _gcs_boost_days is not None and _gcs_boost_days <= 21):
            if _gcs_boost_days <= 7:
                _gcs_mult = 2.2   # muy reciente: ganó la semana pasada
            elif _gcs_boost_days <= 14:
                _gcs_mult = 1.8   # reciente: ≤2 semanas
            else:
                _gcs_mult = 1.5   # moderado: 15-21 días
            final_score *= _gcs_mult
            _gcs_active = True
            analysis_log.append(
                f"GCS_RECENCY_BOOST: tier={_gcs_boost_tier} days={_gcs_boost_days}d "
                f"→ ×{_gcs_mult} final_score (Nodo-60)"
            )

        analysis_log.insert(0, f"Puntuación de Calidad en {normalized_surface}: {final_score:.1f} (Base: {quality_score:.1f}, Partidos: {len(surface_matches)}, Tasa Vic: {win_rate:.1%}, SkillF: {skill_factor:.2f}, Alpha: {surface_alpha:+.1%}, VolConf: {volume_confidence:.2f})")

        # Flag for E-1 dynamic weight boost
        _has_torneo_bonus = any('TORNEO_COMPLETO_BONUS' in l for l in analysis_log)

        return {
            'score':             round(final_score, 4),
            'raw_score':         round(normalized_quality_score, 4),
            'win_rate':          round(win_rate, 4),
            'matches':           len(surface_matches),
            'skill_factor':      round(skill_factor, 4),
            'alpha_bonus':       round(alpha_bonus, 4),
            'volume_confidence': round(volume_confidence, 4),
            'surface_alpha':     round(surface_alpha, 4),
            'torneo_completo':   _has_torneo_bonus,
            'gcs_active':        _gcs_active,
            'gcs_days':          _gcs_boost_days,
        }, analysis_log

    def analyze_surface_performance(self, player_history, player_name):
        """📊 Analiza el rendimiento de un jugador por superficie."""
        surface_stats = {
            'Dura': {'wins': 0, 'losses': 0, 'matches': 0},
            'Arcilla': {'wins': 0, 'losses': 0, 'matches': 0},
            'Hierba': {'wins': 0, 'losses': 0, 'matches': 0},
            'Indoor': {'wins': 0, 'losses': 0, 'matches': 0},
            'Desconocida': {'wins': 0, 'losses': 0, 'matches': 0}
        }

        for match in player_history:
            surface = match.get('superficie', 'Desconocida')
            if surface not in surface_stats:
                surface = 'Desconocida'
            
            surface_stats[surface]['matches'] += 1
            if self.determine_match_winner(match, player_name):
                surface_stats[surface]['wins'] += 1
            else:
                surface_stats[surface]['losses'] += 1
        
        for surface, stats in surface_stats.items():
            if stats['matches'] > 0:
                stats['win_rate'] = round((stats['wins'] / stats['matches']) * 100, 1)
            else:
                stats['win_rate'] = 0.0
        
        return surface_stats

    def analyze_location_factors(self, player_history, player_nationality):
        """🌍 Analiza el factor "Home Advantage" y el rendimiento por región."""
        if not player_nationality or player_nationality == 'N/A':
            return {'home_advantage': None, 'regional_comfort': {}}

        home_stats = {'wins': 0, 'losses': 0, 'matches': 0}
        regional_stats = {}

        for match in player_history:
            match_country = match.get('pais')
            if not match_country:
                continue

            # 1. Home Advantage
            if match_country == player_nationality:
                home_stats['matches'] += 1
                if self.determine_match_winner(match, player_nationality):
                    home_stats['wins'] += 1
                else:
                    home_stats['losses'] += 1
            
            # 2. Regional Comfort
            continent = self.COUNTRY_TO_CONTINENT_MAP.get(match_country, 'Otros')
            if continent not in regional_stats:
                regional_stats[continent] = {'wins': 0, 'losses': 0, 'matches': 0}
            
            regional_stats[continent]['matches'] += 1
            if self.determine_match_winner(match, player_nationality):
                regional_stats[continent]['wins'] += 1
            else:
                regional_stats[continent]['losses'] += 1

        # Calcular tasas de victoria
        if home_stats['matches'] > 0:
            home_stats['win_rate'] = round((home_stats['wins'] / home_stats['matches']) * 100, 1)
        else:
            home_stats['win_rate'] = 0.0

        for region, stats in regional_stats.items():
            if stats['matches'] > 0:
                stats['win_rate'] = round((stats['wins'] / stats['matches']) * 100, 1)
            else:
                stats['win_rate'] = 0.0
        
        return {'home_advantage': home_stats, 'regional_comfort': regional_stats}

    def analyze_rivalry(self, player1_history, player2_history, player1_name, player2_name, player1_form, player2_form, direct_h2h_matches, current_match_context, p1_elo, p2_elo, tournament_name='', optimized_weights=None):
        """⚔️ Realizar análisis completo de rivalidad transitiva"""
        logger.info(f"⚔️ Analizando rivalidad transitiva: {player1_name} vs {player2_name}")
        
        # MEJORADO: Usar get_player_info para una búsqueda de ranking más robusta
        player1_info = self.ranking_manager.get_player_info(player1_name)
        player2_info = self.ranking_manager.get_player_info(player2_name)
        
        player1_rank = player1_info.get('ranking_position') if player1_info else None
        player2_rank = player2_info.get('ranking_position') if player2_info else None
        player1_nationality = player1_info.get('nationality') if player1_info else None
        player2_nationality = player2_info.get('nationality') if player2_info else None

        # Fallback para obtener nacionalidad del historial del oponente
        if not player1_nationality or player1_nationality == 'N/A':
            for match in player2_history:
                if self.ranking_manager.normalize_name(match.get('oponente', '')) == self.ranking_manager.normalize_name(player1_name):
                    if match.get('opponent_nationality') and match.get('opponent_nationality') != 'N/A':
                        player1_nationality = match['opponent_nationality']
                        logger.info(f"   ℹ️ Nacionalidad de {player1_name} inferida del historial de {player2_name}: {player1_nationality}")
                        break
        
        if not player2_nationality or player2_nationality == 'N/A':
            for match in player1_history:
                if self.ranking_manager.normalize_name(match.get('oponente', '')) == self.ranking_manager.normalize_name(player2_name):
                    if match.get('opponent_nationality') and match.get('opponent_nationality') != 'N/A':
                        player2_nationality = match['opponent_nationality']
                        logger.info(f"   ℹ️ Nacionalidad de {player2_name} inferida del historial de {player1_name}: {player2_nationality}")
                        break
        
        logger.info(f"   📊 Rankings: {player1_name} ({player1_rank}, {player1_nationality}) vs {player2_name} ({player2_rank}, {player2_nationality})")
        logger.info(f"   🌍 Contexto Partido Actual: Torneo en {current_match_context.get('country')}, Superficie: {current_match_context.get('surface')}")
        
        # Realizar análisis de superficie y ubicación
        p1_surface_stats = self.analyze_surface_performance(player1_history, player1_name)
        p2_surface_stats = self.analyze_surface_performance(player2_history, player2_name)
        p1_location_stats = self.analyze_location_factors(player1_history, player1_nationality)
        p2_location_stats = self.analyze_location_factors(player2_history, player2_nationality)
        
        common_opponents = self.find_common_opponents(player1_history, player2_history)
        logger.info(f"   🤝 Oponentes comunes encontrados: {len(common_opponents)}")
        if common_opponents:
            logger.info("      Oponentes: " + ", ".join(opp for opp in common_opponents[:10]))
        
        prediction_context = {
            'p1_surface_stats': p1_surface_stats,
            'p2_surface_stats': p2_surface_stats,
            'p1_nationality': player1_nationality,
            'p2_nationality': player2_nationality,
            'current_match_surface': current_match_context.get('surface'),
            'current_match_country': current_match_context.get('country'),
            # F4 (Nodo-46): pasar campos necesarios para Surface Context Discount
            'apply_surface_discount': current_match_context.get('apply_surface_discount', True),
            'tournament_context': current_match_context.get('tournament_context', {}),
        }

        if not common_opponents:
            # Erdős graph con solo historial propio (sin oponentes comunes clásicos)
            try:
                _p1_pts = historial_a_partidos(player1_history, player1_name)
                _p2_pts = historial_a_partidos(player2_history, player2_name)
                _grafo = construir_grafo_victorias(_p1_pts + _p2_pts)
                _erdos = distancia_erdos(player1_name, player2_name, _grafo)
            except Exception:
                _erdos = {'erdos_score': 0.0, 'erdos_score_raw': 0.5, 'paths': [], 'n_paths': 0, 'max_depth_alcanzado': 0}

            # Sin oponentes comunes directos, usar Erdős como señal transitiva única.
            # erdos_score en [-1, +1]: positivo = P1 tiene ventaja, negativo = P2.
            # Escala × 2.0: erdos_score 0.5 → 1.0 pts (rango típico rivalry_score ≈ 0-15).
            _p1_erdos_co = 0.0
            _p2_erdos_co = 0.0
            if _erdos['n_paths'] > 0 and _erdos['erdos_score'] != 0.0:
                _erdos_contrib = abs(_erdos['erdos_score']) * 2.0
                if _erdos['erdos_score'] > 0:
                    _p1_erdos_co = _erdos_contrib
                else:
                    _p2_erdos_co = _erdos_contrib

            return {
                'player1_rank': player1_rank,
                'player2_rank': player2_rank,
            'common_opponents_count': 0,
            'p1_rivalry_score': _p1_erdos_co,
            'p2_rivalry_score': _p2_erdos_co,
            'player1_advantages': [],
            'player2_advantages': [],
            'p1_surface_stats': p1_surface_stats,
            'p2_surface_stats': p2_surface_stats,
            'p1_location_stats': p1_location_stats,
            'p2_location_stats': p2_location_stats,
            'erdos_analysis': _erdos,
            'prediction': self.generate_advanced_prediction(player1_info, player2_info, _p1_erdos_co, _p2_erdos_co, player1_name, player2_name, player1_history, player2_history, 0, 0, player1_form, player2_form, direct_h2h_matches, tournament_name, prediction_context, p1_elo, p2_elo, optimized_weights=optimized_weights, n_common_opponents=0, n_erdos_paths=_erdos.get('n_paths', 0))
        }

        player1_advantages = []
        player2_advantages = []
        p1_common_opponent_score = 0
        p2_common_opponent_score = 0

        # M-28-3 (Nodo-28 Fase 1): superficie del partido objetivo para filtrar common_opponents
        _target_surface_raw = prediction_context.get('current_match_surface', '')
        _target_surface = self.SURFACE_NORMALIZATION_MAP.get(
            (_target_surface_raw or '').lower(), _target_surface_raw
        ) if _target_surface_raw else None

        for common_opponent in common_opponents:
            p1_matches_all = [m for m in player1_history if self.ranking_manager.normalize_name(m.get('oponente', '')) == common_opponent]
            p2_matches_all = [m for m in player2_history if self.ranking_manager.normalize_name(m.get('oponente', '')) == common_opponent]

            # M-28-1: preferir partidos en misma superficie; fallback a todos
            if _target_surface:
                p1_matches_surf = [m for m in p1_matches_all if self.SURFACE_NORMALIZATION_MAP.get((m.get('superficie', '') or '').lower(), m.get('superficie', '')) == _target_surface or m.get('superficie', '') == _target_surface]
                p2_matches_surf = [m for m in p2_matches_all if self.SURFACE_NORMALIZATION_MAP.get((m.get('superficie', '') or '').lower(), m.get('superficie', '')) == _target_surface or m.get('superficie', '') == _target_surface]
                p1_matches = p1_matches_surf if p1_matches_surf else p1_matches_all
                p2_matches = p2_matches_surf if p2_matches_surf else p2_matches_all
            else:
                p1_matches = p1_matches_all
                p2_matches = p2_matches_all

            if p1_matches and p2_matches:
                p1_recent = p1_matches[0]
                p2_recent = p2_matches[0]

                p1_won = self.determine_match_winner(p1_recent, player1_name)
                p2_won = self.determine_match_winner(p2_recent, player2_name)

                opponent_rank = self.ranking_manager.get_player_ranking(p1_recent.get('oponente', ''))

                weight = self.calcular_peso_oponentes_comunes(player1_history, player2_history, common_opponent, player1_name, player2_name)

                # M-28-2: surface_relevance — misma superficie +30%, otra -40%
                if _target_surface:
                    _p1_surf = self.SURFACE_NORMALIZATION_MAP.get((p1_recent.get('superficie', '') or '').lower(), p1_recent.get('superficie', ''))
                    _p2_surf = self.SURFACE_NORMALIZATION_MAP.get((p2_recent.get('superficie', '') or '').lower(), p2_recent.get('superficie', ''))
                    _p1_same = (_p1_surf == _target_surface)
                    _p2_same = (_p2_surf == _target_surface)
                    # Promedio de relevancia de ambos partidos
                    _rel1 = 1.30 if _p1_same else 0.60
                    _rel2 = 1.30 if _p2_same else 0.60
                    surface_relevance = (_rel1 + _rel2) / 2.0
                else:
                    surface_relevance = 1.0
                weight *= surface_relevance

                opponent_display_rank = f"(rank {opponent_rank})" if opponent_rank is not None else "(rank N/A)"
                common_opponent_name = p1_recent.get('oponente', '')

                # Escenario 1: P1 ganó, P2 perdió
                if p1_won and not p2_won:
                    advantage_points = weight
                    if opponent_rank:
                        if opponent_rank <= 15: advantage_points *= 1.5  # AJUSTADO
                        elif 16 <= opponent_rank <= 30: advantage_points *= 1.25 # AJUSTADO
                        elif 31 <= opponent_rank <= 50: advantage_points *= 1.1  # AJUSTADO
                    
                    reason = f"{player1_name} venció a {common_opponent_name} {opponent_display_rank}, mientras que {player2_name} perdió"
                    advantage = {'opponent': common_opponent_name, 'opponent_rank': opponent_rank, 'weight': advantage_points, 'reason': reason, 'player1_result': p1_recent.get('resultado', ''), 'player2_result': p2_recent.get('resultado', ''), 'p1_won': p1_won, 'p2_won': p2_won, 'player1_date': p1_recent.get('fecha', ''), 'player2_date': p2_recent.get('fecha', ''), 'player1_surface': p1_recent.get('superficie', ''), 'player2_surface': p2_recent.get('superficie', '')}
                    player1_advantages.append(advantage)
                    p1_common_opponent_score += advantage_points

                # Escenario 2: P2 ganó, P1 perdió
                elif p2_won and not p1_won:
                    advantage_points = weight
                    if opponent_rank:
                        if opponent_rank <= 15: advantage_points *= 1.5  # AJUSTADO
                        elif 16 <= opponent_rank <= 30: advantage_points *= 1.25 # AJUSTADO
                        elif 31 <= opponent_rank <= 50: advantage_points *= 1.1  # AJUSTADO

                    reason = f"{player2_name} venció a {common_opponent_name} {opponent_display_rank}, mientras que {player1_name} perdió"
                    advantage = {'opponent': common_opponent_name, 'opponent_rank': opponent_rank, 'weight': advantage_points, 'reason': reason, 'player1_result': p1_recent.get('resultado', ''), 'player2_result': p2_recent.get('resultado', ''), 'p1_won': p1_won, 'p2_won': p2_won, 'player1_date': p1_recent.get('fecha', ''), 'player2_date': p2_recent.get('fecha', ''), 'player1_surface': p1_recent.get('superficie', ''), 'player2_surface': p2_recent.get('superficie', '')}
                    player2_advantages.append(advantage)
                    p2_common_opponent_score += advantage_points

                # Escenario 3: Ambos ganaron
                elif p1_won and p2_won:
                    contundencia_p1 = self.analizar_contundencia(p1_recent.get('resultado', ''))
                    contundencia_p2 = self.analizar_contundencia(p2_recent.get('resultado', ''))

                    advantage_points = (contundencia_p1 - contundencia_p2) * weight

                    if advantage_points > 0:
                        reason = f"Ambos vencieron a {common_opponent_name} {opponent_display_rank}, pero {player1_name} fue más contundente (Factor: {contundencia_p1:.1f} vs {contundencia_p2:.1f})"
                        advantage = {'opponent': common_opponent_name, 'opponent_rank': opponent_rank, 'weight': advantage_points, 'reason': reason, 'player1_result': p1_recent.get('resultado', ''), 'player2_result': p2_recent.get('resultado', ''), 'p1_won': p1_won, 'p2_won': p2_won, 'player1_date': p1_recent.get('fecha', ''), 'player2_date': p2_recent.get('fecha', ''), 'player1_surface': p1_recent.get('superficie', ''), 'player2_surface': p2_recent.get('superficie', '')}
                        player1_advantages.append(advantage)
                        p1_common_opponent_score += advantage_points
                    elif advantage_points < 0:
                        reason = f"Ambos vencieron a {common_opponent_name} {opponent_display_rank}, pero {player2_name} fue más contundente (Factor: {contundencia_p2:.1f} vs {contundencia_p1:.1f})"
                        advantage = {'opponent': common_opponent_name, 'opponent_rank': opponent_rank, 'weight': abs(advantage_points), 'reason': reason, 'player1_result': p1_recent.get('resultado', ''), 'player2_result': p2_recent.get('resultado', ''), 'p1_won': p1_won, 'p2_won': p2_won, 'player1_date': p1_recent.get('fecha', ''), 'player2_date': p2_recent.get('fecha', ''), 'player1_surface': p1_recent.get('superficie', ''), 'player2_surface': p2_recent.get('superficie', '')}
                        player2_advantages.append(advantage)
                        p2_common_opponent_score += abs(advantage_points)

                # Escenario 4: Ambos perdieron
                elif not p1_won and not p2_won:
                    resistencia_p1 = self.analizar_resistencia(p1_recent.get('resultado', ''))
                    resistencia_p2 = self.analizar_resistencia(p2_recent.get('resultado', ''))

                    advantage_points = (resistencia_p1 - resistencia_p2) * weight

                    if advantage_points > 0:
                        reason = f"Ambos perdieron con {common_opponent_name} {opponent_display_rank}, pero {player1_name} mostró más resistencia (Factor: {resistencia_p1:.1f} vs {resistencia_p2:.1f})"
                        advantage = {'opponent': common_opponent_name, 'opponent_rank': opponent_rank, 'weight': advantage_points, 'reason': reason, 'player1_result': p1_recent.get('resultado', ''), 'player2_result': p2_recent.get('resultado', ''), 'p1_won': p1_won, 'p2_won': p2_won, 'player1_date': p1_recent.get('fecha', ''), 'player2_date': p2_recent.get('fecha', ''), 'player1_surface': p1_recent.get('superficie', ''), 'player2_surface': p2_recent.get('superficie', '')}
                        player1_advantages.append(advantage)
                        p1_common_opponent_score += advantage_points
                    elif advantage_points < 0:
                        reason = f"Ambos perdieron con {common_opponent_name} {opponent_display_rank}, pero {player2_name} mostró más resistencia (Factor: {resistencia_p2:.1f} vs {resistencia_p1:.1f})"
                        advantage = {'opponent': common_opponent_name, 'opponent_rank': opponent_rank, 'weight': abs(advantage_points), 'reason': reason, 'player1_result': p1_recent.get('resultado', ''), 'player2_result': p2_recent.get('resultado', ''), 'p1_won': p1_won, 'p2_won': p2_won, 'player1_date': p1_recent.get('fecha', ''), 'player2_date': p2_recent.get('fecha', ''), 'player1_surface': p1_recent.get('superficie', ''), 'player2_surface': p2_recent.get('superficie', '')}
                        player2_advantages.append(advantage)
                        p2_common_opponent_score += abs(advantage_points)
        
        if player1_advantages:
            logger.info(f"   ⚡ Ventajas de {player1_name}:")
            for adv in player1_advantages[:3]: logger.info(f"      • {adv['reason']} (Peso: {adv['weight']:.2f})")
        
        if player2_advantages:
            logger.info(f"   ⚡ Ventajas de {player2_name}:")
            for adv in player2_advantages[:3]: logger.info(f"      • {adv['reason']} (Peso: {adv['weight']:.2f})")
        
        logger.info(f"   ⚖️ Peso de rivalidad P1: {p1_common_opponent_score:.2f}, P2: {p2_common_opponent_score:.2f}")

        # Erdős graph — ventaja transitiva a través del grafo completo de victorias
        try:
            _p1_pts = historial_a_partidos(player1_history, player1_name)
            _p2_pts = historial_a_partidos(player2_history, player2_name)
            _grafo = construir_grafo_victorias(_p1_pts + _p2_pts)
            _erdos = distancia_erdos(player1_name, player2_name, _grafo)
            logger.info(f"   🔢 Erdős score: {_erdos['erdos_score']:+.3f} ({_erdos['n_paths']} caminos, depth={_erdos['max_depth_alcanzado']})")
        except Exception as _e:
            logger.warning(f"   ⚠️ Erdős graph error (no crítico): {_e}")
            _erdos = {'erdos_score': 0.0, 'erdos_score_raw': 0.5, 'paths': [], 'n_paths': 0, 'max_depth_alcanzado': 0}

        # Erdős bonus: solo caminos de profundidad ≥2 aportan información nueva
        # (los directos ya están capturados en p1/p2_common_opponent_score).
        # Escala × 1.0: conservador para no sobreponderar sobre los paths directos.
        if _erdos['n_paths'] > 0 and _erdos['max_depth_alcanzado'] >= 2:
            _erdos_bonus = abs(_erdos['erdos_score']) * 1.0
            if _erdos['erdos_score'] > 0:
                p1_common_opponent_score += _erdos_bonus
            elif _erdos['erdos_score'] < 0:
                p2_common_opponent_score += _erdos_bonus

        return {
            'player1_rank': player1_rank,
            'player2_rank': player2_rank,
            'player1_nationality': player1_nationality,
            'player2_nationality': player2_nationality,
            'common_opponents_count': len(common_opponents),
            'common_opponents': common_opponents,
            'p1_rivalry_score': p1_common_opponent_score,
            'p2_rivalry_score': p2_common_opponent_score,
            'player1_advantages': player1_advantages,
            'player2_advantages': player2_advantages,
            'p1_surface_stats': p1_surface_stats,
            'p2_surface_stats': p2_surface_stats,
            'p1_location_stats': p1_location_stats,
            'p2_location_stats': p2_location_stats,
            'erdos_analysis': _erdos,
            'prediction': self.generate_advanced_prediction(player1_info, player2_info, p1_common_opponent_score, p2_common_opponent_score, player1_name, player2_name, player1_history, player2_history, len(player1_advantages), len(player2_advantages), player1_form, player2_form, direct_h2h_matches, tournament_name, prediction_context, p1_elo, p2_elo, optimized_weights=optimized_weights, n_common_opponents=len(common_opponents), n_erdos_paths=_erdos.get('n_paths', 0))
        }

    def generate_basic_prediction(self, rank1, rank2, player1_name, player2_name):
        """🔮 Generar predicción básica solo con rankings"""
        if rank1 is None and rank2 is None: return {'favored_player': 'Empate', 'confidence': 0, 'reasoning': ['Ambos jugadores sin ranking disponible']}
        if rank1 is None: return {'favored_player': player2_name, 'confidence': 60, 'reasoning': [f'{player2_name} tiene ranking ({rank2}), {player1_name} sin ranking']}
        if rank2 is None: return {'favored_player': player1_name, 'confidence': 60, 'reasoning': [f'{player1_name} tiene ranking ({rank1}), {player2_name} sin ranking']}
        
        rank_diff = abs(rank1 - rank2)
        if rank1 < rank2:
            return {'favored_player': player1_name, 'confidence': min(50 + (rank_diff * 2), 90), 'reasoning': [f'{player1_name} tiene mejor ranking ({rank1} vs {rank2})']}
        elif rank2 < rank1:
            return {'favored_player': player2_name, 'confidence': min(50 + (rank_diff * 2), 90), 'reasoning': [f'{player2_name} tiene mejor ranking ({rank2} vs {rank1})']}
        else:
            return {'favored_player': 'Empate', 'confidence': 0, 'reasoning': ['Rankings idénticos']}
    
    def calcular_h2h_immunity(self, direct_h2h_matches: list, favored: str, estado_fav: str) -> dict:
        """
        T19-01 (Nodo-19): Detecta si el favorito tiene patrón de pérdida histórica
        contra este rival específico, incluso cuando está en estado HOT.

        REGLA-T19-2: n_h2h < 3 → immunity_factor = 1.00 (muestra insuficiente)
        REGLA-T19-1: Solo actúa cuando estado_fav == 'HOT'
          HOT + h2h_win_rate < 0.30 → 0.85 (señal de 2do orden negativa)
          HOT + h2h_win_rate > 0.70 → 1.12 (doble confirmación)
          Otros → 1.00

        Retorna: {'h2h_win_rate': float, 'immunity_factor': float, 'n_h2h': int}
        """
        total = len(direct_h2h_matches) if direct_h2h_matches else 0
        if total < 3:
            return {'h2h_win_rate': 0.5, 'immunity_factor': 1.00, 'n_h2h': total}

        wins_fav = sum(1 for m in direct_h2h_matches if m.get('ganador') == favored)
        h2h_win_rate = wins_fav / total

        if estado_fav == 'HOT':
            if h2h_win_rate < 0.30:
                immunity_factor = 0.85
            elif h2h_win_rate > 0.70:
                immunity_factor = 1.12
            else:
                immunity_factor = 1.00
        else:
            immunity_factor = 1.00

        return {
            'h2h_win_rate': round(h2h_win_rate, 3),
            'immunity_factor': immunity_factor,
            'n_h2h': total,
        }

    def classify_tournament(self, tournament_name):
        """
        Categoriza el torneo para ajustar los pesos del análisis.
        T21-01: Delega a detectar_tier() — fuente única de verdad en config.py.
        Retorna: 'grand_slam' | 'atp1000' | 'atp500' | 'challenger' | 'itf'
        """
        from config import detectar_tier
        return detectar_tier(tournament_name or '')

    def generate_advanced_prediction(self, player1_info, player2_info, p1_rivalry_score, p2_rivalry_score, player1_name, player2_name, player1_history, player2_history, player1_advantages_count, player2_advantages_count, player1_form, player2_form, direct_h2h_matches, tournament_name, prediction_context, p1_elo, p2_elo, optimized_weights=None, n_common_opponents=0, n_erdos_paths=0):
        """🎯 Generar predicción avanzada aplicando la fórmula de Peso Final - VERSIÓN REBALANCEADA Y NORMALIZADA."""
        
        reasoning = []

        # Extraer ranks y ELOs
        rank1 = player1_info.get('ranking_position') if player1_info else None
        rank2 = player2_info.get('ranking_position') if player2_info else None
        elo1 = p1_elo
        elo2 = p2_elo
        reasoning.append(f"LOG_ELO_RATINGS: {player1_name}={elo1}, {player2_name}={elo2}")

        # 1. AJUSTE DE PESOS
        if optimized_weights:
            weights = optimized_weights
            reasoning.append(f"LOG_WEIGHTS_OPTIMIZED: {weights}")
        else:
            tournament_category = self.classify_tournament(tournament_name)
            
            # T21-03: Pesos diferenciados por tier (Nodo-21) — SNR por estructura de mercado
            # grand_slam: H2H denso + red Erdős densa → h2h/common_opp altos, form bajo
            # challenger:  red fragmentada, H2H escaso → form/ranking altos, h2h/common_opp bajos
            # itf:         sin red, sin H2H → form_recent dominante
            weights_config = {
                'grand_slam': {
                    'surface_specialization': 0.15, 'form_recent': 0.12, 'common_opponents': 0.22,
                    'h2h_direct': 0.18, 'ranking_momentum': 0.15, 'elo_rating': 0.13, 'home_advantage': 0.05, 'strength_of_schedule': 0.00
                },
                'atp1000': {
                    'surface_specialization': 0.16, 'form_recent': 0.15, 'common_opponents': 0.20,
                    'h2h_direct': 0.14, 'ranking_momentum': 0.17, 'elo_rating': 0.13, 'home_advantage': 0.05, 'strength_of_schedule': 0.00
                },
                'atp500': {
                    'surface_specialization': 0.15, 'form_recent': 0.18, 'common_opponents': 0.15,
                    'h2h_direct': 0.10, 'ranking_momentum': 0.20, 'elo_rating': 0.12, 'home_advantage': 0.05, 'strength_of_schedule': 0.05
                },
                'challenger': {
                    'surface_specialization': 0.20, 'form_recent': 0.22, 'common_opponents': 0.08,
                    'h2h_direct': 0.03, 'ranking_momentum': 0.22, 'elo_rating': 0.15, 'home_advantage': 0.05, 'strength_of_schedule': 0.05
                },
                'itf': {
                    'surface_specialization': 0.15, 'form_recent': 0.28, 'common_opponents': 0.05,
                    'h2h_direct': 0.02, 'ranking_momentum': 0.22, 'elo_rating': 0.15, 'home_advantage': 0.08, 'strength_of_schedule': 0.05
                },
            }
            weights = dict(weights_config.get(tournament_category, weights_config['atp500']))
            reasoning.append(f"LOG_WEIGHTS_STRATEGY: '{tournament_category}' -> {weights}")

            # T21-07 — James-Stein shrinkage (Nodo-21 Fase 2)
            # REGLA-T21-4: pesos nunca ciegos — shrink hacia default cuando n_tier < 20.
            # n_tier leído de calibracion_edge.json por superficie×tier.
            try:
                import json, os as _os
                _cal_path = _os.path.join(_os.path.dirname(__file__), '..', 'data', 'calibracion_edge.json')
                with open(_cal_path) as _f:
                    _cal = json.load(_f)
                _surf_key = prediction_context.get('current_match_surface', 'unknown') or 'unknown'
                _key = f"{_surf_key}_{tournament_category}"
                _tier_state = _cal.get('por_superficie_y_tier', {}).get(_key, {'wins': 0, 'losses': 0})
                _n_tier = _tier_state['wins'] + _tier_state['losses']
                _default_w = weights_config['atp500']
                weights = shrink_weights(weights, _default_w, _n_tier)
                reasoning.append(
                    f"LOG_SHRINKAGE: key={_key} n={_n_tier} "
                    f"factor={round(_n_tier/(_n_tier+20), 3)} weights={weights}"
                )
            except Exception as _shrink_err:
                reasoning.append(f"LOG_SHRINKAGE_SKIP: {_shrink_err}")

            # T21-06 — Density confidence (Nodo-21 Fase 2)
            # REGLA-T21-3: densidad local > categoría discreta.
            # Modula peso common_opponents según densidad real del grafo.
            # El peso sobrante se redistribuye a form_recent (no depende de la red).
            _density = density_confidence(n_common_opponents, n_erdos_paths)
            _w_co_original = weights['common_opponents']
            weights['common_opponents'] = round(_w_co_original * _density, 4)
            weights['form_recent'] = round(weights['form_recent'] + _w_co_original * (1 - _density), 4)
            reasoning.append(
                f"LOG_DENSITY: n_common={n_common_opponents} n_paths={n_erdos_paths} "
                f"density={_density} co_w: {_w_co_original}→{weights['common_opponents']} "
                f"form_w→{weights['form_recent']}"
            )

            # T14-03 — Ajuste de pesos por superficie (Nodo-14, Conexión 3)
            # Alpha validado: ranking es menos predictivo en arcilla lenta (Parry @ 4.50 ganó).
            # Erdős transitivity más fiable en clay → subir common_opponents.
            # En grass más varianza → bajar common_opponents, subir form_recent.
            _surf_adj = prediction_context.get('current_match_surface', '')
            if _surf_adj == 'clay':
                weights['common_opponents'] = round(weights['common_opponents'] + 0.08, 4)  # D56-04
                weights['ranking_momentum']  = round(weights['ranking_momentum']  - 0.08, 4)  # D56-04
                reasoning.append(
                    f"LOG_WEIGHTS_SURFACE_CLAY: common_opp→{weights['common_opponents']} "
                    f"ranking_mom→{weights['ranking_momentum']} (Erdős+clay, Nodo-14)"
                )
            elif _surf_adj == 'grass':
                weights['common_opponents'] = round(weights['common_opponents'] - 0.05, 4)  # D56-04
                weights['form_recent']       = round(weights['form_recent']       + 0.05, 4)  # D56-04
                reasoning.append(
                    f"LOG_WEIGHTS_SURFACE_GRASS: common_opp→{weights['common_opponents']} "
                    f"form_recent→{weights['form_recent']} (alta varianza césped, Nodo-14)"
                )

        # --- OBTENER MULTIPLICADORES Y SCORES BASE ---

        # NUEVO: ANÁLISIS DE ESPECIALIZACIÓN POR SUPERFICIE
        current_surface = prediction_context.get('current_match_surface')
        p1_surface_result, p1_surface_log = self.analyze_surface_specialization(player1_history, current_surface, player1_name)
        p2_surface_result, p2_surface_log = self.analyze_surface_specialization(player2_history, current_surface, player2_name)
        p1_surface_score = p1_surface_result['score']
        p2_surface_score = p2_surface_result['score']
        if p1_surface_log: reasoning.extend([f"P1_LOG_SURF: {log}" for log in p1_surface_log])
        if p2_surface_log: reasoning.extend([f"P2_LOG_SURF: {log}" for log in p2_surface_log])

        # E-1: Dynamic surface weight boost when TORNEO_COMPLETO detected
        # A tournament champion on THIS surface deserves more weight on surface
        _any_torneo = p1_surface_result.get('torneo_completo') or p2_surface_result.get('torneo_completo')
        if _any_torneo and weights.get('form_recent', 0) > 0.10:
            _boost = 0.07
            weights['surface_specialization'] = round(weights.get('surface_specialization', 0.15) + _boost, 4)
            weights['form_recent'] = round(weights['form_recent'] - _boost, 4)
            reasoning.append(
                f"LOG_E1_TORNEO_WEIGHT: surface {weights['surface_specialization']-_boost:.2f}→{weights['surface_specialization']:.2f} "
                f"form {weights['form_recent']+_boost:.2f}→{weights['form_recent']:.2f} "
                f"(tournament champion on this surface)"
            )

        p1_streak_mult, p1_streak_log = self.analyze_streaks_and_consistency(player1_history, player1_name)
        p2_streak_mult, p2_streak_log = self.analyze_streaks_and_consistency(player2_history, player2_name)
        if p1_streak_log: reasoning.extend([f"P1_LOG_MOM: {log}" for log in p1_streak_log])
        if p2_streak_log: reasoning.extend([f"P2_LOG_MOM: {log}" for log in p2_streak_log])

        p1_quality_mult, p1_diversity_mult, p1_adv_log = self.analyze_advanced_player_metrics(player1_history, player1_name)
        p2_quality_mult, p2_diversity_mult, p2_adv_log = self.analyze_advanced_player_metrics(player2_history, player2_name)
        if p1_adv_log: reasoning.extend([f"P1_LOG_DIV: {log}" for log in p1_adv_log])
        if p2_adv_log: reasoning.extend([f"P2_LOG_DIV: {log}" for log in p2_adv_log])

        p1_schedule_score, p1_schedule_log = self.analyze_strength_of_schedule(player1_history, player1_name)
        p2_schedule_score, p2_schedule_log = self.analyze_strength_of_schedule(player2_history, player2_name)
        if p1_schedule_log: reasoning.extend([f"P1_LOG_SoS: {log}" for log in p1_schedule_log])
        if p2_schedule_log: reasoning.extend([f"P2_LOG_SoS: {log}" for log in p2_schedule_log])

        dominance_multiplier_p1 = 1.5 if player1_advantages_count >= 3 else 1.0
        dominance_multiplier_p2 = 1.5 if player2_advantages_count >= 3 else 1.0

        p1_h2h_score, p2_h2h_score, h2h_log = self.analyze_direct_h2h(direct_h2h_matches, player1_name, player2_name)
        if h2h_log: reasoning.extend(h2h_log)

        # --- CÁLCULO DE COMPONENTES RAW (CON LÍMITES) ---
        
        def calculate_raw_scores(player_info, elo, streak_mult, quality_mult, diversity_mult, rivalry_score, dominance_mult, form, h2h_score, schedule_score, player_context):
            raw_scores = {}
            rank = player_info.get('ranking_position') if player_info else None

            # 0a. VENTAJA DE LOCAL (Límite: 100)
            home_advantage_score = 0
            player_nat = player_context.get('nationality', 'N/A')
            match_country = player_context.get('current_match_country', 'N/A')
            if player_nat and player_nat != 'N/A' and match_country and match_country != 'N/A':
                if player_nat.strip().lower() in match_country.strip().lower() or match_country.strip().lower() in player_nat.strip().lower():
                    home_advantage_score += 50
                    reasoning.append(f"LOG_CONTEXT_{player_context['player_name']}: 🏠 Home Advantage Bonus (+50 pts)")
            raw_scores['home_advantage'] = min(home_advantage_score, 100)

            # 0b. ESPECIALIZACIÓN EN SUPERFICIE (Límite: 350)
            surface_quality_score = player_context.get('surface_quality_score', 0)
            raw_scores['surface_specialization'] = min(surface_quality_score, 350)

            # 1. Ranking & Momentum (Límite: 450) - LÓGICA MEJORADA CON DATOS REALES DE FLASHSCORE
            ranking_momentum_score = 0
            player_metrics = self.get_ranking_metrics(player_context['full_name'])

            if player_metrics:
                # A. Puntuación Base (Pts Totales): Calidad y consistencia histórica.
                base_score = math.log1p(player_metrics['pts']) * 20

                # B. Momentum Asegurado (Already Secured): Rendimiento inmediato en el torneo.
                momentum_bonus = math.log1p(player_metrics['already_secured']) * 15

                # C. Potencial y Motivación (Improvement Potential): Techo y motivación.
                potential_bonus = math.log1p(player_metrics['improvement_potential']) * 10

                # D. Índice de Presión (Pressure Index): Factor psicológico.
                # Un índice negativo (ya defendió más de lo necesario) es bueno.
                # Un índice positivo alto (mucha presión) es malo.
                pressure_factor = 0
                if player_metrics['pressure_index'] < 0:
                    pressure_factor = math.log1p(abs(player_metrics['pressure_index'])) * 5 # Bonus por jugar suelto
                else:
                    pressure_factor = -math.log1p(player_metrics['pressure_index']) * 5 # Penalización por presión

                reasoning.append(   
                    f"LOG_RANKING_{player_context['player_name']}: "
                    f"Base({player_metrics['pts']}): {base_score:.2f}, "
                    f"Momentum({player_metrics['already_secured']}): {momentum_bonus:.2f}, "
                    f"Potencial({player_metrics['improvement_potential']}): {potential_bonus:.2f}, "
                    f"Presión({player_metrics['pressure_index']}): {pressure_factor:.2f}"
                )
                
                ranking_momentum_score = base_score + momentum_bonus + potential_bonus + pressure_factor
            
            raw_scores['ranking_momentum'] = min(ranking_momentum_score, 450)

            # 2. Forma Reciente (Límite: 300)
            form_score = 0
            if form and form.get('win_percentage') is not None:
                win_pct = form['win_percentage']
                form_score = (win_pct / 100) * 200  # Puntuación base
                
                if form.get('current_streak_type') == 'victorias':
                    form_score += form.get('current_streak_count', 0) * 15
            raw_scores['form_recent'] = min(form_score * streak_mult, 300)

            # 3. Rivales Comunes (Límite: 400)
            common_opp_score = rivalry_score * quality_mult * dominance_mult
            raw_scores['common_opponents'] = min(common_opp_score, 400)

            # 4. H2H Directo (Límite: 350)
            raw_scores['h2h_direct'] = min(h2h_score * 100, 350)

            # 5. Rating ELO (Límite: 250)
            # floor=1500, cap=250 → rango útil [1500, 1750]
            # B-11 fix: cap explícito como los 7 componentes hermanos
            raw_scores['elo_rating'] = _compute_raw_elo(elo)

            # 6. Strength of Schedule (Límite: 200)
            raw_scores['strength_of_schedule'] = min(schedule_score * diversity_mult, 200)

            # Calcular días desde el último partido
            days_since = -1  # Valor por defecto si no hay datos
            if form and form.get('last_match_date') and form['last_match_date'] != 'N/A':
                try:
                    last_match_date = datetime.strptime(form['last_match_date'], '%d.%m.%Y')
                    days_since = (datetime.now() - last_match_date).days
                except ValueError:
                    days_since = -1

            return raw_scores, days_since

        p1_context = {
            'player_name': 'P1',
            'full_name': player1_name,
            'nationality': prediction_context['p1_nationality'],
            'current_match_country': prediction_context['current_match_country'],
            'surface_quality_score': p1_surface_score
        }
        p2_context = {
            'player_name': 'P2',
            'full_name': player2_name,
            'nationality': prediction_context['p2_nationality'],
            'current_match_country': prediction_context['current_match_country'],
            'surface_quality_score': p2_surface_score
        }

        raw_p1, days_since_p1 = calculate_raw_scores(player1_info, elo1, p1_streak_mult, p1_quality_mult, p1_diversity_mult, p1_rivalry_score, dominance_multiplier_p1, player1_form, p1_h2h_score, p1_schedule_score, p1_context)
        raw_p2, days_since_p2 = calculate_raw_scores(player2_info, elo2, p2_streak_mult, p2_quality_mult, p2_diversity_mult, p2_rivalry_score, dominance_multiplier_p2, player2_form, p2_h2h_score, p2_schedule_score, p2_context)

        # --- MARKOV / PELT CHANGE-POINT (Nodo-02) ---
        # Nodo-32 Fase 3: inicializados a 1.0 para que post-norm los encuentre
        # incluso si el bloque Markov falla por datos insuficientes.
        factor_p1 = 1.0
        factor_p2 = 1.0
        markov_analysis = None
        try:
            from analysis.markov_analyzer import (
                detectar_cambio_regimen, calcular_factor_markov, extraer_resultados_binarios,
                _normalize_surface, _surface_overlap_rate, apply_surface_context_discount,
            )
            resultados_p1 = extraer_resultados_binarios(player1_history, player1_name, n=20)
            resultados_p2 = extraer_resultados_binarios(player2_history, player2_name, n=20)

            markov_p1 = detectar_cambio_regimen(resultados_p1)
            markov_p2 = detectar_cambio_regimen(resultados_p2)

            # Factor simétrico: P1 y P2 se comparan entre sí
            factor_p1 = calcular_factor_markov(markov_p1, markov_p2)
            factor_p2 = calcular_factor_markov(markov_p2, markov_p1)

            # --- H2H IMMUNITY DAMPENER (T19-02, Nodo-19) ---
            # Cruza el estado HOT con el H2H histórico vs este rival específico.
            # HOT pero pierde históricamente a ESTE rival → reducir factor_markov.
            # HOT y domina históricamente → amplificar (doble confirmación).
            immunity_p1 = self.calcular_h2h_immunity(
                direct_h2h_matches, player1_name, markov_p1['estado_actual']
            )
            immunity_p2 = self.calcular_h2h_immunity(
                direct_h2h_matches, player2_name, markov_p2['estado_actual']
            )
            factor_p1 = round(factor_p1 * immunity_p1['immunity_factor'], 4)
            factor_p2 = round(factor_p2 * immunity_p2['immunity_factor'], 4)

            # --- F4 (Nodo-46): Surface Context Discount ---
            # Descuenta factor_markov cuando la racha reciente es de otra superficie.
            # current_match_context['apply_surface_discount']=False activa --no-surface-discount A/B.
            current_surface = _normalize_surface(
                prediction_context.get('current_match_surface', '')
                or prediction_context.get('tournament_context', {}).get('superficie', '')
            )
            season_transition = bool(
                prediction_context.get('tournament_context', {}).get('season_transition_flag', False)
            )
            apply_sd = prediction_context.get('apply_surface_discount', True)

            overlap_p1 = _surface_overlap_rate(player1_history, current_surface)
            overlap_p2 = _surface_overlap_rate(player2_history, current_surface)

            factor_p1, confianza_markov_p1, sd_p1 = apply_surface_context_discount(
                factor_p1, markov_p1['confianza'], overlap_p1,
                markov_p1['estado_actual'], season_transition, apply_sd,
            )
            factor_p2, confianza_markov_p2, sd_p2 = apply_surface_context_discount(
                factor_p2, markov_p2['confianza'], overlap_p2,
                markov_p2['estado_actual'], season_transition, apply_sd,
            )

            # Nodo-32 Fase 3: factor_p1/p2 se aplican POST-normalizacion (ver abajo).
            # PRE-norm causaba compresión por log1p → delta ~0.072 (ruido imperceptible).
            reasoning.append(
                f"LOG_MARKOV_P1: estado={markov_p1['estado_actual']} "
                f"momentum={markov_p1['momentum']} factor={factor_p1} "
                f"wr_rec={markov_p1['win_rate_reciente']} cp={markov_p1['change_point']} "
                f"immunity={immunity_p1['immunity_factor']} h2h_wr={immunity_p1['h2h_win_rate']} "
                f"surf_overlap={overlap_p1} surf_discount={sd_p1}"
            )
            reasoning.append(
                f"LOG_MARKOV_P2: estado={markov_p2['estado_actual']} "
                f"momentum={markov_p2['momentum']} factor={factor_p2} "
                f"wr_rec={markov_p2['win_rate_reciente']} cp={markov_p2['change_point']} "
                f"immunity={immunity_p2['immunity_factor']} h2h_wr={immunity_p2['h2h_win_rate']} "
                f"surf_overlap={overlap_p2} surf_discount={sd_p2}"
            )

            markov_analysis = {
                'jugador1':              markov_p1,
                'jugador2':              markov_p2,
                'factor_markov':         factor_p1,          # perspectiva P1 vs P2
                'h2h_immunity_p1':       immunity_p1,        # T19-03
                'h2h_immunity_p2':       immunity_p2,
                # D46-06: campos de procedencia de superficie (solo registrar, no descontar tier)
                'surface_overlap_rate_p1': round(overlap_p1, 3),
                'surface_overlap_rate_p2': round(overlap_p2, 3),
                'surface_discount_p1':     round(sd_p1, 4),
                'surface_discount_p2':     round(sd_p2, 4),
                'current_surface':         current_surface,
            }
        except Exception as _markov_err:
            reasoning.append(f"LOG_MARKOV_ERROR: {_markov_err}")

        # --- FACTOR TARDÍO (T14-02) — win rate en partidos de 4to/5to set ---
        factor_tardio = 1.0  # Nodo-32 Fase 3: inicializado para post-norm
        tardio_analysis = None
        try:
            from analysis.markov_analyzer import (
                calcular_factor_tardio, calcular_factor_tardio_comparativo
            )
            tardio_p1 = calcular_factor_tardio(player1_history, min_matches=3)
            tardio_p2 = calcular_factor_tardio(player2_history, min_matches=3)

            factor_tardio = calcular_factor_tardio_comparativo(tardio_p1, tardio_p2)

            if factor_tardio != 1.0:
                # Nodo-32 Fase 3: aplicación movida a POST-norm (ver abajo)
                reasoning.append(
                    f"LOG_TARDIO: factor={factor_tardio} "
                    f"wr_tardio_p1={tardio_p1['win_rate_tardio'] if tardio_p1 else 'N/A'} "
                    f"n_p1={tardio_p1['n_partidos_extendidos'] if tardio_p1 else 0} "
                    f"wr_tardio_p2={tardio_p2['win_rate_tardio'] if tardio_p2 else 'N/A'} "
                    f"n_p2={tardio_p2['n_partidos_extendidos'] if tardio_p2 else 0}"
                )

            tardio_analysis = {
                'jugador1':       tardio_p1,
                'jugador2':       tardio_p2,
                'factor_tardio':  factor_tardio,
            }
        except Exception as _tardio_err:
            reasoning.append(f"LOG_TARDIO_ERROR: {_tardio_err}")

        # --- LÓGICA DE PONDERACIÓN DINÁMICA (H2H Antiguo vs. Rivales Comunes) ---
        try:
            if direct_h2h_matches:
                last_match_date = datetime.strptime(direct_h2h_matches[0]['fecha'], '%d.%m.%Y')
                last_match_days_ago = (datetime.now() - last_match_date).days
                H2H_ANTIQUITY_THRESHOLD = 730 # 2 años

                if last_match_days_ago > H2H_ANTIQUITY_THRESHOLD:
                    h2h_winner_is_p1 = raw_p1['h2h_direct'] > raw_p2['h2h_direct']
                    common_opp_winner_is_p1 = raw_p1['common_opponents'] > raw_p2['common_opponents']

                    # Comprobar si hay contradicción
                    if h2h_winner_is_p1 != common_opp_winner_is_p1:
                        # Determinar quién tiene la ventaja en rivales comunes y por cuánto
                        if common_opp_winner_is_p1:
                            advantage_player_raw_common = raw_p1['common_opponents']
                            disadvantage_player_raw_common = raw_p2['common_opponents']
                        else:
                            advantage_player_raw_common = raw_p2['common_opponents']
                            disadvantage_player_raw_common = raw_p1['common_opponents']

                        # Solo aplicar si la ventaja en rivales comunes es significativa
                        if advantage_player_raw_common > disadvantage_player_raw_common * 1.25: # 25% de ventaja
                            reasoning.append(f"LOG_DYNAMIC_WEIGHTING: H2H antiguo ({last_match_days_ago} días) contradicho por una ventaja significativa en Rivales Comunes.")
                            
                            if common_opp_winner_is_p1:
                                # P1 tiene ventaja en rivales, P2 en H2H antiguo
                                reasoning.append("   -> Bonificando Rivales Comunes de P1 y reduciendo H2H de P2.")
                                raw_p1['common_opponents'] *= 1.30 # Bonificación del 30%
                                raw_p2['h2h_direct'] *= 0.15       # Reducción al 15%
                            else:
                                # P2 tiene ventaja en rivales, P1 en H2H antiguo
                                reasoning.append("   -> Bonificando Rivales Comunes de P2 y reduciendo H2H de P1.")
                                raw_p2['common_opponents'] *= 1.30 # Bonificación del 30%
                                raw_p1['h2h_direct'] *= 0.15       # Reducción al 15%
        except Exception as e:
            reasoning.append(f"LOG_DYNAMIC_WEIGHTING_ERROR: {e}")

        # --- CIRCUIT ASYMMETRY DEFLATOR (Nodo-29) ---
        # Detecta asimetría de circuito competitivo y deflacta form/ELO
        # del jugador de circuito inferior para corregir sesgo de circuito.
        # REGLA-N29-5: después de dynamic weighting, antes de normalización.
        circuit_asymmetry = {
            'p1_circuit_tier_index': 0.0,
            'p2_circuit_tier_index': 0.0,
            'asymmetry_ratio': 1.0,
            'deflactor_applied': 1.0,
            'player_deflated': None,
            'signal': 'SYMMETRIC',
        }
        try:
            cti_p1, n_rank_p1 = self.circuit_tier_index(player1_history)
            cti_p2, n_rank_p2 = self.circuit_tier_index(player2_history)
            circuit_asymmetry['p1_circuit_tier_index'] = cti_p1
            circuit_asymmetry['p2_circuit_tier_index'] = cti_p2

            n_min = min(n_rank_p1, n_rank_p2)
            if n_min < 10:
                # REGLA-N29-4: muestra insuficiente → no aplicar
                reasoning.append(
                    f"LOG_CAD_SKIP: n_min={n_min} < 10 (CTI_P1={cti_p1} CTI_P2={cti_p2})"
                )
            else:
                cti_max = max(cti_p1, cti_p2)
                cti_min = max(min(cti_p1, cti_p2), 0.1)
                asimetria = cti_max / cti_min
                circuit_asymmetry['asymmetry_ratio'] = round(asimetria, 2)

                if asimetria >= 5.0:
                    circuit_asymmetry['signal'] = 'STRONG_ASYMMETRY'
                elif asimetria >= 2.0:
                    circuit_asymmetry['signal'] = 'MODERATE_ASYMMETRY'

                # REGLA-N29-1b: el jugador "superior" debe tener CTI ≥ 0.8
                # Esto filtra falsos positivos donde ambos son ITF pero uno
                # tuvo oponentes levemente mejor rankeados (ej: Filip vs Kelm).
                # CTI ≥ 0.8 implica presencia real en circuito Challenger/ATP.
                cad_guardada = cti_max >= 0.8

                if asimetria > 2.0 and cad_guardada:
                    # REGLA-N29-1: solo aplica cuando asimetría > 2.0 y CTI_max ≥ 0.8
                    deflactor = 1.0 / (1.0 + 0.15 * math.log(asimetria))
                    bonificacion = 1.0 + (1.0 - deflactor) * 0.5
                    circuit_asymmetry['deflactor_applied'] = round(deflactor, 4)

                    # Identificar jugador de circuito inferior
                    if cti_p1 < cti_p2:
                        # P1 es el de circuito inferior
                        circuit_asymmetry['player_deflated'] = player1_name
                        raw_p1['form_recent'] = min(raw_p1['form_recent'] * deflactor, 300)
                        raw_p1['elo_rating']  = min(raw_p1['elo_rating']  * deflactor, 250)
                        raw_p2['form_recent'] = min(raw_p2['form_recent'] * bonificacion, 300)
                        raw_p2['elo_rating']  = min(raw_p2['elo_rating']  * bonificacion, 250)
                    else:
                        # P2 es el de circuito inferior
                        circuit_asymmetry['player_deflated'] = player2_name
                        raw_p2['form_recent'] = min(raw_p2['form_recent'] * deflactor, 300)
                        raw_p2['elo_rating']  = min(raw_p2['elo_rating']  * deflactor, 250)
                        raw_p1['form_recent'] = min(raw_p1['form_recent'] * bonificacion, 300)
                        raw_p1['elo_rating']  = min(raw_p1['elo_rating']  * bonificacion, 250)

                    # Fase 3: SoS weight dinámico (toma de form_recent)
                    # REGLA-N29-1: solo en tier donde SoS tiene peso base > 0
                    base_sos_w = weights.get('strength_of_schedule', 0.0)
                    if base_sos_w > 0:
                        sos_multiplier = 1.0 + math.log(asimetria)
                        extra_w = round(base_sos_w * (sos_multiplier - 1.0), 4)
                        extra_w = min(extra_w, weights.get('form_recent', 0.0) * 0.5)
                        weights['strength_of_schedule'] = round(weights['strength_of_schedule'] + extra_w, 4)
                        weights['form_recent'] = round(weights['form_recent'] - extra_w, 4)

                    reasoning.append(
                        f"LOG_CAD: CTI_P1={cti_p1} CTI_P2={cti_p2} "
                        f"asimetria={round(asimetria,2)} deflactor={round(deflactor,4)} "
                        f"bonificacion={round(bonificacion,4)} "
                        f"jugador_deflactado={circuit_asymmetry['player_deflated']} "
                        f"signal={circuit_asymmetry['signal']} "
                        f"sos_w→{weights.get('strength_of_schedule',0):.4f} "
                        f"form_w→{weights.get('form_recent',0):.4f}"
                    )
                elif asimetria > 2.0 and not cad_guardada:
                    reasoning.append(
                        f"LOG_CAD: CTI_P1={cti_p1} CTI_P2={cti_p2} "
                        f"asimetria={round(asimetria,2)} → SKIP (CTI_max={cti_max:.3f} < 0.8, ambos ITF)"
                    )
                else:
                    reasoning.append(
                        f"LOG_CAD: CTI_P1={cti_p1} CTI_P2={cti_p2} "
                        f"asimetria={round(asimetria,2)} → SYMMETRIC (sin deflactor)"
                    )
        except Exception as _cad_err:
            reasoning.append(f"LOG_CAD_ERROR: {_cad_err}")

        # 2. NORMALIZACIÓN DE SCORES — usa normalize_scores() de nivel módulo (Nodo-53 Paso 1)
        # surface_specialization usa normalización LINEAL porque SkillFactor/VolConf
        # (Nodo-28 Fase 1.5) ya controlan la escala. log1p aplasta la señal:
        # raw 86 vs 142 → log 4.47 vs 4.97 (ratio 1.11x vs 1.65x real).
        norm_p1, norm_p2 = normalize_scores(raw_p1, raw_p2)

        # --- Nodo-32 Fase 3: factores Markov + tardío POST-normalizacion (T32-20..T32-27) ---
        # PRE-norm (anterior): log1p(200*1.15)=5.375 vs log1p(200)=5.303 → delta=0.072 (ruido)
        # POST-norm (ahora):   log1p(200)*1.15=6.098 vs log1p(200)=5.303 → delta=0.795 (10x)
        # Cap equivalente al raw cap=300: log1p(300)=5.707
        _norm_cap = math.log1p(300)
        if factor_p1 != 1.0:
            norm_p1['form_recent'] = min(norm_p1['form_recent'] * factor_p1, _norm_cap)
        if factor_p2 != 1.0:
            norm_p2['form_recent'] = min(norm_p2['form_recent'] * factor_p2, _norm_cap)
        if factor_tardio != 1.0:
            norm_p1['form_recent'] = min(norm_p1['form_recent'] * factor_tardio, _norm_cap)
        reasoning.append(
            f"LOG_MARKOV_POST_NORM: factor_p1={factor_p1} factor_p2={factor_p2} "
            f"factor_tardio={factor_tardio} cap={_norm_cap:.3f} "
            f"norm_form_p1={norm_p1['form_recent']:.3f} norm_form_p2={norm_p2['form_recent']:.3f}"
        )

        # --- Nodo-57 D57-01: Decaimiento exponencial de forma por inactividad ---
        # Solo form_recent decae. ELO, H2H, rivales comunes NO expiran en 30-60 días.
        # La respuesta bayesiana correcta: encoger form hacia prior, no penalizar el total.
        def _form_decay_factor(days):
            if days == -1: return 0.70           # fecha desconocida: decay moderado fijo
            if days <= _FORM_GRACE_DAYS: return 1.0
            return max(_FORM_DECAY_FLOOR, math.exp(-_FORM_DECAY_LAMBDA * (days - _FORM_GRACE_DAYS)))

        _fd_p1 = _form_decay_factor(days_since_p1)
        _fd_p2 = _form_decay_factor(days_since_p2)
        if _fd_p1 < 1.0:
            norm_p1 = dict(norm_p1)
            norm_p1['form_recent'] = norm_p1['form_recent'] * _fd_p1
        if _fd_p2 < 1.0:
            norm_p2 = dict(norm_p2)
            norm_p2['form_recent'] = norm_p2['form_recent'] * _fd_p2
        reasoning.append(
            f"LOG_FORM_DECAY: p1_days={days_since_p1} fd_p1={_fd_p1:.3f} "
            f"p2_days={days_since_p2} fd_p2={_fd_p2:.3f}"
        )

        reasoning.append(f"LOG_RAW_SCORES_P1: { {k: round(v, 2) for k, v in raw_p1.items()} }")
        reasoning.append(f"LOG_RAW_SCORES_P2: { {k: round(v, 2) for k, v in raw_p2.items()} }")
        reasoning.append(f"LOG_NORM_SCORES_P1: { {k: round(v, 2) for k, v in norm_p1.items()} }")
        reasoning.append(f"LOG_NORM_SCORES_P2: { {k: round(v, 2) for k, v in norm_p2.items()} }")

        # --- APLICAR PESOS Y CALCULAR PUNTAJE FINAL ---
        # Nodo-57 D57-02: penalty eliminada del total — inactividad ya manejada vía form_decay
        def apply_weights_and_penalties(normalized_scores, weights, days_since):
            weighted_scores = {k: normalized_scores[k] * weights[k] for k in weights}
            final_score = sum(weighted_scores.values())
            return final_score, weighted_scores, 0.0  # penalty=0: inactividad vía form_decay

        final_p1, weighted_p1, penalty_p1 = apply_weights_and_penalties(norm_p1, weights, days_since_p1)
        final_p2, weighted_p2, penalty_p2 = apply_weights_and_penalties(norm_p2, weights, days_since_p2)

        # --- GENERAR DESGLOSE DETALLADO PARA LOGS ---
        def get_breakdown(raw, norm, weighted, penalty, final_score):
            breakdown = {}
            total_weighted = sum(weighted.values())
            if total_weighted > 0:
                for k, v in weighted.items():
                    breakdown[k] = {
                        'raw_score': f"{raw[k]:.1f}",
                        'normalized_score': f"{norm[k]:.2f}",
                        'weight': f"{weights[k]*100:.0f}%",
                        'weighted_score': f"{v:.2f}",
                        'contribution': f"{(v / total_weighted) * 100:.1f}%" if total_weighted > 0 else "0.0%"
                    }
            breakdown['Penalizacion_Inactividad'] = f"{-penalty:.2f} pts"
            breakdown['Puntaje_Final'] = f"{final_score:.2f}"
            return breakdown

        breakdown_p1 = get_breakdown(raw_p1, norm_p1, weighted_p1, penalty_p1, final_p1)
        breakdown_p2 = get_breakdown(raw_p2, norm_p2, weighted_p2, penalty_p2, final_p2)

        # --- DECISIÓN FINAL Y CONFIANZA ---
        score_diff = final_p1 - final_p2
        total_score = final_p1 + final_p2
        
        confidence = 50
        if total_score > 0:
            confidence = 50 + (abs(score_diff) / total_score * 50)

        if score_diff > 0.01: # Umbral mínimo para decidir un ganador
            favored = player1_name
        elif score_diff < -0.01:
            favored = player2_name
        else:
            if rank1 is not None and rank2 is not None:
                favored = player1_name if rank1 < rank2 else player2_name
                confidence = 51
            else:
                favored = 'Empate'
                confidence = 50

        return {
            'favored_player': favored,
            'confidence': round(min(confidence, 95.0), 1),
            'reasoning': reasoning,
            'scores': {
                'p1_final_weight': round(final_p1, 2),
                'p2_final_weight': round(final_p2, 2),
                'score_difference': round(score_diff, 2)
            },
            'score_breakdown': {
                'player1': breakdown_p1,
                'player2': breakdown_p2
            },
            'weights_used': weights,
            '_weights_final': dict(weights),  # D56-01: snapshot post-todas-las-modificaciones (shrinkage+density+surface)
            'markov_analysis': markov_analysis,
            'tardio_analysis': tardio_analysis,
            'circuit_asymmetry': circuit_asymmetry,  # Nodo-29
            'surface_specialization_meta': {          # FIX-1 (Nodo-28 Fase 2)
                'player1': p1_surface_result,
                'player2': p2_surface_result,
            },
            'historial_incompleto': {                 # Nodo-35: flag propagado desde extracción
                'p1': len(player1_history) == 0,
                'p2': len(player2_history) == 0,
            },
        }