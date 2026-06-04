"""
🎯 Módulo de Normalización Estandarizada para Predicciones de Tenis

Este módulo proporciona funciones de normalización consistentes que garantizan
coherencia matemática en todos los cálculos de predicción.

PROBLEMA ORIGINAL:
- Los factores de normalización variaban entre componentes (55.3 para ranking_momentum, 
  39.7 para form_recent, 64.6 para elo_rating)
- No existía documentación sobre cómo se calculaban los factores
- La fórmula de normalización no era reproducible

SOLUCIÓN:
- Normalización min-max con parámetros explícitos
- Documentación completa de cada fórmula
- Validaciones de coherencia integradas
"""

import math
import logging
from typing import Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES DE NORMALIZACIÓN
# ============================================================================

# Límites máximos teóricos para cada componente (usados en normalización)
MAX_RAW_SCORES = {
    'home_advantage': 100,
    'surface_specialization': 350,
    'ranking_momentum': 450,
    'form_recent': 300,
    'common_opponents': 400,
    'h2h_direct': 350,
    'elo_rating': 250,  # ELO normalizado: max(0, elo - 1500)
    'strength_of_schedule': 200
}

# T21-04: Pesos por tier — 5 niveles diferenciados por SNR de estructura de mercado
# Principio: peso ∝ signal-to-noise ratio de la señal en ese mercado
# grand_slam: H2H denso, red Erdős densa → h2h/common_opp altos
# challenger:  red fragmentada, <5% parejas con H2H → form/ranking_momentum altos
# itf:         sin red, primer enfrentamiento >99% → form_recent dominante
DEFAULT_WEIGHTS = {
    'grand_slam': {
        'surface_specialization': 0.15,
        'form_recent': 0.12,
        'common_opponents': 0.22,
        'h2h_direct': 0.18,
        'ranking_momentum': 0.15,
        'elo_rating': 0.13,
        'home_advantage': 0.05,
        'strength_of_schedule': 0.00
    },
    'atp1000': {
        'surface_specialization': 0.16,
        'form_recent': 0.15,
        'common_opponents': 0.20,
        'h2h_direct': 0.14,
        'ranking_momentum': 0.17,
        'elo_rating': 0.13,
        'home_advantage': 0.05,
        'strength_of_schedule': 0.00
    },
    'atp500': {
        'surface_specialization': 0.15,
        'form_recent': 0.18,
        'common_opponents': 0.15,
        'h2h_direct': 0.10,
        'ranking_momentum': 0.20,
        'elo_rating': 0.12,
        'home_advantage': 0.05,
        'strength_of_schedule': 0.05
    },
    'challenger': {
        'surface_specialization': 0.20,
        'form_recent': 0.22,
        'common_opponents': 0.08,
        'h2h_direct': 0.03,
        'ranking_momentum': 0.22,
        'elo_rating': 0.15,
        'home_advantage': 0.05,
        'strength_of_schedule': 0.05
    },
    'itf': {
        'surface_specialization': 0.15,
        'form_recent': 0.28,
        'common_opponents': 0.05,
        'h2h_direct': 0.02,
        'ranking_momentum': 0.22,
        'elo_rating': 0.15,
        'home_advantage': 0.08,
        'strength_of_schedule': 0.05
    },
    # Alias de compatibilidad (tests legacy que referencian 'atp_wta' o 'default')
    'atp_wta': {
        'surface_specialization': 0.15,
        'form_recent': 0.15,
        'common_opponents': 0.20,
        'h2h_direct': 0.15,
        'ranking_momentum': 0.20,
        'elo_rating': 0.10,
        'home_advantage': 0.05,
        'strength_of_schedule': 0.0
    },
    'default': {
        'surface_specialization': 0.15,
        'form_recent': 0.18,
        'common_opponents': 0.15,
        'h2h_direct': 0.10,
        'ranking_momentum': 0.20,
        'elo_rating': 0.12,
        'home_advantage': 0.05,
        'strength_of_schedule': 0.05
    }
}

# ============================================================================
# FUNCIONES DE NORMALIZACIÓN
# ============================================================================

def normalize_min_max(
    raw_score: float, 
    component_name: str,
    max_expected: Optional[float] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Normaliza un puntaje bruto usando el método min-max.
    
    Esta es la función de normalización PRINCIPAL que debe usarse en todo el sistema.
    
    Args:
        raw_score: Puntaje bruto a normalizar
        component_name: Nombre del componente (debe estar en MAX_RAW_SCORES)
        max_expected: Máximo esperado opcional (usa MAX_RAW_SCORES si no se proporciona)
    
    Returns:
        Tuple[float, Dict]: (score_normalizado, metadata_del_cálculo)
    
    Example:
        >>> score, meta = normalize_min_max(225, 'ranking_momentum')
        >>> # score será 0.5 (225 / 450)
        >>> # meta contendrá información del cálculo para logging
    
    FÓRMULA:
        normalized = raw_score / max_expected
    
        Donde max_expected es:
        - Si se proporciona: max_expected
        - Si no: MAX_RAW_SCORES.get(component_name, 100)
    
    NOTA: A diferencia de log1p, esta fórmula es completamente lineal y reproducible.
    """
    # Determinar el máximo esperado
    if max_expected is None:
        max_expected = MAX_RAW_SCORES.get(component_name, 100)
    
    # Evitar división por cero
    if max_expected <= 0:
        logger.warning(f"MAX_RAW_SCORES[{component_name}] <= 0, usando 1 como fallback")
        max_expected = 1
    
    # Normalización lineal (min-max simplificado, asumiendo mínimo de 0)
    normalized = raw_score / max_expected
    
    # Limitar al rango [0, 1] por seguridad
    normalized = max(0.0, min(1.0, normalized))
    
    metadata = {
        'raw_score': raw_score,
        'max_expected': max_expected,
        'normalization_factor': normalized,
        'component': component_name
    }
    
    return normalized, metadata


def normalize_with_log_scale(
    raw_score: float,
    component_name: str,
    scale_factor: float = 10.0
) -> Tuple[float, Dict[str, Any]]:
    """
    Normaliza usando escala logarítmica (para datos con amplio rango).
    
    Útil cuando los puntajes brutos pueden variar ampliamente y se desea
    comprimir los valores altos mientras se preserva la discriminación de valores bajos.
    
    Args:
        raw_score: Puntaje bruto
        component_name: Nombre del componente
        scale_factor: Factor de escala final (default 10 para puntajes 0-10)
    
    Returns:
        Tuple[float, Dict]: (score_normalizado, metadata)
    
    Example:
        >>> score, meta = normalize_with_log_scale(450, 'ranking_momentum')
        >>> # log(1+450) ≈ 6.11, escalado a ~6.11
    
    FÓRMULA:
        normalized = log(1 + raw_score) * (scale_factor / log(1 + max_expected))
    
    DODE max_expected es MAX_RAW_SCORES.get(component_name, 100)
    """
    max_expected = MAX_RAW_SCORES.get(component_name, 100)
    
    # Normalización logarítmica
    log_raw = math.log1p(raw_score)
    log_max = math.log1p(max_expected)
    
    normalized = (log_raw / log_max) * scale_factor
    
    # Limitar al rango [0, scale_factor]
    normalized = max(0.0, min(scale_factor, normalized))
    
    metadata = {
        'raw_score': raw_score,
        'max_expected': max_expected,
        'log_raw': log_raw,
        'log_max': log_max,
        'scale_factor': scale_factor,
        'normalization_factor': normalized,
        'component': component_name
    }
    
    return normalized, metadata


def normalize_elo(elo_rating: float) -> float:
    """
    Normaliza un rating ELO a una escala de 0-10.
    
    Args:
        elo_rating: Rating ELO del jugador (típicamente 1200-2400)
    
    Returns:
        float: ELO normalizado (0-10)
    
    FÓRMULA:
        normalized = (elo_rating - 1200) / (2400 - 1200) * 10
    
    RANGOS:
        ELO 1200 → 0.0
        ELO 1500 → 2.5
        ELO 1800 → 5.0
        ELO 2100 → 7.5
        ELO 2400 → 10.0
    
    Example:
        >>> normalize_elo(1885)
        5.71  # (1885 - 1200) / 1200 * 10
    """
    # Definir límites del ELO
    min_elo = 1200
    max_elo = 2400
    
    # Normalización lineal
    normalized = (elo_rating - min_elo) / (max_elo - min_elo) * 10
    
    # Limitar al rango [0, 10]
    return max(0.0, min(10.0, normalized))


def normalize_percentage(value: float, max_value: float = 100.0) -> float:
    """
    Normaliza un porcentaje o valor similar a escala 0-10.
    
    Args:
        value: Valor a normalizar
        max_value: Valor máximo esperado (default 100 para porcentajes)
    
    Returns:
        float: Valor normalizado a escala 0-10
    
    Example:
        >>> normalize_percentage(70, 100)
        7.0
        >>> normalize_percentage(85, 100)
        8.5
    """
    if max_value <= 0:
        return 0.0
    
    normalized = (value / max_value) * 10
    return max(0.0, min(10.0, normalized))


# ============================================================================
# SISTEMA DE PESOS DINÁMICOS
# ============================================================================

class WeightManager:
    """
    Gestor de pesos dinámicos con redistribución automática.
    
    RESPONSABILIDAD:
    - Mantener pesos base para cada tipo de torneo
    - Redistribuir pesos cuando hay componentes no disponibles
    - Validar coherencia de la suma de pesos
    
    PROBLEMA ORIGINAL:
    - Cuando un componente tenía valor 0 (ej: superficie desconocida),
      el peso simplemente se perdía, reduciendo el puntaje total artificialmente.
    
    SOLUCIÓN:
    - Redistribución proporcional de pesos entre componentes disponibles
    - Validación automática de la suma de pesos
    """
    
    def __init__(self, weights_config: Optional[Dict[str, float]] = None):
        """
        Inicializa el gestor de pesos.
        
        Args:
            weights_config: Diccionario de pesos {componente: peso} o None para usar defaults
        """
        self.base_weights = weights_config.copy() if weights_config else None
        self.adjusted_weights = {}
        self.redistribution_log = []
        
    def calculate_adjusted_weights(
        self,
        available_components: Dict[str, bool],
        weights_config: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calcula pesos ajustados redistribuyendo los pesos de componentes no disponibles.
        
        Args:
            available_components: {componente: está_disponible}
            weights_config: Pesos base a usar (usa self.base_weights si es None)
        
        Returns:
            Dict[str, float]: Pesos ajustados con suma = 1.0
        
        LÓGICA:
            1. Identificar componentes disponibles vs no disponibles
            2. Calcular suma de pesos no disponibles
            3. Redistribuir proporcionalmente entre disponibles
            4. Validar que la suma final sea 1.0
        
        Example:
            weights = {'surface': 0.15, 'form': 0.15, 'h2h': 0.15, 'ranking': 0.20}
            available = {'surface': True, 'form': True, 'h2h': False, 'ranking': True}
            
            Peso no disponible (h2h) = 0.15
            Redistribuir entre surface, form, ranking (suma = 0.85)
            factor = 1 / 0.85 = 1.176
            
            surface = 0.15 * 1.176 = 0.176
            form = 0.15 * 1.176 = 0.176
            ranking = 0.20 * 1.176 = 0.235
            Suma = 0.176 + 0.176 + 0.235 + 0 (h2h no disponible) = 0.587 ✗
            
            ERROR: La suma debe ser 1.0
            Corrección: Los pesos redistribuidos suman 0.587, pero necesitamos que
            los pesos disponibles sumen 1.0 (sin el peso de h2h)
            
            CORRECCIÓN:
            Suma pesos disponibles = 0.15 + 0.15 + 0.20 = 0.50
            Suma pesos no disponibles = 0.15
            factor = 1.0 / 0.50 = 2.0
            
            surface = 0.15 * 2.0 = 0.30
            form = 0.15 * 2.0 = 0.30
            ranking = 0.20 * 2.0 = 0.40
            Suma = 0.30 + 0.30 + 0.40 = 1.0 ✓
        """
        self.redistribution_log = []
        
        if weights_config is None:
            weights_config = self.base_weights
            if weights_config is None:
                raise ValueError("No se proporcionaron pesos y no hay pesos base configurados")
        
        # Identificar componentes disponibles y no disponibles
        available_weights_sum = 0.0
        unavailable_weights_sum = 0.0
        
        for component, weight in weights_config.items():
            if available_components.get(component, True):
                available_weights_sum += weight
            else:
                unavailable_weights_sum += weight
        
        self.redistribution_log.append(
            f"Pesos originales: suma={sum(weights_config.values()):.2f}, "
            f"disponibles={available_weights_sum:.2f}, "
            f"no disponibles={unavailable_weights_sum:.2f}"
        )
        
        # Calcular factor de redistribución
        if available_weights_sum == 0:
            self.redistribution_log.append("ERROR: No hay componentes disponibles")
            return {}
        
        redistribution_factor = 1.0 / available_weights_sum
        
        # Redistribuir pesos
        self.adjusted_weights = {}
        for component, weight in weights_config.items():
            if available_components.get(component, True):
                adjusted_weight = weight * redistribution_factor
                self.adjusted_weights[component] = adjusted_weight
            else:
                self.adjusted_weights[component] = 0.0
                self.redistribution_log.append(
                    f"Componente '{component}' no disponible, peso redistribuido"
                )
        
        # Validar suma
        total_adjusted = sum(self.adjusted_weights.values())
        if abs(total_adjusted - 1.0) > 0.001:
            logger.warning(
                f"Suma de pesos ajustados = {total_adjusted:.4f}, esperado = 1.0. "
                f"Ajustando último componente."
            )
            # Ajustar el último componente para que la suma sea exactamente 1.0
            diff = 1.0 - total_adjusted
            last_component = list(self.adjusted_weights.keys())[-1]
            self.adjusted_weights[last_component] += diff
        
        self.redistribution_log.append(
            f"Pesos ajustados: {self.adjusted_weights}"
        )
        
        return self.adjusted_weights
    
    def validate_weights(self, weights: Dict[str, float]) -> Tuple[bool, str]:
        """
        Valida que los pesos sean coherentes.
        
        Args:
            weights: Diccionario de pesos a validar
        
        Returns:
            Tuple[bool, str]: (es_válido, mensaje)
        """
        total = sum(weights.values())
        
        if abs(total - 1.0) > 0.001:
            return False, f"Suma de pesos = {total:.4f}, esperado = 1.0"
        
        negative_weights = [k for k, v in weights.items() if v < 0]
        if negative_weights:
            return False, f"Pesos negativos encontrados: {negative_weights}"
        
        weights_over_1 = [k for k, v in weights.items() if v > 1.0]
        if weights_over_1:
            return False, f"Pesos > 1.0 encontrados: {weights_over_1}"
        
        return True, "Pesos válidos"


# ============================================================================
# VALIDACIONES DE COHERENCIA
# ============================================================================

def validate_score_calculation(
    raw_scores: Dict[str, float],
    normalized_scores: Dict[str, float],
    weighted_scores: Dict[str, float],
    final_score: float,
    weights: Dict[str, float]
) -> Tuple[bool, list]:
    """
    Valida la coherencia de un cálculo de puntuación.
    
    Args:
        raw_scores: Puntajes brutos
        normalized_scores: Puntajes normalizados
        weighted_scores: Puntajes ponderados
        final_score: Puntaje final
        weights: Pesos usados
    
    Returns:
        Tuple[bool, list]: (es_coherente, lista de advertencias)
    """
    warnings = []
    
    # 1. Verificar que la suma de pesos sea 1.0
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.001:
        warnings.append(
            f"ADVERTENCIA: Suma de pesos = {total_weight:.4f}, esperado = 1.0"
        )
    
    # 2. Verificar que los puntajes normalizados estén en rango [0, 10]
    for component, score in normalized_scores.items():
        if score < 0:
            warnings.append(
                f"ERROR: Score normalizado negativo en '{component}': {score:.2f}"
            )
        elif score > 10:
            warnings.append(
                f"ADVERTENCIA: Score normalizado > 10 en '{component}': {score:.2f}"
            )
    
    # 3. Verificar que los weighted scores sean consistentes
    for component in raw_scores:
        expected_weighted = normalized_scores.get(component, 0) * weights.get(component, 0)
        actual_weighted = weighted_scores.get(component, 0)
        if abs(expected_weighted - actual_weighted) > 0.01:
            warnings.append(
                f"INCONSISTENCIA: '{component}': "
                f"esperado={expected_weighted:.3f}, actual={actual_weighted:.3f}"
            )
    
    # 4. Verificar que el puntaje final sea razonable
    if final_score < 0:
        warnings.append(f"ERROR: Puntaje final negativo: {final_score:.2f}")
    elif final_score > 10:
        warnings.append(f"ADVERTENCIA: Puntaje final > 10: {final_score:.2f}")
    
    is_coherent = len([w for w in warnings if w.startswith('ERROR')]) == 0
    
    return is_coherent, warnings


def calculate_confidence(score_diff: float, total_score: float) -> float:
    """
    Calcula el porcentaje de confianza basado en la diferencia de puntajes.
    
    Args:
        score_diff: Diferencia entre puntajes de los dos jugadores
        total_score: Suma de ambos puntajes
    
    Returns:
        float: Porcentaje de confianza (50-95)
    
    FÓRMULA:
        confidence = 50 + (|score_diff| / total_score) * 50
    
    Example:
        score_diff = 0.5, total_score = 5.0
        confidence = 50 + (0.5 / 5.0) * 50 = 50 + 5 = 55%
    """
    if total_score <= 0:
        return 50.0
    
    confidence = 50 + (abs(score_diff) / total_score) * 50
    return min(95.0, max(50.0, confidence))


# ============================================================================
# INTERFAZ SIMPLIFICADA
# ============================================================================

def normalize_and_weight_scores(
    raw_scores: Dict[str, float],
    weights: Dict[str, float],
    normalization_method: str = 'min_max',
    available_components: Optional[Dict[str, bool]] = None
) -> Dict[str, Any]:
    """
    Interfaz simplificada para normalizar puntajes y aplicar pesos.
    
    Args:
        raw_scores: {componente: puntaje_bruto}
        weights: {componente: peso}
        normalization_method: 'min_max' o 'log_scale'
        available_components: {componente: disponible} o None para todos disponibles
    
    Returns:
        Dict con: normalized, weighted, final, validation_warnings
    
    Example:
        raw = {'ranking': 225, 'form': 180, 'surface': 100}
        weights = {'ranking': 0.20, 'form': 0.15, 'surface': 0.15}
        result = normalize_and_weight_scores(raw, weights)
        # result['final'] = puntaje final normalizado y ponderado
    """
    # Usar todos los componentes como disponibles si no se especifica
    if available_components is None:
        available_components = {k: True for k in raw_scores}
    
    # Calcular pesos ajustados
    weight_manager = WeightManager(weights)
    adjusted_weights = weight_manager.calculate_adjusted_weights(available_components)
    
    # Normalizar puntajes
    normalized = {}
    normalization_metadata = {}
    
    for component, raw_score in raw_scores.items():
        if not available_components.get(component, True):
            normalized[component] = 0.0
            continue
        
        if normalization_method == 'log_scale':
            normalized[component], meta = normalize_with_log_scale(
                raw_score, component
            )
        else:  # min_max por defecto
            normalized[component], meta = normalize_min_max(raw_score, component)
        
        normalization_metadata[component] = meta
    
    # Aplicar pesos
    weighted = {
        component: normalized[component] * adjusted_weights.get(component, 0)
        for component in normalized
    }
    
    # Calcular puntaje final
    final_score = sum(weighted.values())
    
    # Validar coherencia
    is_coherent, validation_warnings = validate_score_calculation(
        raw_scores, normalized, weighted, final_score, adjusted_weights
    )
    
    return {
        'raw_scores': raw_scores,
        'normalized_scores': normalized,
        'weighted_scores': weighted,
        'adjusted_weights': adjusted_weights,
        'final_score': final_score,
        'is_coherent': is_coherent,
        'validation_warnings': validation_warnings,
        'redistribution_log': weight_manager.redistribution_log,
        'normalization_metadata': normalization_metadata
    }