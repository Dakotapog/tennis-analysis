"""
tests/test_nodo113_weather.py — REGLA-T53: B108-06 weather_flag MVP.

Verifica get_weather_flag sin llamadas reales a open-meteo (mock de _fetch_open_meteo)
y que el campo weather_flag aparece en el resultado de calcular_edge_completo.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.weather_client import (
    get_weather_flag,
    _fetch_open_meteo,
    _RAIN_MM_THRESHOLD,
    _WIND_KMH_THRESHOLD,
    _WEATHER_CACHE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_meteo(rain_mm: float, wind_kmh: float):
    """Patch _fetch_open_meteo para devolver valores específicos."""
    return patch(
        'core.weather_client._fetch_open_meteo',
        return_value={'rain_mm': rain_mm, 'wind_kmh': wind_kmh},
    )


# ── T1: país conocido → CLEAR cuando lluvia y viento bajos ───────────────────

def test_pais_conocido_clear():
    """Francia arcilla sin lluvia ni viento → CLEAR."""
    _WEATHER_CACHE.clear()
    with _mock_meteo(0.0, 10.0):
        flag = get_weather_flag('Francia', 'clay', '2026-07-17')
    assert flag == 'CLEAR'


# ── T2: RAIN_RISK cuando lluvia >= threshold ──────────────────────────────────

def test_rain_risk_umbral():
    """Lluvia en umbral exacto → RAIN_RISK."""
    _WEATHER_CACHE.clear()
    with _mock_meteo(_RAIN_MM_THRESHOLD, 5.0):
        flag = get_weather_flag('España', 'arcilla', '2026-07-17')
    assert flag == 'RAIN_RISK'


def test_rain_risk_por_encima_umbral():
    """Lluvia por encima del umbral → RAIN_RISK."""
    _WEATHER_CACHE.clear()
    with _mock_meteo(_RAIN_MM_THRESHOLD + 5.0, 0.0):
        flag = get_weather_flag('Reino Unido', 'grass', '2026-07-17')
    assert flag == 'RAIN_RISK'


# ── T3: WIND_HIGH cuando viento alto sin lluvia ───────────────────────────────

def test_wind_high():
    """Viento >= threshold sin lluvia → WIND_HIGH."""
    _WEATHER_CACHE.clear()
    with _mock_meteo(0.0, _WIND_KMH_THRESHOLD):
        flag = get_weather_flag('Australia', 'hard', '2026-07-17')
    assert flag == 'WIND_HIGH'


# ── T4: RAIN_RISK tiene precedencia sobre WIND_HIGH ──────────────────────────

def test_rain_tiene_precedencia_sobre_viento():
    """Si lluvia Y viento ambos sobre umbral → RAIN_RISK (lluvia primero)."""
    _WEATHER_CACHE.clear()
    with _mock_meteo(_RAIN_MM_THRESHOLD + 2.0, _WIND_KMH_THRESHOLD + 5.0):
        flag = get_weather_flag('Italia', 'clay', '2026-07-17')
    assert flag == 'RAIN_RISK'


# ── T5: país desconocido → UNKNOWN ───────────────────────────────────────────

def test_pais_desconocido_unknown():
    """País no en tabla de coordenadas → UNKNOWN sin crash."""
    _WEATHER_CACHE.clear()
    flag = get_weather_flag('Narnia', 'clay', '2026-07-17')
    assert flag == 'UNKNOWN'


# ── T6: superficie indoor → UNKNOWN (no llamar API) ──────────────────────────

def test_superficie_indoor_unknown():
    """Superficie 'indoor' no es outdoor → UNKNOWN sin llamar open-meteo."""
    _WEATHER_CACHE.clear()
    with patch('core.weather_client._fetch_open_meteo') as mock_fetch:
        flag = get_weather_flag('España', 'indoor', '2026-07-17')
    assert flag == 'UNKNOWN'
    mock_fetch.assert_not_called()


# ── T7: API falla → UNKNOWN sin propagar excepción ───────────────────────────

def test_api_falla_unknown():
    """Si open-meteo falla (excepción) → UNKNOWN sin crash."""
    _WEATHER_CACHE.clear()
    with patch('core.weather_client._fetch_open_meteo', return_value={}):
        flag = get_weather_flag('Francia', 'clay', '2026-07-17')
    assert flag == 'UNKNOWN'


# ── T8: weather_flag aparece en calcular_edge_completo output ────────────────

def test_weather_flag_en_edge_completo():
    """calcular_edge_completo incluye campo weather_flag en resultado."""
    from edge_calculator import calcular_edge_completo

    partido = {
        'jugador1': 'Player A',
        'jugador2': 'Player B',
        'cuota1': 1.80,
        'cuota2': 2.10,
        'superficie': 'clay',
        'pais': 'Francia',
        'cuota_es_real': True,
        'torneo_nombre': 'Roland Garros',
        'torneo': 'Roland Garros',
        'tier': 'grand_slam',
        'ranking_analysis': {
            'prediction': {
                'favored_player': 'Player A',
                'confidence': 65.0,
                'p_model': 0.65,
                'reasoning': [],
                'surface_specialization_meta': {
                    'player1': {'score': 60.0, 'torneo_completo': False, 'gcs_active': False},
                    'player2': {'score': 40.0, 'torneo_completo': False, 'gcs_active': False},
                },
                'h2h_stats': {'wins': 3, 'losses': 2, 'total': 5},
                'historial_incompleto': {'p1': False, 'p2': False},
            }
        },
    }

    _WEATHER_CACHE.clear()
    with _mock_meteo(0.0, 5.0):
        resultado = calcular_edge_completo(partido, calibracion={})

    assert 'weather_flag' in resultado, "campo weather_flag debe estar en resultado"
    assert resultado['weather_flag'] in ('RAIN_RISK', 'WIND_HIGH', 'CLEAR', 'UNKNOWN')
