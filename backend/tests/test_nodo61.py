"""
Tests Nodo-61: GCS Season Window Fix — fecha_partido + season-aware gate

T61-01: Birmingham 2026 vs 2025 desambiguación — gcs_days=15, no gcs_days=380
T61-02: Solo Birmingham 2025 (año diferente) → gcs_active=False (Bug F0 resuelto)
T61-03: Nottingham Jun 10 (days=26) → gcs_extended_active=True, gcs_active=False
T61-04: fecha_partido=date(2026-07-07) → gcs_days=16 para Birmingham Jun 21 2026
T61-05: Torneo fuera de ventana estacional (Octubre) → no GCS ni zona extendida
T61-06: _GCS_EXTENDED_ENABLED=False por default → sin boost en zona extendida
T61-07: LOG_GCS_SHADOW_EXTENDED en zona 22-42d hierba
T61-08: H60-02 en preregistered_hypotheses.json con n_stop=30 y estado=PENDIENTE
T61-09: _is_gcs_season_active() importable como función standalone del módulo
T61-10: Eala R4 simulación — Birmingham Jun 21 2026 + fecha_partido=Jul 6 → gcs_days=15
"""
import json
import os
import sys
import datetime

import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fecha de referencia fija: Wimbledon R4, 2026-07-06
REF_DATE = datetime.date(2026, 7, 6)
REF_DT = datetime.datetime(2026, 7, 6)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_tournament_from_ref(torneo_name, ref_date, days_ago, n_wins=5, surface='Hierba',
                               best_opp_rank=8):
    """Crea partidos de torneo completo con final = ref_date - days_ago."""
    if isinstance(ref_date, datetime.datetime):
        ref_dt = ref_date
    else:
        ref_dt = datetime.datetime.combine(ref_date, datetime.time.min)
    final_date = ref_dt - datetime.timedelta(days=days_ago)
    matches = []
    for i in range(n_wins):
        fecha = (final_date - datetime.timedelta(days=(n_wins - 1 - i))).strftime('%d.%m.%Y')
        matches.append({
            'jugador1': 'TestPlayer',
            'jugador2': f'Opp{i}',
            'resultado': '2-0',
            'ganador': 'TestPlayer',
            'superficie': surface,
            'fecha': fecha,
            'torneo': torneo_name,
            'oponente': f'Opp{i}',
            'opponent_ranking': best_opp_rank if i == 0 else 50,
        })
    return matches


def _make_bg_matches(ref_date, n_wins, n_total, surface='Hierba'):
    """Background matches sin torneo completo (con derrotas mezcladas)."""
    if isinstance(ref_date, datetime.datetime):
        ref_dt = ref_date
    else:
        ref_dt = datetime.datetime.combine(ref_date, datetime.time.min)
    matches = []
    for i in range(n_total):
        ganador = 'TestPlayer' if i < n_wins else 'Opponent'
        fecha = (ref_dt - datetime.timedelta(days=90 + i * 4)).strftime('%d.%m.%Y')
        matches.append({
            'jugador1': 'TestPlayer',
            'jugador2': 'Opponent',
            'resultado': '2-0' if ganador == 'TestPlayer' else '0-2',
            'ganador': ganador,
            'superficie': surface,
            'fecha': fecha,
            'torneo': 'BackgroundEvent',
            'oponente': 'Opponent',
        })
    return matches


def _get_analyzer():
    from analysis.rivalry_analyzer import RivalryAnalyzer
    rm = MagicMock()
    rm.get_player_ranking.return_value = None
    rm.get_player_info.return_value = {'ranking_position': 50, 'ranking_points': 500,
                                       'prox_points': 500, 'max_points': 500, 'defense_points': 0}
    rm.normalize_name.side_effect = lambda n: n.lower() if n else n
    es = MagicMock()
    es.default_rating = 1500
    es.k_factor = 32
    es.expected_score.return_value = 0.5
    es.calculate_rating_change.return_value = 16
    return RivalryAnalyzer(rm, es)


# ── T61-01: Desambiguación año 2025 vs 2026 ──────────────────────────────────

def test_T61_01_year_disambiguation_2025_vs_2026():
    """Bug F0 fix: torneo 'Birmingham' en 2025 (days=380) Y en 2026 (days=15).
    Con fecha_partido=Jul 6 2026, GCS debe usar el de 2026. gcs_days=15, no 380."""
    analyzer = _get_analyzer()

    # Birmingham 2025: finals Jun 21, 2025 → ~380 días antes de Jul 6, 2026
    birmingham_2025 = _make_tournament_from_ref(
        'Birmingham', REF_DATE, days_ago=380, n_wins=5, surface='Hierba'
    )
    # Birmingham 2026: finals Jun 21, 2026 → 15 días antes de Jul 6, 2026
    birmingham_2026 = _make_tournament_from_ref(
        'Birmingham', REF_DATE, days_ago=15, n_wins=5, surface='Hierba'
    )
    historia = birmingham_2025 + birmingham_2026 + _make_bg_matches(REF_DATE, 20, 30, 'Hierba')

    result, log = analyzer.analyze_surface_specialization(
        historia, 'Hierba', 'TestPlayer', fecha_partido=REF_DATE
    )

    assert result['gcs_active'] is True, (
        f"Birmingham 2026 (days=15) debe activar gcs_active. Got {result['gcs_active']}"
    )
    assert result['gcs_days'] == 15, (
        f"gcs_days debe ser 15 (Birmingham 2026), no 380 (Birmingham 2025). Got {result['gcs_days']}"
    )


# ── T61-02: Solo torneo 2025 → gcs_active=False (Bug F0 resuelto) ───────────

def test_T61_02_only_2025_tournament_no_gcs():
    """Con solo Birmingham 2025 (año diferente) y fecha_partido=2026-07-06,
    _is_gcs_season_active devuelve False → gcs_active=False."""
    analyzer = _get_analyzer()

    # Solo Birmingham 2025 (>365 días atrás respecto a fecha_partido=2026-07-06)
    birmingham_2025 = _make_tournament_from_ref(
        'Birmingham', REF_DATE, days_ago=380, n_wins=5, surface='Hierba'
    )
    historia = birmingham_2025 + _make_bg_matches(REF_DATE, 20, 30, 'Hierba')

    result, log = analyzer.analyze_surface_specialization(
        historia, 'Hierba', 'TestPlayer', fecha_partido=REF_DATE
    )

    assert result['gcs_active'] is False, (
        f"Solo torneo 2025 (año diferente) → gcs_active debe ser False. Got {result['gcs_active']}"
    )
    assert result['gcs_extended_active'] is False, (
        f"Torneo 2025 no debe activar zona extendida (año diferente). Got {result['gcs_extended_active']}"
    )


# ── T61-03: Zona extendida (days=26) → gcs_extended_active=True ─────────────

def test_T61_03_extended_zone_detected_days_26():
    """Nottingham Jun 10 2026 (days=26 desde Jul 6) → zona extendida.
    gcs_extended_active=True, gcs_active=False, gcs_days=None."""
    analyzer = _get_analyzer()

    # Nottingham finals Jun 10, 2026 → 26 días antes de Jul 6, 2026
    historia = (
        _make_tournament_from_ref('Nottingham 2026', REF_DATE, days_ago=26, n_wins=5, surface='Hierba') +
        _make_bg_matches(REF_DATE, 20, 30, 'Hierba')
    )

    result, log = analyzer.analyze_surface_specialization(
        historia, 'Hierba', 'TestPlayer', fecha_partido=REF_DATE
    )

    assert result['gcs_active'] is False, (
        f"days=26 > 21 → gcs_active debe ser False. Got {result['gcs_active']}"
    )
    assert result['gcs_days'] is None, (
        f"gcs_days debe ser None (fuera de zona activa ≤21d). Got {result['gcs_days']}"
    )
    assert result.get('gcs_extended_active') is True, (
        f"days=26, mismo año, en temporada → gcs_extended_active debe ser True. Got {result.get('gcs_extended_active')}"
    )
    assert result.get('gcs_extended_days') == 26, (
        f"gcs_extended_days debe ser 26. Got {result.get('gcs_extended_days')}"
    )


# ── T61-04: fecha_partido=Jul 7 → gcs_days=16 para Birmingham Jun 21 ─────────

def test_T61_04_fecha_partido_jul7_gives_gcs_days_16():
    """Nodo-61 D61-F0: con fecha_partido=date(2026-07-07) y Birmingham Jun 21,
    gcs_days debe ser 16 (Jul 7 - Jun 21 = 16d), no 15."""
    analyzer = _get_analyzer()

    ref_jul7 = datetime.date(2026, 7, 7)
    # Birmingham finals Jun 21, 2026 → 16 días antes de Jul 7, 2026
    historia = (
        _make_tournament_from_ref('Birmingham 2026', ref_jul7, days_ago=16, n_wins=5, surface='Hierba') +
        _make_bg_matches(ref_jul7, 20, 30, 'Hierba')
    )

    result, log = analyzer.analyze_surface_specialization(
        historia, 'Hierba', 'TestPlayer', fecha_partido=ref_jul7
    )

    assert result['gcs_active'] is True, (
        f"days=16 ≤ 21 → gcs_active debe ser True. Got {result['gcs_active']}"
    )
    assert result['gcs_days'] == 16, (
        f"Con fecha_partido=Jul 7, gcs_days debe ser 16. Got {result['gcs_days']}"
    )


# ── T61-05: Torneo fuera de ventana estacional → no GCS ─────────────────────

def test_T61_05_out_of_season_window_no_gcs():
    """Torneo ATP500 en Octubre (fuera de ventana Jun 1 - Jul 13) → no activa GCS
    aunque tenga days<=21 respecto al partido."""
    analyzer = _get_analyzer()

    # Partido en Octubre: ref_date = Oct 20, 2026
    ref_oct = datetime.date(2026, 10, 20)
    # Torneo ATP500 hace 10 días (Oct 10) → pero fuera de ventana estacional hierba
    historia = (
        _make_tournament_from_ref('Vienna 2026', ref_oct, days_ago=10, n_wins=5, surface='Hierba') +
        _make_bg_matches(ref_oct, 20, 30, 'Hierba')
    )

    result, log = analyzer.analyze_surface_specialization(
        historia, 'Hierba', 'TestPlayer', fecha_partido=ref_oct
    )

    # Octubre está fuera de la ventana estacional de hierba (Jun 1 - Jul 13)
    assert result['gcs_active'] is False, (
        f"Torneo en Octubre (fuera ventana hierba) → gcs_active debe ser False. Got {result['gcs_active']}"
    )
    assert result.get('gcs_extended_active') is False, (
        f"Torneo fuera de ventana → gcs_extended_active debe ser False. Got {result.get('gcs_extended_active')}"
    )


# ── T61-06: _GCS_EXTENDED_ENABLED=False por default ─────────────────────────

def test_T61_06_gcs_extended_flag_is_false_by_default():
    """Nodo-61 D61-F3: _GCS_EXTENDED_ENABLED debe ser False en producción.
    La zona extendida NO aplica boost hasta que H60-02 gradúe."""
    import analysis.rivalry_analyzer as ra_mod
    assert ra_mod._GCS_EXTENDED_ENABLED is False, (
        "_GCS_EXTENDED_ENABLED debe ser False por default (H60-02 pendiente graduación)"
    )
    # Verificar también _GCS_LOOKBACK_DAYS y _GCS_SEASON_WINDOWS
    assert ra_mod._GCS_LOOKBACK_DAYS == 42, (
        f"_GCS_LOOKBACK_DAYS debe ser 42, got {ra_mod._GCS_LOOKBACK_DAYS}"
    )
    assert 'grass' in ra_mod._GCS_SEASON_WINDOWS, "_GCS_SEASON_WINDOWS debe tener clave 'grass'"
    assert ra_mod._GCS_SEASON_WINDOWS['grass']['dias_max'] == 42, "dias_max debe ser 42"


# ── T61-07: LOG_GCS_SHADOW_EXTENDED en zona extendida hierba ────────────────

def test_T61_07_log_gcs_shadow_extended_fires_in_extended_zone():
    """Nodo-61 D61-F2: LOG_GCS_SHADOW_EXTENDED en log cuando days=28, hierba, _GCS_EXTENDED_ENABLED=False."""
    import analysis.rivalry_analyzer as ra_mod
    assert ra_mod._GCS_EXTENDED_ENABLED is False

    analyzer = _get_analyzer()

    historia = (
        _make_tournament_from_ref('Nottingham 2026', REF_DATE, days_ago=28, n_wins=5, surface='Hierba') +
        _make_bg_matches(REF_DATE, 20, 30, 'Hierba')
    )

    result, log = analyzer.analyze_surface_specialization(
        historia, 'Hierba', 'TestPlayer', fecha_partido=REF_DATE
    )

    shadow_ext = [l for l in log if 'LOG_GCS_SHADOW_EXTENDED' in l]
    assert len(shadow_ext) >= 1, (
        f"LOG_GCS_SHADOW_EXTENDED debe aparecer para days=28 hierba. Log: {log}"
    )
    # Debe mencionar H60-02 o PENDIENTE para indicar que el flag está desactivado
    msg = shadow_ext[0].lower()
    assert 'pendiente' in msg or 'h60-02' in msg or 'sin boost' in msg, (
        f"LOG_GCS_SHADOW_EXTENDED debe mencionar estado pendiente. Got: {shadow_ext[0]}"
    )


# ── T61-08: H60-02 en preregistered_hypotheses.json ─────────────────────────

def test_T61_08_h60_02_in_hypotheses():
    """H60-02 existe en preregistered_hypotheses.json con n_stop=30, estado=PENDIENTE."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hyp_path = os.path.join(base_dir, 'validation', 'preregistered_hypotheses.json')

    assert os.path.exists(hyp_path), "preregistered_hypotheses.json no encontrado"

    with open(hyp_path) as f:
        data = json.load(f)

    hypotheses = data.get('hypotheses', {})
    assert 'H60-02' in hypotheses, "H60-02 no encontrado en hypotheses — Nodo-61 D61-F4"

    h = hypotheses['H60-02']
    assert h.get('n_stop') == 30, f"n_stop debe ser 30, got {h.get('n_stop')}"
    assert h.get('estado') == 'PENDIENTE', f"estado debe ser PENDIENTE, got {h.get('estado')}"

    umbrales = h.get('umbrales_congelados', {})
    assert umbrales.get('dias_min') == 22, f"dias_min debe ser 22, got {umbrales.get('dias_min')}"
    assert umbrales.get('dias_max') == 42, f"dias_max debe ser 42, got {umbrales.get('dias_max')}"
    assert umbrales.get('tier_min') == 'atp500', "tier_min debe ser atp500"

    assert h.get('gated'), "H60-02 debe tener campo 'gated'"
    assert 'preregistrado' in h, "H60-02 debe tener campo 'preregistrado'"


# ── T61-09: _is_gcs_season_active importable como función standalone ─────────

def test_T61_09_is_gcs_season_active_importable():
    """Nodo-61 D61-F1: _is_gcs_season_active() importable desde rivalry_analyzer
    como función standalone (REGLA-T53: invocar función del módulo)."""
    from analysis.rivalry_analyzer import _is_gcs_season_active

    # Caso 1: zona activa — Birmingham Jun 21 2026, partido Jul 6 2026
    is_act, days = _is_gcs_season_active(
        datetime.datetime(2026, 6, 21), datetime.datetime(2026, 7, 6), 'grass'
    )
    assert is_act is True, f"Jun 21 → Jul 6 (15d, mismo año, en temporada) debe ser activo. Got {is_act}"
    assert days == 15, f"days debe ser 15, got {days}"

    # Caso 2: año diferente → False
    is_old, days_old = _is_gcs_season_active(
        datetime.datetime(2025, 6, 21), datetime.datetime(2026, 7, 6), 'grass'
    )
    assert is_old is False, f"Año 2025 vs 2026 → debe ser False. Got {is_old}"

    # Caso 3: zona extendida → False (is_active), pero days en rango
    is_ext, days_ext = _is_gcs_season_active(
        datetime.datetime(2026, 6, 8), datetime.datetime(2026, 7, 6), 'grass'
    )
    assert is_ext is False, f"days=28 > 21 → is_active debe ser False (zona extendida). Got {is_ext}"
    assert days_ext == 28, f"days debe ser 28, got {days_ext}"


# ── T61-10: Eala R4 — Birmingham Jun 21 2026 + fecha_partido=Jul 6 → gcs_days=15 ─

def test_T61_10_eala_r4_simulation_gcs_days_15():
    """Simulación Eala R4 Wimbledon: ganó Birmingham Jun 21 2026.
    Con fecha_partido=date(2026-07-06): gcs_days=15 (no 28 como sin el fix de Bug F0).
    Éste es el caso real que motivó Nodo-61."""
    analyzer = _get_analyzer()

    # Eala ganó Birmingham (WTA 500 = atp500 tier), finals Jun 21, 2026
    # Desde Jul 6, 2026: days = 15 → zona activa, gcs_active=True, boost ×1.8
    birmingham_2026 = _make_tournament_from_ref(
        'Birmingham 2026', REF_DATE, days_ago=15, n_wins=5, surface='Hierba'
    )
    historia = birmingham_2026 + _make_bg_matches(REF_DATE, 20, 30, 'Hierba')

    result, log = analyzer.analyze_surface_specialization(
        historia, 'Hierba', 'Eala', fecha_partido=REF_DATE
    )

    assert result['gcs_active'] is True, (
        f"Eala R4: Birmingham Jun 21 2026 → gcs_active debe ser True. Got {result['gcs_active']}"
    )
    assert result['gcs_days'] == 15, (
        f"Eala R4: gcs_days debe ser 15 (no 28). Got {result['gcs_days']} — Bug F0 fix fallido"
    )
    assert result.get('gcs_extended_active') is False, (
        f"gcs_days=15 (zona activa) → gcs_extended_active debe ser False. Got {result.get('gcs_extended_active')}"
    )
    # Verificar que boost ×1.8 aplica (14 < 15 ≤ 21 → ×1.5; pero usamos days=15 exacto)
    boost_log = [l for l in log if 'GCS_RECENCY_BOOST' in l and 'LOG_GCS_SHADOW' not in l]
    assert len(boost_log) >= 1, f"GCS_RECENCY_BOOST debe estar en log para Eala R4. Log: {log}"
