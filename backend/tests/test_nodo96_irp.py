"""
tests/test_nodo96_irp.py — Nodo-96: IRP Individual Return-from-inactivity Profile

REGLA-T53: todos los tests invocan funciones reales del módulo, nunca hardcodean fórmulas.
"""
import json
import sys
import tempfile
from datetime import date
from pathlib import Path

import pytest

# Asegurar que el backend está en el path
_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from scripts.build_irp_profiles import (
    MIN_RETORNOS,
    RETURN_THRESHOLD_DAYS,
    build_irp_profiles,
    compute_player_irp,
    normalize_for_index,
)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

def _make_rows(*args):
    """
    Crea rows de PlayerDB sintéticos.
    args: lista de (fecha_str, won_bool)
    """
    return [{'fecha': f, 'won': w} for f, w in args]


BUILD_DATE = date(2026, 7, 14)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: cómputo básico de retornos
# ══════════════════════════════════════════════════════════════════════════════

def test_build_profiles_computes_return_correctly():
    """Jugador con gap > 30d entre matches → n_retornos correcto."""
    rows = _make_rows(
        ('2026-01-01', True),   # match 1 (normal)
        ('2026-02-10', True),   # gap 40d → RETURN, won
        ('2026-02-20', False),  # gap 10d → normal
        ('2026-04-01', False),  # gap 39d → RETURN, lost
    )
    profile = compute_player_irp('Test_Player', rows, BUILD_DATE)
    assert profile is not None
    assert profile['n_retornos'] == 2
    assert profile['win_rate_return'] == pytest.approx(0.5, abs=1e-4)   # 1 WON / 2


def test_win_rate_return_vs_normal_computed_separately():
    """win_rate_return y win_rate_normal son independientes."""
    rows = _make_rows(
        ('2026-01-01', True),    # normal (primer match)
        ('2026-01-10', True),    # normal (gap 9d)
        ('2026-01-15', False),   # normal (gap 5d)
        ('2026-04-01', False),   # RETURN (gap 75d), lost
        ('2026-05-15', True),    # RETURN (gap 44d), won
    )
    profile = compute_player_irp('Test_Player', rows, BUILD_DATE)
    assert profile is not None
    # Normal: 2 won, 1 lost → 2/3
    assert profile['win_rate_normal'] == pytest.approx(2 / 3, abs=1e-4)
    # Return: 1 won, 1 lost → 0.5
    assert profile['win_rate_return'] == pytest.approx(0.5, abs=1e-4)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: delta_return negativo cuando rinde peor en retorno
# ══════════════════════════════════════════════════════════════════════════════

def test_delta_return_is_negative_when_worse_on_return():
    """Jugador que siempre pierde al volver → delta_return < 0."""
    rows = _make_rows(
        ('2026-01-01', True),   # normal
        ('2026-01-10', True),   # normal
        ('2026-01-20', True),   # normal
        ('2026-05-01', False),  # RETURN, lost
        ('2026-08-01', False),  # RETURN, lost (gap ~92d)
    )
    profile = compute_player_irp('BadReturner', rows, BUILD_DATE)
    assert profile is not None
    assert profile['delta_return'] < 0


def test_delta_return_positive_when_better_on_return():
    """Jugador que siempre gana al volver → delta_return > 0."""
    rows = _make_rows(
        ('2026-01-01', False),  # normal
        ('2026-01-10', False),  # normal
        ('2026-01-20', False),  # normal
        ('2026-05-01', True),   # RETURN, won
        ('2026-08-01', True),   # RETURN, won (gap ~92d)
    )
    profile = compute_player_irp('GoodReturner', rows, BUILD_DATE)
    assert profile is not None
    assert profile['delta_return'] > 0


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: jugadores sin retornos excluidos
# ══════════════════════════════════════════════════════════════════════════════

def test_player_with_no_returns_excluded():
    """Jugador con todos los gaps <= 30d → n_retornos = 0 → no en profiles."""
    rows = _make_rows(
        ('2026-01-01', True),
        ('2026-01-15', True),   # gap 14d
        ('2026-01-25', False),  # gap 10d
        ('2026-02-05', True),   # gap 11d
    )
    profile = compute_player_irp('NoReturns', rows, BUILD_DATE)
    assert profile is None   # n_retornos < MIN_RETORNOS


def test_player_with_one_return_excluded():
    """Jugador con exactamente 1 retorno → excluido (MIN_RETORNOS = 2)."""
    rows = _make_rows(
        ('2026-01-01', True),
        ('2026-03-15', True),   # gap 73d → RETURN (único)
        ('2026-03-20', False),  # gap 5d → normal
    )
    profile = compute_player_irp('OneReturn', rows, BUILD_DATE)
    assert profile is None


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: name_index normalization
# ══════════════════════════════════════════════════════════════════════════════

def test_normalize_for_index_strips_underscores_and_lowercases():
    assert normalize_for_index('Novak_Djokovic') == 'novak djokovic'


def test_normalize_for_index_handles_accents():
    result = normalize_for_index('Rafael_Nadal')
    assert result == 'rafael nadal'


def test_normalize_for_index_handles_already_spaced():
    assert normalize_for_index('Carlos Alcaraz') == 'carlos alcaraz'


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: _irp_lookup desde edge_calculator
# ══════════════════════════════════════════════════════════════════════════════

def test_irp_lookup_returns_profile_by_name():
    """_irp_lookup encuentra el profile usando name_index."""
    from edge_calculator import _irp_lookup
    irp_data = {
        'name_index': {'novak djokovic': 'Novak_Djokovic'},
        'profiles': {
            'Novak_Djokovic': {
                'slug': 'Novak_Djokovic',
                'n_retornos': 5,
                'win_rate_return': 0.6,
                'delta_return': -0.1,
            }
        }
    }
    result = _irp_lookup('Novak Djokovic', irp_data)
    assert result['n_retornos'] == 5
    assert result['win_rate_return'] == pytest.approx(0.6)


def test_irp_lookup_returns_empty_dict_when_not_found():
    """Nombre desconocido → {} (D96-07: silencioso)."""
    from edge_calculator import _irp_lookup
    irp_data = {'name_index': {}, 'profiles': {}}
    assert _irp_lookup('Unknown Player', irp_data) == {}


def test_irp_lookup_returns_empty_dict_for_empty_irp_data():
    """irp_data vacío → {} sin excepción."""
    from edge_calculator import _irp_lookup
    assert _irp_lookup('Any Player', {}) == {}


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: build_irp_profiles end-to-end con PlayerDB sintético
# ══════════════════════════════════════════════════════════════════════════════

def test_build_irp_profiles_end_to_end():
    """build_irp_profiles lee un PlayerDB sintético y escribe irp_profiles.json."""
    player_db = {
        'version': '1.0',
        'built_at': '2026-07-14T00:00:00',
        'n_players': 2,
        'n_rows_raw': 10,
        'n_rows_deduped': 8,
        'players': {
            'Alice_Smith': {
                'slug': 'Alice_Smith',
                'own_ranking': 50,
                'rows': [
                    {'fecha': '2026-01-01', 'won': True},
                    {'fecha': '2026-03-01', 'won': False},   # gap 59d → RETURN
                    {'fecha': '2026-06-01', 'won': True},    # gap 92d → RETURN
                ]
            },
            'Bob_Jones': {
                'slug': 'Bob_Jones',
                'own_ranking': 100,
                'rows': [
                    {'fecha': '2026-01-01', 'won': True},
                    {'fecha': '2026-01-10', 'won': True},    # gap 9d → normal
                ]
            }
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path  = Path(tmpdir) / 'player_db.json'
        out_path = Path(tmpdir) / 'irp_profiles.json'
        db_path.write_text(json.dumps(player_db), encoding='utf-8')

        result = build_irp_profiles(
            player_db_path=db_path,
            output_path=out_path,
            build_date=BUILD_DATE,
        )

        # Alice tiene 2 retornos → incluida
        assert 'Alice_Smith' in result['profiles']
        # Bob no tiene retornos suficientes → excluido
        assert 'Bob_Jones' not in result['profiles']

        # name_index contiene alice
        assert 'alice smith' in result['name_index']
        assert result['name_index']['alice smith'] == 'Alice_Smith'

        # Archivo escrito
        assert out_path.exists()
        on_disk = json.loads(out_path.read_text(encoding='utf-8'))
        assert on_disk['n_players_with_irp'] == 1
        assert on_disk['return_threshold_days'] == RETURN_THRESHOLD_DAYS


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7: days_since_last computed from last fecha
# ══════════════════════════════════════════════════════════════════════════════

def test_days_since_last_computed_from_last_fecha():
    """days_since_last = build_date - última fecha del jugador."""
    rows = _make_rows(
        ('2026-01-01', True),
        ('2026-03-01', False),   # RETURN
        ('2026-06-01', True),    # RETURN → last fecha
    )
    profile = compute_player_irp('Player', rows, BUILD_DATE)
    assert profile is not None
    expected_days = (BUILD_DATE - date(2026, 6, 1)).days
    assert profile['days_since_last'] == expected_days
    assert profile['last_match_fecha'] == '2026-06-01'


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8: avg_gap_return promedia los gaps de retorno
# ══════════════════════════════════════════════════════════════════════════════

def test_avg_gap_return_is_mean_of_return_gaps():
    """avg_gap_return = promedio de días antes de cada return_match."""
    rows = _make_rows(
        ('2026-01-01', True),
        ('2026-02-10', True),   # gap 40d → RETURN
        ('2026-05-01', False),  # gap 80d → RETURN
    )
    profile = compute_player_irp('GapPlayer', rows, BUILD_DATE)
    assert profile is not None
    expected_avg = (40 + 80) / 2
    assert profile['avg_gap_return'] == pytest.approx(expected_avg, abs=0.5)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 9: apellido fallback en name_index (H96-02)
# ══════════════════════════════════════════════════════════════════════════════

def test_name_index_contiene_apellido_fallback():
    """build_irp_profiles genera name_index con slug completo Y apellido (H96-02)."""
    player_db = {
        'players': {
            'Novak_Djokovic': {
                'slug': 'Novak_Djokovic',
                'rows': [
                    {'fecha': '2026-01-01', 'won': True},
                    {'fecha': '2026-03-15', 'won': False},   # RETURN
                    {'fecha': '2026-07-01', 'won': True},    # RETURN
                ]
            }
        }
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path  = Path(tmpdir) / 'player_db.json'
        out_path = Path(tmpdir) / 'irp_profiles.json'
        db_path.write_text(json.dumps(player_db), encoding='utf-8')
        result = build_irp_profiles(db_path, out_path, BUILD_DATE)

    idx = result['name_index']
    assert 'novak djokovic' in idx, "Slug completo debe estar en name_index"
    assert 'djokovic' in idx,       "Apellido debe estar en name_index (H96-02)"
    assert idx['djokovic'] == 'Novak_Djokovic'


def test_irp_lookup_encuentra_por_apellido(tmp_path):
    """_irp_lookup resuelve cuando edge_calculator pasa solo apellido (H96-02)."""
    from edge_calculator import _irp_lookup
    irp_data = {
        'name_index': {
            'novak djokovic': 'Novak_Djokovic',
            'djokovic':       'Novak_Djokovic',   # apellido fallback
        },
        'profiles': {
            'Novak_Djokovic': {'n_retornos': 8, 'delta_return': -0.12}
        }
    }
    result = _irp_lookup('Djokovic', irp_data)
    assert result.get('n_retornos') == 8, "Lookup por apellido debe encontrar el perfil"
