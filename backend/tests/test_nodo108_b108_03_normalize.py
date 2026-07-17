"""
tests/test_nodo108_b108_03_normalize.py — REGLA-T53: B108-03 name-matching unificado.

Verifica que betslip_registrar._match_stake, ranking_manager.normalize_name
y kambi_tennis._normalize_name producen la misma salida que player_registry.normalize_player_name
para los mismos inputs. Incluye casos con diacríticos, guiones y variaciones de formato.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.player_registry import normalize_player_name
from betslip_registrar import _match_stake


# ── T1-T4: normalize_player_name == ranking_manager.normalize_name ────────────

def test_ranking_manager_delega_a_player_registry():
    """RankingManager.normalize_name delega a normalize_player_name — misma salida."""
    from analysis.ranking_manager import RankingManager
    rm = RankingManager.__new__(RankingManager)  # sin __init__ completo
    casos = [
        "Alcaraz", "García-López", "Müller", "Ñoño Pérez", "van de Zandschulp",
        "Davidovich Fokina", "Pierre-Hugues Herbert", "",
    ]
    for caso in casos:
        assert rm.normalize_name(caso) == normalize_player_name(caso), (
            f"Divergencia en '{caso}': rm={rm.normalize_name(caso)} "
            f"pr={normalize_player_name(caso)}"
        )


# ── T2: kambi_tennis._normalize_name == normalize_player_name ────────────────

def test_kambi_normalize_delega_a_player_registry():
    """kambi_tennis._normalize_name delega a normalize_player_name — misma salida."""
    from scraping.kambi_tennis import _normalize_name as kambi_norm
    casos = [
        "Alcaraz", "García", "Müller", "Ñoño", "van de Zandschulp",
        "Carlos Alcaraz Garfia", "",
    ]
    for caso in casos:
        assert kambi_norm(caso) == normalize_player_name(caso), (
            f"Divergencia en '{caso}': kambi={kambi_norm(caso)} "
            f"pr={normalize_player_name(caso)}"
        )


# ── T3-T6: _match_stake usa normalize_player_name ────────────────────────────

def _plan(*jugadores):
    """Construye un trader_plan mínimo."""
    return {
        "individuales": [
            {"favorito": j, "stake": 1000 * (i + 1), "retorno_potencial": 0}
            for i, j in enumerate(jugadores)
        ]
    }


def test_match_stake_exact_con_diacriticos():
    """_match_stake encuentra jugador con diacríticos (normalización canónica)."""
    plan = _plan("García-López")
    result = _match_stake("García-López", plan)
    assert result["stake"] == 1000


def test_match_stake_diacritico_vs_sin_diacritico():
    """_match_stake encuentra jugador aunque el plan use versión sin acento."""
    plan = _plan("Garcia Lopez")  # plan sin acento
    result = _match_stake("García-López", plan)  # query con acento
    assert result["stake"] == 1000, (
        "normalize_player_name debe equiparar 'García-López' con 'Garcia Lopez'"
    )


def test_match_stake_surname_tier2():
    """Tier 2 surname match funciona con normalización canónica."""
    plan = _plan("Carlos Alcaraz")
    result = _match_stake("Alcaraz", plan)
    assert result["stake"] == 1000


def test_match_stake_sin_match_retorna_cero():
    """Jugador que no está en el plan → stake=0."""
    plan = _plan("Djokovic", "Nadal")
    result = _match_stake("Federer", plan)
    assert result["stake"] == 0


# ── T7: normalize_player_name — casos límite ─────────────────────────────────

def test_normalize_player_name_casos_limite():
    """normalize_player_name maneja casos límite sin crash."""
    assert normalize_player_name("") == ""
    assert normalize_player_name("  Ñ  ") == "n"
    assert normalize_player_name("García-López") == "garcia lopez"
    assert normalize_player_name("Müller") == "muller"
    assert normalize_player_name("Davidovich+2") == "davidovich"  # sufijo numérico


# ── T8: consistencia entre los 3 call-sites ──────────────────────────────────

def test_tres_call_sites_consistentes():
    """Los 3 call-sites migrados producen salida idéntica para inputs con diacríticos."""
    from analysis.ranking_manager import RankingManager
    from scraping.kambi_tennis import _normalize_name as kambi_norm
    rm = RankingManager.__new__(RankingManager)

    casos_diacriticos = ["Ñoño", "García", "Müller", "Pérez", "Ångström"]
    for caso in casos_diacriticos:
        pr = normalize_player_name(caso)
        rm_out = rm.normalize_name(caso)
        k_out = kambi_norm(caso)
        assert pr == rm_out == k_out, (
            f"'{caso}': player_registry={pr} ranking_mgr={rm_out} kambi={k_out}"
        )
