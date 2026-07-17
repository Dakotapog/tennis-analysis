"""
tests/test_nodo107_governor_coverage.py — REGLA-T53: tests invocan función real.

Cubre S107-B (D107-02): cobertura 12/12 estrategias en combo_governor.
Cubre S107-C (D107-03): exposicion_por_jugador cap 5% bankroll.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from combo_governor import (
    _trader_stakes_today,
    _rival_value_stakes_today,
    exposicion_por_jugador,
)


# ── Fixtures helpers ─────────────────────────────────────────────────────────

def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


# ── S107-B: _trader_stakes_today ─────────────────────────────────────────────

def test_trader_stakes_suma_individuales(tmp_path, monkeypatch):
    """_trader_stakes_today suma stakes de individuales correctamente."""
    import combo_governor as mod
    monkeypatch.setattr(mod, "REPORTS_DIR", tmp_path)

    plan = {
        "individuales": [{"favorito": "Rublev", "stake": 3000}, {"favorito": "Alcaraz", "stake": 2000}],
        "cobertura": [],
    }
    _write_json(tmp_path / "trader_plan_20260717_120000.json", plan)

    result = _trader_stakes_today("2026-07-17")
    assert result.get("MOTOR_individual") == 5000


def test_trader_stakes_suma_cobertura(tmp_path, monkeypatch):
    """_trader_stakes_today suma stakes de cobertura correctamente."""
    import combo_governor as mod
    monkeypatch.setattr(mod, "REPORTS_DIR", tmp_path)

    plan = {
        "individuales": [],
        "cobertura": [{"stake": 1500}, {"stake": 1000}],
    }
    _write_json(tmp_path / "trader_plan_20260717_120000.json", plan)

    result = _trader_stakes_today("2026-07-17")
    assert result.get("MOTOR_cobertura") == 2500


def test_trader_stakes_fecha_filtro(tmp_path, monkeypatch):
    """_trader_stakes_today solo lee planes del día indicado."""
    import combo_governor as mod
    monkeypatch.setattr(mod, "REPORTS_DIR", tmp_path)

    plan_hoy = {"individuales": [{"favorito": "X", "stake": 1000}], "cobertura": []}
    plan_ayer = {"individuales": [{"favorito": "Y", "stake": 9999}], "cobertura": []}
    _write_json(tmp_path / "trader_plan_20260717_120000.json", plan_hoy)
    _write_json(tmp_path / "trader_plan_20260716_120000.json", plan_ayer)

    result = _trader_stakes_today("2026-07-17")
    assert result.get("MOTOR_individual") == 1000  # solo hoy


def test_trader_stakes_sin_planes(tmp_path, monkeypatch):
    """_trader_stakes_today retorna dict vacío si no hay planes del día."""
    import combo_governor as mod
    monkeypatch.setattr(mod, "REPORTS_DIR", tmp_path)
    assert _trader_stakes_today("2026-07-17") == {}


# ── S107-B: _rival_value_stakes_today ────────────────────────────────────────

def test_rival_value_stakes_lee_archivo_primario(tmp_path, monkeypatch):
    """_rival_value_stakes_today lee rival_value_plan_*.json."""
    import combo_governor as mod
    monkeypatch.setattr(mod, "REPORTS_DIR", tmp_path)

    plan = {"picks": [{"jugador": "Djokovic", "stake": 2000}, {"jugador": "Medvedev", "stake": 2000}]}
    _write_json(tmp_path / "rival_value_plan_20260717_120000.json", plan)

    result = _rival_value_stakes_today("2026-07-17")
    assert result.get("RIVAL_VALUE") == 4000


def test_rival_value_stakes_fallback_apuestas(tmp_path, monkeypatch):
    """_rival_value_stakes_today cae a apuestas_*.json con tipo=RIVAL_VALUE."""
    import combo_governor as mod
    monkeypatch.setattr(mod, "REPORTS_DIR", tmp_path)

    apuestas = {"picks": [
        {"jugador": "Sinner", "stake": 3000, "tipo": "RIVAL_VALUE"},
        {"jugador": "Ruud",   "stake": 1000, "tipo": "WAS"},
    ]}
    _write_json(tmp_path / "apuestas_20260717_120000.json", apuestas)

    result = _rival_value_stakes_today("2026-07-17")
    assert result.get("RIVAL_VALUE") == 3000


def test_rival_value_stakes_sin_archivos(tmp_path, monkeypatch):
    """_rival_value_stakes_today retorna dict vacío si no hay archivos."""
    import combo_governor as mod
    monkeypatch.setattr(mod, "REPORTS_DIR", tmp_path)
    assert _rival_value_stakes_today("2026-07-17") == {}


# ── Matriz 12/12: todas las estrategias son visibles en el agregado ──────────

def test_matriz_12_12_agregado(tmp_path, monkeypatch):
    """
    Fixture con una entrada por cada estrategia (1-12) → el agregado las ve todas.
    Estrategias 2-7: combo_plan (confianza builder)
    Estrategias 8-11: betplay (apuestas_*.json)
    Estrategia 1: trader_plan (Motor)
    Estrategia 12: rival_value_plan
    """
    import combo_governor as mod
    monkeypatch.setattr(mod, "REPORTS_DIR", tmp_path)

    # Estrategia 1 — MOTOR
    _write_json(tmp_path / "trader_plan_20260717_001.json", {
        "individuales": [{"favorito": "Rublev", "stake": 1000}],
        "cobertura": [{"stake": 500}],
    })
    # Estrategia 12 — RIVAL VALUE
    _write_json(tmp_path / "rival_value_plan_20260717_001.json", {
        "picks": [{"jugador": "Djokovic", "stake": 2000}],
    })
    # Estrategias 8-11 — betplay
    _write_json(tmp_path / "apuestas_20260717_001.json", {
        "picks": [
            {"tipo": "mega",  "stake": 500},
            {"tipo": "safe",  "stake": 1000},
            {"tipo": "games", "stake": 2000},
            {"tipo": "WAS",   "stake": 5000},
        ],
    })

    motor  = _trader_stakes_today("2026-07-17")
    rival  = _rival_value_stakes_today("2026-07-17")
    betply = mod._betplay_stakes_today("2026-07-17")

    total = sum(motor.values()) + sum(rival.values()) + sum(betply.values())

    assert motor.get("MOTOR_individual") == 1000,  "MOTOR individual no encontrado"
    assert motor.get("MOTOR_cobertura")  == 500,   "MOTOR cobertura no encontrada"
    assert rival.get("RIVAL_VALUE")      == 2000,  "RIVAL_VALUE no encontrado"
    assert betply.get("mega")            == 500,   "mega no encontrado"
    assert betply.get("safe")            == 1000,  "safe no encontrado"
    assert betply.get("games")           == 2000,  "games no encontrado"
    assert betply.get("WAS")             == 5000,  "WAS no encontrado"
    assert total == 12000, f"Total 12/12 esperado 12000, got {total}"


# ── S107-C: exposicion_por_jugador ───────────────────────────────────────────

def test_exposicion_pierna_compartida_acumula(monkeypatch):
    """Jugador en 3 combos → su stake se acumula correctamente."""
    capas = [
        {"jugador": "Alcaraz C.", "stake": 2000},
        {"jugador": "Alcaraz C.", "stake": 2000},
        {"jugador": "Alcaraz C.", "stake": 2000},
        {"jugador": "Ruud C.",    "stake": 1000},
    ]
    result = exposicion_por_jugador(capas, bankroll=0)
    # Los nombres normalizados de Alcaraz deben sumar 6000
    alcaraz_total = sum(v for k, v in result.items() if "alcaraz" in k.lower())
    assert alcaraz_total == 6000, f"Alcaraz acumulado: {alcaraz_total}"


def test_exposicion_supera_cap_genera_warning():
    """Jugador con stake > 5% bankroll genera _warnings."""
    capas = [
        {"jugador": "Djokovic N.", "stake": 8000},  # 8000/100000 = 8% > 5%
        {"jugador": "Nadal R.",    "stake": 1000},
    ]
    result = exposicion_por_jugador(capas, bankroll=100_000)
    warnings = result.get("_warnings", [])
    assert warnings, "Debe haber warning por cap 5% superada"
    assert any("djokovic" in w.lower() or "8,000" in w for w in warnings)


def test_exposicion_jugadores_distintos_no_dispara():
    """Jugadores distintos sin concentración no generan warnings."""
    capas = [
        {"jugador": "Sinner J.",   "stake": 2000},
        {"jugador": "Medvedev D.", "stake": 2000},
        {"jugador": "Zverev A.",   "stake": 2000},
    ]
    result = exposicion_por_jugador(capas, bankroll=125_000)
    warnings = result.get("_warnings", [])
    assert not warnings, f"No debe haber warnings: {warnings}"


def test_exposicion_sin_picks_retorna_vacio():
    """Lista vacía de picks retorna dict vacío (sin warnings)."""
    result = exposicion_por_jugador([], bankroll=125_000)
    assert "_warnings" not in result
    assert result == {}
