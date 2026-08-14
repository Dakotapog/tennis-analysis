"""
tests/test_nodo116_antiflood.py — REGLA-T53: Nodo-116 anti-flood + D116-03.

§B tests (5):
  B1. 2do fire mismo event_id → no dispara (de-dup _fired.json)
  B2. Fire #11 → no dispara, retorna mensaje CAP ALCANZADO
  B3. TTL borra .bat de partido iniciado (hora_inicio+15min pasó), respeta futuro
  B4. _generar_html_bat_live escribe en combos_live/ (assert NO Desktop en path)
  B5. _build_combo_live → fila tipo COMBO_LIVE en desk state

§C tests (3):
  C1. best_price con 1 feed plano (schema zita D116-03) → casa única OK
  C2. best_price con 3 feeds → elige la mayor (función pura ya acepta N feeds)
  C3. [skip D116-03] P8 render con feeds multi-casa → N columnas dinámicas
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND = Path(__file__).parent.parent
sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_BACKEND / "scripts"))

from live_edge_monitor import (
    MAX_LIVE_FIRES_DIA,
    _combos_live_dir,
    _fire_break_combos,
    _generar_html_bat_live,
    _ttl_cleanup,
)
from live_desk import _build_combo_live


# ── B1: De-dup por event_id ─────────────���──────────────────────────────────────

def test_dedup_segundo_fire_mismo_event(tmp_path):
    """2do fire mismo partido → no llama subprocess (de-dup _fired.json)."""
    fired_existente = {
        "Alcaraz vs Djokovic": {
            "fired_at": "2099-01-01T10:00:00",
            "hora_inicio": "14:00",
            "drift_pct": 18.0,
        }
    }
    triggers = [{
        "partido": "Alcaraz vs Djokovic",
        "cuota_pre": 1.80, "cuota_live": 1.50,
        "drift_pct": 18.0, "edge_live": 0.08,
        "_fired_prev": False,
    }]

    with patch("live_edge_monitor._BASE_DIR", tmp_path), \
         patch("live_edge_monitor._load_fired", return_value=fired_existente), \
         patch("live_edge_monitor._save_fired") as mock_save, \
         patch("live_edge_monitor._ttl_cleanup"), \
         patch("subprocess.run") as mock_run:
        result = _fire_break_combos(triggers)

    mock_run.assert_not_called()
    mock_save.assert_not_called()
    assert result is None, "De-dup: 2do fire del mismo evento no debe disparar"


# ── B2: Cap diario ─────────────────────────────────────────────────────────────

def test_cap_diario_fire_11_no_dispara(tmp_path):
    """Fire #11 → no llama subprocess, retorna mensaje CAP ALCANZADO."""
    fired_lleno = {
        f"Partido{i}": {"fired_at": "2099-01-01T10:00:00", "hora_inicio": "", "drift_pct": 15.0}
        for i in range(MAX_LIVE_FIRES_DIA)  # 10 entradas = cap completo
    }
    triggers = [{
        "partido": "PartidoNuevo",
        "cuota_pre": 2.0, "cuota_live": 1.65,
        "drift_pct": 17.5, "edge_live": 0.09,
        "_fired_prev": False,
    }]

    with patch("live_edge_monitor._BASE_DIR", tmp_path), \
         patch("live_edge_monitor._load_fired", return_value=fired_lleno), \
         patch("live_edge_monitor._ttl_cleanup"), \
         patch("subprocess.run") as mock_run:
        result = _fire_break_combos(triggers)

    mock_run.assert_not_called()
    assert result is not None
    assert "CAP" in result, "Debe indicar CAP ALCANZADO"
    assert str(MAX_LIVE_FIRES_DIA) in result, "Debe incluir el número del cap"


# ─�� B3: TTL cleanup ─────────────────��──────────────────────────────────────────

def test_ttl_borra_partido_iniciado_respeta_futuro(tmp_path):
    """TTL borra .bat cuyo partido empezó hace >15min; respeta el que aún no empieza."""
    fecha = "2099-01-01"
    # 15:30 — partido_pasado empezó a las 13:00 (13:15 deadline ya pasó)
    #          partido_futuro empieza a las 16:00 (16:15 deadline no llegó)
    now = datetime(2099, 1, 1, 15, 30, 0)

    fired = {
        "bat_pasado": {"fired_at": "2099-01-01T13:00:00", "hora_inicio": "13:00", "drift_pct": 15.0},
        "bat_futuro": {"fired_at": "2099-01-01T15:00:00", "hora_inicio": "16:00", "drift_pct": 15.0},
    }

    with patch("live_edge_monitor._BASE_DIR", tmp_path):
        combos_dir = _combos_live_dir(fecha)
        bat_pasado = combos_dir / "LiveCombo_bat_pasado.bat"
        bat_futuro = combos_dir / "LiveCombo_bat_futuro.bat"
        bat_pasado.write_text("@echo off\r\n", encoding="utf-8")
        bat_futuro.write_text("@echo off\r\n", encoding="utf-8")

        with patch("live_edge_monitor._load_fired", return_value=fired):
            _ttl_cleanup(fecha, now)

    assert not bat_pasado.exists(), ".bat expirado debe haberse borrado"
    assert bat_futuro.exists(), ".bat futuro debe conservarse"


# ── B4: Output a combos_live/, CERO Desktop ─────────���─────────────────────────

def test_generar_html_bat_no_desktop(tmp_path):
    """_generar_html_bat_live escribe en output_dir, NO en Desktop."""
    out_dir = tmp_path / "combos_live" / "2099-01-01"
    out_dir.mkdir(parents=True)

    triggers = [{
        "partido": "Alcaraz vs Djokovic",
        "cuota_pre": 1.80, "cuota_live": 1.50,
        "drift_pct": 18.0, "edge_live": 0.08,
        "betplay_url": "",
    }]
    result = _generar_html_bat_live(triggers, "20990101_1500", output_dir=out_dir)

    assert result is not None
    assert "Desktop" not in result, ".bat NO debe apuntar al Desktop"

    bat_files = list(out_dir.glob("*.bat"))
    assert len(bat_files) == 1, "Debe generarse exactamente 1 .bat"
    bat_content = bat_files[0].read_text(encoding="utf-8")
    assert "Desktop" not in bat_content, "Contenido del .bat NO debe referenciar Desktop"


# ── B5: COMBO_LIVE fila en desk ─────────────────��──────────────────────────────

def test_combo_live_fila_tipo_correcto(tmp_path):
    """_build_combo_live lee _fired.json y retorna fila con tipo=COMBO_LIVE."""
    fecha = "2099-01-01"
    combos_dir = tmp_path / "combos_live" / fecha
    combos_dir.mkdir(parents=True)

    fired = {
        "Alcaraz vs Djokovic": {
            "fired_at": "2099-01-01T14:00:00",
            "hora_inicio": "14:30",
            "drift_pct": 18.5,
        }
    }
    (combos_dir / "_fired.json").write_text(
        json.dumps(fired), encoding="utf-8"
    )

    with patch("live_desk.REPORTS", tmp_path):
        rows = _build_combo_live(fecha)

    assert len(rows) == 1
    assert rows[0]["tipo"] == "COMBO_LIVE"
    assert rows[0]["jugador"] == "Alcaraz vs Djokovic"
    assert rows[0]["hipotesis"] == "H100-01"
    assert "BREAK_CONFIRMADO" in rows[0].get("señales_activas", [])


# ── C1: best_price con feed plano (schema zita sin desglose) ──────────────────

def test_best_price_feed_plano_una_casa():
    """
    D116-03 verificado: zita schema solo tiene cuota1/cuota2, sin desglose por casa.
    best_price() con 1 feed plano retorna esa única casa sin error.
    """
    from scraping.dual_book_client import best_price

    # Simular feed plano del tipo que produce dual_book_client hoy
    # {nombre_lower: {"odds": float, "casa": "flashscore"}}
    feed_plano = {
        "alcaraz": {"odds": 1.55, "casa": "flashscore"},
        "djokovic": {"odds": 2.30, "casa": "flashscore"},
    }
    feeds = {"flashscore": feed_plano}

    resultado = best_price("alcaraz", feeds)

    assert resultado is not None
    assert resultado["casa"] == "flashscore"
    assert resultado["cuota"] == 1.55
    assert resultado["delta_pct"] == 0.0  # única casa → sin divergencia


# ── C2: best_price con 3 feeds elige la mayor ───��─────────────────────────────

def test_best_price_3_feeds_elige_mayor():
    """
    best_price() ya acepta N feeds (funciones puras intactas — Nodo-111).
    Con 3 feeds ficticios elige la cuota más alta.
    """
    from scraping.dual_book_client import best_price

    feeds = {
        "betplay":   {"alcaraz": {"odds": 1.55}},
        "wplay":     {"alcaraz": {"odds": 1.62}},  # mejor
        "rushbet":   {"alcaraz": {"odds": 1.58}},
    }

    resultado = best_price("alcaraz", feeds)

    assert resultado is not None
    assert resultado["casa"] == "wplay", "Debe elegir la casa con mayor cuota"
    assert resultado["cuota"] == 1.62
    assert resultado["delta_pct"] > 0  # hay divergencia entre casas


# ── C3: P8 render multi-columna (gateado D116-03) ────────────────────────────

@pytest.mark.skip(
    reason="D116-03 pendiente — schema zita no tiene desglose por casa. "
           "Implementar cuando scraper Nodo-48 emita {casa: cuota} por partido. "
           "Schema real verificado 2026-07-18: solo cuota1/cuota2, sin campo 'casas'."
)
def test_p8_render_n_casas_n_columnas():
    """P8 render con 3 feeds → 3 columnas dinámicas en HTML del desk."""
    pass
