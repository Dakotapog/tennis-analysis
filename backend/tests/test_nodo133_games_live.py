"""
tests/test_nodo133_games_live.py — REGLA-T53 Nodo-133 Games Live Convergencia

5 tests:
  1. test_clasifica_pre_partido   — hora futura → estado PRE_PARTIDO
  2. test_clasifica_en_vivo       — partido en mock STARTED → EN_VIVO
  3. test_clasifica_terminado     — hora pasada + no en STARTED → TERMINADO
  4. test_convergencia_2_alta     — 2 señales EN_VIVO → Popen llamado 1x
  5. test_antiflood_no_refire     — combo_key ya en fired → Popen no llamado

REGLA-T53: invocan funciones reales del módulo live_desk — nunca hardcodean lógica.
"""
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Añadir backend/ al path si es necesario
sys.path.insert(0, str(Path(__file__).parent.parent))

import live_desk as ld


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_gsr(partidos, tmp_path, gap_juegos=-2.3):
    """Genera un games_signal_report mínimo con señales ALTA."""
    apostar = []
    for p in partidos:
        apostar.append({
            "partido": p["partido"],
            "hora":    p.get("hora", ""),
            "kambi_event_id": p.get("event_id"),
            "games_range": "20-24",
            "señales_optimas": [{
                "apostar": True,
                "confianza_señal": "ALTA",
                "direccion": "UNDER",
                "linea": 21.5,
                "cuota": 1.75,
                "mercado": "juegos",
                "gap_juegos": gap_juegos,
            }]
        })
    data = {"metadata": {"n_partidos": len(partidos), "n_apostar": len(partidos)},
            "apostar": apostar}
    fecha_compact = datetime.now().strftime("%Y%m%d")
    path = tmp_path / f"games_signal_report_{fecha_compact}_test.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ─── Test 1: PRE_PARTIDO ─────────────────────────────────────────────────────

def test_clasifica_pre_partido(tmp_path):
    """Hora futura (+1h) y no en STARTED → estado == PRE_PARTIDO."""
    hora_futura = (datetime.utcnow() + timedelta(hours=1)).strftime("%H:%M")
    _make_gsr([{"partido": "Alcaraz vs Sinner", "hora": hora_futura}], tmp_path)

    fecha = datetime.now().strftime("%Y-%m-%d")
    with patch.object(ld, "REPORTS", tmp_path), \
         patch("live_desk._kambi_started_events", return_value=[]):
        ld._check_games_convergencia(fecha)

    gl_path = tmp_path / f"games_live_{fecha.replace('-', '')}.json"
    assert gl_path.exists(), "games_live debe escribirse"
    gl = json.loads(gl_path.read_text())
    assert gl["signals_alta"][0]["estado"] == "PRE_PARTIDO"
    assert gl["en_vivo_count"] == 0
    assert gl["convergencia_activa"] is False


# ─── Test 2: EN_VIVO ─────────────────────────────────────────────────────────

def test_clasifica_en_vivo(tmp_path):
    """Partido matcheado en mock STARTED events → estado == EN_VIVO."""
    _make_gsr([{"partido": "Alcaraz S. vs Sinner J.", "event_id": 9999}], tmp_path)

    mock_event = {
        "event": {"id": 9999, "state": "STARTED", "homeName": "Alcaraz S.", "awayName": "Sinner J."},
        "betOffers": [],
    }

    fecha = datetime.now().strftime("%Y-%m-%d")
    with patch.object(ld, "REPORTS", tmp_path), \
         patch("live_desk._kambi_started_events", return_value=[mock_event]):
        ld._check_games_convergencia(fecha)

    gl = json.loads((tmp_path / f"games_live_{fecha.replace('-', '')}.json").read_text())
    assert gl["signals_alta"][0]["estado"] == "EN_VIVO"
    assert gl["en_vivo_count"] == 1


# ─── Test 3: TERMINADO ───────────────────────────────────────────────────────

def test_clasifica_terminado(tmp_path):
    """Hora pasada (>130min) y no en STARTED → estado == TERMINADO."""
    hora_pasada = (datetime.utcnow() - timedelta(hours=3)).strftime("%H:%M")
    _make_gsr([{"partido": "Djokovic vs Nadal", "hora": hora_pasada}], tmp_path)

    fecha = datetime.now().strftime("%Y-%m-%d")
    with patch.object(ld, "REPORTS", tmp_path), \
         patch("live_desk._kambi_started_events", return_value=[]):
        ld._check_games_convergencia(fecha)

    gl = json.loads((tmp_path / f"games_live_{fecha.replace('-', '')}.json").read_text())
    assert gl["signals_alta"][0]["estado"] == "TERMINADO"


# ─── Test 4: Convergencia por pierna (D166-01) → Popen ──────────────────────

def test_convergencia_2_alta_dispara(tmp_path):
    """D166-01: reemplaza el disparo agregado en_vivo_count>=2 por gate por
    pierna (convergencia_score>=3). 2 señales EN_VIVO con gap fuerte
    (abs>=4.0 → +2) y cuota_live>=2.00 (+1) alcanzan score=3 individualmente
    → convergencia_activa=True → Popen llamado 1x."""
    _make_gsr([
        {"partido": "Alcaraz vs Sinner",  "event_id": 1001},
        {"partido": "Djokovic vs Medvedev", "event_id": 1002},
    ], tmp_path, gap_juegos=-4.5)

    mock_events = [
        {"event": {"id": 1001, "state": "STARTED", "homeName": "Alcaraz", "awayName": "Sinner"}, "betOffers": []},
        {"event": {"id": 1002, "state": "STARTED", "homeName": "Djokovic", "awayName": "Medvedev"}, "betOffers": []},
    ]
    mock_mkt = {"linea": 21.5, "cuota_under": 2.00, "cuota_over": 1.80,
                "oc_id_under": "OC1", "oc_id_over": "OC2"}

    fecha = datetime.now().strftime("%Y-%m-%d")
    with patch.object(ld, "REPORTS", tmp_path), \
         patch("live_desk._kambi_started_events", return_value=mock_events), \
         patch("live_desk._fetch_live_games_all", return_value=mock_mkt), \
         patch("live_desk.subprocess.Popen") as mock_popen:
        ld._check_games_convergencia(fecha)

    gl = json.loads((tmp_path / f"games_live_{fecha.replace('-', '')}.json").read_text())
    assert gl["signals_alta"][0]["convergencia_score"] >= 3
    assert gl["signals_alta"][1]["convergencia_score"] >= 3
    assert gl["convergencia_activa"] is True
    assert gl["en_vivo_count"] == 2
    mock_popen.assert_called_once()
    # Verificar fired escrito
    fired_path = tmp_path / f"games_live_{fecha.replace('-', '')}_fired.json"
    assert fired_path.exists()


def test_convergencia_en_vivo_sin_score_no_dispara(tmp_path):
    """D166-01: control negativo — 2 señales EN_VIVO pero SIN datos suficientes
    para convergencia_score>=3 (mercado sin cuota, gap débil) → ya NO dispara
    solo por el conteo (comportamiento D133 viejo reemplazado)."""
    _make_gsr([
        {"partido": "Alcaraz vs Sinner",  "event_id": 1001},
        {"partido": "Djokovic vs Medvedev", "event_id": 1002},
    ], tmp_path, gap_juegos=-1.0)  # abs<2.0 → +0 en score

    mock_events = [
        {"event": {"id": 1001, "state": "STARTED", "homeName": "Alcaraz", "awayName": "Sinner"}, "betOffers": []},
        {"event": {"id": 1002, "state": "STARTED", "homeName": "Djokovic", "awayName": "Medvedev"}, "betOffers": []},
    ]

    fecha = datetime.now().strftime("%Y-%m-%d")
    with patch.object(ld, "REPORTS", tmp_path), \
         patch("live_desk._kambi_started_events", return_value=mock_events), \
         patch("live_desk._fetch_live_games_all", return_value=None), \
         patch("live_desk.subprocess.Popen") as mock_popen:
        ld._check_games_convergencia(fecha)

    gl = json.loads((tmp_path / f"games_live_{fecha.replace('-', '')}.json").read_text())
    assert gl["en_vivo_count"] == 2
    assert gl["convergencia_activa"] is False
    mock_popen.assert_not_called()


# ─── Test 5: Anti-flood no refire ────────────────────────────────────────────

def test_antiflood_no_refire(tmp_path):
    """frozenset {A,B} ya en fired → Popen NO llamado segunda vez. D166-01:
    ambas señales deben calificar (score>=3) para que combo_key real
    coincida con el fired pre-poblado — si no calificaran, Popen tampoco se
    llamaría, pero por falta de score, no por el anti-flood bajo prueba."""
    fecha = datetime.now().strftime("%Y-%m-%d")
    fecha_compact = fecha.replace("-", "")

    _make_gsr([
        {"partido": "Alcaraz vs Sinner",  "event_id": 1001},
        {"partido": "Djokovic vs Medvedev", "event_id": 1002},
    ], tmp_path, gap_juegos=-4.5)

    # Pre-poblar fired con ese combo
    fired_key = sorted(["Alcaraz vs Sinner", "Djokovic vs Medvedev"])
    fired_path = tmp_path / f"games_live_{fecha_compact}_fired.json"
    fired_path.write_text(json.dumps([fired_key]), encoding="utf-8")

    mock_events = [
        {"event": {"id": 1001, "state": "STARTED", "homeName": "Alcaraz", "awayName": "Sinner"}, "betOffers": []},
        {"event": {"id": 1002, "state": "STARTED", "homeName": "Djokovic", "awayName": "Medvedev"}, "betOffers": []},
    ]
    mock_mkt = {"linea": 21.5, "cuota_under": 2.00, "cuota_over": 1.80,
                "oc_id_under": "OC1", "oc_id_over": "OC2"}

    with patch.object(ld, "REPORTS", tmp_path), \
         patch("live_desk._kambi_started_events", return_value=mock_events), \
         patch("live_desk._fetch_live_games_all", return_value=mock_mkt), \
         patch("live_desk.subprocess.Popen") as mock_popen:
        ld._check_games_convergencia(fecha)

    mock_popen.assert_not_called()  # Anti-flood funcionó
