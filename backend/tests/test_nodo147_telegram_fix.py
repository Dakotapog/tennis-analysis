"""
tests/test_nodo147_telegram_fix.py — REGLA-T53 D147-07/D157-05

Bug real encontrado 2026-08-02 con evidencia de logs/live_desk.log: `_fire_certeza_alert`
(D147-06) llamaba a `scripts/send_telegram.py`, que nunca existió en el repo ni en su
historial git — el guard `.exists()` lo saltaba en silencio. 6 disparos reales de
CERTEZA MATEMATICA quedaron sin notificar (0 menciones de "telegram" en 8.5MB de log).
El combo ITF live (`_fire_itf_live_games_combo`, 135 disparos en el mismo log) nunca
tuvo integración Telegram en absoluto.

Fix: `_send_telegram_async()` — helper que llama a `utils.telegram._enviar_telegram()`
(el bot real, mismo TG_TOKEN que betplay_combo_builder.py/combo_confianza_builder.py,
confirmado funcional para el sistema de favoritos) en un thread daemon, sin bloquear
el loop de 15s. Wireado en ambos call-sites: `_fire_certeza_alert` (D147-06) y el
guard de señal-nueva del combo ITF live (D157-05).
"""
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import live_desk as ld


def test_147_20_send_telegram_async_llama_enviar_telegram_real(monkeypatch):
    """_send_telegram_async debe invocar utils.telegram._enviar_telegram (el bot
    real), no un script inexistente."""
    calls = []
    monkeypatch.setattr("utils.telegram._enviar_telegram", lambda msg: calls.append(msg) or True)

    ld._send_telegram_async("mensaje de prueba", tag="TEST")

    for _ in range(40):
        if calls:
            break
        time.sleep(0.05)

    assert calls == ["mensaje de prueba"]


def test_147_21_send_telegram_async_no_lanza_si_falla(monkeypatch):
    """Errores de red no deben propagarse al loop de 15s (fire-and-forget)."""
    def _boom(msg):
        raise RuntimeError("network down")
    monkeypatch.setattr("utils.telegram._enviar_telegram", _boom)

    ld._send_telegram_async("mensaje", tag="TEST")
    time.sleep(0.1)  # dar tiempo al thread daemon — no debe crashear el proceso


def test_147_22_fire_certeza_alert_dispara_telegram_real(tmp_path, monkeypatch):
    """D147-07 fix: _fire_certeza_alert ya no referencia scripts/send_telegram.py
    (inexistente) — debe llamar a _send_telegram_async con el mensaje CERTEZA
    MATEMATICA."""
    sent = []
    monkeypatch.setattr(ld, "_send_telegram_async", lambda msg, tag="": sent.append((msg, tag)))

    sig = {
        "partido": "Jodar R. vs Musetti L.", "direccion": "OVER", "linea_t0": 20.5,
        "score_data": {"games_played": 21},
    }
    with patch.object(ld, "REPORTS", tmp_path):
        ld._fire_certeza_alert(sig, "20260802")

    assert len(sent) == 1
    msg, tag = sent[0]
    assert "CERTEZA MATEMATICA" in msg
    assert "Jodar R. vs Musetti L." in msg
    assert "OVER 20.5" in msg
    assert "21 juegos jugados" in msg
    assert tag == "D147-06"


def test_147_23_fire_certeza_alert_fire_once_no_reenvia(tmp_path, monkeypatch):
    """Guard de disparo único (certeza_fired_*.json) sigue intacto — el fix no
    debe reintroducir notificaciones repetidas cada 15s."""
    sent = []
    monkeypatch.setattr(ld, "_send_telegram_async", lambda msg, tag="": sent.append(msg))

    sig = {"partido": "A vs B", "direccion": "UNDER", "linea": 22.5, "score_data": {}}
    with patch.object(ld, "REPORTS", tmp_path):
        ld._fire_certeza_alert(sig, "20260802")
        ld._fire_certeza_alert(sig, "20260802")
        ld._fire_certeza_alert(sig, "20260802")

    assert len(sent) == 1
    guard = json.loads((tmp_path / "certeza_fired_20260802.json").read_text(encoding="utf-8"))
    assert "A vs B_UNDER" in guard


def test_147_24_no_referencia_script_inexistente():
    """Regresión: nadie debe reintroducir la construcción de path hacia
    scripts/send_telegram.py (confirmado inexistente: 0 resultados en
    git log --all -- '*send_telegram*'). El comentario que documenta el bug
    histórico sí puede mencionar el nombre del archivo — lo que no debe
    reaparecer es el patrón de código que lo invoca."""
    src = Path(ld.__file__).read_text(encoding="utf-8")
    assert '"scripts" / "send_telegram.py"' not in src
    assert "send_script.exists()" not in src
