"""
tests/test_nodo168_fix_steam_linea_actual.py — REGLA-T53 Nodo-168

D168-01: _write_games_odds_history() — el guard de deduplicación (evita puntos
repetidos en el historial) saltaba también el cálculo de steam cuando la cuota
no cambió respecto al último punto grabado — condición que ocurre la mayoría
de los ciclos de refresh de 15s. Bug reproducido con evidencia real del
dashboard ("STEAM: -, en toda las filas").

D168-02: itf_live_signals nunca asignaba linea_actual/cuota_actual/oc_id_actual
al dict `best` dentro de _check_games_convergencia() — D158-01 (_fetch_live_games_all)
resuelve esto solo para alta_signals. _build_x3_games() ya lee esas 3 claves
desde itf_s (Nodo-167 D167-01), pero nunca estaban pobladas en el origen.
_check_games_convergencia() es una función monolítica con I/O en vivo real
(HTTP a Kambi) — no aislable sin mockear toda la cadena. Se verifica con
lectura estática del bloque real de código (mismo patrón que Nodo-162:
"regresión estática, lee el archivo real"), acotada a las líneas del loop
ITF (no las de alta_signals, que ya tenían el fix D158-01 antes de este nodo).
"""
import ast
import json
import re
import sys
from datetime import datetime as _real_datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import live_desk as ld


class _FixedDatetime(_real_datetime):
    @classmethod
    def now(cls, tz=None):
        return _real_datetime(2026, 8, 2, 10, 20)


# ── D168-01: steam sobrevive al dedup-guard ──────────────────────────────────

def test_168_01_steam_se_calcula_pese_a_tick_identico_al_ultimo_punto(tmp_path, monkeypatch):
    """Reproduce el bug exacto: historial con 4 puntos previos (suficiente para
    velocity_zscore) y el tick actual IDÉNTICO al último punto grabado (misma
    cuota, mismos games_played) — condición real de la mayoría de los ciclos
    de 15s. Antes del fix, el `continue` de deduplicación saltaba el cálculo
    de steam por completo; después del fix, debe calcularse igual."""
    monkeypatch.setattr(ld, "REPORTS", tmp_path)
    monkeypatch.setattr(ld, "datetime", _FixedDatetime)
    fecha_compact = "20260802"

    hist_path = tmp_path / f"games_odds_history_{fecha_compact}.json"
    pk = "A vs B_UNDER"
    hist_path.write_text(json.dumps({
        pk: [
            {"ts": "10:00", "cuota": 1.95, "games_played": 2},
            {"ts": "10:05", "cuota": 1.90, "games_played": 4},
            {"ts": "10:10", "cuota": 1.85, "games_played": 6},
            {"ts": "10:15", "cuota": 1.55, "games_played": 7},
        ]
    }), encoding="utf-8")

    # tick actual == último punto grabado (10:15) → dispara el guard de dedup
    sig = {
        "partido": "A vs B", "direccion": "UNDER", "estado": "EN_VIVO",
        "cuota_live": 1.55, "score_data": {"games_played": 7},
    }
    ld._write_games_odds_history([sig], fecha_compact)

    assert sig.get("steam_z") is not None, (
        "steam_z debe calcularse aunque el tick actual repita el último punto "
        "del historial — el guard de dedup solo debe bloquear el append, no el cálculo"
    )


def test_168_02_dedup_sigue_sin_duplicar_puntos_en_el_historial(tmp_path, monkeypatch):
    """Control: el fix D168-01 no debe romper la deduplicación original — un
    tick idéntico al último punto NO debe agregarse de nuevo a la lista."""
    monkeypatch.setattr(ld, "REPORTS", tmp_path)
    monkeypatch.setattr(ld, "datetime", _FixedDatetime)
    fecha_compact = "20260802"

    hist_path = tmp_path / f"games_odds_history_{fecha_compact}.json"
    pk = "A vs B_UNDER"
    hist_path.write_text(json.dumps({
        pk: [{"ts": "10:00", "cuota": 1.95, "games_played": 2}]
    }), encoding="utf-8")

    sig = {
        "partido": "A vs B", "direccion": "UNDER", "estado": "EN_VIVO",
        "cuota_live": 1.95, "score_data": {"games_played": 2},
    }
    ld._write_games_odds_history([sig], fecha_compact)

    hist = json.loads(hist_path.read_text(encoding="utf-8"))
    assert len(hist[pk]) == 1, "tick duplicado no debe agregar un segundo punto"


def test_168_03_regresion_sin_historial_suficiente_no_anota_steam(tmp_path, monkeypatch):
    """Regresión directa de test_160_21 (Nodo-160) — debe seguir sin crashear
    ni anotar steam cuando hay menos de 3 puntos totales."""
    monkeypatch.setattr(ld, "REPORTS", tmp_path)
    fecha_compact = "20260802"

    sig = {
        "partido": "C vs D", "direccion": "OVER", "estado": "EN_VIVO",
        "cuota_live": 1.90, "score_data": {"games_played": 3},
    }
    ld._write_games_odds_history([sig], fecha_compact)

    assert "steam_confirmado" not in sig


def test_168_04_steam_confirmado_con_caida_fuerte_sigue_funcionando(tmp_path, monkeypatch):
    """Regresión directa de test_160_20 (Nodo-160) — el caso donde SÍ hay un
    punto nuevo (tick distinto al último) debe seguir marcando steam_confirmado."""
    monkeypatch.setattr(ld, "REPORTS", tmp_path)
    monkeypatch.setattr(ld, "datetime", _FixedDatetime)
    fecha_compact = "20260802"

    hist_path = tmp_path / f"games_odds_history_{fecha_compact}.json"
    pk = "A vs B_UNDER"
    hist_path.write_text(json.dumps({
        pk: [
            {"ts": "10:00", "cuota": 1.95, "games_played": 2},
            {"ts": "10:05", "cuota": 1.94, "games_played": 4},
            {"ts": "10:10", "cuota": 1.955, "games_played": 6},
            {"ts": "10:15", "cuota": 1.94, "games_played": 7},
        ]
    }), encoding="utf-8")

    sig = {
        "partido": "A vs B", "direccion": "UNDER", "estado": "EN_VIVO",
        "cuota_live": 1.55, "score_data": {"games_played": 8},
    }
    ld._write_games_odds_history([sig], fecha_compact)

    assert sig["steam_confirmado"] is True
    assert sig["steam_signal"] == "STEAM"


# ── D168-02: linea_actual/cuota_actual/oc_id_actual poblados para ITF ───────

def _extraer_bloque_check_games_convergencia():
    """Extrae el source completo de _check_games_convergencia() usando ast
    (mismo módulo importado, no una copia en disco) para acotar la búsqueda
    solo a esa función — evita falsos positivos con el bloque D158-01 de
    alta_signals que ya tenía estas 3 asignaciones antes de Nodo-168."""
    src = Path(ld.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_check_games_convergencia":
            start = node.lineno - 1
            end = node.end_lineno
            return "\n".join(lines[start:end])
    raise AssertionError("_check_games_convergencia no encontrada en live_desk.py")


def _bloque_itf_live_signals(func_src):
    """Dentro del source de _check_games_convergencia, acota al bloque que
    construye itf_live_signals (desde 'market = _fetch_live_games_all' hasta
    'itf_live_signals.append(best)') — excluye el bloque D158-01 de alta_signals
    que está antes en el mismo archivo."""
    start_marker = "market = _fetch_live_games_all"
    end_marker = "itf_live_signals.append(best)"
    start = func_src.index(start_marker)
    end = func_src.index(end_marker, start)
    return func_src[start:end]


def test_168_05_itf_best_dict_asigna_linea_actual_cuota_actual_oc_id_actual():
    """Verificación estática del código real (no una copia hardcodeada): el
    bloque que construye `best` dentro del loop ITF de _check_games_convergencia
    debe asignar linea_actual/cuota_actual/oc_id_actual usando las variables
    market_linea/cuota_val/market.get(oc_k) ya frescas ese ciclo (mismo fetch
    _fetch_live_games_all que D158-01 usa para alta_signals) — sin esto,
    _build_x3_games() (que sí lee itf_s.get("linea_actual") desde Nodo-167)
    siempre recibe None para señales ITF_VIVO."""
    func_src = _extraer_bloque_check_games_convergencia()
    itf_block = _bloque_itf_live_signals(func_src)

    assert re.search(r'"linea_actual"\s*:\s*market_linea', itf_block), (
        "best dict debe asignar linea_actual = market_linea (valor ya fresco "
        "de _fetch_live_games_all, sin fetch adicional)"
    )
    assert re.search(r'"cuota_actual"\s*:\s*cuota_val', itf_block), (
        "best dict debe asignar cuota_actual = cuota_val"
    )
    assert re.search(r'"oc_id_actual"\s*:\s*market\.get\(oc_k\)', itf_block), (
        "best dict debe asignar oc_id_actual = market.get(oc_k)"
    )


def test_168_06_build_x3_games_propaga_linea_actual_itf_una_vez_poblado(tmp_path, monkeypatch):
    """Integración con el fix ya wireado en Nodo-167: una vez que itf_live_signals
    trae linea_actual/cuota_actual/oc_id_actual poblados (D168-02), la función
    REAL _build_x3_games() (sin mocks, lee de disco) debe propagarlos al dict
    de señal final vía la ruta ITF_VIVO (líneas 617-619) — cierra el loop
    completo: origen (D168-02) → propagación (D167-01, ya verificada en
    Nodo-167) → disponible para el render."""
    monkeypatch.setattr(ld, "REPORTS", tmp_path)
    fecha = "2026-08-04"
    itf_signal = {
        "partido": "Jugador A vs Jugador B",
        "direccion": "UNDER",
        "linea": 22.5,
        "linea_actual": 21.5,
        "cuota_actual": 1.87,
        "oc_id_actual": 555111222,
        "cuota_live": 1.87,
        "estado": "ITF_VIVO",
        "event_id": 12345,
    }
    gl_path = tmp_path / "games_live_20260804.json"
    gl_path.write_text(json.dumps({"signals_alta": [itf_signal]}), encoding="utf-8")

    x3 = ld._build_x3_games(fecha)
    señal = x3["signals"][0]
    assert señal["linea_actual"] == 21.5
    assert señal["cuota_actual"] == 1.87
    assert señal["oc_id_actual"] == 555111222
