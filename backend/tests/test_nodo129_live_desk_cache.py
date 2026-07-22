"""
Nodo-129 — REGLA-T53: tests para cache en memoria + POST /api/refresh + staleness
D129-01: _get_cached_state() retorna hit sin reconstruir cuando cache es fresco.
D129-02: POST /api/refresh invalida cache (ts=None) → próxima llamada reconstruye.
D129-03: _data_freshness() retorna string "hace Xm Ys" basado en mtime real.
"""
import os
import sys
import time
import tempfile
import threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from live_desk import _STATE_CACHE, _get_cached_state, _data_freshness


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reset_cache():
    with _STATE_CACHE["lock"]:
        _STATE_CACHE["state"] = None
        _STATE_CACHE["ts"] = None


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_cache_memoria_hit(monkeypatch):
    """
    D129-01: Segunda llamada a _get_cached_state() usa cache — no llama build_desk_state().
    Verifica que la segunda llamada retorna en <0.1s (cache hit, sin reconstrucción).
    """
    _reset_cache()
    call_count = [0]

    def _fake_build(fecha):
        call_count[0] += 1
        return {"fecha": fecha, "ts": "fake", "_fake": True}

    monkeypatch.setattr("live_desk.build_desk_state", _fake_build)

    fecha = "2026-07-21"
    r1 = _get_cached_state(fecha)
    assert r1["_fake"] is True
    assert call_count[0] == 1, "Primera llamada debe reconstruir"

    t0 = time.time()
    r2 = _get_cached_state(fecha)
    elapsed = time.time() - t0

    assert r2 is r1, "Cache hit debe retornar la misma referencia"
    assert call_count[0] == 1, "Segunda llamada NO debe reconstruir (cache hit)"
    assert elapsed < 0.1, f"Cache hit debe ser <0.1s, fue {elapsed:.3f}s"


def test_refresh_endpoint_invalida_cache(monkeypatch):
    """
    D129-02: Tras poblar cache, forzar _STATE_CACHE['ts']=None simula POST /api/refresh.
    La próxima llamada a _get_cached_state() debe reconstruir (cache miss).
    """
    _reset_cache()
    call_count = [0]

    def _fake_build(fecha):
        call_count[0] += 1
        return {"fecha": fecha, "build_n": call_count[0]}

    monkeypatch.setattr("live_desk.build_desk_state", _fake_build)

    fecha = "2026-07-21"
    _get_cached_state(fecha)
    assert call_count[0] == 1

    # Simular POST /api/refresh (invalidar cache)
    with _STATE_CACHE["lock"]:
        _STATE_CACHE["ts"] = None

    _get_cached_state(fecha)
    assert call_count[0] == 2, "Tras invalidación, debe reconstruir"


def test_staleness_mtime(tmp_path):
    """
    D129-03: _data_freshness() retorna 'datos de hace Xm Ys'.
    Con archivo mtime = ahora - 135s → debe retornar 'datos de hace 2m 15s'.
    """
    import live_desk as ld

    # Crear archivo temporal con mtime = ahora - 135s
    fake_file = tmp_path / "edge_report_20260721_0001.json"
    fake_file.write_text("{}")
    target_mtime = time.time() - 135
    os.utime(str(fake_file), (target_mtime, target_mtime))

    # Monkeypatch REPORTS para apuntar a tmp_path
    original_reports = ld.REPORTS
    ld.REPORTS = tmp_path
    try:
        result = _data_freshness("2026-07-21")
    finally:
        ld.REPORTS = original_reports

    assert "hace" in result, f"Debe contener 'hace', got: {result}"
    assert "2m" in result, f"Debe indicar 2 minutos, got: {result}"
    assert "15s" in result or "14s" in result or "16s" in result, \
        f"Debe indicar ~15 segundos (tolerancia ±1s), got: {result}"
