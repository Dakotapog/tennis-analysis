"""
Nodo-176 D176-01: picks_mas_cerca() pública en funnel_report.py, expuesta al
dashboard vía _build_que_falta(). REGLA-T53: invoca la función real, no
hardcodea la fórmula de distancia.
"""
from scripts.funnel_report import picks_mas_cerca, _mas_cerca


def test_176_01_alias_apunta_a_la_misma_funcion():
    assert _mas_cerca is picks_mas_cerca


def test_176_02_picks_mas_cerca_ordena_por_distancia_ascendente():
    watchlist = [
        {"partido": "A vs B", "gate_bloqueante": "G_EDGE_MIN", "edge": 0.01},
        {"partido": "C vs D", "gate_bloqueante": "G_EDGE_MIN", "edge": 0.045},
    ]
    resultado = picks_mas_cerca(watchlist, [], top_n=3)
    assert len(resultado) == 2
    # "C vs D" (edge=0.045) está más cerca del umbral EDGE_MIN que "A vs B" (edge=0.01)
    assert resultado[0][1] == "C vs D"


def test_176_03_top_n_limita_resultado():
    watchlist = [
        {"partido": f"P{i} vs R{i}", "gate_bloqueante": "G_EDGE_MIN", "edge": 0.01 * i}
        for i in range(5)
    ]
    resultado = picks_mas_cerca(watchlist, [], top_n=2)
    assert len(resultado) == 2
