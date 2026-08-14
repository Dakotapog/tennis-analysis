"""Tests Nodo-163 — Tier Gap (D163-01) + Superficie Dinámica (D163-02) +
Games Bridge Tuple Crash (D163-03).

REGLA-T53: cada test invoca la función real del módulo, nunca hardcodea la fórmula.
"""
import ast
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── D163-01: --tier default incluye atp1000/atp500 ──────────────────────────

def test_run_daily_tier_default_incluye_atp1000_atp500():
    """Extrae el default real de argparse('--tier') del AST de run_daily.py —
    nunca corrió atp1000/atp500 en PASO 4 porque el default los omitía aunque
    tier_config sí los definía (13/17 picks de 2026-08-03 nunca evaluados)."""
    import run_daily
    src = Path(run_daily.__file__).read_text(encoding='utf-8')
    tree = ast.parse(src)

    tier_default = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, 'attr', None) == 'add_argument':
            args = [a.value for a in node.args if isinstance(a, ast.Constant)]
            if args and args[0] == '--tier':
                for kw in node.keywords:
                    if kw.arg == 'default':
                        tier_default = ast.literal_eval(kw.value)
    assert tier_default is not None, "no se encontró add_argument('--tier', ...) en run_daily.py"
    assert 'atp1000' in tier_default
    assert 'atp500' in tier_default


def test_run_daily_tier_config_tiene_entrada_para_cada_default():
    """Cada tier en el default de --tier debe tener entrada en tier_config —
    de lo contrario el loop `for tier in args.tier: cfg = tier_config.get(tier)`
    lo saltaría silenciosamente (continue) sin correr el trader."""
    import inspect
    import run_daily
    src = inspect.getsource(run_daily.main)
    tree = ast.parse(src)

    tier_default, tier_config_keys = None, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, 'attr', None) == 'add_argument':
            args = [a.value for a in node.args if isinstance(a, ast.Constant)]
            if args and args[0] == '--tier':
                for kw in node.keywords:
                    if kw.arg == 'default':
                        tier_default = ast.literal_eval(kw.value)
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'tier_config' for t in node.targets):
            tier_config_keys = {
                k.value for k in node.value.keys if isinstance(k, ast.Constant)
            }
    assert tier_default is not None
    assert tier_config_keys, "no se encontró el dict tier_config en main()"
    for tier in tier_default:
        assert tier in tier_config_keys, f"'{tier}' está en --tier default pero falta en tier_config"


# ── D163-02: superficie dinámica por tier ────────────────────────────────────

def _edge_report_fixture(tmp_path, picks_por_tier):
    """picks_por_tier: {tier: [superficie, superficie, ...]}"""
    er = {'apostar': [], 'watchlist': [], 'sin_edge': [], 'picks': []}
    for tier, superficies in picks_por_tier.items():
        for sup in superficies:
            er['watchlist'].append({'tier': tier, 'superficie': sup})
    p = tmp_path / 'edge_report_kambi_20260803_100000.json'
    p.write_text(json.dumps(er), encoding='utf-8')
    return p


def test_superficie_dominante_detecta_hard_para_atp1000(tmp_path, monkeypatch):
    from run_daily import _superficie_dominante_tier
    import run_daily
    _edge_report_fixture(tmp_path, {'atp1000': ['hard', 'hard', 'grass']})
    monkeypatch.setattr(run_daily, 'REPORTS_DIR', str(tmp_path))
    result = _superficie_dominante_tier('atp1000', '20260803', fallback='grass')
    assert result == 'hard'


def test_superficie_dominante_fallback_sin_datos(tmp_path, monkeypatch):
    """Si no hay picks para ese tier hoy, usa el fallback estático — nunca lanza."""
    from run_daily import _superficie_dominante_tier
    import run_daily
    _edge_report_fixture(tmp_path, {'itf': ['clay']})
    monkeypatch.setattr(run_daily, 'REPORTS_DIR', str(tmp_path))
    result = _superficie_dominante_tier('atp500', '20260803', fallback='grass')
    assert result == 'grass'


def test_superficie_dominante_sin_edge_report_usa_fallback(tmp_path, monkeypatch):
    from run_daily import _superficie_dominante_tier
    import run_daily
    monkeypatch.setattr(run_daily, 'REPORTS_DIR', str(tmp_path))
    result = _superficie_dominante_tier('grand_slam', '20260803', fallback='clay')
    assert result == 'clay'


# ── D163-03: evaluar_games_bridge tuple unpack ───────────────────────────────

def test_seleccionar_señal_optima_retorna_tupla_de_listas():
    """Confirma el contrato real (Nodo-149 D149-02): 2 listas, nunca una lista plana
    de dicts — es la firma que evaluar_games_bridge.py debe respetar."""
    from games_signal_calculator import _seleccionar_señal_optima
    señales = [
        {'apostar': True, 'mercado_tipo': 'JUEGOS', 'direccion': 'UNDER',
         'gap_juegos': 3.0, 'cuota': 1.85},
    ]
    result = _seleccionar_señal_optima(señales)
    assert isinstance(result, tuple) and len(result) == 2
    juegos_optimas, sets_optimas = result
    assert isinstance(juegos_optimas, list)
    assert all(isinstance(s, dict) for s in juegos_optimas)


def test_evaluar_games_bridge_no_crashea_con_señal_real(tmp_path, monkeypatch):
    """Reproduce el flujo real event_id-encontrado -> optimas -> _res sin el
    AttributeError ('list' object has no attribute 'get') que rompía _save_report
    cada corrida desde Nodo-149."""
    from games_signal_calculator import _seleccionar_señal_optima
    import scripts.evaluar_games_bridge as bridge

    señales = [
        {'apostar': True, 'mercado_tipo': 'JUEGOS', 'direccion': 'UNDER',
         'gap_juegos': 3.0, 'cuota': 1.85},
    ]
    juegos_optimas, _sets_optimas = _seleccionar_señal_optima(señales)
    resultado = {
        'partido': 'A vs B', 'zona_diff': 'DOMINANTE', 'diff_abs': 0.8,
        'predicted_sets': 2, 'games_range': '16-19', 'hora': '10:00',
        'cuota_ml': 1.20, 'confidence': 0.80,
        'señales_optimas': juegos_optimas, 'tiene_mercados': bool(juegos_optimas),
        '_source': 'evaluar_games', '_sb_id': 'EVAL_TEST_001',
    }
    orig = bridge.REPORTS_DIR
    bridge.REPORTS_DIR = tmp_path
    try:
        out_path = bridge._save_report([resultado], '2026-08-03')
        data = json.loads(out_path.read_text(encoding='utf-8'))
        assert data['metadata']['n_picks'] == 1
    finally:
        bridge.REPORTS_DIR = orig
