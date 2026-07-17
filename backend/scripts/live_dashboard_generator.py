"""
live_dashboard_generator.py — Dashboard HTML interactivo para Triple Convergencia Live.
Nodo-100 D100-05.

Lee live_edge_*.json + live_odds_history_*.json + edge_report_*.json.
Genera reports/live_dashboard.html con auto-refresh cada 60s.

Uso:
  python3 scripts/live_dashboard_generator.py           # genera y guarda
  python3 scripts/live_dashboard_generator.py --stdout  # imprime HTML
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Importar fingerprint de señales (usa pick del edge_report cuando el monitor no tiene live data)
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from signal_audit import extract_signal_fingerprint as _extract_fingerprint
except Exception:
    def _extract_fingerprint(pick: dict) -> list:
        return []

SCRIPT_DIR  = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent
REPORTS_DIR = BACKEND_DIR / 'reports'

_ESTADO_COLOR = {
    'NORMAL':            '#6c757d',   # gris
    'BREAK_POSIBLE':     '#fd7e14',   # naranja
    'BREAK_CONFIRMADO':  '#dc3545',   # rojo
}
_ESTADO_LABEL = {
    'NORMAL':            'NORMAL',
    'BREAK_POSIBLE':     'QUIEBRE POSIBLE',
    'BREAK_CONFIRMADO':  'QUIEBRE CONFIRMADO',
}


def _load_latest_edge_report(reports_dir: Path) -> dict[str, dict]:
    """Retorna dict {partido_key: pick} del edge_report más reciente."""
    files = sorted(reports_dir.glob('edge_report_*.json'), reverse=True)
    if not files:
        return {}
    try:
        with open(files[0], encoding='utf-8') as f:
            rep = json.load(f)
        out = {}
        for seg in ('apostar', 'watchlist'):
            for pick in rep.get(seg, []):
                key = (pick.get('partido', '') or '').replace(' ', '_')
                if key:
                    out[key] = pick
        return out
    except Exception:
        return {}


def _load_latest_live_edge(reports_dir: Path) -> dict:
    """Carga el último live_edge_*.json."""
    files = sorted(reports_dir.glob('live_edge_*.json'), reverse=True)
    if not files:
        return {}
    try:
        with open(files[0], encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _load_odds_history(reports_dir: Path) -> dict:
    """Carga live_odds_history_YYYYMMDD.json del día actual."""
    today = datetime.now().strftime('%Y%m%d')
    path  = reports_dir / f'live_odds_history_{today}.json'
    if not path.exists():
        return {}
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def generar_dashboard_html(reports_dir: str = None) -> str:
    """
    Genera reports/live_dashboard.html con estado actual de picks live.
    Retorna path del archivo generado.
    """
    rdir = Path(reports_dir) if reports_dir else REPORTS_DIR

    picks_map    = _load_latest_edge_report(rdir)
    live_snap    = _load_latest_live_edge(rdir)
    odds_history = _load_odds_history(rdir)

    ts_now   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ts_snap  = live_snap.get('ts', 'sin datos')[:19]
    n_trig   = live_snap.get('n_triggers', 0)
    n_break  = live_snap.get('break_confirmados', 0)

    # Construir filas de tabla (picks_chequeados_data = lista; picks_chequeados = int count)
    picks_chequeados = live_snap.get('picks_chequeados_data', [])
    hay_confirmado   = any(p.get('break_state') == 'BREAK_CONFIRMADO'
                           for p in picks_chequeados)

    filas_html = []
    for pc in picks_chequeados:
        partido     = pc.get('partido', '?')
        partido_key = partido.replace(' ', '_')
        estado      = pc.get('break_state', 'NORMAL')
        drift_pct   = pc.get('drift_pct', 0.0) or 0.0
        cuota_pre   = pc.get('cuota_pre', 0.0) or 0.0
        cuota_live  = pc.get('cuota_live', '?')
        edge_live   = pc.get('edge_live', 0.0) or 0.0
        senales_raw = pc.get('senales', [])
        # Fallback: si el monitor no tiene señales (match no live aún), leer del edge_report
        if not senales_raw:
            pick_data_for_sig = picks_map.get(partido_key, {})
            senales_raw = _extract_fingerprint(pick_data_for_sig)

        # Formatear señales con chips de color
        chip_colors = {
            'STRONG':      '#28a745',   # verde
            'MOD':         '#6c757d',   # gris
            'MRK_HOT':     '#fd7e14',   # naranja
            'MRK_COLD':    '#dc3545',   # rojo
            'MRK_NEUTRAL': '#6c757d',   # gris
            'rival_COLD':  '#00d4ff',   # cyan
            'ELO_DOM':     '#9b59b6',   # morado
            'GCS':         '#17a2b8',   # teal
            'COLD_fav':    '#dc3545',   # rojo
            'SURF_HIGH':   '#20c997',   # verde agua
            'SURF_OK':     '#6c757d',   # gris
            'SD_HIGH':     '#ffc107',   # amarillo
            'SD_MED':      '#fd7e14',   # naranja
            'EDGE_HIGH':   '#28a745',   # verde
            'EDGE_MED':    '#6c757d',   # gris
            'CONTRIB_HIGH':'#e83e8c',   # rosa
            'CONTRIB_MED': '#6c757d',   # gris
            'RFI_T':       '#6f42c1',   # índigo
            'IRP_rival':   '#dc3545',   # rojo
            'CAMPEON':     '#ffd700',   # dorado
            'AJUSTE_DIN':  '#17a2b8',   # teal
        }
        chips = []
        for sig in senales_raw:
            if sig.startswith('TIER_'):  # TIER no es señal predictiva — ocultar de chips
                continue
            col = next((v for k, v in chip_colors.items() if sig.startswith(k)), '#adb5bd')
            chips.append(f'<span style="background:{col}33;color:{col};border:1px solid {col};'
                         f'border-radius:3px;padding:1px 5px;font-size:11px;margin:1px;">{sig}</span>')
        senales_html = ' '.join(chips) if chips else '<span style="color:#555">—</span>'

        # score_directo y saque desde edge_report + live
        pick_data = picks_map.get(partido_key, {})
        score     = pick_data.get('score_directo', '-')
        server    = pc.get('server', '')   # 'FAV', 'OPP', o ''
        server_icon = {'FAV': '🎾', 'OPP': '⚽'}.get(server, '')  # 🎾 = FAV sirve

        color = _ESTADO_COLOR.get(estado, '#6c757d')
        label = _ESTADO_LABEL.get(estado, estado)

        blink_style = 'animation: blink 0.8s step-end infinite;' if estado == 'BREAK_CONFIRMADO' else ''
        cuota_live_str = f'{cuota_live:.2f}' if isinstance(cuota_live, float) else (cuota_live if cuota_live != '?' else '—')
        filas_html.append(f"""
        <tr style="background:{color}22; border-left: 4px solid {color};">
          <td><b>{partido}</b><br><small style="color:#888">{pc.get('favorito','')}</small></td>
          <td>{senales_html}</td>
          <td style="text-align:center;">{score}</td>
          <td style="text-align:center;">{cuota_pre:.2f}</td>
          <td style="text-align:center;font-weight:bold;color:{'#fd7e14' if drift_pct>=15 else '#e0e0e0'}">{cuota_live_str}</td>
          <td style="text-align:center;{'color:'+color+';font-weight:bold;' if abs(drift_pct) >= 12 else ''}">{drift_pct:+.1f}%</td>
          <td style="text-align:center;">{server_icon} {edge_live:+.3f}</td>
          <td style="text-align:center;{blink_style} color:{color}; font-weight:bold;">{label}</td>
        </tr>""")

    filas_str = ''.join(filas_html) if filas_html else """
        <tr><td colspan="7" style="text-align:center;padding:20px;color:#999;">
          Sin picks monitoreados — correr edge_calculator.py primero
        </td></tr>"""

    banner_html = ''
    if hay_confirmado:
        banner_html = """
    <div style="background:#dc3545;color:white;padding:16px;margin-bottom:16px;
                border-radius:6px;font-size:18px;font-weight:bold;text-align:center;
                animation: blink 0.8s step-end infinite;">
      BREAK CONFIRMADO — COMBOS LIVE DISPARADOS
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="60">
  <title>Live Dashboard — Triple Convergencia</title>
  <style>
    body {{ font-family: monospace; background: #1a1a2e; color: #e0e0e0;
            padding: 20px; margin: 0; }}
    h1   {{ color: #00d4ff; margin-bottom: 4px; }}
    .subtitle {{ color: #888; font-size: 13px; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th    {{ background: #16213e; color: #00d4ff; padding: 8px 12px;
             text-align: left; border-bottom: 2px solid #00d4ff; }}
    td    {{ padding: 8px 12px; border-bottom: 1px solid #2a2a4a; }}
    .kpi  {{ display: flex; gap: 20px; margin-bottom: 16px; }}
    .kpi-box {{ background: #16213e; border: 1px solid #00d4ff33; border-radius: 6px;
                padding: 10px 16px; min-width: 120px; }}
    .kpi-val {{ font-size: 24px; font-weight: bold; color: #00d4ff; }}
    .kpi-lbl {{ font-size: 11px; color: #888; margin-top: 2px; }}
    @keyframes blink {{ 0%,100% {{ opacity:1 }} 50% {{ opacity:0.3 }} }}
    .refresh {{ color: #555; font-size: 11px; margin-top: 12px; }}
    .btn-live {{ background: #00d4ff; color: #1a1a2e; border: none; border-radius: 6px;
                 padding: 8px 20px; font-size: 14px; font-weight: bold; cursor: pointer;
                 margin-left: 12px; vertical-align: middle; }}
    .btn-live:hover {{ background: #00b8d9; }}
    .btn-live:disabled {{ background: #555; color: #999; cursor: default; }}
  </style>
  <script>
    function runLiveCheck() {{
      var btn = document.getElementById('btnLive');
      btn.disabled = true; btn.textContent = '⏳ Chequeando...';
      fetch('http://localhost:8765/live-check')
        .then(function(r) {{ return r.json(); }})
        .then(function(d) {{
          btn.textContent = '✅ Listo (' + (d.n_triggers||0) + ' triggers)';
          setTimeout(function() {{ location.reload(); }}, 1200);
        }})
        .catch(function() {{
          btn.disabled = false; btn.textContent = '🔴 EN VIVO';
        }});
    }}
  </script>
</head>
<body>
  <h1>Live Dashboard — Triple Convergencia
    <button id="btnLive" class="btn-live" onclick="runLiveCheck()">🔴 EN VIVO</button>
  </h1>
  <div class="subtitle">Nodo-100 | Auto-refresh cada 60s | Ultimo ciclo: {ts_snap}</div>

  {banner_html}

  <div class="kpi">
    <div class="kpi-box">
      <div class="kpi-val">{live_snap.get('picks_monitoreados', 0)}</div>
      <div class="kpi-lbl">Picks monitoreados</div>
    </div>
    <div class="kpi-box">
      <div class="kpi-val" style="color:{'#fd7e14' if n_trig > 0 else '#00d4ff'}">{n_trig}</div>
      <div class="kpi-lbl">Triggers activos</div>
    </div>
    <div class="kpi-box">
      <div class="kpi-val" style="color:{'#dc3545' if n_break > 0 else '#00d4ff'}">{n_break}</div>
      <div class="kpi-lbl">Breaks confirmados</div>
    </div>
    <div class="kpi-box">
      <div class="kpi-val" style="color:{'#28a745' if live_snap.get('stake_permitido') else '#dc3545'}">
        {'OK' if live_snap.get('stake_permitido', True) else 'BLOCK'}
      </div>
      <div class="kpi-lbl">Stake permitido</div>
    </div>
  </div>

  <table>
    <thead>
      <tr>
        <th>Partido / Favorito</th>
        <th>Por qué apostamos</th>
        <th>SD</th>
        <th>Pre</th>
        <th>Live</th>
        <th>Drift</th>
        <th>Edge / Saque</th>
        <th>Estado Break</th>
      </tr>
    </thead>
    <tbody>
      {filas_str}
    </tbody>
  </table>

  <div class="refresh">Generado: {ts_now} | Proximo refresh automatico en ~60s</div>
</body>
</html>"""

    out_path = rdir / 'live_dashboard.html'
    out_path.write_text(html, encoding='utf-8')
    return str(out_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stdout', action='store_true')
    args = parser.parse_args()

    if args.stdout:
        rdir = REPORTS_DIR
        picks_map    = _load_latest_edge_report(rdir)
        live_snap    = _load_latest_live_edge(rdir)
        odds_history = _load_odds_history(rdir)
        print(generar_dashboard_html())
    else:
        path = generar_dashboard_html()
        print(f'[live_dashboard] generado: {path}')


if __name__ == '__main__':
    main()
