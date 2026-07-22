#!/usr/bin/env python3
"""
live_desk.py — Live Trading Desk Dashboard (Nodo-109)

Servidor HTTP en :7780. HTML único auto-refresh 30s con 7 paneles.
NO calcula señales nuevas — solo lee archivos existentes y los presenta
con semántica de mesa de trading.

Paneles:
  P1 TAPE       — momentum de línea (drift% live_edge_monitor)
  P2 BREAK BOARD — rupturas confirmadas (break_state Nodo-100B)
  P3 CONVERGENCE — semáforo de convicción (meta_signal_score H98-01)
  P4 RISK        — governor + exposición + circuit breakers (P4 MANDA)
  P5 EXECUTION   — CLV por pick abierto (shadow_book Momento 2)
  P6 P&L         — blotter por estrategia (shadow_book --report segments)
  P7 CLOCK       — ventanas de acción (zita file + countdown)

Uso:
  python live_desk.py            # :7780
  python live_desk.py --port 7781
  python live_desk.py --fecha 2026-07-14  # día específico para debug
"""

import argparse
import glob
import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
REPORTS = BASE_DIR / "reports"
SB_DIR = REPORTS / "shadow_book"
LOGS_DIR = BASE_DIR / "logs"

PORT_DEFAULT = 7780


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES PURAS (testeables REGLA-T53)
# ══════════════════════════════════════════════════════════════════════════════

_SPARK = '▁▂▃▄▅▆▇█'


def _load_odds_history(fecha: str) -> dict:
    """Lee live_odds_history_YYYYMMDD.json para sparklines U4. REPORTE_SOLO."""
    p = REPORTS / f"live_odds_history_{fecha.replace('-', '')}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _sparkline_drift(jugador: str, history: dict) -> str:
    """
    Últimos 4 drifts del jugador → chars ▁▂▄▇ + flecha tendencia (U4 Nodo-115).
    Retorna "" si no hay historial. REPORTE_SOLO, función pura.
    """
    jug_lower = jugador.lower()
    readings: list = []
    for pk, entry in history.items():
        if jug_lower in pk.lower():
            readings = entry.get('readings', [])[-4:]
            break
    if not readings:
        return ""
    drifts = [r.get('drift', 0) for r in readings]
    max_d = max(abs(d) for d in drifts) or 0.01
    spark = ''.join(_SPARK[min(7, int(abs(d) / max_d * 7))] for d in drifts)
    arrow = '→'
    if len(drifts) >= 2:
        delta = drifts[-1] - drifts[0]
        if delta > 0.01:
            arrow = '↑'
        elif delta < -0.01:
            arrow = '↓'
    return f'{spark} {arrow}'


def _build_conformal_band() -> dict:
    """
    Llama conformal_report() una vez por ciclo. Retorna q_global y gate.
    REPORTE_SOLO — no cambia ningún gate. (U1 Nodo-115 §4)
    """
    try:
        from analysis.conformal_band import conformal_report
        return conformal_report()
    except Exception:
        return {"q_global": None, "gate_ok": False, "n_settled": 0}


def _build_combo_live(fecha: str) -> List[Dict]:
    """
    Lee reports/combos_live/YYYY-MM-DD/_fired.json y retorna lista de dicts
    para filas COMBO_LIVE en accionable_ahora() (D116-01 §B.5).
    REPORTE_SOLO — no cambia gates.
    """
    fired_path = REPORTS / "combos_live" / fecha / "_fired.json"
    if not fired_path.exists():
        return []
    try:
        fired = json.load(fired_path.open(encoding="utf-8"))
    except Exception:
        return []
    rows = []
    combos_dir = REPORTS / "combos_live" / fecha
    for event_id, meta in fired.items():
        # Buscar .bat asociado (nombre contiene timestamp, no event_id directo)
        # Usamos glob para encontrar cualquier .bat — el desk muestra el último
        bat_files = sorted(combos_dir.glob("LiveCombo_*.bat"))
        bat_link = str(bat_files[-1]) if bat_files else ""
        rows.append({
            "tipo": "COMBO_LIVE",
            "jugador": event_id,
            "pick": event_id,
            "hipotesis": "H100-01",
            "n_actual": 0,
            "n_stop": 20,
            "color": "amber",
            "fired_at": meta.get("fired_at", ""),
            "drift_pct": meta.get("drift_pct", 0),
            "bat_link": bat_link,
            "señales_activas": ["BREAK_CONFIRMADO"],
        })
    return rows


def build_desk_state(fecha: Optional[str] = None) -> Dict[str, Any]:
    """
    Agrega las 7 fuentes de datos para el desk. Tolera archivos ausentes.

    Args:
        fecha: YYYY-MM-DD (default: hoy)

    Returns:
        dict con claves: fecha, ts, p1_tape, p2_break, p3_convergence,
        p4_risk, p5_execution, p6_pnl, p7_clock
    """
    if fecha is None:
        fecha = date.today().isoformat()

    # p0_ncal: {nombre_lower: n_calibracion} para U2 en render — no cambia gates
    _ncal: Dict[str, int] = {}
    _er = _latest(str(REPORTS / f"edge_report_{fecha.replace('-', '')}*.json"))
    _er_data = _load_json(_er)
    if _er_data and isinstance(_er_data, dict):
        for _section in ("apostar", "watchlist", "sin_edge", "sin_datos"):
            for _p in _er_data.get(_section, []):
                _jug = (_p.get("favorito_predicho") or "").lower().strip()
                if _jug:
                    _ncal[_jug] = int(_p.get("n_calibracion") or 0)

    state: Dict[str, Any] = {
        "fecha": fecha,
        "ts": datetime.now().isoformat(),
        "data_freshness": _data_freshness(fecha),  # D129-03: mtime real de datos
        "p0_ncal": _ncal,                    # Nodo-115 U2: evidencia por jugador
        "p4_risk": _build_p4_risk(fecha),    # P4 primero — manda
        "p1_tape": _build_p1_tape(fecha),
        "p2_break": _build_p2_break(fecha),
        "p3_convergence": _build_p3_convergence(fecha),
        "p5_execution": _build_p5_execution(fecha),
        "p6_pnl": _build_p6_pnl(fecha),
        "p7_clock": _build_p7_clock(fecha),
        "p8_books": _build_p8_books(fecha),      # Nodo-114 §3: dual-book cache 120s
        "p9_que_falta": _build_que_falta(fecha),       # Nodo-115 §2.4
        "p10_odds_history": _load_odds_history(fecha), # Nodo-115 U4 sparkline
        "p11_combo_live": _build_combo_live(fecha),    # Nodo-116 §B.5
        "p12_conformal": _build_conformal_band(),      # Nodo-115 U1 banda
        "p_data": _build_data_panel(fecha),            # Nodo-118 §5 embudo crosswalk
        "p_games": _build_x3_games(fecha),             # Nodo-40 X3 games signal
        "p_evaluar_games": _build_x4_evaluar_games(fecha),  # Nodo-125 X4 EVALUAR_GAMES
    }
    return state


def accionable_ahora(state: Dict[str, Any]) -> List[Dict]:
    """
    Intersección de la regla 1: BREAK_CONFIRMADO ∩ governor PASS ∩ estrategia graduada.

    Returns lista de picks accionables (puede ser vacía si governor=BLOCK o KGR<0).
    """
    # P4 manda: si BLOCK o KGR<0, nada es accionable (REGLA-HF-5)
    risk = state.get("p4_risk", {})
    if risk.get("governor_code", 0) >= 2:
        return []
    if risk.get("kgr_sesion", 1.0) < 0:
        return []

    accionables = []
    gov_code = risk.get("governor_code", 0)

    # Lookup p3 picks by jugador for meta_score enrichment
    _p3_by_jug = {p["jugador"]: p for p in state.get("p3_convergence", {}).get("picks", [])}

    # P2 breaks confirmados — graduados solo si estrategia tiene gate activo
    for brk in state.get("p2_break", {}).get("breaks", []):
        if brk.get("estado") == "BREAK_CONFIRMADO":
            p3d = _p3_by_jug.get(brk.get("jugador", ""), {})
            edge_live = brk.get("edge_live")
            trigger = brk.get("trigger", False)
            # señales: preferir las del live_edge (HOT/EDGE_MED) sobre p3d vacío
            senales_live = brk.get("senales", [])
            senales = p3d.get("señales_activas") or senales_live
            accionables.append({
                "tipo": "BREAK_CONFIRMADO",
                "p_modelo": brk.get("p_modelo", 0),
                "jugador": brk.get("jugador", ""),
                "pick": brk.get("pick", ""),
                "partido": brk.get("partido", ""),
                "hipotesis": "H100-01",
                "n_actual": brk.get("n_actual", 0),
                "n_stop": 20,
                "color": "amber",  # pre-graduacion: amber siempre
                "governor_code": gov_code,
                "drift_pct": brk.get("drift_pct", 0),
                "cuota_pre": brk.get("cuota_pre"),
                "cuota_live": brk.get("cuota_live"),
                "edge_live": edge_live,
                "trigger": trigger,
                "meta_score": p3d.get("score_directo", 0),
                "señales_activas": senales,
                "n_h2h": p3d.get("n_h2h"),
                "clv": p3d.get("clv"),
            })

    # P3 RIVAL_VALUE con flag activo (H88-01 — pre-graduacion, amber)
    for pick in state.get("p3_convergence", {}).get("picks", []):
        if pick.get("rival_value_flag") and pick.get("score_directo", 0) >= 1:
            accionables.append({
                "tipo": "RIVAL_VALUE",
                "jugador": pick.get("rival", pick.get("jugador", "")),
                "pick": pick.get("jugador", ""),
                "hipotesis": "H88-01",
                "n_actual": 3,
                "n_stop": 30,
                "color": "amber",
                "governor_code": gov_code,
                "meta_score": pick.get("score_directo", 0),
                "señales_activas": pick.get("señales_activas", []),
                "n_h2h": pick.get("n_h2h"),
                "clv": pick.get("clv"),
            })

    # GCS activo (H60-01 GRADUADA → verde)
    for pick in state.get("p3_convergence", {}).get("picks", []):
        if pick.get("gcs_active"):
            accionables.append({
                "tipo": "GCS",
                "jugador": pick.get("jugador", ""),
                "pick": pick.get("jugador", ""),
                "hipotesis": "H60-01",
                "n_actual": 54,
                "n_stop": 54,
                "color": "green",  # GRADUADA
                "governor_code": gov_code,
                "meta_score": pick.get("score_directo", 0),
                "señales_activas": pick.get("señales_activas", []),
                "n_h2h": pick.get("n_h2h"),
                "clv": pick.get("clv"),
            })

    # §4 FAVORITOS_COMPUESTOS primera clase (Nodo-110 / Nodo-114 §4)
    fav_data = _favoritos_hoy(state.get("fecha", ""))
    if fav_data is not None:
        accionables.append({
            "tipo": "FAVORITOS_COMPUESTOS",
            "jugador": f"{fav_data.get('n_combos', 0)} combos",
            "pick": "FAVORITOS_COMPUESTOS",
            "hipotesis": "H110-01",
            "n_actual": 8,   # semilla jul-14/16
            "n_stop": 30,
            "color": "amber",
            "governor_code": gov_code,
            "n_combos": fav_data.get("n_combos", 0),
            "señales_activas": [],
        })
    else:
        # Zero-Null: builder no corrió hoy
        accionables.append({
            "tipo": "FAVORITOS_ZERO",
            "jugador": "sin correr hoy",
            "pick": "",
            "hipotesis": "H110-01",
            "n_actual": 8,
            "n_stop": 30,
            "color": "amber",
            "governor_code": gov_code,
            "señales_activas": [],
            "nota": "python3 favoritos_combo_builder.py --bankroll 125000",
        })

    # COMBO_LIVE del día (D116-01 §B.5) — filas accionables del desk
    for cl in state.get("p11_combo_live", []):
        cl["governor_code"] = gov_code
        accionables.append(cl)

    # Enriquecer todos con mejor_precio de P8
    p8_picks = state.get("p8_books", {}).get("picks", {})
    for a in accionables:
        jug_key = a.get("jugador", "").lower().strip()
        pick_key = a.get("pick", "").lower().strip()
        bp = p8_picks.get(jug_key) or p8_picks.get(pick_key)
        if bp:
            a["mejor_precio"] = bp

    return accionables


def linea_razonamiento(pick: dict) -> str:
    """
    Función pura (REGLA-T53): genera la línea de razonamiento visible para un
    pick accionable. Solo señales presentes, riesgo primero, sin cálculo nuevo.

    Formato:
      [H-XX n/n_stop] TIPO(contexto) + meta_score=N(SIG1,SIG2) + CLV+X% +
      n_h2h=N + governor PASS → mejor precio: casa @X.XX (+Y% vs plan)

    Reglas:
      - Gate siempre al frente
      - Señales pre-graduación: campo, el caller asigna color ámbar
      - Máx 180 chars (resto en title attr del HTML)
    """
    parts = []

    # Gate [H-XX n/n_stop] — siempre primero
    hip = pick.get("hipotesis", "")
    n_a = pick.get("n_actual", 0)
    n_s = pick.get("n_stop", 0)
    gate = f"[{hip} {n_a}/{n_s}]" if hip else ""

    # Tipo de señal + contexto específico
    tipo = pick.get("tipo", "")
    if tipo == "BREAK_CONFIRMADO":
        drift = pick.get("drift_pct")
        cuota_pre = pick.get("cuota_pre")
        cuota_live = pick.get("cuota_live")
        edge_live = pick.get("edge_live")
        trigger = pick.get("trigger", False)
        # cuota pre→live
        cuota_str = (f" cuota {cuota_pre}→{cuota_live}" if cuota_pre and cuota_live else "")
        drift_str = f" drift {drift:+.1f}%" if drift is not None else ""
        edge_str = ""
        if edge_live is not None:
            edge_str = f" edge_live={edge_live:+.2f}"
            if edge_live < 0:
                edge_str += " EDGE NEGATIVO-NO APOSTAR"
        trig_str = "" if trigger else " [monitor: NO DISPARAR]"
        parts.append(f"BREAK_CONFIRMADO{cuota_str}{drift_str}{edge_str}{trig_str}")
    elif tipo == "GCS":
        parts.append("GCS(H60-01 GRADUADA)")
    elif tipo == "RIVAL_VALUE":
        parts.append("RIVAL_VALUE(H88-01 pre-grad)")
    elif tipo == "FAVORITOS_COMPUESTOS":
        nc = pick.get("n_combos", 0)
        parts.append(f"FAVORITOS_COMPUESTOS({nc} combos H110-01 8/8 semilla)")
    elif tipo == "FAVORITOS_ZERO":
        nota = pick.get("nota", "python3 favoritos_combo_builder.py")
        return f"{gate} FAVORITOS: sin correr — {nota}".strip()
    elif tipo:
        parts.append(tipo)

    # Meta-score
    meta = pick.get("meta_score")
    senas = pick.get("señales_activas", [])
    if meta:
        suf = f"({','.join(senas)})" if senas else ""
        parts.append(f"meta_score={meta}{suf}")

    # CLV (si disponible)
    clv = pick.get("clv")
    if clv is not None:
        parts.append(f"CLV{clv:+.1f}%")

    # n_h2h (si disponible)
    n_h2h = pick.get("n_h2h")
    if n_h2h is not None:
        parts.append(f"n_h2h={n_h2h}")

    # Governor
    gcode = pick.get("governor_code", 0)
    gov_str = "PASS" if gcode == 0 else ("WARN" if gcode == 1 else "BLOCK")
    parts.append(f"governor {gov_str}")

    # Mejor precio al final con → (de P8 multi-book)
    precio_str = ""
    mejor = pick.get("mejor_precio", {})
    if mejor and mejor.get("cuota"):
        casa = mejor.get("casa", "?")
        cuota = mejor.get("cuota", 0)
        gain = mejor.get("gain_pct", 0)
        if gain > 0:
            precio_str = f" → mejor precio: {casa} @{cuota} (+{gain:.1f}% vs plan)"
        else:
            precio_str = f" → {casa} @{cuota}"

    body = " + ".join(parts)
    line = f"{gate} {body}{precio_str}".strip() if gate else f"{body}{precio_str}"
    return line[:180]


# ── Nodo-115: helpers de incertidumbre (U2, U3, QUÉ FALTA) ──────────────────

# Constantes copiadas de favoritos_combo_builder.py (REGLA-T53: no importar el módulo)
_FAV_P_MIN = 0.62
_FAV_CUOTA_CLARA_MAX = 1.40
_FAV_RANKING_GAP = 300
_FAV_LEG_MIN = 1.15
_FAV_LEG_MAX = 2.10


def _peso_evidencia(n_cal: int) -> Dict[str, Any]:
    """U2: shrinkage n/(n+20) → barra texto + etiqueta + color."""
    n_cal = max(0, int(n_cal or 0))
    pct = round(n_cal / (n_cal + 20) * 100)
    filled = min(5, round(pct / 20))
    bar = "█" * filled + "░" * (5 - filled)
    if pct < 20:
        color = "#f85149"   # rojo — prior manda
        label = "PRIOR MANDA"
    elif pct < 45:
        color = "#d29922"   # ámbar — moderado
        label = f"{pct}%"
    else:
        color = "#3fb950"   # verde — peso propio
        label = f"{pct}%"
    return {"pct": pct, "bar": bar, "label": label, "color": color, "n": n_cal}


def _gate_barra(n_actual: int, n_stop: int) -> str:
    """U3: barra texto █░ hacia n_stop."""
    n_actual = int(n_actual or 0)
    n_stop = int(n_stop or 0)
    if n_stop <= 0:
        return f"{n_actual}/??"
    if n_actual >= n_stop:
        return "GRADUADA"
    frac = min(n_actual / n_stop, 1.0)
    filled = round(frac * 10)
    bar = "█" * filled + "░" * (10 - filled)
    restantes = n_stop - n_actual
    return f"{bar} {n_actual}/{n_stop} ({restantes} faltan)"


def _build_x3_games(fecha: str) -> Dict[str, Any]:
    """
    X3 Games Signal — lee games_signal_report_{FECHA}*.json (Nodo-40 PASO 3.6).
    Retorna apostar + metadata. REPORTE_SOLO.
    """
    gsr = _latest(str(REPORTS / f"games_signal_report_{fecha.replace('-','')}*.json"))
    if not gsr:
        return {"disponible": False, "fecha": fecha}
    try:
        data = json.loads(Path(gsr).read_text(encoding="utf-8"))
    except Exception:
        return {"disponible": False, "fecha": fecha}
    meta = data.get("metadata", {})
    apostar = data.get("apostar", [])
    # Aplanar señales_optimas: una fila por señal accionable
    signals: List[Dict] = []
    for p in apostar:
        for s in p.get("señales_optimas", []):
            if s.get("apostar"):
                signals.append({
                    "partido":    p.get("partido", ""),
                    "mercado":    s.get("mercado", ""),
                    "direccion":  s.get("direccion", ""),
                    "linea":      s.get("linea"),
                    "cuota":      s.get("cuota"),
                    "gap":        s.get("gap_juegos"),
                    "confianza":  s.get("confianza_señal", ""),
                    "games_range": p.get("games_range", ""),
                })
    # D133-05: enriquecer con estado live si existe games_live_YYYYMMDD.json
    gl_path = REPORTS / f"games_live_{fecha.replace('-', '')}.json"
    gl_data: Dict = {}
    if gl_path.exists():
        try:
            gl_data = json.loads(gl_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    live_idx: Dict[str, Dict] = {
        s.get("partido", ""): s for s in gl_data.get("signals_alta", [])
    }
    for sig in signals:
        live_s = live_idx.get(sig["partido"], {})
        sig["estado_live"] = live_s.get("estado", "PRE_PARTIDO")
        sig["cuota_live"]  = live_s.get("cuota_live")
        sig["drift_pct"]   = live_s.get("drift_pct")

    return {
        "disponible":         True,
        "fecha":              fecha,
        "n_partidos":         meta.get("n_partidos", 0),
        "n_apostar":          meta.get("n_apostar", 0),
        "signals":            signals,
        "fuente":             Path(gsr).name,
        "en_vivo_count":      gl_data.get("en_vivo_count", 0),
        "convergencia_activa": gl_data.get("convergencia_activa", False),
    }


def _build_x4_evaluar_games(fecha: str) -> Dict[str, Any]:
    """
    X4 EVALUAR_GAMES — Nodo-125 D125-04.
    Lee picks EVAL_ (pick_type=evaluar_games) del shadow_book de hoy.
    Enriquece con cuota UNDER de Kambi si existe evaluar_games_signal_FECHA.
    REPORTE_SOLO — sin gates de apuesta.
    """
    sb_path = BASE_DIR / "reports" / "shadow_book" / f"sb_{fecha}.jsonl"
    if not sb_path.exists():
        return {"disponible": False, "fecha": fecha, "picks": [], "n": 0, "n_con_under": 0}

    # Cargar EVAL_ records del shadow_book
    try:
        import shadow_book as _sb
        records = _sb._load_jsonl(sb_path)
    except Exception:
        return {"disponible": False, "fecha": fecha, "picks": [], "n": 0, "n_con_under": 0}

    eg_picks = []
    for sid, rec in records.items():
        if not sid.startswith("EVAL_"):
            continue
        snap = rec.get("pick_snapshot", {})
        if snap.get("pick_type") != "evaluar_games":
            continue
        resol = rec.get("resolucion")
        eg_picks.append({
            "sb_id":    sid,
            "partido":  snap.get("partido", ""),
            "conf":     (lambda c: c / 100 if c and c >= 1 else (c or 0))(snap.get("confidence")),
            "cuota_ml": snap.get("cuota_favorito") or 0,
            "hora":     snap.get("hora"),
            "resultado": resol.get("resultado") if resol else None,
            "cuota_under": None,  # enriquecido abajo si existe señal
        })

    # Enriquecer con cuota UNDER si existe evaluar_games_signal del día
    fecha_compact = fecha.replace("-", "")
    eg_signal_files = sorted(
        (BASE_DIR / "reports").glob(f"evaluar_games_signal_{fecha_compact}*.json"),
        reverse=True,
    )
    if eg_signal_files:
        try:
            eg_data = json.loads(eg_signal_files[0].read_text(encoding="utf-8"))
            # Índice partido → mejor cuota UNDER
            under_idx: Dict[str, float] = {}
            for p in eg_data.get("detalle_completo", []):
                for s in p.get("señales_optimas", []):
                    if s.get("apostar") and s.get("direccion") == "UNDER":
                        cand = float(s.get("cuota") or 0)
                        if cand > under_idx.get(p["partido"], 0):
                            under_idx[p["partido"]] = cand
            for pick in eg_picks:
                pick["cuota_under"] = under_idx.get(pick["partido"])
        except Exception:
            pass

    n_con_under = sum(1 for p in eg_picks if p.get("cuota_under"))
    return {
        "disponible":   True,
        "fecha":        fecha,
        "picks":        eg_picks,
        "n":            len(eg_picks),
        "n_con_under":  n_con_under,
        "fuente":       eg_signal_files[0].name if eg_signal_files else "shadow_book only",
    }


def _build_data_panel(fecha: str) -> Dict[str, Any]:
    """
    Panel DATA — Embudo Nodo-118 §5: estadísticas del ledger crosswalk del día.
    REPORTE_SOLO. Lee data/match_ledger_{YYYYMMDD}.json.
    """
    DATA_DIR = BASE_DIR / "data"
    fecha_compact = fecha.replace("-", "")
    ledger_path = DATA_DIR / f"match_ledger_{fecha_compact}.json"

    if not ledger_path.exists():
        return {"disponible": False, "fecha": fecha}

    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)
    except Exception:
        return {"disponible": False, "fecha": fecha}

    stats = ledger.get("stats", {})
    cuarentena = ledger.get("cuarentena", [])

    # Fuga nominal: primeros 10 con score
    fuga = []
    for c in cuarentena[:10]:
        k = c.get("kambi", c)
        j1 = k.get("jugador1", "?")
        j2 = k.get("jugador2", "?")
        score = c.get("score", c.get("score_mejor", "?"))
        fuga.append({"partido": f"{j1} vs {j2}", "score": score})

    return {
        "disponible": True,
        "fecha": fecha,
        "joins": stats.get("joins_exitosos", 0),
        "cuarentena": stats.get("cuarentena_count", 0),
        "single_kambi": stats.get("single_source_kambi", 0),
        "single_fs": stats.get("single_source_fs", 0),
        "api_total": stats.get("api_total", 0),
        "playwright_total": stats.get("playwright_total", 0),
        "cobertura_pct": stats.get("cobertura_pct", 0.0),
        "fuga_nominal": fuga,
    }


def _build_que_falta(fecha: str) -> List[Dict]:
    """
    Panel QUÉ FALTA (§2.4): picks watchlist con primera condición fallida
    para entrar a FAVORITOS_COMPUESTOS.  REPORTE_SOLO, cero lógica de gate.
    """
    er = _latest(str(REPORTS / f"edge_report_{fecha.replace('-', '')}*.json"))
    data = _load_json(er)
    if not data or isinstance(data, list):
        return []

    resultado: List[Dict] = []
    for p in data.get("watchlist", []):
        p_mod = float(p.get("p_modelo", 0) or 0)
        cuota_fav = float(p.get("cuota_favorito", 99) or 99)
        cuota_riv = float(p.get("cuota_rival", 99) or 99)
        conf = (p.get("confidence_flag") or "").upper()
        rk_fav = int(p.get("ranking_favorito") or 9999)
        rk_riv = int(p.get("ranking_rival") or 9999)
        rk_gap = rk_riv - rk_fav
        nombre = p.get("favorito_predicho") or p.get("partido", "?")

        # Cond 1 — favorito claro
        c1_p    = p_mod >= _FAV_P_MIN
        c1_cuota = (cuota_fav <= _FAV_CUOTA_CLARA_MAX and conf != "LOW")
        c1_rank  = (rk_gap > _FAV_RANKING_GAP and cuota_fav <= 1.60)
        if not (c1_p or c1_cuota or c1_rank):
            delta = _FAV_P_MIN - p_mod
            detalle = f"p_modelo={p_mod:.3f} < {_FAV_P_MIN} (faltan {delta:.3f})"
            if not c1_cuota and cuota_fav < 90:
                detalle += f" | cuota_fav={cuota_fav} > {_FAV_CUOTA_CLARA_MAX} o conf={conf}"
            resultado.append({"jugador": nombre, "condicion": "favorito_claro", "detalle": detalle})
            continue

        # Cond 2 — cuota rango pierna
        if cuota_fav < _FAV_LEG_MIN:
            resultado.append({"jugador": nombre, "condicion": "cuota_rango",
                               "detalle": f"cuota_fav={cuota_fav} < {_FAV_LEG_MIN} (piso pierna)"})
            continue
        if cuota_fav > _FAV_LEG_MAX:
            delta = cuota_fav - _FAV_LEG_MAX
            resultado.append({"jugador": nombre, "condicion": "cuota_rango",
                               "detalle": f"cuota_fav={cuota_fav} > {_FAV_LEG_MAX:.2f} (techo, delta +{delta:.2f})"})
            continue

        # Cond 3 — model=bookie
        if cuota_fav >= cuota_riv:
            resultado.append({"jugador": nombre, "condicion": "model_neq_bookie",
                               "detalle": f"cuota_fav={cuota_fav} >= cuota_rival={cuota_riv} (bookie discrepa)"})
            continue

    return resultado[:10]


def render_html(state: Dict[str, Any]) -> str:
    """Renderiza el HTML completo del desk (no frameworks, no CDN)."""
    risk = state.get("p4_risk", {})
    gov_code = risk.get("governor_code", 0)
    kgr = risk.get("kgr_sesion", 1.0)
    halt = gov_code >= 2 or kgr < 0

    accionables = accionable_ahora(state)

    # Paleta
    BG = "#0d1117"
    PANEL_BG = "#161b22"
    BORDER = "#30363d"
    GREEN = "#3fb950"
    AMBER = "#d29922"
    RED = "#f85149"
    GREY = "#8b949e"
    WHITE = "#e6edf3"
    BLUE = "#58a6ff"

    halt_banner = ""
    if halt:
        reason = "GOVERNOR BLOCK" if gov_code >= 2 else "KGR SESION < 0"
        halt_banner = f"""
        <div style="background:{RED};color:#fff;padding:18px;text-align:center;font-size:1.4em;
                    font-weight:bold;letter-spacing:2px;border-radius:6px;margin-bottom:18px;">
          *** DESK EN HALT — {reason} — REGLA-HF-5 ***<br>
          <span style="font-size:0.7em;font-weight:normal;">Paneles P1-P3 atenuados. No apostar.</span>
        </div>"""

    atenuado = "opacity:0.35;pointer-events:none;" if halt else ""

    def panel(title, content, badge="", badge_color=WHITE):
        bdg = f'<span style="background:{badge_color};color:#000;padding:2px 8px;border-radius:3px;font-size:0.75em;margin-left:8px;font-weight:bold;">{badge}</span>' if badge else ""
        return f"""
        <div style="background:{PANEL_BG};border:1px solid {BORDER};border-radius:8px;padding:16px;margin-bottom:14px;">
          <div style="color:{GREY};font-size:0.8em;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">
            {title}{bdg}
          </div>
          {content}
        </div>"""

    _tbl_n = [0]  # counter para IDs únicos de tablas sortables

    def row(cols, header=False):
        if header:
            cells = "".join(
                f'<th style="padding:4px 10px;color:{GREY};font-weight:bold;'
                f'cursor:pointer;user-select:none;white-space:nowrap;" '
                f'onclick="sortTable(this)" data-dir="">'
                f'{c}&nbsp;<span class="si"></span></th>'
                for c in cols
            )
            return f"<tr>{cells}</tr>"
        cells = "".join(f'<td style="padding:4px 10px;color:{WHITE};">{c}</td>' for c in cols)
        return f"<tr>{cells}</tr>"

    def table(headers, rows_data, empty_msg="Sin datos"):
        if not rows_data:
            return f'<p style="color:{GREY};font-size:0.85em;">{empty_msg}</p>'
        _tbl_n[0] += 1
        tid = f"st{_tbl_n[0]}"
        hdr = row(headers, header=True)
        body = "".join(row(r) for r in rows_data)
        return (
            f'<table id="{tid}" style="border-collapse:collapse;width:100%;font-size:0.85em;">'
            f'<thead>{hdr}</thead><tbody>{body}</tbody></table>'
        )

    # ── P4 RISK ──────────────────────────────────────────────────────────────
    gov_label = ["PASS", "WARN", "BLOCK", "BLOCK"][min(gov_code, 3)]
    gov_color = [GREEN, AMBER, RED, RED][min(gov_code, 3)]
    kgr_color = GREEN if kgr >= 0 else RED
    kill_switches = risk.get("kill_switches", {})
    ks_rows = [[k, ("ON" if v else "off")] for k, v in kill_switches.items()]
    exp_rows = [[e["jugador"], f"{e['pct']*100:.1f}%", ("WARN" if e["pct"] > 0.05 else "ok")]
                for e in risk.get("exposicion", [])]

    p4_content = f"""
      <div style="display:flex;gap:20px;margin-bottom:12px;">
        <div style="text-align:center;">
          <div style="color:{GREY};font-size:0.75em;">GOVERNOR</div>
          <div style="color:{gov_color};font-size:1.6em;font-weight:bold;">{gov_label}</div>
        </div>
        <div style="text-align:center;">
          <div style="color:{GREY};font-size:0.75em;">KGR SESION</div>
          <div style="color:{kgr_color};font-size:1.6em;font-weight:bold;">{kgr:.3f}</div>
        </div>
        <div style="text-align:center;">
          <div style="color:{GREY};font-size:0.75em;">STAKE SESION</div>
          <div style="color:{WHITE};font-size:1.6em;font-weight:bold;">${risk.get('stake_total',0):,.0f}</div>
        </div>
        <div style="text-align:center;">
          <div style="color:{GREY};font-size:0.75em;">BANKROLL</div>
          <div style="color:{WHITE};font-size:1.6em;font-weight:bold;">${risk.get('bankroll',0):,.0f}</div>
        </div>
      </div>
      <div style="display:flex;gap:16px;">
        <div style="flex:1;">
          <div style="color:{GREY};font-size:0.75em;margin-bottom:4px;">Kill-switches</div>
          {table(["Flag","Estado"], ks_rows, "Sin kill-switches")}
        </div>
        <div style="flex:1;">
          <div style="color:{GREY};font-size:0.75em;margin-bottom:4px;">Exposicion por jugador (cap 5%)</div>
          {table(["Jugador","%","Status"], exp_rows, "Sin concentracion")}
        </div>
      </div>"""

    p4_badge = gov_label
    p4_badge_color = gov_color
    p4_panel = panel("P4 RISK — Governor + Exposicion + Circuit Breakers", p4_content, p4_badge, p4_badge_color)

    # ── P2 BREAK BOARD ────────────────────────────────────────────────────────
    breaks = state.get("p2_break", {}).get("breaks", [])
    brk_rows = []
    for b in breaks:
        estado = b.get("estado", "")
        c = RED if estado == "BREAK_CONFIRMADO" else (AMBER if estado == "BREAK_POSIBLE" else GREY)
        brk_rows.append([
            f'<span style="color:{c};">{estado}</span>',
            b.get("jugador", ""),
            b.get("pick", ""),
            f"{b.get('drift_pct',0):.1f}%",
            b.get("hipotesis", "H100-01"),
        ])
    p2_badge = f"{sum(1 for b in breaks if b.get('estado')=='BREAK_CONFIRMADO')} CONFIRMADOS"
    p2_badge_color = RED if any(b.get("estado") == "BREAK_CONFIRMADO" for b in breaks) else GREY
    p2_content = f'<div style="{atenuado}">' + table(
        ["Estado", "Jugador", "Pick", "Drift%", "Gate"],
        brk_rows, "Sin breaks activos hoy"
    ) + "</div>"
    p2_panel = panel("P2 BREAK BOARD — Rupturas confirmadas (Nodo-100B H100-01)", p2_content, p2_badge, p2_badge_color)

    # ── P3 CONVERGENCE ────────────────────────────────────────────────────────
    conv_picks = state.get("p3_convergence", {}).get("picks", [])
    conv_rows = []
    for p in conv_picks:
        score = p.get("score_directo", 0) or 0
        sc_color = GREEN if score >= 3 else (AMBER if score >= 2 else GREY)
        rv = "RIVAL" if p.get("rival_value_flag") else ""
        gcs = "GCS" if p.get("gcs_active") else ""
        badges = " ".join(filter(None, [rv, gcs]))
        dir_ = p.get("direccion", "FAVORITO")
        if dir_ == "SPLIT":
            dir_html = (f'<span style="background:{AMBER};color:#000;padding:1px 4px;'
                        f'border-radius:3px;font-size:0.8em;font-weight:bold;" '
                        f'title="Señales contradictorias: score_fav={score} + rival_value">SPLIT</span>')
        elif dir_ == "RIVAL":
            dir_html = f'<span style="color:{AMBER};">RIVAL</span>'
        else:
            dir_html = f'<span style="color:{GREY};">FAV</span>'
        conv_rows.append([
            p.get("jugador", ""),
            f'<span style="color:{sc_color};font-weight:bold;">{score}</span>',
            p.get("confidence_flag", ""),
            p.get("markov_favorito", ""),
            dir_html,
            badges or "—",
        ])
    p3_panel = panel(
        "P3 CONVERGENCE — Meta-señal H98-01 (score>=3 = fila destacada)",
        f'<div style="{atenuado}">' + table(
            ["Jugador", "Score", "Conf", "Markov", "Dir", "Flags"],
            conv_rows, "Sin datos de convergencia (correr PASO 3b)"
        ) + "</div>"
    )

    # ── P1 TAPE ───────────────────────────────────────────────────────────────
    tape_entries = state.get("p1_tape", {}).get("entries", [])
    tape_rows = []
    for t in tape_entries:
        drift = t.get("drift_pct", 0) or 0
        direction = t.get("direction", "")
        dc = GREEN if direction == "CONFIRMA" else (RED if direction == "ALEJA" else GREY)
        tape_rows.append([
            t.get("jugador", ""),
            f"{drift:+.1f}%",
            f'<span style="color:{dc};">{direction}</span>',
            t.get("vel_zscore", "—"),
            t.get("ts", ""),
        ])
    p1_panel = panel(
        "P1 TAPE — Momentum de línea (drift% + velocity Nodo-71)",
        f'<div style="{atenuado}">' + table(
            ["Jugador", "Drift%", "Dirección", "VelZ", "Hora"],
            tape_rows, "Sin datos live (live_edge_monitor no activo o sin partidos)"
        ) + "</div>"
    )

    # ── P5 EXECUTION (CLV) ────────────────────────────────────────────────────
    clv_picks = state.get("p5_execution", {}).get("picks", [])
    clv_rows = []
    for p in clv_picks:
        clv = p.get("clv", None)
        clv_str = f"{clv:+.1f}%" if clv is not None else "SIN DATO"
        clv_color = GREEN if (clv or 0) > 0 else (RED if (clv or 0) < 0 else GREY)
        clv_rows.append([
            p.get("jugador", ""),
            p.get("cuota_entry", ""),
            p.get("cuota_cierre", "SIN CIERRE"),
            f'<span style="color:{clv_color};">{clv_str}</span>',
            p.get("estrategia", ""),
        ])
    clv_median = state.get("p5_execution", {}).get("clv_median", None)
    med_str = f"CLV mediana: {clv_median:+.1f}%" if clv_median is not None else ""
    p5_panel = panel(
        f"P5 EXECUTION — CLV por pick (Nodo-101) {med_str}",
        table(["Jugador", "Entry", "Cierre", "CLV", "Estrategia"], clv_rows, "Sin picks con snapshot de cierre")
    )

    # ── P6 P&L ────────────────────────────────────────────────────────────────
    pnl_segs = state.get("p6_pnl", {}).get("segmentos", [])
    pnl_rows = []
    for s in pnl_segs:
        roi = s.get("roi", 0) or 0
        rc = GREEN if roi > 0 else (RED if roi < 0 else GREY)
        grad = "GRADUADA" if s.get("graduada") else ("pre-grad" if s.get("n", 0) > 0 else "n/a")
        gc = GREEN if s.get("graduada") else AMBER
        pnl_rows.append([
            s.get("nombre", ""),
            str(s.get("n", 0)),
            s.get("hit_pct", "—"),
            f'<span style="color:{rc};">{roi:+.1f}%</span>',
            f'<span style="color:{gc};">{grad}</span>',
        ])
    p6_panel = panel(
        "P6 P&L — Blotter por estrategia (graduadas arriba)",
        table(["Estrategia", "n", "Hit%", "ROI", "Estado"], pnl_rows,
              "Sin settled picks (correr shadow_book --report)")
    )

    # ── P8 MULTI-BOOK ─────────────────────────────────────────────────────────
    p8_books = state.get("p8_books", {})
    p8_picks = p8_books.get("picks", {})
    p8_rows = []
    for jug_key, bp in p8_picks.items():
        div = bp.get("divergencia_pct", 0) or 0
        div_color = AMBER if div > 8 else WHITE
        div_badge = (
            f' <span style="background:{AMBER};color:#000;padding:1px 5px;'
            f'font-size:0.7em;border-radius:3px;">ATN div {div:.1f}%</span>'
            if div > 8 else ""
        )
        casa_gana = bp.get("casa", "")
        cuota_gana = bp.get("cuota", "")
        gain = bp.get("gain_pct", 0) or 0
        gain_str = f"+{gain:.1f}%" if gain > 0 else f"{gain:.1f}%"
        gain_color = GREEN if gain > 0 else GREY
        # ARB flag
        arb_html = ""
        if bp.get("arb_flag"):
            rc = bp.get("rival_cuota", "?")
            rk = bp.get("rival_casa", "?")
            arb_html = (
                f'<span style="background:#00ff00;color:#000;padding:1px 5px;'
                f'font-size:0.72em;border-radius:3px;font-weight:bold;" '
                f'title="ARB: fav @{cuota_gana} ({casa_gana}) + rival @{rc} ({rk})">ARB</span> '
            )
        _bp_src     = bp.get("_source", "ml")
        _bp_partido = bp.get("_partido", "")
        _bp_mercado = bp.get("_mercado", "")
        if _bp_src == "games" and _bp_partido:
            # GAMES: mostrar partido completo + señal de apuesta
            _jug_cell = (
                f'<span style="font-size:0.75em;color:{GREY};">GAMES</span> '
                f'<b>{_bp_partido}</b>'
                f'<br><span style="color:{AMBER};font-size:0.8em;">➜ {_bp_mercado}</span>'
            )
        else:
            _jug_cell = bp.get("jugador", jug_key)
        p8_rows.append([
            arb_html + _jug_cell,
            str(bp.get("betplay_cuota", "—")),
            str(bp.get("rushbet_cuota", "—")),
            str(bp.get("wplay_cuota", "—")),
            f'<span style="color:{div_color};">{div:.1f}%{div_badge}</span>',
            f'<b style="color:{GREEN};">{casa_gana} @{cuota_gana}</b> '
            f'<span style="color:{gain_color};">({gain_str})</span>',
        ])
    cache_age = p8_books.get("cache_age_s", 0)
    from_cache = p8_books.get("from_cache", False)
    is_stale = from_cache and cache_age > 300
    feeds_str = ", ".join(p8_books.get("feeds", []) or [])
    cache_note = f"cache {cache_age}s" if from_cache else "datos frescos"
    stale_badge = (
        f' <span style="background:{RED};color:#fff;padding:1px 5px;font-size:0.72em;'
        f'border-radius:3px;font-weight:bold;">STALE {cache_age}s</span>'
        if is_stale else ""
    )
    middle_note = (
        f'<p style="color:{GREY};font-size:0.78em;margin:4px 0;">MIDDLE: gateado — sin datos O/U en feeds actuales (D116-03)</p>'
    )
    p8_badge = f"{len(p8_rows)} picks" if p8_rows else "SIN DATOS"
    p8_badge_color = BLUE if p8_rows else GREY
    p8_panel = panel(
        f"P8 MULTI-BOOK — Router X1 Nodo-111 | feeds: {feeds_str or 'ninguno'} | {cache_note} (TTL 10min){stale_badge}",
        f'<div style="{atenuado}">' + table(
            ["Jugador", "betplay", "rushbet", "wplay", "div%", "Mejor precio"],
            p8_rows,
            "Sin datos (dual_book_cache.json vacío — se genera en próximo ciclo de live_edge_monitor)"
        ) + middle_note + "</div>",
        p8_badge, p8_badge_color,
    )

    # ── P9 EXECUTION ROUTER ───────────────────────────────────────────────────
    # Lee p8_books (ya computado, sin nueva llamada de red). Vista acción.
    _p8        = state.get("p8_books", {})
    _p8_picks  = _p8.get("picks", {})
    p9_rows: List = []
    _gains: List[float] = []
    for _jug_key, _bp in _p8_picks.items():
        _jug    = _bp.get("jugador", _jug_key)
        _plan   = _bp.get("cuota_plan", 0)
        _casa   = _bp.get("casa", "?")
        _cuota  = _bp.get("cuota", 0)
        _gain   = float(_bp.get("gain_pct", 0) or 0)
        _arb    = _bp.get("arb_flag", False)
        if _gain > 0:
            _gains.append(_gain)
        _gain_str   = f"+{_gain:.1f}%" if _gain > 0 else f"{_gain:.1f}%"
        _gain_color = GREEN if _gain > 0 else GREY
        _arb_html   = (
            f'<span style="background:#00ff00;color:#000;padding:1px 4px;'
            f'font-size:0.7em;border-radius:3px;font-weight:bold;">ARB</span> '
            if _arb else ""
        )
        p9_rows.append([
            _jug,
            f"@{_plan:.2f}" if _plan else "—",
            f"<b>{_casa}</b>",
            f"@{_cuota:.2f}" if _cuota else "—",
            f'<span style="color:{_gain_color};font-weight:bold;">{_gain_str}</span>{_arb_html}',
        ])
    _roi_extra_p9  = round(sum(_gains) / len(_gains), 2) if _gains else 0.0
    _n_cubiertos   = len(_p8_picks)
    _feeds_p9_str  = ", ".join(_p8.get("feeds", []) or [])
    _p9_badge      = f"+{_roi_extra_p9:.1f}% ROI extra" if _roi_extra_p9 > 0 else f"{_n_cubiertos} picks"
    _p9_badge_col  = GREEN if _roi_extra_p9 > 0 else (BLUE if _n_cubiertos else GREY)
    p9_panel = panel(
        f"P9 EXECUTION ROUTER — dónde ejecutar cada pick | feeds: {_feeds_p9_str or 'ninguno'}",
        table(
            ["Jugador", "Plan @", "Ejecutar en", "Cuota", "+% vs plan"],
            p9_rows,
            "Sin picks cubiertos en feeds (correr live_edge_monitor o PASO 3.7)",
        ),
        _p9_badge, _p9_badge_col,
    )

    # ── X2 STEAM-LAG ─────────────────────────────────────────────────────────
    # Lee p8_books.picks (divergencia_pct) + p1_tape (direction). Render puro.
    # Nodo-111 H111-01: líder mueve ≥15% → rezagada = stale price, ejecutar ahí.
    _tape_dir: Dict[str, str] = {}
    for _te in state.get("p1_tape", {}).get("entries", []):
        _jl = (_te.get("jugador") or "").lower()
        if _jl:
            _tape_dir[_jl] = _te.get("direction", "")

    X2_ALERT_PCT = 15.0   # umbral señal steam (H111-01)
    X2_INFO_PCT  = 10.0   # umbral informativo

    x2_rows: List = []
    x2_alerts = 0
    for _jk, _bp in _p8_picks.items():
        _div = float(_bp.get("divergencia_pct", 0) or 0)
        if _div < X2_INFO_PCT:
            continue
        _jug       = _bp.get("jugador", _jk)
        # Dinámica N-casas: leader=mínimo (ya se movió), stale=máximo (rezagada)
        _cuotas_raw = _bp.get("cuotas", {})
        _cuotas_n   = {k: float(v) for k, v in _cuotas_raw.items()
                       if isinstance(v, (int, float)) and float(v) > 0}
        if len(_cuotas_n) < 2:
            continue
        _leader_casa = min(_cuotas_n, key=_cuotas_n.__getitem__)
        _stale_casa  = max(_cuotas_n, key=_cuotas_n.__getitem__)
        _leader_c    = _cuotas_n[_leader_casa]
        _stale_c     = _cuotas_n[_stale_casa]
        _dir   = _tape_dir.get(_jug.lower(), "")
        _alert = _div >= X2_ALERT_PCT
        if _alert:
            x2_alerts += 1
        # Estado: STEAM OK si confirma modelo, ATN si aleja o sin dato
        if _alert and _dir == "CONFIRMA":
            _estado_html = (f'<span style="background:{GREEN};color:#000;padding:1px 5px;'
                            f'border-radius:3px;font-size:0.72em;font-weight:bold;">STEAM OK</span>')
        elif _alert:
            _dir_label = _dir if _dir else "sin tape"
            _estado_html = (f'<span style="background:{AMBER};color:#000;padding:1px 5px;'
                            f'border-radius:3px;font-size:0.72em;font-weight:bold;">'
                            f'ATN ({_dir_label})</span>')
        else:
            _estado_html = f'<span style="color:{GREY};font-size:0.85em;">gap info</span>'
        _div_color = GREEN if (_alert and _dir == "CONFIRMA") else (AMBER if _alert else GREY)
        x2_rows.append([
            _jug,
            f"{_leader_casa} @{_leader_c:.2f}",
            f'<b style="color:{GREEN};">{_stale_casa} @{_stale_c:.2f}</b>',
            f'<span style="color:{_div_color};font-weight:bold;">{_div:.1f}%</span>',
            f'<span style="color:{GREEN if _dir=="CONFIRMA" else (RED if _dir=="ALEJA" else GREY)};">'
            f'{_dir or "—"}</span>',
            _estado_html,
        ])
    _x2_badge     = f"{x2_alerts} STEAM" if x2_alerts else (f"{len(x2_rows)} gap" if x2_rows else "sin gap ≥10%")
    _x2_badge_col = GREEN if x2_alerts else (AMBER if x2_rows else GREY)
    x2_panel = panel(
        f"X2 STEAM-LAG — Divergencia entre casas (alert≥{X2_ALERT_PCT:.0f}% | info≥{X2_INFO_PCT:.0f}%) | H111-01",
        table(
            ["Jugador", "Leader (movida)", "Rezagada (ejecutar)", "Gap%", "Dirección", "Estado"],
            x2_rows,
            "Sin divergencia ≥10% entre casas (feeds sincronizados o wplay fuera de horario)",
        ),
        _x2_badge, _x2_badge_col,
    )

    # ── X3 GAMES SIGNAL (D133-06: estado live + convergencia) ────────────────
    _gs = state.get("p_games", {})
    x3_rows: List = []
    if _gs.get("disponible"):
        _x3_middles      = 0
        _en_vivo_count   = _gs.get("en_vivo_count", 0)
        _conv_activa     = _gs.get("convergencia_activa", False)

        # Banner CONVERGENCIA ACTIVA (D133-06)
        _conv_banner = ""
        if _conv_activa:
            _conv_banner = (
                f'<div style="background:{RED}22;border:2px solid {RED};border-radius:8px;'
                f'padding:10px 16px;margin-bottom:10px;text-align:center;font-size:1.0rem;'
                f'font-weight:700;color:{RED};">'
                f'CONVERGENCIA GAMES ACTIVA &mdash; {_en_vivo_count} ALTA EN VIVO &mdash; COMBO DISPARADO</div>'
            )
        elif _en_vivo_count > 0:
            _conv_banner = (
                f'<div style="background:{AMBER}22;border:1px solid {AMBER};border-radius:6px;'
                f'padding:8px 14px;margin-bottom:10px;font-size:0.85rem;color:{AMBER};">'
                f'{_en_vivo_count} señal(es) ALTA EN VIVO — esperando ≥2 para combo</div>'
            )

        for _sig in _gs.get("signals", []):
            _conf   = _sig.get("confianza", "")
            _conf_c = GREEN if _conf == "ALTA" else (AMBER if _conf == "MEDIA" else GREY)
            _dir    = _sig.get("direccion", "")
            _dir_c  = GREEN if _dir == "OVER" else (AMBER if _dir == "UNDER" else GREY)
            _gap    = _sig.get("gap")

            # Estado live (D133-05)
            _estado = _sig.get("estado_live", "PRE_PARTIDO")
            _est_c  = GREEN if _estado == "EN_VIVO" else (GREY if _estado == "TERMINADO" else "#8b949e")
            _est_lbl = (
                f'<span style="color:{GREEN};font-weight:bold;animation:blink 1s infinite;">EN VIVO</span>'
                if _estado == "EN_VIVO" else
                f'<span style="color:{GREY};">{_estado}</span>'
            )

            # Cuota live y drift (D133-05)
            _cuota_live = _sig.get("cuota_live")
            _drift      = _sig.get("drift_pct")
            _cuota_live_html = "—"
            if _cuota_live:
                _drift_c   = RED if (_drift or 0) > 10 else (AMBER if (_drift or 0) > 5 else GREEN)
                _drift_str = f' <span style="color:{_drift_c};font-size:0.8em;">({_drift:+.1f}%)</span>' if _drift else ""
                _cuota_live_html = f'@{_cuota_live:.2f}{_drift_str}'

            # Middle candidato
            _gr    = _sig.get("games_range", "")
            _linea = _sig.get("linea")
            _mid_html = "—"
            try:
                _parts = _gr.replace("+", "").split("-")
                _rlo, _rhi = float(_parts[0]), float(_parts[-1])
                if _dir == "OVER" and _linea is not None:
                    _needed = _rhi + 0.5
                    if _rlo >= float(_linea) and _rhi <= _needed:
                        _mid_html = f'<span style="color:{AMBER};font-weight:bold;">UNDER ≥{_needed:.1f}</span>'
                        _x3_middles += 1
                elif _dir == "UNDER" and _linea is not None:
                    _needed = _rlo - 0.5
                    if _rlo >= _needed and _rhi <= float(_linea):
                        _mid_html = f'<span style="color:{AMBER};font-weight:bold;">OVER ≤{_needed:.1f}</span>'
                        _x3_middles += 1
            except Exception:
                pass

            x3_rows.append([
                _sig.get("partido", ""),
                _est_lbl,
                _sig.get("mercado", ""),
                f'<span style="color:{_dir_c};font-weight:bold;">{_dir}</span>',
                str(_sig.get("linea", "—")),
                f'@{_sig["cuota"]:.2f}' if _sig.get("cuota") else "—",
                _cuota_live_html,
                f'{_gap:+.1f}j' if _gap is not None else "—",
                f'<span style="color:{_conf_c};">{_conf}</span>',
                _sig.get("games_range", ""),
                _mid_html,
            ])

        _x3_n       = _gs["n_apostar"]
        _x3_total   = _gs["n_partidos"]
        _mid_note   = f" | {_x3_middles} middle-candidatos" if _x3_middles else ""
        _live_note  = f" | {_en_vivo_count} en vivo" if _en_vivo_count else ""
        _x3_badge   = f"{_x3_n} señales{_mid_note}{_live_note}" if _x3_n else f"0/{_x3_total} sin señal"
        _x3_badge_c = (RED if _conv_activa else (AMBER if _en_vivo_count else (GREEN if _x3_n else GREY)))
        x3_panel = panel(
            f"X3 GAMES SIGNAL — Over/Under mercados Nodo-40 | {_gs['fuente']}",
            _conv_banner + table(
                ["Partido", "Estado", "Mercado", "Dir", "Línea", "Cuota Pre", "Cuota Live", "Gap", "Confianza", "Rango pred.", "Middle?"],
                x3_rows,
                "Sin señales accionables hoy (gap modelo-línea insuficiente)",
            ),
            _x3_badge, _x3_badge_c,
        )
    else:
        x3_panel = panel(
            "X3 GAMES SIGNAL — Over/Under mercados Nodo-40",
            f'<p style="color:{GREY};font-size:0.85em;">Sin reporte (correr PASO 3.6: python3 games_signal_calculator.py)</p>',
        )

    # ── X4 EVALUAR_GAMES — favoritos absolutos → UNDER juegos (Nodo-125) ────
    _x4 = state.get("p_evaluar_games", {})
    _x4_picks = _x4.get("picks", [])
    x4_rows: List = []
    for _p in _x4_picks:
        _res   = _p.get("resultado")
        _res_c = GREEN if _res == "WON" else (RED if _res == "LOST" else GREY)
        _res_s = f'<span style="color:{_res_c};">{_res or "PEND"}</span>'
        _under = _p.get("cuota_under")
        _under_s = f'<span style="color:{GREEN};font-weight:bold;">@{_under:.2f}</span>' if _under else \
                   f'<span style="color:{AMBER};">buscar</span>'
        x4_rows.append([
            _p.get("hora") or "?",
            _p.get("partido", "")[:38],
            f'{_p["conf"]:.0%}',
            f'@{_p["cuota_ml"]:.2f}',
            _under_s,
            _res_s,
        ])
    _x4_n         = _x4.get("n", 0)
    _x4_n_under   = _x4.get("n_con_under", 0)
    _x4_badge     = f"{_x4_n} picks / {_x4_n_under} con UNDER Kambi"
    _x4_badge_col = GREEN if _x4_n_under > 0 else (AMBER if _x4_n > 0 else GREY)
    x4_panel = panel(
        "X4 EVALUAR_GAMES — favoritos absolutos (cuota<1.30) → UNDER juegos (Nodo-125)",
        table(
            ["Hora", "Partido", "Conf", "CuotaML", "CuotaUNDER", "Resultado"],
            x4_rows,
            f'<span style="color:{GREY};font-size:0.85em;">'
            f'Sin picks evaluar_games hoy (correr PASO 3.6b: evaluar_games_bridge.py)</span>',
        ),
        _x4_badge, _x4_badge_col,
    )

    # ── P7 CLOCK ─────────────────────────────────────────────────────────────
    partidos = state.get("p7_clock", {}).get("partidos", [])
    clock_rows = []
    for p in partidos:
        inicio = p.get("inicio", "")
        ventana = p.get("ventana_live", "")
        snap = p.get("snapshot_deadline", "")
        clock_rows.append([p.get("jugador", ""), inicio, ventana, snap])
    p7_panel = panel(
        "P7 CLOCK — Ventanas de acción (live [-30,+45min] | snapshot -15min)",
        table(["Partido", "Inicio", "Ventana live", "Close-snapshot"], clock_rows,
              "Sin partidos (correr PASO 1 para zita file del dia)")
    )

    # ── Nodo-115: n_cal lookup y QUÉ FALTA ──────────────────────────────────
    _ncal_map   = state.get("p0_ncal", {})
    _odds_hist  = state.get("p10_odds_history", {})  # U4 sparkline
    _conf_data  = state.get("p12_conformal", {})      # U1 banda
    _q_global   = _conf_data.get("q_global")          # None si n_settled < gate
    _que_falta = state.get("p9_que_falta", [])

    # ── Accionables con U2+U3+drill-down+facetas (Nodo-115) ─────────────────
    if accionables:
        # Facet buttons por tipo
        tipos_uniq = list(dict.fromkeys(a["tipo"] for a in accionables))
        facet_btns = ""
        for t in ["TODOS"] + tipos_uniq:
            facet_btns += (f'<button class="facet-btn" data-f="{t}" '
                           f'onclick="filtrarTipo(\'{t}\')" '
                           f'style="background:#21262d;color:#e6edf3;border:1px solid #30363d;'
                           f'padding:3px 10px;border-radius:4px;cursor:pointer;margin-right:4px;'
                           f'font-size:0.78em;">{t}</button>')
        facet_bar = f'<div style="margin-bottom:8px;">{facet_btns}</div>'

        # Tabla manual para soportar data-attrs y drill-down
        TD = f'padding:4px 10px;border-bottom:1px solid {BORDER};'
        hdr_cells = "".join(
            f'<td style="{TD}color:{GREY};font-weight:bold;">{h}</td>'
            for h in ["Tipo", "Jugador/Pick", "Evidencia U2", "Gate U3", "Tendencia U4", "Conf U1", "Razonamiento"]
        )
        rows_html = f"<tr><tr>{hdr_cells}</tr></tr>"

        for idx, a in enumerate(accionables):
            ac = GREEN if a.get("color") == "green" else AMBER
            # CSS urgency classes migradas de live_dashboard_generator (Nodo-116 §A)
            tipo_cls = ""
            if a.get("tipo") == "BREAK_CONFIRMADO":
                tipo_cls = " break-confirmado"
            elif a.get("tipo") == "BREAK_POSIBLE":
                tipo_cls = " break-posible"
            elif a.get("tipo") == "COMBO_LIVE":
                tipo_cls = " combo-live"
            razon = linea_razonamiento(a)
            razon_short = razon[:90] + ("…" if len(razon) > 90 else "")

            # U2 — peso evidencia
            n_cal = _ncal_map.get(a.get("jugador", "").lower(), a.get("n_cal", 0))
            ev = _peso_evidencia(n_cal)
            u2_html = (f'<span style="font-family:monospace;color:{ev["color"]};font-size:0.85em;" '
                       f'title="n={ev["n"]} → {ev["pct"]}% peso propio (shrinkage n/(n+20))">'
                       f'{ev["bar"]} {ev["label"]}</span>')

            # U3 — distancia gate
            n_act = int(a.get("n_actual", 0))
            n_stp = int(a.get("n_stop", 0))
            u3_txt = _gate_barra(n_act, n_stp)
            u3_color = GREEN if "GRADUADA" in u3_txt else (AMBER if n_act > 0 else GREY)
            u3_html = (f'<span style="font-family:monospace;color:{u3_color};font-size:0.82em;">'
                       f'{a["hipotesis"]}: {u3_txt}</span>')

            # U4 — sparkline tendencia drift (últimos 4 ciclos)
            u4_spark = _sparkline_drift(a.get("jugador", ""), _odds_hist)
            u4_html = (
                f'<span style="font-family:monospace;color:{GREY};font-size:0.85em;">{u4_spark}</span>'
                if u4_spark else f'<span style="color:{GREY};font-size:0.75em;">–</span>'
            )

            # U1 — banda conformal p=X ±q_global
            p_m = float(a.get("p_modelo") or 0)
            if _q_global is not None and p_m > 0:
                cruza_be = (p_m - _q_global) <= 0.5 <= (p_m + _q_global)
                u1_color = AMBER if cruza_be else GREEN
                u1_label = "BANDA CRUZA BE" if cruza_be else f"±{_q_global:.2f}"
                u1_html  = (
                    f'<span style="font-family:monospace;color:{u1_color};font-size:0.82em;" '
                    f'title="p={p_m:.2f} q={_q_global:.2f}: [{p_m-_q_global:.2f}, {p_m+_q_global:.2f}]">'
                    f'p={p_m:.2f} {u1_label}</span>'
                )
            else:
                u1_html = f'<span style="color:{GREY};font-size:0.75em;">n&lt;{_conf_data.get("n_settled",0)}</span>'

            # Fila principal
            row_id = f"acc-{idx}"
            # Badge EDGE- visible sin abrir drill-down (task #67)
            _el = a.get("edge_live")
            edge_badge = (
                f' <span style="background:{RED};color:#fff;padding:1px 4px;border-radius:3px;'
                f'font-size:0.72em;font-weight:bold;" title="edge_live={_el:.2f} — NO APOSTAR">EDGE-</span>'
                if _el is not None and _el < 0 else ""
            )
            main_row = (
                f'<tr class="acc-row{tipo_cls}" id="{row_id}" data-tipo="{a["tipo"]}" '
                f'style="cursor:pointer;" onclick="toggleDetalle(\'{row_id}\')">'
                f'<td style="{TD}"><span style="color:{ac};font-weight:bold;">{a["tipo"]}</span>{edge_badge}</td>'
                f'<td style="{TD}color:{WHITE};">{a["jugador"]}</td>'
                f'<td style="{TD}">{u2_html}</td>'
                f'<td style="{TD}">{u3_html}</td>'
                f'<td style="{TD}">{u4_html}</td>'
                f'<td style="{TD}">{u1_html}</td>'
                f'<td style="{TD}"><span title="{razon}" style="color:{GREY};font-size:0.82em;">{razon_short}</span></td>'
                f'</tr>'
            )

            # Fila detalle (oculta, expandible al clic)
            mejor = a.get("mejor_precio", {})
            precio_detalle = ""
            if mejor and mejor.get("cuota"):
                precio_detalle = (f'<br><b>P8 mejor precio:</b> {mejor.get("casa","?")} '
                                  f'@{mejor.get("cuota","?")} (+{mejor.get("gain_pct",0):.1f}%)')
            senas = ", ".join(a.get("señales_activas", [])) or "—"
            # Cuota pre→live (BREAK rows)
            cuota_detalle = ""
            if a.get("cuota_pre") and a.get("cuota_live"):
                cp = a["cuota_pre"]; cl = a["cuota_live"]
                edge_live = a.get("edge_live")
                edge_color = RED if (edge_live is not None and edge_live < 0) else GREEN
                edge_txt = (f'<span style="color:{edge_color};"> edge_live={edge_live:+.2f}'
                            f'{"  ⚠ EDGE NEGATIVO — NO APOSTAR" if edge_live < 0 else ""}</span>'
                            ) if edge_live is not None else ""
                trig_txt = (f'<span style="color:{GREY};"> [monitor: NO DISPARAR]</span>'
                            if not a.get("trigger", False) else "")
                cuota_detalle = (f'<br><b>Cuota:</b> pre={cp} → live={cl} '
                                 f'(drift {a.get("drift_pct",0):+.1f}%){edge_txt}{trig_txt}')
            # COMBO_LIVE — bat link y hora de disparo
            bat_html = ""
            if a.get("tipo") == "COMBO_LIVE":
                bat_path = a.get("bat_link", "")
                fired_at = a.get("fired_at", "")
                bat_html = (
                    f'<br><b>Disparado:</b> {fired_at}'
                    + (f'<br><b>Combo .bat:</b> <span style="color:{AMBER};font-family:monospace;">{bat_path}</span>'
                       if bat_path else '<br><span style="color:{GREY};">(.bat no generado — sin trader_plans)</span>')
                )
            det_content = (
                f'<div style="padding:8px 12px;background:#0d1117;border-left:3px solid {ac};'
                f'font-size:0.82em;color:{WHITE};">'
                f'<b>Razonamiento completo:</b><br>'
                f'<span style="color:{GREY};font-family:monospace;">{razon}</span>'
                f'<br><br>'
                f'<b>Evidencia U2:</b> n_calibracion={ev["n"]} → {ev["pct"]}% peso propio'
                f'<br><b>Gate U3:</b> {a["hipotesis"]} {u3_txt}'
                f'<br><b>Señales activas:</b> {senas}'
                f'<br><b>meta_score:</b> {a.get("meta_score", "—")} | '
                f'<b>n_h2h:</b> {a.get("n_h2h", "—")} | '
                f'<b>drift:</b> {a.get("drift_pct", "—")}%'
                f'{cuota_detalle}'
                f'{precio_detalle}'
                f'{bat_html}'
                f'</div>'
            )
            det_row = (
                f'<tr id="det-{row_id}" style="display:none;">'
                f'<td colspan="7" style="padding:0;border-bottom:1px solid {BORDER};">'
                f'{det_content}</td></tr>'
            )
            rows_html += main_row + det_row

        acc_table = (f'<table style="border-collapse:collapse;width:100%;font-size:0.85em;">'
                     f'<tbody id="acc-tbody">{rows_html}</tbody></table>')
        acc_content = facet_bar + acc_table
    else:
        acc_content = f'<p style="color:{GREY};">{"DESK EN HALT — nada accionable" if halt else "Sin señales accionables ahora"}</p>'

    n_real = sum(1 for a in accionables if a.get("tipo") not in ("FAVORITOS_ZERO",))
    acc_panel = panel(
        "ACCIONABLE AHORA (P2∩P4∩estrategia graduada)",
        acc_content,
        f"{n_real} señales",
        GREEN if n_real and not halt else GREY,
    )

    # ── QUÉ FALTA (Nodo-115 §2.4) ────────────────────────────────────────────
    if _que_falta:
        qf_rows = []
        for qf in _que_falta:
            cond_color = {"favorito_claro": AMBER, "cuota_rango": BLUE,
                          "model_neq_bookie": GREY}.get(qf["condicion"], GREY)
            qf_rows.append([
                qf["jugador"],
                f'<span style="color:{cond_color};font-size:0.85em;">{qf["condicion"]}</span>',
                f'<span style="color:{GREY};font-size:0.82em;font-family:monospace;">{qf["detalle"]}</span>',
            ])
        qf_content = table(["Jugador", "Filtro fallido", "Distancia / Detalle"], qf_rows)
    else:
        qf_content = f'<p style="color:{GREY};font-size:0.85em;">Sin casi-accionables en watchlist (correr PASO 3)</p>'

    # ── DATA — Embudo Crosswalk Nodo-118 §5 ──────────────────────────────────
    _d = state.get("p_data", {})
    if _d.get("disponible"):
        _cob = _d["cobertura_pct"]
        _cob_color = GREEN if _cob >= 85 else (AMBER if _cob >= 60 else RED)
        _data_badge = f"{_cob:.1f}%"
        _fuga_rows = [[f["partido"], str(f["score"])] for f in _d.get("fuga_nominal", [])]
        _data_content = (
            f'<div style="margin-bottom:8px;">'
            f'Kambi: <b>{_d["api_total"]}</b> | Playwright: <b>{_d["playwright_total"]}</b> | '
            f'Join auto: <b>{_d["joins"]}</b> | Cuarentena: <b>{_d["cuarentena"]}</b> | '
            f'Single-K: <b>{_d["single_kambi"]}</b> | Single-FS: <b>{_d["single_fs"]}</b>'
            f'</div>'
        ) + table(
            ["Fuga (cuarentena) — partido", "score"],
            _fuga_rows,
            "Sin fugas — cobertura perfecta" if not _fuga_rows else "",
        )
        data_panel = panel(
            f"DATA — Embudo Crosswalk Nodo-118 §5 | {_d['fecha']}",
            _data_content,
            badge=_data_badge,
            badge_color=_cob_color,
        )
    else:
        data_panel = panel(
            "DATA — Embudo Crosswalk Nodo-118 §5",
            "Sin ledger para hoy (correr PASO 1.5: python3 scraping/match_ledger.py --build)",
        )

    que_falta_panel = panel(
        "QUÉ FALTA — casi-accionables (FAVORITOS_COMPUESTOS) + condición exacta",
        qf_content,
        f"{len(_que_falta)} candidatos" if _que_falta else "",
        AMBER if _que_falta else GREY,
    )

    fecha = state.get("fecha", "")
    ts = state.get("ts", "")
    freshness = state.get("data_freshness", "")
    freshness_note = f' | {freshness}' if freshness else ""
    refresh_note = f'<p style="color:{GREY};font-size:0.75em;text-align:right;">Auto-refresh 12s{freshness_note} | <span id="desk-ts">{ts}</span></p>'

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <!-- refresh via fetch() — preserva filtros y drill-downs (§2.5 Nodo-115) -->
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Live Trading Desk — {fecha}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: {BG}; color: {WHITE}; font-family: 'Consolas','Monaco',monospace; padding: 18px; }}
    h1 {{ color: {BLUE}; font-size: 1.1em; margin-bottom: 14px; letter-spacing: 1px; }}
    td {{ border-bottom: 1px solid {BORDER}; }}
    tr:last-child td {{ border-bottom: none; }}
    a {{ color: {BLUE}; }}
    @keyframes blink {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.25; }} }}
    .break-confirmado {{ animation: blink 1s step-start infinite; background: #3a0a0a !important; color: {RED} !important; }}
    .break-posible {{ background: #2d1a00 !important; color: #e67e22 !important; }}
    .combo-live {{ background: #1a1200 !important; border-left: 3px solid #f0a500; }}
  </style>
</head>
<body>
  <h1>LIVE TRADING DESK &nbsp;|&nbsp; {fecha} &nbsp;|&nbsp; Nodo-109</h1>
  {refresh_note}
  {halt_banner}
  {acc_panel}
  {p4_panel}
  {p2_panel}
  {p3_panel}
  {p5_panel}
  {p6_panel}
  {p8_panel}
  {p9_panel}
  {x2_panel}
  {x3_panel}
  {data_panel}
  {que_falta_panel}
  {p1_panel}
  {p7_panel}
  <script>
  // Estado cliente — preservado entre refreshes (§2.5 Nodo-115)
  var _activeFilter = 'TODOS';
  var _openRows = {{}};

  function filtrarTipo(tipo) {{
    _activeFilter = tipo;
    document.querySelectorAll('.acc-row').forEach(function(r) {{
      var t = r.getAttribute('data-tipo');
      r.style.display = (tipo === 'TODOS' || t === tipo) ? '' : 'none';
      var det = document.getElementById('det-' + r.id);
      if (det && !_openRows[r.id]) det.style.display = 'none';
    }});
    document.querySelectorAll('.facet-btn').forEach(function(b) {{
      b.style.background = b.getAttribute('data-f') === tipo ? '#58a6ff' : '#21262d';
      b.style.color = b.getAttribute('data-f') === tipo ? '#000' : '#e6edf3';
    }});
  }}

  function toggleDetalle(rowId) {{
    var d = document.getElementById('det-' + rowId);
    if (!d) return;
    var open = d.style.display === 'table-row';
    d.style.display = open ? 'none' : 'table-row';
    _openRows[rowId] = !open;
  }}

  function _reapplyState() {{
    // Re-aplicar filtro activo
    document.querySelectorAll('.acc-row').forEach(function(r) {{
      var t = r.getAttribute('data-tipo');
      r.style.display = (_activeFilter === 'TODOS' || t === _activeFilter) ? '' : 'none';
    }});
    // Re-abrir filas expandidas
    Object.keys(_openRows).forEach(function(id) {{
      if (_openRows[id]) {{
        var d = document.getElementById('det-' + id);
        if (d) d.style.display = 'table-row';
      }}
    }});
  }}

  function sortTable(th) {{
    var tbl  = th.closest('table');
    var tbody = tbl.querySelector('tbody');
    var ths  = Array.from(th.parentElement.children);
    var col  = ths.indexOf(th);
    var asc  = th.getAttribute('data-dir') !== 'asc';
    var dir  = asc ? 1 : -1;
    ths.forEach(function(h) {{
      h.setAttribute('data-dir', '');
      var si = h.querySelector('.si');
      if (si) si.textContent = '';
    }});
    th.setAttribute('data-dir', asc ? 'asc' : 'desc');
    var si = th.querySelector('.si');
    if (si) si.textContent = asc ? '▲' : '▼';
    Array.from(tbody.querySelectorAll('tr'))
      .sort(function(a, b) {{
        var av = a.children[col] ? a.children[col].textContent.trim() : '';
        var bv = b.children[col] ? b.children[col].textContent.trim() : '';
        var an = parseFloat(av), bn = parseFloat(bv);
        if (!isNaN(an) && !isNaN(bn)) return dir * (an - bn);
        return dir * av.localeCompare(bv);
      }})
      .forEach(function(r) {{ tbody.appendChild(r); }});
  }}

  function autoRefresh() {{
    fetch('/', {{cache: 'no-store'}})
      .then(function(r) {{ return r.text(); }})
      .then(function(html) {{
        var parser = new DOMParser();
        var doc = parser.parseFromString(html, 'text/html');
        // Reemplazar tbody de accionables
        var newTbody = doc.getElementById('acc-tbody');
        var curTbody = document.getElementById('acc-tbody');
        if (newTbody && curTbody) {{
          curTbody.innerHTML = newTbody.innerHTML;
        }}
        // Actualizar timestamp
        var newTs = doc.getElementById('desk-ts');
        var curTs = document.getElementById('desk-ts');
        if (newTs && curTs) curTs.textContent = newTs.textContent;
        // Re-aplicar estado del operador
        _reapplyState();
      }})
      .catch(function() {{}});  // silencioso — reintenta en 12s
    setTimeout(autoRefresh, 12000);
  }}
  setTimeout(autoRefresh, 12000);
  </script>
</body>
</html>"""
    return html


# ══════════════════════════════════════════════════════════════════════════════
# BUILDERS DE PANELES (leen archivos, toleran ausencias)
# ══════════════════════════════════════════════════════════════════════════════

def _latest(pattern: str) -> Optional[Path]:
    files = sorted(glob.glob(pattern))
    return Path(files[-1]) if files else None


def _load_json(path: Optional[Path]) -> Any:
    if not path or not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _build_p4_risk(fecha: str) -> Dict:
    """P4: governor exit code + KGR + kill-switches + exposicion."""
    # Governor
    gov_script = BASE_DIR / "combo_governor.py"
    gov_code = 0
    bankroll = 125000
    stake_total = 0

    # Leer bankroll del trader_plan más reciente
    tp = _latest(str(REPORTS / f"trader_plan_{fecha.replace('-','')}*.json"))
    tp_data = _load_json(tp)
    if tp_data:
        bankroll = tp_data.get("bankroll", bankroll)
        stake_total += sum(p.get("stake", 0) for p in tp_data.get("picks", []))

    # Combos de hoy (apuestas_*.json)
    for ap_file in sorted(glob.glob(str(REPORTS / f"apuestas_{fecha.replace('-','')}*.json"))):
        ap = _load_json(Path(ap_file))
        if ap and isinstance(ap, dict):
            for pick in ap.get("picks", []):
                stake_total += pick.get("stake", 0)

    # Governor subprocess
    try:
        if gov_script.exists():
            res = subprocess.run(
                [sys.executable, str(gov_script), "--bankroll", str(int(bankroll))],
                capture_output=True, text=True, timeout=15,
            )
            gov_code = res.returncode
    except Exception:
        gov_code = 0

    # KGR de sesión desde shadow book del día
    kgr = _kgr_sesion(fecha)

    # Kill-switches (constantes conocidas)
    kill_switches = {
        "MOTOR_DEFENSIVE": True,   # S107-E activo mientras H107-01 acumula
        "GCS_GATE_ENABLED": True,  # H60-01 GRADUADA
        "CAPA2_ENABLED": False,    # gate n>=30 settled CAPA2
    }

    # Exposicion por jugador (lee apuestas del día)
    exposicion = _exposicion_hoy(fecha, bankroll)

    return {
        "governor_code": gov_code,
        "kgr_sesion": kgr,
        "bankroll": bankroll,
        "stake_total": stake_total,
        "kill_switches": kill_switches,
        "exposicion": exposicion,
    }


def _kgr_sesion(fecha: str) -> float:
    """Lee el KGR de la sesión del trader_plan más reciente."""
    tp = _latest(str(REPORTS / f"trader_plan_{fecha.replace('-','')}*.json"))
    data = _load_json(tp)
    if data:
        return float(data.get("kgr", data.get("kgr_sesion", 1.0)))
    return 1.0


def _exposicion_hoy(fecha: str, bankroll: float) -> List[Dict]:
    """Suma stakes por jugador en picks del día."""
    jugador_stake: Dict[str, float] = {}
    for ap_file in sorted(glob.glob(str(REPORTS / f"apuestas_{fecha.replace('-','')}*.json"))):
        ap = _load_json(Path(ap_file))
        if not ap:
            continue
        for pick in (ap.get("picks", []) if isinstance(ap, dict) else []):
            j = pick.get("jugador", pick.get("favorito", "?"))
            jugador_stake[j] = jugador_stake.get(j, 0) + float(pick.get("stake", 0))
    if bankroll <= 0:
        return []
    return [
        {"jugador": j, "stake": s, "pct": s / bankroll}
        for j, s in sorted(jugador_stake.items(), key=lambda x: -x[1])
        if s > 0
    ]


def _build_p1_tape(fecha: str) -> Dict:
    """P1: drift% del live_edge_monitor."""
    le = _latest(str(REPORTS / f"live_edge_{fecha.replace('-','')}*.json"))
    data = _load_json(le)
    if not data:
        return {"entries": [], "source": "SIN DATO — live_edge_monitor no activo"}

    entries = []
    for pick in data.get("picks_chequeados_data", data.get("picks", data.get("partidos", []))):
        if not isinstance(pick, dict):
            continue
        if pick.get("cuota_live") is None:  # sin dato live — no mostrar en tape
            continue
        jugador = pick.get("favorito", pick.get("jugador", pick.get("home", "")))
        if not jugador or jugador == "?":
            continue
        drift = pick.get("drift_pct", pick.get("drift", 0)) or 0
        # Convención monitor: drift = (cuota_pre - cuota_live)/cuota_pre
        # drift > 0 → cuota bajó → CONFIRMA | drift < 0 → cuota subió → ALEJA
        direction = "CONFIRMA" if drift > 0 else ("ALEJA" if drift < 0 else "NEUTRO")
        entries.append({
            "jugador": jugador,
            "drift_pct": drift,
            "direction": direction,
            "vel_zscore": pick.get("velocity_zscore", pick.get("vel_z", "—")),
            "ts": pick.get("ts", data.get("ts", "")),
        })
    return {"entries": entries}


def _build_p2_break(fecha: str) -> Dict:
    """P2: break_state de Nodo-100B."""
    le = _latest(str(REPORTS / f"live_edge_{fecha.replace('-','')}*.json"))
    data = _load_json(le)
    if not data:
        return {"breaks": [], "source": "SIN DATO — live_edge_monitor no activo"}

    breaks = []
    for pick in data.get("picks_chequeados_data", data.get("picks", data.get("partidos", []))):
        if not isinstance(pick, dict):
            continue
        bs = pick.get("break_state", "NORMAL")
        if bs != "NORMAL":
            breaks.append({
                "estado": bs,
                "jugador": pick.get("favorito", pick.get("jugador", "")),
                "pick": pick.get("favorito", pick.get("pick", "")),
                "drift_pct": pick.get("drift_pct", 0),
                "hipotesis": "H100-01",
                "n_actual": pick.get("n_fired", 0),
                "p_modelo": pick.get("p_modelo", 0),
                "cuota_pre": pick.get("cuota_pre"),
                "cuota_live": pick.get("cuota_live"),
                "edge_live": pick.get("edge_live"),
                "trigger": pick.get("trigger", False),
                "senales": pick.get("senales", []),
                "partido": pick.get("partido", ""),
            })
    return {"breaks": breaks}


def _build_p3_convergence(fecha: str) -> Dict:
    """P3: meta_signal_score + rival_value_flag + gcs_active."""
    er = _latest(str(REPORTS / f"edge_report_{fecha.replace('-','')}*.json"))
    data = _load_json(er)
    if not data:
        return {"picks": [], "source": "SIN DATO — correr PASO 3"}

    raw = data if isinstance(data, list) else (data.get("apostar") or []) + (data.get("watchlist") or [])
    picks = []
    for p in raw:
        score = p.get("score_directo", 0) or 0
        rv = bool(p.get("rival_value_flag"))
        gcs = bool(p.get("gcs_active"))
        if score >= 2 or rv or gcs:
            # señales_activas: lista de etiquetas para linea_razonamiento
            senas = []
            if p.get("confidence_flag") == "STRONG":
                senas.append("STRONG")
            if p.get("markov_favorito") in ("HOT", "WARM"):
                senas.append(p["markov_favorito"])
            elo_fav = p.get("elo_favorito") or 0
            elo_riv = p.get("elo_rival") or 0
            if elo_fav - elo_riv > 100:
                senas.append("ELO_DOM")
            if p.get("rfi_fav") and p.get("rfi_tier", 0) != 0:
                senas.append("RFI")
            # direccion: FAVORITO normal | RIVAL cuando solo rv | SPLIT cuando ambos (Nodo-98)
            if score >= 2 and rv:
                direccion = "SPLIT"
            elif rv:
                direccion = "RIVAL"
            else:
                direccion = "FAVORITO"
            picks.append({
                "jugador": p.get("favorito_predicho", p.get("favorito", p.get("jugador", ""))),
                "score_directo": score,
                "señales_activas": senas,
                "n_h2h": p.get("n_h2h"),
                "clv": p.get("clv"),
                "confidence_flag": p.get("confidence_flag", ""),
                "markov_favorito": p.get("markov_favorito", ""),
                "rival_value_flag": rv,
                "rival": p.get("rival", ""),
                "gcs_active": gcs,
                "direccion": direccion,
            })
    picks.sort(key=lambda x: -(x.get("score_directo") or 0))
    return {"picks": picks}


def _build_p5_execution(fecha: str) -> Dict:
    """P5: CLV por pick abierto (Momento 2, Nodo-101)."""
    sb_file = SB_DIR / f"sb_{fecha}.jsonl"
    if not sb_file.exists():
        return {"picks": [], "clv_median": None, "source": "SIN DATO"}

    picks = []
    clvs = []
    try:
        with open(sb_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("event") != "pick_logged":
                    continue
                snap = rec.get("pick_snapshot", {})
                cuota_cierre = snap.get("cierre_kambi")
                cuota_entry = snap.get("cuota_favorito", snap.get("cuota", 0))
                clv = None
                if cuota_cierre and cuota_entry:
                    clv = (float(cuota_entry) / float(cuota_cierre) - 1) * 100
                    clvs.append(clv)
                picks.append({
                    "jugador": rec.get("jugador", ""),
                    "cuota_entry": cuota_entry,
                    "cuota_cierre": cuota_cierre or "pendiente",
                    "clv": round(clv, 2) if clv is not None else None,
                    "estrategia": snap.get("estrategia", "APOSTAR"),
                })
    except Exception:
        pass

    med = sorted(clvs)[len(clvs) // 2] if clvs else None
    return {"picks": picks, "clv_median": round(med, 2) if med else None}


def _build_p6_pnl(fecha: str) -> Dict:
    """P6: segmentos de shadow_book --report (parser multi-línea) + RIVAL_VALUE + MOTOR split."""
    import re as _re
    segmentos = []
    try:
        res = subprocess.run(
            [sys.executable, str(BASE_DIR / "shadow_book.py"), "--report"],
            capture_output=True, text=True, timeout=30, cwd=str(BASE_DIR)
        )
        # Parser de bloques: bloques separados por línea vacía
        blocks: list = []
        current: list = []
        for line in res.stdout.splitlines():
            if line.strip() == "":
                if current:
                    blocks.append(current)
                current = []
            else:
                current.append(line)
        if current:
            blocks.append(current)

        for block in blocks:
            m_seg = _re.match(r'\s+SEGMENTO:\s+(.+)', block[0])
            if not m_seg:
                continue
            nombre = m_seg.group(1).strip()
            n, hit_pct, roi = 0, "?", 0.0
            for bl in block[1:]:
                mn = _re.search(r'n=(\d+)\s+hit%=([\d\.]+)', bl)
                if mn:
                    n, hit_pct = int(mn.group(1)), mn.group(2)
                mr = _re.search(r'ROI flat 1u:\s*([\-\d\.]+)%', bl)
                if mr:
                    roi = float(mr.group(1))
            graduada = any(k in nombre for k in ("H60-01", "GCS", "gcs"))
            segmentos.append({"nombre": nombre, "n": n, "hit_pct": hit_pct, "roi": roi, "graduada": graduada})
    except Exception:
        pass

    # Segmentos adicionales directo de jsonl: RIVAL_VALUE y MOTOR cuota split
    # Schema real: resolucion.resultado='WON'|'LOST', pick_snapshot.* para flags
    rival_n, rival_wins = 0, 0
    motor_lo_n, motor_lo_w = 0, 0   # cuota ≤ 2.5
    motor_hi_n, motor_hi_w = 0, 0   # cuota > 2.5
    for sb_f in sorted(SB_DIR.glob("sb_*.jsonl")):
        try:
            for line in sb_f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                res = r.get("resolucion") or {}
                resultado = res.get("resultado")  # 'WON' | 'LOST' | None (abierto)
                if not resultado:
                    continue  # skip picks sin resolver
                snap = r.get("pick_snapshot", {})
                won = resultado == "WON"
                # RIVAL_VALUE
                if snap.get("rival_value_flag"):
                    rival_n += 1
                    if won:
                        rival_wins += 1
                # MOTOR cuota split
                cuota = float(snap.get("cuota_favorito") or 0)
                if cuota > 0:
                    if cuota <= 2.5:
                        motor_lo_n += 1
                        if won:
                            motor_lo_w += 1
                    else:
                        motor_hi_n += 1
                        if won:
                            motor_hi_w += 1
        except Exception:
            pass
    if rival_n > 0:
        segmentos.append({"nombre": "RIVAL_VALUE (H88-01)", "n": rival_n,
                          "hit_pct": str(round(rival_wins / rival_n * 100, 1)),
                          "roi": 0.0, "graduada": False})
    if motor_lo_n > 0:
        segmentos.append({"nombre": "MOTOR cuota≤2.5", "n": motor_lo_n,
                          "hit_pct": str(round(motor_lo_w / motor_lo_n * 100, 1)),
                          "roi": 0.0, "graduada": False})
    if motor_hi_n > 0:
        segmentos.append({"nombre": "MOTOR cuota>2.5", "n": motor_hi_n,
                          "hit_pct": str(round(motor_hi_w / motor_hi_n * 100, 1)),
                          "roi": 0.0, "graduada": False})

    # Ordenar: graduadas primero, luego por n desc
    segmentos.sort(key=lambda s: (0 if s["graduada"] else 1, -s.get("n", 0)))
    return {"segmentos": segmentos}


def _build_p7_clock(fecha: str) -> Dict:
    """P7: ventanas de acción por partido."""
    # Lee zita file del día
    zita = _latest(str(BASE_DIR / f"data/zita_tennis_matches_{fecha.replace('-','')}*.json"))
    data = _load_json(zita)
    if not data:
        return {"partidos": [], "source": "SIN DATO — correr PASO 1"}

    partidos = []
    matches = data if isinstance(data, list) else data.get("matches", [])
    for m in matches[:20]:  # máx 20 en clock
        inicio = m.get("start_time", m.get("hora", ""))
        jugadores = f"{m.get('home', m.get('jugador_local','?'))} vs {m.get('away', m.get('jugador_visitante','?'))}"
        # Ventana live: inicio - 30min a inicio + 45min
        partidos.append({
            "jugador": jugadores,
            "inicio": inicio,
            "ventana_live": f"-30min a +45min desde {inicio}",
            "snapshot_deadline": f"~15min antes de {inicio}",
        })
    return {"partidos": partidos}


def _favoritos_hoy(fecha: str) -> Optional[Dict]:
    """
    Busca output de favoritos_combo_builder.py del día en apuestas_*.json.
    Retorna dict con n_combos si corrió hoy, None si no corrió.
    """
    if not fecha:
        return None
    fecha_compact = fecha.replace("-", "")
    n_combos = 0
    found = False
    for ap_file in sorted(glob.glob(str(REPORTS / f"apuestas_{fecha_compact}*.json"))):
        ap = _load_json(Path(ap_file))
        if not ap:
            continue
        # favoritos_combo_builder escribe estrategia='FAVORITOS_COMPUESTOS' en root o en picks
        if ap.get("estrategia") == "FAVORITOS_COMPUESTOS":
            n_combos += len(ap.get("combos", ap.get("picks", [])))
            found = True
        elif isinstance(ap.get("picks"), list):
            for p in ap["picks"]:
                if p.get("estrategia") == "FAVORITOS_COMPUESTOS":
                    n_combos += 1
                    found = True
    return {"n_combos": n_combos} if found else None


_DUAL_BOOK_CACHE_PATH = REPORTS / "dual_book_cache.json"
_DUAL_BOOK_TTL_S = 600  # 10 min — reduce 429 Kambi (era 120s, demasiado agresivo)


def _build_p8_books(fecha: str) -> Dict:
    """
    P8 Multi-Book: best-price por pick via dual_book_client, cache TTL 120s.
    Nodo-111 X1 — nunca apuesta automática, solo routing informativo.
    """
    import time as _time

    # Cache check
    cache = _load_json(_DUAL_BOOK_CACHE_PATH)
    if cache and cache.get("ts"):
        try:
            cached_ts = datetime.fromisoformat(cache["ts"])
            age_s = (datetime.now() - cached_ts).total_seconds()
            if age_s < _DUAL_BOOK_TTL_S:
                cache["cache_age_s"] = int(age_s)
                cache["from_cache"] = True
                return cache
        except Exception:
            pass

    # Fresh build — UNA llamada de red por ciclo
    try:
        from scraping.dual_book_client import best_price as _best_price, _norm, es_arb as _es_arb
    except ImportError:
        return {"picks": {}, "feeds": [], "error": "dual_book_client no disponible", "cache_age_s": 0}

    # Book 1+2: odds_aggregator multi-casa (betplay Kambi REST + wplay SSR VERIFIED)
    # Nodo-116 D116-02. fetch_all_odds → {player: {book: entry}} — invertir para best_price()
    _fetch_all_odds = None
    try:
        import sys as _sys
        _scripts_dir = str(BASE_DIR / "scripts")
        if _scripts_dir not in _sys.path:
            _sys.path.insert(0, _scripts_dir)
        from odds_aggregator import fetch_all_odds as _fetch_all_odds
    except ImportError:
        pass

    feeds: Dict[str, Any] = {}
    if _fetch_all_odds:
        try:
            _all_data = _fetch_all_odds(["betplay", "rushbet", "wplay"])
            for _player_key, _books in _all_data.items():
                for _book, _entry in _books.items():
                    if _entry and _entry.get("odds"):
                        # Re-normalizar con dual_book _norm para consistencia con best_price()
                        _dk = _norm(_entry.get("jugador", _player_key))
                        feeds.setdefault(_book, {})[_dk] = _entry
        except Exception:
            pass

    # D128-01: alias nombre-completo → apellido para matchear picks abreviados del edge_report/games
    # Betplay/Wplay devuelven "Botic Van De Zandschulp"; games_report tiene "Van De Zandschulp B."
    # best_price() usa exact-match por _norm(), así que indexamos también por apellido en todos los feeds.
    for _feed_book, _feed_dict in feeds.items():
        _book_aliases: Dict[str, Any] = {}
        for _fk, _fe in _feed_dict.items():
            _fparts = _fk.split()
            if len(_fparts) >= 2:
                _sn = " ".join(_fparts[1:])   # drop primer token (nombre de pila)
                if _sn not in _feed_dict and _sn not in _book_aliases:
                    _book_aliases[_sn] = _fe
        _feed_dict.update(_book_aliases)

    # Fallback betplay: si Kambi devuelve 429/vacío, usar h2h_results_enhanced del pipeline
    _h2h_path = _latest(str(REPORTS / f"h2h_results_enhanced_{fecha.replace('-','')}*.json"))
    _h2h_rows: list = []
    if _h2h_path:
        _h2h_data = _load_json(_h2h_path)
        if isinstance(_h2h_data, list):
            _h2h_rows = _h2h_data
        elif isinstance(_h2h_data, dict):
            _h2h_rows = _h2h_data.get("partidos", _h2h_data.get("matches", []))
    if not feeds.get("betplay") and _h2h_rows:
        _bp_fb: Dict[str, Any] = {}
        for _p in _h2h_rows:
            if _p.get("jugador1") and _p.get("cuota1"):
                _bp_fb[_norm(_p["jugador1"])] = {"odds": _p["cuota1"]}
            if _p.get("jugador2") and _p.get("cuota2"):
                _bp_fb[_norm(_p["jugador2"])] = {"odds": _p["cuota2"]}
        if _bp_fb:
            feeds["betplay"] = _bp_fb

    # Read edge_report for picks to route
    er = _latest(str(REPORTS / f"edge_report_{fecha.replace('-','')}*.json"))
    er_data = _load_json(er) or {}
    all_picks_er = (er_data.get("apostar") or []) + (er_data.get("watchlist") or [])

    # D128-02: incluir jugadores de combos GAMES del día en P8
    # Busca players ATP/WTA de games_signal_report que sí están en los feeds (betplay/wplay)
    # Útil cuando edge_report solo tiene ITF (no en books) pero games tiene ATP
    _all_gr = sorted(glob.glob(str(REPORTS / f"games_signal_report_{fecha.replace('-','')}*.json")))
    _games_players_seen: set = set()
    _games_seen_oids: set = set()
    for _gr_path in _all_gr:
        _gr_data = _load_json(Path(_gr_path)) or {}
        for _gs in (_gr_data.get("apostar") or []):
            for _sig in _gs.get("señales_optimas", []):
                _oid = _sig.get("outcome_id")
                if _oid and _oid in _games_seen_oids:
                    continue
                if _oid:
                    _games_seen_oids.add(_oid)
                _partido = _gs.get("partido", "")
                if " vs " not in _partido:
                    continue
                _p1, _p2 = _partido.split(" vs ", 1)
                for _gp_raw in [_p1.strip(), _p2.strip()]:
                    if not _gp_raw or _gp_raw in _games_players_seen:
                        continue
                    _games_players_seen.add(_gp_raw)
                    # Limpiar inicial trailing "B." → "Van De Zandschulp B." → "Van De Zandschulp"
                    # para que _norm() matchee el alias de apellido en feeds (D128-01)
                    _gp_parts = _gp_raw.split()
                    while _gp_parts and len(_gp_parts[-1].rstrip(".")) <= 1:
                        _gp_parts.pop()
                    _gp = " ".join(_gp_parts) if _gp_parts else _gp_raw
                    all_picks_er.append({
                        "favorito_predicho": _gp,
                        "cuota_favorito": 0,
                        "_source": "games",
                        "_partido": _gs.get("partido", ""),
                        "_mercado": f"{_sig.get('direccion','?')} {_sig.get('linea','?')} @{_sig.get('cuota','?')}",
                    })

    picks_result: Dict[str, Any] = {}
    for p in all_picks_er:
        jug = p.get("favorito_predicho", p.get("favorito", ""))
        if not jug:
            continue
        bp = _best_price(jug, feeds)
        if not bp:
            continue
        base_cuota = float(p.get("cuota_favorito") or 0)
        gain = round((bp["cuota"] / base_cuota - 1) * 100, 2) if base_cuota > 0 else 0.0

        # divergencia entre los feeds disponibles
        cuotas_vals = [v for v in bp.get("cuotas", {}).values() if isinstance(v, (int, float))]
        div_pct = 0.0
        if len(cuotas_vals) >= 2:
            lo, hi = min(cuotas_vals), max(cuotas_vals)
            div_pct = round((hi / lo - 1) * 100, 2) if lo > 0 else 0.0

        cuotas_map = bp.get("cuotas", {})
        _src     = p.get("_source", "ml")
        _partido = p.get("_partido", "")
        _mercado = p.get("_mercado", "")
        # Clave única: partido+jugador para GAMES (evita colisión "Smith" entre 5 partidos)
        _key = f"{_partido}_{jug}".lower() if _src == "games" and _partido else jug.lower()
        picks_result[_key] = {
            "jugador":           jug,
            "casa":              bp.get("casa", ""),
            "cuota":             bp.get("cuota", 0),
            "cuotas":            cuotas_map,
            "gain_pct":          gain,
            "divergencia_pct":   div_pct,
            "betplay_cuota":     cuotas_map.get("betplay", "—"),
            "rushbet_cuota":     cuotas_map.get("rushbet", "—"),
            "wplay_cuota":       cuotas_map.get("wplay", "—"),
            "cuota_plan":        base_cuota,
            "_source":           _src,
            "_partido":          _partido,
            "_mercado":          _mercado,
        }

    # ARB detection: mejor cuota fav (book A) vs mejor cuota rival (book B)
    for p in all_picks_er:
        jug = p.get("favorito_predicho", p.get("favorito", ""))
        rival = p.get("rival", "")
        if not jug or not rival:
            continue
        jug_key = jug.lower()
        if jug_key not in picks_result:
            continue
        rival_bp = _best_price(rival, feeds)
        if rival_bp:
            fav_cuota = picks_result[jug_key].get("cuota", 0)
            arb = _es_arb(fav_cuota, rival_bp["cuota"]) if fav_cuota else False
            picks_result[jug_key]["arb_flag"] = arb
            picks_result[jug_key]["rival_cuota"] = rival_bp["cuota"]
            picks_result[jug_key]["rival_casa"] = rival_bp.get("casa", "")

    result: Dict[str, Any] = {
        "ts":         datetime.now().isoformat(),
        "picks":      picks_result,
        "feeds":      list(feeds.keys()),
        "n_picks":    len(picks_result),
        "cache_age_s": 0,
        "from_cache": False,
        "stale":      False,
    }

    # Persist cache (TTL 120s)
    try:
        _DUAL_BOOK_CACHE_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception:
        pass

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SERVIDOR HTTP
# ══════════════════════════════════════════════════════════════════════════════

_FECHA_OVERRIDE: Optional[str] = None


def _demo_state() -> Dict[str, Any]:
    """Estado sintético realista para demostración a Fable — sin datos reales requeridos."""
    hoy = date.today().isoformat()
    return {
        "fecha": hoy,
        "ts": datetime.now().isoformat(),
        "p0_ncal": {"djokovic": 54, "alcaraz": 12},
        "p9_que_falta": [
            {"jugador": "Holger Rune", "condicion": "favorito_claro",
             "detalle": "p_modelo=0.551 < 0.62 (faltan 0.069) | cuota_fav=1.78 > 1.40 o conf=LOW"},
            {"jugador": "Andrey Rublev", "condicion": "cuota_rango",
             "detalle": "cuota_fav=2.35 > 2.10 (techo, delta +0.25)"},
            {"jugador": "Grigor Dimitrov", "condicion": "model_neq_bookie",
             "detalle": "cuota_fav=1.90 >= cuota_rival=1.85 (bookie discrepa)"},
        ],
        "p4_risk": {
            "governor_code": 0,
            "kgr_sesion": 1.05,
            "bankroll": 125000,
            "stake_total": 7500,
            "kill_switches": {"MOTOR_DEFENSIVE": True, "GCS_GATE_ENABLED": True},
            "exposicion": [
                {"jugador": "Alcaraz", "stake": 5000, "pct": 0.04},
                {"jugador": "Djokovic", "stake": 2500, "pct": 0.02},
            ],
        },
        "p1_tape": {
            "entries": [
                {"jugador": "Alcaraz", "drift_pct": -18.5, "direction": "CONFIRMA", "vel_zscore": 2.3, "ts": datetime.now().isoformat()},
                {"jugador": "Djokovic", "drift_pct": -9.2, "direction": "CONFIRMA", "vel_zscore": 1.1, "ts": datetime.now().isoformat()},
                {"jugador": "Zverev", "drift_pct": 4.1, "direction": "ALEJA", "vel_zscore": -0.5, "ts": datetime.now().isoformat()},
            ]
        },
        "p2_break": {
            "breaks": [{
                "estado": "BREAK_CONFIRMADO",
                "jugador": "Alcaraz",
                "pick": "Alcaraz",
                "drift_pct": -18.5,
                "hipotesis": "H100-01",
                "n_actual": 3,
                "p_modelo": 0.55,
                "cuota_pre": 2.33,
                "cuota_live": 1.54,
                "edge_live": -0.11,
                "trigger": False,
                "senales": ["drift≥15%"],
            }]
        },
        "p3_convergence": {
            "picks": [
                {
                    "jugador": "Djokovic",
                    "score_directo": 4,
                    "confidence_flag": "STRONG",
                    "markov_favorito": "HOT",
                    "rival_value_flag": False,
                    "rival": "Medvedev",
                    "gcs_active": True,
                    "señales_activas": ["STRONG", "HOT", "ELO_DOM"],
                    "n_h2h": 12,
                    "clv": 3.1,
                },
                {
                    "jugador": "Alcaraz",
                    "score_directo": 3,
                    "confidence_flag": "STRONG",
                    "markov_favorito": "HOT",
                    "rival_value_flag": False,
                    "rival": "Zverev",
                    "gcs_active": False,
                    "señales_activas": ["STRONG", "HOT"],
                    "n_h2h": 5,
                    "clv": 1.8,
                },
            ]
        },
        "p5_execution": {
            "picks": [
                {"jugador": "Alcaraz", "cuota_trigger": 2.10, "cuota_actual": 1.95, "clv_pct": -7.1},
                {"jugador": "Djokovic", "cuota_trigger": 1.80, "cuota_actual": 1.75, "clv_pct": -2.8},
            ],
            "clv_median": -4.9,
        },
        "p6_pnl": {
            "segmentos": [
                {"segmento": "GCS (H60-01)", "n": 54, "hit_pct": 64.8, "roi_pct": 31.2, "stake_total": 18000},
                {"segmento": "BREAK H100-01", "n": 3, "hit_pct": 66.7, "roi_pct": 12.5, "stake_total": 7500},
                {"segmento": "FAVORITOS", "n": 8, "hit_pct": 100.0, "roi_pct": 220.0, "stake_total": 27500},
            ]
        },
        "p7_clock": {
            "partidos": [
                {"jugador": "Alcaraz vs Zverev", "inicio": "18:00", "ventana_live": "-30min a +45min desde 18:00", "snapshot_deadline": "~15min antes de 18:00"},
                {"jugador": "Djokovic vs Medvedev", "inicio": "20:30", "ventana_live": "-30min a +45min desde 20:30", "snapshot_deadline": "~15min antes de 20:30"},
            ]
        },
        "p8_books": {
            "picks": {
                "alcaraz": {"jugador": "Alcaraz", "casa": "wplay", "cuota": 2.15, "gain_pct": 2.4, "divergencia_pct": 11.2,
                            "betplay_cuota": 2.10, "rushbet_cuota": 2.08, "wplay_cuota": 2.15, "cuota_plan": 2.10},
                "djokovic": {"jugador": "Djokovic", "casa": "wplay", "cuota": 1.85, "gain_pct": 5.7, "divergencia_pct": 18.5,
                             "betplay_cuota": 1.57, "rushbet_cuota": 1.55, "wplay_cuota": 1.85, "cuota_plan": 1.80},
            },
            "feeds": ["betplay", "rushbet", "wplay"],
            "cache_age_s": 45,
        },
        "p10_odds_history": {
            "Alcaraz vs Zverev": {
                "readings": [
                    {"ts": "16:00:00", "cuota": 2.20, "drift": 0.04},
                    {"ts": "16:05:00", "cuota": 2.10, "drift": 0.09},
                    {"ts": "16:10:00", "cuota": 1.98, "drift": 0.14},
                    {"ts": "16:15:00", "cuota": 1.85, "drift": 0.19},
                ],
                "estado": "BREAK_CONFIRMADO", "fired": False,
            },
            "Djokovic vs Medvedev": {
                "readings": [
                    {"ts": "16:00:00", "cuota": 1.90, "drift": 0.03},
                    {"ts": "16:05:00", "cuota": 1.85, "drift": 0.06},
                    {"ts": "16:10:00", "cuota": 1.80, "drift": 0.08},
                    {"ts": "16:15:00", "cuota": 1.80, "drift": 0.08},
                ],
                "estado": "BREAK_POSIBLE", "fired": False,
            },
        },
        "p_games": {
            "disponible": True, "fecha": date.today().isoformat(),
            "n_partidos": 18, "n_apostar": 2,
            "fuente": "games_signal_report_demo.json",
            "signals": [
                {"partido": "Alcaraz vs Zverev", "mercado": "Total de juegos",
                 "direccion": "OVER", "linea": 20.5, "cuota": 1.88,
                 "gap": 5.5, "confianza": "ALTA", "games_range": "26-32+"},
                {"partido": "Swiatek vs Gauff", "mercado": "Total de sets",
                 "direccion": "OVER", "linea": 2.5, "cuota": 2.10,
                 "gap": None, "confianza": "MEDIA", "games_range": "20-24"},
            ],
        },
        "p11_combo_live": [],
        "p12_conformal": _build_conformal_band(),
    }


_DEMO_MODE: bool = False

# ══════════════════════════════════════════════════════════════════════════════
# D129-01 — Cache en memoria TTL 20s + thread background
# D129-02 — POST /api/refresh para invalidación explícita (n8n push)
# D129-03 — _data_freshness() mtime real de archivos de datos
# ══════════════════════════════════════════════════════════════════════════════

_STATE_CACHE: Dict[str, Any] = {
    "state": None,
    "ts": None,
    "ttl_s": 20,
    "lock": threading.Lock(),
}


def _get_cached_state(fecha: str) -> dict:
    """Retorna estado desde cache si tiene <20s. Cache miss → reconstruye y guarda."""
    with _STATE_CACHE["lock"]:
        now = datetime.now()
        age = (now - _STATE_CACHE["ts"]).total_seconds() if _STATE_CACHE["ts"] else 999
        if _STATE_CACHE["state"] is not None and age < _STATE_CACHE["ttl_s"]:
            return _STATE_CACHE["state"]
    state = build_desk_state(fecha)
    with _STATE_CACHE["lock"]:
        _STATE_CACHE["state"] = state
        _STATE_CACHE["ts"] = datetime.now()
    return state


# ─── D133: Games Live Convergencia ───────────────────────────────────────────

_KAMBI_BASE    = "https://us.offering-api.kambicdn.com/offering/v2018/betplay"
_KAMBI_PARAMS  = "lang=es_CO&market=CO&client_id=2&channel_id=1"
_KAMBI_HDR     = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer":    "https://betplay.com.co/",
    "Accept":     "application/json",
}


def _apellido_games(nombre: str) -> str:
    """Último token no-inicial de un nombre: 'Alcaraz C.' → 'alcaraz'."""
    tokens = [t for t in nombre.split() if len(t) > 2 or not t.rstrip(".").isupper()]
    return tokens[-1].lower().rstrip(".") if tokens else nombre.lower()


def _kambi_started_events() -> list:
    """1 HTTP call → todos los eventos STARTED de tenis en Kambi ahora."""
    # 1. liveEvents.json — eventos dedicados en curso
    try:
        url = f"{_KAMBI_BASE}/liveEvents.json?{_KAMBI_PARAMS}&sport=tennis"
        req = urllib.request.Request(url, headers=_KAMBI_HDR)
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
        events = data.get("liveEvents") or data.get("events") or []
        if events:
            return events
    except Exception:
        pass
    # 2. Fallback: listView filtrado por state=STARTED
    try:
        url = f"{_KAMBI_BASE}/listView/tennis.json?{_KAMBI_PARAMS}"
        req = urllib.request.Request(url, headers=_KAMBI_HDR)
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
        return [e for e in (data.get("events") or [])
                if e.get("event", {}).get("state") == "STARTED"]
    except Exception:
        return []


def _extract_games_cuota_live(event_id: int, direccion: str, linea: Optional[float]) -> Optional[float]:
    """
    D135-01: busca mercado match-level 'Total de juegos' via endpoint betoffer/event/{id}.
    El listView solo devuelve mercados destacados (vacío para ITF). Este endpoint retorna
    todos los betOffers del evento.
    D135-02: excluye mercados set-level ("Total de juegos - Set 3") y juego-level.
    """
    url = (f"{_KAMBI_BASE}/betoffer/event/{event_id}.json"
           f"?{_KAMBI_PARAMS}")
    try:
        req = urllib.request.Request(url, headers=_KAMBI_HDR)
        with urllib.request.urlopen(req, timeout=3) as r:
            offers = json.loads(r.read().decode()).get("betOffers", [])
    except Exception:
        return None

    dir_norm = direccion.upper()
    for bo in offers:
        label = bo.get("criterion", {}).get("label") or ""
        # D135-02: solo mercado match-level, excluir "Total de juegos - Set X"
        if not ("Total de juegos" in label
                and " - Set " not in label
                and "Juego" not in label):
            continue
        for oc in bo.get("outcomes", []):
            oc_label = (oc.get("label") or "").lower()
            oc_line  = oc.get("line", 0) / 1000 if oc.get("line") else None
            is_under = "menos" in oc_label or "under" in oc_label
            is_over  = "más" in oc_label or "over" in oc_label or "mas" in oc_label
            dir_match  = (dir_norm == "UNDER" and is_under) or (dir_norm == "OVER" and is_over)
            line_match = (oc_line is None or linea is None or abs(oc_line - linea) < 1.0)
            if dir_match and line_match:
                odds_raw = oc.get("odds")
                if odds_raw:
                    return round(odds_raw / 1000, 2)
    return None


def _check_games_convergencia(fecha: str) -> None:
    """
    D133-03: clasifica señales ALTA de games_signal_report como EN_VIVO/PRE/TERMINADO.
    Escribe reports/games_live_YYYYMMDD.json.
    Si ≥2 ALTA EN_VIVO → anti-flood → subprocess.Popen(--games) fire-and-forget.
    """
    fecha_compact = fecha.replace("-", "")
    gsr_path = _latest(str(REPORTS / f"games_signal_report_{fecha_compact}*.json"))
    if not gsr_path:
        return

    try:
        data = json.loads(Path(gsr_path).read_text(encoding="utf-8"))
    except Exception:
        return

    # Recoger señales ALTA
    alta_signals: List[Dict] = []
    for p in data.get("apostar", []):
        for s in p.get("señales_optimas", []):
            if s.get("confianza_señal") == "ALTA" and s.get("apostar"):
                alta_signals.append({
                    "partido":    p.get("partido", ""),
                    "direccion":  s.get("direccion", ""),
                    "linea":      s.get("linea"),
                    "cuota_pre":  s.get("cuota"),
                    "hora":       p.get("hora"),
                    "event_id":   p.get("kambi_event_id"),
                    "estado":     "PRE_PARTIDO",
                    "cuota_live": None,
                    "drift_pct":  None,
                })

    if not alta_signals:
        return

    # Obtener eventos STARTED de Kambi (1 HTTP call)
    started_events = _kambi_started_events()

    # Índices por event_id y apellido
    started_by_id:      Dict[int, dict] = {}
    started_by_apellido: Dict[str, dict] = {}
    for ev_wr in started_events:
        ev  = ev_wr.get("event", {}) if isinstance(ev_wr, dict) else {}
        eid = ev.get("id")
        if eid:
            started_by_id[int(eid)] = ev_wr
        for field in ("homeName", "awayName"):
            nombre = ev.get(field, "")
            if nombre:
                started_by_apellido[_apellido_games(nombre)] = ev_wr

    now_utc = datetime.utcnow()

    # Clasificar cada señal
    for sig in alta_signals:
        partido  = sig["partido"]
        eid      = sig.get("event_id")
        matched  = None

        # D133-02: lookup primario por event_id
        if eid:
            matched = started_by_id.get(int(eid))

        # Fallback: apellido de cada jugador
        if not matched:
            partes = [p.strip() for p in partido.replace(" vs. ", " vs ").split(" vs ")]
            for parte in partes:
                ap = _apellido_games(parte)
                if ap in started_by_apellido:
                    matched = started_by_apellido[ap]
                    break

        if matched:
            sig["estado"] = "EN_VIVO"
            cuota_live = _extract_games_cuota_live(matched["event"]["id"], sig["direccion"], sig["linea"])  # D135-01
            if cuota_live and sig.get("cuota_pre"):
                sig["cuota_live"] = cuota_live
                sig["drift_pct"]  = round(
                    (cuota_live - sig["cuota_pre"]) / sig["cuota_pre"] * 100, 1
                )
        else:
            # Clasificar por hora como fallback temporal
            hora_raw = sig.get("hora") or ""
            try:
                if "T" in str(hora_raw):
                    hora_dt = datetime.fromisoformat(
                        str(hora_raw).replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                else:
                    hm = str(hora_raw).split(":")
                    hora_dt = now_utc.replace(
                        hour=int(hm[0]), minute=int(hm[1]), second=0, microsecond=0
                    )
                diff_min = (now_utc - hora_dt).total_seconds() / 60
                sig["estado"] = "TERMINADO" if diff_min > 130 else "PRE_PARTIDO"
            except Exception:
                sig["estado"] = "PRE_PARTIDO"

    en_vivo_count       = sum(1 for s in alta_signals if s["estado"] == "EN_VIVO")
    convergencia_activa = en_vivo_count >= 2

    # Escribir games_live_YYYYMMDD.json (D133-05)
    gl_path = REPORTS / f"games_live_{fecha_compact}.json"
    try:
        gl_path.write_text(
            json.dumps({
                "ts":                 datetime.now().isoformat()[:19],
                "signals_alta":       alta_signals,
                "en_vivo_count":      en_vivo_count,
                "convergencia_activa": convergencia_activa,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    # D133-04: anti-flood + Popen fire-and-forget
    if not convergencia_activa:
        return

    fired_path = REPORTS / f"games_live_{fecha_compact}_fired.json"
    try:
        fired: List[List] = json.loads(fired_path.read_text(encoding="utf-8")) if fired_path.exists() else []
    except Exception:
        fired = []

    if len(fired) >= 10:
        return  # cap diario

    combo_key = sorted(s["partido"] for s in alta_signals if s["estado"] == "EN_VIVO")
    if combo_key in fired:
        return  # ya disparado

    try:
        subprocess.Popen(
            [sys.executable, str(BASE_DIR / "betplay_combo_builder.py"), "--games", "--live"],
            cwd=str(BASE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        fired.append(combo_key)
        fired_path.write_text(json.dumps(fired, ensure_ascii=False), encoding="utf-8")
        logger.info(f"[D133] CONVERGENCIA GAMES: {en_vivo_count} ALTA EN_VIVO → combo disparado")
    except Exception as exc:
        logger.warning(f"[D133] Popen error: {exc}")


def _background_refresh(fecha_fn) -> None:
    """Thread daemon — precalienta cache cada 15s para que el browser reciba <1s."""
    while True:
        try:
            fecha = fecha_fn()
            _get_cached_state(fecha)
            _check_games_convergencia(fecha)   # D133-03: games live convergencia
        except Exception:
            pass
        time.sleep(15)


def _data_freshness(fecha: str) -> str:
    """Retorna antigüedad real del archivo de datos más reciente del día."""
    fecha_compact = fecha.replace("-", "")
    candidates = (
        glob.glob(str(REPORTS / f"live_odds_history_{fecha_compact}*.json"))
        + glob.glob(str(REPORTS / f"edge_report_{fecha_compact}*.json"))
    )
    mtimes = []
    for c in candidates:
        try:
            mtimes.append(os.path.getmtime(c))
        except OSError:
            pass
    if not mtimes:
        return "datos: desconocido"
    age_s = time.time() - max(mtimes)
    return f"datos de hace {int(age_s // 60)}m {int(age_s % 60)}s"


class DeskHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silenciar logs HTTP por defecto

    def do_GET(self):
        if self.path not in ("/", "/desk.html", "/favicon.ico"):
            self.send_response(404)
            self.end_headers()
            return

        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        try:
            if _DEMO_MODE:
                state = _demo_state()
            else:
                fecha = _FECHA_OVERRIDE or date.today().isoformat()
                state = _get_cached_state(fecha)   # D129-01: cache en memoria
            html = render_html(state)
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            err = f"<pre>Error: {e}</pre>".encode()
            self.send_response(500)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(err)

    def do_POST(self):
        """D129-02: POST /api/refresh — invalida cache inmediatamente (llamado por n8n via close_snapshot_server)."""
        if self.path != "/api/refresh":
            self.send_response(404)
            self.end_headers()
            return
        with _STATE_CACHE["lock"]:
            _STATE_CACHE["ts"] = None  # fuerza reconstrucción en próxima request
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    global _FECHA_OVERRIDE, _DEMO_MODE
    parser = argparse.ArgumentParser(description="Live Trading Desk — Nodo-109 :7780")
    parser.add_argument("--port", type=int, default=PORT_DEFAULT)
    parser.add_argument("--fecha", help="Fecha YYYY-MM-DD (default: hoy)")
    parser.add_argument("--once", action="store_true", help="Imprimir HTML y salir (sin servidor)")
    parser.add_argument("--demo", action="store_true", help="Modo demo con datos sintéticos (Fable review)")
    args = parser.parse_args()

    if args.fecha:
        _FECHA_OVERRIDE = args.fecha

    if args.demo:
        _DEMO_MODE = True

    if args.once:
        fecha = args.fecha or date.today().isoformat()
        state = _demo_state() if args.demo else build_desk_state(fecha)
        print(render_html(state))
        return

    # D129-01: thread daemon precalienta cache cada 15s
    _fecha_fn = lambda: _FECHA_OVERRIDE or date.today().isoformat()
    _t = threading.Thread(target=_background_refresh, args=(_fecha_fn,), daemon=True)
    _t.start()
    logger.info("Cache background thread iniciado (TTL 20s, refresh 15s)")

    server = HTTPServer(("0.0.0.0", args.port), DeskHandler)
    logger.info(f"Live Trading Desk en http://localhost:{args.port}/")
    logger.info("Auto-refresh JS: 12s | POST /api/refresh para push desde n8n | Ctrl-C para detener")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Detenido.")


if __name__ == "__main__":
    main()
