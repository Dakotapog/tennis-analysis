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

    state: Dict[str, Any] = {
        "fecha": fecha,
        "ts": datetime.now().isoformat(),
        "p4_risk": _build_p4_risk(fecha),   # P4 primero — manda
        "p1_tape": _build_p1_tape(fecha),
        "p2_break": _build_p2_break(fecha),
        "p3_convergence": _build_p3_convergence(fecha),
        "p5_execution": _build_p5_execution(fecha),
        "p6_pnl": _build_p6_pnl(fecha),
        "p7_clock": _build_p7_clock(fecha),
        "p8_books": _build_p8_books(fecha),  # Nodo-114 §3: dual-book cache 120s
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
            accionables.append({
                "tipo": "BREAK_CONFIRMADO",
                "jugador": brk.get("jugador", ""),
                "pick": brk.get("pick", ""),
                "hipotesis": "H100-01",
                "n_actual": brk.get("n_actual", 0),
                "n_stop": 20,
                "color": "amber",  # pre-graduacion: amber siempre
                "governor_code": gov_code,
                "drift_pct": brk.get("drift_pct", 0),
                "meta_score": p3d.get("score_directo", 0),
                "señales_activas": p3d.get("señales_activas", []),
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
        suf = f"(drift {drift:+.1f}%)" if drift is not None else ""
        parts.append(f"BREAK_CONFIRMADO{suf}")
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

    def row(cols, header=False):
        color = GREY if header else WHITE
        weight = "bold" if header else "normal"
        cells = "".join(f'<td style="padding:4px 10px;color:{color};font-weight:{weight};">{c}</td>' for c in cols)
        return f"<tr>{cells}</tr>"

    def table(headers, rows_data, empty_msg="Sin datos"):
        if not rows_data:
            return f'<p style="color:{GREY};font-size:0.85em;">{empty_msg}</p>'
        hdr = row(headers, header=True)
        body = "".join(row(r) for r in rows_data)
        return f'<table style="border-collapse:collapse;width:100%;font-size:0.85em;"><tr>{hdr}</tr>{body}</table>'

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
        conv_rows.append([
            p.get("jugador", ""),
            f'<span style="color:{sc_color};font-weight:bold;">{score}</span>',
            p.get("confidence_flag", ""),
            p.get("markov_favorito", ""),
            badges or "—",
        ])
    p3_panel = panel(
        "P3 CONVERGENCE — Meta-señal H98-01 (score>=3 = fila destacada)",
        f'<div style="{atenuado}">' + table(
            ["Jugador", "Score", "Conf", "Markov", "Flags"],
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
        p8_rows.append([
            bp.get("jugador", jug_key),
            str(bp.get("betplay_cuota", "—")),
            str(bp.get("flashscore_cuota", "—")),
            f'<span style="color:{div_color};">{div:.1f}%{div_badge}</span>',
            f'<b style="color:{GREEN};">{casa_gana} @{cuota_gana}</b> '
            f'<span style="color:{gain_color};">({gain_str})</span>',
        ])
    cache_age = p8_books.get("cache_age_s", 0)
    from_cache = p8_books.get("from_cache", False)
    feeds_str = ", ".join(p8_books.get("feeds", []) or [])
    cache_note = f"cache {cache_age}s restantes" if from_cache else "datos frescos"
    p8_badge = f"{len(p8_rows)} picks" if p8_rows else "SIN DATOS"
    p8_badge_color = BLUE if p8_rows else GREY
    p8_panel = panel(
        f"P8 MULTI-BOOK — Router X1 Nodo-111 | feeds: {feeds_str or 'ninguno'} | {cache_note} (TTL 120s)",
        f'<div style="{atenuado}">' + table(
            ["Jugador", "betplay", "flashscore", "div%", "Mejor precio"],
            p8_rows,
            "Sin datos (dual_book_cache.json vacío — se genera en próximo ciclo de live_edge_monitor)"
        ) + "</div>",
        p8_badge, p8_badge_color,
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

    # ── Accionables ───────────────────────────────────────────────────────────
    if accionables:
        acc_rows = []
        for a in accionables:
            ac = GREEN if a.get("color") == "green" else AMBER
            razon = linea_razonamiento(a)
            razon_short = razon[:90] + ("…" if len(razon) > 90 else "")
            acc_rows.append([
                f'<span style="color:{ac};font-weight:bold;">{a["tipo"]}</span>',
                a["jugador"],
                a["hipotesis"],
                f'{a["n_actual"]}/{a["n_stop"]}',
                f'<span title="{razon}" style="color:{GREY};font-size:0.82em;">{razon_short}</span>',
            ])
        acc_content = table(["Tipo", "Jugador/Pick", "Gate", "n/n_stop", "Razonamiento"], acc_rows)
    else:
        acc_content = f'<p style="color:{GREY};">{"DESK EN HALT — nada accionable" if halt else "Sin señales accionables ahora"}</p>'

    n_real = sum(1 for a in accionables if a.get("tipo") not in ("FAVORITOS_ZERO",))
    acc_panel = panel(
        "ACCIONABLE AHORA (P2∩P4∩estrategia graduada)",
        acc_content,
        f"{n_real} señales",
        GREEN if n_real and not halt else GREY,
    )

    fecha = state.get("fecha", "")
    ts = state.get("ts", "")
    refresh_note = f'<p style="color:{GREY};font-size:0.75em;text-align:right;">Auto-refresh 30s | {ts}</p>'

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="30">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Live Trading Desk — {fecha}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: {BG}; color: {WHITE}; font-family: 'Consolas','Monaco',monospace; padding: 18px; }}
    h1 {{ color: {BLUE}; font-size: 1.1em; margin-bottom: 14px; letter-spacing: 1px; }}
    td {{ border-bottom: 1px solid {BORDER}; }}
    tr:last-child td {{ border-bottom: none; }}
    a {{ color: {BLUE}; }}
  </style>
</head>
<body>
  <h1>LIVE TRADING DESK &nbsp;|&nbsp; {fecha} &nbsp;|&nbsp; Nodo-109</h1>
  {halt_banner}
  {acc_panel}
  {p4_panel}
  {p2_panel}
  {p3_panel}
  {p5_panel}
  {p6_panel}
  {p8_panel}
  {p1_panel}
  {p7_panel}
  {refresh_note}
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
    le = _latest(str(REPORTS / f"live_edge_state_{fecha.replace('-','')}*.json"))
    data = _load_json(le)
    if not data:
        return {"entries": [], "source": "SIN DATO — live_edge_monitor no activo"}

    entries = []
    for pick in data.get("picks", data.get("partidos", [])):
        drift = pick.get("drift_pct", pick.get("drift", 0)) or 0
        direction = "CONFIRMA" if drift < 0 else ("ALEJA" if drift > 0 else "NEUTRO")
        entries.append({
            "jugador": pick.get("jugador", pick.get("home", "")),
            "drift_pct": drift,
            "direction": direction,
            "vel_zscore": pick.get("velocity_zscore", pick.get("vel_z", "—")),
            "ts": pick.get("ts", pick.get("timestamp", "")),
        })
    return {"entries": entries}


def _build_p2_break(fecha: str) -> Dict:
    """P2: break_state de Nodo-100B."""
    le = _latest(str(REPORTS / f"live_edge_state_{fecha.replace('-','')}*.json"))
    data = _load_json(le)
    if not data:
        return {"breaks": [], "source": "SIN DATO — live_edge_monitor no activo"}

    breaks = []
    for pick in data.get("picks", data.get("partidos", [])):
        bs = pick.get("break_state", "NORMAL")
        if bs != "NORMAL":
            breaks.append({
                "estado": bs,
                "jugador": pick.get("jugador", ""),
                "pick": pick.get("pick", ""),
                "drift_pct": pick.get("drift_pct", 0),
                "hipotesis": "H100-01",
                "n_actual": pick.get("n_fired", 0),
            })
    return {"breaks": breaks}


def _build_p3_convergence(fecha: str) -> Dict:
    """P3: meta_signal_score + rival_value_flag + gcs_active."""
    er = _latest(str(REPORTS / f"edge_report_{fecha.replace('-','')}*.json"))
    data = _load_json(er)
    if not data:
        return {"picks": [], "source": "SIN DATO — correr PASO 3"}

    raw = data if isinstance(data, list) else data.get("picks", [])
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
            picks.append({
                "jugador": p.get("favorito", p.get("jugador", "")),
                "score_directo": score,
                "señales_activas": senas,
                "n_h2h": p.get("n_h2h"),
                "clv": p.get("clv"),
                "confidence_flag": p.get("confidence_flag", ""),
                "markov_favorito": p.get("markov_favorito", ""),
                "rival_value_flag": rv,
                "rival": p.get("rival", ""),
                "gcs_active": gcs,
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
    """P6: segmentos de shadow_book --report (graduadas arriba)."""
    # Intenta correr shadow_book --report y parsear salida
    segmentos = []
    try:
        res = subprocess.run(
            [sys.executable, str(BASE_DIR / "shadow_book.py"), "--report"],
            capture_output=True, text=True, timeout=30, cwd=str(BASE_DIR)
        )
        lines = res.stdout.splitlines()
        # Parsear líneas de segmento: "  LABEL: n=N  hit%=XX  IC95=...  ROI=XX%"
        import re
        for line in lines:
            m = re.search(r"([\w\s/+<>=\-\.]+):\s+n=(\d+)\s+hit%=([\d\.]+)\s+.*ROI=([\-\d\.]+)%", line)
            if m:
                nombre = m.group(1).strip()
                n = int(m.group(2))
                hit = m.group(3)
                roi = float(m.group(4))
                graduada = "GCS" in nombre or "H60-01" in nombre
                segmentos.append({"nombre": nombre, "n": n, "hit_pct": hit, "roi": roi, "graduada": graduada})
    except Exception:
        pass

    # Ordenar: graduadas primero
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
_DUAL_BOOK_TTL_S = 120  # UNA llamada de red por ciclo — respeta rate-limit 429


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
        from scraping.dual_book_client import fetch_kambi, best_price as _best_price, _norm
    except ImportError:
        return {"picks": {}, "feeds": [], "error": "dual_book_client no disponible", "cache_age_s": 0}

    feeds: Dict[str, Any] = {}
    try:
        feeds["betplay"] = fetch_kambi("betplay")
    except Exception:
        feeds["betplay"] = {}

    # Book 2: zita file (FlashScore/Playwright — Nodo-48)
    zita = _latest(str(BASE_DIR / f"data/zita_tennis_matches_{fecha.replace('-','')}*.json"))
    zita_data = _load_json(zita)
    if zita_data and isinstance(zita_data, dict):
        fs: Dict[str, Any] = {}
        for partidos in zita_data.values():
            if not isinstance(partidos, list):
                continue
            for m in partidos:
                if m.get("jugador1") and m.get("cuota1"):
                    fs[_norm(m["jugador1"])] = {"odds": m["cuota1"]}
                if m.get("jugador2") and m.get("cuota2"):
                    fs[_norm(m["jugador2"])] = {"odds": m["cuota2"]}
        if fs:
            feeds["flashscore"] = fs

    # Read edge_report for picks to route
    er = _latest(str(REPORTS / f"edge_report_{fecha.replace('-','')}*.json"))
    er_data = _load_json(er) or {}
    all_picks_er = (er_data.get("apostar") or []) + (er_data.get("watchlist") or [])

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
        picks_result[jug.lower()] = {
            "jugador":           jug,
            "casa":              bp.get("casa", ""),
            "cuota":             bp.get("cuota", 0),
            "cuotas":            cuotas_map,
            "gain_pct":          gain,
            "divergencia_pct":   div_pct,
            "betplay_cuota":     cuotas_map.get("betplay", "—"),
            "flashscore_cuota":  cuotas_map.get("flashscore", "—"),
            "cuota_plan":        base_cuota,
        }

    result: Dict[str, Any] = {
        "ts":         datetime.now().isoformat(),
        "picks":      picks_result,
        "feeds":      list(feeds.keys()),
        "n_picks":    len(picks_result),
        "cache_age_s": 0,
        "from_cache": False,
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
            fecha = _FECHA_OVERRIDE or date.today().isoformat()
            state = build_desk_state(fecha)
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


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    global _FECHA_OVERRIDE
    parser = argparse.ArgumentParser(description="Live Trading Desk — Nodo-109 :7780")
    parser.add_argument("--port", type=int, default=PORT_DEFAULT)
    parser.add_argument("--fecha", help="Fecha YYYY-MM-DD (default: hoy)")
    parser.add_argument("--once", action="store_true", help="Imprimir HTML y salir (sin servidor)")
    args = parser.parse_args()

    if args.fecha:
        _FECHA_OVERRIDE = args.fecha

    if args.once:
        fecha = args.fecha or date.today().isoformat()
        state = build_desk_state(fecha)
        print(render_html(state))
        return

    server = HTTPServer(("0.0.0.0", args.port), DeskHandler)
    logger.info(f"Live Trading Desk en http://localhost:{args.port}/")
    logger.info("Auto-refresh: 30s | Ctrl-C para detener")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Detenido.")


if __name__ == "__main__":
    main()
