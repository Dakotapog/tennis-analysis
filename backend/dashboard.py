"""
dashboard.py — Nodo-58/59: Dashboard de Observabilidad
McLaren Electric Dark Theme — Paneles 1-7

    streamlit run dashboard.py

READ-ONLY absoluto. Nunca escribe, nunca llama APIs externas.
Fuente única: shadow_book.report_dict() + loaders de JSONs locales.
"""

import json
import glob
import os
import sys
from collections import defaultdict
from datetime import datetime, date
from typing import Optional

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

# ── McLaren Electric Palette ─��
_ORANGE  = "#FF8000"
_BLUE    = "#00BFFF"
_TEAL    = "#00E5FF"
_GREEN   = "#00FF87"
_RED     = "#FF1E56"
_DARK    = "#0D0D0D"
_PANEL   = "#1A1A1A"
_CARD    = "#242424"
_TEXT    = "#F0F0F0"
_MUTED   = "#888888"


# ══════════════════════════════════════════════════════════════════════════════
# CSS INJECTION — McLaren Electric Dark
# ══════════════════════════════════════════════════════════════════════════════

def _inject_css() -> None:
    st.markdown(f"""
    <style>
    /* ── Base ── */
    .stApp, .main, section.main {{
        background-color: {_DARK};
        color: {_TEXT};
    }}
    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        background-color: #111111;
        border-right: 2px solid {_ORANGE};
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: {_ORANGE};
    }}
    /* ── Headers ── */
    h1 {{ color: {_ORANGE}; letter-spacing: 1px; font-weight: 800; }}
    h2 {{ color: {_ORANGE}; border-bottom: 2px solid {_ORANGE}44; padding-bottom: 4px; }}
    h3 {{ color: {_BLUE}; }}
    h4 {{ color: {_TEAL}; }}
    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {_PANEL};
        border-radius: 8px;
        gap: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {_MUTED};
        font-weight: 600;
        letter-spacing: 0.5px;
        border-radius: 6px;
        padding: 8px 20px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {_ORANGE}22;
        color: {_ORANGE} !important;
        border-bottom: 2px solid {_ORANGE};
    }}
    /* ── Metrics ── */
    [data-testid="stMetric"] {{
        background-color: {_CARD};
        border: 1px solid {_ORANGE}44;
        border-radius: 10px;
        padding: 12px 16px;
    }}
    [data-testid="stMetricLabel"] {{ color: {_MUTED}; font-size: 0.78rem; text-transform: uppercase; }}
    [data-testid="stMetricValue"] {{ color: {_ORANGE}; font-size: 1.8rem; font-weight: 700; }}
    [data-testid="stMetricDelta"] {{ color: {_GREEN}; }}
    /* ── Alerts ── */
    .stSuccess {{ background-color: {_GREEN}18 !important; border-left: 3px solid {_GREEN}; color: {_GREEN} !important; }}
    .stError   {{ background-color: {_RED}18   !important; border-left: 3px solid {_RED};   color: {_RED}   !important; }}
    .stWarning {{ background-color: {_ORANGE}18 !important; border-left: 3px solid {_ORANGE}; }}
    .stInfo    {{ background-color: {_BLUE}18  !important; border-left: 3px solid {_BLUE}; }}
    /* ── Progress bars ── */
    .stProgress > div > div > div {{ background-color: {_ORANGE}; }}
    /* ── Expander ── */
    details summary {{
        color: {_TEAL};
        font-weight: 600;
        cursor: pointer;
    }}
    details[open] summary {{ color: {_ORANGE}; }}
    details {{ border: 1px solid {_ORANGE}33; border-radius: 8px; padding: 4px 12px; margin: 4px 0; }}
    /* ── Dataframe ── */
    .stDataFrame {{ border: 1px solid {_ORANGE}33; border-radius: 8px; }}
    /* ── Divider ── */
    hr {{ border-color: {_ORANGE}33 !important; }}
    /* ── Buttons ── */
    .stButton > button {{
        background-color: {_ORANGE}22;
        color: {_ORANGE};
        border: 1px solid {_ORANGE};
        border-radius: 6px;
        font-weight: 700;
        transition: all 0.2s;
    }}
    .stButton > button:hover {{
        background-color: {_ORANGE};
        color: {_DARK};
    }}
    /* ── Selectbox ── */
    .stSelectbox label {{ color: {_TEAL}; font-weight: 600; }}
    /* ── Caption ── */
    small, .stCaption {{ color: {_MUTED} !important; }}
    </style>
    """, unsafe_allow_html=True)


def _card(content: str, border_color: str = _ORANGE) -> None:
    """Render un div con estilo card McLaren."""
    st.markdown(
        f'<div style="background:{_CARD};border:1px solid {border_color}44;'
        f'border-radius:10px;padding:16px;margin:8px 0;">{content}</div>',
        unsafe_allow_html=True,
    )


def _badge(text: str, color: str) -> str:
    return (
        f'<span style="background:{color}22;color:{color};border:1px solid {color}66;'
        f'border-radius:4px;padding:2px 8px;font-size:0.78rem;font-weight:700;">{text}</span>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# LOADERS — degradación elegante (T58-01)
# ══════════════════════════════════════════════════════════════════════════════

def load_shadow_report(desde=None, hasta=None) -> dict:
    try:
        from shadow_book import report_dict
        return report_dict(desde=desde, hasta=hasta)
    except Exception:
        return {}


def load_edge_report() -> dict:
    try:
        files = sorted(glob.glob("reports/edge_report_*.json"))
        return json.load(open(files[-1], encoding="utf-8")) if files else {}
    except Exception:
        return {}


def load_trader_plan() -> dict:
    try:
        files = sorted(glob.glob("reports/trader_plan_*.json"))
        return json.load(open(files[-1], encoding="utf-8")) if files else {}
    except Exception:
        return {}


def load_calibracion() -> dict:
    try:
        return json.load(open("data/calibracion_edge.json", encoding="utf-8"))
    except Exception:
        return {}


def load_hypotheses() -> dict:
    try:
        return json.load(open("validation/preregistered_hypotheses.json", encoding="utf-8"))
    except Exception:
        return {}


def load_h2h_report() -> dict:
    """Carga el h2h_results_enhanced más reciente."""
    try:
        files = sorted(glob.glob("reports/h2h_results_enhanced_*.json"))
        return json.load(open(files[-1], encoding="utf-8")) if files else {}
    except Exception:
        return {}


def load_odometer(desde_dt=None) -> dict:
    """Carga datos del odómetro de tokens (D59-01)."""
    try:
        from token_odometer import parse_sessions, DEFAULT_PROJECT_DIR
        return parse_sessions(DEFAULT_PROJECT_DIR, desde=desde_dt)
    except Exception:
        return {}


def load_live_edge() -> dict:
    """Carga el último live_edge_*.json del día."""
    try:
        files = sorted(glob.glob("reports/live_edge_*.json"), reverse=True)
        return json.load(open(files[0], encoding="utf-8")) if files else {}
    except Exception:
        return {}


def load_live_odds_history() -> dict:
    """Carga live_odds_history_YYYYMMDD.json del día actual."""
    try:
        today = datetime.now().strftime('%Y%m%d')
        path  = f"reports/live_odds_history_{today}.json"
        return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    except Exception:
        return {}


def load_meta_signal() -> dict:
    """Carga el meta_signal_*.json más reciente (Nodo-98 PASO 3b)."""
    try:
        files = sorted(glob.glob("reports/meta_signal_*.json"), reverse=True)
        return json.load(open(files[0], encoding="utf-8")) if files else {}
    except Exception:
        return {}


def load_live_sessions_today() -> list:
    """Carga todos los live_edge snapshots del día para tendencias de drift."""
    try:
        today = datetime.now().strftime('%Y%m%d')
        sessions = []
        for f in sorted(glob.glob(f"reports/live_edge_{today}_*.json")):
            try:
                sessions.append(json.load(open(f, encoding='utf-8')))
            except Exception:
                pass
        return sessions
    except Exception:
        return []


def load_apuestas_reales() -> list:
    """Carga picks CERRADO con resultado real de reports/apuestas_*.json."""
    results = []
    try:
        for f in sorted(glob.glob("reports/apuestas_*.json")):
            d = json.load(open(f, encoding="utf-8"))
            if d.get("estado") != "CERRADO":
                continue
            for pick in d.get("picks", []):
                if isinstance(pick, dict) and "correcto" in pick:
                    results.append(pick)
    except Exception:
        pass
    return results


def load_shadow_jsonl_raw(desde=None, hasta=None) -> list:
    """Carga registros crudos del shadow book (para Panel 3 SALUD)."""
    today = datetime.now().strftime('%Y-%m-%d')
    hasta = hasta or today
    desde = desde or today[:8] + "01"
    records = []
    try:
        import re
        for fpath in sorted(glob.glob("reports/shadow_book/sb_*.jsonl")):
            fname = os.path.basename(fpath)
            m = re.match(r'sb_(\d{4}-\d{2}-\d{2})\.jsonl', fname)
            if not m or not (desde <= m.group(1) <= hasta):
                continue
            fecha = m.group(1)
            for line in open(fpath, encoding="utf-8"):
                try:
                    r = json.loads(line.strip())
                    r['_fecha_archivo'] = fecha
                    records.append(r)
                except Exception:
                    pass
    except Exception:
        pass
    return records


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS PUROS — testables (T58-03..05)
# ══════════════════════════════════════════════════════════════════════════════

def _decision_status(n: int, n_stop: int, ic_lower=None,
                     breakeven=None, clv_median=None, extra_gate=True) -> str:
    if not extra_gate:
        return "NO_AUTORIZADO"
    if n < n_stop:
        return "NO_AUTORIZADO"
    if ic_lower is not None and breakeven is not None and ic_lower <= breakeven:
        return "NO_AUTORIZADO"
    if clv_median is not None and clv_median <= 0:
        return "NO_AUTORIZADO"
    return "AUTORIZADO"


def _clv_by_provenance(shadow: dict) -> dict:
    return shadow.get('clv_by_provenance', {})


def _waterfall_steps(wf: dict) -> list:
    if not wf:
        return []
    terminal = wf.get('terminal_reason') or ''
    steps = [
        ("kelly_kl",      f"{wf.get('kelly_kl_report', 0):.4f}"),
        ("raw_stake",     f"${wf.get('raw_stake', 0):,.0f}"),
        ("stake_pre_var", f"${wf.get('stake_pre_var', 0):,.0f}"),
        ("×var_factor",   f"×{wf.get('var_factor', 1):.3f}"),
        ("stake_final",   f"${wf.get('stake_final', 0):,.0f}"),
    ]
    result = []
    for label, val in steps:
        kill = (
            (label == "raw_stake" and wf.get('raw_stake', 0) == 0 and 'MIN_BET' not in terminal)
            or (label == "stake_pre_var" and 'MIN_BET_CLIFF' in terminal)
            or (label == "stake_final" and wf.get('stake_final', 0) == 0)
        )
        result.append((label, val, _RED if kill else _GREEN))
    return result


def _signal_accuracy_table(settled_records: list) -> list:
    """
    D58-06: Acierto direccional por señal × tier × superficie.
    Para cada señal activa en pick_snapshot, calcula n + hit% cuando esa señal
    apuntaba al favorito y el modelo ganó.
    Fuente: pick_snapshot de registros settled del shadow book.
    Retorna lista de dicts lista para st.dataframe().
    """
    from shadow_book import wilson_ci

    # Definición de señales y sus predicados sobre pick_snapshot
    SIGNALS = [
        ("surface_signal>0.5",   lambda s: s.get('surface_signal', 0) > 0.5),
        ("regime_signal>0",      lambda s: s.get('regime_signal', 0) > 0),
        ("markov=HOT",           lambda s: s.get('markov_favorito') == 'HOT'),
        ("markov=COLD rival",    lambda s: s.get('markov_rival') == 'COLD'),
        ("triple_alignment>0",   lambda s: s.get('triple_alignment', 0) > 0),
        ("STRUCTURAL_ALPHA",     lambda s: s.get('alignment_flag') == 'STRUCTURAL_ALPHA'),
        ("bbi_signal>0.5",       lambda s: s.get('bbi_signal', 0) > 0.5),
        ("n_axes>=3",            lambda s: s.get('n_axes_active', 0) >= 3),
        ("golden_zone=True",     lambda s: bool(s.get('golden_zone'))),
        ("edge>=15%",            lambda s: s.get('edge', 0) >= 0.15),
    ]

    non_void = [
        r for r in settled_records
        if r.get('resolucion', {}).get('resultado') in ('WON', 'LOST')
    ]

    rows = []
    for sig_label, sig_pred in SIGNALS:
        active = [r for r in non_void if sig_pred(r.get('pick_snapshot', {}))]
        if not active:
            continue
        n = len(active)
        hits = sum(1 for r in active if r['resolucion']['resultado'] == 'WON')
        hit_pct = round(hits / n * 100, 1) if n else 0.0
        ic = wilson_ci(n, hits)
        rows.append({
            'Señal': sig_label,
            'n': n,
            'Ganadas': hits,
            'hit%': f"{hit_pct:.1f}%{'*' if n < 10 else ''}",
            'IC 95%': f"[{ic[0]:.1f}, {ic[1]:.1f}]",
            'Estado': 'SPARSE*' if n < 10 else ('STRONG' if n >= 30 else 'BUILDING'),
        })

    # Segmento por tier (top 3 tiers con datos)
    tier_counts: dict = defaultdict(lambda: {'n': 0, 'hits': 0})
    for r in non_void:
        tier = r.get('pick_snapshot', {}).get('tier', 'unknown')
        tier_counts[tier]['n'] += 1
        if r['resolucion']['resultado'] == 'WON':
            tier_counts[tier]['hits'] += 1
    for tier, tc in tier_counts.items():
        n = tc['n']
        hits = tc['hits']
        hit_pct = round(hits / n * 100, 1) if n else 0.0
        ic = wilson_ci(n, hits)
        rows.append({
            'Señal': f"[tier={tier}]",
            'n': n,
            'Ganadas': hits,
            'hit%': f"{hit_pct:.1f}%{'*' if n < 10 else ''}",
            'IC 95%': f"[{ic[0]:.1f}, {ic[1]:.1f}]",
            'Estado': 'SPARSE*' if n < 10 else ('STRONG' if n >= 30 else 'BUILDING'),
        })

    return rows


def _component_directions(p1_bd: dict, p2_bd: dict,
                           resultado: str, favored: str, j1: str) -> dict:
    """
    D58-06: Para cada componente, determina si apuntó al ganador real.
    resultado: 'WON' (favorito ganó) | 'LOST' (favorito perdió)
    Retorna {comp_key: 'correct'|'incorrect'|'tie'}
    """
    favorito_won = resultado == 'WON'
    directions = {}
    for comp_key in p1_bd:
        if comp_key in ('Penalizacion_Inactividad', 'Puntaje_Final'):
            continue
        d1 = p1_bd.get(comp_key, {})
        d2 = p2_bd.get(comp_key, {})
        if not isinstance(d1, dict) or not isinstance(d2, dict):
            continue
        try:
            s1 = float(d1.get('weighted_score', 0))
            s2 = float(d2.get('weighted_score', 0))
        except (ValueError, TypeError):
            continue
        comp_favors_j1 = s1 > s2
        j1_is_favorito = (j1 == favored)
        comp_favors_favorito = comp_favors_j1 if j1_is_favorito else not comp_favors_j1
        if abs(s1 - s2) < 0.001:
            directions[comp_key] = 'tie'
        elif comp_favors_favorito == favorito_won:
            directions[comp_key] = 'correct'
        else:
            directions[comp_key] = 'incorrect'
    return directions


def _was_candidates(edge: dict) -> list:
    candidates = []
    for pool in ('apostar', 'watchlist', 'sin_edge'):
        for p in edge.get(pool, []):
            if p.get('edge', 0) >= 0.10 and p.get('cuota_favorito', 0) >= 2.0:
                candidates.append({
                    'partido': p.get('partido', ''),
                    'edge_pct': p.get('edge_pct', ''),
                    'cuota': p.get('cuota_favorito', 0),
                    'markov': p.get('markov_favorito') or '—',
                    'tier': p.get('tier', ''),
                    'was_ok': p.get('markov_favorito') == 'HOT',
                })
    return candidates


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — HOY
# ══════════════════════════════════════════════════════════════════════════════

def panel_hoy(shadow: dict, trader: dict, edge: dict, calibracion: dict = None) -> None:
    st.header("Panel 1 — HOY")

    summary = shadow.get('summary', {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Picks loggeados", summary.get('n_total', 0))
    col2.metric("Settled", summary.get('n_settled', 0))
    col3.metric("Abiertos", summary.get('n_open', 0))

    rm = trader.get('risk_management', {})
    kgr = rm.get('kelly_growth_rate')
    col4.metric("KGR sesión", f"{kgr:.4f}" if kgr is not None else "—",
                delta="OK" if kgr and kgr > 0 else ("RUINA" if kgr and kgr < 0 else None),
                delta_color="normal" if kgr and kgr > 0 else "inverse")

    if kgr is not None and kgr < 0:
        st.error("REGLA-HF-5: KGR < 0 — NO DESPLEGAR esta sesión.")
    if rm.get('var_excedido'):
        var_95 = rm.get('var_95', 0)
        st.warning(f"VaR excedido: ${var_95:,.0f}. Stakes ya ajustados automáticamente.")

    no_data_n = len(edge.get('no_data', []) + edge.get('sin_datos', []))
    if no_data_n:
        st.info(f"{no_data_n} picks NO_DATA en el edge_report — historial vacío por construcción (F2).")

    # ── Cascada de stakes ──
    st.subheader("Cascada de Stakes")
    individuales = trader.get('individuales', [])
    if not individuales:
        st.info("Sin trader plan disponible.")
    else:
        for pick in individuales:
            wf = pick.get('_waterfall', {})
            stake = pick.get('stake', 0)
            partido = pick.get('partido', pick.get('favorito', 'Partido'))
            terminal = (wf or {}).get('terminal_reason', '')
            color = _GREEN if stake > 0 else _RED

            with st.expander(
                f"{partido}  —  stake ${stake:,.0f}  |  {pick.get('edge_pct','?')} edge  |  cuota {pick.get('cuota','?')}"
            ):
                steps = _waterfall_steps(wf)
                cols = st.columns(len(steps) or 1)
                for i, (label, val, c) in enumerate(steps):
                    with cols[i]:
                        st.markdown(
                            f'<div style="text-align:center;background:{_CARD};border:1px solid {c}55;'
                            f'border-radius:8px;padding:10px;">'
                            f'<div style="color:{_MUTED};font-size:0.7rem;text-transform:uppercase;">{label}</div>'
                            f'<div style="color:{c};font-size:1.1rem;font-weight:700;">{val}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                if terminal:
                    st.caption(f"Terminal: {terminal}")

    # ── Candidatos WAS ──
    st.subheader("Candidatos WAS  (edge ≥ 10%, cuota ≥ 2.0)")
    was_cands = _was_candidates(edge)
    if not was_cands:
        st.info("Sin candidatos WAS hoy.")
    else:
        rows = [
            {
                "Partido": c['partido'],
                "Edge": c['edge_pct'],
                "Cuota": f"{c['cuota']:.2f}",
                "Markov": c['markov'],
                "WAS": "ACTIVO" if c['was_ok'] else "sin señal HOT",
                "Tier": c['tier'],
            }
            for c in was_cands
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.divider()

    # ── Señales Activas (I2 Nodo-67) ──
    st.subheader("Señales Activas")
    av  = shadow.get('anchor_variable', {})
    rfi = shadow.get('rfi', {})
    rv  = shadow.get('rival_value', {})
    anc  = av.get('anchor', {})
    var_ = av.get('variable', {})
    rfi_u = rfi.get('ultra', {})
    rfi_t = rfi.get('tier1plus', {})

    def _fmt_seg(d: dict) -> str:
        if not d or d.get('sparse') or d.get('n', 0) == 0:
            return "n insuf."
        return f"n={d['n']} | {d['hit_pct']}% | ROI {d['roi']:+.1f}%"

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("ANCHOR (edge>0)",  _fmt_seg(anc))
    sc2.metric("VARIABLE (edge≤0)", _fmt_seg(var_))
    sc3.metric("RFI Ultra",         _fmt_seg(rfi_u))
    sc4.metric("RFI tier≥1",        _fmt_seg(rfi_t))

    if rv.get('n', 0) > 0:
        _card(
            f'<b style="color:{_TEAL};">RIVAL VALUE H88-01</b>  '
            f'n={rv["n"]}/30 · {rv.get("hit_pct_rival", 0)}% hit_rival · '
            f'ROI {rv.get("roi_rival", 0):+.1f}% · {rv.get("estado", "—")}',
            border_color=_TEAL,
        )

    st.divider()

    # ── Odómetro M0 — Pipeline (I2 Nodo-67) ──
    st.subheader("Odómetro M0 — Pipeline  (READ-ONLY)")
    m0   = shadow.get('m0', {})
    sess = shadow.get('sessions', {})
    summ = shadow.get('summary', {})
    hyps = shadow.get('hypotheses', [])

    oc1, oc2, oc3, oc4 = st.columns(4)
    oc1.metric("Sesiones (periodo)", sess.get('n', 0))
    oc2.metric("Picks loggeados",    summ.get('n_total', 0))
    oc3.metric("Settled",            summ.get('n_settled', 0))
    dias_ss = m0.get('dias_sin_settle', 0)
    oc4.metric(
        "Días sin settle (hist.)", dias_ss,
        delta="settle-guard activo" if dias_ss > 0 else "OK",
        delta_color="inverse" if dias_ss > 0 else "normal",
    )

    oc5, oc6 = st.columns([1, 3])
    gov_exec = m0.get('governor_executions', 0)
    oc5.metric(
        "Ejecuciones Governor", gov_exec,
        delta="gate: 10 sesiones" if gov_exec < 10 else "SIN GATE",
        delta_color="off",
    )

    active_hyps = [h for h in hyps if h.get('estado', 'CONTINUAR') == 'CONTINUAR']
    with oc6:
        if active_hyps:
            st.caption("H-XX activas — n_actual / n_stop")
            for h in active_hyps:
                h_id   = h.get('id', '?')
                n_act  = h.get('n', 0)
                n_stop = h.get('n_stop', 30)
                pct    = min(n_act / n_stop, 1.0) if n_stop else 0
                bar_w  = int(pct * 180)
                color  = _GREEN if pct >= 1.0 else (_ORANGE if pct >= 0.5 else _BLUE)
                st.markdown(
                    f'<div style="margin:3px 0;">'
                    f'<span style="color:{_MUTED};width:80px;display:inline-block;'
                    f'font-size:0.78rem;">{h_id}</span>'
                    f'<div style="display:inline-block;background:{color};'
                    f'width:{bar_w}px;height:9px;border-radius:3px;vertical-align:middle;"></div>'
                    f'<span style="color:{_TEXT};margin-left:8px;font-size:0.78rem;">'
                    f'{n_act}/{n_stop}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Sin hipótesis activas (CONTINUAR) en el período.")

    # ── C4 Nodo-67: Brecha hit%_real vs hit%_shadow (Nodo-86 §1.1) ──
    st.subheader("Brecha hit%_real vs hit%_shadow  (C4 — Nodo-86 §1.1)")

    # hit%_real — de calibracion_edge.json (fuente: betslip_registrar --cerrar)
    _cal  = calibracion or {}
    _glob = _cal.get('global', {})
    _n_real  = _glob.get('wins', 0) + _glob.get('losses', 0)
    _won_real = _glob.get('wins', 0)

    # hit%_shadow — de report_dict() full-history (summary.n_hits / n_settled)
    try:
        from shadow_book import report_dict as _sb_rdict, wilson_ci as _wci
        _sb = _sb_rdict()
        _sb_sum = _sb.get('summary', {})
        _n_sh   = _sb_sum.get('n_settled', 0)
        _won_sh = _sb_sum.get('n_hits', 0)
    except Exception:
        _n_sh, _won_sh = 0, 0
        _wci = None

    def _hit(w, n): return w / n if n else None
    def _ic_c4(w, n):
        try:
            return _wci(w, n) if _wci else (0.0, 1.0)
        except Exception:
            return (0.0, 1.0)

    _hr  = _hit(_won_real, _n_real)
    _hs  = _hit(_won_sh,   _n_sh)
    _icr = _ic_c4(_won_real, _n_real)
    _ics = _ic_c4(_won_sh,   _n_sh)

    bc1, bc2, bc3 = st.columns(3)
    bc1.metric(
        f"hit%_REAL calibracion (n={_n_real})",
        f"{_hr:.1%}" if _hr is not None else "N/A",
        delta=f"IC95 [{_icr[0]:.1%}, {_icr[1]:.1%}]" if _n_real > 0 else None,
        delta_color="off",
    )
    bc2.metric(
        f"hit%_SHADOW settled (n={_n_sh})",
        f"{_hs:.1%}" if _hs is not None else "N/A",
        delta=f"IC95 [{_ics[0]:.1%}, {_ics[1]:.1%}]" if _n_sh > 0 else None,
        delta_color="off",
    )
    if _hr is not None and _hs is not None:
        _brecha = _hs - _hr
        _sign   = f"+{_brecha:.1%}" if _brecha >= 0 else f"{_brecha:.1%}"
        _nota   = "brecha alta — revisar sesgo de seleccion" if abs(_brecha) > 0.10 else "OK"
        bc3.metric("Brecha shadow−real", _sign, delta=_nota,
                   delta_color="inverse" if abs(_brecha) > 0.10 else "normal")
    else:
        bc3.metric("Brecha shadow−real", "N/A", delta="sin datos suficientes", delta_color="off")

    st.caption(
        "hit%_real = calibracion_edge.json global (betslip_registrar --cerrar). "
        "hit%_shadow = shadow book settled (WON/total). "
        "Brecha positiva = shadow sobreestima — revisar sesgo de seleccion."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — HIPÓTESIS
# ══════════════════════════════════════════════════════════════════════════════

def panel_hipotesis(shadow: dict, hypotheses_json: dict) -> None:
    st.header("Panel 2 — HIPÓTESIS")

    n_settled = shadow.get('summary', {}).get('n_settled', 0)
    if n_settled == 0:
        st.warning("Sin registros settled. Correr: `python3 shadow_book.py --settle FECHA`")
        return

    grad = shadow.get('graduation', {})
    if grad.get('any_graduated'):
        st.success(f"GRADUADO: {', '.join(grad.get('graduated_labels', []))} — listo para escalar.")
    else:
        st.info(f"Sin segmento graduado. Más cercano: n={grad.get('nearest_n', 0)}/30")

    # ── CLV por provenance ──
    clv_prov = _clv_by_provenance(shadow)
    if clv_prov:
        st.subheader("CLV por provenance (D52-08 — nunca mezclado)")
        clv_cols = st.columns(max(len(clv_prov), 1))
        for i, (prov, data) in enumerate(clv_prov.items()):
            with clv_cols[i]:
                if data.get('excluded'):
                    st.metric(f"{prov}", f"n={data['n']}", delta="EXCLUIDO de CLV (D52-07)",
                              delta_color="off")
                else:
                    clv_med = data.get('clv_median')
                    label = f"CLV {clv_med:+.1f}%" if clv_med is not None else "CLV sin datos"
                    st.metric(prov, label, f"n={data['n_clv']}/{data['n']} con CLV")
        st.divider()

    # ── Hipótesis ──
    meta_hyps = hypotheses_json.get('hypotheses', {})
    for h in shadow.get('hypotheses', []):
        h_id   = h.get('id', '')
        n      = h.get('n', 0)
        n_stop = h.get('n_stop', 30)
        estado = h.get('estado', 'CONTINUAR')
        hits   = h.get('hits', 0)
        hit_pct= h.get('hit_pct', 0.0)
        ic     = h.get('ic', [0, 100])
        be     = h.get('breakeven')
        clv_m  = h.get('clv_median')
        sparse = h.get('sparse', False)

        meta   = meta_hyps.get(h_id, {})
        nombre = meta.get('nombre', h.get('label', h_id))

        # Color según estado
        if h.get('graduado'):
            border = _GREEN
            badge_html = _badge("GRADUADO", _GREEN)
        elif estado == 'NO_GRADUABLE':
            border = _RED
            badge_html = _badge("NO GRADUABLE", _RED)
        elif estado == 'GRADUABLE':
            border = _ORANGE
            badge_html = _badge("GRADUABLE", _ORANGE)
        else:
            border = _BLUE
            badge_html = _badge(f"n={n}/{n_stop}", _BLUE)

        with st.container():
            st.markdown(
                f'<div style="background:{_CARD};border-left:3px solid {border};'
                f'border-radius:8px;padding:12px 16px;margin:6px 0;">'
                f'<b style="color:{_ORANGE};">{h_id}</b>&nbsp;&nbsp;{badge_html}'
                f'<div style="color:{_TEXT};margin:4px 0;">{nombre}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            prog_cols = st.columns([3, 1, 1, 1])
            with prog_cols[0]:
                st.progress(min(n / n_stop, 1.0) if n_stop > 0 else 0.0,
                            text=f"n={n}/{n_stop}")
            with prog_cols[1]:
                st.metric("hit%", f"{hit_pct:.1f}%{'*' if sparse else ''}")
            with prog_cols[2]:
                ic_str = f"[{ic[0]:.1f}, {ic[1]:.1f}]"
                st.metric("IC 95%", ic_str,
                          f"be={be:.1f}%" if be else "be=?")
            with prog_cols[3]:
                st.metric("CLV", f"{clv_m:.2f}%" if clv_m is not None else "—")

            if h_id == 'H52-05':
                steam = h.get('steam', {})
                drift = h.get('drift', {})
                st.caption(
                    f"STEAM_IN n={steam.get('n',0)} hit%={steam.get('hit_pct',0):.1f}%  |  "
                    f"DRIFT_OUT n={drift.get('n',0)} hit%={drift.get('hit_pct',0):.1f}%"
                )
        st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 3 — SALUD DE DATOS  (D58-04)
# ══════════════════════════════════════════════════════════════════════════════

def panel_salud(shadow: dict, desde_str: str, hasta_str: str, calibracion: dict) -> None:
    st.header("Panel 3 — SALUD DE DATOS")
    st.caption("Habría detectado Nodo-47 en un día. READ-ONLY.")

    records = load_shadow_jsonl_raw(desde=desde_str, hasta=hasta_str)
    picks = [r for r in records if r.get('_type') != 'session_meta']

    if not picks:
        st.info("Sin picks en el rango seleccionado.")
    else:
        # ── history_provenance por día ──
        st.subheader("Distribución history_provenance por día")
        prov_labels = ['ninja_api', 'thf_cache', 'playwright', 'EMPTY', 'unknown']
        prov_colors = [_ORANGE, _BLUE, _TEAL, _RED, _MUTED]

        by_day: dict = defaultdict(lambda: defaultdict(int))
        for r in picks:
            fecha = r.get('_fecha_archivo', '?')
            hp = r.get('pick_snapshot', {}).get('history_provenance', {})
            if not hp:
                by_day[fecha]['EMPTY'] += 1
                continue
            for side in ('p1', 'p2'):
                prov = hp.get(side, 'unknown')
                by_day[fecha][prov] += 1

        if by_day:
            try:
                import plotly.graph_objects as go
                dias = sorted(by_day.keys())
                fig = go.Figure()
                for prov, color in zip(prov_labels, prov_colors):
                    vals = [by_day[d].get(prov, 0) for d in dias]
                    if any(v > 0 for v in vals):
                        fig.add_trace(go.Bar(
                            name=prov, x=dias, y=vals,
                            marker_color=color,
                        ))
                fig.update_layout(
                    barmode='stack',
                    plot_bgcolor=_DARK, paper_bgcolor=_PANEL,
                    font_color=_TEXT,
                    legend=dict(bgcolor=_CARD, bordercolor="rgba(255,128,0,0.27)"),
                    xaxis=dict(gridcolor="rgba(255,128,0,0.13)"),
                    yaxis=dict(gridcolor="rgba(255,128,0,0.13)"),
                    height=320, margin=dict(t=20, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                rows = [{'Fecha': d, **dict(v)} for d, v in sorted(by_day.items())]
                st.dataframe(rows, use_container_width=True, hide_index=True)

        # ── Alerta EMPTY ──
        total_picks = len(picks)
        empty_count = sum(1 for r in picks
                         if not r.get('pick_snapshot', {}).get('history_provenance'))
        if total_picks > 0:
            empty_pct = empty_count / total_picks * 100
            if empty_pct > 10:
                st.error(f"Alerta: {empty_pct:.0f}% de picks con history_provenance EMPTY ({empty_count}/{total_picks}). "
                         f"Posible problema de scraping.")

        # ── ranking_provenance (campo futuro) ──
        st.subheader("ranking_provenance por tier")
        rp_found = any(
            r.get('pick_snapshot', {}).get('ranking_provenance')
            for r in picks
        )
        if not rp_found:
            _card(
                f'<span style="color:{_MUTED}">Campo <code>ranking_provenance</code> aún no presente en picks actuales. '
                f'Cuando esté disponible, mostrará % kambi_estimate por tier '
                f'(firma del bug Nodo-47: Challenger &gt;20% = alerta).</span>',
                border_color=_MUTED,
            )
        else:
            by_tier: dict = defaultdict(lambda: defaultdict(int))
            for r in picks:
                snap = r.get('pick_snapshot', {})
                tier = snap.get('tier', 'unknown')
                rp = snap.get('ranking_provenance', 'unknown')
                by_tier[tier][rp] += 1
            for tier, dist in by_tier.items():
                total = sum(dist.values())
                kambi_pct = dist.get('kambi_estimate', 0) / total * 100 if total else 0
                color = _RED if kambi_pct > 20 else _GREEN
                st.markdown(
                    f'<b style="color:{_ORANGE};">{tier}</b>: '
                    f'kambi_estimate {kambi_pct:.0f}% {_badge("ALERTA", _RED) if kambi_pct > 20 else _badge("OK", _GREEN)}',
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Calibracion por epoch ──
    st.subheader("Calibración por epoch (calibracion_edge.json)")
    if not calibracion:
        st.info("calibracion_edge.json no disponible.")
    else:
        epochs_meta = calibracion.get('_meta', {}).get('calibration_epochs', {})
        col1, col2, col3 = st.columns(3)
        for col, (ep_key, ep_col) in zip([col1, col2, col3],
                                          [('epoch_1', col1), ('epoch_2', col2), ('epoch_3', col3)]):
            ep_data = epochs_meta.get(ep_key, {})
            with ep_col:
                n_obs = ep_data.get('n_observaciones', 0)
                st.metric(
                    ep_key.upper(),
                    f"n={n_obs}",
                    ep_data.get('descripcion', '?')[:40],
                )

        # Desglose por superficie+tier
        st.subheader("n por superficie+tier")
        pst = calibracion.get('por_superficie_y_tier', {})
        if pst:
            rows = []
            for key, val in sorted(pst.items()):
                n = val.get('n', 0)
                p = val.get('p', 0)
                rows.append({'Segmento': key, 'n': n, 'p': f"{p:.3f}",
                             'Estado': 'STRONG' if n >= 30 else ('BUILDING' if n >= 10 else 'SPARSE*')})
            st.dataframe(rows, use_container_width=True, hide_index=True)

    # ── NO_DATA por día ──
    st.subheader("NO_DATA por día")
    nd_by_day: dict = defaultdict(int)
    for r in picks:
        snap = r.get('pick_snapshot', {})
        status_nd = (snap.get('no_data') or snap.get('apostar') is None)
        if snap.get('apostar') is False and not snap.get('watchlist'):
            nd_by_day[r.get('_fecha_archivo', '?')] += 1
    if nd_by_day:
        rows = [{'Fecha': d, 'NO_DATA': n} for d, n in sorted(nd_by_day.items())]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.caption("Sin NO_DATA registrados en el rango.")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 4 — ATRIBUCIÓN POR PARTIDO  (D58-05)
# ══════════════════════════════════════════════════════════════════════════════

def panel_atribucion(h2h: dict, shadow: dict) -> None:
    st.header("Panel 4 — ATRIBUCIÓN POR PARTIDO")
    st.caption("Componentes con _weights_final (D56-01). Penalización de Inactividad visible (D56-05).")

    partidos_raw = h2h.get('partidos', [])
    if not partidos_raw:
        st.info("Sin h2h_results_enhanced disponible. Correr PASO 2 del pipeline.")
        return

    # ── Selector de partido ──
    opciones = []
    for p in partidos_raw:
        j1 = p.get('jugador1', '')
        j2 = p.get('jugador2', '')
        torneo = p.get('torneo_nombre', '')
        if j1 and j2:
            opciones.append(f"{j1} vs {j2} | {torneo}")

    if not opciones:
        st.info("No hay partidos con datos disponibles.")
        return

    sel = st.selectbox("Seleccionar partido", opciones, key="atribucion_partido")
    idx = opciones.index(sel)
    partido = partidos_raw[idx]

    pred = partido.get('ranking_analysis', {}).get('prediction', {}) or {}
    score_bd = pred.get('score_breakdown', {})
    p1_bd = score_bd.get('player1', {})
    p2_bd = score_bd.get('player2', {})

    j1 = partido.get('jugador1', 'J1')
    j2 = partido.get('jugador2', 'J2')
    favored = pred.get('favored_player', j1)
    conf = pred.get('confidence', 0)
    wf = pred.get('_weights_final', {})

    # ── Header tarjeta ──
    _card(
        f'<b style="color:{_ORANGE};font-size:1.1rem;">{j1}  vs  {j2}</b><br>'
        f'<span style="color:{_MUTED};">{partido.get("torneo_nombre","?")}  |  '
        f'{partido.get("tipo_cancha","?")}</span><br>'
        f'<span style="color:{_BLUE};">Favorito: <b>{favored}</b>  |  Confianza: {conf:.1f}%</span>',
    )

    if not p1_bd and not p2_bd:
        st.info("Sin score_breakdown disponible para este partido.")
        return

    # ── Tabla de componentes ──
    st.subheader("Componentes de predicción")

    COMPONENT_LABELS = {
        'surface_specialization': 'Especialización Superficie',
        'form_recent': 'Forma Reciente',
        'common_opponents': 'Rivales Comunes',
        'h2h_direct': 'H2H Directo',
        'ranking_momentum': 'Ranking/Momentum',
        'elo_rating': 'Rating ELO',
        'home_advantage': 'Ventaja Localía',
        'strength_of_schedule': 'Fuerza Calendario',
        'surface_advantage': 'Ventaja Superficie',
    }

    rows = []
    for comp_key, comp_label in COMPONENT_LABELS.items():
        d1 = p1_bd.get(comp_key, {}) if isinstance(p1_bd.get(comp_key), dict) else {}
        d2 = p2_bd.get(comp_key, {}) if isinstance(p2_bd.get(comp_key), dict) else {}
        if not d1 and not d2:
            continue
        weight = (
            wf.get(comp_key, 0) * 100 if wf and comp_key in wf
            else float((d1.get('weight', '0%') or '0%').replace('%', ''))
        )
        rows.append({
            'Componente': comp_label,
            'Peso real': f"{weight:.1f}%",
            f'{j1} raw': d1.get('raw_score', '—'),
            f'{j1} puntaje': d1.get('weighted_score', '—'),
            f'{j1} contrib%': d1.get('contribution', '—'),
            f'{j2} raw': d2.get('raw_score', '—'),
            f'{j2} puntaje': d2.get('weighted_score', '—'),
            f'{j2} contrib%': d2.get('contribution', '—'),
        })

    # ── D56-05: Penalización de Inactividad ──
    pen1 = p1_bd.get('Penalizacion_Inactividad', '0.00 pts')
    pen2 = p2_bd.get('Penalizacion_Inactividad', '0.00 pts')
    try:
        pen1_f = float(str(pen1).replace(' pts', ''))
        pen2_f = float(str(pen2).replace(' pts', ''))
    except (ValueError, TypeError):
        pen1_f = pen2_f = 0.0

    if pen1_f != 0.0 or pen2_f != 0.0:
        rows.append({
            'Componente': 'Penalizacion Inactividad',
            'Peso real': '—',
            f'{j1} raw': '—', f'{j1} puntaje': f"{pen1_f:.4f}", f'{j1} contrib%': '—',
            f'{j2} raw': '—', f'{j2} puntaje': f"{pen2_f:.4f}", f'{j2} contrib%': '—',
        })

    # ── PUNTAJE FINAL ──
    pf1 = pred.get('p1_final_weight', pred.get('score_p1', 0))
    pf2 = pred.get('p2_final_weight', pred.get('score_p2', 0))
    rows.append({
        'Componente': 'PUNTAJE FINAL TOTAL',
        'Peso real': '—',
        f'{j1} raw': '—', f'{j1} puntaje': f"{pf1:.4f}" if pf1 else '—', f'{j1} contrib%': '—',
        f'{j2} raw': '—', f'{j2} puntaje': f"{pf2:.4f}" if pf2 else '—', f'{j2} contrib%': '—',
    })

    st.dataframe(rows, use_container_width=True, hide_index=True)

    # ── D58-06: Post-settlement coloreo por componente ──
    st.subheader("Post-settlement")
    settled_match = None
    match_id = partido.get('match_id') or partido.get('match_url', '')
    all_raw = load_shadow_jsonl_raw()
    for r in all_raw:
        snap = r.get('pick_snapshot', {})
        if (snap.get('match_id') == match_id
                or snap.get('partido', '') == f"{j1} vs {j2}"):
            if 'resolucion' in r:
                settled_match = r
                break

    if settled_match:
        res = settled_match.get('resolucion', {})
        resultado = res.get('resultado', '')
        color_res = _GREEN if resultado == 'WON' else _RED
        clv = res.get('clv_pct')
        clv_str = f"{clv:+.1f}%" if clv is not None else "N/A"
        _card(
            f'<b style="color:{color_res};font-size:1.1rem;">Resultado: {resultado}</b>'
            f'&nbsp;&nbsp;{_badge("CLV " + clv_str, _BLUE if clv and clv > 0 else _MUTED)}'
            f'<br><span style="color:{_MUTED};">Cuota cierre: {res.get("cuota_cierre","N/A")}  '
            f'({res.get("cuota_cierre_provenance","?")})</span>',
            border_color=color_res,
        )

        # Coloreo por componente: verde=apuntó al ganador, rojo=no
        if p1_bd and p2_bd:
            directions = _component_directions(p1_bd, p2_bd, resultado, favored, j1)
            if directions:
                st.markdown(
                    f'<p style="color:{_BLUE};font-weight:600;margin:12px 0 4px;">Acierto direccional por componente</p>',
                    unsafe_allow_html=True,
                )
                COMP_LABELS = {
                    'surface_specialization': 'Sup. Especialización',
                    'form_recent': 'Forma Reciente',
                    'common_opponents': 'Rivales Comunes',
                    'h2h_direct': 'H2H Directo',
                    'ranking_momentum': 'Ranking/Momentum',
                    'elo_rating': 'ELO',
                    'home_advantage': 'Localía',
                    'strength_of_schedule': 'SoS',
                    'surface_advantage': 'Ventaja Sup.',
                }
                badges_html = ""
                for comp_key, direction in directions.items():
                    label = COMP_LABELS.get(comp_key, comp_key)
                    if direction == 'correct':
                        badges_html += _badge(f"✓ {label}", _GREEN) + " "
                    elif direction == 'incorrect':
                        badges_html += _badge(f"✗ {label}", _RED) + " "
                    else:
                        badges_html += _badge(f"~ {label}", _MUTED) + " "
                st.markdown(badges_html, unsafe_allow_html=True)
    else:
        st.caption("Partido aún no settled.")

    # ── Señales especiales ──
    analysis_log = pred.get('analysis_log', [])
    senales = [l for l in analysis_log if any(
        kw in l for kw in ('TORNEO_COMPLETO', 'HOT', 'COLD', 'SURFACE_BONUS',
                           'LOG_FORM_DECAY', 'IMMUNITY', 'PELT')
    )]
    if senales:
        st.subheader("Señales Especiales")
        for s in senales[:10]:
            st.markdown(f'<code style="color:{_TEAL};">{s}</code>', unsafe_allow_html=True)

    st.divider()

    # ── D58-06: Tabla acumulada acierto-por-señal ──
    with st.expander("Tabla acumulada — Acierto por señal × tier (todos los settled)"):
        settled_all = [r for r in all_raw if r.get('resolucion', {}).get('resultado') in ('WON', 'LOST')]
        n_settled_total = len(settled_all)
        st.caption(
            f"n={n_settled_total} picks settled en el rango. "
            f"{'SPARSE* = n<10 — interpretación limitada.' if n_settled_total < 30 else ''}"
            f"{'Con n≥100 este panel es el insumo para recalibrar Nodo-21.' if n_settled_total >= 100 else ''}"
        )
        if n_settled_total == 0:
            st.info("Sin picks settled todavía. Correr --settle después de cada partido.")
        else:
            acc_rows = _signal_accuracy_table(settled_all)
            if acc_rows:
                # Colorear la columna Estado
                st.dataframe(acc_rows, use_container_width=True, hide_index=True)
            else:
                st.info("Sin señales con datos suficientes.")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 5 — RIESGO  (D58-04)
# ══════════════════════════════════════════════════════════════════════════════

def panel_riesgo(trader: dict, shadow: dict) -> None:
    st.header("Panel 5 — RIESGO")
    st.caption("Apuestas REALES y SIMULADAS SIEMPRE en tablas separadas (Regla C). NUNCA mezclar.")

    rm = trader.get('risk_management', {})
    sistema = trader.get('sistema', {})
    meta = trader.get('metadata', {})

    # ── Métricas clave ──
    col1, col2, col3, col4 = st.columns(4)
    kgr = rm.get('kelly_growth_rate')
    var_95 = rm.get('var_95', 0)
    var_excedido = rm.get('var_excedido', False)
    MAX_VAR_PCT = 0.25

    col1.metric("KGR sesión", f"{kgr:.4f}" if kgr is not None else "—",
                delta="CRECIMIENTO" if kgr and kgr > 0 else ("RUINA" if kgr and kgr < 0 else None),
                delta_color="normal" if kgr and kgr > 0 else "inverse")
    col2.metric("VaR 95%", f"${abs(var_95):,.0f}" if var_95 else "—",
                delta="EXCEDIDO" if var_excedido else "OK",
                delta_color="inverse" if var_excedido else "normal")
    col3.metric("Sharpe ratio", f"{rm.get('sharpe_ratio', 0):.2f}" if rm.get('sharpe_ratio') else "—")
    col4.metric("Sessions to double", f"{rm.get('sessions_to_double', '—')}")

    # ── Indicador VaR vs límite ──
    if var_95:
        bankroll_est = trader.get('metadata', {}).get('bankroll', 0) or 0
        if bankroll_est > 0:
            var_pct_actual = abs(var_95) / bankroll_est
            st.markdown(f"**VaR como % bankroll: {var_pct_actual*100:.1f}%  |  Límite: 25%**")
            color_var = _RED if var_pct_actual > MAX_VAR_PCT else _GREEN
            st.markdown(
                f'<div style="background:{_CARD};border-radius:8px;padding:8px;">'
                f'<div style="background:{color_var};height:12px;border-radius:6px;'
                f'width:{min(var_pct_actual/MAX_VAR_PCT*100, 100):.0f}%;"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Stakes simulados (shadow book) ──
    st.subheader("Stakes SIMULADOS — shadow book (pick_snapshot)")
    individuales = trader.get('individuales', [])
    if individuales:
        sim_rows = []
        for p in individuales:
            sim_rows.append({
                'Partido': p.get('partido', p.get('favorito', '?'))[:40],
                'Tier': p.get('superficie', '?'),
                'Cuota': p.get('cuota', '?'),
                'Edge': p.get('edge_pct', '?'),
                'Kelly KL': f"{p.get('kelly_kl', 0):.4f}",
                'Stake $': f"${p.get('stake', 0):,.0f}",
                'Retorno pot.': f"${p.get('retorno_potencial', 0):,.0f}",
            })
        st.dataframe(sim_rows, use_container_width=True, hide_index=True)
    else:
        st.info("Sin trader plan disponible.")

    st.divider()

    # ── Apuestas REALES ──
    st.subheader("Apuestas REALES — betslip_registrar")
    apuestas = load_apuestas_reales()
    if not apuestas:
        st.info("Sin apuestas reales registradas. Usar betslip_registrar.py para registrar.")
    else:
        real_rows = []
        for a in apuestas[:50]:
            real_rows.append({
                'Partido': str(a.get('partido', a.get('combo', '?')))[:40],
                'Stake': f"${a.get('stake', 0):,.0f}",
                'Cuota': a.get('cuota', '?'),
                'Estado': a.get('estado', 'PENDIENTE'),
                'Ganancia': f"${a.get('ganancia', 0):,.0f}" if a.get('ganancia') else '—',
            })
        st.dataframe(real_rows, use_container_width=True, hide_index=True)

        # P&L real
        pnl_total = sum(a.get('ganancia', 0) or 0 for a in apuestas if a.get('estado') == 'GANADA')
        pnl_total -= sum(a.get('stake', 0) or 0 for a in apuestas if a.get('estado') == 'PERDIDA')
        color_pnl = _GREEN if pnl_total >= 0 else _RED
        _card(
            f'<b style="color:{_ORANGE};">P&L REAL acumulado:</b> '
            f'<span style="color:{color_pnl};font-size:1.4rem;font-weight:700;">'
            f'{"+" if pnl_total >= 0 else ""}${pnl_total:,.0f}</span>',
            border_color=color_pnl,
        )

    st.divider()

    # ── Combos de la sesión ──
    combos = trader.get('combos', [])
    if combos:
        st.subheader("Combos de la sesión")
        combo_rows = []
        for c in combos[:10]:
            combo_rows.append({
                'Piernas': len(c.get('picks', [])),
                'Cuota combo': f"{c.get('cuota_total', 0):.2f}",
                'Stake $': f"${c.get('stake', 0):,.0f}",
                'Retorno pot.': f"${c.get('retorno', 0):,.0f}",
            })
        st.dataframe(combo_rows, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 6 — DECISIÓN
# ══════════════════════════════════════════════════════════════════════════════

_DECISION_TABLE = [
    ("BAJAR P_MODELO_MIN en qualifiers",
     "Reducir piso de prob. mínima qualifying 0.55→0.52",
     "H52-07", 50, None),
    ("RELAJAR T33-01  (floor n_h2h=0 ITF)",
     "Habilitar picks ITF sin historial H2H en pool principal",
     "H52-02", 30, None),
    ("FLOOR DE STAKE — MIN_STAKE_APOSTAR",
     "Implementar stake floor $500 para picks aplastados por VaR/MIN_BET",
     "H54-01", 30, None),
    ("ESCALAR SEGMENTO GS",
     "Aumentar kelly_usado en picks grand_slam APOSTAR",
     None, 30, "IC_lower > breakeven + CLV_median > 0"),
    ("TOCAR λ_ITF  (recalibrar aversión ITF)",
     "Cambiar λ_ITF=4.5→3.6",
     "H52-02", 30, "epoch-3 n≥15 ITF"),
    ("ACTIVAR D57-04  (compensación expiry bonus)",
     "Bonus 90-180d ×1.15 para ex-campeones",
     None, 30, "Fase-H Brier pendiente"),
]


def panel_decision(shadow: dict, hypotheses_json: dict, calibracion: dict) -> None:
    st.header("Panel 6 — DECISIÓN")
    st.caption("Semáforos en vivo. La disciplina deja de depender de la memoria.")

    hyps_by_id = {h['id']: h for h in shadow.get('hypotheses', [])}
    meta_hyps = hypotheses_json.get('hypotheses', {})
    epochs = calibracion.get('_meta', {}).get('calibration_epochs', {})
    epoch_3_n = (epochs.get('epoch_3', {}) or {}).get('n_observaciones', 0) or 0

    any_auth = False

    for action, desc, gating_id, n_stop, extra_note in _DECISION_TABLE:
        n = ic_lower = breakeven = clv_med = 0
        extra_gate = True

        if gating_id and gating_id in hyps_by_id:
            h = hyps_by_id[gating_id]
            n = h.get('n', 0)
            ic = h.get('ic', [0, 100])
            ic_lower = ic[0] if ic else 0
            breakeven = h.get('breakeven')
            clv_med = h.get('clv_median')
        else:
            for seg in shadow.get('segments', []):
                if seg.get('label') == 'status=APOSTAR':
                    n = seg.get('n', 0)
                    ic = seg.get('ic', [0, 100])
                    ic_lower = ic[0] if ic else 0
                    breakeven = seg.get('breakeven')
                    clv_med = seg.get('clv_median')
                    break

        if extra_note and "epoch-3" in (extra_note or ""):
            extra_gate = epoch_3_n >= 15

        status = _decision_status(n, n_stop, ic_lower, breakeven, clv_med, extra_gate)
        authorized = status == "AUTORIZADO"
        if authorized:
            any_auth = True

        border = _GREEN if authorized else _RED
        badge_txt = "AUTORIZADO" if authorized else "NO AUTORIZADO"
        badge_color = _GREEN if authorized else _RED

        st.markdown(
            f'<div style="background:{_CARD};border-left:4px solid {border};'
            f'border-radius:8px;padding:14px 18px;margin:8px 0;'
            f'display:flex;justify-content:space-between;align-items:center;">'
            f'<div>'
            f'<b style="color:{_ORANGE};font-size:1rem;">¿{action}?</b><br>'
            f'<span style="color:{_MUTED};font-size:0.82rem;">{desc}</span><br>'
            f'<span style="color:{_BLUE};font-size:0.8rem;">'
            f'{"Gate: " + gating_id + " — " if gating_id else "Gate: segmento APOSTAR — "}'
            f'n={n}/{n_stop}'
            f'{"  |  " + extra_note if extra_note else ""}'
            f'</span>'
            f'</div>'
            f'<div style="text-align:right;min-width:130px;">'
            f'{_badge(badge_txt, badge_color)}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if not any_auth:
        st.info(
            f"Ninguna acción autorizada. "
            f"Más cercano: n={shadow.get('graduation', {}).get('nearest_n', 0)}/30. "
            "Seguir acumulando."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 7 — MOTOR AGÉNTICO (D59-08)
# ══════════════════════════════════════════════════════════════════════════════

def panel_odometro(odo: dict) -> None:
    st.header("Panel 7 — MOTOR AGÉNTICO")
    st.caption("Costo de tokens Claude Code · ROI ledger · Dream M2 · READ-ONLY")

    if not odo:
        st.warning("No se encontraron datos del odómetro. "
                   "Verificar que existen archivos JSONL en ~/.claude/projects/")
        return

    totals = odo.get('totals', {})
    total_cost = totals.get('cost', 0.0)
    total_in = totals.get('input', 0)
    cache_r = totals.get('cache_r', 0)
    cache_c = totals.get('cache_c', 0)
    eligible = total_in + cache_r + cache_c
    cache_pct = (cache_r / eligible * 100) if eligible > 0 else 0.0

    # ── Métricas principales ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Costo Total", f"${total_cost:,.2f}")
    c2.metric("Cache Hit Rate", f"{cache_pct:.1f}%",
              delta="objetivo >90%" if cache_pct >= 90 else "objetivo >90%",
              delta_color="normal" if cache_pct >= 90 else "inverse")
    c3.metric("Sesiones", str(odo.get('n_sessions', 0)))
    c4.metric("Dias activos", str(odo.get('n_days', 0)))

    st.divider()

    # ── Por modelo ─────────────────────────────────────────────────────────────
    st.subheader("Costo por Modelo")
    models = odo.get('models', {})
    if models:
        rows = []
        for model, ma in sorted(models.items(), key=lambda x: -x[1]['cost']):
            label = model.replace('claude-', '').split('-')[0].capitalize() if model else '?'
            pct = ma['cost'] / total_cost * 100 if total_cost else 0
            rows.append({
                'Modelo': label,
                'Costo USD': f"${ma['cost']:,.2f}",
                '% del total': f"{pct:.1f}%",
                'Tokens in (M)': f"{ma['input']/1e6:.2f}",
                'Tokens out (M)': f"{ma['output']/1e6:.2f}",
                'Cache read (M)': f"{ma['cache_r']/1e6:.2f}",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

        # Barras visuales de costo por modelo
        for model, ma in sorted(models.items(), key=lambda x: -x[1]['cost']):
            label = model.replace('claude-', '').split('-')[0].capitalize() if model else '?'
            pct = ma['cost'] / total_cost if total_cost else 0
            bar_color = _ORANGE if 'opus' in model else (_BLUE if 'sonnet' in model else _TEAL)
            st.markdown(
                f'<div style="margin:4px 0;">'
                f'<span style="color:{_MUTED};width:80px;display:inline-block;">{label}</span>'
                f'<div style="display:inline-block;background:{bar_color};'
                f'width:{int(pct*300)}px;height:16px;border-radius:3px;vertical-align:middle;"></div>'
                f'<span style="color:{_TEXT};margin-left:8px;">${ma["cost"]:,.2f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.divider()

    # ── Por tag ─────────────────────────────────────────────────────────────
    col_tag, col_top = st.columns([1, 1])

    with col_tag:
        st.subheader("Costo por Tag")
        tags = odo.get('tags', {})
        if tags:
            untagged_cost = tags.get('untagged', {}).get('cost', 0.0)
            untagged_pct = untagged_cost / total_cost * 100 if total_cost else 0
            for tag, ta in sorted(tags.items(), key=lambda x: -x[1]['cost']):
                pct = ta['cost'] / total_cost * 100 if total_cost else 0
                color = _RED if tag == 'untagged' else _GREEN
                _card(
                    f'<b style="color:{color};">[{tag}]</b> '
                    f'<span style="color:{_TEXT};">${ta["cost"]:,.2f}</span> '
                    f'<span style="color:{_MUTED};">({pct:.1f}%)</span>',
                    border_color=color,
                )
            if untagged_pct > 20:
                st.error(f"%%untagged = {untagged_pct:.0f}%% > 20%% — usar `# TAG: impl/test/audit` al inicio de cada sesión")
            else:
                st.success(f"%%untagged = {untagged_pct:.0f}%% ≤ 20%% OK")

    with col_top:
        st.subheader("Top 5 Sesiones mas Costosas")
        sessions = odo.get('sessions', [])
        top5 = sorted(sessions, key=lambda x: -x['cost'])[:5]
        for i, s in enumerate(top5, 1):
            ts_str = s['ts_first'].strftime('%Y-%m-%d') if s['ts_first'] else '?'
            short_id = (s['session_id'] or '')[:8]
            model_short = (s['model'] or '?').replace('claude-', '').split('-')[0]
            tag = s.get('tag', 'untagged')
            tag_color = _RED if tag == 'untagged' else _TEAL
            _card(
                f'<b style="color:{_ORANGE};">#{i}</b> '
                f'<code style="color:{_MUTED};">{short_id}</code> '
                f'<span style="color:{_TEXT};">{ts_str}</span> · '
                f'<span style="color:{_BLUE};">{model_short}</span> · '
                f'{_badge(tag, tag_color)} '
                f'<b style="color:{_ORANGE};float:right;">${s["cost"]:,.2f}</b>',
            )

    st.divider()

    # ── ROI Ledger ─────────────────────────────────────────────────────────────
    st.subheader("ROI Ledger — Semana 1 (2026-06-03 → 2026-07-03)")
    r1, r2, r3 = st.columns(3)
    r1.metric("Costo tokens", "$1,292.27", delta="Sonnet $609 + Opus $435 + Haiku $206")
    r2.metric("Nodos completados", "~59", delta="1,649 tests acumulados")
    r3.metric("Cache hit rate", "95.7%", delta="sesion mas cara $189.69 Opus")
    _card(
        '<b style="color:#888;">P&L apuestas:</b> separado del costo IA — ver shadow_book --report<br>'
        '<b style="color:#888;">Semana 2:</b> python3 token_odometer.py --report --desde 2026-07-07',
    )

    st.divider()

    # ── Dream M2 ────────────────────────────────────────────────────────────────
    st.subheader("Dream M2 — Skills Candidatos")
    dream_path = os.path.join(os.path.dirname(__file__), 'docs', 'dream-candidates.md')
    if os.path.exists(dream_path):
        with open(dream_path, encoding='utf-8') as f:
            content = f.read()
        n_cands = content.count('### Candidato')
        if n_cands:
            st.info(f"{n_cands} candidatos detectados — revisar docs/dream-candidates.md y aprobar manualmente")
        else:
            st.success("Sin candidatos por ahora — ninguna secuencia aparece en ≥3 sesiones")
        with st.expander("Ver dream-candidates.md"):
            st.code(content, language='markdown')
    else:
        st.caption("Ejecutar `python3 token_odometer.py --dream` para generar candidatos")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 8 — LIVE · Centro de Operaciones Triple Convergencia (Nodo-100)
# ══════════════════════════════════════════════════════════════════════════════

# Colores break state
_CLR_BREAK = {
    'NORMAL':           _MUTED,
    'BREAK_POSIBLE':    _ORANGE,
    'BREAK_CONFIRMADO': _RED,
}
_LBL_BREAK = {
    'NORMAL':           'NORMAL',
    'BREAK_POSIBLE':    'QUIEBRE POSIBLE',
    'BREAK_CONFIRMADO': 'QUIEBRE CONFIRMADO',
}


def _c1_score(senales_live: list, drift_pct: float) -> tuple[int, str, str]:
    """
    Triple Convergencia C1 (Nodo-99 D99-12):
      - STRONG en señales live (confianza modelo alta)
      - IRP_negativo (rival frío, vuelve de inactividad)
      - drift >= 12% sostenido (mercado confirma)
    Retorna (n_cumplidas, label, color).
    """
    has_strong  = 'STRONG' in senales_live
    has_irp_neg = 'IRP_negativo' in senales_live
    has_drift   = abs(drift_pct) >= 12.0
    n = sum([has_strong, has_irp_neg, has_drift])
    if n == 3:
        return 3, 'C1 MAXIMO', _RED
    if n == 2:
        return 2, f'PARCIAL {2}/3', _ORANGE
    if n == 1:
        return 1, f'INICIO {1}/3', _BLUE
    return 0, '—', _MUTED


def panel_live() -> None:
    st.header("Panel 8 — LIVE · Centro de Operaciones")
    st.caption(
        "Puente completo: edge_calculator → meta_signal (score pre-game) → "
        "live_edge_monitor (drift) → break_machine (confirmación) → auto-combo. "
        "Nodo-97 × Nodo-98 × Nodo-99 × Nodo-100."
    )

    # ── Cargar todas las fuentes ──────────────────────────────────────────────
    live           = load_live_edge()
    hist           = load_live_odds_history()
    meta           = load_meta_signal()
    sessions_today = load_live_sessions_today()

    if not live:
        st.warning(
            "Sin datos live. Activar:\n\n"
            "`python3 scripts/live_edge_monitor.py --observe --dashboard`\n\n"
            "o verificar n8n en :5678 → workflow 'Tennis Live Check'."
        )
        return

    ts_snap      = (live.get('ts') or '')[:19]
    observe_only = live.get('observe_only', True)
    stake_ok     = live.get('stake_permitido', True)
    kgr_neg      = live.get('kgr_negativo', False)
    n_mon        = live.get('picks_monitoreados', 0)
    n_trig       = live.get('n_triggers', 0)
    n_break      = live.get('break_confirmados', 0)
    picks        = live.get('picks_chequeados_data', [])

    # Índice meta_signal por partido (para unir pre-game con live)
    meta_idx = {}
    for p in meta.get('todos_los_picks', []):
        key = (p.get('partido') or '').strip()
        if key:
            meta_idx[key] = p

    # ── Barra de estado del sistema ───────────────────────────────────────────
    modo_color   = _MUTED if observe_only else _GREEN
    modo_txt     = 'OBSERVACION' if observe_only else 'ACTIVO'
    stake_color  = _GREEN if stake_ok else _RED
    stake_txt    = 'OK' if stake_ok else 'BLOQUEADO'
    kgr_html     = f'<span style="color:{_RED};font-size:0.82rem;"><b>KGR &lt; 0</b></span>' if kgr_neg else ''
    meta_color   = _GREEN if meta else _MUTED
    meta_txt     = ('OK — ' + meta.get('edge_report_fuente', '?')[:18]) if meta else 'sin datos'
    st.markdown(
        f'<div style="background:{_CARD};border:1px solid {_ORANGE}33;border-radius:8px;'
        f'padding:10px 18px;margin-bottom:14px;display:flex;gap:28px;flex-wrap:wrap;'
        f'align-items:center;">'
        f'<span style="color:{_MUTED};font-size:0.82rem;">Ciclo: '
        f'<b style="color:{_TEAL};">{ts_snap}</b></span>'
        f'<span style="color:{_MUTED};font-size:0.82rem;">Modo: '
        f'<b style="color:{modo_color};">{modo_txt}</b></span>'
        f'<span style="color:{_MUTED};font-size:0.82rem;">Stake: '
        f'<b style="color:{stake_color};">{stake_txt}</b></span>'
        f'{kgr_html}'
        f'<span style="color:{_MUTED};font-size:0.82rem;">Ciclos hoy: '
        f'<b style="color:{_BLUE};">{len(sessions_today)}</b></span>'
        f'<span style="color:{_MUTED};font-size:0.82rem;">Meta-señal: '
        f'<b style="color:{meta_color};">{meta_txt}</b></span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Banner BREAK CONFIRMADO ───────────────────────────────────────────────
    if n_break > 0:
        st.markdown(
            f'<div style="background:{_RED}22;border:2px solid {_RED};border-radius:8px;'
            f'padding:14px;margin-bottom:14px;text-align:center;font-size:1.05rem;'
            f'font-weight:700;color:{_RED};">'
            f'BREAK CONFIRMADO &mdash; COMBOS LIVE DISPARADOS</div>',
            unsafe_allow_html=True,
        )

    # ══ A: KPIs globales ═════════════════════════════════════════════════════
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Picks monitoreados", n_mon)
    k2.metric("Triggers activos", n_trig,
              delta="alerta" if n_trig > 0 else None,
              delta_color="inverse" if n_trig > 0 else "off")
    k3.metric("Breaks confirmados", n_break,
              delta="ACCION" if n_break > 0 else None,
              delta_color="inverse" if n_break > 0 else "off")
    n_score3 = meta.get('n_score_directo_3plus', 0)
    k4.metric("Picks score≥3 hoy", n_score3,
              delta="candidatos C1" if n_score3 > 0 else None)
    n_fired = sum(1 for d in hist.values() if d.get('fired', False))
    k5.metric("H100-01 breaks acum.", f"{n_fired}/20",
              delta="gate≥3" if n_fired < 3 else "gate ABIERTO",
              delta_color="inverse" if n_fired < 3 else "normal")

    st.divider()

    # ══ B: TRIPLE CONVERGENCIA C1 — tabla central ════════════════════════════
    st.subheader("Triple Convergencia C1 — por partido")
    st.caption(
        "C1 = STRONG (confianza modelo) + IRP_negativo (rival frío) + drift≥12% (mercado confirma). "
        "Los 3 juntos = alpha_tier MAXIMO (Nodo-99 D99-12)."
    )

    valid_picks = [pc for pc in picks if pc.get('cuota_live') not in (None, 0, '?')]

    if not valid_picks:
        st.info("Sin picks con cuota live disponible — n8n corriendo cada 2 min.")
    else:
        hdr = st.columns([2.8, 0.8, 2.2, 0.8, 0.9, 0.9, 2.0, 2.2])
        for col, h in zip(hdr, ["Partido", "Score", "Señales live", "Cuota Pre",
                                  "Cuota Live", "Drift%", "Break State", "C1 Convergencia"]):
            col.markdown(
                f'<span style="color:{_BLUE};font-weight:700;font-size:0.75rem;">{h}</span>',
                unsafe_allow_html=True)
        st.markdown(f'<hr style="margin:3px 0;border-color:#333;">', unsafe_allow_html=True)

        for pc in valid_picks:
            partido    = pc.get('partido', '?')
            estado     = pc.get('break_state', 'NORMAL')
            b_color    = _CLR_BREAK.get(estado, _MUTED)
            b_label    = _LBL_BREAK.get(estado, estado)
            drift_pct  = pc.get('drift_pct', 0.0) or 0.0
            cuota_pre  = pc.get('cuota_pre', 0.0) or 0.0
            cuota_live = pc.get('cuota_live', '?')
            senales    = pc.get('senales', [])

            # Score pre-game desde meta_signal
            mp      = meta_idx.get(partido, {})
            score_d = mp.get('score_directo')
            score_str = f"{score_d}/5" if score_d is not None else '—'

            # C1
            n_c1, c1_label, c1_color = _c1_score(senales, drift_pct)
            if n_c1 == 3 and estado == 'BREAK_CONFIRMADO':
                c1_color = _RED  # máximo + confirmado

            senales_str = ' '.join(f'[{s}]' for s in senales) or '—'
            drift_color = (_RED if abs(drift_pct) >= 15
                           else (_ORANGE if abs(drift_pct) >= 10 else _MUTED))

            row = st.columns([2.8, 0.8, 2.2, 0.8, 0.9, 0.9, 2.0, 2.2])
            row[0].markdown(
                f'<span style="color:{b_color};font-weight:700;">{partido}</span>',
                unsafe_allow_html=True)
            row[1].markdown(
                f'<span style="color:{_TEAL};font-weight:700;">{score_str}</span>',
                unsafe_allow_html=True)
            row[2].markdown(
                f'<span style="color:{_MUTED};font-size:0.75rem;">{senales_str}</span>',
                unsafe_allow_html=True)
            row[3].markdown(
                f'<span style="color:{_MUTED};">{cuota_pre:.2f}</span>',
                unsafe_allow_html=True)
            row[4].markdown(
                f'<span style="color:{_TEXT};">{cuota_live}</span>',
                unsafe_allow_html=True)
            row[5].markdown(
                f'<span style="color:{drift_color};font-weight:700;">{drift_pct:+.1f}%</span>',
                unsafe_allow_html=True)
            row[6].markdown(
                f'<span style="color:{b_color};font-weight:600;font-size:0.78rem;">{b_label}</span>',
                unsafe_allow_html=True)
            row[7].markdown(
                f'<span style="color:{c1_color};font-weight:700;">{c1_label}</span>',
                unsafe_allow_html=True)

    st.divider()

    # ══ C: META-SEÑAL PRE-GAME (inteligencia del día) ════════════════════════
    st.subheader("Meta-Señal Pre-Game — score_directo del día (Nodo-98)")
    st.caption(
        "HOT + STRONG + ELO_DOM + RFI_tier + IRP_delta = hasta 5 puntos. "
        "Score≥3 = candidato C1. SPLIT = conflicto, NO combinar."
    )

    if not meta:
        st.info("Sin meta_signal — correr: `python3 scripts/meta_signal_scorer.py`")
    else:
        n_split = meta.get('n_split', 0)

        destacados = meta.get('picks_score_directo_3plus', [])
        splits     = meta.get('picks_split', [])

        if destacados:
            with st.expander(
                f"Picks score_directo≥3 — {len(destacados)} candidatos Triple Convergencia",
                expanded=True,
            ):
                rows = []
                for p in destacados:
                    partido_p = p.get('partido', '?')
                    # Enriquecer con estado live actual si existe
                    live_pc = next(
                        (pc for pc in picks if pc.get('partido') == partido_p), {}
                    )
                    drift_live = live_pc.get('drift_pct')
                    b_state    = live_pc.get('break_state', '—')
                    rows.append({
                        'Partido':      partido_p,
                        'Score':        p.get('score_directo', 0),
                        'Dirección':    p.get('direccion', '?'),
                        'Señales':      ', '.join(p.get('senales_activas_fav', [])),
                        'Edge pre%':    f"{(p.get('edge') or 0)*100:+.1f}%" if p.get('edge') is not None else '—',
                        'Cuota pre':    f"{p.get('cuota') or 0:.2f}",
                        'Drift live':   f'{drift_live:+.1f}%' if drift_live is not None else 'sin dato',
                        'Break state':  b_state,
                        'Sección':      p.get('seccion', '?'),
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("Ningún pick con score_directo≥3 hoy.")

        if splits:
            with st.expander(
                f"Picks SPLIT — conflicto señales ({len(splits)}) — NO apostar en combo mixto"
            ):
                st.warning(
                    "SPLIT = señal pro-favorito activa Y señal pro-rival activa simultáneamente. "
                    "No incluir en combo mixto (Nodo-99 D99-03)."
                )
                rows_s = []
                for p in splits:
                    rows_s.append({
                        'Partido':        p.get('partido', '?'),
                        'Score directo':  p.get('score_directo', 0),
                        'Score rival':    p.get('score_rival_value', 0),
                        'Señales fav':    ', '.join(p.get('senales_activas_fav', [])),
                        'Señales rival':  ', '.join(p.get('senales_activas_rival', [])),
                    })
                st.dataframe(rows_s, use_container_width=True, hide_index=True)

    st.divider()

    # ══ D: EVOLUCIÓN DRIFT (multi-ciclo) ═════════════════════════════════════
    if len(sessions_today) > 1:
        st.subheader(f"Evolución drift — {len(sessions_today)} ciclos hoy")
        st.caption(
            "Fluctuación normal: cuota baja y RECUPERA. "
            "Quiebre real: cuota baja y SE MANTIENE o sigue bajando."
        )
        trend_rows = []
        for sess in sessions_today[-12:]:
            ts_s = (sess.get('ts') or '')[11:16]  # HH:MM
            for pc in sess.get('picks_chequeados_data', []):
                dp = pc.get('drift_pct')
                if dp is not None:
                    trend_rows.append({
                        'Hora':       ts_s,
                        'Partido':    pc.get('partido', '?'),
                        'Drift%':     f'{dp:+.1f}%',
                        'Cuota Live': pc.get('cuota_live', '?'),
                        'Estado':     pc.get('break_state', 'NORMAL'),
                    })
        if trend_rows:
            st.dataframe(trend_rows, use_container_width=True, hide_index=True)

    st.divider()

    # ══ E: BREAK MACHINE HISTORY ══════════════════════════════════════════════
    if hist:
        with st.expander("Break State Machine — historial de lecturas por partido"):
            for partido_key, data in hist.items():
                estado   = data.get('estado', 'NORMAL')
                b_color  = _CLR_BREAK.get(estado, _MUTED)
                fired    = data.get('fired', False)
                readings = data.get('readings', [])
                st.markdown(
                    f'<b style="color:{b_color};">{partido_key.replace("_", " ")}</b>'
                    f' — <span style="color:{b_color};">{_LBL_BREAK.get(estado, estado)}</span>'
                    + (f' <span style="color:{_GREEN};">[FIRED ✓]</span>' if fired else ''),
                    unsafe_allow_html=True,
                )
                if readings:
                    st.dataframe(
                        [{"ts": r.get("ts", "?"),
                          "cuota": r.get("cuota", 0),
                          "drift%": f'{r.get("drift", 0)*100:+.1f}%'}
                         for r in readings],
                        use_container_width=False,
                        hide_index=True,
                    )

    # ══ F: H100-01 — GATE Y EVIDENCIA ════════════════════════════════════════
    st.divider()
    st.subheader("H100-01 — Gate Triple Convergencia")
    st.caption(
        "Hipótesis: picks con score_directo≥2 + break confirmado producen ROI > picks normales. "
        "n_stop=20 | Gate activación: ≥3 breaks confirmados reales."
    )

    g1, g2, g3 = st.columns(3)
    g1.metric("Breaks fired hoy", n_fired)
    g2.metric("Gate (≥3)", "ABIERTO" if n_fired >= 3 else f"faltan {3 - n_fired}",
              delta=None if n_fired >= 3 else f"{3 - n_fired} más",
              delta_color="off" if n_fired >= 3 else "inverse")
    g3.metric("Progreso n_stop", f"{n_fired}/20")

    if n_fired < 3:
        st.info(
            f"En observación — {3 - n_fired} break(s) confirmado(s) más para activar "
            "acumulación formal de H100-01."
        )
    else:
        st.success(
            f"{n_fired} breaks confirmados — evidencia formal activa. "
            f"Acumular hasta n=20 para graduación."
        )

    # ══ G: COMBOS DISPARADOS ═════════════════════════════════════════════════
    combo_out = live.get('combo_break_output')
    combos_hoy = [s.get('combo_break_output') for s in sessions_today
                  if s.get('combo_break_output')]

    if combo_out or combos_hoy:
        st.divider()
        st.subheader("Combos disparados por break confirmado")
        if combo_out:
            _card(
                f'<b style="color:{_GREEN};">Último combo:</b><br>'
                f'<code style="color:{_GREEN};">{combo_out}</code>'
            )
        if len(combos_hoy) > 1:
            with st.expander(f"Historial combos hoy ({len(combos_hoy)})"):
                for c in combos_hoy:
                    st.markdown(f'`{c}`')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title="Tennis Analysis Dashboard",
        page_icon="T",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    # ── Title ──
    st.markdown(
        f'<h1 style="color:{_ORANGE};letter-spacing:2px;margin-bottom:0;">'
        f'TENNIS ANALYSIS</h1>'
        f'<p style="color:{_MUTED};margin-top:0;font-size:0.85rem;">'
        f'Nodo-58 · Dashboard de Observabilidad · '
        f'{datetime.now().strftime("%Y-%m-%d %H:%M")} · READ-ONLY</p>',
        unsafe_allow_html=True,
    )

    # ── Sidebar ──
    with st.sidebar:
        st.markdown(f'<h2 style="color:{_ORANGE};">FILTROS</h2>', unsafe_allow_html=True)
        today = date.today()
        desde = st.date_input("Desde", value=today.replace(day=1))
        hasta = st.date_input("Hasta", value=today)
        st.divider()
        if st.button("Refrescar"):
            st.cache_data.clear()
            st.rerun()
        st.markdown(
            f'<p style="color:{_MUTED};font-size:0.75rem;margin-top:20px;">'
            f'Para actualizar datos:<br>'
            f'<code>python3 shadow_book.py --settle FECHA</code></p>',
            unsafe_allow_html=True,
        )

    desde_str = str(desde)
    hasta_str = str(hasta)

    @st.cache_data(ttl=300)
    def _load(d, h):
        return (
            load_shadow_report(desde=d, hasta=h),
            load_edge_report(),
            load_trader_plan(),
            load_calibracion(),
            load_hypotheses(),
            load_h2h_report(),
        )

    shadow, edge, trader, calibracion, hypotheses_json, h2h = _load(desde_str, hasta_str)

    @st.cache_data(ttl=600)
    def _load_odo():
        return load_odometer()

    odo = _load_odo()

    # ── Tabs ──
    tabs = st.tabs(["HOY", "HIPÓTESIS", "SALUD", "ATRIBUCIÓN", "RIESGO", "DECISIÓN", "MOTOR", "LIVE"])

    with tabs[0]:
        panel_hoy(shadow, trader, edge, calibracion)
    with tabs[1]:
        panel_hipotesis(shadow, hypotheses_json)
    with tabs[2]:
        panel_salud(shadow, desde_str, hasta_str, calibracion)
    with tabs[3]:
        panel_atribucion(h2h, shadow)
    with tabs[4]:
        panel_riesgo(trader, shadow)
    with tabs[5]:
        panel_decision(shadow, hypotheses_json, calibracion)
    with tabs[6]:
        panel_odometro(odo)
    with tabs[7]:
        panel_live()


if __name__ == "__main__":
    main()
