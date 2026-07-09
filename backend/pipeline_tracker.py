#!/usr/bin/env python3
"""
pipeline_tracker.py — Nodo-27: Pipeline Tracker & Observabilidad
READ-ONLY: No modifica ningún archivo de datos.

Uso:
  python3 pipeline_tracker.py                        # todo el histórico
  python3 pipeline_tracker.py --since 2026-06-01     # filtrar por fecha
  python3 pipeline_tracker.py --tier challenger       # filtrar por tier
  python3 pipeline_tracker.py --section confianza     # solo una sección

Fases implementadas:
  Fase 1: S-27-1 (confidence_flag), S-27-2 (cuotas), S-27-3 (tier+superficie)
  Fase 2: S-27-4 (señales: golden_zone, markov, data_completeness, zona_cuota, edge bins)
           S-27-5 (calibración del modelo)
  Fase 3: S-27-6 (evolución temporal por semana), S-27-7 (portfolio: combos vs individuales)
  Fase 4: S-27-8 (shadow book CLV — Nodo-52, --section shadow)
"""

import argparse
import glob
import json
import sys
from datetime import datetime, date
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ── Constantes ────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
REPORTS_DIR = BASE_DIR / "reports"
OUTPUT_FILE = BASE_DIR / "pipeline_tracking.txt"

CUOTA_BINS = [
    (1.50, 2.00, "1.50-2.00"),
    (2.00, 2.50, "2.00-2.50"),
    (2.50, 3.00, "2.50-3.00"),
    (3.00, 4.00, "3.00-4.00"),
    (4.00, 99.0, "4.00+"),
]

EDGE_BINS = [
    (5,  10, " 5-10%"),
    (10, 15, "10-15%"),
    (15, 20, "15-20%"),
    (20, 99, "20%+"),
]

CONF_ORDER = ["STRONG", "MODERATE", "LOW"]
MARKOV_ORDER = ["HOT", "NEUTRAL", "COLD"]
ZONA_ORDER = ["underdog", "slight_underdog", "moderate_favorite"]

# ── Utilidades de display ─────────────────────────────────────────────────────

def _pct(v):
    if v is None:
        return "  N/A "
    return f"{v:5.1f}%"


def _flag(n, threshold=10):
    """Añade * si muestra insuficiente (REGLA-T27-4)."""
    return f"{n}*" if n < threshold else str(n)


def _roi_str(roi):
    if roi is None:
        return "  N/A "
    sign = "+" if roi >= 0 else ""
    return f"{sign}{roi:5.1f}%"


def _row(cols, widths):
    parts = []
    for c, w in zip(cols, widths):
        parts.append(str(c).ljust(w)[:w])
    return "  ".join(parts)


def _header(title):
    line = "=" * 72
    return f"\n{line}\n  {title}\n{line}"


def _subheader(title):
    return f"\n  --- {title} ---"


# ── Carga de datos ────────────────────────────────────────────────────────────

def _fecha_from_filename(filepath):
    """Extrae fecha YYYY-MM-DD del nombre de archivo *_YYYYMMDD_HHMMSS.json."""
    stem = Path(filepath).stem
    parts = stem.rsplit("_", 2)
    if len(parts) >= 3:
        fecha_str = parts[-2]
        if len(fecha_str) == 8 and fecha_str.isdigit():
            try:
                return datetime.strptime(fecha_str, "%Y%m%d").date()
            except ValueError:
                pass
    return None


def _parse_edge_pct(val):
    """Convierte '18.8%' o 0.188 a float 18.8."""
    if val is None:
        return None
    if isinstance(val, str):
        return float(val.replace("%", "").strip())
    if isinstance(val, float) and val < 1.0:
        return val * 100
    return float(val)


def cargar_edge_reports(since=None, tier_filter=None):
    """Carga todos los edge_report_*.json → lista de picks (apostar + watchlist)."""
    picks = []
    files = sorted(glob.glob(str(REPORTS_DIR / "edge_report_*.json")))
    for filepath in files:
        fecha = _fecha_from_filename(filepath)
        if since and fecha and fecha < since:
            continue
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        for lista_key in ("apostar", "watchlist"):
            for p in data.get(lista_key, []):
                t = p.get("tier", "?")
                if tier_filter and t != tier_filter:
                    continue
                picks.append({
                    "fecha": fecha,
                    "fuente_lista": lista_key,
                    "match_id": p.get("match_id"),
                    "favorito": p.get("favorito_predicho"),
                    "partido": p.get("partido"),
                    "cuota": p.get("cuota_favorito"),
                    "edge_pct": _parse_edge_pct(p.get("edge_pct")),
                    "p_modelo": p.get("p_modelo"),
                    "p_blend": p.get("p_blend"),
                    "confidence_flag": p.get("confidence_flag"),
                    "tier": t,
                    "superficie": p.get("superficie", "?"),
                    "zona_cuota": p.get("zona_cuota"),
                    "golden_zone": p.get("golden_zone"),
                    "bbi": p.get("bbi"),
                    "markov_favorito": p.get("markov_favorito"),
                    "data_completeness": p.get("data_completeness"),
                    "kelly_kl": p.get("kelly_kl"),
                    "n_h2h": p.get("n_h2h"),
                    # resultado: se rellena en join
                    "correcto": None,
                    "ganancia": None,
                    "stake": 0,
                })
    return picks


def cargar_resultados_finales(since=None):
    """Carga resultados_finales_*.json → dict match_id → correcto."""
    resultados = {}
    files = sorted(glob.glob(str(REPORTS_DIR / "resultados_finales_*.json")))
    for filepath in files:
        fecha = _fecha_from_filename(filepath)
        if since and fecha and fecha < since:
            continue
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for p in data.get("partidos", []):
            mid = p.get("match_id")
            if mid:
                resultados[mid] = {
                    "correcto": p.get("correcto"),
                    "prediccion": p.get("prediccion"),
                    "superficie": p.get("superficie"),
                    "confianza": p.get("confianza"),
                    "fecha": fecha,
                }
    return resultados


def cargar_apuestas(since=None, tier_filter=None):
    """Carga apuestas_*.json → dict (match_id, jugador) → resultado."""
    apuestas = {}
    files = sorted(glob.glob(str(REPORTS_DIR / "apuestas_*.json")))
    for filepath in files:
        fecha = _fecha_from_filename(filepath)
        if since and fecha and fecha < since:
            continue
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        ts = data.get("ts_registro", "")
        fecha_ap = None
        if ts:
            try:
                fecha_ap = datetime.fromisoformat(ts[:10]).date()
            except ValueError:
                pass
        for p in data.get("picks", []):
            t = p.get("tier", "?")
            if tier_filter and t not in (tier_filter, "?"):
                continue
            mid = p.get("match_id")
            jugador = p.get("jugador")
            correcto = p.get("correcto")
            ganancia = p.get("ganancia", 0) or 0
            stake = p.get("stake", 0) or 0
            if mid:
                key = (mid, jugador)
                apuestas[key] = {
                    "correcto": correcto,
                    "ganancia": ganancia,
                    "stake": stake,
                    "cuota": p.get("cuota"),
                    "edge_pct": _parse_edge_pct(p.get("edge")),
                    "p_modelo": p.get("p_modelo"),
                    "tier": t,
                    "superficie": p.get("superficie", "?"),
                    "fecha": fecha_ap or fecha,
                }
    return apuestas


def join_resultados(picks, resultados_map, apuestas_map):
    """
    Enriquece picks con correcto/ganancia/stake desde:
    1. resultados_finales (por match_id)
    2. apuestas (por match_id + jugador)
    Retorna picks enriquecidos.
    """
    enriched = []
    for p in picks:
        mid = p.get("match_id")
        fav = p.get("favorito")
        p2 = dict(p)

        # Intento 1: resultados_finales
        if mid and mid in resultados_map:
            r = resultados_map[mid]
            # Solo asignar si el predicho coincide con la prediccion guardada
            pred = r.get("prediccion", "")
            if pred and fav and (pred.lower() in fav.lower() or fav.lower() in pred.lower()):
                p2["correcto"] = r["correcto"]
                p2["ganancia"] = 0
                p2["stake"] = 0
                enriched.append(p2)
                continue
            # Si no hay prediccion guardada, igual asignar (solo 1 pick por partido)
            if not pred:
                p2["correcto"] = r["correcto"]
                p2["ganancia"] = 0
                p2["stake"] = 0
                enriched.append(p2)
                continue

        # Intento 2: apuestas (match_id + jugador exact)
        if mid and fav:
            key = (mid, fav)
            if key in apuestas_map:
                a = apuestas_map[key]
                p2["correcto"] = a["correcto"]
                p2["ganancia"] = a["ganancia"]
                p2["stake"] = a["stake"]
                enriched.append(p2)
                continue

        # Sin resultado — incluir igual (correcto=None)
        enriched.append(p2)

    return enriched


def cargar_trader_combos(since=None):
    """Carga trader_plan_*.json → lista de combos con resultado."""
    combos = []
    files = sorted(glob.glob(str(REPORTS_DIR / "trader_plan_*.json")))
    for filepath in files:
        fecha = _fecha_from_filename(filepath)
        if since and fecha and fecha < since:
            continue
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        meta = data.get("metadata", {})
        for combo in data.get("cobertura", []):
            combos.append({
                "fecha": fecha,
                "piernas_n": combo.get("piernas_n", 0),
                "legs": combo.get("legs", []),
                "cuota_combo": combo.get("cuota_combo"),
                "hr_combo": combo.get("hr_combo"),
                "ev_combo": combo.get("ev_combo"),
                "stake": combo.get("stake", 0) or 0,
                "retorno_potencial": combo.get("retorno_potencial", 0),
                "excluidos": combo.get("excluidos", []),
                # resultado: no disponible en trader_plan
                "correcto": None,
            })
    return combos


# ── Estadísticas ──────────────────────────────────────────────────────────────

def _stats(rows, cuota_key="cuota"):
    """
    Calcula n, wins, losses, pendientes, hit%, ROI% para una lista de picks.
    ROI: si stake=0 en todos → proxy (cuota-1)*correcto - (1-correcto) por unidad.
    Si hay stake real → (sum ganancia / sum stake) * 100.
    """
    n = len(rows)
    wins = sum(1 for r in rows if r.get("correcto") is True)
    losses = sum(1 for r in rows if r.get("correcto") is False)
    pending = sum(1 for r in rows if r.get("correcto") is None)

    hit = (wins / (wins + losses) * 100) if (wins + losses) > 0 else None

    # ROI
    total_stake = sum(r.get("stake", 0) or 0 for r in rows)
    if total_stake > 0:
        total_gan = sum(r.get("ganancia", 0) or 0 for r in rows)
        roi = (total_gan / total_stake) * 100
    else:
        # Proxy ROI: unidad = 1, cuota variable
        roi_vals = []
        for r in rows:
            if r.get("correcto") is None:
                continue
            cuota = r.get(cuota_key) or r.get("cuota") or 2.0
            if r["correcto"]:
                roi_vals.append(cuota - 1)  # ganancia neta en unidades
            else:
                roi_vals.append(-1.0)       # pérdida
        roi = (sum(roi_vals) / len(roi_vals) * 100) if roi_vals else None

    avg_edge = None
    edges = [r.get("edge_pct") for r in rows if r.get("edge_pct") is not None]
    if edges:
        avg_edge = sum(edges) / len(edges)

    return {
        "n": n, "wins": wins, "losses": losses, "pending": pending,
        "hit": hit, "roi": roi, "avg_edge": avg_edge,
    }


def _with_resultado(rows):
    """Filtra solo picks con resultado conocido."""
    return [r for r in rows if r.get("correcto") is not None]


# ── Secciones ─────────────────────────────────────────────────────────────────

def seccion_27_1_confianza(picks, out):
    """S-27-1: Rendimiento por nivel de confianza."""
    out.append(_header("S-27-1  Rendimiento por Nivel de Confianza"))
    out.append("  Pregunta: confidence_flag predice correctamente?")

    cols = ["Flag", "N Total", "N Result", "Wins", "Losses", "Hit%", "ROI%(proxy)", "Avg Edge"]
    ws =   [10,     8,         9,          6,      7,         7,      13,             10]
    out.append("")
    out.append("  " + _row(cols, ws))
    out.append("  " + "-" * 75)

    for flag in CONF_ORDER:
        group = [p for p in picks if p.get("confidence_flag") == flag]
        with_r = _with_resultado(group)
        s = _stats(with_r)
        row = [
            flag,
            _flag(len(group)),
            _flag(len(with_r)),
            s["wins"],
            s["losses"],
            _pct(s["hit"]),
            _roi_str(s["roi"]),
            _pct(s["avg_edge"]),
        ]
        out.append("  " + _row(row, ws))

    # Sin flag (campos viejos)
    group_na = [p for p in picks if p.get("confidence_flag") not in CONF_ORDER]
    if group_na:
        with_r = _with_resultado(group_na)
        s = _stats(with_r)
        row = ["(sin flag)", _flag(len(group_na)), _flag(len(with_r)),
               s["wins"], s["losses"], _pct(s["hit"]), _roi_str(s["roi"]), _pct(s["avg_edge"])]
        out.append("  " + _row(row, ws))

    out.append("")
    out.append("  NOTA: ROI proxy = (cuota-1)*wins - losses / N_con_resultado (stake=0 paper trading)")
    out.append("  * = muestra insuficiente (n<10)")


def seccion_27_2_cuotas(picks, out):
    """S-27-2: Rendimiento por rango de cuota."""
    out.append(_header("S-27-2  Rendimiento por Rango de Cuota"))
    out.append("  Pregunta: donde esta el alpha real — underdogs altos o favoritos moderados?")

    cols = ["Rango Cuota", "N Total", "N Result", "Wins", "Losses", "Hit%", "ROI%(proxy)", "Avg Edge"]
    ws =   [12,            8,         9,          6,      7,         7,      13,             10]
    out.append("")
    out.append("  " + _row(cols, ws))
    out.append("  " + "-" * 80)

    for lo, hi, label in CUOTA_BINS:
        group = [p for p in picks if p.get("cuota") and lo <= p["cuota"] < hi]
        with_r = _with_resultado(group)
        s = _stats(with_r)
        row = [
            label,
            _flag(len(group)),
            _flag(len(with_r)),
            s["wins"],
            s["losses"],
            _pct(s["hit"]),
            _roi_str(s["roi"]),
            _pct(s["avg_edge"]),
        ]
        out.append("  " + _row(row, ws))

    out.append("")
    out.append("  Hipotesis: underdogs >=2.00 con edge >5% = alpha estructural (sesiones R4 + Epica)")


def seccion_27_3_tier_superficie(picks, out):
    """S-27-3: Rendimiento por tier y superficie."""
    out.append(_header("S-27-3  Rendimiento por Tier y Superficie"))
    out.append("  Pregunta: Challenger/ITF supera ATP500?")

    cols = ["Tier", "Superficie", "N Total", "N Result", "Wins", "Losses", "Hit%", "ROI%(proxy)", "Avg BBI"]
    ws =   [12,     10,           8,         9,          6,      7,         7,      13,             9]
    out.append("")
    out.append("  " + _row(cols, ws))
    out.append("  " + "-" * 90)

    # Agrupar por tier+superficie
    combos_ts = {}
    for p in picks:
        tier = p.get("tier") or "?"
        sup = p.get("superficie") or "?"
        key = (tier, sup)
        combos_ts.setdefault(key, []).append(p)

    # Ordenar: primero por tier (grand_slam, atp1000, atp500, challenger, itf, ?)
    tier_order = ["grand_slam", "atp1000", "atp500", "challenger", "itf", "?"]
    def sort_key(k):
        tier_idx = tier_order.index(k[0]) if k[0] in tier_order else 99
        return (tier_idx, k[1])

    for (tier, sup) in sorted(combos_ts.keys(), key=sort_key):
        group = combos_ts[(tier, sup)]
        with_r = _with_resultado(group)
        s = _stats(with_r)
        bbis = [p.get("bbi") for p in group if p.get("bbi") is not None]
        avg_bbi = sum(bbis) / len(bbis) if bbis else None
        row = [
            tier,
            sup,
            _flag(len(group)),
            _flag(len(with_r)),
            s["wins"],
            s["losses"],
            _pct(s["hit"]),
            _roi_str(s["roi"]),
            f"{avg_bbi:.3f}" if avg_bbi is not None else "  N/A",
        ]
        out.append("  " + _row(row, ws))

    out.append("")
    out.append("  Hipotesis: BBI alto (bookmaker ciego) → mejor ROI en Challenger/ITF")
    out.append("  Referencia calibracion_edge.json:")

    try:
        with open(BASE_DIR / "data" / "calibracion_edge.json", encoding="utf-8") as f:
            cal = json.load(f)
        g = cal.get("global", {})
        gw, gl = g.get("wins", 0), g.get("losses", 0)
        gn = gw + gl
        ghit = gw / gn * 100 if gn > 0 else 0
        out.append(f"    Global: {gw}W/{gl}L  n={gn}  hit={ghit:.1f}%")
        for key, vals in cal.get("por_superficie_y_tier", {}).items():
            w, l = vals.get("wins", 0), vals.get("losses", 0)
            n = w + l
            if n > 0:
                out.append(f"    {key}: {w}W/{l}L  n={n}  hit={w/n*100:.1f}%")
    except (OSError, json.JSONDecodeError):
        out.append("    (calibracion_edge.json no disponible)")


# ── Fase 2 ────────────────────────────────────────────────────────────────────

def seccion_27_4_senales(picks, out):
    """S-27-4: Rendimiento por señal específica."""
    out.append(_header("S-27-4  Rendimiento por Señal Especifica"))

    # 4a. Golden Zone
    out.append(_subheader("4a. Golden Zone"))
    cols = ["Golden", "N Total", "N Result", "Wins", "Losses", "Hit%", "ROI%(proxy)"]
    ws =   [8,        8,         9,          6,      7,         7,      13]
    out.append("  " + _row(cols, ws))
    out.append("  " + "-" * 65)
    for val, label in [(True, "True"), (False, "False")]:
        group = [p for p in picks if p.get("golden_zone") == val]
        with_r = _with_resultado(group)
        s = _stats(with_r)
        out.append("  " + _row([label, _flag(len(group)), _flag(len(with_r)),
                                 s["wins"], s["losses"], _pct(s["hit"]), _roi_str(s["roi"])], ws))

    # 4b. Markov
    out.append(_subheader("4b. Markov (estado de forma)"))
    cols = ["Markov", "N Total", "N Result", "Wins", "Losses", "Hit%", "ROI%(proxy)"]
    ws =   [9,        8,         9,          6,      7,         7,      13]
    out.append("  " + _row(cols, ws))
    out.append("  " + "-" * 65)
    all_markov = set(p.get("markov_favorito") for p in picks if p.get("markov_favorito"))
    order = [m for m in MARKOV_ORDER if m in all_markov] + \
            [m for m in all_markov if m not in MARKOV_ORDER]
    for mval in order:
        group = [p for p in picks if p.get("markov_favorito") == mval]
        with_r = _with_resultado(group)
        s = _stats(with_r)
        out.append("  " + _row([mval, _flag(len(group)), _flag(len(with_r)),
                                 s["wins"], s["losses"], _pct(s["hit"]), _roi_str(s["roi"])], ws))

    # 4c. Data Completeness
    out.append(_subheader("4c. Data Completeness"))
    dc_bins = [(0, 0.25, "0-25%"), (0.25, 0.50, "25-50%"),
               (0.50, 0.75, "50-75%"), (0.75, 1.01, "75-100%")]
    cols = ["Completeness", "N Total", "N Result", "Wins", "Losses", "Hit%"]
    ws =   [13,             8,         9,          6,      7,         7]
    out.append("  " + _row(cols, ws))
    out.append("  " + "-" * 57)
    for lo, hi, label in dc_bins:
        group = [p for p in picks
                 if p.get("data_completeness") is not None and lo <= p["data_completeness"] < hi]
        with_r = _with_resultado(group)
        s = _stats(with_r)
        out.append("  " + _row([label, _flag(len(group)), _flag(len(with_r)),
                                 s["wins"], s["losses"], _pct(s["hit"])], ws))

    # 4d. Zona de Cuota
    out.append(_subheader("4d. Zona de Cuota"))
    cols = ["Zona", "N Total", "N Result", "Wins", "Losses", "Hit%", "ROI%(proxy)"]
    ws =   [18,     8,         9,          6,      7,         7,      13]
    out.append("  " + _row(cols, ws))
    out.append("  " + "-" * 72)
    all_zonas = set(p.get("zona_cuota") for p in picks if p.get("zona_cuota"))
    order_z = [z for z in ZONA_ORDER if z in all_zonas] + \
               [z for z in all_zonas if z not in ZONA_ORDER]
    for zona in order_z:
        group = [p for p in picks if p.get("zona_cuota") == zona]
        with_r = _with_resultado(group)
        s = _stats(with_r)
        out.append("  " + _row([zona, _flag(len(group)), _flag(len(with_r)),
                                 s["wins"], s["losses"], _pct(s["hit"]), _roi_str(s["roi"])], ws))

    # 4e. Edge Binning
    out.append(_subheader("4e. Edge Binning"))
    cols = ["Edge%", "N Total", "N Result", "Wins", "Losses", "Hit%", "ROI%(proxy)", "Avg Kelly"]
    ws =   [7,       8,         9,          6,      7,         7,      13,             10]
    out.append("  " + _row(cols, ws))
    out.append("  " + "-" * 73)
    for lo, hi, label in EDGE_BINS:
        group = [p for p in picks
                 if p.get("edge_pct") is not None and lo <= p["edge_pct"] < hi]
        with_r = _with_resultado(group)
        s = _stats(with_r)
        kellys = [p.get("kelly_kl") for p in group if p.get("kelly_kl") is not None]
        avg_kelly = sum(kellys) / len(kellys) if kellys else None
        out.append("  " + _row([label, _flag(len(group)), _flag(len(with_r)),
                                 s["wins"], s["losses"], _pct(s["hit"]),
                                 _roi_str(s["roi"]),
                                 f"{avg_kelly:.4f}" if avg_kelly is not None else "  N/A"], ws))


def seccion_27_5_calibracion(picks, out):
    """S-27-5: Calibración del modelo."""
    out.append(_header("S-27-5  Calibracion del Modelo"))
    out.append("  Pregunta: cuando el modelo dice 60%, gana ~60%?")

    p_bins = [
        (0.50, 0.52, "0.50-0.52", 0.51),
        (0.52, 0.55, "0.52-0.55", 0.535),
        (0.55, 0.60, "0.55-0.60", 0.575),
        (0.60, 0.65, "0.60-0.65", 0.625),
        (0.65, 1.01, "0.65+",     0.675),
    ]

    cols = ["p_modelo Bin", "N Total", "N Result", "Actual Hit%", "Esperado", "Diff"]
    ws =   [13,             8,         9,           12,            9,          7]
    out.append("")
    out.append("  " + _row(cols, ws))
    out.append("  " + "-" * 63)

    for lo, hi, label, expected in p_bins:
        group = [p for p in picks
                 if p.get("p_modelo") is not None and lo <= p["p_modelo"] < hi]
        with_r = _with_resultado(group)
        s = _stats(with_r)
        if s["hit"] is not None:
            diff = s["hit"] - expected * 100
            diff_str = f"{diff:+.1f}pp"
        else:
            diff_str = "  N/A"
        out.append("  " + _row([label, _flag(len(group)), _flag(len(with_r)),
                                 _pct(s["hit"]), f"{expected*100:.1f}%", diff_str], ws))

    out.append("")
    out.append("  Calibration error = |actual - esperado|. Si error > 5pp sistematico → modelo sesgado.")


# ── Nodo-40: Sección S-40 Games/Sets Signal Layer ────────────────────────────

def seccion_40_games(out):
    """S-40: Calibración del modelo de totales (games/sets) — Nodo-40."""
    out.append(_header("S-40  Games/Sets Signal Layer — Nodo-40 Calibracion"))
    out.append("  Pregunta: el modelo predice sets y rango de juegos con precision suficiente?")
    out.append("  Hipotesis: games hit% > ganador hit% en zona coinflip (señal ortogonal).")

    cal_path = BASE_DIR / "data" / "calibracion_edge.json"
    if not cal_path.exists():
        out.append("\n  calibracion_edge.json no disponible.")
        return

    try:
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
    except Exception as e:
        out.append(f"\n  Error leyendo calibracion: {e}")
        return

    obs = cal.get("games_calibracion", [])
    thresholds = cal.get("games_thresholds", {})

    out.append(f"\n  Observaciones acumuladas: {len(obs)}")
    out.append(f"  Thresholds activos: DIFF_DOMINANTE={thresholds.get('DIFF_DOMINANTE', 0.35):.2f} | "
               f"DIFF_COINFLIP={thresholds.get('DIFF_COINFLIP', 0.18):.2f}")
    out.append(f"  REGLA-G6: escalar stakes cuando n>=50 (actualmente {'ACTIVA' if len(obs) < 50 else 'INACTIVA'})")

    if not obs:
        out.append("\n  Sin observaciones todavia — cerrar partidos con --cerrar para acumular datos.")
        return

    # S-40-1: Accuracy por zona_diff
    out.append(_subheader("S-40-1  Accuracy de Sets por Zona_diff"))
    out.append("")
    cols = ["Zona",       "N obs", "Sets OK", "Sets%", "Games en rango", "Games%"]
    ws   = [12,           6,       8,          7,       15,               7]
    out.append("  " + _row(cols, ws))
    out.append("  " + "-" * 62)

    zonas = [
        ("dominante",  "DOMINANTE  (|diff|>0.35)  → 2 sets UNDER"),
        ("ajustada",   "AJUSTADA   (0.18-0.35)    → no apostar"),
        ("coinflip",   "COINFLIP   (|diff|<=0.18) → 3 sets OVER"),
    ]

    for zona_key, zona_label in zonas:
        grupo = [o for o in obs if o.get("zona_diff") == zona_key]
        n = len(grupo)
        if n == 0:
            out.append("  " + _row([zona_key, "0", "-", " N/A", "-", " N/A"], ws))
            continue
        sets_ok    = sum(1 for o in grupo if o.get("sets_correcto") is True)
        games_ok   = sum(1 for o in grupo if o.get("games_en_rango") is True)
        games_n    = sum(1 for o in grupo if o.get("games_en_rango") is not None)
        sets_pct   = sets_ok / n * 100 if n else None
        games_pct  = games_ok / games_n * 100 if games_n else None
        flag_s     = "* " if n < 10 else "  "
        flag_g     = "* " if games_n < 10 else "  "
        games_str  = f"{flag_g}{games_ok}/{games_n}" if games_n else "   N/A"
        out.append("  " + _row([
            zona_key,
            f"{flag_s}{n}",
            f"{sets_ok}/{n}",
            _pct(sets_pct),
            games_str,
            _pct(games_pct) if games_pct is not None else " N/A",
        ], ws))

    out.append("")
    out.append("  Target: sets% >= 75% (dominante) | >= 70% (coinflip)")
    out.append("  * = n < 10, muestra insuficiente")

    # S-40-2: Hit% mercado totales vs ganador (hipótesis ortogonalidad)
    out.append(_subheader("S-40-2  Ortogonalidad — Games vs Ganador"))
    out.append("")

    coinflip_obs  = [o for o in obs if o.get("zona_diff") == "coinflip"]
    dominante_obs = [o for o in obs if o.get("zona_diff") == "dominante"]

    def _zona_stats(grupo):
        n = len(grupo)
        if not n:
            return None, None, None
        sets_ok = sum(1 for o in grupo if o.get("sets_correcto") is True)
        games_n = sum(1 for o in grupo if o.get("games_en_rango") is not None)
        games_ok = sum(1 for o in grupo if o.get("games_en_rango") is True)
        return n, sets_ok / n * 100, (games_ok / games_n * 100 if games_n else None)

    n_d, sp_d, gp_d = _zona_stats(dominante_obs)
    n_c, sp_c, gp_c = _zona_stats(coinflip_obs)

    out.append(f"  Dominante  n={n_d or 0}: sets%={_pct(sp_d)}  games_en_rango%={_pct(gp_d)}")
    out.append(f"  Coinflip   n={n_c or 0}: sets%={_pct(sp_c)}  games_en_rango%={_pct(gp_c)}")
    out.append("")
    out.append("  Hipotesis validada cuando:")
    out.append("  - Coinflip sets% > ganador hit% en zona coinflip (ganador es 50/50)")
    out.append("  - games_en_rango% > 60% en ambas zonas con n>=20")

    # S-40-3: Distribucion de diff_abs y accuracy
    if len(obs) >= 5:
        out.append(_subheader("S-40-3  Distribucion diff_abs (calibracion thresholds)"))
        out.append("")
        bins = [(0.0, 0.18, "coinflip"), (0.18, 0.35, "ajustada"), (0.35, 1.0, "dominante")]
        cols3 = ["Rango diff", "N", "Sets OK%", "Avg diff"]
        ws3   = [14, 5, 10, 10]
        out.append("  " + _row(cols3, ws3))
        out.append("  " + "-" * 44)
        for lo, hi, label in bins:
            grupo = [o for o in obs if o.get("diff") is not None and lo <= o["diff"] < hi]
            if not grupo:
                out.append("  " + _row([f"{lo:.2f}-{hi:.2f}", "0", " N/A", " N/A"], ws3))
                continue
            n = len(grupo)
            sets_ok = sum(1 for o in grupo if o.get("sets_correcto") is True)
            avg_diff = sum(o["diff"] for o in grupo) / n
            pct = sets_ok / n * 100 if n else 0
            out.append("  " + _row([f"{lo:.2f}-{hi:.2f}", str(n), _pct(pct), f"{avg_diff:.3f}"], ws3))

    out.append("")
    out.append("  Escalar stakes cuando n>=50 por zona (REGLA-G6).")
    out.append("  Fase 5 ajusta thresholds automaticamente con n>=50.")


# ── Fase 3 ────────────────────────────────────────────────────────────────────

def _semana_iso(d):
    if d is None:
        return "?"
    if isinstance(d, str):
        try:
            d = datetime.strptime(d[:10], "%Y-%m-%d").date()
        except ValueError:
            return "?"
    iso = d.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def seccion_27_6_temporal(picks, out):
    """S-27-6: Evolución temporal por semana."""
    out.append(_header("S-27-6  Evolucion Temporal (Drift Detection)"))
    out.append("  Pregunta: el accuracy mejora o empeora con el tiempo?")

    # Agrupar por semana ISO
    por_semana = {}
    for p in picks:
        semana = _semana_iso(p.get("fecha"))
        por_semana.setdefault(semana, []).append(p)

    cols = ["Semana", "N Total", "N Result", "Wins", "Losses", "Hit%", "ROI%(proxy)"]
    ws =   [10,       8,         9,          6,      7,         7,      13]
    out.append("")
    out.append("  " + _row(cols, ws))
    out.append("  " + "-" * 65)

    for semana in sorted(por_semana.keys()):
        group = por_semana[semana]
        with_r = _with_resultado(group)
        s = _stats(with_r)
        nota = ""
        if s["hit"] is not None and s["hit"] < 50 and len(with_r) >= 5:
            nota = " !! ALERTA drift"
        out.append("  " + _row([semana, _flag(len(group)), _flag(len(with_r)),
                                 s["wins"], s["losses"], _pct(s["hit"]),
                                 _roi_str(s["roi"])], ws) + nota)

    out.append("")
    out.append("  ALERTA: rolling accuracy < 55% con n>=20 → posible model drift.")


def seccion_27_7_portfolio(picks, combos, out):
    """S-27-7: Portfolio — Combos vs Individuales."""
    out.append(_header("S-27-7  Portfolio — Combos vs Individuales"))
    out.append("  Pregunta: los combos agregan valor sobre individuales?")

    # Individuales = picks del edge_report (apostar)
    indiv = [p for p in picks if p.get("fuente_lista") == "apostar"]
    with_r_i = _with_resultado(indiv)
    s_i = _stats(with_r_i)

    out.append("")
    out.append("  Individuales (picks APOSTAR de edge_report):")
    out.append(f"    N total: {len(indiv)} | Con resultado: {len(with_r_i)}")
    out.append(f"    Wins: {s_i['wins']} | Losses: {s_i['losses']} | Hit%: {_pct(s_i['hit'])}")
    out.append(f"    ROI proxy: {_roi_str(s_i['roi'])}")

    # Combos por tipo de piernas
    out.append("")
    out.append("  Combos (por numero de piernas):")

    cols = ["Tipo", "N Combos", "Stake Total", "Retorno Pot."]
    ws =   [12,     9,          12,             13]
    out.append("  " + _row(cols, ws))
    out.append("  " + "-" * 50)

    tipo_map = {
        2: "Safe 2p",
        3: "Combo 3p",
        4: "Combo 4p",
        (5, 99): "Mega 5p+",
    }

    from collections import defaultdict
    por_tipo = defaultdict(list)
    for c in combos:
        n = c.get("piernas_n", 0)
        if n <= 4:
            key = n
        else:
            key = (5, 99)
        por_tipo[key].append(c)

    for tipo_key, label in [(2, "Safe 2p"), (3, "Combo 3p"), (4, "Combo 4p"), ((5, 99), "Mega 5p+")]:
        group = por_tipo.get(tipo_key, [])
        total_stake = sum(c.get("stake", 0) or 0 for c in group)
        total_ret = sum(c.get("retorno_potencial", 0) or 0 for c in group)
        out.append("  " + _row([label, len(group), f"${total_stake:,.0f}", f"${total_ret:,.0f}"], ws))

    out.append("")
    out.append("  NOTA: resultado de combos no disponible automaticamente (requiere --cerrar con todos FT).")
    out.append("  Ver apuestas_*.json estado=CERRADO para P&L real de combos registrados.")


# ── Resumen ejecutivo ─────────────────────────────────────────────────────────

def resumen_ejecutivo(picks, out):
    """Resumen global al inicio del reporte."""
    with_r = _with_resultado(picks)
    s = _stats(with_r)
    apostar = [p for p in picks if p.get("fuente_lista") == "apostar"]

    out.append("=" * 72)
    out.append("  PIPELINE TRACKER — RESUMEN EJECUTIVO")
    out.append(f"  Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out.append("=" * 72)
    out.append("")
    out.append(f"  Edge reports cargados   : {len(set(p.get('fecha') for p in picks if p.get('fecha')))} fechas")
    out.append(f"  Picks totales           : {len(picks)}")
    out.append(f"  Picks APOSTAR           : {len(apostar)}")
    out.append(f"  Picks con resultado     : {len(with_r)}")
    out.append(f"  Picks pendientes        : {len(picks) - len(with_r)}")
    out.append("")
    out.append(f"  Hit% global             : {_pct(s['hit'])} ({s['wins']}W / {s['losses']}L)")
    out.append(f"  ROI proxy global        : {_roi_str(s['roi'])}")
    out.append(f"  Avg edge                : {_pct(s['avg_edge'])}")
    out.append("")
    # Confidence breakdown rápido
    for flag in CONF_ORDER:
        group = _with_resultado([p for p in picks if p.get("confidence_flag") == flag])
        if group:
            sg = _stats(group)
            out.append(f"  {flag:<10}: {sg['wins']}W/{sg['losses']}L  hit={_pct(sg['hit'])}")
    out.append("")


# ── CLI ───────────────────────────────────────────────────────────────────────

def seccion_27_8_shadow(out: list, desde: str = None, hasta: str = None) -> None:
    """
    S-27-8 — Shadow Book CLV Tracking (Nodo-52, D52-05).
    READ-ONLY: delega completamente a shadow_book.report().
    Addendum §G.4: menor invasión — tracker llama al módulo, no duplica lógica.
    """
    try:
        import shadow_book
        texto = shadow_book.report(desde=desde, hasta=hasta)
        out.append(texto)
    except ImportError:
        out.append("  [S-27-8] shadow_book.py no disponible.")
    except Exception as e:
        out.append(f"  [S-27-8] Error al cargar shadow book: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline Tracker & Observabilidad — Nodo-27 (READ-ONLY)"
    )
    parser.add_argument(
        "--since", type=str, default=None,
        help="Fecha mínima YYYY-MM-DD (ej: 2026-06-01)"
    )
    parser.add_argument(
        "--tier", type=str, default=None,
        choices=["grand_slam", "atp1000", "atp500", "challenger", "itf"],
        help="Filtrar por tier"
    )
    parser.add_argument(
        "--section", type=str, default=None,
        choices=["confianza", "cuotas", "tiers", "senales", "calibracion", "drift", "portfolio", "games", "shadow"],
        help="Solo una seccion (shadow = S-27-8 Shadow Book CLV, Nodo-52)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Guardar snapshot JSON en reports/pipeline_tracking_FECHA.json"
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="D58-01: Output snapshot como JSON a stdout para el dashboard"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    since = None
    if args.since:
        try:
            since = datetime.strptime(args.since, "%Y-%m-%d").date()
        except ValueError:
            print(f"ERROR: --since formato invalido: {args.since} (use YYYY-MM-DD)")
            sys.exit(1)

    # Cargar datos
    picks = cargar_edge_reports(since=since, tier_filter=args.tier)
    resultados_map = cargar_resultados_finales(since=since)
    apuestas_map = cargar_apuestas(since=since, tier_filter=args.tier)
    combos = cargar_trader_combos(since=since)

    # Join resultados
    picks = join_resultados(picks, resultados_map, apuestas_map)

    # Verificar datos mínimos
    if not picks:
        print("Sin datos de edge_report disponibles.")
        return

    out = []

    # Resumen ejecutivo siempre
    resumen_ejecutivo(picks, out)

    section = args.section

    # Fase 1
    if section is None or section == "confianza":
        seccion_27_1_confianza(picks, out)

    if section is None or section == "cuotas":
        seccion_27_2_cuotas(picks, out)

    if section is None or section == "tiers":
        seccion_27_3_tier_superficie(picks, out)

    # Fase 2
    if section is None or section == "senales":
        seccion_27_4_senales(picks, out)

    if section is None or section == "calibracion":
        seccion_27_5_calibracion(picks, out)

    # Fase 3
    if section is None or section == "drift":
        seccion_27_6_temporal(picks, out)

    if section is None or section == "portfolio":
        with_r_combos = _with_resultado(picks)  # combos result not tracked yet
        seccion_27_7_portfolio(picks, combos, out)

    if section is None or section == "games":
        seccion_40_games(out)

    if section is None or section == "shadow":
        since_str = str(since) if since else None
        seccion_27_8_shadow(out, desde=since_str)

    out.append("")
    out.append("=" * 72)
    out.append("  FIN DEL REPORTE — pipeline_tracker.py (READ-ONLY, Nodo-27)")
    out.append("=" * 72)

    # Output
    report_text = "\n".join(out)
    print(report_text)

    # Guardar pipeline_tracking.txt (siempre sobreescribe)
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")
        print(f"\n[Guardado: {OUTPUT_FILE}]")
    except OSError as e:
        print(f"\n[WARN: no se pudo guardar {OUTPUT_FILE}: {e}]")

    # D58-01: --json → stdout (no guarda archivo, no imprime reporte)
    if args.json_output:
        snap = {
            "fecha": datetime.now().isoformat(),
            "filtros": {"since": str(since) if since else None, "tier": args.tier},
            "n_picks": len(picks),
            "n_con_resultado": len(_with_resultado(picks)),
            "report": report_text,
        }
        print(json.dumps(snap, ensure_ascii=False, indent=2))
        return

    # Snapshot JSON opcional
    if args.save:
        snap_path = REPORTS_DIR / f"pipeline_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        snap = {
            "fecha": datetime.now().isoformat(),
            "filtros": {"since": str(since) if since else None, "tier": args.tier},
            "n_picks": len(picks),
            "n_con_resultado": len(_with_resultado(picks)),
            "report": report_text,
        }
        try:
            with open(snap_path, "w", encoding="utf-8") as f:
                json.dump(snap, f, ensure_ascii=False, indent=2)
            print(f"[Snapshot: {snap_path}]")
        except OSError as e:
            print(f"[WARN: no se pudo guardar snapshot: {e}]")


if __name__ == "__main__":
    main()
