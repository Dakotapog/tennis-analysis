"""
scraping/match_ledger.py — Nodo-118 F1

Match Ledger Crosswalk: fusiona partidos Playwright (match_ids+URLs) con partidos
API Kambi (cuotas) usando score Fellegi-Sunter simplificado + cuarentena.

PROHIBIDO: joins fuera de fusionar_dia(); auto-join bajo score MIN_SCORE_JOIN.
"""

import json
import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
MIN_SCORE_JOIN = 75       # ≥75 → AUTO-JOIN
MIN_SCORE_QUARANTINE = 55  # 55-74 → CUARENTENA, <55 → single-source
MAX_HORA_DELTA_H = 12      # bloqueo: descartar pares con Δhora > 12h

DATA_DIR = Path("data")
LEDGER_PATTERN = "match_ledger_{fecha}.json"
QUARANTINE_PATTERN = "cuarentena_{fecha}.json"
MERGED_PATTERN = "zita_merged_{fecha}_{ts}.json"


# ── Normalización ─────────────────────────────────────────────────────────────

def _strip_accents(s: str) -> str:
    """Elimina acentos Unicode."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def _normalizar_nombre(nombre: str) -> str:
    """
    Normaliza un nombre de jugador para comparación.

    Detecta formatos comunes y produce tokens ordenados:
      "Badosa P."   → {"badosa", "p"}
      "P. Badosa"   → {"badosa", "p"}
      "Paula Badosa" → {"badosa", "paula"}
      "Badosa Paula" → {"badosa", "paula"}

    Retorna string con tokens ordenados separados por espacio.
    """
    if not nombre:
        return ""

    s = _strip_accents(nombre.lower().strip())
    # quitar puntos y comas, colapsar espacios
    s = re.sub(r"[.,]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    tokens = s.split()
    if not tokens:
        return ""

    # Detectar iniciales (token de 1 char o ya era "x.")
    iniciales = {t for t in tokens if len(t) == 1}
    palabras = [t for t in tokens if len(t) > 1]

    # Reconstruir: palabras primero (apellido/nombre completo), iniciales al final
    resultado = sorted(palabras) + sorted(iniciales)
    return " ".join(resultado)


def _tokens(nombre_norm: str) -> set:
    return set(nombre_norm.split())


# ── Score por componente ──────────────────────────────────────────────────────

def _score_jugador(n_kambi: str, n_fs: str) -> int:
    """
    Score para un par de nombres de jugador (0-35).

    35: hit exacto en tokens normalizados
    30: apellido exacto + inicial compatible
    20: solo apellido exacto (sin inicial en una fuente)
    12: fuzzy token-overlap >= 0.8
     0: sin coincidencia
    """
    nk = _normalizar_nombre(n_kambi)
    nf = _normalizar_nombre(n_fs)

    if not nk or not nf:
        return 0

    tk = _tokens(nk)
    tf = _tokens(nf)

    # Hit exacto de tokens
    if tk == tf:
        return 35

    # Extraer iniciales vs palabras completas
    iniciales_k = {t for t in tk if len(t) == 1}
    palabras_k = {t for t in tk if len(t) > 1}
    iniciales_f = {t for t in tf if len(t) == 1}
    palabras_f = {t for t in tf if len(t) > 1}

    # Apellido exacto = la palabra más larga de cada fuente coincide
    apellido_k = max(palabras_k, key=len) if palabras_k else ""
    apellido_f = max(palabras_f, key=len) if palabras_f else ""
    apellido_ok = apellido_k and apellido_f and apellido_k == apellido_f

    if apellido_ok:
        # Inicial compatible: la inicial de una fuente ∈ iniciales del otro
        # o la inicial del nombre completo del otro
        iniciales_efectivas_k = iniciales_k | {p[0] for p in palabras_k if p != apellido_k}
        iniciales_efectivas_f = iniciales_f | {p[0] for p in palabras_f if p != apellido_f}
        inicial_ok = bool(iniciales_efectivas_k & iniciales_efectivas_f)
        if inicial_ok:
            return 30
        # Apellido solo (sin inicial en una de las fuentes)
        return 20

    # Fuzzy: overlap de tokens
    if tk and tf:
        overlap = len(tk & tf) / max(len(tk), len(tf))
        if overlap >= 0.8:
            return 12

    return 0


def _score_torneo(t_kambi: str, t_fs: str) -> int:
    """Score torneo (0-15): token-overlap >= 0.5 → 15; país/ciudad coincide → 8."""
    if not t_kambi or not t_fs:
        return 0

    tk = set(_strip_accents(t_kambi.lower()).split())
    tf = set(_strip_accents(t_fs.lower()).split())

    # quitar stopwords comunes
    stopwords = {"open", "atp", "wta", "itf", "challenger", "international", "de", "la", "el"}
    tk -= stopwords
    tf -= stopwords

    if not tk or not tf:
        return 0

    overlap = len(tk & tf) / max(len(tk), len(tf))
    if overlap >= 0.5:
        return 15
    if tk & tf:  # al menos un token en común
        return 8
    return 0


def _parse_hora(hora_str: str) -> Optional[float]:
    """Parsea 'HH:MM' o ISO a horas decimales. Retorna None si no parseable."""
    if not hora_str:
        return None
    # Intentar HH:MM
    m = re.search(r"(\d{1,2}):(\d{2})", str(hora_str))
    if m:
        return int(m.group(1)) + int(m.group(2)) / 60
    return None


def _score_hora(hora_k: str, hora_f: str) -> int:
    """Score hora (0-15): Δ≤2h=15, Δ≤6h=8, Δ≤12h=3, >12h=0."""
    hk = _parse_hora(hora_k)
    hf = _parse_hora(hora_f)
    if hk is None or hf is None:
        return 5  # bonus neutro si alguna hora falta
    delta = abs(hk - hf)
    # Considerar wrap-around medianoche (ej: 23:00 vs 01:00 = 2h)
    delta = min(delta, 24 - delta)
    if delta <= 2:
        return 15
    if delta <= 6:
        return 8
    if delta <= MAX_HORA_DELTA_H:
        return 3
    return 0


def score_par(partido_kambi: dict, partido_fs: dict,
              crosswalk: Optional[dict] = None) -> Tuple[int, dict]:
    """
    Score total para un par (kambi, fs). Retorna (score_total, detalle_componentes).

    Componentes:
      jugador1: 0-35
      jugador2: 0-35
      torneo:   0-15
      hora:     0-15
      Total:    0-100
    """
    crosswalk = crosswalk or {}

    # Shortcut: match_id compartido = identidad perfecta (FlashScore ID único por partido)
    mid_k = partido_kambi.get("match_id", "")
    mid_f = partido_fs.get("match_id", "")
    if mid_k and mid_f and mid_k == mid_f:
        return 100, {"jugador1": 35, "jugador2": 35, "torneo": 15, "hora": 15, "match_id": True, "total": 100}

    j1k = partido_kambi.get("jugador1", "") or partido_kambi.get("player1", "")
    j2k = partido_kambi.get("jugador2", "") or partido_kambi.get("player2", "")
    j1f = partido_fs.get("jugador1", "") or partido_fs.get("player1", "")
    j2f = partido_fs.get("jugador2", "") or partido_fs.get("player2", "")

    # Intentar crosswalk primero (lookup exacto, cero fuzzy)
    def crosswalk_hit(nombre: str) -> bool:
        n = _normalizar_nombre(nombre)
        return n in crosswalk.get("_aliases", {})

    # Score directo (j1_kambi vs j1_fs, j2_kambi vs j2_fs)
    s1_directo = 35 if (crosswalk_hit(j1k) and crosswalk_hit(j1f) and
                        crosswalk.get("_aliases", {}).get(_normalizar_nombre(j1k)) ==
                        crosswalk.get("_aliases", {}).get(_normalizar_nombre(j1f))) \
        else _score_jugador(j1k, j1f)

    s2_directo = 35 if (crosswalk_hit(j2k) and crosswalk_hit(j2f) and
                        crosswalk.get("_aliases", {}).get(_normalizar_nombre(j2k)) ==
                        crosswalk.get("_aliases", {}).get(_normalizar_nombre(j2f))) \
        else _score_jugador(j2k, j2f)

    # Score invertido (j1_kambi vs j2_fs, j2_kambi vs j1_fs) — por si el orden difiere
    s1_inv = _score_jugador(j1k, j2f)
    s2_inv = _score_jugador(j2k, j1f)

    # Usar la asignación con mayor score total de jugadores
    if s1_directo + s2_directo >= s1_inv + s2_inv:
        sj1, sj2 = s1_directo, s2_directo
    else:
        sj1, sj2 = s1_inv, s2_inv

    # Cubrir todos los campos de torneo usados en producción (zita usa torneo_nombre/completo)
    def _get_torneo(p):
        return (p.get("torneo") or p.get("torneo_nombre") or p.get("torneo_completo")
                or p.get("torneo_fs") or p.get("tournament") or "")

    tk = _get_torneo(partido_kambi)
    tf = _get_torneo(partido_fs)
    st = _score_torneo(tk, tf)

    def _get_hora(p):
        return (p.get("hora") or p.get("hora_partido") or p.get("hora_inicio") or p.get("time") or "")

    hk = _get_hora(partido_kambi)
    hf = _get_hora(partido_fs)
    sh = _score_hora(str(hk), str(hf))

    total = sj1 + sj2 + st + sh
    detalle = {"jugador1": sj1, "jugador2": sj2, "torneo": st, "hora": sh, "total": total}
    return total, detalle


# ── Fusión ────────────────────────────────────────────────────────────────────

def fusionar_dia(
    kambi_matches: List[dict],
    fs_matches: List[dict],
    fecha: str,
    crosswalk: Optional[dict] = None,
    output_dir: str = "data"
) -> Tuple[str, dict]:
    """
    Fusiona partidos Kambi (cuotas) con partidos FlashScore (match_ids+URLs).

    ÚNICO punto de entrada para joins de identidad — PROHIBIDO hacer joins fuera.

    Args:
        kambi_matches: lista de partidos de extraer_partidos_api.py
        fs_matches:    lista de partidos de extraer_URL_partidos_version2.py
        fecha:         'YYYY-MM-DD'
        crosswalk:     dict de aliases {_aliases: {nombre_norm: canonical_id}}
        output_dir:    directorio de salida

    Returns:
        (merged_path, stats_dict)
    """
    crosswalk = crosswalk or {}
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Índice de FS por partido (para asignación greedy sin duplicados)
    fs_disponibles = list(enumerate(fs_matches))  # (idx_original, partido)

    joins = []
    cuarentena = []
    single_source_kambi = []
    single_source_fs = list(range(len(fs_matches)))  # índices no usados

    # Filtro de bloqueo: Δhora ≤ MAX_HORA_DELTA_H
    def dentro_ventana(pk, pf):
        hk = _parse_hora(str(pk.get("hora", "") or pk.get("time", "")))
        hf = _parse_hora(str(pf.get("hora", "") or pf.get("time", "")))
        if hk is None or hf is None:
            return True  # sin hora → no filtrar
        delta = abs(hk - hf)
        delta = min(delta, 24 - delta)
        return delta <= MAX_HORA_DELTA_H

    # Asignación greedy: para cada partido Kambi, buscar el mejor match FS disponible
    fs_usado = set()

    for pk in kambi_matches:
        # Homónimo guard: si hay dos candidatos FS con score jugador similar,
        # exigir score_torneo > 0 para auto-join; si no, cuarentena
        candidatos = []
        for idx_f, pf in fs_disponibles:
            if idx_f in fs_usado:
                continue
            if not dentro_ventana(pk, pf):
                continue
            s, detalle = score_par(pk, pf, crosswalk)
            if s >= MIN_SCORE_QUARANTINE:
                candidatos.append((s, detalle, idx_f, pf))

        if not candidatos:
            # Sin match viable → single-source Kambi
            single_source_kambi.append({**pk, "join_method": "SINGLE_SOURCE_KAMBI",
                                        "sources": ["kambi"]})
            continue

        # Ordenar por score desc
        candidatos.sort(key=lambda x: x[0], reverse=True)
        mejor_score, mejor_detalle, mejor_idx, mejor_pf = candidatos[0]

        # Homónimo guard: si hay un segundo candidato con score jugador similar
        if len(candidatos) > 1:
            segundo_score = candidatos[1][0]
            diff = mejor_score - segundo_score
            if diff < 10 and mejor_detalle.get("torneo", 0) == 0:
                # Ambigüedad sin discriminador de torneo → cuarentena
                cuarentena.append({
                    "kambi": pk,
                    "candidatos": [{"score": c[0], "fs_jugador1": c[3].get("jugador1"),
                                    "detalle": c[1]} for c in candidatos[:3]],
                    "razon": "homonimo_sin_torneo",
                    "score_mejor": mejor_score
                })
                continue

        if mejor_score >= MIN_SCORE_JOIN:
            # AUTO-JOIN: enriquecer FS con cuotas Kambi
            partido_merged = {
                **mejor_pf,
                "cuota1": pk.get("cuota1"),
                "cuota2": pk.get("cuota2"),
                "outcome_id": pk.get("outcome_id") or pk.get("kambi_event_id"),
                "kambi_event_id": pk.get("kambi_event_id") or pk.get("event_id"),
                "join_method": "AUTO_JOIN",
                "join_score": mejor_score,
                "join_detalle": mejor_detalle,
                "sources": ["kambi", "flashscore"],
                "cuota_es_real": True,
            }
            # D143-01 (Nodo-143): Propagar metadata torneo desde Kambi al join.
            # Solo llena huecos (no sobrescribe) — Kambi gana para tier/torneo,
            # FlashScore gana para match_id/H2H URLs.
            _KAMBI_META_FIELDS = ['tier', 'torneo_nombre', 'torneo_completo',
                                   'pais', 'ranking1', 'ranking2', 'tournament_context']
            for _campo in _KAMBI_META_FIELDS:
                if _campo in pk and pk[_campo] and not partido_merged.get(_campo):
                    partido_merged[_campo] = pk[_campo]
            joins.append(partido_merged)
            fs_usado.add(mejor_idx)
            single_source_fs = [i for i in single_source_fs if i != mejor_idx]
        else:
            # CUARENTENA (55-74)
            cuarentena.append({
                "kambi": pk,
                "fs_candidato": mejor_pf,
                "score": mejor_score,
                "detalle": mejor_detalle,
                "razon": "score_bajo"
            })
            single_source_kambi.append({**pk, "join_method": "SINGLE_SOURCE_KAMBI",
                                         "sources": ["kambi"]})

    # Partidos FS sin join → single-source
    ss_fs_list = [
        {**fs_matches[i], "join_method": "SINGLE_SOURCE_FS", "sources": ["flashscore"]}
        for i in single_source_fs if i not in fs_usado
    ]

    # Construir archivo merged (unión de joins + single-sources)
    merged = joins + single_source_kambi + ss_fs_list

    # Stats
    n_kambi = len(kambi_matches)
    n_fs = len(fs_matches)
    n_joins = len(joins)
    n_cuarentena = len(cuarentena)
    cobertura_pct = round(n_joins / n_kambi * 100, 1) if n_kambi else 0.0

    stats = {
        "fecha": fecha,
        "playwright_total": n_fs,
        "api_total": n_kambi,
        "joins_exitosos": n_joins,
        "cuarentena_count": n_cuarentena,
        "single_source_kambi": len(single_source_kambi),
        "single_source_fs": len(ss_fs_list),
        "cobertura_pct": cobertura_pct,
    }

    # Imprimir embudo (Zero-Null §5 Nodo-118)
    _imprimir_embudo(stats, cuarentena)

    # Serializar
    ts = datetime.now().strftime("%H%M%S")
    merged_path = out_dir / MERGED_PATTERN.format(fecha=fecha.replace("-", ""), ts=ts)
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump({"fecha": fecha, "partidos": merged, "stats": stats}, f,
                  ensure_ascii=False, indent=2)

    ledger = {
        "fecha": fecha,
        "joins": joins,
        "cuarentena": cuarentena,
        "single_source_kambi": single_source_kambi,
        "single_source_fs": ss_fs_list,
        "stats": stats,
    }
    ledger_path = out_dir / LEDGER_PATTERN.format(fecha=fecha.replace("-", ""))
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2)

    qpath = out_dir / QUARANTINE_PATTERN.format(fecha=fecha.replace("-", ""))
    with open(qpath, "w", encoding="utf-8") as f:
        json.dump(cuarentena, f, ensure_ascii=False, indent=2)

    logger.info(f"Ledger escrito: {ledger_path} | Merged: {merged_path}")
    return str(merged_path), stats


def _imprimir_embudo(stats: dict, cuarentena: list) -> None:
    """Imprime embudo Zero-Null en stdout (Nodo-118 §5)."""
    n_kambi = stats["api_total"]
    n_fs = stats["playwright_total"]
    n_joins = stats["joins_exitosos"]
    n_q = stats["cuarentena_count"]
    ss_k = stats["single_source_kambi"]
    ss_f = stats["single_source_fs"]
    cobertura = stats["cobertura_pct"]

    warn = " ⚠️  WARN: cobertura <60%" if cobertura < 60 else ""
    print(f"\nEMBUDO LEDGER {stats.get('fecha', '')}:")
    print(f"  Universo FS (Playwright): {n_fs:>4}   Con cuotas (Kambi): {n_kambi:>3}")
    print(f"  Join auto:               {n_joins:>4}   Cuarentena:        {n_q:>3}   "
          f"Single-source K/FS: {ss_k}/{ss_f}")
    print(f"  Cobertura con-cuotas:   {cobertura:>5.1f}%{warn}")
    if n_q > 0:
        print(f"  FUGA ({n_q}): " +
              " | ".join(f"{c.get('kambi',{}).get('jugador1','?')} vs "
                         f"{c.get('kambi',{}).get('jugador2','?')} "
                         f"(score={c.get('score', c.get('score_mejor', '?'))})"
                         for c in cuarentena[:5]))


# ── Persistencia del ledger ───────────────────────────────────────────────────

# ── Adapter para edge_calculator (Nodo-118 F4) ───────────────────────────────

_REQUIRED_FIELDS_EDGE = ("jugador1", "jugador2", "cuota1", "cuota2",
                          "match_id", "match_url", "superficie", "torneo_nombre")


def exportar_para_edge_calculator(fecha: str, data_dir: str = "data") -> str:
    """
    Lee el ledger del día y exporta un archivo zita_tennis_matches_*_merged.json
    en el schema plano que edge_calculator / extraer_historh2h ya consumen.

    Incluye: joins (cuotas+match_id) + single_source_kambi (cuotas sin match_id)
             + single_source_fs CON cuotas válidas (FlashScore real odds, qualifying rounds).
    Excluye single_source_fs sin cuotas (cuota1=None o cuota1=0). D120-01.

    Retorna el path del archivo escrito.
    """
    ledger = load_ledger(fecha, data_dir)
    if not ledger:
        return ""

    # D120-01: ss_fs con cuotas reales FlashScore (qualifying rounds fuera de Kambi)
    ss_fs_con_cuotas = [
        p for p in ledger.get("single_source_fs", [])
        if p.get("cuota1") and p.get("cuota2")
        and float(p.get("cuota1", 0)) > 0 and float(p.get("cuota2", 0)) > 0
    ]
    joins = ledger.get("joins", [])
    ssk = ledger.get("single_source_kambi", [])
    logger.info(f"   Exportados: {len(joins)} joins + {len(ssk)} kambi + "
                f"{len(ss_fs_con_cuotas)} ss_fs_con_cuotas (Nodo-120)")

    partidos_export = []
    for p in joins + ssk + ss_fs_con_cuotas:
        if not isinstance(p, dict):
            continue
        partidos_export.append({
            "jugador1":      p.get("jugador1", ""),
            "jugador2":      p.get("jugador2", ""),
            "cuota1":        p.get("cuota1"),
            "cuota2":        p.get("cuota2"),
            "match_id":      p.get("match_id", ""),
            "match_url":     p.get("match_url", ""),
            "superficie":    p.get("superficie", ""),
            "torneo_nombre": p.get("torneo_nombre", ""),
            "torneo_completo": p.get("torneo_completo", ""),
            "tier":          p.get("tier", ""),
            "hora":          p.get("hora") or p.get("hora_partido", ""),
            "ranking1":      p.get("ranking1"),
            "ranking2":      p.get("ranking2"),
            "kambi_event_id": p.get("kambi_event_id") or p.get("outcome_id"),
            "_ledger_status": p.get("join_method", ""),
            "_join_score":    p.get("join_score"),
            "_cuota_source":  "flashscore" if p.get("join_method") == "SINGLE_SOURCE_FS" else "kambi",
        })

    ts = datetime.now().strftime("%H%M%S")
    fecha_compact = fecha.replace("-", "")
    out_path = Path(data_dir) / f"zita_tennis_matches_{fecha_compact}_{ts}_merged.json"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(partidos_export, f, ensure_ascii=False, indent=2)
    return str(out_path)


def _buscar_cuota_aggregator(nombre_norm: str, feeds: dict) -> tuple:
    """
    Busca cuota para un jugador en feeds del odds_aggregator.
    Retorna (cuota, bookmaker) o (None, None) si no hay match único.
    D121-03: apellido-first, homónimo = no enriquecer.
    """
    if not nombre_norm:
        return None, None
    token = nombre_norm.split()[0]
    # Match: nombre completo exacto | token apellido exacto | token apellido + espacio
    # Cubre: "aksu a" → "aksu" (clave apellido en aggregator) — D121-03
    candidatos = [k for k in feeds
                  if k == nombre_norm or k == token or k.startswith(token + " ")]
    if len(candidatos) != 1:
        return None, None
    info = feeds[candidatos[0]]
    if not isinstance(info, dict):
        return None, None
    mejor = max(
        [(b, d["odds"]) for b, d in info.items()
         if isinstance(d, dict) and d.get("odds") and float(d["odds"]) > 1.0],
        key=lambda x: x[1],
        default=(None, None),
    )
    return mejor[1], mejor[0]


def enriquecer_ss_fs_con_aggregator(
    fecha: str, data_dir: str = "data", libros: list = None
) -> dict:
    """
    Enriquece ss_fs sin cuotas usando odds_aggregator (betplay+rushbet). D121-01.

    Para cada single_source_fs con cuota1=None, busca los jugadores en el feed
    del odds_aggregator y puebla cuota1/cuota2 con la mejor cuota disponible.
    Guarda el ledger actualizado. Retorna stats.
    """
    try:
        import sys as _sys
        import os as _os
        _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from scripts.odds_aggregator import fetch_all_odds
    except ImportError:
        logger.warning("odds_aggregator no disponible — enriquecimiento omitido")
        return {}

    libros = libros or ["betplay", "rushbet"]
    ledger = load_ledger(fecha, data_dir)
    if not ledger:
        return {}

    ss_fs = ledger.get("single_source_fs", [])
    ss_sin_cuota = [p for p in ss_fs if not p.get("cuota1")]
    if not ss_sin_cuota:
        logger.info("   Enrichment D121-01: 0 ss_fs sin cuota — nada que enriquecer")
        return {"enriquecidos": 0, "sin_match": 0, "homonimos": 0,
                "total_ss_fs": len(ss_fs)}

    feeds = fetch_all_odds(libros)

    enriquecidos = 0
    sin_match = 0
    homonimos = 0

    for partido in ss_sin_cuota:
        j1 = _normalizar_nombre(partido.get("jugador1", ""))
        j2 = _normalizar_nombre(partido.get("jugador2", ""))

        cuota1, book1 = _buscar_cuota_aggregator(j1, feeds)
        cuota2, _     = _buscar_cuota_aggregator(j2, feeds)

        if cuota1 and cuota2:
            partido["cuota1"] = cuota1
            partido["cuota2"] = cuota2
            partido["_cuota_source"] = book1
            partido["_best_book"] = book1
            partido["_enriched_by"] = "D121-01"
            enriquecidos += 1
        else:
            # Distinguir sin_match de homónimo (candidatos > 1)
            j1_norm = _normalizar_nombre(partido.get("jugador1", ""))
            tok = j1_norm.split()[0] if j1_norm else ""
            cands = [k for k in feeds if k == j1_norm or k.startswith(tok + " ")]
            if len(cands) > 1:
                homonimos += 1
                logger.warning(f"   Homónimo D121-03: {partido.get('jugador1')} "
                                f"→ {len(cands)} candidatos, no enriquecido")
            else:
                sin_match += 1

    save_ledger(ledger, fecha, data_dir)
    stats = {"enriquecidos": enriquecidos, "sin_match": sin_match,
             "homonimos": homonimos, "total_ss_fs": len(ss_fs)}
    logger.info(f"   Enrichment D121-01: {enriquecidos}/{len(ss_sin_cuota)} ss_fs enriquecidos "
                f"({sin_match} sin match, {homonimos} homónimos) libros={libros}")
    return stats


def actualizar_cuotas_ledger(
    fecha: str, kambi_matches: List[dict], data_dir: str = "data"
) -> dict:
    """
    PASO 1c: refresca cuota1/cuota2 en el ledger existente con datos Kambi frescos.
    Identifica entradas por kambi_event_id o match_id.
    Persiste el ledger actualizado. Retorna stats.
    """
    ledger = load_ledger(fecha, data_dir)
    if not ledger:
        return {"actualizados": 0, "sin_cambio": 0, "no_encontrado": 0}

    by_event = {(m.get("kambi_event_id") or m.get("event_id")): m
                for m in kambi_matches
                if m.get("kambi_event_id") or m.get("event_id")}
    by_match = {m.get("match_id"): m
                for m in kambi_matches if m.get("match_id")}

    actualizados = 0
    sin_cambio = 0
    no_encontrado = 0
    ts = datetime.now().isoformat()

    for seccion in ("joins", "single_source_kambi"):
        for partido in ledger.get(seccion, []):
            eid = partido.get("kambi_event_id") or partido.get("outcome_id")
            mid = partido.get("match_id")
            fresco = by_event.get(eid) or by_match.get(mid)
            if not fresco:
                no_encontrado += 1
                continue
            c1_nueva, c2_nueva = fresco.get("cuota1"), fresco.get("cuota2")
            if c1_nueva != partido.get("cuota1") or c2_nueva != partido.get("cuota2"):
                partido["cuota1"] = c1_nueva
                partido["cuota2"] = c2_nueva
                partido["cuota_refresh_ts"] = ts
                actualizados += 1
            else:
                sin_cambio += 1

    save_ledger(ledger, fecha, data_dir)
    return {"actualizados": actualizados, "sin_cambio": sin_cambio,
            "no_encontrado": no_encontrado}


def load_ledger(fecha: str, data_dir: str = "data") -> dict:
    """Carga el ledger del día dado. Retorna {} si no existe."""
    path = Path(data_dir) / LEDGER_PATTERN.format(fecha=fecha.replace("-", ""))
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_ledger(ledger: dict, fecha: str, data_dir: str = "data") -> None:
    """Guarda el ledger en disco."""
    path = Path(data_dir) / LEDGER_PATTERN.format(fecha=fecha.replace("-", ""))
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import glob as _glob

    parser = argparse.ArgumentParser(description="Match Ledger Crosswalk (Nodo-118 F1)")
    parser.add_argument("--build", action="store_true",
                        help="Fusionar archivos Playwright+API del día")
    parser.add_argument("--fecha", default=datetime.now().strftime("%Y-%m-%d"),
                        help="Fecha YYYY-MM-DD (default: hoy)")
    parser.add_argument("--playwright", help="Path archivo Playwright (zita_tennis_matches)")
    parser.add_argument("--api", help="Path archivo API (zita_tennis_matches API)")
    parser.add_argument("--data-dir", default="data", help="Directorio datos")
    parser.add_argument("--enrich", action="store_true",
                        help="Enriquecer ss_fs sin cuota via odds_aggregator (D121-02)")
    args = parser.parse_args()

    if args.build:
        import sys

        # Auto-detectar archivos si no se especifican
        def _latest(pattern):
            files = sorted(_glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime,
                           reverse=True)
            return files[0] if files else None

        fecha_compact = args.fecha.replace("-", "")

        def _es_archivo_kambi(path: str) -> bool:
            """Detecta archivo Kambi API por presencia de kambi_event_id u outcome_id (D120-fix)."""
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else [
                    p for v in data.values() for p in (v if isinstance(v, list) else [])
                ]
                return any(
                    p.get("kambi_event_id") or p.get("outcome_id")
                    for p in items if isinstance(p, dict)
                )
            except Exception:
                return False

        # Separar archivos Kambi API (tiene kambi_event_id) de Playwright (no tiene)
        # Nota: AMBOS pueden tener cuotas — la distinción es kambi_event_id (Nodo-120 fix)
        candidatos = sorted(
            _glob.glob(f"{args.data_dir}/zita_tennis_matches_{fecha_compact}*.json"),
            key=lambda p: Path(p).stat().st_mtime
        )
        archivos_kambi = [p for p in candidatos if _es_archivo_kambi(p)]
        archivos_pw    = [p for p in candidatos if not _es_archivo_kambi(p)]

        api_path = args.api or (archivos_kambi[-1] if archivos_kambi else None)
        pw_path = args.playwright or (archivos_pw[-1] if archivos_pw else None)

        if not api_path:
            print("ERROR: no se encontró archivo con cuotas (API). Usa --api.")
            sys.exit(1)
        if not pw_path:
            print("WARN: no se encontró archivo Playwright. Usando solo API.")
            pw_path = api_path

        def _leer_partidos(path: str) -> list:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                partidos = []
                for v in data.values():
                    if isinstance(v, list):
                        partidos.extend(v)
                return partidos
            return []

        kambi = _leer_partidos(api_path)
        fs = _leer_partidos(pw_path)

        merged_path, stats = fusionar_dia(kambi, fs, args.fecha,
                                          output_dir=args.data_dir)
        if args.enrich:
            enrich_stats = enriquecer_ss_fs_con_aggregator(args.fecha, data_dir=args.data_dir)
            n_enr = enrich_stats.get("enriquecidos", 0)
            n_tot = enrich_stats.get("total_ss_fs", 0)
            print(f"Enriched: {n_enr}/{n_tot} ss_fs via aggregator (betplay/rushbet) [D121-01]")
        export_path = exportar_para_edge_calculator(args.fecha, data_dir=args.data_dir)
        print(f"\nMerged: {merged_path}")
        print(f"Export: {export_path}")

    elif args.enrich:
        # --enrich sin --build: enriquecer ledger ya construido
        enrich_stats = enriquecer_ss_fs_con_aggregator(args.fecha, data_dir=args.data_dir)
        n_enr = enrich_stats.get("enriquecidos", 0)
        n_tot = enrich_stats.get("total_ss_fs", 0)
        print(f"Enriched: {n_enr}/{n_tot} ss_fs via aggregator [D121-01]")
        export_path = exportar_para_edge_calculator(args.fecha, data_dir=args.data_dir)
        print(f"Export: {export_path}")
