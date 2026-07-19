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
    args = parser.parse_args()

    if args.build:
        import sys

        # Auto-detectar archivos si no se especifican
        def _latest(pattern):
            files = sorted(_glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime,
                           reverse=True)
            return files[0] if files else None

        fecha_compact = args.fecha.replace("-", "")
        pw_path = args.playwright or _latest(
            f"{args.data_dir}/zita_tennis_matches_{fecha_compact}*.json"
        )
        api_path = args.api or _latest(
            f"{args.data_dir}/zita_tennis_matches_{fecha_compact}*.json"
        )

        if not pw_path or not api_path:
            print("ERROR: no se encontraron archivos. Usa --playwright y --api.")
            sys.exit(1)

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
        print(f"\nMerged: {merged_path}")
