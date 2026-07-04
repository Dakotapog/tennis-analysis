"""
shadow_book.py — Nodo-52: Libro Sombra con CLV Tracking

Implementa Nodo-52 original + Addendum (Nodo-52-ADDENDUM-Integracion-Contexto-Completo.md)

Tres momentos (§1 del spec):
  Momento 1 — LOG:      log_picks()  → reports/shadow_book/sb_YYYY-MM-DD.jsonl
  Momento 2 — SNAPSHOT: close_snapshot()  → añade cierre_kambi a registros abiertos
  Momento 3 — SETTLE:   settle()     → añade bloque 'resolucion'

Addendum B.1 — Schema corregida:
  Guardar el pick COMPLETO del edge_report bajo 'pick_snapshot' (48+ campos).
  Menos código, cero pérdida de información. Los campos derivados (sb_id,
  match_key, es_qualifying, season_transition_flag) son conveniencias top-level.

Addendum B.3 — Session meta:
  Registro '_type: session_meta' por llamada a log_picks() con n por status,
  cv_edge, dispersion_level, session_regime (para V-26-1 y V-26-5b).

INMUTABILIDAD (§1): pick_snapshot NUNCA se edita retroactivamente.
  settle() solo AÑADE el bloque 'resolucion'.

REGLA-T27-2 (heredada de Nodo-27): toda tabla muestra n; bins con n<10 → '*'.
READ-ONLY: este módulo NUNCA modifica edge_report, calibracion_edge.json, ni
  ningún output del pipeline. Es observación pura + registro propio.
ROI simulado siempre flat 1u (nunca mezclar con stakes reales de betslip_registrar).

Hipótesis pre-registradas (Addendum §D, congeladas 2026-07-02):
  H52-01 WAS supera breakeven (D44-03, n=30)
  H52-02 n_h2h=0 + ITF: ¿ELO/Markov discriminan? (Nodo-33 F2, n=30)
  H52-03 STRUCTURAL_ALPHA > LOW hit% (V-28-2, n=20)
  H52-04 Surface discount ON mejora Brier (D46-07, n=5)
  H52-05 STEAM_IN > DRIFT_OUT hit% (V-26-3d, n=20)
  H52-06 p_modelo ranking preservado en sesiones BLIND (V-26-1a/b, n=5 sesiones)
  H52-07 Qualifiers p∈[0.52,0.55) vs principal p≥0.55 (conversación 2026-07-01, n=50)
  H52-08 Zona 2.00-2.50 sigue siendo trampa post-fixes 32/33 (S-27-2, n=30)
"""

import json
import os
import re
import math
import logging
import argparse
import glob as glob_mod
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Directorio de datos — monkeypatcheable en tests
SHADOW_DIR = "reports/shadow_book"

_Z95 = 1.96  # z para IC Wilson 95%

# Status de pick derivado del pick_snapshot
_STATUS_APROBADO = 'APROBADO'
_STATUS_WATCHLIST = 'WATCHLIST'
_STATUS_NO_DATA = 'NO_DATA'  # Nodo-51 F2 / PICK_STATUS_NO_DATA


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES INTERNAS
# ══════════════════════════════════════════════════════════════════════════════

def _slug(text: str, max_len: int = 16) -> str:
    """Normaliza texto a slug URL-safe."""
    s = text.lower().strip()
    s = s.replace('qualifying', 'q').replace('clasificatorios', 'q').replace('qualif', 'q')
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', '-', s)
    s = re.sub(r'-+', '-', s)
    return s[:max_len].rstrip('-')


def _parse_apellido(nombre: str) -> str:
    """Extrae apellido de un nombre de jugador (último token largo)."""
    try:
        from scraping.kambi_tennis import _parse_nombre
        apellido, _ = _parse_nombre(nombre)
        return apellido
    except Exception:
        parts = nombre.lower().strip().split()
        return parts[-1] if parts else nombre.lower()[:8]


def _build_sb_id(fecha: str, torneo: str, p1: str, p2: str, mercado: str = "ML") -> str:
    """
    ID determinista: fecha_torneo-slug_apellido1-apellido2_mercado
    Invariante al orden de p1/p2 (orden alfabético de apellidos).
    """
    t_slug = _slug(torneo, max_len=16)
    a1 = _parse_apellido(p1)
    a2 = _parse_apellido(p2)
    nombres = sorted([a1[:10], a2[:10]])
    return f"{fecha}_{t_slug}_{nombres[0]}-{nombres[1]}_{mercado}"


def _match_key(p1: str, p2: str) -> str:
    """Match key como 'apellido_menor_apellido_mayor' (reutiliza Nodo-48)."""
    try:
        from scraping.kambi_tennis import _build_match_key
        tup = _build_match_key(p1, p2)
        return f"{tup[0]}_{tup[2]}"
    except Exception:
        a1 = _parse_apellido(p1)
        a2 = _parse_apellido(p2)
        return "_".join(sorted([a1, a2]))


def _es_qualifying(torneo: str) -> bool:
    """Detecta si el torneo es clasificatorio (best-effort sin F1)."""
    t = torneo.lower()
    return any(k in t for k in ('qualifying', 'clasificator', 'qualif', ' q.', '-q'))


def _season_transition(fecha: str, superficie: str) -> bool:
    """Detecta transición clay→grass (jun-jul) o grass→hard (ago-sep)."""
    try:
        dt = datetime.strptime(fecha, "%Y-%m-%d")
        if superficie == 'grass' and dt.month in (6, 7):
            return True
        if superficie == 'hard' and dt.month in (8, 9):
            return True
    except Exception:
        pass
    return False


def _sb_status(pick: dict) -> str:
    """Deriva status del shadow book desde el pick_snapshot."""
    if pick.get('status') == _STATUS_NO_DATA:
        return _STATUS_NO_DATA
    if pick.get('apostar', False):
        return _STATUS_APROBADO
    return _STATUS_WATCHLIST


def _gate_from_pick(pick: dict) -> Optional[str]:
    """Extrae código de gate bloqueante del motivo_reclasificacion."""
    motivo = pick.get('motivo_reclasificacion') or ''
    if not motivo:
        return 'EDGE_INSUFICIENTE' if not pick.get('apostar') else None
    m = re.match(r'^([A-Z][A-Z0-9_\-]+)', motivo.strip())
    return m.group(1) if m else motivo[:20]


def _pick_partido_parts(pick: dict) -> Tuple[str, str]:
    """Extrae (p1, p2) del campo 'partido' del pick."""
    partido_str = pick.get('partido', '')
    if ' vs ' in partido_str:
        p1, p2 = partido_str.split(' vs ', 1)
        return p1.strip(), p2.strip()
    return '', ''


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DE REGISTROS (Addendum B.1 — pick_snapshot completo)
# ══════════════════════════════════════════════════════════════════════════════

def _build_record(pick: dict, fecha: str) -> Optional[dict]:
    """
    Construye registro JSONL.
    Addendum B.1: guarda el pick completo bajo 'pick_snapshot'.
    Solo calcula los campos de conveniencia top-level (sb_id, match_key, flags).
    Returns None si faltan campos mínimos para identificar el partido.
    """
    p1, p2 = _pick_partido_parts(pick)
    if not p1 or not p2:
        return None
    torneo = pick.get('torneo', 'Desconocido')
    superficie = pick.get('superficie', 'unknown')

    try:
        sb_id = _build_sb_id(fecha, torneo, p1, p2)
    except Exception:
        nombres = sorted([p1[:8].lower(), p2[:8].lower()])
        sb_id = f"{fecha}_{_slug(torneo, 12)}_{nombres[0]}-{nombres[1]}_ML"

    try:
        mk = _match_key(p1, p2)
    except Exception:
        mk = "_".join(sorted([p1[:8].lower(), p2[:8].lower()]))

    return {
        "sb_id":                  sb_id,
        "logged_at":              datetime.now().astimezone().isoformat(),
        "match_key":              mk,
        "es_qualifying":          _es_qualifying(torneo),
        "season_transition_flag": _season_transition(fecha, superficie),
        "pick_snapshot":          pick,   # Addendum B.1: dict completo, sin mutar
    }


def _build_session_meta(edge_report: dict, fecha: str, session_meta_in: dict) -> dict:
    """
    Addendum B.3: registro de sesión con métricas de nivel de sesión.
    Necesario para V-26-1 y V-26-5b.
    """
    apostar = edge_report.get('apostar') or []
    watchlist = edge_report.get('watchlist') or []
    no_data = edge_report.get('no_data') or []

    # cv_edge: coeficiente de variación del edge en todos los picks con edge > 0
    edges = [p.get('edge', 0) for p in (apostar + watchlist) if p.get('edge', 0) > 0]
    cv_edge = None
    if len(edges) >= 2:
        mean_e = statistics.mean(edges)
        if mean_e > 0:
            cv_edge = round(statistics.stdev(edges) / mean_e, 4)

    return {
        "_type":            "session_meta",
        "sb_id":            f"SESSION_{fecha}",
        "logged_at":        datetime.now().astimezone().isoformat(),
        "fecha":            fecha,
        "n_apostar":        len(apostar),
        "n_watchlist":      len(watchlist),
        "n_no_data":        len(no_data),
        "cv_edge":          cv_edge,
        # Addendum B.3: campos de contexto — None si no viene del caller (betplay_combo_builder)
        "dispersion_level": session_meta_in.get('dispersion_level'),
        "session_regime":   session_meta_in.get('session_regime'),
        "h2h_file":         session_meta_in.get('h2h_file'),
    }


# ══════════════════════════════════════════════════════════════════════════════
# JSONL I/O (append-only, upsert por sb_id)
# ══════════════════════════════════════════════════════════════════════════════

def _jsonl_path(fecha: str) -> str:
    return os.path.join(SHADOW_DIR, f"sb_{fecha}.jsonl")


def _load_jsonl(path: str) -> Dict[str, dict]:
    """Lee JSONL y retorna dict sb_id → record."""
    records: Dict[str, dict] = {}
    if not os.path.exists(path):
        return records
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sid = rec.get('sb_id')
                if sid:
                    records[sid] = rec
            except json.JSONDecodeError:
                continue
    return records


def _save_jsonl(path: str, records: Dict[str, dict]) -> None:
    """Escribe todos los registros al JSONL (uno por línea)."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for rec in records.values():
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


# ══════════════════════════════════════════════════════════════════════════════
# MOMENTO 2 — CLOSE SNAPSHOT (D52-03)
# ══════════════════════════════════════════════════════════════════════════════

def close_snapshot(fecha: Optional[str] = None) -> int:
    """
    Momento 2: captura cuotas de cierre Kambi ~15-30 min antes del inicio.
    Añade campo 'cierre_kambi' top-level a cada registro abierto SIN tocar pick_snapshot.

    Addendum A: reutiliza fetch_kambi_outcomes() (M-26-3, betplay_combo_builder) y
    el mismo name-matching 3-tier: nombre completo → apellido → _fuzzy_name_match (Nodo-36).

    Addendum §4.2: kambi_close es la fuente de ALTA calidad (mismo bookmaker que cuota_tomada).
    settle() la prioriza sobre flashscore_ref para calcular CLV.

    Returns: número de registros actualizados con cierre_kambi.
    """
    fecha = fecha or datetime.now().strftime('%Y-%m-%d')
    path = _jsonl_path(fecha)
    records = _load_jsonl(path)

    open_recs = [
        (sid, rec) for sid, rec in records.items()
        if rec.get('_type') != 'session_meta'
        and 'resolucion' not in rec
        and 'cierre_kambi' not in rec
    ]

    if not open_recs:
        logger.info("[ShadowBook] close_snapshot: sin registros abiertos")
        return 0

    # Fetch Kambi live odds (reutiliza lógica M-26-3)
    try:
        from betplay_combo_builder import fetch_kambi_outcomes
        outcomes_map, _ = fetch_kambi_outcomes()
    except Exception as e:
        logger.error(f"[ShadowBook] close_snapshot: error fetching Kambi: {e}")
        return 0

    if not outcomes_map:
        logger.warning("[ShadowBook] close_snapshot: outcomes_map vacío — ¿sesión terminada?")
        return 0

    # Normalizador consistente con betplay_combo_builder
    try:
        from scraping.kambi_tennis import _normalize_name as _norm_k
    except Exception:
        def _norm_k(x):
            return x.lower().strip()

    captured_at = datetime.now().astimezone().isoformat()
    count = 0

    for _sid, rec in open_recs:
        snap = rec.get('pick_snapshot', {})
        favorito = snap.get('favorito_predicho', '')
        if not favorito:
            continue

        # Tier 1: nombre completo normalizado
        norm_fav = _norm_k(favorito)
        found = outcomes_map.get(norm_fav)

        # Tier 2: apellido solo
        if not found:
            parts = norm_fav.split()
            apellido = parts[-1] if parts else ''
            if apellido:
                found = outcomes_map.get(apellido)

        # Tier 3: _name_tokens / _token_in_kb fallback (Nodo-36)
        if not found:
            for kb_key, outcome in outcomes_map.items():
                if _fuzzy_name_match(kb_key, favorito):
                    found = outcome
                    break

        if found:
            rec['cierre_kambi'] = {
                "cuota":       found.get('odds'),
                "captured_at": captured_at,
                "provenance":  "kambi_close",
            }
            count += 1
            logger.debug(
                f"[ShadowBook] cierre_kambi: {favorito} → {found.get('odds')}"
            )

    if count > 0:
        _save_jsonl(path, records)
        logger.info(f"[ShadowBook] close_snapshot: {count} cierres kambi → {path}")
    else:
        logger.warning("[ShadowBook] close_snapshot: 0 matches con Kambi — verificar nombres")

    return count


# ══════════════════════════════════════════════════════════════════════════════
# MOMENTO 1 — LOG PICKS
# ══════════════════════════════════════════════════════════════════════════════

def log_picks(edge_report: dict, session_meta: dict) -> int:
    """
    Momento 1: registra APROBADO + WATCHLIST + NO_DATA al shadow book.
    Upsert por sb_id — conserva logged_at original si ya existe (inmutabilidad §1).
    Escribe también un registro session_meta (Addendum B.3).

    Args:
        edge_report: output de edge_calculator.procesar_archivo_h2h()
        session_meta: {'fecha': 'YYYY-MM-DD', 'h2h_file': ..., 'dispersion_level': ..., ...}

    Returns: número de registros de pick NUEVOS escritos.
    """
    if not isinstance(edge_report, dict):
        logger.warning("[ShadowBook] log_picks: edge_report no es dict — ignorado")
        return 0

    fecha = session_meta.get('fecha') or datetime.now().strftime('%Y-%m-%d')

    # Addendum B.1: procesar todos los pools incluyendo no_data (F2)
    todos = (
        list(edge_report.get('apostar') or [])
        + list(edge_report.get('watchlist') or [])
        + list(edge_report.get('no_data') or [])
    )

    if not todos:
        logger.info("[ShadowBook] log_picks: sin picks en edge_report")
        return 0

    path = _jsonl_path(fecha)
    existing = _load_jsonl(path)
    nuevos = 0

    for pick in todos:
        if not isinstance(pick, dict):
            continue
        try:
            rec = _build_record(pick, fecha)
            if rec is None:
                continue
            if rec['sb_id'] in existing:
                continue  # Upsert: conservar original (inmutabilidad §1)
            existing[rec['sb_id']] = rec
            nuevos += 1
        except Exception as e:
            logger.warning(
                f"[ShadowBook] Error procesando pick '{pick.get('partido', '?')}': {e}"
            )

    # Addendum B.3: session_meta — upsert por sb_id SESSION_fecha
    try:
        sm_rec = _build_session_meta(edge_report, fecha, session_meta)
        sm_id = sm_rec['sb_id']
        if sm_id not in existing:
            existing[sm_id] = sm_rec
        else:
            # Actualizar conteos (puede re-correrse el mismo día con más picks)
            existing[sm_id].update({
                k: sm_rec[k] for k in ('n_apostar', 'n_watchlist', 'n_no_data', 'cv_edge')
            })
    except Exception as e:
        logger.warning(f"[ShadowBook] Error escribiendo session_meta: {e}")

    if nuevos > 0 or True:  # siempre guardar para actualizar session_meta
        _save_jsonl(path, existing)
        if nuevos > 0:
            logger.info(f"[ShadowBook] {nuevos} nuevos registros → {path}")

    return nuevos


# ══════════════════════════════════════════════════════════════════════════════
# CLV CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def calc_clv(cuota_tomada: float, cuota_cierre: float) -> float:
    """
    CLV% = (cuota_tomada / cuota_cierre − 1) × 100

    Addendum §4.2: NUNCA mezclar kambi_close con flashscore_ref en la misma métrica.
    Ej: @4.90 tomada, @3.80 cierre → CLV = +28.9%
    """
    if cuota_cierre <= 0:
        return 0.0
    return round((cuota_tomada / cuota_cierre - 1) * 100, 2)


def _compute_line_signal(rec: dict) -> str:
    """
    H52-05: deriva la señal de movimiento de línea para un registro settled.

    Misma lógica que line_movement_signal() de betplay_combo_builder (M-26-3, ±4%).
    Fuente preferida para cuota_cierre (Addendum §4.2, prioridad):
      1. cierre_kambi['cuota'] — Momento 2 (ALTA calidad: mismo bookmaker)
      2. resolucion['cuota_cierre'] con provenance != flashscore_ref si kambi_close
         o resolucion directamente (MEDIA: bookmaker distinto, pero proxy válido)

    Returns: "STEAM_IN" | "DRIFT_OUT" | "STABLE" | "NO_DATA"
    """
    cuota_original = rec.get('pick_snapshot', {}).get('cuota_favorito')
    if not cuota_original or cuota_original <= 0:
        return 'NO_DATA'

    # Prioridad 1: cierre_kambi (Momento 2)
    cierre_kambi = rec.get('cierre_kambi', {})
    cuota_cierre = cierre_kambi.get('cuota') if cierre_kambi else None

    # Prioridad 2: cuota_cierre del settlement
    if not cuota_cierre:
        res = rec.get('resolucion', {})
        cuota_cierre = res.get('cuota_cierre')

    if not cuota_cierre or cuota_cierre <= 0:
        return 'NO_DATA'

    delta_pct = (cuota_cierre - cuota_original) / cuota_original
    if delta_pct < -0.04:
        return 'STEAM_IN'
    elif delta_pct > 0.04:
        return 'DRIFT_OUT'
    return 'STABLE'


# ══════════════════════════════════════════════════════════════════════════════
# MOMENTO 3 — SETTLE (Addendum B.2: match_id primary + _name_tokens fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _fuzzy_name_match(candidate: str, pick_name: str) -> bool:
    """
    Addendum B.2: fallback fuzzy con _name_tokens/_token_in_kb (Nodo-36).
    Maneja acentos y apellidos de 2 caracteres correctamente.
    """
    try:
        from scraping.ninja_h2h_parser import _name_tokens, _token_in_kb
        toks = _name_tokens(pick_name)
        return bool(toks) and any(_token_in_kb(tok, candidate) for tok in toks)
    except Exception:
        # Fallback si el import falla
        p = pick_name.lower().strip()
        c = candidate.lower().strip()
        return p in c or c in p


def update_trader_stakes(fecha: str, trader_plan: dict) -> int:
    """
    P54-02 Parte 2: enriquece registros del shadow book con stakes reales del trader.
    Añade 'stake_real' y 'var_flattened' a cada pick APOSTAR del día.
    No modifica pick_snapshot (inmutabilidad §1) — escribe en campo 'trader_deploy'.

    Args:
        fecha: 'YYYY-MM-DD'
        trader_plan: dict con campo 'senales' (output de trader_ev_tenis.py JSON)

    Returns: número de registros actualizados.
    """
    path = _jsonl_path(fecha)
    records = _load_jsonl(path)
    if not records:
        logger.warning(f"[ShadowBook] update_trader_stakes: sin registros para {fecha}")
        return 0

    senales = trader_plan.get('senales', [])
    if not senales:
        logger.info("[ShadowBook] update_trader_stakes: sin senales en trader_plan")
        return 0

    # Índice por partido para match rápido
    senales_idx = {}
    for s in senales:
        partido = s.get('partido', '')
        match_id = s.get('match_id', '')
        if match_id:
            senales_idx[match_id] = s
        if partido:
            senales_idx[partido] = s

    actualizados = 0
    for sb_id, rec in records.items():
        if rec.get('record_type') == 'session_meta':
            continue
        snap = rec.get('pick_snapshot', {})
        match_id = snap.get('match_id', '')
        partido  = snap.get('partido', '')
        senal = senales_idx.get(match_id) or senales_idx.get(partido)
        if senal is None:
            continue
        wf = senal.get('_waterfall', {})
        stake_final = wf.get('stake_final', senal.get('stake', 0))
        var_flattened = wf.get('var_flattened', False)
        rec['trader_deploy'] = {
            'stake_real': stake_final,
            'var_flattened': var_flattened,
            'var_factor': wf.get('var_factor'),
            'terminal_reason': wf.get('terminal_reason'),
            'stake_pre_var': wf.get('stake_pre_var'),
            'updated_at': datetime.now().isoformat(),
        }
        actualizados += 1

    if actualizados > 0:
        _save_jsonl(path, records)
        logger.info(f"[ShadowBook] {actualizados} registros enriquecidos con trader_deploy → {path}")

    return actualizados


def settle(fecha: str, resultados_map: Optional[Dict] = None) -> int:
    """
    Momento 3: settlement post-match.
    Añade bloque 'resolucion' a cada registro abierto.
    Inmutabilidad §1: pick_snapshot NUNCA se toca.

    Addendum B.2 — orden de join:
      1. match_id exacto (pick_snapshot.match_id == fs_match.match_id)
      2. Fallback: _name_tokens/_token_in_kb (Nodo-36)

    Args:
        fecha: "YYYY-MM-DD"
        resultados_map: dict match_key → {ganador, cuota_cierre, provenance, void, match_id}
                        Si None: carga desde resultados_finales + FlashScore.

    Returns: número de registros settled.
    """
    path = _jsonl_path(fecha)
    records = _load_jsonl(path)

    # Filtrar session_meta — no se settle
    pick_records = {k: v for k, v in records.items() if v.get('_type') != 'session_meta'}

    if not pick_records:
        logger.warning(f"[ShadowBook] settle: sin registros pick para {fecha}")
        return 0

    if resultados_map is None:
        resultados_map = _load_resultados(fecha)

    # Construir índice por match_id para join primario (Addendum B.2)
    res_by_match_id: Dict[str, dict] = {}
    for res in resultados_map.values():
        mid = res.get('match_id')
        if mid:
            res_by_match_id[mid] = res

    settled_at = datetime.now().astimezone().isoformat()
    count = 0

    for sb_id, rec in records.items():
        if rec.get('_type') == 'session_meta':
            continue
        if 'resolucion' in rec:
            continue  # Ya settled — inmutabilidad §1

        snap = rec.get('pick_snapshot', {})
        mk = rec.get('match_key', '')

        # Addendum B.2 — Join primario: match_id
        pick_match_id = snap.get('match_id')
        res = None
        if pick_match_id and pick_match_id in res_by_match_id:
            res = res_by_match_id[pick_match_id]
        # Fallback: match_key
        elif mk and mk in resultados_map:
            res = resultados_map[mk]
        # Fallback: _name_tokens (Nodo-36)
        else:
            favorito = snap.get('favorito_predicho', '')
            if favorito:
                for res_candidate in resultados_map.values():
                    g = res_candidate.get('ganador', '') or ''
                    p1_fs = res_candidate.get('p1', '') or ''
                    p2_fs = res_candidate.get('p2', '') or ''
                    if (_fuzzy_name_match(p1_fs, favorito)
                            or _fuzzy_name_match(p2_fs, favorito)):
                        res = res_candidate
                        break

        if res is None:
            continue

        # Addendum §4.2: kambi_close (Momento 2) tiene prioridad sobre flashscore_ref
        # NUNCA mezclar provenances en la misma métrica
        cierre_kambi = rec.get('cierre_kambi', {})
        if cierre_kambi.get('cuota'):
            cuota_cierre_final = cierre_kambi['cuota']
            provenance_final = 'kambi_close'
        else:
            cuota_cierre_final = res.get('cuota_cierre')
            provenance_final = res.get('provenance', 'flashscore_ref')

        if res.get('void', False):
            rec['resolucion'] = {
                "settled_at":              settled_at,
                "resultado":               "VOID",
                "cuota_cierre":            None,
                "cuota_cierre_provenance": provenance_final,
                "clv_pct":                 None,
                "pnl_flat_1u":             0.0,
            }
        else:
            ganador = res.get('ganador', '')
            favorito = snap.get('favorito_predicho', '')
            resultado = 'WON' if _fuzzy_name_match(ganador, favorito) else 'LOST'
            cuota_tomada = snap.get('cuota_favorito', 0)
            clv = calc_clv(cuota_tomada, cuota_cierre_final) if cuota_cierre_final else None
            pnl = round(cuota_tomada - 1, 4) if resultado == 'WON' else -1.0

            rec['resolucion'] = {
                "settled_at":              settled_at,
                "resultado":               resultado,
                "cuota_cierre":            cuota_cierre_final,
                "cuota_cierre_provenance": provenance_final,
                "clv_pct":                 clv,
                "pnl_flat_1u":             pnl,
            }
        count += 1

    if count > 0:
        _save_jsonl(path, records)
        logger.info(f"[ShadowBook] settle: {count} registros settled → {path}")

    return count


def _load_resultados(fecha: str) -> Dict[str, dict]:
    """
    Carga resultados desde reports/resultados_finales_*.json + FlashScore (Nodo-48).
    Usado en producción; en tests se provee resultados_map directamente.
    """
    result_map: Dict[str, dict] = {}

    # 1. resultados_finales_*.json
    for fpath in sorted(glob_mod.glob("reports/resultados_finales_*.json")):
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            items = data if isinstance(data, list) else data.get('resultados', [])
            for r in items:
                r_fecha = str(r.get('fecha', '') or r.get('date', ''))
                if fecha not in r_fecha:
                    continue
                p1 = r.get('jugador1', '') or r.get('p1', '')
                p2 = r.get('jugador2', '') or r.get('p2', '')
                ganador = r.get('ganador', '') or r.get('resultado', '')
                if not p1 or not p2:
                    continue
                try:
                    mk = _match_key(p1, p2)
                    result_map[mk] = {
                        'ganador':    ganador,
                        'cuota_cierre': r.get('cuota_cierre'),
                        'provenance': 'resultados_finales',
                        'void':       r.get('void', False) or str(ganador).upper() == 'VOID',
                        'match_id':   r.get('match_id'),
                        'p1': p1, 'p2': p2,
                    }
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"[ShadowBook] Error leyendo {fpath}: {e}")

    # 2. FlashScore odds de cierre (Nodo-48)
    try:
        from scraping.kambi_tennis import extract_matches_flashscore_only
        target = datetime.strptime(fecha, "%Y-%m-%d").date()
        today = datetime.now().date()
        day_offset = (target - today).days
        _, fs_matches = extract_matches_flashscore_only(day_offset=day_offset)
        for m in fs_matches:
            p1 = m.get('jugador1', '')
            p2 = m.get('jugador2', '')
            if not p1 or not p2:
                continue
            try:
                mk = _match_key(p1, p2)
                ganador_fs = m.get('ganador', '')
                cuota_fs = m.get('cuota1')
                if mk not in result_map and ganador_fs:
                    result_map[mk] = {
                        'ganador':    ganador_fs,
                        'cuota_cierre': cuota_fs,
                        'provenance': 'flashscore_ref',
                        'void':       False,
                        'match_id':   m.get('match_id'),
                        'p1': p1, 'p2': p2,
                    }
                elif mk in result_map and not result_map[mk].get('cuota_cierre') and cuota_fs:
                    result_map[mk]['cuota_cierre'] = cuota_fs
            except Exception:
                continue
    except Exception as e:
        logger.debug(f"[ShadowBook] FlashScore fallback error: {e}")

    return result_map


# ══════════════════════════════════════════════════════════════════════════════
# WILSON CI — implementación directa, sin dependencia nueva
# ══════════════════════════════════════════════════════════════════════════════

def wilson_ci(n: int, hits: int, z: float = _Z95) -> Tuple[float, float]:
    """
    Wilson score interval al nivel z (default 95%).
    Mejor que normal approximation con n pequeño (spec §5).
    Returns: (lower_pct, upper_pct) en porcentaje [0.0, 100.0].

    Verificación: n=34, hits=20 → [42.2, 73.6]
    """
    if n <= 0:
        return (0.0, 100.0)
    p_hat = hits / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n)) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (round(lower * 100, 1), round(upper * 100, 1))


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS DE SEGMENTO (Addendum C — REGLA-T27-2)
# ══════════════════════════════════════════════════════════════════════════════

def _segment_metrics(records: List[dict]) -> dict:
    """
    Calcula métricas de un conjunto de registros settled.
    REGLA-T27-2: bins con n<10 son válidos pero se marcan '*' en el reporte.
    ROI siempre flat 1u (Addendum C).
    """
    non_void = [r for r in records if r.get('resolucion', {}).get('resultado') != 'VOID']
    n_void = len(records) - len(non_void)
    n = len(non_void)

    if n == 0:
        return {
            "n": 0, "void": n_void, "sparse": False,
            "hits": 0, "hit_pct": 0.0,
            "roi": 0.0, "clv_median": None,
            "ic": (0.0, 100.0), "breakeven": None,
        }

    hits = sum(1 for r in non_void if r.get('resolucion', {}).get('resultado') == 'WON')
    pnl_sum = sum(r.get('resolucion', {}).get('pnl_flat_1u', 0) for r in non_void)
    roi = round(pnl_sum / n * 100, 1)

    clv_vals = [
        r['resolucion']['clv_pct']
        for r in non_void
        if r.get('resolucion', {}).get('clv_pct') is not None
    ]
    clv_median = None
    if clv_vals:
        sv = sorted(clv_vals)
        mid = len(sv) // 2
        clv_median = round(
            (sv[mid - 1] + sv[mid]) / 2 if len(sv) % 2 == 0 else sv[mid], 2
        )

    ic = wilson_ci(n, hits)
    cuotas = [
        r.get('pick_snapshot', {}).get('cuota_favorito', 0)
        for r in non_void
        if r.get('pick_snapshot', {}).get('cuota_favorito', 0) > 0
    ]
    breakeven = round(100 / (sum(cuotas) / len(cuotas)), 1) if cuotas else None

    return {
        "n": n, "void": n_void,
        "sparse": n < 10,  # REGLA-T27-2: marcar con '*'
        "hits": hits,
        "hit_pct": round(hits / n * 100, 1),
        "roi": roi,
        "clv_median": clv_median,
        "ic": ic,
        "breakeven": breakeven,
    }


def _graduated(m: dict) -> bool:
    """
    Criterios de graduación §6 (todos simultáneamente):
    1. n ≥ 30
    2. IC Wilson lower > breakeven (1/cuota_media)
    3. CLV mediano > 0
    """
    if m['n'] < 30:
        return False
    ic_lower = m['ic'][0]
    be = m.get('breakeven')
    if be is None or ic_lower <= be:
        return False
    return (m.get('clv_median') or 0) > 0


# ══════════════════════════════════════════════════════════════════════════════
# REPORT — Sección S-27-8 (Addendum §G.4)
# ══════════════════════════════════════════════════════════════════════════════

def _pick_status_sb(rec: dict) -> str:
    """Deriva status del pick desde el registro del shadow book."""
    snap = rec.get('pick_snapshot', {})
    return _sb_status(snap)


def report(desde: Optional[str] = None, hasta: Optional[str] = None) -> str:
    """
    Sección S-27-8: métricas del shadow book por segmento pre-registrado.
    REGLA-T27-2: toda tabla muestra n; bins con n<10 marcados con '*'.
    Addendum D: hipótesis H52-01 a H52-08.
    """
    today = datetime.now().strftime('%Y-%m-%d')
    hasta = hasta or today
    desde = desde or today[:8] + "01"

    all_records: List[dict] = []
    for fpath in sorted(glob_mod.glob(os.path.join(SHADOW_DIR, "sb_*.jsonl"))):
        fname = os.path.basename(fpath)
        m = re.match(r'sb_(\d{4}-\d{2}-\d{2})\.jsonl', fname)
        if not m:
            continue
        if not (desde <= m.group(1) <= hasta):
            continue
        all_records.extend(_load_jsonl(fpath).values())

    # Separar session_meta de picks
    session_metas = [r for r in all_records if r.get('_type') == 'session_meta']
    picks = [r for r in all_records if r.get('_type') != 'session_meta']
    settled = [r for r in picks if 'resolucion' in r]

    n_total = len(picks)
    n_settled = len(settled)

    sep = "═" * 62
    lines: List[str] = [sep]
    lines.append(f"  S-27-8  SHADOW BOOK — {desde} → {hasta}")
    lines.append(sep)
    lines.append(f"  Registros: {n_total}  |  Settled: {n_settled}  |  Abiertos: {n_total - n_settled}")
    if session_metas:
        last_sm = session_metas[-1]
        lines.append(
            f"  Sesiones: {len(session_metas)}  |  "
            f"cv_edge últ: {last_sm.get('cv_edge', 'N/A')}  |  "
            f"regime últ: {last_sm.get('session_regime', 'N/A')}"
        )
    lines.append("")

    if n_settled == 0:
        lines.append("  Sin registros settled.")
        lines.append("  Correr: python3 shadow_book.py --settle FECHA")
        lines.append(sep)
        return "\n".join(lines)

    # ── Segmentos por status ──
    for label, pred in [
        (_STATUS_APROBADO, lambda r: _pick_status_sb(r) == _STATUS_APROBADO),
        (_STATUS_WATCHLIST, lambda r: _pick_status_sb(r) == _STATUS_WATCHLIST),
        (_STATUS_NO_DATA, lambda r: _pick_status_sb(r) == _STATUS_NO_DATA),
    ]:
        _append_segment(settled, lines, f"status={label}", pred)

    # ── Segmentos por tier ──
    for tier in ('grand_slam', 'atp1000', 'atp500', 'challenger', 'itf'):
        _append_segment(settled, lines, f"tier={tier}",
                        lambda r, t=tier: r.get('pick_snapshot', {}).get('tier') == t)

    # ── Segmentos por qualifying / season_transition ──
    _append_segment(settled, lines, "es_qualifying=true",
                    lambda r: r.get('es_qualifying', False))
    _append_segment(settled, lines, "season_transition=true",
                    lambda r: r.get('season_transition_flag', False))

    # ── D54-02: WATCHLIST ∩ tier=grand_slam ∩ edge≥20% (Nodo-55 P54-03) ──
    # Intersección de cortes ya pre-registrados: status × tier × banda de cuota.
    # NO es segmento nuevo — es visualización de uno existente para responder
    # la pregunta de P54-03 sin crear un gate nuevo.
    _append_segment(
        settled, lines,
        "WATCHLIST+grand_slam+edge>=20%",
        lambda r: (
            _pick_status_sb(r) == _STATUS_WATCHLIST
            and r.get('pick_snapshot', {}).get('tier') == 'grand_slam'
            and r.get('pick_snapshot', {}).get('edge', 0) >= 0.20
        ),
    )

    # ── Hipótesis pre-registradas (Addendum §D, congeladas 2026-07-02) ──
    lines.append("  HIPÓTESIS (congeladas 2026-07-02):")

    def _has_was(r):
        snap = r.get('pick_snapshot', {})
        return snap.get('edge', 0) >= 0.10 and snap.get('cuota_favorito', 0) >= 2.0 and snap.get('markov_favorito') == 'HOT'

    def _is_structural_alpha(r):
        return r.get('pick_snapshot', {}).get('alignment_flag') == 'STRUCTURAL_ALPHA'

    def _is_low_confidence(r):
        return r.get('pick_snapshot', {}).get('confidence_flag') == 'LOW'

    def _is_n_h2h_0_itf(r):
        snap = r.get('pick_snapshot', {})
        return snap.get('n_h2h', 1) == 0 and snap.get('tier') == 'itf'

    def _is_qualifying_low_p(r):
        snap = r.get('pick_snapshot', {})
        p = snap.get('p_modelo', 0)
        return r.get('es_qualifying', False) and 0.52 <= p < 0.55

    def _is_main_mid_p(r):
        snap = r.get('pick_snapshot', {})
        p = snap.get('p_modelo', 0)
        return not r.get('es_qualifying', False) and 0.55 <= p < 0.60

    def _is_zona_2_25(r):
        snap = r.get('pick_snapshot', {})
        c = snap.get('cuota_favorito', 0)
        return 2.00 <= c < 2.50

    _append_hypothesis(settled, lines, "H52-01", "WAS supera breakeven", _has_was, n_stop=30)
    _append_hypothesis(settled, lines, "H52-02", "n_h2h=0+ITF discrimina",
                       _is_n_h2h_0_itf, n_stop=30)
    _append_hypothesis(settled, lines, "H52-03", "STRUCTURAL_ALPHA > LOW",
                       lambda r: _is_structural_alpha(r) or _is_low_confidence(r), n_stop=20)
    _append_hypothesis_h52_05(settled, lines)
    _append_hypothesis(settled, lines, "H52-07", "Qualifying p[0.52-0.55) vs principal",
                       _is_qualifying_low_p, n_stop=50)
    _append_hypothesis(settled, lines, "H52-08", "Zona 2.00-2.50 trampa post-fixes",
                       _is_zona_2_25, n_stop=30)

    # ── Graduación ──
    lines.append("")
    lines.append("  GRADUACIÓN (n≥30 + IC_lower>breakeven + CLV_median>0):")
    any_grad = False
    for g_label, g_pred in [
        (_STATUS_APROBADO, lambda r: _pick_status_sb(r) == _STATUS_APROBADO),
        ("WAS", _has_was),
    ]:
        seg = [r for r in settled if g_pred(r)]
        if not seg:
            continue
        gm = _segment_metrics(seg)
        if _graduated(gm):
            lines.append(
                f"    GRADUADO: {g_label} — n={gm['n']}, "
                f"IC=[{gm['ic'][0]},{gm['ic'][1]}], CLV_med={gm['clv_median']}"
            )
            any_grad = True
    if not any_grad:
        nearest_n = max(
            len([r for r in settled if _pick_status_sb(r) == _STATUS_APROBADO]),
            len([r for r in settled if _has_was(r)]),
        )
        lines.append(f"    Sin segmento graduado. Más cercano: n={nearest_n}/30")

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


def _append_segment(settled: list, lines: list, label: str, pred) -> None:
    """Añade sección de segmento al reporte. REGLA-T27-2: n<10 → '*'."""
    seg = [r for r in settled if pred(r)]
    if not seg:
        return
    m = _segment_metrics(seg)
    if m['n'] == 0:
        return
    sparse_mark = '*' if m['sparse'] else ''
    ic_l, ic_u = m['ic']
    be = m.get('breakeven', '?')
    lines.append(f"  SEGMENTO: {label}{sparse_mark}")
    lines.append(f"    n={m['n']}  hit%={m['hit_pct']}  IC95=[{ic_l}, {ic_u}]  breakeven={be}")
    lines.append(f"    ROI flat 1u: {m['roi']}%   CLV mediano: {m['clv_median']}")
    if m['void'] > 0:
        lines.append(f"    VOID excluidos: {m['void']}")
    lines.append("")


def _append_hypothesis(settled: list, lines: list, h_id: str, label: str, pred, n_stop: int) -> None:
    """Añade línea de hipótesis al reporte."""
    seg = [r for r in settled if pred(r)]
    m = _segment_metrics(seg)
    n = m['n']
    if n >= n_stop:
        conclusion = (
            f"GRADUABLE — IC=[{m['ic'][0]},{m['ic'][1]}] breakeven={m.get('breakeven', '?')}"
            if _graduated(m) else
            f"NO GRADUABLE — IC=[{m['ic'][0]},{m['ic'][1]}] breakeven={m.get('breakeven', '?')}"
        )
    else:
        conclusion = f"CONTINUAR (n={n}/{n_stop})"
    sparse_mark = '*' if m['sparse'] else ''
    lines.append(f"    {h_id} [{label}]{sparse_mark}: {conclusion}")


def _append_hypothesis_h52_05(settled: list, lines: list) -> None:
    """
    H52-05: STEAM_IN hit% > DRIFT_OUT hit% (V-26-3d, n_stop=20 picks con delta).

    Compara dos grupos mutuamente excluyentes — estructura distinta a _append_hypothesis().
    Solo picks donde _compute_line_signal() != NO_DATA|STABLE (tienen delta ≥4%).
    REGLA-T27-2: grupos con n<10 marcados con '*'.
    """
    N_STOP = 20

    steam = [r for r in settled if _compute_line_signal(r) == 'STEAM_IN']
    drift  = [r for r in settled if _compute_line_signal(r) == 'DRIFT_OUT']
    n_con_delta = len(steam) + len(drift)

    if n_con_delta == 0:
        lines.append("    H52-05 [STEAM_IN > DRIFT_OUT hit%]: CONTINUAR (n=0/20, sin picks con delta≥4%)")
        return

    ms = _segment_metrics(steam)
    md = _segment_metrics(drift)

    sparse_s = '*' if ms['sparse'] else ''
    sparse_d = '*' if md['sparse'] else ''

    if n_con_delta >= N_STOP:
        # Evaluación: STEAM_IN hit% > DRIFT_OUT hit%
        steam_hit = ms['hit_pct']
        drift_hit = md['hit_pct']
        if ms['n'] > 0 and md['n'] > 0:
            if steam_hit > drift_hit:
                conclusion = (
                    f"CONFIRMADA — STEAM={steam_hit}%{sparse_s} > DRIFT={drift_hit}%{sparse_d} "
                    f"(n_steam={ms['n']}, n_drift={md['n']})"
                )
            else:
                conclusion = (
                    f"NO CONFIRMADA — STEAM={steam_hit}%{sparse_s} <= DRIFT={drift_hit}%{sparse_d} "
                    f"(n_steam={ms['n']}, n_drift={md['n']})"
                )
        elif ms['n'] > 0:
            conclusion = f"STEAM={steam_hit}%{sparse_s}, sin picks DRIFT — no evaluable"
        else:
            conclusion = f"Sin picks STEAM — no evaluable (n_drift={md['n']})"
    else:
        steam_str = f"STEAM: n={ms['n']}{sparse_s} hit%={ms['hit_pct']}" if ms['n'] else "STEAM: n=0"
        drift_str = f"DRIFT: n={md['n']}{sparse_d} hit%={md['hit_pct']}" if md['n'] else "DRIFT: n=0"
        conclusion = f"CONTINUAR (n_delta={n_con_delta}/{N_STOP}) — {steam_str} | {drift_str}"

    lines.append(f"    H52-05 [STEAM_IN > DRIFT_OUT hit%]: {conclusion}")


# ══════════════════════════════════════════════════════════════════════════════
# D58-01: report_dict — expone métricas como dict para el dashboard
# ══════════════════════════════════════════════════════════════════════════════

def report_dict(desde: Optional[str] = None, hasta: Optional[str] = None) -> dict:
    """
    D58-01: Expone métricas del shadow book como dict para el dashboard.
    REGLA: única fuente de verdad — mismos helpers que report().
    Comparte _segment_metrics, _graduated, _compute_line_signal.
    """
    today = datetime.now().strftime('%Y-%m-%d')
    hasta = hasta or today
    desde = desde or today[:8] + "01"

    all_records: List[dict] = []
    for fpath in sorted(glob_mod.glob(os.path.join(SHADOW_DIR, "sb_*.jsonl"))):
        fname = os.path.basename(fpath)
        m_re = re.match(r'sb_(\d{4}-\d{2}-\d{2})\.jsonl', fname)
        if not m_re:
            continue
        if not (desde <= m_re.group(1) <= hasta):
            continue
        all_records.extend(_load_jsonl(fpath).values())

    session_metas = [r for r in all_records if r.get('_type') == 'session_meta']
    picks = [r for r in all_records if r.get('_type') != 'session_meta']
    settled = [r for r in picks if 'resolucion' in r]

    n_total = len(picks)
    n_settled = len(settled)

    result: dict = {
        'range': {'desde': desde, 'hasta': hasta},
        'summary': {'n_total': n_total, 'n_settled': n_settled, 'n_open': n_total - n_settled},
        'sessions': {
            'n': len(session_metas),
            'last_cv_edge': session_metas[-1].get('cv_edge') if session_metas else None,
            'last_regime': session_metas[-1].get('session_regime') if session_metas else None,
        },
        'segments': [],
        'hypotheses': [],
        'graduation': {'any_graduated': False, 'nearest_n': 0, 'graduated_labels': []},
        'clv_by_provenance': {},
    }

    if n_settled == 0:
        return result

    def _seg(label: str, pred) -> Optional[dict]:
        seg = [r for r in settled if pred(r)]
        if not seg:
            return None
        m = _segment_metrics(seg)
        if m['n'] == 0:
            return None
        return {'label': label, **m, 'ic': list(m['ic'])}

    # Segmentos por status
    for label, pred in [
        (_STATUS_APROBADO, lambda r: _pick_status_sb(r) == _STATUS_APROBADO),
        (_STATUS_WATCHLIST, lambda r: _pick_status_sb(r) == _STATUS_WATCHLIST),
        (_STATUS_NO_DATA, lambda r: _pick_status_sb(r) == _STATUS_NO_DATA),
    ]:
        s = _seg(f"status={label}", pred)
        if s:
            result['segments'].append(s)

    # Segmentos por tier
    for tier in ('grand_slam', 'atp1000', 'atp500', 'challenger', 'itf'):
        s = _seg(f"tier={tier}", lambda r, t=tier: r.get('pick_snapshot', {}).get('tier') == t)
        if s:
            result['segments'].append(s)

    # Qualifying / season_transition / watchlist GS
    for label, pred in [
        ("es_qualifying=true", lambda r: r.get('es_qualifying', False)),
        ("season_transition=true", lambda r: r.get('season_transition_flag', False)),
        ("WATCHLIST+grand_slam+edge>=20%", lambda r: (
            _pick_status_sb(r) == _STATUS_WATCHLIST
            and r.get('pick_snapshot', {}).get('tier') == 'grand_slam'
            and r.get('pick_snapshot', {}).get('edge', 0) >= 0.20
        )),
    ]:
        s = _seg(label, pred)
        if s:
            result['segments'].append(s)

    # ── Hipótesis ──
    def _hyp(h_id: str, label: str, pred, n_stop: int) -> dict:
        seg = [r for r in settled if pred(r)]
        m = _segment_metrics(seg)
        n = m['n']
        if n >= n_stop:
            estado = "GRADUABLE" if _graduated(m) else "NO_GRADUABLE"
        else:
            estado = "CONTINUAR"
        return {
            'id': h_id, 'label': label, 'n': n, 'n_stop': n_stop,
            'hits': m['hits'], 'hit_pct': m['hit_pct'],
            'roi': m['roi'], 'clv_median': m['clv_median'],
            'ic': list(m['ic']), 'breakeven': m['breakeven'],
            'sparse': m['sparse'], 'estado': estado,
            'graduado': _graduated(m),
        }

    def _has_was_d(r):
        snap = r.get('pick_snapshot', {})
        return snap.get('edge', 0) >= 0.10 and snap.get('cuota_favorito', 0) >= 2.0 and snap.get('markov_favorito') == 'HOT'

    def _is_n_h2h_0_itf_d(r):
        snap = r.get('pick_snapshot', {})
        return snap.get('n_h2h', 1) == 0 and snap.get('tier') == 'itf'

    def _is_struct_or_low_d(r):
        snap = r.get('pick_snapshot', {})
        return snap.get('alignment_flag') == 'STRUCTURAL_ALPHA' or snap.get('confidence_flag') == 'LOW'

    def _is_qualifying_low_p_d(r):
        snap = r.get('pick_snapshot', {})
        p = snap.get('p_modelo', 0)
        return r.get('es_qualifying', False) and 0.52 <= p < 0.55

    def _is_zona_2_25_d(r):
        snap = r.get('pick_snapshot', {})
        c = snap.get('cuota_favorito', 0)
        return 2.00 <= c < 2.50

    def _is_var_flattened_d(r):
        td = r.get('trader_deploy', {})
        snap = r.get('pick_snapshot', {})
        return bool(td.get('var_flattened')) and bool(snap.get('apostar'))

    result['hypotheses'] = [
        _hyp("H52-01", "WAS supera breakeven", _has_was_d, 30),
        _hyp("H52-02", "n_h2h=0+ITF discrimina", _is_n_h2h_0_itf_d, 30),
        _hyp("H52-03", "STRUCTURAL_ALPHA+LOW combinado", _is_struct_or_low_d, 20),
        _hyp("H52-07", "Qualifying p[0.52-0.55) vs principal", _is_qualifying_low_p_d, 50),
        _hyp("H52-08", "Zona 2.00-2.50 trampa post-fixes", _is_zona_2_25_d, 30),
        _hyp("H54-01", "stake=0 calidad igual que financiados", _is_var_flattened_d, 30),
    ]

    # H52-05: dos grupos (STEAM vs DRIFT)
    steam = [r for r in settled if _compute_line_signal(r) == 'STEAM_IN']
    drift = [r for r in settled if _compute_line_signal(r) == 'DRIFT_OUT']
    ms = _segment_metrics(steam)
    md = _segment_metrics(drift)
    n_delta = ms['n'] + md['n']
    if n_delta >= 20:
        if ms['n'] > 0 and md['n'] > 0:
            h52_05_estado = "GRADUABLE" if ms['hit_pct'] > md['hit_pct'] else "NO_GRADUABLE"
        else:
            h52_05_estado = "NO_GRADUABLE"
    else:
        h52_05_estado = "CONTINUAR"
    result['hypotheses'].append({
        'id': 'H52-05', 'label': 'STEAM_IN > DRIFT_OUT hit%',
        'n': n_delta, 'n_stop': 20, 'estado': h52_05_estado, 'graduado': False,
        'steam': {'n': ms['n'], 'hit_pct': ms['hit_pct'], 'ic': list(ms['ic'])},
        'drift': {'n': md['n'], 'hit_pct': md['hit_pct'], 'ic': list(md['ic'])},
    })

    # ── Graduación ──
    any_grad = False
    nearest_n = 0
    graduated_labels: List[str] = []
    for g_label, g_pred in [
        (_STATUS_APROBADO, lambda r: _pick_status_sb(r) == _STATUS_APROBADO),
        ("WAS", _has_was_d),
    ]:
        seg = [r for r in settled if g_pred(r)]
        gm = _segment_metrics(seg)
        nearest_n = max(nearest_n, gm['n'])
        if _graduated(gm):
            any_grad = True
            graduated_labels.append(g_label)
    result['graduation'] = {
        'any_graduated': any_grad,
        'nearest_n': nearest_n,
        'graduated_labels': graduated_labels,
    }

    # ── CLV por provenance (D52-08) ──
    non_void = [r for r in settled if r.get('resolucion', {}).get('resultado') != 'VOID']
    clv_prov: dict = {}
    for r in non_void:
        res = r.get('resolucion', {})
        prov = res.get('cuota_cierre_provenance', 'unknown')
        clv_pct = res.get('clv_pct')
        if prov not in clv_prov:
            clv_prov[prov] = {'n': 0, 'vals': []}
        clv_prov[prov]['n'] += 1
        if clv_pct is not None:
            clv_prov[prov]['vals'].append(clv_pct)

    clv_by_prov: dict = {}
    for prov, data in clv_prov.items():
        vals = sorted(data['vals'])
        mid = len(vals) // 2
        median = None
        if vals:
            median = round((vals[mid - 1] + vals[mid]) / 2 if len(vals) % 2 == 0 else vals[mid], 2)
        clv_by_prov[prov] = {
            'n': data['n'],
            'n_clv': len(vals),
            'clv_median': median,
            'excluded': prov == 'kambi_inplay',
        }
    result['clv_by_provenance'] = clv_by_prov

    return result


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Shadow Book — Nodo-52 CLV Tracking")
    parser.add_argument('--settle', type=str, metavar='FECHA',
                        help="Settle picks para FECHA (YYYY-MM-DD)")
    parser.add_argument('--report', action='store_true',
                        help="Generar reporte S-27-8 de métricas por segmento")
    parser.add_argument('--desde', type=str, default=None,
                        help="Fecha inicio para reporte (YYYY-MM-DD)")
    parser.add_argument('--hasta', type=str, default=None,
                        help="Fecha fin para reporte (YYYY-MM-DD)")
    parser.add_argument('--close-snapshot', action='store_true',
                        help="Momento 2: capturar cuotas de cierre Kambi ~15-30min antes del inicio")
    parser.add_argument('--fecha', type=str, default=None,
                        help="Fecha para --close-snapshot (YYYY-MM-DD, default: hoy)")
    parser.add_argument('--json', action='store_true', dest='json_output',
                        help="D58-01: Output métricas como JSON para el dashboard")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    if args.json_output:
        import json as _json
        print(_json.dumps(report_dict(desde=args.desde, hasta=args.hasta), ensure_ascii=False, indent=2))
    elif args.settle:
        n = settle(args.settle)
        print(f"Settled: {n} registros para {args.settle}")
    elif args.report:
        print(report(desde=args.desde, hasta=args.hasta))
    elif args.close_snapshot:
        n = close_snapshot(fecha=args.fecha)
        print(f"Cierre Kambi capturado: {n} registros para {args.fecha or 'hoy'}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
