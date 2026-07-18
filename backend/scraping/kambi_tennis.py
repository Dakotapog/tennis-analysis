"""
🎾 Kambi Tennis — Cuotas reales de Betplay + match_ids de FlashScore

Puente NBA→Tenis: misma API Kambi, mismo patrón.
Reemplaza Playwright para PASO 1 del pipeline.

Dos fuentes, un merge:
  Kambi API:         jugadores (nombre completo) + cuotas reales + torneo + hora
  FlashScore feed:   jugadores (abreviado) + match_id + rankings + superficie

Output: data/zita_tennis_matches_FECHA.json — mismo formato que extraer_URL_partidos_version2.py
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from config import FLASHSCORE_BASE, FLASHSCORE_HEADERS
from core.tournament_context import build_tournament_context

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# KAMBI API — Cuotas reales Betplay
# ══════════════════════════════════════════════════════════════════════════════

KAMBI_BASE = "https://us.offering-api.kambicdn.com/offering/v2018/betplay"
KAMBI_PARAMS = "lang=es_CO&market=CO&channel_id=1&client_id=2"
KAMBI_HEADERS = {
    "Referer": "https://betplay.com.co/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

# Mapeo tier Kambi → tier pipeline
KAMBI_TIER_MAP = {
    "atp": "atp",
    "wta": "wta",
    "challenger": "challenger",
    "challenger_qual_": "challenger",
    "wta125": "wta125",
    "wta125_qual_": "wta125",
    "itf_men": "itf",
    "itf_men_qual_": "itf",
    "itf_women": "itf",
    "itf_women_qual_": "itf",
    "atp_doubles": "doubles",
    "wta_doubles": "doubles",
}


def fetch_kambi_tennis() -> List[Dict]:
    """
    Obtiene TODOS los partidos de tenis con cuotas de Betplay.

    Returns:
        Lista de dicts con: jugador1, jugador2, cuota1, cuota2, torneo, hora, tier, kambi_id
    """
    url = f"{KAMBI_BASE}/listView/tennis.json?{KAMBI_PARAMS}"

    try:
        resp = requests.get(url, headers=KAMBI_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"❌ Error consultando Kambi API: {e}")
        return []

    events = data.get("events", [])
    logger.info(f"⚡ Kambi API: {len(events)} eventos de tenis")

    matches = []
    for ev_wrapper in events:
        ev = ev_wrapper.get("event", {})
        offers = ev_wrapper.get("betOffers", [])

        # Solo partidos no iniciados
        if ev.get("state") != "NOT_STARTED":
            continue

        # Extraer tier desde path
        path = ev.get("path", [])
        tier_key = path[1].get("termKey", "") if len(path) > 1 else ""
        tier = KAMBI_TIER_MAP.get(tier_key, tier_key)

        # Excluir dobles
        if tier == "doubles":
            continue

        # Extraer cuotas del Match Odds — usar type para mapear correctamente
        # Kambi NO garantiza outcomes[0]=home: a veces outcomes[0]=OT_TWO (away)
        cuota1 = cuota2 = None
        if offers:
            outcomes = offers[0].get("outcomes", [])
            if len(outcomes) >= 2:
                for oc in outcomes:
                    odds_val = oc.get("odds", 0) / 1000
                    if oc.get("type") == "OT_ONE":
                        cuota1 = odds_val  # homeName
                    elif oc.get("type") == "OT_TWO":
                        cuota2 = odds_val  # awayName

        if not cuota1 or not cuota2:
            continue

        tournament = path[2].get("name", "Unknown") if len(path) > 2 else "Unknown"

        matches.append({
            "jugador1": ev.get("homeName", ""),
            "jugador2": ev.get("awayName", ""),
            "cuota1": cuota1,
            "cuota2": cuota2,
            "torneo_kambi": tournament,
            "tier": tier,
            "hora": ev.get("start", ""),
            "kambi_event_id": ev.get("id"),
            "kambi_name": ev.get("name", ""),
        })

    logger.info(f"   ✅ {len(matches)} singles con cuotas (excluidos dobles + en vivo)")
    return matches


# ══════════════════════════════════════════════════════════════════════════════
# FLASHSCORE FEED — match_ids + rankings + superficie
# ══════════════════════════════════════════════════════════════════════════════

def fetch_flashscore_feed(day_offset: int = 0) -> List[Dict]:
    """
    Obtiene partidos de tenis del feed FlashScore.

    Args:
        day_offset: 0=hoy, 1=mañana, -1=ayer

    Returns:
        Lista de dicts con: jugador1, jugador2, match_id, ranking1, ranking2, torneo, superficie
    """
    ep = f"f_2_{day_offset}_13_es_1"
    url = f"{FLASHSCORE_BASE}/{ep}"

    try:
        resp = requests.get(url, headers=FLASHSCORE_HEADERS, timeout=15)
        if resp.status_code != 200:
            logger.error(f"❌ FlashScore feed HTTP {resp.status_code}")
            return []
    except Exception as e:
        logger.error(f"❌ Error consultando FlashScore feed: {e}")
        return []

    sections = resp.text.split("~")
    logger.info(f"⚡ FlashScore feed (day={day_offset}): {len(sections)} secciones")

    matches = []
    current_tournament = ""
    current_surface = ""

    for sec in sections:
        fields = {}
        for pair in sec.split("¬"):
            if "÷" in pair:
                k, v = pair.split("÷", 1)
                fields[k] = v

        # Tournament header
        if "ZA" in fields:
            current_tournament = fields["ZA"]
            # Extraer superficie del nombre del torneo
            t_lower = current_tournament.lower()
            if "arcilla" in t_lower or "clay" in t_lower:
                current_surface = "clay"
            elif "hierba" in t_lower or "grass" in t_lower:
                current_surface = "grass"
            elif "dura" in t_lower or "hard" in t_lower:
                current_surface = "hard"
            else:
                current_surface = "unknown"
            continue

        # Match record
        if "AA" not in fields:
            continue

        # Solo individuales
        if "DOBLES" in current_tournament or "DOUBLES" in current_tournament:
            continue

        matches.append({
            "jugador1_fs": fields.get("AE", ""),
            "jugador2_fs": fields.get("AF", ""),
            "match_id": fields.get("AA", ""),
            # NOTA B117-01 (Nodo-117): CA/CB son rankings del feed FlashScore (sistema interno),
            # NO posiciones ATP/WTA oficiales. No usar para gaps de ranking — ver
            # data/atp_rankings_complete_*.json y data/wta_rankings_complete_*.json.
            "ranking1": _safe_int(fields.get("CA")),
            "ranking2": _safe_int(fields.get("CB")),
            "torneo_fs": current_tournament,
            "superficie": current_surface,
            "pais1": fields.get("FU", ""),
            "pais2": fields.get("FV", ""),
            "slug1": fields.get("WU", ""),
            "slug2": fields.get("WV", ""),
        })

    logger.info(f"   ✅ {len(matches)} singles en FlashScore feed")
    return matches


def _safe_int(val) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


# ══════════════════════════════════════════════════════════════════════════════
# FUZZY MATCHING — Cruce por nombre de jugador
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_name(name: str) -> str:
    """
    Normaliza nombre para matching: lowercase, sin acentos, sin puntuación.
    B108-03: delega a core.player_registry.normalize_player_name (fuente canónica).
    Fallback inline si el módulo no está disponible (ej. tests aislados).
    """
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _root = str(_Path(__file__).parent.parent)
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from core.player_registry import normalize_player_name
        return normalize_player_name(name)
    except Exception:
        import unicodedata
        name = unicodedata.normalize("NFD", name.lower())
        name = "".join(c for c in name if unicodedata.category(c) != "Mn")
        name = re.sub(r"[^a-z\s]", "", name)
        return name.strip()


# Sufijos que no son parte del apellido (patrón NBA: _parse_nombre)
_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv"})


def _parse_nombre(nombre: str) -> Tuple[str, str]:
    """
    Extrae (apellido, inicial) de cualquier formato de nombre.
    Patrón probado en NBA (decidir_hoy.py _parse_nombre).

    Maneja:
      "Nuno Borges"                    → ("borges", "n")
      "Borges N."                      → ("borges", "n")
      "Alejandro Davidovich Fokina"    → ("fokina", "a")
      "Davidovich A."                  → ("davidovich", "a")   ← este es el problema
      "Pierre-Hugues Herbert"          → ("herbert", "p")
      "Herbert P.H."                   → ("herbert", "p")
      "Botic van de Zandschulp"        → ("zandschulp", "b")
      "Van De Zandschulp B."           → ("zandschulp", "b")
      "Giovanni Mpetshi Perricard"     → ("perricard", "g")
      "Mpetshi Perricard G."           → ("perricard", "g")
    """
    normalized = _normalize_name(nombre)
    raw = normalized.split()
    if not raw:
        return ("", "")

    # Strip suffixes
    while raw and raw[-1] in _SUFFIXES:
        raw.pop()
    if not raw:
        return ("", "")

    # Separar iniciales (tokens de 1 char) del apellido
    iniciales = []
    apellido_parts = []
    for token in raw:
        if len(token) <= 1:
            iniciales.append(token)
        else:
            apellido_parts.append(token)

    if not apellido_parts:
        # Todos son iniciales (raro) — usar el último token original
        return (raw[-1], raw[0][0])

    # El apellido es el ÚLTIMO token largo
    # Esto maneja "Davidovich Fokina" → "fokina" y "Van De Zandschulp" → "zandschulp"
    apellido = apellido_parts[-1]

    # La inicial es: la primera inicial encontrada, o el primer char del primer token largo
    if iniciales:
        inicial = iniciales[0][0]
    else:
        inicial = apellido_parts[0][0]

    return (apellido, inicial)


def _build_match_key(name1: str, name2: str) -> Tuple[str, str, str, str]:
    """
    Construye clave de matching para un par de jugadores.
    Usa (apellido1, inicial1, apellido2, inicial2) ordenados por apellido.
    """
    a1, i1 = _parse_nombre(name1)
    a2, i2 = _parse_nombre(name2)
    # Ordenar por apellido para que el orden de home/away no importe
    if a1 <= a2:
        return (a1, i1, a2, i2)
    return (a2, i2, a1, i1)


def match_players(kambi_matches: List[Dict], fs_matches: List[Dict]) -> List[Dict]:
    """
    Cruza partidos de Kambi con FlashScore por nombre de jugadores.

    Estrategia multi-tier (patrón NBA flashscore_api.py _find_player):
      1. Match exacto por (apellido1+inicial1, apellido2+inicial2)
      2. Fallback: match solo por (apellido1, apellido2) si iniciales no contradicen
      3. Fallback: substring match para tokens ≥5 chars

    Returns:
        Lista de partidos mergeados con cuotas + match_ids + rankings
    """
    # Indexar FlashScore por clave completa y por apellidos solos
    fs_by_full_key: Dict[Tuple, Dict] = {}
    fs_by_surnames: Dict[Tuple[str, str], List[Dict]] = {}

    for fs in fs_matches:
        key = _build_match_key(fs["jugador1_fs"], fs["jugador2_fs"])
        fs_by_full_key[key] = fs

        a1, _ = _parse_nombre(fs["jugador1_fs"])
        a2, _ = _parse_nombre(fs["jugador2_fs"])
        surname_key = tuple(sorted([a1, a2]))
        if surname_key not in fs_by_surnames:
            fs_by_surnames[surname_key] = []
        fs_by_surnames[surname_key].append(fs)

    merged = []
    matched = 0
    unmatched_kambi = []

    for km in kambi_matches:
        km_key = _build_match_key(km["jugador1"], km["jugador2"])
        km_a1, km_i1 = _parse_nombre(km["jugador1"])
        km_a2, km_i2 = _parse_nombre(km["jugador2"])

        # Tier 1: match exacto (apellido + inicial para ambos jugadores)
        fs = fs_by_full_key.get(km_key)

        # Tier 2: solo apellidos (para apellidos compuestos donde FlashScore abrevia diferente)
        if not fs:
            surname_key = tuple(sorted([km_a1, km_a2]))
            candidates = fs_by_surnames.get(surname_key, [])
            if len(candidates) == 1:
                fs = candidates[0]
            elif len(candidates) > 1:
                # Múltiples candidatos — usar iniciales para desambiguar
                for c in candidates:
                    c_key = _build_match_key(c["jugador1_fs"], c["jugador2_fs"])
                    # Verificar que iniciales no contradicen
                    if c_key[1] == km_key[1] and c_key[3] == km_key[3]:
                        fs = c
                        break
                if not fs:
                    fs = candidates[0]  # fallback al primero

        # Tier 3: substring match para nombres compuestos largos
        # "Davidovich Fokina" (Kambi) vs "Davidovich A." (FlashScore) — apellidos NO coinciden
        # pero "davidovich" aparece en ambos como token ≥5 chars
        if not fs:
            km_tokens = set(_normalize_name(km["jugador1"]).split() +
                           _normalize_name(km["jugador2"]).split())
            km_long = {t for t in km_tokens if len(t) >= 5}

            best_score = 0
            best_fs = None
            for fs_candidate in fs_matches:
                fs_tokens = set(
                    _normalize_name(fs_candidate["jugador1_fs"]).split() +
                    _normalize_name(fs_candidate["jugador2_fs"]).split()
                )
                fs_long = {t for t in fs_tokens if len(t) >= 5}
                overlap = len(km_long & fs_long)
                if overlap > best_score and overlap >= 2:
                    best_score = overlap
                    best_fs = fs_candidate
            fs = best_fs

        # Tier 4: single-player fallback para mercados de ronda futura
        # Kambi ofrece cuotas anticipadas (ej: Noskova vs Eala = semifinal)
        # pero FlashScore tiene la ronda actual (Noskova vs Badosa, Svitolina vs Eala).
        # Buscamos cada jugador individualmente → obtenemos ranking, superficie, match_url.
        fs_j1 = None
        fs_j2 = None
        if not fs:
            for fs_candidate in fs_matches:
                fs_s1, fs_i1 = _parse_nombre(fs_candidate["jugador1_fs"])
                fs_s2, fs_i2 = _parse_nombre(fs_candidate["jugador2_fs"])
                # Buscar jugador1 de Kambi
                if not fs_j1:
                    if (km_a1 == fs_s1 and km_i1 == fs_i1) or (km_a1 == fs_s2 and km_i1 == fs_i2):
                        fs_j1 = fs_candidate
                # Buscar jugador2 de Kambi
                if not fs_j2:
                    if (km_a2 == fs_s1 and km_i2 == fs_i1) or (km_a2 == fs_s2 and km_i2 == fs_i2):
                        fs_j2 = fs_candidate

        if fs:
            matched += 1
            # Determinar orden: comparar apellido jugador1 Kambi con jugador1 FS
            km_s1, _ = _parse_nombre(km["jugador1"])
            fs_s1, _ = _parse_nombre(fs["jugador1_fs"])
            if km_s1 == fs_s1:
                r1, r2 = fs["ranking1"], fs["ranking2"]
            else:
                r1, r2 = fs["ranking2"], fs["ranking1"]

            match_url = ""
            if fs.get("slug1") and fs.get("slug2") and fs.get("match_id"):
                match_url = (
                    f"https://www.flashscore.co/match/tennis/"
                    f"{fs['slug1']}-{fs['slug2']}/{fs['match_id']}/#/h2h"
                )

            merged.append({
                "jugador1": km["jugador1"],
                "jugador2": km["jugador2"],
                "cuota1": km["cuota1"],
                "cuota2": km["cuota2"],
                "match_url": match_url,
                "match_id": fs.get("match_id"),
                "ranking1": r1,
                "ranking2": r2,
                "superficie": fs.get("superficie", "unknown"),
                "torneo_nombre": km["torneo_kambi"],
                "torneo_completo": fs.get("torneo_fs", km["torneo_kambi"]),
                "tier": km["tier"],
                "hora": km["hora"],
                "pais": _extract_country_from_tournament(fs.get("torneo_fs", "")),
                "kambi_event_id": km["kambi_event_id"],
                "cuota_es_real": True,
            })
        else:
            # Tier 4 hit: ambos jugadores encontrados individualmente en FS
            if fs_j1 and fs_j2:
                # Extraer ranking de cada jugador desde su partido actual en FS
                j1_s, j1_i = _parse_nombre(km["jugador1"])
                fs1_s1, _ = _parse_nombre(fs_j1["jugador1_fs"])
                r1 = fs_j1["ranking1"] if j1_s == fs1_s1 else fs_j1["ranking2"]

                j2_s, j2_i = _parse_nombre(km["jugador2"])
                fs2_s1, _ = _parse_nombre(fs_j2["jugador1_fs"])
                r2 = fs_j2["ranking1"] if j2_s == fs2_s1 else fs_j2["ranking2"]

                # Superficie: tomar del torneo de cualquier jugador (mismo torneo)
                superficie = fs_j1.get("superficie") or fs_j2.get("superficie") or "unknown"
                torneo_fs = fs_j1.get("torneo_fs") or fs_j2.get("torneo_fs") or km["torneo_kambi"]

                # match_url y match_ids: guardar AMBOS proxies para que el
                # H2H extractor pueda obtener el historial de cada jugador por separado
                # (Nodo-31: evitar contaminación de bloque incorrecto)
                match_url = ""
                ref_fs = fs_j1  # proxy para jugador1
                ref_fs2 = fs_j2  # proxy para jugador2
                if ref_fs.get("slug1") and ref_fs.get("slug2") and ref_fs.get("match_id"):
                    match_url = (
                        f"https://www.flashscore.co/match/tennis/"
                        f"{ref_fs['slug1']}-{ref_fs['slug2']}/{ref_fs['match_id']}/#/h2h"
                    )
                match_url_j2 = ""
                if ref_fs2.get("slug1") and ref_fs2.get("slug2") and ref_fs2.get("match_id"):
                    match_url_j2 = (
                        f"https://www.flashscore.co/match/tennis/"
                        f"{ref_fs2['slug1']}-{ref_fs2['slug2']}/{ref_fs2['match_id']}/#/h2h"
                    )

                matched += 1
                logger.info(
                    f"   🔗 Tier4 (ronda futura): {km['jugador1']} vs {km['jugador2']} "
                    f"→ rankings [{r1},{r2}] sup={superficie} | "
                    f"id_j1={ref_fs.get('match_id')} id_j2={ref_fs2.get('match_id')}"
                )
                merged.append({
                    "jugador1": km["jugador1"],
                    "jugador2": km["jugador2"],
                    "cuota1": km["cuota1"],
                    "cuota2": km["cuota2"],
                    "match_url": match_url,
                    "match_id": ref_fs.get("match_id"),
                    "match_id_j2": ref_fs2.get("match_id"),
                    "match_url_j2": match_url_j2,
                    "ranking1": r1,
                    "ranking2": r2,
                    "superficie": superficie,
                    "torneo_nombre": km["torneo_kambi"],
                    "torneo_completo": torneo_fs,
                    "tier": km["tier"],
                    "hora": km["hora"],
                    "pais": _extract_country_from_tournament(torneo_fs),
                    "kambi_event_id": km["kambi_event_id"],
                    "cuota_es_real": True,
                    "ronda_futura": True,
                })
            elif fs_j1 or fs_j2:
                # Solo un jugador encontrado — parcial
                ref = fs_j1 or fs_j2
                superficie = ref.get("superficie", "unknown")
                torneo_fs = ref.get("torneo_fs", km["torneo_kambi"])

                # Ranking del jugador encontrado
                found_km_name = km["jugador1"] if fs_j1 else km["jugador2"]
                found_s, _ = _parse_nombre(found_km_name)
                ref_s1, _ = _parse_nombre(ref["jugador1_fs"])
                found_rank = ref["ranking1"] if found_s == ref_s1 else ref["ranking2"]

                if fs_j1:
                    r1, r2 = found_rank, None
                else:
                    r1, r2 = None, found_rank

                matched += 1
                proxy_url = ""
                if ref.get("slug1") and ref.get("slug2") and ref.get("match_id"):
                    proxy_url = (
                        f"https://www.flashscore.co/match/tennis/"
                        f"{ref['slug1']}-{ref['slug2']}/{ref['match_id']}/#/h2h"
                    )

                # Bug fix Nodo-49: asignar proxy al jugador CORRECTO.
                # Si solo encontramos j1 → match_id es proxy de P1, j2 no tiene proxy.
                # Si solo encontramos j2 → match_id_j2 es proxy de P2, P1 no tiene proxy.
                # Nunca asignar proxy de P2 a match_id (que _process_ronda_futura usa para P1).
                if fs_j1:
                    proxy_j1_id = ref.get("match_id")
                    proxy_j1_url = proxy_url
                    proxy_j2_id = None
                    proxy_j2_url = ""
                else:
                    proxy_j1_id = None
                    proxy_j1_url = ""
                    proxy_j2_id = ref.get("match_id")
                    proxy_j2_url = proxy_url

                logger.info(
                    f"   🔗 Tier4 (parcial): {km['jugador1']} vs {km['jugador2']} "
                    f"→ ranking parcial [{r1},{r2}] sup={superficie} "
                    f"| proxy_j1={proxy_j1_id} proxy_j2={proxy_j2_id}"
                )
                merged.append({
                    "jugador1": km["jugador1"],
                    "jugador2": km["jugador2"],
                    "cuota1": km["cuota1"],
                    "cuota2": km["cuota2"],
                    "match_url": proxy_j1_url,
                    "match_id": proxy_j1_id,
                    "match_id_j2": proxy_j2_id,
                    "match_url_j2": proxy_j2_url,
                    "ranking1": r1,
                    "ranking2": r2,
                    "superficie": superficie,
                    "torneo_nombre": km["torneo_kambi"],
                    "torneo_completo": torneo_fs,
                    "tier": km["tier"],
                    "hora": km["hora"],
                    "pais": _extract_country_from_tournament(torneo_fs),
                    "kambi_event_id": km["kambi_event_id"],
                    "cuota_es_real": True,
                    "ronda_futura": True,
                })
            else:
                unmatched_kambi.append(km)
                merged.append({
                    "jugador1": km["jugador1"],
                    "jugador2": km["jugador2"],
                    "cuota1": km["cuota1"],
                    "cuota2": km["cuota2"],
                    "match_url": None,
                    "match_id": None,
                    "ranking1": None,
                    "ranking2": None,
                    "superficie": "unknown",
                    "torneo_nombre": km["torneo_kambi"],
                    "torneo_completo": km["torneo_kambi"],
                    "tier": km["tier"],
                    "hora": km["hora"],
                    "pais": "N/A",
                    "kambi_event_id": km["kambi_event_id"],
                    "cuota_es_real": True,
                })

    logger.info(f"🔗 Cruce: {matched}/{len(kambi_matches)} matched | {len(unmatched_kambi)} sin match_id")
    if unmatched_kambi:
        for um in unmatched_kambi[:5]:
            logger.info(f"   ⚠️ Sin FlashScore: {um['jugador1']} vs {um['jugador2']} ({um['torneo_kambi']})")

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# FLASHSCORE ODDS — Cuotas de referencia via Playwright (solo testing)
# ══════════════════════════════════════════════════════════════════════════════

_PLAYWRIGHT_ARGS = [
    "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
    "--disable-software-rasterizer", "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows", "--disable-renderer-backgrounding",
    "--disable-features=TranslateUI", "--disable-extensions", "--no-first-run",
    "--disable-default-apps",
    "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]


def _parse_odd(text: str) -> Optional[float]:
    """Convierte texto de cuota a float. Retorna None si no es válido (≤1.0)."""
    if not text:
        return None
    clean = re.sub(r"[^\d.]", "", text.strip())
    if not clean or not clean.replace(".", "").isdigit():
        return None
    try:
        val = float(clean)
        return val if val > 1.0 else None
    except ValueError:
        return None


async def _scrape_flashscore_odds_async(day_offset: int = 0) -> Dict[Tuple, Tuple[float, float]]:
    """Implementación async interna — usa Playwright para extraer cuotas de FlashScore."""
    import asyncio as _aio
    from playwright.async_api import async_playwright

    odds_map: Dict[Tuple, Tuple[float, float]] = {}

    logger.info(f"🎭 FlashScore odds scraping day_offset={day_offset}")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=_PLAYWRIGHT_ARGS)
        page = await browser.new_page()
        await page.set_viewport_size({"width": 1920, "height": 1080})

        try:
            # Usar dominio colombiano (.co/tenis/) — misma cobertura y estructura HTML
            # que ve el usuario. El .com internacional usa SVG+title (menos cobertura).
            await page.goto(
                "https://www.flashscore.co/tenis/",
                wait_until="domcontentloaded",
                timeout=45000,
            )
            await _aio.sleep(3)

            # Cookie consent
            try:
                btn = await page.wait_for_selector("#onetrust-accept-btn-handler", timeout=7000)
                if btn:
                    await btn.click()
                    await _aio.sleep(2)
            except Exception:
                pass

            # Navegar al día correcto — mismo patrón que extraer_URL_partidos_version2.py
            if day_offset != 0:
                arrow_dir = "next" if day_offset > 0 else "prev"
                day_selectors = [
                    f'button[data-day-picker-arrow="{arrow_dir}"]',
                    f'button.wcl-arrow_YpdN4[aria-label="{"Día siguiente" if day_offset > 0 else "Día anterior"}"]',
                ]
                _navigated = False
                for _step in range(abs(day_offset)):
                    for sel in day_selectors:
                        try:
                            el = await page.query_selector(sel)
                            if el:
                                await el.click()
                                await _aio.sleep(3)
                                logger.info(f"📅 FlashScore odds: click día {'+' if day_offset > 0 else ''}{day_offset} ({sel})")
                                _navigated = True
                                break
                        except Exception:
                            continue
                    if not _navigated:
                        # Fallback por aria-label
                        buttons = await page.query_selector_all("button[aria-label]")
                        for btn in buttons:
                            label = await btn.get_attribute('aria-label') or ''
                            keywords = ['siguiente', 'next', 'tomorrow'] if day_offset > 0 else ['anterior', 'prev', 'yesterday']
                            if any(kw in label.lower() for kw in keywords):
                                await btn.click()
                                await _aio.sleep(3)
                                logger.info(f"📅 FlashScore odds: click día por aria-label '{label}'")
                                _navigated = True
                                break
                    if not _navigated:
                        logger.warning(f"⚠️  FlashScore odds: no se encontró botón de navegación día {day_offset} — usando día actual")

            # Click odds button para activar columnas de cuotas
            for sel in ['text="Odds"', 'text="Cuotas"', '[data-testid*="odds"]',
                        'button:has-text("Odds")', 'a[href*="odds"]']:
                try:
                    el = await page.wait_for_selector(sel, timeout=3000)
                    if el:
                        await el.click()
                        await _aio.sleep(3)
                        break
                except Exception:
                    continue

            import re as _re_odds

            def _extract_odd_from_title(title_str: str) -> Optional[float]:
                """Extrae decimal odd de title '[bl]X[d]Y' (formato .com) o texto plano."""
                _m = _re_odds.search(r'\[d\]([\d.]+)', title_str)
                if _m:
                    return _parse_odd(_m.group(1))
                _m = _re_odds.search(r'\[bl\]([\d.]+)', title_str)
                if _m:
                    return _parse_odd(_m.group(1))
                return _parse_odd(title_str)

            async def _extract_match_data(element) -> None:
                """Extrae mid + cuotas de un elemento y los agrega a odds_map."""
                try:
                    # match_id desde href (?mid=) o desde id del elemento (g_2_XXXX)
                    mid_from_dom = None
                    row_link = await element.query_selector("a.eventRowLink")
                    if row_link:
                        href = await row_link.get_attribute('href') or ""
                        mm = _re_odds.search(r'\?mid=([A-Za-z0-9]+)', href)
                        if mm:
                            mid_from_dom = mm.group(1)
                    if not mid_from_dom:
                        el_id = await element.get_attribute('id') or ""
                        mm = _re_odds.search(r'g_\d+_([A-Za-z0-9]+)$', el_id)
                        if mm:
                            mid_from_dom = mm.group(1)

                    # Jugadores (solo procesar singles — 2 participantes exactos para singles)
                    participants = await element.query_selector_all(".event__participant")
                    if len(participants) < 2:
                        return
                    j1 = (await participants[0].text_content() or "").strip()
                    j2 = (await participants[1].text_content() or "").strip()
                    if not j1 or not j2:
                        return

                    # Cuotas — estructura .co: <span>1.22</span> dentro del div
                    odd1_el = await element.query_selector(".event__odd--odd1")
                    odd2_el = await element.query_selector(".event__odd--odd2")
                    if not odd1_el or not odd2_el:
                        odd1_el = await element.query_selector('[class*="odd1"]')
                        odd2_el = await element.query_selector('[class*="odd2"]')
                    if not odd1_el or not odd2_el:
                        return

                    async def _get_odd_val(el) -> Optional[float]:
                        sp = await el.query_selector("span")
                        if sp:
                            v = _parse_odd(await sp.text_content() or "")
                            if v:
                                return v
                        v = _parse_odd(await el.text_content() or "")
                        if v:
                            return v
                        t = await el.get_attribute('title') or ""
                        return _extract_odd_from_title(t) if t else None

                    c1 = await _get_odd_val(odd1_el)
                    c2 = await _get_odd_val(odd2_el)

                    if c1 and c2:
                        key = _build_match_key(j1, j2)
                        odds_map[key] = (c1, c2)
                        if mid_from_dom:
                            odds_map[('__mid__', mid_from_dom)] = (c1, c2)
                except Exception:
                    pass

            # Sweep con virtual scrolling: recolectar en múltiples posiciones
            # FlashScore usa virtual scroll — elimina elementos fuera de viewport
            # Necesitamos barrer desde arriba hasta abajo recogiendo datos en cada paso
            seen_mids: set = set()
            scroll_height = await page.evaluate("document.body.scrollHeight")
            step_px = 2500
            pos = 0
            rounds = 0
            while pos <= scroll_height + step_px:
                await page.evaluate(f"window.scrollTo(0, {pos})")
                await _aio.sleep(0.8)
                match_elements = await page.query_selector_all(".event__match")
                for element in match_elements:
                    el_id = await element.get_attribute('id') or ""
                    if el_id not in seen_mids:
                        seen_mids.add(el_id)
                        await _extract_match_data(element)
                pos += step_px
                # Recalcular altura en caso de lazy loading
                scroll_height = await page.evaluate("document.body.scrollHeight")
                rounds += 1
                if rounds > 25:  # safety cap
                    break

            logger.info(f"⚡ FlashScore odds: sweep completado ({rounds} rondas, {len(seen_mids)} elementos únicos)")

        finally:
            await browser.close()

    logger.info(f"   ✅ {len(odds_map)} partidos con cuotas extraídos de FlashScore")
    return odds_map


def fetch_flashscore_odds(day_offset: int = 0) -> Dict[Tuple, Tuple[float, float]]:
    """
    Extrae cuotas de FlashScore via Playwright.

    Solo para testing/validación post-hoc — NO son cuotas de Betplay.
    Las cuotas se marcan cuota_es_real=False en el pipeline.

    Args:
        day_offset: 0=hoy, 1=mañana, -1=ayer (debe coincidir con el feed)

    Returns:
        Dict[match_key -> (cuota_j1, cuota_j2)]
        match_key = _build_match_key(j1, j2) — compatible con match_players()
    """
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _scrape_flashscore_odds_async(day_offset))
                return future.result()
        return loop.run_until_complete(_scrape_flashscore_odds_async(day_offset))
    except RuntimeError:
        return asyncio.run(_scrape_flashscore_odds_async(day_offset))


def _load_fs_cache_flat() -> List[Dict]:
    """
    Carga el JSON flashscore-only más reciente del disco como lista plana.
    Nodo-50: fallback de caché para partidos sin match_id.
    """
    data_dir = Path("data")
    files = sorted(data_dir.glob("zita_tennis_matches_*.json"), reverse=True)
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                continue
            flat = []
            for torneo_key, matches in data.items():
                if isinstance(matches, list):
                    for m in matches:
                        m_copy = dict(m)
                        m_copy["_torneo_key"] = torneo_key
                        # Convertir campos al formato de fetch_flashscore_feed
                        if "jugador1" in m_copy and "jugador1_fs" not in m_copy:
                            m_copy["jugador1_fs"] = m_copy["jugador1"]
                            m_copy["jugador2_fs"] = m_copy["jugador2"]
                        flat.append(m_copy)
            if flat:
                logger.info(f"   💾 Caché FS disk: {f.name} ({len(flat)} partidos)")
                return flat
        except Exception:
            continue
    return []


def _enrich_match_ids(merged: List[Dict], day_offset: int) -> List[Dict]:
    """
    Nodo-50: Puente inteligente — enriquece match_id para partidos que fallaron el cruce.

    Para cada partido con match_id=None, intenta 3 fuentes en cascada:
      1. FlashScore feed day+1 (top seeds de GS a veces programados para mañana)
      2. FlashScore feed day-1 (partido ya iniciado, no en feed de hoy)
      3. JSON flashscore-only más reciente en disco (caché)

    No modifica partidos que ya tienen match_id.
    """
    sin_id = [m for m in merged if not m.get("match_id")]
    if not sin_id:
        return merged

    logger.info(f"   🌉 Nodo-50 Bridge: {len(sin_id)} partidos sin match_id — buscando en fuentes alternativas")

    # Construir índices de cada fuente alternativa
    alt_sources: List[Tuple[str, List[Dict]]] = []

    # Fuente A: feed de mañana
    try:
        fs_tomorrow = fetch_flashscore_feed(day_offset + 1)
        if fs_tomorrow:
            alt_sources.append(("feed_day+1", fs_tomorrow))
    except Exception:
        pass

    # Fuente B: feed de ayer (matches empezados tarde)
    if day_offset == 0:
        try:
            fs_yesterday = fetch_flashscore_feed(-1)
            if fs_yesterday:
                alt_sources.append(("feed_day-1", fs_yesterday))
        except Exception:
            pass

    # Fuente C: JSON flashscore-only en disco
    fs_cache = _load_fs_cache_flat()
    if fs_cache:
        alt_sources.append(("disk_cache", fs_cache))

    # Índices por full_key y apellidos para cada fuente
    source_indices = []
    for src_name, src_matches in alt_sources:
        by_full: Dict[Tuple, Dict] = {}
        by_surnames: Dict[Tuple[str, str], List[Dict]] = {}
        for m in src_matches:
            j1 = m.get("jugador1_fs") or m.get("jugador1", "")
            j2 = m.get("jugador2_fs") or m.get("jugador2", "")
            if not j1 or not j2:
                continue
            key = _build_match_key(j1, j2)
            by_full[key] = m
            a1, _ = _parse_nombre(j1)
            a2, _ = _parse_nombre(j2)
            sk = tuple(sorted([a1, a2]))
            by_surnames.setdefault(sk, []).append(m)
        source_indices.append((src_name, by_full, by_surnames, src_matches))

    # Para cada partido sin match_id, buscar en cada fuente
    enriched = 0
    for m in merged:
        if m.get("match_id"):
            continue

        km_key = _build_match_key(m["jugador1"], m["jugador2"])
        km_a1, km_i1 = _parse_nombre(m["jugador1"])
        km_a2, km_i2 = _parse_nombre(m["jugador2"])

        found_fs = None
        found_src = None

        for src_name, by_full, by_surnames, src_matches in source_indices:
            # Tier 1: match exacto
            fs = by_full.get(km_key)
            # Tier 2: solo apellidos
            if not fs:
                sk = tuple(sorted([km_a1, km_a2]))
                candidates = by_surnames.get(sk, [])
                if len(candidates) == 1:
                    fs = candidates[0]
                elif len(candidates) > 1:
                    for c in candidates:
                        j1 = c.get("jugador1_fs") or c.get("jugador1", "")
                        j2 = c.get("jugador2_fs") or c.get("jugador2", "")
                        ck = _build_match_key(j1, j2)
                        if ck[1] == km_key[1] and ck[3] == km_key[3]:
                            fs = c
                            break
                    if not fs:
                        fs = candidates[0]
            if fs:
                found_fs = fs
                found_src = src_name
                break

        if found_fs:
            mid = found_fs.get("match_id") or found_fs.get("match_id")
            if mid:
                j1_name = found_fs.get("jugador1_fs") or found_fs.get("jugador1", "")
                j2_name = found_fs.get("jugador2_fs") or found_fs.get("jugador2", "")
                fs_a1, _ = _parse_nombre(j1_name)
                km_a1_check, _ = _parse_nombre(m["jugador1"])
                is_same_order = (km_a1_check == fs_a1)

                slug1 = found_fs.get("slug1")
                slug2 = found_fs.get("slug2")
                match_url = ""
                if slug1 and slug2 and mid:
                    match_url = (
                        f"https://www.flashscore.co/match/tennis/"
                        f"{slug1}-{slug2}/{mid}/#/h2h"
                    )

                m["match_id"] = mid
                m["match_url"] = match_url or found_fs.get("match_url", "")
                if not m.get("superficie") or m["superficie"] == "unknown":
                    m["superficie"] = found_fs.get("superficie", "unknown")
                if not m.get("ranking1"):
                    m["ranking1"] = found_fs.get("ranking1") if is_same_order else found_fs.get("ranking2")
                if not m.get("ranking2"):
                    m["ranking2"] = found_fs.get("ranking2") if is_same_order else found_fs.get("ranking1")
                if not m.get("torneo_completo") or m["torneo_completo"] == m.get("torneo_nombre"):
                    tc = found_fs.get("_torneo_key") or found_fs.get("torneo_fs") or found_fs.get("torneo_completo", "")
                    if tc:
                        m["torneo_completo"] = tc

                m["match_id_source"] = found_src
                enriched += 1
                logger.info(
                    f"   ✅ Bridge [{found_src}]: {m['jugador1']} vs {m['jugador2']} → id:{mid}"
                )

    if enriched:
        logger.info(f"   🌉 Bridge recuperó {enriched}/{len(sin_id)} match_ids")
    else:
        logger.info(f"   ⚠️ Bridge: 0/{len(sin_id)} recuperados — partidos no encontrados en ninguna fuente")

    return merged


def extract_matches_flashscore_only(
    day_offset: int = 0,
    tiers: Optional[List[str]] = None,
    torneos: Optional[List[str]] = None,
) -> Tuple[str, List[Dict]]:
    """
    Modo testing: FlashScore feed + FlashScore odds — sin depender de Kambi.

    Usa cuota_es_real=False. NO usar para apuestas reales.
    Util para validar el pipeline post-hoc con la jornada completa.

    Args:
        day_offset: 0=hoy, 1=mañana, -1=ayer
        tiers: filtrar por tier (None = todos los singles)
        torneos: filtrar por nombre de torneo, substring case-insensitive (Nodo-50)

    Returns:
        (filename, matches)
    """
    from config import detectar_tier

    # 1. FlashScore feed — match_ids + rankings + superficie
    fs_matches = fetch_flashscore_feed(day_offset)
    if not fs_matches:
        logger.error("❌ FlashScore feed vacío")
        return "", []

    # 2. FlashScore odds — Playwright (mismo day_offset que el feed)
    logger.info("🎭 Iniciando Playwright para cuotas FlashScore...")
    odds_map = fetch_flashscore_odds(day_offset)
    logger.info(f"   💰 {len(odds_map)} pares con cuotas encontrados")

    # 3. Merge: feed + odds — primero por match_id (exacto), luego por nombre (fuzzy)
    merged = []
    con_cuotas = 0
    for m in fs_matches:
        # Prioridad 1: match por match_id (exacto — sin depender de name parsing)
        mid_key = ('__mid__', m.get("match_id", ""))
        cuotas = odds_map.get(mid_key) if m.get("match_id") else None
        # Prioridad 2: match por nombre (fallback)
        if not cuotas:
            key = _build_match_key(m["jugador1_fs"], m["jugador2_fs"])
            cuotas = odds_map.get(key)
        cuota1 = cuotas[0] if cuotas else None
        cuota2 = cuotas[1] if cuotas else None
        if cuota1:
            con_cuotas += 1

        # match_url desde slugs del feed
        match_url = ""
        if m.get("slug1") and m.get("slug2") and m.get("match_id"):
            match_url = (
                f"https://www.flashscore.co/match/tennis/"
                f"{m['slug1']}-{m['slug2']}/{m['match_id']}/#/h2h"
            )

        tier = detectar_tier(m.get("torneo_fs", ""))

        merged.append({
            "jugador1": m["jugador1_fs"],
            "jugador2": m["jugador2_fs"],
            "cuota1": cuota1,
            "cuota2": cuota2,
            "match_url": match_url,
            "match_id": m.get("match_id"),
            "ranking1": m.get("ranking1"),
            "ranking2": m.get("ranking2"),
            "superficie": m.get("superficie", "unknown"),
            "torneo_nombre": m.get("torneo_fs", ""),
            "torneo_completo": m.get("torneo_fs", ""),
            "tier": tier,
            "hora": None,
            "pais": _extract_country_from_tournament(m.get("torneo_fs", "")),
            "kambi_event_id": None,
            "cuota_es_real": False,
        })

    logger.info(f"🔗 FlashScore-only: {len(merged)} partidos | {con_cuotas} con cuotas")

    # 4. Filtrar por tier si se especifica
    if tiers:
        merged = [m for m in merged if m.get("tier") in tiers]
        logger.info(f"   🔍 Filtrado por tiers {tiers}: {len(merged)} partidos")

    # Nodo-50: filtrar por nombre de torneo si se especifica
    if torneos:
        keywords = [k.lower() for k in torneos]
        before = len(merged)
        merged = [
            m for m in merged
            if any(
                kw in (m.get('torneo_nombre') or '').lower()
                or kw in (m.get('torneo_completo') or '').lower()
                for kw in keywords
            )
        ]
        logger.info(f"   🏆 Filtro --torneo {torneos}: {before} → {len(merged)} partidos")

    if not merged:
        logger.warning("⚠️ No hay partidos para guardar")
        return "", []

    # 4.9. Nodo-51 F1: añadir tournament_context a cada match
    merged = _attach_tournament_contexts(merged)

    # 5. Guardar
    filename = save_matches(merged, day_offset)
    return filename, merged


def _extract_country_from_tournament(torneo: str) -> str:
    """Extrae país del nombre del torneo FlashScore."""
    # "ATP - INDIVIDUALES: Stuttgart (Alemania), hierba" → "Alemania"
    match = re.search(r"\(([^)]+)\)", torneo)
    return match.group(1) if match else "N/A"


# ══════════════════════════════════════════════════════════════════════════════
# NODO-51 F1 — TournamentContext como Entidad
# ══════════════════════════════════════════════════════════════════════════════

def _attach_tournament_contexts(
    matches: List[Dict],
    match_date=None,
) -> List[Dict]:
    """
    Nodo-51 F1: añade 'tournament_context' a cada match dict.

    La superficie se infiere de torneo_completo UNA VEZ aquí y viaja con el match.
    Nodo-46 (F4) la leerá de tournament_context — no la reinfiere.

    No modifica ningún campo existente del match dict (compatibilidad hacia atrás).
    """
    for m in matches:
        torneo = m.get("torneo_completo") or m.get("torneo_nombre") or ""
        m["tournament_context"] = build_tournament_context(torneo, match_date=match_date)
    return matches


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT — Mismo formato que extraer_URL_partidos_version2.py
# ══════════════════════════════════════════════════════════════════════════════

def save_matches(matches: List[Dict], day_offset: int = 0) -> str:
    """
    Guarda partidos agrupados por torneo en el formato zita_tennis_matches.

    El formato es un dict {torneo_key: [matches]} para compatibilidad
    con extraer_historh2h.py y el pipeline existente.
    """
    # Agrupar por torneo_completo
    by_tournament: Dict[str, List[Dict]] = {}
    for m in matches:
        key = m.get("torneo_completo", "Unknown")
        if key not in by_tournament:
            by_tournament[key] = []

        entry = {
            "jugador1": m["jugador1"],
            "jugador2": m["jugador2"],
            "cuota1": m["cuota1"],
            "cuota2": m["cuota2"],
            "match_url": m["match_url"],
            "match_id": m.get("match_id"),
            "superficie": m.get("superficie"),
            "torneo_nombre": m.get("torneo_nombre"),
            "torneo_completo": m.get("torneo_completo"),
            "pais": m.get("pais"),
            "ranking1": m.get("ranking1"),
            "ranking2": m.get("ranking2"),
            "tier": m.get("tier"),
            "hora": m.get("hora"),
            "kambi_event_id": m.get("kambi_event_id"),
            "cuota_es_real": True,
            "tournament_context": m.get("tournament_context"),
        }
        # Fix Nodo-49: preservar campos de ronda futura para que ninja_h2h_parser
        # pueda usar los proxies correctos y detectar la ruta _process_ronda_futura.
        if m.get("ronda_futura"):
            entry["ronda_futura"] = True
        if m.get("match_id_j2"):
            entry["match_id_j2"] = m["match_id_j2"]
        if m.get("match_url_j2"):
            entry["match_url_j2"] = m["match_url_j2"]
        by_tournament[key].append(entry)

    # Guardar
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    match_date = (datetime.now() + timedelta(days=day_offset)).strftime("%Y%m%d")
    run_time = datetime.now().strftime("%H%M%S")
    timestamp = f"{match_date}_{run_time}"
    filename = data_dir / f"zita_tennis_matches_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(by_tournament, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in by_tournament.values())
    logger.info(f"💾 {filename}: {total} partidos en {len(by_tournament)} torneos")

    return str(filename)


# ══════════════════════════════════════════════════════════════════════════════
# BETPLAY OFERTAS TXT — Solo ofertas reales disponibles en Betplay
# ══════════════════════════════════════════════════════════════════════════════

def save_betplay_ofertas(matches: List[Dict]) -> str:
    """
    Genera betplay_ofertas.txt — fuente de verdad de ofertas REALES de Betplay.

    Patrón NBA: solo lo que existe en Kambi/Betplay puede apostarse.
    Cualquier partido que no esté aquí NO tiene oferta real.
    """
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    filepath = data_dir / "betplay_ofertas.txt"

    # Agrupar por tier + torneo
    by_tier: Dict[str, Dict[str, List[Dict]]] = {}
    for m in matches:
        tier = m.get("tier", "unknown")
        torneo = m.get("torneo_nombre", m.get("torneo_kambi", "?"))
        if tier not in by_tier:
            by_tier[tier] = {}
        if torneo not in by_tier[tier]:
            by_tier[tier][torneo] = []
        by_tier[tier][torneo].append(m)

    lines = []
    lines.append("=" * 70)
    lines.append("  BETPLAY — OFERTAS REALES DE TENIS")
    lines.append(f"  Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  Total: {len(matches)} partidos con cuotas reales")
    lines.append("=" * 70)
    lines.append("")

    tier_order = ["atp", "wta", "wta125", "challenger", "itf"]
    for tier in tier_order:
        if tier not in by_tier:
            continue
        torneos = by_tier[tier]
        tier_total = sum(len(v) for v in torneos.values())
        lines.append(f"── {tier.upper()} ({tier_total} partidos) ──")
        lines.append("")

        for torneo, partidos in sorted(torneos.items()):
            sup = partidos[0].get("superficie", "?") if partidos else "?"
            lines.append(f"  📍 {torneo} ({sup})")
            for m in partidos:
                j1 = m["jugador1"]
                j2 = m["jugador2"]
                c1 = m["cuota1"]
                c2 = m["cuota2"]
                lines.append(f"     {j1:30s} @{c1:<6.2f}  vs  {j2:30s} @{c2:<6.2f}")
            lines.append("")

    # Tiers no clasificados
    for tier in sorted(by_tier.keys()):
        if tier in tier_order:
            continue
        torneos = by_tier[tier]
        tier_total = sum(len(v) for v in torneos.values())
        lines.append(f"── {tier.upper()} ({tier_total} partidos) ──")
        lines.append("")
        for torneo, partidos in sorted(torneos.items()):
            lines.append(f"  📍 {torneo}")
            for m in partidos:
                lines.append(f"     {m['jugador1']:30s} @{m['cuota1']:<6.2f}  vs  {m['jugador2']:30s} @{m['cuota2']:<6.2f}")
            lines.append("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"📋 {filepath}: {len(matches)} ofertas reales Betplay")
    return str(filepath)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def extract_matches(
    day_offset: int = 0,
    tiers: Optional[List[str]] = None,
    torneos: Optional[List[str]] = None,
) -> Tuple[str, List[Dict]]:
    """
    Pipeline completo: Kambi + FlashScore → merge → save.

    Args:
        day_offset: 0=hoy, 1=mañana
        tiers: filtrar por tier (None = todos los singles)
        torneos: filtrar por nombre de torneo, substring case-insensitive (Nodo-50)

    Returns:
        (filename, matches) — ruta al JSON guardado y lista de partidos
    """
    # 1. Kambi — cuotas reales
    kambi = fetch_kambi_tennis()
    if not kambi:
        logger.error("❌ No se obtuvieron partidos de Kambi")
        return "", []

    # 2. FlashScore feed — match_ids + rankings
    fs = fetch_flashscore_feed(day_offset)

    # 3. Merge
    merged = match_players(kambi, fs)

    # 3.1. Nodo-50 Bridge — enriquecer match_ids faltantes desde fuentes alternativas
    sin_id_before = sum(1 for m in merged if not m.get("match_id"))
    if sin_id_before:
        merged = _enrich_match_ids(merged, day_offset)
        sin_id_after = sum(1 for m in merged if not m.get("match_id"))
        logger.info(f"   📊 Sin match_id: {sin_id_before} → {sin_id_after} tras bridge")

    # 3.5. Filtrar por fecha — Kambi devuelve partidos futuros (días adelante)
    target_date = (datetime.now(timezone.utc) + timedelta(days=day_offset)).date()
    before = len(merged)
    merged = [
        m for m in merged
        if not m.get("hora") or
        datetime.fromisoformat(m["hora"].replace("Z", "+00:00")).date() == target_date
    ]
    if len(merged) < before:
        logger.info(f"   📅 Filtro fecha {target_date}: {before} → {len(merged)} partidos (eliminados {before - len(merged)} futuros/pasados)")

    # 4. Filtrar por tier si se especifica
    if tiers:
        merged = [m for m in merged if m.get("tier") in tiers]
        logger.info(f"   🔍 Filtrado por tiers {tiers}: {len(merged)} partidos")

    # Nodo-50: filtrar por nombre de torneo si se especifica
    if torneos:
        keywords = [k.lower() for k in torneos]
        before = len(merged)
        merged = [
            m for m in merged
            if any(
                kw in (m.get('torneo_nombre') or '').lower()
                or kw in (m.get('torneo_completo') or '').lower()
                for kw in keywords
            )
        ]
        logger.info(f"   🏆 Filtro --torneo {torneos}: {before} → {len(merged)} partidos")

    # 4.9. Nodo-51 F1: añadir tournament_context a cada match
    merged = _attach_tournament_contexts(merged)

    # 5. Guardar
    if not merged:
        logger.warning("⚠️ No hay partidos para guardar")
        return "", []

    filename = save_matches(merged, day_offset)

    # 6. Generar betplay_ofertas.txt — fuente de verdad
    save_betplay_ofertas(merged)

    return filename, merged
