#!/usr/bin/env python3
"""
favoritos_combo_builder.py — Estrategia #13 FAVORITOS_COMPUESTOS (Nodo-110)

Construye combos 3-4 piernas con favoritos claros (p_modelo>=0.62 o cuota<=1.40 o
ranking_gap>300), cuota pierna [1.15, 2.10], cuota combinada [3.5, 7.0].

Evidencia semilla (H110-01): 8/8 combos reales jul-14/16 — cuotas 3.84-6.51x,
stakes $600-680, pago total ~$27,500. Patrón validado por el operador.

D110-01: LEG_MIN_CUOTA=1.15 aplica solo a piernas de combo (no relaja HF-1 para singles).

Uso:
  python favoritos_combo_builder.py                  # modo normal
  python favoritos_combo_builder.py --dry-run        # solo imprimir, sin .bat
  python favoritos_combo_builder.py --telegram       # enviar a Telegram
  python favoritos_combo_builder.py --mega           # incluir piernas spice [2.10,5.00]
  python favoritos_combo_builder.py --override-governor
  python favoritos_combo_builder.py --matches data/zita_tennis_matches_HOY.json  # D110-06 RANKING_ONLY
"""

import argparse
import glob
import json
import logging
import os
import re
import subprocess
import sys
import unicodedata
from datetime import date, datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from scraping.file_utils import select_best_h2h_file as _select_best_h2h  # D154-11
except ImportError:
    _select_best_h2h = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constantes D110-01 ────────────────────────────────────────────────────────
LEG_MIN_CUOTA = 1.15        # D110-01: piso pierna combo (NO relaja HF-1 para singles)
LEG_MAX_CUOTA = 2.10        # techo pierna núcleo
LEG_MAX_SPICE = 5.00        # techo pierna spice (--mega, máx 2 piernas)
COMBO_MIN_CUOTA = 3.5       # cuota combinada mínima
COMBO_MAX_CUOTA = 7.0       # cuota combinada máxima
P_MODELO_MIN = 0.62         # filtro principal probabilidad
CUOTA_CLARA_MAX = 1.40      # alternativa: favorito muy claro
RANKING_GAP_MIN = 300       # alternativa: gap ranking
LEGS_MIN = 3
LEGS_MAX = 4
STAKE_PER_COMBO = 650       # stake fijo por combo ($)
TOPE_SESION = 2000          # tope total sesión estrategia
MAX_LEGS_PER_TORNEO = 2
MAX_COMBOS_TOP = 3          # top-3 combos con solape <=2 piernas
MAX_RANKING_ONLY_PER_COMBO = 2   # D110-06: max piernas RANKING_ONLY por combo
LEG_MAX_CUOTA_RANKING_ONLY = 1.60  # D110-06: sin modelo → favorito más claro exigido
MAX_H2H_MODEL_PER_COMBO = 2     # D146: max piernas H2H_MODEL por combo

# Rutas Windows / WSL
DESKTOP_WIN = Path("/mnt/c/users/hogar/Desktop")
COMBOS_DIR = DESKTOP_WIN / "combos"
CHROME_WIN = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
REDIRECT_BASE = "https://dakotapog.github.io/tennis-analysis/bp/?ids="
BETPLAY_URL_BASE = "https://betplay.com.co/apuestas#home?coupon=combination|"
BETPLAY_URL_TAIL = "||replace"

TG_TOKEN = "8684706586:AAHv4zhjQKvxORf6bnbwCxZQPly9OA7unpY"
TG_CHAT = "8520949513"
TG_URL = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFD", name.lower())
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^a-z\s]", "", name)
    return name.strip()


def _governor_check(bankroll: float, override: bool, builder: str) -> None:
    """Llama al governor y aborta si BLOCK/WARN (a menos que override=True)."""
    gov_script = Path(__file__).parent / "combo_governor.py"
    try:
        result = subprocess.run(
            [sys.executable, str(gov_script), "--bankroll", str(int(bankroll))],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as e:
        logger.warning(f"[governor] No se pudo ejecutar: {e} — continuando")
        return

    if result.returncode == 0:
        return  # PASS

    if override:
        logger.warning(f"[governor] OVERRIDE activo — returncode={result.returncode}")
        log_path = Path(__file__).parent / "logs" / "combo_governor.log"
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, "a") as f:
            ts = datetime.now().isoformat()
            f.write(f"{ts} | OVERRIDE | builder={builder} | code={result.returncode} | {result.stdout.strip()}\n")
        return

    logger.error(f"[governor] BLOCK returncode={result.returncode}: {result.stdout.strip()}")
    sys.exit(result.returncode)


# ── Selección de favoritos ────────────────────────────────────────────────────

def seleccionar_favoritos(edge_report: List[Dict], mega: bool = False) -> Tuple[List[Dict], Dict]:
    """
    Función pura — selecciona piernas candidatas del edge_report.

    Filtros (en orden):
      0. Seguridad: sin NO_DATA, sin phantom, sin historial incompleto
      1. Favorito claro: p_modelo>=0.62 O cuota_fav<=1.40 con conf!=LOW
                         O ranking_gap>300 con cuota<=1.60
      2. Cuota pierna: [1.15, 2.10] núcleo; [2.10, 5.00] spice si mega=True
      3. Favorito del modelo = favorito del bookmaker (cuota_fav < cuota_rival)

    Returns:
        (lista de picks válidos ordenados por p_modelo desc, dict de conteos por filtro)
    """
    conteos = {
        "universo": 0,
        "pass_seguridad": 0,
        "pass_favorito_claro": 0,
        "pass_cuota_rango": 0,
        "pass_model_eq_bookie": 0,
        "descartados_NO_DATA": 0,
        "descartados_phantom": 0,
        "descartados_historial": 0,
        "descartados_no_favorito": 0,
        "descartados_cuota_rango": 0,
        "descartados_model_neq_bookie": 0,
    }

    candidatos = []

    for pick in edge_report:
        # Universo: apostar + watchlist + sin_edge (excluye NO_DATA)
        status = pick.get("status", "")
        if status in ("NO_DATA", "BLOCK"):
            conteos["descartados_NO_DATA"] += 1
            continue
        conteos["universo"] += 1

        # Filtro 0 — seguridad
        if pick.get("phantom_data", False) or pick.get("phantom_identity_risk", False):
            conteos["descartados_phantom"] += 1
            continue
        if pick.get("historial_incompleto", False):
            conteos["descartados_historial"] += 1
            continue
        conteos["pass_seguridad"] += 1

        # Filtro 1 — favorito claro
        p_modelo = float(pick.get("p_modelo", pick.get("probabilidad_modelo", 0)) or 0)
        cuota_fav = float(pick.get("cuota_favorito", 0) or 0)
        cuota_rival = float(pick.get("cuota_rival", 99) or 99)
        ranking_fav = pick.get("ranking_favorito") or pick.get("ranking") or 9999
        ranking_rival = pick.get("ranking_rival") or 9999
        try:
            ranking_fav = int(ranking_fav)
            ranking_rival = int(ranking_rival)
        except (ValueError, TypeError):
            ranking_fav, ranking_rival = 9999, 9999
        ranking_gap = ranking_rival - ranking_fav  # positivo = fav mejor rankeado

        conf = (pick.get("confidence_flag") or "").upper()

        cond_p = p_modelo >= P_MODELO_MIN
        cond_cuota_clara = (cuota_fav <= CUOTA_CLARA_MAX and conf != "LOW")
        cond_ranking = (ranking_gap > RANKING_GAP_MIN and cuota_fav <= 1.60)

        if not (cond_p or cond_cuota_clara or cond_ranking):
            conteos["descartados_no_favorito"] += 1
            continue
        conteos["pass_favorito_claro"] += 1

        # Filtro 2 — cuota pierna en rango
        if cuota_fav < LEG_MIN_CUOTA:
            conteos["descartados_cuota_rango"] += 1
            continue
        cuota_max = LEG_MAX_SPICE if mega else LEG_MAX_CUOTA
        if cuota_fav > cuota_max:
            conteos["descartados_cuota_rango"] += 1
            continue
        conteos["pass_cuota_rango"] += 1

        # Filtro 3 — favorito modelo = favorito bookmaker
        if cuota_fav >= cuota_rival:
            conteos["descartados_model_neq_bookie"] += 1
            continue
        conteos["pass_model_eq_bookie"] += 1

        candidatos.append({**pick, "fuente": pick.get("fuente", "EDGE_REPORT")})

    candidatos.sort(key=lambda p: float(p.get("p_modelo", p.get("probabilidad_modelo", 0)) or 0), reverse=True)
    return candidatos, conteos


# ── Armado de combos ──────────────────────────────────────────────────────────

def armar_combos(picks: List[Dict], mega: bool = False) -> List[Dict]:
    """
    Función pura — genera top-3 combos con solape <=2 piernas entre sí.

    Restricciones:
      - 3-4 piernas (LEGS_MIN/MAX)
      - máx MAX_LEGS_PER_TORNEO por torneo
      - máx 1 pierna por jugador
      - cuota combinada [COMBO_MIN_CUOTA, COMBO_MAX_CUOTA]
      - solape <=2 piernas entre cualquier par de combos seleccionados

    Returns lista de dicts con 'legs', 'cuota_total', 'prob_win', 'stake'.
    """
    if not picks:
        return []

    combos_validos = []

    for n_legs in range(LEGS_MIN, LEGS_MAX + 1):
        for combo_picks in combinations(picks, n_legs):
            # Diversificación: máx MAX_LEGS_PER_TORNEO por torneo y 1 por jugador
            torneo_count: Dict[str, int] = {}
            jugador_seen = set()
            ok = True
            for p in combo_picks:
                # D138-02: 'Desconocido'/'UNK' NO significa "mismo torneo" —
                # significa "torneo sin metadato". Usar partido como clave única
                # para permitir combinación entre matches distintos.
                _TORNEO_GENERICO = {'desconocido', 'unk', '?', '', 'unknown', 'desconocida', 'none'}
                torneo_raw = (p.get("torneo") or p.get("tournament") or "").strip()
                if torneo_raw.lower() in _TORNEO_GENERICO:
                    torneo = f"_match_{p.get('partido', p.get('favorito', str(id(p))))}"
                else:
                    torneo = torneo_raw
                jugador = _normalize_name(p.get("favorito", p.get("jugador", "")))
                if jugador in jugador_seen:
                    ok = False
                    break
                jugador_seen.add(jugador)
                torneo_count[torneo] = torneo_count.get(torneo, 0) + 1
                if torneo_count[torneo] > MAX_LEGS_PER_TORNEO:
                    ok = False
                    break
            if not ok:
                continue

            # D110-06: máx MAX_RANKING_ONLY_PER_COMBO piernas RANKING_ONLY por combo
            n_ronly = sum(1 for p in combo_picks if p.get("fuente") == "RANKING_ONLY")
            if n_ronly > MAX_RANKING_ONLY_PER_COMBO:
                continue

            # Cuota combinada
            cuota_total = 1.0
            for p in combo_picks:
                cuota_total *= float(p.get("cuota_favorito", 1))

            if not (COMBO_MIN_CUOTA <= cuota_total <= COMBO_MAX_CUOTA):
                continue

            # Probabilidad de ganar el combo
            prob_win = 1.0
            for p in combo_picks:
                prob_win *= float(p.get("p_modelo", p.get("probabilidad_modelo", 0.5)) or 0.5)

            combos_validos.append({
                "legs": list(combo_picks),
                "cuota_total": round(cuota_total, 2),
                "prob_win": round(prob_win, 4),
                "stake": STAKE_PER_COMBO,
            })

    if not combos_validos:
        return []

    # Ordenar por prob_win desc
    combos_validos.sort(key=lambda c: c["prob_win"], reverse=True)

    # Seleccionar top-3 con solape <=2 piernas
    selected = []
    for combo in combos_validos:
        combo_jugs = {_normalize_name(p.get("favorito", p.get("jugador", ""))) for p in combo["legs"]}
        ok = True
        for sel in selected:
            sel_jugs = {_normalize_name(p.get("favorito", p.get("jugador", ""))) for p in sel["legs"]}
            if len(combo_jugs & sel_jugs) > 2:
                ok = False
                break
        if ok:
            selected.append(combo)
        if len(selected) >= MAX_COMBOS_TOP:
            break

    return selected


# ── Output / Zero-Null (D90-04) ──────────────────────────────────────────────

def _imprimir_desglose(conteos: Dict, picks_validos: List[Dict]) -> None:
    """Siempre emite — si no hay picks, imprime exactamente qué falta (D90-04)."""
    print("\n=== FAVORITOS_COMPUESTOS — Desglose de filtros ===")
    print(f"  Universo (no NO_DATA):        {conteos['universo']}")
    print(f"  Pass seguridad:               {conteos['pass_seguridad']}"
          f"  (desc phantom={conteos['descartados_phantom']}, hist={conteos['descartados_historial']})")
    print(f"  Pass favorito claro:          {conteos['pass_favorito_claro']}"
          f"  (desc no_fav={conteos['descartados_no_favorito']})")
    print(f"  Pass cuota rango [{LEG_MIN_CUOTA},{LEG_MAX_CUOTA}]:  {conteos['pass_cuota_rango']}"
          f"  (desc={conteos['descartados_cuota_rango']})")
    print(f"  Pass model=bookie:            {conteos['pass_model_eq_bookie']}"
          f"  (desc={conteos['descartados_model_neq_bookie']})")

    if conteos["pass_model_eq_bookie"] < LEGS_MIN:
        falta = LEGS_MIN - conteos["pass_model_eq_bookie"]
        print(f"\n  [INSUFICIENTE] Faltan {falta} piernas para armar combo minimo de {LEGS_MIN}.")
        print("  Accion: revisar si hay partidos sin cuota_rival en edge_report,")
        print("          o ampliar con --mega para incluir cuotas [2.10, 5.00].")


def _build_betplay_url(outcome_ids: List[str]) -> str:
    # REGLA-BAT-1 (CLAUDE.md §9, INMUTABLE): IDs separados por comas, SIN
    # sufijo "|ML/" — ese formato hace que Betplay no parsee el coupon y
    # abra sin piernas cargadas. Ver Nodo-162 (mismo bug en docs/bp/index.html)
    # y Nodo-169 (este archivo nunca tuvo el formato correcto desde su
    # creación en Nodo-146 — bug independiente, no una regresión de Nodo-162).
    ids_str = ",".join(outcome_ids)
    return f"{BETPLAY_URL_BASE}{ids_str}{BETPLAY_URL_TAIL}"


def _build_redirect_url(outcome_ids: List[str]) -> str:
    return REDIRECT_BASE + ",".join(outcome_ids)


def _generar_bat(combos_con_ids: List[Dict], combo_offset: int = 100) -> int:
    """Genera FavComboN.bat en el escritorio (offset 100 para no pisar Combo1-99)."""
    COMBOS_DIR.mkdir(exist_ok=True)
    for old in DESKTOP_WIN.glob("FavCombo*.bat"):
        old.unlink(missing_ok=True)
    for old in COMBOS_DIR.glob("favcombos*.html"):
        old.unlink(missing_ok=True)

    generados = 0
    for i, c in enumerate(combos_con_ids, start=1):
        if not c.get("url"):
            continue
        idx = combo_offset + i
        url = c["url"]
        legs_str = " + ".join(f"{l['jugador']}@{l['cuota']:.2f}" for l in c["legs_display"])

        html_content = (
            f"<html><head><title>FavCombo {idx}</title></head><body>\n"
            f'<script>window.location.replace("{url}");</script>\n'
            f"<p>Redirigiendo... FavCombo {idx}: {legs_str}</p>\n"
            f"</body></html>"
        )
        html_path = COMBOS_DIR / f"favcombo{idx}.html"
        html_path.write_text(html_content, encoding="utf-8")

        html_win = f"C:\\users\\hogar\\Desktop\\combos\\favcombo{idx}.html"
        bat_content = (
            f"@echo off\r\n"
            f'start "" "{CHROME_WIN}" "file:///{html_win}"\r\n'
        )
        bat_path = DESKTOP_WIN / f"FavCombo{idx}.bat"
        bat_path.write_text(bat_content, encoding="utf-8")
        logger.info(f"  FavCombo{idx}.bat — {legs_str} @{c['cuota_total']:.2f}x")
        generados += 1

    return generados


def _registrar_shadow_book(combos: List[Dict], fecha: str) -> None:
    """Registra cada pierna en shadow_book con estrategia=FAVORITOS_COMPUESTOS."""
    try:
        import shadow_book as sb
        for i, combo in enumerate(combos, start=1):
            for pick in combo["legs"]:
                jugador = pick.get("favorito", pick.get("jugador", ""))
                pick_snap = {
                    **pick,
                    "estrategia": "FAVORITOS_COMPUESTOS",
                    "combo_idx": i,
                    "cuota_combo_total": combo["cuota_total"],
                    "stake_combo": combo["stake"],
                    "circuito": pick.get("circuito", pick.get("tier", "")),
                }
                try:
                    sb.log_pick(
                        fecha=fecha,
                        jugador=jugador,
                        cuota=float(pick.get("cuota_favorito", 0)),
                        pick_snapshot=pick_snap,
                    )
                except Exception as e:
                    logger.warning(f"[shadow_book] No se pudo registrar {jugador}: {e}")
    except ImportError:
        logger.warning("[shadow_book] No disponible — picks no registrados")


def _enviar_telegram(combos: List[Dict]) -> None:
    """Envía resumen de combos a Telegram."""
    import urllib.request as ur
    lineas = ["*FAVORITOS COMPUESTOS (estrategia #13)*"]
    for i, c in enumerate(combos, start=1):
        legs_txt = " + ".join(
            f"{p.get('favorito', p.get('jugador','?'))}@{float(p.get('cuota_favorito',0)):.2f}"
            for p in c["legs"]
        )
        lineas.append(
            f"FavCombo{i}: {legs_txt} → *@{c['cuota_total']:.2f}x* ${c['stake']}"
        )
    lineas.append(f"Stake total sesion: ${sum(c['stake'] for c in combos)}")
    msg = "\n".join(lineas)
    payload = json.dumps({"chat_id": TG_CHAT, "text": msg, "parse_mode": "Markdown"}).encode()
    try:
        req = ur.Request(TG_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        ur.urlopen(req, timeout=10)
        logger.info("[telegram] Mensaje enviado")
    except Exception as e:
        logger.warning(f"[telegram] Error: {e}")


# ── Lectura edge_report ───────────────────────────────────────────────────────

def _leer_edge_report(path: Optional[str] = None) -> List[Dict]:
    if path:
        with open(path) as f:
            data = json.load(f)
    else:
        # Excluir edge_report_kambi_* — favoritos necesita el universo completo (307 picks)
        archivos = sorted([f for f in glob.glob("reports/edge_report_*.json") if 'kambi' not in f])
        if not archivos:
            logger.error("[edge_report] No se encontró reports/edge_report_*.json")
            sys.exit(1)
        with open(archivos[-1]) as f:
            data = json.load(f)
        logger.info(f"[edge_report] Usando {archivos[-1]}")

    if isinstance(data, list):
        return data
    # FIX Nodo-110 (Fable 2026-07-17): el schema real del edge_report es
    # {apostar:[], watchlist:[], sin_edge:[]} — sin este merge el universo
    # quedaba SIEMPRE vacío ("0 piernas" con 120 partidos sobre la mesa).
    if any(k in data for k in ("apostar", "watchlist", "sin_edge")):
        return ((data.get("apostar") or [])
                + (data.get("watchlist") or [])
                + (data.get("sin_edge") or []))
    return data.get("picks", data.get("results", []))


# ── RANKING_ONLY universe (D110-06) ──────────────────────────────────────────

def _leer_rankings() -> Dict[str, int]:
    """
    Carga rankings ATP+WTA más recientes → {nombre_normalizado: posicion}.
    Los archivos tienen formato {rankings: [{name: "Surname Firstname", ranking_position: N}]}.
    """
    ranking_map: Dict[str, int] = {}
    for pattern in ("data/atp_rankings_complete_*.json", "data/wta_rankings_complete_*.json"):
        archivos = sorted(glob.glob(pattern))
        if not archivos:
            continue
        try:
            with open(archivos[-1]) as f:
                data = json.load(f)
            for entry in data.get("rankings", []):
                name = entry.get("name", "")
                pos = entry.get("ranking_position")
                if name and pos:
                    ranking_map[_normalize_name(name)] = int(pos)
        except Exception as e:
            logger.warning(f"[rankings] Error leyendo {archivos[-1]}: {e}")
    return ranking_map


def _buscar_ranking(nombre: str, ranking_map: Dict[str, int]) -> Optional[int]:
    """
    Busca posición en ranking_map — normaliza el nombre y prueba variantes.
    Los rankings tienen "Surname Firstname"; los partidos pueden tener "Firstname Surname".
    """
    norm = _normalize_name(nombre)
    if norm in ranking_map:
        return ranking_map[norm]
    parts = norm.split()
    if len(parts) >= 2:
        # Invertir: "Firstname Surname" → "Surname Firstname"
        invertido = " ".join(parts[1:] + [parts[0]])
        if invertido in ranking_map:
            return ranking_map[invertido]
        # Apellido como prefijo o sufijo (fallback para nombres compuestos)
        apellido = parts[-1]
        for k, v in ranking_map.items():
            if k.startswith(apellido + " ") or k.endswith(" " + apellido):
                return v
    return None


def _leer_matches_ranking_only(matches_path: str, edge_picks_set: set) -> List[Dict]:
    """
    D110-06: Lee partidos del archivo PASO 1 que NO están en edge_report.
    Retorna candidatos con fuente=RANKING_ONLY si cumplen TODO:
      (a) ranking_gap > RANKING_GAP_MIN (300)
      (b) cuota_favorito ∈ [LEG_MIN_CUOTA, LEG_MAX_CUOTA_RANKING_ONLY] = [1.15, 1.60]
      (c) favorito por ranking = favorito del book (cuota menor)
    Salvaguarda: max MAX_RANKING_ONLY_PER_COMBO piernas de este tipo por combo (en armar_combos).
    """
    try:
        with open(matches_path) as f:
            matches_data = json.load(f)
    except Exception as e:
        logger.error(f"[ranking_only] Error leyendo {matches_path}: {e}")
        return []

    ranking_map = _leer_rankings()
    if not ranking_map:
        logger.warning("[ranking_only] Rankings vacíos — no se pueden filtrar candidatos RANKING_ONLY")
        return []

    # Normalizar estructura: puede ser dict {torneo: [partidos]} o lista
    if isinstance(matches_data, dict):
        partidos_iter: List[Dict] = []
        for torneo, partidos in matches_data.items():
            if isinstance(partidos, list):
                for p in partidos:
                    p.setdefault("torneo_nombre", torneo)
                    partidos_iter.append(p)
    else:
        partidos_iter = matches_data

    candidatos: List[Dict] = []
    for partido in partidos_iter:
        j1 = partido.get("jugador1", "")
        j2 = partido.get("jugador2", "")
        c1 = partido.get("cuota1")
        c2 = partido.get("cuota2")
        torneo = partido.get("torneo_nombre", partido.get("torneo_completo", "UNK"))
        tier = partido.get("tier", "")

        if not j1 or not j2 or c1 is None or c2 is None:
            continue
        try:
            c1, c2 = float(c1), float(c2)
        except (TypeError, ValueError):
            continue

        # Favorito del book = cuota menor
        if c1 < c2:
            fav_book, rival_book = j1, j2
            cuota_fav, cuota_rival = c1, c2
        elif c2 < c1:
            fav_book, rival_book = j2, j1
            cuota_fav, cuota_rival = c2, c1
        else:
            continue  # cuotas iguales → no hay favorito claro del book

        # (D110-06 §5): partido presente en edge_report NUNCA se duplica
        if _normalize_name(fav_book) in edge_picks_set:
            continue

        # Filtro (b): cuota ∈ [1.15, 1.60] — más estricto que el núcleo [1.15, 2.10]
        if not (LEG_MIN_CUOTA <= cuota_fav <= LEG_MAX_CUOTA_RANKING_ONLY):
            continue

        # Filtro (a): ranking_gap > 300
        # D117-01: preferir siempre rankings ATP/WTA reales (data/atp_rankings_complete_*.json)
        # sobre los valores FlashScore CA/CB del partido (sistema interno ≠ posición oficial).
        r1 = _buscar_ranking(j1, ranking_map) or partido.get("ranking1")
        r2 = _buscar_ranking(j2, ranking_map) or partido.get("ranking2")

        if r1 is None or r2 is None:
            continue  # sin ranking de ambos → no se puede calcular gap

        ranking_gap = abs(int(r1) - int(r2))
        if ranking_gap <= RANKING_GAP_MIN:
            continue

        # Filtro (c): favorito por ranking = favorito del book
        fav_ranking = j1 if int(r1) < int(r2) else j2
        if _normalize_name(fav_ranking) != _normalize_name(fav_book):
            continue

        # p_modelo estimado: probabilidad implícita (sin modelo, solo cuota)
        p_estimado = round(1.0 / cuota_fav, 4)

        candidatos.append({
            "favorito_predicho": fav_book,
            "favorito": fav_book,
            "jugador": fav_book,
            "cuota_favorito": cuota_fav,
            "cuota_rival": cuota_rival,
            "p_modelo": p_estimado,
            "probabilidad_modelo": p_estimado,
            "confianza": "MOD",
            "confidence_flag": "MOD",
            "torneo": torneo,
            "tournament": torneo,
            "tier": tier,
            "fuente": "RANKING_ONLY",
            "ranking_gap": ranking_gap,
            "historial_incompleto": False,
            "phantom_flag": False,
            "no_data": False,
        })

    logger.info(f"[ranking_only] {len(candidatos)} candidatos RANKING_ONLY desde {matches_path}")
    return candidatos


# ── H2H_MODEL universe (D146) ─────────────────────────────────────────────────

def _find_latest_h2h() -> Optional[str]:
    """D146+D154-03: Encuentra el h2h_results_enhanced con más partidos de hoy.

    D154-03: usa select_best_h2h_file() (max n_partidos) en vez de sort
    alfabético — elige API 366p sobre Playwright 36p cuando ambos existen.
    """
    today = date.today().strftime('%Y%m%d')
    if _select_best_h2h is not None:
        return _select_best_h2h(date_str=today, directory='reports')
    # Fallback si import falla
    files = sorted(glob.glob(f"reports/h2h_results_enhanced_{today}_*.json"))
    return files[-1] if files else None


def _leer_h2h_favoritos(h2h_path: str, edge_picks_set: set) -> List[Dict]:
    """
    D146: Lee partidos de h2h_results_enhanced directamente para obtener picks
    con cuota [1.15, 2.10] que edge_calculator descartó por REGLA-HF-1 (cuota < 1.50).
    Usa la predicción real del modelo (ranking_analysis.prediction), no estimación por cuota.

    Filtros:
    - cuota_favorito ∈ [LEG_MIN_CUOTA, LEG_MAX_CUOTA] = [1.15, 2.10]
    - confidence >= 0.55 (MOD+)
    - timing guard: hora ya pasó >15min Colombia (mismo criterio D145-02)
    - deduplicación: no duplicar picks ya en universo (edge_report + RANKING_ONLY)
    """
    try:
        with open(h2h_path) as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"[h2h_favoritos] Error leyendo {h2h_path}: {e}")
        return []

    partidos = data.get("partidos", [])
    if not partidos:
        logger.warning(f"[h2h_favoritos] Sin partidos en {h2h_path}")
        return []

    # Timing guard — Colombia UTC-5 (mismo criterio D145-02)
    _ahora_min: Optional[int] = None
    try:
        import pytz
        _col_tz = pytz.timezone("America/Bogota")
        _ahora = datetime.now(_col_tz)
        _ahora_min = _ahora.hour * 60 + _ahora.minute
    except Exception:
        pass

    candidatos: List[Dict] = []
    for p in partidos:
        # Sin hora → partido sin fecha/hora confirmada (probablemente mañana) → skip
        hora = p.get("hora")
        if not hora:
            continue

        # Timing guard: skip si hora ya pasó >15min (D145-02)
        if hora and _ahora_min is not None:
            try:
                _h, _m = map(int, str(hora).split(":")[:2])
                if _ahora_min > _h * 60 + _m + 15:
                    continue
            except Exception:
                pass

        # Predicción real del modelo
        pred = (p.get("ranking_analysis") or {}).get("prediction") or {}
        favored = pred.get("favored_player", "")
        confidence = float(pred.get("confidence", 0.0) or 0.0)
        # h2h guarda confidence como porcentaje (54.8 = 54.8%), normalizar a [0,1]
        if confidence > 1.0:
            confidence = confidence / 100.0
        if not favored or confidence < 0.55:
            continue

        j1 = p.get("jugador1", "")
        j2 = p.get("jugador2", "")
        c1 = p.get("cuota1")
        c2 = p.get("cuota2")
        if c1 is None or c2 is None:
            continue
        try:
            c1, c2 = float(c1), float(c2)
        except (TypeError, ValueError):
            continue

        # Match favorito predicho → cuota correspondiente (por apellido)
        norm_fav = _normalize_name(favored)
        norm_j1 = _normalize_name(j1)
        norm_j2 = _normalize_name(j2)
        fav_word = norm_fav.split()[0] if norm_fav else ""
        j1_word = norm_j1.split()[0] if norm_j1 else ""
        j2_word = norm_j2.split()[0] if norm_j2 else ""

        if norm_fav == norm_j1 or (fav_word and fav_word == j1_word):
            cuota_fav, cuota_rival, fav_name = c1, c2, j1
        elif norm_fav == norm_j2 or (fav_word and fav_word == j2_word):
            cuota_fav, cuota_rival, fav_name = c2, c1, j2
        else:
            # Fallback: menor cuota = favorito del book
            if c1 <= c2:
                cuota_fav, cuota_rival, fav_name = c1, c2, j1
            else:
                cuota_fav, cuota_rival, fav_name = c2, c1, j2

        # Filtro cuota pierna [1.15, 2.10]
        if not (LEG_MIN_CUOTA <= cuota_fav <= LEG_MAX_CUOTA):
            continue

        # Deduplicación: no añadir si ya está en el universo (edge_report o RANKING_ONLY)
        if _normalize_name(fav_name) in edge_picks_set:
            continue

        torneo = p.get("torneo_nombre", p.get("torneo", ""))
        tier = p.get("tier", "")
        conf_flag = "STRONG" if confidence >= 0.60 else "MOD"

        candidatos.append({
            "favorito_predicho": fav_name,
            "favorito": fav_name,
            "jugador": fav_name,
            "cuota_favorito": cuota_fav,
            "cuota_rival": cuota_rival,
            "p_modelo": round(confidence, 4),
            "probabilidad_modelo": round(confidence, 4),
            "confianza": conf_flag,
            "confidence_flag": conf_flag,
            "torneo": torneo,
            "tournament": torneo,
            "tier": tier,
            "tipo_cancha": p.get("tipo_cancha", "N/A"),
            "fuente": "H2H_MODEL",
            "hora": hora,
            "historial_incompleto": False,
            "phantom_flag": False,
            "no_data": False,
        })

    logger.info(f"[h2h_favoritos] {len(candidatos)} candidatos H2H_MODEL desde {h2h_path}")
    return candidatos


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Favoritos Compuestos — estrategia #13 Nodo-110")
    parser.add_argument("--bankroll", type=float, default=125000)
    parser.add_argument("--file", help="Edge report JSON (default: más reciente)")
    parser.add_argument("--dry-run", action="store_true", help="Solo imprimir, sin .bat ni shadow_book")
    parser.add_argument("--mega", action="store_true", help="Incluir piernas spice [2.10, 5.00]")
    parser.add_argument("--telegram", action="store_true", help="Enviar resumen a Telegram")
    parser.add_argument("--override-governor", action="store_true", dest="override_governor")
    parser.add_argument("--matches", help="D110-06: zita_tennis_matches_*.json para extender universo con RANKING_ONLY")
    args = parser.parse_args()

    # Governor check (S107-D)
    _governor_check(args.bankroll, args.override_governor, "favoritos_combo_builder")

    # Leer edge_report
    edge_report = _leer_edge_report(args.file)
    logger.info(f"[edge_report] {len(edge_report)} picks cargados")

    # Seleccionar
    picks_validos, conteos = seleccionar_favoritos(edge_report, mega=args.mega)
    _imprimir_desglose(conteos, picks_validos)

    # D110-06: extender universo con candidatos RANKING_ONLY si --matches
    if args.matches:
        edge_picks_set = {
            _normalize_name(p.get("favorito_predicho", p.get("favorito", p.get("jugador", ""))))
            for p in picks_validos
        }
        ranking_only = _leer_matches_ranking_only(args.matches, edge_picks_set)
        if ranking_only:
            logger.info(f"[D110-06] {len(ranking_only)} piernas RANKING_ONLY añadidas al universo")
            print(f"\n  [D110-06] {len(ranking_only)} piernas RANKING_ONLY (ranking_gap>300, cuota [1.15,1.60]):")
            for p in ranking_only:
                print(f"    {p['favorito']} @{p['cuota_favorito']:.2f} gap={p['ranking_gap']} ({p['torneo']})")
            picks_validos = picks_validos + ranking_only

    # D146: extender con candidatos H2H_MODEL (picks cuota<1.50 descartados por REGLA-HF-1)
    # Lee h2h_results_enhanced directamente — predicción real del modelo, no estimación.
    h2h_path = _find_latest_h2h()
    if h2h_path:
        edge_picks_set_full = {
            _normalize_name(p.get("favorito_predicho", p.get("favorito", p.get("jugador", ""))))
            for p in picks_validos
        }
        h2h_picks = _leer_h2h_favoritos(h2h_path, edge_picks_set_full)
        if h2h_picks:
            logger.info(f"[D146] {len(h2h_picks)} piernas H2H_MODEL añadidas al universo")
            print(f"\n  [D146] {len(h2h_picks)} piernas H2H_MODEL (cuota [{LEG_MIN_CUOTA},{LEG_MAX_CUOTA}], conf>=55%, modelo real):")
            for p in h2h_picks:
                print(f"    {p['favorito']} @{p['cuota_favorito']:.2f} conf={p['p_modelo']*100:.1f}% ({p['torneo']})")
            picks_validos = picks_validos + h2h_picks

    # Guard: descartar picks sin nombre (edge_report picks con favorito vacío)
    picks_validos = [p for p in picks_validos
                     if (p.get("favorito") or p.get("jugador") or "").strip()]

    # Pre-filtro Kambi: solo piernas apostables en Betplay (D146-fix)
    _outcomes_map: Dict = {}
    _started_map: Dict = {}
    try:
        from betplay_combo_builder import fetch_kambi_outcomes, find_outcome as _find_oc
        _outcomes_map, _started_map = fetch_kambi_outcomes()
        picks_kambi = []
        _excl_favoritos: list = []  # D173-10 (Nodo-173): sin outcome en Kambi
        for p in picks_validos:
            jugador = p.get("favorito", p.get("jugador", ""))
            cuota = float(p.get("cuota_favorito", 0))
            oc, _ = _find_oc(jugador, cuota, _outcomes_map, _started_map,
                              outcome_id_hint=p.get("outcome_id"))  # D174-08 Nodo-174
            if not oc:
                _excl_favoritos.append(p)
            if oc:
                p["_kambi_oid"] = str(oc["outcome_id"])
                # Actualizar cuota con valor real de Kambi (evita drift entre h2h y Betplay)
                kambi_cuota = float(oc.get("odds", cuota))
                if kambi_cuota >= LEG_MIN_CUOTA:
                    p["cuota_favorito"] = kambi_cuota
                picks_kambi.append(p)
        n_antes = len(picks_validos)
        # D173-10 (Nodo-173): rastro auditable de lo descartado. No cambia el gate.
        if _excl_favoritos:
            try:
                from core.combo_exclusions import registrar_exclusiones
                registrar_exclusiones('favoritos', _excl_favoritos,
                                      motivo='sin_outcome_kambi')
            except Exception:  # noqa: BLE001 — observabilidad nunca tumba el builder
                pass
        if picks_kambi:
            picks_validos = picks_kambi
            print(f"\n  [Kambi] {len(picks_kambi)}/{n_antes} piernas confirmadas en Betplay"
                  f" ({n_antes - len(picks_kambi)} descartadas sin cobertura):")
            for p in picks_kambi:
                print(f"    {p.get('favorito', '')} @{p.get('cuota_favorito', 0):.2f}"
                      f"  ({p.get('fuente','edge')})")
        else:
            logger.warning("[kambi-prefilter] Ninguna pierna en Kambi — usando universo sin filtro")
    except ImportError:
        pass

    if len(picks_validos) < LEGS_MIN:
        print(f"\n[FAVORITOS_COMPUESTOS] Sin combo posible hoy ({len(picks_validos)} piernas validas < {LEGS_MIN}).")
        print("  Ver desglose arriba para accion especifica.")
        sys.exit(0)

    # Armar combos
    combos = armar_combos(picks_validos, mega=args.mega)

    if not combos:
        print(f"\n[FAVORITOS_COMPUESTOS] {len(picks_validos)} piernas validas pero ninguna combinacion")
        print(f"  cumple cuota [{COMBO_MIN_CUOTA}, {COMBO_MAX_CUOTA}] con diversificacion por torneo.")
        print("  Probar con --mega para ampliar rango de cuotas.")
        sys.exit(0)

    # Tope sesion
    stake_total = len(combos) * STAKE_PER_COMBO
    if stake_total > TOPE_SESION:
        combos = combos[: TOPE_SESION // STAKE_PER_COMBO]
        stake_total = len(combos) * STAKE_PER_COMBO

    # Obtener outcome IDs de Kambi (reutiliza pre-filtro si ya corrió)
    combos_con_ids: List[Dict] = []
    try:
        from betplay_combo_builder import fetch_kambi_outcomes, find_outcome
        outcomes_map = _outcomes_map if _outcomes_map else fetch_kambi_outcomes()[0]
        started_map = _started_map if _started_map else fetch_kambi_outcomes()[1]

        for i, combo in enumerate(combos, start=1):
            ids = []
            legs_display = []
            for pick in combo["legs"]:
                jugador = pick.get("favorito", pick.get("jugador", ""))
                cuota = float(pick.get("cuota_favorito", 0))
                # Reusar outcome_id del pre-filtro si está disponible (evita re-lookup con cuota drifteada)
                if pick.get("_kambi_oid"):
                    ids.append(pick["_kambi_oid"])
                else:
                    oc, razon = find_outcome(jugador, cuota, outcomes_map, started_map,
                                              outcome_id_hint=pick.get("outcome_id"))  # D174-08 Nodo-174
                    if oc:
                        ids.append(str(oc["outcome_id"]))
                    else:
                        logger.warning(f"  [kambi] {jugador}@{cuota:.2f} → {razon}")
                legs_display.append({"jugador": jugador, "cuota": cuota})

            url = _build_betplay_url(ids) if len(ids) == len(combo["legs"]) else None
            redirect = _build_redirect_url(ids) if ids else None
            combos_con_ids.append({
                **combo,
                "url": url,
                "redirect_url": redirect,
                "outcome_ids": ids,
                "legs_display": legs_display,
                "combo_idx": i,
            })
    except ImportError:
        logger.warning("[kambi] betplay_combo_builder no disponible — sin outcome IDs")
        combos_con_ids = [
            {**c, "url": None, "redirect_url": None, "outcome_ids": [], "legs_display": [
                {"jugador": p.get("favorito", p.get("jugador", "")),
                 "cuota": float(p.get("cuota_favorito", 0))}
                for p in c["legs"]
            ], "combo_idx": i}
            for i, c in enumerate(combos, start=1)
        ]

    # Imprimir resumen
    print("\n=== FAVORITOS_COMPUESTOS — Combos generados ===")
    for c in combos_con_ids:
        legs_str = " + ".join(f"{l['jugador']}@{l['cuota']:.2f}" for l in c["legs_display"])
        print(f"  FavCombo{c['combo_idx']}: {legs_str}")
        print(f"    Cuota: @{c['cuota_total']:.2f}x | P(win): {c['prob_win']*100:.1f}% | Stake: ${c['stake']}")
        if c.get("redirect_url"):
            print(f"    Link: {c['redirect_url']}")
    print(f"\n  Stake total sesion: ${stake_total} (tope ${TOPE_SESION})")
    print("  [H110-01 ACUMULANDO — semilla 8/8 hits jul-14/16]")

    if args.dry_run:
        print("\n  [dry-run] Sin .bat generados.")
        return

    # Generar .bat
    n_bat = _generar_bat(combos_con_ids)
    if n_bat:
        print(f"\n  {n_bat} FavCombo*.bat generados en escritorio.")

    # Registrar en shadow_book
    fecha_hoy = date.today().isoformat()
    _registrar_shadow_book(combos, fecha_hoy)

    # Telegram
    if args.telegram:
        _enviar_telegram(combos_con_ids if combos_con_ids else combos)


if __name__ == "__main__":
    main()
