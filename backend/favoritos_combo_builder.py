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

        candidatos.append(pick)

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
                torneo = p.get("torneo", p.get("tournament", "UNK"))
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
    ids_str = "/".join(f"{oid}|ML" for oid in outcome_ids)
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
        archivos = sorted(glob.glob("reports/edge_report_*.json"))
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


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Favoritos Compuestos — estrategia #13 Nodo-110")
    parser.add_argument("--bankroll", type=float, default=125000)
    parser.add_argument("--file", help="Edge report JSON (default: más reciente)")
    parser.add_argument("--dry-run", action="store_true", help="Solo imprimir, sin .bat ni shadow_book")
    parser.add_argument("--mega", action="store_true", help="Incluir piernas spice [2.10, 5.00]")
    parser.add_argument("--telegram", action="store_true", help="Enviar resumen a Telegram")
    parser.add_argument("--override-governor", action="store_true", dest="override_governor")
    args = parser.parse_args()

    # Governor check (S107-D)
    _governor_check(args.bankroll, args.override_governor, "favoritos_combo_builder")

    # Leer edge_report
    edge_report = _leer_edge_report(args.file)
    logger.info(f"[edge_report] {len(edge_report)} picks cargados")

    # Seleccionar
    picks_validos, conteos = seleccionar_favoritos(edge_report, mega=args.mega)
    _imprimir_desglose(conteos, picks_validos)

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

    # Obtener outcome IDs de Kambi
    combos_con_ids: List[Dict] = []
    try:
        from betplay_combo_builder import fetch_kambi_outcomes, find_outcome
        outcomes_map, started_map = fetch_kambi_outcomes()

        for i, combo in enumerate(combos, start=1):
            ids = []
            legs_display = []
            for pick in combo["legs"]:
                jugador = pick.get("favorito", pick.get("jugador", ""))
                cuota = float(pick.get("cuota_favorito", 0))
                oc, razon = find_outcome(jugador, cuota, outcomes_map, started_map)
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
