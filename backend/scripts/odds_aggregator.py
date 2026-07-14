"""
scripts/odds_aggregator.py — OddsAggregator multi-casa Colombia (D90-08 Nodo-90)

Lee el edge_report más reciente, consulta cuotas en casas colombianas y reporta:
  - best_odds / best_bookmaker por jugador
  - odds_by_book (todas las casas disponibles)
  - clv_potential vs betplay (referencia)

OBSERVACIONAL — no modifica apostar ni kelly. Slot para WPlay/Codere cuando
se confirmen endpoints vía DevTools.

Uso:
  python3 scripts/odds_aggregator.py
  python3 scripts/odds_aggregator.py --bookmakers betplay,betcris,luckia
  python3 scripts/odds_aggregator.py --jugador "Alcaraz"
  python3 scripts/odds_aggregator.py --show-all   # muestra también no-disponibles
"""
import argparse
import json
import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_DIR    = Path(__file__).parent.parent
REPORTS_DIR = BASE_DIR / "reports"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ─── Casas colombianas ────────────────────────────────────────────────────────
#
# api_type:
#   "kambi"   → us.offering-api.kambicdn.com  (offering_key = clave del cliente)
#   "sbtech"  → endpoint SBTech (pendiente verificación DevTools en wplay.co)
#   "custom"  → endpoint propio (pendiente verificación)
#
# offering_key:
#   "VERIFIED"          → confirmado y activo
#   "PENDING_DEVTOOLS"  → bloqueado desde WSL, requiere Chrome DevTools en sitio
#
BOOKMAKERS_CO: Dict[str, Dict] = {
    "betplay": {
        "api_type":     "kambi",
        "offering_key": "betplay",          # VERIFIED — us.offering-api.kambicdn.com
        "base_url":     "https://us.offering-api.kambicdn.com/offering/v2018/betplay",
        "params":       "lang=es_CO&market=CO&channel_id=1&client_id=2",
        "priority":     1,
        "status":       "VERIFIED",
    },
    "betcris": {
        "api_type":     "kambi",
        "offering_key": "PENDING_DEVTOOLS",  # betcris.com.co — IP bloqueada desde WSL
        "base_url":     None,
        "params":       "lang=es_CO&market=CO&channel_id=1&client_id=2",
        "priority":     2,
        "status":       "PENDING_DEVTOOLS",
    },
    "luckia": {
        "api_type":     "kambi",
        "offering_key": "PENDING_DEVTOOLS",  # luckia.co — IP bloqueada desde WSL
        "base_url":     None,
        "params":       "lang=es_CO&market=CO&channel_id=1&client_id=2",
        "priority":     3,
        "status":       "PENDING_DEVTOOLS",
    },
    "sportium": {
        "api_type":     "kambi",
        "offering_key": "PENDING_DEVTOOLS",  # sportium.co — IP bloqueada desde WSL
        "base_url":     None,
        "params":       "lang=es_CO&market=CO&channel_id=1&client_id=2",
        "priority":     4,
        "status":       "PENDING_DEVTOOLS",
    },
    "wplay": {
        "api_type":     "sbtech",
        "offering_key": "PENDING_DEVTOOLS",  # www.wplay.co/sportsbook/tennis — SPA opaca
        "base_url":     None,
        "params":       None,
        "priority":     5,
        "status":       "PENDING_DEVTOOLS",
    },
    "codere": {
        "api_type":     "custom",
        "offering_key": "PENDING_DEVTOOLS",
        "base_url":     None,
        "params":       None,
        "priority":     6,
        "status":       "PENDING_DEVTOOLS",
    },
}

KAMBI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":     "application/json",
}


# ─── Normalización (compatible con betplay_combo_builder._normalize_name) ─────

def _norm(name: str) -> str:
    n = unicodedata.normalize("NFD", name.lower())
    n = "".join(c for c in n if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z\s]", "", n).strip()


# ─── Clientes por API type ─────────────────────────────────────────────────────

def _fetch_kambi(book: str, cfg: Dict) -> Tuple[Dict[str, Dict], str]:
    """
    Obtiene outcomes de tenis de una casa Kambi.
    Retorna (outcomes_map, status_msg).
    outcomes_map: nombre_norm → {odds, jugador, rival, event_id, outcome_id}
    """
    import requests

    if cfg["status"] != "VERIFIED" or not cfg.get("base_url"):
        return {}, f"SKIP ({cfg['status']})"

    url = f"{cfg['base_url']}/listView/tennis.json?{cfg['params']}"
    try:
        resp = requests.get(url, headers=KAMBI_HEADERS, timeout=12)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {}, f"ERROR: {e}"

    outcomes: Dict[str, Dict] = {}
    for ev_wrapper in data.get("events", []):
        ev     = ev_wrapper.get("event", {})
        offers = ev_wrapper.get("betOffers", [])
        if not offers or ev.get("state") != "NOT_STARTED":
            continue
        home = ev.get("homeName", "")
        away = ev.get("awayName", "")
        eid  = ev.get("id")
        for oc in offers[0].get("outcomes", []):
            oc_type = oc.get("type", "")
            if oc_type == "OT_ONE":
                jugador, rival = home, away
            elif oc_type == "OT_TWO":
                jugador, rival = away, home
            else:
                continue
            odds   = oc.get("odds", 0) / 1000
            entry  = {
                "odds":       odds,
                "jugador":    jugador,
                "rival":      rival,
                "event_id":   eid,
                "outcome_id": oc.get("id"),
                "bookmaker":  book,
            }
            key = _norm(jugador)
            outcomes[key] = entry
            apellido = key.split()[-1] if key.split() else key
            if apellido != key:
                outcomes[apellido] = entry

    return outcomes, f"OK ({len(outcomes)} outcomes)"


def _fetch_sbtech(book: str, cfg: Dict) -> Tuple[Dict[str, Dict], str]:
    """
    Slot SBTech para WPlay. Activar cuando se confirme endpoint vía DevTools:
      1. Abrir Chrome → www.wplay.co/sportsbook/tennis
      2. DevTools → Network → filtrar 'api.wplay'
      3. Copiar URL exacta + headers requeridos
      4. Actualizar cfg['base_url'] y STATUS → 'VERIFIED'
    """
    return {}, f"SKIP ({cfg['status']}) — requiere DevTools en www.wplay.co/sportsbook/tennis"


def _fetch_custom(book: str, cfg: Dict) -> Tuple[Dict[str, Dict], str]:
    """Slot custom (codere, etc.). Activar cuando se confirme endpoint."""
    return {}, f"SKIP ({cfg['status']})"


_FETCHERS = {
    "kambi":  _fetch_kambi,
    "sbtech": _fetch_sbtech,
    "custom": _fetch_custom,
}


# ─── Lógica principal ─────────────────────────────────────────────────────────

def fetch_all_odds(bookmakers: Optional[List[str]] = None) -> Dict[str, Dict]:
    """
    Consulta todas las casas activas y retorna odds_by_book por jugador.
    {
      "alcaraz": {
        "betplay": {"odds": 1.55, "outcome_id": "...", ...},
        "betcris":  None,   ← no disponible / skip
      }
    }
    """
    books = bookmakers or sorted(BOOKMAKERS_CO.keys(), key=lambda k: BOOKMAKERS_CO[k]["priority"])
    all_data: Dict[str, Dict[str, Optional[Dict]]] = {}

    for book in books:
        cfg     = BOOKMAKERS_CO.get(book)
        if not cfg:
            logger.warning(f"[odds_agg] Casa desconocida: {book}")
            continue
        fetcher = _FETCHERS.get(cfg["api_type"], _fetch_custom)
        outcomes, msg = fetcher(book, cfg)
        logger.info(f"  {book:<12} {msg}")

        for key, entry in outcomes.items():
            if key not in all_data:
                all_data[key] = {}
            all_data[key][book] = entry

        # Marcar ausentes para jugadores ya vistos
        for key in all_data:
            if book not in all_data[key]:
                all_data[key][book] = None

    return all_data


def build_comparison(all_data: Dict[str, Dict], jugador_filter: Optional[str] = None) -> List[Dict]:
    """
    Para cada jugador retorna:
      jugador, best_odds, best_bookmaker, odds_by_book, clv_potential (vs betplay)
    """
    results = []
    filter_norm = _norm(jugador_filter) if jugador_filter else None

    for key, books_data in all_data.items():
        if filter_norm and filter_norm not in key:
            continue

        available = {b: e for b, e in books_data.items() if e is not None}
        if not available:
            continue

        # Nombre canónico: el de la primera casa disponible
        first = next(iter(available.values()))
        jugador_name = first["jugador"]

        odds_by_book = {b: e["odds"] for b, e in available.items()}
        best_book    = max(odds_by_book, key=lambda b: odds_by_book[b])
        best_odds    = odds_by_book[best_book]

        # CLV potential: % mejora vs betplay (casa de referencia)
        bp_odds = odds_by_book.get("betplay")
        clv_potential = round((best_odds / bp_odds - 1.0), 4) if bp_odds else None

        results.append({
            "jugador":        jugador_name,
            "key":            key,
            "best_odds":      best_odds,
            "best_bookmaker": best_book,
            "betplay_odds":   bp_odds,
            "clv_potential":  clv_potential,
            "odds_by_book":   odds_by_book,
        })

    # Ordenar por CLV descendente
    results.sort(key=lambda r: r["clv_potential"] or 0, reverse=True)
    return results


def _load_edge_report() -> Dict[str, float]:
    """Lee jugadores del edge_report más reciente (apostar=True). {nombre: cuota}"""
    plans = sorted(REPORTS_DIR.glob("edge_report_*.json"), reverse=True)
    if not plans:
        return {}
    try:
        data  = json.loads(plans[0].read_text(encoding="utf-8"))
        picks = data.get("apostar", [])
        return {p["favorito_predicho"]: p.get("cuota_favorito", 0) for p in picks if p.get("favorito_predicho")}
    except Exception:
        return {}


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="OddsAggregator multi-casa Colombia (D90-08)")
    parser.add_argument("--bookmakers", help="Casas a consultar separadas por coma (default: todas)")
    parser.add_argument("--jugador",    help="Filtrar por nombre de jugador")
    parser.add_argument("--show-all",   action="store_true", help="Mostrar también sin CLV gain")
    args = parser.parse_args()

    books = [b.strip() for b in args.bookmakers.split(",")] if args.bookmakers else None
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M")

    print(f"\n{'='*64}")
    print(f"ODDS AGGREGATOR COLOMBIA  {ts}")
    print(f"Casas: {', '.join(books or list(BOOKMAKERS_CO.keys()))}")
    print(f"{'='*64}\n")

    # Estado de casas
    print("Estado de casas:")
    for book, cfg in sorted(BOOKMAKERS_CO.items(), key=lambda x: x[1]["priority"]):
        if books and book not in books:
            continue
        icon = "✓" if cfg["status"] == "VERIFIED" else "○"
        print(f"  {icon} {book:<12} [{cfg['api_type']:<7}] {cfg['status']}")
    print()

    # Consulta
    print("Consultando cuotas...")
    all_data = fetch_all_odds(books)

    # Jugadores del edge_report como contexto
    edge_picks = _load_edge_report()
    if edge_picks:
        print(f"\nEdge report: {len(edge_picks)} picks activos")

    # Comparación
    results = build_comparison(all_data, args.jugador)

    if not results:
        print("Sin datos disponibles — solo betplay activo, demás casas PENDING_DEVTOOLS.")
        print("\nPróximo paso para activar más casas:")
        print("  1. Chrome → www.wplay.co/sportsbook/tennis")
        print("  2. DevTools → Network → filtrar 'api.wplay' o 'sbtech'")
        print("  3. Copiar URL + actualizar BOOKMAKERS_CO['wplay']['base_url']")
        print("  4. Cambiar status a 'VERIFIED'")
    else:
        print(f"\n{'Jugador':<22} {'Best':<8} {'Casa':<12} {'Betplay':<9} {'CLV%':<8}")
        print(f"{'─'*22} {'─'*8} {'─'*12} {'─'*9} {'─'*8}")
        for r in results:
            if not args.show_all and (r["clv_potential"] or 0) <= 0:
                continue
            clv_str = f"+{r['clv_potential']*100:.1f}%" if r["clv_potential"] else "—"
            bp_str  = f"{r['betplay_odds']:.2f}" if r["betplay_odds"] else "—"
            print(f"{r['jugador']:<22} {r['best_odds']:<8.2f} {r['best_bookmaker']:<12} {bp_str:<9} {clv_str}")

    # Guardar JSON
    out = {
        "ts":           ts,
        "bookmakers":   {b: BOOKMAKERS_CO[b]["status"] for b in (books or BOOKMAKERS_CO)},
        "edge_context": edge_picks,
        "results":      results,
    }
    out_path = REPORTS_DIR / f"odds_agg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado: {out_path.name}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
