"""
scraping/dual_book_client.py — Nodo-111 X1: Best-Price Execution Router.

Fuentes:
  Book 1: Kambi offering 'betplay' (verificado HTTP 200 desde WSL y Windows 2026-07-17).
  Book 2: cualquier feed con schema {nombre_normalizado: {"odds": float, ...}} —
          hoy: output del FlashScore odds scraper (Nodo-48) via --book2 archivo.json.
          Mañana: otro offering Kambi si algún key responde (betcris/luckia/sportiumco/
          wplay/rushbet → TODOS 429 sin sesión del sitio, verificado 2026-07-17 desde
          WSL y Windows con headers de navegador — el CDN exige cookie de sesión del skin).

Solo stdlib (urllib) — corre sin venv. Funciones puras testeables (REGLA-T53).
Uso:
  python3 scraping/dual_book_client.py --compare [--book2 reports/flashscore_odds_X.json]
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

KAMBI_URL = ("https://eu-offering-api.kambicdn.com/offering/v2018/{offering}"
             "/listView/tennis.json?lang=es_CO&market=CO&channel_id=1")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def _norm(name: str) -> str:
    """Misma normalización mínima que betplay_combo_builder._normalize_name espera:
    lowercase, sin diacríticos comunes, sin puntuación."""
    s = (name or "").lower().strip()
    for a, b in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ü","u"),("ñ","n"),("ç","c"),("-"," ")]:
        s = s.replace(a, b)
    return " ".join("".join(c for c in s if c.isalnum() or c.isspace()).split())


def fetch_kambi(offering: str = "betplay", retries: int = 2) -> dict:
    """Feed Kambi → {norm_name: {odds, jugador, rival, event_id, outcome_id}}.
    Backoff simple ante 429. Solo eventos NOT_STARTED con betOffers."""
    req = urllib.request.Request(KAMBI_URL.format(offering=offering), headers={
        "User-Agent": UA, "Accept": "application/json",
        "Origin": "https://betplay.com.co", "Referer": "https://betplay.com.co/",
    })
    for i in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.load(r)
            break
        except Exception as e:
            if i == retries:
                print(f"  [dual_book] {offering}: {e}", file=sys.stderr)
                return {}
            time.sleep(3 * (i + 1))
    out = {}
    for ev_w in data.get("events", []):
        ev, offers = ev_w.get("event", {}), ev_w.get("betOffers", [])
        if not offers or ev.get("state") != "NOT_STARTED":
            continue
        home, away = ev.get("homeName", ""), ev.get("awayName", "")
        for oc in offers[0].get("outcomes", []):
            jug = home if oc.get("type") == "OT_ONE" else away if oc.get("type") == "OT_TWO" else None
            if not jug or not oc.get("odds"):
                continue
            out[_norm(jug)] = {"odds": oc["odds"] / 1000, "jugador": jug,
                               "rival": away if jug == home else home,
                               "event_id": ev.get("id"), "outcome_id": oc.get("id")}
    return out


# ── Funciones puras (Nodo-111 §3.2 — testeables sin red) ─────────────────────

def best_price(name: str, feeds: dict) -> dict | None:
    """feeds = {nombre_casa: feed_dict}. Retorna la mejor cuota disponible.
    {'casa','cuota','delta_pct'} — delta vs la peor casa que también lo lista."""
    n = _norm(name)
    quotes = {casa: f[n]["odds"] for casa, f in feeds.items() if n in f and f[n].get("odds")}
    if not quotes:
        return None
    casa = max(quotes, key=quotes.get)
    worst = min(quotes.values())
    return {"casa": casa, "cuota": quotes[casa],
            "delta_pct": round((quotes[casa] / worst - 1) * 100, 2) if worst else 0.0,
            "cuotas": quotes}


def divergencia(o1: float, o2: float) -> float:
    """% de divergencia entre dos cuotas del mismo outcome."""
    lo = min(o1, o2)
    return round((max(o1, o2) / lo - 1) * 100, 2) if lo else 0.0


def es_arb(cuota_a_jug1: float, cuota_b_jug2: float) -> bool:
    """Arb 2-way: mejor cuota jugador1 en casa A + mejor jugador2 en casa B."""
    return cuota_a_jug1 > 1 and cuota_b_jug2 > 1 and (1/cuota_a_jug1 + 1/cuota_b_jug2) < 1.0


def es_middle(linea_over: float, linea_under: float, rango_modelo: tuple) -> bool:
    """OVER linea_over (casa A) + UNDER linea_under (casa B): middle informado si
    la ventana existe (under > over) y el rango modelado de games cae dentro."""
    if linea_under <= linea_over:
        return False
    lo, hi = rango_modelo
    return lo >= linea_over and hi <= linea_under


# ── CLI ──────────────────────────────────────────────────────────────────────

def _latest(pattern: str):
    files = sorted(Path("reports").glob(pattern))
    return files[-1] if files else None


def main():
    args = sys.argv[1:]
    book2_path = args[args.index("--book2") + 1] if "--book2" in args else None
    feeds = {"betplay": fetch_kambi("betplay")}
    if book2_path and Path(book2_path).exists():
        raw = json.loads(Path(book2_path).read_text(encoding="utf-8"))
        # acepta:
        # (a) {partidos:[{jugador,cuota}...]}  — formato explícito
        # (b) {torneo:[{jugador1,cuota1,jugador2,cuota2}...]}  — formato zita (Nodo-48)
        # (c) {nombre: {odds:...}}  — formato plano
        if "partidos" in raw:
            feeds["flashscore"] = {_norm(p.get("jugador","")): {"odds": p.get("cuota")}
                                   for p in raw["partidos"] if p.get("cuota")}
        elif isinstance(raw, dict) and all(isinstance(v, list) for v in raw.values()):
            # formato zita: cada value es lista de partidos con jugador1/cuota1/jugador2/cuota2
            fs = {}
            for partidos in raw.values():
                for m in partidos:
                    if m.get("jugador1") and m.get("cuota1"):
                        fs[_norm(m["jugador1"])] = {"odds": m["cuota1"]}
                    if m.get("jugador2") and m.get("cuota2"):
                        fs[_norm(m["jugador2"])] = {"odds": m["cuota2"]}
            feeds["flashscore"] = fs
        else:
            feeds["flashscore"] = {_norm(k): v for k, v in raw.items()}
    ep = _latest("edge_report_*.json")
    if not ep:
        print("SIN edge_report en reports/ — correr: python3 edge_calculator.py")
        return
    edata = json.loads(ep.read_text(encoding="utf-8"))
    picks = (edata.get("apostar") or []) + (edata.get("watchlist") or [])
    print(f"ROUTER X1 — {ep.name} | feeds: {', '.join(feeds)} "
          f"({', '.join(str(len(f)) for f in feeds.values())} outcomes)")
    mejoras = []
    for p in picks:
        jug = p.get("favorito_predicho", "")
        bp = best_price(jug, feeds)
        if bp:
            base = p.get("cuota_favorito") or 0
            gain = round((bp["cuota"] / base - 1) * 100, 2) if base else 0.0
            mejoras.append(gain if gain > 0 else 0)
            print(f"  {jug:<28} plan @{base:<5} → mejor: {bp['casa']} @{bp['cuota']:<5} "
                  f"(+{gain}% vs plan | libros: {bp['cuotas']})")
        else:
            print(f"  {jug:<28} SIN COBERTURA en feeds activos")
    if mejoras:
        print(f"\n  ROI extra por routing (media picks cubiertos): +{sum(mejoras)/len(mejoras):.2f}%")


if __name__ == "__main__":
    main()
