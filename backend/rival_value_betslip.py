#!/usr/bin/env python3
"""
rival_value_betslip.py — Betslips para señal RIVAL VALUE (H88-01, Nodo-68)

Lee el edge_report más reciente, encuentra picks con rival_value_flag=True
y genera:
  - Stakes individuales con micro-Kelly pre-graduación (n=3/30, shrinkage=5.7%)
  - Link Betplay individual por rival
  - Combo Betplay con TODOS los rivales del día
  - Mensaje Telegram

PROTOCOLO H88-01:
  - Gate: n>=30 picks individuales antes de graduación
  - Actual: n=3 (2026-07-14) — stakes reducidos por shrinkage
  - PROHIBIDO subir stakes antes de n=30
  - Cada apuesta individual cuenta como 1 observación en H88-01

Uso:
  python3 rival_value_betslip.py --bankroll 125000
  python3 rival_value_betslip.py --bankroll 125000 --telegram
  python3 rival_value_betslip.py --bankroll 125000 --dry-run
"""

import argparse
import json
import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# ─── Config (reutiliza constantes de betplay_combo_builder) ───────────────────
KAMBI_BASE    = "https://us.offering-api.kambicdn.com/offering/v2018/betplay"
KAMBI_PARAMS  = "lang=es_CO&market=CO&channel_id=1&client_id=2"
KAMBI_HEADERS = {
    "Referer":    "https://betplay.com.co/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":     "application/json",
}

BETPLAY_URL_BASE = "https://betplay.com.co/apuestas#home?coupon=combination|"
BETPLAY_URL_TAIL = "||replace"
REDIRECT_BASE    = "https://dakotapog.github.io/tennis-analysis/bp/?ids="

TG_TOKEN = "8684706586:AAHv4zhjQKvxORf6bnbwCxZQPly9OA7unpY"
TG_CHAT  = "8520949513"
TG_URL   = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"

REPORTS_DIR  = Path(__file__).parent / "reports"
DESKTOP_WIN  = Path("/mnt/c/users/hogar/Desktop")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ─── H88-01 micro-Kelly pre-graduación ───────────────────────────────────────
H88_N_OBS    = 3    # n_actual actualizado 2026-07-14 (3/3 wins)
H88_K_PRIOR  = 50   # prior conservador pre-graduación
H88_SHRINK   = H88_N_OBS / (H88_N_OBS + H88_K_PRIOR)   # = 0.057 (5.7%)
H88_MAX_PCT  = 0.005  # tope 0.5% bankroll — PROHIBIDO subir antes de n=30
H88_MIN_EDGE = 0.05   # edge mínimo rival para incluir (5%)


def _norm(name: str) -> str:
    """Normaliza nombre: lowercase, sin acentos, solo letras."""
    name = unicodedata.normalize("NFD", name.lower())
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z\s]", "", name).strip()


def _apellido(name: str) -> str:
    """Nodo-156 D156-03: última palabra que NO sea inicial (<=2 chars tras _norm,
    el punto ya fue stripeado). Mismo patrón que games_signal_calculator.py D126-01."""
    parts = _norm(name).split()
    if not parts:
        return _norm(name)
    for p in reversed(parts):
        if len(p) > 2:
            return p
    return parts[0]


# ─── Kambi: mapa favorito_norm → rival outcome_id ─────────────────────────────

def fetch_rival_outcomes() -> Dict[str, Dict]:
    """
    Obtiene de Kambi el outcome_id del RIVAL para cada partido.

    Retorna: {fav_norm → {rival_outcome_id, rival_odds, rival_name, fav_name, event_id}}
    También indexa por apellido del favorito para mayor cobertura.
    """
    url = f"{KAMBI_BASE}/listView/tennis.json?{KAMBI_PARAMS}"
    try:
        resp = requests.get(url, headers=KAMBI_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"  [rival] Kambi error: {e}")
        return {}

    rival_map: Dict[str, Dict] = {}

    for ev_wrapper in data.get("events", []):
        ev     = ev_wrapper.get("event", {})
        offers = ev_wrapper.get("betOffers", [])
        if not offers or ev.get("state") != "NOT_STARTED":
            continue
        home    = ev.get("homeName", "")
        away    = ev.get("awayName", "")
        eid     = ev.get("id")
        offer   = offers[0]

        home_oc = away_oc = None
        for oc in offer.get("outcomes", []):
            t = oc.get("type", "")
            if t == "OT_ONE":
                home_oc = oc
            elif t == "OT_TWO":
                away_oc = oc

        if not home_oc or not away_oc:
            continue

        # Si el favorito es HOME → rival es AWAY (OT_TWO) y viceversa
        for fav_name, fav_oc, rival_name, rival_oc in [
            (home, home_oc, away, away_oc),
            (away, away_oc, home, home_oc),
        ]:
            entry = {
                "rival_outcome_id": rival_oc.get("id"),
                "rival_odds":       rival_oc.get("odds", 0) / 1000,
                "rival_name":       rival_name,
                "fav_name":         fav_name,
                "event_id":         eid,
            }
            for key in [_norm(fav_name), _apellido(fav_name)]:
                if key and key not in rival_map:
                    rival_map[key] = entry

    return rival_map


# ─── Leer edge_report ──────────────────────────────────────────────────────────

def load_rival_picks() -> List[Dict]:
    """Lee el edge_report más reciente y retorna los picks con rival_value_flag=True."""
    reports = sorted(REPORTS_DIR.glob("edge_report_*.json"), reverse=True)
    if not reports:
        logger.warning("  [rival] Sin edge_report en reports/")
        return []
    data = json.loads(reports[0].read_text(encoding="utf-8"))
    all_picks = (
        data.get("apostar", []) +
        data.get("watchlist", []) +
        data.get("sin_edge", [])
    )
    picks = [p for p in all_picks if p.get("rival_value_flag")]
    logger.info(f"  [rival] Edge report: {reports[0].name} — {len(picks)} picks RIVAL VALUE")
    return picks


# ─── Kelly micro pre-graduación ───────────────────────────────────────────────

def micro_kelly(edge_rival: float, cuota_rival: float, bankroll: float) -> float:
    """
    Micro-Kelly H88-01 pre-graduación.

    kelly_raw = edge / (cuota - 1)
    shrinkage = n_obs / (n_obs + K_prior) = 3/53 = 5.7%
    stake = min(kelly_raw * shrinkage, 0.5%) * bankroll
    Redondeado a 500 COP.
    """
    if cuota_rival <= 1.0 or edge_rival < H88_MIN_EDGE:
        return 0.0
    kelly_raw   = edge_rival / (cuota_rival - 1.0)
    kelly_shrunk = kelly_raw * H88_SHRINK
    stake_pct   = min(kelly_shrunk, H88_MAX_PCT)
    stake_raw   = stake_pct * bankroll
    # Redondear a 500 más cercano (mínimo 2000)
    return max(2000.0, round(stake_raw / 500) * 500)


# ─── Generar betslips ──────────────────────────────────────────────────────────

def build_rival_betslips(
    picks: List[Dict],
    rival_map: Dict[str, Dict],
    bankroll: float,
) -> List[Dict]:
    """
    Arma la lista de betslips individuales para cada rival.
    Retorna lista de dicts con todo lo necesario para links y Telegram.
    """
    betslips = []

    for pick in picks:
        fav      = pick.get("favorito_predicho", "")
        partido  = pick.get("partido", "")
        edge_r   = pick.get("edge_vs_mercado_rival", 0.0)
        cuota_r  = pick.get("cuota_rival", 0.0)
        tier     = pick.get("tier", "")
        sup      = pick.get("superficie", "")
        conf     = pick.get("confidence_flag", "")
        edge_fav = pick.get("edge_pct", "")

        if not fav or not cuota_r or edge_r < H88_MIN_EDGE:
            continue

        # Buscar rival outcome en Kambi
        oc_info = (
            rival_map.get(_norm(fav)) or
            rival_map.get(_apellido(fav))
        )
        if not oc_info:
            logger.warning(f"  [rival] Sin outcome_id para rival de {fav}")
            betslips.append({
                "partido":        partido,
                "fav":            fav,
                "rival_name":     pick.get("partido", "?").replace(fav, "").strip(" vs"),
                "cuota_rival":    cuota_r,
                "edge_rival":     edge_r,
                "edge_fav":       edge_fav,
                "tier":           tier,
                "superficie":     sup,
                "conf_flag":      conf,
                "stake":          0,
                "outcome_id":     None,
                "betplay_url":    None,
                "redirect_url":   None,
                "sin_kambi":      True,
            })
            continue

        stake = micro_kelly(edge_r, cuota_r, bankroll)
        oc_id = oc_info["rival_outcome_id"]

        url_individual = f"{BETPLAY_URL_BASE}{oc_id}{BETPLAY_URL_TAIL}"
        redirect_url   = f"{REDIRECT_BASE}{oc_id}"

        betslips.append({
            "partido":        partido,
            "fav":            fav,
            "rival_name":     oc_info["rival_name"],
            "cuota_rival":    cuota_r,
            "kambi_odds":     oc_info["rival_odds"],
            "edge_rival":     edge_r,
            "edge_fav":       edge_fav,
            "tier":           tier,
            "superficie":     sup,
            "conf_flag":      conf,
            "stake":          stake,
            "outcome_id":     oc_id,
            "betplay_url":    url_individual,
            "redirect_url":   redirect_url,
            "sin_kambi":      False,
        })

    return betslips


def build_combo_link(betslips: List[Dict]) -> Optional[str]:
    """Combo Betplay con todos los rivales que tienen outcome_id."""
    ids = [str(b["outcome_id"]) for b in betslips if b.get("outcome_id")]
    if len(ids) < 2:
        return None
    return f"{BETPLAY_URL_BASE}{','.join(ids)}{BETPLAY_URL_TAIL}"


def build_combo_redirect(betslips: List[Dict]) -> Optional[str]:
    ids = [str(b["outcome_id"]) for b in betslips if b.get("outcome_id")]
    if len(ids) < 2:
        return None
    return f"{REDIRECT_BASE}{','.join(ids)}"


# ─── Output ───────────────────────────────────────────────────────────────────

def print_report(betslips: List[Dict], bankroll: float) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    total_stake = sum(b["stake"] for b in betslips)

    print(f"\n{'='*64}")
    print(f"RIVAL VALUE BETSLIPS — H88-01  ({ts})")
    print(f"Bankroll: ${bankroll:,.0f} | n_obs: {H88_N_OBS}/30 | shrinkage: {H88_SHRINK:.1%}")
    print(f"{'='*64}\n")

    for b in betslips:
        estado = "SIN KAMBI" if b.get("sin_kambi") else "OK"
        print(f"  {b['partido']}")
        print(f"  RIVAL: {b['rival_name']} @{b['cuota_rival']:.2f} | edge={b['edge_rival']*100:.1f}% | fav_edge={b['edge_fav']}")
        print(f"  Tier: {b['tier']} | Sup: {b['superficie']} | Conf: {b['conf_flag']}")
        print(f"  Stake: ${b['stake']:,.0f} | Kambi: {estado}")
        if b.get("redirect_url"):
            print(f"  Link: {b['redirect_url']}")
        print()

    combo_url = build_combo_link(betslips)
    combo_red = build_combo_redirect(betslips)
    if combo_url:
        print(f"  COMBO ({len([b for b in betslips if b.get('outcome_id')])} rivales):")
        print(f"  {combo_red}")
        cuota_combo = 1.0
        for b in betslips:
            if b.get("outcome_id"):
                cuota_combo *= b["cuota_rival"]
        print(f"  Cuota combinada: x{cuota_combo:.2f}")

    print(f"\n  Stake total (individuales): ${total_stake:,.0f}")
    print(f"  Pct bankroll: {total_stake/bankroll*100:.2f}%")
    print(f"\n  NOTA: n={H88_N_OBS}/30 — micro-Kelly con shrinkage {H88_SHRINK:.1%}")
    print(f"        Gate H88-01: n>=30 antes de incrementar stakes.")
    print(f"{'='*64}\n")


def build_telegram_msg(betslips: List[Dict], bankroll: float) -> str:
    ts = datetime.now().strftime("%d/%m %H:%M")
    lines = [
        f"RIVAL VALUE H88-01 | {ts}",
        f"n={H88_N_OBS}/30 | shrink={H88_SHRINK:.0%} | bkr=${bankroll:,.0f}",
        "",
    ]
    for b in betslips:
        ok = "" if b.get("sin_kambi") else ""
        lines.append(f"{ok} {b['rival_name']} @{b['cuota_rival']:.2f}")
        lines.append(f"   edge={b['edge_rival']*100:.1f}% | stake=${b['stake']:,.0f}")
        if b.get("redirect_url"):
            lines.append(f"   {b['redirect_url']}")
        lines.append("")

    combo_red = build_combo_redirect(betslips)
    if combo_red:
        cuota_combo = 1.0
        for b in betslips:
            if b.get("outcome_id"):
                cuota_combo *= b["cuota_rival"]
        lines.append(f"COMBO x{cuota_combo:.1f}")
        lines.append(combo_red)

    lines.append("")
    lines.append(f"Gate: {H88_N_OBS}/30 obs. NO subir stakes.")
    return "\n".join(lines)


def enviar_telegram(msg: str) -> bool:
    try:
        r = requests.post(TG_URL, json={
            "chat_id": TG_CHAT,
            "text": msg,
            "disable_web_page_preview": False,
        }, timeout=10)
        return r.ok
    except Exception as e:
        logger.error(f"  [rival] Telegram error: {e}")
        return False


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Rival Value betslips H88-01")
    parser.add_argument("--bankroll", type=float, default=125000,
                        help="Bankroll activo (default: 125000)")
    parser.add_argument("--telegram", action="store_true",
                        help="Enviar resumen a Telegram")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo mostrar, no enviar Telegram")
    parser.add_argument("--override-governor", action="store_true",
                        help="Omitir bloqueo del governor (D107-04) — queda logueado en combo_governor.log")
    args = parser.parse_args()

    # ── Governor soft-veto (S107-D D107-04) ────────────────────────────────
    import subprocess as _sp, sys
    _gov = _sp.run(
        [sys.executable, str(Path(__file__).parent / 'combo_governor.py'),
         '--bankroll', str(args.bankroll)],
        capture_output=True, text=True
    )
    if _gov.returncode != 0:
        _nivel = 'WARN' if _gov.returncode == 1 else 'BLOCK'
        print(_gov.stdout)
        print(f"[rival_value_betslip] Governor [{_nivel}] — presupuesto comprometido.")
        if args.override_governor:
            from datetime import datetime as _dt
            _lp = Path(__file__).parent / 'logs' / 'combo_governor.log'
            _lp.parent.mkdir(exist_ok=True)
            _lp.open('a').write(
                f"[{_dt.now().strftime('%Y-%m-%d %H:%M')}] OVERRIDE por rival_value_betslip nivel={_nivel}\n"
            )
            print("[rival_value_betslip] --override-governor activo — continuando. Override logueado.")
        else:
            print("[rival_value_betslip] Para continuar: agregar --override-governor")
            print("[rival_value_betslip] Para reducir: ver orden de corte arriba.")
            sys.exit(_gov.returncode)

    logger.info("\n[RIVAL VALUE] Cargando picks...")
    picks = load_rival_picks()
    if not picks:
        print("Sin picks RIVAL VALUE hoy.")
        return

    logger.info("[RIVAL VALUE] Consultando Kambi para outcome_ids del rival...")
    rival_map = fetch_rival_outcomes()
    logger.info(f"  Kambi: {len(rival_map)} outcomes cargados")

    betslips = build_rival_betslips(picks, rival_map, args.bankroll)
    if not betslips:
        print("Sin betslips generados (edge < 5% o sin match en Kambi).")
        return

    print_report(betslips, args.bankroll)

    if args.telegram and not args.dry_run:
        msg = build_telegram_msg(betslips, args.bankroll)
        ok = enviar_telegram(msg)
        print(f"  Telegram: {'OK' if ok else 'ERROR'}")


if __name__ == "__main__":
    main()
