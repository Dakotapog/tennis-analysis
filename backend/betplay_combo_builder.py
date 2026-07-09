#!/usr/bin/env python3
"""
betplay_combo_builder.py — Arma combos de Betplay automáticamente

Lee el trader_plan JSON, mapea cada jugador a su outcome_id de Kambi,
genera archivos .bat en el escritorio que abren Chrome con cada combo
pre-cargado en el bet slip de Betplay.

Modo 1 — Chrome .bat (default):
  Genera Combo1.bat ... ComboN.bat en el escritorio.
  Doble clic → Chrome abre Betplay con el combo cargado.
  Usuario: borra ticket anterior (X) → pon stake → apuesta → siguiente .bat

Modo 2 — Consola F12 (--console):
  Imprime los comandos location.hash para pegar en la consola de Betplay.

Modo 3 — Telegram (--telegram):
  Envía resumen de combos a Telegram (sin deep links — solo info).

Modo 4 — WhatsApp (--whatsapp):
  Genera HTML con botones que abren wa.me link por link.
  Cada botón → WhatsApp Web → tap Enviar → link llega al celular.
  Desde el celular: tap link → Chrome → Betplay con combo cargado.
  Patrón VoltSafe RF-06: window.open(whatsapp://send?phone=...)

Uso:
  python betplay_combo_builder.py                        # genera .bat en escritorio
  python betplay_combo_builder.py --console              # comandos para F12
  python betplay_combo_builder.py --telegram             # envía resumen a Telegram
  python betplay_combo_builder.py --whatsapp             # HTML con botones wa.me
  python betplay_combo_builder.py --file trader_plan.json
  python betplay_combo_builder.py --dry-run              # solo mostrar
"""

import argparse
import json
import logging
import os
import re
import sys
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from edge_calculator import GATE_VERSION as _EXPECTED_GATE_VERSION, P_MODELO_MIN_UNDERDOG as _P_MODELO_MIN_UNDERDOG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# KAMBI API — Obtener outcome IDs para los jugadores del trader
# ══════════════════════════════════════════════════════════════════════════════

KAMBI_BASE = "https://us.offering-api.kambicdn.com/offering/v2018/betplay"
KAMBI_PARAMS = "lang=es_CO&market=CO&channel_id=1&client_id=2"
KAMBI_HEADERS = {
    "Referer": "https://betplay.com.co/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

# Betplay coupon URL format
# ||replace → REEMPLAZA betslip en cada carga fresca (no acumula localStorage)
# ||append → ACUMULA sobre estado previo en localStorage (problema multi-combo)
BETPLAY_URL_BASE = "https://betplay.com.co/apuestas#home?coupon=combination|"
BETPLAY_URL_TAIL = "||replace"

# Redirect page para Telegram móvil (GitHub Pages — preserva #hash)
REDIRECT_BASE = "https://dakotapog.github.io/tennis-analysis/bp/?ids="

# Chrome path en Windows (accesible desde WSL vía /mnt/c/)
CHROME_WIN = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

# Escritorio Windows
DESKTOP_WIN = Path("/mnt/c/users/hogar/Desktop")
COMBOS_DIR = DESKTOP_WIN / "combos"

# Telegram
TG_TOKEN = "8684706586:AAHv4zhjQKvxORf6bnbwCxZQPly9OA7unpY"
TG_CHAT = "8520949513"
TG_URL = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"

# WhatsApp — mismo patrón que VoltSafe RF-06 (seguridad.service.ts:172)
WA_PHONE = "573237554356"


def _normalize_name(name: str) -> str:
    """Normaliza nombre: lowercase, sin acentos."""
    name = unicodedata.normalize("NFD", name.lower())
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^a-z\s]", "", name)
    return name.strip()


def fetch_kambi_outcomes() -> tuple[Dict[str, Dict], Dict[str, str]]:
    """
    Obtiene TODOS los outcomes activos de tenis de Kambi.

    Returns:
        Tuple de:
        - Dict mapeando nombre_normalizado → {outcome_id, odds, jugador, rival, ...}
        - Dict mapeando nombre_normalizado → estado del evento (para diagnóstico)
    """
    import requests

    url = f"{KAMBI_BASE}/listView/tennis.json?{KAMBI_PARAMS}"

    try:
        resp = requests.get(url, headers=KAMBI_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"❌ Error consultando Kambi: {e}")
        return {}, {}

    events = data.get("events", [])
    logger.info(f"⚡ Kambi: {len(events)} eventos de tenis")

    outcomes_map = {}
    started_map = {}

    for ev_wrapper in events:
        ev = ev_wrapper.get("event", {})
        offers = ev_wrapper.get("betOffers", [])
        state = ev.get("state", "")

        if not offers:
            continue

        home = ev.get("homeName", "")
        away = ev.get("awayName", "")
        event_id = ev.get("id")

        # Track STARTED events for diagnostics
        if state != "NOT_STARTED":
            for name in (home, away):
                norm = _normalize_name(name)
                started_map[norm] = state
                parts = norm.split()
                if parts and parts[-1] != norm:
                    started_map[parts[-1]] = state
            continue

        outcomes = offers[0].get("outcomes", [])
        for oc in outcomes:
            oc_type = oc.get("type", "")
            outcome_id = oc.get("id")
            odds = oc.get("odds", 0) / 1000

            if oc_type == "OT_ONE":
                jugador, rival = home, away
            elif oc_type == "OT_TWO":
                jugador, rival = away, home
            else:
                continue

            norm = _normalize_name(jugador)
            parts = norm.split()
            apellido = parts[-1] if parts else norm

            entry = {
                "outcome_id": outcome_id,
                "odds": odds,
                "event_id": event_id,
                "event_name": ev.get("name", ""),
                "jugador": jugador,
                "rival": rival,
            }

            outcomes_map[norm] = entry
            if apellido != norm:
                outcomes_map[apellido] = entry

    logger.info(f"   ✅ {len(outcomes_map)} outcomes indexados | {len(started_map)} en juego")
    return outcomes_map, started_map


def find_outcome(jugador: str, cuota: float, outcomes_map: Dict,
                  started_map: Optional[Dict] = None) -> tuple[Optional[Dict], str]:
    """
    Busca el outcome de un jugador. Matching por nombre + cuota ±15%.

    Returns:
        Tuple de (outcome_dict o None, razón del fallo o 'OK')
    """
    norm = _normalize_name(jugador)
    parts = norm.split()
    apellido = parts[-1] if parts else norm

    # Intento 1: nombre completo
    if norm in outcomes_map:
        oc = outcomes_map[norm]
        if abs(oc["odds"] - cuota) / cuota < 0.15:
            return oc, "OK"

    # Intento 2: apellido
    if apellido in outcomes_map:
        oc = outcomes_map[apellido]
        if abs(oc["odds"] - cuota) / cuota < 0.15:
            return oc, "OK"

    # Intento 3: substring match
    for key, oc in outcomes_map.items():
        key_parts = set(key.split())
        norm_parts = set(norm.split())
        long_overlap = {p for p in key_parts & norm_parts if len(p) >= 4}
        if long_overlap and abs(oc["odds"] - cuota) / cuota < 0.15:
            return oc, "OK"

    # Diagnóstico: ¿por qué no se encontró?
    if started_map:
        if norm in started_map:
            return None, f"STARTED ({started_map[norm]})"
        if apellido in started_map:
            return None, f"STARTED ({started_map[apellido]})"

    # Check if name exists but odds differ
    if norm in outcomes_map:
        oc = outcomes_map[norm]
        diff = abs(oc["odds"] - cuota) / cuota * 100
        return None, f"CUOTA_DIFF ({oc['odds']:.2f} vs {cuota:.2f}, diff {diff:.0f}%)"

    return None, "NO_EXISTE en Kambi"


# ══════════════════════════════════════════════════════════════════════════════
# COMBO BUILDER — Mapea trader combos → outcome IDs
# ══════════════════════════════════════════════════════════════════════════════

def build_combo_links(trader_plan: Dict, min_piernas: int = 2) -> List[Dict]:
    """
    Lee el trader_plan y mapea cada combo a outcome IDs de Kambi.

    Si alguna pierna no está disponible (STARTED, no existe, etc.),
    genera el combo parcial con las piernas disponibles (mínimo min_piernas).

    Returns:
        Lista de dicts con outcome_ids, betplay_url, stake, etc.
    """
    combos = trader_plan.get("cobertura", [])
    if not combos:
        logger.warning("⚠️ No hay combos en el trader_plan")
        return []

    outcomes_map, started_map = fetch_kambi_outcomes()
    if not outcomes_map:
        logger.error("❌ No se pudieron obtener outcomes de Kambi")
        return []

    results = []
    seen_combos = set()  # Avoid duplicate partial combos

    for i, combo in enumerate(combos, 1):
        legs = combo.get("legs", [])
        outcome_ids = []
        mapped_legs = []
        skipped_legs = []

        for leg in legs:
            jugador = leg["jugador"]
            cuota = leg["cuota"]
            oc, reason = find_outcome(jugador, cuota, outcomes_map, started_map)

            if oc:
                outcome_ids.append(str(oc["outcome_id"]))
                mapped_legs.append({
                    "jugador": jugador,
                    "cuota": cuota,
                    "cuota_kambi": oc["odds"],
                    "outcome_id": oc["outcome_id"],
                })
                logger.info(f"  ✅ {jugador} @{cuota} → outcome {oc['outcome_id']} (@{oc['odds']})")
            else:
                skipped_legs.append({
                    "jugador": jugador,
                    "cuota": cuota,
                    "outcome_id": None,
                    "error": reason,
                })
                logger.warning(f"  ❌ {jugador} @{cuota} — {reason}")

        # Build URL: full combo if all found, partial if >= min_piernas
        betplay_url = None
        is_partial = len(skipped_legs) > 0
        combo_key = tuple(sorted(outcome_ids))

        if len(outcome_ids) >= min_piernas and combo_key not in seen_combos:
            seen_combos.add(combo_key)
            ids_str = ",".join(outcome_ids)
            betplay_url = f"{BETPLAY_URL_BASE}{ids_str}{BETPLAY_URL_TAIL}"

            # Recalculate combo odds for partial
            cuota_combo = combo.get("cuota_combo", 0)
            if is_partial:
                cuota_combo = 1.0
                for ml in mapped_legs:
                    cuota_combo *= ml["cuota_kambi"]
                cuota_combo = round(cuota_combo, 2)

        results.append({
            "combo_idx": i,
            "piernas": len(mapped_legs),
            "piernas_original": combo.get("piernas_n", len(legs)),
            "legs": mapped_legs,
            "skipped_legs": skipped_legs,
            "outcome_ids": outcome_ids,
            "url": betplay_url,
            "partial": is_partial and betplay_url is not None,
            "stake": combo.get("stake", 0),
            "cuota_combo": cuota_combo if betplay_url else combo.get("cuota_combo", 0),
            "retorno": combo.get("retorno_potencial", 0),
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CHROME .BAT — Archivos en escritorio que abren Chrome con cada combo
# ══════════════════════════════════════════════════════════════════════════════

def generar_bat_chrome(combo_links: List[Dict]) -> int:
    """
    Genera Combo1.bat ... ComboN.bat en el escritorio de Windows.
    Cada .bat abre Chrome con un HTML de redirección que preserva los |
    en la URL de Betplay (cmd.exe rompe | en línea de comandos).

    Returns:
        Cantidad de .bat generados.
    """
    COMBOS_DIR.mkdir(exist_ok=True)

    # Limpiar sesión anterior: borrar Combo*.bat y combo*.html huérfanos
    # Los .bat son efímeros — solo válidos para la sesión activa.
    # Sin limpieza, combos de días anteriores acumulan en el escritorio
    # y pueden abrirse accidentalmente con picks ya jugados.
    for old_bat in DESKTOP_WIN.glob("Combo*.bat"):
        old_bat.unlink(missing_ok=True)
    for old_html in COMBOS_DIR.glob("combo*.html"):
        old_html.unlink(missing_ok=True)
    logger.info("🧹 Escritorio limpio — combos anteriores eliminados")

    valid = [c for c in combo_links if c["url"]]
    if not valid:
        logger.error("❌ No hay combos válidos para generar .bat")
        return 0

    for c in valid:
        idx = c["combo_idx"]
        url = c["url"]
        legs_str = " + ".join(f"{l['jugador']}@{l['cuota']:.2f}" for l in c["legs"])

        # HTML de redirección (preserva | en la URL)
        html_content = (
            f"<html><head><title>Combo {idx}</title></head><body>\n"
            f'<script>window.location.replace("{url}");</script>\n'
            f"<p>Redirigiendo a Betplay... Combo {idx}: {legs_str}</p>\n"
            f"</body></html>"
        )
        html_path = COMBOS_DIR / f"combo{idx}.html"
        html_path.write_text(html_content, encoding="utf-8")

        # .bat que abre Chrome con el HTML local
        html_win_path = f"C:\\users\\hogar\\Desktop\\combos\\combo{idx}.html"
        bat_content = (
            f"@echo off\r\n"
            f'start "" "{CHROME_WIN}" "file:///{html_win_path}"\r\n'
        )
        bat_path = DESKTOP_WIN / f"Combo{idx}.bat"
        bat_path.write_text(bat_content, encoding="utf-8")

        logger.info(f"  📄 Combo{idx}.bat — {legs_str}")

    logger.info(f"✅ {len(valid)} archivos .bat en escritorio")

    # D-BAT-01: Notificar a Windows Explorer que el Desktop cambió.
    # WSL2 escribe por /mnt/c/ — no genera notificaciones shell nativas →
    # Explorer no refresca el Desktop automáticamente.
    # SHChangeNotify(SHCNE_UPDATEDIR, SHCNF_PATH, desktop_path) fuerza el refresh.
    try:
        import subprocess
        ps_cmd = (
            r"$code='using System;using System.Runtime.InteropServices;"
            r"public class SN{[DllImport(""shell32.dll"")]"
            r"public static extern void SHChangeNotify(int e,uint f,IntPtr a,IntPtr b);}'"
            r";Add-Type -TypeDefinition $code;"
            r"[SN]::SHChangeNotify(0x00002000,0x0005,[System.Runtime.InteropServices.Marshal]::StringToHGlobalUni('C:\users\hogar\Desktop'),[IntPtr]::Zero)"
        )
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-c", ps_cmd],
            capture_output=True, timeout=8,
        )
        logger.info("🔔 Windows Explorer notificado — Desktop refrescado")
    except Exception:
        pass  # No bloquear si PowerShell no está disponible

    return len(valid)


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE F12 — Comandos para pegar en la consola de Betplay
# ══════════════════════════════════════════════════════════════════════════════

def mostrar_consola(combo_links: List[Dict]):
    """Imprime comandos location.hash para la consola de Betplay (F12)."""
    valid = [c for c in combo_links if c["url"]]

    print("\n" + "=" * 65)
    print("  COMANDOS PARA CONSOLA DE BETPLAY (F12 > Console)")
    print("  1. Escribe: allow pasting (Enter)")
    print("  2. Pega cada comando → Enter → verifica → apuesta")
    print("  3. Borra ticket (X) antes de cada combo")
    print("=" * 65 + "\n")

    for c in valid:
        ids_str = ",".join(c["outcome_ids"])
        legs_str = " + ".join(f"{l['jugador']}@{l['cuota']:.2f}" for l in c["legs"])
        p = c["piernas"]
        print(f"// Combo {c['combo_idx']} [{p}p] @{c['cuota_combo']:.1f} | ${c['stake']:,} | {legs_str}")
        print(f"location.hash='home?coupon=combination|{ids_str}'")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM — Enviar resumen de combos
# ══════════════════════════════════════════════════════════════════════════════

def _enviar_telegram(msg: str) -> bool:
    """Envía mensaje a Telegram."""
    try:
        params = urllib.parse.urlencode({
            "chat_id": TG_CHAT,
            "text": msg,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        }).encode("utf-8")
        req = urllib.request.Request(TG_URL, data=params, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except Exception as e:
        logger.error(f"❌ Error Telegram: {e}")
        return False


def enviar_combos_telegram(combo_links: List[Dict], metadata: Dict) -> bool:
    """
    Envía combos a Telegram link por link con redirect URLs.

    Cada combo va como mensaje individual con link clickeable.
    El link usa la redirect page de GitHub Pages para preservar el #hash
    en el WebView móvil de Telegram.
    """
    bankroll = metadata.get("bankroll", 0)
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    valid = [c for c in combo_links if c["url"]]

    if not valid:
        return _enviar_telegram("❌ No hay combos válidos hoy")

    # Header
    header = (
        f"🎾 <b>COMBOS BETPLAY</b>\n"
        f"📅 {fecha} | 💰 ${bankroll:,.0f}\n"
        f"📊 {len(valid)} combos listos\n\n"
        f"👇 Tap cada link → Betplay carga el combo"
    )
    ok = _enviar_telegram(header)

    # Cada combo como mensaje individual con redirect link
    for c in valid:
        ids_str = ",".join(c["outcome_ids"])
        redirect_url = f"{REDIRECT_BASE}{ids_str}"
        leg_str = " + ".join(f"{l['jugador']}@{l['cuota']:.2f}" for l in c["legs"])

        msg = (
            f"<b>Combo {c['combo_idx']}</b> [{c['piernas']}p] @{c['cuota_combo']:.1f}\n"
            f"{leg_str}\n"
            f"Stake: ${c['stake']:,.0f} → ${c['retorno']:,.0f}\n\n"
            f"👉 <a href=\"{redirect_url}\">ABRIR COMBO {c['combo_idx']}</a>"
        )
        ok = _enviar_telegram(msg) and ok

    invalid = len(combo_links) - len(valid)
    if invalid:
        _enviar_telegram(f"⚠️ {invalid} combos sin outcome (partidos iniciados/cerrados)")

    return ok


# ══════════════════════════════════════════════════════════════════════════════
# WHATSAPP — Botones wa.me link por link (patrón VoltSafe RF-06)
# ══════════════════════════════════════════════════════════════════════════════

def generar_whatsapp_html(combo_links: List[Dict], metadata: Dict) -> str:
    """
    Genera HTML con un botón por combo que abre wa.me con el link de Betplay.

    Flujo: botón → WhatsApp Web → tap Enviar → link en celular → tap → Betplay.
    Cada combo es un mensaje independiente — si uno falla, los demás siguen.

    Returns:
        Path del HTML generado.
    """
    valid = [c for c in combo_links if c["url"]]
    if not valid:
        logger.error("❌ No hay combos válidos para WhatsApp")
        return ""

    bankroll = metadata.get("bankroll", 0)
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")

    buttons_html = []
    for c in valid:
        idx = c["combo_idx"]
        piernas = c["piernas"]
        cuota = c["cuota_combo"]
        stake = c["stake"]
        retorno = c["retorno"]
        legs_str = " + ".join(f"{l['jugador']}@{l['cuota']:.2f}" for l in c["legs"])

        # Mensaje WhatsApp con el link de Betplay
        msg_lines = [
            f"🎾 *Combo {idx}* [{piernas}p] @{cuota:.1f}",
            f"{legs_str}",
            f"Stake: ${stake:,.0f} → ${retorno:,.0f}",
            "",
            f"{c['url']}",
        ]
        wa_text = urllib.parse.quote("\n".join(msg_lines))
        wa_url = f"https://wa.me/{WA_PHONE}?text={wa_text}"

        buttons_html.append(f"""
    <button class="combo-btn" onclick="window.open('{wa_url}')">
      <b>Combo {idx}</b> — {piernas} piernas @{cuota:.1f}<br>
      <span class="legs">{legs_str}</span><br>
      <span class="stats">Stake: ${stake:,.0f} → ${retorno:,.0f}</span>
    </button>""")

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>WhatsApp Combos Betplay</title>
<style>
  body {{ font-family: Arial; background: #1a1a2e; color: #eee; padding: 20px; max-width: 700px; margin: 0 auto; }}
  h1 {{ color: #25D366; text-align: center; }}
  .info {{ background: #16213e; padding: 12px; border-radius: 8px; border-left: 4px solid #25D366; margin: 15px 0; font-size: 14px; }}
  .combo-btn {{
    display: block; width: 100%; padding: 15px; margin: 8px 0;
    background: #16213e; border: 2px solid #0f3460; border-radius: 10px;
    color: #eee; font-size: 15px; cursor: pointer; text-align: left;
  }}
  .combo-btn:hover {{ background: #0f3460; border-color: #25D366; }}
  .combo-btn b {{ color: #25D366; }}
  .combo-btn .stats {{ color: #ffd700; font-size: 13px; }}
  .combo-btn .legs {{ color: #aaa; font-size: 13px; }}
  .steps {{ background: #0d0d1a; padding: 15px; border-radius: 8px; margin: 15px 0; }}
  .steps li {{ margin: 8px 0; }}
  .steps b {{ color: #25D366; }}
</style>
</head>
<body>

<h1>WhatsApp Combos Betplay</h1>
<p style="text-align:center; color:#888;">{fecha} | Bankroll: ${bankroll:,.0f} | {len(valid)} combos</p>

<div class="info">
  <b>Instrucciones:</b>
  <ol class="steps">
    <li>Clic en cada boton → se abre <b>WhatsApp Web</b> con el mensaje listo</li>
    <li>Tap <b>Enviar</b> en WhatsApp Web</li>
    <li>Desde el celular: tap el <b>link de Betplay</b> → Chrome → combo carga</li>
    <li>Pon stake → apuesta → siguiente link</li>
  </ol>
</div>

{"".join(buttons_html)}

</body>
</html>"""

    out_path = DESKTOP_WIN / "whatsapp_combos.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info(f"✅ WhatsApp HTML generado: {out_path}")

    # También generar .bat para abrir el HTML en Chrome
    bat_content = f'@echo off\r\nstart "" "{CHROME_WIN}" "file:///C:\\users\\hogar\\Desktop\\whatsapp_combos.html"\r\n'
    bat_path = DESKTOP_WIN / "WhatsApp_Combos.bat"
    bat_path.write_text(bat_content, encoding="utf-8")
    logger.info(f"✅ WhatsApp_Combos.bat en escritorio")

    return str(out_path)


# ══════════════════════════════════════════════════════════════════════════════
# SMART COMBO SCORING — Portfolio Theory × Markov × Information Theory
# ══════════════════════════════════════════════════════════════════════════════

_STRATEGY_PARAMS = {
    #                mix_bonus  same_penalty  hot_factor  cold_factor
    "balanced":     (1.20,      0.80,         1.05,       0.90),
    "aggressive":   (1.10,      0.90,         1.08,       0.95),
    "conservative": (1.30,      0.75,         1.03,       0.85),
}

_SAFE_ZONES  = {"heavy_favorite", "moderate_favorite"}
_RISKY_ZONES = {"slight_underdog", "underdog"}


def _score_combo(combo_picks: list, strategy: str = "balanced") -> dict:
    """
    Score a combo using Portfolio Theory + Markov Regime + Information Theory.

    combo_score = combo_EV × diversity_bonus × regime_bonus × alpha_bonus

    combo_EV       = Π(p_modelo_i) × cuota_combo_kambi
    diversity_bonus = mix_bonus si mezcla safe+risky | same_penalty si todo igual zona
    regime_bonus    = Π(hot_factor|cold_factor) por pierna
    alpha_bonus     = 1.0 + min(avg_alpha_vs_elo × 0.5, 0.10)  (cap +10%)
    """
    params = _STRATEGY_PARAMS.get(strategy, _STRATEGY_PARAMS["balanced"])
    mix_bonus, same_penalty, hot_factor, cold_factor = params

    # combo_EV: producto de p_modelo × cuota_combo
    combo_hr = 1.0
    for cp in combo_picks:
        combo_hr *= cp.get("p_modelo", 0.5)
    cuota_combo = 1.0
    for cp in combo_picks:
        cuota_combo *= cp.get("cuota_kambi", cp.get("cuota", 1.5))
    combo_ev = combo_hr * cuota_combo

    # diversity_bonus: Markowitz — mezcla de zonas reduce correlación
    zones = {cp.get("zona_cuota", "slight_underdog") for cp in combo_picks}
    has_safe  = bool(zones & _SAFE_ZONES)
    has_risky = bool(zones & _RISKY_ZONES)
    if has_safe and has_risky:
        diversity_bonus = mix_bonus     # ancla + satélite
    elif len(zones) == 1:
        diversity_bonus = same_penalty  # homogéneo = riesgo correlacionado
    else:
        diversity_bonus = 1.0

    # regime_bonus: Markov momentum compuesto
    regime_bonus = 1.0
    for cp in combo_picks:
        estado = cp.get("markov_favorito")
        if estado == "HOT":
            regime_bonus *= hot_factor
        elif estado == "COLD":
            regime_bonus *= cold_factor

    # alpha_bonus: model alpha beyond ELO (information edge)
    avg_alpha = sum(cp.get("alpha_vs_elo", 0.0) for cp in combo_picks) / len(combo_picks)
    alpha_bonus = 1.0 + min(max(avg_alpha, 0) * 0.5, 0.10)

    combo_score = combo_ev * diversity_bonus * regime_bonus * alpha_bonus

    zones_str  = "+".join(cp.get("zona_cuota", "?")         for cp in combo_picks)
    markov_str = "+".join(cp.get("markov_favorito", "?") or "?" for cp in combo_picks)
    breakdown  = (
        f"EV={combo_ev:.3f} div={diversity_bonus:.2f} "
        f"reg={regime_bonus:.2f} α={alpha_bonus:.2f} "
        f"[{zones_str}] [{markov_str}]"
    )

    return {
        "combo_score":    round(combo_score, 4),
        "combo_ev":       round(combo_ev, 4),
        "combo_hr":       round(combo_hr, 4),
        "diversity_bonus": round(diversity_bonus, 2),
        "regime_bonus":   round(regime_bonus, 2),
        "alpha_bonus":    round(alpha_bonus, 2),
        "breakdown":      breakdown,
    }


def _select_with_cobertura(tier_combos: list, available_picks: list,
                            top_n: int, piernas: int) -> list:
    """
    Selecciona combos con distribución equitativa: ningún jugador aparece
    más de max_app veces. Cuando hay que relajar, excluye activamente
    al jugador más repetido en vez de tomar el siguiente mejor por score.

    max_app = top_n × piernas ÷ n_pool
    Paso 1: selección estricta (respeta max_app)
    Paso 2: busca el mejor combo que EXCLUYA al jugador más repetido
    Paso 3: fallback sin restricción si no hay alternativa
    """
    if not tier_combos:
        return []

    from collections import defaultdict
    n_pool = max(len(available_picks), 1)
    max_app = max(1, (top_n * piernas) // n_pool)

    selected: list = []
    seen: set = set()
    appearances: dict = defaultdict(int)

    def _key(tc):
        return frozenset(cp["jugador"] for cp in tc["picks"])

    def _player_list(tc):
        return [cp["jugador"] for cp in tc["picks"]]

    # Paso 1 — selección estricta con límite max_app
    for tc in tier_combos:
        if len(selected) >= top_n:
            break
        k = _key(tc)
        if k in seen:
            continue
        players = _player_list(tc)
        if any(appearances[p] >= max_app for p in players):
            continue
        selected.append(tc)
        seen.add(k)
        for p in players:
            appearances[p] += 1

    # Paso 2 — rellena priorizando exclusión del jugador más repetido
    while len(selected) < top_n:
        # jugador con más apariciones en los combos ya seleccionados
        top_player = max(appearances, key=lambda p: appearances[p]) if appearances else None
        added = False
        if top_player:
            for tc in tier_combos:
                k = _key(tc)
                if k in seen:
                    continue
                if top_player not in _player_list(tc):
                    selected.append(tc)
                    seen.add(k)
                    for p in _player_list(tc):
                        appearances[p] += 1
                    added = True
                    break
        # Paso 3 — fallback: no hay combo sin top_player, tomar el siguiente mejor
        if not added:
            for tc in tier_combos:
                k = _key(tc)
                if k not in seen:
                    selected.append(tc)
                    seen.add(k)
                    break
            else:
                break  # no quedan combos

    return selected


# ══════════════════════════════════════════════════════════════════════════════
# NODO-26 — CROSS-SECTIONAL SIGNALS + CIRCUIT BREAKER + LINE MOVEMENT
# ══════════════════════════════════════════════════════════════════════════════

MAX_SESSION_LOSS_PCT = 0.04  # 4% bankroll = $5,000 con $125k


def session_budget(bankroll: float, max_loss_pct: float = MAX_SESSION_LOSS_PCT) -> float:
    """Presupuesto máximo de inversión por sesión (M-26-2)."""
    return bankroll * max_loss_pct


def check_budget(n_combos: int, stake: int, bankroll: float) -> tuple:
    """
    Pre-sesión: ¿los combos planificados exceden el budget? (M-26-2)
    Recorta a los top N por score si excede.

    Returns:
        (n_permitidos, mensaje)
    """
    total_risk = n_combos * stake
    budget = session_budget(bankroll)
    if total_risk > budget:
        max_combos = int(budget // stake)
        return max_combos, f"BUDGET LIMIT: {n_combos} combos → recortado a {max_combos} (budget ${budget:,.0f})"
    return n_combos, "OK"


def _find_bankroll_from_plans() -> float:
    """Lee bankroll del trader_plan más reciente (fallback cuando --bankroll=0)."""
    from datetime import timedelta
    reports = Path("reports")
    if not reports.exists():
        return 0.0
    cutoff = datetime.now() - timedelta(hours=24)
    plan_files = sorted(
        [p for p in reports.glob("trader_plan_*.json")
         if p.stat().st_mtime >= cutoff.timestamp() and p.stat().st_size > 100],
        reverse=True,
    )
    for pf in plan_files:
        try:
            plan = json.loads(pf.read_text(encoding="utf-8"))
            br = plan.get("metadata", {}).get("parametros", {}).get("bankroll", 0)
            if br and br > 0:
                return float(br)
        except Exception:
            continue
    return 0.0


def line_movement_signal(cuota_original, cuota_actual) -> tuple:
    """
    Delta de cuota como señal de mercado (M-26-3).
    cuota_original: cuota al momento del edge_calculator (PASO 3)
    cuota_actual:   cuota de Kambi al momento del combo builder (PASO 4.5)

    Returns:
        (factor, signal) where signal is STEAM_IN | DRIFT_OUT | STABLE | NO_DATA
    """
    if cuota_original is None or cuota_original <= 0:
        return 1.00, "NO_DATA"
    delta_pct = (cuota_actual - cuota_original) / cuota_original
    if delta_pct < -0.04:
        return 1.10, "STEAM_IN"
    elif delta_pct > 0.04:
        return 0.85, "DRIFT_OUT"
    else:
        return 1.00, "STABLE"


def ranking_preserved_blend(picks_pool: list, p_historica: float, js_factor: float,
                             amplification: float = 5.0) -> list:
    """
    M-26-1: Preserva el ranking relativo de p_modelo cuando Dispersion Guard = BLIND.

    Solo se activa cuando std(p_blend) < 0.015. En DIFFERENTIATED, James-Stein normal funciona.
    No cambia edge/Kelly — solo modifica p_blend para scoring de combos/megas.

    Returns:
        picks_pool ordenado por p_blend desc (con desempate por cuota asc)
    """
    import numpy as np
    p_modelos = [p.get("p_modelo", 0.5) for p in picks_pool]
    p_mean = float(np.mean(p_modelos))
    for p in picks_pool:
        delta = p.get("p_modelo", 0.5) - p_mean
        p["p_blend"] = float(np.clip(
            p_historica + js_factor * amplification * delta,
            0.40, 0.75,
        ))
    # Sort: p_blend desc, tiebreak cuota asc (menor cuota = favorecido por bookmaker)
    picks_pool.sort(key=lambda x: (-x["p_blend"], x.get("cuota", 99)))
    return picks_pool


def cv_edge_guard(picks_pool: list) -> tuple:
    """
    M-26-5: Coeficiente de Variación de los edges de la sesión.
    CV bajo = el modelo da edge similar a todos los picks = ciego a nivel de sesión.

    Complementa Dispersion Guard (que mide p_blend). Solo evalúa picks con edge > 0.

    Returns:
        (cv_value, status) where status is BLIND_EDGE | LOW_VARIANCE_EDGE | DIVERSE_EDGE | INSUFFICIENT
    """
    import numpy as np
    edges = [p.get("edge", 0) for p in picks_pool if p.get("edge", 0) > 0]
    if len(edges) < 3:
        return None, "INSUFFICIENT"
    mean_e = float(np.mean(edges))
    cv = float(np.std(edges) / mean_e) if mean_e > 0 else 0.0
    if cv < 0.15:
        return round(cv, 4), "BLIND_EDGE"
    elif cv < 0.30:
        return round(cv, 4), "LOW_VARIANCE_EDGE"
    else:
        return round(cv, 4), "DIVERSE_EDGE"


def session_regime(calibracion: dict, lookback: int = 5) -> tuple:
    """
    M-26-4: Meta-Markov — evalúa régimen del modelo basado en sesiones recientes.

    NUNCA aumenta stakes (HOT_MODEL = 1.00). Solo reduce en drawdown.
    Requiere ≥3 sesiones para activar. Con n<3 → INSUFFICIENT, factor=1.0.

    Returns:
        (regime, factor)
    """
    import numpy as np
    history = calibracion.get("session_history", [])
    if len(history) < 3:
        return "INSUFFICIENT", 1.0
    recent = history[-lookback:]
    accs = [s.get("accuracy", 0.5) for s in recent if s.get("accuracy") is not None]
    if len(accs) < 3:
        return "INSUFFICIENT", 1.0
    avg_acc = float(np.mean(accs))
    trend = accs[-1] - accs[0]
    if avg_acc < 0.50:
        return "COLD_MODEL", 0.50
    elif avg_acc < 0.60 and trend < -0.10:
        return "COOLING", 0.75
    elif avg_acc > 0.70:
        return "HOT_MODEL", 1.00
    else:
        return "NEUTRAL", 1.00


# ══════════════════════════════════════════════════════════════════════════════
# NODO-25 — GUARDS (Dispersion, Tournament Concentration, Discipline, Duplicate)
# Post-mortem 2026-06-14: 25 combos, 0 vivos, -$12,500
# ══════════════════════════════════════════════════════════════════════════════


def dispersion_index(picks_pool: list) -> tuple[float, str]:
    """
    Nodo-25 Guard 1: Mide si el modelo distingue entre picks del pool.

    Returns:
        (std_value, classification) where classification is
        BLIND (<0.015) | LOW_SIGNAL (0.015-0.04) | DIFFERENTIATED (>=0.04)
    """
    import numpy as np
    p_blends = [p.get("p_blend", 0.5) for p in picks_pool if p.get("p_blend")]
    if len(p_blends) < 2:
        return 0.0, "BLIND"
    std = float(np.std(p_blends))
    if std < 0.015:
        return std, "BLIND"
    elif std < 0.04:
        return std, "LOW_SIGNAL"
    else:
        return std, "DIFFERENTIATED"


def tournament_concentration_ok(combo_picks: list, max_same: int = 2) -> bool:
    """
    Nodo-25 Guard 2: Max N picks from the same tournament in any combo.
    Returns True if combo passes the guard.
    """
    from collections import Counter
    torneos = Counter()
    for p in combo_picks:
        t = p.get("torneo", "") or p.get("partido", "")
        if t:
            torneos[t] += 1
    if not torneos:
        return True
    return max(torneos.values()) <= max_same


def discipline_check(pick: dict, trader_names: set) -> bool:
    """
    Nodo-25 Guard 3: Only picks from trader_plan enter combos.
    Returns True if pick is in a trader plan.
    """
    return pick.get("jugador", "") in trader_names


def is_duplicate_combo(new_combo_picks: list, existing_combos: list) -> bool:
    """
    Nodo-25 Guard 4: Detects combos with identical picks (order irrelevant).
    Returns True if this combo is a duplicate.
    """
    new_set = frozenset(p.get("jugador", "") for p in new_combo_picks)
    for existing in existing_combos:
        existing_picks = existing.get("picks", existing.get("legs", []))
        existing_set = frozenset(p.get("jugador", "") for p in existing_picks)
        if new_set == existing_set:
            return True
    return False


def _load_trader_names() -> set:
    """Load all player names from trader_plans of last 24h."""
    from datetime import timedelta
    reports = Path("reports")
    cutoff = datetime.now() - timedelta(hours=24)
    names = set()
    for pf in reports.glob("trader_plan_*.json"):
        try:
            if pf.stat().st_mtime < cutoff.timestamp():
                continue
            plan = json.loads(pf.read_text(encoding="utf-8"))
            for p in plan.get("individuales", []):
                name = p.get("favorito", "")
                if name:
                    names.add(name)
            for combo in plan.get("cobertura", []):
                for leg in combo.get("legs", []):
                    name = leg.get("jugador", "")
                    if name:
                        names.add(name)
        except Exception:
            continue
    return names


# ══════════════════════════════════════════════════════════════════════════════
# NODO-25 — SAFE COMBOS (Beta Book)
# 2-leg combos with highest P(both win), cuota 3.0-12.0, cross-tournament
# ══════════════════════════════════════════════════════════════════════════════


def build_safe_combos(stake_per_combo: int = 1000,
                      top_n: int = 8,
                      min_p_both: float = 0.25,
                      max_cuota: float = 12.0) -> tuple[List[Dict], Dict]:
    """
    Nodo-25: Safe Combos — Beta Book.

    Generates 2-leg combos prioritizing P(both win).
    Pool: APOSTAR + WATCHLIST picks from all trader_plans (last 24h).
    Scoring: P(both) + 0.01 × log(cuota) — P dominates, cuota tiebreaker.
    Guard 2 (tournament concentration) enforced: different tournaments required.
    """
    import math

    # 1. Load all picks from trader_plans
    reports = Path("reports")
    from datetime import timedelta
    cutoff = datetime.now() - timedelta(hours=24)
    all_plans = sorted(reports.glob("trader_plan_*.json"), reverse=True)
    plan_files = [
        p for p in all_plans
        if p.stat().st_mtime >= cutoff.timestamp() and p.stat().st_size > 100
    ]

    if not plan_files:
        logger.warning("⚠️ No hay trader_plans en las últimas 24h para safe combos")
        return [], {}

    # 1b. Load edge_report for enrichment
    edge_pick_map = {}
    edge_path = _find_latest_edge_report()
    if edge_path:
        try:
            with open(edge_path, encoding="utf-8") as f:
                edge_data = json.load(f)
            _validate_edge_report_gate(edge_data, edge_path)  # Nodo-32 Acción 3
            for cat in ("apostar", "watchlist"):
                for p in edge_data.get(cat, []):
                    name = p.get("favorito_predicho", "")
                    if name:
                        edge_pick_map[name] = {
                            "tier": p.get("tier", "unknown"),
                            "torneo": p.get("torneo", ""),
                            "gap": abs(p.get("p_blend", 0.5) - p.get("p_modelo", 0.5)),
                            "n_h2h": p.get("n_h2h", 0),
                        }
        except Exception:
            pass

    # 2. Build pool from trader individuales
    pool = []
    seen_names = set()

    for pf in plan_files:
        try:
            plan = json.loads(pf.read_text(encoding="utf-8"))
        except Exception:
            continue

        meta = plan.get("metadata", {})
        params = meta.get("parametros", {})
        plan_tier = params.get("torneo_tipo", "unknown")
        plan_sup = params.get("superficie", "unknown")

        for p in plan.get("individuales", []):
            name = p.get("favorito", "")
            if name and name not in seen_names:
                seen_names.add(name)
                edge_info = edge_pick_map.get(name, {})
                # Build torneo identifier: tier + superficie as fallback
                torneo = edge_info.get("torneo", "")
                if not torneo:
                    torneo = f"{plan_tier}_{plan_sup}"

                pool.append({
                    "jugador":    name,
                    "cuota":      p.get("cuota", 0),
                    "p_blend":    p.get("p_blend", 0.5),
                    "p_modelo":   p.get("p_modelo", 0.5),
                    "tier":       plan_tier if plan_tier != "unknown" else edge_info.get("tier", "unknown"),
                    "torneo":     torneo,
                    "gap":        edge_info.get("gap", abs(p.get("p_blend", 0.5) - p.get("p_modelo", 0.5))),
                    "n_h2h":      edge_info.get("n_h2h", 0),
                    "edge_pct":   p.get("edge_pct", "0%"),
                    "superficie": plan_sup,
                })

    if len(pool) < 2:
        logger.warning(f"⚠️ Pool insuficiente para safe combos: {len(pool)} picks (mínimo 2)")
        return [], {}

    logger.info(f"🛡️ Safe combo pool: {len(pool)} picks de {len(plan_files)} planes")

    # 3. Verify availability in Kambi
    outcomes_map, started_map = fetch_kambi_outcomes()
    if not outcomes_map:
        logger.error("❌ No se pudieron obtener outcomes de Kambi")
        return [], {}

    available_pool = []
    for pick in pool:
        oc, reason = find_outcome(pick["jugador"], pick["cuota"], outcomes_map, started_map)
        if oc:
            pick["outcome_id"] = str(oc["outcome_id"])
            pick["cuota_kambi"] = oc["odds"]
            available_pool.append(pick)
        else:
            logger.info(f"  ⏭️ {pick['jugador']} @{pick['cuota']} — {reason} (excluido de safe)")

    if len(available_pool) < 2:
        logger.warning(f"⚠️ Solo {len(available_pool)} picks disponibles en Kambi (mínimo 2)")
        return [], {}

    logger.info(f"  ✅ {len(available_pool)} picks disponibles para safe combos")

    # 4. Generate all 2-pick pairs, score them
    safe_candidates = []

    for i, p1 in enumerate(available_pool):
        for p2 in available_pool[i + 1:]:
            # Guard 2: different tournaments
            if p1["torneo"] and p2["torneo"] and p1["torneo"] == p2["torneo"]:
                continue

            p_both = p1["p_blend"] * p2["p_blend"]
            cuota_combo = p1["cuota_kambi"] * p2["cuota_kambi"]

            if p_both < min_p_both:
                continue
            if cuota_combo > max_cuota:
                continue

            # Scoring: P(both) dominates, cuota is tiebreaker
            gap_max = max(p1.get("gap", 0), p2.get("gap", 0))
            gap_penalty = min(1.0, 1.15 - gap_max) if gap_max > 0.03 else 1.0
            score = (p_both + 0.01 * math.log(max(cuota_combo, 1.01))) * gap_penalty

            safe_candidates.append({
                "picks":       [p1, p2],
                "p_both":      round(p_both, 4),
                "cuota_combo": round(cuota_combo, 2),
                "score":       round(score, 6),
                "gap_penalty": round(gap_penalty, 3),
                "gap_max":     round(gap_max, 4),
            })

    # Sort by score descending → top N
    safe_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Deduplicate
    safe_combos = []
    for cand in safe_candidates:
        if len(safe_combos) >= top_n:
            break
        if not is_duplicate_combo(cand["picks"], safe_combos):
            # Build URL
            outcome_ids = [p["outcome_id"] for p in cand["picks"]]
            ids_str = ",".join(outcome_ids)
            betplay_url = f"{BETPLAY_URL_BASE}{ids_str}{BETPLAY_URL_TAIL}"

            safe_combos.append({
                "combo_idx":   len(safe_combos) + 1,
                "piernas":     2,
                "picks":       cand["picks"],
                "legs":        [{
                    "jugador":    p["jugador"],
                    "cuota":      p["cuota"],
                    "cuota_kambi": p["cuota_kambi"],
                    "outcome_id": p["outcome_id"],
                    "tier":       p.get("tier", "?"),
                    "torneo":     p.get("torneo", "?"),
                    "gap":        p.get("gap", 0),
                } for p in cand["picks"]],
                "outcome_ids": outcome_ids,
                "url":         betplay_url,
                "partial":     False,
                "stake":       stake_per_combo,
                "cuota_combo": cand["cuota_combo"],
                "retorno":     round(stake_per_combo * cand["cuota_combo"], 0),
                "p_both":      cand["p_both"],
                "score":       cand["score"],
                "gap_penalty": cand["gap_penalty"],
            })

    metadata = {
        "modo":            "SAFE",
        "pool_total":      len(pool),
        "pool_disponible": len(available_pool),
        "n_candidates":    len(safe_candidates),
        "n_safe_combos":   len(safe_combos),
        "stake_per_combo": stake_per_combo,
        "total_stake":     stake_per_combo * len(safe_combos),
        "min_p_both":      min_p_both,
        "max_cuota":       max_cuota,
    }

    logger.info(f"  🛡️ {len(safe_combos)} safe combos generados de {len(safe_candidates)} candidatos")
    return safe_combos, metadata


def _mostrar_safe_combos(safe_links: List[Dict], metadata: Dict):
    """Display safe combos in console."""
    total_stake = metadata.get("total_stake", 0)
    stake_each = metadata.get("stake_per_combo", 1000)

    print()
    print("=" * 70)
    print(f"  🛡️ SAFE COMBOS — Beta Book (2 piernas, P>{metadata.get('min_p_both', 0.25)*100:.0f}%)")
    print(f"  💰 {len(safe_links)} combos × ${stake_each:,} = ${total_stake:,}")
    print(f"  📊 Pool: {metadata.get('pool_disponible', '?')} picks | "
          f"{metadata.get('n_candidates', '?')} pares evaluados")
    print("=" * 70)

    for sc in safe_links:
        cuota = sc["cuota_combo"]
        retorno = sc["retorno"]
        p_both = sc.get("p_both", 0)
        gap_pen = sc.get("gap_penalty", 1.0)

        gap_flag = "" if gap_pen >= 0.99 else f" | gap_pen={gap_pen:.2f}"
        print(f"\n  🛡️ Safe {sc['combo_idx']} — @{cuota:.2f} → ${retorno:,.0f} | P(ambos)={p_both:.1%}{gap_flag}")
        for leg in sc["legs"]:
            gap = leg.get("gap", 0)
            gap_label = "CAL" if gap > 0.12 else "MKT" if gap < 0.08 else "MIX"
            print(f"     ✅ {leg['jugador']:<25} @{leg['cuota_kambi']:.2f}  [{leg.get('tier','?')}] gap={gap:.3f} {gap_label}")

    if safe_links:
        best = max(safe_links, key=lambda x: x["p_both"])
        print(f"\n  🎯 MÁS PROBABLE: Safe {best['combo_idx']} — P(ambos)={best['p_both']:.1%} @{best['cuota_combo']:.2f}")
    print(f"  💰 INVERSIÓN TOTAL: ${total_stake:,}")
    print("=" * 70)


def _generar_bat_safe(safe_links: List[Dict]) -> int:
    """Generate Safe1.bat ... SafeN.bat on Windows desktop."""
    COMBOS_DIR.mkdir(exist_ok=True)

    # Clean previous safe combos
    for old_bat in DESKTOP_WIN.glob("Safe*.bat"):
        old_bat.unlink(missing_ok=True)
    for old_html in COMBOS_DIR.glob("safe*.html"):
        old_html.unlink(missing_ok=True)

    count = 0
    for sc in safe_links:
        url = sc.get("url")
        if not url:
            continue

        idx = sc["combo_idx"]
        cuota = sc["cuota_combo"]
        p_both = sc.get("p_both", 0)

        legs_desc = " + ".join(
            f"{l['jugador']}@{l['cuota_kambi']:.2f}" for l in sc["legs"]
        )
        html_content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Safe {idx}</title></head>
<body style="font-family:monospace;text-align:center;padding:40px">
<h2>🛡️ SAFE {idx} [2p] @{cuota:.2f} — P={p_both:.1%}</h2>
<p>{legs_desc}</p>
<p><a href="{url}" target="_blank" style="font-size:24px;padding:20px;background:#28a745;color:white;text-decoration:none;border-radius:8px">
Abrir en Betplay</a></p>
</body></html>"""

        html_path = COMBOS_DIR / f"safe{idx}.html"
        html_path.write_text(html_content, encoding="utf-8")

        bat_content = (
            f'@echo off\r\n'
            f'start "" "{CHROME_WIN}" '
            f'"file:///C:\\users\\hogar\\Desktop\\combos\\safe{idx}.html"\r\n'
        )
        bat_path = DESKTOP_WIN / f"Safe{idx}.bat"
        bat_path.write_text(bat_content, encoding="utf-8")
        count += 1
        logger.info(f"  📄 Safe{idx}.bat — [2p @{cuota:.2f} P={p_both:.1%}] {legs_desc[:80]}")

    return count


def _enviar_safe_telegram(safe_links: List[Dict], metadata: Dict):
    """Send safe combos summary to Telegram."""
    lines = ["🛡️ *SAFE COMBOS — Beta Book*\n"]
    for sc in safe_links:
        cuota = sc["cuota_combo"]
        retorno = sc["retorno"]
        p_both = sc.get("p_both", 0)
        legs_str = " × ".join(f"{l['jugador']}@{l['cuota_kambi']:.2f}" for l in sc["legs"])
        lines.append(f"*Safe {sc['combo_idx']}* @{cuota:.2f} P={p_both:.0%} → ${retorno:,.0f}")
        lines.append(f"  {legs_str}\n")

    lines.append(f"💰 Total: ${metadata.get('total_stake', 0):,}")
    text = "\n".join(lines)

    try:
        data = json.dumps({"chat_id": TG_CHAT, "text": text, "parse_mode": "Markdown"}).encode()
        req = urllib.request.Request(TG_URL, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        logger.info("📱 Safe combos enviados a Telegram ✅")
    except Exception as e:
        logger.warning(f"⚠️ Error enviando safe a Telegram: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# WAS — Nodo-44: Watchlist Alpha Signal (promo combos, alpha invisible)
# ══════════════════════════════════════════════════════════════════════════════

def _was_qualifies(pick: dict, min_edge: float = 10.0) -> bool:
    """T55-04/05 (Nodo-55 P54-03): gate compuesto WAS extraído para testeo.
    Requiere edge>=min_edge, cuota>=2.0, Y señal Markov explícita.
    Sin señal Markov → False (coin-flip puro, p≈0.51 no es WAS)."""
    edge_raw = pick.get('edge_pct', '0%')
    try:
        edge_val = float(str(edge_raw).replace('%', '').strip())
    except ValueError:
        edge_val = 0.0
    if edge_val < min_edge:
        return False
    if pick.get('cuota_favorito', 0) < 2.0:
        return False
    estado_fav   = pick.get('markov_favorito')
    estado_rival = pick.get('markov_rival')
    conf_fav     = pick.get('markov_conf_fav', 0) or pick.get('conf_fav', 0) or 0
    conf_rival   = pick.get('markov_conf_rival', 0) or pick.get('conf_rival', 0) or 0
    wr_rec_fav   = pick.get('markov_wr_rec_fav') or 0.5
    wr_rec_rival = pick.get('markov_wr_rec_rival') or 0.5
    diff_abs     = abs(wr_rec_fav - wr_rec_rival)
    rival_cold   = estado_rival == 'COLD' and conf_rival >= 0.60
    pick_hot     = estado_fav == 'HOT' and conf_fav >= 0.60
    es_dominante = diff_abs > 0.35
    señal_markov = rival_cold or pick_hot or es_dominante
    return señal_markov


def build_was_combos(stake_per_combo: int = 5000,
                     min_edge: float = 10.0,
                     top_n: int = 5,
                     edge_file: Optional[str] = None) -> tuple[List[Dict], Dict]:
    """
    Nodo-44: Watchlist Alpha Signal (WAS) combos.

    Lee watchlist del edge_report, filtra picks con "alpha invisible":
    - edge >= min_edge (default 10%)
    - cuota_favorito >= 2.0
    - AL MENOS UNA señal Markov:
        rival COLD conf>=0.60 (PCRS)
        OR pick HOT conf>=0.60
        OR zona DOMINANTE (|wr_rec_fav - wr_rec_rival| > 0.35)
        OR COINFLIP (|diff| <= 0.18) + rival COLD conf>=0.60

    Genera combos 2 piernas con cuota_combo >= 4.0x.
    REGLA-WAS-1: solo para promos stake mínimo hasta n>=30.
    Requiere D44-02: edge_report con campos markov_conf_fav/rival + markov_wr_rec_fav/rival.
    """
    import math

    # Load edge_report
    if edge_file:
        edge_path = edge_file
    else:
        edge_path = _find_latest_edge_report()

    if not edge_path or not Path(edge_path).exists():
        logger.error("❌ No se encontró edge_report. Ejecuta: python edge_calculator.py")
        return [], {}

    edge_data = json.loads(Path(edge_path).read_text(encoding="utf-8"))
    _validate_edge_report_gate(edge_data, edge_path)

    watchlist = edge_data.get("watchlist", [])
    if not watchlist:
        logger.warning("⚠️ WAS: edge_report sin picks en watchlist")
        return [], {}

    # Filter WAS candidates (usa _was_qualifies para testabilidad — T55-04/05)
    was_candidates = []
    for pick in watchlist:
        edge_raw = pick.get("edge_pct", "0%")
        try:
            edge_val = float(str(edge_raw).replace("%", "").strip())
        except ValueError:
            edge_val = 0.0
        cuota = pick.get("cuota_favorito", 0)

        if not _was_qualifies(pick, min_edge=min_edge):
            continue

        # Recalcular variables locales para señal_parts (solo picks que pasaron el gate)
        estado_fav   = pick.get("markov_favorito")
        estado_rival = pick.get("markov_rival")
        conf_fav     = pick.get("markov_conf_fav", 0) or 0
        conf_rival   = pick.get("markov_conf_rival", 0) or 0
        wr_rec_fav   = pick.get("markov_wr_rec_fav") or 0.5
        wr_rec_rival = pick.get("markov_wr_rec_rival") or 0.5
        diff_abs_markov = abs(wr_rec_fav - wr_rec_rival)
        es_dominante = diff_abs_markov > 0.35
        es_coinflip  = diff_abs_markov <= 0.18
        rival_cold   = estado_rival == "COLD" and conf_rival >= 0.60
        pick_hot     = estado_fav == "HOT" and conf_fav >= 0.60

        señal_parts = []
        if rival_cold:
            señal_parts.append(f"RIVAL_COLD c={conf_rival:.2f}")
        if pick_hot:
            señal_parts.append(f"HOT c={conf_fav:.2f}")
        if es_dominante:
            señal_parts.append(f"DOMINANTE d={diff_abs_markov:.2f}")
        if es_coinflip and rival_cold:
            señal_parts.append("COINFLIP+COLD")

        was_candidates.append({
            "jugador":         pick.get("favorito_predicho", ""),
            "cuota":           cuota,
            "edge_pct":        edge_raw,
            "edge_val":        edge_val,
            "p_modelo":        pick.get("p_modelo", 0.5),
            "p_blend":         pick.get("p_blend", 0.5),
            "n_h2h":           pick.get("n_h2h", 0),
            "tier":            pick.get("tier", "unknown"),
            "torneo":          pick.get("torneo", ""),
            "superficie":      pick.get("superficie", "unknown"),
            "partido":         pick.get("partido", ""),
            "match_id":        pick.get("match_id", ""),
            "markov_favorito": estado_fav,
            "markov_rival":    estado_rival,
            "conf_fav":        conf_fav,
            "conf_rival":      conf_rival,
            "diff_abs_markov": round(diff_abs_markov, 3),
            "señal_was":       " | ".join(señal_parts),
        })

    if not was_candidates:
        logger.warning(f"⚠️ WAS: 0 picks pasan filtros (edge>={min_edge}% + cuota>=2.0 + señal Markov)")
        return [], {}

    logger.info(f"  WAS candidates: {len(was_candidates)} picks con alpha invisible")
    for p in was_candidates:
        logger.info(f"    {p['jugador']:<25} @{p['cuota']:.2f}  edge={p['edge_pct']}  [{p['señal_was']}]")

    # Verify availability in Kambi
    outcomes_map, started_map = fetch_kambi_outcomes()
    if not outcomes_map:
        logger.error("❌ No se pudieron obtener outcomes de Kambi")
        return [], {}

    available_picks = []
    for pick in was_candidates:
        oc, reason = find_outcome(pick["jugador"], pick["cuota"], outcomes_map, started_map)
        if oc:
            pick["outcome_id"] = str(oc["outcome_id"])
            pick["cuota_kambi"] = oc["odds"]
            available_picks.append(pick)
            logger.info(f"  OK {pick['jugador']:25s} @{pick['cuota']:.2f} -> @{oc['odds']:.2f}")
        else:
            logger.info(f"  NO {pick['jugador']:25s} @{pick['cuota']:.2f} — {reason}")

    if not available_picks:
        logger.warning("⚠️ WAS: 0 picks disponibles en Kambi")
        return [], {}

    # Sort by edge descending
    available_picks.sort(key=lambda x: x["edge_val"], reverse=True)

    # Build 2-leg combos (each pick >= 2.0 so combo >= 4.0x guaranteed)
    was_combos = []
    for i, p1 in enumerate(available_picks):
        for p2 in available_picks[i + 1:]:
            cuota_combo = round(p1["cuota_kambi"] * p2["cuota_kambi"], 2)
            if cuota_combo < 4.0:
                continue
            outcome_ids = [p1["outcome_id"], p2["outcome_id"]]
            ids_str = ",".join(outcome_ids)
            betplay_url = f"{BETPLAY_URL_BASE}{ids_str}{BETPLAY_URL_TAIL}"
            score = (p1["edge_val"] * p2["edge_val"]) * math.log(max(cuota_combo, 1.01))
            was_combos.append({
                "picks":       [p1, p2],
                "cuota_combo": cuota_combo,
                "outcome_ids": outcome_ids,
                "url":         betplay_url,
                "score":       round(score, 4),
            })

    was_combos.sort(key=lambda x: x["score"], reverse=True)

    final_combos = []
    for cand in was_combos:
        if len(final_combos) >= top_n:
            break
        if not is_duplicate_combo(cand["picks"], final_combos):
            final_combos.append({
                "combo_idx":   len(final_combos) + 1,
                "n_piernas":   2,
                "picks":       cand["picks"],
                "legs": [{
                    "jugador":     p["jugador"],
                    "cuota":       p["cuota"],
                    "cuota_kambi": p["cuota_kambi"],
                    "outcome_id":  p["outcome_id"],
                    "edge_pct":    p["edge_pct"],
                    "tier":        p.get("tier", "?"),
                    "torneo":      p.get("torneo", "?"),
                    "señal_was":   p.get("señal_was", ""),
                } for p in cand["picks"]],
                "outcome_ids": cand["outcome_ids"],
                "url":         cand["url"],
                "stake":       stake_per_combo,
                "cuota_combo": cand["cuota_combo"],
                "retorno":     round(stake_per_combo * cand["cuota_combo"]),
                "score":       cand["score"],
            })

    metadata = {
        "modo":             "WAS",
        "fuente":           str(edge_path),
        "min_edge":         min_edge,
        "n_watchlist":      len(watchlist),
        "n_was_candidates": len(was_candidates),
        "n_disponible":     len(available_picks),
        "n_combos":         len(final_combos),
        "stake_per_combo":  stake_per_combo,
        "total_stake":      stake_per_combo * len(final_combos),
        "regla_was1":       "ACTIVA — solo promos stake minimo hasta n>=30",
    }

    logger.info(f"  {len(final_combos)} WAS combos generados (min_edge={min_edge}%, cuota>=2.0)")
    return final_combos, metadata


def _mostrar_was_combos(was_links: List[Dict], metadata: Dict):
    """Display WAS combos in console."""
    total_stake = metadata.get("total_stake", 0)
    stake_each  = metadata.get("stake_per_combo", 5000)
    n_cand      = metadata.get("n_was_candidates", 0)

    print()
    print("=" * 70)
    print(f"  WAS COMBOS — Watchlist Alpha Signal (Nodo-44)")
    print(f"  {len(was_links)} combos x ${stake_each:,} = ${total_stake:,}")
    print(f"  Candidatos WAS: {n_cand} | Disponibles: {metadata.get('n_disponible','?')}")
    print(f"  REGLA-WAS-1: {metadata.get('regla_was1','')}")
    print("=" * 70)

    for wc in was_links:
        cuota = wc["cuota_combo"]
        retorno = wc["retorno"]
        print(f"\n  WAS {wc['combo_idx']} [2p] @{cuota:.2f} -> ${retorno:,.0f}")
        for leg in wc["legs"]:
            señal = leg.get("señal_was", "")
            print(f"     {leg['jugador']:<25} @{leg['cuota_kambi']:.2f}  edge={leg['edge_pct']}  [{señal}]")
            print(f"       tier={leg.get('tier','?')}  {leg.get('torneo','')}")

    if was_links:
        best = max(was_links, key=lambda x: x["cuota_combo"])
        print(f"\n  MAYOR CUOTA: WAS {best['combo_idx']} @{best['cuota_combo']:.2f} -> ${best['retorno']:,.0f}")
    print(f"  INVERSION TOTAL: ${total_stake:,}")
    print("=" * 70)


def _generar_bat_was(was_links: List[Dict]) -> int:
    """Generate WAS1.bat ... WASN.bat on Windows desktop."""
    COMBOS_DIR.mkdir(exist_ok=True)

    for old_bat in DESKTOP_WIN.glob("WAS*.bat"):
        old_bat.unlink(missing_ok=True)
    for old_html in COMBOS_DIR.glob("was_*.html"):
        old_html.unlink(missing_ok=True)

    count = 0
    for wc in was_links:
        idx = wc["combo_idx"]
        html_name = f"was_{idx}"
        cuota = wc["cuota_combo"]
        retorno = wc["retorno"]
        stake = wc["stake"]

        legs_desc = " + ".join(
            f"{l['jugador']} @{l['cuota_kambi']:.2f} (edge={l['edge_pct']})"
            for l in wc["legs"]
        )
        html_content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>WAS{idx}</title></head>
<body style="font-family:monospace;text-align:center;padding:40px">
<h2>WAS {idx} [2p] @{cuota:.2f}</h2>
<p style="font-size:18px">${stake:,} → ${retorno:,.0f}</p>
<p style="font-size:13px">{legs_desc}</p>
<p><a href="{wc['url']}" target="_blank" style="font-size:24px;padding:20px;background:#cc6600;color:white;text-decoration:none;border-radius:8px">
Abrir en Betplay</a></p>
</body></html>"""

        html_path = COMBOS_DIR / f"{html_name}.html"
        html_path.write_text(html_content, encoding="utf-8")

        bat_content = (
            f'@echo off\r\n'
            f'start "" "{CHROME_WIN}" '
            f'"file:///C:\\users\\hogar\\Desktop\\combos\\{html_name}.html"\r\n'
        )
        bat_path = DESKTOP_WIN / f"WAS{idx}.bat"
        bat_path.write_text(bat_content, encoding="utf-8")
        count += 1
        logger.info(f"  WAS{idx}.bat — [2p @{cuota:.2f}] {legs_desc[:80]}")

    return count


# ══════════════════════════════════════════════════════════════════════════════
# GAMES SIGNAL COMBOS — Nodo-40: totales (over/under juegos/sets)
# ══════════════════════════════════════════════════════════════════════════════

def _find_latest_games_signal() -> Optional[Path]:
    """Encuentra el games_signal_report más reciente en reports/."""
    reports = Path("reports")
    files = sorted(reports.glob("games_signal_report_*.json"), reverse=True)
    if files:
        return files[0]
    return None


def build_games_combos(stake_per_combo: int = 2000,
                       games_file: Optional[str] = None) -> tuple[List[Dict], Dict]:
    """
    Nodo-40 Fase 2: Lee games_signal_report y genera combos de totales.

    Arquitectura de combos:
      GamesA: señales ALTA (gap>=3, cuota>=1.80) — máx 3 piernas
      GamesB: señales ALTA + MEDIA (gap>=2, cuota>=1.50) — máx 3 piernas
      GamesC: GamesB con todas las señales disponibles — máx 3 piernas

    REGLA-G4: max 1 señal por partido en cada combo.
    REGLA-G5: máx 3 piernas por combo.
    REGLA-G6: stake máx $2,000 hasta n>=50 calibradas.
    """
    # Load games signal report
    if games_file:
        signal_path = Path(games_file)
    else:
        signal_path = _find_latest_games_signal()

    if not signal_path or not signal_path.exists():
        logger.error("❌ No se encontró games_signal_report. Ejecuta: python games_signal_calculator.py")
        return [], {}

    logger.info(f"📄 Leyendo: {signal_path.name}")
    data = json.loads(signal_path.read_text(encoding="utf-8"))

    apostar = data.get("apostar", [])
    metadata_src = data.get("metadata", {})
    calibracion_n = metadata_src.get("calibracion_n", 0)

    if not apostar:
        logger.warning("⚠️ games_signal_report: no hay señales con apostar=True")
        return [], {}

    # Flatten all señales_optimas respecting REGLA-G4 (1 per partido)
    all_signals = []
    for partido_data in apostar:
        partido = partido_data.get("partido", "")
        zona = partido_data.get("zona_diff", "")
        diff_abs = partido_data.get("diff_abs", 0.0)
        señales = partido_data.get("señales_optimas", [])
        # Take best signal per partido (first = optimal from calculator)
        if señales:
            s = señales[0]
            if s.get("apostar"):
                all_signals.append({
                    "partido": partido,
                    "zona_diff": zona,
                    "diff_abs": diff_abs,
                    "mercado": s.get("mercado", ""),
                    "linea": s.get("linea", 0),
                    "direccion": s.get("direccion", ""),
                    "cuota": s.get("cuota", 1.0),
                    "outcome_id": s.get("outcome_id"),
                    "gap_juegos": s.get("gap_juegos") or 0,
                    "confianza_señal": s.get("confianza_señal", "BAJA"),
                    "razon": s.get("razon", ""),
                })

    if not all_signals:
        logger.warning("⚠️ No hay señales activas para armar combos")
        return [], {}

    # Apply REGLA-G6: cap stake at $2,000 if n < 50
    effective_stake = stake_per_combo
    regla_g6_active = calibracion_n < 50
    if regla_g6_active:
        effective_stake = min(stake_per_combo, 2000)
        logger.info(f"  REGLA-G6 activa: n={calibracion_n}<50 → stake máx $2,000")

    # Build combo tiers
    # ALTA: gap >= 3 AND cuota >= 1.80
    alta = [s for s in all_signals if s["gap_juegos"] >= 3.0 and s["cuota"] >= 1.80]
    # MEDIA: gap >= 2 AND cuota >= 1.50
    media = [s for s in all_signals if s["gap_juegos"] >= 2.0 and s["cuota"] >= 1.50]
    # ALL: any with apostar=True (already filtered above, cuota >= 1.50)
    all_valid = all_signals  # already filtered

    def _make_combo(signals: List[Dict], max_legs: int, label: str, idx: int) -> Optional[Dict]:
        legs = signals[:max_legs]
        if not legs:
            return None
        outcome_ids = [s["outcome_id"] for s in legs if s.get("outcome_id")]
        if not outcome_ids:
            return None
        cuota_combo = 1.0
        for s in legs:
            cuota_combo *= s["cuota"]
        retorno = round(effective_stake * cuota_combo)
        ids_str = ",".join(str(oid) for oid in outcome_ids)
        url = f"{BETPLAY_URL_BASE}{ids_str}{BETPLAY_URL_TAIL}"
        return {
            "combo_idx": idx,
            "label": label,
            "legs": legs,
            "cuota_combo": round(cuota_combo, 2),
            "stake": effective_stake,
            "retorno": retorno,
            "url": url,
            "outcome_ids": outcome_ids,
            "n_piernas": len(legs),
        }

    combos = []
    # GamesA — solo ALTA
    ca = _make_combo(alta, 3, "GamesA", 1)
    if ca:
        combos.append(ca)
    # GamesB — ALTA + MEDIA (sin duplicar partidos de ALTA)
    partidos_alta = {s["partido"] for s in alta}
    media_extra = [s for s in media if s["partido"] not in partidos_alta]
    legs_b = alta[:3] + media_extra
    cb = _make_combo(legs_b, 3, "GamesB", 2)
    if cb:
        combos.append(cb)
    # GamesC — todo (sin duplicar partidos ya incluidos en ALTA+MEDIA)
    partidos_b = {s["partido"] for s in legs_b[:3]}
    extra_c = [s for s in all_valid if s["partido"] not in partidos_b]
    legs_c = legs_b[:3] + extra_c
    cc = _make_combo(legs_c, 3, "GamesC", 3)
    if cc:
        combos.append(cc)

    # Deduplicate: if GamesA == GamesB legs, drop GamesB
    seen_ids = []
    unique_combos = []
    for c in combos:
        key = tuple(sorted(c["outcome_ids"]))
        if key not in seen_ids:
            seen_ids.append(key)
            unique_combos.append(c)

    total_stake = sum(c["stake"] for c in unique_combos)
    metadata = {
        "fuente": str(signal_path.name),
        "n_señales": len(all_signals),
        "stake_per_combo": effective_stake,
        "total_stake": total_stake,
        "calibracion_n": calibracion_n,
        "regla_g6_active": regla_g6_active,
        "n_alta": len(alta),
        "n_media": len(media),
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    logger.info(f"  {len(unique_combos)} combos Games generados | {len(all_signals)} señales | n_cal={calibracion_n}")
    return unique_combos, metadata


def _mostrar_games_combos(games_links: List[Dict], metadata: Dict):
    """Display games combos in console."""
    total_stake = metadata.get("total_stake", 0)
    stake_each = metadata.get("stake_per_combo", 2000)
    cal_n = metadata.get("calibracion_n", 0)
    g6 = " [REGLA-G6 activa]" if metadata.get("regla_g6_active") else ""

    print()
    print("=" * 70)
    print(f"  GAMES COMBOS — Nodo-40 Totales (juegos/sets){g6}")
    print(f"  {len(games_links)} combos × ${stake_each:,} = ${total_stake:,}")
    print(f"  Señales: {metadata.get('n_señales','?')} | ALTA: {metadata.get('n_alta','?')} | MEDIA: {metadata.get('n_media','?')}")
    print(f"  Calibracion: n={cal_n} observaciones (target n>=50 para escalar)")
    print("=" * 70)

    for gc in games_links:
        cuota = gc["cuota_combo"]
        retorno = gc["retorno"]
        label = gc["label"]
        print(f"\n  {label} [{gc['n_piernas']}p] @{cuota:.2f} → ${retorno:,.0f}")
        for leg in gc["legs"]:
            zona_tag = "DOMIN" if leg["zona_diff"] == "dominante" else "COIN"
            print(f"     {leg['direccion']:<5} {leg['linea']:4.1f}  {leg['mercado']:<20}  @{leg['cuota']:.2f}  "
                  f"[gap={leg['gap_juegos']:.1f}] [{leg['confianza_señal']}] [{zona_tag}]")
            print(f"           {leg['partido']}")

    if games_links:
        best = max(games_links, key=lambda x: x["cuota_combo"])
        print(f"\n  MAYOR CUOTA: {best['label']} @{best['cuota_combo']:.2f} → ${best['retorno']:,.0f}")
    print(f"  INVERSION TOTAL: ${total_stake:,}")
    print("=" * 70)


def _generar_bat_games(games_links: List[Dict]) -> int:
    """Generate GamesA.bat, GamesB.bat, GamesC.bat on Windows desktop."""
    COMBOS_DIR.mkdir(exist_ok=True)

    # Clean previous games combos
    for old_bat in DESKTOP_WIN.glob("Games*.bat"):
        old_bat.unlink(missing_ok=True)
    for old_html in COMBOS_DIR.glob("games_*.html"):
        old_html.unlink(missing_ok=True)

    count = 0
    for gc in games_links:
        label = gc["label"]          # "GamesA", "GamesB", "GamesC"
        html_name = label.lower().replace("games", "games_")  # "games_a", etc.
        cuota = gc["cuota_combo"]
        retorno = gc["retorno"]
        stake = gc["stake"]

        legs_desc = " + ".join(
            f"{l['direccion']} {l['linea']} {l['mercado']} @{l['cuota']:.2f} ({l['partido'].split(' vs ')[1] if ' vs ' in l['partido'] else l['partido']})"
            for l in gc["legs"]
        )
        html_content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{label}</title></head>
<body style="font-family:monospace;text-align:center;padding:40px">
<h2>GAMES {label[-1]} [{gc['n_piernas']}p] @{cuota:.2f}</h2>
<p style="font-size:18px">${stake:,} → ${retorno:,.0f}</p>
<p style="font-size:13px">{legs_desc}</p>
<p><a href="{gc['url']}" target="_blank" style="font-size:24px;padding:20px;background:#0066cc;color:white;text-decoration:none;border-radius:8px">
Abrir en Betplay</a></p>
</body></html>"""

        html_path = COMBOS_DIR / f"{html_name}.html"
        html_path.write_text(html_content, encoding="utf-8")

        bat_content = (
            f'@echo off\r\n'
            f'start "" "{CHROME_WIN}" '
            f'"file:///C:\\users\\hogar\\Desktop\\combos\\{html_name}.html"\r\n'
        )
        bat_path = DESKTOP_WIN / f"{label}.bat"
        bat_path.write_text(bat_content, encoding="utf-8")
        count += 1
        logger.info(f"  {label}.bat — [{gc['n_piernas']}p @{cuota:.2f}] {legs_desc[:80]}")

    return count


def _enviar_games_telegram(games_links: List[Dict], metadata: Dict):
    """Send games combos summary to Telegram."""
    cal_n = metadata.get("calibracion_n", 0)
    g6 = " [G6 n<50]" if metadata.get("regla_g6_active") else ""
    lines = [f"*GAMES COMBOS — Totales (Nodo-40)*{g6}\n"]
    for gc in games_links:
        cuota = gc["cuota_combo"]
        retorno = gc["retorno"]
        stake = gc["stake"]
        legs_str = " × ".join(
            f"{l['direccion']} {l['linea']} @{l['cuota']:.2f}" for l in gc["legs"]
        )
        lines.append(f"*{gc['label']}* @{cuota:.2f} → ${retorno:,.0f} (stake ${stake:,})")
        lines.append(f"  {legs_str}\n")

    lines.append(f"Cal n={cal_n} | Total: ${metadata.get('total_stake', 0):,}")
    text = "\n".join(lines)

    try:
        data = json.dumps({"chat_id": TG_CHAT, "text": text, "parse_mode": "Markdown"}).encode()
        req = urllib.request.Request(TG_URL, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        logger.info("Games combos enviados a Telegram")
    except Exception as e:
        logger.warning(f"⚠️ Error enviando games a Telegram: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# LIVE COMBO BUILDER — Re-arma combos con jugadores disponibles AHORA
# ══════════════════════════════════════════════════════════════════════════════

def _es_coinflip_sin_h2h(p_modelo: float, n_h2h: int) -> bool:
    """T33-01/T33-02 (Nodo-33): coin-flip sin H2H directo — bloqueo duro.

    Sin partidos H2H directos (n_h2h==0) y sin convicción real del modelo
    (p_modelo < P_MODELO_MIN_UNDERDOG), el edge proviene del shrinkage
    James-Stein colapsando p_blend hacia p_hist≈0.50 — no es señal real.
    El bloqueo es independiente de cuota: aplica a favoritos (cuota<2.10)
    igual que a underdogs, cerrando la puerta lateral de BUG-33-2.

    Consolida las dos instancias de bloqueo en _build_mega_combos()
    (cobertura legs + sin_edge picks) en una sola función testeable.
    """
    return n_h2h == 0 and p_modelo < _P_MODELO_MIN_UNDERDOG


def _find_latest_edge_report() -> Optional[str]:
    """Encuentra el edge_report más reciente."""
    reports = Path("reports")
    if not reports.exists():
        return None
    files = sorted(reports.glob("edge_report_*.json"), reverse=True)
    return str(files[0]) if files else None


def _validate_edge_report_gate(edge_data: dict, path: str) -> None:
    """Nodo-32 Acción 3: valida que el edge_report fue generado con el gate actual.
    Falla ruidosamente (SystemExit) si gate_version está ausente o no coincide.
    NO hay fallback silencioso — datos de gate viejo producen phantom edges en combos.
    """
    actual = edge_data.get("metadata", {}).get("gate_version")
    if actual != _EXPECTED_GATE_VERSION:
        msg = (
            f"\n{'='*70}\n"
            f"  ERROR: edge_report con gate_version desactualizado o ausente\n"
            f"  Archivo:  {path}\n"
            f"  Versión en archivo: {actual!r}\n"
            f"  Versión esperada:   {_EXPECTED_GATE_VERSION!r}\n"
            f"\n"
            f"  Regenera el edge_report con el H2H más reciente:\n"
            f"      python3 edge_calculator.py\n"
            f"  (usa automáticamente el h2h_results_enhanced_*.json más reciente en reports/)\n"
            f"  O explícito: python3 edge_calculator.py --h2h reports/h2h_results_enhanced_<fecha>.json\n"
            f"\n"
            f"  Un archivo viejo puede contener picks con apostar=True o\n"
            f"  golden_zone=True calculados con un gate anterior (phantom edge).\n"
            f"{'='*70}\n"
        )
        raise SystemExit(msg)


def _save_betslip_index(picks: list) -> str:
    """
    Guarda mapping outcome_id → pick info para betslip_registrar.py.
    Se llama automáticamente al final de build_live_combos().
    """
    index = {}
    for p in picks:
        oid = p.get("outcome_id")
        if oid:
            index[str(oid)] = {
                "jugador":     p["jugador"],
                "cuota":       p["cuota"],
                "cuota_kambi": p.get("cuota_kambi", p["cuota"]),
                "partido":     p.get("partido", ""),
                "match_id":    p.get("match_id", ""),
                "match_url":   p.get("match_url", ""),
                "torneo":      p.get("torneo", ""),
                "superficie":  p.get("superficie", "?"),
                "tier":        p.get("tier", "?"),
                "edge":        p.get("edge", "0%"),
                "p_modelo":    p.get("p_modelo", 0.5),
                "kelly_kl":    p.get("kelly_kl", 0.0),
            }

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"betslip_index_{ts}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "ts":      datetime.now().isoformat(),
            "modo":    "LIVE",
            "n_picks": len(index),
            "index":   index,
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"   💾 betslip_index guardado: {path}")
    return str(path)


def build_live_combos(piernas_min: int = 3, piernas_max: int = 4,
                      top_n: int = 4, min_cuota: float = 1.50,
                      edge_file: Optional[str] = None,
                      strategy: str = "balanced") -> tuple[List[Dict], Dict]:
    """
    Lee cobertura de TODOS los trader_plans del día, verifica disponibilidad
    en Kambi en tiempo real, y conserva stakes originales del trader.

    El trader ya hizo Kelly/VaR/Cobertura Exclusión por tier.
    Esta función solo mapea combos existentes → URLs Betplay y filtra STARTED.

    Fallback: si no hay trader_plans, arma combos desde edge_report (legacy).

    Returns:
        Tuple de (lista de combo dicts, metadata dict)
    """
    # 1. Cargar cobertura + individuales de trader_plans de las últimas 24h
    reports = Path("reports")
    from datetime import timedelta
    cutoff = datetime.now() - timedelta(hours=24)
    all_plans = sorted(reports.glob("trader_plan_*.json"), reverse=True)
    plan_files = [
        p for p in all_plans
        if p.stat().st_mtime >= cutoff.timestamp() and p.stat().st_size > 100
    ]

    merged_cobertura = []
    merged_individuales = []
    total_bankroll = 0

    for pf in plan_files:
        try:
            plan = json.loads(pf.read_text(encoding="utf-8"))
        except Exception:
            continue
        cobertura = plan.get("cobertura", [])
        individuales = plan.get("individuales", [])
        bankroll = plan.get("metadata", {}).get("bankroll", 0)
        tier = plan.get("metadata", {}).get("torneo_tipo", "?")
        if cobertura:
            merged_cobertura.extend(cobertura)
            merged_individuales.extend(individuales)
            total_bankroll += bankroll
            logger.info(f"📄 {pf.name}: {len(cobertura)} combos, {len(individuales)} individuales (tier={tier})")

    if not merged_cobertura:
        logger.warning("⚠️ No hay trader_plans con combos hoy — armando desde edge_report (legacy)")
        return _build_live_combos_legacy(
            piernas_min, piernas_max, top_n, min_cuota, edge_file, strategy)

    logger.info(f"   📦 Total mergeado: {len(merged_cobertura)} combos de {len(plan_files)} planes")

    # 2. Mapear combos a Kambi (filtra STARTED) — reutiliza build_combo_links
    merged_plan = {"cobertura": merged_cobertura, "individuales": merged_individuales}
    combo_links = build_combo_links(merged_plan, min_piernas=piernas_min)

    # 3. Guardar betslip_index desde individuales disponibles
    edge_path = edge_file or _find_latest_edge_report()
    if edge_path:
        with open(edge_path, encoding="utf-8") as f:
            edge_data = json.load(f)
        _validate_edge_report_gate(edge_data, edge_path)  # Nodo-32 Acción 3
        all_picks = []
        for cat in ("apostar", "watchlist"):
            for p in edge_data.get(cat, []):
                cuota = p.get("cuota_favorito", 0)
                if isinstance(cuota, str):
                    cuota = float(cuota)
                if cuota >= min_cuota:
                    all_picks.append({
                        "jugador":     p["favorito_predicho"],
                        "cuota":       cuota,
                        "edge":        p.get("edge_pct", "0%"),
                        "partido":     p.get("partido", ""),
                        "match_id":    p.get("match_id", ""),
                        "match_url":   p.get("match_url", ""),
                        "torneo":      p.get("torneo", ""),
                        "superficie":  p.get("superficie", "unknown"),
                        "tier":        p.get("tier", "unknown"),
                    })
        # Fetch outcomes para betslip_index
        outcomes_map, started_map = fetch_kambi_outcomes()
        available_picks = []
        for pick in all_picks:
            oc, _ = find_outcome(pick["jugador"], pick["cuota"], outcomes_map, started_map)
            if oc:
                pick["outcome_id"] = str(oc["outcome_id"])
                pick["cuota_kambi"] = oc["odds"]
                available_picks.append(pick)
        if available_picks:
            _save_betslip_index(available_picks)

    # 4. Filtrar combos válidos (con URL)
    valid_combos = [c for c in combo_links if c.get("url")]

    metadata = {
        "bankroll":          total_bankroll,
        "modo":              "LIVE",
        "strategy":          strategy,
        "picks_totales":     len(merged_individuales),
        "picks_disponibles": len(valid_combos),
        "planes_mergeados":  len(plan_files),
    }

    logger.info(f"   🏗️ {len(valid_combos)} combos disponibles de {len(merged_cobertura)} totales")
    return valid_combos, metadata


# ══════════════════════════════════════════════════════════════════════════════
# NODO-23 — CROSS-TIER MEGA-COMBOS
# Genera combos de 6-10 piernas cruzando tiers (Challenger + ITF, etc.)
# Sesión épica 2026-06-13: 7p @269.2 → $52,608 | 8p @632.6 → PAGÓ
# ══════════════════════════════════════════════════════════════════════════════

# Correlación cross-tier: picks de tiers/torneos distintos son casi independientes
_RHO_CROSS = {
    "same_tournament": 0.25,
    "same_tier_diff_tournament": 0.10,
    "cross_tier": 0.03,
}

# Mega-combo escalera: piernas → (top_n combos, stake_fraction)
_MEGA_LADDER = {
    6:  3,
    7:  3,
    8:  2,
    9:  1,
    10: 1,
}


def build_mega_combos(stake_per_combo: int = 500,
                      piernas_min: int = 6,
                      piernas_max: int = 10,
                      min_anclas: int = 0,
                      ancla_cuota_max: float = 1.80,
                      min_tiers: int = 2,
                      no_bbi_filter: bool = False) -> tuple[List[Dict], Dict]:
    """
    Nodo-23: Cross-Tier Mega-Combos.

    Mergea picks APOSTAR+watchlist de TODOS los trader_plans (últimas 24h),
    genera combos de 6-10 piernas cruzando tiers, con ≥1 pierna ancla
    (cuota<1.80) y ≥2 tiers distintos.

    Scoring (Nodo-24): (Π mpq_i) × log(cuota_combo) × cross_tier_bonus × gap_penalty
    BBI < 0.40 → excluido del pool mega (bookmaker tiene info suficiente).
    Golden zone picks (Challenger/ITF, cuota≥2.50, n_h2h=0) → bonus ×1.20.
    """
    from itertools import combinations
    import math

    # 1. Cargar todos los trader_plans de las últimas 24h
    reports = Path("reports")
    from datetime import timedelta
    cutoff = datetime.now() - timedelta(hours=24)
    all_plans = sorted(reports.glob("trader_plan_*.json"), reverse=True)
    plan_files = [
        p for p in all_plans
        if p.stat().st_mtime >= cutoff.timestamp() and p.stat().st_size > 100
    ]

    if not plan_files:
        logger.warning("⚠️ No hay trader_plans en las últimas 24h para mega-combos")
        return [], {}

    # 1b. Cargar edge_report para inferir tier per-pick y campos Nodo-24
    edge_tier_map = {}  # jugador → {tier, superficie, bbi, gap_flag, mpq, golden_zone, n_h2h}
    edge_path = _find_latest_edge_report()
    if edge_path:
        try:
            with open(edge_path, encoding="utf-8") as f:
                edge_data = json.load(f)
            _validate_edge_report_gate(edge_data, edge_path)  # Nodo-32 Acción 3
            for cat in ("apostar", "watchlist", "sin_edge"):
                for p in edge_data.get(cat, []):
                    name = p.get("favorito_predicho", "")
                    if name:
                        edge_tier_map[name] = {
                            "tier":         p.get("tier", "unknown"),
                            "superficie":   p.get("superficie", "unknown"),
                            # Nodo-24 campos
                            "bbi":          p.get("bbi", 0.5),
                            "gap_flag":     p.get("gap_flag", "MIXED"),
                            "mpq":          p.get("mpq", 0.0),
                            "golden_zone":  p.get("golden_zone", False),
                            "n_h2h":        p.get("n_h2h", 0),
                            # Nodo-26 M-26-3: cuota original para Line Movement Signal
                            "cuota_original": p.get("cuota_favorito"),
                            "edge":           p.get("edge", 0),
                        }
        except Exception:
            pass

    # 2. Extraer pool unificado de picks con metadata de tier
    pool = []
    seen_names = set()

    for pf in plan_files:
        try:
            plan = json.loads(pf.read_text(encoding="utf-8"))
        except Exception:
            continue

        meta = plan.get("metadata", {})
        params = meta.get("parametros", {})
        tier = params.get("torneo_tipo", "unknown")
        superficie = params.get("superficie", "unknown")

        for p in plan.get("individuales", []):
            name = p.get("favorito", "")
            if name and name not in seen_names:
                seen_names.add(name)
                # Tier: prefer plan metadata → edge_report → pick superficie
                pick_tier = tier if tier != "unknown" else edge_tier_map.get(name, {}).get("tier", "unknown")
                pick_sup = p.get("superficie", superficie)
                if pick_sup == "unknown":
                    pick_sup = edge_tier_map.get(name, {}).get("superficie", "unknown")
                _edge_info = edge_tier_map.get(name, {})
                # Nodo-24: gap = p_blend - p_modelo (calculado localmente si no viene del edge_report)
                _p_blend_v = p.get("p_blend", 0.5)
                _p_modelo_v = p.get("p_modelo", 0.5)
                _gap_local = round(_p_blend_v - _p_modelo_v, 4)
                pool.append({
                    "jugador":    name,
                    "cuota":      p.get("cuota", 0),
                    "p_blend":    _p_blend_v,
                    "p_modelo":   _p_modelo_v,
                    "edge_pct":   p.get("edge_pct", "0%"),
                    "tier":       pick_tier,
                    "superficie": pick_sup,
                    "torneo":     p.get("partido", "").split(" vs ")[0] if " vs " in p.get("partido", "") else "",
                    "zona_cuota": p.get("zona_cuota", "underdog"),
                    "tipo":       "ancla" if p.get("cuota", 99) < ancla_cuota_max else "satelite",
                    # Nodo-24 fields (from edge_report lookup, fallback computed locally)
                    "bbi":         _edge_info.get("bbi", 0.5),
                    "gap_flag":    _edge_info.get("gap_flag", "CALIBRATION_DRIVEN" if _gap_local > 0.12 else "MARKET_DRIVEN" if _gap_local < 0.08 else "MIXED"),
                    "mpq":         _edge_info.get("mpq", 0.0),
                    "golden_zone": _edge_info.get("golden_zone", False),
                    "n_h2h":       _edge_info.get("n_h2h", 0),
                })

        # Also grab watchlist picks from cobertura legs not in individuales
        for combo in plan.get("cobertura", []):
            for leg in combo.get("legs", []):
                name = leg.get("jugador", "")
                if name and name not in seen_names:
                    seen_names.add(name)
                    leg_tier = tier if tier != "unknown" else edge_tier_map.get(name, {}).get("tier", "unknown")
                    leg_sup = edge_tier_map.get(name, {}).get("superficie", superficie)
                    # T33-02 (Nodo-33): lookup real desde edge_report en vez de hardcode 0.55/0.50
                    _info = edge_tier_map.get(name, {})
                    _p_modelo_real = _info.get("p_modelo", 0.50)
                    _n_h2h_real    = _info.get("n_h2h", 0)
                    # Bloqueo duro: n_h2h=0 + p_modelo<0.55 = coin-flip, no entra al pool
                    if _es_coinflip_sin_h2h(_p_modelo_real, _n_h2h_real):
                        logger.info(f"  🚫 {name} @{leg.get('cuota',0)} — n_h2h=0 + p_modelo={_p_modelo_real:.3f} < {_P_MODELO_MIN_UNDERDOG} (coin-flip guard, excluido pool mega)")
                        continue
                    pool.append({
                        "jugador":    name,
                        "cuota":      leg.get("cuota", 0),
                        "p_blend":    _info.get("p_blend", 0.55),
                        "p_modelo":   _p_modelo_real,
                        "n_h2h":      _n_h2h_real,
                        "edge_pct":   _info.get("edge_pct", "?"),
                        "tier":       leg_tier,
                        "superficie": leg_sup,
                        "torneo":     "",
                        "zona_cuota": "slight_underdog",
                        "tipo":       "ancla" if leg.get("cuota", 99) < ancla_cuota_max else "satelite",
                    })

    if len(pool) < piernas_min:
        logger.warning(f"⚠️ Pool insuficiente para mega-combos: {len(pool)} picks (mínimo {piernas_min})")
        return [], {}

    n_anclas_pool = sum(1 for p in pool if p["tipo"] == "ancla")
    n_tiers_pool = len({p["tier"] for p in pool})
    logger.info(f"🎯 Mega-combo pool: {len(pool)} picks | {n_anclas_pool} anclas | {n_tiers_pool} tiers")

    # 2b. Si no hay anclas en pool del trader, buscar en edge_report (sin_edge picks)
    if n_anclas_pool == 0 and edge_path:
        try:
            with open(edge_path, encoding="utf-8") as f:
                edge_data = json.load(f)
            for cat in ("sin_edge",):
                for p in edge_data.get(cat, []):
                    name = p.get("favorito_predicho", "")
                    cuota = p.get("cuota_favorito", 99)
                    if name and name not in seen_names and cuota < ancla_cuota_max:
                        seen_names.add(name)
                        # T33-02 (Nodo-33): usar valores reales del edge_report (sin_edge tiene todos los campos)
                        _p_modelo_se = p.get("p_modelo", 0.50)
                        _n_h2h_se    = p.get("n_h2h", 0)
                        # Bloqueo duro: n_h2h=0 + p_modelo<0.55 = coin-flip
                        if _es_coinflip_sin_h2h(_p_modelo_se, _n_h2h_se):
                            logger.info(f"  🚫 {name} @{cuota} — n_h2h=0 + p_modelo={_p_modelo_se:.3f} < {_P_MODELO_MIN_UNDERDOG} (coin-flip guard sin_edge, excluido pool mega)")
                            continue
                        pool.append({
                            "jugador":    name,
                            "cuota":      cuota,
                            "p_blend":    p.get("p_blend", 0.55),
                            "p_modelo":   _p_modelo_se,
                            "n_h2h":      _n_h2h_se,
                            "edge_pct":   "0%",
                            "tier":       p.get("tier", "unknown"),
                            "superficie": p.get("superficie", "unknown"),
                            "torneo":     p.get("torneo", ""),
                            "zona_cuota": "slight_favorite",
                            "tipo":       "ancla",
                        })
            n_anclas_pool = sum(1 for p in pool if p["tipo"] == "ancla")
            n_tiers_pool = len({p["tier"] for p in pool})
            if n_anclas_pool > 0:
                logger.info(f"  ⚓ Añadidas {n_anclas_pool} anclas desde edge_report (sin_edge)")
        except Exception:
            pass

    if n_tiers_pool < min_tiers:
        logger.warning(f"⚠️ Solo {n_tiers_pool} tier(s) en pool — mega-combos requieren ≥{min_tiers}")
        return [], {}

    # 3. Verificar disponibilidad en Kambi
    outcomes_map, started_map = fetch_kambi_outcomes()
    if not outcomes_map:
        logger.error("❌ No se pudieron obtener outcomes de Kambi")
        return [], {}

    available_pool = []
    for pick in pool:
        oc, reason = find_outcome(pick["jugador"], pick["cuota"], outcomes_map, started_map)
        if oc:
            pick["outcome_id"] = str(oc["outcome_id"])
            pick["cuota_kambi"] = oc["odds"]
            # Nodo-24 Guard: BBI < 0.40 → bookmaker tiene info suficiente → excluir
            _bbi_v = pick.get("bbi", 0.5)
            if not no_bbi_filter and _bbi_v < 0.40:
                logger.info(f"  🚫 {pick['jugador']} @{pick['cuota']} — BBI={_bbi_v:.3f} < 0.40 (bookmaker informado, excluido mega)")
                continue
            # Nodo-26 M-26-3: Line Movement Signal
            _cuota_orig = edge_tier_map.get(pick["jugador"], {}).get("cuota_original")
            _line_factor, _line_signal = line_movement_signal(_cuota_orig, oc["odds"])
            pick["line_factor"] = _line_factor
            pick["line_signal"] = _line_signal
            # Nodo-26 M-26-5: edge value for CV guard
            pick["edge"] = edge_tier_map.get(pick["jugador"], {}).get("edge", 0)
            available_pool.append(pick)
        else:
            logger.info(f"  ⏭️ {pick['jugador']} @{pick['cuota']} — {reason} (excluido de mega)")

    if len(available_pool) < piernas_min:
        logger.warning(f"⚠️ Solo {len(available_pool)} picks disponibles en Kambi (mínimo {piernas_min})")
        return [], {}

    logger.info(f"  ✅ {len(available_pool)} picks disponibles en Kambi para mega-combos")

    # Nodo-26 M-26-5: CV Edge Guard
    cv_val, cv_status = cv_edge_guard(available_pool)
    if cv_val is not None:
        logger.info(f"  📊 CV Edge Guard: CV={cv_val:.4f} → {cv_status}")

    # Nodo-26 M-26-1: Dispersion Guard → Cross-Sectional Ranking (if BLIND)
    disp_std, disp_status = dispersion_index(available_pool)
    logger.info(f"  📊 Dispersion Guard: std={disp_std:.4f} → {disp_status}")

    # If BOTH guards are BLIND → block mega-combos (Nodo-25 + Nodo-26)
    if disp_status == "BLIND" and cv_status == "BLIND_EDGE":
        logger.warning("  🚫 MEGA BLOQUEADO: Dispersion=BLIND + CV=BLIND_EDGE → modelo ciego, skip megas")
        return [], {}

    # If only Dispersion BLIND → apply Cross-Sectional Ranking Preservation
    if disp_status == "BLIND":
        logger.info(f"  🎯 BLIND pool → Cross-Sectional Ranking aplicado (amplification=5.0)")
        std_before = disp_std
        available_pool = ranking_preserved_blend(available_pool, p_historica=0.59, js_factor=0.17)
        import numpy as np
        std_after = float(np.std([p["p_blend"] for p in available_pool]))
        logger.info(f"     std(p_blend): {std_before:.4f} → {std_after:.4f}")

    if cv_status == "BLIND_EDGE":
        logger.warning("  ⚠️ CV=BLIND_EDGE — edges casi idénticos, considera revisar pool")

    # 4. Generar escalera de mega-combos
    mega_combos = []

    for n_piernas in range(piernas_min, min(piernas_max + 1, len(available_pool) + 1)):
        top_n = _MEGA_LADDER.get(n_piernas, 1)

        # Generate all combinations
        candidates = []
        for combo_picks in combinations(available_pool, n_piernas):
            # R-23-4: ≥2 tiers distintos
            tiers_in_combo = {p["tier"] for p in combo_picks}
            if len(tiers_in_combo) < min_tiers:
                continue

            # R-23-5: ≥1 pierna ancla
            anclas_in_combo = sum(1 for p in combo_picks if p["tipo"] == "ancla")
            if anclas_in_combo < min_anclas:
                continue

            # Nodo-24 Scoring: (Π mpq_i) × log(cuota_combo) × cross_tier_bonus × gap_penalty × golden_bonus
            p_todas = 1.0
            cuota_combo = 1.0
            mpq_product = 1.0
            gap_penalty = 1.0
            n_golden = 0
            n_calibration = 0
            for p in combo_picks:
                p_todas *= p["p_blend"]
                cuota_combo *= p["cuota_kambi"]
                # MPQ product (use p_blend fallback if mpq=0)
                _mpq_p = p.get("mpq", 0.0)
                if _mpq_p <= 0:
                    _mpq_p = p["p_blend"] * p.get("bbi", 0.5)
                mpq_product *= max(_mpq_p, 1e-9)
                # gap_penalty: CALIBRATION_DRIVEN piernas reducen score (×0.85 por pierna)
                if p.get("gap_flag") == "CALIBRATION_DRIVEN":
                    gap_penalty *= 0.85
                    n_calibration += 1
                # golden_zone bonus counter
                if p.get("golden_zone", False):
                    n_golden += 1

            cross_tier_bonus = 1.0 + 0.05 * len(tiers_in_combo)
            # Golden zone: si ≥50% de piernas son golden → bonus ×1.20
            golden_bonus = 1.20 if n_golden >= n_piernas / 2 else 1.0
            # Nodo-26 M-26-3: Line Movement product (STEAM_IN boosts, DRIFT_OUT penalizes)
            line_product = 1.0
            for p in combo_picks:
                line_product *= p.get("line_factor", 1.0)
            mega_score = mpq_product * math.log(max(cuota_combo, 1.01)) * cross_tier_bonus * gap_penalty * golden_bonus * line_product

            candidates.append({
                "picks":            list(combo_picks),
                "n_piernas":        n_piernas,
                "p_todas":          p_todas,
                "cuota_combo":      round(cuota_combo, 2),
                "mega_score":       round(mega_score, 12),
                "n_tiers":          len(tiers_in_combo),
                "tiers":            sorted(tiers_in_combo),
                "n_anclas":         anclas_in_combo,
                "cross_tier_bonus": cross_tier_bonus,
                "gap_penalty":      round(gap_penalty, 4),
                "golden_bonus":     golden_bonus,
                "n_golden":         n_golden,
                "n_calibration":    n_calibration,
            })

        # Sort by mega_score descending → top N
        candidates.sort(key=lambda x: x["mega_score"], reverse=True)
        top_candidates = candidates[:top_n]

        for cand in top_candidates:
            outcome_ids = [p["outcome_id"] for p in cand["picks"]]
            ids_str = ",".join(outcome_ids)
            betplay_url = f"{BETPLAY_URL_BASE}{ids_str}{BETPLAY_URL_TAIL}"

            mega_combos.append({
                "combo_idx":   len(mega_combos) + 1,
                "piernas":     cand["n_piernas"],
                "legs":        [{
                    "jugador":    p["jugador"],
                    "cuota":      p["cuota"],
                    "cuota_kambi": p["cuota_kambi"],
                    "outcome_id": p["outcome_id"],
                    "tier":       p["tier"],
                    "tipo":       p["tipo"],
                    # Nodo-24 per-leg fields
                    "bbi":        p.get("bbi", 0.5),
                    "gap_flag":   p.get("gap_flag", "MIXED"),
                    "golden_zone": p.get("golden_zone", False),
                    "mpq":        p.get("mpq", 0.0),
                } for p in cand["picks"]],
                "outcome_ids": outcome_ids,
                "url":         betplay_url,
                "partial":     False,
                "stake":       stake_per_combo,
                "cuota_combo": cand["cuota_combo"],
                "retorno":     round(stake_per_combo * cand["cuota_combo"], 0),
                "p_todas":     round(cand["p_todas"], 6),
                "mega_score":  cand["mega_score"],
                "n_tiers":     cand["n_tiers"],
                "tiers":       cand["tiers"],
                "n_anclas":    cand["n_anclas"],
                # Nodo-24 combo-level fields
                "gap_penalty":    cand.get("gap_penalty", 1.0),
                "golden_bonus":   cand.get("golden_bonus", 1.0),
                "n_golden":       cand.get("n_golden", 0),
                "n_calibration":  cand.get("n_calibration", 0),
            })

        if top_candidates:
            logger.info(f"  📊 {n_piernas}p: {len(candidates)} candidatos → top {len(top_candidates)} (mejor @{top_candidates[0]['cuota_combo']:.1f})")

    metadata = {
        "modo":           "MEGA",
        "pool_total":     len(pool),
        "pool_disponible": len(available_pool),
        "n_tiers":        n_tiers_pool,
        "n_anclas":       n_anclas_pool,
        "stake_per_combo": stake_per_combo,
        "total_stake":    stake_per_combo * len(mega_combos),
        "n_mega_combos":  len(mega_combos),
        "planes_leidos":  len(plan_files),
    }

    logger.info(f"  🚀 {len(mega_combos)} mega-combos generados | stake total ${stake_per_combo * len(mega_combos):,}")
    return mega_combos, metadata


def _build_live_combos_legacy(piernas_min: int = 3, piernas_max: int = 4,
                              top_n: int = 4, min_cuota: float = 1.50,
                              edge_file: Optional[str] = None,
                              strategy: str = "balanced") -> tuple[List[Dict], Dict]:
    """
    Fallback: arma combos desde edge_report cuando no hay trader_plans.
    Lógica original de build_live_combos.
    """
    from itertools import combinations

    edge_path = edge_file or _find_latest_edge_report()
    if not edge_path:
        logger.error("❌ No se encontró edge_report_*.json en reports/")
        return [], {}

    logger.info(f"📄 Edge report: {edge_path}")
    with open(edge_path, encoding="utf-8") as f:
        edge_data = json.load(f)
    _validate_edge_report_gate(edge_data, edge_path)  # Nodo-32 Acción 3

    all_picks = []
    for cat in ("apostar", "watchlist"):
        for p in edge_data.get(cat, []):
            cuota = p.get("cuota_favorito", 0)
            if isinstance(cuota, str):
                cuota = float(cuota)
            if cuota >= min_cuota:
                all_picks.append({
                    "jugador":           p["favorito_predicho"],
                    "cuota":             cuota,
                    "edge":              p.get("edge_pct", "0%"),
                    "tier":              p.get("tier", "?"),
                    "superficie":        p.get("superficie", "?"),
                    "categoria":         cat,
                    "partido":           p.get("partido", ""),
                    "match_id":          p.get("match_id", ""),
                    "match_url":         p.get("match_url", ""),
                    "torneo":            p.get("torneo", ""),
                    "p_modelo":          p.get("p_modelo", 0.5),
                    "zona_cuota":        p.get("zona_cuota", "slight_underdog"),
                    "markov_favorito":   p.get("markov_favorito"),
                    "kelly_kl":          p.get("kelly_kl", 0.0),
                    "alpha_vs_elo":      p.get("alpha_vs_elo", 0.0),
                    "n_h2h":             p.get("n_h2h", 0),
                })

    logger.info(f"   {len(all_picks)} picks con cuota >= {min_cuota}")

    outcomes_map, started_map = fetch_kambi_outcomes()
    if not outcomes_map:
        return [], {}

    available_picks = []
    for pick in all_picks:
        oc, reason = find_outcome(pick["jugador"], pick["cuota"],
                                  outcomes_map, started_map)
        if oc:
            pick["outcome_id"] = str(oc["outcome_id"])
            pick["cuota_kambi"] = oc["odds"]
            available_picks.append(pick)
            logger.info(f"  ✅ {pick['jugador']:25s} @{pick['cuota']:.2f} → @{oc['odds']:.2f} [{pick['categoria']}]")
        else:
            logger.warning(f"  ❌ {pick['jugador']:25s} @{pick['cuota']:.2f} — {reason}")

    if len(available_picks) < piernas_min:
        logger.error(f"❌ Solo {len(available_picks)} picks disponibles, mínimo {piernas_min}")
        return [], {}

    logger.info(f"   🎯 {len(available_picks)} picks disponibles para combos")
    _save_betslip_index(available_picks)

    combos = []
    combo_idx = 0
    for k in range(piernas_min, min(piernas_max, len(available_picks)) + 1):
        tier_combos = []
        for combo_picks in combinations(available_picks, k):
            cuota_combo = 1.0
            for cp in combo_picks:
                cuota_combo *= cp["cuota_kambi"]
            cuota_combo = round(cuota_combo, 2)

            outcome_ids = [cp["outcome_id"] for cp in combo_picks]
            ids_str = ",".join(outcome_ids)

            tier_combos.append({
                "picks": list(combo_picks),
                "cuota_combo": cuota_combo,
                "outcome_ids": outcome_ids,
                "url": f"{BETPLAY_URL_BASE}{ids_str}{BETPLAY_URL_TAIL}",
                "piernas": k,
            })

        for tc in tier_combos:
            tc.update(_score_combo(tc["picks"], strategy=strategy))
        tier_combos.sort(key=lambda x: x["combo_score"], reverse=True)
        selected_tier = _select_with_cobertura(tier_combos, available_picks, top_n, k)
        for tc in selected_tier:
            combo_idx += 1
            legs = [{
                "jugador":     cp["jugador"],
                "cuota":       cp["cuota"],
                "cuota_kambi": cp["cuota_kambi"],
                "outcome_id":  cp["outcome_id"],
                "zona_cuota":  cp.get("zona_cuota", "?"),
                "markov":      cp.get("markov_favorito") or "?",
                "p_modelo":    cp.get("p_modelo", 0.5),
            } for cp in tc["picks"]]

            combos.append({
                "combo_idx":   combo_idx,
                "piernas":     tc["piernas"],
                "legs":        legs,
                "skipped_legs": [],
                "outcome_ids": tc["outcome_ids"],
                "url":         tc["url"],
                "partial":     False,
                "stake":       0,
                "cuota_combo": tc["cuota_combo"],
                "retorno":     0,
                "combo_score": tc.get("combo_score", 0),
                "combo_ev":    tc.get("combo_ev", 0),
                "combo_hr":    tc.get("combo_hr", 0),
                "breakdown":   tc.get("breakdown", ""),
            })

    metadata = {
        "bankroll":          0,
        "modo":              "LIVE_LEGACY",
        "strategy":          strategy,
        "picks_totales":     len(all_picks),
        "picks_disponibles": len(available_picks),
        "edge_report":       edge_path,
    }

    logger.info(f"   🏗️ {len(combos)} combos armados ({piernas_min}-{piernas_max} piernas, top {top_n})")
    return combos, metadata


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def mostrar_combos(combo_links: List[Dict], metadata: Dict):
    """Muestra combos en consola."""
    bankroll = metadata.get("bankroll", 0)

    print("\n" + "=" * 70)
    print("  🎾 BETPLAY COMBO BUILDER")
    print(f"  💰 Bankroll: ${bankroll:,.0f}")
    print("=" * 70)

    for c in combo_links:
        if c["url"]:
            tag = "⚠️ PARCIAL" if c.get("partial") else "✅"
        else:
            tag = "❌"
        print(f"\n  {tag} Combo {c['combo_idx']} [{c['piernas']}p] — @{c['cuota_combo']:.1f} → ${c['retorno']:,.0f}")

        for leg in c["legs"]:
            cuota = leg["cuota"]
            print(f"     ✅ {leg['jugador']:25s} @{cuota:.2f} → id={leg['outcome_id']}")

        for leg in c.get("skipped_legs", []):
            cuota = leg["cuota"]
            print(f"     ❌ {leg['jugador']:25s} @{cuota:.2f} — {leg['error']}")

        if c.get("breakdown"):
            print(f"     Score: {c.get('combo_score', 0):.3f} | HR={c.get('combo_hr', 0):.3f} | {c['breakdown']}")

        if c["url"]:
            print(f"     Stake: ${c['stake']:,.0f}")

    valid = sum(1 for c in combo_links if c["url"])
    partial = sum(1 for c in combo_links if c.get("partial"))
    invalid = len(combo_links) - valid
    print(f"\n  📊 {valid} listos ({partial} parciales) / {invalid} sin outcome")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# MEGA-COMBO DISPLAY + BAT GENERATION (Nodo-23)
# ══════════════════════════════════════════════════════════════════════════════


def _mostrar_mega_combos(mega_links: List[Dict], metadata: Dict):
    """Muestra mega-combos en consola con formato claro."""
    total_stake = metadata.get("total_stake", 0)
    stake_each = metadata.get("stake_per_combo", 500)

    print()
    print("=" * 70)
    print(f"  🚀 MEGA-COMBOS CROSS-TIER (Nodo-23)")
    print(f"  💰 {len(mega_links)} combos × ${stake_each:,} = ${total_stake:,}")
    print(f"  📊 Pool: {metadata.get('pool_disponible', '?')} picks | "
          f"{metadata.get('n_tiers', '?')} tiers | "
          f"{metadata.get('n_anclas', '?')} anclas")
    print("=" * 70)

    for mc in mega_links:
        piernas = mc["piernas"]
        cuota = mc["cuota_combo"]
        retorno = mc["retorno"]
        p_todas = mc.get("p_todas", 0)
        tiers = mc.get("tiers", [])
        n_anclas = mc.get("n_anclas", 0)
        n_golden = mc.get("n_golden", 0)
        n_cal = mc.get("n_calibration", 0)
        gap_pen = mc.get("gap_penalty", 1.0)
        gold_bon = mc.get("golden_bonus", 1.0)

        golden_tag = f" 🌟 EPIC({n_golden}/{piernas})" if gold_bon > 1.0 else ""
        cal_tag = f" ⚠️ CAL:{n_cal}" if n_cal > 0 else ""
        print(f"\n  ✅ Mega {mc['combo_idx']} [{piernas}p] — @{cuota:,.1f} → ${retorno:,.0f}{golden_tag}{cal_tag}")
        print(f"     P(todas)={p_todas:.4f} | Tiers: {'+'.join(tiers)} | gap_penalty={gap_pen:.2f}")
        for leg in mc["legs"]:
            tipo_icon = "⚓" if leg.get("tipo") == "ancla" else "🛰️"
            bbi = leg.get("bbi", 0.5)
            gap_flag = leg.get("gap_flag", "MIXED")
            flag_short = {"MARKET_DRIVEN": "MARKET", "CALIBRATION_DRIVEN": "CAL⚠️", "MIXED": "MIXED"}.get(gap_flag, gap_flag)
            golden_icon = " 🌟" if leg.get("golden_zone") else ""
            print(f"     {tipo_icon} {leg['jugador']:<25} @{leg['cuota_kambi']:.2f}  BBI={bbi:.3f}  {flag_short:<8}{golden_icon}  [{leg['tier']}]")

    print()
    print(f"  💰 INVERSIÓN TOTAL: ${total_stake:,}")
    if mega_links:
        best = max(mega_links, key=lambda x: x["retorno"])
        print(f"  🎯 MEJOR RETORNO: Mega {best['combo_idx']} [{best['piernas']}p] → ${best['retorno']:,.0f}")
    print("=" * 70)


def _generar_bat_mega(mega_links: List[Dict]) -> int:
    """Genera Mega1.bat ... MegaN.bat en el escritorio de Windows."""
    COMBOS_DIR.mkdir(exist_ok=True)

    # Limpiar mega-combos anteriores
    for old_bat in DESKTOP_WIN.glob("Mega*.bat"):
        old_bat.unlink(missing_ok=True)
    for old_html in COMBOS_DIR.glob("mega*.html"):
        old_html.unlink(missing_ok=True)

    count = 0
    for mc in mega_links:
        url = mc.get("url")
        if not url:
            continue

        idx = mc["combo_idx"]
        piernas = mc["piernas"]
        cuota = mc["cuota_combo"]

        # Redirect HTML (same pattern as regular combos)
        legs_desc = " + ".join(
            f"{l['jugador']}@{l['cuota_kambi']:.2f}" for l in mc["legs"]
        )
        html_content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Mega {idx}</title></head>
<body style="font-family:monospace;text-align:center;padding:40px">
<h2>🚀 MEGA {idx} [{piernas}p] @{cuota:,.1f}</h2>
<p>{legs_desc}</p>
<p><a href="{url}" target="_blank" style="font-size:24px;padding:20px;background:#0066ff;color:white;text-decoration:none;border-radius:8px">
Abrir en Betplay</a></p>
</body></html>"""

        html_path = COMBOS_DIR / f"mega{idx}.html"
        html_path.write_text(html_content, encoding="utf-8")

        bat_content = (
            f'@echo off\r\n'
            f'start "" "{CHROME_WIN}" '
            f'"file:///C:\\users\\hogar\\Desktop\\combos\\mega{idx}.html"\r\n'
        )
        bat_path = DESKTOP_WIN / f"Mega{idx}.bat"
        bat_path.write_text(bat_content, encoding="utf-8")
        count += 1
        logger.info(f"  📄 Mega{idx}.bat — [{piernas}p @{cuota:,.1f}] {legs_desc[:80]}")

    return count


def _enviar_mega_telegram(mega_links: List[Dict], metadata: Dict):
    """Envía resumen de mega-combos a Telegram."""
    # Mensaje compacto — una línea por mega para no superar 4096 chars
    lines = ["*MEGA-COMBOS*"]
    for mc in mega_links:
        piernas = mc["piernas"]
        cuota = mc["cuota_combo"]
        retorno = mc["retorno"]
        # Solo jugadores sin cuotas individuales para reducir largo
        names = " + ".join(l['jugador'].split()[-1] for l in mc["legs"])
        lines.append(f"Mega{mc['combo_idx']} [{piernas}p] @{cuota:,.0f} ${retorno:,.0f} | {names}")

    lines.append(f"Total: ${metadata.get('total_stake', 0):,}")
    text = "\n".join(lines)

    try:
        data = json.dumps({"chat_id": TG_CHAT, "text": text, "parse_mode": "Markdown"}).encode()
        req = urllib.request.Request(TG_URL, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        logger.info("📱 Mega-combos enviados a Telegram ✅")
    except Exception as e:
        logger.warning(f"⚠️ Error enviando mega a Telegram: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def find_latest_trader_plan() -> Optional[str]:
    """Encuentra el trader_plan más reciente."""
    reports = Path("reports")
    if not reports.exists():
        return None
    plans = sorted(reports.glob("trader_plan_*.json"), reverse=True)
    return str(plans[0]) if plans else None




def main():
    parser = argparse.ArgumentParser(
        description="Arma combos de Betplay — genera .bat para Chrome"
    )
    parser.add_argument("--file", help="trader_plan JSON específico")
    parser.add_argument("--live", action="store_true",
                        help="Modo LIVE: re-arma combos con jugadores disponibles AHORA en Kambi")
    parser.add_argument("--edge-file", help="edge_report JSON específico (para --live)")
    parser.add_argument("--piernas-min", type=int, default=3, help="Piernas mínimas (default: 3)")
    parser.add_argument("--piernas-max", type=int, default=4, help="Piernas máximas (default: 4)")
    parser.add_argument("--top-n", type=int, default=4, help="Top N combos por pierna (default: 4)")
    parser.add_argument("--min-cuota", type=float, default=1.50, help="Cuota mínima (default: 1.50)")
    parser.add_argument("--strategy", choices=["balanced", "aggressive", "conservative"],
                        default="balanced", help="Estrategia de scoring de combos (default: balanced)")
    parser.add_argument("--mega", action="store_true",
                        help="Nodo-23: Cross-Tier Mega-Combos (6-10 piernas entre tiers)")
    parser.add_argument("--mega-stake", type=int, default=500,
                        help="Stake por mega-combo (default: $500)")
    parser.add_argument("--mega-min", type=int, default=6,
                        help="Piernas mínimas mega-combo (default: 6)")
    parser.add_argument("--mega-max", type=int, default=10,
                        help="Piernas máximas mega-combo (default: 10)")
    parser.add_argument("--safe", action="store_true",
                        help="Nodo-25: Safe Combos — Beta Book (2 piernas, P>25pct)")
    parser.add_argument("--safe-stake", type=int, default=1000,
                        help="Stake por safe combo (default: $1000)")
    parser.add_argument("--safe-top-n", type=int, default=8,
                        help="Top N safe combos (default: 8)")
    parser.add_argument("--was", action="store_true",
                        help="Nodo-44: Watchlist Alpha Signal — combos promo alpha invisible")
    parser.add_argument("--was-stake", type=int, default=5000,
                        help="Stake por WAS combo (default: $5000 — promo stake minimo)")
    parser.add_argument("--was-min-edge", type=float, default=10.0,
                        help="Edge minimo para WAS filter (default: 10.0%%)")
    parser.add_argument("--was-top-n", type=int, default=5,
                        help="Top N WAS combos (default: 5)")
    parser.add_argument("--games", action="store_true",
                        help="Nodo-40: Games/Sets Signal Combos (totales, over/under)")
    parser.add_argument("--games-stake", type=int, default=2000,
                        help="Stake por games combo (default: $2000, REGLA-G6 cap)")
    parser.add_argument("--games-file", help="games_signal_report JSON específico")
    parser.add_argument("--no-dispersion-guard", action="store_true",
                        help="Disable Dispersion Guard (Nodo-25)")
    parser.add_argument("--allow-extra", action="store_true",
                        help="Allow picks outside trader_plan (Nodo-25 Guard 3 override)")
    parser.add_argument("--no-bbi-filter", action="store_true",
                        help="Nodo-24: disable BBI<0.40 filter for mega-combos (permite picks con bookmaker informado)")
    parser.add_argument("--console", action="store_true",
                        help="Mostrar comandos para consola F12 de Betplay")
    parser.add_argument("--telegram", action="store_true",
                        help="Enviar resumen a Telegram")
    parser.add_argument("--whatsapp", action="store_true",
                        help="Generar HTML con botones wa.me link por link")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo mostrar, no generar archivos")
    parser.add_argument("--bankroll", type=float, default=0,
                        help="Nodo-26 M-26-2: bankroll para Circuit Breaker (default: auto desde trader_plan)")
    args = parser.parse_args()

    # ── Nodo-26 M-26-2/4: Session Budget + Meta-Markov ──────────────────────
    _bankroll = args.bankroll if args.bankroll > 0 else _find_bankroll_from_plans()
    _meta_factor = 1.0
    _meta_regime = "INSUFFICIENT"
    if _bankroll > 0:
        _session_budget = session_budget(_bankroll)
        logger.info(f"  💰 Bankroll: ${_bankroll:,.0f} | Session budget (4%): ${_session_budget:,.0f}")
    try:
        _calibracion = json.loads(Path("data/calibracion_edge.json").read_text(encoding="utf-8"))
        _meta_regime, _meta_factor = session_regime(_calibracion)
        if _meta_regime != "INSUFFICIENT":
            logger.info(f"  🧠 Meta-Markov: {_meta_regime} → factor={_meta_factor:.2f}")
            if _meta_factor < 1.0:
                logger.warning(f"  ⚠️ Stakes reducidos ×{_meta_factor:.2f} por régimen del modelo ({_meta_regime})")
    except Exception:
        pass

    # ── MODO LIVE: re-armar combos con jugadores disponibles AHORA ──
    if args.live:
        combo_links, metadata = build_live_combos(
            piernas_min=args.piernas_min,
            piernas_max=args.piernas_max,
            top_n=args.top_n,
            min_cuota=args.min_cuota,
            edge_file=args.edge_file,
            strategy=args.strategy,
        )
        if not combo_links:
            sys.exit(1)

        mostrar_combos(combo_links, metadata)

        if args.dry_run:
            return

        if args.console:
            mostrar_consola(combo_links)
            return

        n = generar_bat_chrome(combo_links)
        if n:
            print(f"\n  🎯 {n} archivos Combo*.bat en tu escritorio")
            print("  Flujo: borra ticket (X) → doble clic .bat → Chrome → stake → apostar")

        if args.telegram:
            ok = enviar_combos_telegram(combo_links, metadata)
            if ok:
                logger.info("📱 Resumen enviado a Telegram ✅")

        # ── MEGA-COMBOS: añadir cross-tier si --mega ──
        if args.mega:
            logger.info("")
            logger.info("🚀 Generando MEGA-COMBOS cross-tier (Nodo-23)...")
            mega_links, mega_meta = build_mega_combos(
                stake_per_combo=args.mega_stake,
                piernas_min=args.mega_min,
                piernas_max=args.mega_max,
                no_bbi_filter=args.no_bbi_filter,
            )
            if mega_links:
                # Nodo-26 M-26-2: Circuit Breaker — recortar si excede budget
                if _bankroll > 0:
                    _eff_stake = int(args.mega_stake * _meta_factor)
                    n_allowed, budget_msg = check_budget(len(mega_links), _eff_stake, _bankroll)
                    if budget_msg != "OK":
                        logger.warning(f"  ⚠️ {budget_msg}")
                        mega_links = mega_links[:n_allowed]
                    if _meta_factor < 1.0:
                        for ml in mega_links:
                            ml["stake"] = _eff_stake
                _mostrar_mega_combos(mega_links, mega_meta)
                if not args.dry_run:
                    n_mega = _generar_bat_mega(mega_links)
                    if n_mega:
                        print(f"\n  🚀 {n_mega} archivos Mega*.bat en tu escritorio")
                        print(f"  Stake: ${args.mega_stake} × {n_mega} = ${args.mega_stake * n_mega:,}")
                    if args.telegram:
                        _enviar_mega_telegram(mega_links, mega_meta)

        # ── SAFE COMBOS: añadir Beta Book si --safe ──
        if args.safe:
            logger.info("")
            logger.info("🛡️ Generando SAFE COMBOS — Beta Book (Nodo-25)...")
            safe_links, safe_meta = build_safe_combos(
                stake_per_combo=args.safe_stake,
                top_n=args.safe_top_n,
            )
            if safe_links:
                _mostrar_safe_combos(safe_links, safe_meta)
                if not args.dry_run:
                    n_safe = _generar_bat_safe(safe_links)
                    if n_safe:
                        print(f"\n  🛡️ {n_safe} archivos Safe*.bat en tu escritorio")
                        print(f"  Stake: ${args.safe_stake} × {n_safe} = ${args.safe_stake * n_safe:,}")
                    if args.telegram:
                        _enviar_safe_telegram(safe_links, safe_meta)

        # ── GAMES COMBOS: añadir Totales si --games ──
        if args.games:
            logger.info("")
            logger.info("Generando GAMES COMBOS — Totales (Nodo-40)...")
            games_links, games_meta = build_games_combos(
                stake_per_combo=args.games_stake,
                games_file=args.games_file,
            )
            if games_links:
                _mostrar_games_combos(games_links, games_meta)
                if not args.dry_run:
                    n_games = _generar_bat_games(games_links)
                    if n_games:
                        print(f"\n  {n_games} archivos Games*.bat en tu escritorio")
                        print(f"  Stake: ${args.games_stake} × {n_games} = ${args.games_stake * n_games:,}")
                    if args.telegram:
                        _enviar_games_telegram(games_links, games_meta)
        return

    # ── MODO MEGA STANDALONE (sin --live) ──
    if args.mega:
        logger.info("🚀 Generando MEGA-COMBOS cross-tier (Nodo-23)...")
        mega_links, mega_meta = build_mega_combos(
            stake_per_combo=args.mega_stake,
            piernas_min=args.mega_min,
            piernas_max=args.mega_max,
            no_bbi_filter=args.no_bbi_filter,
        )
        if not mega_links:
            logger.error("❌ No se pudieron generar mega-combos")
            sys.exit(1)

        # Nodo-26 M-26-2/4: Circuit Breaker + Meta-Markov stake adjustment
        if _bankroll > 0:
            _eff_stake = int(args.mega_stake * _meta_factor)
            n_allowed, budget_msg = check_budget(len(mega_links), _eff_stake, _bankroll)
            if budget_msg != "OK":
                logger.warning(f"  ⚠️ {budget_msg}")
                mega_links = mega_links[:n_allowed]
            if _meta_factor < 1.0:
                for ml in mega_links:
                    ml["stake"] = _eff_stake

        _mostrar_mega_combos(mega_links, mega_meta)

        if args.dry_run:
            return

        n_mega = _generar_bat_mega(mega_links)
        if n_mega:
            print(f"\n  🚀 {n_mega} archivos Mega*.bat en tu escritorio")
            print(f"  Stake: ${args.mega_stake} × {n_mega} = ${args.mega_stake * n_mega:,}")

        if args.telegram:
            _enviar_mega_telegram(mega_links, mega_meta)

        # Safe combos alongside mega
        if args.safe:
            logger.info("")
            logger.info("🛡️ Generando SAFE COMBOS — Beta Book (Nodo-25)...")
            safe_links, safe_meta = build_safe_combos(
                stake_per_combo=args.safe_stake,
                top_n=args.safe_top_n,
            )
            if safe_links:
                _mostrar_safe_combos(safe_links, safe_meta)
                if not args.dry_run:
                    n_safe = _generar_bat_safe(safe_links)
                    if n_safe:
                        print(f"\n  🛡️ {n_safe} archivos Safe*.bat en tu escritorio")
                    if args.telegram:
                        _enviar_safe_telegram(safe_links, safe_meta)
        return

    # ── MODO SAFE STANDALONE ──
    if args.safe:
        logger.info("🛡️ Generando SAFE COMBOS — Beta Book (Nodo-25)...")
        safe_links, safe_meta = build_safe_combos(
            stake_per_combo=args.safe_stake,
            top_n=args.safe_top_n,
        )
        if not safe_links:
            logger.error("❌ No se pudieron generar safe combos")
            sys.exit(1)

        _mostrar_safe_combos(safe_links, safe_meta)

        if args.dry_run:
            return

        n_safe = _generar_bat_safe(safe_links)
        if n_safe:
            print(f"\n  🛡️ {n_safe} archivos Safe*.bat en tu escritorio")
            print(f"  Stake: ${args.safe_stake} × {n_safe} = ${args.safe_stake * n_safe:,}")

        if args.telegram:
            _enviar_safe_telegram(safe_links, safe_meta)
        return

    # ── MODO GAMES STANDALONE ──
    if args.games:
        logger.info("Generando GAMES COMBOS — Totales (Nodo-40)...")
        games_links, games_meta = build_games_combos(
            stake_per_combo=args.games_stake,
            games_file=args.games_file,
        )
        if not games_links:
            logger.error("❌ No se pudieron generar games combos. Verifica games_signal_calculator.py")
            sys.exit(1)

        _mostrar_games_combos(games_links, games_meta)

        if args.dry_run:
            return

        n_games = _generar_bat_games(games_links)
        if n_games:
            print(f"\n  {n_games} archivos Games*.bat en tu escritorio")
            print(f"  Stake: ${args.games_stake} × {n_games} = ${args.games_stake * n_games:,}")

        if args.telegram:
            _enviar_games_telegram(games_links, games_meta)
        return

    # ── MODO WAS STANDALONE (Nodo-44) ──
    if args.was:
        logger.info("Generando WAS COMBOS — Watchlist Alpha Signal (Nodo-44)...")
        was_links, was_meta = build_was_combos(
            stake_per_combo=args.was_stake,
            min_edge=args.was_min_edge,
            top_n=args.was_top_n,
            edge_file=args.edge_file,
        )
        if not was_links:
            logger.error("❌ No se pudieron generar WAS combos. "
                         "Verifica edge_report watchlist con edge>=10% + cuota>=2.0 + señal Markov")
            sys.exit(1)

        _mostrar_was_combos(was_links, was_meta)

        if args.dry_run:
            return

        n_was = _generar_bat_was(was_links)
        if n_was:
            print(f"\n  {n_was} archivos WAS*.bat en tu escritorio")
            print(f"  Stake: ${args.was_stake} × {n_was} = ${args.was_stake * n_was:,}")
            print("  REGLA-WAS-1: solo para promos stake minimo — no escalar a Kelly")
        return

    # ── MODO CLÁSICO: mapear combos del trader_plan ──
    plan_file = args.file or find_latest_trader_plan()
    if not plan_file or not Path(plan_file).exists():
        logger.error("❌ No se encontró trader_plan. Ejecuta primero: python trader_ev_tenis.py --bankroll N")
        sys.exit(1)

    logger.info(f"📄 Leyendo: {plan_file}")
    trader_plan = json.load(open(plan_file, encoding="utf-8"))

    combos = trader_plan.get("cobertura", [])
    if not combos:
        logger.error("❌ El trader_plan no tiene combos de cobertura")
        sys.exit(1)

    logger.info(f"   {len(combos)} combos encontrados")

    # Mapear combos → outcome IDs
    combo_links = build_combo_links(trader_plan)

    # Mostrar en consola
    mostrar_combos(combo_links, trader_plan.get("metadata", {}))

    if args.dry_run:
        logger.info("🔍 Dry run — no se genera nada")
        return

    # Modo consola F12
    if args.console:
        mostrar_consola(combo_links)
        return

    # WhatsApp mode
    if args.whatsapp:
        path = generar_whatsapp_html(combo_links, trader_plan.get("metadata", {}))
        if path:
            print(f"\n  📱 WhatsApp_Combos.bat en tu escritorio")
            print("  Doble clic → Chrome abre la página → clic cada botón → WhatsApp → Enviar")
            print("  Desde el celular: tap link → Chrome → Betplay con combo cargado")
        return

    # Generar .bat para Chrome (default)
    n = generar_bat_chrome(combo_links)
    if n:
        print(f"\n  🎯 {n} archivos Combo*.bat en tu escritorio")
        print("  Flujo: borra ticket (X) → doble clic .bat → Chrome → stake → apostar")

    # Telegram
    if args.telegram:
        ok = enviar_combos_telegram(combo_links, trader_plan.get("metadata", {}))
        if ok:
            logger.info("📱 Resumen enviado a Telegram ✅")


if __name__ == "__main__":
    main()
