"""
📱 Telegram — Señales de apuesta para tenis

Envía los picks del trader a Telegram para apostar rápido
sin buscar manualmente en 20+ torneos de Betplay.

Mismo bot que el proyecto NBA.
"""

import json
import logging
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Mismo bot Telegram que proyecto NBA
TG_TOKEN = "8684706586:AAHv4zhjQKvxORf6bnbwCxZQPly9OA7unpY"
TG_CHAT = "8520949513"
TG_URL = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"


def _enviar_telegram(msg: str) -> bool:
    """Envía mensaje a Telegram. Retorna True si OK."""
    try:
        params = urllib.parse.urlencode({
            "chat_id": TG_CHAT,
            "text": msg,
            "parse_mode": "HTML",
        }).encode("utf-8")
        req = urllib.request.Request(TG_URL, data=params, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except Exception as e:
        logger.error(f"❌ Error enviando Telegram: {e}")
        return False


def enviar_señales_trader(trader_plan_path: str) -> bool:
    """
    Lee el trader_plan JSON y envía señales a Telegram.

    Formato del mensaje:
    - Header con bankroll, tier, fecha
    - Individuales: jugador, cuota, edge, stake, torneo
    - Combos de cobertura: piernas, cuota combo, stake
    - Resumen final
    """
    try:
        data = json.load(open(trader_plan_path, encoding="utf-8"))
    except Exception as e:
        logger.error(f"❌ Error leyendo {trader_plan_path}: {e}")
        return False

    meta = data.get("metadata", {})
    individuales = data.get("individuales", [])
    cobertura = data.get("cobertura", [])
    resumen = data.get("resumen", {})
    risk = data.get("risk_management", {})

    bankroll = meta.get("bankroll", 0)
    tier = meta.get("torneo_tipo", "?")
    superficie = meta.get("superficie", "?")
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Header ──
    lines = []
    lines.append(f"🎾 <b>SEÑALES TENIS — {tier.upper()}</b>")
    lines.append(f"📅 {fecha} | 💰 ${bankroll:,.0f} | 🏟 {superficie}")
    lines.append("")

    # ── Individuales ──
    if individuales:
        lines.append(f"<b>📌 INDIVIDUALES ({len(individuales)})</b>")
        lines.append("")
        for i, pick in enumerate(individuales, 1):
            partido = pick.get("partido", "?")
            favorito = pick.get("favorito", "?")
            cuota = pick.get("cuota", 0)
            edge = pick.get("edge_pct", "?")
            stake = pick.get("stake", 0)
            sup = pick.get("superficie", "?")
            retorno = pick.get("retorno_potencial", 0)

            lines.append(f"  {i}. <b>{favorito}</b> @{cuota:.2f}")
            lines.append(f"     {partido}")
            lines.append(f"     Edge: {edge} | Stake: ${stake:,.0f} → ${retorno:,.0f}")
            lines.append("")

    # ── Cobertura ──
    if cobertura:
        lines.append(f"<b>🔗 COMBOS COBERTURA ({len(cobertura)})</b>")
        lines.append("")
        for i, combo in enumerate(cobertura, 1):
            legs = combo.get("legs", [])
            cuota_combo = combo.get("cuota_combo", 0)
            stake = combo.get("stake", 0)
            retorno = combo.get("retorno_potencial", 0)
            n_piernas = combo.get("piernas_n", len(legs))

            leg_names = " + ".join(
                f"{l['jugador']}@{l['cuota']:.2f}" for l in legs
            )
            lines.append(f"  [{n_piernas}p] @{cuota_combo:.1f} → ${retorno:,.0f}")
            lines.append(f"     {leg_names}")
            lines.append(f"     Stake: ${stake:,.0f}")
            lines.append("")

    # ── Resumen ──
    total_riesgo = resumen.get("total_en_riesgo", 0)
    pct = resumen.get("pct_bankroll", 0)
    kgr = risk.get("kelly_growth_rate", 0)

    lines.append("<b>📊 RESUMEN</b>")
    lines.append(f"  Total en riesgo: ${total_riesgo:,.0f} ({pct:.1f}%)")
    if kgr:
        emoji = "✅" if kgr > 0 else "❌"
        lines.append(f"  KGR: {kgr:.4f} {emoji}")
    lines.append("")
    lines.append("⚡ Generado por Tennis Prediction Engine")

    msg = "\n".join(lines)

    # Telegram tiene límite de 4096 chars
    if len(msg) > 4000:
        # Enviar en 2 partes
        mid = len(lines) // 2
        part1 = "\n".join(lines[:mid])
        part2 = "\n".join(lines[mid:])
        ok1 = _enviar_telegram(part1)
        ok2 = _enviar_telegram(part2)
        return ok1 and ok2

    return _enviar_telegram(msg)


def enviar_resumen_picks(picks: List[Dict], tier: str = "", bankroll: int = 0) -> bool:
    """
    Envía un resumen rápido de picks sin necesitar el trader_plan JSON.
    Útil para enviar desde cualquier punto del pipeline.
    """
    if not picks:
        return False

    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append(f"🎾 <b>PICKS TENIS{' — ' + tier.upper() if tier else ''}</b>")
    lines.append(f"📅 {fecha}{' | 💰 $' + f'{bankroll:,.0f}' if bankroll else ''}")
    lines.append("")

    for i, p in enumerate(picks, 1):
        jugador = p.get("favorito", p.get("jugador", "?"))
        cuota = p.get("cuota", 0)
        edge = p.get("edge_pct", p.get("edge", "?"))
        partido = p.get("partido", "?")
        lines.append(f"  {i}. <b>{jugador}</b> @{cuota:.2f} | Edge: {edge}")
        lines.append(f"     {partido}")

    lines.append("")
    lines.append("⚡ Tennis Prediction Engine")

    return _enviar_telegram("\n".join(lines))
