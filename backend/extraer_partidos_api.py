"""
🎾 PASO 1 API — Extrae partidos de tenis con cuotas reales de Betplay

Reemplaza extraer_URL_partidos_version2.py (Playwright, 8 min, DOM frágil)
con dos API calls HTTP (~2 segundos total):

  1. Kambi API (Betplay)    → jugadores + cuotas REALES donde apostamos
  2. FlashScore feed API    → match_ids (para H2H) + rankings + superficie

Produce: data/zita_tennis_matches_FECHA.json — mismo formato del pipeline.

Uso:
  python3 extraer_partidos_api.py                    # partidos de hoy
  python3 extraer_partidos_api.py --tomorrow          # partidos de mañana
  python3 extraer_partidos_api.py --tier atp wta      # solo ATP + WTA
  python3 extraer_partidos_api.py --tomorrow --tier atp wta challenger
"""

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="PASO 1 API — Partidos de tenis con cuotas reales Betplay"
    )
    parser.add_argument(
        "--tomorrow", action="store_true",
        help="Extraer partidos de mañana (default: hoy)"
    )
    parser.add_argument(
        "--tier", nargs="+", default=None,
        choices=["atp", "wta", "challenger", "wta125", "itf"],
        help="Filtrar por tier (default: todos los singles)"
    )
    args = parser.parse_args()

    day_offset = 1 if args.tomorrow else 0
    dia = "MAÑANA" if args.tomorrow else "HOY"

    logger.info("🎾 PASO 1 API — Partidos + Cuotas Reales Betplay")
    logger.info("=" * 70)
    logger.info(f"📅 Día: {dia} (offset={day_offset})")
    if args.tier:
        logger.info(f"🔍 Tiers: {args.tier}")
    logger.info("=" * 70)

    start = time.time()

    from scraping.kambi_tennis import extract_matches

    filename, matches = extract_matches(
        day_offset=day_offset,
        tiers=args.tier
    )

    elapsed = time.time() - start

    if not matches:
        logger.error("❌ No se encontraron partidos")
        sys.exit(1)

    # Resumen
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"🏁 COMPLETADO en {elapsed:.1f}s")
    logger.info(f"📁 Archivo: {filename}")
    logger.info("")

    # Estadísticas por tier
    by_tier = {}
    with_match_id = 0
    for m in matches:
        t = m.get("tier", "unknown")
        if t not in by_tier:
            by_tier[t] = 0
        by_tier[t] += 1
        if m.get("match_id"):
            with_match_id += 1

    logger.info("📊 RESUMEN:")
    for tier in sorted(by_tier.keys()):
        logger.info(f"   {tier:12s}: {by_tier[tier]:3d} partidos")
    logger.info(f"   {'TOTAL':12s}: {len(matches):3d} partidos")
    logger.info(f"   Con match_id: {with_match_id}/{len(matches)} (para H2H API)")
    logger.info(f"   Cuota real:   {len(matches)}/{len(matches)} (Betplay Kambi)")

    # Mostrar primeros partidos ATP/WTA
    logger.info("")
    logger.info("🎾 PARTIDOS DESTACADOS:")
    shown = 0
    for m in matches:
        if m.get("tier") in ("atp", "wta") and shown < 10:
            c1 = m.get("cuota1", 0)
            c2 = m.get("cuota2", 0)
            mid = "✅" if m.get("match_id") else "⚠️"
            logger.info(
                f"   {mid} {m['jugador1']:25s} @{c1:5.2f} vs {m['jugador2']:25s} @{c2:5.2f} | {m.get('torneo_nombre', '?')}"
            )
            shown += 1

    logger.info("")
    logger.info("👉 Siguiente: python3 extraer_historh2h.py --api-mode --all-tournaments")


if __name__ == "__main__":
    main()
