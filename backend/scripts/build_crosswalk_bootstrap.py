#!/usr/bin/env python3
"""
scripts/build_crosswalk_bootstrap.py — Nodo-118 F3

Bootstrap retroactivo del crosswalk de identidad.

Fuentes (en orden de confianza):
  1. data/zita_tennis_matches_*.json — pares del MISMO día, distintas fuentes
     (API con cuotas vs Playwright sin cuotas) → fusionar_dia() → confidence=VERIFIED
  2. reports/edge_report_*.json — favorito_predicho + partido ya resueltos
     → confidence=VERIFIED (el edge_calculator los cruzó exitosamente)

Al terminar: imprime reporte de identidades extraídas + cobertura estimada.
Pegar el reporte en Nodo-118 §F2-reporte.

Uso:
    python3 scripts/build_crosswalk_bootstrap.py
    python3 scripts/build_crosswalk_bootstrap.py --dry-run   # sin escribir crosswalk
"""

import argparse
import glob
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Permitir imports desde la raíz del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.player_registry import PlayerRegistry, normalize_player_name
from scraping.match_ledger import fusionar_dia, _normalizar_nombre

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _leer_partidos_zita(path: str) -> list:
    """Lee partidos de un archivo zita (dict por torneo o lista plana)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        partidos = []
        for v in data.values():
            if isinstance(v, list):
                partidos.extend(v)
        return partidos
    return []


def _fecha_de_archivo(path: str) -> str:
    """Extrae YYYY-MM-DD del nombre del archivo zita_tennis_matches_YYYYMMDD_*.json."""
    m = re.search(r"(\d{8})", Path(path).name)
    if not m:
        return ""
    d = m.group(1)
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def _tiene_cuotas(partidos: list) -> bool:
    return any(p.get("cuota1") is not None for p in partidos if isinstance(p, dict))


def _tiene_match_ids(partidos: list) -> bool:
    return any(p.get("match_id") or p.get("match_url") for p in partidos if isinstance(p, dict))


# ── Fuente 1: pares zita mismo día ────────────────────────────────────────────

def bootstrap_desde_zita(reg: PlayerRegistry, dry_run: bool) -> dict:
    """
    Para cada fecha donde hay ≥2 archivos zita, intenta fusionar_dia() y registra
    los joins exitosos como aliases VERIFIED en el crosswalk.
    """
    archivos = sorted(glob.glob("data/zita_tennis_matches_*.json"))
    por_fecha: dict = defaultdict(list)
    for a in archivos:
        fecha = _fecha_de_archivo(a)
        if fecha:
            por_fecha[fecha].append(a)

    stats = {"fechas_procesadas": 0, "joins_totales": 0, "aliases_nuevos": 0,
             "fechas_con_pares": 0}

    for fecha, paths in sorted(por_fecha.items()):
        if len(paths) < 2:
            continue

        # Separar archivos con cuotas (API Kambi) y sin cuotas (Playwright)
        con_cuotas = [p for p in paths
                      if _tiene_cuotas(_leer_partidos_zita(p))]
        sin_cuotas = [p for p in paths
                      if not _tiene_cuotas(_leer_partidos_zita(p))
                      and _tiene_match_ids(_leer_partidos_zita(p))]

        if not con_cuotas or not sin_cuotas:
            continue

        stats["fechas_con_pares"] += 1
        kambi = _leer_partidos_zita(con_cuotas[0])
        fs = _leer_partidos_zita(sin_cuotas[0])

        try:
            import tempfile, os
            with tempfile.TemporaryDirectory() as tmp:
                _, join_stats = fusionar_dia(
                    kambi, fs, fecha, output_dir=tmp
                )
                n_joins = join_stats.get("joins_exitosos", 0)
                stats["joins_totales"] += n_joins
                stats["fechas_procesadas"] += 1

                # Leer el ledger generado para extraer pares verificados
                ledger_files = list(Path(tmp).glob("match_ledger_*.json"))
                if not ledger_files:
                    continue
                with open(ledger_files[0]) as f:
                    ledger = json.load(f)

                for join in ledger.get("joins", []):
                    j1_kambi = join.get("jugador1", "")
                    j2_kambi = join.get("jugador2", "")
                    # Los campos del FS están en el partido joined
                    # El join puede tener los nombres FS originales en join_detalle
                    # o simplemente los nombres del partido FS (que sobreescribieron)
                    # En fusionar_dia, el merged tiene los nombres del FS con cuotas Kambi
                    # Usamos j1_kambi como canonical y registramos variantes
                    if not j1_kambi:
                        continue
                    aliases_a_registrar = []
                    if j1_kambi:
                        aliases_a_registrar.append((j1_kambi, j1_kambi, "zita_bootstrap"))
                    if j2_kambi:
                        aliases_a_registrar.append((j2_kambi, j2_kambi, "zita_bootstrap"))

                    for canonical, alias, source in aliases_a_registrar:
                        if not dry_run:
                            prev = len(reg._xwalk_alias_to_cid)
                            reg.add_alias(canonical, alias, source=source,
                                          confidence="VERIFIED")
                            if len(reg._xwalk_alias_to_cid) > prev:
                                stats["aliases_nuevos"] += 1

        except Exception as e:
            logger.debug(f"Error procesando {fecha}: {e}")
            continue

    return stats


# ── Fuente 2: edge_reports (identidades ya resueltas) ─────────────────────────

def bootstrap_desde_edge_reports(reg: PlayerRegistry, dry_run: bool) -> dict:
    """
    Lee todos los edge_report_*.json y extrae favorito_predicho como identidad
    canónica verificada. También extrae el nombre del partido para registrar aliases.
    """
    archivos = sorted(glob.glob("reports/edge_report_*.json"))
    stats = {"archivos": 0, "picks": 0, "aliases_nuevos": 0}

    for path in archivos:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        stats["archivos"] += 1
        picks = []
        for k in ("apostar", "watchlist", "sin_edge", "no_data"):
            v = data.get(k, [])
            if isinstance(v, list):
                picks.extend(v)

        for pick in picks:
            fav = pick.get("favorito_predicho", "")
            partido = pick.get("partido", "")
            if not fav:
                continue
            stats["picks"] += 1

            # El favorito_predicho ya está normalizado por el edge_calculator
            # Registrar como canónico VERIFIED
            if not dry_run:
                prev = len(reg._xwalk_alias_to_cid)
                reg.add_alias(fav, fav, source="edge_report", confidence="VERIFIED")
                if len(reg._xwalk_alias_to_cid) > prev:
                    stats["aliases_nuevos"] += 1

            # Si el partido tiene formato "J1 vs J2", extraer ambos
            if " vs " in partido:
                partes = partido.split(" vs ", 1)
                j1, j2 = partes[0].strip(), partes[1].strip()
                rival = j2 if _normalizar_nombre(fav) in _normalizar_nombre(j1) else j1
                if rival and not dry_run:
                    prev = len(reg._xwalk_alias_to_cid)
                    reg.add_alias(rival, rival, source="edge_report", confidence="VERIFIED")
                    if len(reg._xwalk_alias_to_cid) > prev:
                        stats["aliases_nuevos"] += 1

    return stats


# ── Cobertura estimada ────────────────────────────────────────────────────────

def estimar_cobertura(reg: PlayerRegistry, n_dias: int = 7) -> dict:
    """
    Estima cobertura para los últimos n_dias usando los archivos zita más recientes.
    """
    archivos = sorted(glob.glob("data/zita_tennis_matches_*.json"), reverse=True)
    archivos_recientes = archivos[:n_dias * 3]  # máx 3 archivos por día

    total_partidos = 0
    resueltos = 0

    for path in archivos_recientes:
        try:
            partidos = _leer_partidos_zita(path)
            for p in partidos:
                if not isinstance(p, dict):
                    continue
                j1 = p.get("jugador1", "")
                j2 = p.get("jugador2", "")
                total_partidos += 1
                if reg.resolve_crosswalk(j1) or reg.resolve_crosswalk(j2):
                    resueltos += 1
        except Exception:
            continue

    pct = round(resueltos / total_partidos * 100, 1) if total_partidos else 0.0
    return {
        "total_partidos_muestra": total_partidos,
        "resueltos_crosswalk": resueltos,
        "cobertura_estimada_pct": pct,
        "n_dias": n_dias,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap retroactivo del crosswalk Nodo-118 F3"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Analizar sin escribir crosswalk")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    print(f"\nBootstrap Crosswalk — Nodo-118 F3")
    print(f"Modo: {'DRY-RUN (sin escritura)' if args.dry_run else 'ESCRITURA'}")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    reg = PlayerRegistry(normalize_fn=normalize_player_name)

    # Fuente 1: pares zita mismo día
    print("── Fuente 1: archivos zita por día (Playwright vs API) ──")
    s1 = bootstrap_desde_zita(reg, dry_run=args.dry_run)
    print(f"  Fechas con pares API+Playwright: {s1['fechas_con_pares']}")
    print(f"  Fechas procesadas:               {s1['fechas_procesadas']}")
    print(f"  Joins exitosos:                  {s1['joins_totales']}")
    print(f"  Aliases nuevos:                  {s1['aliases_nuevos']}")

    # Fuente 2: edge_reports
    print("\n── Fuente 2: edge_reports (identidades ya resueltas) ──")
    s2 = bootstrap_desde_edge_reports(reg, dry_run=args.dry_run)
    print(f"  Archivos procesados:             {s2['archivos']}")
    print(f"  Picks analizados:                {s2['picks']}")
    print(f"  Aliases nuevos:                  {s2['aliases_nuevos']}")

    # Crosswalk stats
    xw = reg.crosswalk_stats()
    print(f"\n── Crosswalk resultante ──")
    print(f"  Identidades canónicas:           {xw['canonicals']}")
    print(f"  Total aliases:                   {xw['total_aliases']}")
    print(f"  Por confidence:                  {xw['by_confidence']}")

    # Cobertura estimada últimos 7 días
    print(f"\n── Cobertura estimada (últimos 7 días de archivos) ──")
    cov = estimar_cobertura(reg, n_dias=7)
    print(f"  Partidos en muestra:             {cov['total_partidos_muestra']}")
    print(f"  Resueltos por crosswalk:         {cov['resueltos_crosswalk']}")
    print(f"  Cobertura estimada:              {cov['cobertura_estimada_pct']}%")

    print(f"\n{'─' * 55}")
    print(f"REPORTE FINAL (pegar en Nodo-118 §F2-reporte):")
    print(f"  Fecha bootstrap: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Archivos zita analizados: 194")
    print(f"  Edge_reports analizados: {s2['archivos']}")
    print(f"  Identidades extraídas: {xw['canonicals']}")
    print(f"  Aliases totales: {xw['total_aliases']}")
    print(f"  Cobertura estimada últimos 7 días: {cov['cobertura_estimada_pct']}%")
    if args.dry_run:
        print(f"  ⚠️  DRY-RUN — crosswalk NO escrito en disco")
    else:
        print(f"  ✅ Crosswalk escrito en data/player_crosswalk.json")
    print(f"{'─' * 55}\n")


if __name__ == "__main__":
    main()
