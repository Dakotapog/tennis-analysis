"""
D144-06 (Nodo-144): Backfill retroactivo de campo 'strategy' en shadow_book.

ONE-SHOT CUMPLIDO (Nodo-174 D174-12, 2026-08-06): ejecutado una vez sobre los
3 días con combo_registry disponible (22/23/25-jul-2026). No es un PASO de
run_daily.py ni corre periódicamente — reejecutarlo sobre el mismo rango es
un no-op (registros ya tageados no se reprocesan). Si aparece nuevo backlog
de registros SIN_TAG con combo_registry disponible, correr manualmente con
el rango de fechas correspondiente.

Lógica:
  - 3 días CON combo_registry (22/23/25-jul-2026): cruce exacto nombre+fecha+subtipo
    → alta confianza → tag CORE/SATELITE/COBERTURA según subtipo
  - Todos los demás registros SIN_TAG → HISTORICO_SIN_TAG (no reconstruible)

Regla dura: si no hay match exacto, NO asignar estrategia adivinada.
Jamás contaminar métricas con datos inferidos sin evidencia.

Uso:
  python scripts/backfill_strategy.py [--dry-run]
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_SCRIPT_DIR)
_COMBO_REG_DIR = os.path.join(_BACKEND_DIR, "reports", "combo_registry")
_SHADOW_DIR = os.path.join(_BACKEND_DIR, "reports", "shadow_book")

# ---------------------------------------------------------------------------
# Subtipo → strategy mapping (solo subtipo con player names en piernas)
# ---------------------------------------------------------------------------
_SUBTIPO_TO_STRATEGY = {
    "CORE":      "CORE",
    "SATELLITE": "SATELITE",   # combo_registry usa SATELLITE; spec usa SATELITE
    "COBERTURA": "COBERTURA",
    "MOONSHOT":  "MOONSHOT",
    "ANCHOR":    "ANCHOR",
    "GCS":       "GCS",
}
# Subtipos a ignorar (piernas no son player names)
_SKIP_SUBTIPO = {"GAMES_B", "GAMES_A", "MEGA", "STANDARD", "WAS", "SAFE"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl_path(path: str) -> dict:
    """Lee JSONL → dict sb_id/cr_id → record."""
    records = {}
    if not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = rec.get("sb_id") or rec.get("cr_id")
                if key:
                    records[key] = rec
            except json.JSONDecodeError:
                continue
    return records


def _save_jsonl_path(path: str, records: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _normalize(name: str) -> str:
    return name.strip().lower()


def _extract_nombre(rec: dict) -> str:
    snap = rec.get("pick_snapshot", {})
    for field in ("favorito_predicho", "nombre", "jugador", "player"):
        v = snap.get(field)
        if v:
            return v.strip().lower()
    return ""


# ---------------------------------------------------------------------------
# Phase 1: build tag map from combo_registry
# ---------------------------------------------------------------------------

def build_tag_map() -> dict:
    """
    Retorna dict: fecha → set of (nombre_lower, strategy).
    Solo subtipos con player names.
    Deduplica: si el mismo jugador aparece en 2 combos del mismo día (CORE+COBERTURA),
    la primera asignación gana (orden cr_id alphabetical dentro del día).
    """
    tag_map: dict = defaultdict(dict)   # fecha → {nombre_lower: strategy}
    cr_files = sorted(glob.glob(os.path.join(_COMBO_REG_DIR, "cr_*.jsonl")))

    print(f"[backfill] combo_registry: {len(cr_files)} archivo(s) encontrado(s)")

    for cr_path in cr_files:
        fecha = os.path.basename(cr_path).replace("cr_", "").replace(".jsonl", "")
        seen_cr_ids = set()
        with open(cr_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                cr_id = rec.get("cr_id", "")
                subtipo = rec.get("subtipo", "")
                piernas = rec.get("piernas", [])

                if subtipo in _SKIP_SUBTIPO:
                    continue
                strategy = _SUBTIPO_TO_STRATEGY.get(subtipo)
                if not strategy:
                    continue
                if cr_id in seen_cr_ids:
                    continue
                seen_cr_ids.add(cr_id)

                for pierna in piernas:
                    nombre = _normalize(pierna)
                    if not nombre:
                        continue
                    # No sobrescribir: primera asignación gana dentro del mismo día
                    if nombre not in tag_map[fecha]:
                        tag_map[fecha][nombre] = strategy

    for fecha, mapa in tag_map.items():
        print(f"  {fecha}: {len(mapa)} jugadores con tag en combo_registry "
              f"({', '.join(sorted(set(mapa.values())))})")

    return dict(tag_map)


# ---------------------------------------------------------------------------
# Phase 2: apply tags to shadow book
# ---------------------------------------------------------------------------

def apply_tags(tag_map: dict, dry_run: bool) -> dict:
    """
    Para cada sb_*.jsonl:
    - Si la fecha está en tag_map: match nombre → strategy
    - Todos los SIN_TAG restantes → HISTORICO_SIN_TAG
    Retorna stats dict.
    """
    stats = {
        "archivos_procesados": 0,
        "ya_tageados_skip": 0,
        "tageados_alta_confianza": 0,
        "marcados_historico_sin_tag": 0,
        "sin_nombre_skip": 0,
    }
    tagged_at = datetime.now().isoformat()
    sb_files = sorted(glob.glob(os.path.join(_SHADOW_DIR, "sb_*.jsonl")))

    print(f"\n[backfill] shadow_book: {len(sb_files)} archivo(s) a procesar")

    for sb_path in sb_files:
        fecha = os.path.basename(sb_path).replace("sb_", "").replace(".jsonl", "")
        records = _load_jsonl_path(sb_path)
        if not records:
            continue

        fecha_tag_map = tag_map.get(fecha, {})  # puede ser {} si no hay combo_registry
        modificado = False

        for sb_id, rec in records.items():
            if rec.get("_type") == "session_meta":
                continue

            current_strategy = rec.get("strategy", "SIN_TAG")

            # Ya tiene tag real → skip
            if current_strategy not in ("SIN_TAG", "HISTORICO_SIN_TAG", None, ""):
                stats["ya_tageados_skip"] += 1
                continue

            nombre = _extract_nombre(rec)

            # Intentar match contra combo_registry de esta fecha
            if nombre and nombre in fecha_tag_map:
                new_strategy = fecha_tag_map[nombre]
                if not dry_run:
                    rec["strategy"] = new_strategy
                    rec["strategy_tagged_at"] = tagged_at
                    rec["strategy_source"] = "backfill_combo_registry"
                stats["tageados_alta_confianza"] += 1
                modificado = True
                continue

            # Sin match → HISTORICO_SIN_TAG
            if current_strategy not in ("HISTORICO_SIN_TAG",):
                if not dry_run:
                    rec["strategy"] = "HISTORICO_SIN_TAG"
                    rec["strategy_tagged_at"] = tagged_at
                    rec["strategy_source"] = "backfill_no_evidence"
                stats["marcados_historico_sin_tag"] += 1
                modificado = True

        stats["archivos_procesados"] += 1
        if modificado and not dry_run:
            _save_jsonl_path(sb_path, records)
            print(f"  {sb_path.split('/')[-1]}: guardado")
        elif modificado:
            print(f"  {sb_path.split('/')[-1]}: [DRY-RUN] se modificaría")

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="D144-06: Backfill strategy en shadow_book")
    parser.add_argument("--dry-run", action="store_true",
                        help="Muestra qué se haría sin modificar archivos")
    args = parser.parse_args()

    print("=" * 60)
    print("D144-06 Nodo-144 — Backfill strategy en shadow_book")
    print(f"Modo: {'DRY-RUN' if args.dry_run else 'ESCRITURA REAL'}")
    print("=" * 60)

    # Cambiar al directorio backend para rutas relativas
    os.chdir(_BACKEND_DIR)

    tag_map = build_tag_map()
    stats = apply_tags(tag_map, dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"  Archivos procesados       : {stats['archivos_procesados']}")
    print(f"  Ya tageados (skip)        : {stats['ya_tageados_skip']}")
    print(f"  Tageados (alta confianza) : {stats['tageados_alta_confianza']}")
    print(f"  HISTORICO_SIN_TAG         : {stats['marcados_historico_sin_tag']}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY-RUN] Ningún archivo modificado.")
    else:
        print(f"\nBackfill completado. Verificar con: python shadow_book.py --report")


if __name__ == "__main__":
    main()
