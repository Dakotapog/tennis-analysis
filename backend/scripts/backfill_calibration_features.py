"""
scripts/backfill_calibration_features.py — D173-04 (Nodo-173, BLOQUE C).

El calibrador de D173-05 necesita `score_margin_signed` (D173-03) en cada
registro settled del shadow book para poder ajustarse contra historia real.
Los registros escritos ANTES de D173-03 no tienen ese campo — solo tienen
`p_modelo` (ya comprimido por las fallas F1/F2 del embudo viejo).

Reconstrucción: para registros viejos, se usa `(p_modelo - 0.5)` como proxy
de `score_margin_signed`, marcado `feature_provenance='proxy_normalizado'`
para que D173-05 pueda, si hace falta, tratarlo distinto de la señal cruda.
Registros nuevos (post-D173-03) ya traen `score_margin_signed` real dentro de
`pick_snapshot` — se copian a nivel top-level como `feature_provenance='raw'`,
sin recalcular nada.

Regla dura (igual que D144-06 / backfill_strategy.py): `pick_snapshot` es
inmutable — nunca se escribe ahí. Los campos nuevos van a nivel top-level del
registro (`score_margin_signed`, `feature_provenance`, `features_backfilled_at`).

Idempotente: correr 2 veces no cambia nada la segunda vez.

Uso:
  python scripts/backfill_calibration_features.py            # dry-run (default)
  python scripts/backfill_calibration_features.py --commit    # escribe de verdad
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_SCRIPT_DIR)
_SHADOW_DIR = os.path.join(_BACKEND_DIR, "reports", "shadow_book")


def _load_jsonl_path(path: str) -> dict:
    """Lee JSONL → dict sb_id/cr_id → record. Preserva orden de inserción."""
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
            except json.JSONDecodeError:
                continue
            key = rec.get("sb_id") or rec.get("cr_id")
            if key:
                records[key] = rec
    return records


def _save_jsonl_path(path: str, records: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _es_settled(rec: dict) -> bool:
    res = rec.get("resolucion") or {}
    return isinstance(res, dict) and res.get("resultado") in ("WON", "LOST")


def backfill_record(rec: dict) -> bool:
    """Muta `rec` in-place (nunca `pick_snapshot`). Retorna True si cambió algo."""
    snap = rec.get("pick_snapshot") or {}

    raw = snap.get("score_margin_signed")
    if raw is not None:
        if rec.get("score_margin_signed") != raw or rec.get("feature_provenance") != "raw":
            rec["score_margin_signed"] = raw
            rec["feature_provenance"] = "raw"
            return True
        return False

    p_modelo = snap.get("p_modelo")
    if p_modelo is None:
        return False

    try:
        proxy = round(float(p_modelo) - 0.5, 4)
    except (TypeError, ValueError):
        return False

    if (rec.get("score_margin_signed") == proxy
            and rec.get("feature_provenance") == "proxy_normalizado"):
        return False

    rec["score_margin_signed"] = proxy
    rec["feature_provenance"] = "proxy_normalizado"
    return True


def apply_backfill(dry_run: bool) -> dict:
    stats = {
        "archivos_procesados": 0,
        "settled_total": 0,
        "backfill_raw": 0,
        "backfill_proxy": 0,
        "sin_p_modelo_skip": 0,
        "ya_backfilled_skip": 0,
    }
    stamped_at = datetime.now().isoformat()
    sb_files = sorted(glob.glob(os.path.join(_SHADOW_DIR, "sb_*.jsonl")))

    print(f"\n[backfill_calibration_features] shadow_book: {len(sb_files)} archivo(s)")

    for sb_path in sb_files:
        records = _load_jsonl_path(sb_path)
        if not records:
            continue

        modificado = False

        for rec in records.values():
            if rec.get("_type") == "session_meta":
                continue
            if not _es_settled(rec):
                continue

            stats["settled_total"] += 1
            provenance_antes = rec.get("feature_provenance")
            cambio = backfill_record(rec)

            if not cambio:
                snap = rec.get("pick_snapshot") or {}
                if snap.get("score_margin_signed") is None and snap.get("p_modelo") is None:
                    stats["sin_p_modelo_skip"] += 1
                else:
                    stats["ya_backfilled_skip"] += 1
                continue

            if not dry_run:
                rec["features_backfilled_at"] = stamped_at
            if rec.get("feature_provenance") == "raw" and provenance_antes != "raw":
                stats["backfill_raw"] += 1
            elif rec.get("feature_provenance") == "proxy_normalizado":
                stats["backfill_proxy"] += 1
            modificado = True

        stats["archivos_procesados"] += 1
        if modificado and not dry_run:
            _save_jsonl_path(sb_path, records)
            print(f"  {os.path.basename(sb_path)}: guardado")
        elif modificado:
            print(f"  {os.path.basename(sb_path)}: [DRY-RUN] se modificaría")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="D173-04: Backfill score_margin_signed en shadow_book settled"
    )
    parser.add_argument("--commit", action="store_true",
                        help="Escribe de verdad. Sin este flag, corre en dry-run.")
    args = parser.parse_args()
    dry_run = not args.commit

    print("=" * 60)
    print("D173-04 Nodo-173 — Backfill score_margin_signed en shadow_book")
    print(f"Modo: {'DRY-RUN' if dry_run else 'ESCRITURA REAL'}")
    print("=" * 60)

    os.chdir(_BACKEND_DIR)
    stats = apply_backfill(dry_run)

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:<28} {v}")

    if dry_run:
        print("\n[DRY-RUN] Nada se escribió. Correr con --commit para aplicar.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
