"""
Pattern Recognition Engine — D90-09 (Nodo-90 Sprint 4)
REPORTE_SOLO: lee picks settled del shadow book, reporta candidatos.
NO escribe a preregistered_hypotheses.json — promoción = decisión humana.

Uso:
    python3 scripts/pattern_recognition.py [--min-n 5] [--apostar-only]
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ── Wilson IC95 ──────────────────────────────────────────────────────────────

def _wilson_ic95(wins: int, n: int) -> tuple[float, float]:
    """Wilson score interval (95%). Returns (lower, upper)."""
    if n == 0:
        return (0.0, 1.0)
    z = 1.96  # z_0.975
    p_hat = wins / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


# ── Segment keys ─────────────────────────────────────────────────────────────

_DIMS = [
    "tier",
    "superficie",
    "zona_cuota",
    "markov_favorito",
    "confidence_flag",
]


def _segment_key(snap: dict, dim: str) -> str:
    val = snap.get(dim)
    if val is None:
        return "?"
    return str(val)


# ── Load settled ─────────────────────────────────────────────────────────────

def load_settled(shadow_book_dir: str, apostar_only: bool = False) -> list[dict]:
    records = []
    pattern = os.path.join(shadow_book_dir, "sb_*.jsonl")
    for fpath in sorted(glob.glob(pattern)):
        with open(fpath, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "resolucion" not in rec:
                    continue
                if apostar_only and rec.get("pick_snapshot", {}).get("apostar") is not True:
                    continue
                records.append(rec)
    return records


# ── Segment stats ─────────────────────────────────────────────────────────────

def _compute_segment_stats(
    records: list[dict], dim: str, min_n: int
) -> list[dict]:
    """Compute hit% + Wilson IC95 per segment value for one dimension."""
    buckets: dict[str, list] = defaultdict(list)
    for rec in records:
        snap = rec["pick_snapshot"]
        key = _segment_key(snap, dim)
        won = rec["resolucion"]["resultado"] == "WON"
        cuota = snap.get("cuota_favorito") or 0.0
        buckets[key].append((won, cuota))

    rows = []
    for val, entries in sorted(buckets.items()):
        n = len(entries)
        wins = sum(w for w, _ in entries)
        cuotas = [c for _, c in entries if c > 0]
        avg_cuota = sum(cuotas) / len(cuotas) if cuotas else None
        breakeven = (1.0 / avg_cuota) if avg_cuota else None
        hit_pct = wins / n
        ic_low, ic_high = _wilson_ic95(wins, n)
        candidate = (
            n >= min_n
            and breakeven is not None
            and ic_low > breakeven
        )
        rows.append(
            {
                "dim": dim,
                "value": val,
                "n": n,
                "wins": wins,
                "hit_pct": round(hit_pct, 4),
                "ic95_low": round(ic_low, 4),
                "ic95_high": round(ic_high, 4),
                "avg_cuota": round(avg_cuota, 3) if avg_cuota else None,
                "breakeven": round(breakeven, 4) if breakeven else None,
                "candidate": candidate,
            }
        )
    return rows


# ── Cross-segment (2-way) ─────────────────────────────────────────────────────

def _compute_cross_stats(
    records: list[dict], dim_a: str, dim_b: str, min_n: int
) -> list[dict]:
    """2-way cross of two dimensions."""
    buckets: dict[tuple, list] = defaultdict(list)
    for rec in records:
        snap = rec["pick_snapshot"]
        key = (_segment_key(snap, dim_a), _segment_key(snap, dim_b))
        won = rec["resolucion"]["resultado"] == "WON"
        cuota = snap.get("cuota_favorito") or 0.0
        buckets[key].append((won, cuota))

    rows = []
    for (val_a, val_b), entries in sorted(buckets.items()):
        n = len(entries)
        wins = sum(w for w, _ in entries)
        cuotas = [c for _, c in entries if c > 0]
        avg_cuota = sum(cuotas) / len(cuotas) if cuotas else None
        breakeven = (1.0 / avg_cuota) if avg_cuota else None
        hit_pct = wins / n
        ic_low, ic_high = _wilson_ic95(wins, n)
        candidate = (
            n >= min_n
            and breakeven is not None
            and ic_low > breakeven
        )
        rows.append(
            {
                "dim": f"{dim_a}×{dim_b}",
                "value": f"{val_a}|{val_b}",
                "n": n,
                "wins": wins,
                "hit_pct": round(hit_pct, 4),
                "ic95_low": round(ic_low, 4),
                "ic95_high": round(ic_high, 4),
                "avg_cuota": round(avg_cuota, 3) if avg_cuota else None,
                "breakeven": round(breakeven, 4) if breakeven else None,
                "candidate": candidate,
            }
        )
    return rows


# ── Overall stats ─────────────────────────────────────────────────────────────

def _overall_stats(records: list[dict], min_n: int) -> dict:
    n = len(records)
    wins = sum(1 for r in records if r["resolucion"]["resultado"] == "WON")
    cuotas = [r["pick_snapshot"].get("cuota_favorito") or 0.0 for r in records]
    valid_cuotas = [c for c in cuotas if c > 0]
    avg_cuota = sum(valid_cuotas) / len(valid_cuotas) if valid_cuotas else None
    breakeven = (1.0 / avg_cuota) if avg_cuota else None
    ic_low, ic_high = _wilson_ic95(wins, n)
    return {
        "n": n,
        "wins": wins,
        "hit_pct": round(wins / n, 4) if n else 0,
        "ic95_low": round(ic_low, 4),
        "ic95_high": round(ic_high, 4),
        "avg_cuota": round(avg_cuota, 3) if avg_cuota else None,
        "breakeven": round(breakeven, 4) if breakeven else None,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def run_pattern_recognition(
    shadow_book_dir: str,
    reports_dir: str,
    min_n: int = 5,
    apostar_only: bool = False,
) -> dict:
    records = load_settled(shadow_book_dir, apostar_only=apostar_only)

    if not records:
        print("[WARN] No settled records found.", file=sys.stderr)
        return {}

    overall = _overall_stats(records, min_n)

    # 1-way segments
    all_rows: list[dict] = []
    for dim in _DIMS:
        all_rows.extend(_compute_segment_stats(records, dim, min_n))

    # 2-way cross: pairs with most analytical value
    cross_pairs = [
        ("tier", "superficie"),
        ("tier", "confidence_flag"),
        ("superficie", "markov_favorito"),
        ("zona_cuota", "confidence_flag"),
        ("markov_favorito", "confidence_flag"),
    ]
    cross_rows: list[dict] = []
    for dim_a, dim_b in cross_pairs:
        cross_rows.extend(_compute_cross_stats(records, dim_a, dim_b, min_n))

    candidates_1way = [r for r in all_rows if r["candidate"]]
    candidates_cross = [r for r in cross_rows if r["candidate"]]

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "params": {
            "min_n": min_n,
            "apostar_only": apostar_only,
            "shadow_book_dir": shadow_book_dir,
        },
        "overall": overall,
        "note": (
            "REPORTE_SOLO. Candidatos son señales observacionales. "
            "Promoción a hipótesis = decisión humana vía preregistered_hypotheses.json."
        ),
        "segments_1way": all_rows,
        "segments_cross": cross_rows,
        "candidates_1way": candidates_1way,
        "candidates_cross": candidates_cross,
        "n_candidates": len(candidates_1way) + len(candidates_cross),
    }

    # Write output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(reports_dir, f"pattern_candidates_{ts}.json")
    os.makedirs(reports_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    return report, out_path


# ── CLI print ─────────────────────────────────────────────────────────────────

def _print_report(report: dict, out_path: str) -> None:
    overall = report["overall"]
    print(
        f"\n=== PatternRecognition D90-09 ===\n"
        f"  Settled: {overall['n']} | WON: {overall['wins']} "
        f"| hit%: {overall['hit_pct']:.1%} (IC95 [{overall['ic95_low']:.1%}, {overall['ic95_high']:.1%}])\n"
        f"  Avg cuota: {overall['avg_cuota']} | Breakeven: {overall['breakeven']:.1%}\n"
        f"  Candidatos: {report['n_candidates']} (1-way: {len(report['candidates_1way'])}, cross: {len(report['candidates_cross'])})"
    )

    # 1-way table
    print("\n--- Segmentos 1-way (todos) ---")
    print(f"{'DIM':<22} {'VAL':<20} {'N':>4} {'WON':>4} {'HIT%':>6} {'IC_LOW':>7} {'IC_HI':>7} {'BRK':>6} {'CAND'}")
    print("-" * 95)
    for r in sorted(report["segments_1way"], key=lambda x: (-x["n"], x["dim"])):
        cand = "*** CANDIDATO ***" if r["candidate"] else ""
        bk = f"{r['breakeven']:.1%}" if r["breakeven"] else "  N/A"
        print(
            f"  {r['dim']:<20} {r['value']:<20} {r['n']:>4} {r['wins']:>4} "
            f"{r['hit_pct']:>6.1%} {r['ic95_low']:>7.1%} {r['ic95_high']:>7.1%} {bk:>6} {cand}"
        )

    # Cross table — only show candidate rows to keep output short
    if report["candidates_cross"]:
        print("\n--- Candidatos cross (2-way) ---")
        print(f"{'DIM':<30} {'VAL':<30} {'N':>4} {'HIT%':>6} {'IC_LOW':>7} {'BRK':>6}")
        print("-" * 90)
        for r in sorted(report["candidates_cross"], key=lambda x: -x["n"]):
            bk = f"{r['breakeven']:.1%}" if r["breakeven"] else "  N/A"
            print(
                f"  {r['dim']:<28} {r['value']:<30} {r['n']:>4} "
                f"{r['hit_pct']:>6.1%} {r['ic95_low']:>7.1%} {bk:>6}"
            )
    else:
        print("\n--- Candidatos cross (2-way): ninguno ---")

    print(f"\n  [OK] Escrito en: {out_path}")
    print("  REPORTE_SOLO — promocion a hipotesis = decision humana.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PatternRecognition D90-09 — REPORTE_SOLO")
    parser.add_argument("--min-n", type=int, default=5, help="Mínimo picks para candidato (default 5)")
    parser.add_argument("--apostar-only", action="store_true", help="Solo picks con apostar=True")
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    shadow_dir = str(base / "reports" / "shadow_book")
    reports_dir = str(base / "reports")

    result = run_pattern_recognition(
        shadow_book_dir=shadow_dir,
        reports_dir=reports_dir,
        min_n=args.min_n,
        apostar_only=args.apostar_only,
    )
    if result:
        report, out_path = result
        _print_report(report, out_path)


if __name__ == "__main__":
    main()
