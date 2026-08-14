"""
scripts/update_hypothesis_ledger.py — Nodo-174 D174-03, PASO 10e

Recorre sb_*.jsonl (todo el historial, igual que shadow_book.py::report()),
cuenta n/hits/roi por hipótesis pre-registrada vía
validation/hypothesis_ledger.py::contar_hipotesis(), y escribe el resultado
en validation/preregistered_hypotheses.json vía actualizar_registro().

Uso:
    python3 scripts/update_hypothesis_ledger.py                # escribe (dry_run=False)
    python3 scripts/update_hypothesis_ledger.py --dry-run       # solo muestra el diff
    python3 scripts/update_hypothesis_ledger.py --desde 2026-07-01 --hasta 2026-08-05

REGLA #8: nunca toca umbrales_congelados ni crea hipótesis nuevas — ver
validation/hypothesis_ledger.py::actualizar_registro().
"""
import argparse
import glob as glob_mod
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shadow_book import SHADOW_DIR, _load_jsonl
from validation.hypothesis_ledger import contar_hipotesis, actualizar_registro

_JSON_PATH = Path(__file__).parent.parent / "validation" / "preregistered_hypotheses.json"


def _cargar_settled(desde: str = None, hasta: str = None) -> list:
    """Mismo criterio que shadow_book.py::report() — todos los picks resueltos."""
    all_records = []
    for fpath in sorted(glob_mod.glob(os.path.join(SHADOW_DIR, "sb_*.jsonl"))):
        fname = os.path.basename(fpath)
        m = re.match(r'sb_(\d{4}-\d{2}-\d{2})\.jsonl', fname)
        if not m:
            continue
        if desde and m.group(1) < desde:
            continue
        if hasta and m.group(1) > hasta:
            continue
        all_records.extend(_load_jsonl(fpath).values())
    picks = [r for r in all_records if r.get('_type') != 'session_meta']
    return [r for r in picks if 'resolucion' in r]


def main():
    parser = argparse.ArgumentParser(description="PASO 10e — actualizar n_actual/hits por hipótesis")
    parser.add_argument('--desde', default=None)
    parser.add_argument('--hasta', default=None)
    parser.add_argument('--dry-run', action='store_true', help='Calcula el diff sin escribir')
    args = parser.parse_args()

    settled = _cargar_settled(args.desde, args.hasta)
    conteos = contar_hipotesis(settled)
    diff = actualizar_registro(str(_JSON_PATH), conteos, dry_run=args.dry_run)

    print(f"[HYPOTHESIS_LEDGER] {len(settled)} picks settled analizados, "
          f"{len(conteos)} hipótesis evaluadas, {len(diff)} con cambios.")
    for h_id, d in sorted(diff.items()):
        a, b = d['antes'], d['despues']
        print(f"  {h_id}: n_actual {a['n_actual']}->{b['n_actual']}  hits {a['hits']}->{b['hits']}"
              + (f"  roi_flat_1u->{b['roi_flat_1u']}" if b.get('roi_flat_1u') is not None else ""))
    if args.dry_run and diff:
        print("[HYPOTHESIS_LEDGER] --dry-run: no se escribió el archivo.")


if __name__ == '__main__':
    main()
