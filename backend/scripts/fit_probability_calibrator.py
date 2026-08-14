"""
scripts/fit_probability_calibrator.py — D173-05 (Nodo-173, BLOQUE C).

Ajusta core/probability_calibrator.py contra shadow_book settled y decide,
con un criterio de aceptación NO NEGOCIABLE (spec §D173-05), si se despliega:

    holdout_skill > 0  →  se puede --commit (escribe data/probability_calibrator.json)
    holdout_skill <= 0 →  NO se escribe artefacto, PUERTA 3 no se abre, punto.

Prohibido ajustar la forma funcional / features / min_n para forzar que el
número cruce el umbral. Si falla, el hallazgo se reporta tal cual — falla o no
falla, no se re-intenta con otra configuración para "mejorar" el resultado.

Uso:
  python scripts/fit_probability_calibrator.py --report          # solo imprime métricas
  python scripts/fit_probability_calibrator.py --commit          # escribe SOLO si aprobado
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _BACKEND_DIR)

from core.probability_calibrator import fit_calibrator  # noqa: E402

_SHADOW_DIR = os.path.join(_BACKEND_DIR, 'reports', 'shadow_book')
_ARTIFACT_PATH = os.path.join(_BACKEND_DIR, 'data', 'probability_calibrator.json')


def _cargar_registros_settled() -> list:
    records = []
    for path in sorted(glob.glob(os.path.join(_SHADOW_DIR, 'sb_*.jsonl'))):
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get('_type') == 'session_meta':
                    continue
                records.append(rec)
    return records


def _imprimir_reporte(artifact: dict) -> None:
    print('=' * 60)
    print('D173-05 — Calibrador ancla-mercado: reporte de ajuste')
    print('=' * 60)
    print(f"n_entrenamiento: {artifact['n_entrenamiento']}  n_holdout: {artifact['n_holdout']}")
    print(f"provenance (features): {artifact['feature_provenance_split']}")
    print(f"ventana temporal train: {artifact['ventana_temporal']['train_desde']} -> "
          f"{artifact['ventana_temporal']['train_hasta']}")
    print(f"ventana temporal holdout: {artifact['ventana_temporal']['holdout_desde']} -> "
          f"{artifact['ventana_temporal']['holdout_hasta']}")
    print()
    print('coeficientes:')
    for k, v in artifact['coeficientes'].items():
        print(f'  {k:<8} {v:+.4f}')
    print()
    mh = artifact['metricas_holdout']
    mb = artifact['metricas_baseline_holdout']
    print(f"{'':<20}{'calibrado':>14}{'mercado crudo':>16}")
    print(f"{'brier':<20}{mh['brier']:>14.4f}{mb['brier']:>16.4f}")
    print(f"{'skill (BSS)':<20}{mh['skill']:>14.4f}{mb['skill']:>16.4f}")
    print(f"{'auc':<20}{mh['auc']:>14.4f}{mb['auc']:>16.4f}")
    print()
    if mh['auc'] < 0.55:
        print('⚠ AUC holdout < 0.55 — revisar pipeline antes de confiar en skill.')
    print(f"CRITERIO DURO (skill holdout > 0): {'APROBADO' if artifact['aprobado'] else 'NO APROBADO'}")
    if not artifact['aprobado']:
        print('PUERTA 3 (spec Nodo-173): no se despliega. USE_CALIBRATOR permanece False.')
        print('Cierre del nodo con bloques A/B/E únicamente — FASE 4 (D173-08/09) no procede.')
    print()
    print('reliability bins (holdout, calibrado):')
    for b in mh['bins']:
        print(f"  [{b['lo']:.2f},{b['hi']:.2f}) n={b['n']:>4} p_medio={b['p_medio']:.3f} "
              f"hit_real={b['hit_real']:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description='D173-05: ajustar calibrador ancla-mercado')
    parser.add_argument('--min-n', type=int, default=300)
    parser.add_argument('--report', action='store_true', help='imprime métricas, no escribe nada')
    parser.add_argument('--commit', action='store_true',
                        help='escribe data/probability_calibrator.json SOLO si aprobado=True')
    args = parser.parse_args()

    registros = _cargar_registros_settled()
    print(f'[fit_probability_calibrator] {len(registros)} registros shadow_book cargados')

    try:
        artifact = fit_calibrator(registros, min_n=args.min_n)
    except ValueError as e:
        print(f'[fit_probability_calibrator] {e}')
        return 1

    _imprimir_reporte(artifact)

    if args.commit:
        if not artifact['aprobado']:
            print('\n[fit_probability_calibrator] --commit ignorado: criterio no aprobado.')
            return 2
        os.makedirs(os.path.dirname(_ARTIFACT_PATH), exist_ok=True)
        with open(_ARTIFACT_PATH, 'w', encoding='utf-8') as fh:
            json.dump(artifact, fh, ensure_ascii=False, indent=2)
        print(f'\n[fit_probability_calibrator] artefacto escrito: {_ARTIFACT_PATH}')

    return 0 if artifact['aprobado'] else 3


if __name__ == '__main__':
    sys.exit(main())
