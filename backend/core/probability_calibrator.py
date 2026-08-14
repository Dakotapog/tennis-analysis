"""
core/probability_calibrator.py — D173-05 (Nodo-173, BLOQUE C).

Calibrador ancla-mercado. Reemplaza la lectura directa de `p_modelo` (roto por
F1/F2/F3, ver Nodo-173 §1: AUC 0.575 pero Brier skill -0.0420) por una
regresión logística ANCLADA al mercado:

    p_final = sigmoid(beta0 + beta1*logit(p_implicita) + beta2*score_margin_signed
                       + beta3*rival_ranking_missing + beta4*fav_ranking_missing)

Por qué esta forma y no otra (no sustituir sin reabrir el nodo):
  - Ancla a logit(p_implicita): con beta2=beta3=beta4=0, beta1=1, beta0=0 el
    estimador reproduce el mercado EXACTO — piso de seguridad estructural,
    el calibrador nunca puede ser peor que apostar la cuota implícita.
  - beta1 libre absorbe la sobreconfianza estructural del mercado (§1.5: 0.594).
  - score_margin_signed entra CRUDO/CON SIGNO (no |Δ| normalizado) — corrige
    F1 (piso en 0.50 vía abs()) y F2 (compresión por normalización) de raíz.
  - Los indicadores de ranking ausente reciben coeficiente negativo por
    construcción de los datos (§1.8: confianza fantasma predijo 0.932,
    entregó 0.625) — sin reglas ad-hoc, es la regresión la que lo aprende.

Regla de oro: `p_modelo` NUNCA se sobreescribe. Este módulo es puro (sin I/O);
la persistencia del artefacto vive en scripts/fit_probability_calibrator.py.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Optional

__all__ = [
    'fit_calibrator', 'predict_calibrated', 'evaluate_calibration',
    'extraer_features_registro',
]

_EPS = 1e-6
_N_BINS_DEFAULT = 10


def _logit(p: float, eps: float = _EPS) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def extraer_features_registro(rec: dict) -> Optional[dict]:
    """Extrae features de un registro CRUDO de shadow_book settled.

    Retorna None si el registro no es utilizable (no settled, sin p_implicita,
    o sin score_margin_signed — este último requiere D173-04 backfill corrido).
    """
    res = rec.get('resolucion') or {}
    resultado = res.get('resultado')
    if resultado not in ('WON', 'LOST'):
        return None

    snap = rec.get('pick_snapshot') or {}
    p_implicita = snap.get('p_implicita')
    if p_implicita is None:
        return None

    score_margin_signed = rec.get('score_margin_signed')
    if score_margin_signed is None:
        score_margin_signed = snap.get('score_margin_signed')
    if score_margin_signed is None:
        return None

    try:
        p_implicita_f = float(p_implicita)
        score_margin_f = float(score_margin_signed)
    except (TypeError, ValueError):
        return None

    return {
        'y': 1 if resultado == 'WON' else 0,
        'p_implicita': p_implicita_f,
        'score_margin_signed': score_margin_f,
        'rival_ranking_missing': snap.get('ranking_rival') is None,
        'fav_ranking_missing': snap.get('ranking_favorito') is None,
        'feature_provenance': rec.get('feature_provenance'),
        'logged_at': rec.get('logged_at'),
    }


def _design_row(fila: dict) -> list:
    return [
        _logit(fila['p_implicita']),
        fila['score_margin_signed'],
        1.0 if fila['rival_ranking_missing'] else 0.0,
        1.0 if fila['fav_ranking_missing'] else 0.0,
    ]


def _reliability_bins(y: list, p: list, n_bins: int = _N_BINS_DEFAULT) -> list:
    bins = []
    edge = 1.0 / n_bins
    for i in range(n_bins):
        lo, hi = i * edge, (i + 1) * edge
        idx = [k for k in range(len(p)) if (p[k] >= lo and (p[k] < hi or i == n_bins - 1))]
        if not idx:
            continue
        n_bin = len(idx)
        p_medio = sum(p[k] for k in idx) / n_bin
        hit_real = sum(y[k] for k in idx) / n_bin
        bins.append({'lo': lo, 'hi': hi, 'n': n_bin, 'p_medio': p_medio, 'hit_real': hit_real})
    return bins


def _auc(y: list, p: list) -> float:
    """Mann-Whitney U / n_pos*n_neg — AUC sin dependencias externas."""
    pares = sorted(zip(p, y))
    n = len(pares)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and pares[j][0] == pares[i][0]:
            j += 1
        rank_prom = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = rank_prom
        i = j
    suma_rangos_pos = sum(r for r, (_, yi) in zip(ranks, pares) if yi == 1)
    n_pos = sum(1 for _, yi in pares if yi == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return (suma_rangos_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def evaluate_calibration(y_true: list, p_pred: list) -> dict:
    """Métricas de calibración. Pura, sin I/O.

    'skill' = Brier Skill Score respecto al baseline climatológico (predecir
    siempre la tasa media de la muestra) — 0 = no mejora sobre el promedio,
    negativo = peor que adivinar la tasa base, 1 = perfecto.
    """
    n = len(y_true)
    if n == 0:
        return {'brier': float('nan'), 'brier_baseline': float('nan'),
                'skill': float('nan'), 'auc': float('nan'), 'bins': [], 'n': 0}

    y = [float(v) for v in y_true]
    p = [float(v) for v in p_pred]

    brier = sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / n
    tasa_base = sum(y) / n
    brier_baseline = sum((tasa_base - yi) ** 2 for yi in y) / n
    skill = 1.0 - (brier / brier_baseline) if brier_baseline > 0 else 0.0

    return {
        'brier': brier,
        'brier_baseline': brier_baseline,
        'skill': skill,
        'auc': _auc(y, p),
        'bins': _reliability_bins(y, p),
        'n': n,
    }


def _fit_logistic(X: list, y: list, *, lr: float = 0.1, n_iter: int = 2000,
                   l2: float = 1e-6) -> list:
    """Descenso de gradiente batch simple. Evita depender de sklearn en runtime
    de producción (solo se usa en el script de ajuste offline, pero mantenerlo
    puro en numpy/stdlib evita fragilidad de versión de sklearn en el artefacto).
    """
    n = len(y)
    n_feat = len(X[0]) + 1  # + intercepto
    coef = [0.0] * n_feat  # [beta0, beta1..beta4]

    for _ in range(n_iter):
        grad = [0.0] * n_feat
        for xi, yi in zip(X, y):
            z = coef[0] + sum(c * xv for c, xv in zip(coef[1:], xi))
            p_hat = _sigmoid(z)
            err = p_hat - yi
            grad[0] += err
            for j, xv in enumerate(xi):
                grad[j + 1] += err * xv + l2 * coef[j + 1]
        coef = [c - lr * (g / n) for c, g in zip(coef, grad)]

    return coef


def fit_calibrator(records: list, *, min_n: int = 300) -> dict:
    """Ajusta el calibrador D173-05 sobre registros CRUDOS de shadow_book settled.

    `records` debe venir en orden temporal ascendente (más viejo primero) —
    la partición train/holdout es SIEMPRE temporal (últimos 30% = holdout),
    nunca aleatoria (spec §D173-05, no negociable).

    Levanta ValueError si hay menos de `min_n` registros utilizables.
    """
    filas = [f for f in (extraer_features_registro(r) for r in records) if f is not None]

    con_fecha = [f for f in filas if f.get('logged_at')]
    if len(con_fecha) == len(filas) and filas:
        filas.sort(key=lambda f: f['logged_at'])

    n = len(filas)
    if n < min_n:
        raise ValueError(
            f'D173-05: se requieren >= {min_n} registros utilizables para ajustar '
            f'el calibrador, hay {n} (revisar D173-04 backfill si el número es bajo).'
        )

    n_holdout = max(1, round(n * 0.30))
    n_train = n - n_holdout
    train, holdout = filas[:n_train], filas[n_train:]

    X_train = [_design_row(f) for f in train]
    y_train = [f['y'] for f in train]
    coef = _fit_logistic(X_train, y_train)
    beta0, beta1, beta2, beta3, beta4 = coef

    coeficientes = {'beta0': beta0, 'beta1': beta1, 'beta2': beta2,
                     'beta3': beta3, 'beta4': beta4}

    p_holdout_cal = [
        predict_calibrated(
            {'coeficientes': coeficientes},
            p_implicita=f['p_implicita'],
            score_margin_signed=f['score_margin_signed'],
            rival_ranking_missing=f['rival_ranking_missing'],
            fav_ranking_missing=f['fav_ranking_missing'],
        )
        for f in holdout
    ]
    y_holdout = [f['y'] for f in holdout]
    metricas_holdout = evaluate_calibration(y_holdout, p_holdout_cal)

    p_holdout_mercado = [f['p_implicita'] for f in holdout]
    metricas_baseline_holdout = evaluate_calibration(y_holdout, p_holdout_mercado)

    prov_counts: dict = {}
    for f in filas:
        key = f.get('feature_provenance') or 'desconocido'
        prov_counts[key] = prov_counts.get(key, 0) + 1

    return {
        'coeficientes': coeficientes,
        'n_entrenamiento': n_train,
        'n_holdout': n_holdout,
        'ventana_temporal': {
            'train_desde': train[0].get('logged_at'),
            'train_hasta': train[-1].get('logged_at'),
            'holdout_desde': holdout[0].get('logged_at'),
            'holdout_hasta': holdout[-1].get('logged_at'),
        },
        'metricas_holdout': metricas_holdout,
        'metricas_baseline_holdout': metricas_baseline_holdout,
        'feature_provenance_split': prov_counts,
        'fitted_at': datetime.now().isoformat(),
        'aprobado': bool(metricas_holdout['skill'] > 0),
    }


def predict_calibrated(artifact: dict, *, p_implicita: float, score_margin_signed: float,
                        rival_ranking_missing: bool, fav_ranking_missing: bool) -> float:
    """Aplica el calibrador ya ajustado. Función pura, sin I/O."""
    c = artifact['coeficientes']
    z = (c['beta0']
         + c['beta1'] * _logit(p_implicita)
         + c['beta2'] * float(score_margin_signed)
         + c['beta3'] * (1.0 if rival_ranking_missing else 0.0)
         + c['beta4'] * (1.0 if fav_ranking_missing else 0.0))
    return _sigmoid(z)
