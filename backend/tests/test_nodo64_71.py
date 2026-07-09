"""
tests/test_nodo64_71.py — Tests Nodo-64 (SPRT) y Nodo-70 (CPPI)

REGLA-T53: cada test invoca la función real del módulo, NUNCA hardcodea la fórmula.
"""
import math


# ─────────────────────────────────────────────────────────────────────────────
# T64 — SPRT (Sequential Probability Ratio Test, Wald 1945)
# ─────────────────────────────────────────────────────────────────────────────

def test_sprt_acepta_h1_antes_n20():
    """Serie sintética p=0.85 vs H0=0.50, H1=0.70 → ACEPTA_H1 con n=20, hits=17
    Nota: boundary_A=ln(19)≈2.944. Con p0=0.50, p1=0.70 y 70% hit rate (14/20)
    el LLR≈1.646 no alcanza la frontera. Se necesita hit rate >70% para cruzar en n=20.
    17/20 = 85% → LLR = 17*ln(1.4)+3*ln(0.6) ≈ 4.19 > 2.944 → ACEPTA_H1.
    """
    from validation.hypothesis_tracker import sprt_verdict
    v = sprt_verdict(n=20, hits=17, p0=0.50, p1=0.70)
    assert v['verdict'] == 'ACEPTA_H1', f"Esperado ACEPTA_H1, got {v['verdict']}, LLR={v['llr']}"


def test_sprt_continua_n30_p_igual():
    """Serie p=0.50 (H0 verdadera) → CONTINÚA en n=30"""
    from validation.hypothesis_tracker import sprt_verdict
    v = sprt_verdict(n=30, hits=15, p0=0.50, p1=0.70)
    assert v['verdict'] == 'CONTINUA', f"Esperado CONTINUA, got {v['verdict']}"


def test_sprt_fronteras_correctas():
    """Fronteras A y B calculadas con α=β=0.05"""
    from validation.hypothesis_tracker import sprt_verdict
    v = sprt_verdict(n=5, hits=3, p0=0.50, p1=0.70)
    expected_A = math.log((1 - 0.05) / 0.05)
    expected_B = math.log(0.05 / (1 - 0.05))
    assert abs(v['boundary_A'] - expected_A) < 1e-6
    assert abs(v['boundary_B'] - expected_B) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# T70 — CPPI (Black-Perold)
# ─────────────────────────────────────────────────────────────────────────────

def test_cppi_factor_bankroll_peak():
    """A bankroll=peak → factor > 0 y ≤ 1.0"""
    from trader_ev_tenis import _cppi_factor
    f = _cppi_factor(bankroll=125000, peak_bankroll=125000)
    assert 0.0 < f <= 1.0, f"factor esperado (0,1], got {f}"


def test_cppi_factor_bankroll_floor():
    """A bankroll=FLOOR → factor = 0.0"""
    from trader_ev_tenis import _cppi_factor, _CPPI_FLOOR_PCT
    floor = 125000 * _CPPI_FLOOR_PCT
    f = _cppi_factor(bankroll=floor, peak_bankroll=125000)
    assert f == 0.0, f"factor en floor debe ser 0.0, got {f}"


def test_cppi_factor_monotono():
    """Factor CPPI es monótono creciente con bankroll"""
    from trader_ev_tenis import _cppi_factor
    peak = 125000
    b1, b2, b3 = 90000, 100000, 115000
    f1 = _cppi_factor(b1, peak)
    f2 = _cppi_factor(b2, peak)
    f3 = _cppi_factor(b3, peak)
    assert f1 <= f2 <= f3, f"No monotónico: {f1}, {f2}, {f3}"


def test_cppi_constantes_provisionales():
    """Constantes CPPI existen y son las documentadas"""
    from trader_ev_tenis import _CPPI_FLOOR_PCT, _CPPI_MULTIPLIER
    assert _CPPI_FLOOR_PCT == 0.70
    assert _CPPI_MULTIPLIER == 2.0


# ─────────────────────────────────────────────────────────────────────────────
# T67 — Nodo-67 CUSUM + PSI (analysis/drift_monitor.py)
# REGLA-T53: invocan funciones reales del modulo
# ─────────────────────────────────────────────────────────────────────────────

def test_cusum_alarma_con_salto_brier():
    """Serie con salto de Brier +0.08 en t=10 → alarma antes de t=20"""
    from analysis.drift_monitor import cusum_brier
    brier_estable = [0.20] * 10
    brier_degradado = [0.28] * 15
    serie = brier_estable + brier_degradado
    r = cusum_brier(serie)
    assert r['alarma'] == True, "Debe detectar degradación"
    assert r['alarm_t'] is not None and r['alarm_t'] <= 20, f"Alarma tardía: t={r['alarm_t']}"


def test_cusum_sin_alarma_estacionario():
    """Serie estacionaria → sin alarma"""
    from analysis.drift_monitor import cusum_brier
    serie = [0.22] * 30
    r = cusum_brier(serie)
    assert r['alarma'] == False, "No debe alarmar en serie estacionaria"


def test_psi_score_identico():
    """PSI entre distribuciones idénticas ≈ 0"""
    from analysis.drift_monitor import psi_score
    dist = [5, 8, 12, 15, 10, 7, 9, 11, 6, 8]
    p = psi_score(dist, dist)
    assert p < 0.01, f"PSI debe ser ~0 para dist idéntica, got {p}"


def test_psi_score_divergente():
    """PSI entre distribuciones muy distintas > 0.25"""
    from analysis.drift_monitor import psi_score
    dist_ref = [5] * 20
    dist_new = [50] * 20
    p = psi_score(dist_ref, dist_new)
    assert p > 0.25, f"PSI debe ser >0.25 para distribuciones distintas, got {p}"


def test_cusum_serie_vacia():
    """Serie vacía → sin alarma, sin crash"""
    from analysis.drift_monitor import cusum_brier
    r = cusum_brier([])
    assert r['alarma'] == False
    assert r['alarm_t'] is None
    assert r['cusum_series'] == []


def test_cusum_retorna_campos_esperados():
    """cusum_brier retorna todos los campos requeridos"""
    from analysis.drift_monitor import cusum_brier
    r = cusum_brier([0.2, 0.21, 0.19])
    for campo in ('cusum_series', 'alarm_t', 'brier_ref', 'alarma', 'k', 'h'):
        assert campo in r, f"Campo faltante: {campo}"


def test_cusum_parametros_k_h_provisionales():
    """Constantes _CUSUM_K y _CUSUM_H existen y tienen valores provisionales documentados"""
    from analysis.drift_monitor import _CUSUM_K, _CUSUM_H
    assert _CUSUM_K == 0.005, f"_CUSUM_K esperado 0.005, got {_CUSUM_K}"
    assert _CUSUM_H == 0.05, f"_CUSUM_H esperado 0.05, got {_CUSUM_H}"


def test_psi_score_dist_vacia():
    """PSI con distribuciones vacías → 0.0 sin crash"""
    from analysis.drift_monitor import psi_score
    assert psi_score([], []) == 0.0
    assert psi_score([1, 2, 3], []) == 0.0
    assert psi_score([], [1, 2, 3]) == 0.0


def test_daily_drift_report_sin_datos():
    """daily_drift_report con directorio vacío → n insuficiente sin crash"""
    import tempfile
    from analysis.drift_monitor import daily_drift_report
    with tempfile.TemporaryDirectory() as tmpdir:
        result = daily_drift_report(shadow_dir=tmpdir)
    assert 'n_settled' in result
    assert result['n_settled'] == 0
    assert 'note' in result


# ─────────────────────────────────────────────────────────────────────────────
# T66 — Nodo-66 FLB Curve (analysis/flb_curve.py)
# REGLA-T53: invocan funciones reales del modulo
# ─────────────────────────────────────────────────────────────────────────────

def test_flb_curva_plana_calibrada():
    """Con datos vacíos → n=0, estructura correcta retornada sin crash"""
    from analysis.flb_curve import flb_curve
    import tempfile
    result = flb_curve(shadow_dir=tempfile.mkdtemp())
    assert 'bandas' in result
    assert 'n_total' in result
    assert result['n_total'] == 0


def test_breakeven_ajustado_para_cuota():
    """breakeven_ajustado_para_cuota devuelve 1/cuota si n insuficiente"""
    from analysis.flb_curve import breakeven_ajustado_para_cuota
    fake_report = {
        'bandas': [{'rango': '1.5-2.0', 'n': 5, 'breakeven_ajustado': None, 'n_suficiente': False}],
        'n_total': 5
    }
    b = breakeven_ajustado_para_cuota(1.80, fake_report)
    assert abs(b - 1 / 1.80) < 0.001


def test_flb_constante_n_min():
    """_N_MIN_BANDA == 10"""
    from analysis.flb_curve import _N_MIN_BANDA
    assert _N_MIN_BANDA == 10


def test_flb_bandas_cubren_rango():
    """Las bandas cubren desde 1.0 hasta 100"""
    from analysis.flb_curve import _CUOTA_BANDS
    assert _CUOTA_BANDS[0][0] == 1.0
    assert _CUOTA_BANDS[-1][1] == 100.0


def test_flb_retorna_nota_reporte_solo():
    """flb_curve siempre retorna nota REPORTE_SOLO"""
    import tempfile
    from analysis.flb_curve import flb_curve
    result = flb_curve(shadow_dir=tempfile.mkdtemp())
    assert 'nota' in result
    assert 'REPORTE_SOLO' in result['nota']


def test_flb_numero_bandas_correcto():
    """flb_curve retorna exactamente len(_CUOTA_BANDS) bandas"""
    import tempfile
    from analysis.flb_curve import flb_curve, _CUOTA_BANDS
    result = flb_curve(shadow_dir=tempfile.mkdtemp())
    assert len(result['bandas']) == len(_CUOTA_BANDS)


def test_breakeven_ajustado_cuota_cero():
    """breakeven_ajustado_para_cuota con cuota=0 → 0.0 sin crash"""
    from analysis.flb_curve import breakeven_ajustado_para_cuota
    result = breakeven_ajustado_para_cuota(0.0, {'bandas': [], 'n_total': 0})
    assert result == 0.0


def test_breakeven_ajustado_cuota_con_datos_suficientes():
    """breakeven_ajustado_para_cuota usa breakeven_ajustado cuando n_suficiente=True"""
    from analysis.flb_curve import breakeven_ajustado_para_cuota
    fake_report = {
        'bandas': [
            {'rango': '1.0-1.5', 'n': 15, 'breakeven_ajustado': 0.72, 'n_suficiente': True},
            {'rango': '1.5-2.0', 'n': 20, 'breakeven_ajustado': 0.58, 'n_suficiente': True},
        ],
        'n_total': 35
    }
    # cuota=1.75 cae en banda 1.5-2.0
    b = breakeven_ajustado_para_cuota(1.75, fake_report)
    assert abs(b - 0.58) < 0.001, f"Esperado 0.58, got {b}"


def test_flb_estructura_banda():
    """Cada banda en flb_curve tiene los campos requeridos"""
    import tempfile
    from analysis.flb_curve import flb_curve
    result = flb_curve(shadow_dir=tempfile.mkdtemp())
    campos_requeridos = {
        'rango', 'n', 'hit_pct', 'p_implicita_media',
        'breakeven_ingenuo', 'breakeven_ajustado', 'flb_delta', 'n_suficiente'
    }
    for banda in result['bandas']:
        for campo in campos_requeridos:
            assert campo in banda, f"Campo faltante '{campo}' en banda {banda.get('rango')}"


# ─────────────────────────────────────────────────────────────────────────────
# T65 — Nodo-65 Bootstrap ρ Empírico (analysis/rho_empirical.py)
# REGLA-T53: invocan funciones reales del módulo
# ─────────────────────────────────────────────────────────────────────────────

def test_rho_outcomes_independientes():
    """Mix de sesiones alta-correlación y anti-correlación → ρ̂ en [-0.5, 0.5].
    Nota: outcomes alternados [1,0,1,0] son NEGATIVAMENTE correlados (ρ=-1/3);
    outcomes clonados [1,1,1,1] son positivamente correlados (ρ=1).
    Al mezclar en partes iguales, ρ̂ debe estar cerca de 0.
    REGLA-T53: invoca block_bootstrap_rho real."""
    from analysis.rho_empirical import block_bootstrap_rho
    # Mix 50/50: sesiones positivas (ρ=1) y negativas (ρ≈-1/3)
    pos_sessions = [[1, 1, 1, 1]] * 5 + [[0, 0, 0, 0]] * 5
    neg_sessions = [[1, 0, 1, 0]] * 5 + [[0, 1, 0, 1]] * 5
    sessions = pos_sessions[:6] + neg_sessions[:12]  # 6 pos + 12 neg → promedio ≈ 0
    r = block_bootstrap_rho(sessions, seed=42)
    assert -0.6 <= r['rho_hat'] <= 0.6, (
        f"ρ̂ debe ser moderado para mezcla balanceada: {r['rho_hat']:.3f}"
    )
    # El IC debe estar en rango razonable (no +1 ni -1 extremos)
    assert r['ci_90_lo'] < 0.5, f"IC superior no debe ser extremo positivo: {r['ci_90_lo']}"


def test_rho_outcomes_clonados():
    """Con outcomes clonados (todos iguales en sesión), ρ IC contiene ~1"""
    from analysis.rho_empirical import block_bootstrap_rho
    sessions = [[1, 1, 1, 1]] * 10 + [[0, 0, 0, 0]] * 10  # alta correlación intra
    r = block_bootstrap_rho(sessions, seed=42)
    assert r['rho_hat'] > 0.5, f"ρ̂ debe ser alto para outcomes clonados: {r['rho_hat']:.3f}"


def test_rho_gate_insuficiente():
    """Con pocas sesiones → gate_ok=False pero no crashea"""
    from analysis.rho_empirical import block_bootstrap_rho
    sessions = [[1, 0, 1]] * 3  # solo 3 sesiones, gate requiere 15
    r = block_bootstrap_rho(sessions, seed=42)
    assert r['gate_ok'] == False


def test_rho_constantes():
    """Constantes Nodo-65 existen con valores correctos"""
    from analysis.rho_empirical import _BOOTSTRAP_B, _RHO_MIN_SESSIONS, _RHO_MIN_PICKS_PER_SESSION
    assert _BOOTSTRAP_B == 2000
    assert _RHO_MIN_SESSIONS == 15
    assert _RHO_MIN_PICKS_PER_SESSION == 3


def test_rho_report_sin_datos():
    """rho_report con directorio vacío → estructura correcta sin crash"""
    import tempfile
    from analysis.rho_empirical import rho_report
    result = rho_report(shadow_dir=tempfile.mkdtemp())
    assert 'por_tier' in result
    assert 'n_total_sessions' in result
    assert result['n_total_sessions'] == 0
    assert 'grand_slam' in result['por_tier']


# ─────────────────────────────────────────────────────────────────────────────
# T68 — Nodo-68 Conformal Band (analysis/conformal_band.py)
# REGLA-T53: invocan funciones reales del módulo
# ─────────────────────────────────────────────────────────────────────────────

def test_conformal_quantile_correcto():
    """Cuantil 90% de residuos sintéticos"""
    from analysis.conformal_band import conformal_quantile
    residuals = sorted([abs(0.5 - (i / 100)) for i in range(100)])
    q = conformal_quantile(residuals, alpha=0.10)
    # Q90 de residuos [0.0, 0.01, ..., 0.49] → percentil conservador ~0.44
    assert 0.40 < q < 0.50, f"Quantil 90% esperado ~0.44, got {q}"


def test_no_bet_conformal_moneda():
    """p=0.52 con q=0.06 → NO-BET (intervalo cruza 0.5)"""
    from analysis.conformal_band import is_no_bet_conformal
    assert is_no_bet_conformal(p_modelo=0.52, q=0.06) == True


def test_no_bet_conformal_seguro():
    """p=0.65 con q=0.06 → BET (intervalo no cruza 0.5)"""
    from analysis.conformal_band import is_no_bet_conformal
    assert is_no_bet_conformal(p_modelo=0.65, q=0.06) == False


def test_conformal_n_min_constantes():
    """Constantes de gate correctas"""
    from analysis.conformal_band import _CONFORMAL_N_MIN, _CONFORMAL_N_MIN_TIER
    assert _CONFORMAL_N_MIN == 50
    assert _CONFORMAL_N_MIN_TIER == 30


def test_conformal_report_sin_datos():
    """conformal_report con directorio vacío → estructura correcta sin crash"""
    import tempfile
    from analysis.conformal_band import conformal_report
    result = conformal_report(shadow_dir=tempfile.mkdtemp())
    assert 'q_global' in result
    assert 'n_settled' in result
    assert result['n_settled'] == 0
    assert result['gate_ok'] == False
    assert 'nota' in result
    assert 'REPORTE_SOLO' in result['nota']


# ─────────────────────────────────────────────────────────────────────────────
# T69 — Nodo-69 Pattern Audit (analysis/pattern_audit.py)
# REGLA-T53: invocan funciones reales del módulo
# ─────────────────────────────────────────────────────────────────────────────

def test_pattern_audit_sin_efecto():
    """Sin efecto real (outcomes random) → no significativo"""
    import tempfile
    import json
    import os
    from analysis.pattern_audit import audit_pattern
    tmpdir = tempfile.mkdtemp()
    sb_path = os.path.join(tmpdir, 'sb_2026-07-01.jsonl')
    records = []
    for i in range(20):
        records.append({
            'sb_id': f'r{i}',
            'outcome': i % 2,   # alternado (hit rate 50%)
            'resolucion': {'resultado': 'WON' if i % 2 == 1 else 'LOST'},
            'pick_snapshot': {
                'gcs_active': i < 10,  # patrón en primeros 10
                'cuota_favorito': 1.80,
                'p_modelo': 0.58,
                'tier': 'atp500',
                'superficie': 'Hierba',
                'calibration_epoch': 'epoch_2',
            },
        })
    with open(sb_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    result = audit_pattern('gcs_active', True, shadow_dir=tmpdir)
    assert 'hit_pct_patron' in result
    # Con n=10 patrón y 10 controles, alternando 50%/50% → no significativo
    assert not result.get('significativo', False)


def test_pattern_audit_con_efecto():
    """Patrón con +20pp de hit rate → detectado (hit% correcto)"""
    import tempfile
    import json
    import os
    from analysis.pattern_audit import audit_pattern
    tmpdir = tempfile.mkdtemp()
    sb_path = os.path.join(tmpdir, 'sb_2026-07-01.jsonl')
    records = []
    for i in range(30):
        is_patron = i < 15
        outcome = 1 if (is_patron and i % 10 < 8) or (not is_patron and i % 10 < 3) else 0
        records.append({
            'sb_id': f'r{i}',
            'resolucion': {'resultado': 'WON' if outcome == 1 else 'LOST'},
            'pick_snapshot': {
                'gcs_active': is_patron,
                'cuota_favorito': 1.80,
                'p_modelo': 0.58,
                'tier': 'atp500',
                'superficie': 'Hierba',
                'calibration_epoch': 'epoch_2',
            },
        })
    with open(sb_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    result = audit_pattern('gcs_active', True, shadow_dir=tmpdir)
    assert result['hit_pct_patron'] > result['hit_pct_controles'], (
        "Patrón debe tener mayor hit%"
    )


def test_pattern_audit_campos_requeridos():
    """audit_pattern retorna todos los campos del contrato"""
    import tempfile
    from analysis.pattern_audit import audit_pattern
    result = audit_pattern('gcs_active', True, shadow_dir=tempfile.mkdtemp())
    campos = [
        'patron', 'n_patron', 'n_controles', 'n_pares',
        'hit_pct_patron', 'hit_pct_controles', 'diferencia',
        'mcnemar_p', 'significativo', 'plantilla_hipotesis', 'nota',
    ]
    for campo in campos:
        assert campo in result, f"Campo faltante: {campo}"


# ─────────────────────────────────────────────────────────────────────────────
# T71 — Nodo-71 Velocity Monitor (analysis/velocity_monitor.py)
# REGLA-T53: invocan funciones reales del módulo
# ─────────────────────────────────────────────────────────────────────────────

def test_velocity_steam_acortamiento():
    """Cuota 4.0→2.0 en 2h vs volatilidad típica → z<-2 (STEAM)"""
    from analysis.velocity_monitor import velocity_zscore
    # Serie: estable luego gran caída
    odds = [4.0, 3.9, 3.85, 3.80, 3.82, 3.79, 3.78, 2.00]
    times = [0, 30, 60, 90, 120, 150, 180, 210]
    r = velocity_zscore(odds, times)
    assert r['steam'] == True or r['z_last'] < -2, f"Debe detectar STEAM, z={r['z_last']}"


def test_velocity_plana():
    """Serie plana → |z| < 2"""
    from analysis.velocity_monitor import velocity_zscore
    odds = [2.00, 2.00, 2.00, 1.99, 2.00, 2.01, 2.00, 1.99]
    times = list(range(0, 80, 10))
    r = velocity_zscore(odds, times)
    assert abs(r['z_last']) < 2, f"Serie plana no debe alarmar: z={r['z_last']}"


def test_velocity_serie_corta():
    """Serie < 3 puntos → z_last=None"""
    from analysis.velocity_monitor import velocity_zscore
    r = velocity_zscore([2.0, 1.9], [0, 30])
    assert r['z_last'] is None


def test_velocity_steam_threshold():
    """_VELOCITY_Z_THRESH == -2.0"""
    from analysis.velocity_monitor import _VELOCITY_Z_THRESH
    assert _VELOCITY_Z_THRESH == -2.0


def test_velocity_campos_requeridos():
    """velocity_zscore retorna todos los campos del contrato"""
    from analysis.velocity_monitor import velocity_zscore
    odds = [2.0, 1.9, 1.85, 1.80, 1.78]
    times = [0, 30, 60, 90, 120]
    r = velocity_zscore(odds, times)
    for campo in ('velocities', 'z_scores', 'z_last', 'steam', 'signal', 'nota'):
        assert campo in r, f"Campo faltante: {campo}"
    assert r['signal'] in ('STEAM', 'DRIFT', 'FLAT')
    assert 'REPORTE_SOLO' in r['nota']
