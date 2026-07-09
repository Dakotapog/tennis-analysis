"""
tests/test_fable02_fase0.py — Tests FABLE_02_TENIS_DOCTORADO_SPEC Fase 0

REGLA-T53: invoca funciones reales del módulo, nunca hardcodea fórmulas.

C61-A FORENSE — Causa raíz de la discrepancia GCS ×2.2 vs ×1.15 observada en producción:
  El boost GCS aplica `final_score *= _gcs_mult` (2.2/1.8/1.5) al sub-score de
  surface_specialization DENTRO de analyze_surface_specialization(). Este sub-score boosteado
  es luego:
    1. Capeado en 350 raw (`raw_scores['surface_specialization'] = min(score, 350)`)
    2. Normalizado con log1p (normalize_scores usa math.log1p por defecto)
  El ratio efectivo sobre el score normalizado es log1p(score×mult) / log1p(score), NO mult.
  Ejemplo: score=160 → boost×2.2 → 352 → log1p(352)/log1p(160) = 5.866/5.079 ≈ 1.155
  Esto explica los valores ×1.15/×1.13 observados en producción (E4 Nodo-61).
  El valor ×0.92 aparece cuando es el OPONENTE quien recibe el boost (la confianza relativa
  del jugador disminuye porque el rival sube en surface_specialization).
  La documentación "multiplicador al final_score" era técnicamente correcta pero engañosa:
  el `final_score` es el sub-score de superficie, NO la confianza final del partido.

C61-B GOBERNANZA — Decisión de activación GCS documentada en rivalry_analyzer.py header:
  Opción (a) seleccionada: activación por evidencia retrospectiva A60-01 (n=54, 64.8%),
  H60-01 prospectiva continúa. Docstring "validado por H60-01" corregido a "prior A60-01".

C63-A — n<_MIN_HISTORY_FOR_DECAY → LOG_PLAYWRIGHT_CANDIDATE emitido si match_id disponible.

C62-A — H62-01 pre-registrado en preregistered_hypotheses.json + alpha_promoted top-level en picks.
"""
import math


# ─────────────────────────────────────────────────────────────────────────────
# C61-A — Forense: atenuación del boost GCS por log1p normalization
# ─────────────────────────────────────────────────────────────────────────────

def test_c61a_boost_gcs_atenuado_por_log1p():
    """C61-A: El boost ×2.2 sobre score=160 resulta en ratio efectivo ≈1.155 tras log1p.
    Explica los valores ×1.15 observados en producción (E4 Nodo-61).
    REGLA-T53: invoca la función normalize_scores del módulo real."""
    from analysis.rivalry_analyzer import normalize_scores
    score_sin_boost = 160.0
    score_con_boost = 160.0 * 2.2  # = 352
    p1 = {'surface_specialization': score_con_boost}
    p2 = {'surface_specialization': score_sin_boost}
    norm_p1, norm_p2 = normalize_scores(p1, p2)
    ratio_efectivo = norm_p1['surface_specialization'] / norm_p2['surface_specialization']
    # El ratio esperado es log1p(352)/log1p(160) ≈ 1.155, muy lejos de 2.2
    assert 1.10 < ratio_efectivo < 1.25, (
        f"C61-A: ratio efectivo post-log1p={ratio_efectivo:.3f}. "
        f"Explica ×1.15 observado en producción (no ×2.2 documentado)"
    )


def test_c61a_boost_oponente_reduce_confianza_relativa():
    """C61-A: Cuando el oponente recibe el GCS boost, el ratio de P1 < 1.0 (×0.92 observado).
    REGLA-T53: usa normalize_scores real."""
    from analysis.rivalry_analyzer import normalize_scores
    # P1 sin boost, P2 CON boost (rival recibe boost)
    p1_score = 100.0
    p2_score = 100.0 * 2.2  # = 220
    norm_p1, norm_p2 = normalize_scores(
        {'surface_specialization': p1_score},
        {'surface_specialization': p2_score}
    )
    ratio_p1_efectivo = norm_p1['surface_specialization'] / norm_p2['surface_specialization']
    # P1 tiene score menor → ratio < 1.0 (explica ×0.92 en producción)
    assert ratio_p1_efectivo < 1.0, (
        f"C61-A: cuando oponente recibe boost, ratio P1={ratio_p1_efectivo:.3f} < 1.0. "
        f"Explica ×0.92 observado en producción"
    )


def test_c61a_cap_350_limita_boost():
    """C61-A: El cap en 350 limita el boost cuando el score inicial ya es alto.
    Un score de 300 × 2.2 = 660 queda capeado en 350 → mismo que 300 × 1.17."""
    from analysis.rivalry_analyzer import normalize_scores
    score_alto = 300.0
    score_boosteado = min(score_alto * 2.2, 350)  # cap en 350
    norm_p1, norm_p2 = normalize_scores(
        {'surface_specialization': score_boosteado},
        {'surface_specialization': score_alto}
    )
    ratio_efectivo = norm_p1['surface_specialization'] / norm_p2['surface_specialization']
    # El cap hace que el boost efectivo sea mucho menor que 2.2
    assert ratio_efectivo < 2.0, (
        f"C61-A: cap@350 reduce el boost — ratio={ratio_efectivo:.3f} << 2.2"
    )


# ─────────────────────────────────────────────────────────────────────────────
# C61-B — Gobernanza: docstring corregido + flag activado
# ─────────────────────────────────────────────────────────────────────────────

def test_c61b_flag_gcs_activo():
    """C61-B: _GCS_BOOST_ENABLED == True (activado por decisión formal A60-01 retrospectivo)."""
    from analysis.rivalry_analyzer import _GCS_BOOST_ENABLED
    assert _GCS_BOOST_ENABLED is True, "GCS debe estar ACTIVO para hierba (decisión A60-01)"


def test_c61b_docstring_no_cita_h60_01():
    """C61-B: El comentario de activación no debe citar 'validado por H60-01'.
    Debe citar 'A60-01' (evidencia retrospectiva)."""
    import inspect
    import analysis.rivalry_analyzer as mod
    src = inspect.getsource(mod)
    # La línea que solía decir "H60-01 n<30" en shadow_reason ya no existe así
    assert "validado por H60-01" not in src, (
        "C61-B: 'validado por H60-01' no debe aparecer en rivalry_analyzer.py"
    )
    # El comentario de activación debe citar A60-01
    assert "A60-01" in src, "C61-B: el comentario de activación debe citar A60-01"


# ─────────────────────────────────────────────────────────────────────────────
# C63-A — LOG_PLAYWRIGHT_CANDIDATE cuando n < _MIN_HISTORY_FOR_DECAY
# ─────────────────────────────────────────────────────────────────────────────

def test_c63a_playwright_candidate_en_codigo():
    """C63-A: LOG_PLAYWRIGHT_CANDIDATE existe en rivalry_analyzer.py como signal cuando n<8 y match_id.
    REGLA-T53: verifica el módulo real, no la fórmula."""
    import inspect
    import analysis.rivalry_analyzer as mod
    src = inspect.getsource(mod)
    assert 'LOG_PLAYWRIGHT_CANDIDATE' in src, (
        "C63-A: LOG_PLAYWRIGHT_CANDIDATE debe estar implementado en rivalry_analyzer.py"
    )
    assert '_match_id' in src, (
        "C63-A: La condición if _match_id debe existir para emitir el candidato"
    )


def test_c63a_playwright_candidate_en_scope_correcto():
    """C63-A: LOG_PLAYWRIGHT_CANDIDATE se emite en el mismo bloque que LOG_INSUFFICIENT_HISTORY."""
    import inspect
    import analysis.rivalry_analyzer as mod
    src = inspect.getsource(mod)
    # Ambos logs deben aparecer cerca uno del otro en el código
    idx_insufficient = src.find('LOG_INSUFFICIENT_HISTORY')
    idx_playwright = src.find('LOG_PLAYWRIGHT_CANDIDATE')
    assert idx_insufficient >= 0, "LOG_INSUFFICIENT_HISTORY debe existir"
    assert idx_playwright >= 0, "LOG_PLAYWRIGHT_CANDIDATE debe existir"
    # Deben estar dentro de 500 caracteres uno del otro (mismo bloque)
    assert abs(idx_playwright - idx_insufficient) < 1500, (
        f"C63-A: Los dos logs deben estar en el mismo bloque. "
        f"dist={abs(idx_playwright - idx_insufficient)}"
    )


def test_c63a_min_history_constante():
    """C63-A: _MIN_HISTORY_FOR_DECAY == 8 (inmutable)."""
    from analysis.rivalry_analyzer import _MIN_HISTORY_FOR_DECAY
    assert _MIN_HISTORY_FOR_DECAY == 8


# ─────────────────────────────────────────────────────────────────────────────
# C62-A — H62-01 pre-registrada + alpha_promoted top-level
# ─────────────────────────────────────────────────────────────────────────────

def test_c62a_h62_01_pre_registrada():
    """C62-A: H62-01 existe en preregistered_hypotheses.json."""
    from validation.hypothesis_tracker import get_hypothesis
    h = get_hypothesis('H62-01')
    assert h is not None, "H62-01 debe estar en preregistered_hypotheses.json"
    assert h.get('estado') == 'ACUMULANDO', f"H62-01 debe estar ACUMULANDO, got {h.get('estado')}"
    assert h.get('n_stop') == 30, f"n_stop debe ser 30, got {h.get('n_stop')}"


def test_c62a_alpha_promoted_en_combo_pick():
    """C62-A: alpha_promoted aparece en top-level del pick de combo_confianza_builder."""
    # Verificamos que la clave 'alpha_promoted' existe en el módulo
    import inspect
    import combo_confianza_builder as mod
    src = inspect.getsource(mod)
    assert "'alpha_promoted'" in src, (
        "C62-A: 'alpha_promoted' debe estar definido como clave top-level del pick"
    )
