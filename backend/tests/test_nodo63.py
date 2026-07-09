"""
tests/test_nodo63.py — Nodo-63: Insufficient History Guard + Anchor Combo Builder

PARTE A — Insufficient History Guard (rivalry_analyzer.py):
T63-01: n_partidos=3 → form_decay factor = 1.0 (guard activo, no decay)
T63-02: n_partidos=10 → form_decay factor < 1.0 cuando days=60 (decay normal)
T63-03: n_partidos=3, days=356 → factor es 1.0 (guard activo, no x0.35)
T63-04: n_partidos=7 → guard activo (boundary: < 8)
T63-05: n_partidos=8 → guard NO activo (boundary: exactamente 8 = aplica decay)
T63-06: _MIN_HISTORY_FOR_DECAY == 8 (constante correcta)

PARTE B — Anchor Combo Builder (combo_confianza_builder.py):
T63-07: pick con priority=85, cuota=2.06 → clasificado como ANCLA
T63-08: pick con priority=65, cuota=1.33 → clasificado como BASE
T63-09: _build_anchor_combos con picks válidos → combos_1a3b no vacío cuando hay anclas y bases
T63-10: combo 1A+3B tiene exactamente 1 pick de tipo ancla
T63-11: combo 2A+2B tiene exactamente 2 picks de tipo ancla
T63-12: ANCHOR_CUOTA_MIN == 1.65 y ANCHOR_PRIORITY_MIN == 75.0
"""
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_pick(nombre, cuota, confianza, priority=None, edge_pct='0%', torneo='TestTorneo'):
    """Crea un pick mínimo con campos requeridos por _classify_anchors y _build_anchor_combos."""
    from combo_confianza_builder import _categorizar_pick
    cat = _categorizar_pick(cuota, confianza) or {
        'categoria': 'CAT_A' if cuota < 1.60 else 'CAT_B',
        'combos_permitidos': ['CORE'],
        'pipeline_flag': False,
    }
    return {
        'nombre': nombre,
        'cuota': cuota,
        'confianza': confianza,
        'combo_priority': priority if priority is not None else confianza,
        'torneo': torneo,
        'cat': cat,
        'alpha_senales': [],
        'alpha_score': 0.0,
        'edge_data_ref': {'edge_pct': edge_pct},
    }


# ─────────────────────────────────────────────────────────────────────────────
# PARTE A — Insufficient History Guard
# ─────────────────────────────────────────────────────────────────────────────

class TestInsufficientHistoryGuard:

    def test_T63_01_n3_decay_factor_is_1(self):
        """n=3 (< _MIN_HISTORY_FOR_DECAY=8) → guard activo → factor 1.0."""
        from analysis.rivalry_analyzer import _MIN_HISTORY_FOR_DECAY, _FORM_GRACE_DAYS
        import math
        _FORM_DECAY_LAMBDA = 0.025
        _FORM_DECAY_FLOOR  = 0.35

        def _form_decay_factor(days):
            if days == -1: return 0.70
            if days <= _FORM_GRACE_DAYS: return 1.0
            return max(_FORM_DECAY_FLOOR, math.exp(-_FORM_DECAY_LAMBDA * (days - _FORM_GRACE_DAYS)))

        n_p1 = 3
        days = 60
        fd = 1.0 if n_p1 < _MIN_HISTORY_FOR_DECAY else _form_decay_factor(days)
        assert fd == 1.0, (
            f"T63-01: n={n_p1} < {_MIN_HISTORY_FOR_DECAY} → guard debe forzar fd=1.0, got {fd}"
        )

    def test_T63_02_n10_decay_applied_for_60d(self):
        """n=10 (>= _MIN_HISTORY_FOR_DECAY) + days=60 → decay normal < 1.0."""
        from analysis.rivalry_analyzer import _MIN_HISTORY_FOR_DECAY, _FORM_GRACE_DAYS
        import math
        _FORM_DECAY_LAMBDA = 0.025
        _FORM_DECAY_FLOOR  = 0.35

        def _form_decay_factor(days):
            if days == -1: return 0.70
            if days <= _FORM_GRACE_DAYS: return 1.0
            return max(_FORM_DECAY_FLOOR, math.exp(-_FORM_DECAY_LAMBDA * (days - _FORM_GRACE_DAYS)))

        n_p1 = 10
        days = 60
        fd = 1.0 if n_p1 < _MIN_HISTORY_FOR_DECAY else _form_decay_factor(days)
        assert fd < 1.0, (
            f"T63-02: n={n_p1} >= {_MIN_HISTORY_FOR_DECAY}, days={days} → decay debe aplicar, got fd={fd}"
        )

    def test_T63_03_n3_days356_factor_not_floor(self):
        """n=3, days=356 → guard activo → factor es 1.0, NO el floor 0.35 del decay."""
        from analysis.rivalry_analyzer import _MIN_HISTORY_FOR_DECAY, _FORM_GRACE_DAYS
        import math
        _FORM_DECAY_LAMBDA = 0.025
        _FORM_DECAY_FLOOR  = 0.35

        def _form_decay_factor(days):
            if days == -1: return 0.70
            if days <= _FORM_GRACE_DAYS: return 1.0
            return max(_FORM_DECAY_FLOOR, math.exp(-_FORM_DECAY_LAMBDA * (days - _FORM_GRACE_DAYS)))

        n_p1 = 3
        days = 356
        fd = 1.0 if n_p1 < _MIN_HISTORY_FOR_DECAY else _form_decay_factor(days)
        assert fd == 1.0, (
            f"T63-03: n={n_p1}, days={days} → guard activo → fd debe ser 1.0, no {_FORM_DECAY_FLOOR}, got {fd}"
        )

    def test_T63_04_n7_guard_active(self):
        """n=7 (boundary: 7 < 8) → guard activo."""
        from analysis.rivalry_analyzer import _MIN_HISTORY_FOR_DECAY, _FORM_GRACE_DAYS
        import math
        _FORM_DECAY_LAMBDA = 0.025
        _FORM_DECAY_FLOOR  = 0.35

        def _form_decay_factor(days):
            if days == -1: return 0.70
            if days <= _FORM_GRACE_DAYS: return 1.0
            return max(_FORM_DECAY_FLOOR, math.exp(-_FORM_DECAY_LAMBDA * (days - _FORM_GRACE_DAYS)))

        n_p1 = 7
        days = 90
        fd = 1.0 if n_p1 < _MIN_HISTORY_FOR_DECAY else _form_decay_factor(days)
        assert fd == 1.0, f"T63-04: n=7 < 8 → guard activo, fd debe ser 1.0, got {fd}"

    def test_T63_05_n8_guard_not_active(self):
        """n=8 (exactamente = _MIN_HISTORY_FOR_DECAY) → guard NO activo → decay aplica."""
        from analysis.rivalry_analyzer import _MIN_HISTORY_FOR_DECAY, _FORM_GRACE_DAYS
        import math
        _FORM_DECAY_LAMBDA = 0.025
        _FORM_DECAY_FLOOR  = 0.35

        def _form_decay_factor(days):
            if days == -1: return 0.70
            if days <= _FORM_GRACE_DAYS: return 1.0
            return max(_FORM_DECAY_FLOOR, math.exp(-_FORM_DECAY_LAMBDA * (days - _FORM_GRACE_DAYS)))

        n_p1 = 8
        days = 90
        fd = 1.0 if n_p1 < _MIN_HISTORY_FOR_DECAY else _form_decay_factor(days)
        assert fd < 1.0, (
            f"T63-05: n=8 = _MIN_HISTORY_FOR_DECAY → guard NO activo, decay debe aplicar, got fd={fd}"
        )

    def test_T63_06_min_history_constant(self):
        """_MIN_HISTORY_FOR_DECAY debe ser exactamente 8."""
        from analysis.rivalry_analyzer import _MIN_HISTORY_FOR_DECAY
        assert _MIN_HISTORY_FOR_DECAY == 8, (
            f"T63-06: _MIN_HISTORY_FOR_DECAY debe ser 8, got {_MIN_HISTORY_FOR_DECAY}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PARTE B — Anchor Combo Builder
# ─────────────────────────────────────────────────────────────────────────────

class TestAnchorComboBuilder:

    def test_T63_07_high_priority_high_cuota_is_ancla(self):
        """Pick con priority=85, cuota=2.06 → clasificado como ANCLA."""
        from combo_confianza_builder import _classify_anchors
        pick = _make_pick('Sinner', cuota=2.06, confianza=58.0, priority=85.0)
        anclas, bases = _classify_anchors([pick])
        assert pick in anclas, (
            f"T63-07: priority=85, cuota=2.06 → debe ser ANCLA. anclas={[a['nombre'] for a in anclas]}"
        )
        assert pick not in bases

    def test_T63_08_low_priority_low_cuota_is_base(self):
        """Pick con priority=65, cuota=1.33 → clasificado como BASE."""
        from combo_confianza_builder import _classify_anchors
        pick = _make_pick('Alcaraz', cuota=1.33, confianza=65.0, priority=65.0)
        anclas, bases = _classify_anchors([pick])
        assert pick in bases, (
            f"T63-08: priority=65, cuota=1.33 → debe ser BASE (cuota < ANCHOR_CUOTA_MIN=1.65)"
        )
        assert pick not in anclas

    def test_T63_09_build_anchor_combos_returns_1a3b(self):
        """Con picks válidos (anclas + bases), combos_1a3b no está vacío."""
        from combo_confianza_builder import _build_anchor_combos
        picks = [
            _make_pick('Ancla1', cuota=1.80, confianza=62.0, priority=80.0, torneo='Wimbledon'),
            _make_pick('Base1',  cuota=1.30, confianza=70.0, priority=70.0, torneo='Wimbledon'),
            _make_pick('Base2',  cuota=1.25, confianza=68.0, priority=68.0, torneo='Roland Garros'),
            _make_pick('Base3',  cuota=1.35, confianza=66.0, priority=66.0, torneo='US Open'),
        ]
        result = _build_anchor_combos(picks, bankroll=125000, fase=4)
        assert result['n_anclas'] >= 1, "T63-09: debe haber al menos 1 ancla"
        assert len(result['combos_1a3b']) > 0, (
            f"T63-09: combos_1a3b debe tener al menos 1 combo con 1 ancla + 3 bases"
        )

    def test_T63_10_1a3b_has_exactly_1_ancla(self):
        """Cada combo 1A+3B contiene exactamente 1 pick con cuota >= ANCHOR_CUOTA_MIN."""
        from combo_confianza_builder import _build_anchor_combos, ANCHOR_CUOTA_MIN
        picks = [
            _make_pick('Ancla1', cuota=1.75, confianza=62.0, priority=80.0, torneo='Wimbledon'),
            _make_pick('Base1',  cuota=1.30, confianza=70.0, priority=70.0, torneo='Wimbledon'),
            _make_pick('Base2',  cuota=1.25, confianza=68.0, priority=68.0, torneo='Roland Garros'),
            _make_pick('Base3',  cuota=1.35, confianza=66.0, priority=66.0, torneo='US Open'),
        ]
        result = _build_anchor_combos(picks, bankroll=125000, fase=4)
        assert result['combos_1a3b'], "T63-10: necesitamos al menos 1 combo 1A+3B"

        for combo in result['combos_1a3b']:
            n_ancla = sum(1 for q in combo['cuotas'] if q >= ANCHOR_CUOTA_MIN)
            assert n_ancla >= 1, (
                f"T63-10: combo 1A+3B debe tener >=1 ancla, tiene {n_ancla}: {combo['piernas']}"
            )

    def test_T63_11_2a2b_has_exactly_2_anclas(self):
        """Cada combo 2A+2B contiene exactamente 2 picks con cuota >= ANCHOR_CUOTA_MIN."""
        from combo_confianza_builder import _build_anchor_combos, ANCHOR_CUOTA_MIN
        picks = [
            _make_pick('Ancla1', cuota=1.80, confianza=62.0, priority=85.0, torneo='Wimbledon'),
            _make_pick('Ancla2', cuota=2.10, confianza=61.0, priority=78.0, torneo='Roland Garros'),
            _make_pick('Base1',  cuota=1.30, confianza=70.0, priority=70.0, torneo='Wimbledon'),
            _make_pick('Base2',  cuota=1.25, confianza=68.0, priority=68.0, torneo='US Open'),
        ]
        result = _build_anchor_combos(picks, bankroll=125000, fase=4)
        assert result['combos_2a2b'], "T63-11: necesitamos al menos 1 combo 2A+2B"

        for combo in result['combos_2a2b']:
            n_ancla = sum(1 for q in combo['cuotas'] if q >= ANCHOR_CUOTA_MIN)
            assert n_ancla >= 2, (
                f"T63-11: combo 2A+2B debe tener >=2 anclas, tiene {n_ancla}: {combo['piernas']}"
            )

    def test_T63_12_anchor_constants(self):
        """ANCHOR_CUOTA_MIN == 1.65 y ANCHOR_PRIORITY_MIN == 75.0."""
        from combo_confianza_builder import ANCHOR_CUOTA_MIN, ANCHOR_PRIORITY_MIN
        assert ANCHOR_CUOTA_MIN == 1.65, (
            f"T63-12: ANCHOR_CUOTA_MIN debe ser 1.65, got {ANCHOR_CUOTA_MIN}"
        )
        assert ANCHOR_PRIORITY_MIN == 75.0, (
            f"T63-12: ANCHOR_PRIORITY_MIN debe ser 75.0, got {ANCHOR_PRIORITY_MIN}"
        )
