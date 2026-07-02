"""
tests/test_nodo51_f1.py — Nodo-51 F1: TournamentContext como Entidad

Tests T51-F1-01 → T51-F1-09

Validan:
  - _surface_from_tournament_name: los 3 tipos de parseo
      * Tipo A — keyword explícito en el nombre (clay/grass/hard/arcilla/hierba/dura)
      * Tipo B — tabla de torneos conocidos (Wimbledon, Roland Garros, etc.)
      * Tipo C — unknown cuando no se puede inferir
  - _season_transition_flag: True si <14 días de frontera de calendario
      * 30-jun → True (hierba→hard, frontera July 12: 12 días → dentro)
      * 15-mar → False (far from all boundaries)
      * 01-abr → True  (hard→clay, frontera April 7: 6 días → dentro)
  - build_tournament_context: integración completa del subdict

Detección de mutación real:
  T51-F1-02  FALLA si se elimina la tabla _KNOWN_TOURNAMENT_SURFACES
  T51-F1-04  FALLA si se elimina el fallback 'unknown'
  T51-F1-06  FALLA si season_transition_flag no usa <14 días
  T51-F1-07b FALLA si 30-jun no es True (frontera hierba→hard a 12 días)
  T51-F1-08  FALLA si build_tournament_context no llama detectar_tier()
  T51-F1-09  FALLA si tournament_context no llega al match dict en kambi_tennis
"""
from datetime import date

import pytest

from core.tournament_context import (
    _SURFACE_MAP,
    _surface_from_tournament_name,
    _season_transition_flag,
    build_tournament_context,
    normalize_surface,
)


# ─────────────────────────────────────────────────────────────────────────────
# T51-F1-01 a T51-F1-04 — _surface_from_tournament_name
# ─────────────────────────────────────────────────────────────────────────────

class TestSurfaceFromTournamentName:

    def test_t51_f1_01_type_a_explicit_keyword_clay(self):
        """T51-F1-01a: Keyword 'arcilla' o 'clay' en el nombre → 'clay'.
        Tipo A: keyword explícito en el nombre del torneo."""
        assert _surface_from_tournament_name("ATP Challenger Bogotá (Colombia), arcilla") == "clay"
        assert _surface_from_tournament_name("Roland Garros clay") == "clay"
        assert _surface_from_tournament_name("ITF M15 Monastir - Tierra") == "clay"

    def test_t51_f1_01b_type_a_explicit_keyword_grass(self):
        """T51-F1-01b: Keyword 'hierba' o 'grass' en el nombre → 'grass'."""
        assert _surface_from_tournament_name("ATP - INDIVIDUALES: Stuttgart (Alemania), hierba") == "grass"
        assert _surface_from_tournament_name("Challenger Ilkley grass") == "grass"

    def test_t51_f1_01c_type_a_explicit_keyword_hard(self):
        """T51-F1-01c: Keyword 'dura' o 'hard' en el nombre → 'hard'."""
        assert _surface_from_tournament_name("ATP 500 Viena (Austria), dura") == "hard"
        assert _surface_from_tournament_name("US Open hard court") == "hard"

    def test_t51_f1_02_type_b_known_tournament_wimbledon(self):
        """T51-F1-02: Wimbledon no dice 'grass' en el nombre → tabla conocida.
        FALLA si se elimina _KNOWN_TOURNAMENT_SURFACES."""
        assert _surface_from_tournament_name("Wimbledon") == "grass"
        assert _surface_from_tournament_name("Wimbledon (Gran Bretaña)") == "grass"
        assert _surface_from_tournament_name("ATP - INDIVIDUALES: Wimbledon") == "grass"

    def test_t51_f1_02b_type_b_known_tournament_roland_garros(self):
        """T51-F1-02b: Roland Garros → clay (tabla conocida)."""
        assert _surface_from_tournament_name("Roland Garros") == "clay"
        assert _surface_from_tournament_name("Roland Garros (Francia)") == "clay"

    def test_t51_f1_02c_type_b_known_tournament_grand_slams(self):
        """T51-F1-02c: Australian Open y US Open → hard (tabla conocida)."""
        assert _surface_from_tournament_name("Australian Open") == "hard"
        assert _surface_from_tournament_name("US Open") == "hard"

    def test_t51_f1_03_type_b_known_challenger_cary(self):
        """T51-F1-03: Challenger Cary (USA, hard) — conocido por F1 (caso real Nodo-46)."""
        assert _surface_from_tournament_name("Challenger Cary") == "hard"
        assert _surface_from_tournament_name("ATP Challenger Cary (USA)") == "hard"

    def test_t51_f1_04_type_c_unknown_when_cannot_infer(self):
        """T51-F1-04: Nombre sin keywords y no en tabla → 'unknown'.
        FALLA si se elimina el fallback 'unknown'."""
        assert _surface_from_tournament_name("ATP 250 Generic City") == "unknown"
        assert _surface_from_tournament_name("") == "unknown"
        assert _surface_from_tournament_name(None) == "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# T51-F1-05 — normalize_surface
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeSurface:

    def test_t51_f1_05a_normalize_grass_variants(self):
        """T51-F1-05a: 'Hierba', 'grass', 'herb' → 'grass'."""
        assert normalize_surface("hierba") == "grass"
        assert normalize_surface("Hierba") == "grass"
        assert normalize_surface("grass") == "grass"
        assert normalize_surface("Grass") == "grass"

    def test_t51_f1_05b_normalize_clay_variants(self):
        """T51-F1-05b: 'Arcilla', 'clay', 'tierra' → 'clay'."""
        assert normalize_surface("arcilla") == "clay"
        assert normalize_surface("clay") == "clay"
        assert normalize_surface("tierra") == "clay"

    def test_t51_f1_05c_normalize_hard_variants(self):
        """T51-F1-05c: 'Dura', 'hard', 'hardcourt', 'carpet' → 'hard'."""
        assert normalize_surface("dura") == "hard"
        assert normalize_surface("hard") == "hard"
        assert normalize_surface("hardcourt") == "hard"
        assert normalize_surface("carpet") == "hard"

    def test_t51_f1_05d_unknown_fallback(self):
        """T51-F1-05d: Variante desconocida → 'unknown'."""
        assert normalize_surface("tierra batida") == "unknown"  # multi-word not in map
        assert normalize_surface("") == "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# T51-F1-06 a T51-F1-07 — _season_transition_flag
# ─────────────────────────────────────────────────────────────────────────────

class TestSeasonTransitionFlag:

    def test_t51_f1_06_far_from_all_boundaries_is_false(self):
        """T51-F1-06: Fecha lejos de cualquier frontera → False.
        15-mar: más cercana es April 7 (23 días) → False.
        FALLA si season_transition_flag no usa <14 días."""
        assert _season_transition_flag(date(2026, 3, 15)) is False

    def test_t51_f1_07_june_30_is_true_hierba_hard(self):
        """T51-F1-07: 30-jun → True.
        Frontera Grass→Hard: July 12. Distancia: 12 días < 14 → True.
        FALLA si 30-jun no es True (esta es la case que el spec usa explícitamente)."""
        assert _season_transition_flag(date(2026, 6, 30)) is True

    def test_t51_f1_07b_april_1_is_true_hard_clay(self):
        """T51-F1-07b: 01-abr → True. Frontera Hard→Clay: April 7. Distancia: 6 días < 14 → True."""
        assert _season_transition_flag(date(2026, 4, 1)) is True

    def test_t51_f1_07c_april_7_is_true_on_boundary(self):
        """T51-F1-07c: La frontera exacta (distance=0) → True (0 < 14)."""
        assert _season_transition_flag(date(2026, 4, 7)) is True

    def test_t51_f1_07d_june_2_is_true_clay_grass(self):
        """T51-F1-07d: 02-jun → True. Frontera Clay→Grass: June 2. Distancia: 0 → True."""
        assert _season_transition_flag(date(2026, 6, 2)) is True

    def test_t51_f1_07e_july_27_is_false_far_from_boundary(self):
        """T51-F1-07e: 27-jul → False.
        Frontera más cercana: July 12 (15 días) y Oct 28 (93 días) → 15 ≥ 14 → False."""
        assert _season_transition_flag(date(2026, 7, 27)) is False


# ─────────────────────────────────────────────────────────────────────────────
# T51-F1-08 a T51-F1-09 — build_tournament_context
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildTournamentContext:

    def test_t51_f1_08_wimbledon_context(self):
        """T51-F1-08: Wimbledon → tier='grand_slam', superficie='grass'.
        FALLA si build_tournament_context no llama detectar_tier()."""
        ctx = build_tournament_context("Wimbledon", match_date=date(2026, 7, 1))

        assert ctx["nombre"] == "Wimbledon"
        assert ctx["tier"] == "grand_slam"
        assert ctx["superficie"] == "grass"
        assert "season_transition_flag" in ctx
        assert isinstance(ctx["season_transition_flag"], bool)

    def test_t51_f1_08b_challenger_cary_context(self):
        """T51-F1-08b: Challenger Cary → tier='challenger', superficie='hard'."""
        ctx = build_tournament_context("Challenger Cary (USA)", match_date=date(2026, 6, 29))

        assert ctx["tier"] == "challenger"
        assert ctx["superficie"] == "hard"

    def test_t51_f1_08c_season_flag_propagates_correctly(self):
        """T51-F1-08c: season_transition_flag en el contexto usa la match_date dada."""
        # June 30 → True (hierba→hard, 12 días)
        ctx_near = build_tournament_context("Wimbledon", match_date=date(2026, 6, 30))
        assert ctx_near["season_transition_flag"] is True

        # March 15 → False (lejos de fronteras)
        ctx_far = build_tournament_context("Australian Open", match_date=date(2026, 3, 15))
        assert ctx_far["season_transition_flag"] is False

    def test_t51_f1_08d_empty_torneo_returns_unknown(self):
        """T51-F1-08d: torneo_completo vacío → superficie='unknown', tier default."""
        ctx = build_tournament_context("", match_date=date(2026, 6, 15))
        assert ctx["superficie"] == "unknown"
        assert ctx["nombre"] == ""

    def test_t51_f1_08e_none_match_date_uses_today(self):
        """T51-F1-08e: match_date=None no lanza excepción — usa date.today()."""
        ctx = build_tournament_context("Wimbledon")  # match_date omitido
        assert ctx["superficie"] == "grass"
        assert isinstance(ctx["season_transition_flag"], bool)


class TestTournamentContextIntegrationKambi:
    """
    T51-F1-09: Verifica que los matches que salen de extract_matches_kambi_flashscore()
    incluyen el campo 'tournament_context' en cada dict.

    Usa la función _attach_tournament_contexts() que el módulo kambi_tennis.py
    expone para inyectar el contexto a una lista de matches.
    """

    def test_t51_f1_09_attach_context_adds_subdict(self):
        """T51-F1-09: _attach_tournament_contexts añade 'tournament_context' a cada match.
        FALLA si tournament_context no llega al match dict en kambi_tennis."""
        from scraping.kambi_tennis import _attach_tournament_contexts

        matches = [
            {"jugador1": "A", "jugador2": "B", "torneo_completo": "Wimbledon", "superficie": "grass"},
            {"jugador1": "C", "jugador2": "D", "torneo_completo": "Challenger Cary (USA)", "superficie": "hard"},
            {"jugador1": "E", "jugador2": "F", "torneo_completo": None, "superficie": "unknown"},
        ]

        result = _attach_tournament_contexts(matches, match_date=date(2026, 6, 30))

        for m in result:
            assert "tournament_context" in m, (
                f"Match {m['jugador1']} vs {m['jugador2']} no tiene tournament_context"
            )
            ctx = m["tournament_context"]
            assert "nombre" in ctx
            assert "tier" in ctx
            assert "superficie" in ctx
            assert "season_transition_flag" in ctx

    def test_t51_f1_09b_context_values_correct(self):
        """T51-F1-09b: Valores del context son correctos para cada torneo."""
        from scraping.kambi_tennis import _attach_tournament_contexts

        matches = [
            {"jugador1": "A", "jugador2": "B", "torneo_completo": "Wimbledon"},
            {"jugador1": "C", "jugador2": "D", "torneo_completo": "Challenger Cary (USA)"},
        ]

        result = _attach_tournament_contexts(matches, match_date=date(2026, 6, 30))

        wimbledon = result[0]["tournament_context"]
        assert wimbledon["tier"] == "grand_slam"
        assert wimbledon["superficie"] == "grass"
        assert wimbledon["season_transition_flag"] is True  # 30-jun → True

        cary = result[1]["tournament_context"]
        assert cary["tier"] == "challenger"
        assert cary["superficie"] == "hard"

    def test_t51_f1_09c_existing_fields_unchanged(self):
        """T51-F1-09c: _attach_tournament_contexts NO modifica los campos existentes del match.
        El campo 'superficie' existente queda intacto — tournament_context es ADICIÓN."""
        from scraping.kambi_tennis import _attach_tournament_contexts

        match = {"jugador1": "X", "jugador2": "Y", "torneo_completo": "Wimbledon", "superficie": "grass", "tier": "grand_slam"}
        result = _attach_tournament_contexts([match], match_date=date(2026, 7, 1))

        assert result[0]["superficie"] == "grass"   # campo original intacto
        assert result[0]["tier"] == "grand_slam"     # campo original intacto
        assert "tournament_context" in result[0]     # nuevo campo añadido
