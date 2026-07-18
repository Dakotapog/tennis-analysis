"""
tests/test_nodo110_ranking_only.py — D110-06 RANKING_ONLY universe extension

REGLA-T53: Tests invocan función real del módulo — nunca hardcodean la fórmula.

D110-06 spec:
  Partidos del archivo PASO 1 SIN entrada en edge_report entran al universo
  con fuente=RANKING_ONLY si cumplen TODO:
    (a) ranking_gap > RANKING_GAP_MIN (300)
    (b) cuota_favorito ∈ [1.15, 1.60]
    (c) favorito por ranking = favorito del book
  Salvaguarda: combo nunca lleva >2 piernas RANKING_ONLY.
  Partido presente en edge_report NUNCA se duplica desde --matches.
"""

import json
import tempfile
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent.parent))

from favoritos_combo_builder import (
    _leer_matches_ranking_only,
    armar_combos,
    MAX_RANKING_ONLY_PER_COMBO,
    RANKING_GAP_MIN,
    LEG_MIN_CUOTA,
    LEG_MAX_CUOTA_RANKING_ONLY,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_matches_file(partidos: list, torneo: str = "ITF Fake Open") -> str:
    """Escribe un zita_tennis_matches temporal y devuelve la ruta."""
    data = {torneo: partidos}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        return f.name


def _make_rankings_files(players: list) -> tuple:
    """
    Escribe rankings ATP y WTA temporales y parchea glob en el módulo.
    players: list of (name, position) — formato "Surname Firstname"
    Devuelve (atp_path, wta_path) para cleanup.
    """
    rankings = [
        {"name": name, "ranking_position": pos, "ranking_points": 1000}
        for name, pos in players
    ]
    data = {"metadata": {}, "rankings": rankings}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        return f.name


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRankingOnlyFiltros(unittest.TestCase):
    """REGLA-T53: _leer_matches_ranking_only aplica los 3 filtros de D110-06."""

    def _ranking_map_patch(self, players: list) -> dict:
        """Devuelve un ranking_map en formato esperado por _buscar_ranking."""
        import unicodedata, re
        def norm(n):
            n = unicodedata.normalize("NFD", n.lower())
            n = "".join(c for c in n if unicodedata.category(c) != "Mn")
            return re.sub(r"[^a-z\s]", "", n).strip()
        return {norm(name): pos for name, pos in players}

    def _run_with_ranking_patch(self, matches_path, edge_picks_set, ranking_players):
        """
        Ejecuta _leer_matches_ranking_only con _leer_rankings parchado
        para no depender de archivos reales en disco.
        """
        import favoritos_combo_builder as fcb
        orig = fcb._leer_rankings
        fcb._leer_rankings = lambda: self._ranking_map_patch(ranking_players)
        try:
            result = fcb._leer_matches_ranking_only(matches_path, edge_picks_set)
        finally:
            fcb._leer_rankings = orig
        return result

    def test_pasa_cuando_todos_los_criterios_se_cumplen(self):
        """Partido con gap>300, cuota [1.15,1.60], ranking=book → 1 candidato RANKING_ONLY."""
        partidos = [{
            "jugador1": "Federer Roger",   # ranking 10
            "jugador2": "Novak Djokovic",  # ranking 450
            "cuota1": 1.30,
            "cuota2": 3.50,
            "ranking1": None,
            "ranking2": None,
        }]
        path = _make_matches_file(partidos)
        try:
            result = self._run_with_ranking_patch(
                path,
                edge_picks_set=set(),
                ranking_players=[("Federer Roger", 10), ("Novak Djokovic", 450)],
            )
        finally:
            Path(path).unlink()

        self.assertEqual(len(result), 1, f"esperado 1 candidato, obtuve {len(result)}")
        self.assertEqual(result[0]["fuente"], "RANKING_ONLY")
        self.assertEqual(result[0]["favorito"], "Federer Roger")
        self.assertGreater(result[0]["ranking_gap"], RANKING_GAP_MIN)

    def test_descarta_cuando_cuota_fuera_de_rango(self):
        """cuota_favorito=1.70 > LEG_MAX_CUOTA_RANKING_ONLY → descartado."""
        partidos = [{
            "jugador1": "Alcaraz Carlos",  # ranking 1
            "jugador2": "Muller Thomas",   # ranking 400
            "cuota1": 1.70,                # fuera de [1.15, 1.60]
            "cuota2": 2.10,
            "ranking1": None,
            "ranking2": None,
        }]
        path = _make_matches_file(partidos)
        try:
            result = self._run_with_ranking_patch(
                path,
                edge_picks_set=set(),
                ranking_players=[("Alcaraz Carlos", 1), ("Muller Thomas", 400)],
            )
        finally:
            Path(path).unlink()

        self.assertEqual(len(result), 0, f"cuota 1.70 debe ser descartada (max={LEG_MAX_CUOTA_RANKING_ONLY})")

    def test_descarta_cuando_gap_insuficiente(self):
        """ranking_gap = 200 ≤ RANKING_GAP_MIN (300) → descartado."""
        partidos = [{
            "jugador1": "Smith John",   # ranking 100
            "jugador2": "Brown Mike",   # ranking 300 → gap=200
            "cuota1": 1.35,
            "cuota2": 3.00,
            "ranking1": None,
            "ranking2": None,
        }]
        path = _make_matches_file(partidos)
        try:
            result = self._run_with_ranking_patch(
                path,
                edge_picks_set=set(),
                ranking_players=[("Smith John", 100), ("Brown Mike", 300)],
            )
        finally:
            Path(path).unlink()

        self.assertEqual(len(result), 0, f"gap 200 debe ser descartado (min={RANKING_GAP_MIN})")

    def test_no_duplica_jugadores_ya_en_edge_report(self):
        """Jugador ya en edge_picks_set NO debe aparecer en RANKING_ONLY."""
        partidos = [{
            "jugador1": "Gaines John",  # ya en edge_report
            "jugador2": "Jones Pete",
            "cuota1": 1.25,
            "cuota2": 3.00,
            "ranking1": None,
            "ranking2": None,
        }]
        path = _make_matches_file(partidos)
        # Simular que Gaines ya está en el edge_report
        edge_picks_set = {"gaines john"}
        try:
            result = self._run_with_ranking_patch(
                path,
                edge_picks_set=edge_picks_set,
                ranking_players=[("Gaines John", 50), ("Jones Pete", 450)],
            )
        finally:
            Path(path).unlink()

        self.assertEqual(len(result), 0, "Jugador en edge_report no debe duplicarse desde --matches")


class TestComboMaxRankingOnly(unittest.TestCase):
    """REGLA-T53: armar_combos nunca supera MAX_RANKING_ONLY_PER_COMBO piernas RANKING_ONLY."""

    def _make_pick(self, nombre, cuota, fuente="EDGE_REPORT", torneo="Torneo A", p=0.72):
        return {
            "favorito": nombre, "jugador": nombre, "favorito_predicho": nombre,
            "cuota_favorito": cuota, "cuota_rival": 3.0,
            "p_modelo": p, "torneo": torneo, "tournament": torneo,
            "fuente": fuente, "ranking_gap": 400,
        }

    def test_combo_nunca_lleva_mas_de_2_ranking_only(self):
        """
        Con 3 piernas RANKING_ONLY y 1 EDGE_REPORT, los combos válidos deben
        tener ≤2 piernas RANKING_ONLY (MAX_RANKING_ONLY_PER_COMBO=2).
        """
        picks = [
            self._make_pick("PlayerA", 1.25, "RANKING_ONLY", "Torneo A"),
            self._make_pick("PlayerB", 1.30, "RANKING_ONLY", "Torneo B"),
            self._make_pick("PlayerC", 1.40, "RANKING_ONLY", "Torneo C"),
            self._make_pick("PlayerD", 1.55, "EDGE_REPORT",  "Torneo D"),
        ]
        combos = armar_combos(picks)
        for combo in combos:
            n_ronly = sum(1 for p in combo["legs"] if p.get("fuente") == "RANKING_ONLY")
            self.assertLessEqual(
                n_ronly, MAX_RANKING_ONLY_PER_COMBO,
                f"Combo tiene {n_ronly} piernas RANKING_ONLY (max={MAX_RANKING_ONLY_PER_COMBO})"
            )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
