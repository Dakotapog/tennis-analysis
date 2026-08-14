"""
REGLA-T53 — Regresión: --live-stake no llegaba a _build_live_combos_legacy().

Bug detectado 2026-07-31: cuando no hay trader_plans frescos, build_live_combos()
retorna directo desde _build_live_combos_legacy() (rama "armando desde edge_report
(legacy)") ANTES del bloque que aplica override_stake (D87-08 override manual para
picks sin Kelly-KL calculado). Resultado: los 8 combos WAS generados hoy quedaron
con stake=$0 y retorno=$0 aunque se pasó --live-stake 5000 explícito.

Fix: _build_live_combos_legacy() ahora acepta override_stake y lo usa para poblar
stake/retorno de cada combo, igual que la rama no-legacy.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from edge_calculator import GATE_VERSION


class TestLiveStakeLegacyFix(unittest.TestCase):

    def _edge_report_fixture(self, tmpdir):
        edge = {
            "metadata": {"gate_version": GATE_VERSION},
            "apostar": [],
            "watchlist": [
                {
                    "favorito_predicho": "Jugador Uno",
                    "cuota_favorito": 2.10,
                    "edge_pct": "12%",
                    "tier": "atp250",
                    "superficie": "hard",
                    "partido": "Jugador Uno vs Rival A",
                    "p_modelo": 0.55,
                    "kelly_kl": 0.0,
                },
                {
                    "favorito_predicho": "Jugador Dos",
                    "cuota_favorito": 2.30,
                    "edge_pct": "11%",
                    "tier": "atp250",
                    "superficie": "hard",
                    "partido": "Jugador Dos vs Rival B",
                    "p_modelo": 0.54,
                    "kelly_kl": 0.0,
                },
                {
                    "favorito_predicho": "Jugador Tres",
                    "cuota_favorito": 1.95,
                    "edge_pct": "10%",
                    "tier": "atp250",
                    "superficie": "hard",
                    "partido": "Jugador Tres vs Rival C",
                    "p_modelo": 0.53,
                    "kelly_kl": 0.0,
                },
            ],
        }
        path = Path(tmpdir) / "edge_report_test.json"
        path.write_text(json.dumps(edge), encoding="utf-8")
        return str(path)

    def _fake_outcomes(self):
        outcomes_map = {"dummy": {"outcome_id": "1", "odds": 2.0}}
        started_map = {}
        return outcomes_map, started_map

    def _fake_find_outcome(self, jugador, cuota, outcomes_map, started_map, **kwargs):
        oc = {"outcome_id": f"id_{jugador.replace(' ', '_')}", "odds": cuota}
        return oc, None

    def test_legacy_sin_override_stake_queda_en_cero(self):
        """Comportamiento previo al fix: sin override_stake, stake sigue en 0."""
        from betplay_combo_builder import _build_live_combos_legacy

        with tempfile.TemporaryDirectory() as tmpdir:
            edge_path = self._edge_report_fixture(tmpdir)
            with patch("betplay_combo_builder.fetch_kambi_outcomes", side_effect=self._fake_outcomes), \
                 patch("betplay_combo_builder.find_outcome", side_effect=self._fake_find_outcome), \
                 patch("betplay_combo_builder._save_betslip_index", return_value=""):
                combos, metadata = _build_live_combos_legacy(
                    piernas_min=3, piernas_max=3, top_n=4, min_cuota=1.50,
                    edge_file=edge_path, strategy="balanced",
                )

        self.assertTrue(combos, "debe generar al menos 1 combo con 3 picks disponibles")
        for c in combos:
            self.assertEqual(c["stake"], 0)
            self.assertEqual(c["retorno"], 0)

    def test_legacy_con_override_stake_puebla_stake_real(self):
        """Fix: override_stake=5000 llega a los combos generados en modo legacy."""
        from betplay_combo_builder import _build_live_combos_legacy

        with tempfile.TemporaryDirectory() as tmpdir:
            edge_path = self._edge_report_fixture(tmpdir)
            with patch("betplay_combo_builder.fetch_kambi_outcomes", side_effect=self._fake_outcomes), \
                 patch("betplay_combo_builder.find_outcome", side_effect=self._fake_find_outcome), \
                 patch("betplay_combo_builder._save_betslip_index", return_value=""):
                combos, metadata = _build_live_combos_legacy(
                    piernas_min=3, piernas_max=3, top_n=4, min_cuota=1.50,
                    edge_file=edge_path, strategy="balanced",
                    override_stake=5000,
                )

        self.assertTrue(combos, "debe generar al menos 1 combo con 3 picks disponibles")
        for c in combos:
            self.assertEqual(c["stake"], 5000)
            self.assertAlmostEqual(c["retorno"], round(5000 * c["cuota_combo"], 0))

    def test_build_live_combos_propaga_override_stake_a_legacy(self):
        """build_live_combos() (caller real) debe pasar override_stake al fallback
        cuando el trader_plan del día existe pero con cobertura=[] (0 APOSTAR,
        escenario D148-01 real de 2026-07-31) — la causa raíz del bug."""
        from betplay_combo_builder import build_live_combos

        with tempfile.TemporaryDirectory() as tmpdir:
            edge_path = self._edge_report_fixture(tmpdir)
            # Plan vacío D148-01: existe pero cobertura=[] → merged_cobertura queda
            # vacío tras el merge → build_live_combos debe caer a _build_live_combos_legacy.
            plan_path = Path(tmpdir) / "trader_plan_test.json"
            plan_path.write_text(json.dumps({
                "metadata": {"bankroll": 125000, "torneo_tipo": "atp250"},
                "individuales": [],
                "cobertura": [],
            }), encoding="utf-8")

            with patch("betplay_combo_builder._planes_frescos", return_value=[plan_path]), \
                 patch("betplay_combo_builder.fetch_kambi_outcomes", side_effect=self._fake_outcomes), \
                 patch("betplay_combo_builder.find_outcome", side_effect=self._fake_find_outcome), \
                 patch("betplay_combo_builder._save_betslip_index", return_value=""):
                combos, metadata = build_live_combos(
                    piernas_min=3, piernas_max=3, top_n=4, min_cuota=1.50,
                    edge_file=edge_path, strategy="balanced",
                    override_stake=5000,
                )

        self.assertTrue(combos, "debe caer a legacy y generar combos desde edge_report")
        self.assertTrue(all(c["stake"] == 5000 for c in combos))


if __name__ == "__main__":
    unittest.main()
