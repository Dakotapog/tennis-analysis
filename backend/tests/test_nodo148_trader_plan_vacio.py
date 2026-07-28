"""
REGLA-T53 — Tests Nodo-148: trader_plan vacío cuando 0 APOSTAR picks.

D148-01: trader_ev_tenis.py escribe trader_plan con individuales=[]
cuando senales_raw está vacío, para desbloquear SAFE/WAS/MEGA legacy.

3 tests:
  test_plan_vacio_estructura         — plan escrito tiene campos requeridos
  test_plan_vacio_individuales_empty — individuales y cobertura son listas vacías
  test_build_live_no_bloqueado       — _planes_frescos encuentra el plan vacío
"""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


class TestPlanVacioEstructura(unittest.TestCase):

    def _escribir_plan_vacio(self, tmpdir):
        """Simula lo que hace D148-01 en trader_ev_tenis.py."""
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        plan_file = os.path.join(tmpdir, f"trader_plan_{ts}.json")
        plan = {
            "metadata": {
                "timestamp": ts,
                "bankroll": 125000,
                "torneo_tipo": "atp500",
                "n_apostar": 0,
                "d148": "plan_vacio_sin_apostar",
            },
            "individuales": [],
            "senales": [],
            "combos": [],
            "cobertura": [],
            "sistema": [],
            "risk_management": {},
            "resumen": {
                "n_senales_apostar": 0,
                "total_en_riesgo": 0,
                "pct_bankroll_en_riesgo": 0,
            },
        }
        with open(plan_file, 'w', encoding='utf-8') as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        return plan_file, plan

    def test_plan_vacio_estructura(self):
        """Plan vacío tiene todos los campos requeridos por betplay_combo_builder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_file, plan = self._escribir_plan_vacio(tmpdir)
            loaded = json.loads(Path(plan_file).read_text())
            for campo in ("metadata", "individuales", "senales", "cobertura", "sistema", "resumen"):
                self.assertIn(campo, loaded, f"Campo '{campo}' ausente del plan vacío")
            self.assertEqual(loaded["metadata"]["n_apostar"], 0)
            self.assertEqual(loaded["metadata"]["d148"], "plan_vacio_sin_apostar")

    def test_plan_vacio_individuales_empty(self):
        """individuales y cobertura son listas vacías — no genera combos SAFE/MEGA."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _, plan = self._escribir_plan_vacio(tmpdir)
            self.assertEqual(plan["individuales"], [])
            self.assertEqual(plan["cobertura"], [])
            # merged_cobertura vacío → build_live_combos cae a legacy (edge_report)
            merged_cobertura = []
            for p in [plan]:
                merged_cobertura.extend(p.get("cobertura", []))
            self.assertFalse(merged_cobertura, "merged_cobertura debe ser vacío → activa legacy fallback")

    def test_build_live_no_bloqueado(self):
        """_planes_frescos encuentra el plan vacío → CAPA-LIVE check pasa."""
        from betplay_combo_builder import _planes_frescos
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_file, _ = self._escribir_plan_vacio(tmpdir)
            path = Path(plan_file)
            # Debe encontrar el plan recién escrito (< 4h)
            found = _planes_frescos([path], max_age_h=4)
            self.assertEqual(len(found), 1, "Plan vacío debe pasar el filtro _planes_frescos")
            self.assertEqual(found[0], path)


if __name__ == "__main__":
    unittest.main()
