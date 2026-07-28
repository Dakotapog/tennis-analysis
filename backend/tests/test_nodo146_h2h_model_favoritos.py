"""
REGLA-T53 — Tests Nodo-146: H2H_MODEL universe en favoritos_combo_builder.

D146: _find_latest_h2h() + _leer_h2h_favoritos() leen h2h_results_enhanced
directamente para obtener picks con cuota < 1.50 que edge_calculator descarta
por REGLA-HF-1 pero son válidos como piernas de FAVORITOS_COMPUESTOS (LEG_MIN=1.15).

7 tests:
  test_find_latest_h2h_today
  test_find_latest_h2h_no_file
  test_h2h_favoritos_basic
  test_h2h_favoritos_dedup
  test_h2h_favoritos_cuota_out_of_range
  test_h2h_favoritos_low_confidence
  test_h2h_favoritos_timing_guard
"""

import json
import os
import tempfile
import unittest
from datetime import date
from unittest.mock import patch


# Importar funciones reales del módulo (REGLA-T53: nunca hardcodear fórmula)
from favoritos_combo_builder import (
    _find_latest_h2h,
    _leer_h2h_favoritos,
    LEG_MIN_CUOTA,
    LEG_MAX_CUOTA,
)


def _make_partido(jugador1="Kuramochi K.", jugador2="Smith J.",
                  cuota1=1.25, cuota2=3.50,
                  confidence=0.677, favored="Kuramochi K.",
                  torneo="Vancouver", tipo_cancha="hard",
                  hora=None):  # None → timing guard no aplica (para tests de lógica)
    """Helper: crea un partido h2h con estructura mínima válida."""
    return {
        "jugador1": jugador1,
        "jugador2": jugador2,
        "cuota1": cuota1,
        "cuota2": cuota2,
        "torneo_nombre": torneo,
        "tipo_cancha": tipo_cancha,
        "hora": hora,
        "ranking_analysis": {
            "prediction": {
                "favored_player": favored,
                "confidence": confidence,
            }
        },
    }


def _make_h2h_file(partidos):
    """Helper: escribe h2h_results_enhanced en archivo temporal."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump({"partidos": partidos, "metadata": {}}, f)
    f.close()
    return f.name


class TestFindLatestH2H(unittest.TestCase):

    def test_find_latest_h2h_today(self):
        """Con archivo de hoy disponible, retorna el más reciente."""
        today = date.today().strftime('%Y%m%d')
        fake_files = [
            f"reports/h2h_results_enhanced_{today}_100000.json",
            f"reports/h2h_results_enhanced_{today}_120000.json",
        ]
        with patch("glob.glob", return_value=fake_files):
            result = _find_latest_h2h()
        # sorted(...) → último = 120000
        self.assertEqual(result, fake_files[-1])

    def test_find_latest_h2h_no_file(self):
        """Sin archivos de hoy → retorna None (no explota)."""
        with patch("glob.glob", return_value=[]):
            result = _find_latest_h2h()
        self.assertIsNone(result)


class TestLeerH2HFavoritos(unittest.TestCase):

    def test_h2h_favoritos_basic(self):
        """Partido válido → candidato H2H_MODEL con fuente correcta."""
        # hora="23:59" + datetime mock 09:00 → timing guard no dispara
        p = _make_partido(cuota1=1.25, cuota2=3.50, confidence=0.677,
                          favored="Kuramochi K.", hora="23:59")
        path = _make_h2h_file([p])
        try:
            import favoritos_combo_builder as fcb

            class _FakeDT:
                @staticmethod
                def now(tz=None):
                    class _T:
                        hour = 9
                        minute = 0
                    return _T()

            with patch("favoritos_combo_builder.datetime", _FakeDT):
                result = fcb._leer_h2h_favoritos(path, set())
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["fuente"], "H2H_MODEL")
            self.assertAlmostEqual(result[0]["cuota_favorito"], 1.25)
            self.assertAlmostEqual(result[0]["p_modelo"], 0.677, places=2)
        finally:
            os.unlink(path)

    def test_h2h_favoritos_dedup(self):
        """Candidato ya en edge_picks_set → se omite (sin duplicados)."""
        p = _make_partido(jugador1="Boulais F.", cuota1=1.13, cuota2=5.00,
                          confidence=0.787, favored="Boulais F.")
        path = _make_h2h_file([p])
        try:
            # El favorito ya está en el set del edge_report
            result = _leer_h2h_favoritos(path, {"boulais f"})
            self.assertEqual(len(result), 0, "Duplicado debe ser omitido")
        finally:
            os.unlink(path)

    def test_h2h_favoritos_cuota_out_of_range(self):
        """cuota_favorito < LEG_MIN_CUOTA → descartado."""
        p = _make_partido(cuota1=1.05, cuota2=8.00, confidence=0.90,
                          favored="Djokovic N.")
        path = _make_h2h_file([p])
        try:
            result = _leer_h2h_favoritos(path, set())
            self.assertEqual(len(result), 0,
                             f"cuota 1.05 < LEG_MIN_CUOTA {LEG_MIN_CUOTA} debe descartarse")
        finally:
            os.unlink(path)

    def test_h2h_favoritos_cuota_above_max(self):
        """cuota_favorito > LEG_MAX_CUOTA → descartado."""
        # Kuramochi es el favorito predicho pero su cuota es 2.50 > 2.10
        p = _make_partido(cuota1=2.50, cuota2=1.55, confidence=0.60,
                          favored="Kuramochi K.", jugador1="Kuramochi K.", jugador2="Smith J.")
        path = _make_h2h_file([p])
        try:
            result = _leer_h2h_favoritos(path, set())
            self.assertEqual(len(result), 0,
                             f"cuota 2.50 > LEG_MAX_CUOTA {LEG_MAX_CUOTA} debe descartarse")
        finally:
            os.unlink(path)

    def test_h2h_favoritos_low_confidence(self):
        """confidence < 0.55 → descartado."""
        p = _make_partido(confidence=0.52, favored="Kuramochi K.")
        path = _make_h2h_file([p])
        try:
            result = _leer_h2h_favoritos(path, set())
            self.assertEqual(len(result), 0,
                             "confidence 0.52 < 0.55 debe descartarse")
        finally:
            os.unlink(path)

    def test_h2h_favoritos_timing_guard(self):
        """Partido con hora pasada >15min → timing guard lo descarta."""
        # hora 08:00, simulamos que ahora son 10:30 Colombia (150 min)
        p = _make_partido(hora="08:00", confidence=0.70)
        path = _make_h2h_file([p])
        try:
            with patch("favoritos_combo_builder._find_latest_h2h"):
                # Simular _ahora_min = 630 (10:30 Colombia)
                # Hora 08:00 → inicio_min=480; 630 > 480+15=495 → skip
                import favoritos_combo_builder as fcb
                orig = fcb.datetime

                class FakeDateTime:
                    @staticmethod
                    def now(tz=None):
                        class FakeTime:
                            hour = 10
                            minute = 30
                        return FakeTime()

                with patch("favoritos_combo_builder.datetime", FakeDateTime):
                    result = fcb._leer_h2h_favoritos(path, set())
                self.assertEqual(len(result), 0,
                                 "Partido a las 08:00 con ahora=10:30 debe ser skipeado")
        finally:
            os.unlink(path)

    def test_h2h_favoritos_conf_flag(self):
        """confidence >= 0.60 → STRONG; 0.55-0.59 → MOD."""
        # hora="23:59" + datetime mock 09:00 → timing guard no dispara
        p_strong = _make_partido(confidence=0.65, favored="Kuramochi K.", hora="23:59")
        p_mod = _make_partido(confidence=0.57, favored="Isomura K.",
                              jugador1="Isomura K.", jugador2="Rival B.",
                              hora="23:59")
        path = _make_h2h_file([p_strong, p_mod])
        try:
            import favoritos_combo_builder as fcb

            class _FakeDT:
                @staticmethod
                def now(tz=None):
                    class _T:
                        hour = 9
                        minute = 0
                    return _T()

            with patch("favoritos_combo_builder.datetime", _FakeDT):
                result = fcb._leer_h2h_favoritos(path, set())
            flags = {r["favorito"]: r["confidence_flag"] for r in result}
            self.assertEqual(flags.get("Kuramochi K."), "STRONG")
            self.assertEqual(flags.get("Isomura K."), "MOD")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
