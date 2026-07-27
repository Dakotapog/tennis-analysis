"""
REGLA-T53 — Tests Nodo-144: Trazabilidad de Estrategia en Shadow Book.

D144-01: _build_record() incluye campo 'strategy' top-level.
D144-02: tag_strategy() actualiza registros existentes.
D144-04: log_pick() crea registro individual.

7 tests:
  test_build_record_default_sin_tag
  test_build_record_with_strategy
  test_tag_strategy_updates_record
  test_tag_strategy_no_overwrite
  test_tag_strategy_skip_session_meta
  test_log_pick_creates_record
  test_log_pick_idempotent
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import shadow_book as sb


def _make_minimal_pick(**kwargs) -> dict:
    """Pick mínimo válido para _build_record()."""
    base = {
        "favorito_predicho": "Jugador A",
        "rival":             "Rival B",
        "partido":           "Jugador A vs Rival B",
        "torneo":            "Roland Garros",
        "superficie":        "clay",
        "pick_status":       "APOSTAR",
        "confidence":        65.0,
        "cuota_favorito":    1.80,
    }
    base.update(kwargs)
    return base


class TestBuildRecordStrategyField(unittest.TestCase):
    """D144-01: campo 'strategy' en _build_record()."""

    def test_build_record_default_sin_tag(self):
        """pick sin campo strategy → top-level strategy='SIN_TAG'."""
        pick = _make_minimal_pick()
        self.assertNotIn("strategy", pick)

        rec = sb._build_record(pick, "2026-07-25")

        self.assertIsNotNone(rec, "_build_record retornó None")
        self.assertIn("strategy", rec, "campo 'strategy' ausente en registro")
        self.assertEqual(rec["strategy"], "SIN_TAG")

    def test_build_record_with_strategy(self):
        """pick con strategy='CORE' → top-level preserva el valor, NO queda SIN_TAG."""
        pick = _make_minimal_pick(strategy="CORE")

        rec = sb._build_record(pick, "2026-07-25")

        self.assertIsNotNone(rec)
        self.assertEqual(rec["strategy"], "CORE",
                         "strategy del pick no se preservó en top-level")


class TestTagStrategy(unittest.TestCase):
    """D144-02: tag_strategy() — función de update sin tocar pick_snapshot."""

    def _make_sb_record(self, sb_id: str, nombre: str, strategy: str = "SIN_TAG") -> dict:
        return {
            "sb_id":      sb_id,
            "logged_at":  "2026-07-25T10:00:00",
            "match_key":  nombre.lower().replace(" ", "_"),
            "strategy":   strategy,
            "pick_snapshot": {
                "favorito_predicho": nombre,
                "rival":             "Rival X",
                "partido":           f"{nombre} vs Rival X",
                "torneo":            "Wimbledon",
                "superficie":        "grass",
                "pick_status":       "APOSTAR",
                "confidence":        60.0,
                "cuota_favorito":    2.10,
            },
        }

    def _write_records(self, tmpdir: str, fecha: str, records: list) -> str:
        path = os.path.join(tmpdir, f"sb_{fecha}.jsonl")
        with open(path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return path

    def test_tag_strategy_updates_record(self):
        """tag_strategy() actualiza registro con strategy=SIN_TAG al valor correcto."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fecha = "2026-07-25"
            rec = self._make_sb_record("sb_001", "Federer R.")
            self._write_records(tmpdir, fecha, [rec])

            with patch.object(sb, "SHADOW_DIR", tmpdir):
                n = sb.tag_strategy(fecha, ["Federer R."], "CORE")

            self.assertEqual(n, 1, "debió retornar 1 pick tageado")

            # Verificar que el archivo fue modificado
            path = os.path.join(tmpdir, f"sb_{fecha}.jsonl")
            with open(path) as f:
                saved = json.loads(f.read().strip())
            self.assertEqual(saved["strategy"], "CORE")
            self.assertIn("strategy_tagged_at", saved)
            # pick_snapshot inmutable
            self.assertNotIn("strategy", saved["pick_snapshot"])

    def test_tag_strategy_no_overwrite(self):
        """tag_strategy() NO sobrescribe si strategy ya está asignada (!=SIN_TAG)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fecha = "2026-07-25"
            rec = self._make_sb_record("sb_002", "Nadal R.", strategy="CORE")
            self._write_records(tmpdir, fecha, [rec])

            with patch.object(sb, "SHADOW_DIR", tmpdir):
                n = sb.tag_strategy(fecha, ["Nadal R."], "COBERTURA")

            self.assertEqual(n, 0, "no debió sobrescribir un tag ya asignado")

            path = os.path.join(tmpdir, f"sb_{fecha}.jsonl")
            with open(path) as f:
                saved = json.loads(f.read().strip())
            self.assertEqual(saved["strategy"], "CORE",
                             "el tag original fue sobrescrito incorrectamente")

    def test_tag_strategy_skip_session_meta(self):
        """tag_strategy() no toca registros _type='session_meta'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fecha = "2026-07-25"
            meta = {
                "sb_id":   "META_2026-07-25",
                "_type":   "session_meta",
                "strategy": "SIN_TAG",
                "logged_at": "2026-07-25T10:00:00",
            }
            self._write_records(tmpdir, fecha, [meta])

            with patch.object(sb, "SHADOW_DIR", tmpdir):
                n = sb.tag_strategy(fecha, ["cualquier_nombre"], "CORE")

            self.assertEqual(n, 0, "session_meta no debe ser tageado")


class TestLogPick(unittest.TestCase):
    """D144-04: log_pick() registra un pick individual."""

    def _minimal_snapshot(self, nombre: str) -> dict:
        return {
            "favorito_predicho": nombre,
            "rival":             "Rival Z",
            "partido":           f"{nombre} vs Rival Z",
            "torneo":            "US Open",
            "superficie":        "hard",
            "pick_status":       "APOSTAR",
            "confidence":        62.0,
        }

    def test_log_pick_creates_record(self):
        """log_pick() crea registro en JSONL y retorna sb_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(sb, "SHADOW_DIR", tmpdir):
                sid = sb.log_pick(
                    fecha="2026-07-25",
                    jugador="Djokovic N.",
                    cuota=1.60,
                    pick_snapshot=self._minimal_snapshot("Djokovic N."),
                )

            self.assertIsNotNone(sid, "log_pick debe retornar sb_id")

            path = os.path.join(tmpdir, "sb_2026-07-25.jsonl")
            self.assertTrue(os.path.exists(path), "JSONL no fue creado")

            with open(path) as f:
                saved = json.loads(f.read().strip())

            self.assertEqual(saved["sb_id"], sid)
            # D144-01: campo strategy presente
            self.assertIn("strategy", saved)

    def test_log_pick_idempotent(self):
        """Llamar log_pick() dos veces con el mismo pick = upsert, sin duplicado."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snap = self._minimal_snapshot("Alcaraz C.")
            with patch.object(sb, "SHADOW_DIR", tmpdir):
                sid1 = sb.log_pick("2026-07-25", "Alcaraz C.", 1.75, snap)
                sid2 = sb.log_pick("2026-07-25", "Alcaraz C.", 1.75, snap)

            self.assertEqual(sid1, sid2, "el mismo pick debe generar el mismo sb_id")

            path = os.path.join(tmpdir, "sb_2026-07-25.jsonl")
            with open(path) as f:
                lines = [l for l in f.readlines() if l.strip()]
            self.assertEqual(len(lines), 1, "no debe haber duplicados en JSONL")


if __name__ == "__main__":
    unittest.main()
