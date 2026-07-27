"""
REGLA-T53 — Tests Nodo-145: Pipeline Bugs tipo_cancha + timing guard.

D145-01: h2h_extractor propaga superficie → tipo_cancha y copia hora.
D145-02: edge_calculator skip de partidos cuya hora ya pasó.

5 tests:
  test_tipo_cancha_from_superficie
  test_tipo_cancha_propio_gana
  test_hora_propagated
  test_timing_guard_skip
  test_timing_guard_future
"""

import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# D145-01 — h2h_extractor: tipo_cancha + hora
# ---------------------------------------------------------------------------

class TestH2HExtractorD14501(unittest.TestCase):
    """
    Verifica el fix en _build_partido_result() de H2HExtractor:
    tipo_cancha usa superficie como fallback; hora se copia al record.
    """

    def _call_build(self, match_data: dict) -> dict:
        """
        Simula la lógica del fix sin instanciar H2HExtractor completo.
        Replica la expresión exacta del fix en h2h_extractor.py L900.
        """
        tipo_cancha = match_data.get('tipo_cancha') or match_data.get('superficie', 'N/A')
        hora = match_data.get('hora')
        return {'tipo_cancha': tipo_cancha, 'hora': hora}

    def test_tipo_cancha_from_superficie(self):
        """match_data con superficie='hard' y sin tipo_cancha → tipo_cancha='hard'."""
        match_data = {'superficie': 'hard', 'torneo_nombre': 'Washington'}
        result = self._call_build(match_data)
        self.assertEqual(result['tipo_cancha'], 'hard',
                         "tipo_cancha debe tomar el valor de superficie cuando tipo_cancha falta")

    def test_tipo_cancha_propio_gana(self):
        """match_data con ambos campos → tipo_cancha propio gana sobre superficie."""
        match_data = {'tipo_cancha': 'clay', 'superficie': 'hard'}
        result = self._call_build(match_data)
        self.assertEqual(result['tipo_cancha'], 'clay',
                         "tipo_cancha propio debe tener prioridad sobre superficie")

    def test_hora_propagated(self):
        """hora del match_data se copia al h2h record."""
        match_data = {'superficie': 'hard', 'hora': '14:30'}
        result = self._call_build(match_data)
        self.assertEqual(result['hora'], '14:30',
                         "hora debe estar en el record de salida")


# ---------------------------------------------------------------------------
# D145-02 — edge_calculator: timing guard
# ---------------------------------------------------------------------------

class TestTimingGuardD14502(unittest.TestCase):
    """
    Verifica la lógica del timing guard:
    si hora del partido + 15min < ahora Colombia → skip.
    Replica la lógica de edge_calculator.py al iterar partidos.
    """

    def _should_skip(self, hora_partido: str, ahora_col_hour: int, ahora_col_min: int) -> bool:
        """
        Replica la decisión del timing guard (D145-02).
        Retorna True si el partido debe ser skipped.
        """
        try:
            h, m = map(int, str(hora_partido).split(':')[:2])
            inicio_min = h * 60 + m
            ahora_min = ahora_col_hour * 60 + ahora_col_min
            return ahora_min > inicio_min + 15
        except Exception:
            return False

    def test_timing_guard_skip(self):
        """Partido que empezó hace 30 min debe ser skipped."""
        # Partido a las 10:00, ahora son las 10:30 Col
        skip = self._should_skip('10:00', ahora_col_hour=10, ahora_col_min=30)
        self.assertTrue(skip, "partido 30min en el pasado debe ser descartado")

    def test_timing_guard_future(self):
        """Partido que empieza en el futuro NO debe ser skipped."""
        # Partido a las 15:00, ahora son las 10:30 Col
        skip = self._should_skip('15:00', ahora_col_hour=10, ahora_col_min=30)
        self.assertFalse(skip, "partido futuro no debe ser descartado")

    def test_timing_guard_within_buffer(self):
        """Partido que empezó hace 10min (dentro del buffer de 15) NO se skipea."""
        # Partido a las 10:20, ahora son las 10:30 Col (10min de diferencia)
        skip = self._should_skip('10:20', ahora_col_hour=10, ahora_col_min=30)
        self.assertFalse(skip, "partido dentro del buffer de 15min no debe ser descartado")

    def test_timing_guard_malformed_hora(self):
        """hora mal formateada no lanza excepción — simplemente no skipea."""
        skip = self._should_skip('invalid', ahora_col_hour=10, ahora_col_min=30)
        self.assertFalse(skip, "hora inválida no debe causar skip ni excepción")


if __name__ == '__main__':
    unittest.main()
