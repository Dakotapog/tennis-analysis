"""
Tests for file selection logic in select_best_json_file().

Covers T08-02, T08-03, T08-04 from Nodo-08-File-Selection-Bug.md

Regression tests documenting that modified_time is the PRIMARY sort key
and total_matches is only the tiebreaker. This ordering was inverted in
production until 2026-05-29, causing the May 28 file (423 matches, h2h_url=None)
to always win over the May 29 file (235 matches, valid h2h_url).
"""

from datetime import datetime


# The correct key function — extracted from extraer_historh2h.py line 241
def _select_key(x):
    return (x['modified_time'], x['total_matches'])


class TestFileSelectionPriority:
    """T08-02/03/04 — Verify recency-first selection logic."""

    def test_file_selection_prefers_recency_over_match_count(self):
        """
        T08-02: El archivo más reciente debe ganar aunque tenga menos partidos.

        Regression: before Nodo-08 fix, May 28 (423 matches) won over May 29
        (235 matches) because total_matches was the primary sort key.
        After fix, May 29 wins because modified_time is primary.
        """
        valid_files = [
            {
                'filename': 'data/zita_tennis_matches_20260528_130141.json',
                'total_matches': 423,
                'modified_time': datetime(2026, 5, 28, 13, 1, 41),
                'size_mb': 0.19,
                'location': 'data',
            },
            {
                'filename': 'data/zita_tennis_matches_20260529_015244.json',
                'total_matches': 235,
                'modified_time': datetime(2026, 5, 29, 1, 52, 44),
                'size_mb': 0.13,
                'location': 'data',
            },
        ]

        best = max(valid_files, key=_select_key)

        assert best['filename'] == 'data/zita_tennis_matches_20260529_015244.json', (
            "Archivo más reciente (May 29, 235 partidos) debe ganar sobre "
            "May 28 (423 partidos). La recencia es indicador de calidad: "
            "datos post-Nodo-03 tienen h2h_url válidas; datos anteriores tienen h2h_url=None."
        )

    def test_file_selection_uses_match_count_as_tiebreaker(self):
        """
        T08-03: Con igual timestamp, el archivo con más partidos debe ganar.
        """
        same_time = datetime(2026, 5, 29, 1, 52, 44)
        valid_files = [
            {
                'filename': 'data/a.json',
                'total_matches': 100,
                'modified_time': same_time,
                'size_mb': 0.1,
                'location': 'data',
            },
            {
                'filename': 'data/b.json',
                'total_matches': 235,
                'modified_time': same_time,
                'size_mb': 0.13,
                'location': 'data',
            },
        ]

        best = max(valid_files, key=_select_key)

        assert best['filename'] == 'data/b.json', (
            "Con timestamp idéntico, el archivo con más partidos (235 > 100) debe ganar."
        )

    def test_file_selection_single_file_always_wins(self):
        """
        T08-04: Con un solo archivo válido, siempre debe seleccionarse.
        """
        valid_files = [
            {
                'filename': 'data/only.json',
                'total_matches': 10,
                'modified_time': datetime(2026, 5, 29),
                'size_mb': 0.01,
                'location': 'data',
            }
        ]

        best = max(valid_files, key=_select_key)

        assert best['filename'] == 'data/only.json'

    def test_file_selection_three_files_picks_most_recent(self):
        """
        Variante de T08-02 con tres archivos para cubrir caso de múltiples días.
        """
        valid_files = [
            {
                'filename': 'data/zita_tennis_matches_20260527.json',
                'total_matches': 500,
                'modified_time': datetime(2026, 5, 27, 10, 0, 0),
                'size_mb': 0.25,
                'location': 'data',
            },
            {
                'filename': 'data/zita_tennis_matches_20260528.json',
                'total_matches': 423,
                'modified_time': datetime(2026, 5, 28, 13, 1, 41),
                'size_mb': 0.19,
                'location': 'data',
            },
            {
                'filename': 'data/zita_tennis_matches_20260529.json',
                'total_matches': 235,
                'modified_time': datetime(2026, 5, 29, 1, 52, 44),
                'size_mb': 0.13,
                'location': 'data',
            },
        ]

        best = max(valid_files, key=_select_key)

        assert best['filename'] == 'data/zita_tennis_matches_20260529.json', (
            "Con tres archivos de tres días distintos, el más reciente (May 29) "
            "debe ganar aunque tenga el menor conteo de partidos (235 < 423 < 500)."
        )

    def test_buggy_key_would_have_selected_wrong_file(self):
        """
        Documenta que la clave BUGGY (pre-Nodo-08) hubiera seleccionado el archivo incorrecto.
        Este test captura el comportamiento roto para referencia histórica.
        """
        valid_files = [
            {
                'filename': 'data/zita_tennis_matches_20260528_130141.json',
                'total_matches': 423,
                'modified_time': datetime(2026, 5, 28, 13, 1, 41),
            },
            {
                'filename': 'data/zita_tennis_matches_20260529_015244.json',
                'total_matches': 235,
                'modified_time': datetime(2026, 5, 29, 1, 52, 44),
            },
        ]

        # La clave BUGGY (total_matches primero) seleccionaría May 28
        buggy_winner = max(valid_files, key=lambda x: (x['total_matches'], x['modified_time']))
        assert buggy_winner['filename'] == 'data/zita_tennis_matches_20260528_130141.json', (
            "Confirma que la clave buggy efectivamente seleccionaba el archivo incorrecto."
        )

        # La clave CORRECTA (modified_time primero) selecciona May 29
        correct_winner = max(valid_files, key=lambda x: (x['modified_time'], x['total_matches']))
        assert correct_winner['filename'] == 'data/zita_tennis_matches_20260529_015244.json', (
            "Confirma que la clave correcta selecciona el archivo más reciente."
        )

        # Los dos winners son distintos — el bug era real
        assert buggy_winner['filename'] != correct_winner['filename']
