"""
REGLA-T53 — Nodo-156 D156-03: _apellido() en rival_value_betslip.py descartaba
el nombre completo cuando el formato es 'Nombre I.' (inicial al final),
porque tomaba ciegamente el último token ('Baris O.' -> 'o').

Bug real 2026-07-31: 4/4 candidatos RIVAL VALUE del día quedaron
'Stake:$0 SIN KAMBI' porque rival_map (indexado por apellido real de Kambi)
nunca podía matchear una letra suelta.

Fix: _apellido() ahora recorre los tokens desde el final y descarta los que
sean iniciales (<=2 chars), igual que games_signal_calculator.py (D126-01),
live_desk.py y betplay_combo_builder.py (D154-04).
"""

from rival_value_betslip import _apellido


class TestApellidoInicialTrailingFix:

    def test_nombre_con_inicial_final_devuelve_apellido_real(self):
        assert _apellido("Baris O.") == "baris"
        assert _apellido("Bennani K.") == "bennani"
        assert _apellido("Monday J.") == "monday"
        assert _apellido("Feldbausch K.") == "feldbausch"

    def test_nombre_simple_sin_inicial_sigue_funcionando(self):
        assert _apellido("Musetti L.") == "musetti"
        assert _apellido("Samsonova L.") == "samsonova"

    def test_nombre_de_una_sola_palabra(self):
        assert _apellido("Djokovic") == "djokovic"

    def test_nombre_vacio_no_revienta(self):
        assert _apellido("") == ""

    def test_solo_inicial_no_revienta_usa_primer_token(self):
        # caso degenerado: ningún token >2 chars -> fallback al primero
        assert _apellido("O. K.") == "o"
