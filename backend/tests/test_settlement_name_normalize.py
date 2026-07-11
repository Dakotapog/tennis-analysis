"""
tests/test_settlement_name_normalize.py — _normalize_name_match (Tier 3a en settle)

REGLA-T53: invoca función real del módulo, nunca hardcodea la lógica.
Casos basados en patrones reales del proyecto: acentos, apellidos compuestos con guión,
diferencias FlashScore full-name vs pick apellido solo.
"""
import pytest
from shadow_book import _normalize_name_match


class TestNormalizeNameMatchAccents:
    """Acentos — normalize_player_name convierte á/é/ó/ü antes de comparar."""

    def test_acento_en_pick(self):
        assert _normalize_name_match("Gomez", "Gómez") is True

    def test_acento_en_candidato(self):
        assert _normalize_name_match("Gómez", "Gomez") is True

    def test_ambos_acentuados_distintos_grafias(self):
        assert _normalize_name_match("García", "Garcia") is True

    def test_u_dieresis(self):
        assert _normalize_name_match("Muller", "Müller") is True


class TestNormalizeNameMatchDash:
    """Apellidos compuestos con guión — normalize convierte '-' → espacio."""

    def test_guion_en_candidato(self):
        # Nodo-80: "Dedura-Palomero" (Kambi) vs "Dedura Palomero" (FlashScore)
        assert _normalize_name_match("Dedura-Palomero", "Dedura Palomero") is True

    def test_guion_en_pick(self):
        assert _normalize_name_match("Dedura Palomero", "Dedura-Palomero") is True

    def test_ambos_guionados(self):
        assert _normalize_name_match("Vives-Marcos", "Vives-Marcos") is True


class TestNormalizeNameMatchSubstring:
    """Substring: FlashScore da nombre completo, pick solo apellido (o viceversa)."""

    def test_pick_apellido_en_nombre_completo_fs(self):
        # FlashScore: "Leyton Rivera" — pick favorito: "Rivera"
        assert _normalize_name_match("Leyton Rivera", "Rivera") is True

    def test_nombre_completo_fs_en_pick(self):
        assert _normalize_name_match("Rivera", "Leyton Rivera") is True

    def test_apellido_compuesto_partial(self):
        # FlashScore: "Pedro Vives Marcos" — pick: "Vives Marcos"
        assert _normalize_name_match("Pedro Vives Marcos", "Vives Marcos") is True

    def test_acento_mas_substring(self):
        assert _normalize_name_match("Pedro García", "García") is True


class TestNormalizeNameMatchNegative:
    """Falsos positivos: nombres distintos no deben hacer match."""

    def test_nombres_distintos(self):
        assert _normalize_name_match("Michnev", "Rivera") is False

    def test_candidato_vacio(self):
        assert _normalize_name_match("", "Rivera") is False

    def test_pick_vacio(self):
        assert _normalize_name_match("Rivera", "") is False

    def test_nombres_cortos_sin_relacion(self):
        assert _normalize_name_match("Martinez", "Garcia") is False

    def test_substring_corto_no_matchea(self):
        # "Li" (2 chars) < guardia len>=4 — no debe matchear "Li Na"
        assert _normalize_name_match("Li", "Li Na") is False

    def test_substring_de_3_chars_no_matchea(self):
        # "Lee" (3 chars) < guardia len>=4
        assert _normalize_name_match("Lee", "Lee Smith") is False
