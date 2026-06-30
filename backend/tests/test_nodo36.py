"""
Tests para Nodo-36 — Fix B (acento Unicode) + Fix C (apellidos cortos)
en scraping/ninja_h2h_parser.py

Importa _strip_accents, _name_tokens, _token_in_kb directamente del módulo
para que la detección de mutación sea real — mutar el parser ROMPE estos tests.

Fix B — Bruno Fernandez / Fernández:
  FlashScore retorna KB "Últimos partidos: Fernández B." con tilde.
  Antes: "fernandez" not in "fernández b." → bloque invertido / historial vacío.
  Después: _strip_accents normaliza ambos lados → match correcto.

Fix C — Jing-Jing Lu / Lan Mi:
  "Lu" tiene len=2 → len(t)>2 la excluía → bloque no identificado.
  "Mi" tiene len=2 → mismo problema.
  Después: len(t)>1 incluye tokens de 2 chars.
  Word-boundary guard: "mi" en "michelsen" → NO match (evita falso positivo).

Detección de mutación (real — usa código de producción):
  T36-01 FALLA si se deshabilita _strip_accents en el parser (Fix B rollback).
  T36-02 FALLA si se revierte len(t)>1 a len(t)>2 en el parser (Fix C rollback).
  T36-03 FALLA si se elimina word-boundary .split() del parser (falso positivo michelsen).
"""
import unicodedata
import pytest

from scraping.ninja_h2h_parser import _strip_accents, _name_tokens, _token_in_kb


# ─────────────────────────────────────────────────────────────────────────────
# Helper local (no importado — lógica de composición trivial)
# ─────────────────────────────────────────────────────────────────────────────

def _tokens_match_kb(tokens, kb: str) -> bool:
    return bool(tokens) and any(_token_in_kb(tok, kb) for tok in tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Fix B — Accent normalization (Bruno Fernandez / Fernández)
# ─────────────────────────────────────────────────────────────────────────────

class TestFixB_AccentNormalization:
    """T36-01: Mutation detection — _strip_accents must normalize accents."""

    def test_strip_accents_basic(self):
        """_strip_accents convierte é→e, á→a, etc."""
        assert _strip_accents('fernández') == 'fernandez'
        assert _strip_accents('Fernández') == 'Fernandez'
        assert _strip_accents('garcia') == 'garcia'   # ya sin acento

    def test_fernandez_token_matches_accented_kb(self):
        """T36-01: 'fernandez' (sin tilde) debe encontrarse en KB con tilde."""
        kb = 'Últimos partidos: Fernández B.'
        tokens = _name_tokens('Bruno Fernandez')   # nombre sin tilde
        assert _tokens_match_kb(tokens, kb), (
            "Fix B ROTO: 'fernandez' no matchea 'Fernández' en KB "
            "(revisar _strip_accents en _name_tokens y _token_in_kb)"
        )

    def test_fernandez_accented_name_matches_unaccented_kb(self):
        """Nombre CON tilde también matchea KB sin tilde (bidireccional)."""
        kb = 'Últimos partidos: Fernandez B.'
        tokens = _name_tokens('Bruno Fernández')   # nombre con tilde
        assert _tokens_match_kb(tokens, kb)

    def test_fernandez_mutation_detection(self):
        """T36-01 FALLA si _strip_accents es deshabilitado.
        Simular el comportamiento PRE-fix (sin normalización)."""
        kb = 'Últimos partidos: Fernández B.'
        # Comportamiento antiguo: comparación directa sin normalizar
        old_tok_in_kb = lambda tok, kb: tok in kb.lower()
        tokens_old = [t.lower() for t in 'Bruno Fernandez'.split() if len(t) > 2]
        pre_fix_result = bool(tokens_old) and any(old_tok_in_kb(tok, kb) for tok in tokens_old)
        # Confirmar que SIN el fix falla
        assert not pre_fix_result, "Precondición: comportamiento antiguo NO matchea"
        # Con el fix (código de producción importado) sí funciona
        tokens_new = _name_tokens('Bruno Fernandez')
        post_fix_result = _tokens_match_kb(tokens_new, kb)
        assert post_fix_result, "Fix B debe resolver el mismatch de acentos"

    def test_other_accented_players(self):
        """Cobertura: otros jugadores con acentos comunes en tenis."""
        cases = [
            ('Nicolás Almagro', 'Últimos partidos: Almagro N.'),
            ('João Sousa', 'Últimos partidos: Sousa J.'),
            ('Márton Fucsovics', 'Últimos partidos: Fucsovics M.'),
        ]
        for name, kb in cases:
            tokens = _name_tokens(name)
            assert _tokens_match_kb(tokens, kb), f"Fix B falla para {name!r} vs {kb!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Fix C — Short surname filter (Jing-Jing Lu, Lan Mi)
# ─────────────────────────────────────────────────────────────────────────────

class TestFixC_ShortSurnames:
    """T36-02 y T36-03: apellidos de 2 chars incluidos, con word-boundary guard."""

    def test_lu_included_in_tokens(self):
        """T36-02: 'Lu' (len=2) debe estar en los tokens."""
        tokens = _name_tokens('Jing-Jing Lu')
        assert 'lu' in tokens, (
            "Fix C ROTO: 'Lu' excluido de tokens "
            "(revisar len(t)>1 en _name_tokens)"
        )

    def test_mi_included_in_tokens(self):
        """T36-02: 'Mi' (len=2) debe estar en los tokens."""
        tokens = _name_tokens('Lan Mi')
        assert 'mi' in tokens, (
            "Fix C ROTO: 'Mi' excluido de tokens "
            "(revisar len(t)>1 en _name_tokens)"
        )

    def test_lu_matches_correct_kb(self):
        """'lu' matchea KB 'Últimos partidos: Lu J.' (word-boundary OK)."""
        kb = 'Últimos partidos: Lu J.'
        tokens = _name_tokens('Jing-Jing Lu')
        assert _tokens_match_kb(tokens, kb)

    def test_mi_matches_correct_kb(self):
        """'mi' matchea KB 'Últimos partidos: Mi L.' (word-boundary OK)."""
        kb = 'Últimos partidos: Mi L.'
        tokens = _name_tokens('Lan Mi')
        assert _tokens_match_kb(tokens, kb)

    def test_fix_c_mutation_pre_fix_lu_missing(self):
        """T36-02 FALLA si se revierte len(t)>1 a len(t)>2 en el parser.
        Simula el comportamiento antiguo con len > 2."""
        pre_fix_tokens = [t.lower() for t in 'Jing-Jing Lu'.split() if len(t) > 2]
        assert 'lu' not in pre_fix_tokens, "Precondición: antiguo filtro excluye 'lu'"
        # Con el nuevo filtro (código de producción importado) sí está
        new_tokens = _name_tokens('Jing-Jing Lu')
        assert 'lu' in new_tokens

    def test_word_boundary_guard_mi_vs_michelsen(self):
        """T36-03: 'mi' NO debe matchear 'Michelsen' (falso positivo).
        Sin word-boundary 'mi' in 'michelsen' = True."""
        kb_michelsen = 'Últimos partidos: Michelsen A.'
        tokens_lan_mi = _name_tokens('Lan Mi')
        assert not _tokens_match_kb(tokens_lan_mi, kb_michelsen), (
            "Fix C ROTO: 'mi' matchea 'michelsen' — word-boundary guard faltante"
        )

    def test_word_boundary_guard_lu_vs_lupescu(self):
        """'lu' NO debe matchear 'Lupescu' u otros nombres con 'lu' embedded."""
        kb = 'Últimos partidos: Lupescu A.'
        tokens = _name_tokens('Jing-Jing Lu')
        assert not _tokens_match_kb(tokens, kb), (
            "word-boundary guard debe prevenir 'lu' in 'lupescu'"
        )

    def test_short_token_word_boundary_uses_split(self):
        """_token_in_kb para tok len<=2 usa .split() no 'in'."""
        assert _token_in_kb('lu', 'Lu J.') is True
        assert _token_in_kb('lu', 'Lupescu A.') is False
        assert _token_in_kb('mi', 'Mi L.') is True
        assert _token_in_kb('mi', 'Michelsen A.') is False


# ─────────────────────────────────────────────────────────────────────────────
# No-regresión — nombres sin acento y apellidos compuestos siguen funcionando
# ─────────────────────────────────────────────────────────────────────────────

class TestNoRegression:
    """T36-04/05: fixes no rompen casos existentes."""

    def test_normal_name_no_accent(self):
        """T36-04: apellido sin acento sigue matcheando normalmente."""
        kb = 'Últimos partidos: Djokovic N.'
        tokens = _name_tokens('Novak Djokovic')
        assert _tokens_match_kb(tokens, kb)

    def test_compound_surname_davidovich(self):
        """T36-05: Davidovich Fokina sigue funcionando (Nodo-34 fix)."""
        kb = 'Últimos partidos: Davidovich A.'
        tokens = _name_tokens('Alejandro Davidovich Fokina')
        assert _tokens_match_kb(tokens, kb)

    def test_compound_surname_second_part(self):
        """T36-05: segundo apellido también matchea."""
        kb = 'Últimos partidos: Fokina A.'
        tokens = _name_tokens('Alejandro Davidovich Fokina')
        assert _tokens_match_kb(tokens, kb)

    def test_single_char_tokens_excluded(self):
        """Tokens de 1 char puro siguen excluidos; 'A.' (len=2) incluido pero
        word-boundary evita falsos positivos contra substrings."""
        tokens = _name_tokens('A. Murray')
        assert 'a' not in tokens   # 'a' sola (len=1) excluida
        assert 'murray' in tokens  # apellido incluido normalmente
        # 'A.' tiene len=2 → incluido con Fix C, pero word-boundary lo contiene
        if 'a.' in tokens:
            assert not _token_in_kb('a.', 'Últimos partidos: Alexandrova E.')

    def test_na_name_returns_empty(self):
        """Nombre N/A retorna lista vacía."""
        assert _name_tokens('N/A') == []

    def test_fernandez_no_accent_vs_no_accent_kb(self):
        """Caso sin tilde en ambos lados sigue funcionando."""
        kb = 'Últimos partidos: Fernandez B.'
        tokens = _name_tokens('Bruno Fernandez')
        assert _tokens_match_kb(tokens, kb)

    def test_risk_note_andy_nguyen_ambiguity(self):
        """Riesgo documentado: Andy Nguyen y Avery Nguyen comparten 'nguyen'.
        Ambigüedad conocida — resuelta por match_id, no por nombre."""
        kb = 'Últimos partidos: Nguyen A.'
        tokens_andy = _name_tokens('Andy Nguyen')
        tokens_avery = _name_tokens('Avery Nguyen')
        assert _tokens_match_kb(tokens_andy, kb)
        assert _tokens_match_kb(tokens_avery, kb)
