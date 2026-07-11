"""
tests/test_nodo80_kambi_matching.py — Nodo-80: apellidos compuestos en Kambi lookup

REGLA-T53: invoca función real _apellido_candidates del módulo.
Caso real: Pedro Vives Marcos vs Vlado Jankanj (2026-07-08).
  shadow_book extraía "marcos" → no match en Kambi.
  Con Opción 1: candidatos ["marcos", "vives marcos"] → Kambi tiene "vives marcos" → match.
"""
import pytest
from shadow_book import _apellido_candidates


class TestApellidoCandidatesSimple:
    """Nombres de 2 partes — un solo candidato (comportamiento anterior idéntico)."""

    def test_nombre_dos_partes(self):
        assert _apellido_candidates("leyton rivera") == ["rivera"]

    def test_nombre_dos_partes_kambi(self):
        assert _apellido_candidates("vlado jankanj") == ["jankanj"]


class TestApellidoCandidatesCompuesto:
    """Nombres de 3+ partes — incluye apellido compuesto."""

    def test_tres_partes_genera_dos_candidatos(self):
        # "pedro vives marcos" → ["marcos", "vives marcos"]
        candidates = _apellido_candidates("pedro vives marcos")
        assert candidates == ["marcos", "vives marcos"]

    def test_apellido_compuesto_es_segundo_candidato(self):
        candidates = _apellido_candidates("pedro vives marcos")
        assert "vives marcos" in candidates

    def test_ultimo_token_es_primer_candidato(self):
        # El comportamiento anterior (solo "marcos") sigue siendo el primero
        candidates = _apellido_candidates("pedro vives marcos")
        assert candidates[0] == "marcos"

    def test_cuatro_partes(self):
        # "alejandro davidovich fokina" → ["fokina", "davidovich fokina"]
        candidates = _apellido_candidates("alejandro davidovich fokina")
        assert candidates == ["fokina", "davidovich fokina"]

    def test_no_incluye_nombre_completo(self):
        # El Tier 1 ya probó el nombre completo — Tier 2 no lo repite
        candidates = _apellido_candidates("pedro vives marcos")
        assert "pedro vives marcos" not in candidates


class TestApellidoCandidatesEdgeCases:
    """Casos borde."""

    def test_nombre_una_parte(self):
        # Solo apellido ("Federer") → sin candidatos (Tier 1 lo habrá probado)
        assert _apellido_candidates("federer") == []

    def test_nombre_vacio(self):
        assert _apellido_candidates("") == []


class TestKambiLookupSimulado:
    """
    Simulación del lookup en outcomes_map — demuestra que el fix resuelve el bug real.
    REGLA-T53: usa _apellido_candidates real; el outcomes_map es sintético pero mínimo.
    """

    def test_vives_marcos_finds_compound_key(self):
        """Caso concreto Nodo-80: Kambi indexa 'vives marcos', no 'marcos' solo."""
        outcomes_map = {"vives marcos": {"odds": 2.05, "outcome_id": "kambi_test"}}
        candidates = _apellido_candidates("pedro vives marcos")

        found = None
        for cand in candidates:
            found = outcomes_map.get(cand)
            if found:
                break

        assert found is not None, "El candidato 'vives marcos' debe encontrar match"
        assert found["odds"] == 2.05

    def test_primer_candidato_gana_cuando_existe(self):
        """Si el último token sí está en Kambi, lo encuentra sin llegar al compuesto."""
        outcomes_map = {"marcos": {"odds": 3.10, "outcome_id": "simple"}}
        candidates = _apellido_candidates("pedro vives marcos")

        found = None
        for cand in candidates:
            found = outcomes_map.get(cand)
            if found:
                break

        assert found is not None
        assert found["odds"] == 3.10

    def test_nombre_simple_compatible(self):
        """Nombres de 2 partes siguen funcionando igual que antes."""
        outcomes_map = {"jankanj": {"odds": 1.28, "outcome_id": "simple"}}
        candidates = _apellido_candidates("vlado jankanj")

        found = None
        for cand in candidates:
            found = outcomes_map.get(cand)
            if found:
                break

        assert found is not None
        assert found["odds"] == 1.28
