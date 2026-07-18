"""
REGLA-T53: Tests Nodo-118 F2 — Crosswalk en PlayerRegistry.

Cubre: add_alias persiste y resuelve; VERIFIED no se sobreescribe con AUTO;
MANUAL no se sobreescribe por nada; resolve_crosswalk va antes del fuzzy;
add_alias acumula sin duplicar canonical_id.
"""

import json
import pytest
from pathlib import Path

from core.player_registry import PlayerRegistry, normalize_player_name


def _make_registry(tmp_path) -> PlayerRegistry:
    """Registry limpio sin ranking_manager, con crosswalk en tmp_path."""
    reg = PlayerRegistry(normalize_fn=normalize_player_name)
    # Apuntar crosswalk a directorio temporal
    reg._crosswalk_path = tmp_path / "player_crosswalk.json"
    # Parchar _save/_load para usar tmp
    import core.player_registry as _mod
    _orig_cw = _mod._CROSSWALK_FILE
    _mod._CROSSWALK_FILE = reg._crosswalk_path
    return reg


class TestAddAlias:

    def test_alias_persiste_y_resuelve(self, tmp_path):
        """
        REGLA-T53: add_alias persiste el alias en disco y resolve_crosswalk
        lo encuentra en O(1) en la misma instancia.
        """
        import core.player_registry as _mod
        _mod._CROSSWALK_FILE = tmp_path / "player_crosswalk.json"

        reg = PlayerRegistry(normalize_fn=normalize_player_name)
        reg.add_alias("Paula Badosa", "P. Badosa", source="kambi", confidence="AUTO")

        # resolve_crosswalk debe encontrarlo
        result = reg.resolve_crosswalk("P. Badosa")
        assert result is not None, "resolve_crosswalk debe retornar canonical_id"
        assert "badosa" in result, f"canonical_id debe contener 'badosa', obtenido: {result}"

        # Verificar que se persistió en disco
        cw_file = tmp_path / "player_crosswalk.json"
        assert cw_file.exists(), "crosswalk debe persistirse en disco"
        data = json.loads(cw_file.read_text())
        assert "entries" in data
        assert len(data["entries"]) >= 1

    def test_verified_no_se_sobreescribe_con_auto(self, tmp_path):
        """
        REGLA-T53: un alias con confidence=VERIFIED no puede ser degradado a AUTO.
        """
        import core.player_registry as _mod
        _mod._CROSSWALK_FILE = tmp_path / "player_crosswalk.json"

        reg = PlayerRegistry(normalize_fn=normalize_player_name)
        reg.add_alias("Paula Badosa", "P. Badosa", source="flashscore", confidence="VERIFIED")
        reg.add_alias("Paula Badosa", "P. Badosa", source="kambi", confidence="AUTO")

        # La entrada debe mantener VERIFIED
        cid = reg.resolve_crosswalk("P. Badosa")
        canonical_norm = normalize_player_name("Paula Badosa")
        entry = reg._xwalk.get(canonical_norm, {})
        alias_norm = normalize_player_name("P. Badosa")
        alias_data = entry.get("aliases", {}).get(alias_norm, {})
        assert alias_data.get("confidence") == "VERIFIED", (
            f"VERIFIED no debe sobreescribirse con AUTO: {alias_data}"
        )

    def test_manual_no_se_sobreescribe_por_nada(self, tmp_path):
        """
        REGLA-T53: un alias MANUAL no puede ser sobreescrito por VERIFIED ni AUTO.
        """
        import core.player_registry as _mod
        _mod._CROSSWALK_FILE = tmp_path / "player_crosswalk.json"

        reg = PlayerRegistry(normalize_fn=normalize_player_name)
        reg.add_alias("Iga Swiatek", "I. Swiatek", source="manual", confidence="MANUAL")
        # Intentar sobreescribir con VERIFIED
        reg.add_alias("Iga Swiatek", "I. Swiatek", source="kambi", confidence="VERIFIED")

        canonical_norm = normalize_player_name("Iga Swiatek")
        alias_norm = normalize_player_name("I. Swiatek")
        entry = reg._xwalk.get(canonical_norm, {})
        alias_data = entry.get("aliases", {}).get(alias_norm, {})
        assert alias_data.get("confidence") == "MANUAL", (
            f"MANUAL no debe sobreescribirse con VERIFIED: {alias_data}"
        )

    def test_add_alias_acumula_sin_duplicar(self, tmp_path):
        """
        REGLA-T53: el mismo canonical puede tener múltiples aliases distintos;
        añadir el mismo alias dos veces no duplica entradas.
        """
        import core.player_registry as _mod
        _mod._CROSSWALK_FILE = tmp_path / "player_crosswalk.json"

        reg = PlayerRegistry(normalize_fn=normalize_player_name)
        reg.add_alias("Carlos Alcaraz", "C. Alcaraz", source="kambi", confidence="AUTO")
        reg.add_alias("Carlos Alcaraz", "Alcaraz C.", source="flashscore", confidence="AUTO")
        # Mismo alias dos veces
        reg.add_alias("Carlos Alcaraz", "C. Alcaraz", source="kambi", confidence="AUTO")

        canonical_norm = normalize_player_name("Carlos Alcaraz")
        entry = reg._xwalk.get(canonical_norm, {})
        aliases = entry.get("aliases", {})
        # Solo 2 aliases distintos (no 3)
        assert len(aliases) == 2, (
            f"esperados 2 aliases distintos, obtenidos {len(aliases)}: {list(aliases.keys())}"
        )

    def test_resolve_crosswalk_retorna_none_si_no_existe(self, tmp_path):
        """
        REGLA-T53: resolve_crosswalk retorna None para alias no registrado
        (sin ranking_manager — no hay fuzzy fallback).
        """
        import core.player_registry as _mod
        _mod._CROSSWALK_FILE = tmp_path / "player_crosswalk.json"

        reg = PlayerRegistry(normalize_fn=normalize_player_name)
        result = reg.resolve_crosswalk("Jugador Desconocido XYZ")
        assert result is None

    def test_crosswalk_carga_desde_disco_en_nueva_instancia(self, tmp_path):
        """
        REGLA-T53: el crosswalk persiste entre instancias — una nueva instancia
        carga los aliases escritos por la anterior.
        """
        import core.player_registry as _mod
        cw_file = tmp_path / "player_crosswalk.json"
        _mod._CROSSWALK_FILE = cw_file

        # Primera instancia escribe
        reg1 = PlayerRegistry(normalize_fn=normalize_player_name)
        reg1.add_alias("Novak Djokovic", "N. Djokovic", source="kambi", confidence="VERIFIED")

        # Segunda instancia carga desde disco
        reg2 = PlayerRegistry(normalize_fn=normalize_player_name)
        result = reg2.resolve_crosswalk("N. Djokovic")
        assert result is not None, (
            "Nueva instancia debe resolver alias persistido por instancia anterior"
        )
        assert "djokovic" in result
