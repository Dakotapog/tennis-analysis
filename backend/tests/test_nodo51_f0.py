"""
tests/test_nodo51_f0.py — Nodo-51 F0: PlayerRegistry — Entity Resolution

Tests T51-F0-01 → T51-F0-09

Validan la tabla canónica de jugadores: lookup dos capas (fast O(1) + slow fuzzy),
memoria inmune (alias se registran tras slow path), y la integración con
_inject_kambi_ranking (0 sobreescrituras para jugadores en ATP file).

Los 6 casos de Nodo-47 son el corpus de verdad documentado:
  Glinka, Mayo, Ilagan, Hussey, Manning → fast path (reversed key)
  Watanuki                              → slow path (birth year en clave ATP)
  Unknown ITF                           → kambi_estimate

Detección de mutación real:
  T51-F0-02 FALLA si se elimina el reversed-key del fast path.
  T51-F0-03 FALLA si se elimina el slow path (get_player_info).
  T51-F0-04 FALLA si la memoria inmune no registra el alias tras slow path.
  T51-F0-05 FALLA si is_in_atp_file retorna True para un jugador desconocido.
  T51-F0-07 FALLA si alguno de los 6 casos Nodo-47 no resuelve como atp_file.
  T51-F0-08 FALLA si _inject_kambi_ranking sobreescribe a un jugador del ATP file.
  T51-F0-09 FALLA si _inject_kambi_ranking NO inyecta a un jugador ITF desconocido.
"""
import math
import pytest
from unittest.mock import MagicMock, patch, call

from core.player_registry import PlayerRegistry, normalize_player_name

# ─────────────────────────────────────────────────────────────────────────────
# Corpus de verdad — 6 casos documentados en Nodo-47
# ─────────────────────────────────────────────────────────────────────────────
# Formato real de clave ATP: normalize_name("Glinka Daniil") = "glinka daniil"
# El bug de Nodo-47: normalize_name("Daniil Glinka") = "daniil glinka" → NO matcheaba
# La solución: reversed_key "glinka daniil" → sí matchea → fast path O(1)
#
# Watanuki es el caso especial: key ATP = "watanuki yosuke 1998" (con birth year)
# reversed("yosuke watanuki") = "watanuki yosuke" ≠ "watanuki yosuke 1998" → slow path

# Registros sintéticos ATP (mismo formato que rankings_data real)
_GLINKA_RECORD    = {"name": "Glinka Daniil",    "ranking_position": 174, "ranking_points": 339, "tour": "ATP"}
_MAYO_RECORD      = {"name": "Mayo Aidan",        "ranking_position": 200, "ranking_points": 54,  "tour": "ATP"}
_WATANUKI_RECORD  = {"name": "Watanuki Yosuke (1998)", "ranking_position": 100, "ranking_points": 153, "tour": "ATP"}
_ILAGAN_RECORD    = {"name": "Ilagan Andre",      "ranking_position": 200, "ranking_points": 211, "tour": "ATP"}
_HUSSEY_RECORD    = {"name": "Hussey Giles",      "ranking_position": 197, "ranking_points": 221, "tour": "ATP"}
_MANNING_RECORD   = {"name": "Manning William",   "ranking_position": 200, "ranking_points": 1,   "tour": "ATP"}

# rankings_data usa normalize_name() como clave — reproducimos el resultado exacto:
# normalize_name("Glinka Daniil") = "glinka daniil"  ← ya está en formato Apellido Nombre
_RANKINGS_DATA = {
    "glinka daniil":         _GLINKA_RECORD,
    "mayo aidan":            _MAYO_RECORD,
    "watanuki yosuke 1998":  _WATANUKI_RECORD,   # ← tiene birth year → no hay reversed simple
    "ilagan andre":          _ILAGAN_RECORD,
    "hussey giles":          _HUSSEY_RECORD,
    "manning william":       _MANNING_RECORD,
}


def _make_registry(slow_path_returns=None):
    """
    Construye PlayerRegistry con datos sintéticos de los 6 casos Nodo-47.

    slow_path_returns: dict {player_name: record} que get_player_info simulará encontrar.
    Por default, Watanuki (slow path) retorna su record.
    """
    if slow_path_returns is None:
        slow_path_returns = {"Yosuke Watanuki": _WATANUKI_RECORD}

    mock_rm = MagicMock()
    mock_rm.rankings_data = _RANKINGS_DATA
    mock_rm.normalize_name = normalize_player_name
    mock_rm.get_player_info = MagicMock(
        side_effect=lambda name, tour=None: slow_path_returns.get(name)
    )

    return PlayerRegistry(
        normalize_fn=normalize_player_name,
        ranking_manager=mock_rm,
    ), mock_rm


# ─────────────────────────────────────────────────────────────────────────────
# T51-F0-01 a T51-F0-06 — PlayerRegistry unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPlayerRegistryFastPath:

    def test_t51_f0_01_direct_key_hit(self):
        """T51-F0-01: Alias ya en formato ATP (Apellido Nombre normalizado) → fast path directo.
        FALLA si el bootstrap no carga rankings_data en la alias table."""
        registry, _ = _make_registry()
        # "glinka daniil" ES la clave ATP directamente
        cid = registry.resolve("glinka daniil")
        assert cid is not None
        assert registry.is_in_atp_file("glinka daniil")

    def test_t51_f0_02_reversed_key_hit_nodo47_cases(self):
        """T51-F0-02: Formato Kambi/FlashScore (Nombre Apellido) → reversed_key hit O(1).
        Reproduce el bug de Nodo-47: normalize('Daniil Glinka')='daniil glinka' no matcheaba
        la clave ATP 'glinka daniil'. La fix generalizada es el reversed key en el registry.
        FALLA si se elimina el reversed-key del fast path."""
        registry, mock_rm = _make_registry()

        for player_name_kambi, expected_cid in [
            ("Daniil Glinka",   "glinka daniil"),
            ("Aidan Mayo",      "mayo aidan"),
            ("Andre Ilagan",    "ilagan andre"),
            ("Giles Hussey",    "hussey giles"),
            ("William Manning", "manning william"),
        ]:
            cid = registry.resolve(player_name_kambi)
            assert cid == expected_cid, (
                f"{player_name_kambi}: esperaba canonical_id='{expected_cid}', "
                f"se obtuvo '{cid}'"
            )
            # get_player_info NO debe llamarse — fast path resuelve sin slow path
            mock_rm.get_player_info.assert_not_called()

    def test_t51_f0_02b_reversed_key_marks_atp_file(self):
        """T51-F0-02b: is_in_atp_file=True para los 5 casos de reversed key."""
        registry, _ = _make_registry()
        for name in ["Daniil Glinka", "Aidan Mayo", "Andre Ilagan",
                     "Giles Hussey", "William Manning"]:
            assert registry.is_in_atp_file(name), (
                f"is_in_atp_file debe ser True para '{name}' (reversed key hit)"
            )


class TestPlayerRegistrySlowPath:

    def test_t51_f0_03_slow_path_watanuki_birth_year(self):
        """T51-F0-03: Watanuki tiene birth year en clave ATP → reversed key falla →
        slow path (get_player_info) encuentra y retorna canonical_id.
        FALLA si se elimina el slow path o si no se usa id() para vincular el record."""
        registry, mock_rm = _make_registry()

        cid = registry.resolve("Yosuke Watanuki")

        # Slow path debe haberse invocado
        mock_rm.get_player_info.assert_called_once_with("Yosuke Watanuki")

        assert cid == "watanuki yosuke 1998", (
            f"Slow path debe resolver canonical_id='watanuki yosuke 1998', "
            f"se obtuvo '{cid}'"
        )
        assert registry.is_in_atp_file("Yosuke Watanuki")

    def test_t51_f0_04_immune_memory_second_call_is_o1(self):
        """T51-F0-04: Tras slow path exitoso para Watanuki, el alias 'yosuke watanuki'
        se registra en la alias table. Segunda llamada resuelve sin invocar get_player_info.
        FALLA si se elimina el registro de alias en la memoria inmune."""
        registry, mock_rm = _make_registry()

        # Primera llamada — paga slow path
        cid_first = registry.resolve("Yosuke Watanuki")
        assert mock_rm.get_player_info.call_count == 1

        # Segunda llamada — debe ser O(1) (alias ya registrado)
        cid_second = registry.resolve("Yosuke Watanuki")
        assert mock_rm.get_player_info.call_count == 1, (
            "Segunda llamada no debe invocar get_player_info (memoria inmune activa)"
        )
        assert cid_first == cid_second


class TestPlayerRegistryUnknown:

    def test_t51_f0_05_unknown_player_returns_none(self):
        """T51-F0-05: Jugador no en ATP file y no en slow path → resolve=None, is_in_atp_file=False.
        FALLA si is_in_atp_file retorna True para jugador desconocido."""
        registry, _ = _make_registry(slow_path_returns={})  # slow path vacío

        cid = registry.resolve("Unknown ITF Player")
        assert cid is None, f"resolve debe retornar None, se obtuvo '{cid}'"
        assert not registry.is_in_atp_file("Unknown ITF Player"), (
            "is_in_atp_file debe ser False para jugador desconocido"
        )

    def test_t51_f0_06_register_kambi_estimate(self):
        """T51-F0-06: register_kambi_estimate registra jugador ITF con provenance='kambi_estimate'.
        Tras registro, segunda llamada a resolve es O(1) y provenance no confunde con atp_file."""
        registry, _ = _make_registry(slow_path_returns={})

        cid = registry.register_kambi_estimate("Unknown ITF Player")

        assert cid is not None, "register_kambi_estimate debe retornar un canonical_id"
        assert not registry.is_in_atp_file("Unknown ITF Player"), (
            "is_in_atp_file debe seguir siendo False tras registro como kambi_estimate"
        )
        assert registry.provenance("Unknown ITF Player") == "kambi_estimate"

    def test_t51_f0_06b_kambi_estimate_resolve_is_o1_after_register(self):
        """T51-F0-06b: Tras register_kambi_estimate, resolve es O(1) (alias registrado)."""
        registry, mock_rm = _make_registry(slow_path_returns={})

        registry.register_kambi_estimate("Unknown ITF Player")
        mock_rm.get_player_info.reset_mock()

        cid = registry.resolve("Unknown ITF Player")
        assert cid is not None
        mock_rm.get_player_info.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# T51-F0-07 — Los 6 casos Nodo-47 completos
# ─────────────────────────────────────────────────────────────────────────────

class TestNodo47Cases:

    def test_t51_f0_07_all_six_nodo47_players_are_atp_file(self):
        """T51-F0-07: Los 6 jugadores documentados en Nodo-47 son reconocidos como atp_file.
        Cualquier regresión en la lógica de resolución hace fallar este test.
        FALLA si alguno de los 6 no resuelve correctamente."""
        registry, _ = _make_registry(
            slow_path_returns={"Yosuke Watanuki": _WATANUKI_RECORD}
        )

        cases = [
            ("Daniil Glinka",   "reversed key",  "glinka daniil"),
            ("Aidan Mayo",      "reversed key",  "mayo aidan"),
            ("Yosuke Watanuki", "slow path",     "watanuki yosuke 1998"),
            ("Andre Ilagan",    "reversed key",  "ilagan andre"),
            ("Giles Hussey",    "reversed key",  "hussey giles"),
            ("William Manning", "reversed key",  "manning william"),
        ]

        for player_name, expected_path, expected_cid in cases:
            cid = registry.resolve(player_name)
            assert cid == expected_cid, (
                f"[{expected_path}] {player_name}: "
                f"esperado '{expected_cid}', obtenido '{cid}'"
            )
            assert registry.is_in_atp_file(player_name), (
                f"{player_name} debe ser is_in_atp_file=True"
            )


# ─────────────────────────────────────────────────────────────────────────────
# T51-F0-08 y T51-F0-09 — Integración con _inject_kambi_ranking
# ─────────────────────────────────────────────────────────────────────────────

class TestInjectKambiRankingIntegration:
    """
    Verifica que _inject_kambi_ranking usa PlayerRegistry.is_in_atp_file()
    y no sobreescribe jugadores presentes en el ATP file.

    Esto es el criterio de aceptación de F0: "0 sobreescrituras Kambi en
    jugadores presentes en ATP file".
    """

    def _make_extractor_with_registry(self, rankings_data, slow_path_returns=None):
        """Crea NinjaH2HExtractor mínimo con PlayerRegistry integrado."""
        from scraping.ninja_h2h_parser import NinjaH2HExtractor

        with patch("scraping.ninja_h2h_parser.NinjaH2HExtractor.__init__",
                   return_value=None):
            ext = NinjaH2HExtractor.__new__(NinjaH2HExtractor)

        if slow_path_returns is None:
            slow_path_returns = {}

        mock_rm = MagicMock()
        mock_rm.rankings_data = dict(rankings_data)  # mutable copy
        mock_rm.normalize_name = normalize_player_name
        mock_rm.get_player_info = MagicMock(
            side_effect=lambda name, tour=None: slow_path_returns.get(name)
        )

        ext.ranking_manager = mock_rm
        ext._player_registry = PlayerRegistry(
            normalize_fn=normalize_player_name,
            ranking_manager=mock_rm,
        )
        return ext, mock_rm

    def test_t51_f0_08_no_overwrite_for_atp_player(self):
        """T51-F0-08: _inject_kambi_ranking NO sobreescribe a Daniil Glinka (en ATP file).
        El pts_estimate de Kambi (163pts) NO reemplaza los 339pts del ATP file.
        FALLA si se elimina la guard is_in_atp_file en _inject_kambi_ranking."""
        rankings_data = {"glinka daniil": dict(_GLINKA_RECORD)}
        ext, mock_rm = self._make_extractor_with_registry(rankings_data)

        pts_before = mock_rm.rankings_data["glinka daniil"]["ranking_points"]
        ext._inject_kambi_ranking("Daniil Glinka", 73)
        pts_after = mock_rm.rankings_data["glinka daniil"]["ranking_points"]

        assert pts_after == pts_before, (
            f"Kambi NO debe sobreescribir: {pts_before}pts ATP → se convirtió en {pts_after}pts"
        )

    def test_t51_f0_09_inject_for_unknown_itf_player(self):
        """T51-F0-09: _inject_kambi_ranking SÍ inyecta para jugador ITF desconocido.
        El jugador no está en rankings_data, por lo que recibe pts_estimate de Kambi.
        FALLA si el guard bloquea incorrectamente la inyección de jugadores desconocidos."""
        ext, mock_rm = self._make_extractor_with_registry(rankings_data={})

        ext._inject_kambi_ranking("Mario Arce Fernandez", 500)

        normalized_name = normalize_player_name("Mario Arce Fernandez")
        assert normalized_name in mock_rm.rankings_data, (
            f"El jugador ITF desconocido debe ser inyectado en rankings_data"
        )
        injected = mock_rm.rankings_data[normalized_name]
        expected_pts = max(1, round(700 / math.log1p(500)))
        assert injected["ranking_points"] == expected_pts, (
            f"pts_estimate esperado={expected_pts}, obtenido={injected['ranking_points']}"
        )
        # Debe quedar registrado como kambi_estimate en el registry
        assert ext._player_registry.provenance("Mario Arce Fernandez") == "kambi_estimate"

    def test_t51_f0_08b_no_overwrite_william_manning(self):
        """T51-F0-08b: William Manning (1pt real ATP) NO es sobreescrito por estimate Kambi.
        Este fue el caso de mayor error en Nodo-47: +131pts de sobrestimación."""
        rankings_data = {"manning william": dict(_MANNING_RECORD)}
        ext, mock_rm = self._make_extractor_with_registry(rankings_data)

        pts_before = mock_rm.rankings_data["manning william"]["ranking_points"]
        ext._inject_kambi_ranking("William Manning", 200)
        pts_after = mock_rm.rankings_data["manning william"]["ranking_points"]

        assert pts_after == pts_before, (
            f"Manning: {pts_before}pts reales no deben ser sobreescritos por Kambi"
        )
