"""
tests/test_nodo50.py — Nodo-50: Filtro --torneo en PASO 1

Tests T50-01 → T50-07
Validan la lógica de filtrado por nombre de torneo en extract_matches()
y extract_matches_flashscore_only().

No invoca Playwright ni Kambi — prueba la función de filtrado puro
sobre listas de matches sintéticas.
"""
import pytest
from typing import List, Dict


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_match(torneo_nombre: str, torneo_completo: str, tier: str = "atp") -> Dict:
    return {
        "jugador1": "Player A",
        "jugador2": "Player B",
        "torneo_nombre": torneo_nombre,
        "torneo_completo": torneo_completo,
        "tier": tier,
        "cuota1": 1.80,
        "cuota2": 2.10,
    }


def _apply_torneo_filter(matches: List[Dict], torneos: List[str]) -> List[Dict]:
    """
    Replica exacta de la lógica implementada en kambi_tennis.py (Nodo-50).
    Se mantiene aquí para probar el algoritmo aislado del I/O de Kambi/FlashScore.
    """
    if not torneos:
        return matches
    keywords = [k.lower() for k in torneos]
    return [
        m for m in matches
        if any(
            kw in (m.get('torneo_nombre') or '').lower()
            or kw in (m.get('torneo_completo') or '').lower()
            for kw in keywords
        )
    ]


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_matches():
    return [
        _make_match(
            "Wimbledon",
            "ATP - INDIVIDUALES: Wimbledon (Reino Unido) - hierba",
            tier="atp",
        ),
        _make_match(
            "Roland Garros",
            "ATP - INDIVIDUALES: Roland Garros (Francia) - arcilla",
            tier="atp",
        ),
        _make_match(
            "Nottingham",
            "ATP - INDIVIDUALES: Nottingham (Reino Unido) - hierba",
            tier="atp",
        ),
        _make_match(
            "Skopje",
            "ITF MASCULINO - INDIVIDUALES: M25 Skopje (Macedonia), arcilla",
            tier="itf",
        ),
        _make_match(
            "Wimbledon",
            "WTA - INDIVIDUALES: Wimbledon (Reino Unido) - hierba",
            tier="wta",
        ),
    ]


# ── tests ─────────────────────────────────────────────────────────────────────

class TestFiltroTorneo:

    def test_T50_01_match_torneo_nombre_case_insensitive(self, sample_matches):
        """T50-01: torneo_nombre match — 'Wimbledon' matchea kw='wimbledon'"""
        result = _apply_torneo_filter(sample_matches, ["wimbledon"])
        assert len(result) == 2
        for m in result:
            assert "wimbledon" in (m["torneo_nombre"] or "").lower()

    def test_T50_02_match_torneo_completo_substring(self, sample_matches):
        """T50-02: match por substring en torneo_completo"""
        result = _apply_torneo_filter(sample_matches, ["reino unido"])
        # Wimbledon ATP + Nottingham + Wimbledon WTA = 3
        assert len(result) == 3
        for m in result:
            assert "reino unido" in m["torneo_completo"].lower()

    def test_T50_03_multiple_keywords_or_logic(self, sample_matches):
        """T50-03: múltiples keywords = OR — incluye todos los que coincidan"""
        result = _apply_torneo_filter(sample_matches, ["wimbledon", "roland garros"])
        assert len(result) == 3  # 2 Wimbledon + 1 Roland Garros
        nombres = {m["torneo_nombre"] for m in result}
        assert "Wimbledon" in nombres
        assert "Roland Garros" in nombres

    def test_T50_04_no_match_returns_empty(self, sample_matches):
        """T50-04: sin match → lista vacía (no crashea)"""
        result = _apply_torneo_filter(sample_matches, ["us open"])
        assert result == []

    def test_T50_05_torneos_none_returns_all(self, sample_matches):
        """T50-05: torneos=None → sin filtro, devuelve todos los partidos"""
        result = _apply_torneo_filter(sample_matches, None)
        assert len(result) == len(sample_matches)

    def test_T50_06_combined_tier_and_torneo_and_logic(self, sample_matches):
        """T50-06: AND con tiers — solo ATP Wimbledon (no WTA Wimbledon)"""
        # Simular AND: primero filtrar por tier, luego por torneo
        atp_only = [m for m in sample_matches if m.get("tier") == "atp"]
        result = _apply_torneo_filter(atp_only, ["wimbledon"])
        assert len(result) == 1
        assert result[0]["tier"] == "atp"
        assert result[0]["torneo_nombre"] == "Wimbledon"

    def test_T50_07_torneo_nombre_none_no_crash(self):
        """T50-07: torneo_nombre=None — no crashea, usa torneo_completo como fallback"""
        matches = [
            {
                "jugador1": "A",
                "jugador2": "B",
                "torneo_nombre": None,
                "torneo_completo": "ATP - INDIVIDUALES: Wimbledon (Reino Unido) - hierba",
                "tier": "atp",
            }
        ]
        result = _apply_torneo_filter(matches, ["wimbledon"])
        assert len(result) == 1

    def test_T50_08_empty_string_torneo_nombre_no_crash(self):
        """T50-08: torneo_nombre='' (string vacío) — no crashea"""
        matches = [
            {
                "jugador1": "A",
                "jugador2": "B",
                "torneo_nombre": "",
                "torneo_completo": "ATP - INDIVIDUALES: Wimbledon (Reino Unido) - hierba",
                "tier": "atp",
            }
        ]
        result = _apply_torneo_filter(matches, ["wimbledon"])
        assert len(result) == 1
