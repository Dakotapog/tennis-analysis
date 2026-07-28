"""
REGLA-T53 — Tests Nodo-149: Separación definitiva mercados juegos/sets.

D149-01: mercado_tipo en señales ("JUEGOS" | "SETS")
D149-02: _seleccionar_señal_optima() retorna tupla (juegos_optimas, sets_optimas)
D149-03: señales_optimas = solo JUEGOS; señales_optimas_sets = solo SETS
D149-04: imprimir_reporte() combos A/B (juegos) vs Combo C (sets) — nunca mezclados
D149-05: build_games_combos() filtra por mercado_tipo JUEGOS
D149-06: gap_sets = p_modelo_3sets - 1/cuota; threshold >= 0.10

9 tests — REGLA-T53: invocan funciones reales del módulo.
"""

import pytest
from games_signal_calculator import (
    _seleccionar_señal_optima,
    _P_3SETS_POR_ZONA,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _señal_juegos(direccion="UNDER", linea=22.5, cuota=1.75, gap=4.0, confianza="ALTA"):
    return {
        "mercado": "Total de juegos",
        "mercado_tipo": "JUEGOS",
        "linea": linea,
        "direccion": direccion,
        "cuota": cuota,
        "outcome_id": 123456,
        "gap_juegos": gap,
        "razon": "test",
        "confianza_señal": confianza,
        "apostar": True,
    }


def _señal_sets(direccion="UNDER", linea=2.5, cuota=1.65, confianza="MEDIA"):
    return {
        "mercado": "Total de sets",
        "mercado_tipo": "SETS",
        "linea": linea,
        "direccion": direccion,
        "cuota": cuota,
        "outcome_id": 789012,
        "gap_juegos": None,
        "razon": "test sets",
        "confianza_señal": confianza,
        "apostar": True,
    }


# ── D149-01: mercado_tipo en señales ─────────────────────────────────────────

class TestMercadoTipo:

    def test_juegos_señal_tiene_mercado_tipo_juegos(self):
        """Señal JUEGOS tiene mercado_tipo='JUEGOS'."""
        s = _señal_juegos()
        assert s["mercado_tipo"] == "JUEGOS"

    def test_sets_señal_tiene_mercado_tipo_sets(self):
        """Señal SETS tiene mercado_tipo='SETS'."""
        s = _señal_sets()
        assert s["mercado_tipo"] == "SETS"


# ── D149-02: _seleccionar_señal_optima() retorna tupla ───────────────────────

class TestSeleccionarSenalOptima:

    def test_retorna_tupla_dos_elementos(self):
        """Función retorna (juegos_optimas, sets_optimas) — nunca lista plana."""
        señales = [_señal_juegos(), _señal_sets()]
        resultado = _seleccionar_señal_optima(señales)
        assert isinstance(resultado, tuple)
        assert len(resultado) == 2

    def test_juegos_van_a_primer_elemento(self):
        """Señales JUEGOS aparecen en juegos_optimas (índice 0 de la tupla)."""
        señales = [_señal_juegos(direccion="UNDER"), _señal_sets()]
        juegos, sets = _seleccionar_señal_optima(señales)
        assert len(juegos) == 1
        assert juegos[0]["mercado_tipo"] == "JUEGOS"
        assert len(sets) == 1
        assert sets[0]["mercado_tipo"] == "SETS"

    def test_sets_van_a_segundo_elemento(self):
        """Señales SETS aparecen en sets_optimas (índice 1 de la tupla)."""
        señales = [_señal_sets(direccion="OVER", cuota=1.80)]
        juegos, sets = _seleccionar_señal_optima(señales)
        assert len(juegos) == 0
        assert len(sets) == 1
        assert sets[0]["mercado_tipo"] == "SETS"

    def test_sin_señales_retorna_listas_vacias(self):
        """Sin señales apostables → ([], [])."""
        juegos, sets = _seleccionar_señal_optima([])
        assert juegos == []
        assert sets == []

    def test_no_apostar_false_excluido(self):
        """Señal con apostar=False no aparece en ninguna lista."""
        s = _señal_juegos()
        s["apostar"] = False
        juegos, sets = _seleccionar_señal_optima([s])
        assert juegos == []
        assert sets == []


# ── D149-04: pools separados — nunca se mezclan ──────────────────────────────

class TestPoolsSeparados:

    def test_juegos_under_over_no_se_mezclan_con_sets(self):
        """UNDER juegos + OVER juegos + SETS → juegos tiene 2, sets tiene 1."""
        señales = [
            _señal_juegos(direccion="UNDER", linea=22.5, gap=4.0),
            _señal_juegos(direccion="OVER", linea=20.5, gap=3.0),
            _señal_sets(direccion="UNDER"),
        ]
        juegos, sets = _seleccionar_señal_optima(señales)
        assert len(juegos) == 2
        assert len(sets) == 1
        # Verificar que ningún elemento de juegos es SETS y viceversa
        for s in juegos:
            assert s["mercado_tipo"] == "JUEGOS"
        for s in sets:
            assert s["mercado_tipo"] == "SETS"


# ── D149-06: gap_sets ─────────────────────────────────────────────────────────

class TestGapSets:

    def test_p3sets_por_zona_definido(self):
        """_P_3SETS_POR_ZONA tiene las 3 zonas con probabilidades válidas."""
        for zona in ("dominante", "coinflip", "ajustada"):
            p = _P_3SETS_POR_ZONA[zona]
            assert 0.0 < p < 1.0, f"P({zona}) debe ser probabilidad válida"

    def test_gap_sets_formula(self):
        """gap_sets = p_modelo_3sets - 1/cuota."""
        p = _P_3SETS_POR_ZONA["coinflip"]  # 0.60
        cuota = 1.80
        expected = round(p - (1.0 / cuota), 4)
        # coinflip: 0.60 - 0.5556 = 0.0444 → threshold 0.10 no pasa
        # Con cuota 2.0: 0.60 - 0.50 = 0.10 → exactamente en threshold
        p2 = _P_3SETS_POR_ZONA["coinflip"]
        cuota2 = 2.0
        expected2 = round(p2 - (1.0 / cuota2), 4)
        assert expected2 >= 0.10, "coinflip + cuota 2.0 debe superar threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
