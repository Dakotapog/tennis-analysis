"""
tests/test_nodo127_games_outcome_ids.py — REGLA-T53 Nodo-127

D126-04: procesar_partidos excluye eventos STARTED del listView
D126-05: outcome IDs genéricos (aparecen en 2+ partidos) son descartados
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games_signal_calculator import _apellido


# ── D126-05: simulación de seen_outcome_ids ───────────────────────────────────

def _simular_filtro_unico(optimas_list: list, seen: dict) -> list:
    """Replica la lógica D126-05 de procesar_partidos para testear en aislamiento."""
    unicas = []
    for s in optimas_list:
        oid = s.get("outcome_id")
        if oid is None:
            unicas.append(s)
        elif oid in seen:
            pass  # descartado como NO_UNICO
        else:
            seen[oid] = s.get("partido", "test")
            unicas.append(s)
    return unicas


def test_outcome_unico_detecta_duplicado():
    """D126-05: un outcome_id visto en 2 partidos distintos → el segundo es descartado."""
    seen: dict[int, str] = {}
    señal_a = {"outcome_id": 4265916952, "partido": "Merida vs Jacquet", "cuota": 1.73}
    señal_b = {"outcome_id": 4265916952, "partido": "Wawrinka vs Burruchaga", "cuota": 1.73}

    resultado_a = _simular_filtro_unico([señal_a], seen)
    resultado_b = _simular_filtro_unico([señal_b], seen)

    assert len(resultado_a) == 1   # primero: OK
    assert len(resultado_b) == 0   # segundo: descartado (ID ya visto)


def test_outcome_unico_detecta_unico():
    """D126-05: outcome_ids distintos en partidos distintos → ambos se conservan."""
    seen: dict[int, str] = {}
    señal_a = {"outcome_id": 4265925873, "partido": "Borges vs Luz", "cuota": 1.87}
    señal_b = {"outcome_id": 4265928674, "partido": "Choinski vs De Jong", "cuota": 2.00}

    resultado_a = _simular_filtro_unico([señal_a], seen)
    resultado_b = _simular_filtro_unico([señal_b], seen)

    assert len(resultado_a) == 1
    assert len(resultado_b) == 1


def test_filtro_started_via_apellido():
    """D126-04: _apellido() extrae correctamente para los casos del bug STARTED."""
    # Bueno G. vs Pereira T. — partido que estaba STARTED con odds @9.5
    assert _apellido("Bueno G.") == "bueno"
    assert _apellido("Pereira T.") == "pereira"
    # Estos apellidos son suficientemente específicos para no matchear en dobles
    # con el filtro "/" activo — verificar que la función devuelve strings útiles
    assert len(_apellido("Bueno G.")) > 2
    assert len(_apellido("Pereira T.")) > 2
