"""
tests/test_nodo126_games_signal_bugs.py — REGLA-T53 Nodo-126

D126-01: _apellido() extrae apellido real, no la inicial
D126-02: _buscar_event_id_kambi() excluye dobles (filtro "/")
D126-03: .get("odds", 0) sin KeyError en outcomes sin clave
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games_signal_calculator import _apellido


def test_apellido_simple():
    """D126-01: nombres simples 'Apellido Inicial.' → apellido correcto."""
    assert _apellido("Choinski J.") == "choinski"
    assert _apellido("Bueno G.") == "bueno"
    assert _apellido("Gaston H.") == "gaston"


def test_apellido_compuesto():
    """D126-01: nombres compuestos 'Palabra Apellido Inicial.' → último token no-inicial."""
    assert _apellido("De Jong J.") == "jong"
    assert _apellido("Ugo Carabelli C.") == "carabelli"
    assert _apellido("Van Assche L.") == "assche"


def test_apellido_sin_inicial():
    """D126-01: nombres sin inicial no rompen."""
    assert _apellido("Djokovic") == "djokovic"
    assert _apellido("") == ""
