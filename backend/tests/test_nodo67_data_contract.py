"""
tests/test_nodo67_data_contract.py — C1 Nodo-67: DataContract v2

REGLA-T53: invoca validate_artifact() y DataContractViolation reales del módulo.
Cubre las 6 fronteras de Nodo-86 §4.1.
"""
import pytest
from core.data_contract import validate_artifact, DataContractViolation, ARTIFACT_SCHEMAS


# ─── Artefactos mínimos válidos por frontera ─────────────────────────────────

def _edge_report_ok():
    return {'metadata': {'fecha': '2026-07-13'}, 'apostar': []}

def _trader_plan_ok():
    return {'metadata': {'bankroll': 125000}, 'individuales': []}

def _betslip_index_ok():
    return {'ts': '2026-07-13T10:00:00', 'index': {}}

def _apuestas_ok():
    return {
        'estado': 'ABIERTO',
        'picks': [
            {'jugador': 'Alcaraz', 'cuota': 1.80, 'outcome_id': 'abc123'},
        ],
        'ts_registro': '2026-07-13T10:00:00',
    }

def _sb_jsonl_pick_ok():
    return {'sb_id': 'sb-001', 'partido': 'Alcaraz vs Djokovic', 'pick_snapshot': {}}

def _combo_plan_json_ok():
    return {'fecha': '2026-07-13', 'bankroll': 125000, 'budget': 5000, 'cobertura': []}


# ─── ARTIFACT_SCHEMAS cubre las 6 fronteras ──────────────────────────────────

def test_schemas_contiene_6_fronteras():
    assert len(ARTIFACT_SCHEMAS) == 6

def test_schemas_contiene_todas_las_claves_esperadas():
    esperados = {'edge_report', 'trader_plan', 'betslip_index', 'apuestas', 'sb_jsonl_pick', 'combo_plan_json'}
    assert esperados == set(ARTIFACT_SCHEMAS.keys())


# ─── Artefactos válidos retornan True ────────────────────────────────────────

def test_edge_report_valido():
    assert validate_artifact('edge_report', _edge_report_ok()) is True

def test_trader_plan_valido():
    assert validate_artifact('trader_plan', _trader_plan_ok()) is True

def test_betslip_index_valido():
    assert validate_artifact('betslip_index', _betslip_index_ok()) is True

def test_apuestas_valido():
    assert validate_artifact('apuestas', _apuestas_ok()) is True

def test_sb_jsonl_pick_valido():
    assert validate_artifact('sb_jsonl_pick', _sb_jsonl_pick_ok()) is True

def test_combo_plan_json_valido():
    assert validate_artifact('combo_plan_json', _combo_plan_json_ok()) is True


# ─── Artefacto desconocido → DataContractViolation ───────────────────────────

def test_nombre_desconocido_lanza_violation():
    with pytest.raises(DataContractViolation, match="no registrado"):
        validate_artifact('artefacto_inventado', {})

def test_nombre_vacio_lanza_violation():
    with pytest.raises(DataContractViolation):
        validate_artifact('', {})


# ─── Claves raíz faltantes → DataContractViolation ───────────────────────────

def test_edge_report_sin_apostar_lanza():
    obj = {'metadata': {}}  # falta 'apostar'
    with pytest.raises(DataContractViolation, match="apostar"):
        validate_artifact('edge_report', obj)

def test_edge_report_sin_metadata_lanza():
    obj = {'apostar': []}  # falta 'metadata'
    with pytest.raises(DataContractViolation, match="metadata"):
        validate_artifact('edge_report', obj)

def test_trader_plan_sin_individuales_lanza():
    obj = {'metadata': {}}
    with pytest.raises(DataContractViolation, match="individuales"):
        validate_artifact('trader_plan', obj)

def test_betslip_index_sin_ts_lanza():
    obj = {'index': {}}
    with pytest.raises(DataContractViolation, match="ts"):
        validate_artifact('betslip_index', obj)

def test_apuestas_sin_estado_lanza():
    obj = {'picks': [], 'ts_registro': 'x'}
    with pytest.raises(DataContractViolation, match="estado"):
        validate_artifact('apuestas', obj)

def test_sb_jsonl_sin_sb_id_lanza():
    obj = {'partido': 'x', 'pick_snapshot': {}}
    with pytest.raises(DataContractViolation, match="sb_id"):
        validate_artifact('sb_jsonl_pick', obj)

def test_combo_plan_sin_budget_lanza():
    obj = {'fecha': '2026-07-13', 'bankroll': 125000, 'cobertura': []}
    with pytest.raises(DataContractViolation, match="budget"):
        validate_artifact('combo_plan_json', obj)


# ─── pick_required: validación por pick (frontera apuestas) ──────────────────

def test_apuestas_pick_sin_cuota_lanza():
    obj = {
        'estado': 'ABIERTO',
        'picks': [{'jugador': 'Alcaraz', 'outcome_id': 'abc'}],  # falta cuota
        'ts_registro': 'x',
    }
    with pytest.raises(DataContractViolation, match="cuota"):
        validate_artifact('apuestas', obj)

def test_apuestas_pick_sin_jugador_lanza():
    obj = {
        'estado': 'ABIERTO',
        'picks': [{'cuota': 1.80, 'outcome_id': 'abc'}],  # falta jugador
        'ts_registro': 'x',
    }
    with pytest.raises(DataContractViolation, match="jugador"):
        validate_artifact('apuestas', obj)

def test_apuestas_picks_vacios_es_valido():
    """Lista vacía de picks es válida — no hay nada que validar."""
    obj = {'estado': 'ABIERTO', 'picks': [], 'ts_registro': 'x'}
    assert validate_artifact('apuestas', obj) is True

def test_apuestas_pick_no_dict_se_ignora():
    """Picks que no son dict se saltan (e.g. None o string legacy)."""
    obj = {
        'estado': 'ABIERTO',
        'picks': [None, 'legacy_string'],
        'ts_registro': 'x',
    }
    assert validate_artifact('apuestas', obj) is True

def test_apuestas_segundo_pick_invalido_lanza():
    """Error en el segundo pick — el índice aparece en el mensaje."""
    obj = {
        'estado': 'ABIERTO',
        'picks': [
            {'jugador': 'Alcaraz', 'cuota': 1.80, 'outcome_id': 'abc'},
            {'jugador': 'Djokovic', 'outcome_id': 'xyz'},  # falta cuota
        ],
        'ts_registro': 'x',
    }
    with pytest.raises(DataContractViolation, match="cuota"):
        validate_artifact('apuestas', obj)


# ─── Mensaje de error es informativo (fail-loud) ─────────────────────────────

def test_violation_mensaje_contiene_nombre_artefacto():
    try:
        validate_artifact('edge_report', {})
    except DataContractViolation as e:
        assert 'edge_report' in str(e)
    else:
        pytest.fail("Debería haber lanzado DataContractViolation")

def test_violation_hereda_de_exception():
    assert issubclass(DataContractViolation, Exception)
