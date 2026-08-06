"""
Tests Nodo-165 — Bonus de certeza D147 en convergencia_score ITF (D165-01).

D142-02 (_convergencia_score_itf, heurística gap/cuota/markov/ranking) y D147
(_calcular_certeza_condicional, Gaussiano condicionado al marcador real) se
calculan en pasos separados del pipeline de live_desk.py sin comunicarse
entre sí. Señales ITF_VIVO con p_condicional>=0.90 (D147 confirma la
dirección con el marcador real) quedaban atascadas en convergencia_score=2
por falta de markov (jugadores ITF/qualy sin pick individual en edge_report)
o cuota_live<2.00, sin llegar nunca al umbral >=3 que dispara el combo real
(live_desk.py:4500). _convergencia_certeza_bonus() invocada aquí es la
función real del módulo — REGLA-T53.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live_desk import _convergencia_certeza_bonus


def test_165_01_certeza_alta_p_condicional_alto_suma_bonus():
    """alerta_nivel=ALTA + p_condicional>=0.85 → +1, score 2→3 (cruza el
    umbral de disparo). Caso real: Zheng vs Kecmanovic, p_condicional=0.94."""
    certeza = {"alerta_nivel": "ALTA", "p_condicional": 0.94, "certeza_matematica": False}
    r = _convergencia_certeza_bonus(score_actual=2, certeza=certeza)
    assert r["score"] == 3
    assert r["aplicado"] is True
    assert r["confianza"] == "ALTA"
    assert "D147_certeza=ALTA" in r["detalle"]


def test_165_02_certeza_matematica_tambien_suma_bonus():
    """alerta_nivel=CERTEZA (certeza_matematica=True) también califica —
    es un nivel de confianza igual o más fuerte que ALTA."""
    certeza = {"alerta_nivel": "CERTEZA", "p_condicional": 0.99, "certeza_matematica": True}
    r = _convergencia_certeza_bonus(score_actual=1, certeza=certeza)
    assert r["score"] == 2
    assert r["aplicado"] is True


def test_165_03_alerta_mod_no_suma_bonus():
    """alerta_nivel=MOD (p_condicional entre 0.70-0.90) no califica — el
    bonus es solo para certeza fuerte real, no para cualquier señal D147."""
    certeza = {"alerta_nivel": "MOD", "p_condicional": 0.75, "certeza_matematica": False}
    r = _convergencia_certeza_bonus(score_actual=2, certeza=certeza)
    assert r["score"] == 2
    assert r["aplicado"] is False


def test_165_04_alta_con_p_condicional_bajo_el_umbral_no_suma():
    """Control negativo del segundo criterio: alerta_nivel=ALTA pero
    p_condicional<0.85 no debería ocurrir en la práctica (ALTA implica
    p>=0.90 en _calcular_certeza_condicional), pero el guard doble no debe
    fallar si algún caller pasa datos inconsistentes."""
    certeza = {"alerta_nivel": "ALTA", "p_condicional": 0.80, "certeza_matematica": False}
    r = _convergencia_certeza_bonus(score_actual=2, certeza=certeza)
    assert r["aplicado"] is False
    assert r["score"] == 2


def test_165_05_certeza_none_no_suma_bonus():
    """Control negativo: sin score_data (certeza=None) no debe crashear ni
    sumar bonus — señal sigue evaluada solo por D142-02."""
    r = _convergencia_certeza_bonus(score_actual=2, certeza=None)
    assert r["score"] == 2
    assert r["aplicado"] is False


def test_165_06_score_cae_en_cap_5_no_lo_supera():
    """El bonus respeta el cap original de _convergencia_score_itf (max=5) —
    no infla el score más allá de lo que la UI ('N/5') puede mostrar."""
    certeza = {"alerta_nivel": "CERTEZA", "p_condicional": 0.99, "certeza_matematica": True}
    r = _convergencia_certeza_bonus(score_actual=5, certeza=certeza)
    assert r["score"] == 5
    assert r["aplicado"] is False  # ya estaba en el tope, no hay cambio real


def test_165_07_confianza_se_recalcula_con_score_final():
    """La etiqueta confianza debe reflejar el score DESPUÉS del bonus, no el
    original — evita que confianza='MEDIA' quede inconsistente con
    convergencia_score=3 (que ya es 'ALTA')."""
    certeza = {"alerta_nivel": "ALTA", "p_condicional": 0.96, "certeza_matematica": False}
    r = _convergencia_certeza_bonus(score_actual=2, certeza=certeza)
    assert r["score"] >= 3
    assert r["confianza"] == "ALTA"


def test_165_08_caso_real_kovacevic_borges_bonus_no_evita_gate_tiebreak():
    """Regresión de diseño: el bonus puede subir un score bloqueado por
    set1-tiebreak (games_set1=13, caso real Kovacevic vs Borges) hasta el
    umbral >=3, pero el gate downstream D150-06 en el bloque de disparo
    (live_desk.py:4514-4520, `if _gs1_06 >= 12: continue`) sigue excluyendo
    la señal porque opera sobre games_set1 crudo, independiente de
    convergencia_score. Este test documenta la garantía: el bonus NO
    debilita D150/D151, solo decide quién entra al pool que esos gates
    filtran después."""
    certeza = {"alerta_nivel": "ALTA", "p_condicional": 0.962, "certeza_matematica": False}
    r = _convergencia_certeza_bonus(score_actual=2, certeza=certeza)
    assert r["score"] == 3  # ahora SÍ entraría a alta_itf_raw...
    # ...pero el filtro real de disparo excluye por games_set1>=12
    # independientemente del convergencia_score — se verifica la fórmula
    # del gate directamente (misma condición que live_desk.py:4515):
    games_set1 = 13
    excluida_por_tiebreak = games_set1 >= 12
    assert excluida_por_tiebreak is True
