"""
core/live_signal_bridge.py — D160-05 (Nodo-160 §6)

Puente GAMES-EN-VIVO ↔ GANADOR-EN-VIVO. Nodo-40 demostró que el mercado de
juegos/sets es alfa ortogonal al ganador — fusionar ambos scores en un solo
número sería estadísticamente incorrecto (se promediarían dos variables no
intercambiables). Este módulo trata las dos señales como evidencia
independiente que se REFUERZA cuando coincide, sin penalizar cuando diverge
(mismo patrón que score_directo/score_rival_value en Nodo-98: campos
separados, clasificación por casos, nunca suma ciega).

No abre apuestas de GANADOR en vivo por sí solo (el proyecto no tiene ese
rail hoy) — solo produce un estado de reconciliación (CONVERGENCIA_FUERTE/
DIVERGENCIA/NEUTRO) para: (a) contexto en el dashboard live_desk, (b) booster
de confianza en X2 steam-lag (Nodo-111) y n_obs efectivo en D160-04, y (c)
acumulación de evidencia para H160-01.
"""
from typing import Dict, Optional

_SCORE_DIRECTO_MIN = 3
_DRIFT_DIVERGENCIA = 0.05  # drift_pct > esto = cuota del favorito subiendo (mercado dudando)
_BREAK_STATES_CONFIRMANDO = ("BREAK_POSIBLE", "BREAK_CONFIRMADO")


def reconciliar_senales_partido(partido_key: str, games_state: Dict, winner_state: Dict) -> Dict:
    """
    games_state: {direccion, certeza_matematica, p_condicional, zona,
                  break_situation, serving} — de _check_games_convergencia().
                  Se asume (mismo supuesto que D150/D151 gates) que la señal
                  ya está referenciada al favorito del partido.
    winner_state: {score_directo, break_state, drift_pct, direccion_favorito}
                  — score_directo pre-partido (Nodo-98) + break_state/drift_pct
                  en vivo (detect_break_state, Nodo-100). drift_pct con
                  convención del proyecto (D150-01/D150-05): negativo = cuota
                  bajando = mercado confirma al favorito; positivo = cuota
                  subiendo = mercado dudando.

    Retorna {"partido_key", "estado", "razon"} — estado en
    CONVERGENCIA_FUERTE/DIVERGENCIA/NEUTRO. Nunca lanza — datos insuficientes
    en cualquiera de los dos lados degrada a NEUTRO sin efecto.
    """
    games_state = games_state or {}
    winner_state = winner_state or {}

    zona = games_state.get("zona")
    certeza = games_state.get("certeza_matematica")
    break_situation = games_state.get("break_situation")
    games_dominante = bool(zona == "DOMINANTE" and certeza and break_situation)

    if not games_dominante:
        return {"partido_key": partido_key, "estado": "NEUTRO", "razon": "games_sin_certeza_dominante"}

    drift_pct = winner_state.get("drift_pct")
    score_directo = winner_state.get("score_directo")
    break_state = winner_state.get("break_state")

    if drift_pct is not None and drift_pct > _DRIFT_DIVERGENCIA:
        return {
            "partido_key": partido_key, "estado": "DIVERGENCIA",
            "razon": f"games_dominante_pero_drift_favorito={drift_pct:.1%}_contra",
        }

    if (score_directo is not None and score_directo >= _SCORE_DIRECTO_MIN
            and break_state in _BREAK_STATES_CONFIRMANDO):
        return {
            "partido_key": partido_key, "estado": "CONVERGENCIA_FUERTE",
            "razon": f"games_dominante+score_directo={score_directo}+break_state={break_state}",
        }

    return {"partido_key": partido_key, "estado": "NEUTRO", "razon": "sin_convergencia_suficiente"}
