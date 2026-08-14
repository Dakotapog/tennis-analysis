"""core/p_wave.py — Nodo-181 D181-03.

Detector de ONDA P: el inicio del movimiento de cuota, no su consumación.
Onda S (lo que `certeza_matematica` ya detecta) dispara DESPUÉS de que el
mercado ya reprecio — D181-01 midio 0.0% de movimiento posterior en 4 de 6
disparos historicos. Este modulo existe para capturar el onset, mientras
todavia queda movimiento por atrapar.

Reusa `analysis.velocity_monitor.velocity_zscore` (Nodo-71 / D168-01) para
el z-score de velocidad — REGLA-T53, no se reimplementa esa formula aqui.

Puro, sin I/O, sin `datetime.now()`. Patron `core/games_live_model.py`.
NO REPORTE_SOLO: a diferencia de `scripts/lead_time_report.py`, este modulo
si puede alimentar gates (D181-04 quorum_sensores, D181-06 panel P_VENTANA)
— por eso NO importa nada de `scripts/lead_time_report.py` (que si es
REPORTE_SOLO estricto y no puede ser leido por ningun gate).
"""
from typing import Dict, List, Optional

from analysis.velocity_monitor import velocity_zscore

_MOV_ACUMULADO_MAX_PCT = 15.0  # >= esto ya es onda S, no P (criterio 2)
_GAMES_PLAYED_FRACCION_MAX = 0.6  # criterio 3: todavia debe quedar partido


def _opuesta(direccion: str) -> str:
    return "UNDER" if direccion.upper() == "OVER" else "OVER"


def _minutos_relativos(ventana: List[dict]) -> List[float]:
    """Minutos transcurridos desde el primer punto de la ventana, con guard
    de rollover de medianoche (mismo principio que D181-01, reescrito local
    porque este modulo no puede importar scripts/lead_time_report.py —
    REPORTE_SOLO estricto, D181-01)."""
    out = []
    day_offset = 0
    prev_mins = None
    base_mins = None
    for punto in ventana:
        h, m = map(int, punto["ts"].split(":"))
        mins = h * 60 + m
        if prev_mins is not None and mins < prev_mins:
            day_offset += 1440
        prev_mins = mins
        abs_mins = mins + day_offset
        if base_mins is None:
            base_mins = abs_mins
        out.append(float(abs_mins - base_mins))
    return out


def detectar_onda_p(serie: List[dict], linea: float, direccion: str,
                     z_min: float = 2.0, n_min: int = 4) -> Dict:
    """{"detectada": bool, "ts_onset": str|None, "z": float,
    "magnitud_acumulada_pct": float, "direccion_implicita": "OVER"|"UNDER"|None,
    "n_puntos": int}

    Los tres criterios de onset, simultaneos (Nodo-181 D181-03):
    1. |z| de velocity_zscore sobre los ultimos `n_min` puntos >= z_min.
    2. Movimiento acumulado en esa ventana < 15% (si ya se movio mas, es
       onda S consumada, no el inicio).
    3. games_played del ultimo punto < linea*0.6 (todavia queda partido).

    `direccion_implicita` se deriva del signo de z: cuota de esta serie
    acortandose (z<0, convencion STEAM de Nodo-71) confirma `direccion`;
    cuota alargandose (z>0) implica la direccion opuesta. `detectada` solo
    es True cuando la onda confirma la `direccion` pedida — un onset que
    fade la direccion no es una oportunidad para ese lado.

    Contrato de `serie`: es el historial de cuota de UN solo lado del
    mercado (mismo shape que `games_odds_history_*.json`, clave
    "Partido_DIRECCION" — igual que D181-01). Pasar la serie de un lado
    junto con la `direccion` del lado contrario es un error del caller,
    no algo que este modulo pueda detectar por si solo.
    """
    n_puntos = len(serie)
    vacio = {
        "detectada": False, "ts_onset": None, "z": 0.0,
        "magnitud_acumulada_pct": 0.0, "direccion_implicita": None,
        "n_puntos": n_puntos,
    }
    if n_puntos < n_min:
        return vacio

    ventana = serie[-n_min:]
    odds = [p["cuota"] for p in ventana]
    tiempos = _minutos_relativos(ventana)
    z_last = velocity_zscore(odds, tiempos)["z_last"]
    if z_last is None:
        return vacio

    cuota_onset = ventana[0]["cuota"]
    cuota_final = ventana[-1]["cuota"]
    magnitud_acumulada_pct = (abs(cuota_final - cuota_onset) / cuota_onset * 100
                               if cuota_onset else 0.0)

    _dir = direccion.upper()
    direccion_implicita = _dir if z_last < 0 else _opuesta(_dir) if z_last > 0 else None

    games_played_final = serie[-1].get("games_played")
    criterio_1 = abs(z_last) >= z_min
    criterio_2 = magnitud_acumulada_pct < _MOV_ACUMULADO_MAX_PCT
    criterio_3 = (games_played_final is not None
                  and games_played_final < linea * _GAMES_PLAYED_FRACCION_MAX)

    detectada = bool(criterio_1 and criterio_2 and criterio_3 and direccion_implicita == _dir)

    return {
        "detectada": detectada,
        "ts_onset": ventana[0].get("ts") if detectada else None,
        "z": round(z_last, 2),
        "magnitud_acumulada_pct": round(magnitud_acumulada_pct, 2),
        "direccion_implicita": direccion_implicita,
        "n_puntos": len(ventana),
    }


# ---------------------------------------------------------------------------
# D181-04 — quorum_sensores
# ---------------------------------------------------------------------------

_DRIFT_PCT_MIN = 5.0  # puntos porcentuales; mismo umbral que core/live_signal_bridge._DRIFT_DIVERGENCIA (0.05)
_EDGE_LIVE_MIN = 0.0  # cualquier edge positivo cuenta como sensor activo

_FAMILIAS_SENSORES = {
    "MERCADO": ("p_wave_detectada", "steam_confirmado", "drift_pct"),
    "MODELO": ("mc_p_condicional", "p_condicional", "edge_live"),
    "ESTADO": ("break_situation", "serving", "games_set1", "score_data"),
}


def _sensor_activo(nombre: str, valor) -> bool:
    """Definicion de 'sensor activo' por tipo de dato (D181-04, decision de
    implementacion — el spec nombra los sensores pero no fija el umbral de
    cada uno):

    - bool: activo si es True.
    - drift_pct: activo si |valor| >= _DRIFT_PCT_MIN (5 puntos, mismo umbral
      que ya usa core/live_signal_bridge para "mercado dudando").
    - mc_p_condicional / p_condicional: activo si > 0.5 (favorece esta
      direccion, no un simple 50/50).
    - edge_live: activo si > 0 (cualquier edge positivo).
    - games_set1 / score_data: activo si el valor esta presente (not None /
      no vacio) — para esta familia, tener dato observado del marcador ya
      es la señal (a diferencia de MERCADO/MODELO que necesitan una
      direccion o magnitud).
    """
    if valor is None:
        return False
    if isinstance(valor, bool):
        return valor
    if nombre == "drift_pct":
        return abs(valor) >= _DRIFT_PCT_MIN
    if nombre in ("mc_p_condicional", "p_condicional"):
        return valor > 0.5
    if nombre == "edge_live":
        return valor > _EDGE_LIVE_MIN
    if nombre in ("games_set1", "score_data"):
        return bool(valor)
    return bool(valor)


def quorum_sensores(senal: Dict) -> Dict:
    """Nodo-181 D181-04 — quorum por familias independientes.

    Clasifica la evidencia disponible en `senal` en tres familias
    (MERCADO/MODELO/ESTADO, ver `_FAMILIAS_SENSORES`) y exige >=1 sensor
    activo en >=2 familias distintas para `quorum_ok` (3/3 recomendado
    para nivel ACCION en D181-06 — esa comparacion vive en el caller, no
    aqui).

    No reemplaza `convergencia_score` (D165/D166): lo complementa como
    requisito adicional de independencia — dos sensores de la misma
    familia pueden ser el mismo dato contado dos veces. Los gates
    D150/D151/D164 quedan intactos.

    Returns: {"familias_activas": [str], "n_familias": int,
              "quorum_ok": bool, "detalle": {familia: {sensor: bool}}}
    """
    detalle: Dict[str, Dict[str, bool]] = {}
    familias_activas = []
    for familia, sensores in _FAMILIAS_SENSORES.items():
        estado_sensores = {s: _sensor_activo(s, senal.get(s)) for s in sensores}
        detalle[familia] = estado_sensores
        if any(estado_sensores.values()):
            familias_activas.append(familia)

    n_familias = len(familias_activas)
    return {
        "familias_activas": familias_activas,
        "n_familias": n_familias,
        "quorum_ok": n_familias >= 2,
        "detalle": detalle,
    }
