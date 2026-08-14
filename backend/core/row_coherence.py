"""core/row_coherence.py — Nodo-181 D181-13.

Gate de coherencia de fila, BLOQUEANTE para el badge de pick. No borra la
fila del panel (X3 GAMES es REPORTE_SOLO / uso manual en Betplay) — la
demueve a INCOHERENTE con motivo explícito visible, nunca se muestra como
pick. Mismo patrón que check_contradictions.py (auditoría, no gate de
disparo), aplicado a la fila renderizada en vez de al documento.

Caso motivador (Nodo-181 §3.B.1): Nally C. vs Kessler M., 6:3 5:2 break
point a favor de Nally — dirección apostada OVER pero banner "CONFIRMAR
UNDER". Esa fila nunca debió salir como pick.

Puro, sin I/O.
"""
from typing import Optional

MOTIVO_DIRECCION_CONTRADICE_BANNER = "direccion_contradice_banner"
MOTIVO_EDGE_INSUFICIENTE = "edge_insuficiente"
MOTIVO_ZONA_INALCANZABLE = "zona_inalcanzable"
MOTIVO_ETIQUETA_NO_CORRESPONDE = "etiqueta_no_corresponde_numero"
MOTIVO_SIN_DATOS = "sin_datos_suficientes"

EDGE_MINIMO_PCT = 0.0  # edge negativo nunca es coherente con un pick

_ETIQUETAS_SCORE5 = (  # (piso_score_inclusive, etiqueta_esperada) — score 0-5
    (3, "ALTA"), (2, "MEDIA"), (0, "BAJA"),
)


def _etiqueta_esperada_score5(score: int) -> str:
    for piso, etiqueta in _ETIQUETAS_SCORE5:
        if score >= piso:
            return etiqueta
    return "BAJA"


def evaluar_coherencia_fila(
    direccion: Optional[str] = None,
    banner_direccion: Optional[str] = None,
    edge_pct: Optional[float] = None,
    zona_lo: Optional[float] = None,
    total_max: Optional[int] = None,
    label_cualitativa: Optional[str] = None,
    score_num: Optional[int] = None,
    score_max: Optional[int] = None,
) -> tuple:
    """(estado, motivo). estado en {"OK", "INCOHERENTE"}.

    Las 4 condiciones de D181-13 son un OR — basta con que una dispare para
    que la fila caiga. Cada condición se evalúa solo si sus propios datos
    están presentes (la tabla X3 no siempre calcula edge_pct explícito para
    cada fila). Si NINGUNA de las 4 tiene datos suficientes, fail-closed:
    la fila se trata como INCOHERENTE, no como válida (D181-13 verbatim).
    """
    evaluable = False

    if direccion and banner_direccion:
        evaluable = True
        if direccion.upper() not in banner_direccion.upper():
            return "INCOHERENTE", MOTIVO_DIRECCION_CONTRADICE_BANNER

    if edge_pct is not None:
        evaluable = True
        if edge_pct < EDGE_MINIMO_PCT:
            return "INCOHERENTE", MOTIVO_EDGE_INSUFICIENTE

    if zona_lo is not None and total_max is not None:
        evaluable = True
        if zona_lo > total_max:
            return "INCOHERENTE", MOTIVO_ZONA_INALCANZABLE

    if label_cualitativa and score_num is not None and score_max == 5:
        evaluable = True
        if label_cualitativa.upper() != _etiqueta_esperada_score5(score_num):
            return "INCOHERENTE", MOTIVO_ETIQUETA_NO_CORRESPONDE

    if not evaluable:
        return "INCOHERENTE", MOTIVO_SIN_DATOS

    return "OK", ""
