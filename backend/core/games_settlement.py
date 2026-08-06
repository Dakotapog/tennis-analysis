"""
core/games_settlement.py — Nodo-159 D159-01: settlement puro para picks de
juegos totales (OVER/UNDER games_live).

Función pura, sin I/O — la persistencia del score final vive en
live_desk.py (_snapshot_live_score) y la lectura en shadow_book.py
(_load_games_final_score).
"""

from typing import Tuple


def settle_games_outcome(direccion: str, linea: float, final_games: int) -> Tuple[bool, str]:
    """OVER/UNDER total de juegos. Sin ambigüedad de push — líneas son .5."""
    if direccion == "OVER":
        win = final_games > linea
    else:  # UNDER
        win = final_games < linea
    return win, f"{direccion} {linea} vs final={final_games}"
