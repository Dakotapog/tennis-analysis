"""core/games_arithmetic.py — Nodo-181 D181-12.

Cota DETERMINISTA de juegos restantes en un partido de tenis en vivo — sin
probabilidad, sin modelo. Solo aritmética de formato (misma familia exógena
que `perdida_matematica` de D180-03; ver Nodo-181 §3.B.3).

Caso motivador (§3.B.1): Nally C. vs Kessler M., 6:3 5:2 con break point a
favor de Nally. La dashboard mostraba zona "26-32+ juegos" — una distribución
de partido completo a 3 sets — cuando el partido estaba a UN juego de
terminar en 17. Este módulo existe para que esa distancia se calcule, no se
adivine.

Puro, sin I/O — mismo patrón que `core/games_settlement.py` / `core/monte_carlo_games.py`.
"""


def _games_min_para_ganar_set(favor: int, contra: int) -> int:
    """Mínimo de juegos adicionales para que el lado con `favor` juegos gane
    el set en curso, asumiendo que el rival (con `contra` juegos) no gana
    ninguno más — el escenario más rápido posible. Formato estándar ATP/WTA:
    6 juegos con 2 de margen, o tiebreak en 6-6 (7-6)."""
    if favor >= 6 and favor - contra >= 2:
        return 0
    if favor == 6 and contra == 6:
        return 1
    return max(6 - favor, contra + 2 - favor, 0)


def juegos_restantes_min(sets_ganados_home: int, sets_ganados_away: int,
                          juegos_home: int, juegos_away: int,
                          sets_a_ganar: int = 2) -> int:
    """Mínimo absoluto de juegos que faltan para que el partido termine.

    Toma el camino más rápido posible entre los dos jugadores: cualquier
    intercalado de sets ganados por el otro lado solo puede sumar juegos,
    nunca restar del total — por eso el mínimo real es el menor de los dos
    caminos directos, no una combinación de ambos.
    """
    home_sets_faltan = max(sets_a_ganar - sets_ganados_home, 0)
    away_sets_faltan = max(sets_a_ganar - sets_ganados_away, 0)

    camino_home = (_games_min_para_ganar_set(juegos_home, juegos_away)
                   + max(home_sets_faltan - 1, 0) * 6)
    camino_away = (_games_min_para_ganar_set(juegos_away, juegos_home)
                   + max(away_sets_faltan - 1, 0) * 6)

    return min(camino_home, camino_away)


_MAX_JUEGOS_POR_SET = 13  # 6-6 + tiebreak (7-6) = techo estandar ATP/WTA de un set


def total_alcanzable(juegos_jugados_total: int,
                      sets_ganados_home: int, sets_ganados_away: int,
                      juegos_home: int, juegos_away: int,
                      sets_a_ganar: int = 2) -> tuple:
    """(total_min, total_max_set_actual).

    total_min: juegos_jugados_total + el mínimo absoluto de juegos_restantes_min.

    total_max_set_actual: techo determinista del PARTIDO COMPLETO, condicionado
    al marcador real ya jugado (no una distribución ciega de partido a 3 sets
    desde cero) — juegos ya jugados en sets completados + techo del set en
    curso (13 juegos: 6-6 y tiebreak 7-6) + techo de cualquier set decisivo
    que aún pueda hacer falta bajo el formato (también 13 cada uno). Es más
    estrecho que "26-32+" solo cuando el marcador ya descarta sets futuros —
    exactamente el error que §3.B.2 documenta.
    """
    minimo_adicional = juegos_restantes_min(sets_ganados_home, sets_ganados_away,
                                             juegos_home, juegos_away, sets_a_ganar)
    total_min = juegos_jugados_total + minimo_adicional

    juegos_previos_completados = juegos_jugados_total - juegos_home - juegos_away
    sets_totales_formato = 2 * sets_a_ganar - 1
    sets_ya_contados = sets_ganados_home + sets_ganados_away + 1  # + el set en curso
    sets_futuros_max = max(sets_totales_formato - sets_ya_contados, 0)

    total_max_set_actual = (juegos_previos_completados
                             + _MAX_JUEGOS_POR_SET
                             + sets_futuros_max * _MAX_JUEGOS_POR_SET)

    return total_min, total_max_set_actual


def estado_linea(linea: float, direccion: str, juegos_jugados: int,
                  total_min: int, total_max: int) -> str:
    """"IMPOSIBLE" | "RESUELTO" | "VIVO" para una línea de juegos dada la
    cota determinista de total_alcanzable(). Sin probabilidad: solo compara
    la línea contra los límites aritméticos ya calculados.
    """
    if direccion == "OVER":
        if juegos_jugados > linea:
            return "RESUELTO"
        if total_max < linea:
            return "IMPOSIBLE"
        return "VIVO"
    if direccion == "UNDER":
        if total_min > linea:
            return "IMPOSIBLE"
        if total_max <= linea:
            return "RESUELTO"
        return "VIVO"
    raise ValueError(f"direccion invalida: {direccion!r} (esperado OVER/UNDER)")
