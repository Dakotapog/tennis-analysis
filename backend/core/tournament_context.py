"""
core/tournament_context.py — Nodo-51 F1: TournamentContext como Entidad

Hace que el torneo sea un objeto con atributos en lugar de un string suelto.
En PASO 1, cada match dict recibe un subdict 'tournament_context' con:

    {
        "nombre":               str,   # torneo_completo limpio
        "tier":                 str,   # grand_slam | atp1000 | atp500 | challenger | itf
        "superficie":           str,   # clay | grass | hard | unknown
        "season_transition_flag": bool # True si <14 días de frontera de superficie
    }

La superficie se resuelve UNA vez aquí y viaja con el match.
Nodo-46 (F4) la lee de tournament_context — no la infiere de nuevo.

Principio rector (MM-4 — Nicho Ecológico):
    El estado Markov no es del jugador — es del par (jugador, hábitat).
    TournamentContext define el hábitat: (tier, superficie, temporada).
    Sin él, el modelo aplica forma de hierba a un partido en hard.
"""
import re
from datetime import date
from typing import Optional

from config import detectar_tier


# ══════════════════════════════════════════════════════════════════════════════
# _SURFACE_MAP — fuente canónica de normalización de superficie (Nodo-46 spec)
# Normaliza cualquier variante a: 'clay' | 'grass' | 'hard' | 'unknown'
# ══════════════════════════════════════════════════════════════════════════════

_SURFACE_MAP: dict = {
    # Grass
    'hierba':       'grass',
    'grass':        'grass',
    'herb':         'grass',
    # Hard
    'dura':         'hard',
    'hard':         'hard',
    'hardcourt':    'hard',
    'indoor hard':  'hard',
    'carpet':       'hard',
    # Clay
    'arcilla':      'clay',
    'clay':         'clay',
    'tierra':       'clay',
}

# ══════════════════════════════════════════════════════════════════════════════
# _KNOWN_TOURNAMENT_SURFACES — torneos cuya superficie no aparece en el nombre
# ══════════════════════════════════════════════════════════════════════════════

_KNOWN_TOURNAMENT_SURFACES: dict = {
    # Grand Slams
    'wimbledon':        'grass',
    'roland garros':    'clay',
    'french open':      'clay',
    'australian open':  'hard',
    'us open':          'hard',
    # ATP 1000
    'indian wells':     'hard',
    'miami':            'hard',
    'monte-carlo':      'clay',
    'monte carlo':      'clay',
    'madrid':           'clay',
    'rome':             'clay',
    'canada':           'hard',
    'toronto':          'hard',
    'montreal':         'hard',
    'cincinnati':       'hard',
    'shanghai':         'hard',
    'paris masters':    'hard',
    # ATP 500 / WTA
    'queen':            'grass',   # Queen's Club
    'queens club':      'grass',
    'halle':            'grass',
    'eastbourne':       'grass',
    'birmingham':       'grass',
    'berlin':           'clay',
    'barcelona':        'clay',
    'hamburg':          'clay',
    'geneva':           'clay',
    'lyon':             'clay',
    # Challengers / ITF conocidos
    'ilkley':           'grass',
    'nottingham':       'grass',
    'surbiton':         'grass',
    'rosmalen':         'grass',
    'cary':             'hard',
}

# ══════════════════════════════════════════════════════════════════════════════
# Calendario de transiciones de superficie — (month, day) aprox. anuales
#
#   Hard → Clay:  April 7   (Monte-Carlo, inicio arcilla)
#   Clay → Grass: June 2    (Queen's / Halle)
#   Grass → Hard: July 12   (post-Wimbledon; US Open Series)
#   Hard → Indoor: October 28 (Vienna/Paris indoor swing)
#
# Invariante de test: June 30 → True (Grass→Hard frontera July 12, 12 días < 14)
# ══════════════════════════════════════════════════════════════════════════════

_TRANSITION_BOUNDARIES: list = [
    (4,  7),   # Hard → Clay
    (6,  2),   # Clay → Grass
    (7, 12),   # Grass → Hard
    (10, 28),  # Hard → Indoor
]

_TRANSITION_WINDOW_DAYS: int = 14


# ══════════════════════════════════════════════════════════════════════════════
# API PÚBLICA
# ══════════════════════════════════════════════════════════════════════════════

def normalize_surface(surface_str: str) -> str:
    """
    Normaliza cualquier variante de superficie a canonical: clay | grass | hard | unknown.

    Usado por Nodo-46 (F4) para asegurar comparación correcta entre
    la superficie del torneo actual y las superficies del historial H2H.
    """
    if not surface_str:
        return 'unknown'
    return _SURFACE_MAP.get(surface_str.lower().strip(), 'unknown')


def _surface_from_tournament_name(torneo_completo: Optional[str]) -> str:
    """
    Infiere superficie desde el nombre del torneo. Prioridad:

    1. Tabla de torneos conocidos (Wimbledon, Roland Garros, Cary...)
       — cubrre torneos cuyo nombre no incluye la superficie.
    2. Keywords explícitos en el nombre (clay, grass, hard, arcilla, hierba, dura...)
       — el FlashScore feed incluye la superficie en el nombre para muchos torneos.
    3. 'unknown' si no se puede inferir.

    La superficie se determina UNA vez en PASO 1 y viaja con el match.
    """
    if not torneo_completo:
        return 'unknown'

    t_lower = torneo_completo.lower()

    # 1. Tabla de torneos conocidos (prioridad — evita falsos positivos por keywords)
    # Word-boundary matching prevents 'halle' from matching inside 'challenger'
    for keyword, surface in _KNOWN_TOURNAMENT_SURFACES.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', t_lower):
            return surface

    # 2. Keywords explícitos en el nombre
    for keyword, surface in _SURFACE_MAP.items():
        if keyword in t_lower:
            return surface

    return 'unknown'


def _days_to_nearest_boundary(reference_date: date) -> int:
    """Devuelve la distancia en días a la frontera de temporada más cercana."""
    year = reference_date.year
    min_days = 999
    for month, day in _TRANSITION_BOUNDARIES:
        for y in (year - 1, year, year + 1):
            try:
                boundary = date(y, month, day)
                days = abs((boundary - reference_date).days)
                if days < min_days:
                    min_days = days
            except ValueError:
                pass
    return min_days


def _season_transition_flag(reference_date: Optional[date] = None) -> bool:
    """
    True si la fecha está a menos de _TRANSITION_WINDOW_DAYS días de una frontera
    de superficie del calendario ATP.

    Caso documentado en spec: June 30 → True (hierba→hard, frontera July 12: 12 días).
    """
    if reference_date is None:
        reference_date = date.today()
    return _days_to_nearest_boundary(reference_date) < _TRANSITION_WINDOW_DAYS


def build_tournament_context(
    torneo_completo: Optional[str],
    match_date: Optional[date] = None,
) -> dict:
    """
    Construye el subdict 'tournament_context' para un match dict del PASO 1.

    Args:
        torneo_completo: nombre completo del torneo (desde FlashScore o Kambi).
        match_date: fecha del partido; None → usa date.today().

    Returns:
        {
            "nombre":               str,
            "tier":                 str,   # detectar_tier() desde config.py
            "superficie":           str,   # _surface_from_tournament_name()
            "season_transition_flag": bool
        }
    """
    nombre = (torneo_completo or '').strip()
    return {
        "nombre":                 nombre,
        "tier":                   detectar_tier(nombre),
        "superficie":             _surface_from_tournament_name(nombre),
        "season_transition_flag": _season_transition_flag(match_date),
    }
