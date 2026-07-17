"""
core/weather_client.py — B108-06: Weather MVP observacional (Nodo-113)

Obtiene pronóstico de lluvia/viento para la ciudad del torneo usando open-meteo
(gratuito, sin API key). Devuelve weather_flag: 'RAIN_RISK'|'WIND_HIGH'|'CLEAR'|'UNKNOWN'.

Solo observacional — NO ajusta p_modelo. La hipótesis H113-01 acumula para
determinar si hay señal real antes de cualquier ajuste.

Diseño:
- get_weather_flag(pais, ciudad, superficie, fecha_iso) → str
- Cache en memoria por (lat, lon, fecha) para evitar llamadas duplicadas por sesión
- Si open-meteo falla → 'UNKNOWN' sin propagar excepción
"""
import logging
import urllib.request
import json
from functools import lru_cache
from datetime import date as _date

logger = logging.getLogger(__name__)

# ── Coordenadas aproximadas por país (lat, lon) ───────────────────────────────
# Cubre los ~30 países más frecuentes en el circuito ATP/WTA/ITF.
# Si el país no está → (None, None) → weather_flag='UNKNOWN'

_COUNTRY_COORDS: dict[str, tuple[float, float]] = {
    # Europa
    'Francia':         (48.85, 2.35),
    'Reino Unido':     (51.51, -0.13),
    'España':          (40.42, -3.70),
    'Italia':          (41.90, 12.50),
    'Alemania':        (52.52, 13.41),
    'Austria':         (48.21, 16.37),
    'Suiza':           (46.95, 7.44),
    'Países Bajos':    (52.37, 4.90),
    'Bélgica':         (50.85, 4.35),
    'Polonia':         (52.23, 21.01),
    'República Checa': (50.08, 14.44),
    'Rumania':         (44.43, 26.10),
    'Rumanía':         (44.43, 26.10),
    'Hungría':         (47.50, 19.04),
    'Croacia':         (45.81, 15.98),
    'Serbia':          (44.80, 20.47),
    'Grecia':          (37.98, 23.73),
    'Portugal':        (38.72, -9.14),
    'Suecia':          (59.33, 18.07),
    'Noruega':         (59.91, 10.75),
    'Dinamarca':       (55.68, 12.57),
    'Finlandia':       (60.17, 24.94),
    'Eslovaquia':      (48.15, 17.11),
    'Bulgaria':        (42.70, 23.32),
    'Ucrania':         (50.45, 30.52),
    'Rusia':           (55.75, 37.62),
    'Turquía':         (39.93, 32.86),
    'Eslovenia':       (46.05, 14.51),
    'Macedonia del Norte': (41.99, 21.43),
    'Bosnia':          (43.84, 18.36),
    'Serbia y Montenegro': (44.80, 20.47),
    # Americas
    'Estados Unidos':  (34.05, -118.24),
    'Canada':          (43.65, -79.38),
    'México':          (19.43, -99.13),
    'Argentina':       (-34.61, -58.38),
    'Brasil':          (-15.78, -47.93),
    'Chile':           (-33.46, -70.65),
    'Colombia':        (4.71, -74.07),
    'Perú':            (-12.04, -77.03),
    'Ecuador':         (-0.22, -78.51),
    'Bolivia':         (-16.50, -68.15),
    # Asia/Oceania
    'Australia':       (-33.87, 151.21),
    'Japón':           (35.69, 139.69),
    'China':           (39.91, 116.39),
    'India':           (28.61, 77.21),
    'Tailandia':       (13.75, 100.52),
    'Kazajistán':      (51.18, 71.45),
    # Africa
    'Marruecos':       (33.99, -6.85),
    'Sudáfrica':       (-25.75, 28.19),
    'Túnez':           (36.82, 10.17),
    'Egipto':          (30.06, 31.22),
}

# ── Thresholds ────────────────────────────────────────────────────────────────
_RAIN_MM_THRESHOLD  = 3.0   # mm/día → RAIN_RISK
_WIND_KMH_THRESHOLD = 30.0  # km/h   → WIND_HIGH

# ── Superficies que se juegan al aire libre ───────────────────────────────────
_OUTDOOR_SURFACES = {'clay', 'grass', 'arcilla', 'hierba', 'hard', 'dura', 'tierra'}

# Cache en memoria: (lat, lon, fecha_iso) → weather_flag
_WEATHER_CACHE: dict[tuple, str] = {}


def _get_coords(pais: str, ciudad: str | None = None) -> tuple[float | None, float | None]:
    """Resolución país → (lat, lon). ciudad ignorada por ahora (MVP)."""
    if not pais:
        return None, None
    # Normalizar: quitar paréntesis, strip
    pais_clean = pais.strip().split('(')[0].strip()
    return _COUNTRY_COORDS.get(pais_clean, (None, None))


@lru_cache(maxsize=64)
def _fetch_open_meteo(lat: float, lon: float, fecha_iso: str) -> dict:
    """
    Llama a open-meteo y retorna {'rain_mm': float, 'wind_kmh': float}.
    lru_cache evita llamadas duplicadas dentro de la misma sesión.
    Retorna {} en caso de error.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=precipitation_sum,wind_speed_10m_max"
        f"&timezone=auto&start_date={fecha_iso}&end_date={fecha_iso}"
    )
    try:
        req = urllib.request.urlopen(url, timeout=5)
        raw = json.loads(req.read().decode())
        daily = raw.get('daily', {})
        precip = (daily.get('precipitation_sum') or [0])[0] or 0.0
        wind   = (daily.get('wind_speed_10m_max') or [0])[0] or 0.0
        return {'rain_mm': float(precip), 'wind_kmh': float(wind)}
    except Exception as exc:
        logger.debug(f"[weather_client] open-meteo error: {exc}")
        return {}


def get_weather_flag(
    pais: str,
    superficie: str,
    fecha_iso: str | None = None,
    ciudad: str | None = None,
) -> str:
    """
    Retorna weather_flag para el partido.

    Args:
        pais:       país del torneo (campo 'pais' del zita file)
        superficie: tipo de cancha ('clay'/'grass'/'arcilla'/'hierba'/'hard'/...)
        fecha_iso:  fecha YYYY-MM-DD (default: hoy)
        ciudad:     ciudad (no usada en MVP, reservado para geocodificación futura)

    Returns:
        'RAIN_RISK' | 'WIND_HIGH' | 'CLEAR' | 'UNKNOWN'
        'UNKNOWN' cuando no hay coordenadas o la API falla.
    """
    # Solo outdoor
    if (superficie or '').lower() not in _OUTDOOR_SURFACES:
        return 'UNKNOWN'

    if not fecha_iso:
        fecha_iso = _date.today().isoformat()

    lat, lon = _get_coords(pais, ciudad)
    if lat is None:
        logger.debug(f"[weather_client] País sin coordenadas: '{pais}' → UNKNOWN")
        return 'UNKNOWN'

    cache_key = (lat, lon, fecha_iso)
    if cache_key in _WEATHER_CACHE:
        return _WEATHER_CACHE[cache_key]

    meteo = _fetch_open_meteo(lat, lon, fecha_iso)
    if not meteo:
        _WEATHER_CACHE[cache_key] = 'UNKNOWN'
        return 'UNKNOWN'

    rain_mm  = meteo.get('rain_mm', 0.0)
    wind_kmh = meteo.get('wind_kmh', 0.0)

    if rain_mm >= _RAIN_MM_THRESHOLD:
        flag = 'RAIN_RISK'
    elif wind_kmh >= _WIND_KMH_THRESHOLD:
        flag = 'WIND_HIGH'
    else:
        flag = 'CLEAR'

    _WEATHER_CACHE[cache_key] = flag
    logger.debug(
        f"[weather_client] {pais} {fecha_iso}: rain={rain_mm}mm wind={wind_kmh}km/h → {flag}"
    )
    return flag
