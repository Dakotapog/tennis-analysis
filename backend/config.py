"""
config.py — Constantes de configuración centralizadas del pipeline de tenis.

D-17: Centraliza constantes que estaban dispersas en múltiples archivos:
  - FLASHSCORE_BASE, FLASHSCORE_HEADERS ← validar_con_api.py
  - TOTAL_MATCHES_TO_PROCESS, BROWSER_HEADLESS, BROWSER_SLOW_MO ← scraping/h2h_extractor.py

NO incluye MAX_RAW_SCORES / DEFAULT_WEIGHTS — están co-ubicados con la lógica
de normalización en normalization.py y no son "config dispersa".
"""

# ──────────────────────────────────────────────────────────────────────────────
# FlashScore Ninja API
# ──────────────────────────────────────────────────────────────────────────────

FLASHSCORE_BASE = "https://global.flashscore.ninja/202/x/feed"

FLASHSCORE_HEADERS = {
    "X-Fsign": "SW9D1eZo",
    "Referer": "https://www.flashscore.co/",
    "Origin": "https://www.flashscore.co",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "*/*",
}

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline — parámetros de scraping
# ──────────────────────────────────────────────────────────────────────────────

TOTAL_MATCHES_TO_PROCESS = 80
BROWSER_HEADLESS = True          # Ejecutar navegador sin interfaz gráfica (WSL)
BROWSER_SLOW_MO = 250            # Retardo entre acciones de Playwright (ms)

# ──────────────────────────────────────────────────────────────────────────────
# Clasificación de tier de torneo — fuente única de verdad (T21-02)
# Usada por edge_calculator.py (λ), rivalry_analyzer.py (pesos), trader (ρ)
# ──────────────────────────────────────────────────────────────────────────────

def detectar_tier(torneo_completo: str) -> str:
    """
    Detecta el tier del torneo desde su nombre completo.
    Retorna: 'grand_slam' | 'atp1000' | 'atp500' | 'challenger' | 'itf'

    Fuente única de verdad para clasificación de tier — importar desde aquí,
    nunca duplicar lógica en otros módulos.
    """
    t = (torneo_completo or '').lower()
    if any(gs in t for gs in ('roland garros', 'french open', 'wimbledon',
                               'australian open', 'us open', 'grand slam')):
        return 'grand_slam'
    # ITF antes de ATP1000/Challenger para evitar que keywords de ciudad (madrid, rome)
    # matcheen atp1000 cuando el nombre incluye "ITF" o "M15/M25/W15/W25"
    if any(k in t for k in ('itf', 'm15', 'm25', 'm75', 'w15', 'w25',
                              'w35', 'w50', 'w60', 'w75', 'w100', 'w125')):
        return 'itf'
    if any(k in t for k in ('atp 1000', 'atp1000', 'masters 1000', 'masters1000',
                              'indian wells', 'miami open', 'monte-carlo', 'madrid',
                              'rome', 'canada', 'cincinnati', 'shanghai', 'paris masters',
                              'toronto', 'montreal')):
        return 'atp1000'
    if any(k in t for k in ('atp 500', 'atp500', '500')):
        return 'atp500'
    # Challenger — incluye keywords de ciudades conocidas de circuito Challenger
    if any(k in t for k in ('challenger', 'heilbronn', 'perugia', 'prostejov',
                              'foggia', 'makarska', 'kayseri', 'monastir', 'lakewood',
                              'centurion', 'caltanissetta', 'cuiaba', 'banja luka',
                              'brasilia', 'focsani', 'ontinyent', 'kursumlijska',
                              'tsaghkadzor', 'szentendre', 'rosbach', 'ljubljana',
                              'sumter', 'tyler')):
        return 'challenger'
    return 'atp500'  # fallback conservador (ni GS ni Challenger confirmado)
