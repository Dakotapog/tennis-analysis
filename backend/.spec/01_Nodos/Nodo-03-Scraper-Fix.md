# Nodo-03: Fix extraer_URL_partidos_version2.py (3 Bugs Críticos)

> **Wikilinks:** [[Mandatos-No-Negociables]] | [[Sprint-Pipeline]] | [[Pipeline-Arquitectura]] | [[Grafo-Dependencias-Datos]] | [[Fuentes-Datos]] | [[Nodo-01-Edge-Calculator]] | [[Nodo-02-Markov-Changepoint]] | [[Nodo-03-Scraper-Fix]] | [[Nodo-04-Dataset-Fix]] | [[Nodo-05-Validacion-API]] | [[Nodo-06-Erdos-Graph]]
> ✅ Estado 2026-05-29: Bugs T03-01/02/03/04 RESUELTOS — 235 partidos con h2h_url válidas, 33 torneos, superficie clay/hard/grass funcionando. Scraper corregido con selector `.headerLeague__title`.
> **Evidencia:** 423 partidos scraped el 2026-05-28, los 3 bugs confirmados en data real

**Prioridad:** ALTA — bloquea surface_specialization y el análisis de torneo
**Archivo:** `extraer_URL_partidos_version2.py`
**Evidencia:** 423 partidos scraped el 2026-05-28, los 3 bugs confirmados en logs

---

## Contrato de Señal (Signal Contract)

```
PRODUCE:  S1_MATCH_LIST → data/zita_tennis_matches_FECHA.json
            h2h_url: "https://.../#/h2h/overall/"   (Bug 1 fix)
            match_id: "rDQ3y6to"                     (Bug 2 fix)
            torneo: "Roland Garros (France)"          (Bug 3 fix)
            superficie: "clay"                        (Bug 3 fix)

CONSUME:  FlashScore.com DOM (Playwright)
            .event__match → elementos de partido
            .event__header → headers de torneo

PREREQUISITO: Playwright instalado y chromium disponible
              FlashScore.com accesible (sin bloqueo geográfico)
```

---

## Conexiones Cross-Nodo (CX)

| CX | De → A | Impacto |
|---|---|---|
| CX-01 | [[Nodo-03-Scraper-Fix]] Bug2 → [[Nodo-05-Validacion-API]] | match_id real habilita dc_1_{event_id} |
| CX-03 | [[Nodo-03-Scraper-Fix]] Bug3 → rivalry_analyzer.py | superficie limpia → surface_specialization>0% |
| CX-05 | superficie → [[Nodo-04-Dataset-Fix]] features | surface como feature ML válida (antes era garbage) |
| CX-06 | [[Nodo-05-Validacion-API]] accuracy_clay → [[Nodo-01-Edge-Calculator]] | calibrar p_historica por superficie |

---

---

## Bug 1: h2h_url = None (0/423 partidos)

**Causa:** El scraper extrae `match_url` pero nunca construye `h2h_url`.
**Impacto:** `extraer_historh2h.py` no puede navegar a la sección H2H.

**Fix:**
```python
# Después de extraer match_url, construir h2h_url:
match_url_limpia = match_url.split('?')[0].rstrip('/')
h2h_url = match_url_limpia + '/#/h2h/overall/'

# Ejemplo:
# match_url: https://www.flashscore.com/match/tennis/cobolli-flavio-zDtaCcPe/wu-yibing-8ASNvPfK/?mid=rDQ3y6to
# h2h_url:   https://www.flashscore.com/match/tennis/cobolli-flavio-zDtaCcPe/wu-yibing-8ASNvPfK/#/h2h/overall/
```

## Bug 2: match_id = "tennis" (no el ID real del partido)

**Causa:** El scraper extrae solo el tipo de deporte de la URL en lugar del ID único.
**Impacto:** Sin match_id real, no se puede usar la FlashScore Ninja API (`dc_1_{event_id}`).

**Fix:**
```python
import re

# Extraer event_id del parámetro ?mid= de la match_url
match_event_id = None
mid_match = re.search(r'\?mid=([^&]+)', match_url)
if mid_match:
    match_event_id = mid_match.group(1)

# Ejemplo:
# match_url: ...?mid=rDQ3y6to
# match_event_id: rDQ3y6to
```

## Bug 3: torneo = "Sin Torneo Asignado" (423/423 partidos)

**Causa:** El parser del DOM no extrae correctamente el nombre del torneo desde FlashScore.
**Impacto:** 
- `tipo_cancha` = HTML garbage → `surface_specialization = 0%` en el modelo
- No se puede filtrar por torneo ni superficie

**Fix:** Antes de procesar partidos, extraer el torneo del elemento DOM de cada sección de torneo:
```python
# En el bucle principal de extracción, leer el header del torneo:
# El DOM de FlashScore estructura los partidos bajo elementos de torneo
# Selector real (verificado 2026-05-29): .headerLeague__title
# NOTA: .event__header y .tournament__name NO son los selectores reales en producción

torneo_actual = "Desconocido"
for elemento in elementos:
    # Si es un header de torneo (no un partido)
    if elemento.get_attribute('class') and 'headerLeague__title' in elemento.get_attribute('class'):
        torneo_texto = elemento.inner_text().strip()
        if torneo_texto:
            torneo_actual = torneo_texto
    # Si es un partido, asignar el torneo_actual
    elif es_partido(elemento):
        partido['torneo'] = torneo_actual
        partido['superficie'] = extraer_superficie(torneo_actual)
```

**Extracción de superficie del nombre del torneo:**
```python
def extraer_superficie(torneo_texto: str) -> str:
    """Roland Garros → clay | Wimbledon → grass | Australian/US Open → hard"""
    t = torneo_texto.lower()
    if 'roland garros' in t or 'french open' in t:
        return 'clay'
    if 'wimbledon' in t:
        return 'grass'
    if 'australian open' in t or 'us open' in t or 'united states' in t:
        return 'hard'
    if 'clay' in t or 'arcilla' in t:
        return 'clay'
    if 'grass' in t or 'hierba' in t:
        return 'grass'
    if 'hard' in t or 'dura' in t:
        return 'hard'
    return 'unknown'
```

## Output esperado después del fix

```json
{
    "jugador1": "Arnaldi M.",
    "jugador2": "Tsitsipas S.",
    "cuota1": 2.55,
    "cuota2": 1.48,
    "match_url": "https://www.flashscore.com/match/tennis/arnaldi-matteo-XXXXX/tsitsipas-stefanos-YYYYY/?mid=ZZZZZ",
    "h2h_url": "https://www.flashscore.com/match/tennis/arnaldi-matteo-XXXXX/tsitsipas-stefanos-YYYYY/#/h2h/overall/",
    "match_id": "ZZZZZ",
    "torneo": "Roland Garros (France)",
    "superficie": "clay"
}
```

## Tests Requeridos

```python
# tests/test_url_scraper_output.py
def test_h2h_url_derivada_de_match_url():
    match_url = "https://www.flashscore.com/match/tennis/a-b-ID1/c-d-ID2/?mid=rDQ3y6to"
    h2h_url = match_url.split('?')[0].rstrip('/') + '/#/h2h/overall/'
    assert h2h_url == "https://www.flashscore.com/match/tennis/a-b-ID1/c-d-ID2/#/h2h/overall/"

def test_match_id_extraido_correctamente():
    import re
    match_url = "https://www.flashscore.com/match/tennis/a-b-ID1/c-d-ID2/?mid=rDQ3y6to"
    mid = re.search(r'\?mid=([^&]+)', match_url).group(1)
    assert mid == "rDQ3y6to"

def test_superficie_roland_garros():
    assert extraer_superficie("Roland Garros (France)") == "clay"
    assert extraer_superficie("Wimbledon (UK)") == "grass"
    assert extraer_superficie("Australian Open (Australia)") == "hard"
```

---

## Ciclo de Vida

```
Estado:   ROTO — confirmado en datos reales 2026-05-28 (423 partidos, 0 con h2h_url)
Fix:      ~4 horas (3 bugs independientes en el mismo archivo)
Tests:    ~1 hora (3 tests unitarios ya especificados arriba)
Deploy:   python3 extraer_URL_partidos_version2.py → verificar primer partido tiene h2h_url
Validar:  grep "h2h_url" data/zita_tennis_matches_HOY.json | head -5
          → debe mostrar URLs reales, no null

Impacto post-fix:
  surface_specialization: 0% → >0% (en siguientes partidos)
  match_id: "tennis" → "rDQ3y6to" (habilita Nodo-05)
  torneo: "Sin Torneo" → "Roland Garros (France)" (habilita home_advantage)
```
