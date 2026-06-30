# Sources: Fuentes de Datos del Sistema

> **Wikilinks:** [[Grafo-Dependencias-Datos]] | [[Pipeline-Arquitectura]] | [[Mandatos-No-Negociables]] | [[Sprint-Pipeline]] | [[Nodo-01-Edge-Calculator]] | [[Nodo-02-Markov-Changepoint]] | [[Nodo-03-Scraper-Fix]] | [[Nodo-04-Dataset-Fix]] | [[Nodo-05-Validacion-API]] | [[Nodo-06-Erdos-Graph]] | [[Nodo-09-API-Status-Keys]]

---

## Fuente 1: FlashScore.com (Playwright)

**Tipo:** Web scraping con browser headless
**Latencia:** 8 min / 423 partidos (lista) | 30-60 min / 28 H2H profundos
**Fiabilidad:** Alta — datos oficiales FlashScore

### Datos que produce

| Campo | Script | Estado |
|---|---|---|
| Lista de partidos del día | `extraer_URL_partidos_version2.py` | ⚠️ torneo=Sin Torneo, h2h_url=None |
| H2H profundo entre jugadores | `extraer_historh2h.py` | ✅ funcionando |
| Rankings ATP completos | `extraer_ranking_atp_version2.py` | ✅ funcionando |
| Resultados post-partido | `resultados_finales.py` | ✅ funcionando |

### Selectores DOM Conocidos

```python
# Partidos del día (FlashScore /tennis/)
'.event__match'          → elemento de partido
'.event__header'         → header de torneo/competición
'.event__participant'    → nombre del jugador
'.event__odd'            → cuota del partido
'.tournament__name'      → nombre del torneo (alternativa)

# H2H (/match/ID/#/h2h/overall/)
'.h2h__section'          → sección de historial
'.h2h__row'              → partido individual del historial
```

### Bugs Activos (ver [[Nodo-03-Scraper-Fix]])

```
h2h_url:  nunca se construye desde match_url
match_id: extrae "tennis" (tipo deporte) en lugar del event_id del ?mid=
torneo:   DOM parser no lee el header de torneo → "Sin Torneo Asignado"
```

---

## Fuente 2: FlashScore Ninja API

**Tipo:** REST API propietaria (sin auth pública)
**Latencia:** <1 seg / petición
**Formato:** `KEY÷VALUE¬KEY÷VALUE` (propietario)

### Endpoints Confirmados

```
BASE: https://global.flashscore.ninja/202/x/feed/

dc_1_{event_id}                     → score, estado, timestamp
  HTTP: 200 ✅ para tenis
  Claves reales (verificadas 2026-05-29 — Nodo-09):
    DJ = ganador: 'H'=local ganó, 'A'=visitante ganó, ''=no terminado
    DE = sets ganados por local (jugador1)
    DF = sets ganados por visitante (jugador2)
    DC = Unix timestamp del inicio programado
    DV = constante tipo partido (2=tenis, NO indica estado)
  FT detection: DJ in ('H', 'A')
  NS detection: DJ=='' AND datetime.fromtimestamp(DC) > now()
  LIVE detection: DJ=='' AND datetime.fromtimestamp(DC) <= now()
  INCORRECTO (documentación anterior errónea): ~AA, ~BH, ~BI — no existen en este endpoint

dc_h2h_1_{event_id}                 → H2H entre jugadores
  HTTP: 404 ❌ para tenis (solo NBA funciona)

df_psn_1_{event_id}                 → box score jugador
  HTTP: 200 ✅ para NBA, no probado para tenis

t_3_200_{tournId}_-5_es-co_1        → partidos del torneo
  HTTP: 200 ✅ para NBA (tournId=IBmris38)
  Tennis tournId: pendiente descubrir
```

### Autenticación

```python
HEADERS = {
    "X-Fsign": "SW9D1eZo",        # fijo, no rota
    "Referer": "https://www.flashscore.co/",  # .co NO .com
    "Origin": "https://www.flashscore.co",
    "User-Agent": "Mozilla/5.0 (compatible)"
}
```

### Uso en el Pipeline

```python
# Validación post-partido (ver [[Nodo-05-Validacion-API]])
url = f"https://global.flashscore.ninja/202/x/feed/dc_1_{match_id}"
r = requests.get(url, headers=HEADERS)
# match_id debe ser el event_id real (ej: "rDQ3y6to"), NO "tennis"
# Prerequisito: [[Nodo-03-Scraper-Fix]] Bug 2 resuelto
```

---

## Fuente 3: ATP Rankings (FlashScore)

**Script:** `extraer_ranking_atp_version2.py` / `CompleteRankingScraper`
**Output:** `data/atp_rankings_complete_FECHA.json`
**Frecuencia:** Semanal (rankings ATP actualizan los lunes)

### Estructura del Output

```json
{
  "jugadores": [
    {
      "nombre": "Sinner J.",
      "ranking": 1,
      "puntos": 11380,
      "pais": "ITA",
      "edad": 22
    }
  ],
  "fecha_scrape": "2026-01-13T19:53:12",
  "total_jugadores": 200
}
```

---

## Fuente 4: Datos Históricos Locales

**Tipo:** JSON files generados por el pipeline
**Ubicación:** `reports/h2h_results_enhanced_FECHA.json`

### Campos Disponibles (confirmados en datos reales)

```json
{
  "jugador1": "Arnaldi M.",
  "jugador2": "Tsitsipas S.",
  "cuota1": 2.55,
  "cuota2": 1.48,
  "match_url": "https://www.flashscore.com/match/tennis/.../",
  "h2h_url": null,              ← BUG Nodo-03
  "match_id": "tennis",         ← BUG Nodo-03
  "torneo": "Sin Torneo Asignado", ← BUG Nodo-03
  "tipo_cancha": "<HTML garbage>", ← consecuencia del bug de torneo
  "ranking_analysis": {
    "prediction": {
      "favored_player": "Tsitsipas S.",
      "confidence": 59.2,
      "method": "multi_factor"
    },
    "surface_specialization": 0.0  ← consecuencia del bug HTML garbage
  },
  "prediccion_ganador": null    ← SIEMPRE null — no usar
}
```

---

## Estrategia Híbrida: API + Playwright

```
Tarea                     API (<1s)     Playwright (min)   Decisión
──────────────────────────────────────────────────────────────────
Lista partidos del día    ❌ (no impl)  ✅ 8min            Playwright
H2H histórico             ❌ 404 tenis  ✅ 30-60min        Playwright
Resultado partido          ✅ dc_1_     —                   API
Score en tiempo real       ✅ dc_1_     —                   API
Rankings ATP               —            ✅ semanal          Playwright
```

**Principio:** Usar API para todo lo que esté disponible. Playwright solo cuando la API
retorna 404 o los datos no están disponibles. Confirmar cada endpoint antes de asumir.
