# Nodo-16 — Multi-Torneo Pipeline

> **Estado:** ✅ IMPLEMENTADO — 2026-06-02
> **Wikilinks:** [[MOC-Principal]] | [[Sprint-Pipeline]] | [[Nodo-03-Scraper-Fix]] | [[Nodo-07-Strangler-Fig]] | [[Nodo-15-Portfolio-HedgeFund]] | [[Inventario-Deuda-Tecnica]]
> **Tests:** 875 passed, 0 failed (baseline mantenido)

---

## Problema

El pipeline original estaba hardcoded para Roland Garros:

1. **Bug silencioso en `h2h_extractor.py`**: Si el archivo `zita_tennis_matches_*.json` contenía partidos del French Open, el filtro Roland Garros se activaba automáticamente y **ignoraba todos los Challengers/ATP** sin avisar.

2. **Sin límite de partidos en scraper**: `extraer_URL_partidos_version2.py` extraía todos los partidos disponibles (~150+) sin forma de controlar el volumen.

3. **Sin modo multi-torneo**: No había forma de correr el pipeline con Challengers (Birmingham/grass, Centurion 2/hard) junto al Grand Slam.

4. **Confusión PASO 1 vs PASO 2**: El usuario esperaba que el scraper tardara mucho. Aclaración: PASO 1 (~30s) solo lee una página; PASO 2 (~30-60 min) visita cada partido individualmente.

---

## Solución implementada

### `extraer_URL_partidos_version2.py`

**Nuevo CLI arg `--max-matches N`:**
```bash
# Sin límite (comportamiento original):
python3 extraer_URL_partidos_version2.py

# Limitar a 80 partidos individuales (dobles siempre excluidos):
python3 extraer_URL_partidos_version2.py --max-matches 80
```

El límite se aplica en `extract_matches_from_dom()` — detiene el loop cuando `len(matches) >= max_matches`. Los dobles seguían filtrados antes por el flag `is_doubles_tournament`.

**Nuevo método `navigate_to_tomorrow()`** (experimental):
```bash
python3 extraer_URL_partidos_version2.py --tomorrow --max-matches 80
```
Intenta navegar a los partidos del día siguiente vía selectores de FlashScore. Si falla, usa los partidos del día actual y muestra diagnóstico de elementos encontrados en el DOM.

> ⚠️ Estado del `--tomorrow`: pendiente de validación. Los selectores de FlashScore pueden variar. Ver screenshots/ para diagnóstico.

---

### `scraping/h2h_extractor.py`

**Nuevo atributo `all_tournaments` (bool):**

```python
# Modo default (sin cambio):
if roland_garros:
    target = singles_RG
else:
    target = valid_matches   # comportamiento original

# Modo --all-tournaments:
if self.all_tournaments:
    target = [m for m in valid_matches if es_singles_cuadro_principal(m)]
    # → filtra por cuotas + URL válida, ignora torneo
```

`es_singles_cuadro_principal(match)`:
- URL contiene `/match/tennis/` con >20 chars → es singles
- `cuota1 is not None` → tiene cuotas (excluye juniors/calificación sin odds)

---

### `extraer_historh2h.py`

**Nuevos CLI args:**
```bash
# Multi-torneo (desactiva filtro Roland Garros):
python3 extraer_historh2h.py --all-tournaments

# Especificar archivo manualmente:
python3 extraer_historh2h.py --file data/zita_tennis_matches_FECHA.json

# Combinado:
python3 extraer_historh2h.py --all-tournaments --file data/zita_tennis_matches_FECHA.json
```

---

## Diagnóstico de tiempos — PASO 1 vs PASO 2

| Paso | Script | Tiempo | Por qué |
|---|---|---|---|
| PASO 1 | `extraer_URL_partidos_version2.py` | ~30 segundos | Lee **una sola página** de FlashScore (el listado) |
| PASO 2 | `extraer_historh2h.py` | ~30-60 min | Visita la página H2H de **cada partido** individualmente (~1 min/partido × 50-80 partidos) |

**Conclusión:** 30 segundos en PASO 1 es correcto. No indica error.

---

## Flujo completo multi-torneo

```bash
# PASO 1 — ~30 segundos
python3 extraer_URL_partidos_version2.py --max-matches 80
# → data/zita_tennis_matches_FECHA.json (80 individuales, 0 dobles)

# PASO 2 — ~50-80 minutos
python3 extraer_historh2h.py --all-tournaments
# → reports/h2h_results_enhanced_FECHA.json

# PASO 3
python3 edge_calculator.py
# → reports/edge_report_FECHA.json

# PASO 4 — ρ=0.15 para sesión mixta multi-torneo
python3 trader_ev_tenis.py --bankroll 125000 --torneo-tipo atp500 --superficie clay
# → reports/trader_plan_FECHA.json + .txt
```

---

## Validación en prod (2026-06-02)

```
PASO 1 ejecutado: 80 partidos individuales, 0 dobles, 12 torneos
  French Open (France)       clay   19 partidos
  Centurion 2 (South Africa) hard    4 partidos (Challenger)
  Heilbronn (Germany)        clay    5 partidos (Challenger)
  Perugia (Italy)            clay    5 partidos (Challenger)
  Prostejov (Czech Republic) clay    4 partidos (Challenger)
  Tyler (USA)                hard    5 partidos (Challenger)
  Birmingham (UK) - Qual.    grass   6 partidos
  ... + Qualifications de cada Challenger

Archivo: data/zita_tennis_matches_20260601_203502.json
875 tests passed, 0 failed — baseline mantenido
```

---

## Reglas nuevas

**REGLA-S1-1: Modo de pipeline según torneo**
```
Grand Slam solo:   PASO 1 sin flags | PASO 2 sin flags
Multi-torneo:      PASO 1 --max-matches 80 | PASO 2 --all-tournaments
```

**REGLA-S1-2: --torneo-tipo para multi-torneo**
```
Sesión mixta (Challenger + ATP + GS mismo día):
  --torneo-tipo atp500  →  ρ=0.15  (picks más independientes entre torneos)
No usar grand_slam (ρ=0.25) cuando hay Challengers — sobrepenaliza portfolio Kelly.
```

**REGLA-S1-3: Dobles nunca pasan**
```
El filtro is_doubles_tournament en el scraper es automático.
No requiere flag. 'dobles'/'doubles' en nombre del torneo → partido descartado.
```
