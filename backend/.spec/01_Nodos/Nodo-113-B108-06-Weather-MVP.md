# Nodo-113 — B108-06: Weather MVP observacional

> Creado: 2026-07-17
> Estado: IMPLEMENTADO
> Wikilinks: [[Nodo-108]] [[Nodo-64]] [[Nodo-96]]

---

## Problema

Ningún componente del pipeline consideraba condiciones meteorológicas. El bookmaker
tampoco las modela bien para partidos outdoor en tiempo de lluvia/viento fuerte.
Potencial señal de valor no explotada.

## Solución MVP

### D113-01 — `core/weather_client.py`

Módulo puro sin dependencias externas. Usa open-meteo (gratuito, sin API key):
- `get_weather_flag(pais, superficie, fecha_iso, ciudad)` → `str`
- Retorna: `'RAIN_RISK'` | `'WIND_HIGH'` | `'CLEAR'` | `'UNKNOWN'`
- Solo aplica a superficies outdoor (clay/grass/hard y variantes en español)
- Coordenadas hardcodeadas por país (~45 entradas) — MVP sin geocodificación
- Cache `lru_cache` por `(lat, lon, fecha_iso)` para evitar duplicados por sesión
- Thresholds: RAIN ≥ 3.0 mm/día | WIND ≥ 30.0 km/h
- Si open-meteo falla → `'UNKNOWN'` sin propagar excepción (no bloquea pipeline)

### D113-02 — Campo `weather_flag` en `edge_calculator.py`

Añadido después de `irp_fav`/`irp_rival` (línea ~1188):

```python
resultado['weather_flag'] = get_weather_flag(
    pais=partido.get('pais', ''),
    superficie=partido.get('superficie', partido.get('tipo_cancha', '')),
)
```

**OBSERVACIONAL PURO**: No modifica `p_modelo`, `kelly_kl`, `apostar` ni ningún gate.
La hipótesis H113-01 acumula para determinar si hay señal antes de cualquier ajuste.

### D113-03 — Hipótesis H113-01 pre-registrada

Ver `validation/preregistered_hypotheses.json`. n_stop=40, ACUMULANDO.

## Datos de entrada

Campo `pais` del zita file (`data/zita_tennis_matches_*.json`). Este campo está
disponible en todos los partidos extraídos por Playwright (extraer_URL_partidos_version2.py).

## Pendientes

- D113-04: cuando H113-01 GRADUADA, evaluar ajuste de p_modelo ±2% (clay RAIN_RISK).
- D113-05: geocodificación por ciudad (reemplazar tabla estática) — baja prioridad MVP.
- D113-06: cobertura WTA indoor (si `tipo_cancha='indoor'` → UNKNOWN sin llamada API).
