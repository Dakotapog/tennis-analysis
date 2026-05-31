# Nodo-10: Surface Propagation Bug

> **Wikilinks:** [[Inventario-Deuda-Tecnica]] | [[Pipeline-Arquitectura]] | [[Grafo-Dependencias-Datos]] | [[Nodo-03-Scraper-Fix]] | [[Nodo-07-Strangler-Fig]] | [[Sprint-Pipeline]]
> **Estado:** 2026-05-30 — RESUELTO ✅ | Verificado en run real (16 partidos Roland Garros)
> **Impacto P&L:** 15% del motor de predicción recuperado — surface_specialization activo en producción

---

## Síntoma Original (2026-05-29)

```
tipo_cancha: Desconocida  ← en h2h_results_enhanced_20260529_221441.json
surface_specialization: 0  en TODOS los partidos
Sup: unknown  ← en edge_calculator output
```

Ocurría aunque S1 extraía correctamente `superficie=clay` para French Open.

---

## Root Cause (diagnosticado 2026-05-29)

**S1 producía superficie correcta. S2 no la consumía.**

```
extraer_URL_partidos_version2.py
  → "superficie": "clay"  ✅

extraer_historh2h.py (versión monolito 3717 líneas)
  → re-extraía tipo_cancha desde página H2H de FlashScore
  → página H2H no expone superficie → tipo_cancha = "Desconocida"
  → rivalry_analyzer recibía superficie="Desconocida" → surface_specialization = 0
```

---

## Resolución Real — Efecto colateral de Nodo-07 Fase 1

**El fix NO requirió código adicional.** Nodo-07 Fase 1 (Strangler Fig, 2026-05-29) reescribió `process_single_match` en `extraer_historh2h.py`. El nuevo código ya incluye propagación correcta en dos puntos:

```python
# Línea 1205 — contexto para rivalry_analyzer
current_match_context = {
    'country': match_data.get('pais', 'N/A'),
    'surface': match_data.get('superficie') or match_data.get('tipo_cancha') or 'N/A'
}
# ↑ lee 'superficie' de S1 primero → si S1 tiene clay → rivalry_analyzer recibe clay

# Línea 1243 — campo en el resultado guardado
'tipo_cancha': match_data.get('superficie') or current_match_info.get('tipo_cancha') or match_data.get('tipo_cancha', 'N/A'),
# ↑ mismo patrón: S1 tiene prioridad → 'clay' propagado al JSON de salida
```

Con S1 produciendo `superficie=clay` (Nodo-03 ✅) y S2 leyendo `match_data.get('superficie')` (Nodo-07 ✅), la cadena se cierra sola.

---

## Evidencia del Run de Producción (2026-05-30)

```
Archivo: h2h_results_enhanced_20260530_053518.json
Partidos: 16 (Roland Garros, cuadro principal)

surface_specialization > 0:  16/16 ✅
surface_specialization = 0:   0/16

Muestra de surf_w (weighted score):
  Cerundolo F. vs Svajda Z.      tipo_cancha: clay  surf_w: 0.62
  Cobolli F. vs Tien L.          tipo_cancha: clay  surf_w: 0.63
  Auger-Aliassime F. vs Nakashima tipo_cancha: clay  surf_w: 0.64
  Sabalenka A. vs Kasatkina D.   tipo_cancha: clay  surf_w: 0.64
  Gauff C. vs Potapova A.        tipo_cancha: clay  surf_w: 0.69

Rango: 0.49 – 0.69  (era 0.00 en todos los runs previos a Nodo-07)
```

---

## Tasks — Estado Final

| Task | Descripción | Estado |
|---|---|---|
| T10-01 | Propagar `superficie` de S1 a S2 | ✅ Resuelto por Nodo-07 Fase 1 (código ya correcto) |
| T10-02 | Normalizar superficie para rivalry_analyzer | ✅ "clay"→"clay" funcionando |
| T10-03 | Tests de verificación | ✅ Verificado empíricamente en prod — tests formales pendientes en Nodo-07 Fase 2 |

---

## Impacto Conseguido

```
Antes (pre-Nodo-07):
  surface_specialization = 0  en TODOS los partidos
  15% del motor = 0 contribución

Después (post-Nodo-07, verificado 2026-05-30):
  surface_specialization activo en todos los partidos con superficie conocida
  Roland Garros: surf_w 0.49–0.69 según historial en arcilla de cada jugador
  15% del motor ahora contribuye a predicciones reales
```

---

## Vinculación

- [[Nodo-07-Strangler-Fig]] — la migración Strangler Fig que resolvió el bug como efecto colateral
- [[Nodo-03-Scraper-Fix]] — produce `superficie=clay` en S1 (precondición)
- [[Grafo-Dependencias-Datos]] — S1→S2 costura donde se propagaba el bug
- [[Pipeline-Arquitectura]] — surface_specialization es 15% del motor rivalry_analyzer
- [[Sprint-Pipeline]] — T-SF-05 confirmado en mismo run
