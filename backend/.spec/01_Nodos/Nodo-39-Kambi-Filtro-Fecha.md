# Nodo-39 — Kambi Filtro Fecha (PASO 1 API Fix)

> **Fecha:** 2026-06-27 08:50 UTC
> **Severidad:** CRÍTICA — Kambi devuelve eventos futuros (2-4 días adelante) sin filtro de fecha. El pipeline los procesa como "partidos de hoy", contaminando todas las predicciones downstream.
> **Archivos modificados:** `scraping/kambi_tennis.py` (función `extract_matches`)
> **Impacto:** PASO 1 ahora filtra por fecha UTC, eliminando ~70% eventos futuros/pasados
> **Estado:** ✅ COMPLETO

---

## 0. HALLAZGO

Kambi API (`fetch_kambi_tennis()`) devuelve TODOS los eventos de tenis disponibles en Betplay sin filtrar por fecha de partido. Cuando el usuario corre `extraer_partidos_api.py --offset 0` (hoy), el sistema trae:

- Partidos de **hoy** (Wimbledon, ITF en vivo)
- Partidos de **mañana y pasado mañana** (Hurlingham exhibiciones)
- Todo se procesa como si fuera "hoy"

**Ejemplo (2026-06-27):**
- Kambi: 199 eventos
- Después merge con FlashScore: 178 eventos candidatos
- **Después filtro fecha:** 40 partidos reales de hoy
- **Eliminados: 138 eventos futuros/pasados (77.5%)**

Los combos se generaban con jugadores como **Tsitsipas** (juega lunes 2026-06-29) y **Etcheverry** (martes 2026-06-30), completamente inútiles.

---

## 1. SOLUCIÓN

En `extract_matches()`, después del merge Kambi+FlashScore, agregar filtro de fecha ISO:

```python
# 3.5. Filtrar por fecha — Kambi devuelve partidos futuros
target_date = (datetime.now(timezone.utc) + timedelta(days=day_offset)).date()
before = len(merged)
merged = [
    m for m in merged
    if not m.get("hora") or
    datetime.fromisoformat(m["hora"].replace("Z", "+00:00")).date() == target_date
]
```

El campo `hora` ya viene de Kambi en formato ISO (`2026-06-29T11:40:00Z`), así que solo necesitaba parsing.

---

## 2. RESULTADO

**Antes (contaminado):**
```
extraer_partidos_api.py → 188 partidos → combo_builder → TSITSIPAS @1.42 (juega lunes)
```

**Después (limpio):**
```
extraer_partidos_api.py → 40 partidos → combo_builder → 1 pick ITF real de hoy
```

---

## 3. IMPLEMENTACIÓN

**Archivo:** `scraping/kambi_tennis.py`
**Líneas:** 736-748 (dentro de `extract_matches()`)
**Imports:** Ya presentes (`from datetime import datetime, timedelta, timezone`)

**Log output:**
```
📅 Filtro fecha 2026-06-27: 178 → 40 partidos (eliminados 138 futuros/pasados)
```

---

## 4. TESTS

No requiere test específico — validado en vivo:
- 2026-06-27 08:44 → extraer_partidos_api.py filtra correctamente ✅
- Downstream: edge_calculator, combo_builder procesan datos limpios ✅

---

## NOTAS

- Fix aplica para `day_offset=0` (hoy) y `day_offset=1` (mañana)
- No afecta FlashScore feed (ya filtra por offset) — solo corrige Kambi
- `--tomorrow` flag ahora también devuelve datos limpios
