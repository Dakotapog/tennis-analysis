# Nodo-106 — Retroactive Settle Workflow (extensión T9)

> **Fecha:** 2026-07-16 | **Autor:** Sonnet 4.6
> **Wikilinks directos:** [[Nodo-66-Plan-Trabajo-Semanal-Sonnet]] | [[Nodo-52-Shadow-Book-CLV-Tracking]] | [[Nodo-81-Settlement-Name-Normalize]] | [[Nodo-36-Unicode-Acento-Apellidos-Cortos]]
> **Wikilinks huérfanos** (este nodo referencia pero ellos no backlinkan aún): [[Nodo-87-Fixes-Auditoria-D87]] | [[Nodo-51-Plan-Estrategico-Data-Layer-Torneo]]
> **Recibe backlink de:** [[Nodo-66-Plan-Trabajo-Semanal-Sonnet]] (addendum T9-ext añadido 2026-07-16)
> **Estado:** DOCUMENTADO — 302 settled / 71 abiertos (57 permanentes) tras ejecución 2026-07-16

---

## 1. PROBLEMA RAÍZ

`shadow_book.py --settle FECHA` retorna `Settled: 0` cuando:

1. **Filtro de fecha en `_load_resultados()`** — solo acepta `resultados_finales_*.json` cuyo nombre contenga la fecha exacta o `fecha+1d`. Un archivo generado 2+ días después es rechazado silenciosamente.
2. **Ventana FlashScore limitada** — feed solo cubre `day_offset` ≥ −2. Para `day=−3` datos parciales; `day≤−4` sin feed.
3. **H2H file incorrecto** — el fallback fuzzy falla cuando `resultados_finales` se generó sobre un H2H diferente al que produjo los picks del shadow book.

**Diagnóstico rápido:**
```python
# Confirmar que hay picks sin resolucion
from shadow_book import _load_jsonl, _jsonl_path
recs = _load_jsonl(_jsonl_path('2026-07-14'))
abiertos = [v for k,v in recs.items() if 'pick_snapshot' in v and 'resolucion' not in v]
print(f'Abiertos: {len(abiertos)}')  # > 0 → activar workflow §2

# Obtener el H2H file exacto de la sesión
import json
with open('reports/shadow_book/sb_2026-07-14.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r.get('_type') == 'session_meta':
            print('h2h_file:', r.get('h2h_file'))
            break
```

---

## 2. WORKFLOW COMPLETO (3 pasos)

### Paso 1 — Identificar H2H file de cada sesión pendiente

```python
import json, glob
for f in sorted(glob.glob('reports/shadow_book/sb_*.jsonl')):
    fecha = f.split('sb_')[1].replace('.jsonl','')
    with open(f) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get('_type') == 'session_meta':
                print(f'{fecha} → {r.get("h2h_file")}')
                break
```

### Paso 2 — Resultados via Ninja API (--no-cal para no contaminar calibración)

```bash
python3 validar_con_api.py --h2h reports/h2h_results_enhanced_FECHA_HHMMSS.json --no-cal
# → reports/resultados_finales_HOY_HHMMSS.json  (formato: detailed_results)
```

### Paso 3 — Inyección directa en settle()

```python
import sys; sys.path.insert(0, '.')
from shadow_book import settle
import json

with open('reports/resultados_finales_20260716_HHMMSS.json') as f:
    data = json.load(f)

result_map = {}
for r in data.get('detailed_results', []):
    mi = r.get('match_info', {})
    ar = r.get('actual_result', {})
    j1, j2 = mi.get('jugador1',''), mi.get('jugador2','')
    ganador = ar.get('actual_winner','')
    url = mi.get('match_url','')
    mid = url.rstrip('/').split('/')[-1] if url else None
    if not ganador: continue
    k1, k2 = j1.split()[-1].lower(), j2.split()[-1].lower()
    result_map[f"{min(k1,k2)}_{max(k1,k2)}"] = {
        'ganador': ganador, 'cuota_cierre': None,
        'provenance': 'validar_api_retroactivo', 'void': False,
        'match_id': mid, 'p1': j1, 'p2': j2,
    }

n = settle('2026-07-14', resultados_map=result_map)
print(f'Settled: {n}')
```

### Paso 3b — Picks GS/ATP no resueltos por API

Usar **WebSearch directamente** (no delegar al usuario — lección 2026-07-16):
```
WebSearch: "Krueger Kostyuk Wimbledon 2026 resultado"
```
Construir `result_map` con `provenance='manual_lookup'` e inyectar igual.  
**Para void** (partido no disputado): `ganador=''` + `void=True`.

---

## 3. EJECUCIÓN 2026-07-16

| Fecha | H2H file | n API results | Settled API | Settled manual |
|-------|----------|--------------|-------------|----------------|
| 07-13 | h2h_..._20260713_083345.json | 66 | 4 | 0 |
| 07-14 | h2h_..._20260714_073915.json | 130 | 16 | 0 |
| 07-15 | h2h_..._20260715_225410.json | 287 | 22 | 0 |
| 07-05→07-12 GS/ATP/CHA | manual WebSearch | 15 resultados | 0 | **18** |
| **Total** | | | **42** | **18** = **+60** |

Shadow book: 242 → **302 settled** | 131 → **71 abiertos**

---

## 4. LOS 71 ABIERTOS — Clasificación final

| Tier | n | Causa | Acción |
|------|---|-------|--------|
| ITF M15/W15 minors | ~57 | Sin cobertura Ninja API — gap permanente | Aceptar (igual que T9 §9) |
| Partidos 07-16 activos | ~14 | En curso — feed activo | `/settle-retry` día siguiente |

Los picks GS/ATP/Challenger que tenían datos disponibles fueron **todos resueltos** en esta sesión.

---

## 5. POR QUÉ FALLA EL SETTLE AUTOMÁTICO EN GS/ATP

El matching de `settle()` tiene 3 capas (ver [[Nodo-81-Settlement-Name-Normalize]]):

1. **match_id exacto** — falla si `resultados_finales` fue generado desde H2H diferente al del pick
2. **match_key** — falla con apellidos compuestos (Van Der Meerschen, St. Hilaire)
3. **Fuzzy nombre** — falla con acentos e iniciales (ver [[Nodo-36-Unicode-Acento-Apellidos-Cortos]])

Para GS la raíz fue **capa 1**: el archivo `resultados_finales_20260713_153750.json` cubría Istanbul 2 (Turquía), mientras los picks del sb eran Wimbledon — H2H files distintos.

---

## 6. PATRÓN PREVENTIVO (operativo desde 2026-07-16)

```
Cada lunes: /settle-retry → ITF rezagados ≤48h
Si settle retorna 0 con picks abiertos:
  └─ Paso 1: extraer h2h_file de session_meta
  └─ Paso 2: validar_con_api.py --no-cal
  └─ Paso 3: inyección programática
Si quedan abiertos GS/ATP/Challenger:
  └─ WebSearch directamente (NO delegar al usuario)
  └─ provenance='manual_lookup'
Si quedan abiertos ITF minors:
  └─ Gap permanente — documentar en addendum de este nodo
```

---

## Addendum — Gaps permanentes ITF (2026-07-16)

Sin cobertura Ninja API ni FlashScore histórico:

| Torneo | Fecha | n picks |
|--------|-------|---------|
| M15 Kursumlijska Banja (Serbia) | jul-08→jul-11 | ~6 |
| M15/W15 Wuning/Luan (China) | jul-08, jul-14 | ~8 |
| W15 Monastir 22 (Tunisia) | jul-14 | ~4 |
| M15 Uslar (Germany) | jul-14 | ~2 |
| M15 Castelo Branco (Portugal) | jul-14 | ~2 |
| W15 Hillcrest 3 (South Africa) | jul-14 | ~1 |
| Varios M15/W15 | jul-02→07-12 | ~34 |
| **Total permanentes** | | **~57** |
