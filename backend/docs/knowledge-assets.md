# knowledge-assets.md — URLs, Selectores y Formatos Críticos

> **Nodo:** [[Nodo-59-Motor-Agentico-Odometro-Dream]]
> **REGLA-DELETE-KNOWLEDGE:** antes de eliminar cualquier scraper o parser, extraer y documentar aquí las URLs, selectores CSS, formatos de respuesta y tokens de autenticación que contiene.
> **Por qué:** el conocimiento técnico de scraping se pierde cuando se borra el código. Este archivo es el repositorio permanente de ese conocimiento — el código puede regenerarse, el selector correcto no.

---

## 1. Kambi API — Betplay (Fuente de Cuotas Reales)

**Base URL:** `https://eu-offering-api.kambicdn.com/offering/v2018/betplay/`

**Endpoints principales:**
```
/listView/tennis.json?lang=es_CO&market=CO&client_id=2&channel_id=1&ncid=1&category=&...
```

**Headers requeridos:**
- `User-Agent`: browser real (no-bot)
- No requiere token de autenticación

**Campo `||replace`:** El flag correcto para localStorage en Betplay es `||replace`, NO `||append`. Usar `||append` acumula picks entre tabs → REGLA-KAMBI-1.

**Archivo:** `scraping/kambi_tennis.py`
**Fuente:** `cuota_es_real=True` → estas cuotas son las reales de Betplay donde se apuesta.

**NUNCA derivar URLs de browser desde URLs de API** (REGLA-URL-1): son sistemas completamente distintos. `global.flashscore.ninja/202/x/feed` ≠ `www.flashscore.com/tennis/`.

---

## 2. FlashScore Ninja H2H API

**Base URL:** `https://global.flashscore.ninja/202/x/feed/`

**Tipo de feed:**
- `tipo=13` → 23→146 singles mañana (activar con type=13 en filtros)
- H2H Ninja: `https://global.flashscore.ninja/202/x/feed/0/1/0/{match_id}/0/0`

**Formato de respuesta:** JSON con estructura nested. Ver `scraping/ninja_h2h_parser.py` para parser completo.

**Rate limiting:** ~0.5s/partido recomendado para no ser bloqueado.

**Archivo:** `scraping/ninja_h2h_parser.py` (62 tests en test_nodo31.py lo blindan)

---

## 3. FlashScore Web (DOM — fallback)

**URL:** `https://www.flashscore.com/tennis/`

**Selectores CSS críticos (verificados 2026-06-07):**
```css
.event__match--live    /* partidos en vivo */
.event__match          /* todos los partidos */
.event__participant    /* nombres de jugadores */
.event__score          /* marcadores */
```

**Playwright required:** DOM con JavaScript. Usar `scraping/browser_manager.py`.
**Tiempo:** ~8 min para 80 partidos (vs ~45s con API).

---

## 4. FlashScore H2H Directo (Playwright fallback)

**URL pattern:** `https://www.flashscore.com/match/tennis/{player1}-{player2}/{match_id}/#/h2h`

**Selector H2H table:**
```css
.h2h__section .h2h__row
```

**Archivo:** `scraping/h2h_extractor.py` (fallback de `extraer_historh2h.py`)

---

## 5. Formatos de Fecha

| Sistema | Formato | Ejemplo |
|---------|---------|---------|
| FlashScore matches | `DD.MM.YYYY` | `30.06.2026` |
| H2H results JSON | ISO 8601 | `2026-06-30T00:00:00` |
| Shadow book JSONL | ISO 8601 UTC | `2026-07-02T04:00:00+00:00` |
| Calibracion edge | `YYYY-MM-DD` | `2026-06-30` |

**CRÍTICO para parser:** El año en FlashScore está en `[-4:]` del string `DD.MM.YYYY`, NO en `[:4]` (bug histórico Nodo-30 T30-13).

---

## 6. calibracion_edge.json — Estructura

```json
{
  "por_superficie_y_tier": {
    "clay_grand_slam": {"p": 0.758, "n": 31, "wins": 25, "losses": 6}
  },
  "fallback_por_tier": {
    "grand_slam": {"p": 0.65, "n": 45}
  },
  "por_superficie": {
    "clay": {"p": 0.60, "n": 200}
  },
  "global": {"p": 0.55, "n": 706}
}
```

**Jerarquía de lookup:** `por_superficie_y_tier` (n≥10) → `fallback_por_tier` (clamped si diverge >0.03) → `por_superficie` → `global`.

**NUNCA modificar manualmente** — solo via `validar_con_api.py` o `betslip_registrar.py --cerrar`.

---

## 7. Claude Code JSONL — Estructura para Odómetro

```json
{
  "type": "assistant",
  "message": {
    "model": "claude-sonnet-4-6",
    "usage": {
      "input_tokens": 587,
      "output_tokens": 42,
      "cache_read_input_tokens": 28084,
      "cache_creation_input_tokens": 671
    }
  },
  "timestamp": "2026-06-03T18:34:56.000Z",
  "sessionId": "b945e471-...",
  "type": "user" // para mensajes de usuario
}
```

**Directorio JSONL:** `~/.claude/projects/-mnt-c-users-hogar-tennis-analysis-backend/*.jsonl`
**Parser:** `token_odometer.py` (D59-01)

---

## 8. Betslip Registrar — Formato de Apuesta

**Puerto:** 5001 (localhost)
**Endpoint POST:** `http://localhost:5001/registrar`

**Bookmarklet:** Ver `docs/bp/index.html` en GitHub Pages — botón con `||replace` + `target="_blank"`.

**Estados de betslip:** `PENDIENTE` → `CERRADO` (tras `--cerrar`) | `VOID` (resultado incierto)

---

## Regla de Obsolescencia

Cuando un módulo de scraping es reemplazado o eliminado:
1. Extraer URLs, selectores y formatos al bloque correspondiente de este archivo
2. Marcar el nodo que lo especificó con: `> SUPERSEDED por [[Nodo-XX]]`
3. Nunca borrar el bloque de conocimiento — solo etiquetar como LEGACY si cambia

Ejemplo:
```markdown
## 3b. FlashScore Web v1 (LEGACY — superseded por Ninja API §2)
[datos históricos del selector que ya no funciona...]
```
