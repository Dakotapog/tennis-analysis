# Nodo-50: Filtro de Torneo en PASO 1 (`--torneo`)

> **Wikilinks:** [[Nodo-22-API-Integration-Kambi-Ninja]] | [[Nodo-48-FlashScore-Odds-Scraper-Testing]]
> **Fecha de creación:** 2026-07-01
> **Estado:** ✅ IMPLEMENTADO 2026-07-01

**Prioridad:** MEDIA — calidad de vida operacional, no P&L directo
**Archivos modificados:**
- `scraping/kambi_tennis.py` — `extract_matches()` + `extract_matches_flashscore_only()`
- `extraer_partidos_api.py` — nuevo flag `--torneo`
- `tests/test_nodo50.py` — 7 tests T50-01 → T50-07

---

## 0. El Problema

Sin filtro de torneo en PASO 1, `extraer_partidos_api.py --tier atp` trae TODOS los
torneos ATP del día (Wimbledon + Queen's Club + Challenger + ITF dentro de ATP).
PASO 2 (`extraer_historh2h.py`) procesa todos esos partidos innecesariamente.

Para focalizarse en Wimbledon, el único filtro disponible era `--tier atp wta`
(que igual trae todos los ATP/WTA) + confiar en que `--torneo-tipo grand_slam` en el
trader filtra en PASO 4. Pero PASO 2 tardaba el tiempo completo sin beneficio.

---

## 1. La Solución — Mismo Patrón que `--tier`

### Campos disponibles en cada match dict
| Campo | Ejemplo Wimbledon |
|---|---|
| `torneo_nombre` | `"Wimbledon"` |
| `torneo_completo` | `"ATP - INDIVIDUALES: Wimbledon (Reino Unido) - Fase previa, hierba"` |
| `tier` | `"atp"` |

### Lógica del filtro
```
--torneo wimbledon  →  kw="wimbledon" busca en torneo_nombre OR torneo_completo (OR)
--torneo wimbledon "us open"  →  múltiples keywords (OR entre torneos)
--torneo + --tier   →  AND (ambos deben cumplirse)
Sin --torneo        →  sin cambio — comportamiento idéntico al actual
```

Matching: **substring case-insensitive** sobre `torneo_nombre` y `torneo_completo`.
Misma filosofía que `detectar_tier()` en `config.py`.

---

## 2. Implementación

### `scraping/kambi_tennis.py`

**`extract_matches()` (línea ~1112):** añadir `torneos` parameter:

```python
def extract_matches(
    day_offset: int = 0,
    tiers: Optional[List[str]] = None,
    torneos: Optional[List[str]] = None,   # Nodo-50
) -> Tuple[str, List[Dict]]:
```

Después del filtro de tiers (línea ~1155):

```python
    if torneos:
        keywords = [k.lower() for k in torneos]
        before = len(merged)
        merged = [
            m for m in merged
            if any(
                kw in (m.get('torneo_nombre') or '').lower()
                or kw in (m.get('torneo_completo') or '').lower()
                for kw in keywords
            )
        ]
        logger.info(f"   🏆 Filtro --torneo {torneos}: {before} → {len(merged)} partidos")
```

**`extract_matches_flashscore_only()` (línea ~886):** mismo cambio idéntico.

### `extraer_partidos_api.py`

Nuevo argumento:
```python
parser.add_argument(
    "--torneo", nargs="+", default=None,
    metavar="NOMBRE",
    help="Filtrar por nombre de torneo (substring, case-insensitive). "
         "Ej: --torneo wimbledon | --torneo 'us open' 'roland garros'"
)
```

Pasar a ambas rutas:
```python
# extract_matches(..., torneos=args.torneo)
# extract_matches_flashscore_only(..., torneos=args.torneo)
```

Log en header cuando se activa:
```python
if args.torneo:
    logger.info(f"🏆 Torneos: {args.torneo}")
```

---

## 3. Uso

```bash
# Solo Wimbledon (ATP + WTA)
python3 extraer_partidos_api.py --tier atp wta --torneo wimbledon

# Grand Slams combinados (Roland Garros + Wimbledon)
python3 extraer_partidos_api.py --torneo wimbledon "roland garros"

# Wimbledon en modo testing (sin Kambi)
python3 extraer_partidos_api.py --flashscore-only --tier atp wta --torneo wimbledon

# Sin --torneo → comportamiento idéntico al actual
python3 extraer_partidos_api.py --tier atp wta
```

---

## 4. Tests (T50-01 → T50-07)

| Test | Qué prueba |
|---|---|
| T50-01 | `torneo_nombre` match case-insensitive: "Wimbledon" → kw="wimbledon" ✅ |
| T50-02 | `torneo_completo` match substring: match si kw está en torneo_completo |
| T50-03 | Múltiples keywords = OR: "wimbledon" OR "us open" → incluye ambos |
| T50-04 | Sin match → lista vacía (no crashea, el guard `if not merged:` maneja) |
| T50-05 | `torneos=None` → sin filtro, devuelve todos los partidos |
| T50-06 | AND con tiers: `--tier atp --torneo wimbledon` → solo ATP Wimbledon |
| T50-07 | `torneo_nombre=None` (campo ausente) → no crashea, usa torneo_completo |

---

## 5. Vacíos Anticipados

| Vacío | Manejo |
|---|---|
| `torneo_nombre` puede ser `None` | `(m.get('torneo_nombre') or '').lower()` |
| Match semánticamente ambiguo ("open" matchea todo) | Responsabilidad del usuario — documentado |
| 0 partidos después del filtro | `if not merged: sys.exit(1)` ya existe en ambas funciones |
| `--flashscore-only` + `--torneo` | Mismo parámetro propagado a `extract_matches_flashscore_only()` |
| Acentos ("Spaña" vs "España") | Búsqueda sobre strings originales — usuario debe usar el string correcto |

---

## 6. Impacto en Pipeline

| Paso | Cambio |
|---|---|
| PASO 1 | Archivo `zita_tennis_matches_FECHA.json` solo con torneos filtrados |
| PASO 2 | `extraer_historh2h.py` procesa menos partidos → más rápido |
| PASO 3-4 | Sin cambio — el filtro ya ocurrió upstream |
| `--flashscore-only` | Compatible — mismo parámetro |
| Tests existentes | Sin cambio — `torneos=None` por default |

**Tests baseline:** 1438 passed (no deben bajar).
