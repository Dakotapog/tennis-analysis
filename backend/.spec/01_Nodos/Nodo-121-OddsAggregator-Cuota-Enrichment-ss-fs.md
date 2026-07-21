# Nodo-121 — OddsAggregator Cuota Enrichment para ss_fs

> Estado: PENDIENTE IMPLEMENTACIÓN
> Fecha spec: 2026-07-20
> Extiende: [[Nodo-120-FS-Single-Source-Cuotas-Qualifying-Flow]] (complementario)
> Tests: 3 REGLA-T53 planificados — `tests/test_nodo121_aggregator_enrichment.py`

---

## 0. Diagnóstico en lenguaje simple

Nodo-120 abrió la compuerta para qualifying picks **cuando FlashScore tiene cuotas**.
Pero descubrimos algo más profundo: los mismos partidos qualifying que FlashScore
no tiene cuotas están **SÍ disponibles en Betplay y Rushbet** — el scraper de PASO 1
simplemente no los alcanza.

**Evidencia directa (2026-07-20 00:00 CO):**

```
Ledger ss_fs → Aksu vs Costoulas → cuota1=None  ← PASO 1 no lo capturó
odds_aggregator.fetch_all_odds(['betplay','rushbet']):
    Aksu:      betplay=3.0  |  rushbet=2.95
    Costoulas: betplay=1.38 |  rushbet=1.37
```

El `odds_aggregator` llama a la misma API Kambi con parámetros más amplios:
PASO 1 → ~122 partidos | `odds_aggregator` → ~237 partidos (474 outcomes / 2).
Los qualifying rounds están en Betplay — solo que PASO 1 no los ve.

**Solución**: después del `--build`, enriquecer los ss_fs con cuota1=None
buscando sus nombres en el feed del odds_aggregator. Las cuotas son reales,
apuestas operacionales en Betplay o Rushbet.

---

## 1. Root cause técnico

```
PASO 1 (extraer_partidos_api.py) → 122 partidos Kambi (parámetros limitados)
  → fusionar_dia() → 11 ss_fs con cuota1=None
  → exportar_para_edge_calculator() → 11 excluidos por D120-03 (cuota=None)
  → edge_calculator: 0 picks qualifying

PERO:
odds_aggregator.fetch_all_odds(['betplay','rushbet']) → 474 outcomes (~237 partidos)
  → Aksu/Costoulas (y otros 10 ss_fs) SÍ aparecen CON cuotas reales
  → nunca se consultó el aggregator para enriquecer el ledger
RESULTADO: dinero perdido por gap de cobertura API
```

---

## 2. Decisiones

### D121-01 — `enriquecer_ss_fs_con_aggregator()` en match_ledger.py
**Qué**: Nueva función. Carga el ledger del día, toma los ss_fs con cuota1=None,
llama `fetch_all_odds(['betplay','rushbet'])`, hace match por nombre normalizado,
y popula cuota1/cuota2 en el ledger. Guarda ledger actualizado.

**Por qué**: La misma API Kambi que usa odds_aggregator tiene los qualifying rounds.
Solo faltaba conectar los dos módulos.

**Límite**: Solo enriquece ss_fs (Playwright sin contraparte Kambi). No toca joins ni ss_kambi.

### D121-02 — Flag `--enrich` en CLI de match_ledger
**Qué**: `python3 scraping/match_ledger.py --build --enrich --fecha 2026-07-20`
O separado: `python3 scraping/match_ledger.py --enrich --fecha 2026-07-20`
Llama `enriquecer_ss_fs_con_aggregator()` tras `fusionar_dia()`.

**Integración run_daily**: PASO 1.5 en `run_daily.py` pasa a ser:
```bash
python3 scraping/match_ledger.py --build --enrich --fecha FECHA
```

### D121-03 — Name matching strategy (apellido-first)
Match entre ledger ss_fs y odds_aggregator keys usando `_normalizar_nombre()`
existente en match_ledger + lógica de apellido de [[Nodo-80-Kambi-Name-Matching]].

**Algoritmo**:
1. `_normalizar_nombre("Aksu A.")` → `"aksu a"`
2. Buscar en feeds: clave exacta O clave que empiece con primer token (`"aksu"`)
3. Si dos candidatos (homónimo) → no enriquecer (cuarentena silenciosa + log WARN)
4. Si exacto o único → poblar cuota1/cuota2 + guardar event_id del aggregator

### D121-04 — `_cuota_source` extendido
- `'flashscore'` → ss_fs con cuota de Playwright (Nodo-120)
- `'betplay'` → ss_fs enriquecido desde aggregator, mejor precio en betplay
- `'rushbet'` → ss_fs enriquecido, mejor precio en rushbet
- `_best_book` → campo adicional: qué casa tiene la mejor cuota para ese pick

**Best price**: usar `best_price()` de [[Nodo-111-Dual-Book-Live-Intelligence]]
(función pura existente) para seleccionar la casa con mayor cuota.

### D121-05 — Log en embudo
```
Enriched desde aggregator: 8/11 ss_fs (3 sin match / 0 homónimos)
Exportados (Nodo-120+121): 85 joins + 37 kambi + 8 ss_fs_con_cuotas
```

### D121-06 — NO tocar fusionar_dia()
El enriquecimiento ocurre POST-fusión. `fusionar_dia()` determina identidades;
`enriquecer_ss_fs_con_aggregator()` rellena cuotas faltantes. Separación de responsabilidades.

---

## 3. Hipótesis pre-registrada

### H121-01 — ss_fs enriched via aggregator superan breakeven
- **Predicado**: picks enriquecidos por odds_aggregator (cuota de betplay/rushbet real)
  tienen hit% > breakeven de su cuota media (estimado ~40% para cuotas ~2.5–3.5)
- **Gate**: n_stop=20 settled picks con `_cuota_source` en `['betplay','rushbet']` + `_ledger_status='SINGLE_SOURCE_FS'`
- **Kill-switch**: hit% < 30% con n≥15 → revisar cobertura qualifying en API Kambi
- **Estado**: PENDIENTE (implementación no iniciada)
- **Diferencia de H120-01**: H120-01 trackea cuotas FlashScore; H121-01 trackea cuotas betplay/rushbet reales

---

## 4. Código a escribir

### Archivo 1: `scraping/match_ledger.py` — +~60 líneas

**Nueva función** (insertar después de `exportar_para_edge_calculator()` ~L521):

```python
def enriquecer_ss_fs_con_aggregator(fecha: str, data_dir: str = "data",
                                     libros: list = None) -> dict:
    """
    Enriquece ss_fs sin cuotas usando odds_aggregator (betplay+rushbet).
    D121-01: conecta el ledger con el feed más amplio de la API Kambi.
    Retorna stats: {enriquecidos, sin_match, homónimos, total_ss_fs}.
    """
    from scripts.odds_aggregator import fetch_all_odds
    from analysis.dual_book_client import best_price  # Nodo-111
    libros = libros or ['betplay', 'rushbet']

    ledger = load_ledger(fecha, data_dir)
    if not ledger:
        return {}

    ss_fs = ledger.get('single_source_fs', [])
    ss_sin_cuota = [p for p in ss_fs if not p.get('cuota1')]
    if not ss_sin_cuota:
        return {'enriquecidos': 0, 'sin_match': 0, 'homónimos': 0, 'total_ss_fs': len(ss_fs)}

    feeds = fetch_all_odds(libros)

    enriquecidos = 0
    sin_match = 0
    homonimos = 0

    for partido in ss_sin_cuota:
        j1 = _normalizar_nombre(partido.get('jugador1', ''))
        j2 = _normalizar_nombre(partido.get('jugador2', ''))

        # Match jugador1
        cuota1, book1 = _buscar_cuota_aggregator(j1, feeds)
        cuota2, book2 = _buscar_cuota_aggregator(j2, feeds)

        if cuota1 and cuota2:
            partido['cuota1'] = cuota1
            partido['cuota2'] = cuota2
            partido['_cuota_source'] = book1  # 'betplay' o 'rushbet'
            partido['_best_book'] = book1
            partido['_enriched_by'] = 'D121-01'
            enriquecidos += 1
        else:
            sin_match += 1

    save_ledger(ledger, fecha, data_dir)
    stats = {'enriquecidos': enriquecidos, 'sin_match': sin_match,
             'homónimos': homonimos, 'total_ss_fs': len(ss_fs)}
    logger.info(f"   Enrichment D121-01: {enriquecidos}/{len(ss_sin_cuota)} ss_fs enriquecidos "
                f"({sin_match} sin match, {homonimos} homónimos)")
    return stats


def _buscar_cuota_aggregator(nombre_norm: str, feeds: dict) -> tuple:
    """Busca cuota para un jugador en feeds del aggregator. Retorna (cuota, bookmaker)."""
    token = nombre_norm.split()[0] if nombre_norm else ''
    candidatos = [k for k in feeds if nombre_norm in k or k.startswith(token + ' ')]
    if len(candidatos) == 1:
        info = feeds[candidatos[0]]
        # Elegir mejor precio entre libros
        mejor = max(
            [(b, d['odds']) for b, d in info.items() if isinstance(d, dict) and d.get('odds')],
            key=lambda x: x[1], default=(None, None)
        )
        return mejor[1], mejor[0]
    return None, None
```

**CLI** — añadir `--enrich` flag (~L598):
```python
parser.add_argument('--enrich', action='store_true',
                    help='Enriquecer ss_fs sin cuota via odds_aggregator (D121-02)')
# ... después de fusionar_dia():
if args.enrich:
    enrich_stats = enriquecer_ss_fs_con_aggregator(args.fecha, data_dir=args.data_dir)
    print(f"Enriched: {enrich_stats.get('enriquecidos',0)} ss_fs via aggregator")
```

**run_daily.py** — PASO 1.5 actualizado:
```python
# PASO 1.5 — Ledger crosswalk + enrichment (Nodo-118 + Nodo-121)
python3 scraping/match_ledger.py --build --enrich --fecha FECHA
```

### Archivo 2: `tests/test_nodo121_aggregator_enrichment.py` — 3 tests REGLA-T53

```
test_enriquece_ss_fs_sin_cuota()
    ss_fs con cuota1=None, feeds mock con cuota real
    → después de enriquecer: cuota1=3.0, _cuota_source='betplay'

test_ss_fs_con_cuota_no_se_toca()
    ss_fs que ya tiene cuota1=1.8 (Nodo-120)
    → enriquecer no modifica, cuota1 sigue 1.8

test_homonimo_no_enriquece()
    feeds tiene 'aksu' apuntando a DOS partidos distintos
    → sin match, cuota1 sigue None, homonimos=1
```

### Archivo 3: `validation/preregistered_hypotheses.json`
Añadir H121-01 (ver §3).

---

## 5. Lo que NO cambia

- `fusionar_dia()` — sin tocar
- `exportar_para_edge_calculator()` — ya maneja ss_fs con cuota (Nodo-120)
- `edge_calculator.py` — sin cambios
- `odds_aggregator.py` — consumido como función pura
- `shadow_book.py` — `_cuota_source` ya es informacional

---

## 6. Flujo completo post-implementación

```
PASO 1a  extraer_partidos_api.py      → Kambi 122 partidos
PASO 1b  extraer_URL_partidos.py      → Playwright 96 partidos
PASO 1.5 match_ledger --build --enrich:
    fusionar_dia()                    → 85 joins + 37 ss_k + 11 ss_fs(cuota=None)
    enriquecer_ss_fs_con_aggregator() → odds_aggregator 237 partidos
                                        → 8/11 enriquecidos (cuota betplay/rushbet)
    exportar_para_edge_calculator()   → 85+37+8 = 130 picks al edge_calculator
                                        (+8 qualifying que antes eran 0)
PASO 2   extraer_historh2h.py        → H2H para los 130
PASO 3   edge_calculator.py          → picks con _cuota_source='betplay'/'rushbet'
```

---

## §WIKILINKS COMPLETOS

### Forward links (este nodo depende de)
- [[Nodo-120-FS-Single-Source-Cuotas-Qualifying-Flow]] — compuerta ss_fs abierta; este nodo la completa para cuota=None
- [[Nodo-118-Match-Ledger-Crosswalk-Identidad-Fusion-Definitiva]] — fusionar_dia() + save_ledger() + load_ledger()
- [[Nodo-90-Auditoria-Fable-Nodo89]] — D90-08: odds_aggregator.py origen (fetch_all_odds)
- [[Nodo-111-Dual-Book-Live-Intelligence]] — best_price() funciones puras para elegir mejor casa
- [[Nodo-116-Entierro-Dashboard-Vieja-AutoCombo-AntiFlood-P8-MultiCasa]] — rushbet VERIFIED 2026-07-19 + wplay VERIFIED 2026-07-14
- [[Nodo-80-Kambi-Name-Matching]] — patrón apellido→nombre para name matching

### Back links (nodos que deben conocer este)
- [[Nodo-120-FS-Single-Source-Cuotas-Qualifying-Flow]] ← complementario: enriquece lo que Nodo-120 no puede (cuota=None)
- [[Nodo-118-Match-Ledger-Crosswalk-Identidad-Fusion-Definitiva]] ← F7 addendum: enriquecimiento post-fusión

### Huérfanos operacionales
- `nodos_index.json` — reindexar tras implementación (118 nodos)
- `validation/preregistered_hypotheses.json` — H121-01 a registrar
- `run_daily.py` — PASO 1.5 actualizar flag `--enrich`
