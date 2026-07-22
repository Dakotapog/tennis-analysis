# Nodo-132 — Activación ComboRegistry: Conectar Builders a log_combo() para P&L Real

> **Estado:** PROPUESTA — pendiente auditoría Fable
> **Tipo:** INTEGRATION — Nodo-76 existe al 100%, cero activaciones
> **Trigger:** 2026-07-21 — `combo_registry.py --report` retorna "Sin registros" después de semanas de combos generados
> **Autor:** Sonnet 4.6 (auditoría graphify + pipeline real)
> **Hallazgo graphify:** `ComboRegistry.log_combo()` / `.settle_date()` / `._settle_pierna()` existen en `combo_registry.py:L89` — FALSO PENDIENTE por falta de integración
> **Para auditoría:** Fable debe verificar (1) mapeo correcto de subtipos, (2) que log_combo se llame DESPUÉS del BAT (no antes), (3) coherencia stake vs. Kelly real

---

## Wikilinks

| Link | Rol |
|------|-----|
| [[Nodo-76-Combo-PnL-Registry]] | Padre — ComboRegistry implementado, nunca activado |
| [[Nodo-100-Taxonomia-Estrategias-Generacion-Combos]] | 12 estrategias — mapa de subtipos |
| [[Nodo-52-Shadow-Book-CLV-Tracking]] | shadow_book.py — referencia de arquitectura append-only |
| [[Nodo-27-Pipeline-Tracker-Observabilidad]] | pipeline_tracker.py --section confianza — consumidor del reporte |
| [[Nodo-109-Live-Trading-Desk-Dashboard]] | live_desk.py — panel P&L combos en dashboard |

**Wikilinks totales: 5 | Huérfanos: 0**

---

## §1. El gap real — evidencia directa

```bash
$ python3 combo_registry.py --report
COMBO P&L REGISTRY
Sin registros.
```

`reports/combo_registry/` = 0 archivos JSON a pesar de semanas de combos generados.

**Causa:** Ningún builder llama `ComboRegistry.log_combo()`. El sistema está desconectado.

### §1.1 Lo que YA existe en Nodo-76 (no reimplementar)

```python
ComboRegistry.log_combo(tipo, subtipo, bat_name, piernas, cuotas, stake, fecha_jornada)
  → escribe cr_{fecha}.jsonl en reports/combo_registry/
  → retorna cr_id único

ComboRegistry.settle_date(fecha, resultados_path)
  → lee cr_{fecha}.jsonl + resultados_finales_{fecha}.json
  → settle per pierna via ._settle_pierna()
  → calcula pnl = stake * cuota_compuesta si GANO, else -stake

combo_registry.py --settle YYYY-MM-DD   → CLI settle
combo_registry.py --report              → P&L histórico por tipo/subtipo
```

### §1.2 Lo que FALTA (Nodo-132)

3 cosas quirúrgicas:
1. Llamar `log_combo()` en cada función que genera BAT
2. Llamar `combo_registry.py --settle` en `run_daily.py` post-settle
3. Mostrar resumen en `pipeline_tracker --section confianza` y `live_desk` panel

---

## §2. Mapa de integración — 1 llamada por builder

### D132-01 — `combo_confianza_builder.py`: función `_generate_bat_files()` (~L1590)

```python
# D132-01a: al generar cada bat principal (CORE/SATELLITE/MOONSHOT/COBERTURA)
from combo_registry import ComboRegistry as _CR
_cr = _CR()
for idx, combo in enumerate(valid_combos, 1):
    # ... código existente de bat ...
    bat_path = DESKTOP_WIN / f"{prefix}{idx}.bat"
    # NUEVO: log_combo inmediatamente después de escribir el bat
    _cr.log_combo(
        tipo="CC",
        subtipo=combo.get("categoria", "CORE"),   # CORE|SATELLITE|MOONSHOT|COBERTURA
        bat_name=f"{prefix}{idx}",
        piernas=[leg["jugador"] for leg in combo["legs"]],
        cuotas=[leg["cuota"] for leg in combo["legs"]],
        stake=combo.get("stake", 2000),
    )
```

```python
# D132-01b: en _build_anchor_bat() (~L920) para combos ANCHOR
_cr.log_combo(
    tipo="AC",
    subtipo=combo.get("tipo", "ANCHOR"),  # ANCHOR_1A3B|ANCHOR_2A2B|ANCHOR_3A2B
    bat_name=f"AC{idx}",
    piernas=[leg["jugador"] for leg in combo["legs"]],
    cuotas=[leg["cuota"] for leg in combo["legs"]],
    stake=1500,
)
```

### D132-02 — `betplay_combo_builder.py`: 6 puntos de log

```python
# D132-02a: build_combo_links() → combos generales (~L390) — tipo "Combo", subtipo "STANDARD"
# D132-02b: _generate_safe_bat() (~L1315) — tipo "Safe", subtipo "SAFE"
# D132-02c: build_was_combos() bat generation — tipo "WAS", subtipo "WAS"
# D132-02d: build_mega_combos() bat generation — tipo "Mega", subtipo "MEGA"
# D132-02e: build_games_combos() → GamesA/B/C — tipo "Games", subtipo "GAMES_A|B|C"
# D132-02f: build_evaluar_games_combos() → EvalGamesA — tipo "Games", subtipo "EVALUAR"

# Ejemplo para Safe:
from combo_registry import ComboRegistry as _CR
_cr = _CR()
bat_path = DESKTOP_WIN / f"Safe{idx}.bat"
# ... código existente ...
_cr.log_combo(
    tipo="Safe",
    subtipo="SAFE",
    bat_name=f"Safe{idx}",
    piernas=[leg["jugador"] for leg in combo["legs"]],
    cuotas=[leg["cuota"] for leg in combo["legs"]],
    stake=stake_per_combo,
)
```

### D132-03 — `run_daily.py`: settle automático después de `shadow_book --settle`

```python
# D132-03: después del PASO 10a (shadow_book --settle)
_run_step(
    f"python3 combo_registry.py --settle {fecha}",
    f"PASO 10b-combo: Combo Registry settle {fecha}",
    optional=True,
)
```

### D132-04 — `pipeline_tracker.py`: sección combo en `--section confianza`

```python
# D132-04: en seccion_27_6_temporal() o nueva seccion_combo()
import subprocess, json
result = subprocess.run(
    ["python3", "combo_registry.py", "--report", "--json"],
    capture_output=True, text=True
)
if result.returncode == 0:
    data = json.loads(result.stdout)
    # mostrar tabla: tipo | n | hit% | ROI | stake_total | pnl_total
```

### D132-05 — `combo_registry.py`: añadir flag `--json` para output machine-readable

```python
# D132-05: en main(), si --json: print(json.dumps(report_dict()))
# Permite consumo desde pipeline_tracker y live_desk sin parsear texto
```

---

## §3. Mapeo canónico de subtipos

| Builder | Función | tipo | subtipo |
|---------|---------|------|---------|
| combo_confianza_builder | _generate_bat_files (CORE) | CC | CORE |
| combo_confianza_builder | _generate_bat_files (SATELLITE) | CC | SATELLITE |
| combo_confianza_builder | _generate_bat_files (MOONSHOT) | CC | MOONSHOT |
| combo_confianza_builder | _generate_bat_files (COBERTURA) | CC | COBERTURA |
| combo_confianza_builder | _build_anchor_bat | AC | ANCHOR |
| betplay_combo_builder | build_combo_links | Combo | STANDARD |
| betplay_combo_builder | _generate_safe_bat | Safe | SAFE |
| betplay_combo_builder | build_was_combos | WAS | WAS |
| betplay_combo_builder | build_mega_combos | Mega | MEGA |
| betplay_combo_builder | build_games_combos | Games | GAMES_A/B/C |
| betplay_combo_builder | build_evaluar_games_combos | Games | EVALUAR |
| favoritos_combo_builder | main | Fav | FAVORITOS |

---

## §4. Conexiones ocultas que graphify reveló

### §4.1 `pipeline_tracker` ya planea una sección de combos
`seccion_27_6_temporal()` existe en `pipeline_tracker.py:L766` — graphify la detectó. Su nombre sugiere que es temporal y espera datos de combo_registry para volverse permanente.

### §4.2 `docs/ROI-LEDGER.md` referenciado en graphify como nodo
Graphify detectó `"ROI Ledger (docs/ROI-LEDGER.md, actualización semanal, 10 min)"`. Ese archivo probablemente fue diseñado para recibir output de combo_registry `--report`. La pipeline de datos ya fue arquitectada — falta el dato.

### §4.3 `combo_governor.py` puede consumir combo P&L para soft-veto dinámico
`combo_governor.py` (C63-B) actualmente veta por presupuesto fijo. Con combo_registry activo, puede comparar ROI real vs. esperado por estrategia y ajustar umbrales de veto dinámicamente. Esto es el alpha operativo de mediano plazo.

### §4.4 El stake real vs. Kelly no está en ningún registro actual
Cuando `trader_ev_tenis.py` (EL MOTOR) aprueba un pick, el stake Kelly se calcula. Ese stake llega a `shadow_book.py` via `log_picks()`. Pero los combos usan stakes fijos ($1k, $2k, $5k) que NUNCA llegan a ningún registro. `combo_registry.log_combo(stake=...)` cierra ese hueco.

---

## §5. Preguntas abiertas para Fable

1. **¿El log_combo debe ir ANTES o DESPUÉS de verificar que Chrome abrió el BAT?**
   Si el usuario no abre el BAT (decide no apostar), el combo queda registrado pero nunca fue apostado. ¿Log en generación (intención) o en apertura (ejecución)?

2. **¿Stake fijo o stake Kelly para los combo logs?**
   Los builders usan stakes fijos (`stake_per_combo=1000`). El Kelly real para ese combo sería diferente. ¿Registrar el stake fijo del builder o calcular Kelly por combo en el momento del log?

3. **¿Cómo manejar combos con cuotas parcialmente desconocidas al momento del log?**
   `build_safe_combos()` calcula cuotas en el momento. `build_mega_combos()` puede tener cuotas @100x+ estimadas. ¿Son confiables para P&L?

4. **¿`combo_registry --settle` puede compartir datos con `shadow_book --settle`?**
   Los resultados de `resultados_finales.py` son la misma fuente para ambos. ¿Un solo settle que alimente ambos, o mantener separados (cada uno lee por su cuenta)?

5. **¿El panel en live_desk debe ser P9 (nuevo) o extender P6 (P&L existente)?**
   P6 ya muestra segmentos de shadow_book. ¿Combos = subsección de P6 o panel dedicado P9-COMBO-PNL?

---

## §6. Hipótesis pre-registrada

```json
{
  "id": "H132-01",
  "descripcion": "Combos CC (CORE+SATELLITE) tienen hit_rate > 20% (P_combo 4-leg promedio breakeven)",
  "formula": "hit_rate_CC > 1 / cuota_combo_promedio_CC",
  "n_stop": 30,
  "baseline": "hit_rate_Mega + hit_rate_Safe",
  "estado": "PENDIENTE_IMPLEMENTACION",
  "fecha_registro": "2026-07-21"
}
```

---

## §7. Tests REGLA-T53 — `tests/test_nodo132_combo_registry_activation.py`

```python
def test_D132_01_combo_confianza_logs_core_combo()
    # después de _generate_bat_files(), cr_{fecha}.jsonl existe con subtipo=CORE

def test_D132_02_betplay_logs_safe_combo()
    # después de _generate_safe_bat(), cr_{fecha}.jsonl tiene tipo=Safe

def test_D132_02_betplay_logs_games_combo()
    # después de build_games_combos(), tipo=Games subtipo=GAMES_A

def test_D132_03_run_daily_settle_calls_combo_registry()
    # run_daily paso 10b-combo ejecuta combo_registry --settle

def test_D132_05_combo_registry_json_flag_machine_readable()
    # combo_registry --report --json retorna JSON parseable con claves tipo/subtipo/hit_pct/pnl

def test_H132_01_hypothesis_registered()
    # H132-01 en preregistered_hypotheses.json
```

---

## §8. Orden de implementación

| Fix | Archivo | Complejidad | Prioridad |
|-----|---------|-------------|-----------|
| D132-05 | `combo_registry.py` | 5 líneas — flag --json | HOY — desbloquea todo |
| D132-01a/b | `combo_confianza_builder.py` | 8 líneas por punto | ALTA |
| D132-02 (Safe+Games) | `betplay_combo_builder.py` | 6 × 5 líneas | ALTA |
| D132-03 | `run_daily.py` | 3 líneas | ALTA |
| D132-04 | `pipeline_tracker.py` | 15 líneas | MEDIA |
| D132-02 (WAS+Mega+Std) | `betplay_combo_builder.py` | 3 × 5 líneas | MEDIA |

---

## §9. Resumen ejecutivo para Fable

**Qué pasó:** `combo_registry.py` (Nodo-76) fue implementado con toda la lógica necesaria: `log_combo()`, `settle_date()`, `_settle_pierna()`, `--report`. Pero ningún builder llama esas funciones. Resultado: 0 registros de combos en semanas de operación.

**Por qué importa:** Sin registros, no podemos responder: ¿cuál estrategia tiene mejor ROI? ¿CORE bate a MOONSHOT? ¿SAFE tiene hit% > breakeven? ¿EVALUAR_GAMES como combo supera EVALUAR individual? Todo eso queda en especulación.

**Qué se propone:** 6 puntos de integración (D132-01 a D132-05). El más simple es D132-05 (flag --json, 5 líneas). El más impactante es D132-01 + D132-02 (conectar todos los builders). Total: ~60 líneas de código para activar meses de arquitectura ya construida.

**Conexiones ocultas descubiertas:** `seccion_27_6_temporal()` en pipeline_tracker.py espera datos de combo_registry. `docs/ROI-LEDGER.md` fue diseñado para recibir este output. `combo_governor.py` puede usar el P&L real para veto dinámico. Tres sistemas listos para consumir datos que aún no llegan.

---

**Wikilinks totales: 5 | Huérfanos: 0**

[[Nodo-76-Combo-PnL-Registry]] | [[Nodo-100-Taxonomia-Estrategias-Generacion-Combos]] | [[Nodo-52-Shadow-Book-CLV-Tracking]] | [[Nodo-27-Pipeline-Tracker-Observabilidad]] | [[Nodo-109-Live-Trading-Desk-Dashboard]]
