# Nodo-174 — Auditoría de Conexiones Rotas: el Lazo de Validación Abierto (D174-01 → D174-14)

> **Estado:** SPEC — pendiente implementación.
> **Origen:** auditoría 2026-08-05, derivada del trabajo de campo de [[Nodo-173]].
> **Tipo:** auditoría de integración (mismo patrón que [[Nodo-86]], [[Nodo-99]], [[Nodo-119]], [[Nodo-154]]).
> **Naturaleza de los hallazgos:** ninguno es un bug de lógica. Todos son **desconexiones** —
> código que existe, corre bien, y cuyo output no llega a ningún consumidor.

---

## 0. Tesis de la auditoría

[[Nodo-173]] explica por qué el sistema **no apuesta**. Este nodo explica por qué el sistema
**no aprende**.

El proyecto tiene una constitución sólida (REGLA #8: ninguna hipótesis sin pre-registro) y la
cumple religiosamente: 39 hipótesis declaradas con umbrales congelados. Pero **el mecanismo que
convierte observaciones en conteos está roto o ausente en 18 de esas 39**. El resultado es un
registro de intenciones que nunca se cierra: se declaran hipótesis, se construyen las señales, se
serializan al reporte… y ahí mueren.

Consecuencia práctica: en 3 meses **una sola hipótesis graduó** (H60-01/GCS, n=54), y lo hizo por
una ruta que sí estaba conectada. Todas las demás están en `n=0` o cerca, no porque falten
partidos, sino porque **nadie las cuenta**.

> **Esto es la explicación estructural de por qué el sistema no mejora solo.** Nodo-173 abre el
> embudo; Nodo-174 conecta el lazo de retroalimentación. Sin ambos, Nodo-173 produciría más
> volumen sin producir más conocimiento.

---

## 1. Hallazgos — Bloque A: el lazo de validación está abierto

### A1 — 18 de 39 hipótesis no tienen ninguna ruta de medición

`shadow_book.report()` segmenta 25 IDs de hipótesis. El JSON declara 39. Intersección: 21.

**Declaradas y sin segmento — estructuralmente incapaces de acumular, nunca:**

```
H60-01  H60-02  H77-01  H77-03  H96-01  H97-01  H103-01 H110-01
H111-01 H113-01 H120-01 H121-01 H125-01 H132-01 H139-01 H139-02
H165-01 H166-01
```

Incluye:
- **H60-01 (GCS)** — la *única* estrategia GRADUADA del sistema, según CLAUDE.md §11, con hit
  64.8% y n=54. **No tiene segmento en `report()`.** Su `n=54` está en el JSON como literal.
  La graduación fue manual. No hay forma de verificarla ni de detectar si se degrada.
- **H96-01 (IRP)** — Sprint 5 completo, 4361 perfiles construidos, `irp_fav`/`irp_rival`
  serializados en el 100% de los picks. Cero medición.
- **H113-01 (weather)** — Nodo-113 completo, `weather_flag` en el 100% de los picks. Cero medición.
- **H111-01 (dual-book X1)**, **H110-01**, **H165-01**, **H166-01**, **H139-01/02** — nodos
  declarados completos, hipótesis declaradas ACUMULANDO, sin ruta de conteo.

**Reproducción:**

```bash
python3 - <<'PY'
import json,re
decl={k for k in json.load(open('validation/preregistered_hypotheses.json'))['hypotheses'] if re.match(r'H\d+-\d+',k)}
seg=set(re.findall(r'H\d{2,3}-\d{2}',open('shadow_book.py',encoding='utf-8',errors='ignore').read()))
print("sin segmento:", " ".join(sorted(decl-seg)))
print("sin pre-registro:", " ".join(sorted(seg-decl)))
PY
```

### A2 — 4 hipótesis medidas sin pre-registro (violación inversa de REGLA #8)

```
H147-01  H150-01  H151-01  H152-01
```

`shadow_book.py` las segmenta y les asigna `p0`/`p1` (ej. `:1576` → `("H152-01", ..., 0.385, 0.55)`),
pero **no están en `validation/preregistered_hypotheses.json`**. Los umbrales viven hardcodeados en
el código del reporte, donde nada impide moverlos después de ver los datos. Es la puerta de atrás
al p-hacking que la REGLA #8 existe para cerrar.

Los specs de Nodo-147/150/151/152 afirman que fueron pre-registradas. No lo están.

### A3 — `n_actual` no lo escribe nadie

Búsqueda exhaustiva de escrituras a `n_actual` fuera de tests: **cero**. Las únicas apariciones son
un `st.caption()` en `dashboard.py:596` y literales en `live_desk.py`.

`git log` sobre `validation/preregistered_hypotheses.json`: última modificación **2026-07-22**, y
fue para *añadir* H139-01/02, no para actualizar conteos. **El archivo es un registro de solo
escritura manual.**

### A4 — El panel de validación del live desk muestra números inventados

`live_desk.py:135, 253, 271, 289, 303`:

```python
"n_actual": 0,
"n_actual": 3,     # H88-01
"n_actual": 54,    # H60-01
"n_actual": 8,     # semilla jul-14/16
"n_actual": 8,
```

Las barras de progreso `n_actual/n_stop` que el usuario ve en `:7780` son **literales tipeados a
mano en julio**. No se han movido desde entonces y no se moverán. El instrumento que debería
decir "faltan 27 para graduar H88-01" dice 3/30 desde hace tres semanas porque el 3 está escrito
en el código fuente.

---

## 2. Hallazgos — Bloque B: señales producidas y tiradas

Verificado contra `reports/edge_report_20260805_105416.json` (46 picks) y grep de consumidores
reales (excluyendo el productor y los tests):

| señal | producida | consumidores reales | veredicto |
|---|---|---|---|
| `weather_flag` | **46/46** | **0** | **muerta** |
| `irp_fav` / `irp_rival` | **46/46** | **0** (solo su propio builder) | **muerta** |
| `hcuc_convergence` | **0/46** | shadow_book (lector) | **nunca producida** |
| `outcome_id` | **0/46** | 3 combo builders (esperándola) | **nunca producida** |
| `alpha_vs_elo` | 46/46 | `betplay_combo_builder.py:755` (bonus) | viva ✓ |
| `p_elo_base` | 46/46 | 0 fuera de `alpha_vs_elo` | insumo interno, aceptable |
| `rfi_tier` | 46/46 | 7 archivos | viva pero **no-nula solo 3/46** ([[Nodo-173]] D173-09) |
| `score_directo` | 46/46 | meta_signal | viva, no-nula 21/46 |
| `meta_score` | **0/46** | — | ausente (Nodo-98 parcialmente wireado) |

### B1 — `_calc_hcuc_convergence` nunca fue escrita

```bash
grep -c "hcuc" edge_calculator.py              # → 0
git log --all --oneline -S "_calc_hcuc_convergence" -- edge_calculator.py   # → vacío
```

**La función nunca fue commiteada, en ninguna rama, en ningún momento.**

Sin embargo `CLAUDE.md` documenta: *"D155-02: `_calc_hcuc_convergence(resultado, _surf_fav,
_surf_dog)` en `edge_calculator.py` … 11 tests REGLA-T53 — 11/11 PASS"*, y
`shadow_book.py:1566` lee `pick_snapshot.hcuc_convergence` para poblar el segmento de H152-01.

Como el campo nunca existe, ese segmento siempre recibe lista vacía → **H152-01 está clavada en
n=0 por construcción**, y lo estará para siempre.

### B2 — `outcome_id` en el edge_report: marcado ✅, nunca implementado

```bash
grep -n "'outcome_id'" edge_calculator.py      # → sin resultados
git log --all --oneline -S "outcome_id" -- edge_calculator.py   # → vacío
```

`CLAUDE.md` tabla Nodo-154: *"G2 | Gap | `edge_calculator.py:~1560` outcome_id ausente del output
dict | D154-06 | ✅"*.

**Nunca existió.** Y es caro: por eso `betplay_combo_builder`, `combo_confianza_builder` y
`favoritos_combo_builder` tienen que **re-matchear por apellido contra Kambi en cada corrida** —
el mecanismo frágil que produjo el bug B4 de Nodo-154 (2/13 matches), que se parcheó con
`_PARTICLES` (D154-04) llevándolo a ≥8/13. **D154-06 era el fix estructural que hacía innecesario
ese parche.** Se marcó hecho sin hacerse, y el sistema sigue pagando el impuesto del
name-matching todos los días.

### B3 — Nodo-96 y Nodo-113: trabajo completo, valor cero

`irp_fav`/`irp_rival` (4361 perfiles, PASO 0c diario) y `weather_flag` (llamada a open-meteo por
partido) se computan, se serializan, y **nadie los lee jamás** — ni un gate, ni un builder, ni un
segmento del shadow book. Es costo de cómputo y de API sin retorno posible.

No es que las señales sean malas. Es que **no hay forma de saber si son buenas**.

---

## 3. Hallazgos — Bloque C: el suite de tests no corre

```
collected 2571 items / 1 error
ERROR tests/test_nodo155_hcuc_convergence.py
ImportError: cannot import name '_calc_hcuc_convergence' from 'edge_calculator'
!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!
```

**pytest aborta en la fase de colección.** No ejecuta ni un test. El comando documentado en
`CLAUDE.md §8` como paso obligatorio de baseline (`python -m pytest tests/ --no-cov -q`) devuelve
exit code 2 sin correr nada.

Con el módulo roto excluido, el suite **sí corre**, y el resultado real (2026-08-05) es:

```
python3 -m pytest tests/ --no-cov -q --ignore=tests/test_nodo155_hcuc_convergence.py
→ 2541 passed, 28 failed, 2 skipped
```

Dos lecturas honestas:

1. **El conteo `2541 passed` de CLAUDE.md es correcto** — no fue inventado. Se obtiene con
   `--ignore`, no con el comando documentado.
2. **Los `27 failed` documentados como "pre-existentes sin relación" son en realidad 28, y nunca
   se enumeraron.** Repartidos así:

```
5  test_nodo147_telegram_fix.py          4  test_nodo141_kambi_only_report.py
4  test_nodo160_p_bridge_wiring.py       3  test_nodo40.py
3  test_nodo156_rival_apellido_fix.py    2  test_nodo42.py
2  test_nodo159_games_settlement.py      1  test_nodo51_f3.py
1  test_nodo160_winner_market_wiring.py  1  test_nodo154_pipeline_integrity.py
1  test_nodo146_h2h_model_favoritos.py   1  test_nodo135_games_live_api.py
```

Nueve de esos 28 (`nodo159`, `nodo160` ×5, `nodo154`, `nodo141`, `nodo147`) tocan **wiring de
señales live y de disponibilidad Kambi** — exactamente el territorio de los hallazgos A y B de
este nodo. Llamarlos "sin relación" fue una suposición, no una verificación.

Esto también explica la nota de deuda *"test_nodo155 import error pre-existing"*: se registró como
molestia conocida sin advertir que **rompe el comando de baseline de la constitución del proyecto**.

---

## 4. Hallazgos — Bloque D: estrategias en producción sin cobertura

| función | archivos de test | stake típico |
|---|---:|---|
| `build_system_combos` (Sistema Leave-One-Out) | **0** | $3,500 |
| `build_safe_combos` (SAFE) | **0** | $1,000 |
| `build_was_combos` (WAS) | **0** | $5,000 |
| `build_mega_combos` | 1 | $500 |
| `build_ancla_segura_combos` | 1 | $3,000 |

`CLAUDE.md` sobre Nodo-156-B: *"8 tests REGLA-T53 pendientes sesión futura."* La sesión futura no
llegó. `ls tests/ | grep 156` devuelve solo `test_nodo156_rival_apellido_fix.py`, que es otro nodo.

**WAS mueve el stake más alto del sistema ($5,000) y no tiene un solo test.**

---

## 5. Hallazgos — Bloque E: instrumentos huérfanos

Módulos que ningún otro archivo importa ni invoca:

| archivo | origen | estado |
|---|---|---|
| `scripts/audit_phantom_history.py` | Nodo-152 D152-06 (auditoría retroactiva 30 días) | **nunca ejecutado** |
| `scripts/backfill_evaluar_shadow.py` | — | huérfano |
| `scripts/nodo_pagerank.py` | — | huérfano |
| `scripts/backfill_strategy.py` | Nodo-144 D144-06 | one-shot, ya cumplió — OK |

`audit_phantom_history.py` es el que importa: se construyó específicamente para detectar
retroactivamente contaminaciones tipo Vesantera (historial top-ATP asignado a un ITF M15, que
costó AC1–AC12 con edge inventado del 39.5%). **Existe, nunca corrió, y nadie sabe cuántos
Vesanteras hay en los 3838 resultados de calibración.**

---

## 6. Hallazgos — Bloque F: el embudo de disponibilidad

`kambi_disponible` no-nulo en **17 de 46 picks (37%)**. El 63% del universo diario **no es
apostable en Betplay**, y esa exclusión es silenciosa (ver [[Nodo-173]] D173-10).

Combinado con el embudo de gates de Nodo-173, el pipeline real es:

```
268 analizados → 46 serializados → 17 apostables en Betplay → 2 apostar → 0-1 tras REGLA-HF-1/KGR
```

**El gate de disponibilidad Kambi es del mismo orden de magnitud que T32-01**, y hasta ahora
nadie lo había medido.

---

## 7. Deliverables

### PRIORIDAD 0 — Desbloquear la puerta de calidad

#### D174-01 — Reparar la colección de pytest

Dos opciones; **elegir (a)**, no (b):

**(a) Implementar `_calc_hcuc_convergence` según Nodo-155 D155-02** — la función está
especificada en `.spec/01_Nodos/Nodo-155-HCUC-Pipeline-Integration.md`, junto a
`_calc_elo_dominance_axis` en `edge_calculator.py`, 100% observacional (no toca
`edge`/`kelly_kl`/`apostar`). Asignar `hcuc_convergence` / `hcuc_signals` al final de
`calcular_edge_completo()`. Esto además desbloquea H152-01, que está en n=0 por esta causa.

**(b) NO** borrar ni skipear el test. El test es correcto — está detectando ausencia real de
código. Skipearlo enterraría el hallazgo.

**Aceptación:** `python3 -m pytest tests/ --no-cov -q` colecciona los 2571 items y reporta un
número. Registrar ese número como **baseline verificado** en CLAUDE.md, reemplazando los conteos
no verificables.

#### D174-02 — Baseline honesto del suite + triaje de los 28 fallos

Tras D174-01, correr `pytest tests/ --no-cov -q` **sin `--ignore`** y documentar en CLAUDE.md §5
el conteo real, con los fallos **nombrados uno por uno**. Prohibido escribir
"N pre-existentes sin relación" sin enumerarlos y sin haber leído cada uno — esa formulación es
lo que permitió que el error de colección sobreviviera semanas.

Triaje obligatorio de los 28 (lista en §3). Prioridad a los 9 que tocan wiring live/Kambi
(`nodo159`, `nodo160`×5, `nodo154`, `nodo141`, `nodo147`): para cada uno, decidir y anotar —
**(i)** el test está mal y se corrige, **(ii)** el código está mal y se abre deliverable, o
**(iii)** el test quedó obsoleto por un cambio de diseño posterior y se retira citando el nodo que
lo superó. Ninguna otra salida es válida; "pre-existente" no es un diagnóstico.

---

### PRIORIDAD 1 — Cerrar el lazo de validación

#### D174-03 — `hypothesis_ledger`: conteo automático

**Archivo nuevo:** `validation/hypothesis_ledger.py`

```python
def contar_hipotesis(settled: list[dict]) -> dict[str, dict]:
    """Para cada H-XX con predicado registrado, cuenta n/hits/roi sobre los liquidados.
    Devuelve {h_id: {'n':int,'hits':int,'roi':float,'sprt':dict|None}}.
    Función pura — no lee ni escribe archivos."""

def actualizar_registro(path_json: str, conteos: dict, *, dry_run: bool = True) -> dict:
    """Escribe n_actual/hits/roi_flat_1u en preregistered_hypotheses.json.
    NUNCA toca 'umbrales_congelados' ni 'preregistrado' (inmutables, anti p-hacking).
    Devuelve el diff aplicado."""
```

**Registro de predicados** — un único diccionario que mapea `h_id → predicado(record) -> bool`,
sustituyendo los segmentos dispersos y hardcodeados de `shadow_book.report()`:

```python
PREDICADOS = {
    'H96-01':  lambda r: bool(_snap(r).get('irp_fav')),
    'H113-01': lambda r: bool(_snap(r).get('weather_flag')),
    'H60-01':  lambda r: bool(_snap(r).get('gcs_active')),
    ...
}
```

**Invariante que un test debe verificar:** `set(PREDICADOS) == set(hipótesis declaradas)`.
Falla si alguien declara una hipótesis sin predicado (A1) o mide una sin declarar (A2).

**Integrar como PASO 10e en `run_daily.py`**, después del settle: cada liquidación actualiza los
conteos automáticamente.

#### D174-04 — Predicados para las 18 hipótesis sin ruta

Escribir el predicado de cada una de las 18 de §A1. Para las que dependen de un campo que **no se
serializa**, el deliverable incluye serializarlo. Casos conocidos:

- `H60-01` — requiere `gcs_active` en `pick_snapshot`. **Verificar que se serializa**; es la
  única hipótesis GRADUADA y su graduación es hoy inauditable.
- `H111-01` — requiere el resultado del router dual-book (PASO 3.7) en el snapshot.
- `H139-01/02` — requieren un tag de origen Kambi-First en el pick.
- `H165-01` / `H166-01` — señales live; el predicado vive sobre `shadow_book` de picks live
  (`pick_type='live'`, Nodo-101).

#### D174-05 — Pre-registrar las 4 hipótesis huérfanas

Mover `H147-01`, `H150-01`, `H151-01`, `H152-01` a
`validation/preregistered_hypotheses.json` con **exactamente** los `p0`/`p1` hoy hardcodeados en
`shadow_book.py` (no re-derivarlos — congelarlos tal cual están, que es el punto de la regla), y
sustituir los literales por lectura del JSON.

#### D174-06 — El live desk deja de mentir

**Archivo:** `live_desk.py:135, 253, 271, 289, 303`

Eliminar todos los `n_actual` literales. El panel lee de
`validation/preregistered_hypotheses.json` tras D174-03. Si el archivo no tiene conteo fresco
(`fitted_at` con más de 48h), mostrar **`n=?`** explícito, nunca un número viejo.

**Principio:** un instrumento sin dato debe decir que no tiene dato. Mostrar el último valor
conocido sin marcarlo como viejo es peor que no mostrar nada.

---

### PRIORIDAD 2 — Conectar las señales muertas

#### D174-07 — Segmentar `weather_flag` e `irp_*` en el shadow book

Con D174-03 ya construido, esto es añadir dos predicados. Coste marginal casi nulo, y convierte
Nodo-96 y Nodo-113 de "trabajo hecho sin valor" a "hipótesis acumulando".

**No añadir gates.** Ambas siguen REPORTE_SOLO hasta que sus H-XX alcancen `n_stop`.

#### D174-08 — Implementar `outcome_id` en el edge_report (D154-06 real)

**Archivo:** `edge_calculator.py`, dict de salida por pick (~`:1560`).

El `outcome_id` ya viaja en el registro h2h tras el ledger (Nodo-118/143). Propagarlo:

```python
'outcome_id': registro_h2h.get('outcome_id'),
```

Luego, en los tres builders, **preferir `outcome_id` cuando esté presente** y caer al
name-matching solo si falta. Registrar la tasa de uso: `outcome_id_hit_rate` por corrida.

**Impacto esperado:** elimina el impuesto diario de name-matching y su fragilidad. Métrica de
éxito: `outcome_id_hit_rate ≥ 0.80` a los 7 días.

**Estado de implementación (2026-08-06): COMPLETO en los 3 builders del texto + 1 gap adicional encontrado y cerrado.**

`edge_calculator.py` propaga `outcome_id` en cada pick de salida (`apostar`/`watchlist`).
Los 3 builders nombrados en el spec preferían `outcome_id_hint` en `find_outcome()`/`_find_outcome()`
antes de caer a name-matching, con contador global `_OUTCOME_ID_STATS` (`hint_used`/`name_matched`/
`no_match`) y `_outcome_id_hit_rate()` en `betplay_combo_builder.py`:

- `betplay_combo_builder.py` — 8/8 call sites de `find_outcome()` con hint, 7/7 puntos de
  propagación de `outcome_id`, telemetría en `main()`. 5 tests REGLA-T53.
- `combo_confianza_builder.py` — 4 call sites de `_find_outcome(..., outcome_id_hint=_hint)`
  (líneas 976, 1724, 1792, 1847), `_hint` leído de `combo.get('outcome_ids')` (campo plural del
  combo, construido internamente desde el edge_report). Tests verificados sin regresión.
- `favoritos_combo_builder.py` — no tiene matcher propio, reusa `find_outcome()` importado de
  `betplay_combo_builder.py` en 2 sitios de `main()` (pre-filtro Kambi, resolución de outcome_ids
  del combo). `seleccionar_favoritos()`/`armar_combos()` ya preservaban `outcome_id` vía spread
  `**pick`, sin whitelisting — solo hicieron falta los 2 puntos de invocación del matcher. 2 tests
  REGLA-T53 nuevos.

**Gap encontrado durante el cierre — 4º generador fuera del texto literal ("los tres builders")
pero con el mismo patrón exacto de fuga:** `trader_ev_tenis.py::_build_cobertura()` construye
`all_plan` con legs whitelisteados a solo `{'jugador', 'cuota'}` (L871-872 antes del fix),
strippeando `outcome_id` aunque cada `l` del pool ya lo trae (viene directo de
`reporte.get('apostar', [])`, el mismo edge_report ya arreglado). `cobertura_plan` se escribe al
`trader_plan_*.json` bajo la clave `"cobertura"` (L1415), y el consumidor real no es
`combo_confianza_builder.py` sino `betplay_combo_builder.py::build_combo_links()` (:373-395,
lee `trader_plan.get("cobertura", [])` y llama `find_outcome()` por cada leg) — ese call site
nunca pasaba `outcome_id_hint`. Fix de 2 líneas: `trader_ev_tenis.py` L871-872 ahora incluye
`'outcome_id': l.get('outcome_id')` por leg; `build_combo_links()` L392-395 ahora pasa
`outcome_id_hint=leg.get("outcome_id")`. El contador `_OUTCOME_ID_STATS` se beneficia
automáticamente (vive dentro de `find_outcome()`, no en el call site). 2 tests REGLA-T53 nuevos
(`test_174_08_build_combo_links_pasa_outcome_id_hint_desde_leg`,
`test_174_08_build_combo_links_leg_sin_outcome_id_hint_es_none`).

Suite completa verificada tras cada builder cerrado — 0 regresiones (único fallo: el preexistente
conocido `test_nodo42.py::test_t42_07_superficie_filter_excluye_clay`, sin relación).

#### D174-09 — Ejecutar `audit_phantom_history.py`

Correrlo sobre los 30 días de shadow book y **reportar el resultado**. Si encuentra
contaminaciones, cada resultado afectado debe marcarse en `calibracion_edge.json` para que no
siga envenenando los priors (es dinero real ya perdido alimentando el modelo con la etiqueta
equivocada).

Si no encuentra nada, documentarlo — un negativo verificado también cierra la deuda.

Luego: integrarlo como PASO semanal en `run_daily.py --fase noche` o retirarlo formalmente.
**Un script de auditoría que nunca corre no es una defensa, es documentación.**

**Estado de implementación (2026-08-06): COMPLETO.**

Ejecución sobre 30 días de shadow book (`reports/audit_phantom_history_20260806_142232.json`):
36 jugadores con historial contaminado (thf_cache asignó historial top-ATP/GS a jugadores
ITF/Challenger por matching de apellido sin validar circuito, Nodo-152), de los cuales 3
tienen hits reales en `reports/shadow_book/sb_*.jsonl`: 1 sin resultado aún (Antoni Kasperski),
2 con `resolucion.resultado=LOST` y `pnl_flat_1u=-1.0` — dinero real perdido con edge calculado
sobre historial fantasma:

- **Ariana Morris** (`match_key="kha_morris"`, `sb_2026-07-08.jsonl`) — superficie=hard, tier=itf.
- **Alexander Weis** (`match_key="ravel_weis"`, `sb_2026-07-11.jsonl`) — superficie=clay, tier=itf.

Corrección aplicada a `data/calibracion_edge.json` (backup previo en
`data/calibracion_edge.json.bak_d174_09_20260806`, nota `_nota_d174_09_phantom_correction`
documenta el detalle): decrementados los `losses` agregados contaminados por estos 2 casos —
`global` (-2), `por_superficie.hard`/`clay` (-1 c/u), `por_superficie_y_tier.hard_itf`/`clay_itf`
(-1 c/u, incluyendo `era_v2_losses` porque ambos settlements ocurrieron después de
`era_v2_start`). El caso sin resultado (Kasperski) no requiere corrección — no ha alimentado
priors todavía.

Integración semanal: PASO 3.8 en `run_daily.py`, insertado entre PASO 3.7 (Dual-Book Router,
Nodo-111) y el `return` de fin de fase noche — corre `scripts/audit_phantom_history.py --days 30`
con guard `datetime.now().weekday() == 0` (solo lunes, mismo día que `check_contradictions.py`
por convención existente en el proyecto) y `optional=True` (no bloquea el pipeline nocturno si
falla, mismo patrón que PASO 3K/D141-02 y PASO 3.9/D154-08). El reporte se genera pero la
corrección de `calibracion_edge.json` sigue siendo **manual** — un hallazgo de contaminación
no se auto-aplica a los priors sin revisión humana (riesgo de falso positivo en
`_validate_circuit_consistency()` sobre casos límite). 2 tests REGLA-T53 nuevos en
`tests/test_nodo174_hypothesis_ledger.py` (inspección de fuente, patrón D141-02):
`test_174_09_run_daily_tiene_paso_audit_phantom_semanal`,
`test_174_09_paso_audit_phantom_es_optional` — 2/2 PASS.

---

### PRIORIDAD 3 — Cobertura y limpieza

#### D174-10 — Tests para las 3 estrategias sin cobertura

`build_system_combos` (8 tests, la deuda declarada de Nodo-156-B), `build_safe_combos`,
`build_was_combos`. REGLA-T53. Mínimo por estrategia:
- respeta REGLA-HF-1 (cuota ≥ 1.50),
- respeta REGLA-BAT-1 (coupon comma-joined, `||replace`, sin `|ML/`),
- retorna `[]` sin crash cuando el pool es insuficiente (fail-loud, patrón D172-01),
- filtra por `kambi_disponible`,
- no reutiliza `outcome_id` entre piernas.

**Estado de implementación (2026-08-06): COMPLETO.**

Auditoría previa al test confirmó el gap real: `build_system_combos` (Nodo-156-B)
y `build_was_combos` (Nodo-44) ya tenían guard explícito REGLA-HF-1
(`cuota < 1.50: continue` / gate `cuota_favorito >= 2.0` dentro de
`_was_qualifies()`), pero **`build_safe_combos` (Nodo-25) no lo tenía** —
única de las 3 sin defensa contra un pick heavy-fav colándose en el pool.
Fix aplicado antes de escribir el test correspondiente (2 líneas, mismo
patrón que `build_system_combos`, `betplay_combo_builder.py` dentro del loop
de construcción del pool en `build_safe_combos`):

```python
if p.get("cuota", 0) < 1.50:  # REGLA-HF-1 — nunca en pool (Nodo-174 D174-10)
    continue
```

18 tests REGLA-T53 nuevos en `tests/test_nodo174_10_combos_sin_cobertura.py`
(6 por estrategia: pool insuficiente→`[]` sin crash, REGLA-HF-1, filtro
`kambi_disponible`, no reutilización de `outcome_id`, formato coupon
REGLA-BAT-1, más un guard extra por función — Leave-One-Out matemático en
`build_system_combos`, "sin señal Markov no califica" en `build_was_combos`,
Guard 2 torneos-distintos en `build_safe_combos`). Patrón de mock reutilizado
de `test_nodo172_ancla_segura.py` (`patch.object` sobre `_find_latest_edge_report`/
`fetch_kambi_outcomes`/`find_outcome`); `build_safe_combos` requirió además
`monkeypatch.chdir(tmp_path)` porque lee `Path("reports").glob(...)` con ruta
relativa al cwd, y un `edge_report` con `torneo` distinto por jugador — sin
eso, todos los picks de un mismo `trader_plan` caen en el mismo fallback
`f"{tier}_{superficie}"` y el Guard 2 (torneos distintos) excluye todos los
pares (2 tests fallaron en el primer intento por esta razón, corregido
ajustando el fixture, no el código de producción). 18/18 PASS.

#### D174-11 — Telemetría del gate Kambi

Complementa [[Nodo-173]] D173-10. Añadir a `metadata` del edge_report:

```python
'kambi': {'n_disponibles': int, 'n_no_disponibles': int, 'n_desconocido': int}
```

63% de exclusión merece una línea en el reporte diario, no un descubrimiento por auditoría.

**Estado de implementación (2026-08-06): COMPLETO.**

`graphify query "metadata kambi_disponible n_disponibles telemetria edge_report"`
confirmó el gap real (REGLA-GRAPHIFY-FIRST): el campo por-pick `kambi_disponible`
ya existía desde D90-01 (`_annotate_kambi()`, `edge_calculator.py:917-931`), pero
no había ningún agregado en `metadata` — el % de exclusión solo era visible
contando manualmente sobre `resultados`, exactamente el "descubrimiento por
auditoría" que el nodo pide evitar.

Fix en `edge_calculator.py`, mismo patrón de agregación que D173-01
(`_funnel`), insertado justo antes de construir el dict `output`:

```python
# ── D174-11 (Nodo-174): telemetría del gate Kambi ────────────────────────
_kambi_disp = sum(1 for _r in resultados if _r.get('kambi_disponible') is True)
_kambi_no_disp = sum(1 for _r in resultados if _r.get('kambi_disponible') is False)
_kambi_desconocido = len(resultados) - _kambi_disp - _kambi_no_disp
_kambi_telemetry = {
    'n_disponibles':    _kambi_disp,
    'n_no_disponibles': _kambi_no_disp,
    'n_desconocido':    _kambi_desconocido,
}
```

y añadido a `metadata` junto a `funnel`:

```python
    'funnel':         _funnel,
    # D174-11: telemetría del gate Kambi — {n_disponibles,n_no_disponibles,n_desconocido}
    'kambi':          _kambi_telemetry,
```

`n_desconocido` cubre el caso `_kambi_coverage_cache` vacío (sin cobertura
Betplay cargada ese día) — `_annotate_kambi()` retorna `None` en ese caso, no
`False`, para no confundir "sin datos" con "confirmado no disponible".

4 tests REGLA-T53 nuevos en `tests/test_nodo174_11_kambi_telemetry.py`:
`test_174_11_sin_coverage_todo_desconocido` (cache vacío → todo cae en
`n_desconocido`), `test_174_11_disponible_y_no_disponible_se_cuentan_por_separado`
(favorito en `players_normalized` → disponible, favorito ausente →
no_disponible), `test_174_11_invariante_suma_igual_procesados` (suma de los 3
contadores == `metadata.n_procesados`, mismo invariante que el funnel D173-01),
`test_174_11_campo_kambi_disponible_en_cada_resultado_individual` (el campo
por-pick D90-01 sigue presente sin cambios — la telemetría es un agregado
nuevo, no un reemplazo). 4/4 PASS aislado; verificado además sin regresiones
con suite targeted `test_edge_calculator.py` + `test_nodo92_d90_01.py` +
`test_nodo94_sprint3.py` + `test_nodo173_calibracion.py` +
`test_nodo174_11_kambi_telemetry.py` + `test_nodo163_tier_gap_superficie_bridge.py`
→ 147/147 PASS.

#### D174-12 — Resolver módulos huérfanos

`scripts/backfill_evaluar_shadow.py` y `scripts/nodo_pagerank.py`: **decidir explícitamente** —
conectar (con su PASO en `run_daily.py`) o retirar (con nota en el nodo correspondiente).
No dejarlos en el limbo. `backfill_strategy.py` es one-shot cumplido: marcar como tal en su
docstring.

**Estado de implementación (2026-08-06): COMPLETO.**

`graphify query "backfill_evaluar_shadow nodo_pagerank run_daily orquestador huerfano"`
confirmó el gap real: ambos scripts existen como nodos aislados en el grafo
(sin arista `CALLS` entrante desde `run_daily.py`), y `grep -rln` sobre todo
el repo confirma cero referencias fuera de sí mismos. Decisión explícita para
cada uno, documentada como addendum en su Nodo de origen (no en código, no
son bugs):

- **`scripts/backfill_evaluar_shadow.py`** (D124-05, [[Nodo-124]]) — **RETIRAR**
  de huérfano. Su propio docstring ("Recupera picks EVALUAR históricos...
  Uso: `--fecha YYYY-MM-DD`") ya lo declara herramienta de recuperación
  retroactiva sobre una fecha puntual, no un PASO diario — mismo patrón que
  `scripts/audit_phantom_history.py` ([[Nodo-152]] D152-06). Conectarlo a
  `run_daily.py` sería trabajo redundante: `apostar`/`watchlist` ya se loguean
  en shadow_book durante el pipeline normal, solo `sin_edge` necesita backfill,
  y reprocesar el mismo rango de fechas cada día no aporta nada nuevo tras la
  primera pasada. Addendum §7 insertado en `Nodo-124-EvalTracker-TablaFavoritos-ShadowBook.md`.
- **`scripts/nodo_pagerank.py`** (D105-03, [[Nodo-105]]) — **RETIRAR** de
  huérfano. Herramienta de mantenimiento del vault `.spec/` (PageRank sobre
  wikilinks entre Nodos), no del pipeline de trading — se ejecuta manualmente
  al auditar salud documental, igual que `graphify update .`. Mezclarlo con
  `run_daily.py` confundiría higiene documental con orquestación de apuestas.
  Addendum §5 insertado en `Nodo-105-Knowledge-Graph-Navigation-Zettelkasten.md`.
- **`scripts/backfill_strategy.py`** (D144-06, [[Nodo-144]]) — docstring
  actualizado con nota "ONE-SHOT CUMPLIDO (Nodo-174 D174-12)": ejecutado una
  vez sobre los 3 días con combo_registry disponible (22/23/25-jul-2026), no
  es un PASO recurrente, reejecutar sobre el mismo rango es no-op.

Sin cambios de código en producción (solo docstrings/specs) — no requiere
tests REGLA-T53 nuevos, no hay comportamiento nuevo que verificar.

#### D174-13 — Regla anti-regresión: "✅" exige evidencia

Extender `check_contradictions.py` (cron lunes) con un **Bloque D**: para cada deliverable marcado
`✅` en CLAUDE.md que nombre una función o un campo, verificar que ese símbolo **existe en el
código**. Reportar los que no.

Esta auditoría encontró dos (`_calc_hcuc_convergence`, `outcome_id`) sin buscarlos
específicamente. Es razonable suponer que hay más.

##### Addendum — Estado de implementación (2026-08-06): COMPLETO

Implementado como **Bloque E** (no Bloque D) en `check_contradictions.py` — la letra D ya
la ocupa el ritual de huérfanos existente (D105-05). Documentado en un comentario en el
propio código para que la colisión de nomenclatura entre el spec y el código no se repita
como confusión en una sesión futura.

Nueva función pura `_check_simbolos_verificables()`: para cada fila de tabla de CLAUDE.md
marcada `| ... | ✅ |` (scope acotado — no la narrativa libre, mismo patrón de
precisión-sobre-recall que `_check_huerfanos()`), extrae símbolos de código que nombra —
funciones `nombre(...)` vía regex que exige guión bajo en el identificador (evita falsos
positivos de `str(`/`len(`), y campos `snake_case` bare vía regex que excluye tokens
envueltos en backticks/precedidos de punto o slash (evita capturar `edge_calculator.py:1575`
como si fuera un campo). Verifica cada uno contra el código real: funciones vía
`def nombre(` en algún `.py` de producción, campos vía heurístico de substring
(`'nombre'`/`.nombre` en el código concatenado).

Dos bugs encontrados y corregidos durante la implementación:
1. **Rendimiento** — el primer intento escaneaba `venv/` completo (16,079 archivos .py,
   timeout >90s). Fix: `_EXCLUDE_DIRS` filtra `.spec/graphify-out/tests/.git/venv/.venv/
   env/node_modules/__pycache__`. Corrida real tras el fix: ~27-33s.
2. **Falsos positivos de campo por mención de módulo en prosa** — filas de CLAUDE.md que
   mencionan un archivo sin backticks ni `.py` (ej. "games_signal universo distinto al
   edge_calculator", filas B9/G5 de la tabla D154) se capturaban como si "games_signal"/
   "edge_calculator" fueran campos de dato reclamados. Fix: excluir cualquier token que
   sea, o prefije con guión bajo, el `stem` de algún archivo `.py` real del repo.

Verificación real contra el CLAUDE.md y código de producción del repo (no un fixture):
`[PASS] Todos los símbolos marcados ✅ tienen evidencia en código.` — 0 símbolos sin
evidencia, exit 0. 8 tests REGLA-T53 en `tests/test_nodo174_13_simbolos_verificables.py`
(7 con fixtures temporales aislados vía monkeypatch de `CLAUDE_MD`/`BASE_DIR` + 1 de
regresión de alcance completo contra el repo real) — 8/8 PASS.

Limitación conocida y aceptada (documentada en el docstring de la función): el chequeo de
campo es un heurístico de substring, no prueba que el campo esté en el archivo/línea
específico que CLAUDE.md reclama — confirma ausencia con certeza, presencia con alta
probabilidad, no con prueba formal. Suficiente para el objetivo del deliverable: atrapar
casos como `_calc_hcuc_convergence`/`outcome_id` (ausencia total), no auditar precisión
línea por línea.

#### D174-14 — Rectificar CLAUDE.md

Corregir las entradas desmentidas por esta auditoría:
- Nodo-155 D155-02 → `PENDIENTE` (no implementado, nunca commiteado)
- Nodo-154 D154-06 → `PENDIENTE` (nunca implementado)
- Nodo-155 "11 tests 11/11 PASS" → el módulo no colecciona
- Conteos de suite desde Nodo-155 → marcar como no verificados hasta D174-02
- §5 "Nodos completos" → Nodo-155 sale de la lista hasta que D174-01 cierre

**Esto no es cosmético.** CLAUDE.md es la vista que orienta cada sesión futura; entradas falsas
hacen que Sonnet asuma infraestructura que no existe, que es exactamente cómo se llegó a que
`shadow_book` lea un campo que nadie produce.

##### Addendum — Estado de implementación (2026-08-06): COMPLETO (reconciliado, no "PENDIENTE")

El texto original de este deliverable asumía que Nodo-155 D155-02/Nodo-154 D154-06 seguirían
sin implementar al momento de rectificar CLAUDE.md. No fue así: **D174-01** (esta misma
auditoría) implementó `_calc_hcuc_convergence()` de verdad en `edge_calculator.py:1003`, y
**D174-08** implementó la propagación real de `outcome_id` en `edge_calculator.py:1462`. Por lo
tanto la corrección correcta no es marcar esas dos entradas `PENDIENTE` — sería falso en sentido
contrario, la misma clase de error que esta auditoría existe para prevenir — sino documentar la
reconciliación: lo que la auditoría original encontró roto, un deliverable posterior de este
mismo Nodo lo cerró.

Cambios reales aplicados a CLAUDE.md:
1. Header (línea 3, cifra `"2609 tests totales"`) → reemplazado por el baseline real verificado
   en esta sesión tras cerrar el único fallo pendiente del suite completo (ver triaje abajo):
   **2699 passed, 0 failed, 3 skipped** (`python -m pytest tests/ --no-cov -q`, corrida completa
   324.28s, sin overlap con cambios de este Nodo — el fallo cerrado era un test pre-existente
   no relacionado con D174-01..13).
2. Nodo-155 D155-02 / Nodo-154 D154-06 (narrativa del header y tabla §5) — **NO se marcan
   PENDIENTE**: ambas ya son ciertas hoy, verificadas por lectura directa del código
   (`_calc_hcuc_convergence` línea 1003, `outcome_id` línea 1462 de `edge_calculator.py`), no
   por confianza en el texto viejo.
3. Se agrega esta entrada de Nodo-174 al header de CLAUDE.md documentando el cierre de los 14
   deliverables.

**Triaje D174-02 de un hallazgo adicional durante esta verificación final:** el suite completo
tenía exactamente 1 fallo real (no "pre-existentes sin relación" sin enumerar — prohibido por
este mismo Nodo): `tests/test_nodo42.py::test_t42_07_superficie_filter_excluye_clay`. Categoría
(i) — test mal, no código mal. Causa: `_extract_and_categorize()` llama internamente a
`_load_edge_report_index()`, que lee el `edge_report_*.json` real de producción del día sin
mockear; el fixture ficticio del test usaba el apellido `"Ghetu"`, que coincidió por azar con un
jugador real del edge_report de hoy con `apostar=False` — el gate G1 de Nodo-103
(`_apply_combo_gates`) lo bloqueó del pool esperado por esa colisión, no por el gate EV_LEG_MIN
de D143-01 que el comentario original del test citaba incorrectamente (cálculo real:
`ev_leg=0.51*2.10=1.071≥1.02`, ese gate nunca bloqueaba este fixture). Fix: aislar también
`_load_edge_report_index` vía `monkeypatch.setattr('combo_confianza_builder._load_edge_report_index',
lambda: {})`, mismo patrón ya usado en el test para aislar `load_coverage` (gate Kambi D140-04);
corregidos los 3 comentarios que atribuían la causa al gate equivocado. Verificado:
`tests/test_nodo42.py` → 8 passed, 1 skipped. Riesgo estructural anotado para auditoría futura
(fuera de alcance de este triaje puntual): cualquier otro test que dependa de
`_extract_and_categorize()`/`_load_edge_report_index()` sin mockear tiene el mismo riesgo de
no-determinismo por colisión de nombres con datos reales de producción.

---

## 8. Orden de ejecución

```
D174-01  reparar coleccion pytest          ← BLOQUEANTE de todo lo demas
D174-02  baseline honesto del suite
    ↓
D174-03  hypothesis_ledger + PASO 10e
D174-04  18 predicados faltantes
D174-05  pre-registrar las 4 huerfanas
D174-06  live_desk deja de mentir
    ↓ PUERTA: set(PREDICADOS) == set(declaradas). n_actual se mueve solo tras un settle.
D174-07  segmentar weather + irp
D174-08  outcome_id real (D154-06)
D174-09  ejecutar audit_phantom_history
    ↓
D174-10  tests de las 3 estrategias
D174-11  telemetria gate Kambi
D174-12  huerfanos: conectar o retirar
D174-13  check_contradictions Bloque D
D174-14  rectificar CLAUDE.md
```

**Relación con [[Nodo-173]]:** son complementarios y el orden entre ellos importa poco, salvo por
D174-01 y D174-02, que deben ir **antes que cualquier fase de Nodo-173** — sin un suite que corra,
las puertas de calidad de Nodo-173 no son verificables.

---

## 9. Tests

**`tests/test_nodo174_ledger.py`**
1. `test_174_01` — `set(PREDICADOS) == set(hipótesis declaradas)`. Falla si alguien declara sin
   predicado o mide sin declarar. **Es el test que evita que A1/A2 se repitan.**
2. `test_174_02` — `contar_hipotesis()` sobre records sintéticos devuelve n/hits exactos.
3. `test_174_03` — `actualizar_registro(dry_run=True)` no escribe.
4. `test_174_04` — `actualizar_registro()` **nunca** modifica `umbrales_congelados` ni
   `preregistrado`, aunque se le pasen valores nuevos.
5. `test_174_05` — cada `H-XX` con `n>=n_stop` recibe veredicto SPRT.

**`tests/test_nodo174_integridad.py`**
6. `test_174_06` — para cada campo que `shadow_book.py` lee de `pick_snapshot` en un segmento de
   hipótesis, existe un productor en `edge_calculator.py`. **Es el test que habría atrapado B1.**
7. `test_174_07` — `outcome_id` presente en el dict de salida de `calcular_edge_completo()`.
8. `test_174_08` — `live_desk.py` no contiene literales `"n_actual": <int>` distintos de 0.

**`tests/test_nodo174_estrategias.py`** — los 8+ de D174-10 (ver §7 D174-10 para el mínimo).

---

## 10. Lo que esta auditoría dice del proceso

Tres hallazgos comparten una firma: **fueron declarados completos sin que existiera el código**
(`_calc_hcuc_convergence`, `outcome_id`, los 8 tests de Nodo-156-B). Y uno de ellos —
`_calc_hcuc_convergence` — tenía tests escritos que lo probaban, tests que **nunca pudieron
correr** porque el módulo no importaba.

La causa raíz no es descuido: es que **la verificación se hizo por archivo de test individual, no
por suite completa**, y el conteo global se copió de la sesión anterior. Un `pytest tests/` que
aborta en colección devuelve exit code 2 sin listar nada — es fácil de leer mal si uno espera ver
fallos y ve un error.

D174-13 (verificar que los `✅` correspondan a símbolos reales) y D174-02 (enumerar los fallos
pre-existentes uno por uno, prohibido agregarlos) son las dos correcciones de proceso que impiden
la recurrencia. Recomiendo tratarlas como parte de `PRE_IMPLEMENTATION_CHECKLIST.md`, no como
deliverables de un solo nodo.

---

## 11. Wikilinks

[[Nodo-173]] (embudo de decisión — este nodo cierra el lazo que aquel abre) ·
[[Nodo-155]] (D155-02 nunca implementado — origen de B1 y C1) ·
[[Nodo-154]] (D154-06 nunca implementado — origen de B2; mismo patrón de auditoría) ·
[[Nodo-156-B]] (8 tests declarados pendientes, nunca escritos) ·
[[Nodo-152]] (D152-06 `audit_phantom_history.py` nunca ejecutado) ·
[[Nodo-96]] (IRP — señal producida, cero consumidores) ·
[[Nodo-113]] (weather — señal producida, cero consumidores) ·
[[Nodo-98]] (meta_score ausente del reporte) ·
[[Nodo-101]] (picks live — base del predicado de H165/H166) ·
[[Nodo-51]] (F5 `hypothesis_tracker.py` — la pieza que existe y funciona; le falta el contador) ·
[[Nodo-86]] / [[Nodo-99]] / [[Nodo-119]] (auditorías previas — mismo método) ·
[[Nodo-78]] (6 reglas de auditoría SDD: *"nunca ejecutado" = hallazgo*, aplicada aquí a
`audit_phantom_history.py`)
