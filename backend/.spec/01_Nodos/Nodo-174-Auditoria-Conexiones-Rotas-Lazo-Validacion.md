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

#### D174-09 — Ejecutar `audit_phantom_history.py`

Correrlo sobre los 30 días de shadow book y **reportar el resultado**. Si encuentra
contaminaciones, cada resultado afectado debe marcarse en `calibracion_edge.json` para que no
siga envenenando los priors (es dinero real ya perdido alimentando el modelo con la etiqueta
equivocada).

Si no encuentra nada, documentarlo — un negativo verificado también cierra la deuda.

Luego: integrarlo como PASO semanal en `run_daily.py --fase noche` o retirarlo formalmente.
**Un script de auditoría que nunca corre no es una defensa, es documentación.**

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

#### D174-11 — Telemetría del gate Kambi

Complementa [[Nodo-173]] D173-10. Añadir a `metadata` del edge_report:

```python
'kambi': {'n_disponibles': int, 'n_no_disponibles': int, 'n_desconocido': int}
```

63% de exclusión merece una línea en el reporte diario, no un descubrimiento por auditoría.

#### D174-12 — Resolver módulos huérfanos

`scripts/backfill_evaluar_shadow.py` y `scripts/nodo_pagerank.py`: **decidir explícitamente** —
conectar (con su PASO en `run_daily.py`) o retirar (con nota en el nodo correspondiente).
No dejarlos en el limbo. `backfill_strategy.py` es one-shot cumplido: marcar como tal en su
docstring.

#### D174-13 — Regla anti-regresión: "✅" exige evidencia

Extender `check_contradictions.py` (cron lunes) con un **Bloque D**: para cada deliverable marcado
`✅` en CLAUDE.md que nombre una función o un campo, verificar que ese símbolo **existe en el
código**. Reportar los que no.

Esta auditoría encontró dos (`_calc_hcuc_convergence`, `outcome_id`) sin buscarlos
específicamente. Es razonable suponer que hay más.

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
