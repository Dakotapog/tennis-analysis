# Nodo-96 — IRP: Individual Return-from-inactivity Profile

> **Wikilinks:** [[Nodo-64-RFI-Return-From-Inactivity]] | [[Nodo-57-Penalizacion-Inactividad-Campeon-Validacion]] | [[Nodo-93-Sprint2-Implementado]] | [[Nodo-89-Sistema-Inteligencia-Integral]] | [[Nodo-90-Auditoria-Fable-Nodo89]]
> **Fecha:** 2026-07-14 | **Sprint:** 5 (Fable5 roadmap Nodo-90 §5)
> **Principio:** REPORTE_SOLO — no modifica edge, kelly ni ninguna decisión de apuesta.
> **Auditoría:** 2026-07-14 — ver §8. 15 tests passing. 2 hallazgos reales (H_IRP + apellido fallback). 1 falso positivo del revisor inicial corregido.

---

## 1. PROBLEMA

El RFI actual (D64-01, Nodo-64) aplica penalización de form_decay **uniforme para todos los jugadores**:
los mismos umbrales (90d/180d/365d) y la misma curva `exp(-0.025*(days-30))` independientemente del
historial individual. Pero los jugadores responden diferente a la inactividad:

- Un jugador con historial de 80% win-rate al volver de gaps > 30d no merece la misma penalización
  que uno que cae a 20% en los mismos escenarios.
- La calibración actual no puede capturar esto — es agregada por tier, no por jugador.

IRP mide empíricamente, para cada jugador en PlayerDB, cómo rinde en "partidos de retorno"
(primer match después de un gap > 30d) vs su rendimiento normal.

---

## 2. DECISIONES

| ID | Decisión |
|---|---|
| D96-01 | IRP es REPORTE_SOLO — `irp_fav`/`irp_rival` se serializan en el pick del edge_report pero NO tocan edge, kelly_kl, p_modelo ni ningún gate de apuesta |
| D96-02 | Umbral de retorno: **30 días** (mismo inflection point de form_decay, Nodo-57). Gap > 30d entre matches consecutivos = "return match" |
| D96-03 | Solo jugadores con `n_retornos >= 2` entran al profile — un solo retorno no establece patrón |
| D96-04 | `delta_return = win_rate_return - win_rate_normal`. Negativo = rinde peor al volver. Positivo = rinde mejor |
| D96-05 | Gate de activación como señal real: n ≥ 30 jugadores con IRP activo en edge_report + hipótesis H_IRP pre-registrada. Hasta entonces, solo observar |
| D96-06 | `build_irp_profiles.py` se ejecuta como **PASO 0c** en run_daily (0a=ATP, 0b=WTA, 0c=IRP — el spec original decía "0b" por error; la implementación usa 0c, que es la numeración correcta) |
| D96-07 | Si `data/irp_profiles.json` no existe o player no encontrado → `irp_fav = {}` (silencioso, no bloquea pipeline) |

---

## 3. IMPLEMENTACIÓN

### 3.1 `scripts/build_irp_profiles.py`

Lee `data/player_db.json`. Para cada jugador:
1. Ordena `rows` por `fecha` ascendente
2. Calcula gap en días entre fechas consecutivas
3. Clasifica `return_match = True` si gap anterior > 30d
4. Computa stats separadas:
   - `n_retornos`: matches donde `return_match = True`
   - `win_rate_return`: `wins_en_retorno / n_retornos`
   - `win_rate_normal`: `wins_en_otros / n_otros`
   - `delta_return`: `win_rate_return - win_rate_normal`
   - `avg_gap_return`: promedio de gaps que precedieron los retornos
   - `last_match_fecha`: fecha del match más reciente en PlayerDB
   - `days_since_last`: `(build_date - last_match_fecha).days`

Output: `data/irp_profiles.json`
```json
{
  "built_at": "2026-07-14T...",
  "n_players_with_irp": 1240,
  "return_threshold_days": 30,
  "profiles": { "Slug": { ... } },
  "name_index": { "normalized name": "Slug" }
}
```

### 3.2 `edge_calculator.py`

Carga `data/irp_profiles.json` una vez al inicio del módulo (igual que `calibracion_edge.json`).
Función `_irp_lookup(nombre, irp_data)` — normaliza nombre → busca en `name_index` → retorna profile dict o `{}`.

Campos añadidos al pick:
```python
pick['irp_fav']   = _irp_lookup(nombre_favorito, _IRP_DATA)
pick['irp_rival'] = _irp_lookup(nombre_rival, _IRP_DATA)
```

### 3.3 `run_daily.py`

```bash
# PASO 0c — IRP profiles (Nodo-96, solo si player_db existe)
python3 scripts/build_irp_profiles.py
```
Se ejecuta después de PASO 0a (rankings ATP) y PASO 0b (rankings WTA), antes del PASO 1 (scraping).
**Corrección spec:** el doc original decía PASO 0b — error de anticipación. La implementación usa 0c (correcto).

---

## 4. SCHEMA `irp_profiles.json`

```json
{
  "built_at": "ISO datetime",
  "n_players_with_irp": 1240,
  "return_threshold_days": 30,
  "profiles": {
    "Novak_Djokovic": {
      "slug": "Novak_Djokovic",
      "n_matches": 150,
      "n_retornos": 12,
      "win_rate_return": 0.583,
      "win_rate_normal": 0.712,
      "delta_return": -0.129,
      "avg_gap_return": 54.2,
      "last_match_fecha": "2026-07-01",
      "days_since_last": 13
    }
  },
  "name_index": {
    "novak djokovic": "Novak_Djokovic",
    "djokovic": "Novak_Djokovic"
  }
}
```

**⚠️ Limitación de cobertura (Hallazgo H96-02):** la implementación original solo indexa el slug
completo normalizado. Si `edge_calculator` recibe `favored='Djokovic'` o `'N. Djokovic'`,
`_irp_lookup` retorna `{}` silenciosamente. Fix pendiente: agregar apellido (última palabra) como
key adicional en `name_index` — ver §8.2 y fix en `build_irp_profiles.py` L192.
```

---

## 5. PRECONDICIONES

- `data/player_db.json` debe existir (Sprint 2, Nodo-93). Si no existe, PASO 0b se salta silenciosamente.
- Baseline tests: 1969 passed antes de tocar código.

---

## 6. INVARIANTES REPORTE_SOLO

1. `irp_fav`/`irp_rival` en el pick son dicts (vacíos si no hay datos) — nunca modifican `edge`, `p_modelo`, `kelly_kl` ni `apostar`.
2. `build_irp_profiles.py` es idempotente — puede correr múltiples veces sin efecto secundario.
3. `data/irp_profiles.json` NO se commitea al repositorio (datos generados).
4. La promoción de IRP a señal real (afectar edge) requiere H_IRP pre-registrada + n≥30 jugadores con señal activa.

---

## 7. TESTS (REGLA-T53)

`tests/test_nodo96_irp.py` — **15 tests** (implementados, todos passing 2026-07-14).
Todos invocan funciones reales del módulo: `compute_player_irp`, `normalize_for_index`,
`build_irp_profiles`, `_irp_lookup` (importada de edge_calculator). Ninguno hardcodea fórmulas.

Cobertura:
- Cómputo básico de retornos (n_retornos, win_rate_return)
- win_rate_return vs win_rate_normal independientes
- delta_return negativo / positivo
- Jugadores sin retornos excluidos (n_retornos=0 y n_retornos=1)
- normalize_for_index: underscores, acentos, espacios
- _irp_lookup: encontrado, no encontrado, irp_data vacío
- build_irp_profiles end-to-end con PlayerDB sintético en tempdir
- days_since_last y avg_gap_return

**Gap de cobertura (pendiente):** ningún test verifica apellido-only lookup tras el fix H96-02.
Añadir en la sesión de fix.

---

## 8. AUDITORÍA 2026-07-14 (Sonnet 4.6 — revisión forense post-implementación)

### 8.1 Metodología

Revisión doble: subagente inicial + auditoría forense manual sobre código real
(`edge_calculator.py` L1159-1165, `build_irp_profiles.py`, `tests/test_nodo96_irp.py`,
`run_daily.py`, `validation/preregistered_hypotheses.json`).

### 8.2 Hallazgos

| ID | Hallazgo | Tipo | Severidad | Estado |
|---|---|---|---|---|
| H96-00 | "Bug inversión jugadores" (subagente inicial) | **FALSO POSITIVO** | — | Cerrado |
| H96-01 | H_IRP no pre-registrada en `preregistered_hypotheses.json` | REAL — deuda técnica | BAJA (REPORTE_SOLO) | **PENDIENTE FIX** |
| H96-02 | `name_index` sin apellido fallback — cobertura baja en producción | REAL — limitación de diseño | MEDIA | **PENDIENTE FIX** |
| H96-03 | PASO 0b vs 0c en spec original | REAL — error de documentación | BAJA | Corregido en §3.3 y D96-06 |
| H96-04 | Tests spec decía "8 tests mínimos" — implementación tiene 15 | REAL — subdocumentación | BAJA | Corregido en §7 |

### 8.3 Detalle H96-00 — FALSO POSITIVO

El subagente inicial reportó: *"Si el favorito es jugador2, irp_rival apunta al jugador incorrecto"*.

Verificación forense (`edge_calculator.py` L880-891 + L1163):
```python
if favored == jugador1:   → player_key_sb = 'player1'
elif favored == jugador2: → player_key_sb = 'player2'

_nombre_rival = jugador2 if player_key_sb == 'player1' else jugador1
# fav=j1 → key='player1' → rival=jugador2  ✓
# fav=j2 → key='player2' → rival=jugador1  ✓
```
La lógica es correcta en ambos casos. El subagente confundió la condición del ternario.

### 8.4 Detalle H96-01 — H_IRP no pre-registrada

`grep -c "H_IRP\|H96\|irp" validation/preregistered_hypotheses.json` → **0**

D96-05 exige H_IRP pre-registrada antes de escalar IRP a señal activa. La hipótesis aún no existe.
No bloquea el despliegue actual (REPORTE_SOLO), pero es deuda técnica obligatoria antes del gate.

**Fix:** añadir entrada en `validation/preregistered_hypotheses.json`:
```json
"H96-01": {
  "nombre": "IRP Individual Return Profile",
  "hipotesis": "Jugadores con delta_return < -0.10 tienen win_rate menor en partidos de retorno",
  "gate": "n >= 30 jugadores con irp_fav activo en edge_report",
  "estado": "ACUMULANDO",
  "fecha_registro": "2026-07-14"
}
```

### 8.5 Detalle H96-02 — Apellido fallback en name_index

`_irp_lookup` normaliza y hace lookup exacto en `name_index`. El name_index solo contiene
slugs completos normalizados. En producción, `favored` en edge_calculator proviene del H2H
y puede ser solo apellido (`'Djokovic'`) o inicial+apellido (`'N. Djokovic'`).

Tasa de miss estimada: alta en jugadores de circuito ITF/Challenger cuyos nombres en H2H
son abreviados. El fallo es silencioso (`{}`) — no bloquea, pero IRP no aporta datos reales.

**Fix:** en `build_irp_profiles.py` L192, además del slug completo, añadir apellido como key:
```python
_full = normalize_for_index(slug)
name_index[_full] = slug
_parts = _full.split()
if len(_parts) > 1:
    name_index[_parts[-1]] = slug   # apellido → mismo slug
```
Riesgo de colisión (dos jugadores con mismo apellido): el último en procesarse gana.
Mitigación: es REPORTE_SOLO — colisión = dato erróneo observacional, no decisión de apuesta.

### 8.6 Veredicto final

**GO para producción como REPORTE_SOLO.** Aplicar H96-01 y H96-02 antes de cerrar el nodo.
