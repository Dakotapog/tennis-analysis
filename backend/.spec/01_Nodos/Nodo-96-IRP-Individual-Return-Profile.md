# Nodo-96 — IRP: Individual Return-from-inactivity Profile

> **Wikilinks:** [[Nodo-64-RFI-Return-From-Inactivity]] | [[Nodo-57-Penalizacion-Inactividad]] | [[Nodo-93-Sprint2-PlayerDB]] | [[Nodo-89-Sistema-Inteligencia-Integral]] | [[Nodo-90-Auditoria-Fable-Nodo89]]
> **Fecha:** 2026-07-14 | **Sprint:** 5 (Fable5 roadmap Nodo-90 §5)
> **Principio:** REPORTE_SOLO — no modifica edge, kelly ni ninguna decisión de apuesta.

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
| D96-06 | `build_irp_profiles.py` se ejecuta como PASO 0b en run_daily (después de rankings, antes del scraping) |
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
# PASO 0b — IRP profiles (Nodo-96, solo si player_db existe)
python3 scripts/build_irp_profiles.py
```
Añadir después del PASO 0 (rankings), antes del PASO 1 (scraping).

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
    "novak djokovic": "Novak_Djokovic"
  }
}
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

`tests/test_nodo96_irp.py` — 8 tests mínimos, todos invocando funciones reales del módulo.
Ver implementación en `scripts/build_irp_profiles.py`.
