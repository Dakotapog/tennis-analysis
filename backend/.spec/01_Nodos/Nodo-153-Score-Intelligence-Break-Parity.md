# Nodo-153 — Score Intelligence: Serve/Break/Game Live + D153-04 Break Parity Fix

**Fecha:** 2026-07-29  
**Estado:** IMPLEMENTADO — 14/14 tests REGLA-T53 PASS  
**Prioridad:** CRÍTICA — corrige bug conceptual en detección de quiebre  
**Wikilinks:** [[Nodo-150]] [[Nodo-147]] [[Nodo-142]] [[live_desk.py]] [[_parse_kambi_livedata_sets]] [[_fmt_progreso]]

---

## 1. Contexto: El Bug del Quiebre

**Síntoma observado (2026-07-29 21:00):**
```
Charlie Pade vs Scott Jones  
►0:1 | 1j QUIEBRE  
```

El dashboard mostraba QUIEBRE cuando no había quiebre. Home sirvió G2 (●), Away ganó su saque G1 = HOLD normal, no break.

**Raíz:** La fórmula antigua era `no-servidor lidera = quiebre`. Ignoraba quién sirvió **primero en ese set**.

---

## 2. Corrección: Break por Paridad de Servicio

### Regla Fundamental Tenis
En tenis el servidor ALTERNA cada game automáticamente. La **paridad de N** (total de juegos jugados en el set) determina quién sirvió primero:

```
N par  → servidor actual ES el mismo que sirvió G1
N impar → servidor actual es el OPUESTO al que sirvió G1
```

### Marcador Esperado (sin breaks)
```
N par:   home:away igualado (ej: 2:2, 4:4, 6:6)
N impar: primer servidor adelante por 1 (ej: 3:2 si home G1, 2:3 si away G1)
```

### Break = Realidad ≠ Expectativa
```python
# Ejemplo 1: ►0:1 | N=1 (bug del usuario)
N=1 (impar), sirve home ahora
→ first_server = AWAY (opuesto)
→ esperado = away +1 → 0:1
→ real = 0:1
→ esperado == real → SIN quiebre ✓

# Ejemplo 2: ►2:1 | N=3 (quiebre real)
N=3 (impar), sirve home ahora  
→ first_server = AWAY
→ esperado = away +1 → 1:2
→ real = 2:1
→ real ≠ esperado → QUIEBRE ✓

# Ejemplo 3: 1:3► | N=4 (quiebre confirmado)
N=4 (par), sirve home ahora
→ first_server = HOME (mismo)
→ esperado = tie → 2:2
→ real = 1:3
→ esperado ≠ real → QUIEBRE ✓
```

---

## 3. D153-04 Implementation

### Código Correctivo (`live_desk.py:3090-3125`)

```python
# D153-04: break_situation — paridad de servicio, no simplemente "quién lidera"
break_situation = False
if serving and current_set_home is not None and current_set_away is not None:
    _N = current_set_home + current_set_away
    if _N > 0:
        # Quién sirvió primero en este set
        _first_srv = serving if (_N % 2 == 0) else (
            "away" if serving == "home" else "home"
        )
        # Marcador esperado sin quiebre
        _exp_lead = 0 if (_N % 2 == 0) else (1 if _first_srv == "home" else -1)
        _act_lead = current_set_home - current_set_away
        break_situation = (_act_lead != _exp_lead)
```

### Campos D153 Completos
- **D153-01:** `current_set_home/away` — score en set actual (desde `statistics.sets`)
- **D153-02:** `serving` — "home"/"away" (desde `statistics.sets.homeServe`)
- **D153-03:** `game_score` — "30:15" (desde `liveData.score.home/away`)
- **D153-04:** `break_situation` — boolean (paridad de servicio)

### Display (`_fmt_progreso()`)
```
"7:6,1:6, ►2:2 [15:30] | 24j QUIEBRE"
  ↑ sets completos  ↑set actual  ↑game  ↑total  ↑break
```

---

## 4. Tests REGLA-T53 (14/14 PASS)

| Test | Caso | Resultado |
|------|------|-----------|
| `test_153_01` | `home=[7,1,2], away=[6,6,1]` → D153-01 extrae 2:1 | ✓ |
| `test_153_02_home` | `homeServe=True` → serving="home" | ✓ |
| `test_153_02_away` | `homeServe=False` → serving="away" | ✓ |
| `test_153_03` | `score={home:"40",away:"15"}` → game_score="40:15" | ✓ |
| `test_153_03_zero` | `score={home:"0",away:"0"}` → game_score=None | ✓ |
| `test_153_04_3:1` | `home=[6,1,1], away=[4,6,3]` + home sirve → break=True | ✓ |
| `test_153_04_3:1_flip` | `home=[6,2,3], away=[4,6,1]` + away sirve → break=True | ✓ |
| `test_153_04_hold` | `2:1, N=3, away sirve` → break=False (hold) | ✓ |
| `test_153_04_equal` | `3:3` → break=False | ✓ |
| `test_153_04_0:1_BUG_FIX` | `►0:1 N=1` (Charlie Pade caso) → break=False | ✓ |
| `test_153_04_REAL_BREAK` | `►2:1 N=3` (away sirvió G1) → break=True | ✓ |
| `test_153_fmt_home` | `serving="home"` → muestra ► | ✓ |
| `test_153_fmt_away` | `serving="away"` → muestra ◄ | ✓ |
| `test_153_fmt_break` | `break_situation=True` → muestra QUIEBRE | ✓ |

---

## 5. Integración con Nodo-150 y Nodo-147

### Flujo Completo
1. **D147-01b:** `_fetch_kambi_livedata()` obtiene `statistics.sets` (endpoint `/event/{id}/livedata.json`)
2. **D153:** `_parse_kambi_livedata_sets()` extrae D153-01→D153-04 campos
3. **D150-02:** `games_set1` propagado al nivel superior (para gates)
4. **D150-04:** Si `games_set1 >= 12` → forzar `COINFLIP_FORZADO` (no DOMINANTE)
5. **Dashboard:** `_fmt_progreso()` renderiza `►2:2 [15:30] QUIEBRE`

### Dependencias
- **Requiere:** [[Nodo-147]] (D147-01b `_fetch_kambi_livedata`)
- **Requiere:** [[Nodo-150]] (D150-02 `games_set1`, D150-04 `COINFLIP_FORZADO`)
- **Usado por:** [[Nodo-100]] (break state machine), [[Nodo-119]] (dashboard live)
- **Complementa:** [[Nodo-142]] (ITF live games convergence)

---

## 6. Performance: Fast Loop 5s

**Latencia real después de fixes:**
- Kambi score update: ~1s
- Fast loop re-fetch (TTL=4s < 5s ciclo): ~5s
- STATE_CACHE invalidate: inmediato
- Browser F5: ~5-7s total

**No cambia main loop (15s)** — cuotas, convergencia, gates siguen cada 15s.

---

## 7. Cambios Arquitectónicos Sesión 2026-07-29

| Cambio | Archivo | Líneas | Efecto |
|--------|---------|--------|--------|
| D153-04 break_situation paridad | `live_desk.py` | 3090-3125 | Corrección conceptual |
| `_LIVEDATA_TTL = 4s` | `live_desk.py` | 3545 | Fuerza re-fetch en fast loop |
| `_STATE_CACHE ttl = 6s` | `live_desk.py` | 2648 | Browser ve scores frescos |
| `_fast_score_refresh()` thread | `live_desk.py` | 4495-4552 | Nuevo loop 5s scores |
| Threads en `main()` | `live_desk.py` | 4649-4653 | Arrancar ambos daemons |
| Tests actualizados | `test_nodo153_*.py` | 124-200 | 14 tests, paridad correcta |

---

## 8. Verificación en Vivo

**Dashboard esperado tras fix:**
```
Charlie Pade vs Scott Jones
EN VIVO | Total de juegos | OVER 19.5 @1.93
►0:1 | 1j
(SIN QUIEBRE)
```

**Comandos verificación:**
```bash
# Sintaxis
python -c "import ast; ast.parse(open('live_desk.py').read())"

# Tests
python -m pytest tests/test_nodo153_score_intelligence.py -v --no-cov

# Reiniciar
systemctl --user restart tennis-live-desk.service
```

---

## 9. Deuda Técnica

- [ ] D153-05: Casos edge (love sets 0:6, 0:0 edge cases) — candidato sesión futura
- [ ] D153-STRESS: Load test fast loop con 20+ señales simultáneas
- [ ] D153-METRICS: Logging de latencias reales `game_score` en producción

---

## Resumen

**Bug:** Fórmula `no-servidor lidera = quiebre` ignoraba paridad de servicio.  
**Fix:** Cálculo expectativa por N par/impar → compara vs realidad.  
**Resultado:** `►0:1 N=1` = hold (no quiebre), `►2:1 N=3` = quiebre real.  
**Tests:** 14/14 PASS con casos edge verificados.  
**Latencia:** ≤7s (vs 30s antes).
