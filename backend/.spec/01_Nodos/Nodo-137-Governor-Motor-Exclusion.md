# Nodo-137 — Governor: Exclusión MOTOR del Gate de Combos

**Fecha:** 2026-07-22
**Estado:** IMPLEMENTADO — commit `5437262`
**Wikilinks:** [[Nodo-100]] [[Nodo-107]] [[Nodo-138]] [[Nodo-65]] [[combo_governor.py]]

---

## 1. Diagnóstico

El `combo_governor.py` incluía el stake del MOTOR (`trader_ev_tenis.py`) en el cálculo
de exposición total comparado contra el budget de combos. Esto provocaba un BLOCK permanente:

```
MOTOR:          $65,000  (Kelly-KL + CPPI — gestión de riesgo propia)
Budget combos:  $15,000
Total contado: $65,000 > $15,000 → BLOCK en TODAS las estrategias
```

**Resultado:** 0 combos desplegados en toda sesión, imposibilitando validar las 12 estrategias.

---

## 2. Raíz del Problema

El MOTOR y los combos son **sistemas de riesgo independientes**:

| Sistema | Gestión de riesgo | Gate apropiado |
|---------|-------------------|----------------|
| MOTOR | Kelly-KL + VaR + CPPI | Propio (trader_ev_tenis.py) |
| Combos | Budget por sesión | combo_governor.py |

Sumar ambos en un único gate mezcla dos monedas diferentes. El MOTOR ya tiene
su propio VaR y CPPI — el governor de combos no debe interferir.

---

## 3. Fix — D137-01

**Archivo:** `combo_governor.py` — L359

**Antes:**
```python
all_stakes = {**stakes_motor, **stakes_confianza, **stakes_rival, **stakes_betplay}
total = sum(all_stakes.values())
```

**Después:**
```python
# D137-01: MOTOR excluido del gate de combos — tiene su propio Kelly-KL/VaR/CPPI.
all_stakes = {**stakes_confianza, **stakes_rival, **stakes_betplay}
total = sum(all_stakes.values())
total_motor = sum(stakes_motor.values())
```

El MOTOR se muestra como referencia con WARN si supera 40% bankroll, pero **no entra
en el cálculo del budget de combos**.

---

## 4. Fix colateral — rival_value_betslip.py

`rival_value_betslip.py` llamaba `combo_governor.py` como binario sin intérprete Python,
causando `OSError: [Errno 8] Exec format error` en WSL.

**Fix:** Prefijo `sys.executable` en `subprocess.run()` + import `sys` añadido.

```python
import subprocess as _sp, sys
_gov = _sp.run(
    [sys.executable, str(Path(__file__).parent / 'combo_governor.py'),
     '--bankroll', str(args.bankroll)],
    capture_output=True, text=True
)
```

---

## 5. Impacto

- Governor pasa de BLOCK permanente → permite combos cuando `total_combos ≤ budget`
- MOTOR sigue visible en output para auditoría de exposición total del bankroll
- `rival_value_betslip.py` puede llamar al governor sin crash en WSL

---

## 6. Archivos modificados

- `combo_governor.py` — D137-01: separar MOTOR de combo budget gate
- `rival_value_betslip.py` — fix OSError subprocess sin intérprete Python
