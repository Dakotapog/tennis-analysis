# Nodo-76 — Combo Registry: P&L por tipo de combo

**Fecha:** 2026-07-09 (retroactivo — archivo creado en commit 1d7565c)
**Rama:** main
**Estado:** BORRADOR — esperando confirmación
**Cierra:** huérfano SDD + conflicto de numeración detectados en auditoría 2026-07-09

---

## Problema

`combo_registry.py` tenía en su docstring `"Nodo-64: Combo P&L Registry"`. Pero
**Nodo-64 en el spec es SPRT** (`hypothesis_tracker.py`). Conflicto de numeración:
el archivo reclamaba un número ya ocupado, lo que impedía que cualquier grep de
trazabilidad lo encontrara correctamente.

Consecuencia: ningún `Nodo-*.md` lo mencionaba con el número correcto → huérfano
en el índice SDD.

Este Nodo-76 es el número correcto retroactivo. El docstring de `combo_registry.py`
fue actualizado de "Nodo-64" a "Nodo-76" en este mismo commit.

---

## Arquitectura real

`combo_registry.py` registra y settlea combos multi-pierna ejecutados.
Es el análogo del shadow_book pero para combos (no picks individuales).

```
combo_confianza_builder / betplay_combo_builder
    │  genera picks de combo
    ▼
ComboRegistry.log_combo(fecha, tipo, piernas, stake)
    │  escribe en reports/combo_registry/YYYY-MM-DD.jsonl (append-only)
    ▼
ComboRegistry.settle_date(fecha)
    │  lee output de resultados_finales.py
    │  verifica cada pierna (ganado/perdido)
    │  escribe resolución por combo
    ▼
ComboRegistry.report([fecha])
    │  P&L histórico por tipo: CORE, SAT, MOONSHOT, mega, safe, etc.
```

Sin imports del modelo de predicción — dependencia cero del motor de Kelly/EV.

---

## Archivos

| Archivo | Rol |
|---|---|
| `combo_registry.py` | Implementación (log + settle + report + normalización nombres) |
| `reports/combo_registry/YYYY-MM-DD.jsonl` | Registro append-only por día |

---

## Validación en producción

```
Comando: ls reports/combo_registry/ 2>/dev/null
Output:  directorio no existe aún — 0 datos reales (2026-07-09)
```

El módulo existe pero nunca ha sido invocado en producción. No está integrado
en `run_daily.py`. No hay datos históricos de combos settlados via este registry.

---

## Tests

```
Comando: grep -rn "ComboRegistry\|combo_registry" tests/test_nodo63.py
Output:  no encontrado en test_nodo63.py

Comando: grep -n "combo_registry" .spec/01_Nodos/Nodo-64*.md
Output:  Nodo-64 no menciona combo_registry ✅ (confirmado: no hay colisión activa)
```

No existen tests específicos de ComboRegistry. El commit 1d7565c los anuncia en el
mensaje pero no los incluye para este módulo. Tests T76-XX pendientes.

| Test | Caso | Esperado |
|---|---|---|
| T76-01 | log_combo → archivo JSONL creado | archivo existe con entrada correcta |
| T76-02 | settle_date con resultados fixture | ganado/perdido correcto por pierna |
| T76-03 | report() sin datos | no crashea, retorna string vacío o "sin datos" |
| T76-04 | _names_match normalización | "Rodríguez" == "rodriguez" == "RODRIGUEZ" |

---

## Limitaciones conocidas

- Docstring original decía "Nodo-64" — corregido a "Nodo-76" en commit de este Nodo.
- No está integrado en `run_daily.py` — requiere llamada manual.
- No hay tests de integración reales (reports/combo_registry/ vacío).
- `combo_governor.py` (Nodo-74) no depende de ComboRegistry — lee archivos
  combo_plan_*.txt directamente. Son módulos paralelos, no en cadena.

---

## Auditoría de cierre

- [x] Nodo-64.md no menciona combo_registry (verificado: no hay colisión activa)
- [x] tests/test_nodo63.py no incluye tests de ComboRegistry (0 tests específicos)
- [x] reports/combo_registry/ no existe aún (0 datos reales — módulo sin invocar)
- [x] Docstring de combo_registry.py corregido de "Nodo-64" a "Nodo-76"
- [ ] T76-01 a T76-04 implementados y pasando
- [ ] Integración con run_daily.py (o decisión explícita de no integrar)
