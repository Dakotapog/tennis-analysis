# Nodo-75 — nodos_index.json: Índice de Trazabilidad SDD

**Fecha:** 2026-07-09  
**Rama:** main  
**Estado:** BORRADOR — esperando confirmación antes de generar el índice  
**Cierra:** gap detectado en auditoría 2026-07-09 (combo_governor.py, combo_registry.py sin mención en Nodo)

---

## Problema

El barrido de auditoría (2026-07-09) encontró 7 archivos `.py` sin mención en ningún Nodo:

```
HUERFANO: ./combo_registry.py
HUERFANO: ./extraer_ranking_atp_version2.py
HUERFANO: ./extraer_ranking_wta_version2.py
HUERFANO: ./Intelligent_ml_enhancer.py
HUERFANO: ./n8n_push_workflow.py
HUERFANO: ./session_compiler.py
HUERFANO: ./utils/feature_engineering.py
```

El script `check_contradictions.py` solo compara headers de Nodo-XX.md contra CLAUDE.md. No detecta:
- Archivos de código sin Nodo correspondiente
- Nodos que mencionan archivos que ya no existen (nombre cambiado, eliminado)

El resultado: violaciones SDD pasan desapercibidas hasta que alguien pregunta manualmente.

---

## Solución

### D75-A: `nodos_index.json`

Archivo JSON generado automáticamente desde `.spec/01_Nodos/Nodo-*.md`. Estructura:

```json
{
  "generated_at": "2026-07-09T14:00:00",
  "nodos": [
    {
      "id": "Nodo-74",
      "file": ".spec/01_Nodos/Nodo-74-Combo-Governor.md",
      "estado": "BORRADOR",
      "archivos_mencionados": ["combo_governor.py"],
      "fecha": "2026-07-09"
    }
  ],
  "huerfanos": [
    "combo_registry.py",
    "n8n_push_workflow.py"
  ]
}
```

El campo `huerfanos` se calcula en el momento de la generación: archivos `.py` en el proyecto que no aparecen en ningún Nodo.

### D75-B: Sincronización — Opción B (aprobada)

Chequeo de desactualización integrado en `check_contradictions.py` (ya aprobado por el proyecto para correr semanalmente):

```python
# Al inicio de check_contradictions.py
idx_path = BASE_DIR / "nodos_index.json"
if idx_path.exists():
    newest_nodo = max(SPEC_DIR.glob("Nodo-*.md"), key=lambda f: f.stat().st_mtime)
    if newest_nodo.stat().st_mtime > idx_path.stat().st_mtime:
        print(f"WARNING: nodos_index.json desactualizado.")
        print(f"  Nodo más reciente: {newest_nodo.name}")
        print(f"  Correr: python3 scripts/rebuild_nodos_index.py")
```

**Por qué Opción B sobre hook pre-commit:** los hooks de git son locales y no se versionan sin un `install-hooks.sh` separado. La verificación en `check_contradictions.py` va en el repo, corre el lunes vía cron, y es visible para cualquier sesión futura de Claude Code. Trade-off aceptado: hasta 6 días de índice desactualizado entre lunes — tolerable dado el volumen de 1-2 nodos/semana.

### D75-C: Script `scripts/rebuild_nodos_index.py`

Script de regeneración (~40 líneas):
- Lee todos `Nodo-*.md`
- Extrae: id, estado (regex header), archivos mencionados (grep `\.py`)
- Lista archivos `.py` del proyecto (excluyendo tests/, venv/, _*)
- Calcula `huerfanos` = py_files - mencionados_en_algun_nodo
- Escribe `nodos_index.json`

**Lista de exclusiones conocidas** (archivos cuyo nombre difiere del Nodo que los documenta):
```python
_EXCLUSIONES = {
    # archivo.py → Nodo que lo documenta (para no marcarlo huérfano)
    "rivalry_analyzer.py":  "Nodo-32",   # motor principal, mencionado indirectamente
    "edge_calculator.py":   "Nodo-35",
    "trader_ev_tenis.py":   "Nodo-55",
    "shadow_book.py":       "Nodo-27",
    "combo_confianza_builder.py": "Nodo-62",
    "betplay_combo_builder.py":   "Nodo-26",
    "extraer_URL_partidos_version2.py": "Nodo-51",
    "extraer_historh2h.py": "Nodo-49",
}
```

Esta lista se construyó con los resultados del barrido real de auditoría + verificación manual en DECISION-LOG.md.

---

## Archivos

| Archivo | Rol |
|---|---|
| `nodos_index.json` | Índice generado (en raíz del proyecto) |
| `scripts/rebuild_nodos_index.py` | Generador (~40 líneas) |
| `check_contradictions.py` | Ya existente — recibe chequeo de stale (Bloque B ya implementado) |

---

## Tests

| Test | Caso | Esperado |
|---|---|---|
| T75-01 | Generar índice con Nodos reales | JSON válido con todos los Nodos |
| T75-02 | combo_governor.py mencionado → no huérfano | `huerfanos` no incluye combo_governor.py |
| T75-03 | Archivo sin Nodo → aparece en huérfanos | `huerfanos` lo incluye |
| T75-04 | Nodo más reciente > índice → check_contradictions emite WARNING | WARNING presente en stdout |
| T75-05 | Índice actualizado → check_contradictions no emite WARNING | sin WARNING de stale |

---

## Limitaciones conocidas

- El parser de "archivos mencionados" usa grep de `\.py` en el texto del Nodo. Puede tener falsos negativos si el Nodo menciona el módulo sin la extensión (ej. "combo_governor" sin ".py"). La lista `_EXCLUSIONES` cubre los casos conocidos.
- El índice NO reemplaza la lectura de los Nodos individuales — es un índice de trazabilidad, no un resumen de contenido.
- Nodos retroactivos (Nodo-74) con `Estado: BORRADOR` aparecerán en el índice con ese estado — correcto.

---

## Auditoría de cierre

- [ ] `nodos_index.json` generado y válido JSON
- [ ] T75-01 a T75-05 pasan
- [ ] `check_contradictions.py` emite WARNING cuando índice está desactualizado
- [ ] 7 archivos huérfanos del barrido 2026-07-09 aparecen en `nodos_index.json["huerfanos"]`
- [ ] Cron semanal del lunes detecta índice stale si se agrega un Nodo sin regenerar
