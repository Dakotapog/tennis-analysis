# Nodo-11: Inventario y Decisión de Scripts Legado

> **Wikilinks:** [[Inventario-Deuda-Tecnica]] | [[Pipeline-Arquitectura]] | [[Sprint-Pipeline]] | [[Nodo-12-Inventario-Infraestructura-Legado]]
> **Estado:** 2026-05-30 — CERRADO ✅ | Todas las decisiones ejecutadas y verificadas en disco
> **Metodología:** TTC + Marco de Tres Expertos (Mandato 9)

---

## Resumen de Decisiones Ejecutadas

| Archivo | Decisión | Estado disco | Verificado |
|---|---|---|---|
| `extraer_cuotas_partidos.py` | ELIMINAR | ✅ No existe | 2026-05-30 |
| `generar_tabla_favoritos.py` | ELIMINAR (D-13) | ✅ No existe | 2026-05-30 |
| `extraer_URL_partidos_en_vivo.py` | MANTENER + fix h2h_url | ✅ Existe | 2026-05-30 |
| `ml_trainer.py` | SUSPENDER | ✅ Existe | 2026-05-30 |
| `consultar_resultados_historicos.py` | MANTENER | ✅ Existe | 2026-05-30 |

**Disco y spec alineados al 100%.**

---

## Archivo 1: `extraer_cuotas_partidos.py` — ELIMINADO ✅

**Por qué:** Versión anterior de ZitaScraper. No extrae h2h_url, match_id ni superficie. Bug crítico en `remove_duplicates` (return dentro del for → retorna tras primer match). Totalmente supersedida por `extraer_URL_partidos_version2.py` con todos los fixes de Nodo-03. Cero importadores activos.

---

## Archivo 2: `generar_tabla_favoritos.py` (v1) — ELIMINADO ✅ (D-13)

**Por qué:** Reporte manual legacy. `generar_tabla_favoritos2.py` (v2) validada en producción 2026-05-29 y ejecutada exitosamente. v1 tiene bug `score_breakdown` variable scope que rompía el análisis de desglose. D-13 cumplida.

---

## Archivo 3: `extraer_URL_partidos_en_vivo.py` — MANTENER (fix pendiente T12-D)

**Qué hace:** `ZitaScraper` variante LIVE — filtra solo partidos con `event__match--live`. Propósito único: monitoreo en tiempo real de cuotas durante partidos activos.

**Fix pendiente (T12-D):**
```python
# INCORRECTO — formato antiguo:
match_data['h2h_url'] = f"https://www.flashscore.co/partido/tenis/{match_data['match_id']}/#/h2h/general"

# CORRECTO (aplicar fix Nodo-03):
match_data['h2h_url'] = match_url.split('?')[0] + '/#/h2h/overall/'
```
Ejecutar cuando se integre al pipeline LIVE (post validación P&L con n≥30).

---

## Archivo 4: `ml_trainer.py` — SUSPENDIDO

**Qué hace:** `RandomForestClassifier` + `LogisticRegression` básicos. Lógica anti-overfitting para datasets pequeños (<100 samples). Lee `ml_datasets/enhanced/` (actualmente vacío). No puede ejecutarse hasta que S8 (`generar_dataset_plus.py`) produzca datos limpios en producción.

**Valor único vs `aplicar_enhancer.py`:** anti-overfitting explícito para datasets pequeños — `aplicar_enhancer.py` no cubre este caso. Conservar para cuando S8 tenga datos post-Nodo-03.

**Acción futura:** Crear `Nodo-ML-Trainer` cuando S8 esté activo con n≥30 partidos limpios.

---

## Archivo 5: `consultar_resultados_historicos.py` — MANTENER

**Qué hace:** `ResultVerifier` — Playwright para extraer resultado final desde `match_url`. Fallback de Playwright para verificar resultados de **datos históricos** donde la API FlashScore ya no tiene el evento activo.

**Valor único vs `validar_con_api.py`:** funciona con cualquier `match_url` válida sin necesitar `match_id` real. Cubre datos pre-Nodo-03 (match_id="tennis") que la API no puede resolver.

---

## Tarea Pendiente

| Task | Qué | Cuándo |
|---|---|---|
| T12-D | Fix h2h_url en `extraer_URL_partidos_en_vivo.py` | Pre-integración pipeline LIVE |

---

## Vinculación

- [[Inventario-Deuda-Tecnica]] — D-13 (generar_tabla_favoritos v1) ✅ eliminado
- [[Nodo-12-Inventario-Infraestructura-Legado]] — auditoría infraestructura paralela (Flask/Selenium)
- [[Pipeline-Arquitectura]] — S8 pipeline ML (ml_trainer.py entra aquí)
- [[Sprint-Pipeline]] — T12-D en backlog
