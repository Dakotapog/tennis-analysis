# Nodo-110 — MODO OPERADOR: Favoritos Compuestos (la estrategia #13 que el usuario ya validó con dinero real)

> **Wikilinks:** [[Nodo-107-Riesgo-Agregado-Motor-Reconciliacion]] | [[Nodo-100-Taxonomia-Estrategias-Generacion-Combos]] | [[Nodo-90-Auditoria-Fable-Nodo89]] | [[Nodo-55-Respuesta-Fable-Funnel-Deploy]] | [[Nodo-117-Auditoria-Scraping-Rankings-Cobertura-H2H]]
> **Fecha:** 2026-07-17 | **Autor:** Fable 5 | **Prioridad:** MÁXIMA — es el dolor persistente #1 del proyecto
> **Evidencia de origen (dinero real, adjunta por el operador):** 8/8 combos ganados (jul-14 y jul-16), cuotas 3.84-6.51x, stakes $600-680, pago total ~$27,500. Construidos A MANO desde el output de tabla_favoritos mientras el sistema entregaba 0 picks APOSTAR (jul-17: 120+ singles → 0 apuestas, 1 games combo).

---

## §1. EL DIAGNÓSTICO EN UNA FRASE
**El patrón ganador del operador está EXCLUIDO por diseño, tres veces, antes de poder existir:**

| Excluidor | Por qué mata el patrón |
|---|---|
| Puerta económica (edge>5%) | Favoritos claros @1.15-1.60 casi nunca muestran edge vs mercado → jamás son APOSTAR. El patrón del operador NO es un patrón de edge: es de **probabilidad conjunta alta** (4 favoritos p~0.75-0.85 → P(combo)~40-50% a cuota 4-6x = EV+) |
| REGLA-HF-1 (cuota<1.50 fuera del pool) | Las piernas ancla del operador (McNeil @1.23, Grubor @1.32, Galarneau @1.15, Batin @1.15) son ilegales en el pool. HF-1 nació del KGR de heavy-fav **como SINGLE** (-0.5085) — nunca se decidió para PIERNAS de combo. Precedente interno: las piernas VARIABLE @1.18-1.35 ya se apuestan en combos de confianza (D87-08) |
| N28F2/T33 + CAPA2 estrecha | ITF/W15 con n_h2h=0 y 1 eje → watchlist LOW. CAPA2 (H89-01) exige cuota≥1.50 Y n_h2h≥1 — deja fuera justo esta población |

Sus 8 combos comparten estructura reproducible: **3-4 piernas, favoritos multi-tier (ITF+Challenger+WTA), cuota pierna [1.15, 2.10], máx 1-2 piernas por torneo, cuota combinada 3.8-6.5x, stake ~$600-680**.

## §2. DECISIONES
- **D110-01:** Aclaración constitucional (requiere OK usuario en DECISION-LOG): REGLA-HF-1 aplica a SINGLES y al pool del trader; las piernas de combo tienen su propio piso `LEG_MIN_CUOTA=1.15`. No es relajación: es codificar lo que D87-08 ya practica y el operador ya validó.
- **D110-02:** Nueva estrategia #13 FAVORITOS COMPUESTOS en la taxonomía §11 — generador propio, governor la cuenta (extiende matriz 12/12→13/13 de D107-02).
- **D110-03:** Las apuestas del operador dejan de ser invisibles: todo combo generado se registra en shadow book + betslip_index (si él apuesta manual, la calibración lo captura — fin del bucket `?`).
- **D110-04:** H110-01 pre-registrada con las 8 observaciones reales como semilla (n_actual=8, hits=8 a nivel combo; ~30 piernas ganadas).

## §3. SPEC PARA SONNET — `favoritos_combo_builder.py` (nuevo)
**Selección (función pura `seleccionar_favoritos(edge_report) -> list`):** universo = apostar+watchlist+sin_edge. Filtros: (1) seguridad SIEMPRE: sin NO_DATA, sin phantom, sin historial_incompleto; (2) favorito claro: `p_modelo >= 0.62` O (`cuota_favorito <= 1.40` Y `confidence_flag != LOW`) O (`ranking_favorito` mejor por >300 puestos Y `cuota <= 1.60`); (3) cuota ∈ [1.15, 2.10]; (4) el favorito del modelo = favorito del bookmaker (cuota_fav < cuota_rival) — este patrón NO caza upsets. Ordenar por p_modelo desc.

**Armado (`armar_combos(picks) -> list`):** 3-4 piernas; máx 2 por torneo y máx 1 por jugador (exposición D107-03); cuota combinada objetivo [3.5, 7.0]; generar top-3 combos con solape ≤2 piernas entre sí (como hace el operador: variaciones del mismo núcleo). Stake: $650 fijo por combo, tope sesión $2,000 para la estrategia, governor con veto (Nodo-107) lo suma ANTES de emitir.

**Output (Zero-Null D90-04):** SIEMPRE emite — si hoy no hay 3 piernas válidas, imprime cuántas pasaron cada filtro y cuál falta. Links Betplay + .bat (reusar `find_outcome`/`generar_bat_chrome` de betplay_combo_builder). Registro automático: shadow book (pick_snapshot por pierna, `estrategia=FAVORITOS_COMPUESTOS`) + betslip_index.

**Integración:** `run_daily.py` tras PASO 4.3. Tests REGLA-T53 (~8): los 4 filtros de selección; diversificación por torneo; solape ≤2; fixture con el edge_report del 2026-07-16 → debe reproducir ≥3 de las piernas reales ganadas (Gaines/McNeil/Forbes/Bynoe si están en el reporte); governor BLOCK → no emite.

**H110-01 (JSON, tras OK del usuario):** predicción: combos de 3-4 favoritos seleccionados por esta función logran hit% combo ≥ 25% (breakeven @4.0 media) — semilla retrospectiva 8/8 jul-14/16; n_stop=30 combos prospectivos; kill-switch hit%<15% con n≥15 → OFF.

**PROHIBIDO:** tocar HF-1 para singles, tocar gates del edge_calculator, escalar stake >$650 antes de graduación.

**Criterio de éxito:** correr con los datos de HOY (jul-17, 120+ singles) → ≥1 combo emitido con piernas nombradas, o el desglose exacto de por qué no — nunca más "no hay nada" con 120 partidos sobre la mesa.

## §4. 🚨 CRÍTICO — Bug Fix Nodo-110: El Loader Roto (2026-07-17)

**Diagnóstico ejecutado:** "no hay nada" con 120 partidos en la mesa el 2026-07-17 02:30 UTC. Sonnet reportó "próxima ventana fuerte mañana". **FALSO.** El problema no era ausencia de señal — era un loader completamente roto + pipeline alimentado a medias.

### Causa raíz: `_leer_edge_report()` línea 423, favoritos_combo_builder.py

**ANTES (roto):**
```python
if isinstance(data, list):
    return data
return data.get("picks", data.get("results", []))  # ← JAMÁS existe en real edge_report.json
```

**El schema REAL de edge_report.json:**
```json
{
  "apostar": [...],           # candidatos con edge>threshold
  "watchlist": [...],         # edge marginal o baja confianza
  "sin_edge": [...],          # picks sin edge (pero registro en shadow book)
  "no_data": [...]            # poblaciones sin H2H
}
```

Sonnet (implementador de Nodo-110) nunca testeó `_leer_edge_report` contra un edge_report REAL. El código buscaba claves "picks"/"results" que NO EXISTEN. Resultado: `_leer_edge_report()` retornaba `[]` siempre, universo vacío, ningún combo generado. Silenciosamente.

### Solución (APLICADA, FIX Nodo-110):
```python
# FIX Nodo-110 (Fable 2026-07-17): el schema real del edge_report es
# {apostar:[], watchlist:[], sin_edge:[]} — sin este merge el universo
# quedaba SIEMPRE vacío ("0 piernas" con 120 partidos sobre la mesa).
if any(k in data for k in ("apostar", "watchlist", "sin_edge")):
    return ((data.get("apostar") or [])
            + (data.get("watchlist") or [])
            + (data.get("sin_edge") or []))
return data.get("picks", data.get("results", []))  # ← fallback para esquemas heredados
```

### Evidencia de impacto:
- **ANTES:** `universo = []` → dry-run con 120+ singles → "0 piernas" → "0 apuestas" → operador construye combos A MANO.
- **DESPUÉS:** `universo = 15 candidatos` (de 120+) → filtra a 1 pierna válida para combo (mínimo 3 requeridas) → explica POR QUÉ no hay combo ("2 piernas faltantes, mejor edge mañana").
- **Análisis profundo:** El edge_calculator genera solo ~25 picks de 120+ partidos (20% coverage). NO es que haya "día sin señal" — es que el pipeline mismo ejecuta a media potencia. Problema separado: ver `run_daily.py --fase noche` para pre-extraction.

### Test REGLA-T53 (REQUERIDO):

Nueva prueba en `tests/test_nodo110_favoritos_builder.py`:

```python
def test_leer_edge_report_merge_trois_listas():
    """REGLA-T53: _leer_edge_report DEBE leer el schema real y merguear las 3 listas."""
    import tempfile
    import json
    from favoritos_combo_builder import _leer_edge_report
    
    # Fixture: edge_report con schema correcto
    edge_report = {
        "apostar": [
            {"favorito_predicho": "Gaines", "cuota_favorito": 1.23, "confianza": "STRONG"},
        ],
        "watchlist": [
            {"favorito_predicho": "McNeil", "cuota_favorito": 1.32, "confianza": "MOD"},
        ],
        "sin_edge": [
            {"favorito_predicho": "Forbes", "cuota_favorito": 1.45, "confianza": "LOW"},
        ],
        "no_data": []
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(edge_report, f)
        f.flush()
        
        result = _leer_edge_report(f.name)
        
    # Assertion: len(result) == len(apostar) + len(watchlist) + len(sin_edge)
    assert len(result) == 3, f"esperado 3, obtuve {len(result)}"
    nombres = [p.get("favorito_predicho") for p in result]
    assert "Gaines" in nombres
    assert "McNeil" in nombres
    assert "Forbes" in nombres
    
    # Test edge case: si todas las listas están vacías, retorna []
    edge_report_vacío = {"apostar": [], "watchlist": [], "sin_edge": [], "no_data": []}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(edge_report_vacío, f)
        f.flush()
        
        result = _leer_edge_report(f.name)
        
    assert len(result) == 0, f"vacío: esperado 0, obtuve {len(result)}"
```

**Propósito:** Garantizar que el contrato interno entre edge_calculator y favoritos_combo_builder nunca se rompa por confusión de esquema. El loader es invisible pero CRÍTICO — si falla, el universo desaparece.

### Diagnóstico de Sonnet (refutado):

> "la próxima ventana fuerte es mañana — hoy sin divergencia de modelo"

**FALSO.** No fue ausencia de divergencia. Fue un loader que retornaba 0 candidatos independientemente de los datos. La divergencia estaba en los datos, pero nunca llegó a la función de selección.

---

## §4. ADDENDUM 2026-07-17 — Dimensión real del alcance (evidencia adicional del operador)
Dos combos más, mismo patrón a mayor escala:
- **jun-23, Wimbledon qualy femenino: 8 piernas, @99.94x (222.9 con nula), $500 → $49,970.** 7 ganadas + 1 nula. Piernas @1.29-2.48 + una @4.90 (Jorge). TODAS del mismo evento (qualy GS) — la diversificación fue por PARTIDO, no por torneo.
- **jun-19, ITF femenino: 5 piernas @11.19x, $500 → $5,596.** Piernas @1.45-1.76, 4 torneos ITF distintos.

**Implicaciones para la spec (D110-05, variante MEGA-OPERADOR):**
1. El patrón escala a 5-8 piernas cuando hay un evento denso (qualy GS = 32+ partidos same-day) — relajar "máx 2 por torneo" a "máx 2 por torneo SALVO qualy GS/eventos ≥16 partidos, donde el límite es por partido".
2. El rango de pierna se extiende: núcleo [1.15, 2.10] + hasta 2 piernas "spice" [2.10, 5.00] SOLO si son watchlist con p_modelo≥0.55 o rival_value_flag (la Jorge @4.90 y Mikulskyte @2.48 fueron el multiplicador).
3. Segmento propio en H110-01: `FAVORITOS_COMPUESTOS_MEGA` (5-8 piernas) separado del núcleo (3-4) — n_stop y kill-switch independientes; stake fijo $500, jamás escalar por euforia (el 99x es cola derecha, no esperanza).
4. Población WTA/GS-qualy femenino entra explícitamente al universo (las 3 evidencias mayores son femeninas — serializar `circuito` en el registro para medir si el alpha es específico WTA/ITF-F).

## §5. D110-06 — UNIVERSO EXTENDIDO: el patrón no necesita H2H (2026-07-17 noche)

**Evidencia (corrida nocturna para 2026-07-18):** 66 partidos extraídos → solo 20 con H2H (30%) → edge_report: 0 apostar, 9 watchlist. El fix §4 funciona (los 9 watchlist SÍ entran al universo), pero **46 partidos con cuotas Kambi reales se descartan en el cruce FlashScore de PASO 2, antes de cualquier análisis**. El cuello ya no es el loader — es que edge_report solo contiene la minoría de partidos que consiguió match_id.

**Insight:** el patrón del operador (8/8 con dinero real) se construyó SIN H2H profundo — con ranking + cuota desde tabla_favoritos. El propio filtro (3) de §3 ya lo admite: `ranking_favorito mejor por >300 puestos Y cuota ≤ 1.60`. Obligar a la estrategia #13 a pasar por edge_report es alimentarla con el 30% de su universo natural.

**D110-06 (spec para Sonnet):** `favoritos_combo_builder.py --matches <zita_tennis_matches_*.json>`:
1. Partidos del archivo PASO 1 SIN entrada en edge_report entran al universo con `fuente=RANKING_ONLY` si cumplen TODO: (a) ranking gap > 300 puestos (leer rankings ATP/WTA ya extraídos en PASO 0 — cero red nueva), (b) `cuota_favorito ∈ [1.15, 1.60]` (más estricto que el rango núcleo: sin modelo, exigimos favorito más claro), (c) favorito por ranking = favorito del book (cuota menor).
2. Salvaguardas: máx 2 piernas RANKING_ONLY por combo (el ancla del combo siempre lleva ≥1 pierna con análisis completo si existe); campo `fuente` serializado en shadow book.
3. Segmento separado en H110-01: `FAVORITOS_RANKING_ONLY` — medir si estas piernas diluyen o suman ANTES de igualarlas al núcleo. Kill-switch independiente: hit% pierna <55% con n≥20 → OFF (una pierna @1.15-1.60 necesita ~65-87% para pagar).
4. Tests REGLA-T53 (~4): fixture zita con ranking gap → pasa/no pasa cada uno de los 3 criterios; combo nunca lleva >2 piernas RANKING_ONLY; partido presente en edge_report NUNCA se duplica desde --matches.
5. **PROHIBIDO:** que RANKING_ONLY entre a ninguna otra estrategia (1-12); que el trader o los gates lean estas piernas; relajar el gap <300 antes de graduar el segmento.

**Por qué es seguro:** no toca edge_calculator ni gates; amplía solo el universo del builder #13, con stake fijo $650, tope sesión $2,000 y governor 13/13 contando. Es codificar lo que el operador ya hace a mano con tabla_favoritos — pero registrado, medido y con kill-switch.
