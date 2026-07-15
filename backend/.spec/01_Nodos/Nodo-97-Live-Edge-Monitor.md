# Nodo-97 — Live Edge Monitor: seguimiento de cuotas en vivo para combos intrapartido

> **Wikilinks:** [[Nodo-90-Auditoria-Fable-Nodo89]] | [[Nodo-74-Combo-Governor]] | [[Nodo-73-n8n-CloseSnapshot-Timing]] | [[Nodo-65-Convergencia-Multi-Senal]] | [[Nodo-95-Sprint4-PatternRecognition]]
> **Fecha:** 2026-07-14 | **Autor:** Sonnet 4.6 (emergencia del pipeline — conexión oculta identificada)
> **Principio:** El mercado en vivo tarda 2-3 minutos en reaccionar. Si el modelo pre-partido dijo STRONG y la cuota empieza a bajar, hay una ventana de ~90s donde el edge sigue intacto y se puede apostar.

---

## 1. PROBLEMA

El sistema actualmente apuesta solo **pre-partido**. Pero existe una ineficiencia temporal
que el pipeline ya tiene todos los ingredientes para explotar:

1. El modelo dice "Boogaard gana, edge 22%, STRONG, HOT" a las 9am.
2. El partido empieza a las 2pm. La cuota es 3.55.
3. A los 5 minutos del primer set, Boogaard gana 3-0. El mercado aún no reaccionó — cuota 3.20.
4. Nuestro edge a 3.20 sigue siendo ~15%. Ventana de 60-90 segundos.
5. Ningún sistema alerta esto. El trader no lo ve. La ventana se cierra.

**El Live Edge Monitor cierra esa ventana.**

---

## 2. DECISIONES

| ID | Decisión |
|---|---|
| D97-01 | Polling Kambi cada 60s para picks STRONG pre-registrados del día — usando el cliente parametrizado de D90-08 (OddsAggregator) |
| D97-02 | Trigger de alerta: cuota actual ≤ cuota_pre_partido × 0.85 (bajó ≥15%) Y edge_live = (p_modelo - 1/cuota_live) > 0.05 |
| D97-03 | Alert vía Telegram: partido, cuota_pre, cuota_live, edge_live, señales activas (STRONG/HOT/IRP) |
| D97-04 | Combo sugerido automático: los picks STRONG+HOT con edge_live > 0 se agrupan en combo 2-3 patas |
| D97-05 | Output: `reports/live_edge_FECHA_HHMMSS.json` con snapshot de cada check |
| D97-06 | Ventana de monitoreo: desde 30min antes del partido hasta 45min después del inicio (primer set) |
| D97-07 | No apuesta automática — alerta humana siempre. La decisión final es del trader |
| D97-08 | Gate: solo picks con `confidence_flag=STRONG` o `markov_favorito=HOT` del edge_report del día |
| D97-13 | Shadow book para picks live: campo `pick_type='live'` + `cuota_trigger` como cuota de apertura para CLV. CLV_live = (cuota_trigger - cuota_cierre) / cuota_cierre — separado de CLV_pregame en dashboard (D99-02) |
| D97-14 | Live combo respeta Combo Governor (Nodo-74): si `governor.budget_restante > 0` → stake Kelly live shrink 5%; si budget=0 → stake=$0 + banner "OBSERVACION — budget diario agotado". Live combo es ADICIONAL solo si KGR>0 Y budget>0 (D99-06) |

---

## 3. ARQUITECTURA

### 3.1 `scripts/live_edge_monitor.py`

```python
# Flujo principal:
# 1. Leer edge_report del día → filtrar STRONG o HOT
# 2. Para cada pick activo (partido en ventana horaria):
#    a. Fetch cuota live desde Kambi (cliente D90-08)
#    b. Calcular edge_live = p_modelo - 1/cuota_live
#    c. Calcular drift = (cuota_pre - cuota_live) / cuota_pre
#    d. Si drift >= 0.15 AND edge_live > 0.05 → TRIGGER
# 3. Si TRIGGER: construir combo_live + enviar Telegram
# 4. Escribir snapshot en reports/live_edge_*.json
```

### 3.2 Integración n8n (PRIMARIO — cron independiente = FALLBACK)

**D99-11:** n8n es la arquitectura primaria. El workflow de Nodo-73 ya tiene cron, retry logic y bridge :8765 activos.
Añadir live monitor como nodo adicional en el MISMO workflow n8n:
cron cada 60s entre 9am-11pm → POST a bridge `:8765/live-check` → ejecuta monitor.
Fallback: `close_snapshot_trigger.py`-style cron (`*/1 9-23 * * *`) si n8n cae — misma lógica que Nodo-73.

### 3.3 Telegram alert format

```
🎾 LIVE EDGE DETECTADO
Boogaard vs Onclin | Challenger
Pre-partido: 3.55 → Live: 2.90 (drift -18.3%)
Edge live: +12.4% | Señales: STRONG + HOT + IRP_OK
p_modelo: 0.432 vs p_implícita: 0.345

COMBO SUGERIDO:
• Boogaard 2.90
• Dodig 2.80
Kelly sugerido: $850 total (shrink 5%)
```

---

## 4. CONEXIÓN CON HERRAMIENTAS EXISTENTES

| Herramienta | Rol en Live Edge |
|---|---|
| `edge_report_FECHA.json` | Fuente de p_modelo + picks STRONG/HOT |
| `OddsAggregator._fetch_kambi()` (D90-08) | Fetch cuota live — ya parametrizado |
| `betplay_combo_builder.py --live` | Construir combo con cuotas live actuales |
| n8n workflow | Orquestar el cron de polling |
| Telegram bot | Alert al trader |
| `shadow_book.py` | Log automático del pick live para CLV tracking |

**Cero infraestructura nueva** — solo conectar piezas existentes.

---

## 5. OUTPUT EVIDENCIA (obligatorio, D97-05)

`reports/live_edge_YYYYMMDD_HHMMSS.json`:
```json
{
  "ts": "2026-07-14T14:05:33",
  "picks_monitoreados": 3,
  "triggers": [
    {
      "partido": "Boogaard vs Onclin",
      "cuota_pre": 3.55,
      "cuota_live": 2.90,
      "drift_pct": -18.3,
      "p_modelo": 0.432,
      "edge_live": 0.124,
      "senales": ["STRONG", "HOT"],
      "irp_delta": -0.02,
      "alerta_enviada": true
    }
  ],
  "combo_sugerido": { "patas": 2, "stake_total": 850 }
}
```

---

## 6. PRECONDICIONES

- `edge_report_FECHA.json` del día debe existir (PASO 3 pipeline)
- `data/odds_agg_*.json` o cliente Kambi activo (D90-08 verificado)
- Bot Telegram configurado (ya existe desde combo_confianza_builder)
- `data/irp_profiles.json` existe (Nodo-96)

---

## 7. TESTS (REGLA-T53)

`tests/test_nodo97_live_edge.py` — mínimo 8 tests:
1. `test_trigger_cuando_drift_supera_umbral` — drift ≥15% + edge_live > 5% → trigger=True
2. `test_no_trigger_si_edge_negativo` — cuota bajó pero p_modelo no cubre → trigger=False
3. `test_no_trigger_si_drift_insuficiente` — cuota bajó solo 5% → trigger=False
4. `test_solo_picks_strong_hot_monitoreados` — filtro correcto por confidence_flag
5. `test_ventana_horaria_activa` — ventana ASIMÉTRICA [-30min pre, +45min post]: pick 60min antes → excluido; pick 20min antes → incluido; pick 50min después → excluido; pick 30min después → incluido (D99-05: corregido de ±75min simétrico)
6. `test_combo_live_construido_con_2_triggers` — 2 picks con trigger → combo 2 patas
7. `test_output_json_escrito_en_reports` — snapshot escrito en path correcto
8. `test_edge_live_formula_correcta` — edge_live = p_modelo - 1/cuota_live

---

## 8. GATE DE ACTIVACIÓN

Primer ciclo: modo observación (log sin Telegram) durante 5 sesiones.
Si en ≥3/5 sesiones hay al menos 1 trigger real → activar Telegram.

---

## 9. GAPS CERRADOS PRE-AUDITORÍA (2026-07-14)

### D97-09 — Velocidad target: < 30 segundos trigger → link móvil
El edge live se cierra en 60-90s. El pipeline debe completar:
1. Fetch Kambi live (≤5s) → 2. Calcular edge (< 1s) → 3. Construir HTML + .bat Desktop (≤10s)
→ 4. Telegram con link redirect (≤5s) → **Total: < 30s desde detección.**
Output: HTML en Desktop (mismo patrón betplay_combo_builder) + Telegram link.
No solo texto — link clickeable inmediatamente en móvil.

### D97-10 — Kambi LIVE usa endpoint diferente al pre-game
D90-08 usa el offering pre-partido: `/offering/api/v3/{offering}/listView/...`
El mercado in-play usa: `/event/{eventId}/livedata` o el feed live de Kambi.
**Precondición:** verificar endpoint live por DevTools antes de implementar.
Si endpoint no confirmado → modo fallback: re-fetch pre-game offering (cuota de apertura de segunda mitad).
Fable debe confirmar el endpoint real antes del Sprint de implementación.

### D97-11 — KGR < 0 no bloquea el live monitor
El live edge monitor es observacional durante KGR < 0 (modo `--observe`):
- Log del trigger en `live_edge_*.json` → sí
- Telegram alert → sí, pero con banner "KGR < 0 — observación, no ejecutar"
- Stake sugerido → 0 (trader decide si ignora la guardia manualmente)
Razón: la señal de convergencia es válida aunque el contexto de portafolio sea negativo.
El KGR mide el pasado del día; el live edge mide el futuro del partido.

### D97-12 — Output = HTML Desktop + .bat + Telegram link (no solo texto)
El combo live genera los mismos artefactos que betplay_combo_builder:
- `Desktop/combo_live_FECHA.html` con links Betplay por cuota
- `Desktop/combo_live_FECHA.bat` para abrir en Chrome
- Telegram: link a GitHub Pages redirect con los picks
Reutilizar `_generar_html_combo()` de betplay_combo_builder.py — cero código nuevo.
