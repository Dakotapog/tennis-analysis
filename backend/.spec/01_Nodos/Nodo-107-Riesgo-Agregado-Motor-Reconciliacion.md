# Nodo-107 — Reconciliación Embudo (Nodo-54/55) vs Riesgo Real (O-01): el veto va en el AGREGADO

> **Wikilinks:** [[Nodo-55-Respuesta-Fable-Funnel-Deploy]] | [[Nodo-54-Brief-Fable-Funnel-Deploy]] | [[Nodo-74-Combo-Governor]] | [[Nodo-100-Taxonomia-Estrategias-Generacion-Combos]] | [[Nodo-86-Auditoria-Fable5]] | [[Nodo-90-Auditoria-Fable-Nodo89]]
> **Fecha:** 2026-07-17 | **Autor:** Fable 5 (diagnóstico con datos reales, sesión PowerShell+wsl)
> **Responde a:** FABLE_BRIEF 2026-07-17 — "no presupongas la respuesta, diagnostica primero"
> **Veredicto: HIPÓTESIS C — pero con la Hipótesis A INVERTIDA respecto a lo que el operador asume.**

---

## §1. LA EVIDENCIA (números reales, no intuición)

### E1 — "El Motor" SÍ dispara. Y cuando dispara, PIERDE.
CLAUDE.md §11 (shadow book 2026-07-01→14, settled):

| Segmento | n | Hit% | ROI flat | IC 95% |
|---|---|---|---|---|
| **EL MOTOR (APROBADO)** | **20** | **30.0%** | **−21.1%** | [14.5, 51.9] |
| GCS (graduada) | 54 | 64.8% | — | — |
| Grand Slam | 19 | 47.4% | **+40.2%** | [27.3, 68.3] |
| RIVAL VALUE | 3 | 100% | +275% | pre-grad |
| ANCHOR (edge>0) | 207 | 34.3% | −5.7% | — |

**La queja del operador ("nunca produce una apuesta") y el dato son dos cosas distintas:** el Motor generó ~20 picks APROBADOS settled en 14 días (~1.4/día). Lo que casi nunca produce es **dinero desplegado** (KGR<0, VaR, MIN_BET cliff — el waterfall del Nodo-55/P54-02) — y lo que desplegó **perdió**. El segmento del Motor es HOY el peor de los 12. La frustración es legítima pero el diagnóstico "aflojar los gates del Motor" multiplicaría una población perdedora.

### E2 — O-01 ocurrió en las estrategias satélite, y el governor lo VIO sin poder actuar.
`logs/combo_governor.log` (7 sesiones registradas):
```
2026-07-10 WARN  total=$15,500  budget=$15,000  (103.3%)
2026-07-13 BLOCK total=$83,500  budget=$5,000   (1670.0%)  fecha=2026-06-26  ← O-01 retroactivo
2026-07-13 WARN  total=$17,500  budget=$15,000  (116.7%)
2026-07-14 WARN  total=$15,500  budget=$15,000  (103.3%)
2026-07-17 PASS  total=$0
```
El sobre-stake 16.7× del 2026-06-26 salió de los combos (estrategias 2-11), NO del Motor. El governor (Nodo-74) lo detecta — pero es READ-ONLY: imprime WARN y el pipeline sigue. Tres sesiones en sobrepresupuesto (103-117%) pasaron sin freno.

### E3 — O-01 NO EXISTE como registro escrito. Verificado: `grep -rn "O-01" docs/ .spec/` → 0 resultados en DECISION-LOG.md. El incidente que motivó todo este brief vive solo en memoria conversacional y en una línea del Nodo-86 §6 ("el incidente 10× invalida VaR/CPPI"). Un proyecto SDD no puede gobernar riesgo sobre un incidente sin registro.

### E4 — El governor tiene HUECOS de cobertura además de no tener autoridad.
`combo_governor.py`: suma `combo_plan_*` (estrategias 2-7, `:97-123`) + `apuestas_*.json` del día (8-11 **solo si se registraron manualmente**, `:125-142`); del `trader_plan` toma únicamente el **bankroll** (`:147-151`). **Fuera de la suma agregada: EL MOTOR (#1, individuales+cobertura del trader_plan) y RIVAL VALUE (#12, rival_value_betslip.py).** Y ninguna vista suma exposición POR JUGADOR a través de estrategias — el mecanismo exacto de O-01 (7 combos con piernas compartidas).

## §2. VEREDICTO POR HIPÓTESIS

- **Hipótesis B: CONFIRMADA — es la urgencia.** El control de riesgo agregado existe (Nodo-74) pero (a) sin autoridad de veto, (b) sin cubrir 2 de las 12 estrategias, (c) sin dimensión de concentración por pierna compartida. O-01 es exactamente eso: nadie sumó.
- **Hipótesis A: CONFIRMADA PERO INVERTIDA.** El Motor no está "tan bien calibrado que nunca dispara" (versión Nodo-55) ni "tan estricto que hay que aflojarlo" (versión operador). Sus gates seleccionan una población de alta cuota que pierde (30% hit, −21% ROI). El fallo de calibración es de **selección**, no de restricción. Aflojar = perder más rápido. La respuesta correcta es dato-gobernada (H107-01, §4).
- **Nodo-55 queda parcialmente SUPERSEDED:** su "el embudo es el ratio normal del negocio" era correcto con los datos de julio-03; con n=20 settled el embudo del Motor no es estrecho-y-bueno, es estrecho-y-negativo. Su prohibición de aflojar gates SE MANTIENE; su implicación de que el Motor es la señal de más alta convicción NO — hoy las señales rentables son GCS, GS y (pre-grad) RIVAL VALUE, todas de rutas satélite.

## §3. DECISIONES

| ID | Decisión |
|---|---|
| D107-01 | **O-01 se formaliza en DECISION-LOG.md** (texto exacto en §5). Sin registro no hay gobernanza. |
| D107-02 | **Governor cubre las 12 estrategias:** añade al agregado los stakes del `trader_plan` del día (individuales + cobertura) y los `rival_value_betslip` — matriz de cobertura 12/12 verificada por test. |
| D107-03 | **Exposición por pierna compartida:** el governor calcula stake total POR JUGADOR sumando todas las capas del día; cap = 5% del bankroll de sesión por jugador. Es el guard anti-O-01 real. |
| D107-04 | **Autoridad de veto GRADUAL:** con 7/10 sesiones logueadas, el governor pasa a **soft-veto YA** (estado WARN/BLOCK → los builders exigen flag explícito `--override-governor` para continuar, y el override queda logueado). Hard-veto automático (exit sin flag posible) al completar 10 sesiones con las nuevas dimensiones activas. NO se espera a la sesión 10 para el soft-veto: el BLOCK 1670% ya pagó esa matrícula. |
| D107-05 | **El Motor NO se afloja NI se apaga.** H107-01 pre-registrada (§4) decide con datos si el problema es el tramo de cuota alta. Mientras ROI del segmento MOTOR < 0 con n≥30: stakes individuales del trader × 0.5 (defensive sizing, reversible por graduación). |

## §4. H107-01 (pre-registro — ✅ APROBADA por el usuario y registrada en `preregistered_hypotheses.json` el 2026-07-17; JSON validado, 25 hipótesis. S107-F COMPLETADO — Sonnet ya no debe insertarla, solo leerla)
```json
{"id": "H107-01", "nombre": "MOTOR split por cuota", "prediccion": "los picks APROBADOS del Motor con cuota<=2.50 logran hit% >= breakeven de su cuota media; los de cuota>2.50 son la fuente del ROI negativo del segmento", "segmentos": ["MOTOR_cuota_baja", "MOTOR_cuota_alta"], "n_stop": 30, "accion_si_exito": "gate de cuota máxima en pool individuales del trader", "accion_si_fracaso": "revisar selección completa del Motor (no los umbrales)", "estado": "ACUMULANDO", "fecha_registro": "2026-07-17"}
```
NOTA: este bloque era el BORRADOR. La versión vigente y ya registrada vive en `validation/preregistered_hypotheses.json` (clave `H107-01`, estado ACUMULANDO, JSON validado 2026-07-17) — ante cualquier diferencia, manda el archivo de hipótesis.

## §5. TEXTO PARA DECISION-LOG.md (añadir tal cual)
```markdown
## O-01 — Sobre-stake 16.7× por agregación ciega de combos (2026-06-26, registrado 2026-07-17)
**Qué pasó:** 7 combos con piernas compartidas sumaron $83,500 desplegados contra un budget de sesión de $5,000 (1670%) — detectado retroactivamente por combo_governor 2026-07-13. Ninguna estrategia individual violó sus propios límites; nadie sumaba el total ni la concentración por jugador.
**Causa raíz:** control de riesgo per-estrategia sin control agregado cross-estrategia (governor READ-ONLY + cobertura 10/12 + sin dimensión por-jugador).
**Regla derivada:** REGLA-O1: ningún peso se despliega sin pasar por el agregado del governor (12/12 estrategias + cap 5% bankroll por jugador). Ver [[Nodo-107-Riesgo-Agregado-Motor-Reconciliacion]].
```

## §6. SPEC PARA SONNET (orden de implementación)

**Baseline primero:** `pytest tests/ --no-cov -q` → anotar el conteo actual (≥1945 post Nodo-95) — es el nuevo baseline, no romper.

1. **S107-A (D107-01):** añadir O-01 al DECISION-LOG (§5 tal cual). Sin código.
2. **S107-B (D107-02):** `combo_governor.py` — nueva función pura `_trader_stakes_today(fecha) -> dict[str,int]` (lee `trader_plan_*.json` del día: `individuales[].stake` + `cobertura[].stake`) y `_rival_value_stakes_today(fecha)` (mismo patrón sobre el output de rival_value_betslip). Sumarlas al total en `main()`. Test T53: matriz 12/12 — un fixture por estrategia, el agregado los ve todos.
3. **S107-C (D107-03):** función pura `exposicion_por_jugador(capas: list[dict]) -> dict[str, int]` — suma stake por jugador normalizado (usar `core/player_registry.normalize_player_name`, NO otra normalización nueva — C2 Nodo-67). WARN/BLOCK si algún jugador > 5% bankroll sesión. Tests: pierna compartida en 3 combos dispara; jugadores distintos no.
4. **S107-D (D107-04):** exit code del governor: 0=PASS, 1=WARN, 2=BLOCK. En `combo_confianza_builder.py`, `betplay_combo_builder.py` y `rival_value_betslip.py`: al inicio de la generación con dinero real, invocar governor (subprocess o import); si exit≥1 y no hay `--override-governor` → abortar con el reporte del governor impreso (mensaje accionable, nunca silencio — coherente con Zero-Null D90-04: el output explica qué bloqueó y cuánto habría que reducir). Override → línea en `combo_governor.log` con quién/cuánto.
5. **S107-E (D107-05):** `trader_ev_tenis.py` — factor 0.5 sobre stakes individuales mientras `MOTOR_DEFENSIVE=True` (constante con comentario a H107-01); banner en output. + `shadow_book.py` report: segmentos `MOTOR_cuota<=2.5` / `MOTOR_cuota>2.5` (patrón de segmentos existente).
6. **S107-F:** pedir OK del usuario para H107-01 (§4) e insertarla en `preregistered_hypotheses.json`.

**PROHIBIDO en este nodo:** tocar los gates del edge_calculator, λ_tier, EDGE_MIN/KELLY_KL_MIN (re-litigar Nodo-55 sin los datos de H107-01), o convertir el soft-veto en hard-veto antes de 10 sesiones.

**Criterio de éxito:** re-simular la sesión 2026-06-26 (los combo_plans existen en reports/) → el governor la BLOQUEA con detalle por jugador; sesión normal (2026-07-17) → PASS sin fricción; suite completa verde.
