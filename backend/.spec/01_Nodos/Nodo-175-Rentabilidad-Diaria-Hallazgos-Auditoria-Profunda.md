# Nodo-175 — Rentabilidad Diaria: Síntesis de la Auditoría Profunda (Nodo-163→174) y Plan de Mejora

> Estado: **PENDIENTE DE IMPLEMENTAR**. Escrito en modo síntesis (2% tokens disponibles) para no perder el
> conocimiento acumulado en ~15 sesiones de auditoría de código real. Cada hallazgo cita archivo/patrón para
> que una sesión futura pueda verificar con grep puntual antes de tocar código (GIT-FIRST, GREP-FIRST).
> No se ejecutó código nuevo para escribir este nodo — es síntesis de lo ya verificado en Nodos 154-174.

## 0. Por qué este nodo

Los últimos ~10 nodos (163-174) repararon **infraestructura de medición**: gates tier-agnósticos, calibración
p_modelo, hypothesis_ledger que por fin escribe n_actual, symbol-audit que impide afirmaciones falsas en
CLAUDE.md. El sistema ahora es medible y transparente. **Lo que falta es traducir esa medibilidad en más
apuestas rentables ejecutadas por día** — no más arquitectura, sino cerrar la brecha señal→stake real.

## 1. Diagnóstico consolidado — dónde está el dinero real (shadow book, tabla §11 CLAUDE.md)

| Segmento | Hit% | ROI | n | Lectura |
|---|---|---|---|---|
| GCS (grass) | 64.8% | — | 54 | ÚNICA graduada, pero ventana de temporada muy angosta (solo hierba) |
| RIVAL VALUE | 100% | +275% | 3 | Mejor ROI absoluto pero **n=3, sin significancia** — gate n=30 lejos |
| Grand Slam | 47.4% | +40.2% | 19 | Mejor ROI real con n decente — infra-explotado (pocos GS/año) |
| EL MOTOR (producto núcleo) | 30.0% | **-21.1%** | 20 | El generador de picks individuales pierde dinero incluso "APROBADO" |
| Challenger | 37.2% | -5.5% | 78 | Volumen alto, ROI negativo |
| ITF | 36.9% | **-13.5%** | 111 | Mayor volumen de TODOS los tiers, mayor pérdida flat-1u |
| WATCHLIST | 34.9% | -5.2% | 149 | n grande, sigue sin editar tras Kelly-KL supuestamente compensar con λ=4.5x en ITF |

**Lectura crítica:** los tiers de mayor volumen (ITF, Challenger, Watchlist) son los de peor ROI. Kelly-KL con
λ alto (ITF=4.5x) reduce el stake pero **no convierte una señal mala en buena** — si el modelo pierde plata en
flat-1u en ITF, sigue perdiendo con Kelly, solo que más despacio. La pregunta de rentabilidad real no es "qué
tan grande apostar" sino "¿por qué el modelo tiene edge negativo estructural en el tier de mayor volumen?".

## 2. Gaps activos con impacto directo en rentabilidad (priorizados)

### G175-01 — Ceguera de P&L a nivel de COMBO (el más importante)
La tabla §11 mide picks **individuales** en shadow_book. Pero la mayoría del stake real sale por
`betplay_combo_builder.py`/`combo_confianza_builder.py` (SAFE/WAS/MEGA/GAMES/SISTEMA/ANCLA SEGURA/CORE/etc,
13 estrategias). No existe un reporte que cruce `ComboRegistry` (Nodo-144 D144-08 `strategy` tag) contra
resultado liquidado por combo → no sabemos si SISTEMA Leave-One-Out o MEGA realmente ganan dinero, solo que
sus componentes individuales tienen tal o cual hit rate aislado. El caso Nodo-172 (combo real @49.33x ganó
por varianza, no señal — 3 de 4 piernas LOW confidence) demuestra que un combo puede "ganar" sin que el motor
tenga edge real. **Sin este reporte, cualquier decisión de rentabilidad es ciega.**
→ Acción: nuevo script `combo_pnl_report.py` que lea `data/combo_registry.json` (o el jsonl equivalente),
cruce con resultados liquidados (mismo mecanismo que `shadow_book.py --settle`), y agregue hit%/ROI/CLV **por
estrategia de combo**, no solo por pick individual.

### G175-02 — Filtro de calidad de fillers (confidence_flag) solo existe en 1 de 13 estrategias
Nodo-172 creó `build_ancla_segura_combos()` con el filtro STRONG/MODERATE-only para fillers porque
`build_live_combos()` (D133-04) y `build_system_combos()` (D156-B-01) **nunca leían `confidence_flag`** —
solo aplicaban REGLA-HF-1 (cuota≥1.50) y disponibilidad Kambi. Esto significa que SAFE, WAS, MEGA, GAMES y
SISTEMA (5 de 8 estrategias en betplay_combo_builder.py) **siguen aceptando fillers LOW confidence hoy**. El
mismo patrón de riesgo que causó la auditoría de Nodo-172 está presente, sin cerrar, en el resto del builder.
→ Acción: replicar el gate `confidence_flag in (STRONG, MODERATE)` de D172-01 a las 5 estrategias restantes
(no debería requerir nueva arquitectura, es el mismo patrón — bajo esfuerzo, alto impacto en varianza).

### G175-03 — Señales REPORTE_SOLO nunca graduadas a gate real (alpha dejado en la mesa)
Varias señales fueron construidas, verificadas con datos reales, y **nunca se usaron para decidir nada**:
- `irp_fav`/`irp_rival` (Nodo-96, 4361 perfiles) — solo serializado en edge_report, sin gate ni bonus Kelly.
- HCUC convergence (Nodo-155 D155-02, `_calc_hcuc_convergence` en `edge_calculator.py:1003`) — acumulando
  n desde D174-03, pero H152-01 aún sin graduar; cuando gradúe, falta el paso "convertir en gate/bonus".
- Monte Carlo condicional (Nodo-160 D160-02, `core/monte_carlo_games.py`) — anota `mc_p_condicional` en la
  señal, explícitamente REPORTE_SOLO, nunca gate (decisión deliberada H160-02, pero nunca revisada post-hoc).
- Dual-Book Router X1 (Nodo-111, `scraping/dual_book_client.py --compare`) — imprime tabla de mejor cuota por
  casa. **No verificado si el stake real se enruta a la mejor cuota o si solo es un reporte informativo** —
  si es solo reporte, cada pick que podría haberse jugado en la casa con mejor precio es ROI regalado.
- Triple Convergencia C1 (Nodo-99, STRONG+rival_COLD+drift_live) — descrita como "alpha oculto más puro",
  sin evidencia en memoria de que tenga tracking dedicado en shadow_book (a diferencia de H88-01/H98-01/etc).
→ Acción: para cada una, verificar (grep puntual) si connecta a alguna decisión de stake/filtro. Las que no
graduaron por n insuficiente, dejarlas — las que ya tienen n razonable y nunca se evaluó gate, evaluarlas.

### G175-04 — Propagación incompleta del calibrador de Nodo-173
Nodo-173 corrigió la raíz real de "3 meses sin apostar" (p_modelo no era probabilidad real: AUC 0.575 pero
Brier skill −0.042 → calibrador ancla-mercado). BLOQUE C se cerró legítimamente por PUERTA 3 (skill≤0) — pero
**no quedó verificado en esta sesión si el p_modelo calibrado llega a TODOS los consumidores** (Kelly sizing
en trader_ev_tenis, ranking de fillers en combo builders, shadow_book CLV) o solo a los tocados en BLOQUE A/B.
Dado el patrón repetido en Nodo-174 (símbolos marcados ✅ que nunca se conectaron), este es el riesgo más alto
de "arreglado en el papel, no en producción".
→ Acción: `grep -rn "probability_calibrator\|p_modelo_calibrado" *.py core/` y verificar cada call-site que
LEE p_modelo para decisión de stake — confirmar que usa la versión calibrada, no la cruda.

### G175-05 — EL MOTOR (single-bet engine) pierde dinero pese a "APROBADO"
30% hit / −21.1% ROI con n=20, gates ya estrictos. Es el único generador que no combina cuotas (combo
multiplica, MOTOR es cuota única) — su ROI negativo no se explica por "cuotas altas normales", es la métrica
más directa de si el modelo tiene edge real. Con hypothesis_ledger (D174-03) ahora escribiendo n_actual,
esta cifra se puede seguir acumulando automáticamente — falta decidir un kill-switch o revisión de gates si
n crece y el ROI sigue negativo (ya existe el patrón kill-switch en H151-01/H165-01, replicar aquí).
→ Acción: pre-registrar hipótesis H175-XX con n_stop y umbral kill-switch para EL MOTOR específicamente
(hoy no tiene una H-XX dedicada pese a ser el producto núcleo).

### G175-06 — Ventana de rentabilidad real es angosta (GCS = grass only)
La única estrategia graduada y positiva (GCS, 64.8%) solo aplica en hierba, ATP500+. Fuera de temporada de
hierba (semanas específicas del año), el sistema no tiene ninguna estrategia graduada con evidencia formal
de rentabilidad — todo lo demás está en pre-graduación o es negativo en flat-1u. Esto es más un hallazgo de
honestidad que un bug: **el "hedge fund activo" descrito en CLAUDE.md §1 no tiene todavía una estrategia
graduada de uso diario/todo el año.**
→ Acción: no hay fix de código — es priorización: acelerar la acumulación de n en Grand Slam (mejor ROI real
tras GCS) y RIVAL VALUE (mejor ROI absoluto) en vez de seguir construyendo nuevas estrategias sin graduar.

## 3. Deliverables propuestos (orden sugerido por impacto/esfuerzo)

| ID | Deliverable | Esfuerzo | Impacto P&L |
|---|---|---|---|
| D175-01 | `combo_pnl_report.py` — hit%/ROI/CLV por estrategia de combo (G175-01) | Medio | ALTO — sin esto, todo lo demás es a ciegas |
| D175-02 | Replicar `confidence_flag` gate de D172-01 a SAFE/WAS/MEGA/GAMES/SISTEMA (G175-02) | Bajo | Medio-Alto — reduce varianza en 5/8 estrategias |
| D175-03 | Auditar propagación real de calibrador Nodo-173 a todos los consumidores de p_modelo (G175-04) | Bajo (grep) | ALTO si hay brecha — replica el patrón exacto de Nodo-174 |
| D175-04 | Verificar si dual_book_client X1 enruta stake real o solo reporta (G175-03) | Bajo | Medio — ROI gratis si no está enrutando |
| D175-05 | Pre-registrar H175-XX kill-switch para EL MOTOR (G175-05) | Bajo | Alto a mediano plazo (evita seguir perdiendo en el producto núcleo) |
| D175-06 | Evaluar graduación/gate de IRP, HCUC, MC-condicional con n actual (G175-03) | Medio | Depende de n — puede no ser accionable aún |

## 4. Hipótesis a pre-registrar (validation/preregistered_hypotheses.json)

- **H175-01**: EL MOTOR (picks individuales, cuota única, gates D151/D164 activos) — hit rate ≥ breakeven
  implícito de la cuota promedio del segmento. n_stop=30 (actual n=20). Kill-switch: si con n≥30 el ROI
  flat-1u sigue <−15%, congelar EL MOTOR como generador de stake real y dejarlo solo como insumo de combos.
- **H175-02**: combos SAFE/WAS/MEGA/GAMES/SISTEMA con gate `confidence_flag` (post D175-02) vs su propio
  historial pre-fix — comparar hit%/ROI antes/después del filtro, n_stop=15 por estrategia.

## 5. Nota de honestidad SDD

Este nodo es síntesis, no auditoría fresca — algunos hallazgos (G175-03, G175-04) requieren un grep de
verificación de ~2 minutos antes de asumir que el gap sigue abierto (el código pudo cambiar entre la sesión
que generó cada Nodo fuente y hoy). Aplicar GIT-FIRST/GREP-FIRST antes de cada D175-XX, como manda §2 regla 9
de CLAUDE.md — no repetir el patrón que Nodo-174 encontró (símbolos ✅ nunca commiteados).
