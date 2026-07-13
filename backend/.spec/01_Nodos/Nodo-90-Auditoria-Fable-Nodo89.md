# Nodo-90 — Auditoría Fable del Nodo-89 (Sistema de Inteligencia Integral)

> **Wikilinks:** [[Nodo-89-Sistema-Inteligencia-Integral]] | [[Nodo-86-Auditoria-Fable5]] | [[Nodo-87-Fixes-Auditoria-D87]] | [[Nodo-72-Phantom-Identity-Guard]] | [[Nodo-64-RFI-Return-From-Inactivity]] | [[Nodo-68]] (Rival Value Flip)
> **Fecha:** 2026-07-12 | **Autor:** Fable 5 (auditoría solo-lectura, PowerShell — sin ejecución de tests)
> **Estado:** AUDITORÍA COMPLETA. Nodo-89 queda como historia inmutable; las correcciones de este nodo tienen precedencia (§10 CLAUDE.md).
> **Veredicto global:** El diagnóstico del Nodo-89 es correcto en dirección (el sistema falla en silencio en días de calificación) pero contiene **7 errores factuales contra el código real** y **4 decisiones que violan reglas constitucionales o repiten errores históricos ya documentados**. Con las correcciones de este nodo, Sprint 1 es implementable sin romper los 1822 tests.

---

## §1. CORRECCIONES FACTUALES (Nodo-89 vs código real, con evidencia)

### C-1. §6.3 del Nodo-89 describe mal `n_axes_active` — CRÍTICO para D89-11
Cálculo real: `triple_alignment_score()` en `edge_calculator.py:531-606`. Umbral único `_AXIS_THRESHOLD=0.50` sobre las señales **normalizadas**:
- Surface activo ⇔ `|alpha_vs_elo| / 0.25 > 0.50` ⇔ `|alpha_vs_elo| > 0.125` (no "surface_signal > 0.15")
- Regime activo ⇔ Markov HOT **Y** `delta_wr_markov > 0.15` (cada uno aporta 0.5; se necesita 1.0 > 0.5 — ambos)
- BBI activo ⇔ `bbi / 0.70 > 0.50` ⇔ `bbi > 0.35` (no "siempre activo si bbi != 0.5")

La tabla del caso Arseneault (§2 Nodo-89) marca Surface=0.163 como eje ACTIVO — falso (0.163 < 0.50). El propio edge_report dice `n_axes_active=1` (solo BBI). Además el BBI aparece como 0.563 en la tabla y 0.394 en el texto — inconsistencia interna sin resolver.

### C-2. La implicación no vista de §6.2: CAPA 2 debe RE-EVALUAR los gates de seguridad
Correcto que N28F2 (`:985`), HOT_sin_BBI (`:996`) y T33-01 (`:1006`) solo corren `if resultado.get('apostar')`. Implicación que el Nodo-89 no extrae: **un pick bloqueado en la puerta económica base (`:479`, edge>5% ∧ kelly>2%) jamás pasó por los gates de seguridad**. Un candidato CAPA 2 (apostar=False en base) llega "sin filtrar". La CAPA 2 debe re-aplicar explícitamente: NO_DATA (`:950-961`), phantom (`:963-980`), HOT_sin_BBI, y T33-01 (auto-satisfecho si el gate capa2 exige `n_h2h>=1` y `p>=0.60`). N28F2 se relaja **deliberada y documentadamente** (esa es la esencia de la capa), con kill-switch (§3 R1).

### C-3. D89-02 (filtrar por cobertura Kambi ANTES del edge) — RECHAZADO, viola el instrumento de medición
El shadow book necesita el universo COMPLETO de picks para CLV, calibración e hipótesis (H77, H88, RFI…). Filtrar en origen por cobertura de UNA casa mata la medición y sesga la calibración. Además PASO 1 primario es Playwright (Constitución 6), no el API. **Corrección (D90-01):** `kambi_disponible: bool` como campo observacional del edge_report (side-car `kambi_coverage_FECHA.json` generado por un fetch Kambi independiente); el filtro se aplica solo en trader/combo builder, nunca en edge_calculator ni shadow book.

### C-4. D89-03 — las "tres clases de fallo" de nombres son UNA SOLA, y el fix es quirúrgico
Evidencia: `analysis/ranking_manager.py:get_player_info()`. El paso 3 (`:455-472`) solo soporta **UNA** inicial final (`len(name_parts[-1]) == 1`). Con "Hsu Y. H." → normalizado `hsu y h`:
- Paso 3: `last_name_parts=['hsu','y']` — exige que `y` sea parte del apellido en ranked_parts → miss.
- Paso 4: matchea 3/3 por prefijo pero `is_safe_match` exige apellido en última posición o formato 2×2 → miss.
- Paso 5: excluye tokens `len<=2` (BUG-34-2 Fix B) → solo cuenta `hsu` → 1 < 2 matches → miss.

`Trotter J. K.` y `Burruchaga R. A.` fallan por el mismo camino. **Fix real (D90-02): ~15 líneas** en el paso 3 — extraer TODOS los tokens finales de 1 carácter como iniciales y matchear cada una contra las partes restantes del nombre ranked. El `CanonicalNameResolver` de 6 estrategias del Nodo-89 es sobre-ingeniería para este bug, y su estrategia 5 (Soundex/Metaphone) **CONTRADICE Nodo-72** (Phantom Identity nació de matching laxo que confundía homónimos) — ELIMINADA. La alias-table desde 112 archivos H2H es válida pero pertenece a PlayerDB (D89-05), no al camino caliente de resolución.

### C-5. D89-05 PlayerDB — la fuente de datos correcta está DENTRO de los H2H, no en los partidos del día
El schema propuesto en Nodo-89 acumula los partidos *programados* (result: "pending") — pobre. La riqueza real: cada `h2h_results_enhanced_*.json` contiene bloques `historial_<Jugador>` con partidos **ya settled**: `fecha, oponente, resultado (sets), outcome, torneo, superficie, opponent_ranking, opponent_weight` (verificado en `h2h_results_enhanced_20260712_104518.json`). Con ~112 archivos → decenas de miles de filas settled. **Correcciones (D90-03):**
- Deduplicar por `(jugador_canónico, fecha, oponente_canónico)` — el mismo historial se re-extrae cada día.
- `opponent_ranking` es el ranking al momento de la **extracción**, no del partido → guardar `ranking_asof = fecha_extraccion` del archivo fuente; brackets de RankGap toleran drift ±meses, aceptable con caveat serializado.
- No existe `own_ranking` histórico del jugador en sus filas → derivarlo del ranking del archivo del día (mismo caveat).
- Cruce con shadow book solo aporta los picks nuestros (~200 filas) — es la capa de P&L, no la de historial.

### C-6. D89-06 PlayerIntelligence — 2 de 7 dimensiones no tienen datos hoy
- **VAP (Dim 6): SIN DATOS.** El historial tiene resultado a nivel de sets ("2-0", "2-1"), no juegos por set → `avg_games_per_set` incomputable. DIFERIDA hasta que exista fuente de scores game-level.
- **PRS (Dim 4): PARCIAL.** Sin tie-breaks ni marcadores por set. Computable: win-rate en partidos a 3 sets (proxy de partidos tensos), win-rate como underdog de ranking. Redefinir así.
- **MQI (Dim 3):** sin ELO histórico del rival → proxy `opponent_ranking` + `opponent_weight` ya presentes.
- **RankGap (Dim 1) y SVI (Dim 2): 100% cubiertas** — son el MVP correcto.
- **CFS (Dim 5):** computable (torneo string en cada fila). **IRP (Dim 7):** extiende RFI (D64-01 ya serializa `rfi_*` — `edge_calculator.py:894-938`); el perfil por-jugador requiere PlayerDB primero.

### C-7. Detalles menores pero que romperían la implementación
- `from open_meteo import Client` (D89-07) no es el paquete real — usar `requests` directo a `api.open-meteo.com` (sin key, sin dependencia nueva).
- D89-08 punto 2 innecesario: `extraer_historh2h.py` consume el archivo de partidos más reciente (`select_best_json_file`, `:203`) — si PASO 1 corrió con `--tomorrow`, el H2H ya procesa mañana. `run_daily.py` ya tiene `--tomorrow` (`:221, :254-256`).
- Staleness: `build_live_combos()` YA filtra trader_plans a 24h (`betplay_combo_builder.py:2006-2009`) — D89-01 es bajar ese cutoff a 4h + mensaje accionable, no crear un mecanismo nuevo. Y `find_outcome()` ya rechaza drift de cuota >15%/25% (`:200-221`) — el caso "Martin Espinar 69%" fue el guard funcionando; lo que faltó fue la alternativa propuesta en el output.
- Los offering-keys Kambi de betcris/luckia/sportium y el API SBTech de WPlay (D89-09) son **supuestos no verificados** — precondición dura: verificación empírica de endpoints (curl) antes de escribir una línea del OddsAggregator.

---

## §2. CONFLICTOS CONSTITUCIONALES Y SU RESOLUCIÓN

### MANDATO-01 (Zero-Null) vs REGLA-HF-5 (KGR<0 → NO DESPLEGAR)
Compatibles solo con esta redacción vinculante (D90-04):
> El sistema SIEMPRE entrega una **respuesta completa**: el mejor menú de acción disponible (capa activada, picks, stakes, y por qué cada candidato bloqueado se bloqueó). Solo HF-1, HF-5, NO_DATA y PHANTOM pueden dejar el menú sin apuestas ejecutables de dinero real — y en ese caso el output OBLIGATORIAMENTE lista: (a) qué bloqueó cada candidato, (b) la alternativa concreta (games/otra casa/hora exacta de reintento). El silencio o la lista vacía sin explicación es bug. Esto satisface el espíritu del mandato del usuario sin tocar las dos reglas anti-ruina que el propio Nodo-89 §4 declara intocables.

### D89-04 CAPA 2 sin pre-registro viola Constitución 8
La CAPA 2 apuesta dinero real en una población nueva (picks sin edge formal, p>=0.60). Exige **H89-01 pre-registrada** antes del primer peso real (JSON exacto en Nodo-91 §6). La CAPA 2 nace con kill-switch: hit% < 45% con n>=20 → auto-OFF (flag), reportado en `shadow_book --report` segmento CAPA2.

### D89-10 auto-registro de hipótesis — RECHAZADO
`validation/preregistered_hypotheses.json` dice "NO modificar sin decisión". El PatternRecognitionEngine escribe candidatos a `reports/pattern_candidates_FECHA.json`; la promoción a hipótesis es decisión humana (D90-09). El engine es INSTRUMENTO REPORTE_SOLO (patrón Fase 4), lee solo picks settled.

### D89-11 (ELO como cuarto eje) — aprobado con re-diseño
NO tocar `triple_alignment_score()` (el producto de 3 normas es la definición de alignment; un 4º factor cambia la semántica y los umbrales calibrados con el caso Eala). En su lugar: `elo_dominance_axis: bool` calculado aparte (`elo_gap > +50 en dirección contraria al ranking` — datos ya en resultado: `elo_favorito`, `elo_rival`, `ranking_favorito`, `ranking_rival`), y el gate N28F2 (`:985`) pasa a `n_axes_efectivo = n_axes_active + int(elo_dominance_axis)`. OBSERVACIONAL 2 semanas primero (serializado, H89-02), luego activación — el 29% hit de "BBI sola" NO se midió con ELO-dominance como condición, pero tampoco sabemos que ELO+BBI > 50% hasta medirlo.

---

## §3. LOS 3 RIESGOS MÁS GRANDES DE IMPLEMENTACIÓN

**R1 — CAPA 2 apuesta la población que N28F2 bloqueó por evidencia (29% hit).** La única defensa real: gate estricto (p>=0.60, n_h2h>=1, cuota 1.50-2.80, HOT_sin_BBI re-evaluado), stake 25% del kelly, tope diario de capa ($5,000 challenger / $2,000 ITF), H89-01 con kill-switch automático. Si a n=20 el hit% < 45%, la capa se apaga sola y queda solo shadow.

**R2 — PlayerDB con resolución de nombres defectuosa = GIGO que contamina todo lo de arriba.** El proyecto YA vivió esto dos veces (Nodo-34 corrupción H2H, Nodo-72 phantom identity). Precondición dura de Sprint 2: el fix multi-inicial (D90-02) mergeado + tests con nombres reales extraídos de los 112 archivos + toda fila de PlayerDB lleva `resolution_confidence` (exact/reversed/fuzzy) y las fuzzy se cuarentenan de las estadísticas agregadas.

**R3 — OddsAggregator + RealTime Intelligence = superficie de scraping frágil que puede robarle semanas al pipeline core.** Endpoints supuestos, ToS, mantenimiento de venue-coords, RSS que cambian. Mitigación: staged (Kambi-family con un solo cliente parametrizado → verificar por curl ANTES → WPlay solo si el endpoint se confirma), agregación primero como medidor de CLV (shadow), ejecución multi-casa solo cuando existan cuentas reales en cada casa. Time-box: si un endpoint no se confirma en 1 sesión, se descarta esa casa y se sigue.

---

## §4. DECISIONES DE ESTE NODO

| ID | Decisión | Reemplaza/modifica |
|---|---|---|
| D90-01 | `kambi_disponible` observacional en edge_report; filtro solo en trader/combos; side-car coverage JSON | D89-02 (rechazado como filtro en origen) |
| D90-02 | Fix multi-inicial en `ranking_manager.get_player_info()` paso 3 (~15 líneas); Soundex eliminado | D89-03 (reducido; alias-table pasa a PlayerDB) |
| D90-03 | PlayerDB se construye desde los bloques `historial_*` settled internos de los H2H + dedupe canónico + `ranking_asof` | D89-05 (fuente corregida) |
| D90-04 | Capas implementadas en **trader + run_daily** (pool fallback); edge_calculator solo anota `capa2_candidate`; puerta `:479` intocada; redacción vinculante Zero-Null compatible con HF-5 | D89-04 (ubicación) + MANDATO-01 (redacción) |
| D90-05 | VAP diferida (sin datos); PRS redefinida (3-sets + underdog); MQI con proxy opponent_ranking | D89-06 |
| D90-06 | RealTime signals: OBSERVACIONAL serializado + hipótesis, nunca ajuste directo de p_modelo pre-graduación; open-meteo vía requests | D89-07 |
| D90-07 | `run_daily.py --fase noche\|manana` (noche=PASOS 0-3.6 con --tomorrow; mañana=PASO 4+); sin scripts nuevos | D89-08 |
| D90-08 | OddsAggregator staged: verificación curl → cliente Kambi parametrizado → CLV shadow → ejecución con cuentas | D89-09 |
| D90-09 | PatternRecognition escribe `pattern_candidates_*.json`; promoción a hipótesis = decisión humana; REPORTE_SOLO, solo settled | D89-10 |
| D90-10 | ELO_DOMINANCE como eje aditivo externo (`n_axes_efectivo`), observacional 2 semanas (H89-02), sin tocar triple_alignment_score | D89-11 |
| D90-11 | N28F2 re-calibración por tier: BLOQUEADA hasta n>=30 Challenger-qualy settled post-Sprint1 (los datos los genera la propia CAPA 2 en shadow) | D89-12 |

---

## §5. ROADMAP CORREGIDO (para Sonnet)

- **Sprint 1 (Nodo-91 — spec completa, implementable ya):** capas fallback (D90-04 + D89-01 staleness 4h) + fix nombres multi-inicial (D90-02, promovido a Sprint 1: 15 líneas, alto impacto) + pipeline nocturno (D90-07) + pre-registros H89-01/H89-02 + segmento CAPA2 en shadow report.
- **Sprint 2:** PlayerDB (D90-03) con `scripts/build_player_db.py` batch + incremental; `kambi_disponible` side-car (D90-01); alias-table como subproducto del batch.
- **Sprint 3:** PlayerIntelligence Dim 1+2 (RankGap, SVI) leyendo PlayerDB → serializadas al edge_report como observacionales; ELO_DOMINANCE activación si H89-02 sana; OddsAggregator fase curl+cliente Kambi parametrizado.
- **Sprint 4:** PatternRecognition (D90-09) cuando el shadow book supere ~150 picks settled; RealTime MVP observacional (D90-06); MQI/PRS/CFS.
- **Sprint 5:** N28F2 por tier (D90-11), IRP por jugador, ejecución multi-casa.

**Precondiciones por sprint:** S1: ninguna (baseline pytest 1822). S2: D90-02 mergeado + verificado. S3: PlayerDB con >=30 días y spot-check manual de 20 jugadores. S4: n settled suficiente por instrumento. S5: datos generados por S1-S3.

---

*Auditoría estática desde PowerShell (sin venv). Toda línea citada fue leída en esta sesión. Los tests los ejecuta Sonnet en WSL: baseline `pytest tests/ --no-cov -q` = 1822 passed ANTES de tocar código (Protocolo §8).*
