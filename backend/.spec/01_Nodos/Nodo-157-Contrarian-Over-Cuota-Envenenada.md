# Nodo-157 — Contrarian OVER Signal en Ruta Cuota Envenenada (D157-01)

**Estado:** COMPLETO
**Fecha:** 2026-07-31
**Módulo principal:** `live_desk.py`

---

## Contexto / Gap

D150-01 (`cuota_envenenada`) y la ruta `linea_envenenada` (D150-02/03) detectan el mismo
fenómeno de mercado por dos vías distintas — la línea sube (`linea_drift>2j`) o la cuota
sube (`cuota_drift>+15%`). Solo la ruta `linea_envenenada` calculaba `over_candidato`
(el trade contrarian OVER cuando el partido se está alargando a tercer set). La ruta
`cuota_envenenada` marcaba la señal como envenenada pero nunca ofrecía el contrarian,
perdiendo el caso donde el mercado revalúa vía precio antes que vía línea.

## Implementación

- **D157-01:** `_calc_over_candidato()` extraído como closure reutilizable dentro de
  `_check_games_convergencia()` (antes solo inline en la rama `linea_drift>2.0`).
  Misma fórmula: `cuota_over` en rango `[1.30, 2.80]`, `p_model_over` vía Normal(midpoint,σ=3.5),
  `edge_over = p_model_over - 1/cuota_over`; candidato si `edge_over > 5%`.
- La rama `cuota_envenenada` (D150-01) ahora llama `_calc_over_candidato()` cuando
  `conv_dir=="UNDER"` y `over_candidato` aún no fue fijado por la ruta de línea.
- Badge X3: nueva rama `elif _cuota_envenenada and _over_cand` — mismo badge azul
  "OVER — TERCER SET" ya usado en la ruta `linea_envenenada`, evitando que la señal
  se muestre solo como "CUOTA ENVENENADA" (roja, sin acción) cuando sí hay un
  contrarian con edge>5%.
- Gate de combo (`_fire_itf_live_games_combo`, línea ~4425) sin cambios — ya lee
  `s.get("over_candidato")` genérico, por lo que el override D150-07 style
  (`certeza_matematica` OR `over_candidato`) ahora también cubre este segundo camino.

- **D157-02 (Bug Fix — 2026-08-01):** outcome_ids de mercados EN_VIVO expiran/rotan en segundos.
  Guard anti-flood `itf_fired_path` bloqueaba la reescritura del `.bat`/`.html` tras el primer
  disparo, dejando un coupon con ID muerto (síntoma: bat abre Betplay sin pick cargado si usuario
  no hacía clic al instante). **Fix:** mueve `_fire_itf_live_games_combo(alta_itf, ...)` FUERA del
  guard — ahora se reescribe con ID fresco cada ciclo (15s) mientras señal siga EN_VIVO;
  guard solo controla el log/notificación de "nueva señal" (cap 10/día). `live_desk.py` L4500-4512.

## No-Goals

- No se creó una tercera categoría de señal ni se tocó el gate `_edge_live_gate`/
  `_score_null_gate`/`_zona_direccion_gate` ([[Nodo-151]]) — puramente aditivo sobre
  campos ya existentes (`over_candidato`, `cuota_over_live`, `oc_id_over_live`, `edge_over`).
- No se modificó el umbral `CUOTA_ENVENENADA_UMBRAL=15.0` ni el rango de cuota
  `[1.30,2.80]` ni el edge mínimo `5.0%` — mismos valores ya validados en [[Nodo-150]].

## Tests

`tests/test_nodo157_contrarian_over_cuota.py` — 4 tests REGLA-T53 (simulan la fórmula
real con las mismas constantes de producción, siguiendo el patrón ya establecido en
`test_nodo150_live_risk_intelligence.py` para lógica embebida en `_check_games_convergencia`,
función monolítica no aislable sin mockear HTTP en vivo).

## Wikilinks

- [[Nodo-150]] — `cuota_envenenada` (D150-01), `_calcular_certeza_condicional()` (D150-02/03)
- [[Nodo-151]] — `_edge_live_gate`/`_score_null_gate`/`_zona_direccion_gate` gates
- [[Nodo-133]] — `_check_games_convergencia()` ciclo EN_VIVO (15s) donde D157-01 actúa
- [[Nodo-73]] — Nodo-n8n close-snapshot timing exacto, contexto n8n systemd
- Rival: [[Nodo-147]] — `_calcular_certeza_condicional()` Modelo Gaussiano (mismo usado en D157-01)
