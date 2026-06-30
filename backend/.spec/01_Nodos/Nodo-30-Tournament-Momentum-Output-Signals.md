# Nodo-30 --- Tournament Momentum + Output Signals

> **Estado:** IMPLEMENTADO — 2026-06-20 | Tests: 1113→1143 passed | 30 nuevos tests T30-01 a T30-30 | F6 player_profitability.py ✅ | F7 JUGADOR RENTABLE en SENALES ESPECIALES ✅
> **Wikilinks:** [[MOC-Principal]] | [[Sprint-Normalizacion-19jun]] | [[Nodo-28-Conditional-Decomposition-Metamodel]] | [[Nodo-21-Pesos-Diferenciados-Por-Tier]]
> **Origen:** Sprint-Normalizacion-19jun (caso Eala vs Svitolina) + caso Carnicella vs Miroshnichenko (19-jun-2026). E-1/E-2 implementados ad-hoc en el sprint; este Nodo formaliza, cierra gaps, y extiende con features pendientes.
> **Prioridad:** ALTA --- afecta transparencia del output para clientes + corrige bugs activos en E-2

---

## Problema

### 3 gaps encontrados durante Sprint-Normalizacion-19jun

**Gap 1 --- TORNEO_COMPLETO sin limite de edad (BUG --- YA CORREGIDO)**
Miroshnichenko recibio x1.4 bonus por W15 Los Angeles ganado en julio 2025 (hace 1 ano). El gate de >90 dias fue implementado en sesion 19-jun. Formalizar y testear.

**Gap 2 --- Output no refleja senales del modelo**
La tabla de favoritos (`generar_tabla_favoritos2.py`) mostraba:
- Pesos INICIALES (surface 15%, form 18%) en vez de los FINALES ajustados (22%, 22%)
- Desglose superficie como "N/A" (buscaba `surface_advantage` en vez de `surface_specialization`)
- TORNEO_COMPLETO y E-1 enterrados en 100+ lineas de logs tecnicos
- Sin seccion SENALES ESPECIALES visible para clientes

**Gap 3 --- Jugadores rentables no trackeados**
Caso Carnicella: ranking bajo, nos ha dado profit multiples veces, vencio a la favorita del torneo. El pipeline no tiene memoria de que jugadores han sido rentables historicamente. Requiere datos de `betslip_registrar.py`.

---

## Alcance y Limites

### EN SCOPE (Sonnet implementa)

| Fase | Que | Archivo | Riesgo |
|---|---|---|---|
| F1 | Tests para TORNEO_COMPLETO expiry (>90d = sin bonus) | `tests/test_nodo30.py` | BAJO --- logica ya implementada, solo faltan tests |
| F2 | Tests para E-1 (weight shift cuando torneo_completo) | `tests/test_nodo30.py` | BAJO --- logica ya implementada |
| F3 | Tests para E-2 (bonus dinamico escalado) | `tests/test_nodo30.py` | BAJO --- logica ya implementada |
| F4 | Tests para get_weights_from_reasoning() pesos finales | `tests/test_nodo30.py` | BAJO --- logica ya implementada |
| F5 | Tests para SENALES ESPECIALES (scalp tier-relativo, torneo expirado) | `tests/test_nodo30.py` | BAJO --- logica ya implementada |
| F6 | Player Profitability Tracker --- lectura de betslip historicos | `analysis/player_profitability.py` | MEDIO --- feature nuevo, requiere datos de betslip_registrar |
| F7 | Integracion profitability en output tabla favoritos | `generar_tabla_favoritos2.py` | BAJO --- solo agregar senal si player tiene historial |

### FUERA DE SCOPE (no tocar)

- No cambiar la logica de prediccion en `rivalry_analyzer.py` (E-1/E-2 ya implementados y validados)
- No cambiar `edge_calculator.py` ni `trader_ev_tenis.py`
- No cambiar el flujo del pipeline (PASO 1-4)
- No agregar nuevos componentes de peso al modelo
- No modificar tests existentes (1113 deben seguir pasando)

---

## Fase 1-5: Tests de logica ya implementada

### Contexto para Sonnet

La logica de E-1, E-2, expiry, pesos finales, y senales ya esta en produccion. Lo que falta son tests unitarios que cubran los casos edge. El codigo vive en:

- `analysis/rivalry_analyzer.py` lineas 788-857 (TORNEO_COMPLETO + E-2)
- `analysis/rivalry_analyzer.py` lineas 1401-1412 (E-1 weight shift)
- `generar_tabla_favoritos2.py` funcion `get_weights_from_reasoning()` (pesos finales)
- `generar_tabla_favoritos2.py` bloque SENALES ESPECIALES (lineas ~806-850)

### Tests requeridos (archivo: `tests/test_nodo30.py`)

```
T30-01: TORNEO_COMPLETO se dispara con >=4W, 0L en mismo torneo+anio
T30-02: TORNEO_COMPLETO NO se dispara con 3W-0L (threshold minimo)
T30-03: TORNEO_COMPLETO NO se dispara con 5W-1L (tiene derrota)
T30-04: TORNEO_COMPLETO_EXPIRADO con torneo >90 dias --- bonus = 1.0 (sin efecto)
T30-05: TORNEO_COMPLETO reciente <=14d --- bonus base 1.3 + recency 0.2 = 1.5
T30-06: TORNEO_COMPLETO reciente <=14d + final(>=5W) --- bonus = 1.6
T30-07: TORNEO_COMPLETO reciente <=14d + top10 + final --- bonus = 1.7
T30-08: TORNEO_COMPLETO bonus cap --- no puede exceder 2.0
T30-09: TORNEO_COMPLETO con fecha=None --- no se dispara (sin fecha no hay recencia)
T30-10: E-1 weight shift --- torneo_completo=True + form_recent>0.10 -> surface +0.07, form -0.07
T30-11: E-1 NO se dispara si torneo_completo=False
T30-12: E-1 NO se dispara si form_recent <= 0.10 (no hay de donde quitar)
T30-13: E-2 fecha parsing --- DD.MM.YYYY usa [-4:] para anio, no [:4]
T30-14: get_weights_from_reasoning con LOG_E1_TORNEO_WEIGHT --- pesos finales reflejan el shift
T30-15: get_weights_from_reasoning con LOG_WEIGHTS_SURFACE_GRASS --- common_opp y form ajustados
T30-16: get_weights_from_reasoning con LOG_DENSITY --- co_w y form_w ajustados
T30-17: get_weights_from_reasoning sin ajustes --- devuelve pesos iniciales del strategy
T30-18: get_weights_from_reasoning cadena completa (strategy + density + grass + E1) --- pesos finales correctos
T30-19: SENALES ESPECIALES detecta TORNEO_COMPLETO_BONUS en reasoning
T30-20: SENALES ESPECIALES detecta TORNEO_COMPLETO_EXPIRADO en reasoning
T30-21: SENALES ESPECIALES scalp tier-relativo --- ITF threshold=100, GS threshold=10
T30-22: SENALES ESPECIALES scalp NO se muestra si rank > threshold del tier
```

### Patron de test para T30-01 a T30-09 (guia para Sonnet)

```python
# Instanciar RivalryAnalyzer con mocks minimos
# Construir surface_matches con campos: torneo, fecha (DD.MM.YYYY), resultado, ranking_oponente
# Llamar analyze_surface_specialization(surface_matches, 'hard', 'Player A')
# Verificar que analysis_log contiene/no contiene 'TORNEO_COMPLETO_BONUS'
# Verificar que el score refleja el multiplicador esperado
```

### Patron de test para T30-10 a T30-12 (guia para Sonnet)

```python
# Construir prediction con p1_surface_result y p2_surface_result que incluyan torneo_completo=True/False
# Simular weights dict con form_recent y surface_specialization
# Verificar que weights se ajustan/no segun las condiciones
# NOTA: E-1 esta inline en generate_prediction(), no es funcion separada.
#       Testear via generate_prediction() con mocks de historial.
```

### Patron de test para T30-14 a T30-18 (guia para Sonnet)

```python
# Construir lista de reasoning strings simulados
# Llamar get_weights_from_reasoning(reasoning)
# Verificar que los pesos devueltos son los FINALES (post-ajustes), no los iniciales
```

---

## Fase 6: Player Profitability Tracker

### Problema

Carnicella (ranking ~600 WTA) nos ha dado profit multiples veces en apuestas ITF. El pipeline no tiene memoria de esto. Cuando aparece de nuevo, la tratamos como "jugadora random de ranking bajo" en vez de "jugadora que consistentemente supera su cuota".

### Datos disponibles

`betslip_registrar.py --cerrar` ya guarda resultados en `data/calibracion_edge.json` y en archivos de betslip. Lo que NO hace es agregar por jugador.

### Solucion: `analysis/player_profitability.py`

```python
def build_player_profitability(betslip_dir='reports/'):
    """
    Lee todos los betslip_*.json cerrados y agrega por jugador:
    - n_apostado: veces que apostamos A FAVOR de este jugador
    - n_ganado: veces que gano cuando apostamos
    - profit_total: suma de (stake * (cuota-1)) si gano, -stake si perdio
    - roi: profit_total / total_apostado
    - avg_cuota: cuota promedio cuando apostamos
    - last_seen: ultima fecha
    
    Retorna dict {nombre_normalizado: {stats}}
    Persiste en data/player_profitability.json
    """
```

### Integracion en output (Fase 7)

En `generar_tabla_favoritos2.py`, seccion SENALES ESPECIALES:

```
  >> JUGADOR RENTABLE: Kaitlyn Carnicella --- 5 apuestas, 4 ganadas, ROI +62%, cuota prom 2.85
```

Solo mostrar si `n_apostado >= 3` y `roi > 0`.

### Limites de F6/F7

- READ-ONLY sobre betslips --- no modifica ningun archivo de apuesta
- Solo lee betslips con estado "cerrado" (ganado/perdido confirmado)
- Nombre normalizado debe usar la misma logica de matching que `scraping/kambi_tennis.py`
- Si no hay betslips cerrados, la feature no muestra nada (graceful degradation)
- No afecta la prediccion del modelo --- es solo una senal informativa en el output

### Tests requeridos para F6/F7

```
T30-23: build_player_profitability con 0 betslips --- retorna dict vacio
T30-24: build_player_profitability con 1 betslip cerrado (ganado) --- stats correctas
T30-25: build_player_profitability con betslip pendiente (no cerrado) --- lo ignora
T30-26: build_player_profitability con multiples apuestas al mismo jugador --- agrega correctamente
T30-27: ROI calculado correctamente --- profit / total_apostado
T30-28: SENALES ESPECIALES muestra JUGADOR RENTABLE solo si n>=3 y roi>0
T30-29: SENALES ESPECIALES NO muestra JUGADOR RENTABLE si roi<=0
T30-30: SENALES ESPECIALES NO muestra JUGADOR RENTABLE si n<3
```

---

## Validacion del Nodo

| ID | Criterio | Como verificar | Responsable |
|---|---|---|---|
| V-30-1 | 1113 tests existentes siguen pasando | 1143 passed ✅ | Haiku |
| V-30-2 | T30-01 a T30-22 pasan (logica ya implementada) | 22/22 passed ✅ | Haiku |
| V-30-3 | T30-23 a T30-30 pasan (player profitability) | 8/8 passed ✅ | Haiku |
| V-30-4 | Output tabla favoritos muestra pesos FINALES | Inspeccionar `analisis_partidos_pandas.txt` | Opus |
| V-30-5 | TORNEO_COMPLETO_EXPIRADO visible en output | grep EXPIRADO en output | Opus |
| V-30-6 | Scalp tier-relativo correcto (ITF=100, GS=10) | Inspeccionar SENALES ESPECIALES | Opus |
| V-30-7 | No regresion en predicciones clay GS | Comparar con calibracion_edge.json | Opus |

---

## Workflow de implementacion

```
1. Sonnet: Implementar tests T30-01 a T30-22 (logica existente)
   -> Opus audita: tests cubren los edge cases? faltan escenarios?

2. Haiku: Correr tests, reportar resultados
   -> Si fallan: Sonnet corrige logica o test segun diagnóstico de Opus

3. Sonnet: Implementar F6 (player_profitability.py) + F7 (integracion output)
   -> Opus audita: read-only? graceful degradation? name matching correcto?

4. Sonnet: Implementar tests T30-23 a T30-30
   -> Haiku: Correr tests completos (existentes + nuevos)

5. Haiku: Actualizar Sprint-Pipeline.md y MOC-Principal con estado del Nodo
```

---

## Conexion con Nodos anteriores

- **Sprint-Normalizacion-19jun:** E-1 y E-2 nacieron ahi. Este Nodo formaliza con tests.
- **Nodo-28 Fase 1.5:** SkillFactor/AlphaBonus crean la senal que TORNEO_COMPLETO amplifica.
- **Nodo-21:** `detectar_tier()` determina el threshold de scalp por tier.
- **Nodo-27 (Pipeline Tracker):** Player Profitability complementa la observabilidad.

---

## Reglas para implementacion

1. **NO tocar rivalry_analyzer.py** excepto para corregir bugs que los tests revelen
2. **NO tocar edge_calculator.py ni trader_ev_tenis.py**
3. **Tests primero** --- escribir el test, verificar que falla por la razon correcta, luego implementar
4. **player_profitability.py es READ-ONLY** --- lee betslips, no escribe ni modifica nada del pipeline
5. **Graceful degradation** --- si no hay betslips, si no hay datos, si el archivo no existe: silencio, no crash
6. **Name matching** --- usar la misma normalizacion que `scraping/kambi_tennis.py` para evitar duplicados
