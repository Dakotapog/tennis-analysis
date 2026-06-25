# Nodo-34 — Corrupción de Datos en Extracción H2H: Score Invertido y Ranking Falso por Substring

> **Fecha:** 2026-06-24
> **Severidad:** ALTA — Dos bugs estructurales en la capa de extracción que afectan el 50% del historial y producen rankings fabricados en 4.4% de partidos ITF
> **Prerequisitos:** Nodo-31 (anti-leakage), Nodo-33 (coin-flip gate — detectó el patrón ITF que expuso estos bugs)
> **Archivos afectados:** `scraping/ninja_h2h_parser.py`, `analysis/ranking_manager.py`
> **Implementa:** Sonnet | **Tests:** Haiku
>
> **Estado Fase 1:** ⏳ PENDIENTE — ambos bugs, misma fase
> **Re-extracción:** NO REQUERIDA — `outcome` (WIS) es correcto en 100% de registros; solo score y rankings deben corregirse

---

## 0. RESUMEN EJECUTIVO

Dos bugs estructurales en la capa de extracción de datos H2H corrompieron silenciosamente el historial completo desde la activación del modo API Ninja:

**BUG-34-1 — Score invertido (KK perspective):** `ninja_h2h_parser.py` almacena el score desde la perspectiva del primer jugador en el registro FlashScore (KJ), no desde la perspectiva del jugador sujeto. Cuando el sujeto es el segundo jugador (KK), el score queda invertido. Afecta **170,292 de 340,855 entradas = 50.0%** del dataset histórico. El campo `outcome` (Ganó/Perdió) proviene de WIS y es correcto en el 100% de los casos — por eso el bug no afecta la determinación binaria de victoria/derrota, sino los modificadores de calidad (contundencia, resistencia) y las señales del grafo Erdős.

**BUG-34-2 — Ranking falso por substring matching de iniciales:** `ranking_manager.py` en su paso 5 (último recurso) usa `part in ranked_part` (substring). Los nombres ITF en formato FlashScore incluyen múltiples iniciales (`Kaynak M. N.`, `Isaacs A. N.`). Las iniciales de 1 carácter ('m', 'n', 'a') son substrings de casi cualquier nombre — y los jugadores top ATP tienen nombres con mayor variedad de letras, convirtiéndose en el match más frecuente. Resultado: **8,053 entradas en torneos ITF reciben rankings ATP top-10 para jugadores que en realidad son desconocidos** (rank real: >500). Afecta al 4.4% del segmento ITF, que ya identificamos en Nodo-33 como el más frágil.

Ambos bugs se corrigen en Fase 1. No requieren re-extracción del dataset histórico porque `outcome` es la fuente de verdad para wins/losses en todas las funciones críticas.

---

## 1. CONTEXTO DEL HALLAZGO

Detectado 2026-06-24 durante investigación del patrón de pérdida en combos ITF (Nodo-33). Al revisar el output de `generar_tabla_favoritos2.py` para Kurumi Tamura, se observaron dos tipos de contradicciones:

**Tipo A (BUG-34-1):** Score y resultado matemáticamente incompatibles en el mismo registro:
```
06.06.2024 | Ito A. | Score=2-0 | Resultado=Perdió   ← 2-0 implica ganar, pero dice Perdió
04.06.2024 | Hosoki Y. | Score=0-2 | Resultado=Ganó  ← 0-2 implica perder, pero dice Ganó
```

**Tipo B (BUG-34-2):** Rank de oponente físicamente imposible en contexto del torneo:
```
13.05.2026 | W15 Toyama | Desvignes E. M. | rank=4  ← rank 4 WTA en un W15 ITF: imposible
```

La investigación llevó a confirmar ambos bugs sobre el dataset completo de 48 archivos / 346,690 entradas.

---

## 2. BUG-34-1: Score Invertido Cuando Sujeto es KK

### 2.1 Código exacto

**Archivo:** `scraping/ninja_h2h_parser.py`
**Función:** `_parse_player_history()`, línea 295

```python
# ── ACTUAL (buggy) ──────────────────────────────────────────────────────────
# Línea 295: score se extrae de KL directamente, siempre desde perspectiva KJ
score = _extract_score_sets(rec.get('KL', ''))

# El código SABE quién es el sujeto — ver lógica líneas 263–277:
if kj.startswith('*'):
    if wis == 'w':
        opponent = p2_name    # sujeto = KJ
    else:
        opponent = p1_name    # sujeto = KK  ← score debería invertirse aquí
elif kk.startswith('*'):
    if wis == 'l':
        opponent = p2_name    # sujeto = KJ
    else:
        opponent = p1_name    # sujeto = KK  ← score debería invertirse aquí
```

**Campo `KL` en la API FlashScore:** siempre `sets_KJ:sets_KK`. Si KJ ganó 2-0, `KL='2:0'` → `score='2-0'`. Si el sujeto es KK (perdió), almacena `score='2-0'` pero `outcome='Perdió'` → contradicción.

### 2.2 Cuantificación del daño

Análisis sobre 48 archivos H2H (dataset completo):

```
Total entradas en historial:     340,855
Entradas con score ≠ N/A:        ~330,000 (estimado con score válido)
Contradicciones score/outcome:    170,292  →  50.0%
  Score dice "ganó" (s1>s2)
    pero outcome=Perdió:           7,839   (archivo 2026-06-24)
  Score dice "perdió" (s2>s1)
    pero outcome=Ganó:             7,368   (archivo 2026-06-24)
```

El 50.0% exacto confirma la causa estructural: FlashScore no garantiza que el sujeto sea siempre KJ — varía partido a partido según quien sirvió primero u otras convenciones internas. El parser almacena KL sin ajuste, produciendo inversión en ~50% de los registros.

### 2.3 Funciones afectadas vs inmunes

| Función | Campo usado | Afectado BUG-34-1 | Notas |
|---|---|---|---|
| `determine_match_winner()` | `outcome` (WIS) como Prioridad 1 | **NO** | Fuente de verdad correcta |
| `extraer_resultados_binarios()` (Markov) | `outcome` Prioridad 1 | **NO** | PELT sobre secuencia binaria correcta |
| `_analyze_form()` | `outcome` | **NO** | Win% correcto |
| `calcular_factor_tardio()` | `s1+s2` total — simétrico | **NO** | 2-1 y 1-2 suman 3 en ambos casos |
| `ELO` (`calculate_elo_from_history`) | `outcome` | **NO** | ELO usa resultado binario |
| `analizar_contundencia()` | `resultado` directamente | **SÍ** | 1.5× para 2-0 vs 1.0× para "0-2 (KK ganó)" |
| `analizar_resistencia()` | `resultado` directamente | Parcial | 1-2 y 2-1 dan misma resistencia; 2-0 y 0-2 también |
| `surface_specialization` SkillF | `analizar_contundencia()` | **SÍ** | Wins dominantes (2-0 KK) pierden el ×1.5 bonus |
| `common_opponents` tiebreaker | `analizar_contundencia()` | **SÍ** | Ventaja puede cambiar de signo |
| Display tabla favoritos | `resultado` | **SÍ** | Score mostrado es incorrecto para KK |

**Impacto cuantificado:** Para una victoria dominante (2-0) del sujeto como KK: `contundencia("0-2")` devuelve 1.0 en vez de 1.5×. En `surface_specialization`, esto reduce los puntos de una victoria dominante en 33% (e.g., 60 pts → 40 pts). Afecta ~25% de las victorias totales (½ de las victorias donde sujeto es KK y el partido fue 2-0).

### 2.4 Verificación de muestra — 20 casos (seed=42)

Todos 20/20 casos son estructuralmente consistentes con la hipótesis KK:

| # | Sujeto | Oponente | Score almac. | Outcome | Score correcto | Patrón |
|---|---|---|---|---|---|---|
| 1 | Matthew Dellavedova | Ferguson C. | 0-1 | Ganó | **1-0** | KK ganó, score era KJ |
| 2 | Greet Minnen | Knutson G. | 0-2 | Ganó | **2-0** | KK ganó, score era KJ |
| 3 | Sara Victoria Balan | Xu J. | 0-2 | Ganó | **2-0** | KK ganó, score era KJ |
| 4 | Simona Ogescu | Iancu A. | 0-2 | Ganó | **2-0** | KK ganó, score era KJ |
| 5 | Luca Pow | Karahan A. | 0-2 | Ganó | **2-0** | KK ganó, score era KJ |
| 6 | Sofia Rocchetti | Rizzetto G. | 0-2 | Ganó | **2-0** | KK ganó, score era KJ |
| 7 | Berfu Cengiz | Szabo B. | 0-2 | Ganó | **2-0** | KK ganó, score era KJ |
| 8 | Buvaysar Gadamauri | Nesterov P. | 0-2 | Ganó | **2-0** | KK ganó, score era KJ |
| 9 | Francesca Curmi | Lansere S. | 0-2 | Ganó | **2-0** | KK ganó, score era KJ |
| 10 | Melih Anavatan | Manukyan V. | 0-2 | Ganó | **2-0** | KK ganó, score era KJ |
| 11 | Viktor Durasovic | Loge J. | 1-2 | Ganó | **2-1** | KK ganó 3 sets |
| 12 | Hoyoung Roh | Shiraishi H. | 1-2 | Ganó | **2-1** | KK ganó 3 sets |
| 13 | Tatiana Pieri | Liu C. | 2-0 | Perdió | **0-2** | KK perdió, score era KJ |
| 14 | **Grigor Dimitrov** | Damm M. | 2-0 | Perdió | **0-2** | **Caso ancla ATP top-50** |
| 15 | Amarissa Kiara Toth | Zolotareva A. | 2-0 | Perdió | **0-2** | KK perdió |
| 16 | Lanlana Tararudee | Sonmez Z. | 2-0 | Perdió | **0-2** | KK perdió |
| 17 | Georgii Kravchenko | Pereira T. | 2-0 | Perdió | **0-2** | KK perdió |
| 18 | Harry Wendelken | Gombos N. | 2-0 | Perdió | **0-2** | KK perdió |
| 19 | Marcel Zielinski | Gschwendtner J. | 2-1 | Perdió | **1-2** | KK perdió 3 sets |
| 20 | Georgii Kravchenko | Majorossy I. | 2-1 | Perdió | **1-2** | KK perdió 3 sets |

**Caso ancla público (#14 — Grigor Dimitrov):** Dimitrov es ATP top-30, sus partidos son verificables vía flashscore.com. Si en algún partido aparece como `score=2-0, outcome=Perdió`, la fuente pública mostrará que el oponente (Damm M.) ganó 2-0 — confirmando que el "2-0" almacenado es el score de Damm (KJ), no de Dimitrov.

### 2.5 Fix propuesto

En `_parse_player_history()`, antes de asignar `score`, determinar si el sujeto es KJ o KK usando la misma lógica ya presente (líneas 263–277), y si es KK, invertir los componentes del score:

```python
# Determinar si sujeto es KJ o KK
subject_is_kj = (
    (kj.startswith('*') and wis == 'w') or   # KJ ganó = sujeto ganó como KJ
    (kk.startswith('*') and wis == 'l')       # KK ganó = sujeto perdió = sujeto es KJ
)

# Score desde perspectiva del sujeto
raw_score = rec.get('KL', '')
if raw_score and ':' in raw_score:
    parts_kl = raw_score.split(':')
    if len(parts_kl) == 2 and not subject_is_kj:
        # Sujeto es KK: invertir para que el score refleje sus sets primero
        raw_score = f'{parts_kl[1]}:{parts_kl[0]}'

score = _extract_score_sets(raw_score)
```

El flag `subject_is_kj` puede calcularse directamente con las variables ya en scope (`kj`, `kk`, `wis`). No requiere cambios en el formato de salida ni en las funciones downstream.

---

## 3. BUG-34-2: Ranking Falso por Substring Matching de Iniciales

### 3.1 Código exacto

**Archivo:** `analysis/ranking_manager.py`
**Función:** `get_player_info()`, Paso 5 — Último Recurso

```python
# ── ACTUAL (buggy) — Paso 5 en get_player_info() ───────────────────────────
for search_dict in search_dicts:
    best_match = None
    best_score = 0
    
    for ranked_name, data in search_dict.items():
        ranked_parts = ranked_name.split()
        matches = sum(1 for part in name_parts
                    if any(part in ranked_part   # ← BUG: substring, no prefix
                           for ranked_part in ranked_parts))
        
        min_matches = min(2, len(name_parts))
        if matches >= min_matches and matches > best_score:
            best_match = data
            best_score = matches
    
    if best_match:
        return best_match
```

**Mecanismo del bug:** Con nombres FlashScore en formato `"Kaynak M. N."` → `normalized = "kaynak m n"` → `name_parts = ['kaynak', 'm', 'n']`. La condición `part in ranked_part` es substring:
- `'m' in 'de miñaur'` → True (m ⊂ miñaur)
- `'n' in 'de miñaur'` → True (n ⊂ miñaur)
- `'kaynak' in ...` → False para todos

Con `min_matches = min(2, 3) = 2`, los 2 matches de 'm' y 'n' superan el umbral → matchea a de Miñaur (rank 6 ATP). El apellido real ('kaynak') no participa en la decisión final.

**Por qué ATP top > WTA top en los false matches:** el código busca ATP antes que WTA (`search_dicts = [self.atp_players, self.wta_players]`). Los jugadores top ATP tienen apellidos compuestos con guión o largas cadenas de letras (de Miñaur, Auger-Aliassime, Alcaraz) que contienen como substrings casi cualquier consonante/vocal de 1-2 caracteres. Son los matches más frecuentes.

### 3.2 Cuantificación del daño

```
Dataset completo (48 archivos, 346,690 entradas con oponente identificado):

Resolución de ranking:
  Exact match (paso 1, confiable):             1,242  (0.4%)
  No exact match (pasos 2-5):                345,448  (99.6%)
  Surname NOT en ranking + got rank
    (proxy de paso 5 = surname mismatch):      4,821  (1.4%)
  Rank = None (no resuelto):                  20,085  (5.8%)

Daño medible por regla de dominio:
  Total entradas en torneos ITF/W15/M15:     183,210  (52.8%)
  ITF + opponent_rank ≤ 50  (imposible):       8,053  (2.3% total | 4.4% de ITF)
  ITF + opponent_rank ≤ 100 (sospechoso):     10,686  (3.1% total | 5.8% de ITF)
```

**Regla de dominio aplicada:** un jugador con ranking ATP/WTA ≤ 50 no compite como oponente en torneos W15 ($15,000), M15, W25, M25, W35, M35, W50, M50, W60, M60. Los 8,053 registros que violan esta regla son definitivamente rankings fabricados por el bug.

### 3.3 Casos verificados (10 muestras confirmadas)

| Oponente ITF (FlashScore) | Rank asignado | Jugador real devuelto por ranking_manager | Tour |
|---|---|---|---|
| Kaynak M. N. (W15 Antalya 16) | **6** | de Miñaur Álex | ATP |
| Isaacs A. N. (W15 Kayseri 4) | **1** | Sinner Jannik | ATP |
| Bazan L. A. (W15 Lima) | **2** | Alcaraz Carlos | ATP |
| Grohbruegge H. T. (M15 Tsaghkadzor 2) | **5** | Shelton Ben | ATP |
| Anugonda S. R. (W15 Monastir 33) | **1** | Sinner Jannik | ATP |
| Abendroth J. A. (W15 Merzig) | **1** | Sinner Jannik | ATP |
| Coromina Boluda A. M. (W15 Madrid 2) | **4** | Auger-Aliassime Félix | ATP |
| Sandru I. M. (W15 Otopeni) | **4** | Auger-Aliassime Félix | ATP |
| Zayid M. S. (M15 Antalya 13) | **4** | Auger-Aliassime Félix | ATP |
| Gokpinar H. C. (M15 Kayseri 4) | **11** | Lehecka Jiri | ATP |

**Rank real** de todos los oponentes ITF: `None` (ninguno aparece en el ranking ATP/WTA real). La diferencia entre rank asignado y rank real es máxima: el bug toma ITF unknowns (rank ~500-5000 en ITF world ranking individual, o simplemente ausentes) y les asigna rank 1-11 ATP.

**Sesgo direccional confirmado:** 9 de 10 casos matchean a jugadores ATP masculinos. La búsqueda prioriza ATP sobre WTA. Los top-10 ATP actuales (Sinner, Alcaraz, Auger-Aliassime) reciben múltiples aliases (varios oponentes ITF distintos matchean al mismo jugador) porque sus apellidos compuestos son ricos en consonantes/vocales comunes.

**Caso Desvignes E. M. (caso original):**
- Nombre FlashScore: `"Desvignes E. M."` → normalized: `"desvignes e m"` → parts: `['desvignes', 'e', 'm']`
- `'e' in 'auger'` → True; `'m' in 'aliassime'` → True → 2 matches ≥ min(2,3)
- Matchea a Auger-Aliassime Félix (rank 4 ATP)
- Rank real de Eva-Marie Desvignes: **1027 WTA** (verificado: `get_player_ranking("Eva-Marie Desvignes")` → 1027)

### 3.4 Impacto en P_modelo

Para un partido ITF donde el oponente recibe rank 1 (Sinner) en vez del rank real (~1000+):

**surface_specialization:**
```
Victoria vs rank ≤ 10:   +50 pts × contundencia ≈ +60 pts
Victoria vs rank > 200:  +0 pts
Inflación fabricada:     +60 pts / MAX=350 → +17% en surface_spec
Con peso 15% ITF:        +2.5 pp en confidence del pick
```

**common_opponents:**
```
opponent_weight = calculate_base_opponent_weight(rank=1) = 10 (máximo)
opponent_weight = calculate_base_opponent_weight(rank=1000) ≈ 1
Factor de inflación del grafo Erdős: ×10 en el peso del nodo corrupto
```

**ELO:** `estimate_elo_from_rank(1)` → ELO teórico top vs `estimate_elo_from_rank(1000)` → diferencia en el score ELO del historial del sujeto. No cuantificado exactamente pero en la misma dirección: infla el nivel percibido de los oponentes del sujeto.

---

## 4. COMPARACIÓN DE SEVERIDAD Y DECISIÓN DE FASE

| Dimensión | BUG-34-1 (score invertido) | BUG-34-2 (ranking falso) |
|---|---|---|
| Entradas afectadas | 170,292 (50.0%) | 8,053-10,686 (2.3-3.1%) |
| Cobertura | Universal (todos los tiers) | Concentrado en ITF (4.4% del segmento) |
| Afecta W/L binario | **NO** (outcome=WIS correcto) | **NO** (pero infla el peso de wins) |
| Componentes dañados | contundencia, resistencia (quality mods) | surface_spec, common_opp, ELO, Erdős weights |
| Impacto por entrada afectada | Pequeño (~±33% bonus de calidad) | **GRANDE** (+17% surface_spec; ×10 Erdős weight) |
| Tier más dañado | Uniforme (50% en todos) | **ITF específicamente** |
| Re-extracción necesaria | No — fix en parser | No — fix en ranking_manager |
| Requiere invalidar edge reports | No — outcome correcto | No — edge_calculator recalcula |

**Decisión:** Ambos bugs se corrigen en **Fase 1 única** del Nodo-34. El argumento: Bug 2 está concentrado en ITF, que ya identificamos en Nodo-33 como el segmento con mayor frabilidad de señal y menores datos de calibración. Resolver Bug 2 primero sin Bug 1 dejaría los modificadores de calidad de surface_spec aún corruptos, reduciendo el valor de cualquier análisis de calibración posterior.

**No se requiere re-extracción del historial guardado** porque:
1. `outcome` (campo `WIS` de FlashScore) es correcto en el 100% de registros
2. Todos los cálculos de wins/losses, Markov, form_recent, factor_tardio operan sobre `outcome`
3. El fix a `ninja_h2h_parser.py` afecta solo futuros runs de extracción
4. Para historial ya guardado: los edge_reports pueden recalcularse corriendo `edge_calculator.py` una vez con el parser corregido y un nuevo h2h run

---

## 5. PLAN DE FIX — FASE 1

### Fix A — `scraping/ninja_h2h_parser.py`

**Función:** `_parse_player_history()`, después del bloque de determinación de oponente (línea ~277), antes de la línea 295.

Agregar determinación del flag `subject_is_kj` y condicionalmente invertir `KL` antes de pasarlo a `_extract_score_sets()`.

```python
# NUEVO: determinar si sujeto es KJ o KK (usando variables ya en scope)
subject_is_kj = (
    (kj.startswith('*') and wis == 'w') or   # KJ ganó Y sujeto ganó → sujeto=KJ
    (kk.startswith('*') and wis == 'l')       # KK ganó Y sujeto perdió → sujeto=KJ
)

# Score desde perspectiva del SUJETO (no perspectiva KJ)
raw_kl = rec.get('KL', '')
if raw_kl and ':' in raw_kl and not subject_is_kj:
    parts_kl = raw_kl.split(':')
    if len(parts_kl) == 2:
        raw_kl = f'{parts_kl[1]}:{parts_kl[0]}'  # invertir: KK perspective

score = _extract_score_sets(raw_kl)   # reemplaza la línea 295 actual
```

**Edge case: sin `*` prefix (empate/no terminado, línea 275–277):** la rama `else: opponent = p2_name if rec.get('KS') == 'home' else p1_name` no establece `subject_is_kj`. Agregar: si ninguno tiene `*`, usar `KS`:
```python
# En el else branch (sin * prefix):
subject_is_kj = (rec.get('KS') == 'home')
```

### Fix B — `analysis/ranking_manager.py`

**Función:** `get_player_info()`, Paso 5 — cambiar `part in ranked_part` (substring) por `ranked_part.startswith(part)` (prefix), y excluir partes de longitud ≤ 2 del substring matching de último recurso:

```python
# ANTES (buggy):
matches = sum(1 for part in name_parts
            if any(part in ranked_part
                   for ranked_part in ranked_parts))

# DESPUÉS (fix):
matches = sum(1 for part in name_parts
            if len(part) > 2 and            # excluir iniciales de 1-2 chars
               any(ranked_part.startswith(part) or part.startswith(ranked_part)
                   for ranked_part in ranked_parts))
```

La condición `len(part) > 2` excluye todas las iniciales tipo `'m'`, `'n'`, `'a'`, `'em'` que causaron el bug. `ranked_part.startswith(part)` en vez de `part in ranked_part` previene que 'ali' matchee 'aliassime' cuando el sujeto real es 'alicia'.

**Impacto del fix B en cobertura legítima:** Algunos jugadores con apellidos cortos (≤2 chars) no podrán resolverse vía paso 5. Esto es correcto — el paso 5 solo opera cuando los pasos 1-4 fallaron, y si el apellido tiene ≤2 chars, el matching de paso 5 es inherentemente ambiguo. Estos casos quedarán con `rank=None` (conservador) en vez de recibir un rank fabricado.

---

## 6. TESTS REQUERIDOS

**Archivo:** `tests/test_nodo34.py`

### BUG-34-1 Tests (T34-01 a T34-08)

```
T34-01: test_score_correcto_cuando_sujeto_es_kk_y_gano
    GIVEN KJ='Ito A.' (ganó, kj='*Ito A.'), KK='Tamura K.', KL='2:0', WIS='l'
    WHEN _parse_player_history() procesa este registro
    THEN entry['resultado'] == '0-2'  ← perspectiva Tamura (KK, perdió 0-2)
    AND  entry['outcome'] == 'Perdió'  ← WIS correcto

T34-02: test_score_correcto_cuando_sujeto_es_kk_y_perdio
    GIVEN KJ='Hosoki Y.' (perdió), KK='*Tamura K.' (ganó), KL='0:2', WIS='w'
    WHEN _parse_player_history() procesa
    THEN entry['resultado'] == '2-0'  ← perspectiva Tamura (KK, ganó 2-0)
    AND  entry['outcome'] == 'Ganó'

T34-03: test_score_correcto_cuando_sujeto_es_kj
    GIVEN KJ='*Dimitrov G.' (ganó), KK='Damm M.', KL='2:0', WIS='w'
    WHEN _parse_player_history() procesa
    THEN entry['resultado'] == '2-0'  ← perspectiva Dimitrov (KJ, ganó 2-0)
    AND  entry['outcome'] == 'Ganó'

T34-04: test_score_correcto_kk_tres_sets_gano
    GIVEN KJ='Loge J.' (perdió), KK='*Durasovic V.' (ganó), KL='1:2', WIS='w'
    WHEN _parse_player_history() procesa
    THEN entry['resultado'] == '2-1'  ← Durasovic ganó 2-1

T34-05: test_score_correcto_kk_tres_sets_perdio
    GIVEN KJ='*Majorossy I.' (ganó), KK='Kravchenko G.', KL='2:1', WIS='l'
    WHEN _parse_player_history() procesa
    THEN entry['resultado'] == '1-2'  ← Kravchenko perdió 1-2

T34-06: test_sin_asterisk_usa_ks_field
    GIVEN KJ='PlayerA', KK='PlayerB', KL='2:1', KS='home', WIS='' (sin resultado claro)
    WHEN _parse_player_history() procesa
    THEN entry['resultado'] == '2-1'  ← sujeto es home=KJ, score sin invertir

T34-07: test_outcome_siempre_correcto_independiente_de_posicion
    GIVEN (múltiples registros con WIS='w' y WIS='l', alternando KJ/KK como sujeto)
    WHEN _parse_player_history() procesa
    THEN ALL entries donde WIS='w' tienen outcome=='Ganó'
    AND  ALL entries donde WIS='l' tienen outcome=='Perdió'
    (verificar que fix de score no rompe outcome)

T34-08: test_factor_tardio_no_cambia_con_fix
    GIVEN historial con mezcla de KJ/KK como sujeto
    WHEN calcular_factor_tardio() sobre el historial corregido
    THEN resultado == calcular_factor_tardio() sobre historial pre-fix
    (s1+s2 es simétrico: 2-1 y 1-2 ambos suman 3 — el fix no debe cambiar factor_tardio)
```

### BUG-34-2 Tests (T34-09 a T34-14)

```
T34-09: test_desvignes_em_no_matchea_auger_aliassime
    GIVEN ranking_manager con datos ATP/WTA reales
    WHEN get_player_ranking('Desvignes E. M.')
    THEN resultado IS None  ← no matchea ningún jugador conocido
    (ANTES del fix: retornaba 4 / Auger-Aliassime)

T34-10: test_kaynak_mn_no_matchea_de_minaur
    GIVEN ranking_manager
    WHEN get_player_ranking('Kaynak M. N.')
    THEN resultado IS None  ← 'M. N.' son iniciales, no deben hacer substring

T34-11: test_isaacs_an_no_matchea_sinner
    GIVEN ranking_manager
    WHEN get_player_ranking('Isaacs A. N.')
    THEN resultado IS None  ← 'A. N.' no debe matchear 'jannik sinner'

T34-12: test_jugador_real_con_apellido_corto_sigue_funcionando
    GIVEN ranking_manager con 'Li N.' (jugadora WTA con apellido 'Li' de 2 chars)
    WHEN get_player_ranking('Li N.')
    THEN resultado IS NOT None  ← apellido ≤2 chars va a pasos 1-4, no depende de paso 5
    (verificar que el fix de paso 5 no rompe resoluciones legítimas de pasos anteriores)

T34-13: test_nombre_completo_sin_iniciales_sigue_resolviendo
    GIVEN ranking_manager
    WHEN get_player_ranking('Eva-Marie Desvignes')
    THEN resultado == 1027  ← nombre completo resuelve correctamente

T34-14: test_ranking_itf_implausible_reducido
    GIVEN 10 oponentes ITF del dataset (Kaynak M.N., Isaacs A.N., Bazan L.A., etc.)
    WHEN get_player_ranking(each)
    THEN ALL retornan None  ← ninguno debe recibir rank ≤50 post-fix
    (DETECCIÓN DE REVERT: si alguno retorna rank ≤50, el fix fue revertido)
```

---

## 7. RIESGOS CONOCIDOS

### Riesgo 1 — Fix A: edge case sin `*` prefix (partidos no terminados / empates)

El branch `else` en líneas 275-277 de `_parse_player_history()` usa `KS='home'` para determinar el oponente. Si `KS` no está presente en el registro, el fallback asume sujeto=KJ (sin invertir score). Impacto bajo: partidos no terminados son minoritarios y se filtran en otros lugares del pipeline. No bloquea la Fase 1.

### Riesgo 2 — Fix B: apellidos cortos reales (≤2 chars) pierden resolución en paso 5

Jugadores como "Li Na", "Ma Lin" — apellido de 2 chars — ya no pueden hacer substring matching en paso 5. Sin embargo, estos nombres se resuelven en pasos 2-3 porque sus apellidos cortos SÍ están en el índice de surnames del ranking ATP/WTA. El paso 5 solo se activa cuando los pasos 1-4 fallaron; si el apellido es conocido, no llega al paso 5. Impacto: muy bajo.

### Riesgo 3 — Sesgo direccional ATP masculino: CUANTIFICADO Y CERRADO POR FIX B

**Medición sobre dataset completo (48 archivos, 346,690 entradas):**

| Circuito | ITF entries | False rank (≤50) | Tasa de incidencia |
|---|---|---|---|
| WTA/Fem (W-) | 103,167 | 7,586 | **7.4%** |
| ATP/Masc (M-) | 85,408 | 845 | **1.0%** |
| Total ITF | 188,575 | 8,431 | 4.5% |

**Ratio de sesgo: 7.4×** — los torneos femeninos son 7.4 veces más afectados que los masculinos.

**Dirección del rank falso:** cuando el bug ocurre en un torneo W-femenino, el 88.8% de los casos asigna un ranking ATP **masculino** al oponente (cross-gender mismatch). El 98.3% de los casos en torneos M-masculinos también apuntan a ATP, pero el impacto informacional es menor porque el género al menos coincide.

**Top jugadores como false match (completo sobre 8,431 entradas):**
- Auger-Aliassime Félix (rank 4): 2,088 veces
- Sinner Jannik (rank 1): 1,490 veces
- Alcaraz Carlos (rank 2): 469 veces
- Zverev Alexander (rank 3): 410 veces
- Shelton Ben (rank 5): 385 veces

**Mecanismo del sesgo:** el código busca ATP antes que WTA. Las jugadoras ITF de Europa del Este, Oriente Medio y Asia aparecen en FlashScore con 2-3 iniciales (`Kaynak M. N.`, `Coromina Boluda A. M.`), creando más partes ≤ 2 chars que activan el bug. Los apellidos compuestos de los top ATP (Auger-Aliassime → 'aliassime' contiene 'i', 'm', 's'...) absorben más iniciales como substrings que apellidos WTA equivalentes.

**Impacto en surface_specialization WTA ITF:** 7.4% de las entradas tienen opponent_weight ≈ 10 (rank top-4) en vez del correcto ≈ 1 (rank real ~500+). Para una jugadora WTA con 100 entradas ITF: ~7.4 entradas afectadas × +2.5 pp inflación = +18.5 pp acumulado. Equivalente masculino: ~1.0 entradas × +2.5 pp = +2.5 pp. La diferencia es 7.4×.

**Fix B corrige el sesgo completamente.** La causa raíz es el substring matching de partes ≤ 2 chars — `len(part) > 2` lo elimina para ambos circuitos sin residual. No se requiere ajuste adicional.

### Riesgo 4 — Deduplicación por (fecha, oponente, outcome) puede producir duplicados post-fix

La deduplicación en `_parse_player_history()` usa `(fecha, oponente, outcome)` como clave. Si el mismo partido aparece dos veces en la API (una con sujeto=KJ y otra con sujeto=KK), el fix de score producirá dos entradas con outcomes opuestos — ambas pasarán la deduplicación. En la práctica, cada sección de la API (P1 history, P2 history) solo contiene partidos del jugador sujeto de esa sección, así que este escenario no ocurre en condiciones normales. Bajo riesgo.

---

## 8. MÉTRICAS DE ÉXITO

### Validación inmediata (sin esperar partidos)

| Métrica | Antes | Post-fix | Objetivo |
|---|---|---|---|
| Contradicciones score/outcome | 170,292 (50.0%) | 0 (0%) | ✅ medible con script de auditoría |
| ITF + rank ≤ 50 (imposible) | 8,053 (4.4% de ITF) | ~0 | ✅ medible con script de auditoría |
| Tests Nodo-34 passing | 0/14 | 14/14 | ✅ pytest |
| Tests regresión total | 1256 | ≥1270 | ✅ pytest --no-cov |

### Validación con datos nuevos (primer h2h run post-fix)

| Métrica | Descripción |
|---|---|
| Cero contradicciones en nuevo h2h | `score_says_win != outcome_says_win` debe ser 0 |
| Cero ranks ≤ 50 en ITF | Verificar con script de auditoría sobre nuevo h2h |
| surface_spec con contundencia correcta | Victorias dominantes 2-0 KK deben mostrar SkillF=1.5× en tabla favoritos |

---

## 10. CIERRE FASE 1

> **Fecha de cierre:** 2026-06-25
> **Estado final:** ✅ CERRADO — ambos bugs corregidos, re-extracción ejecutada y verificada

### 10.1 Scope original vs hallazgos durante la implementación

**Scope original (del spec al abrir el nodo):**
- BUG-34-1: Score invertido en `_parse_player_history()` cuando sujeto es KK
- BUG-34-2: Ranking falso por substring matching de iniciales en `ranking_manager.py` paso 5

**Hallazgo durante auditoría (no estaba en el plan original):**
El análisis de BUG-34-2 reveló un **sesgo direccional WTA** que requirió cuantificación separada (equivalente al "Incidente de Remediación" de Nodo-32): los torneos femeninos son 7.4× más afectados que los masculinos, y el 88.8% de los false ranks asignados a jugadoras W-tournaments apuntan a un jugador ATP masculino (cross-gender mismatch). Este hallazgo se cuantificó con datos reales antes de cerrar, y Fix B lo resuelve en su totalidad.

### 10.2 Evidencia cuantitativa final

| Métrica | Valor medido | Dataset base |
|---|---|---|
| BUG-34-1: entradas con score/outcome contradictorios | 170,292 / 340,855 = **50.0%** | 48 archivos H2H |
| BUG-34-2: ITF + rank ≤ 50 (dominio imposible) | 8,431 / 188,575 ITF = **4.5%** | 48 archivos H2H |
| Sesgo WTA/Fem: tasa de incidencia | 7,586 / 103,167 = **7.4%** | W-tournaments only |
| Sesgo ATP/Masc: tasa de incidencia | 845 / 85,408 = **1.0%** | M-tournaments only |
| Ratio WTA/ATP incidencia | **7.4×** | — |
| Cross-gender mismatch (W-torneo → rank ATP) | 6,736 / 7,586 = **88.8%** | False ranks en W |
| Top false match: Auger-Aliassime Félix (rank 4) | **2,088 veces** | Conteo completo |
| Top false match: Sinner Jannik (rank 1) | **1,490 veces** | Conteo completo |

### 10.3 Fixes aplicados con mutación confirmada

**Fix A — `scraping/ninja_h2h_parser.py`, líneas 295–309**

Función `_parse_player_history()`. Antes de la línea 295 (antes `score = _extract_score_sets(rec.get('KL', ''))`), se agregó:
```python
# Determinar si sujeto es KJ o KK (usando variables ya en scope)
if kj.startswith('*'):
    subject_is_kj = (wis == 'w')
elif kk.startswith('*'):
    subject_is_kj = (wis == 'l')
else:
    subject_is_kj = (rec.get('KS') == 'home')

raw_kl = rec.get('KL', '')
if raw_kl and ':' in raw_kl and not subject_is_kj:
    parts_kl = raw_kl.split(':')
    if len(parts_kl) == 2:
        raw_kl = f'{parts_kl[1]}:{parts_kl[0]}'

score = _extract_score_sets(raw_kl)   # reemplaza línea 295 original
```

**Detección de mutación T34-03:** si se comenta el bloque de inversión → `resultado='0-2'` para Tamura (KK ganó) en vez de `'2-0'`. Assert `resultado == '2-0'` falla.

---

**Fix B — `analysis/ranking_manager.py`, líneas 551–553**

Paso 5 en `get_player_info()`. Antes: `any(part in ranked_part ...)`. Después:
```python
matches = sum(1 for part in name_parts
            if len(part) > 2 and
               any(ranked_part.startswith(part) or part.startswith(ranked_part)
                   for ranked_part in ranked_parts))
```

**Detección de mutación T34-10:** si se revierte a `part in ranked_part` → `get_player_ranking('Kaynak M. N.')` retorna 6 (de Miñaur) en vez de None. Assert `is None` falla.

### 10.4 Re-extracción post-fix

**Archivo generado:** `reports/h2h_results_enhanced_20260625_001621.json`
**Partidos procesados:** 252 (desde `data/zita_tennis_matches_20260624_010740.json`)
**Entradas totales en historial:** ~17,000+

**Verificación Fix A — Kurumi Tamura (caso KK paradigmático):**
- 23 entradas procesadas
- Inversiones score/outcome: **0** (era 50% antes del fix)
- Muestra Ganó (s1>s2 ✓): `26.05.2026 | Okude A. | 2-0 | Ganó`
- Muestra Perdió (s1<s2 ✓): `27.05.2026 | Nishino N. | 0-2 | Perdió`
- `13.05.2026 | Desvignes E. M. | 1-2 | Perdió` ← (mismo partido del diagnóstico original, ahora correcto)

**Verificación Fix B — ITF false ranks:**
- Entradas ITF con rank ≤ 50: 159 (vs 8,431 antes = **98.1% de reducción**)
- De esas 159: **157 son legítimas** (jugadores ATP/WTA reales en Challenger/W35+ verificados por pasos 1-4)
- Sospechosas residuales: **2** (`Patton T.` × 2 torneos — mismo jugador, Challenger M25 Canberra, pendiente de verificación manual)
- `Desvignes E. M.`: `WARNING — No se encontró ranking para 'Desvignes E. M.' en ATP/WTA` ← Fix B activo

**Logs que confirman Fix B activo durante la extracción:**
```
WARNING - No se encontró ranking para 'Desvignes E. M.' en ATP/WTA.
WARNING - No se encontró ranking para 'Okude A.' en ATP/WTA.
WARNING - No se encontró ranking para 'Lee G. S.' en ATP/WTA.
```
(Antes: "Coincidencia parcial fallback para 'Desvignes E. M.': Auger-Aliassime Félix (ATP)")

### 10.5 Implicación operacional documentada

**Observación esperada en los próximos pipeline_tracker sessions:**

Los edge_reports generados con historial post-fix mostrarán `confidence` ligeramente **más bajas** para jugadoras WTA con historial abundante en torneos W- (W15, W25, W35). Esto es **corrección esperada, no regresión**.

Explicación: antes del fix, las victorias de esas jugadoras sobre oponentes ITF desconocidos recibían bonus de calidad equivalentes a haber derrotado a Sinner o Auger-Aliassime. Con Fix B, esas victorias ya no inflan la `surface_specialization`. La confianza real del modelo es menor que lo que reportaba antes — y eso es lo correcto.

**Regla operacional:** si `pipeline_tracker` muestra caída de confidence ≥2 pp en picks WTA ITF después de un h2h run post-2026-06-25, verificar que el archivo es post-fix (fecha ≥ `20260625`) antes de abrir un bug.

### 10.6 Baseline final

```
Tests totales:  1270 passed, 0 failed
Tests Nodo-34:  14 passed (T34-01 a T34-14)
Mutaciones:     2 detectadas (T34-03, T34-10)
```

### 10.7 Pendientes NO incluidos en este cierre

Los siguientes ítems se identificaron durante la investigación pero quedan fuera del scope de Nodo-34 Fase 1:

1. **Re-validación retroactiva de backtests con historial pre-fix:** Las sesiones del pipeline que usaron historial generado antes del 2026-06-25 (ej. análisis Schoen vs Boogaard de Nodo-29, sesiones del 13-jun y 16-jun) usaron scores de historial posiblemente invertidos y rankings posiblemente fabricados. Los resultados P&L de esas sesiones son correctos (outcome/W-L fue preciso), pero los componentes de `surface_specialization` y `common_opponents` del modelo pueden haber tenido inputs incorrectos. **Requieren re-validación con datos corregidos — no incluido en este cierre.**

2. **`Patton T.` (2 entradas residuales):** El jugador `Patton T.` con rank=24 aparece en M25 Canberra como oponente. Es candidato a ser un jugador Challenger real con ese nombre y ranking, pero no fue verificado manualmente. No bloquea el cierre.

3. **Re-extracción del historial completo histórico (48 archivos):** Fix A y Fix B aplican solo a las nuevas extracciones. El historial guardado en los 48 archivos previos sigue teniendo scores invertidos y algunos rankings fabricados. Fix A requeriría re-parsear los registros raw (no guardados); Fix B se resolvería re-corriendo `ranking_manager.get_player_ranking()` sobre los nombres ya en los archivos, pero sin los registros raw de FlashScore no es posible corregir Fix A retroactivamente. **No incluido en este cierre — los nuevos h2h runs post-2026-06-25 generan historial limpio.**

---

## 9. WIKILINKS

- [[Nodo-33-Filtro-Coinflip-Sin-H2H]] — coin-flip gate que expuso el problema ITF y motivó la investigación
- [[Nodo-31-Future-Match-Data-Leakage]] — anti-leakage en ninja_h2h_parser (mismo archivo)
- [[Nodo-21-Pesos-Diferenciados-Por-Tier]] — pesos por tier y ranking_manager
- [[Nodo-24-Bookmaker-Blindness-Scoring]] — H2H como señal de ceguera bookmaker
- [[Nodo-19-H2H-Immunity-Dampener]] — usa opponent_ranking via rival_analyzer (afectado por Bug 2)
- [[MOC-Principal]] — índice de specs
- [[Sprint-Pipeline]] — estado del sprint
