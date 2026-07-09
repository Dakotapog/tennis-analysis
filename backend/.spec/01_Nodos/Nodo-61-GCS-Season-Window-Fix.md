# Nodo-61 — GCS Season Window: Fix de Fecha y Ventana Season-Aware

> **Wikilinks:** [[Nodo-60-GCS-Grass-Surface-Champion-Signal]] | [[Nodo-60-ADDENDUM-FABLE-Auditoria-Tres-Carriles]] | [[Nodo-57-Penalizacion-Inactividad-Campeon-Validacion]] | [[Nodo-52-Shadow-Book-CLV-Tracking]] | [[Nodo-58-Dashboard-Observabilidad]]
> **Fecha spec:** 2026-07-05
> **Estado:** ✅ IMPLEMENTADO 2026-07-06 — 1669 tests (10 nuevos T61-01→T61-10). Bug F0+F1 cerrados.
> **Prioridad:** ALTA — bug activo en producción + señal real perdida en Wimbledon QF/SF/F

---

## §0. Veredicto Ejecutivo

| Problema | Tipo | Impacto hoy | Fix |
|---|---|---|---|
| **Bug F0: `gcs_days` usa año incorrecto** | Bug de datos | Eala R4 aparece con 28d en vez de 15d. GCS clasificado como EXPIRADO incorrectamente. | Anclar búsqueda de `torneo_reciente_ganado` a los últimos 42 días, no al historial completo |
| **Bug F1: Gate 21d excluye Semana 1 sistemáticamente** | Fallo de diseño | Birmingham/Nottingham/Halle/Queens nunca cubren Wimbledon QF/SF/F | Reemplazar contador de días por gate season-aware (42d + mismo año) |
| **Gap F2: Sin datos en zona 22-42d** | Ceguera metodológica | H60-01 no puede probar ni refutar la extensión porque LOG_GCS_SHADOW no cubre hierba >21d | LOG_GCS_SHADOW_EXTENDED para colección gratuita |
| **H60-02: hipótesis para la extensión** | Pre-registro | Sin pre-registro no se puede activar nada | Registrar H60-02 antes de acumular datos |

**El principio violado:** el sistema está descartando información válida y conocida (Eala ganó Birmingham 2026, no 2025) por un bug de búsqueda de fecha. Adicionalmente, la arquitectura del gate era estructuralmente ciega a los torneos de Semana 1 de hierba desde el primer día.

---

## §1. Evidencia — Los Datos que Prueban el Bug

### Evidencia E1 — Inconsistencia de `gcs_days` en producción

```
PARTIDO:   Jasmine Paolini vs Alexandra Eala (Wimbledon)
MATCH_ID:  G0wm3Eid

Análisis R3 (edge_report 2026-07-04):   gcs_days = 13d  →  gcs_active = True   →  APOSTAR  edge=23.9%
Análisis R4 (edge_report 2026-07-06):   gcs_days = 28d  →  gcs_active = False  →  sin sección en edge report
```

Dos días de diferencia real (Jul 4 → Jul 6) **no pueden producir 15 días de diferencia en `gcs_days`**.

La única explicación coherente: el extractor de H2H cambió entre análisis (FlashScore H2H para R3 vs Kambi H2H para R4) y entregó una historia de partidos en Birmingham con orden o profundidad temporal diferente. El algoritmo encontró Birmingham 2025 (~Jun 8) en lugar de Birmingham 2026 (~Jun 21).

### Evidencia E2 — El Calendar Math Confirma el Bug

```
Birmingham 2026 final estimado: ~21 Junio 2026  (según gcs_days=13 en R3, que es correcto)
Birmingham 2025 final estimado: ~8 Junio 2025   (según gcs_days=28 en R4, que es incorrecto)

Jul 4 - Jun 21 = 13 días  ✅ CORRECTO (el modelo funcionó bien en R3)
Jul 6 - Jun 21 = 15 días  ✅ CORRECTO si el fix se aplica  (debería mostrarse esto)
Jul 6 - Jun  8 = 28 días  ❌ INCORRECTO (año equivocado — Birmingham 2025)
```

### Evidencia E3 — La Tabla de Calendario Muestra el Fallo Estructural

```
Días transcurridos desde torneo previo hasta cada ronda de Wimbledon 2026:

Torneo         Final     R1   R2   R3   R4   QF   SF    F   Cubierto (≤21d)
────────────────────────────────────────────────────────────────────────────
Nottingham    Jun 15     15*  17*  19*  21*  23   25   28   R1-R4 solo (50%)
Birmingham    Jun 21     9*   11*  13*  15*  17*  19*  22   R1-F  (100%)  ← si fix correcto
Birmingham    Jun  8    22   24   26   28   30   32   35   NINGUNA (0%)  ← con bug
Halle         Jun 15    15*  17*  19*  21*  23   25   28   R1-R4 solo (50%)
Queens        Jun 15    15*  17*  19*  21*  23   25   28   R1-R4 solo (50%)
Eastbourne    Jun 28     2*   4*   6*   8*  10*  12*  15*  R1-F  (100%)
Bad Homburg   Jun 28     2*   4*   6*   8*  10*  12*  15*  R1-F  (100%)
Ilkley        Jun 21     9*  11*  13*  15*  17*  19*  22   R1-F  (100%)

* = dentro del gate actual ≤21d
```

Con la fecha correcta de Birmingham 2026 (Jun 21), el gate de 21 días YA cubre toda la primera ronda hasta el final. **El bug de fecha es el problema prioritario — corregirlo resuelve parcialmente el problema calendario para este año.**

Sin embargo, el fallo estructural persiste para Nottingham/Halle/Queens (finales Jun 15): sus campeones pierden el GCS boost en QF/SF/F de Wimbledon bajo el gate actual de 21 días. Eso requiere F1 (extensión a 42d).

### Evidencia E4 — Los 11 Casos GCS Observados en Producción

```
Fuente: grep en todos los edge_reports/  (comando de verificación en §7)

Partido                              Días  Boost   Edge    Tier         Sección
─────────────────────────────────────────────────────────────────────────────
Alexandra Eala vs Iga Swiatek        13d   1.15x   23.9%   grand_slam   APOSTAR   ← caso fundacional
Ashlyn Krueger vs Marta Kostyuk      21d   1.13x   17.1%   grand_slam   WATCHLIST
Marie Bouzkova vs Elise Mertens      14d   1.03x   3.8%    grand_slam   WATCHLIST
Madison Keys vs Linda Noskova         8d   0.92x   -10.4%  grand_slam   SIN_EDGE
Ashlyn Krueger vs Marta Kostyuk      21d   1.11x   15.3%   grand_slam   WATCHLIST (×4 runs)
Simona Waltert vs Katarzyna Kawa     21d   1.03x   4.1%    challenger   WATCHLIST
```

Observaciones clave:
1. El único caso APOSTAR (Eala 13d, 23.9% edge) usó la fecha CORRECTA de Birmingham 2026
2. Todos los casos con días ≤21 tienen gcs_active=True y aparecen en edge_report
3. No existe NI UN SOLO caso con días entre 22-42 en ningún edge_report — el LOG_GCS_SHADOW no cubre hierba >21d
4. Keys con gcs_boost=0.92 (NEGATIVO) merece auditoría separada (ver §2.4)

---

## §2. Diagnóstico Técnico

### §2.1 Bug F0 — La Causa Raíz en el Código

En `analysis/rivalry_analyzer.py`, función `analyze_surface_specialization()`, la detección de `torneo_reciente_ganado` busca la secuencia de victorias más larga en un mismo torneo dentro del historial de partidos del jugador.

**El problema:** el historial de partidos del jugador puede contener múltiples ediciones del mismo torneo (Birmingham 2024, Birmingham 2025, Birmingham 2026). La búsqueda sin filtro de fecha de recencia puede retornar la edición más antigua o la más fácil de detectar según la ordenación de la API.

**Comportamiento actual (pseudocódigo):**
```python
# BUGGY
for partido in historial_completo:   # historial puede ser 2-3 años
    if torneo_actual == torneo_anterior and ganó:
        streak += 1
    torneo_reciente_ganado_fecha = fecha_primer_partido_de_streak
    # PROBLEMA: si el historial no está ordenado por recencia exacta,
    # puede tomar Birmingham 2025 en lugar de Birmingham 2026
```

**Comportamiento correcto:**
```python
# CORRECTO
LOOKBACK_DAYS = 42  # solo buscar dentro de los últimos 42 días
cutoff = fecha_partido - timedelta(days=LOOKBACK_DAYS)
for partido in [p for p in historial if p.fecha >= cutoff]:
    # solo considera partidos recientes → no puede confundir años
```

### §2.2 Bug F1 — Gate 21d Estructura-Ciego

El parámetro `dias_max=21` en `analyze_surface_specialization()` fue establecido en Nodo-60 como una estimación conservadora. Nunca fue validado contra la estructura del calendario de tenis.

**El fallo:** La "grass season" tiene una estructura natural de 6 semanas (Jun 1 - Jul 13). Un campeón de Birmingham (semana 2 del swing, final ~Jun 21) llega a Wimbledon con 9 días — dentro del gate. Pero un campeón de Nottingham (semana 1, final ~Jun 15) llega a QF Wimbledon con 23 días — fuera del gate. La diferencia entre los dos torneos es de 6 días de calendario, no de calidad de preparación.

**La señal que se pierde:** Nottingham/Halle/Queens son torneos en hierba ATP500 o WTA500 de primer nivel. Un campeón de Queen's Club tiene históricamente una de las correlaciones más altas con el título de Wimbledon masculino. El sistema los descarta silenciosamente en rondas tardías.

### §2.3 Gap F2 — Ceguera Metodológica en Hierba >21d

El `LOG_GCS_SHADOW` (Nodo-60 ADDENDUM) se dispara para clay/hard cuando `gcs_active=True` (señal detectada) pero la superficie no es grass. Para hierba con días >21, `gcs_active=False` y el LOG no se dispara en absoluto.

**Consecuencia:** El shadow book no tiene NI UN registro de "jugador con torneo ganado en hierba hace 22-35 días". La hipótesis H60-02 (extensión del gate) no puede ser evaluada prospectivamente porque los datos simplemente no se recolectan.

### §2.4 Anomalía — Keys con gcs_boost=0.92 (Negativo)

Madison Keys aparece con boost NEGATIVO (0.92). Esto es contraditorio con la lógica GCS (el boost debería ser ≥1.0 siempre que gcs_active=True). Requiere auditoría:

```bash
# Comando de auditoría
grep -A10 "Keys\|Noskova" reports/edge_report_2026070*.json | grep -E "gcs|boost|score"
```

Si es un bug de cálculo donde el multiplicador se aplica en sentido incorrecto (dividiendo en vez de multiplicando, o aplicando al rival en vez de al jugador), debe corregirse en este nodo.

---

## §3. Deliverables

| ID | Descripción | Archivo | Prioridad |
|---|---|---|---|
| D61-F0 | Fix bug `torneo_reciente_ganado_fecha` — búsqueda anclada a últimos 42d | `analysis/rivalry_analyzer.py` | CRÍTICA |
| D61-F1 | Season-aware gate — reemplaza `dias_max=21` con `_is_gcs_active()` season-aware | `analysis/rivalry_analyzer.py` | ALTA |
| D61-F2 | LOG_GCS_SHADOW_EXTENDED — colección datos hierba 22-42d | `analysis/rivalry_analyzer.py` + `edge_calculator.py` | ALTA |
| D61-F3 | `_GCS_EXTENDED_ENABLED = False` — flag explícito de producción | `analysis/rivalry_analyzer.py` | ALTA |
| D61-F4 | H60-02 pre-registrada en `preregistered_hypotheses.json` | `validation/preregistered_hypotheses.json` | ALTA |
| D61-F5 | `gcs_extended_active` + `gcs_extended_days` en edge_report | `edge_calculator.py` | MEDIA |
| D61-F6 | Auditoría Keys boost negativo + fix si corresponde | `analysis/rivalry_analyzer.py` | MEDIA |
| D61-F7 | Tests T61-01 → T61-10 | `tests/test_nodo61.py` | OBLIGATORIA |
| D61-F8 | Panel 6 Nodo-58: fila H60-02 | `dashboard.py` | MEDIA |

---

## §4. Implementación Detallada

### D61-F0 — Fix bug `torneo_reciente_ganado_fecha`

**Archivo:** `analysis/rivalry_analyzer.py`
**Función:** `analyze_surface_specialization()` (buscar `torneo_reciente_ganado` o `_detect_recent_champion`)

**Cambio:**

```python
# CONSTANTE NUEVA — buscar solo en este horizonte
_GCS_LOOKBACK_DAYS = 42   # máximo histórico que consideramos "reciente"

# En la función de detección de campeón reciente:
# ANTES (buggy):
matches_en_torneo = [m for m in historial if m['torneo'] == nombre_torneo]

# DESPUÉS (correcto):
from datetime import timedelta
cutoff_date = fecha_partido - timedelta(days=_GCS_LOOKBACK_DAYS)
matches_en_torneo = [
    m for m in historial
    if m['torneo'] == nombre_torneo
    and m['fecha'] >= cutoff_date    # ← ESTA ES LA LÍNEA NUEVA CRÍTICA
]
```

**Precaución de implementación:** La `fecha_partido` puede venir como string ISO, datetime o date. Normalizar antes de comparar. Si `historial` no tiene fechas parseable, usar año de la temporada actual como fallback (`año == año_partido`).

**Resultado esperado post-fix:**
```
Eala R4 (Jul 6):  torneo_reciente_ganado = Birmingham 2026 (Jun 21)
                  gcs_days = 15  (no 28)
                  gcs_active = True (15 ≤ 21 — sin necesitar el gate extendido)
```

### D61-F1 — Season-Aware Gate

**Archivo:** `analysis/rivalry_analyzer.py`
**Constantes nuevas:**

```python
# Ventanas de season por superficie
# Formato: (mes_inicio, día_inicio, mes_fin, día_fin, dias_max_gcs)
_GCS_SEASON_WINDOWS = {
    'grass':  {'start': (6, 1),  'end': (7, 13), 'dias_max': 42},
    'hierba': {'start': (6, 1),  'end': (7, 13), 'dias_max': 42},  # alias
    # clay y hard: sin season definida aún — usar legacy 21d
}

# Flag explícito para la extensión 22-42d
_GCS_EXTENDED_ENABLED = False   # OFF por default hasta H60-02 gradúe
```

**Función nueva `_is_gcs_active(torneo_fecha, partido_fecha, superficie)`:**

```python
def _is_gcs_active(torneo_fecha, partido_fecha, superficie_normalizada):
    """
    Determina si el GCS signal está activo para este partido.
    
    Reglas:
    1. El torneo ganado debe ser de la misma superficie.
    2. El torneo ganado debe haber ocurrido dentro de la season window del año actual.
    3. Los días transcurridos deben estar dentro del límite de la season.
    4. Si días > 21 y _GCS_EXTENDED_ENABLED=False → gcs_active=True pero boost=1.0
       (ver _gcs_boost_multiplier).
    """
    sup = superficie_normalizada.lower()
    window = _GCS_SEASON_WINDOWS.get(sup)
    
    if window is None:
        # Superficie sin season definida: usar gate legacy 21d
        dias = (partido_fecha - torneo_fecha).days
        return dias <= 21, dias
    
    # Verificar que el torneo ganado está dentro de la season window del año en curso
    year = partido_fecha.year
    season_start = date(year, window['start'][0], window['start'][1])
    season_end   = date(year, window['end'][0],   window['end'][1])
    
    if not (season_start <= torneo_fecha <= season_end):
        # El torneo fue en otra season (año pasado, o fuera del swing de hierba)
        return False, (partido_fecha - torneo_fecha).days
    
    dias = (partido_fecha - torneo_fecha).days
    active = dias <= window['dias_max']
    return active, dias
```

**Función modificada `_gcs_boost_multiplier(gcs_days, gcs_active)`:**

```python
def _gcs_boost_multiplier(gcs_days, gcs_active):
    """
    Retorna el multiplicador GCS para el final_score.
    
    Días 1-21: boost activo (validado por H60-01, 64.8% hit rate).
    Días 22-42: boost GATED (pendiente H60-02).
                Si _GCS_EXTENDED_ENABLED=False → retorna 1.0 y dispara LOG_GCS_SHADOW_EXTENDED.
    Días >42: sin boost.
    """
    if not gcs_active:
        return 1.0
    if gcs_days <= 7:
        return _GCS_MULT_RECENT    # 2.2
    if gcs_days <= 14:
        return _GCS_MULT_MID       # 1.8
    if gcs_days <= 21:
        return _GCS_MULT_BASE      # 1.5
    if gcs_days <= 42:
        # Zona gated: señal detectada pero boost suspendido hasta H60-02
        if _GCS_EXTENDED_ENABLED:
            if gcs_days <= 28: return 1.3
            if gcs_days <= 35: return 1.15
            return 1.05
        else:
            return 1.0   # LOG_GCS_SHADOW_EXTENDED se dispara en calling code
    return 1.0
```

### D61-F2 — LOG_GCS_SHADOW_EXTENDED

**Archivo:** `analysis/rivalry_analyzer.py`
**Dónde:** En `analyze_surface_specialization()`, después de calcular el boost.

```python
# Cuando gcs_active=True pero gcs_days > 21 y _GCS_EXTENDED_ENABLED=False:
if gcs_active and gcs_days > 21 and not _GCS_EXTENDED_ENABLED:
    mult_si_activo = 1.3 if gcs_days <= 28 else (1.15 if gcs_days <= 35 else 1.05)
    log_msg = (
        f"LOG_GCS_SHADOW_EXTENDED: {jugador_nombre} días={gcs_days}d, "
        f"torneo={torneo_reciente}, "
        f"boost_si_H60_02_graduara=×{mult_si_activo} — "
        f"pendiente graduación H60-02 (n_stop=30)"
    )
    surface_log.append(log_msg)
    # También serializar en el resultado para el shadow book
    result['gcs_extended_active'] = True
    result['gcs_extended_days'] = gcs_days
    result['gcs_extended_mult_potencial'] = mult_si_activo
```

**Archivo:** `edge_calculator.py`
**Dónde:** En la serialización del pick (donde ya se serializa `gcs_bonus`, `gcs_days`, etc.)

```python
# Agregar al pick_dict:
pick['gcs_extended_active'] = partido.get('gcs_extended_active', False)
pick['gcs_extended_days']   = partido.get('gcs_extended_days', None)
```

### D61-F3 — `_GCS_EXTENDED_ENABLED` Flag Explícito

Ya incluido en D61-F1. Verificar que el default sea `False` y que sea visible en el código como constante de módulo (no enterrada en una función).

```python
# Línea explícita al nivel del módulo en rivalry_analyzer.py:
_GCS_EXTENDED_ENABLED = False   # GCS 22-42d GATED: activar solo tras H60-02 graduación
```

### D61-F4 — H60-02 Pre-Registro

**Archivo:** `validation/preregistered_hypotheses.json`
**Agregar** después del bloque H60-01:

```json
"H60-02": {
  "nombre": "GCS signal persiste en zona 22-42d dentro de la misma grass season",
  "hipotesis": "Picks con TORNEO_COMPLETO_BONUS (grass, tier>=atp500), torneo ganado EN LA MISMA GRASS SEASON (Jun 1 - Jul 13 mismo año), dias en [22, 42] → hit% sigue elevado vs baseline sin GCS",
  "descripcion": "Extension de H60-01 para cubrir campeones de Semana 1 del swing (Nottingham/Birmingham/Halle/Queens) jugando en Wimbledon QF/SF/F donde el gate de 21d los excluía sistematicamente.",
  "origen_deuda": "Nodo-61 — analisis calendario Wimbledon 2026-07-05 + bug gcs_days Eala R4",
  "preregistrado": "2026-07-05",
  "umbrales_congelados": {
    "tier_min": "atp500",
    "superficie": "grass",
    "dias_min": 22,
    "dias_max": 42,
    "season_start": "Jun 1",
    "season_end":   "Jul 13",
    "mismo_anyo": true
  },
  "multiplicadores_propuestos": {
    "22-28d": 1.3,
    "29-35d": 1.15,
    "36-42d": 1.05,
    "nota": "Propuesta inicial — PROHIBIDO activar antes de exito=true. Magnitudes calibradas conservadoramente respecto a H60-01."
  },
  "metrica": "hit% con IC Wilson 95%",
  "exito": "limite inferior IC Wilson 95% > 1/cuota_media del segmento Y Brier_con < Brier_sin",
  "corte_secundario_preregistrado": "mismo segmento AND edge_vs_mercado >= 0.08",
  "n_stop": 30,
  "estado": "PENDIENTE",
  "estado_inicial": "n=0 — sin datos prospectivos. El gate de 21d cegaba esta zona hasta Nodo-61.",
  "gated": "GCS_EXTENDED_ENABLED permanece False hasta exito=true. LOG_GCS_SHADOW_EXTENDED acumula datos.",
  "n_actual": 0,
  "hits": 0,
  "nota": "Contexto: Nottingham/Halle/Queens (finales ~Jun 15) quedan sistematicamente fuera del gate en Wimbledon QF+ bajo el limite de 21d. Con fix D61-F0 (fecha correcta), Birmingham 2026 ya es <=21d en Wimbledon R4 — H60-02 aplica principalmente a Semana 1 torneos en rondas tardias."
}
```

### D61-F5 — Serialización en edge_report

**Archivo:** `edge_calculator.py`

Agregar en la construcción del pick_dict (donde se serializa `gcs_bonus`, `gcs_days`, `gcs_score_boost`):

```python
# Campos nuevos D61-F5
pick_dict['gcs_extended_active']        = partido.get('ranking_analysis', {}) \
                                            .get('prediction', {}) \
                                            .get('gcs_extended_active', False)
pick_dict['gcs_extended_days']          = partido.get('ranking_analysis', {}) \
                                            .get('prediction', {}) \
                                            .get('gcs_extended_days', None)
pick_dict['gcs_extended_mult_potencial']= partido.get('ranking_analysis', {}) \
                                            .get('prediction', {}) \
                                            .get('gcs_extended_mult_potencial', None)
```

### D61-F6 — Auditoría Keys boost negativo

Antes de implementar, verificar:

```bash
python3 -c "
import json
data = json.load(open('reports/edge_report_20260705_223706.json'))
for section in ['apostar','watchlist','sin_edge']:
    for p in data.get(section, []):
        if 'keys' in p.get('partido','').lower() or 'noskova' in p.get('partido','').lower():
            print(p.get('partido'))
            print('  gcs_bonus:', p.get('gcs_bonus'))
            print('  gcs_score_boost:', p.get('gcs_score_boost'))
            print('  gcs_days:', p.get('gcs_days'))
            print('  p_modelo:', p.get('p_modelo'))
"
```

Si `gcs_score_boost < 1.0`, el bug está en la serialización: el campo se calcula para el jugador equivocado (rival en vez del jugador predicho). Fix: verificar que `gcs_active` y `gcs_score_boost` correspondan al `favorito_predicho`, no al rival.

---

## §5. Tests T61-01 → T61-10

**Archivo:** `tests/test_nodo61.py`

**REGLA-T53:** Todos los tests deben invocar la función real del módulo. No hardcodear fórmulas.

```python
# tests/test_nodo61.py
"""
Tests Nodo-61: GCS Season Window Fix
Verifica fix del bug de fecha, gate season-aware, LOG_GCS_SHADOW_EXTENDED y H60-02.
"""
import pytest
from datetime import date
from analysis.rivalry_analyzer import analyze_surface_specialization
from validation.hypothesis_tracker import get_hypothesis

# ─── T61-01: Fix fecha Birmingham 2026 ──────────────────────────────────────

def test_T61_01_gcs_days_usa_birmingham_2026_no_2025():
    """
    Con fix D61-F0: Eala con Birmingham 2026 (Jun 21) → gcs_days=15 para Jul 6.
    El análisis NO debe usar Birmingham 2025 (Jun 8 → 28 días).
    """
    # Construir historial con AMBAS ediciones de Birmingham
    historial_con_dos_anios = [
        # Birmingham 2026 (el correcto)
        {'torneo': 'Birmingham', 'fecha': date(2026, 6, 21), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': 'Birmingham', 'fecha': date(2026, 6, 20), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': 'Birmingham', 'fecha': date(2026, 6, 18), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': 'Birmingham', 'fecha': date(2026, 6, 17), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': 'Birmingham', 'fecha': date(2026, 6, 16), 'resultado': 'win', 'superficie': 'grass'},
        # Birmingham 2025 (el que NO debe usarse)
        {'torneo': 'Birmingham', 'fecha': date(2025, 6, 8),  'resultado': 'win', 'superficie': 'grass'},
        {'torneo': 'Birmingham', 'fecha': date(2025, 6, 7),  'resultado': 'win', 'superficie': 'grass'},
    ]
    fecha_partido = date(2026, 7, 6)
    result = analyze_surface_specialization(
        historial_con_dos_anios,
        superficie='grass',
        tier='grand_slam',
        fecha_partido=fecha_partido
    )
    # Con fix: debe usar Birmingham 2026 (Jun 21) → gcs_days=15
    assert result['gcs_days'] == 15, (
        f"gcs_days={result['gcs_days']} — el fix debe usar Birmingham 2026 (15d), "
        f"no Birmingham 2025 (28d)"
    )
    assert result['gcs_active'] is True, "15 días ≤ 21 → debe estar activo"


# ─── T61-02: Sin confusión de año anterior ──────────────────────────────────

def test_T61_02_torneo_anio_anterior_no_activa_gcs():
    """
    Si el único Birmingham disponible es de 2025 (hace >365d), NO debe activar GCS.
    La búsqueda está limitada a los últimos 42 días.
    """
    historial_solo_2025 = [
        {'torneo': 'Birmingham', 'fecha': date(2025, 6, 8), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': 'Birmingham', 'fecha': date(2025, 6, 7), 'resultado': 'win', 'superficie': 'grass'},
    ]
    fecha_partido = date(2026, 7, 6)
    result = analyze_surface_specialization(
        historial_solo_2025,
        superficie='grass',
        tier='grand_slam',
        fecha_partido=fecha_partido
    )
    assert result['gcs_active'] is False, (
        "Birmingham 2025 está fuera del lookback de 42 días — no debe activar GCS"
    )


# ─── T61-03: Season-aware activa Birmingham Semana 1 en QF Wimbledon ────────

def test_T61_03_season_aware_nottingham_activo_en_wimbledon_qf():
    """
    Nottingham (final ~Jun 15) → Wimbledon QF (Jul 8) = 23 días.
    Con gate legacy 21d: INACTIVO.
    Con gate season-aware 42d + mismo año: ACTIVO.
    """
    historial = [
        {'torneo': 'Nottingham', 'fecha': date(2026, 6, 15), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': 'Nottingham', 'fecha': date(2026, 6, 14), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': 'Nottingham', 'fecha': date(2026, 6, 12), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': 'Nottingham', 'fecha': date(2026, 6, 11), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': 'Nottingham', 'fecha': date(2026, 6, 10), 'resultado': 'win', 'superficie': 'grass'},
    ]
    fecha_partido = date(2026, 7, 8)  # Wimbledon QF
    result = analyze_surface_specialization(
        historial,
        superficie='grass',
        tier='grand_slam',
        fecha_partido=fecha_partido
    )
    assert result['gcs_active'] is True, (
        f"Nottingham Jun 15 → QF Jul 8 = 23d. Season-aware gate (42d) debe estar activo. "
        f"gcs_days={result['gcs_days']}"
    )
    assert result.get('gcs_extended_active') is True or result['gcs_days'] <= 21, (
        "Días 22-42 en season deben marcarse como gcs_extended_active o gcs_active"
    )


# ─── T61-04: Torneo clay no activa GCS en partido grass ─────────────────────

def test_T61_04_clay_tournament_no_activa_gcs_en_grass():
    """
    Torneo ganado en clay dentro de los últimos 42d NO activa GCS en partido de grass.
    El season-aware gate requiere MISMA superficie.
    """
    historial = [
        {'torneo': 'Roland Garros', 'fecha': date(2026, 6, 8), 'resultado': 'win', 'superficie': 'clay'},
        {'torneo': 'Roland Garros', 'fecha': date(2026, 6, 7), 'resultado': 'win', 'superficie': 'clay'},
    ]
    fecha_partido = date(2026, 7, 4)  # Wimbledon R3
    result = analyze_surface_specialization(
        historial,
        superficie='grass',
        tier='grand_slam',
        fecha_partido=fecha_partido
    )
    assert result['gcs_active'] is False, (
        "Torneo ganado en clay no debe activar GCS para partido en grass"
    )


# ─── T61-05: LOG_GCS_SHADOW_EXTENDED se dispara en zona 22-42d ──────────────

def test_T61_05_log_gcs_shadow_extended_dias_22_42():
    """
    Nottingham (23d antes de QF Wimbledon): con _GCS_EXTENDED_ENABLED=False,
    el boost es 1.0 pero debe aparecer LOG_GCS_SHADOW_EXTENDED en los logs.
    """
    historial = [
        {'torneo': 'Nottingham', 'fecha': date(2026, 6, 15), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': 'Nottingham', 'fecha': date(2026, 6, 14), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': 'Nottingham', 'fecha': date(2026, 6, 12), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': 'Nottingham', 'fecha': date(2026, 6, 11), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': 'Nottingham', 'fecha': date(2026, 6, 10), 'resultado': 'win', 'superficie': 'grass'},
    ]
    fecha_partido = date(2026, 7, 8)
    result = analyze_surface_specialization(
        historial,
        superficie='grass',
        tier='grand_slam',
        fecha_partido=fecha_partido
    )
    # El boost no debe cambiar el score si el flag está OFF
    assert result.get('gcs_score_boost', 1.0) == 1.0, (
        f"Con _GCS_EXTENDED_ENABLED=False, boost debe ser 1.0 para días 22-42. "
        f"Actual: {result.get('gcs_score_boost')}"
    )
    # Pero debe existir el campo gcs_extended_active
    assert result.get('gcs_extended_active') is True, (
        "gcs_extended_active debe ser True para señalar que hay una señal potencial no activada"
    )
    # Y debe haber un log con el texto correcto
    logs = result.get('logs', [])
    shadow_ext_log = [l for l in logs if 'LOG_GCS_SHADOW_EXTENDED' in str(l)]
    assert len(shadow_ext_log) > 0, (
        "Debe existir al menos un LOG_GCS_SHADOW_EXTENDED en los logs del análisis"
    )


# ─── T61-06: Flag OFF → final_score idéntico para días 22-42 ────────────────

def test_T61_06_extended_flag_off_no_cambia_final_score():
    """
    Con _GCS_EXTENDED_ENABLED=False, el final_score de un pick en zona 22-42d
    debe ser idéntico al final_score sin ningún código GCS.
    REGLA-T53: comparar invocando la función con el flag ON vs OFF.
    """
    from analysis.rivalry_analyzer import analyze_surface_specialization, _GCS_EXTENDED_ENABLED
    historial = [
        {'torneo': 'Nottingham', 'fecha': date(2026, 6, 15), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': 'Nottingham', 'fecha': date(2026, 6, 14), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': 'Nottingham', 'fecha': date(2026, 6, 12), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': 'Nottingham', 'fecha': date(2026, 6, 11), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': 'Nottingham', 'fecha': date(2026, 6, 10), 'resultado': 'win', 'superficie': 'grass'},
    ]
    fecha_partido = date(2026, 7, 8)
    assert _GCS_EXTENDED_ENABLED is False, (
        "_GCS_EXTENDED_ENABLED debe ser False por default en producción"
    )
    result = analyze_surface_specialization(
        historial, superficie='grass', tier='grand_slam', fecha_partido=fecha_partido
    )
    assert result.get('gcs_score_boost', 1.0) == 1.0, (
        "Con flag OFF, boost debe ser exactamente 1.0 — el final_score no puede cambiar"
    )


# ─── T61-07: H60-02 pre-registrada correctamente ────────────────────────────

def test_T61_07_h60_02_en_preregistered_hypotheses():
    """
    H60-02 debe existir en preregistered_hypotheses.json con los campos obligatorios
    y estado PENDIENTE.
    """
    import json, os
    path = os.path.join(os.path.dirname(__file__), '../validation/preregistered_hypotheses.json')
    data = json.load(open(path))
    hyp = data.get('hypotheses', {}).get('H60-02')
    assert hyp is not None, "H60-02 no encontrada en preregistered_hypotheses.json"
    assert hyp.get('n_stop') == 30, "n_stop debe ser 30"
    assert hyp.get('estado') == 'PENDIENTE', "estado debe ser PENDIENTE al inicio"
    assert 'gated' in hyp, "Campo 'gated' obligatorio — describe condición de activación"
    assert hyp.get('gcs_extended_enabled_default') != True, (
        "gcs_extended_enabled no debe ser True por default"
    )
    dias_min = hyp.get('umbrales_congelados', {}).get('dias_min')
    dias_max = hyp.get('umbrales_congelados', {}).get('dias_max')
    assert dias_min == 22, "dias_min debe ser 22"
    assert dias_max == 42, "dias_max debe ser 42"


# ─── T61-08: Eala R4 con fix → gcs_days correcto ────────────────────────────

def test_T61_08_eala_r4_gcs_days_15_con_fix():
    """
    Simula el análisis de Eala R4 (Jul 6) con el fix aplicado.
    Con Birmingham 2026 (Jun 21): gcs_days debe ser 15, gcs_active=True.
    El boost para días 15-21 es ×1.5 (dentro del gate activo de H60-01).
    """
    historial_eala_r4 = [
        {'torneo': 'Birmingham', 'fecha': date(2026, 6, 21), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': 'Birmingham', 'fecha': date(2026, 6, 20), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': 'Birmingham', 'fecha': date(2026, 6, 18), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': 'Birmingham', 'fecha': date(2026, 6, 17), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': 'Birmingham', 'fecha': date(2026, 6, 16), 'resultado': 'win', 'superficie': 'grass'},
        # Viejo Birmingham 2025 (NO debe usarse)
        {'torneo': 'Birmingham', 'fecha': date(2025, 6, 8), 'resultado': 'win', 'superficie': 'grass'},
    ]
    fecha_partido = date(2026, 7, 6)
    result = analyze_surface_specialization(
        historial_eala_r4,
        superficie='grass',
        tier='grand_slam',
        fecha_partido=fecha_partido
    )
    assert result['gcs_days'] == 15, (
        f"gcs_days esperado: 15. Actual: {result['gcs_days']}. "
        f"El fix debe usar Birmingham 2026 (Jun 21) no 2025 (Jun 8)."
    )
    assert result['gcs_active'] is True, (
        "Con 15 días y tier=grand_slam+grass, gcs_active debe ser True"
    )
    boost = result.get('gcs_score_boost', 1.0)
    assert 1.4 <= boost <= 1.6, (
        f"Boost para 15-21d debe ser ×1.5 (rango 1.4-1.6 por posible cálculo). Actual: {boost}"
    )


# ─── T61-09: Semana 1 completa en Wimbledon F ────────────────────────────────

def test_T61_09_queen_club_activo_en_wimbledon_final():
    """
    Queen's Club (final ~Jun 15) → Wimbledon Final (Jul 12) = 27 días.
    Con season-aware gate (42d): debe estar ACTIVO (gcs_extended_active=True).
    Con _GCS_EXTENDED_ENABLED=False: boost=1.0 pero señal registrada.
    """
    historial = [
        {'torneo': "Queen's Club", 'fecha': date(2026, 6, 15), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': "Queen's Club", 'fecha': date(2026, 6, 14), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': "Queen's Club", 'fecha': date(2026, 6, 12), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': "Queen's Club", 'fecha': date(2026, 6, 11), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': "Queen's Club", 'fecha': date(2026, 6, 10), 'resultado': 'win', 'superficie': 'grass'},
    ]
    fecha_partido = date(2026, 7, 12)  # Final de Wimbledon
    result = analyze_surface_specialization(
        historial,
        superficie='grass',
        tier='grand_slam',
        fecha_partido=fecha_partido
    )
    assert result.get('gcs_extended_active') is True, (
        f"Queen's Club (27d) en Final Wimbledon debe tener gcs_extended_active=True. "
        f"gcs_days={result.get('gcs_days')}"
    )
    assert result.get('gcs_score_boost', 1.0) == 1.0, (
        "Con flag OFF, boost debe ser 1.0 aunque la señal esté detectada"
    )


# ─── T61-10: Torneo fuera de season window no activa GCS extended ──────────

def test_T61_10_torneo_fuera_de_season_window_no_gcs():
    """
    Torneo ganado el 1 Mayo (antes de Jun 1, fuera de la grass season window)
    NO debe activar GCS aunque sea en hierba y tenga <42 días.
    """
    historial = [
        {'torneo': 'Stuttgart', 'fecha': date(2026, 5, 1), 'resultado': 'win', 'superficie': 'grass'},
        {'torneo': 'Stuttgart', 'fecha': date(2026, 4, 30), 'resultado': 'win', 'superficie': 'grass'},
    ]
    fecha_partido = date(2026, 6, 2)  # 32 días después, pero Stuttgart fuera de [Jun 1, Jul 13]
    result = analyze_surface_specialization(
        historial,
        superficie='grass',
        tier='atp500',
        fecha_partido=fecha_partido
    )
    assert result['gcs_active'] is False, (
        "Torneo ganado en Mayo (fuera de grass season window) no debe activar GCS"
    )
    assert result.get('gcs_extended_active', False) is False, (
        "gcs_extended_active tampoco debe activarse fuera de la season window"
    )
```

---

## §6. Orden de Implementación (Estricto)

```
S61-1  D61-F0 — Fix bug fecha (CRÍTICO, implementar primero)
       Verificar: python3 -c "from analysis.rivalry_analyzer import ..."
       Gate: T61-01 y T61-02 deben pasar antes de continuar

S61-2  D61-F1 — Season-aware gate (_GCS_SEASON_WINDOWS + _is_gcs_active)
       Gate: T61-03, T61-04, T61-06 deben pasar

S61-3  D61-F2 + D61-F3 — LOG_GCS_SHADOW_EXTENDED + flag explícito
       Gate: T61-05 debe pasar

S61-4  D61-F6 — Auditoría y fix Keys boost negativo (si existe)
       Verificar antes de implementar con el comando de auditoría del §2.4

S61-5  D61-F4 — H60-02 en preregistered_hypotheses.json
       Gate: T61-07 debe pasar

S61-6  D61-F5 — Serialización gcs_extended_* en edge_calculator
       Gate: T61-08 debe pasar (prueba end-to-end)

S61-7  D61-F7 — Todos los tests T61-01 → T61-10
       Gate final: python -m pytest tests/test_nodo61.py -v → 10 passed, 0 failed

S61-8  Baseline regresión: python -m pytest tests/ --no-cov -q → 0 failed
       Nota: baseline antes de empezar debe ser 1659 passed
```

**PROHIBIDO en este nodo:**
- Activar `_GCS_EXTENDED_ENABLED = True` en producción
- Cambiar los multiplicadores GCS_MULT_RECENT/MID/BASE (2.2/1.8/1.5) — son de H60-01
- Modificar umbrales de H60-01 (n_stop=30, tier_min=atp500)
- Implementar la ponderación-en-origen del §3 de Nodo-60-ADDENDUM (requiere H60-01 graduado)

---

## §7. Checklist de Auditoría Fable (post-implementación)

Cuando Sonnet reporte "terminado", verificar CADA punto con el comando indicado:

```bash
# ── A61-01: Bug F0 resuelto — historial acotado a 42 días ──────────────────
python3 -m pytest tests/test_nodo61.py::test_T61_01_gcs_days_usa_birmingham_2026_no_2025 -v
# Debe: PASSED

# ── A61-02: Año anterior no contamina ──────────────────────────────────────
python3 -m pytest tests/test_nodo61.py::test_T61_02_torneo_anio_anterior_no_activa_gcs -v
# Debe: PASSED

# ── A61-03: GCS_EXTENDED_ENABLED es False en el módulo ─────────────────────
python3 -c "
from analysis.rivalry_analyzer import _GCS_EXTENDED_ENABLED
print('_GCS_EXTENDED_ENABLED:', _GCS_EXTENDED_ENABLED)
assert _GCS_EXTENDED_ENABLED is False, 'FALLO: flag debe ser False en producción'
print('OK — flag en False (producción segura)')
"

# ── A61-04: LOG_GCS_SHADOW_EXTENDED aparece en análisis de Nottingham QF ───
python3 -c "
from datetime import date
from analysis.rivalry_analyzer import analyze_surface_specialization
historial = [
    {'torneo': 'Nottingham', 'fecha': date(2026, 6, 15), 'resultado': 'win', 'superficie': 'grass'},
    {'torneo': 'Nottingham', 'fecha': date(2026, 6, 14), 'resultado': 'win', 'superficie': 'grass'},
    {'torneo': 'Nottingham', 'fecha': date(2026, 6, 12), 'resultado': 'win', 'superficie': 'grass'},
    {'torneo': 'Nottingham', 'fecha': date(2026, 6, 11), 'resultado': 'win', 'superficie': 'grass'},
    {'torneo': 'Nottingham', 'fecha': date(2026, 6, 10), 'resultado': 'win', 'superficie': 'grass'},
]
result = analyze_surface_specialization(historial, superficie='grass', tier='grand_slam', fecha_partido=date(2026,7,8))
logs = result.get('logs', [])
ext_logs = [l for l in logs if 'SHADOW_EXTENDED' in str(l)]
print('Logs GCS_SHADOW_EXTENDED:', ext_logs)
assert len(ext_logs) > 0, 'FALLO: no se encontró LOG_GCS_SHADOW_EXTENDED'
assert result.get('gcs_score_boost', 1.0) == 1.0, 'FALLO: boost debe ser 1.0 con flag OFF'
print('OK — LOG_GCS_SHADOW_EXTENDED presente, boost=1.0')
"

# ── A61-05: Eala R4 con historial correcto → gcs_days=15 ──────────────────
python3 -c "
from datetime import date
from analysis.rivalry_analyzer import analyze_surface_specialization
historial = [
    {'torneo': 'Birmingham', 'fecha': date(2026, 6, 21), 'resultado': 'win', 'superficie': 'grass'},
    {'torneo': 'Birmingham', 'fecha': date(2026, 6, 20), 'resultado': 'win', 'superficie': 'grass'},
    {'torneo': 'Birmingham', 'fecha': date(2026, 6, 18), 'resultado': 'win', 'superficie': 'grass'},
    {'torneo': 'Birmingham', 'fecha': date(2026, 6, 17), 'resultado': 'win', 'superficie': 'grass'},
    {'torneo': 'Birmingham', 'fecha': date(2026, 6, 16), 'resultado': 'win', 'superficie': 'grass'},
    {'torneo': 'Birmingham', 'fecha': date(2025, 6, 8),  'resultado': 'win', 'superficie': 'grass'},
]
result = analyze_surface_specialization(historial, superficie='grass', tier='grand_slam', fecha_partido=date(2026,7,6))
print(f'gcs_days: {result[\"gcs_days\"]}  (esperado: 15)')
print(f'gcs_active: {result[\"gcs_active\"]}  (esperado: True)')
assert result['gcs_days'] == 15, f'FALLO: gcs_days={result[\"gcs_days\"]}, esperado 15'
assert result['gcs_active'] is True, 'FALLO: debe estar activo con 15 días'
print('OK — fix funciona para Eala R4')
"

# ── A61-06: H60-02 en preregistered_hypotheses.json ──────────────────────
python3 -c "
import json
data = json.load(open('validation/preregistered_hypotheses.json'))
h = data['hypotheses'].get('H60-02')
assert h is not None, 'FALLO: H60-02 no encontrada'
assert h['estado'] == 'PENDIENTE', f'FALLO: estado={h[\"estado\"]}, esperado PENDIENTE'
assert h['umbrales_congelados']['dias_min'] == 22, 'FALLO: dias_min debe ser 22'
assert h['umbrales_congelados']['dias_max'] == 42, 'FALLO: dias_max debe ser 42'
print('H60-02:', h['nombre'])
print('Estado:', h['estado'])
print('Umbrales:', h['umbrales_congelados'])
print('OK — H60-02 correctamente pre-registrada')
"

# ── A61-07: Todos los tests del nodo pasan ────────────────────────────────
python3 -m pytest tests/test_nodo61.py -v
# Debe: 10 passed, 0 failed

# ── A61-08: Baseline de regresión — ningún test roto ─────────────────────
python3 -m pytest tests/ --no-cov -q
# Debe: 1659+ passed, 0 failed
# (puede ser más si tests previos crecieron)

# ── A61-09: Corrida real — Eala en edge report con fix ────────────────────
# (ejecutar después de tener el H2H actualizado con cuotas Kambi)
python3 edge_calculator.py --h2h reports/h2h_results_enhanced_20260705_223642.json
python3 -c "
import json
data = json.load(open(sorted(__import__('glob').glob('reports/edge_report_2026*.json'))[-1]))
all_picks = data.get('apostar',[]) + data.get('watchlist',[]) + data.get('sin_edge',[])
for p in all_picks:
    if 'eala' in p.get('partido','').lower() or 'paolini' in p.get('partido','').lower():
        print('Partido:', p.get('partido'))
        print('  gcs_bonus:', p.get('gcs_bonus'))
        print('  gcs_days:', p.get('gcs_days'), '(esperado: ~15, no 28)')
        print('  gcs_extended_active:', p.get('gcs_extended_active'))
        print('  edge:', p.get('edge_pct'))
"
# Debe: gcs_days ≈ 15, gcs_active=True

# ── A61-10: Keys boost negativo auditado ─────────────────────────────────
python3 -c "
import json, glob
for fname in sorted(glob.glob('reports/edge_report_2026*.json'))[-3:]:
    data = json.load(open(fname))
    for section in ['apostar','watchlist','sin_edge']:
        for p in data.get(section,[]):
            if 'keys' in p.get('partido','').lower():
                boost = p.get('gcs_score_boost', 1.0)
                print(f'{fname}: {p[\"partido\"]} | gcs_boost={boost}')
                if boost < 1.0:
                    print('  *** BOOST NEGATIVO DETECTADO — verificar fix D61-F6 ***')
                else:
                    print('  boost OK (>=1.0)')
"
```

---

## §8. Relación con Otros Nodos

| Nodo | Relación |
|---|---|
| [[Nodo-60-GCS-Grass-Surface-Champion-Signal]] | Fix del sistema que creó. El bug F0 es un defecto de implementación de D60-02. |
| [[Nodo-60-ADDENDUM-FABLE-Auditoria-Tres-Carriles]] | H60-01 sigue activa sin cambios. H60-02 es hipótesis hermana independiente. |
| [[Nodo-57-Penalizacion-Inactividad-Campeon-Validacion]] | D57-03 (_MIN_WINS_CHAMPION por tier) es prerequisito de GCS — no tocarlo. |
| [[Nodo-52-Shadow-Book-CLV-Tracking]] | LOG_GCS_SHADOW_EXTENDED debe alimentar el shadow book igual que LOG_GCS_SHADOW. |
| [[Nodo-58-Dashboard-Observabilidad]] | Panel 6 debe mostrar fila H60-02 con estado PENDIENTE / n_actual=0. |

---

## §9. Impacto Esperado Post-Fix

**Inmediato (este Wimbledon):**
- Eala R4: `gcs_days=15` (no 28) → `gcs_active=True` → GCS_RECENCY_BOOST ×1.5 activo
- Eala como favorita @1.58 → p_implicita=63.3% — el GCS boost mejora el score pero el mercado puede seguir sin ofrecer edge explotable en individual
- En QF/SF: si Eala llega, tendrá ~21-25 días → zona activa o extended, señal registrada en H60-02

**Medio plazo (resto de grass season):**
- Todos los campeones de Semana 1 (Nottingham/Birmingham 2026, Halle/Queens si hay champions en Wimbledon) ahora están correctamente calculados
- LOG_GCS_SHADOW_EXTENDED comienza a acumular datos para H60-02 en cada partido donde días ∈ [22, 42]

**Largo plazo:**
- Cuando H60-02 alcance n≥30 prospectivo y gradúe: `_GCS_EXTENDED_ENABLED = True` con UNA línea
- Los multiplicadores 22-28d → ×1.3, 29-35d → ×1.15, 36-42d → ×1.05 se activan
- Los campeones de Queen's Club y Halle comienzan a tener señal en SF/F de Wimbledon

---

*Spec firmado: 2026-07-05. Evidencia: edge_report_20260704 (gcs_days=13, APOSTAR) vs edge_report_20260706 (gcs_days=28, sin sección). Bug reproductible. Implementación bloqueada a Sonnet hasta confirmación de comprensión del análisis.*
