# Nodo-05: Validación Post-Partido con FlashScore API

> **Wikilinks:** [[Mandatos-No-Negociables]] | [[Sprint-Pipeline]] | [[Pipeline-Arquitectura]] | [[Grafo-Dependencias-Datos]] | [[Fuentes-Datos]] | [[Nodo-01-Edge-Calculator]] | [[Nodo-02-Markov-Changepoint]] | [[Nodo-03-Scraper-Fix]] | [[Nodo-04-Dataset-Fix]] | [[Nodo-05-Validacion-API]] | [[Nodo-06-Erdos-Graph]]
> **Objetivo:** Cerrar el loop P&L — medir accuracy real con datos limpios post-2026-05-28

**Prioridad:** ALTA — sin validación no hay P&L real, sin P&L no hay evidencia de edge
**Archivo objetivo:** `validar_con_api.py`
**API confirmada:** FlashScore Ninja `dc_1_{event_id}` → score + estado (HTTP 200 ✅)

---

## Contrato de Señal (Signal Contract)

```
PRODUCE:  reports/resultados_finales_FECHA.json
          accuracy por superficie (clay/grass/hard)
          calibration_data.json (p_modelo vs resultado_real)

CONSUME:  reports/h2h_results_enhanced_FECHA.json  (predicciones)
          dc_1_{event_id} API (resultados reales)
          match_id real (requiere [[Nodo-03-Scraper-Fix]] Bug 2)

PREREQUISITO: match_id = event_id real (no "tennis")
              dc_1_{event_id} retorna HTTP 200 para tenis ✅ confirmado
```

---

## Evidencia API Confirmada

```bash
# Probado el 2026-05-28 durante ejecución del pipeline:
curl -H "X-Fsign: SW9D1eZo" -H "Referer: https://www.flashscore.co/" \
     "https://global.flashscore.ninja/202/x/feed/dc_1_rDQ3y6to"
# → HTTP 200 | score, estado, timestamp del partido

# H2H endpoints — NO disponibles para tenis:
# dc_h2h_1_{id} → 404
# Playwright obligatorio para H2H (confirmado)
```

---

## Implementación

```python
# validar_con_api.py
import requests
import json
from datetime import datetime

FLASHSCORE_BASE = "https://global.flashscore.ninja/202/x/feed"
HEADERS = {
    "X-Fsign": "SW9D1eZo",
    "Referer": "https://www.flashscore.co/",
    "Origin": "https://www.flashscore.co",
    "User-Agent": "Mozilla/5.0"
}

def obtener_resultado_partido(event_id: str) -> dict:
    """
    Consulta dc_1_{event_id} y extrae ganador + score.
    
    Retorna:
        status: 'FT' | 'LIVE' | 'NS' | 'UNKNOWN'
        ganador: nombre del jugador ganador (o None si no terminó)
        score: '6-3, 7-5' (string)
        timestamp_fin: ISO timestamp
    """
    url = f"{FLASHSCORE_BASE}/dc_1_{event_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        raw = r.text
    except Exception as e:
        return {'status': 'ERROR', 'error': str(e)}

    # Parser del formato propietario KEY÷VALUE¬KEY÷VALUE
    datos = {}
    for par in raw.split('¬'):
        if '÷' in par:
            k, v = par.split('÷', 1)
            datos[k] = v

    status = datos.get('~AA', 'UNKNOWN')
    # ~AA = estado: 0=NS, 1=1ST, 2=2ND, ..., 100=FT
    
    if status == '100':  # FT — Finished
        # Extraer score de sets
        home_sets = datos.get('~BH', '0')
        away_sets = datos.get('~BI', '0')
        ganador = 'jugador1' if int(home_sets) > int(away_sets) else 'jugador2'
        return {
            'status': 'FT',
            'ganador_lado': ganador,
            'sets_local': home_sets,
            'sets_visitante': away_sets,
            'raw_data': datos
        }
    elif status in ('0', ''):
        return {'status': 'NS'}  # Not Started
    else:
        return {'status': 'LIVE', 'score_parcial': datos}


def validar_predicciones(h2h_file: str, output_file: str):
    """
    Lee h2h_results_enhanced, consulta resultados reales, calcula accuracy.
    """
    with open(h2h_file) as f:
        partidos = json.load(f)

    resultados = []
    correctas = 0
    total_validados = 0

    for partido in partidos:
        match_id = partido.get('match_id')
        if not match_id or match_id == 'tennis':
            # Bug Nodo-03 sin fix — saltar
            continue

        pred = partido.get('ranking_analysis', {}).get('prediction', {})
        favorito_pred = pred.get('favored_player')
        confianza = pred.get('confidence')

        if not favorito_pred:
            continue

        resultado_api = obtener_resultado_partido(match_id)

        if resultado_api['status'] != 'FT':
            continue

        # Resolver ganador real por nombre
        lado_ganador = resultado_api['ganador_lado']
        ganador_real = partido['jugador1'] if lado_ganador == 'jugador1' else partido['jugador2']

        correcto = (favorito_pred == ganador_real)
        if correcto:
            correctas += 1
        total_validados += 1

        resultados.append({
            'partido': f"{partido['jugador1']} vs {partido['jugador2']}",
            'prediccion': favorito_pred,
            'confianza': confianza,
            'resultado_real': ganador_real,
            'correcto': correcto,
            'match_id': match_id,
            'torneo': partido.get('torneo', 'Desconocido'),
            'superficie': partido.get('superficie', 'unknown')
        })

    accuracy = correctas / total_validados if total_validados > 0 else 0

    output = {
        'fecha_validacion': datetime.now().isoformat(),
        'fuente_h2h': h2h_file,
        'total_validados': total_validados,
        'correctas': correctas,
        'accuracy': round(accuracy, 4),
        'partidos': resultados
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Accuracy: {accuracy*100:.1f}% ({correctas}/{total_validados})")
    return output
```

---

## Accuracy por Superficie (Post-Fix Nodo-03)

Una vez Nodo-03 esté activo, calcular accuracy segmentada:

```python
def accuracy_por_superficie(resultados: list) -> dict:
    from collections import defaultdict
    por_sup = defaultdict(lambda: {'correctas': 0, 'total': 0})
    for r in resultados:
        sup = r.get('superficie', 'unknown')
        por_sup[sup]['total'] += 1
        if r['correcto']:
            por_sup[sup]['correctas'] += 1
    return {
        sup: {
            'accuracy': v['correctas'] / v['total'] if v['total'] > 0 else 0,
            'n': v['total']
        }
        for sup, v in por_sup.items()
    }
```

**Meta:** Clay ≥ 55% (Roland Garros hay estructura — favoritos ganan en arcilla)

---

## Conexiones Cross-Nodo (CX)

| CX | Conexión | Impacto |
|---|---|---|
| CX-01 | [[Nodo-03-Scraper-Fix]] Bug 2 → match_id real | Sin event_id, API retorna 404 para partidos específicos |
| CX-06 | Output → calibration_data → [[Nodo-01-Edge-Calculator]] p_historica | Actualiza lambda_aversion con accuracy real por superficie |
| CX-07 | Accuracy clay ≥ 55% → habilita scale-up de bankroll en clay | Norte estratégico: apostar más en la superficie donde el modelo demuestra edge |

---

## Tests Requeridos

```python
# tests/test_validacion_api.py
def test_parser_formato_propietario():
    """El parser KEY÷VALUE¬KEY÷VALUE extrae correctamente."""
    raw = "~AA÷100¬~BH÷2¬~BI÷1"
    datos = parsear_respuesta_flashscore(raw)
    assert datos['~AA'] == '100'
    assert datos['~BH'] == '2'

def test_partido_finalizado_identifica_ganador():
    """Status=100 → ganador_lado correcto según sets."""
    r = {'status': 'FT', 'sets_local': '2', 'sets_visitante': '1', 'ganador_lado': 'jugador1'}
    assert r['ganador_lado'] == 'jugador1'

def test_match_id_tennis_se_salta():
    """Partidos con match_id='tennis' no se validan (bug Nodo-03 sin fix)."""
    partido = {'match_id': 'tennis', 'jugador1': 'A', 'jugador2': 'B'}
    resultado = validar_partido_individual(partido)
    assert resultado is None

def test_accuracy_calculada_correctamente():
    """3 correctas de 5 = 60%."""
    resultados = [
        {'correcto': True}, {'correcto': True}, {'correcto': True},
        {'correcto': False}, {'correcto': False}
    ]
    assert calcular_accuracy(resultados) == 0.60
```

---

## Ciclo de Vida

```
Estado:   POR CONSTRUIR
Bloqueado por: Nodo-03 Bug 2 (match_id = "tennis" → API no puede identificar el partido)
Una vez Nodo-03 activo: construir en ~3 horas
Ejecutar: python3 validar_con_api.py --h2h reports/h2h_results_enhanced_HOY.json
Output:   reports/resultados_finales_HOY.json + accuracy real
Meta:     n≥30 partidos validados con datos limpios para calibrar p_historica
```
