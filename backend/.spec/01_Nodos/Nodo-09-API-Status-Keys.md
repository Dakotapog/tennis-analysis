# Nodo-09: Bug de Claves de Estado — FlashScore DC_1 API

> **Wikilinks:** [[Mandatos-No-Negociables]] | [[Pipeline-Arquitectura]] | [[Sprint-Pipeline]] | [[Grafo-Dependencias-Datos]] | [[Fuentes-Datos]] | [[Nodo-05-Validacion-API]] | [[Nodo-08-File-Selection-Bug]]
> Estado: 2026-05-29 | Severidad: CRÍTICA | Tipo: Bug de Producción | Archivo: `validar_con_api.py`

---

## Descripción del Bug

**Archivo:** `validar_con_api.py` — función `obtener_resultado_partido()`
**Impacto:** `validar_con_api.py` retorna siempre `status='NS'` para partidos ya terminados.
Pipeline de validación post-partido completamente ciego — no puede detectar ningún resultado real.

```python
# ❌ BUGGY — líneas 96-100 (claves inexistentes en la respuesta real):
status_raw = datos.get('~AA', '')   # ~AA nunca existe → siempre '' → siempre NS
home_sets  = datos.get('~BH', '0') # ~BH nunca existe → siempre '0'
away_sets  = datos.get('~BI', '0') # ~BI nunca existe → siempre '0'

# ✅ FIX — claves reales del endpoint dc_1 (verificadas 2026-05-29):
dj         = datos.get('DJ', '')   # 'H'=local ganó, 'A'=visitante ganó, ''=en curso
home_sets  = datos.get('DE', '0') # sets ganados por local
away_sets  = datos.get('DF', '0') # sets ganados por visitante
```

---

## Evidencia — Respuesta Real del Endpoint

**Endpoint:** `GET https://global.flashscore.ninja/202/x/feed/dc_1_ETKIzZPG`
**Partido:** Teichmann J. vs Muchova K. — Roland Garros 2026-05-29
**Resultado real:** Teichmann ganó 2-0 (7-5, 6-?)

```
DA÷3¬DZ÷3¬DB÷3¬DD÷1780063821¬AW÷1¬DC÷1780057200¬DS÷0¬DE÷2¬
DEI÷https://...¬DF÷0¬DG÷2¬DH÷0¬DI÷-1¬DJ÷H¬DK÷1780063829¬
DL÷3¬DM÷¬DN÷7¬DO÷5¬DP÷13¬DQ÷6¬DR÷0¬DS÷0¬DT÷¬DV÷2¬
DX÷ST,MH,MC,OD,HH,TTS,DR¬AZ÷1¬A1÷950183c0...¬~
```

**Cross-validación con 2 partidos más (misma sesión):**

| Partido | mid | DJ | DE | DF | Resultado |
|---|---|---|---|---|---|
| Teichmann J. vs Muchova K. | ETKIzZPG | **H** | 2 | 0 | Teichmann ganó 2-0 ✅ |
| Tirante T.A. vs Carreno-Busta P. | CW6rxGQF | **A** | 1 | 3 | Carreno-Busta ganó 3-1 ✅ |
| Wang Xiy. vs Starodubtseva Y. | nB2WmfO0 | **H** | 2 | 0 | Wang ganó 2-0 ✅ |

**Mapa completo de claves verificadas:**

| Clave API | Significado | Valores |
|---|---|---|
| `DJ` | Ganador del partido | `'H'`=local ganó, `'A'`=visitante ganó, `''`=no terminado |
| `DE` | Sets ganados por local (jugador1) | entero como string |
| `DF` | Sets ganados por visitante (jugador2) | entero como string |
| `DC` | Unix timestamp del inicio programado | ej. `1780057200` |
| `DN` | Games del local en set 1 | entero como string |
| `DO` | Games del visitante en set 1 | entero como string |
| `DV` | Constante de tipo partido | `2` siempre en tenis (no es estado) |
| `DS` | Siempre `0` en respuesta observada | no usar para estado |
| `DR` | Siempre `0` en respuesta observada | no usar para estado |

**Por qué `DV=2` confundió inicialmente:** cuando se consultó el partido de Teichmann
durante la segunda manga, `DV=2` parecía significar "segundo set en juego". Pero en los
tres partidos ya terminados también muestra `DV=2`. Es una constante de tipo (tenis=2),
no un indicador de estado.

---

## Causa Raíz

El docstring de `parsear_respuesta_flashscore` usaba como ejemplo el formato `~AA÷100¬~BH÷2¬~BI÷1`,
que corresponde a una versión ANTERIOR o a un endpoint DIFERENTE de FlashScore. El endpoint
`dc_1_{event_id}` actual usa claves de dos letras (`DJ`, `DE`, `DF`) sin prefijo `~`.

El parser (`parsear_respuesta_flashscore`) es correcto — parsea cualquier formato `KEY÷VALUE¬`.
El bug está en `obtener_resultado_partido()` que busca claves que no existen.

---

## Fix

### `validar_con_api.py` — función `obtener_resultado_partido()`

```python
# ANTES (buggy):
status_raw = datos.get('~AA', '')
if status_raw == '100':
    home_sets = datos.get('~BH', '0')
    away_sets = datos.get('~BI', '0')
    ganador_lado = 'jugador1' if int(home_sets) > int(away_sets) else 'jugador2'
    return {'status': 'FT', ...}
if status_raw in ('0', ''):
    return {'status': 'NS', 'raw_data': datos}
return {'status': 'LIVE', 'score_parcial': datos}

# DESPUÉS (correcto):
dj = datos.get('DJ', '')
if dj in ('H', 'A'):
    home_sets = datos.get('DE', '0')
    away_sets = datos.get('DF', '0')
    ganador_lado = 'jugador1' if dj == 'H' else 'jugador2'
    return {'status': 'FT', 'ganador_lado': ganador_lado,
            'sets_local': home_sets, 'sets_visitante': away_sets, 'raw_data': datos}
# NS vs LIVE: usar DC (timestamp programado) como discriminador
try:
    dc_ts = int(datos.get('DC', '0'))
    if dc_ts and datetime.fromtimestamp(dc_ts) > datetime.now():
        return {'status': 'NS', 'raw_data': datos}
except (ValueError, TypeError):
    pass
return {'status': 'LIVE', 'score_parcial': datos}
```

### `tests/test_validacion_api.py` — 5 tests de `TestObtenerResultado`

Los mocks usaban el formato antiguo (`~AA÷100¬~BH÷2¬~BI÷1`). Actualizar a formato real:
- FT jugador1: `"DJ÷H¬DE÷2¬DF÷1¬DC÷1577836800"` (DC en el pasado)
- FT jugador2: `"DJ÷A¬DE÷0¬DF÷2¬DC÷1577836800"`
- NS: `"DJ÷¬DC÷4070908800"` (DC en el futuro, año 2099)
- LIVE: `"DJ÷¬DC÷1577836800"` (DC en el pasado, DJ vacío)

---

## Tests Nuevos Requeridos

```python
# Agregar a tests/test_validacion_api.py

def test_formato_real_dc1_endpoint():
    """Parser maneja correctamente el formato real del endpoint dc_1."""
    raw = "DA÷3¬DC÷1780057200¬DE÷2¬DF÷0¬DJ÷H¬DN÷7¬DO÷5¬DV÷2¬~"
    d = parsear_respuesta_flashscore(raw)
    assert d['DJ'] == 'H'
    assert d['DE'] == '2'
    assert d['DF'] == '0'
    assert d['DC'] == '1780057200'

def test_dv_no_es_indicador_de_estado():
    """DV=2 es constante de tipo (tenis), no indica 'segundo set en juego'."""
    # Tres partidos terminados reales muestran DV=2 todos — no es estado
    raw_ft = "DJ÷H¬DE÷2¬DF÷0¬DC÷1577836800¬DV÷2"
    with patch('validar_con_api.requests.get') as mock_get:
        mock = MagicMock()
        mock.text = raw_ft
        mock.raise_for_status.return_value = None
        mock_get.return_value = mock
        r = obtener_resultado_partido('ETKIzZPG')
    assert r['status'] == 'FT'   # DV=2 NO debe interpretarse como LIVE
    assert r['ganador_lado'] == 'jugador1'
```

---

## Tareas

| ID | Tarea | Estado |
|---|---|---|
| T09-01 | Actualizar `obtener_resultado_partido()`: `~AA`→`DJ`, `~BH`→`DE`, `~BI`→`DF` | ✅ 2026-05-29 |
| T09-02 | Actualizar mocks en 5 tests de `TestObtenerResultado` al formato real | ✅ 2026-05-29 |
| T09-03 | Añadir `test_formato_real_dc1_endpoint` | ✅ 2026-05-29 |
| T09-04 | Añadir `test_dv_no_es_indicador_de_estado` | ✅ 2026-05-29 |
| T09-05 | Verificar `pytest tests/ --no-cov -q` → ≥699 passed | ✅ 2026-05-29 |

---

## Vinculación

- [[Nodo-05-Validacion-API]] — este bug vive en el script del Nodo-05; es su único bloqueante restante
- [[Fuentes-Datos]] — FlashScore Ninja API dc_1: claves `DJ`/`DE`/`DF` son el nuevo contrato documentado
- [[Grafo-Dependencias-Datos]] — S6_RESULTADO_REAL depende de este fix para cerrarse
- [[Sprint-Pipeline]] — T09-01 a T09-05 completan la validación post-partido
- [[Mandatos-No-Negociables]] — Mandato 6: tests antes que código, fix documentado con evidencia real
- [[Nodo-08-File-Selection-Bug]] — bug análogo: clave incorrecta en búsqueda de datos
