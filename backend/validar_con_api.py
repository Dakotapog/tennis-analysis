"""
Nodo-05 — Validación Post-Partido con FlashScore Ninja API

Cierra el loop P&L: compara predicciones del modelo con resultados reales.

Flujo:
  1. Lee h2h_results_enhanced_FECHA.json (predicciones)
  2. Consulta dc_1_{event_id} por cada partido con match_id real
  3. Compara prediccion vs resultado real → accuracy
  4. Segmenta accuracy por superficie
  5. Exporta resultados_finales_FECHA.json + calibracion actualizada

API confirmada: dc_1_{event_id} → HTTP 200 para tenis
Auth: X-Fsign: SW9D1eZo | Referer: https://www.flashscore.co/
H2H endpoints → 404 para tenis (Playwright sigue siendo necesario para H2H)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

import requests

# ──────────────────────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────────────────────

FLASHSCORE_BASE = "https://global.flashscore.ninja/202/x/feed"
HEADERS = {
    "X-Fsign": "SW9D1eZo",
    "Referer": "https://www.flashscore.co/",
    "Origin": "https://www.flashscore.co",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "*/*",
}
DELAY_ENTRE_REQUESTS = 0.5   # segundos — no martillar la API
CALIBRACION_FILE = "data/calibracion_edge.json"


# ──────────────────────────────────────────────────────────────────────────────
# Parser del formato propietario FlashScore
# ──────────────────────────────────────────────────────────────────────────────

def parsear_respuesta_flashscore(raw: str) -> dict:
    """
    Convierte el formato propietario KEY÷VALUE¬KEY÷VALUE en un dict.

    Ejemplo (formato real dc_1, verificado 2026-05-29):
        "DA÷3¬DC÷1780057200¬DE÷2¬DF÷0¬DJ÷H¬DV÷2¬~"
        → {'DA': '3', 'DC': '1780057200', 'DE': '2', 'DF': '0', 'DJ': 'H', 'DV': '2'}

    Claves relevantes del endpoint dc_1_{event_id} para tenis:
        DJ = ganador: 'H'=local ganó, 'A'=visitante ganó, ''=no terminado
        DE = sets ganados por local (jugador1)
        DF = sets ganados por visitante (jugador2)
        DC = Unix timestamp del inicio programado
        DV = constante de tipo partido (2=tenis, no es indicador de estado)
    """
    datos: dict = {}
    for par in raw.split('¬'):
        if '÷' in par:
            k, v = par.split('÷', 1)
            datos[k] = v
    return datos


# ──────────────────────────────────────────────────────────────────────────────
# Consulta a la API
# ──────────────────────────────────────────────────────────────────────────────

def obtener_resultado_partido(event_id: str) -> dict:
    """
    Consulta dc_1_{event_id} y extrae ganador + estado del partido.

    Retorna:
        status:        'FT' | 'LIVE' | 'NS' | 'ERROR' | 'UNKNOWN'
        ganador_lado:  'jugador1' | 'jugador2' | None
        sets_local:    str  (número de sets ganados por el local)
        sets_visitante: str
        raw_data:      dict completo para debugging
    """
    if not event_id or event_id in ('tennis', ''):
        return {'status': 'INVALID_ID', 'error': f'match_id inválido: {event_id!r}'}

    url = f"{FLASHSCORE_BASE}/dc_1_{event_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        raw = r.text
    except requests.exceptions.HTTPError as e:
        return {'status': 'ERROR', 'error': f'HTTP {e.response.status_code}'}
    except Exception as e:
        return {'status': 'ERROR', 'error': str(e)}

    datos = parsear_respuesta_flashscore(raw)

    # Claves reales del endpoint dc_1 (verificadas 2026-05-29 — Nodo-09):
    # DJ = ganador: 'H'=local ganó, 'A'=visitante ganó, ''=no terminado
    # DE = sets ganados por local, DF = sets ganados por visitante
    # DC = Unix timestamp del inicio programado (para distinguir NS vs LIVE)
    dj = datos.get('DJ', '')

    if dj in ('H', 'A'):
        home_sets = datos.get('DE', '0')
        away_sets = datos.get('DF', '0')
        ganador_lado = 'jugador1' if dj == 'H' else 'jugador2'
        return {
            'status': 'FT',
            'ganador_lado': ganador_lado,
            'sets_local': home_sets,
            'sets_visitante': away_sets,
            'raw_data': datos,
        }

    # DJ vacío: no iniciado o en curso — usar DC para distinguir
    try:
        dc_ts = int(datos.get('DC', '0'))
        if dc_ts and datetime.fromtimestamp(dc_ts) > datetime.now():
            return {'status': 'NS', 'raw_data': datos}
    except (ValueError, TypeError):
        pass

    return {'status': 'LIVE', 'score_parcial': datos}


# ──────────────────────────────────────────────────────────────────────────────
# Validación individual (testeable sin I/O)
# ──────────────────────────────────────────────────────────────────────────────

def validar_partido_individual(partido: dict, resultado_api: Optional[dict] = None) -> Optional[dict]:
    """
    Valida UN partido comparando predicción con resultado real.

    Args:
        partido:       dict del h2h_results_enhanced (con ranking_analysis, match_id, etc.)
        resultado_api: si se pasa, evita la llamada HTTP (útil en tests)

    Retorna None si no se puede validar (match_id inválido, partido no terminado,
    sin predicción, etc.).
    """
    match_id = partido.get('match_id')
    if not match_id or match_id in ('tennis', '', None):
        return None

    pred = partido.get('ranking_analysis', {}).get('prediction', {})
    favorito_pred = pred.get('favored_player')
    if not favorito_pred:
        return None

    if resultado_api is None:
        resultado_api = obtener_resultado_partido(match_id)

    if resultado_api.get('status') != 'FT':
        return None

    lado_ganador = resultado_api.get('ganador_lado')
    if lado_ganador is None:
        return None

    jugador1 = partido.get('jugador1', '')
    jugador2 = partido.get('jugador2', '')
    ganador_real = jugador1 if lado_ganador == 'jugador1' else jugador2

    correcto = (favorito_pred.strip().lower() == ganador_real.strip().lower())

    return {
        'partido': f"{jugador1} vs {jugador2}",
        'prediccion': favorito_pred,
        'confianza': pred.get('confidence'),
        'resultado_real': ganador_real,
        'correcto': correcto,
        'match_id': match_id,
        'torneo': partido.get('torneo', 'Desconocido'),
        'superficie': partido.get('superficie', 'unknown'),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Métricas agregadas (testeables sin I/O)
# ──────────────────────────────────────────────────────────────────────────────

def calcular_accuracy(resultados: list[dict]) -> float:
    """Accuracy global: fracción de predicciones correctas."""
    if not resultados:
        return 0.0
    return sum(1 for r in resultados if r.get('correcto')) / len(resultados)


def accuracy_por_superficie(resultados: list[dict]) -> dict:
    """
    Segmenta accuracy por superficie.

    Retorna:
        {'clay': {'accuracy': 0.62, 'n': 18, 'correctas': 11}, ...}
    """
    por_sup: dict = defaultdict(lambda: {'correctas': 0, 'total': 0})
    for r in resultados:
        sup = r.get('superficie', 'unknown')
        por_sup[sup]['total'] += 1
        if r.get('correcto'):
            por_sup[sup]['correctas'] += 1

    return {
        sup: {
            'accuracy': round(v['correctas'] / v['total'], 4) if v['total'] > 0 else 0.0,
            'n': v['total'],
            'correctas': v['correctas'],
        }
        for sup, v in por_sup.items()
    }


# ──────────────────────────────────────────────────────────────────────────────
# Actualización de calibración (feed para edge_calculator.py)
# ──────────────────────────────────────────────────────────────────────────────

def actualizar_calibracion_desde_resultados(resultados: list[dict]) -> None:
    """
    Actualiza data/calibracion_edge.json con los resultados validados.
    Conecta CX-06: accuracy real → p_historica en Kelly-KL.
    """
    if not resultados:
        return

    try:
        with open(CALIBRACION_FILE) as f:
            cal = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cal = {"global": {"wins": 0, "losses": 0}, "por_superficie": {}}

    for r in resultados:
        if r.get('correcto') is None:
            continue
        sup = r.get('superficie', 'unknown')
        key = 'wins' if r['correcto'] else 'losses'

        cal.setdefault('global', {'wins': 0, 'losses': 0})
        cal['global'][key] += 1

        cal.setdefault('por_superficie', {})
        cal['por_superficie'].setdefault(sup, {'wins': 0, 'losses': 0})
        cal['por_superficie'][sup][key] += 1

    cal['ultima_actualizacion'] = datetime.now().isoformat()

    os.makedirs(os.path.dirname(CALIBRACION_FILE), exist_ok=True)
    with open(CALIBRACION_FILE, 'w') as f:
        json.dump(cal, f, indent=2, ensure_ascii=False)

    print(f"  Calibración actualizada → {CALIBRACION_FILE}")


# ──────────────────────────────────────────────────────────────────────────────
# Orquestador principal
# ──────────────────────────────────────────────────────────────────────────────

def validar_predicciones(h2h_file: str, output_file: str, actualizar_cal: bool = True) -> dict:
    """
    Lee h2h_results_enhanced, consulta API, calcula accuracy y exporta JSON.

    Args:
        h2h_file:       ruta a h2h_results_enhanced_FECHA.json
        output_file:    ruta de salida para resultados_finales_FECHA.json
        actualizar_cal: si True, actualiza calibracion_edge.json (CX-06)
    """
    with open(h2h_file, encoding='utf-8') as f:
        raw = json.load(f)
    partidos = raw.get('partidos', raw) if isinstance(raw, dict) else raw

    print(f"Validando {len(partidos)} partidos desde {h2h_file} ...")

    resultados: list[dict] = []
    saltados = 0

    for i, partido in enumerate(partidos, 1):
        match_id = partido.get('match_id')
        if not match_id or match_id in ('tennis', ''):
            saltados += 1
            continue

        resultado_api = obtener_resultado_partido(match_id)

        r = validar_partido_individual(partido, resultado_api)
        if r is None:
            # No terminado, sin predicción, o match_id inválido
            if resultado_api.get('status') not in ('NS', 'LIVE'):
                saltados += 1
            continue

        resultados.append(r)
        estado = '✅' if r['correcto'] else '❌'
        print(f"  [{i:3d}] {estado} {r['partido']} → {r['resultado_real']} (pred: {r['prediccion']})")

        time.sleep(DELAY_ENTRE_REQUESTS)

    accuracy = calcular_accuracy(resultados)
    por_sup = accuracy_por_superficie(resultados)

    output = {
        'fecha_validacion': datetime.now().isoformat(),
        'fuente_h2h': h2h_file,
        'total_partidos': len(partidos),
        'total_validados': len(resultados),
        'saltados': saltados,
        'correctas': sum(1 for r in resultados if r['correcto']),
        'accuracy': round(accuracy, 4),
        'accuracy_por_superficie': por_sup,
        'partidos': resultados,
    }

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Accuracy: {accuracy*100:.1f}% ({output['correctas']}/{output['total_validados']})")
    for sup, datos in por_sup.items():
        print(f"   {sup:8s}: {datos['accuracy']*100:.1f}% (n={datos['n']})")
    print(f"   Exportado → {output_file}")

    if actualizar_cal and resultados:
        actualizar_calibracion_desde_resultados(resultados)

    return output


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _find_latest_h2h() -> Optional[str]:
    """Encuentra el h2h_results_enhanced más reciente en reports/."""
    import glob
    files = glob.glob('reports/h2h_results_enhanced_*.json')
    return max(files, key=os.path.getmtime) if files else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nodo-05 — Validación post-partido FlashScore')
    parser.add_argument('--h2h', help='Ruta a h2h_results_enhanced_FECHA.json (auto si omite)')
    parser.add_argument('--output', help='Ruta de salida (auto si omite)')
    parser.add_argument('--no-cal', action='store_true', help='No actualizar calibracion_edge.json')
    args = parser.parse_args()

    h2h_file = args.h2h or _find_latest_h2h()
    if not h2h_file:
        print("❌ No se encontró ningún h2h_results_enhanced_*.json en reports/")
        raise SystemExit(1)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = args.output or f'reports/resultados_finales_{ts}.json'

    validar_predicciones(h2h_file, output_file, actualizar_cal=not args.no_cal)
