"""
core/combo_exclusions.py — D173-10 (Nodo-173, BLOQUE D).

Observabilidad del gate Kambi en los combo builders.

Problema (Nodo-173 §1.11 Caso C): `kambi_disponible=False` excluye picks de TODOS
los combo builders sin dejar rastro consultable. Cuando el usuario pregunta "el
pipeline vio 268 partidos y los combos no armaron nada, ¿por qué?", la respuesta
requería una sesión de depuración en vez de un archivo.

Este módulo NO cambia el comportamiento del gate — el filtro Kambi es correcto:
no se puede apostar lo que la casa no lista. Solo lo hace **auditable**.

Contrato:
  - Append-only sobre `reports/combo_exclusions_{YYYYMMDD}.json`.
  - Una entrada por (builder, corrida). Varias corridas del mismo builder en el
    mismo día se acumulan — el archivo es un log, no un snapshot.
  - Fail-soft absoluto: cualquier excepción de I/O se traga. Un problema de
    observabilidad NUNCA puede tumbar la generación de combos.

Consumido por `scripts/funnel_report.py` (D173-11).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Iterable, Optional

__all__ = ['exclusion_record', 'registrar_exclusiones', 'ruta_exclusiones',
           'leer_exclusiones']

REPORTS_DIR = 'reports'


def ruta_exclusiones(fecha_compact: Optional[str] = None) -> str:
    """Ruta del archivo de exclusiones del día (formato YYYYMMDD)."""
    fc = fecha_compact or datetime.now().strftime('%Y%m%d')
    return os.path.join(REPORTS_DIR, f'combo_exclusions_{fc}.json')


def exclusion_record(pick: Any, motivo: str) -> dict:
    """Normaliza un pick (dict del edge_report) a un registro de exclusión.

    Tolera dicts incompletos y objetos que no son dict — en ese caso registra
    solo el motivo, que sigue siendo información útil para el conteo.
    """
    if not isinstance(pick, dict):
        return {'partido': str(pick)[:120], 'motivo': motivo}

    partido = (pick.get('partido')
               or pick.get('match')
               or pick.get('jugador')
               or pick.get('favorito')
               or pick.get('nombre')
               or '?')

    rec: dict = {'partido': str(partido)[:120], 'motivo': motivo}

    cuota = pick.get('cuota_favorito', pick.get('cuota'))
    if cuota is not None:
        try:
            rec['cuota'] = round(float(cuota), 2)
        except (TypeError, ValueError):
            pass

    p_mod = pick.get('p_modelo')
    if p_mod is not None:
        try:
            rec['p_modelo'] = round(float(p_mod), 3)
        except (TypeError, ValueError):
            pass

    for k in ('edge_pct', 'tier', 'torneo_nombre'):
        v = pick.get(k)
        if v is not None:
            rec[k] = v

    return rec


def registrar_exclusiones(builder: str,
                          excluidos: Iterable[Any],
                          motivo: str = 'kambi_no_disponible',
                          fecha_compact: Optional[str] = None) -> int:
    """Anexa las exclusiones de un builder al log del día. Retorna cuántas escribió.

    `excluidos` puede ser una lista de picks (dicts del edge_report) o de
    registros ya normalizados (dicts con clave 'motivo').

    Fail-soft: ante cualquier error retorna 0 sin propagar.
    """
    try:
        registros = []
        for item in (excluidos or []):
            if isinstance(item, dict) and 'motivo' in item and 'partido' in item:
                registros.append(item)
            else:
                registros.append(exclusion_record(item, motivo))

        if not registros:
            return 0

        path = ruta_exclusiones(fecha_compact)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        data: dict = {}
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                data = {}
        if not isinstance(data, dict):
            data = {}
        entradas = data.get('entradas')
        if not isinstance(entradas, list):
            entradas = []

        entradas.append({
            'builder':   builder,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'n':         len(registros),
            'excluidos': registros,
        })
        data['entradas'] = entradas
        data['generado'] = datetime.now().isoformat(timespec='seconds')

        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

        return len(registros)
    except Exception:  # noqa: BLE001 — observabilidad nunca tumba el pipeline
        return 0


def leer_exclusiones(fecha_compact: Optional[str] = None) -> list:
    """Lee las entradas del día. Retorna [] si no hay archivo o está corrupto."""
    path = ruta_exclusiones(fecha_compact)
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        entradas = data.get('entradas') if isinstance(data, dict) else None
        return entradas if isinstance(entradas, list) else []
    except (json.JSONDecodeError, OSError):
        return []
