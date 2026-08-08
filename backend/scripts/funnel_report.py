"""
scripts/funnel_report.py — D173-11 (Nodo-173, BLOQUE E).

Reporte diario del embudo de decisión. Responde la queja de fondo del usuario:
"hoy no hubo señales" no es una respuesta — es una no-respuesta.

Consume:
  - `metadata.funnel` del edge_report del día (D173-01: gate_ledger/gate_bloqueante
    por pick, agregado por `calcular_edge_completo`).
  - `reports/combo_exclusions_{fecha}.json` (D173-10: exclusiones por gate Kambi
    en los combo builders — partidos que SÍ pasaron edge_calculator pero no
    llegaron a un combo).

Requisito de diseño explícito (spec §D173-11, no negociable): el reporte NUNCA
puede terminar sin contenido accionable. Cuando SOBREVIVEN=0, la sección
"LOS 3 QUE MÁS CERCA ESTUVIERON" es obligatoria, con la distancia exacta al
umbral que los bloqueó. Cumple `feedback_zero_response_prohibition` /
MANDATO-01→06 de Nodo-89.

No es un gate. No decide nada. Solo hace legible lo que `calcular_edge_completo`
ya calculó y `combo_exclusions` ya registró.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPORTS_DIR = 'reports'

# Distancias al umbral se calculan SOLO para los 3 gates numéricos de
# calcular_edge (D173-01). Los demás gates (phantom, N28F2, contaminación...)
# son binarios/estructurales — no tienen una "distancia en pp" comparable, así
# que se muestran con su motivo textual tal cual lo escribió registrar_gate().
try:
    from edge_calculator import EDGE_MIN, KELLY_KL_MIN
except Exception:  # pragma: no cover — el reporte no puede depender de imports frágiles
    EDGE_MIN, KELLY_KL_MIN = 0.05, 0.02

try:
    from config import P_MODELO_MIN_UNDERDOG
except Exception:  # pragma: no cover
    P_MODELO_MIN_UNDERDOG = 0.55


def _fecha_compact(fecha: Optional[str] = None) -> str:
    return fecha or datetime.now().strftime('%Y%m%d')


def find_latest_edge_report(fecha_compact: Optional[str] = None,
                            directory: str = REPORTS_DIR) -> Optional[str]:
    """Ubica el edge_report completo (NO el kambi-filtrado) más reciente del día.

    El funnel de D173-01 se calcula sobre TODOS los partidos procesados, antes
    del filtro Kambi (D141) — usar el kambi-only aquí subreportaría el embudo.
    """
    fc = _fecha_compact(fecha_compact)
    candidatos = sorted(
        f for f in glob.glob(os.path.join(directory, f'edge_report_{fc}*.json'))
        if 'kambi' not in os.path.basename(f)
    )
    return candidatos[-1] if candidatos else None


def _cargar_json(path: str) -> Optional[dict]:
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _distancia_a_umbral(pick: dict) -> Optional[tuple]:
    """Retorna (distancia_pp, etiqueta) para el gate bloqueante de `pick`, o
    None si el gate no admite una distancia numérica comparable.

    Distancia siempre en PUNTOS PORCENTUALES, siempre positiva (qué tan lejos
    estaba de cruzar el umbral) — nunca negativa, nunca None disfrazado de 0.
    """
    gate = pick.get('gate_bloqueante')
    if gate == 'G_T32_01':
        p_modelo = pick.get('p_modelo')
        if p_modelo is None:
            return None
        dist = (P_MODELO_MIN_UNDERDOG - float(p_modelo)) * 100
        return (max(dist, 0.0), f'p_modelo {float(p_modelo)*100:.1f}% (faltaron {dist:.1f}pp)')
    if gate == 'G_EDGE_MIN':
        edge = pick.get('edge')
        if edge is None:
            return None
        dist = (EDGE_MIN - float(edge)) * 100
        return (max(dist, 0.0), f'edge {float(edge)*100:.1f}% (faltaron {dist:.1f}pp)')
    if gate == 'G_KELLY_MIN':
        kelly = pick.get('kelly_kl_ajustado')
        if kelly is None:
            return None
        dist = (KELLY_KL_MIN - float(kelly)) * 100
        return (max(dist, 0.0), f'kelly_kl {float(kelly)*100:.2f}% (faltaron {dist:.2f}pp)')
    return None


def _nombre_partido(pick: dict) -> str:
    # `partido` es el campo real serializado por calcular_edge (D173-01 output).
    # jugador1/jugador2/favorito son fallback para inputs de test o formatos antiguos.
    partido = pick.get('partido')
    if partido:
        return str(partido)
    j1 = pick.get('jugador1') or pick.get('favorito_predicho') or pick.get('favorito') or '?'
    j2 = pick.get('jugador2') or pick.get('rival') or ''
    return f'{j1} vs {j2}' if j2 else str(j1)


def picks_mas_cerca(watchlist: list, sin_edge: list, top_n: int = 3) -> list:
    """D173-11: candidatos con distancia numérica al umbral, ordenados asc.

    Fallback para gates sin distancia numérica (phantom/N28F2/contaminación):
    se incluyen al final con motivo textual, para que la sección NUNCA quede
    vacía mientras haya picks bloqueados de algún tipo.
    """
    con_distancia = []
    sin_distancia = []
    for p in (watchlist or []) + (sin_edge or []):
        if not isinstance(p, dict):
            continue
        d = _distancia_a_umbral(p)
        if d is not None:
            con_distancia.append((d[0], _nombre_partido(p), d[1]))
        else:
            motivo = None
            ledger = p.get('gate_ledger') or []
            for entry in ledger:
                if isinstance(entry, dict) and entry.get('gate') == p.get('gate_bloqueante'):
                    motivo = entry.get('motivo')
                    break
            sin_distancia.append((float('inf'), _nombre_partido(p), motivo or p.get('motivo_reclasificacion') or '?'))

    con_distancia.sort(key=lambda x: x[0])
    resultado = con_distancia[:top_n]
    if len(resultado) < top_n:
        resultado += sin_distancia[:top_n - len(resultado)]
    return resultado


_mas_cerca = picks_mas_cerca  # alias D176-01 — nombre interno preservado


def _barra(n: int, max_n: int, ancho: int = 20) -> str:
    if max_n <= 0:
        return ''
    largo = max(1, round((n / max_n) * ancho)) if n > 0 else 0
    return '▓' * largo


def generar_reporte(fecha_compact: Optional[str] = None,
                    edge_report_path: Optional[str] = None) -> str:
    """Construye el reporte de embudo como texto. Nunca retorna string vacío."""
    fc = _fecha_compact(fecha_compact)
    path = edge_report_path or find_latest_edge_report(fc)

    if not path:
        return (f'EMBUDO {fc}\n'
                f'{"─"*70}\n'
                f'  Sin edge_report para hoy — el pipeline PASO 3 no ha corrido aún,\n'
                f'  o corrió con 0 partidos. ACCIÓN: correr `python3 edge_calculator.py`.\n')

    data = _cargar_json(path)
    if not isinstance(data, dict):
        return (f'EMBUDO {fc}\n'
                f'{"─"*70}\n'
                f'  edge_report en {path} no se pudo leer (corrupto o vacío).\n'
                f'  ACCIÓN: revisar {path} o re-correr `python3 edge_calculator.py`.\n')

    metadata = data.get('metadata') or {}
    funnel = metadata.get('funnel') or {}
    n_procesados = funnel.get('n_procesados', 0)
    por_gate = funnel.get('por_gate') or {}
    n_sobrevive = funnel.get('n_sobrevive', 0)

    from core.combo_exclusions import leer_exclusiones
    excl_entradas = leer_exclusiones(fc)
    n_kambi_excl = sum(e.get('n', 0) for e in excl_entradas if isinstance(e, dict))

    lineas = []
    lineas.append(f'EMBUDO {fc}{" "*10}{n_procesados} partidos analizados')
    lineas.append('─' * 70)

    if n_procesados == 0:
        lineas.append('  edge_report existe pero procesó 0 partidos.')
        lineas.append('  ACCIÓN: revisar PASO 1/PASO 2 (extracción de partidos/H2H).')
        return '\n'.join(lineas) + '\n'

    max_n = max([n_procesados] + list(por_gate.values()) + [n_kambi_excl]) or 1
    _ETIQUETAS = {
        'G_EDGE_MIN':         'edge <= 5%',
        'G_KELLY_MIN':        'kelly_kl <= 2%',
        'G_T32_01':           'T32-01 / p_modelo',
        'G_N28F2':            'N28F2 (n_axes < 2)',
        'G_PHANTOM':          'phantom identity',
        'G_HIST_CONTAM':      'contaminación historial',
        'G_ELO_INCOHERENTE':  'ELO-ranking incoherente',
        'G_SIN_DATOS':        'sin datos',
        'G_HOT_SIN_BBI':      'hot sin BBI',
        'G_T33_01':           'T33-01',
    }
    for gate, n in por_gate.items():
        etiqueta = _ETIQUETAS.get(gate, gate)
        lineas.append(f'  {etiqueta:<28} {n:>5}   {_barra(n, max_n)}')
    if n_kambi_excl:
        lineas.append(f'  {"kambi no disponible":<28} {n_kambi_excl:>5}   {_barra(n_kambi_excl, max_n)}')

    lineas.append('  ' + '─' * 34)
    lineas.append(f'  SOBREVIVEN{" "*17}{n_sobrevive:>5}')
    lineas.append('')

    watchlist = data.get('watchlist') or []
    sin_edge = data.get('sin_edge') or []
    cercanos = _mas_cerca(watchlist, sin_edge)

    # Requisito duro D173-11: esta sección NUNCA se omite mientras haya
    # candidatos bloqueados que mostrar — ni siquiera cuando SOBREVIVEN > 0.
    if cercanos:
        lineas.append('  LOS 3 QUE MÁS CERCA ESTUVIERON:')
        for _dist, nombre, detalle in cercanos:
            lineas.append(f'    {nombre:<28} {detalle}')
    elif n_sobrevive == 0:
        lineas.append('  Ningún pick bloqueado tiene distancia calculable al umbral')
        lineas.append('  (revisar sin_datos/phantom — bloqueo estructural, no de calibración).')

    return '\n'.join(lineas) + '\n'


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description='D173-11: reporte diario de embudo')
    parser.add_argument('--fecha', default=None, help='YYYYMMDD, default hoy')
    parser.add_argument('--file', default=None, help='ruta explícita a edge_report')
    parser.add_argument('--out', default=None, help='si se pasa, también escribe a este archivo')
    args = parser.parse_args()

    texto = generar_reporte(args.fecha, args.file)
    print(texto)

    if args.out:
        try:
            os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
            with open(args.out, 'w', encoding='utf-8') as fh:
                fh.write(texto)
        except OSError as e:
            print(f'[funnel_report] WARN no se pudo escribir {args.out}: {e}', file=sys.stderr)

    return 0


if __name__ == '__main__':
    sys.exit(main())
