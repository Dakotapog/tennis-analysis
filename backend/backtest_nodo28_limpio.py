#!/usr/bin/env python3
"""
Backtest Nodo-28: Re-calcula predicciones SIN data leakage.

Problema detectado: el H2H de las 19:54 incluye resultados de partidos
de hoy (18.06.2026) en los historiales, inflando accuracy.

Este script:
1. Carga el H2H contaminado
2. Filtra entradas con fecha de hoy de TODOS los historiales
3. Recalcula ELO + form + predicciones desde cero (sin leak)
4. Compara contra resultados reales
5. Reporta accuracy limpio vs contaminado
"""

import json
import sys
import os
from copy import deepcopy
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.rivalry_analyzer import RivalryAnalyzer
from analysis.ranking_manager import RankingManager
from analysis.elo_system import EloRatingSystem


FECHA_HOY = "18.06.2026"
H2H_FILE = "reports/h2h_results_enhanced_20260618_195412.json"
RESULTS_FILE = "reports/resultados_finales_20260618_211312.json"


def filtrar_historial(historial, fecha_excluir):
    """Quitar partidos de fecha_excluir del historial."""
    return [m for m in historial if m.get('fecha') != fecha_excluir]


def recalcular_form(historial, player_name, n=10):
    """Recalcular form_analysis desde historial limpio (últimos n partidos)."""
    recent = historial[:n]
    wins = sum(1 for m in recent if m.get('outcome') == 'Ganó')
    losses = len(recent) - wins
    return {
        'player_name': player_name,
        'recent_matches_count': len(recent),
        'wins': wins,
        'losses': losses,
        'win_percentage': round(wins / len(recent) * 100, 1) if recent else 0
    }


def main():
    # ── Cargar datos ──
    with open(H2H_FILE, 'r', encoding='utf-8') as f:
        h2h_data = json.load(f)

    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    # Build lookup de resultados reales
    resultados_reales = {}
    for p in results_data['partidos']:
        resultados_reales[p['match_id']] = {
            'ganador_real': p['resultado_real'],
            'correcto_original': p['correcto'],
            'confianza_original': p['confianza'],
            'prediccion_original': p['prediccion']
        }

    # ── Inicializar motor de predicción ──
    ranking_manager = RankingManager()
    elo_system = EloRatingSystem()
    rivalry_analyzer = RivalryAnalyzer(ranking_manager, elo_system)

    partidos = h2h_data['partidos']

    print("=" * 80)
    print("BACKTEST NODO-28 — PREDICCIONES SIN DATA LEAKAGE")
    print(f"Fecha excluida de historiales: {FECHA_HOY}")
    print(f"Partidos en H2H: {len(partidos)}")
    print(f"Partidos con resultado real: {len(resultados_reales)}")
    print("=" * 80)

    # ── Contadores ──
    total_validados = 0
    aciertos_limpios = 0
    aciertos_originales = 0
    cambios = []
    entradas_filtradas_total = 0

    detalles = []

    for match in partidos:
        match_id = match.get('match_id')
        if match_id not in resultados_reales:
            continue

        p1 = match['jugador1']
        p2 = match['jugador2']
        p1_key = p1.replace(' ', '_').replace('.', '')
        p2_key = p2.replace(' ', '_').replace('.', '')

        # ── Historiales originales y limpios ──
        hist1_original = match.get(f'historial_{p1_key}', [])
        hist2_original = match.get(f'historial_{p2_key}', [])

        hist1_limpio = filtrar_historial(hist1_original, FECHA_HOY)
        hist2_limpio = filtrar_historial(hist2_original, FECHA_HOY)

        filtradas_1 = len(hist1_original) - len(hist1_limpio)
        filtradas_2 = len(hist2_original) - len(hist2_limpio)
        entradas_filtradas_total += filtradas_1 + filtradas_2

        # ── Recalcular form desde historial limpio ──
        form1_limpio = recalcular_form(hist1_limpio, p1)
        form2_limpio = recalcular_form(hist2_limpio, p2)

        # ── Recalcular ELO desde historial limpio ──
        elo1_limpio = rivalry_analyzer.calculate_elo_from_history(p1, hist1_limpio)
        elo2_limpio = rivalry_analyzer.calculate_elo_from_history(p2, hist2_limpio)

        # ── Filtrar H2H directo también (quitar enfrentamiento de hoy) ──
        h2h_directo = match.get('enfrentamientos_directos', [])
        h2h_limpio = [m for m in h2h_directo if m.get('fecha') != FECHA_HOY]

        # ── Contexto del partido ──
        current_context = {
            'country': match.get('pais', 'N/A'),
            'surface': match.get('tipo_cancha', 'N/A')
        }

        # ── Recalcular predicción ──
        try:
            new_analysis = rivalry_analyzer.analyze_rivalry(
                hist1_limpio, hist2_limpio,
                p1, p2,
                form1_limpio, form2_limpio,
                h2h_limpio, current_context,
                elo1_limpio, elo2_limpio,
                match.get('torneo_completo', '')
            )

            new_pred = new_analysis['prediction']
            new_favored = new_pred['favored_player']
            new_confidence = new_pred['confidence']
        except Exception as e:
            print(f"  ERROR recalculando {p1} vs {p2}: {e}")
            continue

        # ── Comparar ──
        resultado = resultados_reales[match_id]
        ganador_real = resultado['ganador_real']
        pred_original = resultado['prediccion_original']
        conf_original = resultado['confianza_original']
        correcto_original = resultado['correcto_original']

        correcto_limpio = (new_favored == ganador_real)

        total_validados += 1
        if correcto_limpio:
            aciertos_limpios += 1
        if correcto_original:
            aciertos_originales += 1

        # Track cambios
        cambio_pred = (new_favored != pred_original)
        cambio_resultado = (correcto_limpio != correcto_original)

        detalle = {
            'partido': f"{p1} vs {p2}",
            'superficie': match.get('tipo_cancha', '?'),
            'pred_original': pred_original,
            'conf_original': conf_original,
            'pred_limpia': new_favored,
            'conf_limpia': round(new_confidence, 1),
            'ganador_real': ganador_real,
            'correcto_original': correcto_original,
            'correcto_limpio': correcto_limpio,
            'cambio_pred': cambio_pred,
            'filtradas': filtradas_1 + filtradas_2
        }
        detalles.append(detalle)

        if cambio_resultado:
            emoji_orig = "OK" if correcto_original else "FAIL"
            emoji_limp = "OK" if correcto_limpio else "FAIL"
            cambios.append(f"  {p1} vs {p2}: {emoji_orig} -> {emoji_limp} | "
                          f"pred: {pred_original}({conf_original}%) -> {new_favored}({new_confidence:.1f}%) | "
                          f"real: {ganador_real} | filtradas: {filtradas_1+filtradas_2}")

    # ── Reporte ──
    print()
    print("=" * 80)
    print("RESULTADOS")
    print("=" * 80)

    acc_original = aciertos_originales / total_validados * 100 if total_validados else 0
    acc_limpio = aciertos_limpios / total_validados * 100 if total_validados else 0

    print(f"\nPartidos validados: {total_validados}")
    print(f"Entradas de hoy filtradas de historiales: {entradas_filtradas_total}")
    print()
    print(f"  ACCURACY ORIGINAL (con leak):  {aciertos_originales}/{total_validados} = {acc_original:.1f}%")
    print(f"  ACCURACY LIMPIO   (sin leak):  {aciertos_limpios}/{total_validados} = {acc_limpio:.1f}%")
    print(f"  DIFERENCIA:                    {acc_limpio - acc_original:+.1f} puntos porcentuales")
    print()

    # Partidos que cambiaron resultado
    pred_cambiaron = sum(1 for d in detalles if d['cambio_pred'])
    print(f"Predicciones que cambiaron de jugador: {pred_cambiaron}/{total_validados}")

    if cambios:
        print(f"\nPartidos cuyo acierto/fallo CAMBIO ({len(cambios)}):")
        for c in cambios:
            print(c)

    # ── Desglose por superficie ──
    print("\n" + "=" * 80)
    print("DESGLOSE POR SUPERFICIE")
    print("=" * 80)
    superficies = {}
    for d in detalles:
        sup = d['superficie']
        if sup not in superficies:
            superficies[sup] = {'total': 0, 'ok_orig': 0, 'ok_limpio': 0}
        superficies[sup]['total'] += 1
        if d['correcto_original']:
            superficies[sup]['ok_orig'] += 1
        if d['correcto_limpio']:
            superficies[sup]['ok_limpio'] += 1

    for sup, stats in sorted(superficies.items()):
        acc_o = stats['ok_orig'] / stats['total'] * 100
        acc_l = stats['ok_limpio'] / stats['total'] * 100
        print(f"  {sup:8s}: original {stats['ok_orig']}/{stats['total']} ({acc_o:.0f}%) | limpio {stats['ok_limpio']}/{stats['total']} ({acc_l:.0f}%) | diff {acc_l-acc_o:+.0f}pp")

    # ── Desglose por confianza ──
    print("\n" + "=" * 80)
    print("DESGLOSE POR BANDA DE CONFIANZA (LIMPIA)")
    print("=" * 80)
    bandas = [(50, 52, '50-52%'), (52, 55, '52-55%'), (55, 60, '55-60%'), (60, 70, '60-70%'), (70, 100, '70%+')]
    for lo, hi, label in bandas:
        bucket = [d for d in detalles if lo <= d['conf_limpia'] < hi]
        if bucket:
            ok = sum(1 for d in bucket if d['correcto_limpio'])
            print(f"  {label:12s}: {ok}/{len(bucket)} ({ok/len(bucket)*100:.0f}%)")

    # ── Lista completa ──
    print("\n" + "=" * 80)
    print("DETALLE COMPLETO")
    print("=" * 80)
    print(f"{'Partido':50s} {'Sup':6s} {'Pred Orig':20s} {'Conf':5s} {'Pred Limpia':20s} {'Conf':5s} {'Real':20s} {'Orig':5s} {'Limp':5s} {'Filt':4s}")
    print("-" * 150)
    for d in detalles:
        orig_mark = "OK" if d['correcto_original'] else "FAIL"
        limp_mark = "OK" if d['correcto_limpio'] else "FAIL"
        cambio_mark = " *" if d['cambio_pred'] else ""
        print(f"{d['partido']:50s} {d['superficie']:6s} {d['pred_original']:20s} {d['conf_original']:5.1f} {d['pred_limpia']:20s} {d['conf_limpia']:5.1f} {d['ganador_real']:20s} {orig_mark:5s} {limp_mark:5s} {d['filtradas']:4d}{cambio_mark}")


if __name__ == '__main__':
    main()
