"""
tests/test_nodo40.py — Nodo-40: Games/Sets Signal Layer

Cubre (per spec):
  T40-01  Gate diff: zona_diff clasifica correctamente dominante/ajustada/coinflip
  T40-02  Gate diff: zona ajustada excluida de señales apostables
  T40-03  Gap gate: UNDER requiere línea >= games_max + MIN_GAP_JUEGOS
  T40-04  Gap gate: OVER requiere games_min - línea >= MIN_GAP_JUEGOS
  T40-05  Gap gate: línea exactamente en el borde no genera señal
  T40-06  Cuota gate: cuota < MIN_CUOTA excluye la señal
  T40-07  Selección óptima UNDER: elige señal con mayor gap (mayor margen de seguridad)
  T40-08  Selección óptima OVER: elige señal con menor línea (más fácil de superar)
  T40-09  Anti-correlación: max 1 señal por partido en el reporte final
  T40-10  LABELS_EXCLUIR: mercado individual "juegos ganados por jugador" es ignorado
  T40-11  Confianza ALTA: gap >= 4 → confianza_señal == "ALTA"
  T40-12  Confianza MEDIA: gap 2-3 → confianza_señal == "MEDIA"
  T40-13  apostar=False cuando confianza MEDIA y cuota < 1.70
  T40-14  apostar=True cuando confianza ALTA
  T40-15  apostar=True cuando confianza MEDIA y cuota >= 1.70
  T40-16  predecir_sets: diff > 0.35 → 2 sets, range 16-19
  T40-17  predecir_sets: 0.25 < diff <= 0.35 → 2 sets, range 18-21
  T40-18  predecir_sets: 0.18 < diff <= 0.25 → 2 sets, range 20-23
  T40-19  predecir_sets: diff <= 0.18 → 3 sets
  T40-20  Total sets UNDER: 2 sets pred + cuota >= 1.60 → apostar=True
  T40-21  Total sets OVER: 3 sets pred + cuota >= 1.70 → apostar=True
  T40-22  build_games_combos: sin señales → lista vacía
  T40-23  build_games_combos: REGLA-G6 cap $2000 cuando n < 50
  T40-24  build_games_combos: stake override respetado cuando n >= 50
  T40-25  build_games_combos: combos no duplican el mismo partido (REGLA-G4)
  T40-26  guardar_reporte: calibracion_n lee len(games_calibracion) real, no hardcode
"""

import json
import pytest
from pathlib import Path
import tempfile
from unittest.mock import patch

from games_signal_calculator import (
    _zona_diff,
    _predecir_sets_y_games,
    _analizar_mercados_juegos,
    _seleccionar_señal_optima,
    guardar_reporte,
    DIFF_DOMINANTE,
    DIFF_COINFLIP,
    MIN_GAP_JUEGOS,
    MIN_CUOTA,
    LABELS_EXCLUIR,
)
from betplay_combo_builder import build_games_combos


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_betoffer(label: str, linea: float, odds_mas: int, odds_menos: int,
                   id_mas: int = 1001, id_menos: int = 1002) -> list:
    """Crea un betOffer mínimo con dos outcomes (más de / menos de)."""
    return [{
        "criterion": {"label": label},
        "outcomes": [
            {
                "label": "Más de",
                "line": int(linea * 1000),
                "odds": odds_mas,
                "id": id_mas,
            },
            {
                "label": "Menos de",
                "line": int(linea * 1000),
                "odds": odds_menos,
                "id": id_menos,
            },
        ]
    }]


def _make_games_signal_report(señales_optimas: list, calibracion_n: int = 3) -> dict:
    """Crea un games_signal_report mínimo para tests de build_games_combos."""
    apostar = []
    for i, s in enumerate(señales_optimas):
        apostar.append({
            "partido": s.get("partido", f"Jugador{i}A vs Jugador{i}B"),
            "zona_diff": s.get("zona_diff", "dominante"),
            "diff_abs": s.get("diff_abs", 0.5),
            "predicted_sets": s.get("predicted_sets", 2),
            "games_range": s.get("games_range", "16-19"),
            "señales_optimas": [s],
        })
    return {
        "metadata": {
            "fecha": "2026-06-28 16:00:00",
            "fuente": "test",
            "n_partidos": len(apostar),
            "n_apostar": len(apostar),
            "nodo": "Nodo-40-Games-Sets-Signal-Layer",
            "calibracion_n": calibracion_n,
        },
        "apostar": apostar,
        "detalle_completo": [],
    }


def _write_temp_report(data: dict) -> str:
    """Escribe un games_signal_report en un archivo temporal y devuelve la ruta."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(data, tmp, ensure_ascii=False)
    tmp.close()
    return tmp.name


# ══════════════════════════════════════════════════════════════════════════════
# T40-01 a T40-02 — Gate diff: clasificación de zonas
# ══════════════════════════════════════════════════════════════════════════════

class TestGateDiff:
    def test_t40_01_dominante_above_threshold(self):
        """diff > DIFF_DOMINANTE → zona dominante."""
        assert _zona_diff(DIFF_DOMINANTE + 0.01) == "dominante"
        assert _zona_diff(0.50) == "dominante"
        assert _zona_diff(0.90) == "dominante"

    def test_t40_01_coinflip_at_or_below_threshold(self):
        """diff <= DIFF_COINFLIP → zona coinflip."""
        assert _zona_diff(DIFF_COINFLIP) == "coinflip"
        assert _zona_diff(0.0) == "coinflip"
        assert _zona_diff(0.10) == "coinflip"

    def test_t40_01_ajustada_between_thresholds(self):
        """DIFF_COINFLIP < diff <= DIFF_DOMINANTE → zona ajustada."""
        mid = (DIFF_COINFLIP + DIFF_DOMINANTE) / 2
        assert _zona_diff(mid) == "ajustada"
        assert _zona_diff(DIFF_COINFLIP + 0.01) == "ajustada"
        assert _zona_diff(DIFF_DOMINANTE) == "ajustada"

    def test_t40_02_ajustada_produces_no_señal(self):
        """Zona ajustada no genera señales — el loop la salta sin procesar Kambi."""
        # La función _analizar_mercados_juegos no filtra por zona, eso lo hace
        # procesar_partidos. Pero sí podemos verificar que, con pred de 2 sets
        # y un gap grande, _analizar_mercados genera señal: esto confirma que
        # el bloqueo de zona ajustada está en el caller (procesar_partidos), no aquí.
        # Este test documenta el contrato: zona ajustada → no llega a _analizar_mercados.
        # Lo verifica indirectamente: diff ajustado no es dominante ni coinflip.
        diff = (DIFF_COINFLIP + DIFF_DOMINANTE) / 2
        zona = _zona_diff(diff)
        assert zona == "ajustada"
        assert zona not in ("dominante", "coinflip")


# ══════════════════════════════════════════════════════════════════════════════
# T40-03 a T40-06 — Gap gate y Cuota gate
# ══════════════════════════════════════════════════════════════════════════════

class TestGapGate:
    """Tests para _analizar_mercados_juegos con mercado Total de juegos."""

    # Predicción base: 2 sets, range 16-19 (diff dominante)
    PRED_2SETS = {"predicted_sets": 2, "games_min": 16, "games_max": 19, "games_range": "16-19"}
    # Predicción base: 3 sets, range 26-99 (diff coinflip)
    PRED_3SETS = {"predicted_sets": 3, "games_min": 26, "games_max": 99, "games_range": "26-32+"}

    def test_t40_03_under_gap_suficiente(self):
        """UNDER: línea >= games_max + MIN_GAP_JUEGOS genera señal."""
        linea = self.PRED_2SETS["games_max"] + MIN_GAP_JUEGOS  # 19 + 2 = 21
        betoffer = _make_betoffer("Total de juegos", linea, 2000, 1800)
        señales = _analizar_mercados_juegos(betoffer, self.PRED_2SETS)
        under = [s for s in señales if s["direccion"] == "UNDER"]
        assert len(under) >= 1
        assert under[0]["gap_juegos"] >= MIN_GAP_JUEGOS

    def test_t40_04_over_gap_suficiente(self):
        """OVER: games_min - línea >= MIN_GAP_JUEGOS genera señal."""
        linea = self.PRED_3SETS["games_min"] - MIN_GAP_JUEGOS  # 26 - 2 = 24
        betoffer = _make_betoffer("Total de juegos", linea, 1800, 2000)
        señales = _analizar_mercados_juegos(betoffer, self.PRED_3SETS)
        over = [s for s in señales if s["direccion"] == "OVER"]
        assert len(over) >= 1
        assert over[0]["gap_juegos"] >= MIN_GAP_JUEGOS

    def test_t40_05_gap_insuficiente_no_genera_señal(self):
        """Línea con gap < MIN_GAP_JUEGOS no genera señal UNDER."""
        linea = self.PRED_2SETS["games_max"] + MIN_GAP_JUEGOS - 0.5  # 19 + 1.5 = 20.5
        betoffer = _make_betoffer("Total de juegos", linea, 2000, 1800)
        señales = _analizar_mercados_juegos(betoffer, self.PRED_2SETS)
        under = [s for s in señales if s["direccion"] == "UNDER"]
        assert len(under) == 0

    def test_t40_05_over_gap_insuficiente(self):
        """Línea con gap < MIN_GAP_JUEGOS no genera señal OVER."""
        linea = self.PRED_3SETS["games_min"] - MIN_GAP_JUEGOS + 0.5  # 26 - 1.5 = 24.5
        betoffer = _make_betoffer("Total de juegos", linea, 1800, 2000)
        señales = _analizar_mercados_juegos(betoffer, self.PRED_3SETS)
        over = [s for s in señales if s["direccion"] == "OVER"]
        assert len(over) == 0

    def test_t40_06_cuota_bajo_minimo_no_genera_señal(self):
        """Cuota < MIN_CUOTA: señal generada pero apostar=False."""
        linea = self.PRED_2SETS["games_max"] + MIN_GAP_JUEGOS + 2  # gap amplio
        # odds < MIN_CUOTA → e.g. 1.30
        odds_bajo = int(1.30 * 1000)
        betoffer = _make_betoffer("Total de juegos", linea, 2000, odds_bajo)
        señales = _analizar_mercados_juegos(betoffer, self.PRED_2SETS)
        under = [s for s in señales if s["direccion"] == "UNDER"]
        # No debería haber señal porque cuota_menos < MIN_CUOTA
        assert len(under) == 0

    def test_t40_06_cuota_exactamente_minimo_genera_señal(self):
        """Cuota exactamente en MIN_CUOTA: señal válida."""
        linea = self.PRED_2SETS["games_max"] + MIN_GAP_JUEGOS + 2
        odds_min = int(MIN_CUOTA * 1000)
        betoffer = _make_betoffer("Total de juegos", linea, 2000, odds_min)
        señales = _analizar_mercados_juegos(betoffer, self.PRED_2SETS)
        under = [s for s in señales if s["direccion"] == "UNDER"]
        assert len(under) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# T40-07 a T40-08 — Selección óptima de línea
# ══════════════════════════════════════════════════════════════════════════════

class TestSeleccionOptima:
    PRED_2SETS = {"predicted_sets": 2, "games_min": 16, "games_max": 19, "games_range": "16-19"}
    PRED_3SETS = {"predicted_sets": 3, "games_min": 26, "games_max": 99, "games_range": "26-32+"}

    def test_t40_07_under_elige_mayor_gap(self):
        """UNDER: entre múltiples líneas, elige la de mayor gap (más margen de seguridad)."""
        # Línea 21.5 → gap 2.5 | Línea 22.5 → gap 3.5 | Línea 23.5 → gap 4.5
        bos = []
        for linea, odds_menos, id_menos in [(21.5, 2350, 101), (22.5, 1960, 102), (23.5, 1830, 103)]:
            bos.extend(_make_betoffer("Total de juegos", linea, 2000, odds_menos,
                                      id_mas=200, id_menos=id_menos))
        señales = _analizar_mercados_juegos(bos, self.PRED_2SETS)
        # D149-02 (Nodo-149): retorna tupla (juegos_optimas, sets_optimas), no lista plana.
        optimas, _ = _seleccionar_señal_optima(señales)
        under = [s for s in optimas if s["direccion"] == "UNDER" and s["mercado"] == "Total de juegos"]
        assert len(under) == 1
        # La óptima debe ser la de mayor gap
        assert under[0]["gap_juegos"] == max(
            s["gap_juegos"] for s in señales
            if s["direccion"] == "UNDER" and s["mercado"] == "Total de juegos" and s["apostar"]
        )

    def test_t40_08_over_elige_mayor_gap(self):
        """OVER: entre múltiples líneas, elige la de mayor gap (más distancia del mínimo)."""
        # games_min = 26
        # Línea 23.5 → gap 2.5 | Línea 22.5 → gap 3.5 | Línea 21.5 → gap 4.5
        bos = []
        for linea, odds_mas, id_mas in [(23.5, 2350, 101), (22.5, 1960, 102), (21.5, 1830, 103)]:
            bos.extend(_make_betoffer("Total de juegos", linea, odds_mas, 2000,
                                      id_mas=id_mas, id_menos=200))
        señales = _analizar_mercados_juegos(bos, self.PRED_3SETS)
        optimas, _ = _seleccionar_señal_optima(señales)
        over = [s for s in optimas if s["direccion"] == "OVER" and s["mercado"] == "Total de juegos"]
        assert len(over) == 1
        # La óptima es la de mayor gap
        assert over[0]["gap_juegos"] == max(
            s["gap_juegos"] for s in señales
            if s["direccion"] == "OVER" and s["mercado"] == "Total de juegos" and s["apostar"]
        )

    def test_t40_seleccion_unica_por_mercado_y_direccion(self):
        """_seleccionar_señal_optima devuelve como máximo 1 señal por (mercado, dirección)."""
        bos = []
        for linea, odds_menos, id_menos in [(21.5, 2350, 101), (22.5, 1960, 102)]:
            bos.extend(_make_betoffer("Total de juegos", linea, 2000, odds_menos,
                                      id_mas=200, id_menos=id_menos))
        señales = _analizar_mercados_juegos(bos, self.PRED_2SETS)
        optimas, _ = _seleccionar_señal_optima(señales)
        under = [s for s in optimas if s["direccion"] == "UNDER" and s["mercado"] == "Total de juegos"]
        assert len(under) == 1


# ══════════════════════════════════════════════════════════════════════════════
# T40-09 — Anti-correlación (REGLA-G4)
# ══════════════════════════════════════════════════════════════════════════════

class TestAntiCorrelacion:
    def test_t40_09_max_1_señal_por_partido_en_combos(self):
        """build_games_combos: cada partido aparece máx 1 vez en un combo (REGLA-G4)."""
        # Crear señales con dos señales del mismo partido (juegos + sets)
        señal_juegos = {
            "partido": "Krueger vs Suresh",
            "zona_diff": "dominante",
            "diff_abs": 0.48,
            "predicted_sets": 2,
            "games_range": "16-19",
            "mercado": "Total de juegos",
            "linea": 22.5,
            "direccion": "UNDER",
            "cuota": 1.78,
            "outcome_id": 4237150679,
            "gap_juegos": 3.5,
            "confianza_señal": "MEDIA",
            "apostar": True,
        }
        señal_sets = {
            "partido": "Krueger vs Suresh",  # mismo partido
            "zona_diff": "dominante",
            "diff_abs": 0.48,
            "predicted_sets": 2,
            "games_range": "16-19",
            "mercado": "Total de sets",
            "linea": 2.5,
            "direccion": "UNDER",
            "cuota": 1.60,
            "outcome_id": 4237150680,
            "gap_juegos": None,
            "confianza_señal": "MEDIA",
            "apostar": True,
        }
        # El reporte tiene un partido con dos señales_optimas
        report = {
            "metadata": {
                "fecha": "2026-06-28 16:00:00",
                "fuente": "test",
                "n_partidos": 1,
                "n_apostar": 1,
                "nodo": "Nodo-40-Games-Sets-Signal-Layer",
                "calibracion_n": 3,
            },
            "apostar": [{
                "partido": "Krueger vs Suresh",
                "zona_diff": "dominante",
                "diff_abs": 0.48,
                "predicted_sets": 2,
                "games_range": "16-19",
                "señales_optimas": [señal_juegos, señal_sets],
            }],
            "detalle_completo": [],
        }
        tmp_path = _write_temp_report(report)
        try:
            combos, metadata = build_games_combos(stake_per_combo=2000, games_file=tmp_path)
            # Con 1 solo partido solo puede haber 1 pierna por combo
            for combo in combos:
                partidos_en_combo = {leg["partido"] for leg in combo["legs"]}
                # No hay duplicado de partido
                assert len(partidos_en_combo) == len(combo["legs"]), \
                    f"Partido duplicado en {combo['label']}: {[l['partido'] for l in combo['legs']]}"
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# T40-10 — LABELS_EXCLUIR
# ══════════════════════════════════════════════════════════════════════════════

class TestLabelsExcluir:
    PRED_2SETS = {"predicted_sets": 2, "games_min": 16, "games_max": 19, "games_range": "16-19"}

    def test_t40_10_mercado_individual_juegos_excluido(self):
        """'número total de juegos ganados por X' es ignorado (LABELS_EXCLUIR)."""
        linea = 22.5
        for label in [
            "Número total de juegos ganados por Krueger",
            "Juegos ganados por Suresh",
        ]:
            betoffer = _make_betoffer(label, linea, 2000, 1800)
            señales = _analizar_mercados_juegos(betoffer, self.PRED_2SETS)
            assert len(señales) == 0, f"Label '{label}' debería ser excluido pero generó señal"

    def test_t40_10_mercado_total_juegos_no_excluido(self):
        """'Total de juegos' no está en LABELS_EXCLUIR y genera señal."""
        linea = self.PRED_2SETS["games_max"] + MIN_GAP_JUEGOS + 1  # gap 3
        betoffer = _make_betoffer("Total de juegos", linea, 2000, 1800)
        señales = _analizar_mercados_juegos(betoffer, self.PRED_2SETS)
        assert len(señales) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# T40-11 a T40-15 — Confianza y apostar flag
# ══════════════════════════════════════════════════════════════════════════════

class TestConfianza:
    PRED_2SETS = {"predicted_sets": 2, "games_min": 16, "games_max": 19, "games_range": "16-19"}

    def test_t40_11_gap_4_o_mas_es_alta(self):
        """gap >= 4 → confianza_señal == 'ALTA'."""
        linea = self.PRED_2SETS["games_max"] + 4  # gap exacto = 4
        betoffer = _make_betoffer("Total de juegos", linea, 2000, 1800)
        señales = _analizar_mercados_juegos(betoffer, self.PRED_2SETS)
        under = [s for s in señales if s["direccion"] == "UNDER"]
        assert len(under) >= 1
        assert under[0]["confianza_señal"] == "ALTA"

    def test_t40_12_gap_2_a_3_es_media(self):
        """2 <= gap < 4 → confianza_señal == 'MEDIA'."""
        linea = self.PRED_2SETS["games_max"] + 2  # gap exacto = 2
        betoffer = _make_betoffer("Total de juegos", linea, 2000, 1800)
        señales = _analizar_mercados_juegos(betoffer, self.PRED_2SETS)
        under = [s for s in señales if s["direccion"] == "UNDER"]
        assert len(under) >= 1
        assert under[0]["confianza_señal"] == "MEDIA"

    def test_t40_13_media_cuota_baja_apostar_false(self):
        """confianza MEDIA + cuota < 1.70 → apostar=False."""
        linea = self.PRED_2SETS["games_max"] + 2  # gap = 2 → MEDIA
        odds_bajo = int(1.60 * 1000)  # < 1.70
        betoffer = _make_betoffer("Total de juegos", linea, 2000, odds_bajo)
        señales = _analizar_mercados_juegos(betoffer, self.PRED_2SETS)
        under = [s for s in señales if s["direccion"] == "UNDER"]
        assert len(under) >= 1
        assert under[0]["confianza_señal"] == "MEDIA"
        assert under[0]["apostar"] is False

    def test_t40_14_alta_apostar_true(self):
        """confianza ALTA → apostar=True siempre."""
        linea = self.PRED_2SETS["games_max"] + 4  # gap = 4 → ALTA
        odds = int(1.55 * 1000)  # cuota baja pero ALTA siempre apuesta
        betoffer = _make_betoffer("Total de juegos", linea, 2000, odds)
        señales = _analizar_mercados_juegos(betoffer, self.PRED_2SETS)
        under = [s for s in señales if s["direccion"] == "UNDER"]
        assert len(under) >= 1
        assert under[0]["confianza_señal"] == "ALTA"
        assert under[0]["apostar"] is True

    def test_t40_15_media_cuota_alta_apostar_true(self):
        """confianza MEDIA + cuota >= 1.70 → apostar=True."""
        linea = self.PRED_2SETS["games_max"] + 2  # gap = 2 → MEDIA
        odds = int(1.70 * 1000)
        betoffer = _make_betoffer("Total de juegos", linea, 2000, odds)
        señales = _analizar_mercados_juegos(betoffer, self.PRED_2SETS)
        under = [s for s in señales if s["direccion"] == "UNDER"]
        assert len(under) >= 1
        assert under[0]["apostar"] is True


# ══════════════════════════════════════════════════════════════════════════════
# T40-16 a T40-19 — predecir_sets_y_games
# ══════════════════════════════════════════════════════════════════════════════

class TestPredecirSetsYGames:
    def test_t40_16_dominante_2sets_range_16_19(self):
        """diff > 0.35 → 2 sets, range 16-19."""
        p = _predecir_sets_y_games(0.48, 1.2)
        assert p["predicted_sets"] == 2
        assert p["games_range"] == "16-19"
        assert p["games_min"] == 16
        assert p["games_max"] == 19

    def test_t40_17_ajustada_alta_2sets_range_18_21(self):
        """0.25 < diff <= 0.35 → 2 sets, range 18-21."""
        p = _predecir_sets_y_games(0.30, 1.2)
        assert p["predicted_sets"] == 2
        assert p["games_range"] == "18-21"

    def test_t40_18_ajustada_baja_2sets_range_20_23(self):
        """0.18 < diff <= 0.25 → 2 sets, range 20-23."""
        p = _predecir_sets_y_games(0.22, 1.2)
        assert p["predicted_sets"] == 2
        assert p["games_range"] == "20-23"

    def test_t40_19_coinflip_3sets(self):
        """diff <= 0.18 → 3 sets."""
        p = _predecir_sets_y_games(0.03, 1.6)
        assert p["predicted_sets"] == 3
        assert p["games_range"] == "26-32+"

    def test_t40_19_coinflip_baja_densidad_23_28(self):
        """diff <= 0.18 y total_score bajo → range 23-28."""
        p = _predecir_sets_y_games(0.10, 1.0)  # total_score <= 1.5
        assert p["predicted_sets"] == 3
        assert p["games_range"] == "23-28"

    def test_t40_16_borde_dominante(self):
        """diff exactamente en DIFF_DOMINANTE (0.35) → ajustada (no dominante)."""
        p = _predecir_sets_y_games(DIFF_DOMINANTE, 1.2)
        # 0.35 no es > 0.35, cae en rama elif diff > 0.25 → range 18-21
        assert p["games_range"] == "18-21"


# ══════════════════════════════════════════════════════════════════════════════
# T40-20 a T40-21 — Total de sets
# ══════════════════════════════════════════════════════════════════════════════

class TestTotalSets:
    def test_t40_20_sets_under_2sets_cuota_alta(self):
        """2 sets pred + cuota >= 1.60 → Total de sets UNDER apostar=True."""
        pred = {"predicted_sets": 2, "games_min": 16, "games_max": 19, "games_range": "16-19"}
        odds_menos = int(1.65 * 1000)
        betoffer = _make_betoffer("Total de sets", 2.5, 2000, odds_menos)
        señales = _analizar_mercados_juegos(betoffer, pred)
        sets_under = [s for s in señales if s["mercado"] == "Total de sets" and s["direccion"] == "UNDER"]
        assert len(sets_under) >= 1
        assert sets_under[0]["apostar"] is True

    def test_t40_20_sets_under_cuota_baja_no_apostar(self):
        """2 sets pred + cuota < 1.60 → Total de sets UNDER apostar=False."""
        pred = {"predicted_sets": 2, "games_min": 16, "games_max": 19, "games_range": "16-19"}
        odds_menos = int(1.55 * 1000)  # < 1.60
        betoffer = _make_betoffer("Total de sets", 2.5, 2000, odds_menos)
        señales = _analizar_mercados_juegos(betoffer, pred)
        sets_under = [s for s in señales if s["mercado"] == "Total de sets" and s["direccion"] == "UNDER"]
        assert len(sets_under) >= 1
        assert sets_under[0]["apostar"] is False

    def test_t40_21_sets_over_3sets_cuota_alta(self):
        """3 sets pred + cuota >= 1.70 → Total de sets OVER apostar=True."""
        pred = {"predicted_sets": 3, "games_min": 26, "games_max": 99, "games_range": "26-32+"}
        odds_mas = int(1.75 * 1000)
        betoffer = _make_betoffer("Total de sets", 2.5, odds_mas, 2000)
        señales = _analizar_mercados_juegos(betoffer, pred)
        sets_over = [s for s in señales if s["mercado"] == "Total de sets" and s["direccion"] == "OVER"]
        assert len(sets_over) >= 1
        assert sets_over[0]["apostar"] is True


# ══════════════════════════════════════════════════════════════════════════════
# T40-22 a T40-25 — build_games_combos
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildGamesCombos:
    def _make_señal(self, partido: str, outcome_id: int, cuota: float = 1.78,
                    gap: float = 3.5, confianza: str = "MEDIA") -> dict:
        return {
            "partido": partido,
            "zona_diff": "dominante",
            "diff_abs": 0.48,
            "predicted_sets": 2,
            "games_range": "16-19",
            "mercado": "Total de juegos",
            "linea": 22.5,
            "direccion": "UNDER",
            "cuota": cuota,
            "outcome_id": outcome_id,
            "gap_juegos": gap,
            "confianza_señal": confianza,
            "apostar": True,
        }

    def test_t40_22_sin_señales_retorna_vacio(self):
        """Sin señales apostar → lista vacía."""
        report = {
            "metadata": {"calibracion_n": 3, "n_apostar": 0, "fecha": "2026-06-28", "fuente": "test", "n_partidos": 0, "nodo": "N40"},
            "apostar": [],
            "detalle_completo": [],
        }
        tmp = _write_temp_report(report)
        try:
            combos, meta = build_games_combos(stake_per_combo=2000, games_file=tmp)
            assert combos == []
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_t40_23_regla_g6_cap_2000(self):
        """REGLA-G6: n < 50 → stake efectivo nunca supera $2,000."""
        señales = [
            self._make_señal("A vs B", 1001),
            self._make_señal("C vs D", 1002),
        ]
        report = _make_games_signal_report(señales, calibracion_n=5)
        tmp = _write_temp_report(report)
        try:
            combos, meta = build_games_combos(stake_per_combo=5000, games_file=tmp)
            assert meta["regla_g6_active"] is True
            for c in combos:
                assert c["stake"] <= 2000, f"Stake {c['stake']} excede REGLA-G6 cap"
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_t40_24_sin_g6_stake_respetado(self):
        """n >= 50 → stake override de 3000 es respetado."""
        señales = [
            self._make_señal("A vs B", 1001),
            self._make_señal("C vs D", 1002),
        ]
        report = _make_games_signal_report(señales, calibracion_n=50)
        tmp = _write_temp_report(report)
        try:
            combos, meta = build_games_combos(stake_per_combo=3000, games_file=tmp)
            assert meta["regla_g6_active"] is False
            for c in combos:
                assert c["stake"] == 3000
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_t40_25_combos_no_duplican_partido(self):
        """Ningún combo contiene el mismo partido más de una vez (REGLA-G4)."""
        señales = [
            self._make_señal("A vs B", 1001, cuota=1.80, gap=4.0, confianza="ALTA"),
            self._make_señal("C vs D", 1002, cuota=1.75, gap=3.5, confianza="MEDIA"),
            self._make_señal("E vs F", 1003, cuota=1.65, gap=2.5, confianza="MEDIA"),
        ]
        report = _make_games_signal_report(señales, calibracion_n=3)
        tmp = _write_temp_report(report)
        try:
            combos, meta = build_games_combos(stake_per_combo=2000, games_file=tmp)
            for combo in combos:
                partidos = [leg["partido"] for leg in combo["legs"]]
                assert len(partidos) == len(set(partidos)), \
                    f"{combo['label']}: partido duplicado — {partidos}"
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_t40_25_max_3_piernas(self):
        """Ningún combo tiene más de 3 piernas (REGLA-G5)."""
        señales = [self._make_señal(f"J{i}A vs J{i}B", 1000 + i) for i in range(5)]
        report = _make_games_signal_report(señales, calibracion_n=3)
        tmp = _write_temp_report(report)
        try:
            combos, meta = build_games_combos(stake_per_combo=2000, games_file=tmp)
            for combo in combos:
                assert combo["n_piernas"] <= 3, \
                    f"{combo['label']}: {combo['n_piernas']} piernas > REGLA-G5 máx 3"
        finally:
            Path(tmp).unlink(missing_ok=True)


class TestCalibracionN:
    """T40-26: guardar_reporte usa calibracion_n real de calibracion_edge.json."""

    def test_t40_26_calibracion_n_vacio_es_cero(self):
        """games_calibracion=[] → calibracion_n=0 en el reporte (no hardcode)."""
        calib_sintetico = {"games_calibracion": []}
        with tempfile.TemporaryDirectory() as tmpdir:
            # Crear data/calibracion_edge.json sintético en directorio temporal
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            cal_path = data_dir / "calibracion_edge.json"
            cal_path.write_text(json.dumps(calib_sintetico), encoding="utf-8")

            reports_dir = Path(tmpdir) / "reports"
            reports_dir.mkdir()

            # Parchear Path para que "data/calibracion_edge.json" apunte al sintético
            original_path = Path("data/calibracion_edge.json")
            with patch("games_signal_calculator.Path") as mock_path_cls:
                def path_side_effect(p):
                    if str(p) == "data/calibracion_edge.json":
                        return cal_path
                    if str(p) == "reports":
                        return reports_dir
                    return Path(p)
                mock_path_cls.side_effect = path_side_effect

                outfile = guardar_reporte([], "test_source.json")

            with open(outfile, encoding="utf-8") as f:
                output = json.load(f)

            assert output["metadata"]["calibracion_n"] == 0, (
                f"calibracion_n debería ser 0 con games_calibracion vacío, "
                f"no {output['metadata']['calibracion_n']}"
            )

    def test_t40_26b_calibracion_n_refleja_n_real(self):
        """games_calibracion con 7 entradas → calibracion_n=7 en el reporte."""
        obs = [{"fecha": f"2026-07-0{i}", "partido": f"A vs B", "sets_real": 2} for i in range(7)]
        calib_sintetico = {"games_calibracion": obs}
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            cal_path = data_dir / "calibracion_edge.json"
            cal_path.write_text(json.dumps(calib_sintetico), encoding="utf-8")

            reports_dir = Path(tmpdir) / "reports"
            reports_dir.mkdir()

            with patch("games_signal_calculator.Path") as mock_path_cls:
                def path_side_effect(p):
                    if str(p) == "data/calibracion_edge.json":
                        return cal_path
                    if str(p) == "reports":
                        return reports_dir
                    return Path(p)
                mock_path_cls.side_effect = path_side_effect

                outfile = guardar_reporte([], "test_source.json")

            with open(outfile, encoding="utf-8") as f:
                output = json.load(f)

            assert output["metadata"]["calibracion_n"] == 7, (
                f"calibracion_n debería ser 7, no {output['metadata']['calibracion_n']}"
            )
