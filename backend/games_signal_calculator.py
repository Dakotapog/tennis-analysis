"""
games_signal_calculator.py — Nodo-40 Fase 1
Alpha ortogonal al ganador: mercado de totales (games/sets) basado en score_difference.

Flujo:
  1. Lee h2h_results_enhanced_*.json más reciente
  2. Calcula zona_diff por partido (dominante / ajustada / coinflip)
  3. Consulta Kambi API → betOffers de Total de juegos / Total de sets por evento
  4. Selecciona la línea óptima con gap suficiente vs rango pronosticado
  5. Genera reports/games_signal_report_FECHA.json

Uso:
  python3 games_signal_calculator.py
  python3 games_signal_calculator.py --file reports/h2h_results_enhanced_FECHA.json
  python3 games_signal_calculator.py --min-cuota 1.60 --min-gap 3

Reglas (Nodo-40 spec):
  REGLA-G1: solo apostar zona dominante (|diff|>0.35) o coinflip (|diff|<=0.18)
  REGLA-G2: gap mínimo de 2 juegos entre límite del rango y la línea del mercado
  REGLA-G3: cuota mínima 1.50
  REGLA-G4: no combinar ganador + totales del mismo partido en el mismo combo
  REGLA-G5: máximo 3 piernas por combo de totales
  REGLA-G6: stakes máx $2,000 hasta n>=50 observaciones calibradas
"""

import json
import glob
import logging
import argparse
import requests
from datetime import datetime
from pathlib import Path

from scraping.kambi_tennis import KAMBI_BASE, KAMBI_PARAMS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Constantes de calibración (Nodo-40 spec) ─────────────────────────────────
DIFF_DOMINANTE  = 0.35   # |diff| > 0.35 → 2 sets dominantes → señal UNDER games
DIFF_COINFLIP   = 0.18   # |diff| <= 0.18 → 3 sets casi garantizado → señal OVER

MIN_GAP_JUEGOS  = 2      # brecha mínima entre límite de rango y línea del mercado
MIN_CUOTA       = 1.50   # cuota mínima para que haya EV

# N mínimo de observaciones por zona antes de ajustar thresholds (Fase 5)
MIN_N_CALIBRACION = 50


# ── Fase 5: Auto-calibración de thresholds ───────────────────────────────────
def _cargar_thresholds_calibrados() -> dict:
    """
    Lee calibracion_edge.json["games_thresholds"] y actualiza las constantes globales
    si existen thresholds guardados de una calibración previa (n>=50).
    Devuelve el dict de thresholds activos.
    """
    global DIFF_DOMINANTE, DIFF_COINFLIP
    try:
        cal_path = Path("data/calibracion_edge.json")
        if not cal_path.exists():
            return {}
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        t = cal.get("games_thresholds", {})
        if t.get("DIFF_DOMINANTE") and t.get("n_dominante", 0) >= MIN_N_CALIBRACION:
            DIFF_DOMINANTE = float(t["DIFF_DOMINANTE"])
            logger.info(f"   Fase 5: DIFF_DOMINANTE={DIFF_DOMINANTE:.3f} (n={t['n_dominante']} calibrado)")
        if t.get("DIFF_COINFLIP") and t.get("n_coinflip", 0) >= MIN_N_CALIBRACION:
            DIFF_COINFLIP = float(t["DIFF_COINFLIP"])
            logger.info(f"   Fase 5: DIFF_COINFLIP={DIFF_COINFLIP:.3f} (n={t['n_coinflip']} calibrado)")
        return t
    except Exception as e:
        logger.warning(f"   Fase 5: no se pudo cargar thresholds ({e}), usando defaults")
        return {}


def auto_calibrar_thresholds() -> dict:
    """
    Fase 5: Analiza games_calibracion en calibracion_edge.json.
    Si n_zona >= 50, ajusta los thresholds DIFF_DOMINANTE y DIFF_COINFLIP
    buscando el valor que maximiza sets_correcto% en cada zona.
    Guarda el resultado en calibracion_edge.json["games_thresholds"].

    Lógica de ajuste:
    - Para zona dominante: el threshold óptimo es el valor de diff donde
      sets_correcto pasa de False a True con mayor frecuencia.
      Si la accuracy real es buena (>75%), bajar threshold para capturar más partidos.
      Si la accuracy real es mala (<65%), subir threshold para ser más selectivo.
    - Para zona coinflip: misma lógica invertida.

    Devuelve dict con los thresholds calculados y metadata.
    """
    global DIFF_DOMINANTE, DIFF_COINFLIP

    cal_path = Path("data/calibracion_edge.json")
    if not cal_path.exists():
        logger.warning("   Fase 5: calibracion_edge.json no encontrado")
        return {}

    try:
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"   Fase 5: error leyendo calibracion: {e}")
        return {}

    obs = cal.get("games_calibracion", [])
    if not obs:
        logger.info("   Fase 5: sin observaciones en games_calibracion, nada que calibrar")
        return {}

    # Separar por zona y verificar n mínimo
    dominante_obs = [o for o in obs if o.get("zona_diff") == "dominante" and o.get("diff") is not None]
    coinflip_obs  = [o for o in obs if o.get("zona_diff") == "coinflip"  and o.get("diff") is not None]

    thresholds = cal.get("games_thresholds", {
        "DIFF_DOMINANTE": DIFF_DOMINANTE,
        "DIFF_COINFLIP":  DIFF_COINFLIP,
        "n_dominante": 0,
        "n_coinflip":  0,
        "ultima_calibracion": None,
    })

    cambios = []

    # ── Calibrar DIFF_DOMINANTE ───────────────────────────────────────────────
    n_d = len(dominante_obs)
    thresholds["n_dominante"] = n_d
    if n_d >= MIN_N_CALIBRACION:
        sets_ok_d = sum(1 for o in dominante_obs if o.get("sets_correcto") is True)
        acc_d = sets_ok_d / n_d

        # Si accuracy > 80%: podemos bajar el threshold para capturar más partidos
        # Si accuracy < 65%: subir threshold para mayor precision
        # Ajuste gradual de 0.02 para evitar oscilaciones
        nuevo_dominante = DIFF_DOMINANTE
        if acc_d > 0.80 and DIFF_DOMINANTE > 0.28:
            nuevo_dominante = round(DIFF_DOMINANTE - 0.02, 3)
            cambios.append(f"DIFF_DOMINANTE {DIFF_DOMINANTE:.3f}→{nuevo_dominante:.3f} (acc={acc_d:.1%} > 80%)")
        elif acc_d < 0.65 and DIFF_DOMINANTE < 0.50:
            nuevo_dominante = round(DIFF_DOMINANTE + 0.02, 3)
            cambios.append(f"DIFF_DOMINANTE {DIFF_DOMINANTE:.3f}→{nuevo_dominante:.3f} (acc={acc_d:.1%} < 65%)")
        else:
            cambios.append(f"DIFF_DOMINANTE sin cambio (acc={acc_d:.1%}, en rango 65-80%)")

        thresholds["DIFF_DOMINANTE"] = nuevo_dominante
        thresholds["acc_dominante"]  = round(acc_d, 4)
        DIFF_DOMINANTE = nuevo_dominante

    # ── Calibrar DIFF_COINFLIP ────────────────────────────────────────────────
    n_c = len(coinflip_obs)
    thresholds["n_coinflip"] = n_c
    if n_c >= MIN_N_CALIBRACION:
        sets_ok_c = sum(1 for o in coinflip_obs if o.get("sets_correcto") is True)
        acc_c = sets_ok_c / n_c

        nuevo_coinflip = DIFF_COINFLIP
        if acc_c > 0.75 and DIFF_COINFLIP < 0.25:
            nuevo_coinflip = round(DIFF_COINFLIP + 0.02, 3)
            cambios.append(f"DIFF_COINFLIP {DIFF_COINFLIP:.3f}→{nuevo_coinflip:.3f} (acc={acc_c:.1%} > 75%)")
        elif acc_c < 0.60 and DIFF_COINFLIP > 0.05:
            nuevo_coinflip = round(DIFF_COINFLIP - 0.02, 3)
            cambios.append(f"DIFF_COINFLIP {DIFF_COINFLIP:.3f}→{nuevo_coinflip:.3f} (acc={acc_c:.1%} < 60%)")
        else:
            cambios.append(f"DIFF_COINFLIP sin cambio (acc={acc_c:.1%}, en rango 60-75%)")

        thresholds["DIFF_COINFLIP"] = nuevo_coinflip
        thresholds["acc_coinflip"]  = round(acc_c, 4)
        DIFF_COINFLIP = nuevo_coinflip

    if cambios:
        thresholds["ultima_calibracion"] = datetime.now().isoformat()
        thresholds["log"] = cambios

        # Guardar en calibracion_edge.json
        cal["games_thresholds"] = thresholds
        cal_path.write_text(json.dumps(cal, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info("   Fase 5: thresholds calibrados:")
        for c in cambios:
            logger.info(f"     {c}")
    else:
        logger.info(f"   Fase 5: n insuficiente (dom={n_d}, coin={n_c}), se necesitan {MIN_N_CALIBRACION}/zona")

    return thresholds

# Labels Kambi de mercados relevantes
# IMPORTANTE: excluir "número total de juegos ganados por X" (mercado individual de jugador)
LABELS_TOTAL_JUEGOS = ["total de juegos"]
LABELS_TOTAL_SETS   = ["total de sets"]
LABELS_EXCLUIR      = ["número total de juegos ganados", "juegos ganados por", "- set", "con quiebre"]

# Rango de games según subzona (reutiliza lógica de generar_tabla_favoritos2.py)
def _predecir_sets_y_games(diff_abs: float, total_score: float) -> dict:
    """Replica predecir_sets_y_games() de generar_tabla_favoritos2.py."""
    if diff_abs > 0.18:
        predicted_sets = 2
        if diff_abs > 0.35:
            games_min, games_max = 16, 19
            games_range = "16-19"
        elif diff_abs > 0.25:
            games_min, games_max = 18, 21
            games_range = "18-21"
        else:
            games_min, games_max = 20, 23
            games_range = "20-23"
    else:
        predicted_sets = 3
        if total_score > 1.5:
            games_min, games_max = 26, 99   # 26-32+ → usamos 99 como infinito
            games_range = "26-32+"
        else:
            games_min, games_max = 23, 28
            games_range = "23-28"
    return {
        "predicted_sets": predicted_sets,
        "games_min": games_min,
        "games_max": games_max,
        "games_range": games_range,
    }


def _zona_diff(diff_abs: float) -> str:
    if diff_abs > DIFF_DOMINANTE:
        return "dominante"
    elif diff_abs > DIFF_COINFLIP:
        return "ajustada"
    else:
        return "coinflip"


# ── Carga de datos ────────────────────────────────────────────────────────────
def _cargar_h2h(file_path: str | None) -> tuple[list, str]:
    if file_path:
        path = file_path
    else:
        archivos = sorted(glob.glob("reports/h2h_results_enhanced_*.json"))
        if not archivos:
            raise FileNotFoundError("No se encontró ningún h2h_results_enhanced_*.json en reports/")
        path = archivos[-1]
    logger.info(f"📂 Cargando: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    partidos = data.get("partidos", [])
    logger.info(f"   {len(partidos)} partidos cargados")
    return partidos, path


# ── Kambi: betOffers de un evento ────────────────────────────────────────────
def _fetch_betoffer_event(ev_id: int) -> list:
    """Devuelve lista de betOffers para un evento Kambi."""
    url = f"{KAMBI_BASE}/betoffer/event/{ev_id}.json?{KAMBI_PARAMS}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json().get("betOffers", [])
    except Exception as e:
        logger.warning(f"   Kambi error ev_id={ev_id}: {e}")
        return []


def _apellido(nombre: str) -> str:
    """Extrae el apellido de un nombre tipo 'Choinski J.' o 'De Jong J.'
    Toma la última palabra que NO sea una inicial (una letra + punto)."""
    words = (nombre or "").split()
    for w in reversed(words):
        if not (len(w) <= 2 and w.endswith(".")):
            return w.lower()
    return (words[0] if words else "").lower()


def _buscar_event_id_kambi(partido: dict) -> int | None:
    """Busca el event_id de Kambi usando el listView de tenis."""
    j1 = _apellido(partido.get("jugador1", ""))
    j2 = _apellido(partido.get("jugador2", ""))
    if not j1 or not j2:
        return None
    try:
        url = f"{KAMBI_BASE}/listView/tennis.json?{KAMBI_PARAMS}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        events = r.json().get("events", [])
        for ev in events:
            name = ev.get("event", {}).get("name", "").lower()
            if "/" in name:  # excluir dobles (formato "A/B - C/D")
                continue
            if j1 in name and j2 in name:
                return ev.get("event", {}).get("id")
    except Exception as e:
        logger.warning(f"   Kambi listView error: {e}")
    return None


# ── Análisis de mercados disponibles ─────────────────────────────────────────
def _analizar_mercados_juegos(betoffer: list, pred: dict) -> list:
    """
    Recorre los betOffers de un evento y devuelve señales válidas.
    pred contiene: predicted_sets, games_min, games_max, games_range
    """
    señales = []
    games_min = pred["games_min"]
    games_max = pred["games_max"]
    predicted_sets = pred["predicted_sets"]

    for bo in betoffer:
        label = bo.get("criterion", {}).get("label", "").lower()
        outcomes = bo.get("outcomes", [])
        if len(outcomes) != 2:
            continue

        # Obtener línea del betOffer
        linea = None
        for out in outcomes:
            raw_line = out.get("line")
            if raw_line:
                linea = raw_line / 1000
                break
        if linea is None:
            continue

        # Excluir mercados individuales de jugador
        if any(k in label for k in LABELS_EXCLUIR):
            continue

        o_mas  = next((o for o in outcomes if o.get("label","").lower() == "más de"), None)
        o_menos= next((o for o in outcomes if o.get("label","").lower() == "menos de"), None)

        # ── Total de juegos ──────────────────────────────────────────────────
        if any(k in label for k in LABELS_TOTAL_JUEGOS):
            cuota_mas   = (o_mas.get("odds", 0)   / 1000) if o_mas   and o_mas.get("odds",0)   > 0 else None
            cuota_menos = (o_menos.get("odds", 0) / 1000) if o_menos and o_menos.get("odds",0) > 0 else None

            # UNDER: para partidos dominantes (2 sets, pocos games)
            if predicted_sets == 2:
                gap = linea - games_max
                if gap >= MIN_GAP_JUEGOS and cuota_menos and cuota_menos >= MIN_CUOTA:
                    confianza = "ALTA" if gap >= 4 else "MEDIA"
                    señales.append({
                        "mercado": "Total de juegos",
                        "linea": linea,
                        "direccion": "UNDER",
                        "cuota": cuota_menos,
                        "outcome_id": o_menos["id"] if o_menos else None,
                        "gap_juegos": round(gap, 1),
                        "razon": f"modelo dice {pred['games_range']} games, línea {linea} tiene gap +{gap:.1f} sobre el máximo del rango",
                        "confianza_señal": confianza,
                        "apostar": confianza == "ALTA" or (confianza == "MEDIA" and cuota_menos >= 1.70),
                    })

            # OVER: para partidos reñidos (3 sets, muchos games)
            elif predicted_sets == 3:
                gap = games_min - linea
                if gap >= MIN_GAP_JUEGOS and cuota_mas and cuota_mas >= MIN_CUOTA:
                    confianza = "ALTA" if gap >= 3 else "MEDIA"
                    señales.append({
                        "mercado": "Total de juegos",
                        "linea": linea,
                        "direccion": "OVER",
                        "cuota": cuota_mas,
                        "outcome_id": o_mas["id"] if o_mas else None,
                        "gap_juegos": round(gap, 1),
                        "razon": f"modelo dice {pred['games_range']} games, línea {linea} tiene gap -{gap:.1f} bajo el mínimo del rango",
                        "confianza_señal": confianza,
                        "apostar": confianza == "ALTA" or (confianza == "MEDIA" and cuota_mas >= 1.70),
                    })

        # ── Total de sets ────────────────────────────────────────────────────
        elif any(k in label for k in LABELS_TOTAL_SETS):
            # La línea siempre es 2.5 en tennis best-of-3
            cuota_mas   = (o_mas.get("odds", 0)   / 1000) if o_mas   and o_mas.get("odds",0)   > 0 else None
            cuota_menos = (o_menos.get("odds", 0) / 1000) if o_menos and o_menos.get("odds",0) > 0 else None

            cuota_mas_sets   = (o_mas.get("odds", 0)   / 1000) if o_mas   and o_mas.get("odds",0)   > 0 else None
            cuota_menos_sets = (o_menos.get("odds", 0) / 1000) if o_menos and o_menos.get("odds",0) > 0 else None

            if predicted_sets == 2 and cuota_menos_sets and cuota_menos_sets >= MIN_CUOTA:
                señales.append({
                    "mercado": "Total de sets",
                    "linea": linea,
                    "direccion": "UNDER",
                    "cuota": cuota_menos_sets,
                    "outcome_id": o_menos.get("id") if o_menos else None,
                    "gap_juegos": None,
                    "razon": f"modelo predice 2 sets, UNDER {linea} sets",
                    "confianza_señal": "MEDIA",
                    "apostar": cuota_menos_sets >= 1.60,
                })
            elif predicted_sets == 3 and cuota_mas_sets and cuota_mas_sets >= MIN_CUOTA:
                señales.append({
                    "mercado": "Total de sets",
                    "linea": linea,
                    "direccion": "OVER",
                    "cuota": cuota_mas_sets,
                    "outcome_id": o_mas.get("id") if o_mas else None,
                    "gap_juegos": None,
                    "razon": f"modelo predice 3 sets, OVER {linea} sets",
                    "confianza_señal": "MEDIA",
                    "apostar": cuota_mas_sets >= 1.70,
                })

    return señales


def _seleccionar_señal_optima(señales: list) -> list:
    """
    Por mercado (juegos / sets), devuelve la señal óptima:
    - UNDER: línea más alta con mayor cuota (más margen de seguridad)
    - OVER: línea más baja con mayor cuota (más fácil de superar)
    Solo devuelve señales con apostar=True.
    """
    apostar = [s for s in señales if s["apostar"]]
    if not apostar:
        return []

    optimas = []

    # Total de juegos UNDER → máxima línea (más margen) entre las apostables
    juegos_under = [s for s in apostar if s["mercado"] == "Total de juegos" and s["direccion"] == "UNDER"]
    if juegos_under:
        mejor = max(juegos_under, key=lambda s: (s["gap_juegos"] or 0, s["cuota"]))
        optimas.append(mejor)

    # Total de juegos OVER → mínima línea (más fácil) entre las apostables
    juegos_over = [s for s in apostar if s["mercado"] == "Total de juegos" and s["direccion"] == "OVER"]
    if juegos_over:
        mejor = max(juegos_over, key=lambda s: (s["gap_juegos"] or 0, s["cuota"]))
        optimas.append(mejor)

    # Total de sets → incluir si apostable
    sets_señales = [s for s in apostar if s["mercado"] == "Total de sets"]
    if sets_señales:
        mejor = max(sets_señales, key=lambda s: s["cuota"])
        optimas.append(mejor)

    return optimas


# ── Procesamiento principal ───────────────────────────────────────────────────
def procesar_partidos(partidos: list, min_cuota: float, min_gap: float) -> list:
    global MIN_CUOTA, MIN_GAP_JUEGOS
    MIN_CUOTA = min_cuota
    MIN_GAP_JUEGOS = min_gap

    resultados = []
    # Cache del listView para no hacer 14 llamadas
    _listview_cache = None

    def _get_listview():
        nonlocal _listview_cache
        if _listview_cache is None:
            try:
                url = f"{KAMBI_BASE}/listView/tennis.json?{KAMBI_PARAMS}"
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                _listview_cache = r.json().get("events", [])
            except Exception as e:
                logger.warning(f"   Error listView: {e}")
                _listview_cache = []
        return _listview_cache

    # Cargar betslip_index más reciente para cruzar outcome_ids ya conocidos
    betslip_index = {}
    betslip_files = sorted(glob.glob("reports/betslip_index_*.json"))
    if betslip_files:
        try:
            with open(betslip_files[-1], encoding="utf-8") as f:
                raw_idx = json.load(f).get("index", {})
            for oid, meta in raw_idx.items():
                mid = meta.get("match_id", "")
                if mid:
                    betslip_index[mid] = meta.get("kambi_event_id") or meta.get("event_id")
            logger.info(f"   betslip_index cargado: {len(betslip_index)} match_ids")
        except Exception:
            pass

    for i, p in enumerate(partidos):
        j1 = p.get("jugador1", "")
        j2 = p.get("jugador2", "")
        nombre = f"{j1} vs {j2}"
        logger.info(f"[{i+1}/{len(partidos)}] {nombre}")

        # 1. Extraer score_difference y scores del modelo
        pred_block = p.get("ranking_analysis", {}).get("prediction", {})
        scores = pred_block.get("scores", {})
        score_difference = scores.get("score_difference", 0) or 0
        p1_score = scores.get("p1_final_weight", 0) or 0
        p2_score = scores.get("p2_final_weight", 0) or 0
        total_score = p1_score + p2_score
        diff_abs = abs(score_difference)

        # 2. Calcular zona y predicción
        zona = _zona_diff(diff_abs)
        pred = _predecir_sets_y_games(diff_abs, total_score)

        resultado_base = {
            "partido": nombre,
            "jugador1": j1,
            "jugador2": j2,
            "torneo": p.get("torneo_nombre", p.get("torneo_completo", "")),
            "superficie": p.get("tipo_cancha", "unknown"),
            "tier": p.get("tier", ""),
            "match_id": p.get("match_id", ""),
            "score_difference": round(score_difference, 4),
            "diff_abs": round(diff_abs, 4),
            "zona_diff": zona,
            "predicted_sets": pred["predicted_sets"],
            "games_range": pred["games_range"],
            "señales": [],
            "señales_optimas": [],
            "tiene_mercados_kambi": False,
        }

        # 3. Solo procesar si tiene señal (no zona ajustada sin cuota alta)
        if zona == "ajustada":
            logger.info(f"   ⏭️  zona ajustada (diff={diff_abs:.2f}) — no apostar totales")
            resultados.append(resultado_base)
            continue

        # 4. Buscar event_id en Kambi
        match_id = p.get("match_id", "")
        ev_id = None

        # Intento 1: betslip_index (más confiable — IDs ya resueltos hoy)
        if match_id and match_id in betslip_index and betslip_index[match_id]:
            ev_id = betslip_index[match_id]
            logger.debug(f"   betslip_index hit: ev_id={ev_id}")

        # Intento 2: listView por apellido (funciona para partidos futuros)
        if not ev_id:
            events = _get_listview()
            j1_parts = j1.lower().split()
            j2_parts = j2.lower().split()
            apellido1 = j1_parts[-1] if j1_parts else ""
            apellido2 = j2_parts[-1] if j2_parts else ""
            for ev in events:
                ev_name = ev.get("event", {}).get("name", "").lower()
                if apellido1 in ev_name and apellido2 in ev_name:
                    ev_id = ev.get("event", {}).get("id")
                    break

        # Intento 3: búsqueda amplia en listView por primer apellido
        if not ev_id:
            events = _get_listview()
            j1_parts = j1.lower().split()
            apellido1 = j1_parts[-1] if j1_parts else ""
            for ev in events:
                ev_name = ev.get("event", {}).get("name", "").lower()
                if apellido1 in ev_name:
                    ev_id = ev.get("event", {}).get("id")
                    logger.debug(f"   apellido1 fallback: {ev_name}")
                    break

        if not ev_id:
            logger.info(f"   ⚠️  no encontrado en Kambi (partidos ya jugados o sin mercado)")
            resultados.append(resultado_base)
            continue

        # 5. Obtener betOffers
        betoffer = _fetch_betoffer_event(ev_id)
        resultado_base["tiene_mercados_kambi"] = bool(betoffer)
        resultado_base["kambi_event_id"] = ev_id

        if not betoffer:
            logger.info(f"   ⚠️  sin betOffers en Kambi")
            resultados.append(resultado_base)
            continue

        # 6. Analizar señales disponibles
        señales = _analizar_mercados_juegos(betoffer, pred)
        optimas = _seleccionar_señal_optima(señales)

        resultado_base["señales"] = señales
        resultado_base["señales_optimas"] = optimas

        n_apostar = len(optimas)
        if n_apostar:
            for s in optimas:
                logger.info(f"   ✅ {s['mercado']} {s['direccion']} {s['linea']} @{s['cuota']:.2f} [{s['confianza_señal']}] gap={s.get('gap_juegos','N/A')}")
        else:
            logger.info(f"   — sin señales apostables ({len(señales)} candidatas descartadas)")

        resultados.append(resultado_base)

    return resultados


# ── Output y reporte ──────────────────────────────────────────────────────────
def imprimir_reporte(resultados: list):
    apostar = [r for r in resultados if r["señales_optimas"]]
    candidatas = [r for r in resultados if r["señales"] and not r["señales_optimas"]]

    print()
    print("═" * 66)
    print("  GAMES SIGNAL CALCULATOR — Nodo-40")
    print("═" * 66)
    print(f"  Partidos analizados : {len(resultados)}")
    print(f"  Con señales APOSTAR : {len(apostar)}")
    print(f"  Candidatas (no aptos): {len(candidatas)}")
    print()

    if apostar:
        print("  ✅ SEÑALES APOSTABLES:")
        print()
        for r in apostar:
            print(f"  🎾 {r['partido']}")
            print(f"     Zona: {r['zona_diff'].upper()} | diff={r['diff_abs']:.2f} | "
                  f"Modelo: {r['predicted_sets']} sets, {r['games_range']} games")
            for s in r["señales_optimas"]:
                print(f"     → {s['mercado']} {s['direccion']} {s['linea']} "
                      f"@{s['cuota']:.2f} [{s['confianza_señal']}] "
                      f"gap={s.get('gap_juegos', 'N/A')} | id={s['outcome_id']}")
                print(f"       {s['razon']}")
            print()

    if candidatas:
        print("  👀 CANDIDATAS (sin señal suficiente):")
        for r in candidatas:
            total = len(r["señales"])
            print(f"     {r['partido']} — {total} señal(es) bajo threshold")
    print()

    # Combos sugeridos — REGLA-G4: max 1 señal por partido en el mismo combo
    # Tomar la mejor señal por partido (mayor confianza, luego mayor cuota)
    def _ranking_señal(s):
        conf_ord = {"ALTA": 2, "MEDIA": 1}
        return (conf_ord.get(s["confianza_señal"], 0), s["cuota"])

    mejores_por_partido = []
    for r in apostar:
        if r["señales_optimas"]:
            mejor = max(r["señales_optimas"], key=_ranking_señal)
            mejores_por_partido.append((r, mejor))

    if len(mejores_por_partido) >= 2:
        print("  📦 COMBOS SUGERIDOS:")
        # Combo A: 2 mejores partidos distintos
        if len(mejores_por_partido) >= 2:
            a, b = mejores_por_partido[0], mejores_por_partido[1]
            cuota_combo = round(a[1]["cuota"] * b[1]["cuota"], 2)
            ids = f"{a[1]['outcome_id']},{b[1]['outcome_id']}"
            print(f"     Combo A (2p @{cuota_combo}x): "
                  f"{a[0]['partido']} {a[1]['direccion']} {a[1]['linea']} + "
                  f"{b[0]['partido']} {b[1]['direccion']} {b[1]['linea']}")
            print(f"       IDs Kambi: {ids}")
        # Combo B: 3 partidos distintos
        if len(mejores_por_partido) >= 3:
            a, b, c = mejores_por_partido[0], mejores_por_partido[1], mejores_por_partido[2]
            cuota_combo = round(a[1]["cuota"] * b[1]["cuota"] * c[1]["cuota"], 2)
            ids = f"{a[1]['outcome_id']},{b[1]['outcome_id']},{c[1]['outcome_id']}"
            print(f"     Combo B (3p @{cuota_combo}x): "
                  f"{a[0]['partido']} {a[1]['direccion']} {a[1]['linea']} + "
                  f"{b[0]['partido']} {b[1]['direccion']} {b[1]['linea']} + "
                  f"{c[0]['partido']} {c[1]['direccion']} {c[1]['linea']}")
            print(f"       IDs Kambi: {ids}")
        print()

    print("  📌 REGLA-G6: stakes máx $2,000/combo hasta n≥50 observaciones")
    print("═" * 66)


def guardar_reporte(resultados: list, source_file: str) -> str:
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"games_signal_report_{ts}.json"

    # Leer n real de observaciones acumuladas desde calibracion_edge.json
    _n_cal = 0
    try:
        _cal_path = Path("data/calibracion_edge.json")
        if _cal_path.exists():
            _cal = json.loads(_cal_path.read_text(encoding="utf-8"))
            _n_cal = len(_cal.get("games_calibracion", []))
    except Exception:
        pass

    # Solo incluir partidos con mercados o señales
    resumen_apostar = [
        {
            "partido": r["partido"],
            "zona_diff": r["zona_diff"],
            "diff_abs": r["diff_abs"],
            "predicted_sets": r["predicted_sets"],
            "games_range": r["games_range"],
            "señales_optimas": r["señales_optimas"],
        }
        for r in resultados if r["señales_optimas"]
    ]

    output = {
        "metadata": {
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fuente": source_file,
            "n_partidos": len(resultados),
            "n_apostar": len(resumen_apostar),
            "nodo": "Nodo-40-Games-Sets-Signal-Layer",
            "calibracion_n": _n_cal,
        },
        "apostar": resumen_apostar,
        "detalle_completo": resultados,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"💾 Reporte guardado: {filename}")
    return str(filename)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Nodo-40: calcula señales de mercado de totales (games/sets)"
    )
    parser.add_argument("--file", type=str, default=None,
                        help="Ruta al h2h_results_enhanced_*.json (default: más reciente)")
    parser.add_argument("--min-cuota", type=float, default=1.50,
                        help="Cuota mínima para señal (default: 1.50)")
    parser.add_argument("--min-gap", type=float, default=2.0,
                        help="Gap mínimo de juegos entre rango y línea (default: 2.0)")
    parser.add_argument("--calibrar", action="store_true",
                        help="Fase 5: recalibrar thresholds DIFF_DOMINANTE/COINFLIP con datos acumulados")
    args = parser.parse_args()

    # Fase 5: cargar thresholds calibrados al inicio (usa defaults si n<50)
    _cargar_thresholds_calibrados()

    # Fase 5: recalibrar explícitamente si se pide
    if args.calibrar:
        auto_calibrar_thresholds()
        return

    partidos, source = _cargar_h2h(args.file)
    resultados = procesar_partidos(partidos, args.min_cuota, args.min_gap)
    imprimir_reporte(resultados)
    guardar_reporte(resultados, source)


if __name__ == "__main__":
    main()
