#!/usr/bin/env python3
"""D181-01 — lead_time_report.py: el reloj del sismografo (Nodo-181 SS3).

Cruza el instante de disparo de certeza_matematica (`reports/certeza_fired_{fecha}.json`)
con la serie de cuotas en vivo (`reports/games_odds_history_{fecha}.json`) para medir,
por primera vez, cuanto dura la ventana de oportunidad despues de cada disparo.

REPORTE_SOLO estricto (Nodo-181 D181-01): este script no debe ser importado ni leido
por ningun gate, calculo de stake o generador de combos. Su unico consumidor es el
propio reporte JSON/texto y el ojo humano.
"""
import argparse
import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def _parse_serie_datetimes(serie, fecha_base):
    """Convierte los ts 'HH:MM' de una serie en datetimes absolutos.

    Suma 1 dia cada vez que el reloj retrocede respecto al punto anterior —
    sin este guard las series que cruzan medianoche producen ventanas negativas
    (Nodo-181 D181-01, requisito duro).
    """
    out = []
    day_offset = 0
    prev_mins = None
    for punto in serie:
        h, m = map(int, punto["ts"].split(":"))
        mins = h * 60 + m
        if prev_mins is not None and mins < prev_mins:
            day_offset += 1
        prev_mins = mins
        out.append(fecha_base + timedelta(days=day_offset, hours=h, minutes=m))
    return out


def procesar_disparo(hora_disparo_iso, serie, fecha_base):
    hora_disparo = datetime.fromisoformat(hora_disparo_iso)
    dts = _parse_serie_datetimes(serie, fecha_base)
    cuotas = [p["cuota"] for p in serie]

    cuota_t0 = cuotas[0]
    antes_idx = [i for i, dt in enumerate(dts) if dt <= hora_disparo]
    if antes_idx:
        cuota_fire = cuotas[antes_idx[-1]]
        n_antes = len(antes_idx)
    else:
        cuota_fire = cuota_t0
        n_antes = 0
    cuota_final = cuotas[-1]
    n_despues = len(serie) - n_antes

    mov_antes_pct = abs(cuota_fire - cuota_t0) / cuota_t0 * 100 if cuota_t0 else 0.0
    mov_despues_pct = abs(cuota_final - cuota_fire) / cuota_fire * 100 if cuota_fire else 0.0
    ventana_min = max((dts[-1] - hora_disparo).total_seconds() / 60.0, 0.0)

    if cuota_final < cuota_fire:
        direccion = "A_FAVOR"
    elif cuota_final > cuota_fire:
        direccion = "EN_CONTRA"
    else:
        direccion = "PLANO"

    return {
        "hora_disparo": hora_disparo_iso,
        "cuota_t0": cuota_t0,
        "cuota_fire": cuota_fire,
        "cuota_final": cuota_final,
        "mov_antes_pct": round(mov_antes_pct, 2),
        "mov_despues_pct": round(mov_despues_pct, 2),
        "ventana_min": round(ventana_min, 1),
        "n_puntos_antes": n_antes,
        "n_puntos_despues": n_despues,
        "direccion_movimiento": direccion,
    }


def procesar_dia(fecha_compact):
    """fecha_compact: 'YYYYMMDD'."""
    fired_path = REPORTS_DIR / f"certeza_fired_{fecha_compact}.json"
    if not fired_path.exists():
        return []
    fired = json.loads(fired_path.read_text(encoding="utf-8"))

    odds_path = REPORTS_DIR / f"games_odds_history_{fecha_compact}.json"
    odds = json.loads(odds_path.read_text(encoding="utf-8")) if odds_path.exists() else {}

    fecha_base = datetime.strptime(fecha_compact, "%Y%m%d")
    resultados = []
    for clave, valor in fired.items():
        # D180-06 escribe {"ts":..., "direccion":...}; archivos historicos
        # anteriores a ese cambio guardan el ISO como string plano. Ambos
        # formatos deben leerse — Nodo-181 D181-01 no puede depender de que
        # nadie migre el histórico retroactivamente.
        hora_disparo_iso = valor["ts"] if isinstance(valor, dict) else valor
        serie = odds.get(clave)
        if not serie:
            resultados.append({
                "clave": clave, "fecha": fecha_compact,
                "categoria": "SIN_HISTORIAL", "hora_disparo": hora_disparo_iso,
            })
            continue
        r = procesar_disparo(hora_disparo_iso, serie, fecha_base)
        r["clave"] = clave
        r["fecha"] = fecha_compact
        r["categoria"] = "MEDIDO"
        resultados.append(r)
    return resultados


def _percentil(vals_ordenados, p):
    if not vals_ordenados:
        return None
    k = (len(vals_ordenados) - 1) * p
    f, c = int(k), min(int(k) + 1, len(vals_ordenados) - 1)
    if f == c:
        return vals_ordenados[f]
    return vals_ordenados[f] + (vals_ordenados[c] - vals_ordenados[f]) * (k - f)


def calcular_agregados(resultados):
    medidos = [r for r in resultados if r["categoria"] == "MEDIDO"]
    n_total, n_medidos = len(resultados), len(medidos)
    base = {"n_total": n_total, "n_medidos": n_medidos, "n_sin_historial": n_total - n_medidos}
    if not medidos:
        return {**base, "pct_ventana_cero": None, "ventana_min_mediana": None,
                "ventana_min_p25": None, "ventana_min_p75": None,
                "mov_despues_pct_mediana": None, "mov_despues_pct_p25": None,
                "mov_despues_pct_p75": None, "n_a_favor": 0, "n_en_contra": 0, "n_plano": 0}

    ventanas = sorted(r["ventana_min"] for r in medidos)
    movs = sorted(r["mov_despues_pct"] for r in medidos)
    n_cero = sum(1 for r in medidos if r["mov_despues_pct"] < 1.0)
    return {
        **base,
        "pct_ventana_cero": round(n_cero / n_medidos * 100, 1),
        "ventana_min_mediana": round(statistics.median(ventanas), 1),
        "ventana_min_p25": round(_percentil(ventanas, 0.25), 1),
        "ventana_min_p75": round(_percentil(ventanas, 0.75), 1),
        "mov_despues_pct_mediana": round(statistics.median(movs), 1),
        "mov_despues_pct_p25": round(_percentil(movs, 0.25), 1),
        "mov_despues_pct_p75": round(_percentil(movs, 0.75), 1),
        "n_a_favor": sum(1 for r in medidos if r["direccion_movimiento"] == "A_FAVOR"),
        "n_en_contra": sum(1 for r in medidos if r["direccion_movimiento"] == "EN_CONTRA"),
        "n_plano": sum(1 for r in medidos if r["direccion_movimiento"] == "PLANO"),
    }


def _leer_apuestas_registradas(fecha_compact):
    """D181-11: lee todos los `reports/apuestas_{fecha_compact}_*.json` del dia
    (un archivo por cada captura del bookmarklet, `betslip_registrar.py::registrar()`
    D87-09) y devuelve `[(ts_registro_iso, partido), ...]` — solo picks con
    `partido` no vacio son matcheables (los `fuera_de_index=True` no traen
    partido resuelto, se descartan aqui sin inventar un valor)."""
    out = []
    for path in sorted(REPORTS_DIR.glob(f"apuestas_{fecha_compact}_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        ts_registro = data.get("ts_registro")
        if not ts_registro:
            continue
        for p in data.get("picks", []):
            partido = p.get("partido")
            if partido:
                out.append((ts_registro, partido))
    return out


def procesar_alerta_a_apuesta(fecha_compact):
    """D181-11: mide el tiempo entre el disparo de alerta VENTANA (`fire_ledger`,
    D181-02/08) y el momento en que el usuario registra la apuesta que
    realmente coloco (`reports/apuestas_*.json`).

    Match a nivel de PARTIDO, no de pierna/direccion: `apuestas_*.json` no
    guarda que direccion (OVER/UNDER) se jugo, solo outcome_id+jugador+partido
    (formato heredado de picks ganador-de-partido, D87-09). Emparejar por
    direccion inventaria un campo que el dato de origen no tiene — se declara
    la limitacion en vez de forzar un match falso.

    REPORTE_SOLO (mismo limite D181-01): ningun gate/stake lee esta funcion.
    """
    ledger_path = REPORTS_DIR / f"fire_ledger_{fecha_compact}.jsonl"
    if not ledger_path.exists():
        return []

    disparos = []
    with ledger_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entrada = json.loads(line)
            except Exception:
                continue
            if entrada.get("tipo") == "VENTANA":
                disparos.append(entrada)

    apuestas = _leer_apuestas_registradas(fecha_compact)

    resultados = []
    for d in disparos:
        clave = d.get("clave", "")
        partido = clave.rsplit("_", 1)[0] if "_" in clave else clave
        ts_alerta_iso = d.get("ts_iso")
        if not ts_alerta_iso:
            continue
        try:
            ts_alerta = datetime.fromisoformat(ts_alerta_iso)
        except Exception:
            continue

        candidatas = []
        for ts_reg_iso, p_apuesta in apuestas:
            if p_apuesta != partido:
                continue
            try:
                ts_reg = datetime.fromisoformat(ts_reg_iso)
            except Exception:
                continue
            if ts_reg >= ts_alerta:
                candidatas.append(ts_reg)

        if not candidatas:
            resultados.append({
                "clave": clave, "ts_alerta": ts_alerta_iso,
                "categoria": "SIN_APUESTA_REGISTRADA",
            })
            continue

        ts_apuesta = min(candidatas)
        minutos = round((ts_apuesta - ts_alerta).total_seconds() / 60.0, 1)
        resultados.append({
            "clave": clave, "ts_alerta": ts_alerta_iso,
            "ts_apuesta": ts_apuesta.isoformat(), "minutos_friccion": minutos,
            "categoria": "MEDIDO",
        })
    return resultados


def calcular_agregados_friccion(resultados):
    medidos = [r for r in resultados if r["categoria"] == "MEDIDO"]
    n_total, n_medidos = len(resultados), len(medidos)
    base = {"n_total": n_total, "n_medidos": n_medidos,
            "n_sin_apuesta_registrada": n_total - n_medidos}
    if not medidos:
        return {**base, "minutos_friccion_mediana": None,
                "minutos_friccion_p25": None, "minutos_friccion_p75": None}
    vals = sorted(r["minutos_friccion"] for r in medidos)
    return {
        **base,
        "minutos_friccion_mediana": round(statistics.median(vals), 1),
        "minutos_friccion_p25": round(_percentil(vals, 0.25), 1),
        "minutos_friccion_p75": round(_percentil(vals, 0.75), 1),
    }


def _rango_fechas_compact(desde, hasta):
    d0 = datetime.strptime(desde, "%Y-%m-%d")
    d1 = datetime.strptime(hasta, "%Y-%m-%d")
    out, d = [], d0
    while d <= d1:
        out.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return out


def main():
    ap = argparse.ArgumentParser(description="D181-01: reloj del sismografo (REPORTE_SOLO)")
    ap.add_argument("--desde", help="YYYY-MM-DD")
    ap.add_argument("--hasta", help="YYYY-MM-DD")
    args = ap.parse_args()

    if args.desde and args.hasta:
        fechas = _rango_fechas_compact(args.desde, args.hasta)
    else:
        fechas = [datetime.now().strftime("%Y%m%d")]

    resultados = []
    for fecha in fechas:
        resultados.extend(procesar_dia(fecha))

    agregados = calcular_agregados(resultados)

    resultados_friccion = []
    for fecha in fechas:
        resultados_friccion.extend(procesar_alerta_a_apuesta(fecha))
    agregados_friccion = calcular_agregados_friccion(resultados_friccion)

    salida = {
        "generado": datetime.now().isoformat(),
        "fechas": fechas,
        "agregados": agregados,
        "disparos": resultados,
        "alerta_a_apuesta": {  # D181-11
            "agregados": agregados_friccion,
            "disparos": resultados_friccion,
        },
    }

    out_path = REPORTS_DIR / f"lead_time_report_{fechas[-1]}.json"
    out_path.write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"D181-01 lead_time_report — {agregados['n_total']} disparos "
          f"({agregados['n_medidos']} medidos, {agregados['n_sin_historial']} SIN_HISTORIAL)")
    if agregados["n_medidos"]:
        print(f"  ventana_min: mediana={agregados['ventana_min_mediana']} "
              f"p25={agregados['ventana_min_p25']} p75={agregados['ventana_min_p75']}")
        print(f"  mov_despues_pct: mediana={agregados['mov_despues_pct_mediana']} "
              f"p25={agregados['mov_despues_pct_p25']} p75={agregados['mov_despues_pct_p75']}")
        print(f"  pct_ventana_cero (mov<1%): {agregados['pct_ventana_cero']}%")
        print(f"  direccion: A_FAVOR={agregados['n_a_favor']} "
              f"EN_CONTRA={agregados['n_en_contra']} PLANO={agregados['n_plano']}")

    print(f"D181-11 alerta→apuesta: {agregados_friccion['n_total']} disparos VENTANA "
          f"({agregados_friccion['n_medidos']} con apuesta registrada matcheada)")
    if agregados_friccion["n_medidos"]:
        print(f"  minutos_friccion: mediana={agregados_friccion['minutos_friccion_mediana']} "
              f"p25={agregados_friccion['minutos_friccion_p25']} "
              f"p75={agregados_friccion['minutos_friccion_p75']}")

    print(f"Guardado: {out_path}")


if __name__ == "__main__":
    main()
