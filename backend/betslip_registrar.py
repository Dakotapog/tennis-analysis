"""
betslip_registrar.py — Cierra el loop: bookmarklet → apuestas → resultados

Flujo completo:
  PASO 1: python3 betplay_combo_builder.py --live --telegram
          → genera reports/betslip_index_FECHA.json automáticamente

  PASO 2: python3 betslip_registrar.py --listen   ← dejar corriendo
          Cargar combo en Betplay → click bookmarklet "Leer Betslip" (antes de confirmar)
          → POST automático → "Apuesta registrada" → reports/apuestas_YYYYMMDD_HHMMSS.json

  PASO 4 (post-partido): python3 betslip_registrar.py --cerrar
          → consulta FlashScore por resultado real de cada pick
          → registra win/loss → actualiza data/calibracion_edge.json

  UTILIDAD: python3 betslip_registrar.py --estado
          → lista todas las apuestas pendientes de cerrar
"""

import json
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import FLASHSCORE_BASE, FLASHSCORE_HEADERS as HEADERS

LISTEN_PORT = 5001

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

REPORTS_DIR    = Path("reports")
CALIBRACION    = Path("data/calibracion_edge.json")
DELAY_REQUESTS = 0.3  # segundos entre requests FlashScore


# ══════════════════════════════════════════════════════════════════════════════
# Trader plan — stake Kelly por jugador
# ══════════════════════════════════════════════════════════════════════════════

def _cargar_trader_plan() -> dict:
    """Carga el trader_plan más reciente de reports/."""
    planes = sorted(REPORTS_DIR.glob("trader_plan_*.json"), reverse=True)
    if not planes:
        return {}
    return json.loads(planes[0].read_text(encoding="utf-8"))


def _match_stake(jugador: str, trader_plan: dict) -> dict:
    """
    Busca el stake Kelly para un jugador en trader_plan["individuales"].
    Match 3-tier: exact → surname-only → substring ≥4 chars.
    Retorna {"stake": N, "retorno_potencial": N}.
    """
    individuales = trader_plan.get("individuales", [])
    if not individuales:
        return {"stake": 0, "retorno_potencial": 0}

    jugador_lower = jugador.lower().strip()
    jugador_parts = jugador_lower.split()
    apellido = jugador_parts[-1] if jugador_parts else ""

    # Tier 1: exact match
    for ind in individuales:
        if ind.get("favorito", "").lower().strip() == jugador_lower:
            return {"stake": ind.get("stake", 0), "retorno_potencial": ind.get("retorno_potencial", 0)}

    # Tier 2: surname match
    for ind in individuales:
        fav = ind.get("favorito", "").lower()
        if apellido and apellido in fav.split():
            return {"stake": ind.get("stake", 0), "retorno_potencial": ind.get("retorno_potencial", 0)}

    # Tier 3: substring ≥4 chars
    for ind in individuales:
        fav = ind.get("favorito", "").lower()
        if apellido and len(apellido) >= 4 and apellido in fav:
            return {"stake": ind.get("stake", 0), "retorno_potencial": ind.get("retorno_potencial", 0)}

    return {"stake": 0, "retorno_potencial": 0}


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades FlashScore — misma lógica que resultados_finales.py
# ══════════════════════════════════════════════════════════════════════════════

def _obtener_resultado(match_id: str) -> dict:
    """
    Consulta FlashScore dc_1_{match_id} y retorna resultado.
    Returns: {"status": "FT"|"LIVE"|"NS"|"ERROR", "ganador_lado": "jugador1"|"jugador2",
              "_raw": datos_completos}
    """
    import requests, time

    if not match_id:
        return {"status": "NO_MATCH_ID"}

    url = f"{FLASHSCORE_BASE}/dc_1_{match_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}

    datos = {}
    for par in r.text.split("¬"):
        if "÷" in par:
            k, v = par.split("÷", 1)
            datos[k] = v

    dj = datos.get("DJ", "")
    if dj in ("H", "A"):
        return {
            "status":       "FT",
            "ganador_lado": "jugador1" if dj == "H" else "jugador2",
            "sets_local":   datos.get("DE", "0"),
            "sets_visit":   datos.get("DF", "0"),
            "_raw":         datos,
        }

    try:
        dc_ts = int(datos.get("DC", "0"))
        if dc_ts and datetime.fromtimestamp(dc_ts) > datetime.now():
            return {"status": "NS"}
    except (ValueError, TypeError):
        pass

    return {"status": "LIVE"}


def _calcular_juegos_totales(datos_raw: dict) -> Optional[int]:
    """
    Intenta sumar los juegos por set desde los campos FlashScore.
    FlashScore tennis usa pares consecutivos para los marcadores por set:
      EG/EH = set 1 (home/away), EI/EJ = set 2, EK/EL = set 3
    Si el patrón no se encuentra, devuelve None.
    """
    pares_por_set = [("EG", "EH"), ("EI", "EJ"), ("EK", "EL")]
    total = 0
    encontrado = False
    for home_key, away_key in pares_por_set:
        h = datos_raw.get(home_key)
        a = datos_raw.get(away_key)
        if h is None or a is None:
            continue
        try:
            total += int(h) + int(a)
            encontrado = True
        except ValueError:
            continue
    return total if encontrado else None


def _check_games_en_rango(games_real: Optional[int], games_range: str) -> Optional[bool]:
    """Verifica si games_real cae en el rango pronosticado (ej. '16-19', '26-32+')."""
    if games_real is None or not games_range:
        return None
    try:
        if "+" in games_range:
            g_min = int(games_range.replace("+", "").split("-")[0])
            return games_real >= g_min
        parts = games_range.split("-")
        g_min, g_max = int(parts[0]), int(parts[1])
        return g_min <= games_real <= g_max
    except (ValueError, IndexError):
        return None


def _cargar_games_signal_index() -> dict:
    """
    Carga el games_signal_report más reciente → dict indexado por match_id y partido.
    Busca en detalle_completo (que sí contiene match_id).
    """
    import glob as _glob
    files = sorted(_glob.glob(str(REPORTS_DIR / "games_signal_report_*.json")), reverse=True)
    if not files:
        return {}
    try:
        data = json.loads(Path(files[0]).read_text(encoding="utf-8"))
    except Exception:
        return {}
    index = {}
    for entry in data.get("detalle_completo", []):
        mid = entry.get("match_id", "")
        if mid:
            index[mid] = entry
        partido = entry.get("partido", "")
        if partido:
            index[partido] = entry
    return index


# ══════════════════════════════════════════════════════════════════════════════
# Betslip index — carga el más reciente
# ══════════════════════════════════════════════════════════════════════════════

def _cargar_betslip_index(ts_bookmarklet: Optional[str] = None) -> tuple[dict, str]:
    """
    Carga el betslip_index más reciente (o el más cercano al timestamp del bookmarklet).

    Returns:
        (index_dict, path_string) — index mapeando outcome_id → pick_info
    """
    REPORTS_DIR.mkdir(exist_ok=True)
    indices = sorted(REPORTS_DIR.glob("betslip_index_*.json"), reverse=True)

    if not indices:
        logger.error("❌ No hay betslip_index en reports/. Ejecuta primero: python3 betplay_combo_builder.py --live")
        return {}, ""

    # Si hay timestamp del bookmarklet, usar el índice más cercano anterior
    if ts_bookmarklet:
        try:
            ts_bet = datetime.fromisoformat(ts_bookmarklet.replace("Z", "+00:00"))
            # Ordenar por proximidad temporal — tomar el más reciente antes del bookmarklet
            for idx_path in indices:
                idx_data = json.loads(idx_path.read_text(encoding="utf-8"))
                ts_idx = datetime.fromisoformat(idx_data.get("ts", "").replace("Z", "+00:00"))
                if ts_idx <= ts_bet:
                    logger.info(f"   📎 Usando betslip_index: {idx_path.name}")
                    return idx_data.get("index", {}), str(idx_path)
        except Exception:
            pass  # Fallback al más reciente

    # Fallback: el más reciente
    idx_path = indices[0]
    logger.info(f"   📎 Usando betslip_index: {idx_path.name}")
    idx_data = json.loads(idx_path.read_text(encoding="utf-8"))
    return idx_data.get("index", {}), str(idx_path)


# ══════════════════════════════════════════════════════════════════════════════
# MODO 1 — Registrar apuesta desde JSON del bookmarklet
# ══════════════════════════════════════════════════════════════════════════════

def registrar(bookmarklet_json: str):
    """
    Recibe el JSON copiado del bookmarklet y lo mapea a picks con nombres.
    Guarda reports/apuestas_FECHA.json.
    """
    try:
        data = json.loads(bookmarklet_json)
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON inválido: {e}")
        logger.error("   Asegurate de pegar el JSON completo entre comillas simples.")
        sys.exit(1)

    raw_picks = data.get("picks", [])
    ts_bookmarklet = data.get("ts", datetime.now().isoformat())

    if not raw_picks:
        logger.error("❌ El JSON no tiene picks. ¿Había un combo cargado cuando hiciste click?")
        sys.exit(1)

    logger.info(f"📋 {len(raw_picks)} Kambi outcome IDs recibidos")

    # Cargar betslip_index
    betslip_index, idx_path = _cargar_betslip_index(ts_bookmarklet)
    if not betslip_index:
        sys.exit(1)

    # Mapear cada outcome_id → pick_info
    picks_registrados = []
    no_encontrados = []

    for raw in raw_picks:
        oid = str(raw.get("id", ""))
        if not oid:
            continue

        info = betslip_index.get(oid)
        if info:
            picks_registrados.append({
                "outcome_id":        oid,
                "jugador":           info["jugador"],
                "cuota":             info["cuota"],
                "cuota_kambi":       info.get("cuota_kambi", info["cuota"]),
                "partido":           info.get("partido", ""),
                "match_id":          info.get("match_id", ""),
                "match_url":         info.get("match_url", ""),
                "torneo":            info.get("torneo", ""),
                "superficie":        info.get("superficie", "unknown"),
                "tier":              info.get("tier", "unknown"),
                "edge":              info.get("edge", "0%"),
                "p_modelo":          info.get("p_modelo", 0.5),
                "kelly_kl":          info.get("kelly_kl", 0.0),
                "stake":             0,
                "retorno_potencial": 0,
                "resultado_real":    None,
                "correcto":          None,
                "ganancia":          None,
            })
            logger.info(f"   ✅ {info['jugador']:25s} @{info['cuota']:.2f} — {info.get('partido','')}")
        else:
            no_encontrados.append(oid)
            logger.warning(f"   ⚠️  outcome_id {oid} no está en el betslip_index (pick manual o combo viejo)")

    if not picks_registrados:
        logger.error("❌ Ningún pick mapeado. ¿Usaste el combo correcto del índice?")
        sys.exit(1)

    if no_encontrados:
        logger.warning(f"   {len(no_encontrados)} picks sin mapear — omitidos del registro")

    # Agregar stake desde trader_plan
    plan = _cargar_trader_plan()
    if plan:
        for pick in picks_registrados:
            s = _match_stake(pick["jugador"], plan)
            pick["stake"] = s["stake"]
            pick["retorno_potencial"] = s["retorno_potencial"]
            if s["stake"]:
                logger.info(f"   💰 {pick['jugador']:25s} stake=${s['stake']:,.0f} → retorno=${s['retorno_potencial']:,.0f}")
    else:
        logger.warning("   ⚠️  Sin trader_plan — stake=0. Corré trader_ev_tenis.py primero.")

    # Guardar apuestas_FECHA.json
    REPORTS_DIR.mkdir(exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"apuestas_{ts_str}.json"

    apuesta = {
        "ts_registro":      datetime.now().isoformat(),
        "ts_bookmarklet":   ts_bookmarklet,
        "estado":           "PENDIENTE",
        "betslip_index":    idx_path,
        "n_picks":          len(picks_registrados),
        "picks":            picks_registrados,
    }

    out_path.write_text(json.dumps(apuesta, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"\n✅ Apuesta registrada: {out_path}")
    logger.info(f"   {len(picks_registrados)} picks mapeados")
    logger.info(f"   Cuando termine(n) el/los partido(s), ejecuta:")
    logger.info(f"   python3 betslip_registrar.py --cerrar")


# ══════════════════════════════════════════════════════════════════════════════
# MODO 2 — Cerrar sesión post-partido
# ══════════════════════════════════════════════════════════════════════════════

def cerrar(archivo: Optional[str] = None):
    """
    Lee apuestas_*.json PENDIENTES, consulta FlashScore por resultado,
    actualiza calibracion_edge.json.
    """
    import time

    # Encontrar archivo(s) pendientes
    if archivo:
        targets = [Path(archivo)]
    else:
        targets = sorted(REPORTS_DIR.glob("apuestas_*.json"), reverse=True)
        targets = [t for t in targets
                   if json.loads(t.read_text(encoding="utf-8")).get("estado") == "PENDIENTE"]

    if not targets:
        logger.info("✅ No hay apuestas PENDIENTES para cerrar.")
        return

    logger.info(f"🔍 {len(targets)} archivo(s) pendiente(s) para cerrar\n")

    # Cargar calibracion
    calibracion = json.loads(CALIBRACION.read_text(encoding="utf-8")) if CALIBRACION.exists() else {
        "global": {"wins": 0, "losses": 0},
        "por_superficie": {},
        "por_superficie_y_tier": {},
        "fallback_por_tier": {},
    }

    # Nodo-40 Fase 3: índice de predicciones de games/sets para ground truth
    games_index = _cargar_games_signal_index()
    if games_index:
        logger.info(f"   Nodo-40: games_signal_index cargado — {len([k for k in games_index if len(k) <= 10])} match_ids")

    total_wins = 0
    total_losses = 0
    total_pendientes = 0

    for target in targets:
        apuesta = json.loads(target.read_text(encoding="utf-8"))
        picks = apuesta.get("picks", [])
        alguno_pendiente = False

        logger.info(f"📄 {target.name} — {len(picks)} picks")

        for pick in picks:
            if pick.get("correcto") is not None:
                estado = "✅" if pick["correcto"] else "❌"
                logger.info(f"   {estado} {pick['jugador']} — ya cerrado")
                continue

            match_id = pick.get("match_id", "")
            resultado = _obtener_resultado(match_id)
            status = resultado.get("status")

            if status == "FT":
                # Determinar ganador: jugador1 = home (AE en FlashScore)
                # La API retorna ganador_lado = "jugador1" | "jugador2"
                # pick["partido"] = "Alex Bolt vs Keegan Smith" → jugador1=Alex Bolt
                partido = pick.get("partido", "")
                partes = partido.split(" vs ")
                jugador1 = partes[0].strip() if len(partes) >= 1 else ""
                jugador2 = partes[1].strip() if len(partes) >= 2 else ""

                ganador_lado = resultado["ganador_lado"]
                ganador_real = jugador1 if ganador_lado == "jugador1" else jugador2

                # Verificar si nuestro pick ganó
                pick_lower = pick["jugador"].split(" ")[0].lower()
                correcto = pick_lower in ganador_real.lower()

                pick["resultado_real"] = ganador_real
                pick["correcto"] = correcto
                pick["sets"] = f"{resultado['sets_local']}-{resultado['sets_visit']}"

                # ── Nodo-40 Fase 3: games_ground_truth ──────────────────────
                sets_real = int(resultado.get("sets_local", 0)) + int(resultado.get("sets_visit", 0))
                games_real = _calcular_juegos_totales(resultado.get("_raw", {}))
                games_pred = games_index.get(match_id) or games_index.get(pick.get("partido", ""))
                if games_pred:
                    zona_diff   = games_pred.get("zona_diff", "")
                    sets_pred   = games_pred.get("predicted_sets")
                    games_range = games_pred.get("games_range", "")
                    diff        = games_pred.get("diff_abs")
                    gt = {
                        "fecha":           datetime.now().strftime("%Y-%m-%d"),
                        "partido":         pick.get("partido", ""),
                        "sets_real":       sets_real,
                        "sets_pred":       sets_pred,
                        "games_range_pred": games_range,
                        "games_real":      games_real,
                        "diff":            diff,
                        "zona_diff":       zona_diff,
                        "sets_correcto":   sets_real == sets_pred,
                        "games_en_rango":  _check_games_en_rango(games_real, games_range),
                    }
                    pick["games_ground_truth"] = gt
                    calibracion.setdefault("games_calibracion", []).append(gt)
                    tag = "sets OK" if gt["sets_correcto"] else "sets FALLO"
                    g_tag = f" | games {games_real} {'OK' if gt['games_en_rango'] else 'FALLO'}" if games_real else ""
                    logger.info(f"   N40 {tag} (pred={sets_pred} real={sets_real}){g_tag}")

                stake = pick.get("stake", 0)
                if correcto:
                    pick["ganancia"] = round(stake * (pick["cuota"] - 1))
                else:
                    pick["ganancia"] = -stake

                pl_str = f"  +${pick['ganancia']:,.0f}" if correcto and stake else (f"  -${stake:,.0f}" if stake else "")
                estado = "✅ GANÓ" if correcto else "❌ PERDIÓ"
                logger.info(f"   {estado} {pick['jugador']:25s} → {ganador_real} ({pick['sets']}){pl_str}")

                # Actualizar calibracion
                sup = pick.get("superficie", "unknown")
                tier = pick.get("tier", "unknown")

                if correcto:
                    total_wins += 1
                    calibracion["global"]["wins"] = calibracion["global"].get("wins", 0) + 1
                    calibracion.setdefault("por_superficie", {}).setdefault(sup, {"wins": 0, "losses": 0})
                    calibracion["por_superficie"][sup]["wins"] += 1
                    key_tier = f"{sup}_{tier}"
                    calibracion.setdefault("por_superficie_y_tier", {}).setdefault(key_tier, {"wins": 0, "losses": 0})
                    calibracion["por_superficie_y_tier"][key_tier]["wins"] += 1
                    # FIX-5: acumular era_v2 (datos post-normalización-fix 2026-06-19)
                    calibracion["por_superficie_y_tier"][key_tier].setdefault("era_v2_wins", 0)
                    calibracion["por_superficie_y_tier"][key_tier].setdefault("era_v2_losses", 0)
                    calibracion["por_superficie_y_tier"][key_tier].setdefault("era_v2_start", "2026-06-19")
                    calibracion["por_superficie_y_tier"][key_tier]["era_v2_wins"] += 1
                else:
                    total_losses += 1
                    calibracion["global"]["losses"] = calibracion["global"].get("losses", 0) + 1
                    calibracion.setdefault("por_superficie", {}).setdefault(sup, {"wins": 0, "losses": 0})
                    calibracion["por_superficie"][sup]["losses"] += 1
                    key_tier = f"{sup}_{tier}"
                    calibracion.setdefault("por_superficie_y_tier", {}).setdefault(key_tier, {"wins": 0, "losses": 0})
                    calibracion["por_superficie_y_tier"][key_tier]["losses"] += 1
                    # FIX-5: acumular era_v2 (datos post-normalización-fix 2026-06-19)
                    calibracion["por_superficie_y_tier"][key_tier].setdefault("era_v2_wins", 0)
                    calibracion["por_superficie_y_tier"][key_tier].setdefault("era_v2_losses", 0)
                    calibracion["por_superficie_y_tier"][key_tier].setdefault("era_v2_start", "2026-06-19")
                    calibracion["por_superficie_y_tier"][key_tier]["era_v2_losses"] += 1

            elif status in ("LIVE", "NS"):
                logger.info(f"   ⏳ {pick['jugador']:25s} — partido {status}, reintentar más tarde")
                alguno_pendiente = True
                total_pendientes += 1

            else:
                logger.warning(f"   ⚠️  {pick['jugador']:25s} — {status} (match_id: {match_id or 'vacío'})")
                alguno_pendiente = True
                total_pendientes += 1

            time.sleep(DELAY_REQUESTS)

        # P&L de la sesión
        pl_sesion = sum(p.get("ganancia", 0) or 0 for p in picks if p.get("correcto") is not None)
        invertido = sum(p.get("stake", 0) or 0 for p in picks if p.get("correcto") is not None)
        if invertido:
            signo = "+" if pl_sesion >= 0 else ""
            logger.info(f"   💰 P&L sesión: {signo}${pl_sesion:,.0f}  (invertido ${invertido:,.0f})")

        # Actualizar estado del archivo
        todos_cerrados = all(p.get("correcto") is not None for p in picks)
        apuesta["estado"] = "PENDIENTE" if alguno_pendiente else "CERRADO"
        apuesta["pl_sesion"] = pl_sesion
        apuesta["ts_cierre"] = datetime.now().isoformat() if not alguno_pendiente else None
        target.write_text(json.dumps(apuesta, ensure_ascii=False, indent=2), encoding="utf-8")

        if todos_cerrados:
            logger.info(f"   💾 {target.name} → CERRADO\n")
        else:
            logger.info(f"   ⏳ {target.name} → aún PENDIENTE ({total_pendientes} sin resultado)\n")

    # Guardar calibracion actualizada
    if total_wins + total_losses > 0:
        calibracion["ultima_actualizacion"] = datetime.now().isoformat()
        g = calibracion["global"]
        n = g["wins"] + g["losses"]
        p = g["wins"] / n if n > 0 else 0
        CALIBRACION.write_text(json.dumps(calibracion, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info("=" * 55)
        logger.info(f"📊 SESIÓN: +{total_wins}W / -{total_losses}L")
        logger.info(f"📊 GLOBAL: {g['wins']}W / {g['losses']}L — p={p:.3f} (n={n})")
        if total_pendientes:
            logger.info(f"⏳ {total_pendientes} pick(s) sin resultado — volvé a correr --cerrar más tarde")
        logger.info("=" * 55)
    else:
        if total_pendientes:
            logger.info(f"⏳ {total_pendientes} pick(s) sin resultado — volvé a correr --cerrar más tarde")
        else:
            logger.info("ℹ️  Sin cambios en calibración.")


# ══════════════════════════════════════════════════════════════════════════════
# MODO 3 — Estado: listar apuestas pendientes
# ══════════════════════════════════════════════════════════════════════════════

def estado():
    """Lista todas las apuestas PENDIENTES con sus picks."""
    archivos = sorted(REPORTS_DIR.glob("apuestas_*.json"), reverse=True)
    pendientes = []

    for f in archivos:
        data = json.loads(f.read_text(encoding="utf-8"))
        if data.get("estado") == "PENDIENTE":
            pendientes.append((f, data))

    if not pendientes:
        logger.info("✅ No hay apuestas pendientes.")
        return

    logger.info(f"⏳ {len(pendientes)} apuesta(s) PENDIENTE(S):\n")
    for f, data in pendientes:
        ts = data.get("ts_registro", "")[:16]
        picks = data.get("picks", [])
        logger.info(f"  📄 {f.name} — {ts} — {len(picks)} picks")
        for p in picks:
            cerrado = "✅" if p.get("correcto") is True else ("❌" if p.get("correcto") is False else "⏳")
            logger.info(f"     {cerrado} {p['jugador']:25s} @{p['cuota']:.2f} — {p.get('partido','')}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# MODO 4 — Servidor local: bookmarklet hace POST automático
# ══════════════════════════════════════════════════════════════════════════════

def escuchar(port: int = LISTEN_PORT):
    """
    Levanta un servidor HTTP en localhost:{port}.
    El bookmarklet hace POST /registrar con el JSON del betslip.
    No requiere copiar/pegar nada.

    Uso:
      python3 betslip_registrar.py --listen
    Luego click en el bookmarklet "Leer Betslip" — se registra solo.
    """
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # silenciar logs HTTP crudos

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_OPTIONS(self):
            self.send_response(200)
            self._cors()
            self.end_headers()

        def do_POST(self):
            if self.path != "/registrar":
                self.send_response(404)
                self.end_headers()
                return

            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")

            try:
                registrar(body)
                resp = json.dumps({"ok": True}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._cors()
                self.end_headers()
                self.wfile.write(resp)
            except SystemExit:
                resp = json.dumps({"ok": False, "error": "ver consola"}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self._cors()
                self.end_headers()
                self.wfile.write(resp)

    logger.info(f"🟢 Servidor escuchando en http://localhost:{port}/registrar")
    logger.info("   Click en el bookmarklet 'Leer Betslip' para registrar automáticamente.")
    logger.info("   Ctrl+C para detener.\n")
    HTTPServer(("localhost", port), Handler).serve_forever()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Registra apuestas Betplay y cierra el loop con FlashScore"
    )
    parser.add_argument("json_bookmarklet", nargs="?",
                        help="JSON del bookmarklet (pegado entre comillas simples)")
    parser.add_argument("--cerrar", action="store_true",
                        help="Consulta resultados y cierra apuestas pendientes")
    parser.add_argument("--estado", action="store_true",
                        help="Lista apuestas pendientes de cerrar")
    parser.add_argument("--archivo",
                        help="Archivo apuestas_*.json específico para --cerrar")
    parser.add_argument("--listen", action="store_true",
                        help=f"Servidor local en localhost:{LISTEN_PORT} — bookmarklet hace POST automático")
    parser.add_argument("--port", type=int, default=LISTEN_PORT,
                        help=f"Puerto para --listen (default: {LISTEN_PORT})")
    args = parser.parse_args()

    if args.estado:
        estado()
    elif args.cerrar:
        cerrar(args.archivo)
    elif args.listen:
        escuchar(args.port)
    elif args.json_bookmarklet:
        registrar(args.json_bookmarklet)
    else:
        parser.print_help()
        print("\nEjemplos:")
        print("  python3 betslip_registrar.py --listen          # RECOMENDADO — bookmarklet POST automático")
        print("  python3 betslip_registrar.py --estado")
        print("  python3 betslip_registrar.py --cerrar")
