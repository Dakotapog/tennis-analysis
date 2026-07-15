"""
Nodo-97 — Live Edge Monitor: seguimiento de cuotas en vivo para combos intrapartido.
D97-01→D97-14 | Auditoría: Nodo-99 D99-01→D99-11.

Flujo:
  1. Lee edge_report del día → filtra STRONG o HOT (D97-08)
  2. Para cada pick en ventana [-30min, +45min] (D97-06, D99-05):
     a. Fetch cuota live via KambiLiveClient (adapter, D99-01)
     b. drift = (cuota_pre - cuota_live) / cuota_pre
     c. edge_live = p_modelo - 1/cuota_live
     d. TRIGGER si drift >= 0.15 AND edge_live > 0.05 (D97-02)
  3. Si TRIGGER:
     - KGR < 0 → stake=0, banner OBSERVACION (D97-11)
     - Combo Governor budget check (D97-14)
     - Genera HTML+bat Desktop (D97-12)
     - Telegram alert (D97-03)
  4. Escribe reports/live_edge_YYYYMMDD_HHMMSS.json (D97-05)

D99-01 (BLOCKER): KambiLiveClientFallback usa re-fetch pre-game como proxy.
                  KambiLiveClientReal requiere endpoint DevTools (pendiente).
"""
from __future__ import annotations

import glob
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ─── Constantes ──────────────────────────────────────────────────────────────
DRIFT_MIN         = 0.15   # drift >= 15% para trigger (D97-02)
EDGE_LIVE_MIN     = 0.05   # edge_live > 5% para trigger (D97-02)
VENTANA_PRE_MIN   = 30     # minutos ANTES del inicio (D99-05)
VENTANA_POST_MIN  = 45     # minutos DESPUÉS del inicio (D99-05)

# Telegram (mismo bot que betplay_combo_builder)
_TG_TOKEN = "8684706586:AAHv4zhjQKvxORf6bnbwCxZQPly9OA7unpY"
_TG_CHAT  = "8520949513"
_TG_URL   = f"https://api.telegram.org/bot{_TG_TOKEN}/sendMessage"

# Output Desktop (mismo patrón betplay_combo_builder)
_DESKTOP   = Path("/mnt/c/users/hogar/Desktop")
_COMBOS    = _DESKTOP / "combos"
_CHROME    = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


# ─── KambiLiveClient adapter ─────────────────────────────────────────────────

class KambiLiveClientFallback:
    """
    D99-01 FALLBACK: re-fetch del offering pre-game de Kambi como proxy de cuota live.
    Acepta partidos en estado NOT_STARTED y STARTED para capturar cuotas
    que ya cambiaron desde la apertura pre-game pero antes de que cierren.

    Usar hasta que D97-15 confirme el endpoint real via DevTools.
    """
    _BASE = "https://us.offering-api.kambicdn.com/offering/v2018/betplay"
    _PARAMS = "lang=es_CO&market=CO&client_id=2&channel_id=1&ncid=1&category=tennis"
    _HEADERS = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    def get_live_odds(self, jugador: str) -> Optional[float]:
        """
        Busca la cuota actual del jugador en el offering pre-game.
        Retorna float o None si no encontrado / error.
        """
        url = f"{self._BASE}/listView/tennis.json?{self._PARAMS}"
        try:
            req = urllib.request.Request(url, headers=self._HEADERS)
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
        except Exception:
            return None

        jugador_norm = jugador.lower().strip()
        for ev_wrapper in data.get("events", []):
            ev    = ev_wrapper.get("event", {})
            state = ev.get("state", "")
            if state not in ("NOT_STARTED", "STARTED"):
                continue
            offers = ev_wrapper.get("betOffers", [])
            if not offers:
                continue
            home = ev.get("homeName", "").lower()
            away = ev.get("awayName", "").lower()
            for oc in offers[0].get("outcomes", []):
                oc_type = oc.get("type", "")
                nombre = (home if oc_type == "OT_ONE" else
                          away if oc_type == "OT_TWO" else "").lower()
                if jugador_norm in nombre or nombre in jugador_norm:
                    odds_raw = oc.get("odds")
                    if odds_raw:
                        return round(odds_raw / 1000, 2)  # Kambi: odds en milésimas
        return None


class KambiLiveClientMock:
    """
    Mock para tests — retorna cuota configurable.
    Nunca usar en producción.
    """
    def __init__(self, cuota: Optional[float] = None):
        self._cuota = cuota

    def get_live_odds(self, jugador: str) -> Optional[float]:
        return self._cuota


# ─── Helpers de cálculo ──────────────────────────────────────────────────────

def calc_drift(cuota_pre: float, cuota_live: float) -> float:
    """drift = (cuota_pre - cuota_live) / cuota_pre. Positivo = cuota bajó."""
    if not cuota_pre or cuota_pre <= 0:
        return 0.0
    return round((cuota_pre - cuota_live) / cuota_pre, 4)


def calc_edge_live(p_modelo: float, cuota_live: float) -> float:
    """edge_live = p_modelo - 1/cuota_live. (D97-02)"""
    if not cuota_live or cuota_live <= 0:
        return 0.0
    return round(p_modelo - 1.0 / cuota_live, 4)


def es_trigger(drift: float, edge_live: float) -> bool:
    """TRIGGER si drift >= 0.15 AND edge_live > 0.05. (D97-02)"""
    return drift >= DRIFT_MIN and edge_live > EDGE_LIVE_MIN


def en_ventana(inicio_partido: datetime, ahora: Optional[datetime] = None) -> bool:
    """
    Ventana ASIMÉTRICA [-30min, +45min] del inicio del partido. (D97-06 / D99-05)
    Pick 60min antes → excluido | 20min antes → incluido
    Pick 50min después → excluido | 30min después → incluido
    """
    ahora = ahora or datetime.now()
    minutos = (ahora - inicio_partido).total_seconds() / 60
    return -VENTANA_PRE_MIN <= minutos <= VENTANA_POST_MIN


def get_p_modelo(pick: dict) -> float:
    """Extrae p_modelo del pick con fallback desde edge + cuota_fav."""
    if pick.get('p_modelo'):
        return float(pick['p_modelo'])
    conf = pick.get('confidence')
    if conf is not None:
        return float(conf) / 100.0
    edge     = pick.get('edge', 0.0) or 0.0
    cuota_fav = pick.get('cuota_fav') or pick.get('cuota_favorito') or 0.0
    if cuota_fav > 0:
        return edge + 1.0 / cuota_fav
    return 0.0


# ─── Filtro picks monitoreados ────────────────────────────────────────────────

def filtrar_picks_monitoreados(picks: list[dict]) -> list[dict]:
    """
    D97-08: solo picks STRONG o HOT del edge_report.
    Excluye picks con status NO_DATA o phantom_data.
    """
    result = []
    for p in picks:
        if p.get('status') == 'NO_DATA' or p.get('phantom_data'):
            continue
        flag = p.get('confidence_flag', '')
        markov = p.get('markov_favorito', '')
        if flag == 'STRONG' or markov == 'HOT':
            result.append(p)
    return result


# ─── Leer edge_report del día ────────────────────────────────────────────────

def load_picks_del_dia(reports_dir: str = 'reports') -> list[dict]:
    """Lee el edge_report más reciente y retorna picks STRONG/HOT."""
    pattern = os.path.join(reports_dir, 'edge_report_*.json')
    files = sorted(glob.glob(pattern), reverse=True)
    if not files:
        return []
    with open(files[0], encoding='utf-8') as f:
        report = json.load(f)
    todos = (
        list(report.get('apostar', []))
        + list(report.get('watchlist', []))
    )
    return filtrar_picks_monitoreados(todos)


# ─── Output HTML + bat Desktop (D97-12) ──────────────────────────────────────

def _generar_html_bat_live(triggers: list[dict], timestamp: str) -> Optional[str]:
    """
    Genera live_combo_TIMESTAMP.html en Desktop/combos y LiveCombo_TIMESTAMP.bat.
    Reutiliza el patrón de betplay_combo_builder (D97-12 / D99-11).
    Retorna ruta del .bat generado, o None si no hay triggers.
    """
    if not triggers:
        return None

    _COMBOS.mkdir(parents=True, exist_ok=True)
    html_name = f"live_combo_{timestamp}.html"
    bat_name  = f"LiveCombo_{timestamp}.bat"
    html_path = _COMBOS / html_name
    bat_path  = _DESKTOP / bat_name

    lines = ["<html><head><meta charset='utf-8'><title>LIVE EDGE</title></head><body>"]
    lines.append("<h2>LIVE EDGE DETECTADO</h2><ul>")
    for t in triggers:
        lines.append(
            f"<li><b>{t['partido']}</b> | "
            f"Pre: {t['cuota_pre']:.2f} → Live: {t['cuota_live']:.2f} "
            f"(drift {t['drift_pct']:+.1f}%) | "
            f"edge_live: {t['edge_live']:+.3f}</li>"
        )
    lines.append("</ul></body></html>")
    html_path.write_text("\n".join(lines), encoding="utf-8")

    html_win = f"C:\\users\\hogar\\Desktop\\combos\\{html_name}"
    bat_content = (
        f"@echo off\r\n"
        f'start "" "{_CHROME}" "file:///{html_win}"\r\n'
    )
    bat_path.write_text(bat_content, encoding="utf-8")
    return str(bat_path)


# ─── Telegram alert (D97-03) ─────────────────────────────────────────────────

def _enviar_telegram_live(triggers: list[dict], kgr_negativo: bool = False) -> bool:
    """Envía alerta Telegram. Con banner KGR<0 si corresponde. (D97-11)"""
    if not triggers:
        return False

    banner = "\n[KGR < 0 — OBSERVACION, no ejecutar]" if kgr_negativo else ""
    header = f"LIVE EDGE DETECTADO{banner}\n"
    partes = []
    for t in triggers:
        partes.append(
            f"{t['partido']}\n"
            f"Pre: {t['cuota_pre']:.2f} -> Live: {t['cuota_live']:.2f} "
            f"(drift {t['drift_pct']:+.1f}%)\n"
            f"edge_live: {t['edge_live']:+.3f} | Senales: {', '.join(t['senales'])}"
        )
    msg = header + "\n---\n".join(partes)

    try:
        params = urllib.parse.urlencode({
            "chat_id": _TG_CHAT,
            "text":    msg,
        }).encode("utf-8")
        req = urllib.request.Request(_TG_URL, data=params, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception:
        return False


# ─── Loop principal ───────────────────────────────────────────────────────────

def run(
    reports_dir:      str  = 'reports',
    cliente:          object = None,
    ahora:            Optional[datetime] = None,
    kgr_sesion:       Optional[float] = None,
    budget_restante:  Optional[float] = None,
    observe_only:     bool  = True,   # Gate: primeras 5 sesiones en observación
    telegram:         bool  = False,  # Gate: activar después de ≥3/5 sesiones con trigger
) -> dict:
    """
    Corre un ciclo de chequeo live. Retorna el snapshot del ciclo.

    Args:
        cliente:         KambiLiveClient a usar (KambiLiveClientFallback por defecto)
        ahora:           datetime de referencia (override para tests)
        kgr_sesion:      KGR del día (None = no disponible)
        budget_restante: Combo Governor budget restante (D97-14)
        observe_only:    True = log sin Telegram (primeras 5 sesiones)
        telegram:        True = enviar Telegram cuando TRIGGER
    """
    if cliente is None:
        cliente = KambiLiveClientFallback()

    ahora = ahora or datetime.now()
    timestamp = ahora.strftime('%Y%m%d_%H%M%S')

    picks_monitoreados = load_picks_del_dia(reports_dir)
    triggers = []
    picks_chequeados = []

    for pick in picks_monitoreados:
        favorito = pick.get('favorito_predicho', '')
        cuota_pre = pick.get('cuota_fav') or pick.get('cuota_favorito') or 0.0
        p_modelo  = get_p_modelo(pick)

        # Ventana horaria [-30min, +45min] (D97-06 / D99-05)
        inicio_raw = pick.get('inicio_partido') or pick.get('hora_inicio')
        if inicio_raw:
            try:
                if isinstance(inicio_raw, str):
                    inicio_dt = datetime.fromisoformat(inicio_raw)
                else:
                    inicio_dt = datetime.fromtimestamp(inicio_raw)
                if not en_ventana(inicio_dt, ahora):
                    continue
            except (ValueError, TypeError, OSError):
                pass  # sin fecha → no filtrar por ventana

        # Fetch cuota live
        cuota_live = cliente.get_live_odds(favorito)
        if cuota_live is None or cuota_live <= 0:
            picks_chequeados.append({'partido': pick.get('partido', favorito), 'cuota_live': None})
            continue

        drift      = calc_drift(cuota_pre, cuota_live)
        edge_live  = calc_edge_live(p_modelo, cuota_live)
        es_trig    = es_trigger(drift, edge_live)

        check_info = {
            'partido':      pick.get('partido', favorito),
            'favorito':     favorito,
            'cuota_pre':    cuota_pre,
            'cuota_live':   cuota_live,
            'drift_pct':    round(drift * 100, 2),
            'p_modelo':     round(p_modelo, 4),
            'edge_live':    edge_live,
            'trigger':      es_trig,
            'senales':      _senales_activas(pick),
        }
        picks_chequeados.append(check_info)

        if es_trig:
            triggers.append(check_info)

    # ── KGR y Combo Governor (D97-11 / D97-14) ───────────────────────────────
    kgr_negativo    = (kgr_sesion is not None and kgr_sesion < 0)
    budget_agotado  = (budget_restante is not None and budget_restante <= 0)
    stake_permitido = not kgr_negativo and not budget_agotado

    # ── Output HTML + bat + Telegram (D97-12) ─────────────────────────────────
    bat_path = None
    alerta_enviada = False
    if triggers:
        bat_path = _generar_html_bat_live(triggers, timestamp)
        if telegram and not observe_only and stake_permitido:
            alerta_enviada = _enviar_telegram_live(triggers, kgr_negativo=False)
        elif telegram and kgr_negativo:
            alerta_enviada = _enviar_telegram_live(triggers, kgr_negativo=True)

    # ── Snapshot JSON (D97-05) ────────────────────────────────────────────────
    snapshot = {
        'ts':                 ahora.isoformat(),
        'picks_monitoreados': len(picks_monitoreados),
        'picks_chequeados':   len(picks_chequeados),
        'triggers':           triggers,
        'n_triggers':         len(triggers),
        'combo_sugerido':     {'patas': len(triggers)} if triggers else None,
        'kgr_negativo':       kgr_negativo,
        'budget_agotado':     budget_agotado,
        'stake_permitido':    stake_permitido,
        'observe_only':       observe_only,
        'alerta_enviada':     alerta_enviada,
        'bat_path':           bat_path,
    }

    out_path = os.path.join(reports_dir, f'live_edge_{timestamp}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    print(f'[live_edge_monitor] ts={ahora.strftime("%H:%M:%S")} | '
          f'monitoreados={len(picks_monitoreados)} | triggers={len(triggers)}')
    return snapshot


def _senales_activas(pick: dict) -> list[str]:
    """Extrae etiquetas de señales activas del pick para el alert."""
    s = []
    if pick.get('confidence_flag') == 'STRONG':
        s.append('STRONG')
    if pick.get('markov_favorito') == 'HOT':
        s.append('HOT')
    if (pick.get('rfi_tier') or 0) >= 1:
        s.append(f"RFI_tier{pick.get('rfi_tier')}")
    irp = pick.get('irp_rival') or {}
    if (irp.get('delta_return') or 0.0) < -0.10:
        s.append('IRP_negativo')
    return s


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Live Edge Monitor — Nodo-97')
    ap.add_argument('--observe', action='store_true', default=True,
                    help='Modo observación (sin Telegram efectivo)')
    ap.add_argument('--telegram', action='store_true', default=False,
                    help='Activar envío Telegram (post-gate 3/5 sesiones)')
    args = ap.parse_args()
    run(observe_only=args.observe, telegram=args.telegram)
