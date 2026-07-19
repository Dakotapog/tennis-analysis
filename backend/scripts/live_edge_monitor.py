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

# Chrome (abridor de .bat)
_CHROME        = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
_REDIRECT_BASE = "https://dakotapog.github.io/tennis-analysis/bp/?ids="

# Anti-flood D116-01 — CERO Desktop, todo a reports/combos_live/
_BASE_DIR           = Path(__file__).parent.parent
MAX_LIVE_FIRES_DIA  = 10   # cap diario; fire #11+ solo log


def _combos_live_dir(fecha: str) -> Path:
    """Destino único D116-01: reports/combos_live/YYYY-MM-DD/."""
    d = _BASE_DIR / "reports" / "combos_live" / fecha
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fired_path(fecha: str) -> Path:
    return _combos_live_dir(fecha) / "_fired.json"


def _load_fired(fecha: str) -> dict:
    """Carga mapa {event_id: {fired_at, hora_inicio}} desde disco."""
    p = _fired_path(fecha)
    if p.exists():
        try:
            return json.load(p.open(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_fired(fecha: str, fired: dict) -> None:
    _fired_path(fecha).write_text(
        json.dumps(fired, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _ttl_cleanup(fecha: str, now_dt: "datetime") -> None:
    """Borra .bat de partidos cuya hora_inicio + 15min ya pasó (D116-01)."""
    fired = _load_fired(fecha)
    combos_dir = _combos_live_dir(fecha)
    for bat in list(combos_dir.glob("LiveCombo_*.bat")):
        event_id = bat.stem.replace("LiveCombo_", "")
        meta = fired.get(event_id, {})
        hora_inicio_str = meta.get("hora_inicio")
        if hora_inicio_str:
            try:
                h, m = map(int, hora_inicio_str.split(":"))
                match_dt = now_dt.replace(hour=h, minute=m, second=0, microsecond=0)
                deadline = match_dt + timedelta(minutes=15)
                if now_dt >= deadline:
                    bat.unlink(missing_ok=True)
                continue  # hora_inicio presente → no usar fallback
            except Exception:
                pass
        # Fallback: borrar si tiene más de 3 horas (solo cuando hora_inicio ausente/inválida)
        age_h = (now_dt.timestamp() - bat.stat().st_mtime) / 3600
        if age_h > 3:
            bat.unlink(missing_ok=True)


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


class KambiLiveClientReal:
    """
    D99-01 RESUELTO: endpoint dedicado liveEvents.json (eventos en curso Betplay).

    Cadena de lookup:
    1. /liveEvents.json?sport=tennis — solo partidos en vivo (Kambi LIVE real)
    2. /listView/tennis.json STARTED — fallback: pre-game offering con filtro state
    KambiLiveClientFallback queda como tercer nivel si este falla completamente.

    D97-15: operativo desde 2026-07-14.
    """
    _BASE    = "https://us.offering-api.kambicdn.com/offering/v2018/betplay"
    _PARAMS  = "lang=es_CO&market=CO&client_id=2&channel_id=1"
    _HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer":    "https://betplay.com.co/",
        "Accept":     "application/json",
    }

    def __init__(self):
        # Cache de datos de score/saque por jugador (jugador_norm → dict)
        self._score_cache: dict = {}

    def get_score_data(self, jugador: str) -> dict:
        """Retorna {server:'FAV'|'OPP'|None, fav_games:int, opp_games:int} del último fetch."""
        return self._score_cache.get(jugador.lower().strip(), {})

    def get_live_odds(self, jugador: str) -> Optional[float]:
        """Retorna cuota live del jugador o None si no encontrado."""
        jugador_norm = jugador.lower().strip()

        # 1. liveEvents.json — eventos Kambi en curso (endpoint dedicado live)
        odds = self._fetch_live_events(jugador_norm)
        if odds:
            return odds

        # 2. listView STARTED — fallback pre-game con filtro state=STARTED
        return self._fetch_listview_started(jugador_norm)

    def _fetch_live_events(self, jugador_norm: str) -> Optional[float]:
        url = f"{self._BASE}/liveEvents.json?{self._PARAMS}&sport=tennis"
        try:
            req = urllib.request.Request(url, headers=self._HEADERS)
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
            # Kambi puede devolver "liveEvents" o "events" según versión
            events = data.get("liveEvents") or data.get("events") or []
            return self._search_events(events, jugador_norm)
        except Exception:
            return None

    def _fetch_listview_started(self, jugador_norm: str) -> Optional[float]:
        url = f"{self._BASE}/listView/tennis.json?{self._PARAMS}"
        try:
            req = urllib.request.Request(url, headers=self._HEADERS)
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
            events = [e for e in (data.get("events") or [])
                      if e.get("event", {}).get("state") == "STARTED"]
            return self._search_events(events, jugador_norm)
        except Exception:
            return None

    def _search_events(self, events: list, jugador_norm: str) -> Optional[float]:
        for ev_wrapper in events:
            ev     = ev_wrapper.get("event", {}) if isinstance(ev_wrapper, dict) else {}
            offers = ev_wrapper.get("betOffers", []) if isinstance(ev_wrapper, dict) else []
            if not offers:
                continue
            home = ev.get("homeName", "").lower()
            away = ev.get("awayName", "").lower()

            # Determinar si nuestro jugador es home o away
            is_home = bool(jugador_norm in home or (home and home in jugador_norm))

            # ── Extraer score y saque de liveData ──────────────────────────────
            live_data = ev.get("liveData") or {}

            # Quién saca: Kambi usa "currentServer" o "server" con valores HOME/AWAY
            server_raw = (live_data.get("currentServer") or
                          live_data.get("server") or "").upper()
            if server_raw in ("HOME", "PLAYER1", "1"):
                server = "FAV" if is_home else "OPP"
            elif server_raw in ("AWAY", "PLAYER2", "2"):
                server = "FAV" if not is_home else "OPP"
            else:
                server = None

            # Score del set actual (Kambi puede usar dict o string)
            score_raw = live_data.get("score") or {}
            if isinstance(score_raw, dict):
                home_g = int(score_raw.get("home") or score_raw.get("homeScore") or 0)
                away_g = int(score_raw.get("away") or score_raw.get("awayScore") or 0)
            else:
                # Intentar parsear string "3:2" o "3-2"
                import re as _re
                m = _re.search(r"(\d+)[:\-](\d+)", str(score_raw))
                home_g, away_g = (int(m.group(1)), int(m.group(2))) if m else (0, 0)

            # Guardar en cache (por jugador_norm)
            self._score_cache[jugador_norm] = {
                "server":    server,
                "fav_games": home_g if is_home else away_g,
                "opp_games": away_g if is_home else home_g,
            }
            # ─────────────────────────────────────────────────────────────────

            for oc in offers[0].get("outcomes", []):
                oc_type = oc.get("type", "")
                nombre  = (home if oc_type == "OT_ONE" else
                           away if oc_type == "OT_TWO" else "").lower()
                if jugador_norm in nombre or nombre in jugador_norm:
                    odds_raw = oc.get("odds")
                    if odds_raw:
                        return round(odds_raw / 1000, 2)
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
    D97-08: picks STRONG/HOT del edge_report.
    D97-08b: también incluye picks ATP500/GS/ATP1000/Challenger — tiers que
             Kambi cubre en vivo (ITF no aparece en STARTED; estos sí).
    Excluye picks con status NO_DATA o phantom_data.
    """
    result = []
    for p in picks:
        if p.get('status') == 'NO_DATA' or p.get('phantom_data'):
            continue
        flag = p.get('confidence_flag', '')
        markov = p.get('markov_favorito', '')
        tier = p.get('tier', '')
        kambi_visible = tier in ('atp500', 'gs', 'atp1000', 'challenger')
        if flag == 'STRONG' or markov == 'HOT' or kambi_visible:
            result.append(p)
    return result


# ─── Leer edge_report del día ────────────────────────────────────────────────

def load_betslip_index(reports_dir: str = 'reports') -> dict:
    """
    Mergea TODOS los betslip_index del día → mapa jugador_norm → betplay_url.
    Usa todos los archivos del día para no perder picks entre runs de combo_confianza
    y betplay_combo_builder (cada uno genera su propio index).
    """
    today = datetime.now().strftime('%Y%m%d')
    files = sorted(glob.glob(os.path.join(reports_dir, f'betslip_index_{today}_*.json')))
    result = {}
    for f in files:
        try:
            with open(f, encoding='utf-8') as fp:
                data = json.load(fp)
            index = data.get('index', {})
            for oid, info in index.items():
                jugador = info.get('jugador', '').lower().strip()
                if jugador and jugador not in result:
                    result[jugador] = f"{_REDIRECT_BASE}{oid}"
        except Exception:
            continue
    return result


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

def _generar_html_bat_live(
    triggers: list[dict], timestamp: str, output_dir: "Optional[Path]" = None
) -> Optional[str]:
    """
    Genera live_combo_TIMESTAMP.html + LiveCombo_TIMESTAMP.bat en output_dir.
    D116-01: output_dir = reports/combos_live/YYYY-MM-DD/ (CERO Desktop).
    Retorna ruta del .bat generado, o None si no hay triggers.
    """
    if not triggers:
        return None

    fecha = datetime.now().strftime("%Y-%m-%d")
    out   = output_dir or _combos_live_dir(fecha)
    out.mkdir(parents=True, exist_ok=True)

    html_name = f"live_combo_{timestamp}.html"
    bat_name  = f"LiveCombo_{timestamp}.bat"
    html_path = out / html_name
    bat_path  = out / bat_name

    lines = [
        "<html><head><meta charset='utf-8'><title>LIVE EDGE</title>",
        "<style>body{font-family:monospace;background:#1a1a2e;color:#e0e0e0;padding:20px;}",
        "h2{color:#dc3545;} .pick{background:#16213e;border-left:4px solid #dc3545;",
        "padding:12px 16px;margin:10px 0;border-radius:4px;}",
        ".btn{display:inline-block;background:#00d4ff;color:#1a1a2e;padding:10px 24px;",
        "font-size:15px;font-weight:bold;border-radius:6px;text-decoration:none;margin-top:8px;}",
        ".btn:hover{background:#00b8d9;} .odds{color:#fd7e14;font-size:18px;font-weight:bold;}",
        "</style></head><body>",
        "<h2>LIVE EDGE DETECTADO</h2>",
    ]
    for t in triggers:
        url = t.get('betplay_url', '')
        btn_html = (f'<a class="btn" href="{url}" target="_blank">APOSTAR EN BETPLAY</a>'
                    if url else '<span style="color:#888">Sin link Betplay</span>')
        lines.append(
            f"<div class='pick'>"
            f"<b>{t['partido']}</b><br>"
            f"Pre: {t['cuota_pre']:.2f} -&gt; Live: {t['cuota_live']:.2f} "
            f"(drift {t['drift_pct']:+.1f}%) | edge_live: {t['edge_live']:+.3f}<br>"
            f"{btn_html}"
            f"</div>"
        )
    lines.append("</body></html>")
    html_path.write_text("\n".join(lines), encoding="utf-8")

    # Windows path: /mnt/c/foo → C:\foo
    html_win = str(html_path).replace("/mnt/c/", "C:\\").replace("/", "\\")
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
        cliente = KambiLiveClientReal()  # D97-15: usa liveEvents.json (D99-01 resuelto)

    ahora = ahora or datetime.now()
    timestamp = ahora.strftime('%Y%m%d_%H%M%S')

    # Break state machine — cargar history del día (D100-06)
    history = load_odds_history(reports_dir)

    betslip_map       = load_betslip_index(reports_dir)
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

        # Fetch cuota live + datos de score/saque
        cuota_live = cliente.get_live_odds(favorito)
        if cuota_live is None or cuota_live <= 0:
            picks_chequeados.append({'partido': pick.get('partido', favorito), 'cuota_live': None})
            continue

        drift      = calc_drift(cuota_pre, cuota_live)
        edge_live  = calc_edge_live(p_modelo, cuota_live)
        es_trig    = es_trigger(drift, edge_live)

        # Score data (server + games) — disponible si KambiLiveClientReal resolvió saque
        score_data = {}
        if hasattr(cliente, 'get_score_data'):
            score_data = cliente.get_score_data(favorito)

        partido_key = pick.get('partido', favorito).replace(' ', '_')
        break_state = detect_break_state(partido_key, drift, cuota_live, history,
                                         score_data=score_data or None)
        # Marcar si ya estaba fired (para single-fire logic)
        _fired_prev = history.get(partido_key, {}).get('fired', False)

        betplay_url = betslip_map.get(favorito.lower().strip(), '')
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
            'break_state':  break_state,
            '_fired_prev':  _fired_prev,
            'betplay_url':  betplay_url,
            'server':       score_data.get('server', '') if score_data else '',
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

    # ── Break confirmados → auto-combo (D100-04) ─────────────────────────────
    breaks_confirmados = [c for c in picks_chequeados
                          if c.get('break_state') == 'BREAK_CONFIRMADO']
    combo_break_output = None
    if breaks_confirmados and stake_permitido and not observe_only:
        combo_break_output = _fire_break_combos(breaks_confirmados, reports_dir)
        # Marcar fired=True en history para los partidos disparados
        for c in breaks_confirmados:
            if not c.get('_fired_prev'):
                pk = c['partido'].replace(' ', '_')
                if pk in history:
                    history[pk]['fired'] = True

    # Persistir history actualizado
    save_odds_history(history, reports_dir)

    # ── Snapshot JSON (D97-05) ────────────────────────────────────────────────
    snapshot = {
        'ts':                 ahora.isoformat(),
        'picks_monitoreados':      len(picks_monitoreados),
        'picks_chequeados':        len(picks_chequeados),
        'picks_chequeados_data':   picks_chequeados,   # lista completa para dashboard
        'triggers':           triggers,
        'n_triggers':         len(triggers),
        'break_confirmados':  len(breaks_confirmados),
        'combo_sugerido':     {'patas': len(triggers)} if triggers else None,
        'combo_break_output': combo_break_output,
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
    """
    Extrae TODAS las señales que justifican el pick — visibles en dashboard/panel.
    Orden: confianza → markov → superficie → score_directo → ELO → RFI → IRP → GCS
    """
    s = []
    # Confianza base
    flag = pick.get('confidence_flag', '')
    if flag == 'STRONG':
        s.append('STRONG')
    elif flag == 'MOD':
        s.append('MOD')

    # Markov (estado de forma)
    markov = pick.get('markov_favorito', '')
    if markov == 'HOT':
        s.append('HOT')
    elif markov == 'COLD':
        s.append('COLD_fav')   # favorito frío — señal débil
    markov_r = pick.get('markov_rival', '')
    if markov_r == 'COLD':
        s.append('rival_COLD')  # rival frío — alpha (WAS/ANCHOR)

    # Superficie especialización
    surf = pick.get('surface_specialization') or {}
    surf_score = surf.get('raw_score') or surf.get('score') or 0.0
    if surf_score >= 0.65:
        s.append(f'SURF{surf_score:.0%}')
    elif surf_score >= 0.55:
        s.append('SURF_ok')

    # Score directo convergencia (Nodo-98)
    sd = pick.get('score_directo') or 0
    if sd >= 4:
        s.append(f'SD{sd}★')
    elif sd >= 3:
        s.append(f'SD{sd}')

    # ELO dominance
    elo_d = pick.get('elo_dominance', '')
    if elo_d in ('STRONG', 'DOMINANT'):
        s.append('ELO_DOM')

    # Edge bruto — formato consistente con signal_audit (EDGE_HIGH/EDGE_MED)
    edge = pick.get('edge') or 0.0
    if edge >= 0.20:
        s.append('EDGE_HIGH')
    elif edge >= 0.10:
        s.append('EDGE_MED')

    # RFI (Return from Inactivity)
    rfi = pick.get('rfi_tier') or 0
    if rfi >= 2:
        s.append(f'RFI_T{rfi}')
    elif rfi == 1:
        s.append('RFI')

    # IRP rival (Individual Return Profile negativo del rival)
    irp = pick.get('irp_rival') or {}
    delta = irp.get('delta_return') or 0.0
    if delta < -0.15:
        s.append('IRP_rival--')
    elif delta < -0.10:
        s.append('IRP_rival-')

    # GCS (Grass Court Specialist)
    if pick.get('gcs_active'):
        s.append('GCS')

    # Contribution% al puntaje (generar_tabla_favoritos2.py → rivalry_analyzer)
    contrib = float(pick.get('contribution') or 0.0)
    if contrib >= 0.70:
        s.append('CONTRIB_HIGH')
    elif contrib >= 0.50:
        s.append('CONTRIB_MED')

    # Señales especiales del analyzer (special_signals + reasoning)
    special_raw = list(pick.get('special_signals') or [])
    for reason in (pick.get('reasoning') or []):
        special_raw.append(str(reason))
    for raw in special_raw:
        ru = raw.upper()
        if 'CAMPEON' in ru and 'SUPERFICIE' in ru and 'CAMPEON_SUPERF' not in s:
            s.append('CAMPEON_SUPERF')
        if 'CAMPEON DE TORNEO' in ru and 'CAMPEON_TORNEO' not in s:
            s.append('CAMPEON_TORNEO')
        if 'AJUSTE DINAMICO' in ru and 'AJUSTE_DIN' not in s:
            s.append('AJUSTE_DIN')

    return s


# ─── Break State Machine (D100-01 → D100-07) ─────────────────────────────────

_BREAK_POSIBLE_DRIFT    = 0.15   # drift >= 15% → BREAK_POSIBLE
_BREAK_CONFIRM_DRIFT    = 0.12   # drift >= 12% en ciclo siguiente → CONFIRMADO
_BREAK_RECOVERY_DRIFT   = 0.10   # drift < 10% → recovery → NORMAL


def _history_path(reports_dir: str) -> str:
    today = datetime.now().strftime('%Y%m%d')
    return os.path.join(reports_dir, f'live_odds_history_{today}.json')


def load_odds_history(reports_dir: str = 'reports') -> dict:
    """Carga el historial de odds del día. Retorna {} si no existe."""
    path = _history_path(reports_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def save_odds_history(history: dict, reports_dir: str = 'reports') -> None:
    """Persiste el historial de odds del día."""
    path = _history_path(reports_dir)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def detect_break_state(partido_key: str, current_drift: float,
                       current_cuota: float, history: dict,
                       score_data: dict = None) -> str:
    """
    Máquina de estado Break por partido (D100-01→D100-07).

    MODO PRECISO (cuando score_data tiene server + games):
      NORMAL → BREAK_POSIBLE   = FAV gana un game en el SAQUE del OPP (quiebre real)
      BREAK_POSIBLE → BREAK_CONFIRMADO = FAV gana el game siguiente en su PROPIO saque
      Cualquier estado → NORMAL si OPP quiebra de vuelta (contra-break)

    MODO FALLBACK (sin score_data de Kambi):
      Usa umbrales de drift (comportamiento original D100-01→D100-07).

    Actualiza history[partido_key] in-place. Retorna nuevo estado.
    """
    now_ts = datetime.now().strftime('%H:%M:%S')
    entry  = history.setdefault(partido_key, {
        'readings': [], 'estado': 'NORMAL', 'fired': False,
        'last_fav_games': None, 'last_opp_games': None, 'last_server': None,
        'break_pending_confirm': False,
    })
    estado_prev = entry.get('estado', 'NORMAL')

    if entry.get('fired'):
        return 'BREAK_CONFIRMADO'

    entry['readings'].append({
        'ts': now_ts, 'cuota': current_cuota, 'drift': round(current_drift, 4)
    })
    entry['readings'] = entry['readings'][-10:]

    # ── MODO PRECISO: quiebre real por score + saque ──────────────────────────
    if score_data and score_data.get('server'):
        fav_g    = score_data.get('fav_games', 0)
        opp_g    = score_data.get('opp_games', 0)
        server   = score_data.get('server')          # 'FAV' o 'OPP'
        last_fav = entry.get('last_fav_games')
        last_opp = entry.get('last_opp_games')
        last_srv = entry.get('last_server')

        if last_fav is not None and last_opp is not None and last_srv is not None:
            fav_won = fav_g > last_fav   # FAV ganó un game
            opp_won = opp_g > last_opp   # OPP ganó un game

            # QUIEBRE: FAV gana un game en el saque del OPP
            if fav_won and last_srv == 'OPP' and estado_prev == 'NORMAL':
                entry['estado'] = 'BREAK_POSIBLE'
                entry['break_pending_confirm'] = True

            # CONFIRMACIÓN: después del quiebre, FAV sostiene su propio saque
            elif (fav_won and last_srv == 'FAV'
                  and entry.get('break_pending_confirm')
                  and estado_prev == 'BREAK_POSIBLE'):
                entry['estado'] = 'BREAK_CONFIRMADO'
                entry['break_pending_confirm'] = False

            # CONTRA-BREAK: OPP quiebra de vuelta → volver a NORMAL
            elif opp_won and last_srv == 'FAV':
                entry['estado'] = 'NORMAL'
                entry['break_pending_confirm'] = False

        entry['last_fav_games'] = fav_g
        entry['last_opp_games'] = opp_g
        entry['last_server']    = server

    # ── MODO FALLBACK: drift-based ────────────────────────────────────────────
    else:
        if estado_prev == 'NORMAL':
            if current_drift >= _BREAK_POSIBLE_DRIFT:
                entry['estado'] = 'BREAK_POSIBLE'
        elif estado_prev == 'BREAK_POSIBLE':
            if current_drift < _BREAK_RECOVERY_DRIFT:
                entry['estado'] = 'NORMAL'
            elif current_drift >= _BREAK_CONFIRM_DRIFT:
                entry['estado'] = 'BREAK_CONFIRMADO'

    return entry['estado']


def _fire_break_combos(triggers_confirmados: list, reports_dir: str = 'reports') -> Optional[str]:
    """
    Dispara betplay_combo_builder.py --live cuando hay breaks confirmados.
    D116-01 anti-flood: de-dup por event_id (_fired.json), cap MAX_LIVE_FIRES_DIA,
    TTL cleanup de .bat expirados, output a reports/combos_live/ (CERO Desktop).
    """
    if not triggers_confirmados:
        return None

    fecha   = datetime.now().strftime("%Y-%m-%d")
    now_dt  = datetime.now()
    out_dir = _combos_live_dir(fecha)

    # TTL cleanup al inicio de cada ciclo
    _ttl_cleanup(fecha, now_dt)

    # Cargar registro de fires del día
    fired_map = _load_fired(fecha)

    # Filtrar: sólo triggers nuevos (no fired hoy Y no fired_prev en sesión)
    nuevos = [
        t for t in triggers_confirmados
        if not t.get('_fired_prev', False) and t.get('partido', '') not in fired_map
    ]
    if not nuevos:
        return None

    # Cap diario
    fires_hoy = len(fired_map)
    if fires_hoy >= MAX_LIVE_FIRES_DIA:
        msg = (f'[live_edge_monitor] CAP ALCANZADO ({fires_hoy}/{MAX_LIVE_FIRES_DIA}) — '
               f'revisar manualmente: {", ".join(t["partido"] for t in nuevos)}')
        print(msg)
        return msg

    partidos = [t['partido'] for t in nuevos]
    print(f'[live_edge_monitor] BREAK CONFIRMADO en: {", ".join(partidos)} — disparando combos')

    try:
        r = subprocess.run(
            [sys.executable, str(_BASE_DIR / 'betplay_combo_builder.py'),
             '--live', '--telegram',
             '--output-dir', str(out_dir)],
            capture_output=True, text=True,
            cwd=str(_BASE_DIR), timeout=120
        )
        output = (r.stdout + r.stderr).strip()
        print(f'[live_edge_monitor] combo_builder rc={r.returncode} | {output[:200]}')

        # Actualizar _fired.json con los nuevos fires
        for t in nuevos:
            fired_map[t['partido']] = {
                'fired_at':    now_dt.isoformat(),
                'hora_inicio': t.get('hora', ''),
                'drift_pct':   t.get('drift_pct', 0),
            }
        _save_fired(fecha, fired_map)

        # D101-05: auto-log al shadow book (D99-02)
        try:
            import sys as _sys
            _sys.path.insert(0, str(_BASE_DIR))
            import shadow_book as _sb
            for t in nuevos:
                _pick_live = {
                    'partido':           t.get('partido', ''),
                    'favorito_predicho': t.get('favorito', ''),
                    'cuota_favorito':    t.get('cuota_pre', 0),
                    'p_modelo':          t.get('p_modelo', 0),
                    'edge':              t.get('edge_live', 0),
                    'break_state':       'BREAK_CONFIRMADO',
                    'drift_pct':         t.get('drift_pct', 0),
                    'pick_type':         'live',
                }
                _sb.log_live_pick(_pick_live, cuota_trigger=t.get('cuota_live', t.get('cuota_pre', 0)))
        except Exception as _e:
            print(f'[live_edge_monitor] shadow_book log_live error: {_e}')

        return output
    except Exception as e:
        print(f'[live_edge_monitor] ERROR _fire_break_combos: {e}')
        return None


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Live Edge Monitor — Nodo-97')
    ap.add_argument('--observe', action='store_true', default=True,
                    help='Modo observación (sin Telegram efectivo)')
    ap.add_argument('--telegram', action='store_true', default=False,
                    help='Activar envío Telegram (post-gate 3/5 sesiones)')
    ap.add_argument('--dashboard', action='store_true', default=False,
                    help='Generar live_dashboard.html después del ciclo (Nodo-100)')
    args = ap.parse_args()
    resultado = run(observe_only=args.observe, telegram=args.telegram)
    if args.dashboard:
        print('[dashboard] SUPERSEDED — usar live_desk :7780 (Nodo-109). Flag --dashboard es no-op.')
