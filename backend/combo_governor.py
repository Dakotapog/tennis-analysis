"""
combo_governor.py — Governor único de presupuesto de combos (C63-B)
Cierra V-26-2a: suma TODAS las capas de combos contra session_budget M-26-2.

Modo REPORTE (READ-ONLY — no cambia stakes ni bloquea builders).
Correr ANTES de emitir cualquier .bat:
  python3 combo_governor.py --bankroll 125000

Capas supervisadas:
  Confianza builder: CORE | Satellite (SAT_*) | MOONSHOT | COBERTURA | ANCHOR_*
  Betplay builder:   mega | safe | games | WAS (detecta si hay apuestas_* del día)

Orden de corte si excede budget (mayor varianza primero):
  1. ANCHOR_3A2B  2. ANCHOR_2A2B  3. ANCHOR_1A3B  4. MOONSHOT  5. SAT_*  6. COB_*  7. CORE

Uso:
  python3 combo_governor.py                   # lee bankroll de trader_plan reciente
  python3 combo_governor.py --bankroll 125000 # bankroll explícito
  python3 combo_governor.py --fecha 2026-07-08 # día específico (default: hoy)
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR    = Path(__file__).parent
REPORTS_DIR = BASE_DIR / "reports"

# M-26-2 session budgets por fuente:
#   combo_confianza: determinado por PHASE_CONFIG[fase]['max_daily_pct'] (2/4/7/12%)
#   betplay: MAX_SESSION_LOSS_PCT = 0.04 (4%)
# El governor usa el budget declarado en el combo_plan como referencia unificada
# (campo "Budget diario: $X" en el RESUMEN). Si betplay suma por encima → WARN/BLOCK.
_BETPLAY_MAX_PCT = 0.04  # fallback si no hay combo_plan

# Orden de corte por varianza (mayor primero)
_CORTE_ORDEN = [
    "ANCHOR_3A2B",
    "ANCHOR_2A2B",
    "ANCHOR_1A3B",
    "MOONSHOT",
    "SAT",
    "COB",
    "CORE",
]

# ─── Parsers ────────────────────────────────────────────────────────────────

def _parse_combo_plan(path: Path) -> tuple[dict[str, int], int]:
    """
    Lee un combo_plan_*.txt y extrae stakes por capa + budget declarado.
    Retorna (dict {nombre_capa: stake_total}, budget_declarado).
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    stakes: dict[str, int] = {}
    declared_budget = 0

    # Extraer budget declarado del encabezado
    m_bud = re.search(r'Budget diario:\s+\$([0-9,]+)', text)
    if m_bud:
        declared_budget = int(m_bud.group(1).replace(",", ""))

    # Detectar contexto de la línea de STAKE
    current_label = "UNKNOWN"
    for line in text.splitlines():
        stripped = line.strip()

        # Detectar etiqueta de sección
        if stripped.startswith("CORE "):
            current_label = "CORE"
        elif stripped.startswith("MOONSHOT "):
            current_label = "MOONSHOT"
        elif stripped.startswith("[SAT_"):
            m = re.search(r'\[SAT_(\d+)\]', stripped)
            current_label = f"SAT_{m.group(1)}" if m else "SAT"
        elif stripped.startswith("[COB_"):
            m = re.search(r'\[COB_(\w+)\]', stripped)
            current_label = f"COB_{m.group(1)}" if m else "COB"
        elif stripped.startswith("[ANCHOR_"):
            m = re.search(r'\[ANCHOR_([A-Z0-9]+)_(\d+)\]', stripped)
            if m:
                current_label = f"ANCHOR_{m.group(1)}_{m.group(2)}"
            else:
                current_label = "ANCHOR"

        # Extraer stake
        m_stake = re.search(r'STAKE:\s+\$([0-9,]+)', stripped)
        if m_stake:
            amount = int(m_stake.group(1).replace(",", ""))
            stakes[current_label] = stakes.get(current_label, 0) + amount

    return stakes, declared_budget


def _latest_combo_plans(fecha: str) -> list[Path]:
    """Devuelve combo_plan_*.txt del día, más reciente al final."""
    plans = sorted(REPORTS_DIR.glob(f"combo_plan_{fecha.replace('-', '')}*.txt"))
    return plans


def _parse_combo_plan_json(path: Path) -> tuple[dict[str, int], int]:
    """
    I3 Nodo-67: Lee un combo_plan_*.json y extrae stakes por capa + budget.
    Fuente estructurada — sin regex frágil.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    stakes: dict[str, int] = {}
    for item in data.get('cobertura', []):
        nombre = item.get('nombre', 'UNKNOWN')
        stake = int(item.get('stake', 0))
        if stake:
            stakes[nombre] = stakes.get(nombre, 0) + stake
    declared_budget = int(data.get('budget', 0))
    return stakes, declared_budget


def _latest_combo_plan_json(fecha: str) -> "Path | None":
    """Retorna el combo_plan_*.json más reciente del día, o None."""
    plans = sorted(REPORTS_DIR.glob(f"combo_plan_{fecha.replace('-', '')}*.json"))
    return plans[-1] if plans else None


def _betplay_stakes_today(fecha: str) -> dict[str, int]:
    """
    Intenta leer stakes de apuestas betplay del día desde apuestas_*.json.
    Betplay no persiste stake totals → este parser es best-effort.
    """
    stakes: dict[str, int] = {}
    fecha_str = fecha.replace("-", "")
    for path in REPORTS_DIR.glob(f"apuestas_{fecha_str}*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # apuestas_*.json tiene picks individuales; stake en cada pick si existe
            for pick in data.get("picks", []):
                stake = pick.get("stake", 0) or 0
                layer = pick.get("tipo", "betplay")
                if stake:
                    stakes[layer] = stakes.get(layer, 0) + int(stake)
        except Exception:
            continue
    return stakes


def _trader_stakes_today(fecha: str) -> dict[str, int]:
    """
    S107-B (D107-02): Lee trader_plan_*.json del día y suma stakes
    individuales[].stake + cobertura[].stake — estrategia #1 EL MOTOR.
    """
    stakes: dict[str, int] = {}
    fecha_str = fecha.replace("-", "")
    for path in sorted(REPORTS_DIR.glob(f"trader_plan_{fecha_str}*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for pick in data.get("individuales", []):
                s = int(pick.get("stake", 0) or 0)
                if s:
                    stakes["MOTOR_individual"] = stakes.get("MOTOR_individual", 0) + s
            for pick in data.get("cobertura", []):
                s = int(pick.get("stake", 0) or 0)
                if s:
                    stakes["MOTOR_cobertura"] = stakes.get("MOTOR_cobertura", 0) + s
        except Exception:
            continue
    return stakes


def _rival_value_stakes_today(fecha: str) -> dict[str, int]:
    """
    S107-B (D107-02): Lee output de rival_value_betslip del día.
    Patrón primario: rival_value_plan_{fecha}*.json.
    Fallback: apuestas_{fecha}*.json con tipo=RIVAL_VALUE — best-effort.
    """
    stakes: dict[str, int] = {}
    fecha_str = fecha.replace("-", "")
    for path in sorted(REPORTS_DIR.glob(f"rival_value_plan_{fecha_str}*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for pick in data.get("picks", data.get("betslips", [])):
                s = int(pick.get("stake", 0) or 0)
                if s:
                    stakes["RIVAL_VALUE"] = stakes.get("RIVAL_VALUE", 0) + s
        except Exception:
            continue
    if not stakes:
        for path in sorted(REPORTS_DIR.glob(f"apuestas_{fecha_str}*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                for pick in data.get("picks", []):
                    if str(pick.get("tipo", "")).upper() == "RIVAL_VALUE":
                        s = int(pick.get("stake", 0) or 0)
                        if s:
                            stakes["RIVAL_VALUE"] = stakes.get("RIVAL_VALUE", 0) + s
            except Exception:
                continue
    return stakes


def exposicion_por_jugador(
    capas: list[dict], bankroll: float = 0
) -> dict[str, int]:
    """
    S107-C (D107-03): Suma stake total por jugador normalizado a través de
    todas las capas del día. Usa core/player_registry.normalize_player_name
    (C2 Nodo-67) — NO otra normalización.

    Retorna dict {nombre_normalizado: stake_total}.
    Si bankroll > 0: añade clave '_warnings' con jugadores > 5% bankroll.
    """
    try:
        from core.player_registry import normalize_player_name
    except ImportError:
        normalize_player_name = lambda x: x.strip().lower()  # noqa: E731 — fallback solo en tests

    totales: dict[str, int] = {}
    for capa in capas:
        jugador = capa.get("jugador") or capa.get("player") or ""
        stake = int(capa.get("stake", 0) or 0)
        if not jugador or not stake:
            continue
        nombre = normalize_player_name(jugador)
        totales[nombre] = totales.get(nombre, 0) + stake

    if bankroll > 0:
        cap = bankroll * 0.05
        warnings = [
            f"{j}: ${s:,} ({s/bankroll*100:.1f}% > 5%)"
            for j, s in totales.items() if s > cap
        ]
        if warnings:
            totales["_warnings"] = warnings  # type: ignore[assignment]

    return totales


def _bankroll_from_plan() -> float:
    """Lee bankroll del trader_plan más reciente (últimas 24h)."""
    from datetime import timedelta
    cutoff = datetime.now() - timedelta(hours=24)
    plans = sorted(
        [p for p in REPORTS_DIR.glob("trader_plan_*.json")
         if p.stat().st_mtime >= cutoff.timestamp()],
        reverse=True,
    )
    for p in plans:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            br = (d.get("metadata", {}) or {}).get("parametros", {}).get("bankroll", 0)
            if br and float(br) > 0:
                return float(br)
        except Exception:
            continue
    return 0.0


# ─── Lógica de corte ────────────────────────────────────────────────────────

def _corte_plan(stakes: dict[str, int], exceso: int) -> list[str]:
    """
    Devuelve lista de capas a recortar para reducir el exceso,
    empezando por mayor varianza (ANCHOR_3A2B primero).
    """
    por_cortar = []
    restante = exceso
    for prefix in _CORTE_ORDEN:
        if restante <= 0:
            break
        matching = {k: v for k, v in stakes.items() if k.startswith(prefix)}
        for capa, stake in sorted(matching.items(), reverse=True):
            if restante <= 0:
                break
            por_cortar.append(f"  Cortar {capa}: ${stake:,} (ahorra ${min(stake, restante):,})")
            restante -= stake
    return por_cortar


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Governor de presupuesto de combos (C63-B / V-26-2a)"
    )
    parser.add_argument("--bankroll", type=float, default=0,
                        help="Bankroll actual (default: lee de trader_plan reciente)")
    parser.add_argument("--fecha", type=str, default=datetime.now().strftime("%Y-%m-%d"),
                        help="Fecha a auditar (default: hoy)")
    args = parser.parse_args()

    fecha = args.fecha
    bankroll = args.bankroll or _bankroll_from_plan()

    if bankroll <= 0:
        print("[governor] ERROR: bankroll desconocido. Usar --bankroll 125000")
        sys.exit(1)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*60}")
    print(f"COMBO GOVERNOR  {ts}  |  Fecha: {fecha}")
    print(f"Bankroll: ${bankroll:,.0f}")
    print(f"{'='*60}")

    # ── 1. Confianza builder — JSON primero (I3), .txt como fallback ───────
    plans     = _latest_combo_plans(fecha)
    json_plan = _latest_combo_plan_json(fecha)
    stakes_confianza: dict[str, int] = {}
    declared_budget = 0

    if json_plan:
        stakes_confianza, declared_budget = _parse_combo_plan_json(json_plan)
        print(f"\nCombo Confianza Builder ({json_plan.name}  [JSON]):")
        for capa, stake in sorted(stakes_confianza.items()):
            print(f"  {capa:<22} ${stake:>8,}")
        # Cross-verify con .txt cuando ambos existen
        if plans:
            stakes_txt, _ = _parse_combo_plan(plans[-1])
            total_json = sum(stakes_confianza.values())
            total_txt  = sum(stakes_txt.values())
            if total_json == total_txt:
                print(f"  [I3] cross-verify OK — JSON=${total_json:,} == TXT=${total_txt:,}")
            else:
                print(f"  [I3] cross-verify WARN — JSON=${total_json:,} != TXT=${total_txt:,} (usando JSON)")
    elif plans:
        stakes_confianza, declared_budget = _parse_combo_plan(plans[-1])
        print(f"\nCombo Confianza Builder ({plans[-1].name}  [TXT fallback]):")
        for capa, stake in sorted(stakes_confianza.items()):
            print(f"  {capa:<22} ${stake:>8,}")
    else:
        print(f"\nCombo Confianza Builder: sin combo_plan para {fecha}")

    # ── 2. Motor + Rival Value (S107-B D107-02: cobertura 12/12) ───────────
    stakes_motor = _trader_stakes_today(fecha)
    if stakes_motor:
        print(f"\nMotor EL MOTOR (#1):")
        for capa, stake in sorted(stakes_motor.items()):
            print(f"  {capa:<22} ${stake:>8,}")
    else:
        print(f"\nMotor EL MOTOR (#1): sin trader_plan para {fecha}")

    stakes_rival = _rival_value_stakes_today(fecha)
    if stakes_rival:
        print(f"\nRival Value (#12):")
        for capa, stake in sorted(stakes_rival.items()):
            print(f"  {capa:<22} ${stake:>8,}")
    else:
        print(f"\nRival Value (#12): sin rival_value_plan para {fecha} (best-effort)")

    # ── 3. Betplay builder ──────────────────────────────────────────────────
    stakes_betplay = _betplay_stakes_today(fecha)
    if stakes_betplay:
        print(f"\nBetplay Builder:")
        for capa, stake in sorted(stakes_betplay.items()):
            print(f"  {capa:<22} ${stake:>8,}")
    else:
        print(f"\nBetplay Builder: sin apuestas con stake registrado para {fecha}")
        print("  (betplay no persiste totals — verificar manualmente si corrió --live)")

    # ── 4. Total cruzado 12/12 ──────────────────────────────────────────────
    # D137-01: MOTOR excluido del gate de combos — tiene su propio Kelly-KL/VaR/CPPI.
    # Se muestra como referencia pero NO cuenta contra el budget de combos (#2-#12).
    all_stakes = {**stakes_confianza, **stakes_rival, **stakes_betplay}
    total = sum(all_stakes.values())
    total_motor = sum(stakes_motor.values())
    # Budget unificado: declarado en combo_plan (Fase-N × bankroll) + betplay (4% extra)
    betplay_budget = bankroll * _BETPLAY_MAX_PCT
    budget = (declared_budget or bankroll * _BETPLAY_MAX_PCT) + (
        betplay_budget if stakes_betplay else 0
    )
    pct = total / budget * 100 if budget > 0 else 0

    print(f"\n{'─'*60}")
    if total_motor:
        motor_pct = total_motor / bankroll * 100
        motor_warn = "  ⚠️ WARN >40%" if motor_pct > 40 else ""
        print(f"MOTOR (ref, Kelly-KL):    ${total_motor:>10,}  ({motor_pct:.1f}% bankroll){motor_warn}")
    budget_label = f"confianza ${declared_budget:,} + betplay ${betplay_budget if stakes_betplay else 0:,.0f}" if declared_budget else f"betplay ${betplay_budget:,.0f}"
    print(f"Budget unificado: ${budget:,.0f}  ({budget_label})")
    print(f"TOTAL COMBINADO:  ${total:>10,}  ({pct:.1f}% del budget)")

    if pct <= 100:
        nivel = "PASS"
        msg = "Dentro del budget. Proceder."
    elif pct <= 120:
        nivel = "WARN"
        msg = f"Excede budget en ${total - budget:,.0f} ({pct - 100:.1f}%). Revisar antes de apostar."
    else:
        nivel = "BLOCK"
        msg = f"Excede budget en ${total - budget:,.0f} ({pct - 100:.1f}%). NO emitir .bat sin cortes."

    print(f"Estado: [{nivel}]  {msg}")

    if nivel in ("WARN", "BLOCK") and all_stakes:
        exceso = int(total - budget)
        cortes = _corte_plan(all_stakes, exceso)
        print(f"\nOrden de corte sugerido (mayor varianza primero):")
        for c in cortes:
            print(c)

    # ── Exposición por jugador (S107-C D107-03: cap 5% bankroll) ───────────
    # Construir lista de picks con jugador+stake desde todas las capas
    _picks_para_exposicion: list[dict] = []
    try:
        fecha_str = fecha.replace("-", "")
        # trader_plan individuales
        for p in sorted(REPORTS_DIR.glob(f"trader_plan_{fecha_str}*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            for pick in d.get("individuales", []):
                jugador = pick.get("favorito") or pick.get("jugador") or ""
                if jugador and pick.get("stake", 0):
                    _picks_para_exposicion.append({"jugador": jugador, "stake": pick["stake"]})
            for pick in d.get("cobertura", []):
                for leg in pick.get("legs", []):
                    jugador = leg.get("jugador") or leg.get("player") or ""
                    stake_leg = int(pick.get("stake", 0) or 0)
                    if jugador and stake_leg:
                        _picks_para_exposicion.append({"jugador": jugador, "stake": stake_leg})
        # combo_plan picks (betslip_index del día)
        for p in sorted(REPORTS_DIR.glob(f"betslip_index_{fecha_str}*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            for oid, pick in d.get("index", {}).items():
                jugador = pick.get("jugador") or pick.get("player") or ""
                stake = int(pick.get("stake", 0) or 0)
                if jugador and stake:
                    _picks_para_exposicion.append({"jugador": jugador, "stake": stake})
    except Exception:
        pass

    if _picks_para_exposicion:
        exposicion = exposicion_por_jugador(_picks_para_exposicion, bankroll)
        warnings_exp = exposicion.pop("_warnings", [])
        if warnings_exp:
            print(f"\n[EXPOSICION POR JUGADOR] WARN — cap 5% superada:")
            for w in warnings_exp:
                print(f"  {w}")
        elif exposicion:
            top = sorted(exposicion.items(), key=lambda x: -x[1])[:5]
            print(f"\n[EXPOSICION POR JUGADOR] OK — top 5:")
            for jugador, stake in top:
                print(f"  {jugador:<28} ${stake:>8,}  ({stake/bankroll*100:.1f}%)")

    print(f"{'='*60}\n")

    # Log
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_entry = f"[{ts}] {nivel} total=${total:,} budget=${budget:,.0f} ({pct:.1f}%) fecha={fecha}\n"
    (log_dir / "combo_governor.log").open("a").write(log_entry)

    sys.exit(0 if nivel == "PASS" else (2 if nivel == "BLOCK" else 1))


if __name__ == "__main__":
    main()
