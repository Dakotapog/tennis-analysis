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

    # ── 1. Confianza builder ────────────────────────────────────────────────
    plans = _latest_combo_plans(fecha)
    stakes_confianza: dict[str, int] = {}
    declared_budget = 0
    if plans:
        stakes_confianza, declared_budget = _parse_combo_plan(plans[-1])
        print(f"\nCombo Confianza Builder ({plans[-1].name}):")
        for capa, stake in sorted(stakes_confianza.items()):
            print(f"  {capa:<22} ${stake:>8,}")
    else:
        print(f"\nCombo Confianza Builder: sin combo_plan para {fecha}")

    # ── 2. Betplay builder ──────────────────────────────────────────────────
    stakes_betplay = _betplay_stakes_today(fecha)
    if stakes_betplay:
        print(f"\nBetplay Builder:")
        for capa, stake in sorted(stakes_betplay.items()):
            print(f"  {capa:<22} ${stake:>8,}")
    else:
        print(f"\nBetplay Builder: sin apuestas con stake registrado para {fecha}")
        print("  (betplay no persiste totals — verificar manualmente si corrió --live)")

    # ── 3. Total cruzado ────────────────────────────────────────────────────
    all_stakes = {**stakes_confianza, **stakes_betplay}
    total = sum(all_stakes.values())
    # Budget unificado: declarado en combo_plan (Fase-N × bankroll) + betplay (4% extra)
    betplay_budget = bankroll * _BETPLAY_MAX_PCT
    budget = (declared_budget or bankroll * _BETPLAY_MAX_PCT) + (
        betplay_budget if stakes_betplay else 0
    )
    pct = total / budget * 100 if budget > 0 else 0

    print(f"\n{'─'*60}")
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

    print(f"{'='*60}\n")

    # Log
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_entry = f"[{ts}] {nivel} total=${total:,} budget=${budget:,.0f} ({pct:.1f}%) fecha={fecha}\n"
    (log_dir / "combo_governor.log").open("a").write(log_entry)

    sys.exit(0 if nivel == "PASS" else (2 if nivel == "BLOCK" else 1))


if __name__ == "__main__":
    main()
