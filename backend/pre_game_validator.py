"""
pre_game_validator.py — Validación local pre-partido
Reemplaza la lógica que FABLE_02 ponía en GitHub Actions.
Corre localmente donde viven los datos (cron/manual).

Detecta antes de apostar:
  - kelly_kl == 0.0  → BLOCK (no desplegar)
  - n_partidos < 8   → WARN  (Nodo-63 Insufficient History Guard)
  - phantom identity → WARN  (ranking=None + n_history>20)
  - picks sin cuota real → WARN

Uso:
  python3 pre_game_validator.py              # valida edge_report más reciente
  python3 pre_game_validator.py --fixture    # crea fixture kelly_kl=0.0 para test
  python3 pre_game_validator.py --path FILE  # valida archivo específico
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
REPORTS_DIR = BASE_DIR / "reports"

# ─── Códigos de resultado ────────────────────────────────────────────────────
PASS  = "PASS"
WARN  = "WARN"
BLOCK = "BLOCK"


def _latest_edge_report() -> Path | None:
    files = sorted(REPORTS_DIR.glob("edge_report_*.json"))
    return files[-1] if files else None


def _validate_pick(pick: dict) -> list[tuple[str, str, str]]:
    """Devuelve lista de (nivel, codigo, mensaje) para un pick."""
    issues = []
    nombre = pick.get("jugador") or pick.get("player") or pick.get("nombre") or "?"

    # BLOCK: kelly_kl == 0.0
    kl = pick.get("kelly_kl", None)
    if kl is not None and float(kl) == 0.0:
        issues.append((BLOCK, "KELLY_ZERO",
                       f"{nombre}: kelly_kl=0.0 — NO DESPLEGAR (REGLA-HF-5)"))

    # WARN: Insufficient History (Nodo-63)
    n = pick.get("n_partidos") or pick.get("n_h2h") or pick.get("historial_n")
    if n is not None and int(n) < 8:
        issues.append((WARN, "INSUFFICIENT_HISTORY",
                       f"{nombre}: n_partidos={n} < 8 — datos incompletos, no inactividad real"))

    # WARN: Phantom Identity signal
    ranking = pick.get("ranking") or pick.get("ranking_actual")
    if ranking is None and n is not None and int(n) > 20:
        fecha_min = pick.get("fecha_mas_antigua") or pick.get("oldest_match")
        issues.append((WARN, "PHANTOM_IDENTITY_SIGNAL",
                       f"{nombre}: ranking=None + n={n} > 20 — posible homónimo veterano"))
        if fecha_min:
            issues.append((WARN, "PHANTOM_IDENTITY_SIGNAL",
                           f"  fecha_más_antigua={fecha_min} — verificar con Playwright"))

    # WARN: sin cuota real
    cuota_real = pick.get("cuota_es_real")
    if cuota_real is False:
        issues.append((WARN, "CUOTA_SIMULADA",
                       f"{nombre}: cuota_es_real=False — Kambi no tiene precio, cuota estimada"))

    # PASS: si no hay problemas
    if not issues:
        kl_str = f"{kl:.4f}" if kl is not None else "?"
        conf = pick.get("confianza") or pick.get("confidence") or "?"
        issues.append((PASS, "OK", f"{nombre}: kelly_kl={kl_str} conf={conf}"))

    return issues


def validate_file(path: Path) -> int:
    """Valida un edge_report. Retorna exit code (0=OK, 1=WARN, 2=BLOCK)."""
    with open(path) as f:
        data = json.load(f)

    # Soporta distintos formatos de edge_report
    picks = []
    if isinstance(data, list):
        picks = data
    elif "picks" in data:
        picks = data["picks"]
    elif "apuestas" in data:
        picks = data["apuestas"]
    elif "edge_picks" in data:
        picks = data["edge_picks"]
    else:
        # Busca listas de picks en el root
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                picks = v
                break

    if not picks:
        print(f"[validator] {path.name}: sin picks encontrados")
        return 0

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*60}")
    print(f"PRE-GAME VALIDATOR  {ts}")
    print(f"Archivo: {path.name}  ({len(picks)} picks)")
    print(f"{'='*60}")

    # ── Guard C63-A: cola Playwright con candidatos pero sin consumidor ────
    _pq = Path(__file__).parent / "data" / "playwright_queue.json"
    if _pq.exists() and _pq.stat().st_size > 2:
        print(f"  [WARN]  PLAYWRIGHT_QUEUE_PENDIENTE: data/playwright_queue.json "
              f"tiene contenido — cola C63-A sin consumidor, revisar manualmente")

    exit_code = 0
    blocks = 0
    warns  = 0

    for pick in picks:
        status = pick.get("status") or pick.get("estado") or ""
        # Solo validar picks candidatos a apuesta
        if status.upper() not in ("APOSTAR", "WATCHLIST", "APOSTAR_EDGE", ""):
            continue

        issues = _validate_pick(pick)
        for nivel, codigo, msg in issues:
            if nivel == BLOCK:
                print(f"  [BLOCK] {codigo}: {msg}")
                blocks += 1
                exit_code = 2
            elif nivel == WARN:
                print(f"  [WARN]  {codigo}: {msg}")
                warns += 1
                if exit_code < 1:
                    exit_code = 1
            else:
                print(f"  [PASS]  {msg}")

    print(f"\nResumen: {blocks} BLOCK | {warns} WARN | picks={len(picks)}")
    if blocks > 0:
        print("ACCION: NO DESPLEGAR picks con BLOCK. Revisar kelly_kl en edge_calculator.")
    elif warns > 0:
        print("ACCION: Revisar WARNs antes de apostar. Sin BLOCK = pipeline listo.")
    else:
        print("ACCION: Pipeline OK. Proceder con PASO 4.")
    print(f"{'='*60}\n")

    return exit_code


def create_fixture() -> Path:
    """Crea un edge_report fixture con kelly_kl=0.0 para test del validador."""
    fixture = {
        "picks": [
            {
                "jugador": "FIXTURE_TestPlayer_A",
                "kelly_kl": 0.0,
                "confianza": 0.61,
                "cuota": 1.85,
                "n_partidos": 15,
                "ranking": 45,
                "status": "APOSTAR",
                "cuota_es_real": True,
            },
            {
                "jugador": "FIXTURE_TestPlayer_B",
                "kelly_kl": 0.045,
                "confianza": 0.58,
                "cuota": 2.10,
                "n_partidos": 3,
                "ranking": None,
                "status": "APOSTAR",
                "cuota_es_real": True,
            },
            {
                "jugador": "FIXTURE_TestPlayer_C",
                "kelly_kl": 0.031,
                "confianza": 0.55,
                "cuota": 1.95,
                "n_partidos": 25,
                "ranking": None,
                "status": "WATCHLIST",
                "cuota_es_real": False,
            },
        ]
    }
    out = REPORTS_DIR / "edge_report_fixture_test.json"
    REPORTS_DIR.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"[validator] Fixture creado: {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Pre-game validator — valida edge_report antes de apostar")
    parser.add_argument("--fixture", action="store_true", help="Crear y validar fixture kelly_kl=0.0 (test)")
    parser.add_argument("--path", type=str, help="Ruta a edge_report específico")
    args = parser.parse_args()

    if args.fixture:
        path = create_fixture()
    elif args.path:
        path = Path(args.path)
        if not path.exists():
            print(f"[validator] ERROR: archivo no encontrado: {path}")
            sys.exit(1)
    else:
        path = _latest_edge_report()
        if path is None:
            print(f"[validator] No hay edge_report en {REPORTS_DIR}/. Correr PASO 3 primero.")
            sys.exit(0)

    code = validate_file(path)
    sys.exit(code)


if __name__ == "__main__":
    main()
