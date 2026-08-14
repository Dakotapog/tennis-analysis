"""core/fire_ledger.py — Nodo-181 D181-02.

Cierra el gap descrito en Nodo-181 §1.6: `core/fire_guard.py` (Nodo-161) persiste
solo `List[List[str]]` — claves sin timestamp. `certeza_fired_*.json` es el unico
registro con hora ISO, y solo cubre el disparo D147-06. Los disparos de combos
live (D133-04, D150/D157) no quedan unibles con el historial de cuotas.

Este modulo es ADITIVO: no sustituye a `fire_guard`. `should_fire()`/`mark_fired()`
conservan su contrato y su archivo actual sin cambio alguno — el anti-flood no
depende de este ledger. `fire_ledger` es solo trazabilidad para D181-01.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

TIPOS_VALIDOS = {"CERTEZA", "GAMES_LIVE", "ITF_LIVE", "COMBO"}


def registrar_disparo(
    fecha: str,
    clave: str,
    tipo: str,
    cuota_al_disparo: Optional[float] = None,
    contexto: Optional[dict] = None,
) -> None:
    """Append de una linea al ledger unificado `reports/fire_ledger_{fecha}.jsonl`.

    fecha: 'YYYYMMDD'. tipo debe estar en TIPOS_VALIDOS (no se valida en runtime
    para no bloquear un disparo por un typo — ver nota best-effort abajo).

    Best-effort (mismo patron defensivo que `fire_guard.mark_fired`): un fallo
    de escritura NUNCA puede impedir ni retrasar el disparo real. Cualquier
    excepcion se traga en silencio.
    """
    try:
        ctx = contexto or {}
        entrada = {
            "ts_iso": datetime.now().isoformat(),
            "clave": clave,
            "tipo": tipo,
            "cuota": cuota_al_disparo,
            "linea": ctx.get("linea"),
            "games_played": ctx.get("games_played"),
            "contexto": ctx,
        }
        path = REPORTS_DIR / f"fire_ledger_{fecha}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
    except Exception:
        pass
