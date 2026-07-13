#!/usr/bin/env python3
"""
combo_registry.py — Nodo-76: Combo P&L Registry

Registra combos generados por combo_confianza_builder y betplay_combo_builder,
los settlea contra resultados_finales y genera reportes de P&L por tipo.

SOLO importa: json, pathlib, datetime, unicodedata, re, argparse, glob
NO importa ningún módulo del modelo de predicción.

CLI:
  python3 combo_registry.py --settle 2026-07-07   # settlea una fecha
  python3 combo_registry.py --report               # reporte histórico
  python3 combo_registry.py --report --fecha 2026-07-07  # reporte un día
"""

import argparse
import glob
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# I7 Nodo-67: usar normalize canónica de player_registry cuando esté disponible.
# Fallback a implementación local si el stack del modelo no está accesible.
try:
    from core.player_registry import normalize_player_name as _pr_normalize
except Exception:
    _pr_normalize = None

# ─────────────────────────────────────────────────────────────────────────────
# Directorio de datos
# ─────────────────────────────────────────────────────────────────────────────

COMBO_REGISTRY_DIR = Path("reports/combo_registry")
REPORTS_DIR = Path("reports")


# ─────────────────────────────────────────────────────────────────────────────
# Normalización de nombres (para join con resultados_finales)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_name(nombre: str) -> str:
    """Quita acentos, lowercase, colapsa espacios (implementación local stdlib)."""
    nfkd = unicodedata.normalize("NFKD", nombre)
    sin_acentos = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", sin_acentos.lower().strip())


def _canon(nombre: str) -> str:
    """
    I7 Nodo-67: normalización canónica unificada.
    Usa player_registry.normalize_player_name si está disponible (misma clave
    que rankings_data), sino cae al _normalize_name local (NFKD — compatible).
    """
    if _pr_normalize is not None:
        return _pr_normalize(nombre)
    return _normalize_name(nombre)


def _names_match(pick_name: str, result_winner: str) -> bool:
    """
    True si el pick y el ganador se refieren a la misma persona.
    Estrategia: normalizar con _canon() (unificada), luego:
    1. Nombre completo de pick en resultado (o viceversa)
    2. Apellido exacto (última palabra) coincide
    """
    if not pick_name or not result_winner:
        return False
    p = _canon(pick_name)
    r = _canon(result_winner)
    # Coincidencia substring
    if p in r or r in p:
        return True
    # Apellido exacto (última palabra)
    apellido_p = p.split()[-1] if p.split() else ""
    apellido_r = r.split()[-1] if r.split() else ""
    if apellido_p and len(apellido_p) > 3 and apellido_p == apellido_r:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Clase principal
# ─────────────────────────────────────────────────────────────────────────────

class ComboRegistry:
    """Registro de combos con P&L tracking."""

    def __init__(self, registry_dir: Optional[Path] = None):
        self.registry_dir = Path(registry_dir) if registry_dir else COMBO_REGISTRY_DIR
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    # ── I/O ──────────────────────────────────────────────────────────────────

    def _registry_path(self, fecha: str) -> Path:
        return self.registry_dir / f"cr_{fecha}.jsonl"

    def _load_registry(self, fecha: str) -> List[dict]:
        path = self._registry_path(fecha)
        if not path.exists():
            return []
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return records

    def _save_registry(self, fecha: str, records: List[dict]) -> None:
        path = self._registry_path(fecha)
        lines = [json.dumps(r, ensure_ascii=False) for r in records]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _append_record(self, fecha: str, record: dict) -> None:
        path = self._registry_path(fecha)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── Log ──────────────────────────────────────────────────────────────────

    def log_combo(
        self,
        tipo: str,
        subtipo: str,
        bat_name: str,
        piernas: List[str],
        cuotas: List[float],
        stake: float,
        fecha_jornada: Optional[str] = None,
    ) -> str:
        """
        Registra un combo al momento de generar el BAT.

        tipo: "CC" | "AC" | "Combo" | "Mega" | "Safe" | "Games"
        subtipo: "CORE" | "SATELLITE" | "MOONSHOT" | "COBERTURA" |
                 "ANCHOR_1A3B" | "ANCHOR_2A2B" | "ANCHOR_3A2B" |
                 "STANDARD" | "MEGA" | "SAFE" | "GAMES_A" | "GAMES_B" | "GAMES_C"
        piernas: lista de nombres / señales
        cuotas: lista de cuotas por pierna
        stake: stake en COP/USD
        Retorna: cr_id generado
        """
        fecha_jornada = fecha_jornada or datetime.now().strftime("%Y-%m-%d")

        # Construir cr_id
        piernas_slug = "-".join(
            _normalize_name(p).replace(" ", "")[:10] for p in piernas[:3]
        )
        cr_id = f"{fecha_jornada}_{tipo}_{bat_name}_{piernas_slug}"

        # Calcular cuota compuesta
        cuota_compuesta: float = 1.0
        for c in cuotas:
            cuota_compuesta *= c
        cuota_compuesta = round(cuota_compuesta, 4)

        retorno_potencial = round(stake * cuota_compuesta, 2)

        record = {
            "cr_id": cr_id,
            "logged_at": datetime.now().isoformat(timespec="seconds"),
            "fecha_jornada": fecha_jornada,
            "tipo": tipo,
            "subtipo": subtipo,
            "bat_name": bat_name,
            "n_piernas": len(piernas),
            "piernas": list(piernas),
            "cuotas": list(cuotas),
            "cuota_compuesta": cuota_compuesta,
            "stake": stake,
            "retorno_potencial": retorno_potencial,
            "settled": False,
            "resultado": None,
            "piernas_resultado": {},
            "pnl": None,
            "settled_at": None,
        }

        self._append_record(fecha_jornada, record)
        return cr_id

    # ── Settle ────────────────────────────────────────────────────────────────

    def _find_resultados_finales(self, fecha: str) -> Optional[dict]:
        """
        Encuentra el resultados_finales más reciente para una fecha dada.
        Primero busca archivos con la fecha en el nombre, luego el más reciente.
        """
        reports = REPORTS_DIR
        if self.registry_dir != COMBO_REGISTRY_DIR:
            # En tests, buscar desde el directorio padre del registry_dir
            reports = self.registry_dir.parent.parent / "reports"

        # Buscar archivos que contengan la fecha
        fecha_compact = fecha.replace("-", "")
        pattern_dated = list(reports.glob(f"resultados_finales_{fecha_compact}*.json"))
        if pattern_dated:
            return json.loads(sorted(pattern_dated, reverse=True)[0].read_text(encoding="utf-8"))

        # Fallback: el más reciente
        all_files = sorted(reports.glob("resultados_finales_*.json"), reverse=True)
        if all_files:
            return json.loads(all_files[0].read_text(encoding="utf-8"))
        return None

    def _build_winner_set(self, resultados: dict) -> Dict[str, bool]:
        """
        Construye un dict {nombre_normalizado: hit} desde detailed_results.
        hit=True → ese jugador GANÓ.
        """
        winner_set: Dict[str, bool] = {}
        for dr in resultados.get("detailed_results", []):
            actual = dr.get("actual_result", {})
            verification = dr.get("verification", {})
            actual_winner = actual.get("actual_winner", "")
            hit = verification.get("hit", False)
            if actual_winner:
                winner_set[_normalize_name(actual_winner)] = hit
        return winner_set

    def _settle_pierna(
        self, nombre_pierna: str, winner_set: Dict[str, bool]
    ) -> Optional[str]:
        """
        Retorna "GANO", "PERDIO" o None (OPEN) para una pierna.
        Para señales games (contienen "UNDER"/"OVER"/"under"/"over"),
        siempre retorna None (no podemos settlearlas desde resultados_finales).
        """
        norm_pierna = _normalize_name(nombre_pierna)
        # Señales games — no son nombres de jugadores
        if any(kw in norm_pierna for kw in ("under", "over", "games", "sets")):
            return None

        for winner_norm, hit in winner_set.items():
            if _names_match(nombre_pierna, winner_norm):
                # Si hit=True → el modelo predijo este jugador Y ganó → GANO
                # Si hit=False → el modelo predijo este jugador Y perdió → PERDIO
                return "GANO" if hit else "PERDIO"

            # También buscar el perdedor: si el nombre de la pierna NO es el ganador
            # pero aparece en el partido (hit=False → el rival era el favorito)
        return None

    def settle_date(self, fecha: str) -> dict:
        """
        Settlea todos los combos de una fecha contra resultados_finales.
        Retorna {settled, win, loss, open, pnl_total}.
        """
        records = self._load_registry(fecha)
        if not records:
            return {"settled": 0, "win": 0, "loss": 0, "open": 0, "pnl_total": 0.0}

        resultados = self._find_resultados_finales(fecha)
        if not resultados:
            return {"settled": 0, "win": 0, "loss": 0, "open": 0, "pnl_total": 0.0}

        winner_set = self._build_winner_set(resultados)

        stats = {"settled": 0, "win": 0, "loss": 0, "open": 0, "pnl_total": 0.0}

        updated = []
        for rec in records:
            if rec.get("settled"):
                updated.append(rec)
                # Aún contar en stats
                res = rec.get("resultado")
                if res == "WIN":
                    stats["win"] += 1
                    stats["settled"] += 1
                    if rec.get("pnl") is not None:
                        stats["pnl_total"] += rec["pnl"]
                elif res == "LOSS":
                    stats["loss"] += 1
                    stats["settled"] += 1
                    if rec.get("pnl") is not None:
                        stats["pnl_total"] += rec["pnl"]
                else:
                    stats["open"] += 1
                continue

            # Calcular resultado de cada pierna
            piernas_resultado: Dict[str, str] = {}
            for nombre in rec["piernas"]:
                pierna_status = self._settle_pierna(nombre, winner_set)
                if pierna_status is not None:
                    piernas_resultado[nombre] = pierna_status

            # Determinar resultado del combo
            todas_gano = all(
                piernas_resultado.get(n) == "GANO" for n in rec["piernas"]
            )
            alguna_perdio = any(
                piernas_resultado.get(n) == "PERDIO" for n in rec["piernas"]
            )
            alguna_open = any(
                piernas_resultado.get(n) is None
                or piernas_resultado.get(n, "OPEN") == "OPEN"
                for n in rec["piernas"]
                if n not in piernas_resultado
            )
            # Revisamos si alguna pierna no fue encontrada (OPEN)
            piernas_no_encontradas = [
                n for n in rec["piernas"] if n not in piernas_resultado
            ]

            if alguna_perdio:
                resultado = "LOSS"
                pnl = -float(rec["stake"])
                is_settled = True
            elif not piernas_no_encontradas and todas_gano:
                resultado = "WIN"
                pnl = round(float(rec["stake"]) * rec["cuota_compuesta"] - float(rec["stake"]), 2)
                is_settled = True
            else:
                resultado = "OPEN"
                pnl = None
                is_settled = False

            rec["piernas_resultado"] = piernas_resultado
            rec["resultado"] = resultado
            rec["pnl"] = pnl
            rec["settled"] = is_settled
            rec["settled_at"] = datetime.now().isoformat(timespec="seconds") if is_settled else None

            updated.append(rec)

            if is_settled:
                stats["settled"] += 1
                if resultado == "WIN":
                    stats["win"] += 1
                    stats["pnl_total"] += pnl
                elif resultado == "LOSS":
                    stats["loss"] += 1
                    stats["pnl_total"] += pnl
            else:
                stats["open"] += 1

        # Reescribir el archivo completo con los registros actualizados
        self._save_registry(fecha, updated)
        stats["pnl_total"] = round(stats["pnl_total"], 2)
        return stats

    # ── Report ────────────────────────────────────────────────────────────────

    def _load_all_records(self, fecha: Optional[str] = None) -> List[dict]:
        """Carga todos los registros para una fecha o para todas las fechas."""
        if fecha:
            return self._load_registry(fecha)
        all_records = []
        for path in sorted(self.registry_dir.glob("cr_*.jsonl")):
            all_records.extend(self._load_registry(path.stem.replace("cr_", "")))
        return all_records

    def report(self, fecha: Optional[str] = None) -> str:
        """
        Genera reporte de P&L por tipo de combo.
        Si fecha=None: agrega todos los días disponibles.
        """
        records = self._load_all_records(fecha)

        if not records:
            fecha_label = fecha or "Historico"
            return (
                f"COMBO P&L REGISTRY\n"
                f"==================\n"
                f"Fecha: {fecha_label}\n\n"
                f"Sin registros.\n"
            )

        # Agrupar por tipo
        from collections import defaultdict
        agrupado: Dict[str, dict] = defaultdict(lambda: {
            "n": 0, "win": 0, "loss": 0, "open": 0,
            "stake_total": 0.0, "retorno": 0.0, "pnl": 0.0,
        })

        for rec in records:
            tipo = rec.get("tipo", "?")
            g = agrupado[tipo]
            g["n"] += 1
            g["stake_total"] += float(rec.get("stake", 0))
            resultado = rec.get("resultado")
            pnl = rec.get("pnl")
            if resultado == "WIN":
                g["win"] += 1
                g["retorno"] += float(rec.get("retorno_potencial", 0))
                if pnl is not None:
                    g["pnl"] += pnl
            elif resultado == "LOSS":
                g["loss"] += 1
                if pnl is not None:
                    g["pnl"] += pnl
            else:
                g["open"] += 1

        fecha_label = fecha or "Historico"
        lines = [
            "COMBO P&L REGISTRY",
            "==================",
            f"Fecha: {fecha_label}",
            "",
        ]

        # Cabecera de tabla
        header = (
            f"{'TIPO':<10} | {'N':>8} | {'WIN':>5} | {'LOSS':>5} | {'OPEN':>5} | "
            f"{'Stake total':>14} | {'Retorno':>12} | {'P&L':>12} | {'ROI%':>7}"
        )
        sep = "-" * len(header)
        lines.append(header)
        lines.append(sep)

        total_n = total_win = total_loss = total_open = 0
        total_stake = total_retorno = total_pnl = 0.0

        for tipo in sorted(agrupado.keys()):
            g = agrupado[tipo]
            roi = (g["pnl"] / g["stake_total"] * 100) if g["stake_total"] > 0 else 0.0
            pnl_sign = "+" if g["pnl"] >= 0 else ""
            lines.append(
                f"{tipo:<10} | {g['n']:>8} | {g['win']:>5} | {g['loss']:>5} | {g['open']:>5} | "
                f"${g['stake_total']:>13,.0f} | ${g['retorno']:>11,.0f} | "
                f"${pnl_sign}{g['pnl']:>10,.0f} | {roi:>6.1f}%"
            )
            total_n += g["n"]
            total_win += g["win"]
            total_loss += g["loss"]
            total_open += g["open"]
            total_stake += g["stake_total"]
            total_retorno += g["retorno"]
            total_pnl += g["pnl"]

        lines.append(sep)
        total_roi = (total_pnl / total_stake * 100) if total_stake > 0 else 0.0
        pnl_sign = "+" if total_pnl >= 0 else ""
        lines.append(
            f"{'TOTAL':<10} | {total_n:>8} | {total_win:>5} | {total_loss:>5} | {total_open:>5} | "
            f"${total_stake:>13,.0f} | ${total_retorno:>11,.0f} | "
            f"${pnl_sign}{total_pnl:>10,.0f} | {total_roi:>6.1f}%"
        )

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Combo P&L Registry — Nodo-64")
    parser.add_argument("--settle", metavar="FECHA", help="Settlear combos de YYYY-MM-DD")
    parser.add_argument("--report", action="store_true", help="Generar reporte P&L")
    parser.add_argument("--fecha", metavar="FECHA", help="Fecha para filtrar reporte (YYYY-MM-DD)")
    args = parser.parse_args()

    registry = ComboRegistry()

    if args.settle:
        print(f"Settling combos para {args.settle}...")
        stats = registry.settle_date(args.settle)
        print(
            f"  settled={stats['settled']} | win={stats['win']} | "
            f"loss={stats['loss']} | open={stats['open']} | "
            f"pnl_total={stats['pnl_total']:+,.0f}"
        )
        if stats["settled"] == 0 and stats["open"] == 0:
            print("  (Sin combos registrados para esta fecha — OK)")

    if args.report:
        print(registry.report(args.fecha))

    if not args.settle and not args.report:
        parser.print_help()


if __name__ == "__main__":
    main()
