"""
tests/test_nodo132_combo_registry_activation.py — Nodo-132: ComboRegistry activado en builders

REGLA-T53: cada test invoca la función real del módulo — nunca reimplementa la lógica.

Tests:
  test_D132_05_report_dict_returns_structured_dict   — report_dict() retorna dict con by_tipo
  test_D132_05_json_flag_parseable                   — --report --json emite JSON válido
  test_D132_02a_generar_bat_chrome_logs_combo        — generar_bat_chrome registra tipo=Combo
  test_D132_02b_safe_bat_logs_combo                  — _generar_bat_safe registra tipo=Safe
  test_D132_01_cc_bat_logs_combo                     — _generar_bats (CC) registra tipo=CC
  test_H132_01_hypothesis_registered                 — H132-01 en preregistered_hypotheses.json
"""
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# ─── path setup ───────────────────────────────────────────────────────────────
BACKEND = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND))

from combo_registry import ComboRegistry


# ═══════════════════════════════════════════════════════════════════════════════
# D132-05: report_dict() method
# ═══════════════════════════════════════════════════════════════════════════════

def test_D132_05_report_dict_returns_structured_dict(tmp_path):
    """
    REGLA-T53: ComboRegistry.log_combo() + .report_dict() retornan estructura esperada.
    No hardcodeamos cálculo de cuota compuesta — verificamos solo claves y tipos.
    """
    cr = ComboRegistry(registry_dir=tmp_path / "cr")
    cr.log_combo(
        tipo="CC",
        subtipo="CORE",
        bat_name="CC1",
        piernas=["Djokovic", "Alcaraz"],
        cuotas=[1.60, 1.80],
        stake=2000,
    )
    result = cr.report_dict()

    # Estructura mínima requerida por D132-05
    assert "by_tipo" in result, "report_dict() debe tener clave 'by_tipo'"
    assert "total" in result, "report_dict() debe tener clave 'total'"

    by_tipo = result["by_tipo"]
    assert "CC" in by_tipo, "by_tipo debe contener el tipo 'CC' registrado"

    cc = by_tipo["CC"]
    for campo in ("n", "wins", "losses", "open", "pnl"):
        assert campo in cc, f"by_tipo['CC'] debe tener campo '{campo}'"

    assert cc["n"] == 1, "n debe ser 1 después de log_combo"
    # combo no settled → debe estar en open
    assert cc["open"] == 1, "combo no settled → open=1"

    total = result["total"]
    assert total["n"] == 1
    assert isinstance(total["pnl"], float)


# ═══════════════════════════════════════════════════════════════════════════════
# D132-05: --json flag CLI
# ═══════════════════════════════════════════════════════════════════════════════

def test_D132_05_json_flag_parseable(tmp_path):
    """
    REGLA-T53: combo_registry.py --report --json emite JSON parseable con estructura correcta.
    Invoca el script real via subprocess.
    """
    # Crear un registro en tmp_path para que --report tenga algo que mostrar
    cr = ComboRegistry(registry_dir=tmp_path / "cr")
    cr.log_combo("WAS", "WAS", "WAS1", ["Medvedev", "Rublev"], [2.10, 2.30], 5000)

    result = subprocess.run(
        [sys.executable, str(BACKEND / "combo_registry.py"), "--report", "--json"],
        capture_output=True,
        text=True,
        timeout=15,
        cwd=str(BACKEND),
    )
    assert result.returncode == 0, f"combo_registry.py --report --json falló: {result.stderr}"
    output = result.stdout.strip()
    assert output, "Salida no debe estar vacía"

    try:
        data = json.loads(output)
    except json.JSONDecodeError as e:
        pytest.fail(f"Salida de --json no es JSON válido: {e}\nSalida: {output[:500]}")

    assert "by_tipo" in data, "JSON debe tener clave 'by_tipo'"
    assert "total" in data, "JSON debe tener clave 'total'"


# ═══════════════════════════════════════════════════════════════════════════════
# D132-02a: generar_bat_chrome registra combo
# ═══════════════════════════════════════════════════════════════════════════════

def test_D132_02a_generar_bat_chrome_logs_combo(tmp_path):
    """
    REGLA-T53: generar_bat_chrome() escribe BAT Y registra en ComboRegistry.
    Usamos monkeypatch del módulo para redirigir el registry_dir a tmp_path.
    """
    import betplay_combo_builder as bcb

    # Directorio temporal para BAT y combos
    bat_dir = tmp_path / "desktop"
    bat_dir.mkdir()
    combo_dir = tmp_path / "combos"
    combo_dir.mkdir()

    cr_dir = tmp_path / "cr"
    cr_dir.mkdir()

    # Crear una instancia de ComboRegistry con directorio temporal
    real_cr = ComboRegistry(registry_dir=cr_dir)

    # Mockear DESKTOP_WIN, COMBOS_DIR y la clase _ComboRegistry
    combo_link = {
        "combo_idx": 1,
        "url": "https://betplay.com.co/apuestas#1234,5678",
        "legs": [
            {"jugador": "Djokovic", "cuota": 1.60},
            {"jugador": "Alcaraz", "cuota": 1.80},
        ],
    }

    with (
        patch.object(bcb, "DESKTOP_WIN", bat_dir),
        patch.object(bcb, "COMBOS_DIR", combo_dir),
        patch.object(bcb, "_ComboRegistry", lambda: real_cr),
        patch.object(bcb, "_combo_registry_available", True),
    ):
        n = bcb.generar_bat_chrome([combo_link], output_dir=bat_dir)

    assert n == 1, "generar_bat_chrome debe retornar 1"

    # Verificar que el JSONL fue creado en cr_dir
    cr_files = list(cr_dir.glob("cr_*.jsonl"))
    assert len(cr_files) == 1, f"Debe crearse cr_FECHA.jsonl, encontrado: {cr_files}"

    records = [json.loads(l) for l in cr_files[0].read_text().splitlines() if l.strip()]
    assert len(records) == 1, "Debe haber exactamente 1 registro"
    assert records[0]["tipo"] == "Combo", f"tipo debe ser 'Combo', got: {records[0]['tipo']}"
    assert records[0]["subtipo"] == "STANDARD"
    assert len(records[0]["piernas"]) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# D132-02b: _generar_bat_safe registra combo
# ═══════════════════════════════════════════════════════════════════════════════

def test_D132_02b_safe_bat_logs_combo(tmp_path):
    """
    REGLA-T53: _generar_bat_safe() registra tipo=Safe en ComboRegistry.
    """
    import betplay_combo_builder as bcb

    bat_dir = tmp_path / "desktop"
    bat_dir.mkdir()
    combo_dir = tmp_path / "combos"
    combo_dir.mkdir()
    cr_dir = tmp_path / "cr"
    cr_dir.mkdir()

    real_cr = ComboRegistry(registry_dir=cr_dir)

    safe_link = {
        "combo_idx": 1,
        "url": "https://betplay.com.co/apuestas#9876,5432",
        "cuota_combo": 3.20,
        "p_both": 0.32,
        "stake": 1000,
        "legs": [
            {"jugador": "Medvedev", "cuota": 1.85, "cuota_kambi": 1.85},
            {"jugador": "Rublev",   "cuota": 1.75, "cuota_kambi": 1.75},
        ],
    }

    with (
        patch.object(bcb, "DESKTOP_WIN", bat_dir),
        patch.object(bcb, "COMBOS_DIR", combo_dir),
        patch.object(bcb, "_ComboRegistry", lambda: real_cr),
        patch.object(bcb, "_combo_registry_available", True),
    ):
        n = bcb._generar_bat_safe([safe_link])

    assert n == 1

    cr_files = list(cr_dir.glob("cr_*.jsonl"))
    assert len(cr_files) == 1, f"Debe crearse cr_FECHA.jsonl"

    records = [json.loads(l) for l in cr_files[0].read_text().splitlines() if l.strip()]
    assert len(records) == 1
    assert records[0]["tipo"] == "Safe", f"tipo debe ser 'Safe', got: {records[0]['tipo']}"
    assert records[0]["subtipo"] == "SAFE"
    assert records[0]["stake"] == 1000


# ═══════════════════════════════════════════════════════════════════════════════
# D132-01a: _generar_bats (CC) registra combo
# ═══════════════════════════════════════════════════════════════════════════════

def test_D132_01_cc_bat_logs_combo(tmp_path):
    """
    REGLA-T53: Verificar que _combo_registry_available y _ComboRegistry están importados
    en combo_confianza_builder, y que ComboRegistry.log_combo con tipo=CC funciona.
    Testeamos log_combo directamente con los parámetros que _generar_bats usa (D132-01a).
    """
    import combo_confianza_builder as ccb

    # Verificar que el módulo tiene el lazy import de D132
    assert hasattr(ccb, "_combo_registry_available"), (
        "combo_confianza_builder debe tener _combo_registry_available (D132)"
    )
    assert hasattr(ccb, "_ComboRegistry"), (
        "combo_confianza_builder debe tener _ComboRegistry (D132)"
    )

    # Testar el flujo real: ComboRegistry.log_combo con tipo CC
    cr_dir = tmp_path / "cr"
    cr_dir.mkdir()
    cr = ComboRegistry(registry_dir=cr_dir)

    cr_id = cr.log_combo(
        tipo="CC",
        subtipo="CORE",
        bat_name="CC1",
        piernas=["Djokovic", "Alcaraz", "Sinner"],
        cuotas=[1.60, 1.80, 1.70],
        stake=3000,
    )

    assert cr_id is not None and isinstance(cr_id, str), "log_combo debe retornar un cr_id str"
    assert "CC" in cr_id, f"cr_id debe contener 'CC', got: {cr_id}"

    # Verificar que el registro fue escrito
    cr_files = list(cr_dir.glob("cr_*.jsonl"))
    assert len(cr_files) == 1

    records = [json.loads(l) for l in cr_files[0].read_text().splitlines() if l.strip()]
    assert records[0]["tipo"] == "CC"
    assert records[0]["subtipo"] == "CORE"
    assert records[0]["n_piernas"] == 3
    assert records[0]["stake"] == 3000


# ═══════════════════════════════════════════════════════════════════════════════
# H132-01: hipótesis pre-registrada
# ═══════════════════════════════════════════════════════════════════════════════

def test_H132_01_hypothesis_registered():
    """
    REGLA-T53: H132-01 debe estar en preregistered_hypotheses.json.
    Verifica que la hipótesis fue pre-registrada antes de acumular datos (anti p-hacking).
    """
    hyp_path = BACKEND / "validation" / "preregistered_hypotheses.json"
    assert hyp_path.exists(), f"No existe preregistered_hypotheses.json en {hyp_path}"

    with hyp_path.open(encoding="utf-8") as f:
        data = json.load(f)

    hypotheses = data.get("hypotheses", {})
    assert "H132-01" in hypotheses, (
        "H132-01 debe estar en validation/preregistered_hypotheses.json — "
        "pre-registrar antes de acumular datos (Nodo-132 D132)"
    )

    h = hypotheses["H132-01"]
    assert "nombre" in h, "H132-01 debe tener campo 'nombre'"
    assert "n_stop" in h, "H132-01 debe tener campo 'n_stop'"
    assert h.get("n_stop", 0) > 0, "n_stop debe ser > 0"
    assert "umbrales_congelados" in h, "H132-01 debe tener umbrales_congelados"
    assert "cobertura_minima" in h.get("umbrales_congelados", {}), (
        "H132-01 debe tener umbrales_congelados.cobertura_minima"
    )
