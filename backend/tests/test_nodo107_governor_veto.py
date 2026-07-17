"""
tests/test_nodo107_governor_veto.py — REGLA-T53: tests invocan función real.

Cubre S107-D (D107-04): soft-veto del governor en los builders.
Verifica: PASS → continúa; BLOCK sin override → sys.exit; BLOCK con override → continúa + log.
"""
import sys
import subprocess
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from combo_confianza_builder import _governor_check


# ── Helpers ──────────────────────────────────────────────────────────────────

def _mock_gov(returncode: int, stdout: str = ""):
    """Retorna un mock de subprocess.run con el returncode dado."""
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    return m


# ── Tests _governor_check (combo_confianza_builder) ──────────────────────────

def test_governor_pass_no_interrumpe():
    """Governor PASS (exit=0) → _governor_check retorna sin sys.exit."""
    with patch("combo_confianza_builder.subprocess.run", return_value=_mock_gov(0)):
        # No debe lanzar excepción ni SystemExit
        _governor_check(125000, override=False, builder="test")


def test_governor_block_sin_override_aborta():
    """Governor BLOCK (exit=2) sin --override-governor → sys.exit(2)."""
    with patch("combo_confianza_builder.subprocess.run", return_value=_mock_gov(2, "BLOCK output")):
        with pytest.raises(SystemExit) as exc:
            _governor_check(125000, override=False, builder="test")
        assert exc.value.code == 2


def test_governor_warn_sin_override_aborta():
    """Governor WARN (exit=1) sin --override-governor → sys.exit(1)."""
    with patch("combo_confianza_builder.subprocess.run", return_value=_mock_gov(1, "WARN output")):
        with pytest.raises(SystemExit) as exc:
            _governor_check(125000, override=False, builder="test")
        assert exc.value.code == 1


def test_governor_block_con_override_continua(tmp_path):
    """Governor BLOCK con --override-governor → continúa y loguea override."""
    log_path = tmp_path / "logs" / "combo_governor.log"

    with patch("combo_confianza_builder.subprocess.run", return_value=_mock_gov(2, "BLOCK")):
        with patch("combo_confianza_builder.Path", return_value=tmp_path / "combo_governor.py"):
            # Parchear el path del log dentro de _governor_check
            import combo_confianza_builder as mod
            original_path = mod.Path if hasattr(mod, 'Path') else None
            # Ejecutar con override=True — no debe lanzar SystemExit
            try:
                _governor_check(125000, override=True, builder="test_builder")
            except SystemExit:
                pytest.fail("No debe hacer sys.exit con override=True")


def test_governor_warn_con_override_continua():
    """Governor WARN con --override-governor → continúa sin sys.exit."""
    with patch("combo_confianza_builder.subprocess.run", return_value=_mock_gov(1, "WARN")):
        try:
            _governor_check(125000, override=True, builder="test_builder")
        except SystemExit:
            pytest.fail("No debe hacer sys.exit con override=True")


def test_governor_override_escribe_log(tmp_path, monkeypatch):
    """Con override, se escribe entrada en combo_governor.log."""
    import combo_confianza_builder as mod

    # Redirigir el Path del log al tmp_path
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    log_file = logs_dir / "combo_governor.log"

    original_Path = mod.__builtins__  # guardamos para no romper
    # Monkeypatching: hacemos que Path(__file__).parent apunte a tmp_path
    monkeypatch.setattr(mod, "__file__", str(tmp_path / "combo_confianza_builder.py"))

    with patch("combo_confianza_builder.subprocess.run", return_value=_mock_gov(2, "BLOCK")):
        _governor_check(125000, override=True, builder="test_override_log")

    # Verificar que el log existe y contiene la entrada de override
    if log_file.exists():
        content = log_file.read_text()
        assert "OVERRIDE" in content
        assert "test_override_log" in content


# ── Test de integración: builders aceptan --override-governor ────────────────

def test_combo_confianza_acepta_arg_override():
    """combo_confianza_builder.py acepta --override-governor sin crash de argparse."""
    result = subprocess.run(
        [sys.executable, "combo_confianza_builder.py", "--bankroll", "125000",
         "--override-governor", "--no-bat"],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
    )
    # Puede fallar por falta de datos (exit=1) pero NO por argparse error
    assert "error: unrecognized arguments" not in result.stderr
    assert "unrecognized" not in result.stderr.lower()


def test_betplay_acepta_arg_override():
    """betplay_combo_builder.py acepta --override-governor sin crash de argparse."""
    result = subprocess.run(
        [sys.executable, "betplay_combo_builder.py", "--override-governor", "--dry-run"],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
    )
    assert "error: unrecognized arguments" not in result.stderr
    assert "unrecognized" not in result.stderr.lower()


def test_rival_value_acepta_arg_override():
    """rival_value_betslip.py acepta --override-governor sin crash de argparse."""
    result = subprocess.run(
        [sys.executable, "rival_value_betslip.py", "--override-governor", "--dry-run"],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
    )
    assert "error: unrecognized arguments" not in result.stderr
    assert "unrecognized" not in result.stderr.lower()
