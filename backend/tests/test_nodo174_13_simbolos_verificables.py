"""REGLA-T53 — Nodo-174 D174-13: "✅" exige evidencia en check_contradictions.py.

Invoca la función real `_check_simbolos_verificables()` sobre fixtures de
CLAUDE.md + árbol de código temporales (monkeypatch de CLAUDE_MD/BASE_DIR),
nunca hardcodea el resultado esperado a mano.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_contradictions as cc  # noqa: E402


def _setup(tmp_path, claude_md_text, py_files):
    """Crea CLAUDE.md + .py files bajo tmp_path y monkeypatch-ea las
    constantes del módulo real."""
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(claude_md_text, encoding="utf-8")
    for rel_path, contenido in py_files.items():
        f = tmp_path / rel_path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(contenido, encoding="utf-8")
    return claude_md


def test_174_13_campo_faltante_se_reporta(tmp_path, monkeypatch):
    claude_md = _setup(
        tmp_path,
        "| X | campo_totalmente_inventado | ✅ |\n",
        {"modulo.py": "def otra_cosa():\n    pass\n"},
    )
    monkeypatch.setattr(cc, "CLAUDE_MD", claude_md)
    monkeypatch.setattr(cc, "BASE_DIR", tmp_path)

    n_faltantes, faltantes = cc._check_simbolos_verificables()

    assert n_faltantes >= 1
    assert any("campo_totalmente_inventado" in m for m in faltantes)


def test_174_13_campo_existente_no_se_reporta(tmp_path, monkeypatch):
    claude_md = _setup(
        tmp_path,
        "| X | campo_real_existente | ✅ |\n",
        {"modulo.py": "resultado['campo_real_existente'] = 1\n"},
    )
    monkeypatch.setattr(cc, "CLAUDE_MD", claude_md)
    monkeypatch.setattr(cc, "BASE_DIR", tmp_path)

    n_faltantes, faltantes = cc._check_simbolos_verificables()

    assert not any("campo_real_existente" in m for m in faltantes)


def test_174_13_funcion_faltante_se_reporta(tmp_path, monkeypatch):
    claude_md = _setup(
        tmp_path,
        "| X | usa `funcion_inventada_xyz()` | ✅ |\n",
        {"modulo.py": "def otra_funcion():\n    pass\n"},
    )
    monkeypatch.setattr(cc, "CLAUDE_MD", claude_md)
    monkeypatch.setattr(cc, "BASE_DIR", tmp_path)

    n_faltantes, faltantes = cc._check_simbolos_verificables()

    assert n_faltantes >= 1
    assert any("funcion_inventada_xyz" in m and "función" in m for m in faltantes)


def test_174_13_funcion_existente_no_se_reporta(tmp_path, monkeypatch):
    claude_md = _setup(
        tmp_path,
        "| X | usa `funcion_real_definida()` | ✅ |\n",
        {"modulo.py": "def funcion_real_definida():\n    pass\n"},
    )
    monkeypatch.setattr(cc, "CLAUDE_MD", claude_md)
    monkeypatch.setattr(cc, "BASE_DIR", tmp_path)

    n_faltantes, faltantes = cc._check_simbolos_verificables()

    assert not any("funcion_real_definida" in m for m in faltantes)


def test_174_13_fila_sin_check_se_ignora(tmp_path, monkeypatch):
    """Una fila sin ✅ al final no debe escanearse (scope acotado)."""
    claude_md = _setup(
        tmp_path,
        "| X | campo_en_fila_pendiente | ⏳ |\n",
        {"modulo.py": "def otra_cosa():\n    pass\n"},
    )
    monkeypatch.setattr(cc, "CLAUDE_MD", claude_md)
    monkeypatch.setattr(cc, "BASE_DIR", tmp_path)

    n_faltantes, faltantes = cc._check_simbolos_verificables()

    assert not any("campo_en_fila_pendiente" in m for m in faltantes)


def test_174_13_excluye_prefijo_de_nombre_de_archivo_real(tmp_path, monkeypatch):
    """Mención bare de módulo en prosa (ej. 'games_signal sin --file') no debe
    reportarse como campo faltante -- es nombre de archivo truncado, no dato."""
    claude_md = _setup(
        tmp_path,
        "| X | games_signal universo distinto al edge_calculator | ✅ |\n",
        {"games_signal_calculator.py": "x = 1\n", "edge_calculator.py": "y = 2\n"},
    )
    monkeypatch.setattr(cc, "CLAUDE_MD", claude_md)
    monkeypatch.setattr(cc, "BASE_DIR", tmp_path)

    n_faltantes, faltantes = cc._check_simbolos_verificables()

    assert not any("games_signal" in m for m in faltantes)
    assert not any("edge_calculator" in m for m in faltantes)


def test_174_13_excluye_directorio_venv(tmp_path, monkeypatch):
    """Definiciones dentro de venv/ no cuentan como evidencia -- si el único
    'def' real vive en venv/, el símbolo se sigue reportando como faltante."""
    claude_md = _setup(
        tmp_path,
        "| X | usa `solo_en_venv_func()` | ✅ |\n",
        {"venv/lib/paquete.py": "def solo_en_venv_func():\n    pass\n"},
    )
    monkeypatch.setattr(cc, "CLAUDE_MD", claude_md)
    monkeypatch.setattr(cc, "BASE_DIR", tmp_path)

    n_faltantes, faltantes = cc._check_simbolos_verificables()

    assert any("solo_en_venv_func" in m for m in faltantes)


def test_174_13_pass_real_contra_claude_md_del_repo():
    """Regresión de alcance completo: corriendo contra el CLAUDE.md y el
    código real del repo, no debe haber símbolos ✅ sin evidencia (verificado
    manualmente 2026-08-06 tras el fix de exclusión de stems -- ver Nodo-174
    addendum D174-13)."""
    n_faltantes, faltantes = cc._check_simbolos_verificables()

    assert n_faltantes == 0, f"Símbolos sin evidencia: {faltantes}"
