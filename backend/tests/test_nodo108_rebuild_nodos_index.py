"""
tests/test_nodo108_rebuild_nodos_index.py — REGLA-T53: tests invocan función real.

Cubre B108-01: rebuild_nodos_index detecta números de nodo duplicados y falla
ruidosamente (sys.exit(2)). Valida que el índice real no tiene duplicados.
"""
import sys
import json
import tempfile
import textwrap
from pathlib import Path

import pytest

# Importar la función real (REGLA-T53)
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.rebuild_nodos_index import _parse_nodo, build_index


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_nodo_md(tmp_path: Path, filename: str, content: str) -> Path:
    """Crea un archivo Nodo-*.md temporal con el contenido dado."""
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ── Tests _parse_nodo ─────────────────────────────────────────────────────────

def test_parse_nodo_extrae_id(tmp_path):
    """_parse_nodo extrae correctamente el nodo_id desde el nombre de archivo."""
    md = _make_nodo_md(tmp_path, "Nodo-42-Ejemplo.md", """\
        # Nodo-42 — Ejemplo
        **Fecha:** 2026-07-17
    """)
    result = _parse_nodo(md)
    assert result is not None
    assert result["nodo_id"] == "42"


def test_parse_nodo_no_nodo_archivo(tmp_path):
    """_parse_nodo retorna None para archivos sin patrón Nodo-N."""
    md = tmp_path / "README.md"
    md.write_text("# No es un nodo", encoding="utf-8")
    assert _parse_nodo(md) is None


# ── Tests detección de duplicados (B108-01) ───────────────────────────────────

def test_duplicados_causa_exit_2(tmp_path, monkeypatch):
    """build_index debe fallar con sys.exit(2) si hay dos Nodo-*.md con el mismo número."""
    spec_dir = tmp_path / ".spec" / "01_Nodos"
    spec_dir.mkdir(parents=True)

    # Crear dos archivos con el mismo número (100)
    _make_nodo_md(spec_dir, "Nodo-100-Taxonomia.md", """\
        # Nodo-100 — Taxonomia
        **Fecha:** 2026-07-17
    """)
    _make_nodo_md(spec_dir, "Nodo-100-Triple-Convergencia.md", """\
        # Nodo-100 — Triple Convergencia
        **Fecha:** 2026-07-17
    """)

    # Parchear BASE_DIR y SPEC_DIR para apuntar al tmp
    import scripts.rebuild_nodos_index as mod
    monkeypatch.setattr(mod, "SPEC_DIR", spec_dir)
    monkeypatch.setattr(mod, "OUTPUT", tmp_path / "nodos_index.json")
    monkeypatch.setattr(mod, "BASE_DIR", tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        mod.build_index()

    assert exc_info.value.code == 2, "Debe salir con código 2 en duplicados"


def test_sin_duplicados_no_falla(tmp_path, monkeypatch):
    """build_index no debe fallar si todos los números de nodo son únicos."""
    spec_dir = tmp_path / ".spec" / "01_Nodos"
    spec_dir.mkdir(parents=True)

    _make_nodo_md(spec_dir, "Nodo-100-Taxonomia.md", """\
        # Nodo-100 — Taxonomia
        **Fecha:** 2026-07-17
    """)
    _make_nodo_md(spec_dir, "Nodo-100B-Triple-Convergencia.md", """\
        # Nodo-100B — Triple Convergencia
        **Fecha:** 2026-07-17
    """)

    import scripts.rebuild_nodos_index as mod
    monkeypatch.setattr(mod, "SPEC_DIR", spec_dir)
    monkeypatch.setattr(mod, "OUTPUT", tmp_path / "nodos_index.json")
    monkeypatch.setattr(mod, "BASE_DIR", tmp_path)

    # No debe lanzar SystemExit
    result = mod.build_index()
    assert isinstance(result, dict)


# ── Validación del índice real (sin duplicados en producción) ─────────────────

def test_indice_real_sin_duplicados():
    """El nodos_index.json real no debe tener nodo_ids duplicados."""
    index_path = Path(__file__).parent.parent / "nodos_index.json"
    if not index_path.exists():
        pytest.skip("nodos_index.json no encontrado — correr rebuild_nodos_index primero")

    data = json.loads(index_path.read_text(encoding="utf-8"))
    nodos = data.get("nodos", [])
    ids = [n["nodo_id"] for n in nodos]
    duplicados = [nid for nid in set(ids) if ids.count(nid) > 1]
    assert not duplicados, f"nodos_index.json tiene nodo_ids duplicados: {duplicados}"


def test_nodo_100B_existe_en_indice_real():
    """Nodo-100B-Triple-Convergencia-Live.md debe estar en el índice (B108-01 aplicado)."""
    index_path = Path(__file__).parent.parent / "nodos_index.json"
    if not index_path.exists():
        pytest.skip("nodos_index.json no encontrado")

    data = json.loads(index_path.read_text(encoding="utf-8"))
    archivos = [n["nodo_archivo"] for n in data.get("nodos", [])]
    assert "Nodo-100B-Triple-Convergencia-Live.md" in archivos, (
        "Nodo-100B no encontrado en índice — rebuild_nodos_index debe ejecutarse tras el rename"
    )


def test_nodo_100_triple_no_existe_en_indice_real():
    """El archivo colisionante Nodo-100-Triple-Convergencia-Live.md NO debe estar en el índice."""
    index_path = Path(__file__).parent.parent / "nodos_index.json"
    if not index_path.exists():
        pytest.skip("nodos_index.json no encontrado")

    data = json.loads(index_path.read_text(encoding="utf-8"))
    archivos = [n["nodo_archivo"] for n in data.get("nodos", [])]
    assert "Nodo-100-Triple-Convergencia-Live.md" not in archivos, (
        "El archivo colisionante aún aparece en el índice — rename no aplicado"
    )
