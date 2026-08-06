"""
tests/test_nodo161_fire_guard.py — REGLA-T53 D161-01

core/fire_guard.py extrae el guard de disparo-único duplicado literalmente
entre el fire de games_live (D133-04) y el de itf_live_games (D150/D157) en
live_desk.py — misma forma de dato (lista de listas, cap diario) y misma
lógica. Estos tests invocan should_fire()/mark_fired() reales, nunca
hardcodean la condición de disparo.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import fire_guard


def test_161_01_should_fire_sin_historial_previo(tmp_path):
    path = tmp_path / "fired.json"
    assert fire_guard.should_fire(path, ["Alice", "Bob"], cap=10) is True


def test_161_02_should_fire_key_ya_registrada_no_dispara(tmp_path):
    path = tmp_path / "fired.json"
    path.write_text(json.dumps([["Alice", "Bob"]]), encoding="utf-8")
    assert fire_guard.should_fire(path, ["Alice", "Bob"], cap=10) is False


def test_161_03_should_fire_key_distinta_si_dispara(tmp_path):
    path = tmp_path / "fired.json"
    path.write_text(json.dumps([["Alice", "Bob"]]), encoding="utf-8")
    assert fire_guard.should_fire(path, ["Carol", "Dave"], cap=10) is True


def test_161_04_cap_alcanzado_bloquea_incluso_key_nueva(tmp_path):
    path = tmp_path / "fired.json"
    historial = [[f"P{i}"] for i in range(10)]
    path.write_text(json.dumps(historial), encoding="utf-8")
    assert fire_guard.should_fire(path, ["Nunca Vista"], cap=10) is False


def test_161_05_mark_fired_persiste_append(tmp_path):
    path = tmp_path / "fired.json"
    fire_guard.mark_fired(path, ["Alice", "Bob"])
    fire_guard.mark_fired(path, ["Carol", "Dave"])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == [["Alice", "Bob"], ["Carol", "Dave"]]


def test_161_06_mark_fired_luego_should_fire_bloquea(tmp_path):
    path = tmp_path / "fired.json"
    key = ["Eve", "Frank"]
    assert fire_guard.should_fire(path, key, cap=10) is True
    fire_guard.mark_fired(path, key)
    assert fire_guard.should_fire(path, key, cap=10) is False


def test_161_07_json_corrupto_se_trata_como_sin_historial(tmp_path):
    path = tmp_path / "fired.json"
    path.write_text("{esto no es json valido", encoding="utf-8")
    assert fire_guard.should_fire(path, ["Alice"], cap=10) is True


def test_161_08_mark_fired_no_lanza_si_directorio_no_existe(tmp_path):
    path = tmp_path / "no_existe" / "fired.json"
    fire_guard.mark_fired(path, ["Alice"])  # best-effort, no debe propagar excepción
