"""REGLA-T53 — Nodo-171 (D171-01/02/03)

Usuario reportó "los combos solo abren la pagina de betplay, no existen ningun
pick" en el flujo GAMES --live --telegram, disparado por D166-01. Investigación
encontró 3 bugs distintos en betplay_combo_builder.py:

D171-01: trader_plans stale (>4h) hacía sys.exit(1) en el bloque --live ANTES
de llegar a la sección --games/--mega/--safe/--sistema, aunque esos modos son
motores independientes que no dependen de trader_plans.

D171-02: outcome_ids reales vienen como int (Kambi), no str — ",".join(ids)
crasheaba con TypeError en 5 sitios.

D171-03 (el más grave): cuando build_games_combos_live() (motor en vivo,
respeta D150/D151, usa linea_actual/cuota_actual/oc_id_actual frescos) no
encontraba candidatos válidos, el código caía silenciosamente a
build_games_combos() (reporte estático games_signal_report_*.json, a veces
de horas atrás) — mandando combos con outcome_id de partidos ya cerrados en
Betplay. Esto probablemente explica el mensaje roto que disparó el reporte
del usuario.

Estos tests invocan main() real con sys.argv mockeado (no re-implementan la
lógica de dispatch) — REGLA-T53.
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import betplay_combo_builder as bcb


def test_171_01_games_no_sale_por_sys_exit_con_trader_plans_stale():
    """D171-01: --games --live no debe sys.exit(1) cuando build_live_combos()
    retorna vacío (trader_plans stale) — debe seguir hasta la sección GAMES."""
    argv = ["betplay_combo_builder.py", "--live", "--games", "--dry-run"]
    with patch.object(sys, "argv", argv), \
         patch.object(bcb, "build_live_combos", return_value=([], {})), \
         patch.object(bcb, "build_games_combos_live", return_value=([], {})) as mock_live_games:
        bcb.main()
        mock_live_games.assert_called_once()


def test_171_02_join_outcome_ids_int_no_crashea():
    """D171-02: outcome_ids con enteros reales (Kambi) no deben crashear el
    ",".join() en _enviar_safe_telegram (mismo patrón en los otros 4 sitios)."""
    safe_links = [{
        "combo_idx": 1, "cuota_combo": 3.2, "retorno": 6400,
        "p_both": 0.30,
        "outcome_ids": [4285326120, 4285229734],  # int reales, no str
        "legs": [
            {"jugador": "Alcaraz C.", "cuota_kambi": 1.4},
            {"jugador": "Sinner J.", "cuota_kambi": 1.5},
        ],
    }]
    metadata = {"total_stake": 2000}

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        status = 200

    with patch("urllib.request.urlopen", return_value=_FakeResp()):
        # No debe lanzar TypeError: sequence item 0: expected str instance, int found
        bcb._enviar_safe_telegram(safe_links, metadata)


def test_171_03_live_games_no_cae_a_reporte_estatico_cuando_live_vacio():
    """D171-03: cuando build_games_combos_live() retorna vacío, main() NO debe
    llamar a build_games_combos() (reporte estático, potencialmente stale)."""
    argv = ["betplay_combo_builder.py", "--live", "--games", "--dry-run"]
    with patch.object(sys, "argv", argv), \
         patch.object(bcb, "build_live_combos", return_value=([], {})), \
         patch.object(bcb, "build_games_combos_live", return_value=([], {})), \
         patch.object(bcb, "build_games_combos") as mock_static_games:
        bcb.main()
        mock_static_games.assert_not_called()


def test_171_04_live_games_usa_combo_live_cuando_hay_candidatos():
    """Control positivo: cuando build_games_combos_live() SÍ retorna candidatos,
    esos son los que se muestran/envían (no se ignoran ni se mezclan con
    build_games_combos estático)."""
    fake_links = [{
        "combo_idx": 1, "cuota_combo": 3.52, "retorno": 7048, "stake": 2000,
        "outcome_ids": [111, 222, 333],
        "legs": [
            {"jugador": "Loge J.", "cuota_kambi": 1.61},
            {"jugador": "Sonego L.", "cuota_kambi": 1.44},
            {"jugador": "Shang J.", "cuota_kambi": 1.52},
        ],
    }]
    fake_meta = {"total_stake": 2000}

    argv = ["betplay_combo_builder.py", "--live", "--games", "--dry-run"]
    with patch.object(sys, "argv", argv), \
         patch.object(bcb, "build_live_combos", return_value=([], {})), \
         patch.object(bcb, "build_games_combos_live", return_value=(fake_links, fake_meta)), \
         patch.object(bcb, "build_games_combos") as mock_static_games, \
         patch.object(bcb, "_mostrar_games_combos") as mock_mostrar:
        bcb.main()
        mock_static_games.assert_not_called()
        mock_mostrar.assert_called_once_with(fake_links, fake_meta)
