"""
REGLA-T53: Tests Nodo-120 — FS Single-Source Cuotas, exportar_para_edge_calculator.

Verifica D120-01 (ss_fs con cuotas incluido), D120-02 (_cuota_source trazabilidad),
D120-03 (ss_fs sin cuotas excluido).
"""

import json
from pathlib import Path

from scraping.match_ledger import exportar_para_edge_calculator, save_ledger


def _make_ledger(ss_fs_cuota1=1.80, ss_fs_cuota2=2.10):
    """
    Ledger sintético: 1 join + 1 single_source_kambi + 1 single_source_fs.
    ss_fs tiene cuota1=ss_fs_cuota1 (None para simular pick sin cuotas).
    """
    return {
        "fecha": "2026-07-20",
        "joins": [
            {
                "jugador1": "Paula Badosa", "jugador2": "Tamara Zidansek",
                "cuota1": 1.27, "cuota2": 3.50,
                "match_id": "abc123", "match_url": "https://flashscore.com/1",
                "superficie": "clay", "torneo_nombre": "Iasi",
                "torneo_completo": "Iasi (Romania)", "tier": "wta",
                "hora": "14:00", "ranking1": 15, "ranking2": 88,
                "join_method": "AUTO_JOIN", "join_score": 100,
                "sources": ["kambi", "flashscore"],
            }
        ],
        "single_source_kambi": [
            {
                "jugador1": "Carlos Alcaraz", "jugador2": "Novak Djokovic",
                "cuota1": 1.55, "cuota2": 2.40,
                "match_id": "", "match_url": "",
                "superficie": "grass", "torneo_nombre": "Wimbledon",
                "torneo_completo": "Wimbledon (UK)", "tier": "grand_slam",
                "hora": "16:00", "ranking1": 2, "ranking2": 1,
                "join_method": "SINGLE_SOURCE_KAMBI", "sources": ["kambi"],
            }
        ],
        "single_source_fs": [
            {
                "jugador1": "Kyrian Jacquet", "jugador2": "Taro Daniel",
                "cuota1": ss_fs_cuota1, "cuota2": ss_fs_cuota2,
                "match_id": "xyz789", "match_url": "https://flashscore.com/2",
                "superficie": "clay", "torneo_nombre": "Estoril - Qual",
                "torneo_completo": "Estoril (Portugal) - Qualification",
                "tier": "atp_qual_", "hora": "06:00",
                "ranking1": 88, "ranking2": 55,
                "join_method": "SINGLE_SOURCE_FS", "sources": ["flashscore"],
            }
        ],
        "cuarentena": [],
        "stats": {"joins_exitosos": 1, "single_source_kambi": 1, "single_source_fs": 1},
    }


class TestNodo120SsFsCuotas:

    def test_ss_fs_con_cuotas_incluido_en_export(self, tmp_path):
        """
        REGLA-T53 D120-01: ss_fs con cuota1>0 y cuota2>0 debe aparecer en el
        archivo exportado para edge_calculator (total = joins + kambi + ss_fs = 3).
        """
        save_ledger(_make_ledger(ss_fs_cuota1=1.80, ss_fs_cuota2=2.10),
                    "2026-07-20", data_dir=str(tmp_path))

        out_path = exportar_para_edge_calculator("2026-07-20", data_dir=str(tmp_path))

        assert out_path, "Debe retornar path no vacío"
        data = json.loads(Path(out_path).read_text())
        assert len(data) == 3, (
            f"Esperados 3 registros (1 join + 1 kambi + 1 ss_fs), obtenidos {len(data)}"
        )
        jugadores = {(r["jugador1"], r["jugador2"]) for r in data}
        assert ("Kyrian Jacquet", "Taro Daniel") in jugadores, \
            "El pick ss_fs (Jacquet vs Daniel) debe estar en el export"

    def test_ss_fs_sin_cuotas_excluido_del_export(self, tmp_path):
        """
        REGLA-T53 D120-03: ss_fs con cuota1=None debe ser excluido del export
        (solo deben quedar joins + single_source_kambi = 2 registros).
        """
        save_ledger(_make_ledger(ss_fs_cuota1=None, ss_fs_cuota2=None),
                    "2026-07-20", data_dir=str(tmp_path))

        out_path = exportar_para_edge_calculator("2026-07-20", data_dir=str(tmp_path))

        assert out_path
        data = json.loads(Path(out_path).read_text())
        assert len(data) == 2, (
            f"ss_fs sin cuotas debe excluirse → 2 registros, obtenidos {len(data)}"
        )
        jugadores = {(r["jugador1"], r["jugador2"]) for r in data}
        assert ("Kyrian Jacquet", "Taro Daniel") not in jugadores, \
            "ss_fs sin cuotas NO debe aparecer en el export"

    def test_ss_fs_cuota_source_es_flashscore(self, tmp_path):
        """
        REGLA-T53 D120-02: el campo _cuota_source del registro exportado debe ser
        'flashscore' para ss_fs y 'kambi' para joins/single_source_kambi.
        """
        save_ledger(_make_ledger(ss_fs_cuota1=1.80, ss_fs_cuota2=2.10),
                    "2026-07-20", data_dir=str(tmp_path))

        out_path = exportar_para_edge_calculator("2026-07-20", data_dir=str(tmp_path))
        data = json.loads(Path(out_path).read_text())

        by_jugador = {r["jugador1"]: r for r in data}

        # Join debe tener _cuota_source='kambi'
        assert by_jugador["Paula Badosa"]["_cuota_source"] == "kambi", \
            "joins deben tener _cuota_source='kambi'"

        # Single_source_kambi debe tener _cuota_source='kambi'
        assert by_jugador["Carlos Alcaraz"]["_cuota_source"] == "kambi", \
            "single_source_kambi debe tener _cuota_source='kambi'"

        # Single_source_fs debe tener _cuota_source='flashscore'
        assert by_jugador["Kyrian Jacquet"]["_cuota_source"] == "flashscore", \
            "single_source_fs debe tener _cuota_source='flashscore'"
