"""
Tests Nodo-121 — OddsAggregator Cuota Enrichment para ss_fs
REGLA-T53: invocan función real del módulo, nunca hardcodean lógica.
"""
import json
import pytest
from unittest.mock import patch
from scraping.match_ledger import (
    enriquecer_ss_fs_con_aggregator,
    save_ledger,
    load_ledger,
)


def _make_ledger(ss_fs_list, fecha="2099-01-01"):
    return {
        "fecha": fecha,
        "joins": [],
        "cuarentena": [],
        "single_source_kambi": [],
        "single_source_fs": ss_fs_list,
        "stats": {},
    }


MOCK_FEEDS = {
    # Apellido corto (clave alias)
    "aksu": {
        "betplay": {"odds": 3.0, "jugador": "Ayla Aksu", "rival": "Sofia Costoulas",
                    "event_id": 1111, "outcome_id": 2222, "bookmaker": "betplay"},
        "rushbet": {"odds": 2.95, "jugador": "Ayla Aksu", "rival": "Sofia Costoulas",
                    "event_id": 1111, "outcome_id": 2222, "bookmaker": "rushbet"},
    },
    "ayla aksu": {
        "betplay": {"odds": 3.0, "jugador": "Ayla Aksu", "rival": "Sofia Costoulas",
                    "event_id": 1111, "outcome_id": 2222, "bookmaker": "betplay"},
    },
    "costoulas": {
        "betplay": {"odds": 1.38, "jugador": "Sofia Costoulas", "rival": "Ayla Aksu",
                    "event_id": 1111, "outcome_id": 3333, "bookmaker": "betplay"},
        "rushbet": {"odds": 1.37, "jugador": "Sofia Costoulas", "rival": "Ayla Aksu",
                    "event_id": 1111, "outcome_id": 3333, "bookmaker": "rushbet"},
    },
    "sofia costoulas": {
        "betplay": {"odds": 1.38, "jugador": "Sofia Costoulas", "rival": "Ayla Aksu",
                    "event_id": 1111, "outcome_id": 3333, "bookmaker": "betplay"},
    },
}


class TestNodo121AggregatorEnrichment:

    def test_enriquece_ss_fs_sin_cuota(self, tmp_path):
        """D121-01: ss_fs con cuota1=None + match en aggregator → cuota poblada."""
        fecha = "2099-01-01"
        ledger = _make_ledger([{
            "jugador1": "Aksu A.",
            "jugador2": "Costoulas S.",
            "cuota1": None,
            "cuota2": None,
            "join_method": "SINGLE_SOURCE_FS",
            "torneo_nombre": "WTA Qualifying",
        }], fecha)
        save_ledger(ledger, fecha, str(tmp_path))

        with patch("scripts.odds_aggregator.fetch_all_odds", return_value=MOCK_FEEDS):
            import scraping.match_ledger as ml
            stats = ml.enriquecer_ss_fs_con_aggregator(fecha, data_dir=str(tmp_path))

        ledger_actualizado = load_ledger(fecha, str(tmp_path))
        ss = ledger_actualizado["single_source_fs"]

        assert stats["enriquecidos"] == 1
        assert stats["sin_match"] == 0
        assert ss[0]["cuota1"] == 3.0
        assert ss[0]["cuota2"] == 1.38
        assert ss[0]["_cuota_source"] in ("betplay", "rushbet")
        assert ss[0]["_enriched_by"] == "D121-01"

    def test_ss_fs_con_cuota_no_se_toca(self, tmp_path):
        """D121-01 límite: ss_fs que ya tiene cuota1 (Nodo-120) no debe modificarse."""
        fecha = "2099-01-02"
        ledger = _make_ledger([{
            "jugador1": "Aksu A.",
            "jugador2": "Costoulas S.",
            "cuota1": 1.8,   # YA tiene cuota de FlashScore (Nodo-120)
            "cuota2": 2.1,
            "join_method": "SINGLE_SOURCE_FS",
            "_cuota_source": "flashscore",
        }], fecha)
        save_ledger(ledger, fecha, str(tmp_path))

        with patch("scripts.odds_aggregator.fetch_all_odds", return_value=MOCK_FEEDS):
            import scraping.match_ledger as ml
            stats = ml.enriquecer_ss_fs_con_aggregator(fecha, data_dir=str(tmp_path))

        ledger_actualizado = load_ledger(fecha, str(tmp_path))
        ss = ledger_actualizado["single_source_fs"]

        # nada que enriquecer — cuota1 no era None
        assert stats["enriquecidos"] == 0
        assert ss[0]["cuota1"] == 1.8      # sin cambio
        assert ss[0]["_cuota_source"] == "flashscore"  # sin cambio
        assert "_enriched_by" not in ss[0]  # no marcado como enriquecido

    def test_homonimo_no_enriquece(self, tmp_path):
        """D121-03: si hay 2 candidatos para el mismo apellido → no enriquecer."""
        fecha = "2099-01-03"
        ledger = _make_ledger([{
            "jugador1": "Smith J.",
            "jugador2": "Garcia R.",
            "cuota1": None,
            "cuota2": None,
            "join_method": "SINGLE_SOURCE_FS",
        }], fecha)
        save_ledger(ledger, fecha, str(tmp_path))

        # Feed con dos "smith" → homónimo
        feeds_homonimo = {
            "smith john": {"betplay": {"odds": 2.0, "jugador": "John Smith",
                                       "rival": "X", "event_id": 1, "outcome_id": 2,
                                       "bookmaker": "betplay"}},
            "smith jane": {"betplay": {"odds": 1.5, "jugador": "Jane Smith",
                                       "rival": "Y", "event_id": 3, "outcome_id": 4,
                                       "bookmaker": "betplay"}},
            "garcia": {"betplay": {"odds": 1.8, "jugador": "Garcia R.",
                                   "rival": "Smith", "event_id": 5, "outcome_id": 6,
                                   "bookmaker": "betplay"}},
        }

        with patch("scripts.odds_aggregator.fetch_all_odds", return_value=feeds_homonimo):
            import scraping.match_ledger as ml
            stats = ml.enriquecer_ss_fs_con_aggregator(fecha, data_dir=str(tmp_path))

        ledger_actualizado = load_ledger(fecha, str(tmp_path))
        ss = ledger_actualizado["single_source_fs"]

        assert stats["homonimos"] == 1
        assert stats["enriquecidos"] == 0
        assert ss[0]["cuota1"] is None   # no modificado
