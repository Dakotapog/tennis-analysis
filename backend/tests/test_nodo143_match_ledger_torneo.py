"""
REGLA-T53: Tests Nodo-143 — Match Ledger: propagación de metadata torneo en joins.

D143-01: fusionar_dia() debe copiar tier/torneo_nombre/torneo_completo/pais/
         ranking1/ranking2/tournament_context desde Kambi al join cuando el
         campo no existe en el registro FlashScore (fill-gaps, no overwrite).
"""

import json
import pytest
from pathlib import Path

from scraping.match_ledger import fusionar_dia


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kambi(j1="Fearnley J.", j2="Wolf J.", hora="15:00",
           tier="challenger", torneo_nombre="Bloomfield Hills",
           torneo_completo="US - Bloomfield Hills Challenger",
           pais="USA", ranking1=95, ranking2=140,
           tournament_context="challenger_hard",
           cuota1=1.50, cuota2=2.60, event_id=9001):
    return {
        "jugador1": j1, "jugador2": j2, "hora": hora,
        "cuota1": cuota1, "cuota2": cuota2,
        "kambi_event_id": event_id, "outcome_id": event_id,
        "tier": tier,
        "torneo_nombre": torneo_nombre,
        "torneo_completo": torneo_completo,
        "pais": pais,
        "ranking1": ranking1,
        "ranking2": ranking2,
        "tournament_context": tournament_context,
    }


def _fs(j1="Fearnley J.", j2="Wolf J.", hora="15:00",
        match_url="https://flashscore.com/m/abc", match_id="abc123",
        h2h_url="https://flashscore.com/h2h/abc"):
    """FlashScore record: tiene URLs/IDs pero no metadata de torneo."""
    return {
        "jugador1": j1, "jugador2": j2, "hora": hora,
        "match_url": match_url, "match_id": match_id,
        "h2h_url": h2h_url,
        "cuota1": None, "cuota2": None,
        "resultado": "-", "estado": "Programado 15:00",
    }


def _run_join(kambi_list, fs_list, tmp_path):
    """Ejecuta fusionar_dia y devuelve solo los joins AUTO_JOIN del archivo escrito.

    El archivo merged usa 'partidos' (lista plana: joins + single-sources).
    Filtramos por join_method=='AUTO_JOIN' para aislar solo los joins.
    """
    fecha = "2026-07-25"
    merged_path, stats = fusionar_dia(
        kambi_matches=kambi_list,
        fs_matches=fs_list,
        fecha=fecha,
        output_dir=str(tmp_path),
    )
    data = json.loads(Path(merged_path).read_text(encoding="utf-8"))
    all_partidos = data.get("partidos", [])
    joins = [p for p in all_partidos if p.get("join_method") == "AUTO_JOIN"]
    return joins, stats


# ---------------------------------------------------------------------------
# Tests D142-01
# ---------------------------------------------------------------------------

class TestD14201TorneoMetadataPropagation:

    def test_join_preserves_torneo_nombre(self, tmp_path):
        """Join con FS sin torneo + Kambi con torneo → join tiene torneo_nombre correcto."""
        kambi = [_kambi(torneo_nombre="Bloomfield Hills")]
        fs = [_fs()]
        joins, stats = _run_join(kambi, fs, tmp_path)

        assert stats["joins_exitosos"] >= 1, "Debe haber al menos un join"
        join = joins[0]
        assert join.get("torneo_nombre") == "Bloomfield Hills", (
            f"torneo_nombre perdido en join: {join.get('torneo_nombre')}"
        )

    def test_join_preserves_tier(self, tmp_path):
        """El campo tier de Kambi se propaga correctamente al join."""
        kambi = [_kambi(tier="challenger")]
        fs = [_fs()]
        joins, stats = _run_join(kambi, fs, tmp_path)

        assert stats["joins_exitosos"] >= 1
        join = joins[0]
        assert join.get("tier") == "challenger", (
            f"tier perdido en join: {join.get('tier')}"
        )

    def test_join_preserves_all_meta_fields(self, tmp_path):
        """Los 7 campos de metadata Kambi están todos en el join."""
        kambi = [_kambi(
            tier="challenger",
            torneo_nombre="Bloomfield Hills",
            torneo_completo="US - Bloomfield Hills Challenger",
            pais="USA",
            ranking1=95,
            ranking2=140,
            tournament_context="challenger_hard",
        )]
        fs = [_fs()]
        joins, stats = _run_join(kambi, fs, tmp_path)

        assert stats["joins_exitosos"] >= 1
        join = joins[0]

        assert join.get("tier") == "challenger"
        assert join.get("torneo_nombre") == "Bloomfield Hills"
        assert join.get("torneo_completo") == "US - Bloomfield Hills Challenger"
        assert join.get("pais") == "USA"
        assert join.get("ranking1") == 95
        assert join.get("ranking2") == 140
        assert join.get("tournament_context") == "challenger_hard"

    def test_no_overwrite_existing_field(self, tmp_path):
        """Si FS ya tiene un campo (ej. ranking1), Kambi no lo sobrescribe."""
        kambi = [_kambi(ranking1=95)]
        # FS con ranking1 propio
        fs_record = _fs()
        fs_record["ranking1"] = 50  # FS tiene dato propio
        joins, stats = _run_join(kambi, [fs_record], tmp_path)

        assert stats["joins_exitosos"] >= 1
        join = joins[0]
        # El valor de FS (50) debe prevalecer sobre el de Kambi (95)
        assert join.get("ranking1") == 50, (
            f"Kambi sobrescribió ranking1 de FS: {join.get('ranking1')}"
        )

    def test_handles_none_kambi_meta_field(self, tmp_path):
        """Campo Kambi con valor None no se copia al join (no genera KeyError)."""
        kambi_record = _kambi(torneo_nombre="Bloomfield Hills")
        kambi_record["pais"] = None  # campo explícitamente None
        fs = [_fs()]
        joins, stats = _run_join([kambi_record], fs, tmp_path)

        assert stats["joins_exitosos"] >= 1
        join = joins[0]
        # torneo_nombre sí se copia (no None)
        assert join.get("torneo_nombre") == "Bloomfield Hills"
        # pais=None no se copia (campo ausente o None en join)
        assert not join.get("pais"), (
            f"pais=None de Kambi no debe copiarse al join: {join.get('pais')}"
        )
