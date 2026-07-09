"""
tests/test_nodo52.py — Nodo-52: Shadow Book CLV Tracking
Tests T52-01 a T52-10 (incluye Addendum T52-09, T52-10) + T52-11→T52-15 (D52-03 close_snapshot)

Baseline esperado: 1491 tests pasan.
"""
import json
import os
import pytest
from unittest.mock import patch

import shadow_book
from shadow_book import (
    log_picks,
    settle,
    report,
    calc_clv,
    wilson_ci,
    _build_sb_id,
    _match_key,
    _sb_status,
    _build_record,
    _segment_metrics,
    _compute_line_signal,
    close_snapshot,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

def _pick_aprobado():
    """Pick que pasa todos los gates — status APROBADO."""
    return {
        "partido":           "Greet Minnen vs Fiona Ferro",
        "favorito_predicho": "Greet Minnen",
        "cuota_favorito":    1.29,
        "cuota_rival":       3.50,
        "torneo":            "Wimbledon Qualifying",
        "superficie":        "grass",
        "tier":              "grand_slam",
        "edge":              0.062,
        "edge_pct":          "6.2%",
        "kelly_kl":          0.041,
        "p_modelo":          0.712,
        "p_implicita":       0.775,
        "apostar":           True,
        "motivo_reclasificacion": None,
        "n_h2h":             4,
        "match_id":          "fs_abc123",
        "match_url":         "https://flashscore.com/abc123",
        "cuota_es_real":     True,
        "markov_favorito":   "HOT",
        "markov_rival":      "COLD",
        "markov_conf_fav":   0.66,
        "markov_conf_rival": 0.71,
        "bbi":               0.35,
        "mpq":               0.0123,
        "golden_zone":       False,
        "gap_flag":          "MIXED",
        "calibration_gap":   0.02,
        "n_axes_active":     3,
        "alignment_flag":    "PARTIAL_ALIGNMENT",
        "confidence_flag":   "STRONG",
        "data_completeness": 0.85,
        "zona_cuota":        "favorite",
        "p_historica_usada": 0.758,
    }


def _pick_watchlist_t33():
    """Pick bloqueado por T33-01 (n_h2h=0, p<0.55) — status WATCHLIST."""
    return {
        "partido":           "Carlos Alcaraz vs Alex Michelsen",
        "favorito_predicho": "Carlos Alcaraz",
        "cuota_favorito":    2.50,
        "cuota_rival":       1.60,
        "torneo":            "Wimbledon",
        "superficie":        "grass",
        "tier":              "grand_slam",
        "edge":              0.045,
        "edge_pct":          "4.5%",
        "kelly_kl":          0.018,
        "p_modelo":          0.520,
        "p_implicita":       0.400,
        "apostar":           False,
        "motivo_reclasificacion": "T33-01: n_h2h=0 + p_modelo=0.520<0.55 (coin-flip bloqueado)",
        "n_h2h":             0,
        "match_id":          "fs_def456",
        "match_url":         None,
        "cuota_es_real":     True,
        "markov_favorito":   "NEUTRAL",
        "markov_rival":      "NEUTRAL",
        "markov_conf_fav":   0.5,
        "markov_conf_rival": 0.5,
        "bbi":               0.55,
        "mpq":               0.0089,
        "golden_zone":       False,
        "gap_flag":          "MARKET_DRIVEN",
        "calibration_gap":   0.01,
        "n_axes_active":     1,
        "alignment_flag":    "PARTIAL_ALIGNMENT",
        "confidence_flag":   "LOW",
        "data_completeness": 0.90,
        "zona_cuota":        "slight_underdog",
        "p_historica_usada": 0.758,
    }


def _make_edge_report(apostar=None, watchlist=None, no_data=None):
    return {
        "metadata": {"fecha": "2026-07-02", "gate_version": "v3"},
        "apostar":  apostar or [],
        "watchlist": watchlist or [],
        "sin_edge": [],
        "sin_datos": [],
        "no_data": no_data or [],
    }


@pytest.fixture
def shadow_dir(tmp_path, monkeypatch):
    """Redirige SHADOW_DIR a tmp_path para tests aislados."""
    d = str(tmp_path / "shadow_book")
    monkeypatch.setattr(shadow_book, "SHADOW_DIR", d)
    return d


FECHA = "2026-07-02"


# ═══════════════════════════════════════════════════════════════════════════════
# T52-01: log_picks escribe APROBADO + WATCHLIST con gate_bloqueante correcto
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogPicks:
    def test_t52_01_escribe_aprobado_y_watchlist(self, shadow_dir):
        """log_picks escribe APROBADO + WATCHLIST; gate_bloqueante correcto."""
        aprobado = _pick_aprobado()
        watchlist = _pick_watchlist_t33()
        report_dict = _make_edge_report(apostar=[aprobado], watchlist=[watchlist])

        n = log_picks(report_dict, {'fecha': FECHA})

        assert n == 2

        # Leer JSONL y verificar
        path = os.path.join(shadow_dir, f"sb_{FECHA}.jsonl")
        assert os.path.exists(path)
        records = {}
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                records[r['sb_id']] = r

        # Filtrar session_meta
        picks = {k: v for k, v in records.items() if v.get('_type') != 'session_meta'}
        assert len(picks) == 2

        # Encontrar el pick aprobado
        aprobado_rec = next(
            r for r in picks.values()
            if r['pick_snapshot']['apostar'] is True
        )
        assert _sb_status(aprobado_rec['pick_snapshot']) == 'APROBADO'
        assert aprobado_rec['pick_snapshot']['favorito_predicho'] == 'Greet Minnen'

        # Encontrar el pick watchlist
        watchlist_rec = next(
            r for r in picks.values()
            if r['pick_snapshot']['apostar'] is False
        )
        assert _sb_status(watchlist_rec['pick_snapshot']) == 'WATCHLIST'
        # Gate bloqueante se extrae del motivo_reclasificacion
        gate = watchlist_rec['pick_snapshot'].get('motivo_reclasificacion', '')
        assert 'T33-01' in gate

    def test_t52_01b_no_data_pool_incluido(self, shadow_dir):
        """log_picks también procesa el pool no_data (Nodo-51 F2)."""
        no_data_pick = {
            "partido": "Player A vs Player B",
            "favorito_predicho": "Player A",
            "cuota_favorito": 1.80,
            "torneo": "ITF Bogota",
            "superficie": "clay",
            "tier": "itf",
            "edge": 0.03,
            "kelly_kl": 0.01,
            "p_modelo": 0.520,
            "apostar": False,
            "status": "NO_DATA",
            "motivo_reclasificacion": "HISTORIAL_NO_EXTRAIDO: sin datos de Player A",
            "n_h2h": 0,
            "match_id": "fs_nd001",
            "cuota_es_real": True,
        }
        report_dict = _make_edge_report(no_data=[no_data_pick])

        n = log_picks(report_dict, {'fecha': FECHA})
        assert n == 1

        path = os.path.join(shadow_dir, f"sb_{FECHA}.jsonl")
        picks_lines = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get('_type') != 'session_meta':
                    picks_lines.append(r)
        assert len(picks_lines) == 1
        assert picks_lines[0]['pick_snapshot']['status'] == 'NO_DATA'


# ═══════════════════════════════════════════════════════════════════════════════
# T52-02: sb_id determinista — doble corrida no duplica, conserva logged_at
# ═══════════════════════════════════════════════════════════════════════════════

class TestSbIdDeterminista:
    def test_t52_02_upsert_conserva_logged_at(self, shadow_dir):
        """Doble corrida: 0 duplicados, logged_at original conservado."""
        aprobado = _pick_aprobado()
        report_dict = _make_edge_report(apostar=[aprobado])

        # Primera corrida
        n1 = log_picks(report_dict, {'fecha': FECHA})
        assert n1 == 1

        # Leer logged_at original
        path = os.path.join(shadow_dir, f"sb_{FECHA}.jsonl")
        with open(path) as f:
            rec_orig = next(
                json.loads(line) for line in f
                if json.loads(line).get('_type') != 'session_meta'
            )
        logged_at_orig = rec_orig['logged_at']

        # Segunda corrida — mismo día
        import time; time.sleep(0.01)
        n2 = log_picks(report_dict, {'fecha': FECHA})
        assert n2 == 0  # sin nuevos

        # logged_at original conservado (inmutabilidad §1)
        with open(path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        picks = [r for r in lines if r.get('_type') != 'session_meta']
        assert len(picks) == 1
        assert picks[0]['logged_at'] == logged_at_orig

    def test_t52_02b_sb_id_invariante_orden_p1_p2(self):
        """sb_id es el mismo independientemente del orden p1/p2."""
        id1 = _build_sb_id(FECHA, "Wimbledon Qualifying", "Greet Minnen", "Fiona Ferro")
        id2 = _build_sb_id(FECHA, "Wimbledon Qualifying", "Fiona Ferro", "Greet Minnen")
        assert id1 == id2


# ═══════════════════════════════════════════════════════════════════════════════
# T52-03: settle cruza match_key y marca WON/LOST (fixture 3 partidos)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSettle:
    def _escribir_picks(self, shadow_dir, picks_data):
        """Helper: escribe picks al JSONL directamente."""
        path = os.path.join(shadow_dir, f"sb_{FECHA}.jsonl")
        os.makedirs(shadow_dir, exist_ok=True)
        with open(path, 'w') as f:
            for rec in picks_data:
                f.write(json.dumps(rec) + '\n')

    def test_t52_03_settle_won_lost_correcto(self, shadow_dir):
        """settle con 3 partidos fixture: 2 WON, 1 LOST."""
        # 3 picks pre-escritos en el JSONL
        snap1 = _pick_aprobado()   # Minnen → gana
        snap2 = _pick_watchlist_t33()  # Alcaraz → pierde en fixture
        snap3 = {**_pick_aprobado(),
                 "partido": "Rafael Nadal vs Novak Djokovic",
                 "favorito_predicho": "Rafael Nadal",
                 "cuota_favorito": 2.20,
                 "match_id": "fs_ghi789",
                 }

        mk1 = _match_key("Greet Minnen", "Fiona Ferro")
        mk2 = _match_key("Carlos Alcaraz", "Alex Michelsen")
        mk3 = _match_key("Rafael Nadal", "Novak Djokovic")

        picks_jsonl = [
            {
                "sb_id": "2026-07-02_wimbledon-q_ferro-minnen_ML",
                "logged_at": "2026-07-02T04:00:00+00:00",
                "match_key": mk1,
                "es_qualifying": True,
                "season_transition_flag": True,
                "pick_snapshot": snap1,
            },
            {
                "sb_id": "2026-07-02_wimbledon_alcaraz-michelsen_ML",
                "logged_at": "2026-07-02T04:01:00+00:00",
                "match_key": mk2,
                "es_qualifying": False,
                "season_transition_flag": False,
                "pick_snapshot": snap2,
            },
            {
                "sb_id": "2026-07-02_wimbledon_djokovic-nadal_ML",
                "logged_at": "2026-07-02T04:02:00+00:00",
                "match_key": mk3,
                "es_qualifying": False,
                "season_transition_flag": False,
                "pick_snapshot": snap3,
            },
        ]
        self._escribir_picks(shadow_dir, picks_jsonl)

        # Mapa de resultados (fixture)
        resultados_map = {
            mk1: {'ganador': 'Greet Minnen', 'cuota_cierre': 1.22, 'provenance': 'flashscore_ref', 'void': False},
            mk2: {'ganador': 'Alex Michelsen', 'cuota_cierre': 1.65, 'provenance': 'flashscore_ref', 'void': False},
            mk3: {'ganador': 'Rafael Nadal', 'cuota_cierre': 2.10, 'provenance': 'flashscore_ref', 'void': False},
        }

        n = settle(FECHA, resultados_map=resultados_map)

        assert n == 3

        # Verificar WON / LOST
        path = os.path.join(shadow_dir, f"sb_{FECHA}.jsonl")
        recs = {}
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                recs[r['sb_id']] = r

        assert recs["2026-07-02_wimbledon-q_ferro-minnen_ML"]['resolucion']['resultado'] == 'WON'
        assert recs["2026-07-02_wimbledon_alcaraz-michelsen_ML"]['resolucion']['resultado'] == 'LOST'
        assert recs["2026-07-02_wimbledon_djokovic-nadal_ML"]['resolucion']['resultado'] == 'WON'


# ═══════════════════════════════════════════════════════════════════════════════
# T52-04: CLV — cuota_tomada=4.90, cierre=3.80 → clv_pct ≈ 28.9
# ═══════════════════════════════════════════════════════════════════════════════

class TestCLV:
    def test_t52_04_clv_formula(self):
        """(4.90 / 3.80 − 1) × 100 ≈ 28.9%"""
        clv = calc_clv(4.90, 3.80)
        assert abs(clv - 28.9) < 0.1, f"Esperado ≈28.9, obtenido {clv}"

    def test_t52_04b_clv_negativo(self):
        """Cuota tomada < cierre → CLV negativo (perdiste ventaja)."""
        clv = calc_clv(1.50, 1.80)
        assert clv < 0

    def test_t52_04c_clv_cero_cierre_invalido(self):
        """cuota_cierre=0 → CLV=0 (sin crash)."""
        assert calc_clv(2.00, 0) == 0.0

    def test_t52_04d_provenance_separada(self, shadow_dir):
        """
        Addendum §4.2: kambi_close y flashscore_ref nunca se mezclan en métrica.
        Verificar que el campo cuota_cierre_provenance se preserva.
        """
        snap = _pick_aprobado()
        mk = _match_key("Greet Minnen", "Fiona Ferro")
        rec = {
            "sb_id": "test_prov",
            "logged_at": "2026-07-02T04:00:00+00:00",
            "match_key": mk,
            "es_qualifying": True,
            "season_transition_flag": False,
            "pick_snapshot": snap,
        }
        path = os.path.join(shadow_dir, f"sb_{FECHA}.jsonl")
        os.makedirs(shadow_dir, exist_ok=True)
        with open(path, 'w') as f:
            f.write(json.dumps(rec) + '\n')

        resultados_map = {
            mk: {
                'ganador': 'Greet Minnen',
                'cuota_cierre': 1.22,
                'provenance': 'kambi_close',  # Momento 2
                'void': False,
            }
        }
        settle(FECHA, resultados_map=resultados_map)

        with open(path) as f:
            settled_rec = next(
                json.loads(l) for l in f if json.loads(l).get('sb_id') == 'test_prov'
            )
        assert settled_rec['resolucion']['cuota_cierre_provenance'] == 'kambi_close'


# ═══════════════════════════════════════════════════════════════════════════════
# T52-05: VOID excluido de hit% y ROI
# ═══════════════════════════════════════════════════════════════════════════════

class TestVoid:
    def test_t52_05_void_excluido_de_metricas(self):
        """VOID excluido de hit% y ROI flat; contabilizado aparte."""
        records_settled = [
            {
                "pick_snapshot": {"cuota_favorito": 2.00, "apostar": True},
                "resolucion": {"resultado": "WON",  "pnl_flat_1u": 1.00, "clv_pct": 5.0},
            },
            {
                "pick_snapshot": {"cuota_favorito": 1.80, "apostar": True},
                "resolucion": {"resultado": "LOST", "pnl_flat_1u": -1.0, "clv_pct": -2.0},
            },
            {
                "pick_snapshot": {"cuota_favorito": 3.00, "apostar": True},
                "resolucion": {"resultado": "VOID", "pnl_flat_1u": 0.0,  "clv_pct": None},
            },
        ]

        m = _segment_metrics(records_settled)

        assert m['n'] == 2        # VOID excluido
        assert m['void'] == 1     # VOID contabilizado aparte
        assert m['hits'] == 1
        assert m['hit_pct'] == 50.0
        # ROI: (1.00 − 1.00) / 2 × 100 = 0.0%
        assert m['roi'] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# T52-06: Wilson CI — n=34, hits=20 → IC ≈ [42.2, 73.6]
# ═══════════════════════════════════════════════════════════════════════════════

class TestWilsonCI:
    def test_t52_06_wilson_ejemplo_spec(self):
        """n=34, hits=20 → IC95 ≈ [42.2, 73.6] (spec §5)."""
        lower, upper = wilson_ci(34, 20)
        assert abs(lower - 42.2) < 0.2, f"lower={lower}"
        assert abs(upper - 73.6) < 0.2, f"upper={upper}"

    def test_t52_06b_wilson_n_cero(self):
        """n=0 → retorna (0.0, 100.0) sin crash."""
        assert wilson_ci(0, 0) == (0.0, 100.0)

    def test_t52_06c_wilson_todos_ganadores(self):
        """n=10, hits=10 → IC upper debe ser 100.0."""
        lower, upper = wilson_ci(10, 10)
        assert lower > 65.0
        assert upper == 100.0

    def test_t52_06d_wilson_ninguno_gana(self):
        """n=10, hits=0 → IC lower debe ser 0.0."""
        lower, upper = wilson_ci(10, 0)
        assert lower == 0.0
        assert upper < 35.0


# ═══════════════════════════════════════════════════════════════════════════════
# T52-07: Inmutabilidad — pick_snapshot no se toca en settle
# ═══════════════════════════════════════════════════════════════════════════════

class TestInmutabilidad:
    def test_t52_07_pick_snapshot_intacto_post_settle(self, shadow_dir):
        """settle añade 'resolucion' pero pick_snapshot no se modifica."""
        snap = _pick_aprobado()
        snap_original = json.dumps(snap, sort_keys=True)

        mk = _match_key("Greet Minnen", "Fiona Ferro")
        rec = {
            "sb_id": "test_inmut",
            "logged_at": "2026-07-02T04:00:00+00:00",
            "match_key": mk,
            "es_qualifying": True,
            "season_transition_flag": False,
            "pick_snapshot": snap,
        }
        path = os.path.join(shadow_dir, f"sb_{FECHA}.jsonl")
        os.makedirs(shadow_dir, exist_ok=True)
        with open(path, 'w') as f:
            f.write(json.dumps(rec) + '\n')

        resultados_map = {
            mk: {'ganador': 'Greet Minnen', 'cuota_cierre': 1.22,
                 'provenance': 'flashscore_ref', 'void': False}
        }
        settle(FECHA, resultados_map=resultados_map)

        with open(path) as f:
            settled_rec = next(
                json.loads(l) for l in f if json.loads(l).get('sb_id') == 'test_inmut'
            )

        # 'resolucion' añadido
        assert 'resolucion' in settled_rec

        # pick_snapshot sin cambios
        snap_post = json.dumps(settled_rec['pick_snapshot'], sort_keys=True)
        assert snap_post == snap_original, "pick_snapshot fue mutado — viola inmutabilidad §1"

    def test_t52_07b_settle_doble_no_sobreescribe(self, shadow_dir):
        """Segunda llamada a settle no sobreescribe resolucion ya existente."""
        snap = _pick_aprobado()
        mk = _match_key("Greet Minnen", "Fiona Ferro")
        rec = {
            "sb_id": "test_doble",
            "logged_at": "2026-07-02T04:00:00+00:00",
            "match_key": mk,
            "es_qualifying": False,
            "season_transition_flag": False,
            "pick_snapshot": snap,
            "resolucion": {   # ya settled
                "settled_at": "ORIGINAL_TIMESTAMP",
                "resultado": "WON",
                "cuota_cierre": 1.22,
                "cuota_cierre_provenance": "kambi_close",
                "clv_pct": 5.74,
                "pnl_flat_1u": 0.29,
            }
        }
        path = os.path.join(shadow_dir, f"sb_{FECHA}.jsonl")
        os.makedirs(shadow_dir, exist_ok=True)
        with open(path, 'w') as f:
            f.write(json.dumps(rec) + '\n')

        resultados_map = {
            mk: {'ganador': 'Greet Minnen', 'cuota_cierre': 1.30,
                 'provenance': 'flashscore_ref', 'void': False}
        }
        n = settle(FECHA, resultados_map=resultados_map)
        assert n == 0  # sin nuevos settlements

        with open(path) as f:
            post_rec = next(json.loads(l) for l in f if json.loads(l).get('sb_id') == 'test_doble')

        # settled_at original conservado
        assert post_rec['resolucion']['settled_at'] == "ORIGINAL_TIMESTAMP"
        assert post_rec['resolucion']['cuota_cierre'] == 1.22  # no sobreescrita


# ═══════════════════════════════════════════════════════════════════════════════
# T52-08: Hook — edge_report malformado no crashea el PASO 3
# ═══════════════════════════════════════════════════════════════════════════════

class TestHookSafety:
    def test_t52_08_edge_report_none(self, shadow_dir):
        """log_picks con None → 0 nuevos, sin exception."""
        result = log_picks(None, {'fecha': FECHA})
        assert result == 0

    def test_t52_08b_edge_report_vacio(self, shadow_dir):
        """log_picks con {} → 0 nuevos, sin exception."""
        result = log_picks({}, {'fecha': FECHA})
        assert result == 0

    def test_t52_08c_picks_con_partido_invalido(self, shadow_dir):
        """Picks sin campo 'partido' no crashean — se ignoran."""
        bad_pick = {"favorito_predicho": "Player A", "apostar": True}
        report_dict = _make_edge_report(apostar=[bad_pick])
        result = log_picks(report_dict, {'fecha': FECHA})
        assert result == 0

    def test_t52_08d_hook_excepcion_interna(self, shadow_dir):
        """Si _build_record lanza excepción internamente, log_picks sigue sin crash."""
        malformed_pick = {
            "partido": "A vs B",
            "favorito_predicho": "A",
            "torneo": None,    # torneo None → puede causar error en _slug
            "apostar": True,
        }
        report_dict = _make_edge_report(apostar=[malformed_pick])
        # No debe levantar excepción
        result = log_picks(report_dict, {'fecha': FECHA})
        # Puede ser 0 o 1 dependiendo de si _slug maneja None
        assert isinstance(result, int)


# ═══════════════════════════════════════════════════════════════════════════════
# T52-09: session_meta registrado (Addendum B.3)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionMeta:
    def test_t52_09_session_meta_escrito(self, shadow_dir):
        """log_picks escribe un registro session_meta con _type y n por status."""
        aprobado = _pick_aprobado()
        watchlist = _pick_watchlist_t33()
        report_dict = _make_edge_report(apostar=[aprobado], watchlist=[watchlist])

        log_picks(report_dict, {
            'fecha': FECHA,
            'dispersion_level': 'DIFFERENTIATED',
            'session_regime': 'BULLISH',
        })

        path = os.path.join(shadow_dir, f"sb_{FECHA}.jsonl")
        with open(path) as f:
            lines = [json.loads(l) for l in f if l.strip()]

        session_metas = [r for r in lines if r.get('_type') == 'session_meta']
        assert len(session_metas) == 1

        sm = session_metas[0]
        assert sm['fecha'] == FECHA
        assert sm['n_apostar'] == 1
        assert sm['n_watchlist'] == 1
        assert sm['n_no_data'] == 0
        assert sm['dispersion_level'] == 'DIFFERENTIATED'
        assert sm['session_regime'] == 'BULLISH'

    def test_t52_09b_cv_edge_calculado(self, shadow_dir):
        """cv_edge se calcula desde los edges de los picks."""
        pick1 = {**_pick_aprobado(), "edge": 0.10}
        pick2 = {**_pick_watchlist_t33(), "edge": 0.05}
        report_dict = _make_edge_report(apostar=[pick1], watchlist=[pick2])

        log_picks(report_dict, {'fecha': FECHA})

        path = os.path.join(shadow_dir, f"sb_{FECHA}.jsonl")
        with open(path) as f:
            sm = next(json.loads(l) for l in f if json.loads(l).get('_type') == 'session_meta')

        assert sm['cv_edge'] is not None
        assert sm['cv_edge'] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# T52-10: pick_snapshot preserva los campos completos (Addendum B.1)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPickSnapshot:
    def test_t52_10_pick_snapshot_completo(self, shadow_dir):
        """pick_snapshot preserva TODOS los campos del pick sin mutar ni omitir."""
        aprobado = _pick_aprobado()
        # Campos clave que el spec anterior habría omitido
        campos_clave = [
            'bbi', 'mpq', 'golden_zone', 'gap_flag', 'calibration_gap',
            'n_axes_active', 'alignment_flag', 'confidence_flag',
            'data_completeness', 'zona_cuota', 'p_historica_usada',
            'markov_conf_fav', 'markov_conf_rival',
        ]

        report_dict = _make_edge_report(apostar=[aprobado])
        log_picks(report_dict, {'fecha': FECHA})

        path = os.path.join(shadow_dir, f"sb_{FECHA}.jsonl")
        with open(path) as f:
            rec = next(
                json.loads(l) for l in f
                if json.loads(l).get('_type') != 'session_meta'
            )

        snap = rec['pick_snapshot']
        for campo in campos_clave:
            assert campo in snap, f"Campo '{campo}' falta en pick_snapshot"
            assert snap[campo] == aprobado[campo], (
                f"Campo '{campo}' mutado: {snap[campo]} != {aprobado[campo]}"
            )

    def test_t52_10b_pick_snapshot_no_es_copia_modificada(self, shadow_dir):
        """pick_snapshot es el dict original sin transformaciones."""
        aprobado = _pick_aprobado()
        original_keys = set(aprobado.keys())

        report_dict = _make_edge_report(apostar=[aprobado])
        log_picks(report_dict, {'fecha': FECHA})

        path = os.path.join(shadow_dir, f"sb_{FECHA}.jsonl")
        with open(path) as f:
            rec = next(
                json.loads(l) for l in f
                if json.loads(l).get('_type') != 'session_meta'
            )

        snap_keys = set(rec['pick_snapshot'].keys())
        # No se añadieron ni quitaron campos
        assert snap_keys == original_keys


# ═══════════════════════════════════════════════════════════════════════════════
# H52-05: _compute_line_signal + _append_hypothesis_h52_05 (V-26-3d)
# ═══════════════════════════════════════════════════════════════════════════════

class TestH5205LineSignal:
    def _make_settled(self, cuota_favorito, cuota_cierre, resultado='WON',
                      provenance='flashscore_ref', via_kambi=False):
        """Helper: construye un registro settled con cuota original y de cierre."""
        rec = {
            'sb_id': f'test_{cuota_favorito}_{cuota_cierre}',
            'pick_snapshot': {'cuota_favorito': cuota_favorito, 'apostar': True},
            'resolucion': {
                'resultado': resultado,
                'pnl_flat_1u': cuota_favorito - 1 if resultado == 'WON' else -1.0,
                'clv_pct': None,
                'cuota_cierre': None if via_kambi else cuota_cierre,
                'cuota_cierre_provenance': provenance,
            },
        }
        if via_kambi:
            rec['cierre_kambi'] = {'cuota': cuota_cierre, 'provenance': 'kambi_close'}
        return rec

    def test_steam_in_delta_mayor_4pct(self):
        """Cuota baja >4% → STEAM_IN (mercado acepta nuestra posición)."""
        rec = self._make_settled(cuota_favorito=2.00, cuota_cierre=1.85)  # -7.5%
        assert _compute_line_signal(rec) == 'STEAM_IN'

    def test_drift_out_delta_mayor_4pct(self):
        """Cuota sube >4% → DRIFT_OUT (mercado se aleja)."""
        rec = self._make_settled(cuota_favorito=2.00, cuota_cierre=2.20)  # +10%
        assert _compute_line_signal(rec) == 'DRIFT_OUT'

    def test_stable_dentro_del_umbral(self):
        """Delta ≤4% → STABLE."""
        rec = self._make_settled(cuota_favorito=2.00, cuota_cierre=2.03)  # +1.5%
        assert _compute_line_signal(rec) == 'STABLE'

    def test_no_data_sin_cuota_cierre(self):
        """Sin cuota_cierre → NO_DATA."""
        rec = {
            'pick_snapshot': {'cuota_favorito': 2.00},
            'resolucion': {'resultado': 'WON', 'cuota_cierre': None},
        }
        assert _compute_line_signal(rec) == 'NO_DATA'

    def test_no_data_sin_cuota_favorito(self):
        """Sin cuota_favorito en pick_snapshot → NO_DATA."""
        rec = {
            'pick_snapshot': {},
            'resolucion': {'cuota_cierre': 1.80},
        }
        assert _compute_line_signal(rec) == 'NO_DATA'

    def test_kambi_close_tiene_prioridad(self):
        """cierre_kambi tiene prioridad sobre resolucion.cuota_cierre."""
        # resolucion.cuota_cierre daría DRIFT (2.30 > 2.00 × 1.04)
        # cierre_kambi daría STEAM (1.85 < 2.00 × 0.96)
        rec = self._make_settled(
            cuota_favorito=2.00,
            cuota_cierre=2.30,
            via_kambi=False,
        )
        rec['resolucion']['cuota_cierre'] = 2.30     # DRIFT si se usa esto
        rec['cierre_kambi'] = {'cuota': 1.85}        # STEAM — debe ganar
        assert _compute_line_signal(rec) == 'STEAM_IN'

    def test_h52_05_report_sin_delta(self, shadow_dir):
        """Con picks settled pero sin cuota_cierre, H52-05 muestra n_delta=0."""
        from shadow_book import _append_hypothesis_h52_05
        # Picks settled sin cuota de cierre (no hubo Momento 2 ni settlement con cuota)
        sin_delta = [
            {
                'pick_snapshot': {'cuota_favorito': 2.00, 'apostar': True},
                'resolucion': {'resultado': 'WON', 'pnl_flat_1u': 1.0, 'cuota_cierre': None},
            }
            for _ in range(3)
        ]
        lines = []
        _append_hypothesis_h52_05(sin_delta, lines)
        txt = ' '.join(lines)
        assert 'H52-05' in txt
        assert 'n=0' in txt or 'sin picks con delta' in txt

    def test_h52_05_steam_mayor_drift(self, shadow_dir):
        """Con STEAM>DRIFT hit%, reporte indica CONFIRMADA cuando n≥20."""
        from shadow_book import _append_hypothesis_h52_05
        # 12 STEAM ganados + 8 DRIFT perdidos = n_delta=20
        steam_wins = [
            self._make_settled(2.00, 1.85, 'WON') for _ in range(12)
        ]
        drift_losses = [
            self._make_settled(2.00, 2.20, 'LOST') for _ in range(8)
        ]
        settled = steam_wins + drift_losses
        lines = []
        _append_hypothesis_h52_05(settled, lines)
        txt = ' '.join(lines)
        assert 'CONFIRMADA' in txt
        assert 'STEAM' in txt
        assert 'DRIFT' in txt

    def test_h52_05_no_confirmada_steam_menor(self, shadow_dir):
        """Cuando STEAM <= DRIFT hit%, indica NO CONFIRMADA."""
        from shadow_book import _append_hypothesis_h52_05
        # 8 STEAM perdidos + 12 DRIFT ganados
        steam_losses = [self._make_settled(2.00, 1.85, 'LOST') for _ in range(8)]
        drift_wins   = [self._make_settled(2.00, 2.20, 'WON')  for _ in range(12)]
        lines = []
        _append_hypothesis_h52_05(steam_losses + drift_wins, lines)
        txt = ' '.join(lines)
        assert 'NO CONFIRMADA' in txt

    def test_h52_05_continuar_con_n_insuficiente(self, shadow_dir):
        """Con n_delta < 20, indica CONTINUAR."""
        from shadow_book import _append_hypothesis_h52_05
        picks = [self._make_settled(2.00, 1.85, 'WON') for _ in range(5)]
        lines = []
        _append_hypothesis_h52_05(picks, lines)
        txt = ' '.join(lines)
        assert 'CONTINUAR' in txt
        assert '5/20' in txt


# ═══════════════════════════════════════════════════════════════════════════════
# Match key y sb_id helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestHelpers:
    def test_match_key_orden_invariante(self):
        """_match_key es invariante al orden de jugadores."""
        mk1 = _match_key("Greet Minnen", "Fiona Ferro")
        mk2 = _match_key("Fiona Ferro", "Greet Minnen")
        assert mk1 == mk2

    def test_sb_status_deriva_correctamente(self):
        """_sb_status deriva APROBADO/WATCHLIST/NO_DATA del pick."""
        assert _sb_status({"apostar": True}) == "APROBADO"
        assert _sb_status({"apostar": False}) == "WATCHLIST"
        assert _sb_status({"apostar": False, "status": "NO_DATA"}) == "NO_DATA"
        assert _sb_status({}) == "WATCHLIST"


# ═══════════════════════════════════════════════════════════════════════════════
# T52-11→T52-15: close_snapshot (D52-03 — Momento 2)
# ═══════════════════════════════════════════════════════════════════════════════

def _fake_outcomes_map():
    """Simula respuesta de fetch_kambi_outcomes con 2 jugadores."""
    return {
        "greet minnen":  {"odds": 1.22, "jugador": "Greet Minnen"},
        "fiona ferro":   {"odds": 3.10, "jugador": "Fiona Ferro"},
        "carlos alcaraz": {"odds": 1.30, "jugador": "Carlos Alcaraz"},
    }


class TestCloseSnapshot:
    def _open_rec(self, shadow_dir, sb_id, favorito, cuota):
        """Escribe un registro abierto (sin resolucion, sin cierre_kambi)."""
        snap = {**_pick_aprobado(), "favorito_predicho": favorito, "cuota_favorito": cuota}
        mk = _match_key(favorito, "Fiona Ferro")
        rec = {
            "sb_id": sb_id,
            "logged_at": "2026-07-02T04:00:00+00:00",
            "match_key": mk,
            "es_qualifying": False,
            "season_transition_flag": False,
            "pick_snapshot": snap,
        }
        path = os.path.join(shadow_dir, f"sb_{FECHA}.jsonl")
        os.makedirs(shadow_dir, exist_ok=True)
        existing = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    existing[r['sb_id']] = r
        existing[sb_id] = rec
        with open(path, 'w') as f:
            for r in existing.values():
                f.write(json.dumps(r) + '\n')

    def test_t52_11_cierre_kambi_anadido_por_nombre(self, shadow_dir):
        """T52-11: match por nombre completo → cierre_kambi con cuota y provenance=kambi_close."""
        self._open_rec(shadow_dir, "sb_minnen_test", "Greet Minnen", 1.29)

        with patch("betplay_combo_builder.fetch_kambi_outcomes",
                   return_value=(_fake_outcomes_map(), {})):
            n = close_snapshot(fecha=FECHA)

        assert n == 1
        path = os.path.join(shadow_dir, f"sb_{FECHA}.jsonl")
        with open(path) as f:
            rec = next(json.loads(l) for l in f if json.loads(l).get('sb_id') == 'sb_minnen_test')

        assert 'cierre_kambi' in rec
        assert rec['cierre_kambi']['cuota'] == 1.22
        assert rec['cierre_kambi']['provenance'] == 'kambi_close'
        assert 'captured_at' in rec['cierre_kambi']

    def test_t52_12_pick_snapshot_inmutable(self, shadow_dir):
        """T52-12: cierre_kambi es top-level — pick_snapshot no se toca (inmutabilidad §1)."""
        self._open_rec(shadow_dir, "sb_immut_test", "Greet Minnen", 1.29)
        snap_original = None
        path = os.path.join(shadow_dir, f"sb_{FECHA}.jsonl")
        with open(path) as f:
            for l in f:
                r = json.loads(l)
                if r.get('sb_id') == 'sb_immut_test':
                    snap_original = json.dumps(r['pick_snapshot'], sort_keys=True)

        with patch("betplay_combo_builder.fetch_kambi_outcomes",
                   return_value=(_fake_outcomes_map(), {})):
            close_snapshot(fecha=FECHA)

        with open(path) as f:
            rec = next(json.loads(l) for l in f if json.loads(l).get('sb_id') == 'sb_immut_test')

        assert json.dumps(rec['pick_snapshot'], sort_keys=True) == snap_original

    def test_t52_13_ya_settled_no_se_actualiza(self, shadow_dir):
        """T52-13: registros ya settled o con cierre_kambi se saltan — retorna 0 nuevos."""
        snap = _pick_aprobado()
        mk = _match_key("Greet Minnen", "Fiona Ferro")
        settled_rec = {
            "sb_id": "sb_settled",
            "logged_at": "2026-07-02T04:00:00+00:00",
            "match_key": mk,
            "es_qualifying": False,
            "season_transition_flag": False,
            "pick_snapshot": snap,
            "resolucion": {"resultado": "WON", "cuota_cierre": 1.18},
        }
        already_cierre = {
            "sb_id": "sb_cierre_exist",
            "logged_at": "2026-07-02T04:01:00+00:00",
            "match_key": mk,
            "es_qualifying": False,
            "season_transition_flag": False,
            "pick_snapshot": {**snap, "favorito_predicho": "Greet Minnen"},
            "cierre_kambi": {"cuota": 1.20, "provenance": "kambi_close", "captured_at": "T"},
        }
        path = os.path.join(shadow_dir, f"sb_{FECHA}.jsonl")
        os.makedirs(shadow_dir, exist_ok=True)
        with open(path, 'w') as f:
            f.write(json.dumps(settled_rec) + '\n')
            f.write(json.dumps(already_cierre) + '\n')

        with patch("betplay_combo_builder.fetch_kambi_outcomes",
                   return_value=(_fake_outcomes_map(), {})):
            n = close_snapshot(fecha=FECHA)

        assert n == 0  # nada que actualizar

    def test_t52_14_kambi_fetch_falla_no_crashea(self, shadow_dir):
        """T52-14: si fetch_kambi_outcomes lanza excepción → retorna 0 sin crash."""
        self._open_rec(shadow_dir, "sb_crash_test", "Greet Minnen", 1.29)

        with patch("betplay_combo_builder.fetch_kambi_outcomes",
                   side_effect=Exception("timeout")):
            n = close_snapshot(fecha=FECHA)

        assert n == 0
        # El JSONL no fue modificado — sin cierre_kambi
        path = os.path.join(shadow_dir, f"sb_{FECHA}.jsonl")
        with open(path) as f:
            rec = next(json.loads(l) for l in f if json.loads(l).get('sb_id') == 'sb_crash_test')
        assert 'cierre_kambi' not in rec

    def test_t52_15_sin_registros_abiertos_retorna_0(self, shadow_dir):
        """T52-15: si no hay registros abiertos → retorna 0 sin llamar Kambi."""
        # JSONL vacío (solo session_meta)
        path = os.path.join(shadow_dir, f"sb_{FECHA}.jsonl")
        os.makedirs(shadow_dir, exist_ok=True)
        sm = {"_type": "session_meta", "sb_id": f"SESSION_{FECHA}", "fecha": FECHA}
        with open(path, 'w') as f:
            f.write(json.dumps(sm) + '\n')

        called = []
        def fake_fetch():
            called.append(True)
            return ({}, {})

        with patch("betplay_combo_builder.fetch_kambi_outcomes", side_effect=fake_fetch):
            n = close_snapshot(fecha=FECHA)

        assert n == 0
        assert not called  # Kambi no fue contactado
