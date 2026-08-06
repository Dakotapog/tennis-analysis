"""
tests/test_nodo87_fixes.py — Nodo-87: cobertura de los 12 fixes de la auditoría D87

Los 1804 tests existentes (verificados 2026-07-11) NO cubrían ninguno de estos caminos —
por eso los bugs vivieron meses sin detectarse (Nodo-86). Cada clase de aquí corresponde
a un ID D87-xx / D64-01 de la tabla en Nodo-87-Fixes-Auditoria-D87.md.

REGLA-T53: cada test invoca la función real del módulo — nunca reimplementa la fórmula.
Las aserciones son estructurales/de umbral (ej. "stake==0", "gap>1.0") en vez de recalcular
valores exactos con la misma fórmula que se está probando.

Gap conocido y documentado: D87-07 (CPPI también cubre cobertura) y D87-10 (--all-picks
default False) quedan embebidos en trader_ev_tenis.main() sin extracción a función testeable
— no cubiertos aquí. Candidatos a refactor futuro si se quiere blindarlos también.
"""
import json
import os
from pathlib import Path

import pytest

import shadow_book
from shadow_book import settle, update_alpha_flags, _match_key

from trader_ev_tenis import _p_blend, _print_individuales

from edge_calculator import calcular_edge_completo

from betplay_combo_builder import _save_betslip_index

from betslip_registrar import _backfill_desde_edge

from pre_game_validator import validate_file


# ─────────────────────────────────────────────────────────────────────────────
# Fixture compartida — partido mínimo para calcular_edge_completo()
# (mismo patrón que tests/test_nodo33.py _make_partido)
# ─────────────────────────────────────────────────────────────────────────────

CALIB_MINIMAL = {
    "global": {"wins": 50, "losses": 20},
    "por_superficie": {
        "clay": {"wins": 31, "losses": 10},
    },
    "por_superficie_y_tier": {},
    "fallback_por_tier": {
        "grand_slam": 0.758,
        "atp1000":    0.65,
        "atp500":     0.62,
        "challenger": 0.55,
        "itf":        0.50,
    },
}


def _make_partido(
    jugador1="PlayerA",
    jugador2="PlayerB",
    cuota1=2.50,
    cuota2=1.55,
    confidence=65.0,
    n_h2h=2,
    elo_fav=1400,
    elo_rival=1600,
    torneo="ITF Testing",
    superficie="clay",
    phantom_p1=False,
    gcs_active_p1=False,
    form_decay_meta=None,
):
    """Partido mínimo para calcular_edge_completo(). favored = jugador1 siempre.

    elo_fav < elo_rival deliberadamente: el favorito del modelo tiene ELO más bajo
    que el rival (simula un pick tipo GCS/surface-specialist), lo que activa el eje
    'surface' del triple alignment (alpha_vs_elo alto) — necesario para que
    n_axes_active>=2 y el guard FIX-3 (N28F2) no bloquee 'apostar' antes de tiempo
    en los tests que no están probando ese guard específico.
    """
    ra = {
        "prediction": {
            "favored_player": jugador1,
            "confidence": confidence,
            "markov_analysis": None,
            "surface_specialization_meta": {
                "player1": {"gcs_active": gcs_active_p1},
                "player2": {"gcs_active": False},
            },
            "circuit_asymmetry": None,
            "score_breakdown": {},
            "reasoning": [],
        },
    }
    if form_decay_meta:
        ra["prediction"]["form_decay_meta"] = form_decay_meta
    if phantom_p1:
        ra["phantom_identity_p1"] = {"phantom": True, "type": "HOMONYM_GAP"}
    ra[f"{jugador1}_elo"] = elo_fav
    ra[f"{jugador2}_elo"] = elo_rival
    _sanit1 = jugador1.replace(" ", "_").replace(".", "_").strip("_")
    _sanit2 = jugador2.replace(" ", "_").replace(".", "_").strip("_")
    ra[f"{_sanit1}_ranking"] = 100
    ra[f"{_sanit2}_ranking"] = 80

    return {
        "jugador1": jugador1,
        "jugador2": jugador2,
        "cuota1": cuota1,
        "cuota2": cuota2,
        "torneo_completo": torneo,
        "superficie": superficie,
        "enfrentamientos_directos": [{"winner": jugador1}] * n_h2h,
        "ranking_analysis": ra,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# D87-01 — update_alpha_flags matchea por 'favorito_predicho' (H62-01 puede acumular)
# ═══════════════════════════════════════════════════════════════════════════════

class TestD87_01AlphaFlags:

    @pytest.fixture
    def shadow_dir(self, tmp_path, monkeypatch):
        d = str(tmp_path / "shadow_book")
        monkeypatch.setattr(shadow_book, "SHADOW_DIR", d)
        return d

    def _escribir_pick(self, shadow_dir, fecha, sb_id, favorito_predicho):
        path = os.path.join(shadow_dir, f"sb_{fecha}.jsonl")
        os.makedirs(shadow_dir, exist_ok=True)
        rec = {
            "sb_id": sb_id,
            "logged_at": f"{fecha}T04:00:00+00:00",
            "match_key": _match_key(favorito_predicho, "Rival Player"),
            "es_qualifying": False,
            "season_transition_flag": False,
            "pick_snapshot": {
                "partido": f"{favorito_predicho} vs Rival Player",
                "favorito_predicho": favorito_predicho,
                "cuota_favorito": 1.90,
                "apostar": True,
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return path

    def test_marca_alpha_promoted_por_favorito_predicho(self, shadow_dir):
        """Pre-fix: buscaba 'nombre'/'jugador'/'player' — no existe en pick_snapshot,
        0 registros marcados nunca. Post-fix: matchea por 'favorito_predicho'."""
        fecha = "2026-07-11"
        path = self._escribir_pick(shadow_dir, fecha, "sb_test_1", "Max Wiskandt")

        n_marcados = update_alpha_flags(fecha, ["Max Wiskandt"])

        assert n_marcados == 1
        with open(path, encoding="utf-8") as f:
            rec = json.loads(f.readline())
        assert rec["combo_flags"]["alpha_promoted"] is True

    def test_no_marca_nombre_no_listado(self, shadow_dir):
        fecha = "2026-07-11"
        self._escribir_pick(shadow_dir, fecha, "sb_test_2", "Otro Jugador")

        n_marcados = update_alpha_flags(fecha, ["Max Wiskandt"])

        assert n_marcados == 0


# ═══════════════════════════════════════════════════════════════════════════════
# D87-02 — Gate GCS no revive picks bloqueados por NO_DATA/phantom
# ═══════════════════════════════════════════════════════════════════════════════

class TestD87_02GateGCSRespetaGuards:

    def test_gcs_no_revive_pick_phantom(self):
        """Pick con phantom_identity detectado + gcs_active + edge>=0.15: las condiciones
        de GCS se cumplen (gcs_bonus=True) pero el guard NO_DATA/motivo_reclasificacion
        debe impedir que gcs_gate_applied/apostar se reactiven."""
        partido = _make_partido(
            jugador1="Phantom Player", jugador2="Real Rival",
            cuota1=2.50, cuota2=1.55, confidence=65.0,
            n_h2h=2, gcs_active_p1=True, phantom_p1=True,
        )

        r = calcular_edge_completo(partido, CALIB_MINIMAL)

        assert r is not None
        # Confirma que las condiciones de GCS SÍ se cumplían (si no, el test no probaría nada)
        assert r["gcs_bonus"] is True, "gcs_active debe detectarse pese al phantom"
        assert r["edge"] >= 0.15, "el edge debe superar el umbral GCS para que el gate se evalúe"
        # El guard D87-02 debe bloquear la reactivación
        assert r["status"] == "NO_DATA"
        assert r["phantom_data"] is True
        assert r["apostar"] is False
        assert r["gcs_gate_applied"] is False, (
            "D87-02: el gate GCS no debe reactivar un pick bloqueado por PHANTOM_IDENTITY"
        )

    def test_gcs_funciona_normalmente_sin_guard_activo(self):
        """Control: el mismo pick SIN phantom_identity sí debe poder activar el gate GCS
        — confirma que el fix no rompió el camino feliz (H60-01 graduada)."""
        partido = _make_partido(
            jugador1="Clean Player", jugador2="Real Rival",
            cuota1=2.50, cuota2=1.55, confidence=65.0,
            n_h2h=2, gcs_active_p1=True, phantom_p1=False,
        )

        r = calcular_edge_completo(partido, CALIB_MINIMAL)

        assert r is not None
        assert r["gcs_bonus"] is True
        assert r.get("status") != "NO_DATA"
        assert r["gcs_gate_applied"] is True, (
            "sin guard activo, el gate GCS debe seguir funcionando (H60-01 graduada)"
        )
        assert r["apostar"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# D87-04 — normalización de superficie '?' (bookmarklet Kambi) en edge_calculator
# ═══════════════════════════════════════════════════════════════════════════════

class TestD87_04NormalizacionSuperficie:

    def test_superficie_interrogacion_se_normaliza_a_unknown(self):
        partido = _make_partido(superficie="?")

        r = calcular_edge_completo(partido, CALIB_MINIMAL)

        assert r is not None
        assert r["superficie"] == "unknown", (
            "D87-04: '?' debe normalizarse a 'unknown', no crear un bucket nuevo"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# D87-05 — _p_blend: el prior no puede inflar por encima de p_modelo
# ═══════════════════════════════════════════════════════════════════════════════

class TestD87_05PBlendNoInfla:

    def test_prior_alto_se_clampea_al_modelo_con_n_h2h_cero(self):
        """Caso real del bug: n_h2h=0, prior=0.758 (accuracy del tier) > p_modelo=0.56.
        Pre-fix: p_blend=0.758 (EV ficticio). Post-fix: p_blend=0.56 (=p_modelo)."""
        p_blend = _p_blend(0.56, 0, 0.758)
        assert p_blend == pytest.approx(0.56)

    def test_prior_bajo_no_se_ve_afectado(self):
        """Cuando el prior YA es más conservador que el modelo, el comportamiento
        original (blend hacia el prior) se preserva sin cambios.
        pytest.approx: (3*0.40)/3 no necesariamente vuelve a dar exactamente 0.40
        en IEEE754 — comparar floats con == es incorrecto independientemente del fix."""
        p_blend = _p_blend(0.56, 0, 0.40)
        assert p_blend == pytest.approx(0.40)

    def test_p_blend_nunca_excede_p_modelo(self):
        """Invariante estructural del fix: para cualquier prior >= p_modelo, el blend
        resultante no puede superar p_modelo, sin importar n_h2h."""
        p_modelo = 0.55
        for n_h2h in (0, 3, 10):
            for prior in (0.60, 0.758, 0.90):
                assert _p_blend(p_modelo, n_h2h, prior) <= p_modelo + 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# D87-06 — pre_game_validator valida apostar+watchlist+sin_edge (schema real)
# ═══════════════════════════════════════════════════════════════════════════════

class TestD87_06ValidatorSchemaReal:

    def test_watchlist_con_kelly_cero_dispara_block(self, tmp_path):
        """Pre-fix: solo se escaneaba 'apostar' (fallback genérico a la primera lista).
        Un kelly_kl=0.0 en watchlist nunca se detectaba. Post-fix: los 3 pools se validan."""
        edge_report = {
            "apostar": [
                {"favorito_predicho": "Player OK", "kelly_kl": 0.04,
                 "cuota_favorito": 1.90, "n_h2h": 10, "ranking_favorito": 50,
                 "cuota_es_real": True},
            ],
            "watchlist": [
                {"favorito_predicho": "Player Bloqueado", "kelly_kl": 0.0,
                 "cuota_favorito": 2.10, "n_h2h": 12, "ranking_favorito": 80,
                 "cuota_es_real": True},
            ],
            "sin_edge": [],
        }
        path = tmp_path / "edge_report_test.json"
        path.write_text(json.dumps(edge_report), encoding="utf-8")

        exit_code = validate_file(path)

        assert exit_code == 2, (
            "D87-06: kelly_kl=0.0 en watchlist debe disparar BLOCK (exit_code=2) — "
            "antes del fix, watchlist nunca se escaneaba"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# D87-08 — betslip_index persiste p_modelo/kelly_kl para picks VARIABLE (cuota<1.50)
# ═══════════════════════════════════════════════════════════════════════════════

class TestD87_08BetslipIndexCubreVariable:

    def test_pick_cuota_baja_entra_al_index_con_p_modelo_y_kelly(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        picks = [{
            "outcome_id": "999001",
            "jugador": "Underdog Legs",
            "cuota": 1.20,
            "cuota_kambi": 1.18,
            "partido": "Underdog Legs vs Favorite Rival",
            "match_id": "fs_1",
            "superficie": "clay",
            "tier": "itf",
            "edge": "-13.0%",
            "p_modelo": 0.65,
            "kelly_kl": 0.0,
        }]

        path = _save_betslip_index(picks)
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        entry = data["index"]["999001"]
        assert entry["p_modelo"] == 0.65, (
            "D87-08: el index debe persistir p_modelo real, no el default 0.5"
        )
        assert entry["kelly_kl"] == 0.0
        assert entry["superficie"] == "clay"
        assert entry["tier"] == "itf"


# ═══════════════════════════════════════════════════════════════════════════════
# D87-09 — backfill automático de campos degradados desde el edge_report
# ═══════════════════════════════════════════════════════════════════════════════

class TestD87_09BackfillDesdeEdge:

    def _edge_idx(self):
        return {
            "max wiskandt": {
                "favorito_predicho": "Max Wiskandt",
                "superficie": "clay",
                "tier": "itf",
                "match_id": "fs_9001",
                "match_url": "https://flashscore.com/9001",
                "torneo": "M25 Marburg",
                "partido": "Max Wiskandt vs Rival",
                "p_modelo": 0.86,
                "kelly_kl": 0.041,
                "edge_pct": "33.9%",
            }
        }

    def test_completa_campos_degradados(self):
        pick = {
            "jugador": "Max Wiskandt",
            "superficie": "?",
            "tier": "?",
            "match_id": "",
            "p_modelo": 0.5,
            "kelly_kl": 0.0,
            "edge": "0%",
        }

        filled = _backfill_desde_edge(pick, self._edge_idx())

        assert filled is True
        assert pick["superficie"] == "clay"
        assert pick["tier"] == "itf"
        assert pick["match_id"] == "fs_9001"
        assert pick["p_modelo"] == 0.86
        assert pick["kelly_kl"] == 0.041

    def test_no_completa_si_jugador_no_esta_en_edge_report(self):
        pick = {"jugador": "Jugador Desconocido", "superficie": "?", "tier": "?",
                 "p_modelo": 0.5, "kelly_kl": 0.0}

        filled = _backfill_desde_edge(pick, self._edge_idx())

        assert filled is False
        assert pick["superficie"] == "?"

    def test_no_sobreescribe_campos_ya_reales(self):
        """Solo debe rellenar valores degradados, nunca pisar un dato real ya presente."""
        pick = {
            "jugador": "Max Wiskandt",
            "superficie": "hard",  # ya real — distinto del edge_report ('clay')
            "tier": "?",
            "p_modelo": 0.5,
            "kelly_kl": 0.0,
        }

        _backfill_desde_edge(pick, self._edge_idx())

        assert pick["superficie"] == "hard", "no debe sobreescribir un valor no degradado"
        assert pick["tier"] == "itf", "sí debe completar el campo degradado"


# ═══════════════════════════════════════════════════════════════════════════════
# D87-11 — settle() exige que el rival también coincida (no solo el favorito)
# ═══════════════════════════════════════════════════════════════════════════════

class TestD87_11SettleExigeRival:

    @pytest.fixture
    def shadow_dir(self, tmp_path, monkeypatch):
        d = str(tmp_path / "shadow_book")
        monkeypatch.setattr(shadow_book, "SHADOW_DIR", d)
        return d

    def test_settle_matchea_contra_el_rival_correcto(self, shadow_dir):
        """Caso Nodo-86 §4.4: un jugador ('Carlos Alcaraz') con DOS partidos el mismo día
        (u homónimo) — el settle fallback por nombre debe elegir el resultado cuyo RIVAL
        también coincide, no el primero que matchee solo por el favorito."""
        fecha = "2026-07-11"
        path = os.path.join(shadow_dir, f"sb_{fecha}.jsonl")
        os.makedirs(shadow_dir, exist_ok=True)

        pick_snapshot = {
            "partido": "Carlos Alcaraz vs John Rival",
            "favorito_predicho": "Carlos Alcaraz",
            "cuota_favorito": 1.30,
            "cuota_rival": 3.20,
            "apostar": True,
            "match_id": None,
        }
        rec = {
            "sb_id": "sb_test_rival",
            "logged_at": f"{fecha}T04:00:00+00:00",
            # match_key deliberadamente NO usado como clave de resultados_map abajo,
            # fuerza el fallback tier-3 por nombre (Addendum B.2)
            "match_key": _match_key("Carlos Alcaraz", "John Rival"),
            "es_qualifying": False,
            "season_transition_flag": False,
            "pick_snapshot": pick_snapshot,
        }
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        # Dos resultados del mismo día, mismo favorito, rivales DISTINTOS.
        # El primero (rival equivocado) aparece primero en el diccionario —
        # pre-fix, el settle lo habría tomado por ser el primer match de nombre.
        resultados_map = {
            "resultado_rival_equivocado": {
                "p1": "Carlos Alcaraz", "p2": "Alex Michelsen",
                "ganador": "Alex Michelsen", "cuota_cierre": 1.60,
                "provenance": "flashscore_ref", "void": False, "match_id": None,
            },
            "resultado_rival_correcto": {
                "p1": "Carlos Alcaraz", "p2": "John Rival",
                "ganador": "Carlos Alcaraz", "cuota_cierre": 1.25,
                "provenance": "flashscore_ref", "void": False, "match_id": None,
            },
        }

        n = settle(fecha, resultados_map=resultados_map)
        assert n == 1

        with open(path, encoding="utf-8") as f:
            settled_rec = json.loads(f.readline())

        # Debe haber matcheado contra "resultado_rival_correcto" (cuota_cierre=1.25,
        # resultado=WON) — NO contra el de rival equivocado (que habría dado LOST).
        assert settled_rec["resolucion"]["resultado"] == "WON", (
            "D87-11: debe settlear contra el partido cuyo RIVAL también coincide"
        )
        assert settled_rec["resolucion"]["cuota_cierre"] == 1.25


# ═══════════════════════════════════════════════════════════════════════════════
# D87-03 — floor MIN_BET eliminado con EV<=0 o presupuesto agotado
# ═══════════════════════════════════════════════════════════════════════════════

class TestD87_03NoStakeFantasma:

    def test_kelly_cero_no_fuerza_min_bet(self):
        """EV negativo (p_blend*cuota-1<=0) → kelly=0 → stake debe ser 0,
        NO el floor de $1,000 que se forzaba antes del fix."""
        senal = {
            "partido": "Test A vs Test B",
            "favorito_predicho": "Test A",
            "p_modelo": 0.30,
            "cuota_favorito": 1.50,
            "n_h2h": 0,
            "p_historica_usada": 0.30,
            "edge_pct": "-5.0%",
            "kelly_kl": 0.0,
            "zona_cuota": "moderate_favorite",
            "superficie": "clay",
        }

        gastado, enriched = _print_individuales([senal], bankroll=100000, budget=40000)

        assert enriched[0]["stake"] == 0
        assert gastado == 0

    def test_budget_agotado_no_fuerza_min_bet(self):
        """kelly>0 (EV positivo) pero budget=0 → capped_stake<=0 → stake debe ser 0,
        NO el floor forzado por encima del presupuesto disponible."""
        senal = {
            "partido": "Test C vs Test D",
            "favorito_predicho": "Test C",
            "p_modelo": 0.75,
            "cuota_favorito": 2.00,
            "n_h2h": 20,
            "p_historica_usada": 0.70,
            "edge_pct": "25.0%",
            "kelly_kl": 0.20,
            "zona_cuota": "underdog",
            "superficie": "grass",
        }

        gastado, enriched = _print_individuales([senal], bankroll=100000, budget=0)

        assert enriched[0]["stake"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# D64-01 — señal RFI serializada en edge_report (acumulación automática H76-01)
# ═══════════════════════════════════════════════════════════════════════════════

class TestD64_01SenalRFI:

    def test_caso_michnev_rivera_produce_rfi_ultra(self):
        """Reproduce el caso semilla de Nodo-64: Michnev (inactivo 273d, favorito
        bookmaker @1.17) vs Rivera (activo 8d, underdog @4.35, favorecido por el modelo).

        jugador1=Rivera → mapea a form_decay_meta['p1']; jugador2=Michnev → mapea a 'p2'
        (calcular_edge_completo usa 'p1'/'p2' según cuál de los dos es jugador1/jugador2,
        no según quién es el favorito — ver edge_calculator.py bloque D64-01)."""
        form_decay_meta = {
            "p1": {"days_since": 8,   "fd": 1.00, "n": 10},   # Rivera — activo
            "p2": {"days_since": 273, "fd": 0.35, "n": 10},   # Michnev — inactivo
        }
        partido = _make_partido(
            jugador1="Leyton Rivera", jugador2="Petr Michnev",
            cuota1=4.35, cuota2=1.17,
            confidence=62.7, n_h2h=2, superficie="clay",
            form_decay_meta=form_decay_meta,
        )

        r = calcular_edge_completo(partido, CALIB_MINIMAL)

        assert r is not None
        assert r["rfi_days_inactive"] == 273
        assert r["rfi_tier"] == 2, "180<=273<365 → RFI-2"
        assert r["rfi_is_bookie_fav"] is True, "Michnev (inactivo) tiene la cuota más baja"
        assert r["rfi_model_picks_active"] is True, "el modelo favorece a Rivera (activo)"
        assert r["rfi_ultra"] is True, (
            "H76-01: rfi_tier>=2 + inactivo favorito bookmaker + modelo va al activo"
        )
        assert r["rfi_decay_gap"] > 1.5, "Rivera (fd=1.0) debe verse mucho más fresco que Michnev (fd=0.35)"

    def test_ambos_jugadores_frescos_no_activa_rfi_ultra(self):
        """Control: sin inactividad relevante (<90 días ambos), rfi_ultra debe ser False."""
        form_decay_meta = {
            "p1": {"days_since": 10, "fd": 1.00, "n": 10},
            "p2": {"days_since": 15, "fd": 1.00, "n": 10},
        }
        partido = _make_partido(
            jugador1="Player Fresh1", jugador2="Player Fresh2",
            cuota1=1.80, cuota2=2.00,
            confidence=58.0, n_h2h=2,
            form_decay_meta=form_decay_meta,
        )

        r = calcular_edge_completo(partido, CALIB_MINIMAL)

        assert r is not None
        assert r["rfi_tier"] == 0
        assert r["rfi_ultra"] is False
