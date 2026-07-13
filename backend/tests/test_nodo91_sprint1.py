"""tests/test_nodo91_sprint1.py — REGLA-T53 Sprint 1 (Nodo-91)

Cada test invoca la función real del módulo; nunca reimplementa la fórmula.
Baseline: 1827 tests passed antes de este archivo.
"""
import pytest
from unittest.mock import patch
from datetime import datetime, timedelta
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# S1-E — Multi-iniciales: get_player_info() paso 3 (D90-02)
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiIniciales:
    """Inyectamos un dict sintético en atp_players para aislar el paso 3."""

    def _make_manager(self, synthetic_dict: dict):
        """Crea un RankingManager con atp_players sintético sin tocar disco."""
        from analysis.ranking_manager import RankingManager
        rm = RankingManager.__new__(RankingManager)
        rm.atp_players = synthetic_dict
        rm.wta_players = {}
        rm.combined_players = {}
        return rm

    def test_multi_iniciales_hsu(self):
        """'Hsu Y. H.' debe encontrar 'hsu yu hsiou' con 2 iniciales. (D90-02)"""
        rm = self._make_manager({'hsu yu hsiou': {'name': 'Hsu Yu Hsiou', 'ranking': 150}})
        result = rm.get_player_info('Hsu Y. H.')
        assert result is not None
        assert result['name'] == 'Hsu Yu Hsiou'

    def test_multi_iniciales_burruchaga(self):
        """'Burruchaga R. A.' debe encontrar 'burruchaga roman andres'. (D90-02)"""
        rm = self._make_manager({'burruchaga roman andres': {'name': 'Burruchaga Roman Andres', 'ranking': 300}})
        result = rm.get_player_info('Burruchaga R. A.')
        assert result is not None
        assert result['name'] == 'Burruchaga Roman Andres'

    def test_una_inicial_sigue_funcionando(self):
        """Regresión: 'Cerundolo F.' (1 inicial) sigue funcionando tras el cambio. (D90-02)"""
        rm = self._make_manager({'cerundolo francisco': {'name': 'Cerundolo Francisco', 'ranking': 35}})
        result = rm.get_player_info('Cerundolo F.')
        assert result is not None
        assert result['name'] == 'Cerundolo Francisco'


# ─────────────────────────────────────────────────────────────────────────────
# S1-A — evaluar_capa2() (D90-04 / H89-01)
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluarCapa2:
    """Tests para la función pura evaluar_capa2() añadida en edge_calculator.py."""

    @pytest.fixture(autouse=True)
    def import_fn(self):
        from edge_calculator import evaluar_capa2
        self.evaluar_capa2 = evaluar_capa2

    def test_capa2_rechaza_no_data(self):
        """Picks con status NO_DATA nunca son CAPA 2."""
        from core.data_contract import PICK_STATUS_NO_DATA
        resultado = {'status': PICK_STATUS_NO_DATA}
        assert self.evaluar_capa2(resultado, 0.65, 2.0, 3) is False

    def test_capa2_rechaza_phantom(self):
        """Picks con phantom_data=True nunca son CAPA 2 (Nodo-72)."""
        resultado = {'phantom_data': True}
        assert self.evaluar_capa2(resultado, 0.65, 2.0, 3) is False

    def test_capa2_rechaza_hot_sin_bbi(self):
        """HOT + BBI < 0.50 bloquea CAPA 2 (re-check gate N28F2)."""
        resultado = {'markov_favorito': 'HOT', 'bbi': 0.3}
        assert self.evaluar_capa2(resultado, 0.65, 2.0, 3) is False

    def test_capa2_rechaza_p_bajo(self):
        """p_modelo < 0.60 no alcanza el gate de confianza."""
        resultado = {}
        assert self.evaluar_capa2(resultado, 0.58, 2.0, 3) is False

    def test_capa2_rechaza_sin_h2h(self):
        """n_h2h=0 falla T33 (n_h2h >= 1 requerido)."""
        resultado = {}
        assert self.evaluar_capa2(resultado, 0.62, 2.0, 0) is False

    def test_capa2_rechaza_cuota_hf1(self):
        """Cuota fuera del rango [1.50, 2.80] rechazada — HF-1 y tope superior."""
        resultado = {}
        assert self.evaluar_capa2(resultado, 0.62, 1.40, 2) is False  # < 1.50
        assert self.evaluar_capa2(resultado, 0.62, 3.10, 2) is False  # > 2.80

    def test_capa2_acepta_caso_valido(self):
        """Caso limpio: p=0.62, cuota 2.0, n_h2h=2 → True."""
        resultado = {}
        assert self.evaluar_capa2(resultado, 0.62, 2.0, 2) is True

    def test_capa2_no_duplica_capa1(self):
        """Si apostar=True (ya es CAPA 1) → evaluar_capa2 retorna False."""
        resultado = {'apostar': True}
        assert self.evaluar_capa2(resultado, 0.65, 2.0, 3) is False


# ─────────────────────────────────────────────────────────────────────────────
# S1-A — elo_dominance_axis (D90-10 observacional)
# ─────────────────────────────────────────────────────────────────────────────

class TestEloDominance:
    """
    elo_dominance_axis = True cuando:
      - elo_favorito - elo_rival > 50
      - ranking_favorito > ranking_rival  (número peor = más alto, Nodo-91 §S1-A)
    """

    def _resultado_elo(self, elo_f, elo_r, rk_f, rk_r):
        return {
            'elo_favorito': elo_f,
            'elo_rival': elo_r,
            'ranking_favorito': rk_f,
            'ranking_rival': rk_r,
        }

    def _get_axis_from_edge(self, resultado):
        """Invoca edge_calculator para calcular elo_dominance_axis."""
        from edge_calculator import _calc_elo_dominance_axis
        return _calc_elo_dominance_axis(resultado)

    def test_elo_dominance_activo(self):
        """ELO gap > 50 y ranking_favorito > ranking_rival → True."""
        r = self._resultado_elo(elo_f=1560, elo_r=1464, rk_f=1188, rk_r=200)
        assert self._get_axis_from_edge(r) is True

    def test_elo_dominance_inactivo_sin_gap(self):
        """ELO gap ≤ 50 → False aunque ranking difiera."""
        r = self._resultado_elo(elo_f=1520, elo_r=1490, rk_f=1188, rk_r=200)
        assert self._get_axis_from_edge(r) is False

    def test_elo_dominance_inactivo_ranking_consistente(self):
        """Favorito con mejor ELO Y mejor ranking → False (no hay anomalía)."""
        r = self._resultado_elo(elo_f=1560, elo_r=1464, rk_f=50, rk_r=200)
        assert self._get_axis_from_edge(r) is False


# ─────────────────────────────────────────────────────────────────────────────
# S1-B — _pool_capa2() en trader_ev_tenis.py
# ─────────────────────────────────────────────────────────────────────────────

class TestPoolCapa2:
    """Tests para la función pura _pool_capa2(watchlist, sin_edge)."""

    @pytest.fixture(autouse=True)
    def import_fn(self):
        from trader_ev_tenis import _pool_capa2
        self._pool_capa2 = _pool_capa2

    def test_pool_capa2_filtra_flag(self):
        """Solo picks con capa2_candidate=True aparecen en el pool."""
        watchlist = [
            {'jugador': 'A', 'capa2_candidate': True},
            {'jugador': 'B', 'capa2_candidate': False},
        ]
        sin_edge = [
            {'jugador': 'C', 'capa2_candidate': True},
            {'jugador': 'D'},
        ]
        pool = self._pool_capa2(watchlist, sin_edge)
        jugadores = [p['jugador'] for p in pool]
        assert 'A' in jugadores
        assert 'C' in jugadores
        assert 'B' not in jugadores
        assert 'D' not in jugadores

    def test_pool_capa2_vacio_sin_candidatos(self):
        """Pool vacío cuando ningún pick tiene capa2_candidate=True."""
        pool = self._pool_capa2([], [{'jugador': 'X'}])
        assert pool == []


# ─────────────────────────────────────────────────────────────────────────────
# S1-D — _planes_frescos() en betplay_combo_builder.py
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanesFrescos:
    """Test para la función pura _planes_frescos(paths, max_age_h)."""

    @pytest.fixture(autouse=True)
    def import_fn(self):
        from betplay_combo_builder import _planes_frescos
        self._planes_frescos = _planes_frescos

    def _fake_path(self, hours_ago: float) -> Path:
        """Crea un Path con stem trader_plan_YYYYMMDD_HHMMSS con la antigüedad indicada."""
        ts = datetime.now() - timedelta(hours=hours_ago)
        stem = f"trader_plan_{ts.strftime('%Y%m%d_%H%M%S')}"
        p = Path(f"/tmp/{stem}.json")
        return p

    def test_planes_frescos_corta_4h(self):
        """Plan de hace 5h se descarta; plan de hace 1h se conserva."""
        p_5h = self._fake_path(5.0)
        p_1h = self._fake_path(1.0)
        result = self._planes_frescos([p_5h, p_1h], max_age_h=4)
        assert p_1h in result
        assert p_5h not in result

    def test_planes_frescos_todos_viejos(self):
        """Si todos los planes son > max_age_h → lista vacía."""
        p_old = self._fake_path(10.0)
        result = self._planes_frescos([p_old], max_age_h=4)
        assert result == []

    def test_planes_frescos_todos_frescos(self):
        """Si todos los planes son recientes → todos pasan."""
        p1 = self._fake_path(0.5)
        p2 = self._fake_path(1.5)
        result = self._planes_frescos([p1, p2], max_age_h=4)
        assert len(result) == 2
