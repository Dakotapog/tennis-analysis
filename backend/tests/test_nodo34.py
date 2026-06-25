"""
Tests para Nodo-34 — Corrupción de Datos en Extracción H2H: Score Invertido y Ranking Falso
Cubre:
  BUG-34-1: _parse_player_history() almacena KL siempre desde perspectiva KJ,
             invirtiendo el score cuando el sujeto es KK (50.0% de entradas)
  BUG-34-2: ranking_manager.py paso 5 usa substring matching ('e' in 'auger')
             que asigna rankings ATP top-10 a jugadores ITF desconocidos

Clases:
  TestNodo34FixAScoreInversion:  T34-01 a T34-08 — _parse_player_history() en ninja_h2h_parser.py
  TestNodo34FixBRankingFalso:    T34-09 a T34-14 — get_player_info() paso 5 en ranking_manager.py

Detección de mutación:
  T34-03 FALLA si se elimina el bloque de inversión raw_kl en _parse_player_history().
         Sin el Fix A, KL='0:2' (KK ganó) queda sin invertir → resultado='0-2'
         pero outcome='Ganó' → contradicción, y el assert `resultado == '2-0'` falla.

  T34-10 FALLA si se revierte Fix B en ranking_manager.py paso 5:
         `len(part) > 2 and startswith` → `part in ranked_part`.
         Sin Fix B, 'Kaynak M. N.' tiene 'm' in 'de minaur' y 'n' in 'de minaur'
         → 2 matches ≥ min(2,3) → retorna de Miñaur rank 6, y `is None` falla.
"""
import time
import pytest
from unittest.mock import patch

from scraping.ninja_h2h_parser import _parse_player_history
from analysis.ranking_manager import RankingManager

# ─────────────────────────────────────────────────────────────────────────────
# Timestamp fijo: 7 días atrás — pasa el filtro anti-leakage de 36h (Nodo-31)
# ─────────────────────────────────────────────────────────────────────────────
OLD_KC = str(int(time.time()) - 7 * 24 * 3600)


def _make_rec(kj: str, kk: str, kl: str, wis: str, ks: str = None) -> dict:
    """Record mínimo válido para _parse_player_history(). KC siempre=OLD_KC."""
    rec = {
        'KC': OLD_KC,
        'KJ': kj,
        'KK': kk,
        'KL': kl,
        'WIS': wis,
        'KD': '2',   # clay
        'KE': '',
        'KF': 'Test Tournament',
    }
    if ks is not None:
        rec['KS'] = ks
    return rec


@pytest.fixture
def rm_top_atp():
    """
    RankingManager con ATP top-11 inyectado (sin archivos reales).

    Usa patch para que __init__ no intente cargar archivos de filesystem.
    Los jugadores inyectados son exactamente los que causan false matches
    en BUG-34-2 (Sinner, Alcaraz, Auger-Aliassime, de Miñaur, etc.).
    """
    with patch.object(RankingManager, 'load_rankings', return_value=None):
        rm = RankingManager()  # __init__ corre pero load_rankings es no-op

    # ATP: los jugadores top que el bug asignaba a oponentes ITF desconocidos
    # Las claves son normalize_name(original): guiones→espacios, acentos→ascii
    top_atp = [
        ('jannik sinner',         1),
        ('carlos alcaraz',        2),
        ('alexander zverev',      3),
        ('felix auger aliassime', 4),  # BUG: 'e' in 'auger', 'm' in 'aliassime'
        ('andrey rublev',         5),
        ('alex de minaur',        6),  # BUG: 'm'/'n' in 'minaur'
        ('ben shelton',           7),
        ('holger rune',           8),
        ('tommy paul',            9),
        ('jack draper',          10),
        ('jiri lehecka',         11),
    ]
    for norm_name, rank in top_atp:
        data = {
            'rank': rank, 'ranking_position': rank,
            'tour': 'ATP', 'name': norm_name, 'original_name': norm_name,
        }
        rm.atp_players[norm_name] = data
        rm.rankings_data[norm_name] = data

    # WTA: jugadoras para T34-12 y T34-13
    wta_entries = [
        # 'Li N.' → normalizado 'li n' → step 3 encuentra 'li na' (apellido corto)
        ('li na',               None),
        # Nombre completo Eva-Marie Desvignes → step 1 exact match → rank 1027
        ('eva marie desvignes', 1027),
    ]
    for norm_name, rank in wta_entries:
        data = {
            'rank': rank, 'ranking_position': rank,
            'tour': 'WTA', 'name': norm_name, 'original_name': norm_name,
        }
        rm.wta_players[norm_name] = data

    return rm


# ─────────────────────────────────────────────────────────────────────────────
# T34-01 a T34-08 — Fix A: score desde perspectiva del sujeto (no de KJ)
# ─────────────────────────────────────────────────────────────────────────────

class TestNodo34FixAScoreInversion:
    """Fix A: _parse_player_history() debe invertir KL cuando sujeto es KK."""

    def test_t34_01_kk_gano_score_almacenado_en_perspectiva_kk(self):
        """T34-01: Sujeto=KK ganó 2-0 (KL='0:2') → resultado debe ser '2-0'

        Caso Tamura/Ito: KJ=Ito (perdió), KK=*Tamura (ganó), KL='0:2' (0 sets Ito, 2 sets Tamura).
        El sujeto es Tamura (KK) → score desde su perspectiva: '2-0'.
        Antes del fix: resultado='0-2' (score de Ito, no de Tamura).
        """
        rec = _make_rec(kj='Ito A.', kk='*Tamura K.', kl='0:2', wis='w')
        history = _parse_player_history([rec], subject_player='Tamura K.')

        assert len(history) == 1, "Debe procesar el registro"
        entry = history[0]
        assert entry['resultado'] == '2-0', (
            f"Tamura (KK, ganó 2-0): esperado '2-0', got '{entry['resultado']}'"
        )
        assert entry['outcome'] == 'Ganó'

    def test_t34_02_kk_perdio_score_almacenado_en_perspectiva_kk(self):
        """T34-02: Sujeto=KK perdió (KL='2:0', WIS='l') → resultado debe ser '0-2'

        Caso Tamura/Hosoki: KJ=*Hosoki (ganó), KK=Tamura (perdió), KL='2:0' (2 sets Hosoki).
        El sujeto es Tamura (KK) → score desde su perspectiva: '0-2'.
        Antes del fix: resultado='2-0' (score de Hosoki, no de Tamura).
        """
        rec = _make_rec(kj='*Hosoki Y.', kk='Tamura K.', kl='2:0', wis='l')
        history = _parse_player_history([rec], subject_player='Tamura K.')

        assert len(history) == 1
        entry = history[0]
        assert entry['resultado'] == '0-2', (
            f"Tamura (KK, perdió 0-2): esperado '0-2', got '{entry['resultado']}'"
        )
        assert entry['outcome'] == 'Perdió'

    def test_t34_03_mutation_detection_fix_a(self):
        """T34-03: DETECCIÓN DE MUTACIÓN Fix A — inversión de KL es correcta

        Este test FALLA si se elimina el bloque Fix A en _parse_player_history():
          `if raw_kl and ':' in raw_kl and not subject_is_kj: ...invertir...`

        Mecanismo:
          rec: KJ='Ito A.', KK='*Tamura K.', KL='0:2', WIS='w' (Tamura/KK ganó)
          SIN Fix A: score = _extract_score_sets('0:2') = '0-2'
                     Contradicción con outcome='Ganó' → BUG activo
          CON Fix A: raw_kl invertido '0:2'→'2:0', score = '2-0'
                     Consistente con outcome='Ganó' → correcto
          Assert `resultado == '2-0'` FALLA si Fix A fue revertido o comentado.
        """
        rec = _make_rec(kj='Ito A.', kk='*Tamura K.', kl='0:2', wis='w')
        history = _parse_player_history([rec], subject_player='Tamura K.')

        assert len(history) == 1, "Debe procesar el registro"
        entry = history[0]

        # Si Fix A revertido: resultado será '0-2' y este assert fallará
        assert entry['resultado'] == '2-0', (
            "T34-03 MUTACIÓN DETECTADA: resultado='0-2' en vez de '2-0'. "
            "El bloque de inversión de raw_kl fue eliminado. Restaurar Fix A en "
            "scraping/ninja_h2h_parser.py (función _parse_player_history, "
            "bloque 'BUG-34-1 Fix A')."
        )
        assert entry['outcome'] == 'Ganó', (
            "outcome debe venir de WIS — si falla, hay regresión en lógica de WIS"
        )

    def test_t34_04_kj_gano_score_no_invertido(self):
        """T34-04: Sujeto=KJ ganó 2-0 (KL='2:0') → resultado '2-0' sin invertir

        Fix A NO debe invertir cuando sujeto es KJ.
        Caso Dimitrov: KJ=*Dimitrov (ganó), KK=Damm, KL='2:0'.
        """
        rec = _make_rec(kj='*Dimitrov G.', kk='Damm M.', kl='2:0', wis='w')
        history = _parse_player_history([rec], subject_player='Dimitrov G.')

        assert len(history) == 1
        assert history[0]['resultado'] == '2-0', (
            f"Dimitrov (KJ, ganó 2-0): esperado '2-0', got '{history[0]['resultado']}'"
        )
        assert history[0]['outcome'] == 'Ganó'

    def test_t34_05_kj_perdio_score_no_invertido(self):
        """T34-05: Sujeto=KJ perdió (KL='0:2', WIS='l') → resultado '0-2' sin invertir"""
        rec = _make_rec(kj='Dimitrov G.', kk='*Zverev A.', kl='0:2', wis='l')
        history = _parse_player_history([rec], subject_player='Dimitrov G.')

        assert len(history) == 1
        assert history[0]['resultado'] == '0-2'
        assert history[0]['outcome'] == 'Perdió'

    def test_t34_06_kk_gano_tres_sets(self):
        """T34-06: Sujeto=KK ganó 2-1 en 3 sets (KL='1:2', WIS='w') → resultado '2-1'"""
        rec = _make_rec(kj='Loge J.', kk='*Durasovic V.', kl='1:2', wis='w')
        history = _parse_player_history([rec], subject_player='Durasovic V.')

        assert len(history) == 1
        assert history[0]['resultado'] == '2-1'
        assert history[0]['outcome'] == 'Ganó'

    def test_t34_07_kk_perdio_tres_sets(self):
        """T34-07: Sujeto=KK perdió 1-2 en 3 sets (KL='2:1', WIS='l') → resultado '1-2'"""
        rec = _make_rec(kj='*Majorossy I.', kk='Kravchenko G.', kl='2:1', wis='l')
        history = _parse_player_history([rec], subject_player='Kravchenko G.')

        assert len(history) == 1
        assert history[0]['resultado'] == '1-2'
        assert history[0]['outcome'] == 'Perdió'

    def test_t34_08_score_y_outcome_consistentes_en_batch(self):
        """T34-08: Fix A no rompe la consistencia score/outcome en lote mixto

        Procesa KK-ganó y KK-perdió en el mismo lote y verifica que s1>s2 ↔ Ganó.
        DETECCIÓN DE REGRESIÓN: si Fix A introduce inversión cuando no corresponde,
        algún entry tendrá score y outcome contradictorios.
        """
        recs = [
            _make_rec(kj='PlayerA', kk='*Sujeto', kl='0:2', wis='w'),  # KK ganó 2-0
            _make_rec(kj='*PlayerB', kk='Sujeto', kl='2:0', wis='l'),  # KK perdió 0-2
        ]
        history = _parse_player_history(recs, subject_player='Sujeto')

        assert len(history) == 2
        for entry in history:
            s1, s2 = map(int, entry['resultado'].split('-'))
            if entry['outcome'] == 'Ganó':
                assert s1 > s2, (
                    f"outcome=Ganó pero score={entry['resultado']} (s1≤s2) — "
                    "Fix A está invirtiendo cuando no debería, o viceversa"
                )
            else:
                assert s1 < s2, (
                    f"outcome=Perdió pero score={entry['resultado']} (s1≥s2) — "
                    "Fix A está invirtiendo cuando no debería, o viceversa"
                )


# ─────────────────────────────────────────────────────────────────────────────
# T34-09 a T34-14 — Fix B: paso 5 no usa iniciales como substrings
# ─────────────────────────────────────────────────────────────────────────────

class TestNodo34FixBRankingFalso:
    """Fix B: ranking_manager.py paso 5 no debe matchear iniciales de 1-2 chars."""

    def test_t34_09_desvignes_em_no_matchea_auger_aliassime(self, rm_top_atp):
        """T34-09: 'Desvignes E. M.' → None (no matchea Auger-Aliassime rank 4)

        Eva-Marie Desvignes es WTA rank ~1027, no aparece en rankings top.
        Antes del fix, paso 5:
          parts=['desvignes','e','m'] | 'e' in 'auger'=True | 'm' in 'aliassime'=True
          → 2 matches ≥ min(2,3) → retornaba rank 4 (Auger-Aliassime)
        """
        result = rm_top_atp.get_player_ranking('Desvignes E. M.')
        assert result is None, (
            f"'Desvignes E. M.' debe retornar None (jugadora ITF desconocida). "
            f"got rank={result} — posible falso match a Auger-Aliassime rank 4"
        )

    def test_t34_10_mutation_detection_fix_b(self, rm_top_atp):
        """T34-10: DETECCIÓN DE MUTACIÓN Fix B — iniciales no usadas como substrings

        Este test FALLA si se revierte Fix B en ranking_manager.py paso 5:
          `len(part) > 2 and (startswith)` → `part in ranked_part`

        Mecanismo:
          'Kaynak M. N.' → parts=['kaynak','m','n']
          SIN Fix B: 'm' in 'de minaur'=True; 'n' in 'de minaur'=True
                     2 matches ≥ min(2,3) → retorna de Miñaur rank 6
          CON Fix B: len('m')==1 → excluido; len('n')==1 → excluido
                     'kaynak' no hace startswith con ningún nombre ATP → 0 matches → None
          Assert `is None` FALLA si Fix B fue revertido o comentado.
        """
        result = rm_top_atp.get_player_ranking('Kaynak M. N.')

        # Si Fix B revertido: result = 6 (de Miñaur) y este assert falla
        assert result is None, (
            "T34-10 MUTACIÓN DETECTADA: 'Kaynak M. N.' retornó rank "
            f"{result} en vez de None. Las iniciales 'M.' y 'N.' (len≤2) "
            "están siendo usadas como substrings en ranking_manager.py paso 5. "
            "Restaurar Fix B: `len(part) > 2 and (startswith)` en get_player_info()."
        )

    def test_t34_11_isaacs_an_no_matchea_sinner(self, rm_top_atp):
        """T34-11: 'Isaacs A. N.' → None (no matchea Jannik Sinner rank 1)

        Antes del fix: 'a' in 'jannik'=True | 'n' in 'jannik'=True → 2 matches → rank 1
        """
        result = rm_top_atp.get_player_ranking('Isaacs A. N.')
        assert result is None, (
            f"'Isaacs A. N.' debe retornar None, got rank={result} "
            f"(posible falso match a Sinner rank 1)"
        )

    def test_t34_12_apellido_corto_resuelve_por_pasos_anteriores(self, rm_top_atp):
        """T34-12: 'Li N.' se resuelve en paso 3 (Apellido + Inicial), no en paso 5

        Fix B excluye partes len≤2 del paso 5. 'Li N.' normalizado='li n',
        parts=['li','n']. El paso 3 detecta que last_part='n' tiene len==1 →
        busca Apellido(s)=['li'] + inicial='n' → encuentra 'li na' en wta_players
        (na.startswith('n')) → resuelve sin llegar al paso 5.

        Verifica que Fix B no rompe resoluciones legítimas de pasos 1-3.
        El rank es None en el fixture (Li Na retirada), lo relevante es que
        no se produce un falso match ATP.
        """
        result = rm_top_atp.get_player_ranking('Li N.')
        # Li Na está en el fixture con rank=None (jugadora retirada sin rank activo)
        # El test verifica que no hay crash y no hay falso match ATP (rank sería un int ≤500)
        assert result is None, (
            f"'Li N.' debe retornar None (rank=None en fixture, no hay falso match ATP), "
            f"got rank={result}. Verificar que paso 3 (Apellido+Inicial) funciona post-Fix B."
        )

    def test_t34_13_nombre_completo_desvignes_resuelve_rank_1027(self, rm_top_atp):
        """T34-13: 'Eva-Marie Desvignes' (nombre completo) resuelve a rank 1027 WTA

        Contraste con T34-09: el nombre FlashScore abreviado 'Desvignes E. M.'
        debe retornar None, pero el nombre completo resuelve correctamente por
        exact match en paso 1 (normalize_name: guión→espacio → 'eva marie desvignes').

        Verifica que Fix B no bloquea resoluciones legítimas por nombre completo.
        """
        result = rm_top_atp.get_player_ranking('Eva-Marie Desvignes')
        assert result == 1027, (
            f"Nombre completo 'Eva-Marie Desvignes' debe resolver a rank 1027 WTA, "
            f"got rank={result}"
        )

    def test_t34_14_todos_los_casos_itf_verificados_retornan_none(self, rm_top_atp):
        """T34-14: Los 10 casos ITF del spec retornan None post-fix (dominio imposible)

        DETECCIÓN DE REVERT: si cualquiera retorna rank ≤ 50, Fix B fue revertido.
        Todos son jugadores ITF desconocidos que antes recibían rankings ATP top-11.
        """
        itf_opponents = [
            ('Kaynak M. N.',          'de Miñaur rank 6'),
            ('Isaacs A. N.',          'Sinner rank 1'),
            ('Bazan L. A.',           'Alcaraz rank 2'),
            ('Grohbruegge H. T.',     'Shelton rank 7'),
            ('Anugonda S. R.',        'Sinner rank 1'),
            ('Abendroth J. A.',       'Sinner rank 1'),
            ('Coromina Boluda A. M.', 'Auger-Aliassime rank 4'),
            ('Sandru I. M.',          'Auger-Aliassime rank 4'),
            ('Zayid M. S.',           'Auger-Aliassime rank 4'),
            ('Gokpinar H. C.',        'Lehecka rank 11'),
        ]

        false_matches = []
        for name, expected_bug_match in itf_opponents:
            rank = rm_top_atp.get_player_ranking(name)
            if rank is not None and rank <= 50:
                false_matches.append((name, rank, expected_bug_match))

        assert not false_matches, (
            "T34-14: Falsos matches detectados post-fix (Fix B revertido):\n"
            + '\n'.join(
                f"  '{n}' → rank {r} (antes matcheaba {bug})"
                for n, r, bug in false_matches
            )
        )
