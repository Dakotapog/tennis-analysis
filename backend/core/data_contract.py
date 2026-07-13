"""
core/data_contract.py — Nodo-51 F2: Contrato de Completitud + Procedencia

Implementa MM-1 (Inversión de Jacobi/Munger): en vez de más fallbacks,
garantizar que NUNCA se calcule un edge sobre historial EMPTY.

La regla es dura: si history_provenance es EMPTY para cualquier jugador,
el pick sale con status='NO_DATA' y edge=None — nunca p_modelo=0.500.
Un coin-flip por ignorancia no es una probabilidad.

Conexión con C2: `cuota_es_real` (Nodo-48) es la semilla de este sistema.
Cada campo crítico lleva su procedencia — el trader deja de adivinar la
calidad del dato: la lee.

Campos de procedencia:
    history_provenance   ninja_api | thf_cache | playwright_dom | EMPTY
    ranking_provenance   atp_file  | kambi_estimate
    odds_provenance      kambi_live | flashscore_ref   (cuota_es_real compat.)
"""

# ─── Procedencia de historial de partidos ───────────────────────────────────
PROVENANCE_NINJA_API      = 'ninja_api'       # FlashScore Ninja H2H API (fast path)
PROVENANCE_THF_CACHE      = 'thf_cache'       # Temporal History Fallback (Nodo-45)
PROVENANCE_PLAYWRIGHT_DOM = 'playwright_dom'  # Playwright DOM scraping (slow path)
PROVENANCE_EMPTY          = 'EMPTY'           # Sin historial en ninguna fuente

VALID_HISTORY_PROVENANCES = frozenset({
    PROVENANCE_NINJA_API,
    PROVENANCE_THF_CACHE,
    PROVENANCE_PLAYWRIGHT_DOM,
    PROVENANCE_EMPTY,
})

# ─── Procedencia de ranking ──────────────────────────────────────────────────
RANK_PROVENANCE_ATP_FILE       = 'atp_file'        # Archivo ATP/WTA real
RANK_PROVENANCE_KAMBI_ESTIMATE = 'kambi_estimate'  # Estimado desde Kambi (Nodo-47/F0)

# ─── Procedencia de cuotas ───────────────────────────────────────────────────
ODDS_PROVENANCE_KAMBI_LIVE    = 'kambi_live'      # Cuotas reales Betplay/Kambi
ODDS_PROVENANCE_FLASHSCORE_REF = 'flashscore_ref' # Referencia FlashScore (no real)

# ─── Status de pick ──────────────────────────────────────────────────────────
PICK_STATUS_NO_DATA = 'NO_DATA'  # Historial EMPTY — excluido de todos los pools


# ═══════════════════════════════════════════════════════════════════════════════
# API PÚBLICA
# ═══════════════════════════════════════════════════════════════════════════════

def has_empty_history(partido_or_pick: dict) -> bool:
    """
    True si algún jugador tiene historia EMPTY — pick no puede tener edge.

    Acepta dos tipos de dict:
      - Partido dict (H2H JSON): lee `ranking_analysis.prediction.historial_incompleto`
      - Pick dict (edge report):  lee `status == 'NO_DATA'`

    El campo `historial_incompleto` lo produce rivalry_analyzer (Nodo-35):
        {'p1': len(player1_history) == 0, 'p2': len(player2_history) == 0}

    FALLA DE MUTACIÓN: si se elimina la lectura de historial_incompleto,
    picks con historial vacío pasarán al pool como si tuvieran datos.
    """
    # Caso 1: pick ya procesado por edge_calculator con status explícito
    if partido_or_pick.get('status') == PICK_STATUS_NO_DATA:
        return True

    # Caso 2: partido dict del H2H JSON (antes de edge_calculator)
    pred = (
        partido_or_pick
        .get('ranking_analysis', {})
        .get('prediction', {})
    )
    hi = pred.get('historial_incompleto', {})
    return bool(hi.get('p1')) or bool(hi.get('p2'))


def completeness_score(partido_or_pick: dict) -> float:
    """
    Score de completitud de datos: [0.0, 1.0].

    0.0 → algún jugador no tiene historial. Edge PROHIBIDO.
    1.0 → ambos jugadores tienen datos.

    Invariante: completeness_score == 0.0 ↔ has_empty_history == True.

    En fases futuras puede tener valores intermedios según calidad de
    fuente (ninja_api > thf_cache > playwright_dom). Por ahora: binario.
    """
    if has_empty_history(partido_or_pick):
        return 0.0
    return 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# C1 Nodo-67: DataContract v2 — schema por artefacto
# Cierra las 6 fronteras de Nodo-86 §4.1 con UN mecanismo.
# ═══════════════════════════════════════════════════════════════════════════════

class DataContractViolation(Exception):
    """Fallo de contrato de artefacto — fail-loud (no silencioso)."""


# Schemas: cada entrada define las claves requeridas en el top-level del artefacto.
# 'required'       — claves obligatorias en el objeto raíz
# 'pick_required'  — claves obligatorias en cada pick de la lista indicada
# 'list_key'       — nombre de la lista de picks (para pick_required)
ARTIFACT_SCHEMAS: dict = {
    # Frontera 1: edge_calculator → trader_ev_tenis
    'edge_report': {
        'required': ['metadata', 'apostar'],
    },
    # Frontera 2: trader_ev_tenis → combo_governor / dashboard
    'trader_plan': {
        'required': ['metadata', 'individuales'],
    },
    # Frontera 3: bookmarklet → betslip_registrar --listen
    'betslip_index': {
        'required': ['ts', 'index'],
    },
    # Frontera 4: betslip_registrar --listen → --cerrar
    'apuestas': {
        'required': ['estado', 'picks', 'ts_registro'],
        'list_key': 'picks',
        'pick_required': ['jugador', 'cuota', 'outcome_id'],
    },
    # Frontera 5: shadow_book --settle → report_dict / dashboard
    'sb_jsonl_pick': {
        'required': ['sb_id', 'partido', 'pick_snapshot'],
    },
    # Frontera 6 (I3 Nodo-67): combo_confianza_builder → combo_governor
    'combo_plan_json': {
        'required': ['fecha', 'bankroll', 'budget', 'cobertura'],
    },
}


def validate_artifact(name: str, obj: dict) -> bool:
    """
    Valida que `obj` cumple el schema registrado para `name`.
    Lanza DataContractViolation si falla (fail-loud — no retorna False en silencio).
    Retorna True si el artefacto es válido.

    Uso:
        from core.data_contract import validate_artifact
        validate_artifact('edge_report', edge_data)   # lanza si falta 'apostar'
    """
    schema = ARTIFACT_SCHEMAS.get(name)
    if schema is None:
        raise DataContractViolation(
            f"Artefacto '{name}' no registrado en ARTIFACT_SCHEMAS. "
            f"Añadir schema en core/data_contract.py antes de consumir."
        )

    # Verificar claves requeridas en raíz
    missing = [k for k in schema.get('required', []) if k not in obj]
    if missing:
        raise DataContractViolation(
            f"[{name}] Claves requeridas ausentes: {missing}. "
            f"Artefacto recibido con claves: {list(obj.keys())}"
        )

    # Verificar claves requeridas en cada pick
    list_key = schema.get('list_key')
    pick_required = schema.get('pick_required', [])
    if list_key and pick_required:
        picks = obj.get(list_key, [])
        for i, pick in enumerate(picks):
            if not isinstance(pick, dict):
                continue
            missing_pick = [k for k in pick_required if k not in pick]
            if missing_pick:
                raise DataContractViolation(
                    f"[{name}] Pick [{i}] ({pick.get('jugador', pick.get('partido', '?'))}) "
                    f"faltan campos: {missing_pick}"
                )

    return True
