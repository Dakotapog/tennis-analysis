"""
core/player_registry.py — Nodo-51 F0: Registro Canónico de Jugadores

Entity resolution layer: todo componente del pipeline resuelve identidad de
jugador a través de esta capa — nunca mediante strings ad-hoc.

Arquitectura (MM-3, patrón sistema inmune dos capas):

  Fast path O(1):
    1. Direct key lookup — alias ya normalizado en tabla
    2. Reversed key lookup — absorbe el fix de Nodo-47 (glinka daniil vs daniil glinka)

  Slow path O(n):
    3. get_player_info() del RankingManager (fuzzy matching inteligente)
       → En hit: registra alias en tabla (memoria inmune) → próxima vez O(1)

  Provenance tracking:
    Cada canonical_id tiene provenance: 'atp_file' | 'kambi_estimate' | 'unknown'
    Esto es la semilla del sistema de procedencia por campo (C2, F2).

Invariante de seguridad:
  is_in_atp_file(x) == True  ↔  x está en el archivo ATP/WTA real.
  register_kambi_estimate(x) → provenance='kambi_estimate', is_in_atp_file → False.
  Nunca se puede confundir una fuente con la otra.

NOTA SOBRE NORMALIZACIÓN:
  normalize_player_name() replica la lógica de RankingManager.normalize_name().
  Si normalize_name() cambia en ranking_manager.py, actualizar aquí también.
  B108-03 (F0-DEUDA CERRADO 2026-07-17): RankingManager.normalize_name(),
  kambi_tennis._normalize_name() y betslip_registrar._match_stake() ya delegan aquí.
"""
import json
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZACIÓN — fuente canónica para F0
# ══════════════════════════════════════════════════════════════════════════════

def normalize_player_name(name: str) -> str:
    """
    Normalización canónica de nombre de jugador.
    Replica exactamente RankingManager.normalize_name() para garantizar que
    las claves del registry coincidan con las de rankings_data.

    Si esta función diverge de normalize_name(), el fast path fallará silenciosamente.
    """
    if not name:
        return ""
    normalized = name.lower().strip()
    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u',
        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
        'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',
        'ñ': 'n', 'ç': 'c'
    }
    for char, replacement in replacements.items():
        normalized = normalized.replace(char, replacement)
    normalized = re.sub(r'[+\-]\d+$', '', normalized)
    normalized = normalized.replace('-', ' ')
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = ' '.join(normalized.split())
    return normalized


# ══════════════════════════════════════════════════════════════════════════════
# PLAYER REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

_PROVENANCE_ATP     = 'atp_file'
_PROVENANCE_KAMBI   = 'kambi_estimate'
_PROVENANCE_UNKNOWN = 'unknown'

# ── Crosswalk (Nodo-118 F2) ───────────────────────────────────────────────────
_CROSSWALK_FILE   = Path("data/player_crosswalk.json")
_CONF_HIERARCHY   = {"MANUAL": 3, "VERIFIED": 2, "AUTO": 1}  # mayor = más confiable


class PlayerRegistry:
    """
    Registro canónico de jugadores con resolución de identidad en dos capas.

    Parámetros:
        normalize_fn: Callable — función de normalización de nombres.
                      En producción: ranking_manager.normalize_name.
                      Si None: usa normalize_player_name() del módulo.
        ranking_manager: instancia de RankingManager para el slow path.
                         Si None: solo opera con el fast path.

    Uso típico en NinjaH2HExtractor:
        self._player_registry = PlayerRegistry(
            normalize_fn=self.ranking_manager.normalize_name,
            ranking_manager=self.ranking_manager,
        )

    Uso en tests:
        registry = PlayerRegistry(normalize_fn=normalize_player_name)
        # bootstrap manual:
        registry._bootstrap_from_dict(synthetic_rankings_data)
    """

    def __init__(
        self,
        normalize_fn: Optional[Callable[[str], str]] = None,
        ranking_manager=None,
    ):
        self._normalize_fn: Callable[[str], str] = normalize_fn or normalize_player_name
        self._ranking_manager = ranking_manager

        # alias normalizado → canonical_id
        self._alias_to_cid: Dict[str, str] = {}

        # canonical_id → provenance ('atp_file' | 'kambi_estimate')
        self._cid_to_provenance: Dict[str, str] = {}

        # id(info_dict) → canonical_id  — permite slow path sin búsqueda lineal
        # (id() funciona porque get_player_info devuelve el MISMO objeto que está
        # en rankings_data, no una copia)
        self._info_id_to_cid: Dict[int, str] = {}

        if ranking_manager is not None:
            self._bootstrap(ranking_manager)

        # ── Crosswalk (Nodo-118 F2) ───────────────────────────────────────────
        # {canonical_id: {canonical, aliases:{norm_alias: {source, confidence, added}},
        #                 last_seen}}
        self._xwalk: Dict[str, dict] = {}
        # Índice plano norm_alias → canonical_id para O(1) lookup
        self._xwalk_alias_to_cid: Dict[str, str] = {}
        self._load_crosswalk()

    # ── Bootstrap ────────────────────────────────────────────────────────────

    def _bootstrap(self, ranking_manager) -> None:
        """
        Carga todos los jugadores de rankings_data en la alias table.
        La clave de rankings_data ya está normalizada (es el canonical_id).
        """
        for normalized_key, info in ranking_manager.rankings_data.items():
            self._register_atp_entry(normalized_key, info)

        logger.debug(
            f"[PlayerRegistry] Bootstrap: {len(self._alias_to_cid)} aliases "
            f"desde {len(self._cid_to_provenance)} jugadores ATP/WTA."
        )

    def _register_atp_entry(self, canonical_key: str, info: dict) -> None:
        """Registra una entrada ATP/WTA en la alias table y el id mapping."""
        self._alias_to_cid[canonical_key] = canonical_key
        self._cid_to_provenance[canonical_key] = _PROVENANCE_ATP
        self._info_id_to_cid[id(info)] = canonical_key

    # ── Resolución ───────────────────────────────────────────────────────────

    def _normalize(self, name: str) -> str:
        return self._normalize_fn(name)

    def resolve(self, player_name: str) -> Optional[str]:
        """
        Retorna canonical_id para player_name, o None si no se encuentra.

        Capas de búsqueda (en orden de costo creciente):
        1. Direct alias lookup O(1)
        2. Reversed-key lookup O(1)  ← absorbe el bug de Nodo-47
        3. Fuzzy matching via ranking_manager.get_player_info()  ← slow path
           → En hit: registra alias para futuras llamadas (memoria inmune)
        """
        if not player_name:
            return None

        normalized = self._normalize(player_name)

        # ── Capa 1: direct lookup ──────────────────────────────────────────
        if normalized in self._alias_to_cid:
            return self._alias_to_cid[normalized]

        # ── Capa 2: reversed key (Nodo-47 fix generalizado) ───────────────
        # ATP indexa "Apellido Nombre"; Kambi/FlashScore usan "Nombre Apellido".
        # Para nombres de 2 tokens: probar la inversión en O(1).
        parts = normalized.split()
        if len(parts) == 2:
            reversed_key = f"{parts[1]} {parts[0]}"
            if reversed_key in self._alias_to_cid:
                cid = self._alias_to_cid[reversed_key]
                # Memoria inmune: registrar alias normalizado para próxima vez
                self._alias_to_cid[normalized] = cid
                logger.debug(
                    f"[PlayerRegistry] Reversed-key hit: '{player_name}' "
                    f"→ '{reversed_key}' → cid='{cid}'"
                )
                return cid

        # ── Capa 3: slow path — fuzzy matching ────────────────────────────
        # Para nombres con birth year (Watanuki Yosuke (1998) → key de 3 tokens)
        # o apellidos compuestos (Davidovich Fokina), donde la inversión simple falla.
        if self._ranking_manager is not None:
            info = self._ranking_manager.get_player_info(player_name)
            if info is not None:
                cid = self._info_id_to_cid.get(id(info))
                if cid:
                    # Memoria inmune: registrar alias para próxima vez → O(1)
                    self._alias_to_cid[normalized] = cid
                    logger.debug(
                        f"[PlayerRegistry] Slow path hit: '{player_name}' "
                        f"→ cid='{cid}' (alias registrado)"
                    )
                    return cid

        logger.debug(
            f"[PlayerRegistry] No encontrado: '{player_name}' "
            f"(normalized='{normalized}')"
        )
        return None

    # ── API pública ──────────────────────────────────────────────────────────

    def is_in_atp_file(self, player_name: str) -> bool:
        """
        True si el jugador está en el archivo ATP/WTA real.
        False si es desconocido o solo tiene estimate de Kambi.

        Usado por _inject_kambi_ranking para el guard de no-sobreescritura.
        """
        cid = self.resolve(player_name)
        if cid is None:
            return False
        return self._cid_to_provenance.get(cid) == _PROVENANCE_ATP

    def provenance(self, player_name: str) -> str:
        """
        Retorna la provenance del jugador:
          'atp_file'       — dato real del archivo ATP/WTA
          'kambi_estimate' — estimado inyectado desde ranking Kambi
          'unknown'        — no encontrado en ninguna fuente
        """
        cid = self.resolve(player_name)
        if cid is None:
            return _PROVENANCE_UNKNOWN
        return self._cid_to_provenance.get(cid, _PROVENANCE_UNKNOWN)

    def register_kambi_estimate(self, player_name: str) -> str:
        """
        Registra un jugador ITF/desconocido que Kambi conoce pero ATP no indexa.
        Devuelve canonical_id. Provenance='kambi_estimate' — is_in_atp_file → False.

        Llamado por _inject_kambi_ranking DESPUÉS de inyectar en rankings_data.
        Garantiza que futuras llamadas a resolve() sean O(1).
        """
        normalized = self._normalize(player_name)
        if normalized not in self._alias_to_cid:
            cid = normalized
            self._alias_to_cid[normalized] = cid
            self._cid_to_provenance[cid] = _PROVENANCE_KAMBI
            logger.debug(
                f"[PlayerRegistry] Kambi estimate registrado: '{player_name}' "
                f"→ cid='{cid}'"
            )
        return self._alias_to_cid[normalized]

    def stats(self) -> dict:
        """Estadísticas del registry — útil para logging y debugging."""
        atp_count = sum(
            1 for p in self._cid_to_provenance.values() if p == _PROVENANCE_ATP
        )
        kambi_count = sum(
            1 for p in self._cid_to_provenance.values() if p == _PROVENANCE_KAMBI
        )
        return {
            "total_aliases":   len(self._alias_to_cid),
            "atp_file":        atp_count,
            "kambi_estimate":  kambi_count,
            "immune_memory":   len(self._alias_to_cid) - len(self._cid_to_provenance),
        }

    # ── Crosswalk — Nodo-118 F2 ──────────────────────────────────────────────

    def _load_crosswalk(self, path: Path = None) -> None:
        """Carga data/player_crosswalk.json en memoria. No-op si no existe."""
        p = path or _CROSSWALK_FILE
        if not p.exists():
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._xwalk = data.get("entries", {})
            # Reconstruir índice plano
            self._xwalk_alias_to_cid = {}
            for cid, entry in self._xwalk.items():
                for alias_norm in entry.get("aliases", {}):
                    self._xwalk_alias_to_cid[alias_norm] = cid
            logger.debug(
                f"[Crosswalk] Cargado: {len(self._xwalk)} canónicos, "
                f"{len(self._xwalk_alias_to_cid)} aliases"
            )
        except Exception as e:
            logger.warning(f"[Crosswalk] Error cargando {p}: {e}")

    def _save_crosswalk(self, path: Path = None) -> None:
        """Persiste el crosswalk en disco."""
        p = path or _CROSSWALK_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"entries": self._xwalk}, f, ensure_ascii=False, indent=2)

    def add_alias(
        self,
        canonical_id: str,
        alias: str,
        source: str,
        confidence: str,
        crosswalk_path: Path = None,
    ) -> None:
        """
        Registra un alias en el crosswalk y lo persiste.

        Jerarquía: MANUAL > VERIFIED > AUTO.
        - MANUAL nunca se sobreescribe (protección contra automatismos).
        - VERIFIED sobreescribe AUTO.
        - AUTO no sobreescribe VERIFIED ni MANUAL.
        - Acumula: un canonical_id puede tener múltiples aliases de distintas fuentes.

        Args:
            canonical_id: forma canónica del jugador (ej. "paula badosa")
            alias: alias a registrar (ej. "P. Badosa", "Badosa P.")
            source: "kambi" | "flashscore" | "atp" | "manual"
            confidence: "MANUAL" | "VERIFIED" | "AUTO"
        """
        canonical_id = self._normalize(canonical_id)
        alias_norm = self._normalize(alias)
        if not canonical_id or not alias_norm:
            return

        new_level = _CONF_HIERARCHY.get(confidence, 0)

        # Inicializar entrada si no existe
        if canonical_id not in self._xwalk:
            self._xwalk[canonical_id] = {
                "canonical": canonical_id,
                "aliases": {},
                "last_seen": datetime.now().strftime("%Y-%m-%d"),
            }

        entry = self._xwalk[canonical_id]
        existing = entry["aliases"].get(alias_norm)

        if existing:
            existing_level = _CONF_HIERARCHY.get(existing.get("confidence", "AUTO"), 0)
            # MANUAL nunca se sobreescribe; tampoco se baja de nivel
            if existing_level >= new_level:
                logger.debug(
                    f"[Crosswalk] add_alias: '{alias_norm}' ya existe con "
                    f"{existing.get('confidence')} ≥ {confidence} — no se sobreescribe"
                )
                return

        entry["aliases"][alias_norm] = {
            "source": source,
            "confidence": confidence,
            "added": datetime.now().strftime("%Y-%m-%d"),
        }
        entry["last_seen"] = datetime.now().strftime("%Y-%m-%d")

        # Actualizar índice plano
        self._xwalk_alias_to_cid[alias_norm] = canonical_id

        # También registrar en _alias_to_cid para que resolve() lo encuentre en O(1)
        if alias_norm not in self._alias_to_cid:
            self._alias_to_cid[alias_norm] = canonical_id

        self._save_crosswalk(crosswalk_path)
        logger.debug(
            f"[Crosswalk] add_alias: '{alias}' → '{canonical_id}' "
            f"({source}, {confidence})"
        )

    def resolve_crosswalk(self, name: str) -> Optional[str]:
        """
        Resuelve identidad consultando PRIMERO el crosswalk (aliases persistidos),
        luego cae al resolve() normal (ATP/WTA + fuzzy).

        Retorna canonical_id o None.
        """
        if not name:
            return None
        alias_norm = self._normalize(name)
        # Capa 0: crosswalk lookup O(1) — antes del fuzzy
        if alias_norm in self._xwalk_alias_to_cid:
            return self._xwalk_alias_to_cid[alias_norm]
        # Capa 1-3: resolve() existente
        return self.resolve(name)

    def crosswalk_stats(self) -> dict:
        """Estadísticas del crosswalk para el embudo Nodo-118 §5."""
        total_aliases = sum(
            len(e.get("aliases", {})) for e in self._xwalk.values()
        )
        by_confidence = {}
        for entry in self._xwalk.values():
            for a in entry.get("aliases", {}).values():
                c = a.get("confidence", "AUTO")
                by_confidence[c] = by_confidence.get(c, 0) + 1
        return {
            "canonicals": len(self._xwalk),
            "total_aliases": total_aliases,
            "by_confidence": by_confidence,
            "index_size": len(self._xwalk_alias_to_cid),
        }
