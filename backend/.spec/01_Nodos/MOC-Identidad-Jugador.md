# MOC — Identidad de Jugador

> **Tipo:** Map of Content | **Creado:** 2026-07-16 (D105-02)
> **Wikilinks:** [[Nodo-72-Phantom-Identity-Guard]] | [[Nodo-80-Kambi-Name-Matching]] | [[Nodo-81-Settlement-Name-Normalize]] | [[Nodo-82-Kambi-Match-ID-Structural]]

---

## Nodos en este cluster

| Nodo | Tema | Estado |
|---|---|---|
| [[Nodo-72-Phantom-Identity-Guard]] | `_detect_phantom_identity()` — LOG_PLAYWRIGHT_CANDIDATE cuando ranking=None + n>20 + oldest>365d | completo |
| [[Nodo-80-Kambi-Name-Matching]] | `core/player_registry.py` — resolución canónica de nombres, fuzzy matching, alias table | completo |
| [[Nodo-81-Settlement-Name-Normalize]] | Phantom Guard activo — homónimos API vs Playwright, wikilink ↔ Nodo-82 | completo |
| [[Nodo-82-Kambi-Match-ID-Structural]] | Forma canónica definitiva: `player_registry` es la única fuente de verdad de IDs | completo |

## Relación con el sistema

- Alimenta: `edge_calculator.py` PASO 3 (necesita identidad resuelta para buscar H2H)
- Afecta: `extraer_historh2h.py` (Playwright PRIMARIO porque la API es vulnerable a homónimos)
- Gate: Phantom Identity bloqueó 3 errores de ruina antes de que llegaran a Kelly-KL
