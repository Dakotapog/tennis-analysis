# PRE_IMPLEMENTATION_CHECKLIST — GIT-FIRST como Proceso

> Nodo-51 F-Meta — Costo de cumplir: 5 minutos.
> Costo de ignorar, ya pagado dos veces: nodos declarados BLOQUEADOS con solución existente, y edges fantasma en producción.

## Checklist Obligatorio (antes de implementar cualquier nodo con scraping/datos)

```bash
[ ] git log --all --oneline -- '*<keyword>*'
    # ¿Existe código previo del usuario para este problema?
    # Caso real: extraer_cuotas_partidos.py (23d2d91) resolvía Nodo-48 — declarado BLOQUEADO por error.

[ ] git grep <keyword> $(git rev-list --all) 2>/dev/null | head -20
    # ¿La solución ya se escribió en algún commit anterior?

[ ] ¿La URL que voy a usar es del NAVEGADOR o de una API interna?
    # global.flashscore.ninja/202/x/feed = Ninja API (JSON, sin cuotas) — NUNCA usar como URL de browser
    # www.flashscore.com/tennis/ = sitio web real (DOM, Playwright)
    # Error raíz de Nodo-48 y Nodo-49 — no repetir.

[ ] ¿El dato "inexistente" existe en otra fuente ya conocida?
    # Fuentes conocidas: THF cache (Nodo-45), FlashScore DOM (Nodo-49), git history, calibracion_edge.json
    # Orden de búsqueda: cache → API → Playwright → NO_DATA (nunca phantom edge)
```

## Regla de Evidencia (MM-5, Jerarquía Clínica)

```
case report (n=2)   → hipótesis pre-registrada (validation/preregistered_hypotheses.json)
cohorte (n≥30)      → calibración (calibracion_edge.json)

NUNCA calibrar con n=1 aunque el efecto sea espectacular.
NUNCA ajustar umbrales mirando resultados intermedios (p-hacking).
```

## Regla de Contexto (MM-1, Inversión)

Antes de preguntar "¿cómo consigo el resultado?", preguntar:
**"¿Qué garantiza que NUNCA se calcule un edge sobre `p_modelo=0.500` por datos ausentes?"**

La respuesta es F2 (Nodo-51): el contrato de completitud.
Si el contrato no está firmado → `status='NO_DATA'` → excluido de TODOS los pools.
NO hace falta más lógica. El contrato es dueño de la garantía.

## Regla de URLs (error crítico documentado)

| URL | Tipo | Uso correcto |
|-----|------|--------------|
| `global.flashscore.ninja/202/x/feed/...` | API interna Ninja | Datos JSON para Ninja H2H parser |
| `www.flashscore.co/partido/tenis/{match_id}/#/h2h/general` | DOM navegador | Playwright H2H fallback |
| `api.kambi.com/offering/...` | API Kambi/Betplay | Cuotas reales para edge_calculator |
| `www.flashscore.com/tennis/` | DOM navegador | Cuotas/odds via Playwright (Nodo-48) |

**Nunca derivar URLs de browser desde URLs de API — son sistemas completamente distintos.**
