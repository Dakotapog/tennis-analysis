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

## REGLA-T53 — Tests de Bug (Nodo-53, elevada a regla permanente)

```
Ningún test de bug reproduce la fórmula manualmente en el test.
SIEMPRE invocar la función del módulo real.

MAL:
  max_surface = 350
  norm = min(raw/max_surface, 1.0) * math.log1p(max_surface)  # hardcodea la fórmula buggy
  assert norm > 0.40  # permanece FAIL después del fix — el test no detecta el cambio

BIEN:
  from analysis.rivalry_analyzer import normalize_scores
  norm_p1, _ = normalize_scores({'surface_specialization': 33.49}, {'surface_specialization': 10.89})
  assert norm_p1['surface_specialization'] > 0.40  # FAIL antes, PASS después

Por qué: un test que hardcodea la fórmula buggy permanece en FAIL después del fix.
Sonnet concluye que el fix no funcionó y elimina el test. El contrato FAIL→PASS
solo es válido si el test llama al código real.
Tercera ocurrencia del mismo error en el ciclo Nodo-53.
```

## REGLA-T53 — Tests de Bug Deben Llamar al Módulo Real

```
Ningún test de bug reproduce la fórmula manualmente. SIEMPRE invocar la función del módulo.

MAL — test en FAIL permanente (antes Y después del fix):
  norm = min(raw/350, 1.0) * math.log1p(350)  # hardcodea fórmula buggy
  assert norm > 0.40  # sigue fallando después del fix — no detecta el cambio

BIEN — contrato real FAIL→PASS:
  from analysis.rivalry_analyzer import normalize_scores  # función de módulo
  norm_p1, _ = normalize_scores({'surface_specialization': 33.49}, {...})
  assert norm_p1['surface_specialization'] > 0.40  # FAIL antes, PASS después del fix

Si la función es anidada (no importable) → extraerla a nivel módulo como parte del fix.
Tercera ocurrencia del mismo error en Nodo-53. Elevada a regla permanente.
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
