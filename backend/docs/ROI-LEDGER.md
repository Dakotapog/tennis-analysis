# ROI-LEDGER.md — Ledger de ROI: Tokens vs Entregables vs P&L

> **Nodo:** [[Nodo-59-Motor-Agentico-Odometro-Dream]]
> **Actualización:** semanal (~10 min), cada lunes
> **Honestidad estructural:** el costo de tokens se mide en entregables verificables y horas ahorradas. El P&L de apuestas se reporta al lado SIN mezclarse — mismo principio que simulado≠real del shadow book. Si en 4 semanas costo > valor + P&L negativo, el ledger lo dirá sin anestesia.

---

## Semana 1 — 2026-06-03 → 2026-07-03 (acumulado inicial)

| Campo | Valor | Fuente |
|-------|-------|--------|
| **Costo tokens USD** | $1,292.27 | `token_odometer.py --report` |
| **Horas humanas estimadas** | ~120h (4 semanas × ~30h/semana) | Estimación operador |
| **Modelos usados** | Sonnet $609 + Opus $435 + Haiku $206 | odómetro |
| **Cache hit rate** | 95.7% | odómetro |
| **Sesión más cara** | b5cc2e2b Opus [impl] $189.69 (2026-06-07) | odómetro |

### Entregables verificables (nodos completados)

| Nodo | Descripción | Tests agregados |
|------|-------------|-----------------|
| Nodo-01 | Edge Calculator + Kelly-KL | — |
| Nodo-17/18/19/20/21 | Pesos 5-tier + density + shrinkage + H2H Immunity + PELT | — |
| Nodo-27/28/29/30 | Observabilidad + Circuit Asymmetry + Tournament Momentum | — |
| Nodo-32 | Markov POST-NORM (motor nodo32-fase3-markov-postnorm) | — |
| Nodo-38/39/40/41 | Combo Confianza + Playwright + Games Signal + ML Dataset | — |
| Nodo-44/45/46/47 | WAS + THF + auditoría + bug ranking inject | — |
| Nodo-48/49/50/51 | FlashScore odds + Playwright H2H + filtro torneo + Data Layer | — |
| Nodo-52 | Shadow Book CLV | — |
| Nodo-55 | Fable Funnel Deploy + Stake Waterfall | +5 tests |
| Nodo-56 | Bugs display pesos | +3 tests |
| Nodo-57 | Form decay + champion gate | +11 tests |
| Nodo-59 | Motor agéntico + odómetro | +4 tests |
| **TOTAL** | **~59 nodos** | **1,616+ tests** |

### P&L Apuestas (SEPARADO — no mezclar con ROI IA)

| Campo | Valor | Fuente |
|-------|-------|--------|
| Bankroll actual | $125,000+ | betslip_registrar |
| P&L acumulado | +$25,000 (estimado) | pipeline_tracker |
| Hit rate global | 63.2% (n=1,241) | calibracion_edge.json |
| Shadow book settled | n=0 (inicializado 2026-07-02) | shadow_book.py --report |

> **Nota:** Shadow book iniciado en Nodo-52 (2026-07-02). Primeros 30 días de CLV tracking están en curso (H52-01→H52-08 pre-registradas). El ROI de IA no se mezcla con el P&L — son métricas independientes.

---

## Semana 2 — [2026-07-07 → 2026-07-13] (pendiente)

| Campo | Valor | Fuente |
|-------|-------|--------|
| Costo tokens USD | `python3 token_odometer.py --report --desde 2026-07-07` | — |
| Entregables | — | — |
| P&L de apuestas | — | `shadow_book.py --report` |

---

## Semana 3 — [2026-07-14 → 2026-07-20] (pendiente)

*(completar cada lunes)*

---

## Semana 4 — [2026-07-21 → 2026-07-27] (pendiente)

*(completar cada lunes — primer ciclo de hipótesis del shadow book)*

---

## Notas de Metodología

### Cómo medir horas humanas
Estimación por sesión: cada sesión de Claude Code que no es Haiku automático = ~1-2h humanas (revisar, aprobar, testear). Total semanal estimado.

### Criterio de continuación (post-semana 4)
- Si costo tokens > valor estimado de entregables + P&L positivo → revisar routing (D59-02)
- Si %untagged > 20% → disciplina de tags (convención en MODEL-ROUTING.md)
- Si cache hit rate < 50% → sesiones demasiado cortas o muchos `/clear` innecesarios

### Cómo calcular valor de entregables
- Cada test agregado → ~$5-10 USD de valor de verificación futura
- Cada nodo completado → horas evitadas de debugging futuro (~5-20h/nodo)
- Cada skill/comando → minutos ahorrados × frecuencia semanal × semanas restantes
