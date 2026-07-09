Resumen de sesión activa: picks, stakes, combos y señales del día.

## Qué hace
Lee los archivos más recientes de la sesión y produce un resumen ejecutivo de 5 minutos:

1. **Picks con edge** — del `reports/edge_report_*.json` más reciente:
   - Jugadores con status=APOSTAR, confianza, cuota, kelly_kl, stake sugerido
   - Picks con GCS activo (gcs_active=True)
   - Picks en watchlist (status=WATCHLIST)

2. **Plan de combos** — del `reports/combo_plan_*.txt` más reciente (si existe)
   - CORE combo seleccionado, cuota combinada, stake
   - Universo GCS separado si hay picks grass

3. **Señales alpha** — picks con alpha_score > 20 (Signal Bridge Nodo-62)

4. **Riesgo** — del `reports/trader_plan_*.json` más reciente (si existe):
   - KGR (si < 0 → NO DESPLEGAR en negrita)
   - VaR% del bankroll
   - CPPI factor (si < 0.5 → bankroll cerca del floor)

5. **Estado hipótesis críticas**:
   - H60-01 (GCS grass): n_actual / 30
   - H62-01 (alpha_promoted): n_actual / 30

## Cómo leer los archivos
```bash
import glob, json
edge = sorted(glob.glob('reports/edge_report_*.json'))[-1]  # más reciente
trader = sorted(glob.glob('reports/trader_plan_*.json'), default=[None])
combo = sorted(glob.glob('reports/combo_plan_*.txt'), default=[None])
```

Mostrar en formato tabla clara. Texto plano, sin emojis.
Si no hay archivos del día, indicar "Sin sesión activa — correr PASO 3 (edge_calculator.py)".
