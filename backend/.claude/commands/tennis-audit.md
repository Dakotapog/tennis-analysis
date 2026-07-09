Auditoría pre-partido: valida picks del edge_report antes de apostar.

## Qué hace
1. Lee el `reports/edge_report_*.json` más reciente
2. Para cada pick con `status=APOSTAR`, verifica:
   - `kelly_kl > 0` (si es 0.0 → alerta KELLY_ZERO, no desplegar)
   - `n_partidos >= 8` (si < 8 → alerta INSUFFICIENT_HISTORY per Nodo-63)
   - `history_provenance` no es `api_only` cuando `n_partidos > 20` (señal phantom identity)
   - `ranking` no es None cuando `n_partidos > 20` (otro indicador phantom)
3. Lee `validation/preregistered_hypotheses.json` y muestra estado de H60-01, H62-01, H54-01
4. Corre `python3 pipeline_tracker.py --section shadow` para hit% actual

## Cómo ejecutar
```bash
# Leer edge_report más reciente
import glob, json
f = sorted(glob.glob('reports/edge_report_*.json'))[-1]
data = json.load(open(f))
```

## Output esperado
Tabla con columna PASS/WARN/BLOCK por pick. Cualquier BLOCK = no apostar ese pick.
Resumen de hipótesis activas al final.

Si el usuario pasa un nombre de jugador como argumento (ej: `/tennis-audit Alcaraz`), filtra solo ese pick.

Reporta en texto plano (sin emojis — salen como ? en terminal).
