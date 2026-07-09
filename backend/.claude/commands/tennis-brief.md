Brief diario del sistema: salud del pipeline, shadow book y calibración.

## Qué hace
Equivalente a `run_daily.py` en modo lectura pura — sin ejecutar nada, solo reportar estado.

1. **Pipeline hoy** — corre internamente:
   ```bash
   python3 pipeline_tracker.py --since $(date +%Y-%m-%d) --section confianza
   ```

2. **Shadow book** — últimas 7 sesiones:
   ```bash
   python3 shadow_book.py --report
   ```
   Muestra: hit% por tier, CLV median, IC Wilson. Alerta si ITF hit% < 35%.

3. **Drift monitor** — si existe `analysis/drift_monitor.py`:
   ```python
   from analysis.drift_monitor import daily_drift_report
   r = daily_drift_report()
   # mostrar cusum_alarm, psi_alarm
   ```

4. **Calibración** — lee `data/calibracion_edge.json`:
   - Clay GS: p actual, n muestras
   - Alerta si n < 30 en cualquier tier activo

5. **Hipótesis activas** — de `validation/preregistered_hypotheses.json`:
   - Todas las hipótesis en estado ACUMULANDO con su n_actual / n_stop
   - Cualquier hipótesis que cruzó n_stop → alerta DECISION PENDIENTE

6. **Bankroll implícito** — del `reports/trader_plan_*.json` más reciente:
   - Último bankroll registrado y fecha

## Frecuencia recomendada
Correr una vez por día antes del PASO 1 (extraer partidos). Toma 30 segundos.

Texto plano, sin emojis. Columnas alineadas con espacios para legibilidad en terminal.
Si algún comando falla, mostrar el error y continuar con los demás (no abortar).
