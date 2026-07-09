Ejecuta el pipeline diario completo de análisis de tenis.

Corre `python3 run_daily.py --bankroll 125000` para ejecutar PASO 0→4.3 + settle de ayer + daily_brief.

Si el usuario pasa argumentos adicionales (como `--tomorrow` o `--bankroll N`), inclúyelos en el comando.

Después de correr, muestra el contenido de `reports/daily_brief_*.txt` más reciente para revisión humana.

Si hay errores en la ejecución, reporta cuál paso falló y sugiere el comando de corrección específico según CLAUDE.md §4.
