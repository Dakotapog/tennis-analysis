Captura snapshot de cuota de cierre Kambi (Momento 2 del shadow book).

Corre: `python3 shadow_book.py --close-snapshot`

Ejecutar ~15 minutos ANTES del inicio de cada partido para capturar la cuota de cierre real de Kambi.

Horarios recomendados por tier (REGLA-SB-1):
- Grand Slam: 08:30 AM (partidos empiezan ~10:00)
- ATP1000/500: según horario del torneo
- ITF/Challenger: 12:30 PM (partidos suelen empezar 14:00+)

Sin este paso, el CLV se calcula solo con cuota de entrada (menos preciso). Con snapshot de cierre, CLV es cuota_entrada vs cuota_cierre → métrica real de valor capturado.

Ver `docs/knowledge-assets.md §8` para estructura del betslip.
