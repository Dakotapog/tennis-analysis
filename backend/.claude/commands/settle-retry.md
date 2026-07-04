Settle con retry para partidos ITF/Challenger rezagados hasta 48h.

Corre settlement para las últimas 2 fechas:
```bash
python3 shadow_book.py --settle $(date -d "yesterday" +%Y-%m-%d)
python3 shadow_book.py --settle $(date -d "2 days ago" +%Y-%m-%d)
```

Luego muestra el reporte actualizado:
```bash
python3 shadow_book.py --report
```

Contexto: los partidos ITF y Challenger a veces terminan tarde o tienen resultados pendientes. Este skill los recoge sin requerir la fecha exacta.

Si el usuario especifica una fecha concreta (ej. `2026-07-05`), usa esa fecha en lugar de yesterday/2-days-ago.
