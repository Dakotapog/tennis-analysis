Genera artículo de audit-trail para la sesión actual en .spec/01_Nodos/audit-trail/.

## Qué hace

Ejecuta `session_compiler.py` y guarda el artículo de la sesión:

```bash
cd /mnt/c/users/hogar/tennis-analysis/backend
python3 session_compiler.py --horas 8
```

El artículo incluye:
- Commits de las últimas 8 horas (o desde `--desde YYYY-MM-DD`)
- Archivos modificados
- Entradas nuevas D/E/C del DECISION-LOG.md
- Estado de tests (desde git log o CLAUDE.md)

## Variantes

```bash
python3 session_compiler.py --horas 24           # últimas 24h
python3 session_compiler.py --desde 2026-07-08   # desde fecha
python3 session_compiler.py --tema "GCS forense" # tema explícito
python3 session_compiler.py --dry-run            # preview sin guardar
```

## Cuándo usar

Al final de cada sesión de trabajo, antes de cerrar Claude Code,
para generar el audit trail en .spec/01_Nodos/audit-trail/.
