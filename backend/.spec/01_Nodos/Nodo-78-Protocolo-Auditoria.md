# Nodo-78 — Protocolo de Auditoría SDD (derivado de sesión 2026-07-09)

**Fecha:** 2026-07-09
**Estado:** PROPUESTO — para aplicar tal cual, sin reinterpretar
**Rama:** main
**Origen:** auditoría extendida (11 rondas) sobre C62-A, C63-A, C63-B, Nodo-75, Strangler Fig
**Cierra:** necesidad de que futuras sesiones de Claude Code arranquen con esta disciplina ya incorporada, sin reconstruirla desde cero.

---

## Por qué existe este Nodo

Durante la auditoría del 2026-07-09, cada conclusión que se aceptó en el primer intento resultó parcial o incorrecta al menos una vez. Cada vez que se pidió "un comando más" antes de cerrar, la conclusión cambió — a veces de forma decisiva (código que parecía muerto resultó ser dependencia activa; una "reversión" que parecía real nunca ocurrió). Este Nodo documenta el patrón para que deje de repetirse desde cero en cada sesión.

## Regla 1 — Git archaeology antes de declarar código muerto

Antes de recomendar eliminar, deprecar o ignorar un archivo por "no tener consumidor":
```bash
git log --all --oneline -- <archivo>
git log --all --grep="<nombre_del_módulo>" -i --oneline
```
Un archivo sin import visible en un grep superficial puede tener un consumidor indirecto (una clase que lo importa, que a su vez es importada por el entry point real). Verificar la cadena completa de dependencias, no solo el primer nivel. Caso real: `browser_manager.py`/`data_parser.py` parecían huérfanos porque nadie en el "pipeline primario" los importaba directamente — el consumidor real (`h2h_extractor.py` → `H2HExtractor`) estaba un nivel más adentro.

## Regla 2 — Un comando más antes de cerrar una conclusión

Ninguna conclusión de auditoría se acepta como final si se sostiene en un solo comando o en inferencia sin verificar el otro extremo del flujo. Ejemplos reales de esta sesión donde la primera conclusión cambió con un comando adicional:
- "combo_governor.py está untracked" → `git log --oneline -1 -- archivo` mostró que ya estaba versionado.
- "el log de Playwright candidate no tiene cola" → un segundo grep mostró que sí llama a `_enqueue_playwright_candidate` en la misma rama de código.
- "H2HExtractor fue revertido a Playwright inline" → revisar la línea real (`extractor = H2HExtractor()`) mostró que nunca se revirtió.

Regla operativa: si una conclusión se basa en la ausencia de algo (ningún import, ningún test, ninguna mención), verificar la ausencia con al menos dos ángulos de búsqueda distintos antes de reportarla como hallazgo.

## Regla 3 — Distinguir "huérfano real" de "omisión de nombre"

Un archivo puede estar completamente cubierto por un Nodo en términos conceptuales, pero aparecer como huérfano en cualquier índice automatizado si el Nodo lo referencia por nombre de clase (`BrowserManager`) en vez de nombre de archivo (`browser_manager.py`). Antes de crear un Nodo retroactivo nuevo para un "huérfano" detectado por herramienta automática, verificar primero si el Nodo que debería cubrirlo ya existe y solo le falta la fila con el nombre de archivo exacto — la corrección correcta en ese caso es un adendo con fecha al Nodo existente, no un Nodo nuevo.

## Regla 4 — No asumir que el contenido pegado llegó

Si un reporte de auditoría afirma "el contenido está pegado arriba" pero el destinatario no lo ve, no se asume que es un problema del destinatario. Se repite la solicitud en fragmentos más pequeños o se verifica el tamaño/existencia del archivo antes de reintentar pegarlo completo.

## Regla 5 — "Nunca se ejecutó" es un hallazgo, no un cierre limpio

Un mecanismo de control (governor, guard, validador) que existe en el código, pasa sus tests, y suma/verifica correctamente — pero que tiene **0 ejecuciones reales en producción** — no se reporta como "implementado, listo". Se reporta como lo que es: un control sin datos de comportamiento real, con una condición explícita de cuántas ejecuciones reales se necesitan antes de escalar su nivel de autonomía (ej. de modo reporte a modo bloqueo).

## Regla 6 — Preferir la corrección mínima sobre la documentación nueva

Antes de crear un Nodo retroactivo completo para cerrar un gap SDD, verificar si el gap es en realidad una omisión pequeña en un Nodo ya existente (una fila faltante en una tabla, un nombre de archivo no mencionado). La corrección mínima con un adendo fechado es preferible a generar documentación nueva para algo que ya estaba conceptualmente cubierto. Reservar Nodos nuevos para arquitectura genuinamente sin cobertura previa.

---

## Aplicación

Este protocolo aplica a cualquier sesión futura de Claude Code que realice auditorías de trazabilidad SDD (comparación CLAUDE.md vs Nodos vs código real), no solo a la de esta fecha. `check_contradictions.py` y cualquier script equivalente futuro deben citar este Nodo en su docstring como referencia de la disciplina esperada.

## Regla 7 — Un respaldo no verificado bajo fallo real es indistinguible de no tener respaldo

Nunca confiar en que "cron de respaldo" o "supervisado por systemd" significa que funciona, sin una prueba de fallo real o simulada.

Caso real (2026-07-10 — simulacro de apagón WSL2): el cron `*/10` de `close_snapshot_trigger.py` corrió silenciosamente sin numpy disponible durante semanas. Nadie lo notó porque n8n (mecanismo primario) nunca falló, así que el respaldo roto nunca se puso a prueba. Adicionalmente, `tennis-snapshot-bridge.service` nunca se registró en systemd: el servidor en :8765 corría como proceso huérfano (PID 208) sin supervisor real, arrancado manualmente y sobreviviendo por linger — no por diseño. Un apagón real habría dejado ambos mecanismos de respaldo inoperativos simultáneamente, con n8n como único punto de fallo no redundado.

Fixes aplicados en la misma sesión: (1) cron actualizado a `venv/bin/python3` explícito, restringido a 9-23h; (2) unit `tennis-snapshot-bridge.service` creado, registrado y habilitado bajo systemd real (PID 208 → PID 4519); (3) guard `if not match_id: return False` en `_already_processed()` (`close_snapshot_server.py:87`) — bug latente porque JSON puede traer `match_id: null` aunque el contrato de tipo diga `str`.

Regla operativa: cada mecanismo de respaldo debe tener al menos una prueba de fallo documentada (real o simulada) antes de considerarse operativo. La existencia del código no equivale a la existencia del respaldo.

---

## Vinculación

Relacionado con el trabajo de auditoría 2026-07-09: Nodo-74, Nodo-75, Nodo-76, adendos a Nodo-07 y Nodo-73, entrada C-07 en `docs/DECISION-LOG.md`.
Regla 7 derivada del simulacro de apagón 2026-07-10.
