# PROMPT PARA FABLE — Nodo de Tenis: Auditoría, Expansión a Nivel Doctorado y Diseño del Protocolo de Verificación

> Instrucciones de uso: pega este prompt en una sesión nueva con Fable, y adjunta junto con él dos archivos: (1) `FABLE_02_TENIS.md` (la versión ya trabajada con Sonnet, con las conexiones de Graphify, codebase-memory-mcp, Hermes, MCP, n8n) y (2) el `CLAUDE.md` real del proyecto (714 líneas, Nodo-60-ADDENDUM, 1659 tests). Fable necesita los dos para no repetir análisis ya hecho y para no trabajar contra un estado desactualizado del pipeline.

---

## ROL

Eres el analista de mayor nivel disponible para este proyecto — doctorado en la práctica, con acceso a razonamiento extendido (test-time compute) que un modelo de implementación como Sonnet no tiene. Tu trabajo no es escribir código. Tu trabajo es pensar más profundo, más tiempo, y con más amplitud de dominios de lo que cualquier sesión de Sonnet puede permitirse — y dejar ese pensamiento convertido en una especificación que Sonnet pueda ejecutar sin ambigüedad, por el tiempo que haga falta, sin presión de sprints de horas.

No tienes límite de tiempo de razonamiento para esta tarea. Tómate el que necesites. Es preferible que entregues un análisis correcto y completo en una sesión larga a que entregues algo rápido e incompleto.

---

## CONTEXTO QUE DEBES ABSORBER ANTES DE EMPEZAR

1. **El proyecto real**: un motor de apuestas de tenis que opera como un hedge fund cuantitativo de corta duración (cada partido = activo financiero con vida útil de 2-3 horas). Fundamentos ya implementados: Kelly ajustado por divergencia Kullback-Leibler, Portfolio Kelly multi-activo con correlación ρ por tier, Sistema de Cobertura por Exclusión C(N,K), VaR/CVaR, grafo de rivalidad transitiva (Erdős + PageRank), cadenas de Markov con PELT cambio de régimen, H2H Immunity Dampener, y un sistema de pre-registro de hipótesis (H52-01 a H60-01) que impide el p-hacking retroactivo. 1659 tests pasando. Disciplina Spec-Driven Development estricta — nada se implementa sin Nodo documentado.

2. **El documento `FABLE_02_TENIS.md` adjunto**: contiene ya una primera capa de integración de herramientas (Graphify, codebase-memory-mcp, vault de Obsidian como audit trail, Hermes local para entity resolution y fallback de parsing, MCP de Playwright, n8n para automatizar el closing snapshot). Esa capa fue construida por una sesión de Claude Sonnet con investigación web real, pero Sonnet tiene un techo de profundidad analítica que tú no tienes. No repitas ese trabajo — óptimizalo, corrígelo donde esté equivocado o incompleto, y constrúyele encima.

3. **Lo que Sonnet no pudo ver por su naturaleza de modelo de implementación**: conexiones que requieren puentear literatura de dominios lejanos entre sí — teoría de portafolios, teoría de la información, microestructura de mercados, inferencia causal, teoría de juegos, detección de anomalías bayesiana, diseño experimental secuencial, econometría de series de tiempo financieras aplicada a un dominio no financiero. Esa es exactamente tu ventaja frente a Sonnet en esta tarea — úsala.

---

## LO QUE DEBES HACER — EN ESTE ORDEN

### Fase 1 — Auditoría del documento existente

Lee `FABLE_02_TENIS.md` completo y el `CLAUDE.md` real completo. Antes de agregar nada nuevo, identifica:
- Qué recomendaciones de la versión de Sonnet están bien fundamentadas y deben quedarse
- Qué recomendaciones son superficiales, genéricas, o no están realmente ancladas a un módulo/Nodo específico del proyecto — y deben profundizarse o eliminarse
- Qué contradice o queda desactualizado frente al estado real del CLAUDE.md (ya se marcó una: el bug `kelly_kl=0.0` puede estar resuelto; verifica si hay más discrepancias de este tipo)
- Qué partes del pipeline real (constantes hardcodeadas, guards, hipótesis pre-registradas, arquitectura GCS de tres carriles) todavía no tienen ninguna conexión con las herramientas propuestas, y deberían tenerla

Documenta esta auditoría explícitamente al inicio de tu respuesta — es la base de todo lo que sigue.

### Fase 2 — Búsqueda de conexiones ocultas adicionales, con tu nivel de análisis

Este es el corazón de la tarea. No repitas Graphify/Hermes/MCP/n8n — ya están cubiertos. Busca en las capas que Sonnet no tiene la profundidad de dominio para encontrar:

- **Validación estadística rigurosa de las constantes hardcodeadas del pipeline.** El proyecto ya tiene ρ por tier (0.25/0.20/0.15/0.10/0.05), λ por tier en Kelly-KL (1.0×/1.6×/2.4×/3.6×/4.5×), K-factor ELO por tier, y GCS_RECENCY_BOOST (×2.2/×1.8/×1.5). Todas fueron asumidas o calibradas con muestras pequeñas. ¿Qué método estadístico (bootstrap, validación cruzada temporal walk-forward, test de sensibilidad) permitiría a Sonnet auditar estas constantes contra el shadow book acumulado sin caer en el mismo p-hacking que el proyecto ya se cuida de evitar con el sistema de hipótesis pre-registradas?

- **Teoría de portafolios más allá de Kelly clásico.** El proyecto ya implementó Portfolio Kelly con factor de correlación simple. ¿Hay algo de la literatura de gestión de riesgo cuantitativo (Markowitz, Black-Litterman, Kelly fraccionado dinámico, control de drawdown tipo CPPI) que aplique de forma no obvia a un portafolio de picks de tenis con vida útil de horas en vez de portafolios de acciones con vida útil de meses?

- **Detección de anomalías y drift de modelo.** El pipeline tiene un Circuit Breaker (Nodo-26) y detección de cambios de régimen vía Markov/PELT a nivel de jugador. ¿Existe un método de monitoreo de drift a nivel de MODELO completo (no solo por jugador) — algo del dominio de MLOps/monitoreo de modelos en producción financiera — que detecte si el edge del sistema completo se está degradando estructuralmente, antes de que se vea en el P&L acumulado?

- **Inferencia causal para separar señal real de artefacto de selección.** El caso GCS (Grass/Surface Champion Signal) tuvo un proceso de auditoría en tres carriles precisamente para descartar sesgo de supervivencia. ¿Qué técnicas de diseño cuasi-experimental (matching, regresión discontinua si aplica, o simplemente un framework más formal de causal inference) fortalecerían ese proceso para el próximo patrón que se descubra, en vez de repetir el proceso ad-hoc de auditoría manual que tomó GCS?

- **Microestructura del mercado de apuestas.** El proyecto ya distingue cuota de entrada vs cuota de cierre (CLV real). Esto es exactamente el concepto de "toxic flow" y "informed trading" de microestructura de mercados financieros aplicado a bookmakers. ¿Qué se puede tomar prestado de esa literatura (por ejemplo, modelos de descubrimiento de precio, o el concepto de que el movimiento de línea ANTES del partido es información, no solo ruido) que el pipeline no está explotando todavía?

- **Cualquier otra conexión de dominio cruzado que tu conocimiento identifique** — no te limites a esta lista, es un punto de partida, no un techo. El objetivo es que encuentres al menos 3-5 conexiones genuinamente nuevas que ni la primera ni la segunda ronda de este documento contemplaron.

Para cada conexión que propongas: nómbrala, explica el principio de origen (de qué campo viene), y ánclala a un archivo/Nodo/constante específica del CLAUDE.md real — si no puedes anclarla a algo concreto del pipeline, no la incluyas.

### Fase 3 — Reescritura del documento a nivel doctorado

Con la auditoría (Fase 1) y las conexiones nuevas (Fase 2), reescribe y expande `FABLE_02_TENIS.md` en un documento nuevo. Exigencias de nivel:
- Cada recomendación debe tener justificación matemática o estadística explícita cuando aplique, no solo descripción cualitativa
- Cada conexión nueva debe decir exactamente qué Nodo nuevo crear, qué archivo modifica, y qué test la valida — con el mismo rigor que el propio CLAUDE.md real usa para documentar sus Nodos (ver Nodo-60 como el estándar de calidad a igualar: hipótesis, método, resultado numérico, guard, test)
- Mantén y profundiza lo que Sonnet ya dejó bien hecho (Hermes, MCP, n8n) — no lo borres, constrúyele encima donde tu análisis lo mejore
- El documento final debe ser autosuficiente: alguien que solo lea este documento nuevo y el CLAUDE.md real debe poder ejecutar sin tener que volver a `FABLE_02_TENIS.md` original

### Fase 4 — SPEC.md ejecutable, sin marco de tiempo fijo

Genera el SPEC.md que Sonnet va a ejecutar. Elimina cualquier estructura de "Sprint de N horas" o "Semana X" — en su lugar, organiza por **fases con dependencias explícitas**, donde cada fase se cierra cuando sus criterios de verificación pasan, sin importar cuánto tiempo real tome. Cada fase debe tener:
- Prerrequisitos (qué fase anterior debe estar cerrada)
- Archivos exactos a crear o modificar
- Comandos exactos en orden
- Criterio de verificación objetivo y medible (no "debería funcionar bien" — sino "el test T6X-01 pasa" o "el valor de ρ recalculado cae dentro del intervalo de confianza reportado")
- Qué hacer si el criterio de verificación falla (rollback, o escalar de vuelta a ti para reanálisis)

Recuerda las restricciones no negociables del proyecto real que cualquier fase debe respetar: baseline de pytest antes y después de cada cambio (1659 passed hoy, verificar el número real al momento de ejecutar), REGLA-T53 (ningún test hardcodea la fórmula, siempre invoca la función real), GIT-FIRST (buscar en git history antes de reimplementar), y cero impacto en producción durante la migración.

### Fase 5 — Diseño del protocolo de auditoría exhaustiva que TÚ vas a ejecutar después

Esta es la fase que hace que todo lo anterior sea verificable y no solo una lista de buenas intenciones. Diseña ahora — antes de que Sonnet implemente nada — el checklist exacto que usarás en la auditoría posterior para confirmar si cada punto que definiste en la Fase 4 fue implementado al nivel de detalle que tú especificaste, ni más ni menos.

El protocolo de auditoría debe incluir, por cada fase del SPEC.md:
- **Criterio binario de cumplimiento**: qué evidencia exacta (output de comando, contenido de archivo, resultado de test) confirma que la fase está "implementada correctamente" vs "implementada parcialmente" vs "no implementada"
- **Comandos de verificación que tú mismo vas a correr o pedir que se te muestren** en la sesión de auditoría — no debe depender de que Sonnet te reporte "ya lo hice", debe ser verificable con evidencia objetiva (output real de pytest, contenido real de un archivo, un query real a codebase-memory-mcp)
- **Preguntas de auditoría forense** para las conexiones nuevas de la Fase 2 específicamente — dado que son las más sofisticadas, necesitan una verificación más rigurosa que un simple "el archivo existe": por ejemplo, para la validación estadística de constantes, la auditoría debe incluir que tú reproduzcas el cálculo, no solo que confirmes que un número cambió
- **Clasificación de severidad de desviaciones**: si Sonnet implementó algo distinto a lo especificado, define de antemano qué desviaciones son aceptables (mejoras justificadas) y cuáles son motivo de rechazo total de la fase (atajos que comprometen el rigor cuantitativo del proyecto, como saltarse el pre-registro de hipótesis)

Entrega este protocolo de auditoría como un documento separado, `AUDITORIA_FABLE_TENIS.md`, para que se use en la sesión de revisión posterior sin tener que reconstruir los criterios desde cero.

---

## FORMATO DE ENTREGA

Entrega tres artefactos diferenciados en tu respuesta:
1. **Auditoría de la versión anterior** (Fase 1) — puede ser breve, es diagnóstico
2. **`FABLE_02_TENIS_DOCTORADO.md`** — el documento expandido completo (Fases 2 y 3) con el SPEC.md ejecutable incluido (Fase 4)
3. **`AUDITORIA_FABLE_TENIS.md`** — el protocolo de verificación posterior (Fase 5)

No resumas ni comprimas por brevedad — este documento existe para que Sonnet lo ejecute sin tener que volver a preguntar, y para que la auditoría posterior sea objetiva. Prefiere extensión y precisión sobre concisión en esta tarea específica.
