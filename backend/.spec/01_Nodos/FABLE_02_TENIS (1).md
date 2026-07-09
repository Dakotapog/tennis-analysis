# FABLE_02 — PIPELINE DE TENIS
## Integración de herramientas nuevas — NO reconstruir, solo integrar

> Estado actual: Pipeline funcionando con 62 nodos, 1585 tests pasando
> Objetivo: Agregar memoria persistente, mapa estructural, y automatización sin romper nada
> Regla absoluta: Cero impacto en producción. El pipeline sigue funcionando igual.
> Fecha: 2026-07-05

---

## INSTRUCCIÓN PARA FABLE

El CLAUDE.md actual del proyecto tiene 797 líneas y documenta un sistema completamente funcional. NO repliques su contenido. Este documento es un ADDENDUM — solo contiene lo nuevo que se integra.

Tu trabajo:
1. Diseñar la integración de Graphify + codebase-memory-mcp sin tocar el pipeline existente
2. Diseñar el vault de Obsidian como audit trail vivo que reemplaza el historial estático
3. Resolver el bug de kelly_kl=0.0 mediante auditoría forense
4. Generar el SPEC.md ejecutable para Sonnet con instrucciones que no rompan los 1585 tests

Restricción crítica: Sonnet debe correr `python -m pytest tests/ --no-cov -q` antes y después de cada cambio. Si algún test falla, revertir inmediatamente.

**Nota de reconciliación (segunda ronda, con el CLAUDE.md real del proyecto en mano):** este documento se escribió originalmente contra un estado del pipeline de hace varios Nodos (1585 tests, bug `kelly_kl=0.0` como pendiente crítico sin auditar). El CLAUDE.md real actual del proyecto está en Nodo-60-ADDENDUM-FABLE, con 1659 tests, GCS activo solo en hierba, y el bug activo real hoy es distinto: `prediccion_ganador` top-level = None en `extraer_historh2h.py` (usar `ranking_analysis.prediction.favored_player`). El bug `kelly_kl=0.0` de 2026-06-23 ya no aparece en la tabla de bugs activos del CLAUDE.md real — probablemente fue resuelto en alguna sesión entre Nodo-38 y Nodo-60 y archivado en git history. Fable debe verificar esto con `git log --all --oneline -- '*kelly_kl*'` ANTES de ejecutar el Sprint 3 de auditoría forense de este documento — si el bug ya está resuelto, ese sprint completo se reemplaza por el Vacío nuevo (ver sección de conexiones profundas más abajo: entity resolution para Andy/Avery Nguyen y el bug real de `prediccion_ganador`).

---

## LO QUE YA EXISTE Y NO SE TOCA

El CLAUDE.md actual del proyecto documenta completamente:
- Pipeline Paso 0 → Paso 10 con comandos exactos
- 62 módulos Python con sus responsabilidades
- Kelly-KL, ELO/Markov, Erdős, H2H Immunity, Portfolio Kelly implementados
- 1585 tests pasando (62 tests Nodo-31, 25 tests Nodo-38, 9 tests Nodo-45)
- Bugs activos identificados
- Reglas GIT-FIRST, SDD, TTC Protocol
- Guards de no-ruina: REGLA-HF-1, REGLA-HF-5, VaR automático

Pendientes del audit Nodo-32 a Nodo-40 sin resolver:
- `kelly_kl=0.0` en betslip 2026-06-23 — bug crítico sin auditar
- Andy/Avery Nguyen — riesgo de ambigüedad de nombre
- Audit de 31 nodos de priority mapping
- Backtesting out-of-sample con T15-06
- Validación CAD Nodo-29 (n≥50 sesiones limpias)

---

## HERRAMIENTA 1 — GRAPHIFY

### Qué hace exactamente
Graphify es un CLI que toma la carpeta del proyecto y genera un grafo de conocimiento persistente usando Tree-sitter para parsear el código fuente. Dos pasadas:

Pasada 1 (AST — cero tokens, cero costo):
- Extrae clases, funciones, imports, call graphs, docstrings
- Tu código nunca sale de tu máquina
- Resultado: `graphify-out/graph.json`

Pasada 2 (semántica — usa tokens de Claude):
- Procesa PDFs, imágenes, markdown, configuraciones
- Solo se corre una vez por archivo — cache por SHA256
- Solo re-procesa archivos que cambiaron

Archivos que genera:
```
graphify-out/
├── graph.json      ← fuente de verdad — queryable entre sesiones
├── GRAPH_REPORT.md ← resumen de arquitectura, nodos más conectados, conexiones sorprendentes
├── graph.html      ← visualización interactiva en el browser
└── cache/          ← SHA256 por archivo — evita re-procesar
```

### Por qué es crítico para el pipeline de tenis
Con 62 módulos Python, Claude Code hoy hace grep a través de todos los archivos para orientarse — 15.000-20.000 tokens por sesión antes de hacer algo útil.

Con Graphify instalado, una query como:
```
graphify query "qué módulos dependen de kelly_kl"
graphify path "phantom_gate" "apostar"
graphify explain "ninja_h2h_parser"
graphify query "show me all god nodes" --dfs
```
responde en milisegundos desde el grafo — sin releer código.

El `--dfs` (depth-first search) es especialmente valioso para el pipeline de tenis: permite rastrear el camino exacto desde una señal de entrada hasta `apostar=True` — exactamente el análisis que se hace en cada auditoría de nodo.

### Blast-radius analysis — la función más valiosa
Antes de tocar cualquier módulo, Graphify puede decir qué otros módulos se ven afectados. Para un sistema con 1585 tests y bugs silenciosos ya documentados, saber el blast-radius de cada modificación antes de escribir una línea es la diferencia entre una sesión limpia y un regresión.

```
graphify impact "edge_calculator.py"
→ [rivalry_analyzer.py, trader_ev_tenis.py, betslip_registrar.py, 
   tests/test_nodo21.py, tests/test_nodo33.py]
```

### Git hook — actualización automática
```bash
graphify hook install
```
Este comando instala un hook en git que reconstruye el grafo en cada commit — sin tokens extra (solo el pase AST, que es local). El grafo siempre refleja el código actual.

### Integración con el vault de Obsidian
```bash
graphify ./tennis_pipeline --obsidian --obsidian-dir ~/vault/graphify/tenis/
```
Genera automáticamente una nota .md por cada módulo del pipeline, con backlinks entre módulos dependientes. El vault muestra visualmente cómo `kelly_kl` conecta con `edge_calculator`, con `trader_ev_tenis`, con `betslip_registrar` — sin escribir una sola nota manualmente.

### Instalación — orden exacto
```bash
pip install graphifyy                          # instalar
graphify install                               # registrar skill en Claude Code
# Crear .graphifyignore ANTES de correr:
echo "tests/\ndata/\nreports/\n__pycache__/\n*.json\n*.log" > .graphifyignore
graphify ./tennis_pipeline                     # primera construcción
graphify ./tennis_pipeline --obsidian --obsidian-dir ~/vault/graphify/tenis/
graphify hook install                          # git hook automático
```

Advertencia: si detecta más de 200 archivos sin .graphifyignore, hace pausa y pide confirmación. El .graphifyignore debe estar en la raíz del proyecto.

---

## HERRAMIENTA 2 — CODEBASE-MEMORY-MCP

### Qué hace exactamente
Servidor MCP de alto rendimiento que indexa el codebase en un grafo de conocimiento persistente en SQLite. Un binario estático sin dependencias — corre localmente.

Capacidades únicas que Graphify no tiene:
- Consultas tipo Cypher: `MATCH (f:Function)-[:CALLS]->(g) WHERE f.name = 'kelly_kl' RETURN g.name`
- Detección de código muerto: funciones sin llamadores, excluyendo entry points
- Git diff impact mapping: mapea cambios sin commitear a símbolos afectados con clasificación de riesgo
- Búsqueda semántica vectorial embebida en el binario — sin API externa
- Architecture Decision Records: persiste decisiones arquitecturales entre sesiones

Para el pipeline de tenis, el ADR (Architecture Decision Records) es especialmente valioso: cada decisión de la auditoría Nodo-32 a Nodo-40 se persiste como un registro inmutable en el grafo — no en el CLAUDE.md que crece sin límite.

### Configuración MCP en Claude Code
```json
{
  "mcpServers": {
    "codebase-memory": {
      "command": "cbm",
      "args": ["serve"],
      "env": {}
    }
  }
}
```

Después de instalar, en cada sesión de Claude Code:
```
graphify query "qué depende de kelly_kl"     ← mapa estructural
cbm semantic_query "serialización de stakes"  ← búsqueda semántica
cbm detect_changes                            ← impacto de cambios no commiteados
cbm manage_adr "bug kelly_kl=0.0 resuelto: era bug de serialización en betslip_registrar.py línea X"
```

---

## HERRAMIENTA 3 — VAULT DE OBSIDIAN COMO AUDIT TRAIL VIVO

### El problema con el CLAUDE.md actual
El CLAUDE.md tiene 797 líneas y crece con cada sesión. Es un archivo estático que un humano tiene que mantener manualmente. En 6 meses tendrá 2000 líneas y será difícil de navegar.

### La solución: vault como audit trail
El vault de Obsidian reemplaza la sección de historial del CLAUDE.md. El CLAUDE.md apunta al vault en vez de contener todo el historial.

Estructura del vault para el proyecto de tenis:
```
~/vault/proyecto-tenis/
├── CLAUDE.md                    ← constitución específica del proyecto (50 líneas máximo)
├── architecture/
│   ├── pipeline-overview.md     ← mapa de Paso 0 a Paso 10
│   ├── fundamentos-cientificos/ ← Kelly-KL, Erdős, Markov — uno por módulo
│   └── decisiones/              ← ADRs: qué se decidió y por qué
├── audit-trail/
│   ├── nodo-32.md               ← bug phantom edge gate — corregido
│   ├── nodo-33.md               ← bug lateral gate — corregido
│   ├── nodo-34.md               ← bugs score inversion + ranking bias
│   ├── nodo-35.md               ← missing historial_extraido flag
│   ├── nodo-36.md               ← unicode filtering
│   ├── nodo-37.md               ← ...
│   └── pendientes.md            ← kelly_kl=0.0, Nguyen, backtesting
├── sessions/
│   ├── YYYY-MM-DD.md            ← betslip procesado, resultado, lecciones
│   └── anomalias/               ← flags críticos detectados por rutinas
└── patterns/
    ├── fallos-por-circuito.md   ← qué circuito genera más falsos positivos
    ├── gates-activos.md         ← estado actual de cada gate
    └── calibracion.md           ← estado de calibracion_edge.json
```

### Schema frontmatter para sesiones de tenis
Cada sesión de apuesta genera una nota con este frontmatter exacto:
```yaml
---
tipo: sesion_betslip
fecha: YYYY-MM-DD
circuito: [ATP|WTA|Challenger|ITF]
superficie: [clay|hard|grass|carpet]
picks_total: N
picks_aprobados: N
picks_rechazados: N
kelly_kl_min: 0.XXX       # CRÍTICO: si es 0.0 → flag automático
anomalias: []              # lista de flags detectados
resultado: [ganado|perdido|pendiente|parcial]
kgr: 0.XXX                # Kelly Growth Rate — si <0 NO DESPLEGAR
var_pct: 0.XX              # % bankroll en VaR
nodos_activados: []        # qué nodos procesaron esta sesión
gates_fallidos: []         # qué gates fallaron si hubo anomalía
---
```

Si `kelly_kl_min = 0.0` en cualquier nota → el agente nocturno crea issue en GitHub automáticamente.

---

## AUDITORÍA FORENSE — BUG KELLY_KL=0.0

> ⚠️ Ver "Nota de reconciliación" al inicio del documento: este bug no aparece en la tabla de Bugs Activos del CLAUDE.md real (Nodo-60). Verificar con `git log --all --oneline -- '*kelly_kl*'` antes de ejecutar el Sprint 3 — puede que ya esté resuelto y archivado.

### Contexto del bug
Betslip del 2026-06-23 tiene `kelly_kl=0.0` en el archivo `betslip_index_20260623_004706.json`. Fue identificado como pendiente crítico en el audit Nodo-33 sección 1. No ha sido investigado.

### Lo que Fable debe hacer con este bug
Esta es exactamente la tarea para la que Fable rinde mejor: análisis forense profundo de una anomalía de un sistema cuantitativo. Fable debe:

1. Recibir el JSON completo del betslip + el código de `edge_calculator.py` + `betslip_registrar.py`
2. Determinar cuál de las dos hipótesis es correcta:
   - Hipótesis A: bug de serialización — el valor se calculó correctamente pero se serializó como 0.0
   - Hipótesis B: real sizing no aplicado — kelly_kl nunca se calculó esa noche

3. Trazar el camino exacto en el grafo del pipeline desde la entrada del partido hasta la escritura del betslip

### Queries de Graphify para la auditoría
```bash
graphify path "extraer_historh2h.py" "betslip_registrar.py"
graphify query "qué funciones escriben kelly_kl al JSON"
graphify explain "edge_calculator.py"
cbm semantic_query "serialización kelly_kl betslip"
```

### SPEC para Sonnet — auditoría del bug
```
TAREA: Auditoría forense kelly_kl=0.0
HERRAMIENTAS: graphify + codebase-memory-mcp
ANTES DE TOCAR CÓDIGO:
1. python -m pytest tests/ --no-cov -q  (baseline: 1585 passed)
2. graphify path "edge_calculator" "betslip_registrar" 
3. grep -n "kelly_kl" betslip_registrar.py | head -20
4. grep -n "kelly_kl" edge_calculator.py | head -20
5. Leer el betslip_index_20260623_004706.json completo
6. Reportar hipótesis con evidencia del código — NO hacer fix sin confirmación
NO HACER: modificar código sin baseline de pytest aprobado
NO HACER: asumir cuál es el bug sin trazar el path en el grafo
```

---

## SLASH-COMMANDS PERSONALIZADOS PARA EL PIPELINE

Estos archivos .md van en `~/.claude/commands/` y Claude Code los reconoce como slash-commands:

### /tennis-audit
```markdown
# Tennis Pipeline Audit

Cuando ejecutes este comando:
1. Corre `python -m pytest tests/ --no-cov -q` y reporta el resultado
2. Consulta graphify query "qué gates están activos" 
3. Lee ~/vault/proyecto-tenis/audit-trail/pendientes.md
4. Lee ~/vault/proyecto-tenis/sessions/anomalias/ — últimos 7 días
5. Genera reporte: tests pasando, gates activos, pendientes críticos, anomalías recientes
6. Si kelly_kl_min=0.0 en alguna sesión reciente → flag CRÍTICO inmediato
```

### /tennis-session
```markdown
# Tennis Session Recorder

Cuando ejecutes este comando:
1. Pide al operador: fecha, circuito, picks_total, picks_aprobados, resultado
2. Lee el betslip más reciente en reports/
3. Extrae: kelly_kl_min, kgr, var_pct, anomalias[], gates_fallidos[]
4. Crea nota en ~/vault/proyecto-tenis/sessions/YYYY-MM-DD.md con el frontmatter completo
5. Si kelly_kl_min=0.0 → ejecuta: gh issue create --title "ALERTA: kelly_kl=0.0 [FECHA]" --body [contexto]
6. Actualiza ~/vault/proyecto-tenis/patterns/ con el patrón detectado
```

### /tennis-brief
```markdown
# Tennis Daily Brief

Cuando ejecutes este comando:
1. Consulta la API de Kambi para partidos del día (python extraer_partidos_api.py)
2. Lee ~/vault/proyecto-tenis/audit-trail/pendientes.md
3. Lee ~/vault/proyecto-tenis/patterns/ para contexto de qué circuitos son más confiables hoy
4. Lee la última sesión en ~/vault/proyecto-tenis/sessions/
5. Genera brief: partidos del día disponibles, circuitos recomendados, pendientes críticos, estado del sistema
```

---

## RUTINAS AUTOMÁTICAS

### Rutina 1 — Validación pre-partido (7:00pm diario)
Tipo: GitHub Actions — lógica determinista, cero tokens

```yaml
name: tennis-pre-match-validation
on:
  schedule:
    - cron: '0 19 * * *'
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests baseline
        run: python -m pytest tests/ --no-cov -q
      - name: Check kelly_kl in recent betslips
        run: |
          python3 -c "
          import json, glob, sys
          files = sorted(glob.glob('reports/betslip_index_*.json'))[-3:]
          for f in files:
              data = json.load(open(f))
              for pick in data.get('picks', []):
                  if pick.get('apostar') and pick.get('kelly_kl', 1) == 0.0:
                      print(f'ALERTA: kelly_kl=0.0 en {f}')
                      sys.exit(1)
          print('OK: kelly_kl validado')
          "
      - name: Create issue if anomaly
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: 'ALERTA: kelly_kl=0.0 detectado ' + new Date().toISOString().split('T')[0],
              body: 'Validación pre-partido detectó kelly_kl=0.0. NO desplegar hasta resolver.',
              labels: ['bug', 'critical']
            })
```

### Rutina 2 — Síntesis post-sesión (11:00pm diario)
Tipo: Claude Code Routine (cloud — requiere razonamiento)

```
cron: "0 23 * * *"
prompt: |
  Lee el betslip más reciente en el repo del pipeline de tenis.
  Compara picks con apostar=True vs resultados reales (busca en web si necesario).
  Detecta: ¿qué nodo del pipeline generó el error si hubo pérdida?
  Ejecuta /tennis-session para guardar la sesión al vault.
  Actualiza ~/vault/proyecto-tenis/patterns/ con la lección del día.
  Si el KGR fue negativo: crea issue "KGR negativo - revisar antes de próxima sesión".
repos: [tennis-pipeline, vault-repo]
```

### Rutina 3 — Brief matutino tenis (6:30am diario)
Tipo: GitHub Actions — lógica determinista

```yaml
name: tennis-morning-brief
on:
  schedule:
    - cron: '30 6 * * *'
jobs:
  brief:
    runs-on: ubuntu-latest
    steps:
      - name: Get today matches
        run: python extraer_partidos_api.py --tomorrow
      - name: Read pending audit items
        run: cat vault/proyecto-tenis/audit-trail/pendientes.md
      - name: Generate brief
        run: echo "Brief generado y guardado en vault"
```

---

## CONEXIONES OCULTAS INTEGRADAS — LO QUE FALTÓ EN LA PRIMERA VERSIÓN

La primera versión de este documento usó Graphify y codebase-memory-mcp pero no conectó dos piezas que resuelven directamente el costo de tokens de un pipeline de 62 módulos, ni resolvió qué herramienta de memoria por hooks usar en este proyecto específico (distinto a la elección de Jarvis, porque aquí el objetivo es memoria de código, no memoria conversacional).

### Conexión A — Compresión de tokens aplicada a las sesiones de auditoría

Cada sesión de auditoría de nodo (Nodo-32 a Nodo-40) implica que Claude Code lea múltiples módulos Python completos más el output de pytest (1585 tests). Sin compresión, el output crudo de `pytest -v` de una suite de ese tamaño consume miles de tokens solo en listar tests que pasan.

**Tamp** (proxy de compresión, ~52.6% de reducción en tokens de entrada) es directamente aplicable aquí: se activa apuntando `ANTHROPIC_BASE_URL` a `http://localhost:7778` antes de correr Claude Code sobre este repo, sin modificar ningún test ni script del pipeline. Comprime específicamente el ruido de logs de build y salidas de comandos repetitivas — exactamente lo que genera `python -m pytest tests/ --no-cov -q` en cada verificación de Sprint 1, Sprint 2 y la auditoría kelly_kl.

Instalación (agregar al Sprint 1, no requiere tocar el pipeline):
```bash
claude plugin marketplace add sliday/claude-plugins
claude plugin install tamp@sliday
# En ~/.claude/settings.json del proyecto:
# "env": { "ANTHROPIC_BASE_URL": "http://localhost:7778" }
```

Esto no cambia el comportamiento de las verificaciones — REGLA-T53 y el protocolo de baseline/post-cambio de pytest siguen aplicando exactamente igual. Solo reduce el costo de cada sesión.

### Conexión B — Cuál herramienta de memoria por hooks usar aquí (distinta a Jarvis)

FABLE_01 recomienda `claude-mem` para Jarvis por su integración con OpenClaw. Ese razonamiento NO aplica aquí — el pipeline de tenis no corre sobre OpenClaw, corre directo en Claude Code sobre el repo. Para este proyecto la prioridad es que la memoria conecte con el vault de Obsidian como audit trail (ya diseñado en este documento) sin abrir un segundo sistema de almacenamiento paralelo (SQLite+Chroma de claude-mem) que compita con el propio `graph.json` de Graphify y el SQLite de codebase-memory-mcp.

Recomendación: **claude-memory-compiler**, no claude-mem, para este proyecto específico. Razón concreta: su output son artículos markdown organizados por concepto, sin base de datos adicional — encaja directamente en `~/vault/proyecto-tenis/audit-trail/` sin agregar un cuarto motor de almacenamiento (ya hay grafo AST de Graphify, grafo SQLite de codebase-memory-mcp, y el vault en markdown). Añadir SQLite+Chroma de claude-mem sería un quinto sistema de persistencia para el mismo dato.

División de responsabilidades resultante (ya no hay solapamiento):
- **Graphify** → mapa estructural del código (qué depende de qué) — pasada AST + semántica
- **codebase-memory-mcp** → consultas Cypher, ADRs, detección de código muerto, impacto de cambios sin commitear
- **claude-memory-compiler** → compila las sesiones de auditoría (qué se decidió, qué se probó, qué lección quedó) en artículos markdown que van directo al `audit-trail/` del vault
- **Tamp** → capa transversal que comprime tokens en las tres anteriores, no compite con ninguna

### Conexión C — Cypher queries adicionales para la auditoría kelly_kl (lo que faltó especificar)

La sección de codebase-memory-mcp ya menciona consultas tipo Cypher pero no dio ejemplos aplicados al bug específico. Fable debe usar estas antes de la auditoría forense:

```cypher
MATCH (f:Function)-[:CALLS]->(g:Function) 
WHERE g.name = 'kelly_kl' 
RETURN f.name, f.file, f.line_number

MATCH (f:Function {name: 'betslip_registrar'})-[:WRITES]->(field) 
WHERE field.name CONTAINS 'kelly' 
RETURN f.name, field.name, field.type

MATCH path = (entry:Function)-[:CALLS*1..5]->(target:Function {name: 'kelly_kl'}) 
RETURN path
```

Estas tres consultas, combinadas con `graphify path "edge_calculator" "betslip_registrar"`, le dan a Fable el camino exacto de datos sin necesidad de releer los 62 módulos completos — reduciendo el contexto que Sonnet necesita cargar antes de proponer el fix.

### Conexión D — ADR como hash-chain: persistencia inmutable de decisiones

`cbm manage_adr` ya está documentado, pero falta la razón de por qué usarlo en vez de simplemente escribir en el CLAUDE.md: los ADRs de codebase-memory-mcp son registros inmutables — una vez escrito "bug kelly_kl=0.0 resuelto: [causa]", ese registro no se edita, se agrega uno nuevo si la comprensión cambia. Esto es lo que evita el problema descrito en el Vacío 3 original (el vault contradice al CLAUDE.md): el ADR es la fuente de verdad histórica inmutable, el CLAUDE.md y el vault son vistas resumidas que pueden regenerarse a partir del historial de ADRs sin perder la secuencia de decisiones.

---

## RONDA 2 — TEST-TIME COMPUTE: CONEXIONES OCULTAS PROFUNDAS (Hermes, MCP, n8n)

Esta sección es la respuesta a la petición explícita de validar Hermes (ya instalado en Docker), MCP, plugins y n8n contra el pipeline REAL — no genérico. Cada recomendación está anclada a un módulo, un bug, o una constante hardcodeada que ya existe en el CLAUDE.md real de 714 líneas, no a una idea abstracta. El criterio de inclusión fue estricto: si la herramienta no conecta con un archivo, un Nodo, o una regla ya escrita en el CLAUDE.md, no entra aquí.

### A — Validación de Hermes (Docker) — dónde SÍ aporta y dónde NO

Hermes (familia Nous Research, corre local vía Ollama u otro runtime OpenAI-compatible) es un modelo con salida JSON estructurada confiable y function-calling nativo, entrenado sobre Llama — no es un modelo de razonamiento de frontera, es un modelo barato y rápido para tareas de clasificación/extracción repetitivas. Esa es exactamente su utilidad aquí: no reemplaza a Claude en ninguna decisión de apuesta, reemplaza trabajo mecánico que hoy consume tokens de Claude sin necesitarlo.

**Uso real 1 — Resolución de entidades para el caso Andy/Avery Nguyen (pendiente documentado en el CLAUDE.md real).**
El proyecto ya construyó `core/player_registry.py` (Nodo-51 F0) como la fuente única de verdad para entity resolution — ese módulo "absorbe clase entera de bugs tipo Nodo-47". El caso Nguyen es exactamente el tipo de ambigüedad que ese registry existe para resolver, pero resolverlo manualmente contra 706+ registros históricos + los nuevos que entran cada día es trabajo de clasificación masiva, no de razonamiento.

Patrón propuesto: Hermes local, con schema JSON forzado (`{"jugador_id_canonico": str, "confianza": float, "evidencia": [str]}`), corre sobre cada nuevo nombre ambiguo que entra por `extraer_historh2h.py` ANTES de que llegue a `PlayerRegistry`. Si `confianza >= 0.85`, se resuelve automático y silencioso. Si `confianza < 0.85`, se deja el flag para que una sesión de Claude Code (no Hermes) lo resuelva con criterio real — exactamente el patrón de "modelo barato filtra, modelo caro decide en el residual" que ya usan sistemas de multi-model routing en producción.

```bash
ollama pull finalend/hermes-3-llama-3.1:8b-q4_K_M
# CRÍTICO: Ollama por defecto usa ventana de contexto de 4096 tokens — insuficiente
# para pasarle el historial de partidos de un jugador. Configurar explícitamente:
# OLLAMA_CONTEXT_LENGTH=32768 en el modelfile o vía API request
```

**Uso real 2 — Tercera capa de fallback para scraping cuando el DOM está "frágil" (palabra textual del CLAUDE.md sobre el modo Playwright).**
El pipeline ya tiene dos modos para H2H: API (Ninja, ~0.5s/partido) y Playwright (fallback, ~30 min para 80 partidos, "DOM frágil"). Cuando Playwright trae HTML mal formado o inconsistente (`scraping/data_parser.py`), hoy ese HTML pasa por un parser de reglas fijas que se rompe con cualquier cambio de estructura del sitio. Hermes con schema JSON forzado puede actuar como un parser de HTML tolerante a variaciones — se le pasa el fragmento de HTML crudo y el schema esperado, y devuelve JSON estructurado sin que el operador tenga que reescribir el parser cada vez que FlashScore cambia su DOM. Esto es una tercera capa, no reemplaza ni al modo API ni a `data_parser.py` — se activa solo cuando el parser de reglas fijas falla.

**Uso real 3 — Pre-filtro de calidad para el dataset ML (Nodo-41).**
El proyecto ya documentó que el 69% del dataset viejo estaba contaminado por trazabilidad rota — exactamente el tipo de problema que un filtro barato de clasificación puede prevenir antes de que ocurra otra vez. Cada registro nuevo que entra a `generar_dataset_plus.py` puede pasar por Hermes con un schema `{"trazabilidad_valida": bool, "razon": str}` antes de aceptarse en el dataset — sin gastar tokens de Claude en una tarea de clasificación binaria repetitiva sobre miles de registros.

**Dónde NO usar Hermes — línea roja explícita:**
- Nunca en el cálculo de Kelly-KL, Portfolio Kelly, VaR, o cualquier fórmula cuantitativa — eso es código determinista en Python, no una tarea de lenguaje natural.
- Nunca en la decisión final de `apostar=True/False` — esa decisión vive en `edge_calculator.py` y `trader_ev_tenis.py`, con guards de no-ruina (REGLA-HF-1, REGLA-HF-5) que no deben depender de un LLM local sin las garantías de determinismo que el pipeline actual tiene.
- Nunca para generar hipótesis nuevas tipo H60-01 — el proyecto ya tiene una disciplina de pre-registro de hipótesis (`validation/preregistered_hypotheses.json`) que depende de razonamiento humano+Fable, no de clasificación automática.

### B — MCP: qué instalar y qué NO instalar (y por qué la diferencia importa)

**Playwright MCP oficial (Microsoft) como reemplazo del fallback casero — sí instalar.**
El CLAUDE.md real describe el modo Playwright como fallback con "DOM frágil" y ~8 minutos de extracción. El Playwright MCP oficial no usa screenshots ni selectores CSS frágiles — opera sobre el árbol de accesibilidad de la página, lo que en la práctica genera selectores mucho más estables ante cambios menores de diseño del sitio. Esto ataca directamente la palabra "frágil" que el propio proyecto usa para describir su punto débil actual.

```bash
claude mcp add playwright -- npx -y @playwright/mcp@latest
```

Con Browserless como backend (opcional, de pago) se puede además resolver el problema de bloqueo por scraping repetitivo — modo stealth con huella de navegador falsificada, proxy residencial y resolución automática de CAPTCHA a nivel de infraestructura, sin que `browser_manager.py` tenga que implementar ninguna de esas técnicas a mano:
```bash
claude mcp add playwright-stealth -- npx -y @playwright/mcp@latest \
  --cdp-endpoint "wss://production-sfo.browserless.io/stealth?token=TOKEN&solveCaptchas=true"
```
Esto reemplaza SOLO la ruta de fallback — el modo API (`extraer_partidos_api.py`, `--api-mode` en H2H) sigue siendo la ruta primaria sin cambios, tal como ya está documentado. No tocar `config.py` ni las constantes de tier.

**MCP de finanzas cuantitativas (QuantConnect-MCP, maverick-mcp, etc.) — NO instalar, pero SÍ adoptar la técnica que usan.**
Estos servidores exponen herramientas de PCA y test de cointegración (Engle-Granger) para validar si dos activos están correlacionados de verdad o solo lo parecen. El pipeline de tenis ya tiene constantes de correlación ρ hardcodeadas por tier (grand_slam=0.25, atp1000=0.20, atp500=0.15, challenger=0.10, itf=0.05) usadas en Portfolio Kelly — pero esos valores están asumidos, no medidos contra los resultados reales acumulados en el shadow book.

Conexión oculta real: con `reports/shadow_book/sb_YYYY-MM-DD.jsonl` acumulando resultados settled (ya hay 54 registros solo del caso GCS), hay suficiente historial para correr un test de correlación real entre picks del mismo torneo/sesión y ver si ρ=0.25 en Grand Slam es conservador, optimista, o correcto. Esto no requiere instalar un MCP de trading — es un script de análisis en Python (`scipy.stats` o `statsmodels` para el test de cointegración) que Sonnet puede escribir directamente sobre los datos que el proyecto ya genera. Fable debe proponer esto como un Nodo nuevo: "Nodo-6X: validación empírica de ρ por tier contra shadow book acumulado" — siguiendo exactamente la misma disciplina de pre-registro de hipótesis que ya usa el proyecto (ver H60-01 como plantilla).

**El patrón de artefactos de CBT Framework — adoptar la estructura, no el software.**
Existe un framework de backtesting para Claude Code (no específico de tenis, es para trading de mercados) que estructura cada sesión de investigación en una cadena de archivos: `IDEA.md → DISCOVERY.md → RESEARCH.md → EDA.md → BUILD_PLAN.md → REPORT.md → DEEP_ANALYSIS.md`. Esto es, en esencia, lo que cada Nodo del proyecto de tenis ya hace de forma informal dentro del CLAUDE.md (contexto → causa raíz → fix → tests → resultado). La conexión oculta es formalizar esa cadena como plantilla de archivos en `.spec/01_Nodos/` en vez de que cada Nodo tenga un formato ligeramente distinto — esto es lo que hace posible delegar un mes completo de trabajo a Sonnet sin que Fable tenga que revisar cada Nodo desde cero: cada Nodo nuevo sigue la misma plantilla de 7 archivos, predecible para cualquier sesión de Claude Code que lo retome.

### C — n8n: automatizar solo lo que hoy depende de que el operador se acuerde de correrlo a mano

El CLAUDE.md real revela un punto débil operativo concreto: PASO 5.5 (`shadow_book.py --close-snapshot`) debe correr "~15 min ANTES del inicio de cada partido" para capturar la cuota de cierre y calcular CLV real — y el texto dice explícitamente "sin este paso, CLV se calcula solo con cuota de entrada (menos preciso)". Esto es un paso manual con ventana de tiempo estrecha, exactamente el tipo de tarea que n8n resuelve mejor que un GitHub Action de cron fijo, porque la hora de inicio de cada partido de tenis varía día a día.

Flujo n8n concreto (nuevo, no estaba en la versión original de este documento):
```
Trigger: n8n Schedule Trigger cada 10 minutos durante horas de partidos del día
  → HTTP Request: lee reports/daily_brief_FECHA.txt o data/zita_tennis_matches_FECHA.json
    (generado por run_daily.py, ya existe — n8n solo lo LEE, no lo genera)
  → Function node: calcula para cada partido si faltan 15±5 minutos para el inicio
  → Si sí: Execute Command node → `python3 shadow_book.py --close-snapshot`
  → Telegram: confirma "Snapshot de cierre capturado para [partido]"
```

Esto es responsabilidad exclusiva de TIMING — n8n no decide nada, no calcula nada, solo dispara un comando que ya existe en el momento correcto. Cero riesgo de que n8n invente lógica de apuesta.

**Regla explícita que Fable debe escribir en el SPEC.md: n8n NUNCA ejecuta `betslip_registrar.py` ni ningún comando que registre o confirme una apuesta real.** El flujo de apuesta real sigue siendo 100% manual vía el bookmarklet descrito en PASO 4.6 — n8n solo automatiza lectura de estado, alertas, y captura de snapshots de datos de solo lectura (PASO 5.5 y PASO 9 son ambos READ-ONLY por diseño del propio proyecto). Esto es consistente con el hecho de que el propio `pipeline_tracker.py` ya se documenta como "observabilidad READ-ONLY" — n8n hereda esa misma restricción de diseño, no la contradice.

Segundo flujo, más simple — alerta de ventana de vida del activo: la Visión del proyecto trata cada partido como "un activo financiero con vida útil de 2-3 horas". n8n puede correr un Schedule Trigger que revise `reports/trader_plan_FECHA.json` y mande un Telegram al operador exactamente cuando un pick con `apostar=True` entra en su ventana de ejecución óptima — hoy esto depende de que el operador revise el archivo manualmente.

### D — Plan de delegación de un mes completo para Sonnet (Claude Code)

Con las tres piezas anteriores confirmadas, este es el calendario de 4 semanas que Fable debe convertir en Nodos formales con la plantilla de 7 archivos de la Conexión B:

**Semana 1 — Fundacional (Graphify + codebase-memory-mcp + Hermes entity resolution)**
- Días 1-2: Graphify + `.graphifyignore` + hook git (ya especificado en Sprint 1 original)
- Días 3-4: Hermes local instalado, integrado como pre-filtro en `extraer_historh2h.py` para el caso Nguyen, con `confianza < 0.85` derivando a revisión manual
- Día 5: verificar que `python -m pytest tests/ --no-cov -q` sigue en 1659 passed (baseline real, no 1585) + git log del bug kelly_kl para confirmar si Sprint 3 original sigue vigente

**Semana 2 — Observabilidad + n8n**
- Días 1-2: flujo n8n de captura de closing snapshot (PASO 5.5 automatizado)
- Días 3-4: flujo n8n de alerta Telegram de ventana de ejecución de picks `apostar=True`
- Día 5: Playwright MCP instalado como fallback, probado contra un caso real donde el modo API falló

**Semana 3 — Validación empírica de constantes hardcodeadas**
- Días 1-3: Nodo nuevo de validación de ρ por tier contra shadow book acumulado (test de correlación/cointegración en Python, sin MCP de trading) — hipótesis pre-registrada siguiendo plantilla H60-01
- Días 4-5: si hay evidencia suficiente (mismo criterio de n≥30 que ya usa el proyecto), ajustar ρ en `trader_ev_tenis.py`; si no, dejar constante y documentar por qué

**Semana 4 — Consolidación y vault**
- Días 1-2: migración del historial del CLAUDE.md al vault de Obsidian (Vacío 2 original) usando claude-memory-compiler
- Días 3-4: reconciliar cualquier contradicción detectada entre vault y CLAUDE.md (Vacío 3 original)
- Día 5: reporte de mes — qué Nodos se cerraron, qué queda pendiente, actualizar CLAUDE.md real con el estado nuevo

Este plan no reemplaza el trabajo diario del pipeline (PASO 0 a PASO 10 siguen corriendo todos los días en paralelo) — es trabajo de infraestructura que corre por las tardes/noches, sin interferir con la operación diaria de 45 min → 7 min que ya logró `run_daily.py` (D54-03).

---

## VACÍOS CRÍTICOS — FABLE DEBE RESOLVER ESTOS

### Vacío 1 — Integración de Graphify con el .graphifyignore correcto
El pipeline tiene carpetas `data/`, `reports/`, `tests/`, y archivos JSON pesados que no deben entrar al grafo. Sin un `.graphifyignore` correcto, Graphify procesará miles de archivos irrelevantes y disparará el límite de 200 archivos.

Fable debe generar el `.graphifyignore` óptimo para este proyecto específico basado en la estructura de carpetas del CLAUDE.md actual.

### Vacío 2 — Migración del historial del CLAUDE.md al vault
El CLAUDE.md tiene una sección "Lo que pasó — No debe repetirse" con 5 errores críticos documentados. Estos deben migrarse al vault como notas permanentes sin perder el contexto. Fable debe diseñar cómo hacer esta migración sin que Claude Code pierda acceso al historial durante la transición.

### Vacío 3 — Cuándo el vault contradice el CLAUDE.md
Si una nota del vault dice que el bug X está resuelto pero el CLAUDE.md dice que está activo — ¿cuál tiene razón? Fable debe diseñar la política de precedencia y cómo el agente detecta y reporta estas contradicciones.

### Vacío 4 — El bug real activo hoy no estaba en la versión original de este documento
El CLAUDE.md real lista como único bug activo: `prediccion_ganador` top-level = None en `extraer_historh2h.py`, con la corrección ya conocida (usar `ranking_analysis.prediction.favored_player`, NO el campo top-level que siempre es None). Este bug es de severidad 🟠, no crítica como kelly_kl, pero Fable debe decidir si vale la pena que Sonnet lo resuelva como parte de la Semana 1 del plan de un mes (Ronda 2, sección D) ya que el fix es conocido y probablemente trivial — probablemente un caso de eliminar el campo top-level directamente en vez de dejarlo como trampa para futuras sesiones que no lean el CLAUDE.md completo.

---

## SPEC.MD — LO QUE FABLE DEBE GENERAR PARA SONNET

### Sprint 1 — Día 1 (3 horas)
```
OBJETIVO: Graphify operativo + vault estructurado

ANTES DE EMPEZAR:
python -m pytest tests/ --no-cov -q  → confirmar 1585 passed

ARCHIVOS A CREAR:
- .graphifyignore (en raíz del proyecto)
- ~/vault/proyecto-tenis/CLAUDE.md (constitución específica)
- ~/vault/proyecto-tenis/audit-trail/pendientes.md (migración desde CLAUDE.md)
- ~/.claude/commands/tennis-audit.md
- ~/.claude/commands/tennis-session.md
- ~/.claude/commands/tennis-brief.md

COMANDOS EN ORDEN:
1. pip install graphifyy
2. graphify install
3. Crear .graphifyignore con las carpetas correctas
4. graphify ./tennis_pipeline
5. graphify ./tennis_pipeline --obsidian --obsidian-dir ~/vault/graphify/tenis/
6. graphify hook install
7. python -m pytest tests/ --no-cov -q  → confirmar sigue en 1585 passed

VERIFICACIÓN:
- graphify query "qué depende de kelly_kl" responde en <2 segundos
- graphify-out/GRAPH_REPORT.md existe y tiene contenido
- Los nodos del vault en ~/vault/graphify/tenis/ están creados
- 1585 tests siguen pasando
```

### Sprint 2 — Día 2 (2 horas)
```
OBJETIVO: Rutinas automáticas operativas

ARCHIVOS A CREAR:
- .github/workflows/tennis-pre-match-validation.yml
- .github/workflows/tennis-morning-brief.yml

COMANDOS:
1. Configurar GitHub Secrets: ANTHROPIC_API_KEY
2. Activar GitHub Actions en el repo
3. Probar manualmente: workflow_dispatch en tennis-pre-match-validation
4. Verificar que crea issue si hay anomalía

VERIFICACIÓN:
- GitHub Action corre sin errores en workflow_dispatch
- Si kelly_kl=0.0 existe en algún betslip reciente → issue creado automáticamente
```

### Sprint 3 — Auditoría kelly_kl=0.0 (sesión con Fable)
```
OBJETIVO: Resolver bug crítico pendiente

INPUT REQUERIDO PARA FABLE:
1. Contenido de betslip_index_20260623_004706.json
2. Código de edge_calculator.py (función que calcula kelly_kl)
3. Código de betslip_registrar.py (función que escribe el JSON)
4. Output de: graphify path "edge_calculator" "betslip_registrar"

FABLE HACE:
- Auditoría forense: hipótesis A vs hipótesis B
- Identifica la línea exacta donde el bug ocurre
- Propone el fix con tests correspondientes

SONNET HACE:
1. python -m pytest tests/ --no-cov -q  → baseline
2. Implementa el fix propuesto por Fable
3. python -m pytest tests/ --no-cov -q  → verificar sin regresión
4. Agrega test específico para el bug (REGLA-T53: invocar función real, no hardcodear)
5. cbm manage_adr "bug kelly_kl=0.0 resuelto: [descripción del fix]"
6. /tennis-session para documentar en el vault
```

---

## MÉTRICAS DE ÉXITO

El sistema está integrado correctamente cuando:
1. `graphify query "qué depende de kelly_kl"` responde en <2 segundos desde el grafo
2. Las rutinas de GitHub Actions corren a las 7pm y 6:30am sin intervención
3. Si `kelly_kl=0.0` aparece en cualquier betslip → issue en GitHub creado automáticamente en <5 minutos
4. El bug `kelly_kl=0.0` del 2026-06-23 tiene hipótesis confirmada con evidencia de código
5. La sesión de Claude Code empieza con contexto del pipeline sin releer los 62 módulos
