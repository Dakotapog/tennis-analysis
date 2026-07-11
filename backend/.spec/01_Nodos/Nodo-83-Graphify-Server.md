# Nodo-83 — Graphify: Servidor HTTP local (:7779) + fetch() dinámico

**Fecha:** 2026-07-11
**Estado:** IMPLEMENTADO
**Rama:** main

---

## Problema

`graphify-out/graph.html` embebía los 1686 nodos y 2753 aristas directamente en el HTML
(1.4 MB). Dos consecuencias:
1. **Datos estáticos:** `graphify update .` regenera `graph.json` pero no el HTML. El grafo
   mostrado en el browser era siempre el de la última generación completa, nunca el nuevo.
2. **LEGEND vacía:** `const LEGEND = [];` en línea 71 del viejo HTML. El panel Communities
   mostraba checkboxes vacíos — funcionalidad de filtrado inoperativa.

Además, abrir `file:///` en Chrome bloquea `fetch()` por política CORS, por lo que el grafo
debía verse con `file://` embebido o via HTTP.

---

## Solución

### Archivo 1: `graphify_server.py` (nuevo)

Servidor HTTP stdlib (sin dependencias externas) en puerto **7779**:
- Sirve `graphify-out/` completo
- `graph.json` → `Cache-Control: no-cache, no-store, must-revalidate`
- Resto de archivos → `Cache-Control: public, max-age=3600`
- Logs filtrados: solo errores 4xx/5xx

### Archivo 2: `graphify-out/graph.html` (reescrito)

Reemplaza los bloques de datos embebidos por `fetch()` asíncrono:

```javascript
async function init() {
  const data = await fetch('graph.json?t=' + Date.now()).then(r => r.json());
  RAW_NODES = data.nodes;
  RAW_EDGES = data.links.map(l => ({   // graph.json usa links.source/target
    from: l.source, to: l.target, ...
  }));
  // LEGEND computado dinámicamente desde nodos (antes era [])
  ...
  document.getElementById('stats').textContent = `${RAW_NODES.length} nodes...`;
  nodesDS = new vis.DataSet(...);
  network = new vis.Network(...);
  // Todos los handlers, search, legend — dentro de init()
}
init().catch(err => { document.getElementById('stats').textContent = 'Error: ' + err.message; });
```

Cambios clave:
- `graph.json` usa `links[].source/target` → se mapea a `from/to` para vis.js
- `LEGEND` se computa desde `RAW_NODES.community` → Communities sidebar funciona
- `#stats` se actualiza con el conteo real fetched (no hardcoded)
- `focusNode` y `toggleAllCommunities` expuestos como `window.*` para handlers inline
- Ambos `<script>` bloques consolidados en uno (`hyperedges` dentro de `init()`)
- File size: 14 KB (vs 1.4 MB antes)

### Archivo 3: `~/.config/systemd/user/graphify.service`

Mismo patrón que `tamp.service`:
- `Type=simple`, `Restart=always`, `RestartSec=5`
- `WantedBy=default.target`
- `ExecStart`: venv python3 del proyecto

---

## Criterio de aceptación (verificado 2026-07-11)

```
curl -si http://localhost:7779/graph.html  → HTTP 200, 14360 bytes
curl -si http://localhost:7779/graph.json  → Cache-Control: no-cache, no-store, must-revalidate
graph.json: 1686 nodes, 2753 links
systemctl --user status graphify.service  → active (running)
```

**F5 sin regenerar**: el HTML fetched siempre `graph.json?t=<timestamp>` → bypass de
caché del browser → el grafo mostrado es siempre el de `graphify-out/graph.json` actual.

**Después de `graphify update .`**: solo `graph.json` cambia (HTML no se regenera). F5 en
`http://localhost:7779/graph.html` muestra los nuevos nodos/aristas sin paso extra.

---

## URL de acceso

```
http://localhost:7779/graph.html
```

Comandos de gestión:
```bash
systemctl --user start|stop|restart|status graphify.service
```

---

## Vinculación

- `graphify_server.py` — servidor HTTP principal
- `graphify-out/graph.html` — cliente fetch() dinámico
- `~/.config/systemd/user/graphify.service` — unit systemd
- **Fase 2 (migración 3D):** gateada — requiere confirmación explícita antes de implementar
- Nodo-75: índice de nodos — graphify.service registrado como infraestructura Nodo-83
