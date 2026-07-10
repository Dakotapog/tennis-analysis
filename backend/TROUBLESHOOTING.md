# TROUBLESHOOTING — Tennis Analysis Backend

## Claude Code no puede hablar con la API / sesion congelada

**Causa probable:** el proxy local Tamp (puerto 7778) se cayó.

`ANTHROPIC_BASE_URL=http://localhost:7778` está seteado en el entorno de Claude Code.
Sin Tamp corriendo, Claude Code NO tiene fallback automatico a api.anthropic.com — es una dependencia dura.

**Diagnostico y arreglo:**

```bash
# 1. Verificar si Tamp responde
curl -s localhost:7778/health | python3 -m json.tool

# 2. Si no responde, reiniciar
systemctl --user restart tamp
sleep 2
curl -s localhost:7778/health

# 3. Si falla el restart, ver logs
journalctl --user -u tamp -n 50

# 4. Si el servicio no existe (post-reboot perdio el unit)
tamp install-service
systemctl --user enable tamp
systemctl --user start tamp
```

**Linger habilitado** (`loginctl enable-linger mikata` — 2026-07-10): Tamp deberia sobrevivir reboots sin login interactivo.

**Si necesitas trabajar SIN Tamp temporalmente** (mientras lo arreglas):
```bash
unset ANTHROPIC_BASE_URL
# Claude Code usara api.anthropic.com directo hasta que reabras la terminal
```

---

## Graphify / Tamp no indexan archivos nuevos

```bash
# Rebuild grafo (tras cambios en .py)
graphify update .

# Rebuild indice de nodos (tras anadir Nodo-XX.md)
python3 scripts/rebuild_nodos_index.py

# Verificar consistencia CLAUDE.md vs nodos
python3 check_contradictions.py --quick
```
