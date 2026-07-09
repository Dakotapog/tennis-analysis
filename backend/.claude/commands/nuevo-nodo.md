Crea el archivo spec para un nuevo nodo siguiendo el template estándar SDD.

Pide al usuario:
1. Número de nodo (ej. 60)
2. Nombre del nodo (ej. "Calibración Automática Post-Tourney")
3. Nodos relacionados para wikilinks

Luego crea `.spec/01_Nodos/Nodo-{N}-{Nombre}.md` con este template:

```markdown
# Nodo-{N} — {Nombre}

> **Wikilinks:** [[Nodo-XX-...]] | [[Nodo-YY-...]]
> **Fecha:** {fecha actual}
> **Estado:** 📋 ABIERTO
> **Deuda técnica de:** [nodo previo que dejó este pendiente, o N/A]

---

## 1. Problema

[Descripción concisa del problema que resuelve este nodo]

---

## 2. Solución Propuesta

[Descripción de la solución]

---

## 3. Implementación

### Archivos afectados
- `archivo.py` — [qué cambia]

### Cambios específicos
[Código o pseudocódigo de los cambios]

---

## 4. Tests (REGLA-T53)

- T{N}-01: [descripción — invoca función real, no hardcodea fórmula]
- T{N}-02: [...]

---

## 5. PRE-IMPLEMENTATION CHECKLIST (F-Meta)

- [ ] GIT-FIRST: `git log --all --oneline -- '*keyword*'`
- [ ] URL check: ¿la URL es de navegador o API? NO mezclar.
- [ ] knowledge-assets.md actualizado si toca scraping
- [ ] detectar_tier() en config.py — no duplicar lógica de tier
- [ ] REGLA-T53: tests invocan función real
```

Actualiza `.spec/MOC-Principal.md` añadiendo el nuevo nodo al índice con estado ABIERTO.
