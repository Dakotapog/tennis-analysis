# Nodo-162 — Redirect Page: Formato de Coupon Betplay Roto (D162-01)

**Estado:** IMPLEMENTADO (fix local, push pendiente — decisión explícita del usuario)
**Fecha:** 2026-08-02
**Módulo principal:** `docs/bp/index.html` (raíz del repo, fuera de `backend/`, servido por GitHub Pages)

---

## Contexto / Bug reportado

Usuario reportó: el link "ABRIR GamesLive" del mensaje Telegram de GAMES COMBOS
(D133-04, disparado por `betplay_combo_builder.py --games --live --telegram`)
abre Betplay pero **sin ninguna pierna cargada en el coupon** — solo la página
de inicio.

## Root cause

`docs/bp/index.html` es la página GitHub Pages a la que apuntan TODOS los links
de Telegram (`REDIRECT_BASE = "https://dakotapog.github.io/tennis-analysis/bp/?ids="`),
usada por:
- `betplay_combo_builder.py:524` (SAFE/otros combos)
- `betplay_combo_builder.py:2119` (`_enviar_games_telegram`, D157-06)
- `combo_confianza_builder.py:1796` (CORE/pre-match)

El commit `4ae668d` ("fix: bp/index.html formato coupon Betplay (ID|ML/ID|ML vs
ID,ID)", 2026-07-28) cambió deliberadamente el JS de esta página del formato
correcto (`ids` tal cual, comma-joined, pasado directo a la URL) a:

```js
var couponPart = idArr.map(function(id) { return id + '|ML'; }).join('/');
var betplayUrl = '...coupon=combination|' + couponPart + '||replace';
```

Esto produce `combination|ID1|ML/ID2|ML||replace` — **viola directamente
REGLA-BAT-1** (CLAUDE.md §9, marcada INMUTABLE): *"Formato coupon:
`combination|ID1,ID2,ID3||replace` (IDs separados por comas, SIN sufijo
`|ML/`)"*. `betplay_combo_builder.py`'s `BETPLAY_URL_BASE`/`BETPLAY_URL_TAIL`
(pipeline `.bat` local) nunca tuvieron este bug — solo la página GitHub Pages.
Betplay no puede parsear el coupon `|ML/`-separado y silenciosamente no carga
ninguna pierna — el mismo síntoma exacto reportado por el usuario.

Verificado con `git log --all -- docs/bp/index.html`: el formato comma-joined
fue el original desde `f2f5d48` (creación de la página) hasta `b880384`
(`||append`→`||replace`); `4ae668d` es el único commit que introdujo `|ML/`.
Ningún commit posterior lo revirtió — el bug lleva activo desde 2026-07-28
(~5 días), afectando **todos** los combos enviados por Telegram (no solo
GAMES live), no solo el caso ITF que motivó el reporte.

## Fix

`docs/bp/index.html` revertido al formato REGLA-BAT-1:

```js
var n = ids.split(',').length;
var betplayUrl = 'https://betplay.com.co/apuestas#home?coupon=combination|' + ids + '||replace';
```

Comentario añadido en el código citando REGLA-BAT-1 y el commit que rompió el
formato, para que una futura sesión no repita el mismo error.

## Deploy — decisión explícita del usuario

`docs/bp/index.html` se sirve vía GitHub Pages — el fix local no tiene efecto
en el link real de Telegram hasta hacer `git push`. El branch local está 213
commits adelante de `origin/main` (sin sincronizar hace tiempo). Se preguntó
al usuario cómo proceder; eligió **"No hacer push — yo lo hago manualmente"**.
El fix queda committeado localmente, sin push. Los links de Telegram
seguirán rotos hasta que el usuario publique el commit.

## Tests

`tests/test_nodo162_redirect_coupon_format.py` — test de regresión estático
(lee `docs/bp/index.html` desde disco, no requiere servidor ni browser):
confirma que el JS no contiene el sufijo `|ML` y que construye la URL con
`ids` comma-joined tal cual, per REGLA-BAT-1.

## Wikilinks

- [[Nodo-157]] — D157-06 añadió el link REDIRECT_BASE a `_enviar_games_telegram()`,
  sin saber que la página destino ya estaba rota desde 3 días antes
- CLAUDE.md §9 REGLA-BAT-1 — fuente de verdad del formato, ahora también
  aplicada a la página de redirect (antes solo mencionaba el pipeline `.bat` local)
