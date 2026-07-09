Checklist F-Meta: ejecutar ANTES de implementar cualquier nodo o feature.

Lee el nodo especificado por el usuario en `.spec/01_Nodos/` y ejecuta estos pasos en orden:

## 1. GIT-FIRST (OBLIGATORIO)

Busca si ya existe una implementación previa:
```bash
git log --all --oneline -- '*{keyword}*'
```
Si existe → recuperar con `git show COMMIT:backend/archivo.py`. NO reinventar.

## 2. Baseline de tests

```bash
python -m pytest tests/ --no-cov -q | tail -3
```
Debe mostrar 0 failed antes de empezar.

## 3. Verificar URLs

¿El nodo toca scraping? Revisar `docs/knowledge-assets.md`:
- `global.flashscore.ninja/202/x/feed` = API Ninja (JSON)
- `www.flashscore.com/tennis/` = DOM web (Playwright)
- NUNCA derivar URL de browser desde URL de API

## 4. Tier detection

¿El nodo clasifica torneos? Verificar que usa `detectar_tier()` de `config.py` — no duplicar.

## 5. REGLA-T53

¿Los tests nuevos invocan la función real del módulo? No hardcodear fórmulas en tests.

## 6. knowledge-assets.md

¿El nodo elimina o modifica un scraper? Extraer primero URLs/selectores/formatos.

Reporta el resultado de cada check (OK / REQUIERE ACCIÓN) antes de proceder con la implementación.
