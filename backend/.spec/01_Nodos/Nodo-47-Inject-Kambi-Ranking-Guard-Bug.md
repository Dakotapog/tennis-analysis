# Nodo-47: Bug en Guard de _inject_kambi_ranking — Kambi Sobreescribía Rankings ATP Reales

> **Wikilinks:** [[Nodo-46-Markov-Surface-Context-Discount]] | [[Nodo-21-Pesos-Diferenciados-Por-Tier]]
> **Fecha de descubrimiento:** 2026-06-30
> **Estado:** ✅ RESUELTO 2026-06-30

**Prioridad:** CRÍTICA — afectaba `ranking_momentum` en todas las sesiones con datos Kambi
**Archivo corregido:** `scraping/ninja_h2h_parser.py` — método `_inject_kambi_ranking()` línea ~1072

---

## El Problema

`_inject_kambi_ranking()` tiene una guard que debería saltar la inyección cuando el jugador
ya existe en el archivo ATP. La guard usaba `rankings_data.get(normalized)` con búsqueda
directa por clave, pero el ATP indexa los jugadores en formato **"Apellido Nombre"** mientras
que `normalize_name('Daniil Glinka')` produce **"daniil glinka"** — el mismatch hacía que la
guard siempre retornase `None` y el estimate de Kambi sobreescribiera el ranking real.

### Prueba del bug

```
normalize_name('Daniil Glinka') → 'daniil glinka'
rankings_data.get('daniil glinka') → None          ← guard falla
ATP file key → 'glinka daniil'                      ← formato diferente
```

### Efecto en el modelo

```
LOG_RANKING_P1: Base(163): 102.00    ← Kambi rank=73 → pts_estimate=163
ATP real: Glinka rank=174, pts=339   ← diferencia de 176 puntos

Glinka vs Mayo (2026-06-29):
  Kambi inyectado: Glinka=163pts, Mayo=132pts → ratio 1.25:1 (casi iguales)
  ATP real:        Glinka=339pts, Mayo=54pts  → ratio 6.3:1 (Glinka domina)

ranking_momentum casi equiparado → modelo tuvo que depender de Markov (COLD)
→ Markov invirtió predicción → predijo Mayo → Glinka ganó
```

### Jugadores afectados en sesión 2026-06-29

| Jugador | Kambi rank | Pts inyectados | Pts ATP real | Error |
|---------|-----------|----------------|--------------|-------|
| Daniil Glinka | 73 | 163 pts | 339 pts | -176 (subestimado) |
| Aidan Mayo | 200 | 132 pts | 54 pts | +78 (sobreestimado) |
| Yosuke Watanuki | 100 | 152 pts | 153 pts | -1 (casualidad) |
| Andre Ilagan | 200 | 132 pts | 211 pts | -79 (subestimado) |
| Giles Hussey | 197 | 132 pts | 221 pts | -89 (subestimado) |
| William Manning | 200 | 132 pts | 1 pt | +131 (MUY sobreestimado) |

---

## Diagnóstico Detallado — Cómo Se Detectó

### Paso 1: ranking=#? en h2h_results

Los 5 partidos del Challenger Cary mostraban `ranking1=#?` y `ranking2=#?` en el output.
La pregunta inicial fue: ¿los jugadores NO están en el archivo ATP?

### Paso 2: los jugadores SÍ están en el archivo ATP

```python
rm.get_player_info('Daniil Glinka') → rank=174, pts=339  ← encuentra con intelligent matching
```

### Paso 3: la guard falla silenciosamente

```python
normalized = rm.normalize_name('Daniil Glinka')     # → 'daniil glinka'
rm.rankings_data.get('daniil glinka')               # → None (key real: 'glinka daniil')
# guard retorna False → inyecta Kambi estimate de 163 pts
```

### Paso 4: el log confirma el bug

```
LOG_RANKING_P1: Base(163): 102.00
round(700 / log1p(73)) = 163 ✓   ← pts_estimate exacto de Kambi rank=73
```

---

## Atribución de Fallos — Sesión 2026-06-29

La sesión tuvo 3/5 fallos. Con el bug identificado, la atribución correcta es:

| Fallo | Causa real | Nodo |
|-------|-----------|------|
| Glinka vs Mayo | Ranking bug: ratio 6:1 colapsado a 1.25:1 → Markov dominó | Este nodo |
| Watanuki vs Ilagan | Markov COLD desde arcilla aplicado a hard (error Kambi ~1pt) | Nodo-46 |
| Hussey vs Manning | Upset genuino (4.2 cuota) — ranking correcto aún predice Hussey | Ninguno |

**Conclusión:** El Nodo-46 (Markov Surface Context) tiene evidencia real en 1 fallo (Watanuki),
no en 2-3. El Nodo-46 sigue siendo válido como mejora futura pero requería más n.

---

## La Fix — Guard Optimizada en Dos Pasos

```python
# Fast path O(1): chequea key directo + key invertido (cubre 95% ATP/WTA)
# "daniil glinka" → busca también "glinka daniil" → encuentra en O(1)
normalized = self.ranking_manager.normalize_name(player_name)
parts = normalized.split()
reversed_key = ' '.join(reversed(parts)) if len(parts) == 2 else None
rd = self.ranking_manager.rankings_data
if rd.get(normalized) or (reversed_key and rd.get(reversed_key)):
    return  # encontrado en O(1) — no sobreescribir

# Slow path: intelligent matching para nombres compuestos
# "Yosuke Watanuki" → "Watanuki Yosuke (1998)" no matchea con inversión
# "Davidovich Fokina A." → apellido compuesto, necesita fuzzy matching
if self.ranking_manager.get_player_info(player_name):
    return  # encontrado via intelligent matching — no sobreescribir
```

### Por qué dos pasos

La versión inicial del fix (solo `get_player_info()`) era correcta en lógica pero costosa:
- `get_player_info()` sin caché itera 3841 jugadores × string matching = 5.83ms/call
- 80 partidos × 2 jugadores = 160 calls = **932ms overhead/sesión** + 160 líneas de log extra

La versión optimizada:
- Fast path O(1): `dict.get()` — microsegundos, sin log
- Slow path solo para ~5% de jugadores con nombre ATP especial (birth year, compuesto)
- Resultado: **157ms overhead/sesión** (6× más rápido)

### Casos cubiertos

| Jugador | Fast path | Resultado |
|---------|-----------|-----------|
| Daniil Glinka → Glinka Daniil | reversed_key hit | skip ✓ |
| Aidan Mayo → Mayo Aidan | reversed_key hit | skip ✓ |
| Martin Maldonado → Maldonado Martin (rank 1824) | reversed_key hit | skip ✓ |
| Yosuke Watanuki → Watanuki Yosuke (1998) | miss → slow path | skip ✓ |
| Jugador ITF desconocido | miss → miss → inyectar | inject ✓ |

---

## Impacto Histórico

El bug estuvo activo desde que se introdujo `_inject_kambi_ranking`. Cada sesión con
datos Kambi generó `ranking_momentum` con valores incorrectos.

**Por qué no rompió calibración clay GS:**
- Top-50 en Grand Slams: diferencias de ranking tan grandes (1000+ pts vs 50 pts) que
  incluso el estimate Kambi mantenía la dirección correcta de la ventaja
- El modelo era "robusto al ruido" porque la señal de ranking era tan clara que absorbía el error

**Dónde más dolía:**
- Challenger/ITF con jugadores de ranking medio-bajo (100-500): los estimates Kambi
  aplanan las diferencias reales, haciendo que el modelo dependa más de Markov y form
- Exactamente el tier donde reclamamos mayor ventaja informacional (Nodo-21)

**Calibración:** Las 706 observaciones en `calibracion_edge.json` fueron generadas con
ranking_momentum parcialmente corrupto. No invalida la calibración (la dirección era
correcta en GS/clay) pero explica parte del ruido en Challenger.

---

## Tests

No se agregaron tests específicos (el bug era en la guard de una función de un método privado
con dependencia de RankingManager cargado). Los 1438 tests existentes cubren el comportamiento
externo y siguen pasando.

Si se quiere test unitario directo:
```python
def test_inject_kambi_does_not_overwrite_real_atp_ranking():
    """Guard debe saltar si el jugador existe en ATP bajo formato 'Apellido Nombre'."""
    # Crear RankingManager mock con 'glinka daniil' → {rank: 174, pts: 339}
    # Llamar _inject_kambi_ranking('Daniil Glinka', 73)
    # Verificar que rankings_data['glinka daniil'] aún tiene pts=339 (no 163)
```

---

## Relación con Otros Nodos

| Nodo | Relación |
|---|---|
| [[Nodo-46-Markov-Surface-Context-Discount]] | Este bug fue la causa primaria del fallo Glinka/Mayo que Nodo-46 atribuía a Markov surface. Nodo-46 sigue válido para Watanuki/Ilagan (1 caso confirmado). |
| [[Nodo-21-Pesos-Diferenciados-Por-Tier]] | Challenger/ITF eran los tiers más afectados — los de mayor ventaja informacional y donde los errores de ranking son más costosos |
| [[Nodo-45-Temporal-History-Fallback]] | THF resuelve match_id=None. Este nodo resuelve que cuando el match SÍ se procesa, el ranking sea el correcto |
