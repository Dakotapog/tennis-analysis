# Nodo-37 — Combo Confianza Builder: Portfolio de Combos Basado en Señal de Confianza

> **Fecha inicio:** 2026-06-25
> **Severidad:** OPORTUNIDAD — El pipeline decía "0 apuestas" pero la señal de confianza ≥53% tenía 95.2% accuracy real ese mismo día. El capital estaba disponible y no se desplegó.
> **Prerequisitos:** Nodo-15 (Cobertura por Exclusión), Nodo-32 (Phantom Edge), Nodo-35 (Historial Vacío Flag)
> **Archivos nuevos:** `combo_confianza_builder.py`
> **Archivos NO modificados:** `edge_calculator.py`, `trader_ev_tenis.py`, `betplay_combo_builder.py`
> **Implementa:** Sonnet | **Tests:** pendiente
>
> **Estado:** SUPERSEDED por [[Nodo-38-Portfolio-Aislamiento-Riesgo]] — arquitectura CORE/Satellite/Moonshot reemplaza progresión C5→C20

---

## 0. RESUMEN EJECUTIVO

El pipeline principal (`edge_calculator.py` → `trader_ev_tenis.py`) opera sobre señal de **edge Kelly-KL**: solo apuesta cuando `P_modelo - P_implícita > 5%` y `Kelly-KL > 2%`. El 25-jun-2026 ese threshold no se alcanzó en ningún partido → output: "0 apuestas".

Sin embargo, ese mismo día se validó empíricamente:

```
Picks con confianza ≥ 53%: 21 partidos verificados
Aciertos: 20 | Fallos: 1
Accuracy: 95.2%

Picks con confianza < 53%: 17 partidos verificados
Aciertos: 10 | Fallos: 7
Accuracy: 58.8%
```

El único fallo ≥53% fue Bronzetti (55.1%) — perdió 0-2. Los 7 fallos estaban todos bajo 53%.

**La señal existe. El pipeline la ignora porque el gate Kelly-KL es demasiado estricto para días sin cuotas extremas.** `combo_confianza_builder.py` es la capa paralela que explota esta señal directamente.

---

## 1. HALLAZGO QUE MOTIVÓ ESTE NODO

### 1.1 El problema del combo de 15-19 piernas

El usuario apostó manualmente $500 × 4 combos = $2,000. Resultado: $0 retornado.

Causa estructural: combos de 15-19 piernas con heavy favorites (cuotas 1.12-1.29). Un solo fallo (Zeitune 51.7%) destruyó todas las apuestas. El sistema de cobertura por exclusión (Nodo-15) no se usó.

### 1.2 La oportunidad perdida

Con filtro confianza ≥53% y combos de 5-8 piernas con cobertura:
- 20 de 21 picks eran ganadores
- Combos C5 y C7 habrían ganado aun con el fallo de Bronzetti (excluida en cobertura)
- P&L positivo garantizado con la estructura correcta

### 1.3 Por qué esto es paralelo al pipeline principal y NO lo reemplaza

| Sistema | Señal | Gate | Cuándo aplica |
|---|---|---|---|
| `edge_calculator` → `trader` | Edge Kelly-KL vs bookmaker | `P_modelo - P_implícita > 5%` | Días con cuotas ineficientes |
| `combo_confianza_builder` | Confianza del modelo ≥53% | Accuracy empírica validada | Todos los días con ≥5 picks |

No son competidores. El edge pipeline es más preciso cuando hay señal fuerte. El combo confianza es la capa base que siempre tiene picks.

---

## 2. DESCUBRIMIENTO CLAVE: MODELO CONSERVADOR EN CONFIANZA

El sistema calcula P(combo gana) = ∏P(pick_i gana) usando las confianzas individuales del modelo. Para un C7 con picks entre 61-95% de confianza, P(C7) ≈ 11.4%.

Pero la accuracy empírica observada es **95.2% por pick** (no 61-95%). Esto significa que el modelo es sistemáticamente conservador en sus estimados. La confianza de 61% puede reflejar incertidumbre del modelo, no la probabilidad real de acierto.

Con accuracy empírica 95.2%:
- P(C5) = 0.952^5 = **77.4%**
- P(C7) = 0.952^7 = **71.7%**
- P(C9) = 0.952^9 = **64.7%**

Esto cambia radicalmente el EV de los combos. El parámetro `--p-empirica` permite usar la accuracy histórica calibrada en lugar de las confianzas del modelo.

**ADVERTENCIA:** La accuracy del 95.2% viene de n=21 en un solo día. Requiere acumulación histórica antes de ser confiable. El default del script sigue siendo las confianzas del modelo (conservador). `--p-empirica` es opt-in explícito.

---

## 3. DISEÑO DEL SISTEMA

### 3.1 Tiers de confianza

```
DIAMOND: ≥75% — incluir en TODOS los combos (C5+)
GOLD:    65-75% — incluir desde C5
SILVER:  57-65% — incluir desde C7
BRONZE:  53-57% — incluir desde C9, candidatos prioritarios a exclusión en cobertura
```

### 3.2 Portfolio progresivo con stake decreciente

```
C5:  Top 5 picks (DIAMOND+GOLD)  → stake_base × 1.00  ($20,000)
C7:  Top 7 picks                 → stake_base × 0.75  ($15,000)
C9:  Top 9 picks                 → stake_base × 0.50  ($10,000)
C11: Top 11 picks                → stake_base × 0.35  ($7,000)
C13: Top 13 picks                → stake_base × 0.25  ($5,000)
C15: Top 15 picks                → stake_base × 0.15  ($3,000)
C20: Top 20 picks                → stake_base × 0.05  ($1,000)
```

**Lógica decreciente:** A más piernas, mayor varianza. Los stakes grandes van a los combos más cortos donde P(win) es mayor.

### 3.3 Cobertura por exclusión (extensión de Nodo-15)

Para cada tamaño de combo:
- Generar hasta 5 combos de cobertura que excluyen 1-2 picks de MENOR confianza dentro del pool
- El pick excluido es reemplazado por el siguiente pick del pool de reserva
- Stake por cobertura = stake_principal × 0.25 / n_combos_cobertura
- Garantía: si el pick excluido falla, ese combo de cobertura gana

```
Ejemplo C7 con pool de 10 picks:
  Principal:   [P1, P2, P3, P4, P5, P6, P7]  (top 7 por confianza)
  Cobertura A: [P1, P2, P3, P4, P5, P6, P8]  (excluye P7, entra P8)
  Cobertura B: [P1, P2, P3, P4, P5, P7, P8]  (excluye P6, entra P8)
  Cobertura C: [P1, P2, P3, P4, P5, P8, P9]  (excluye P6+P7, entran P8+P9)
```

### 3.4 Filtros de seguridad

```python
--threshold 53    # confianza mínima (default 53%)
--min-cuota 1.10  # cuota mínima del favorito (default 1.10)
                  # Picks con cuota < 1.10 no aportan odds suficientes al combo
--p-empirica 0.0  # si > 0, usar esta accuracy histórica para calcular EV
                  # (default 0 = usar confianzas individuales del modelo)
```

**REGLA COMBO-1:** Picks con cuota < 1.10 se excluyen del pool aunque tengan confianza ≥53%. Reducen el odds total del combo sin aportar valor esperado suficiente.

**REGLA COMBO-2:** No construir un tamaño de combo si no hay suficientes picks elegibles. C5 requiere ≥5 picks, C7 requiere ≥7, etc.

**REGLA COMBO-3:** El C5 (spine) solo incluye DIAMOND (≥75%) + GOLD (65-75%). No mezclar BRONZE en el combo de mayor stake.

---

## 4. FLUJO DE USO

```bash
# PASO 1 — Pipeline normal de mañana
python3 extraer_partidos_api.py
python3 extraer_historh2h.py --api-mode --all-tournaments

# PASO 2 — Revisar tabla (opcional, para validación humana)
python3 generar_tabla_favoritos2.py

# PASO 3 — Construir combos de confianza (NUEVO — Nodo-37)
python3 combo_confianza_builder.py
# → reports/combo_plan_FECHA.txt

# Variantes:
python3 combo_confianza_builder.py --threshold 55 --min-cuota 1.15
python3 combo_confianza_builder.py --p-empirica 0.952 --stake-base 15000

# PASO 4 — (Paralelo) Si hay picks con edge Kelly-KL, correr trader también
python3 edge_calculator.py
python3 trader_ev_tenis.py --bankroll 125000 --torneo-tipo challenger
```

---

## 5. CALIBRACIÓN DE `--p-empirica`

La accuracy empírica por día debe acumularse en `data/calibracion_edge.json` bajo una nueva sección:

```json
"combo_confianza": {
    "threshold_53": {
        "n_dias": 1,
        "n_picks_total": 21,
        "n_aciertos": 20,
        "accuracy": 0.952,
        "ultima_actualizacion": "2026-06-25"
    }
}
```

Cuando `n_picks_total >= 100`, la accuracy es confiable para usar como `--p-empirica`. Con n=21 (un día), es indicativa pero no estadísticamente robusta (IC 95%: 75%-99%).

---

## 6. PENDIENTES (scope de este nodo)

| # | Acción | Estado |
|---|---|---|
| 1 | `combo_confianza_builder.py` base — portfolio progresivo + cobertura + min-cuota | ✅ Implementado |
| 2 | Parámetro `--p-empirica` para usar accuracy histórica en cálculo de EV | 🔄 En curso |
| 3 | Acumular accuracy en `calibracion_edge.json` sección `combo_confianza` | ⏳ Pendiente |
| 4 | Tests `tests/test_nodo37.py` — mutación en filtros, construcción de combos, cobertura | ⏳ Pendiente |
| 5 | Integrar con `betslip_registrar.py` para cerrar loop de calibración | ⏳ Futuro |

---

## 7. RIESGO PRINCIPAL

**Overfitting a un día (n=21):** La accuracy del 95.2% observada el 25-jun-2026 puede ser excepcionalmente buena. Necesita validación en ≥10 días con ≥200 picks antes de ser confiable para `--p-empirica`. Hasta entonces, el default conservador (confianzas del modelo) es el correcto para stakes grandes.

**Andy Nguyen / Avery Nguyen (Nodo-36):** Ambas aparecen en el mismo torneo ITF Claremont. Si el sistema asigna el pick incorrecto entre las dos, el combo incluye el favorito equivocado. Monitorear manualmente cuando ambas juegan el mismo día.

---

## 8. WIKILINKS

- [[Nodo-15-Portfolio-Kelly-Cobertura]] — Cobertura por Exclusión C(N,K) — lógica base reutilizada
- [[Nodo-32-Calibracion-Pipeline-Señales-Rotas]] — Phantom edge gate — razón por la que el pipeline dice "0 apuestas"
- [[Nodo-35-Historial-Vacio-Flag-Pipeline]] — Gate historial vacío — picks sin historial no deben entrar al pool
- [[Nodo-36-Unicode-Acento-Apellidos-Cortos]] — Corrección de matching de nombres — afecta qué picks se extraen
- [[MOC-Principal]] — índice de specs
- [[Sprint-Pipeline]] — estado del sprint
