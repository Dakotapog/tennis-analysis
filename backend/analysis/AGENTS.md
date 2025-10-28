# 🧪 GUÍA DE TESTS - Tennis Analysis

## 🎯 OBJETIVO
Tests de calidad que validen funcionalidad sin bloquear progreso.

---

## 📊 COBERTURA: ENFOQUE PRAGMÁTICO

### ⚠️ REGLA ESTRICTA - NO NEGOCIABLE
```ini
# pytest.ini
--cov-fail-under=50  # MÍNIMO ABSOLUTO durante refactorización
```

**PROHIBIDO bajar de 50% sin autorización explícita.**

### Si Tests No Pasan con 50%:
1. ❌ **NO bajar umbral a 25% o menos**
2. ✅ **SÍ mejorar los tests:**
   - Agregar más casos de prueba
   - Usar mocks si es necesario
   - Simplificar código complejo
   - Pedir ayuda antes de bajar estándares

### Estrategia por Tipo de Código

**Código NUEVO (copiado y refactorizado):**
- Meta durante refactor: **50%+** OBLIGATORIO
- Meta final: **70%+** (antes de 2025-11-30)
- Prioridad: Tests funcionales primero, cobertura después
- Tests: Happy path + edge cases básicos + error handling

**Código LEGACY (aún en extraer_historh2h.py):**
- Meta: **0%** (no tocar hasta copiar)
- Tests: Ninguno hasta después de copiar
- Estado: INTOCABLE

**Código CRÍTICO (cálculos ELO, rankings):**
- Meta: **90%+**
- Prioridad: Máxima
- Tests: Exhaustivos desde el inicio

---

## 📝 PLANTILLA DE TEST

```python
"""Tests para NombreClase."""
import pytest
from analysis.nombre_clase import NombreClase

class TestNombreClase:
    """Suite de tests para NombreClase."""
    
    @pytest.fixture
    def instance(self):
        """Instancia limpia para tests."""
        return NombreClase(param=valor)
    
    # HAPPY PATH (caso normal)
    def test_metodo_caso_normal(self, instance):
        """Test: Operación exitosa con datos válidos."""
        result = instance.metodo(input_valido)
        assert result == esperado
    
    # EDGE CASES (límites)
    def test_metodo_valor_vacio(self, instance):
        """Test: Manejo de entrada vacía."""
        result = instance.metodo("")
        assert result == resultado_esperado
    
    def test_metodo_valor_none(self, instance):
        """Test: Manejo de None."""
        result = instance.metodo(None)
        assert result is None  # o lo que corresponda
    
    # ERROR HANDLING (errores)
    def test_metodo_raise_error(self, instance):
        """Test: Lanza excepción con input inválido."""
        with pytest.raises(ValueError, match="mensaje esperado"):
            instance.metodo(input_invalido)
```

---

## 🎨 BUENAS PRÁCTICAS

### 1. Nombres Descriptivos
```python
# ❌ Malo
def test_1(): pass
def test_method(): pass

# ✅ Bueno
def test_calculate_elo_winner_gains_points(): pass
def test_add_player_raises_error_negative_points(): pass
```

### 2. Un Assert por Test (preferido)
```python
# ❌ Evitar
def test_player_stats(player):
    assert player.name == "Nadal"
    assert player.points == 2000
    assert player.ranking == 1

# ✅ Mejor
def test_player_name(player):
    assert player.name == "Nadal"

def test_player_points(player):
    assert player.points == 2000
```

### 3. Fixtures Reutilizables
```python
@pytest.fixture
def sample_data():
    """Datos de prueba reutilizables."""
    return {
        'player1': 'Nadal',
        'player2': 'Federer',
        'score': '6-4 6-4'
    }

def test_a(sample_data):
    assert sample_data['player1'] == 'Nadal'

def test_b(sample_data):
    assert len(sample_data) == 3
```

### 4. Docstrings Claros
```python
def test_calculate_elo(self, elo_system):
    """
    Test: ELO se calcula correctamente para ganador.
    
    Dado: Dos jugadores con ratings 1500
    Cuando: Jugador 1 gana
    Entonces: Rating jugador 1 > 1500, jugador 2 < 1500
    """
```

---

## 🔍 CASOS A CUBRIR

### Mínimo por Método Público
1. **Happy path**: Caso exitoso normal
2. **Edge case**: Al menos 1 (vacío/None/límite)
3. **Error case**: Al menos 1 (input inválido)

### Ejemplo Completo
```python
class TestRankingManager:
    @pytest.fixture
    def ranking(self):
        return RankingManager()
    
    # HAPPY PATH
    def test_add_player_success(self, ranking):
        """Test: Agregar jugador exitosamente."""
        ranking.add_player("Nadal", 2000)
        assert "Nadal" in ranking.players
    
    # EDGE CASES
    def test_add_player_duplicate_updates(self, ranking):
        """Test: Jugador duplicado actualiza puntos."""
        ranking.add_player("Nadal", 2000)
        ranking.add_player("Nadal", 2500)
        assert ranking.get_points("Nadal") == 2500
    
    def test_get_player_not_found(self, ranking):
        """Test: Jugador inexistente retorna None."""
        assert ranking.get_player("Desconocido") is None
    
    # ERROR HANDLING
    def test_add_player_negative_points_raises(self, ranking):
        """Test: Puntos negativos lanzan ValueError."""
        with pytest.raises(ValueError):
            ranking.add_player("Nadal", -100)
```

---

## 🚦 EJECUTAR TESTS

### Comandos Básicos
```bash
# SIEMPRE con PYTHONPATH
PYTHONPATH=. pytest tests/ -v

# Test específico (ejemplo: RankingManager en proceso)
PYTHONPATH=. pytest tests/test_ranking_manager.py -v

# Ver detalles de fallos
PYTHONPATH=. pytest tests/test_ranking_manager.py -vv

# Con cobertura
PYTHONPATH=. pytest tests/ --cov=analysis --cov-report=term-missing

# Solo tests que fallaron antes
PYTHONPATH=. pytest tests/ --lf

# Detener en primer fallo
PYTHONPATH=. pytest tests/ -x
```

### Interpretar Resultados
```bash
# ✅ Exitoso - CONTINUAR con siguiente clase
3 passed in 0.5s
Coverage: 55%

# ❌ Fallido - CORREGIR antes de avanzar
2 passed, 1 failed
FAILED tests/test_ranking_manager.py::test_method - AssertionError

# ⚠️ Coverage bajo pero tests pasan - OK temporal
3 passed in 0.5s
Coverage: 45% (bajo pero aceptable en refactor)
```

### Flujo de Validación
```bash
# 1. Correr tests de clase actual
PYTHONPATH=. pytest tests/test_ranking_manager.py -v

# 2. Si fallan: corregir y repetir hasta pasar
# 3. Si pasan: ejecutar suite completa
PYTHONPATH=. pytest tests/ -v

# 4. Si suite pasa: avanzar a siguiente clase
# 5. Si suite falla: corregir regresiones
```

---

## 📈 ESTRATEGIA DE MEJORA

### Fase 1: Refactorización (ACTUAL)
- Cobertura: 50% mínimo
- Foco: Tests básicos funcionando
- Prioridad: Velocidad de refactorización

### Fase 2: Consolidación (Próxima)
- Cobertura: 60% objetivo
- Foco: Edge cases importantes
- Prioridad: Estabilidad

### Fase 3: Excelencia (Final)
- Cobertura: 70% requerido
- Foco: Tests exhaustivos
- Prioridad: Calidad total

**Deadline Fase 3:** 2025-11-30

---

## 🚨 CUÁNDO DETENER Y PREGUNTAR

**SIEMPRE preguntar ANTES de:**
- ❌ Bajar cobertura < 50%
- ❌ Modificar `pytest.ini`
- ❌ Cambiar estándares de calidad
- ❌ Saltarse tests que fallan

**PREGUNTAR si:**
- Clase con >10 métodos públicos
- Lógica muy compleja (múltiples if/loops anidados)
- Dependencias externas (DB, APIs)
- Tests que requieren mocks complicados
- Incertidumbre sobre qué testear
- **Tests no alcanzan 50% de cobertura**

**NUNCA hacer sin consultar:**
- Bajar umbral de cobertura
- Comentar tests que fallan
- Marcar tests como `@pytest.mark.skip`
- Cambiar configuración de pytest

**Mejor preguntar 10 veces que bajar estándares 1 vez.**

---

## ✅ CHECKLIST PRE-COMMIT

**Por cada clase refactorizada:**

- [ ] Código copiado de `extraer_historh2h.py`
- [ ] Tests escritos en `tests/test_*.py`
- [ ] `PYTHONPATH=. pytest tests/test_*.py` ejecutado
- [ ] **TODOS los tests de esta clase PASAN** (verde) ⚠️ CRÍTICO
- [ ] Suite completa ejecutada: `PYTHONPATH=. pytest tests/`
- [ ] No hay regresiones (tests anteriores siguen pasando)
- [ ] Cobertura ≥ 50% (temporal)
- [ ] Docstrings en tests claros
- [ ] Sin código comentado
- [ ] `extraer_historh2h.py` INTACTO

**NO avanzar a siguiente clase si:**
- ❌ Tests de clase actual fallan
- ❌ Suite completa tiene regresiones
- ❌ No se validó integridad del original

---

## 💡 FILOSOFÍA

**"Tests son documentación ejecutable"**

Un buen test debe:
- Explicar QUÉ hace el código
- Validar que funciona
- Servir de ejemplo de uso
- Fallar claramente cuando algo rompe

**Calidad > Cantidad**
- 10 tests buenos > 50 tests malos
- 50% bien testeado > 70% superficial

---

*Actualizado: 2025-10-20 | v1.0*