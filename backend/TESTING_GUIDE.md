# 🧪 Guía de Testing - Tennis Analysis

## 📋 Tabla de Contenidos
1. [Instalación](#instalación)
2. [Ejecución de Tests](#ejecución-de-tests)
3. [Estructura de Tests](#estructura-de-tests)
4. [Cobertura de Código](#cobertura-de-código)
5. [Mejores Prácticas](#mejores-prácticas)

---

## 🚀 Instalación

### 1. Activar Entorno Virtual
```bash
# En WSL/Ubuntu
cd /mnt/c/users/hogar/tennis-analysis/backend
source venv/bin/activate
```

### 2. Instalar Dependencias de Testing
```bash
pip install -r requirements-testing.txt
```

---

## ▶️ Ejecución de Tests

### Tests Básicos
```bash
# Ejecutar todos los tests
pytest

# Ejecutar con verbosidad
pytest -v

# Ejecutar tests específicos
pytest tests/test_extraer_historh2h.py

# Ejecutar una clase de tests específica
pytest tests/test_extraer_historh2h.py::TestEloRatingSystem

# Ejecutar un test individual
pytest tests/test_extraer_historh2h.py::TestEloRatingSystem::test_initialization
```

### Tests por Marcadores
```bash
# Solo tests unitarios
pytest -m unit

# Solo tests de integración
pytest -m integration

# Excluir tests lentos
pytest -m "not slow"

# Tests de humo (smoke tests)
pytest -m smoke
```

### Tests con Cobertura
```bash
# Ejecutar con reporte de cobertura
pytest --cov=backend --cov-report=html

# Ver reporte en navegador
# El reporte HTML se genera en: htmlcov/index.html
```

### Tests en Modo Watch (Desarrollo)
```bash
# Instalar pytest-watch
pip install pytest-watch

# Ejecutar en modo watch
ptw
```

---

## 📁 Estructura de Tests

```
tennis-analysis/
├── backend/
│   ├── extraer_historh2h.py
│   ├── generar_dataset_plus.py
│   └── ...
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Fixtures globales
│   ├── test_extraer_historh2h.py      # Tests para scraper H2H
│   ├── test_generar_dataset_plus.py   # Tests para dataset
│   ├── test_aplicar_enhancer.py       # Tests para enhancer
│   └── fixtures/                       # Datos de prueba
│       ├── sample_matches.json
│       ├── sample_rankings.json
│       └── ...
├── pytest.ini                          # Configuración de pytest
└── requirements-testing.txt            # Dependencias de testing
```

---

## 📊 Cobertura de Código

### Objetivo de Cobertura
- **Mínimo aceptable:** 70%
- **Objetivo ideal:** 85%
- **Crítico (clases core):** 90%+

### Ver Reporte de Cobertura
```bash
# Generar reporte
pytest --cov=backend --cov-report=term-missing

# Reporte HTML detallado
pytest --cov=backend --cov-report=html
open htmlcov/index.html  # En tu navegador
```

### Ejemplo de Salida
```
Name                           Stmts   Miss  Cover   Missing
------------------------------------------------------------
backend/extraer_historh2h.py     850     85    90%   234-245, 567-589
backend/generar_dataset_plus.py  420     63    85%   123-134, 456-478
------------------------------------------------------------
TOTAL                           1270    148    88%
```

---

## ✅ Mejores Prácticas

### 1. Principios AAA (Arrange-Act-Assert)
```python
def test_elo_rating_update():
    # Arrange - Preparar datos
    elo = EloRatingSystem(k_factor=32)
    
    # Act - Ejecutar acción
    elo.update_ratings("Winner", "Loser")
    
    # Assert - Verificar resultado
    assert elo.ratings["Winner"] > 1500
```

### 2. Usar Fixtures para Datos Reutilizables
```python
@pytest.fixture
def sample_player():
    return {
        'name': 'Rafael Nadal',
        'rank': 2,
        'points': 7000
    }

def test_player_ranking(sample_player):
    assert sample_player['rank'] == 2
```

### 3. Mocking de Dependencias Externas
```python
@patch('extraer_historh2h.requests.get')
def test_fetch_data(mock_get):
    mock_get.return_value.json.return_value = {'data': 'test'}
    result = fetch_player_data('test_id')
    assert result == {'data': 'test'}
```

### 4. Tests Parametrizados
```python
@pytest.mark.parametrize("rank,expected_elo", [
    (1, 2400),
    (10, 2200),
    (50, 2000),
    (100, 1800)
])
def test_elo_estimation(rank, expecte