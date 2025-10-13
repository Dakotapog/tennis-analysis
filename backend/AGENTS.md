# Instrucciones Generales para el Agente de IA

Este archivo contiene las directrices generales para la asistencia de IA en todo el proyecto de análisis de tenis.

## Filosofía del Proyecto

El núcleo de este proyecto es un pipeline "anti-colapso" que transforma datos brutos de scraping en datasets altamente enriquecidos y validados, listos para múltiples tareas de ML. La IA debe apoyar esta filosofía en cada paso.

1.  **Extracción Profunda (`extraer_historh2h.py`):** No solo extraemos datos, sino que generamos conocimiento a través del `RivalryAnalyzer` y el sistema ELO.
2.  **Generación Robusta (`generar_dataset_plus.py`):** La creación del dataset inicial se enfoca en la ingeniería de características que capturan la complejidad del tenis para evitar predicciones triviales.
3.  **Mejora con IA (`Intelligent_ml_enhancer.py`):** Aplicamos técnicas avanzadas de IA (imputación KNN, `IsolationForest`, RFE, SMOTE) para limpiar, optimizar y enriquecer el dataset.
4.  **Orquestación Multi-Formato (`aplicar_enhancer.py`):** El resultado final no es un único dataset, sino múltiples formatos especializados para clasificación, regresión y Deep Learning.

## Directrices Principales

- **Eficiencia del Código:** Prioriza siempre la eficiencia en el código Python, especialmente en funciones de procesamiento de datos. Utiliza operaciones vectorizadas con `pandas` y `numpy` siempre que sea posible.
- **Estándar de Documentación:** Todos los comentarios y docstrings del código deben seguir la sintaxis de `pydoc` para mantener la consistencia y facilitar la generación automática de documentación.
- **Manejo de Dependencias:** Antes de introducir una nueva librería, verifica si la funcionalidad ya puede ser cubierta por las dependencias existentes (`pandas`, `numpy`, `scikit-learn`, `playwright`, `imblearn`).
- **Inmutabilidad de Datos:** En los scripts de preprocesamiento, trata los DataFrames de entrada como inmutables. Crea copias explícitas antes de realizar transformaciones para evitar efectos secundarios inesperados.
- **Registro (Logging):** Utiliza el módulo `logging` para registrar los eventos importantes del proceso, advertencias y errores. Evita el uso de `print()` para la depuración en el código final.
- **Modularidad:** Fomenta la creación de clases y funciones reutilizables, siguiendo el ejemplo de `IntelligentMLEnhancer` y `RivalryAnalyzer`.
