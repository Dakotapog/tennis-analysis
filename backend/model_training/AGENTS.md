# Instrucciones de IA para Entrenamiento de Modelos

Estas instrucciones reflejan el enfoque de "generación multi-formato" del proyecto y anulan las directrices generales cuando se trabaja en la carpeta `model_training` o en scripts como `aplicar_enhancer.py`.

- **Enfoque Multi-Target:** El objetivo no es un único modelo, sino varios modelos especializados. La IA debe adaptar sus sugerencias al tipo de tarea:
    - **Clasificación:** Predecir el `ganador_real`. Usar `RandomForestClassifier` o `GradientBoostingClassifier` como base.
    - **Regresión:** Predecir `p1_prob_win` o `match_duration_score`. Usar `GradientBoostingRegressor` o `RandomForestRegressor`.
    - **Deep Learning:** Preparar datos para frameworks como TensorFlow/Keras, con normalización `MinMaxScaler` y `one-hot encoding`.

- **Selección de Features Avanzada:** La selección de características debe realizarse con métodos robustos como `RFE` (Recursive Feature Elimination) para encontrar el subconjunto de features más predictivo y evitar el ruido.

- **Benchmarking y Validación Cruzada:** La evaluación de un modelo debe realizarse comparando el rendimiento de varios algoritmos (`RandomForest`, `GradientBoosting`, etc.) mediante validación cruzada estratificada (`StratifiedKFold`). Esto asegura que la elección del modelo final sea basada en evidencia y no en una única partición de datos.

- **Optimización de Hiperparámetros:** Fomenta la implementación de técnicas de búsqueda de hiperparámetros como `GridSearchCV` o `RandomizedSearchCV` para optimizar el rendimiento del modelo seleccionado.

- **Análisis de Overfitting:** Incluye siempre una comparación entre el rendimiento en el conjunto de entrenamiento y el de validación para detectar y cuantificar el sobreajuste. El objetivo es encontrar modelos que generalicen bien a datos no vistos.
