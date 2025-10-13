# Instrucciones de IA para Preprocesamiento y Mejora de Datos

Estas instrucciones reflejan el enfoque "anti-colapso" del proyecto y anulan las directrices generales cuando se trabaja en la carpeta `data_preprocessing` o en scripts como `generar_dataset_plus.py` y `Intelligent_ml_enhancer.py`.

- **Ingeniería de Características Contextual:** La generación de features debe ir más allá de simples cálculos. El objetivo es crear variables que capturen la dinámica del tenis, como `ranking_diff`, `elo_diff`, `match_competitiveness` y ventajas por superficie. La IA debe proponer features que ayuden a prevenir predicciones triviales basadas solo en el ranking.

- **Imputación Inteligente:** Para el manejo de valores nulos, el método preferido es la imputación contextual con `KNNImputer` en lugar de la media simple. Esto preserva las relaciones complejas entre las variables.

- **Detección de Anomalías:** Utiliza enfoques multivariados como `IsolationForest` o `LocalOutlierFactor` para detectar y limpiar outliers, ya que las anomalías en este dominio pueden ser sutiles y no detectables con métodos univariados.

- **Balanceo de Clases Avanzado:** Cuando se necesite balancear el dataset, prioriza técnicas de sobremuestreo inteligentes como `SMOTE` o `ADASYN` (de la librería `imblearn`) para generar muestras sintéticas realistas en la clase minoritaria, en lugar de un submuestreo simple que descarte datos valiosos.

- **Validación Preventiva:** Antes de finalizar un script de preprocesamiento, incluye siempre una fase de validación que simule un entrenamiento de ML simple para detectar problemas como baja diversidad de predicciones o alta correlación entre features. El objetivo es anticipar y corregir el colapso del modelo.
