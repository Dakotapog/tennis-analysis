# Proceso de Scraping H2H: de Monolítico a Modular

Este documento describe el flujo de trabajo del proceso de scraping para extraer datos de enfrentamientos directos (H2H) de partidos de tenis, detallando la transición desde el script original `extraer_historh2h.py` hacia una arquitectura modular y refactorizada en la carpeta `scraping`.

## 1. Visión General del Proceso

El objetivo principal del sistema es navegar a una página de partidos de tenis, acceder a la sección H2H, extraer el historial de partidos de ambos jugadores y sus enfrentamientos directos, y finalmente, enriquecer estos datos con rankings y análisis avanzados para generar una predicción.

El flujo general se puede resumir en los siguientes pasos:

1.  **Inicialización y Configuración**: Preparación del entorno, incluyendo la gestión del navegador y la carga de datos iniciales.
2.  **Selección de Partidos**: Identificación y carga de los partidos a procesar desde un archivo JSON.
3.  **Navegación y Extracción**: Uso de un navegador automatizado para visitar las URLs de los partidos y extraer el contenido HTML relevante.
4.  **Parseo de Datos**: Conversión del HTML extraído en datos estructurados (historiales de partidos, H2H directo).
5.  **Enriquecimiento y Análisis**: Integración de datos de ranking, cálculo de métricas (ELO, forma) y análisis de rivalidades.
6.  **Generación de Reportes**: Almacenamiento de los resultados consolidados en un archivo JSON.

---

## 2. Arquitectura Refactorizada (`/scraping`)

El código original en `extraer_historh2h.py` fue refactorizado en módulos con responsabilidades claras, siguiendo las mejores prácticas de desarrollo de software. Esta modularización se encuentra en la carpeta `scraping/`.

### Módulos Principales:

-   **`h2h_extractor.py` (El Orquestador)**:
    -   Es el coordinador principal que dirige todo el flujo de trabajo.
    -   Inicializa y utiliza los otros módulos (`BrowserManager`, `DataParser`, `RankingManager`, etc.).
    -   Gestiona la cola de partidos a procesar y el ciclo de vida de la extracción.
    -   **Equivalente a**: La clase `SequentialH2HExtractor` y la función `main` en el script original.

-   **`browser_manager.py` (El Navegador)**:
    -   **Responsabilidad**: Gestionar todo lo relacionado con el navegador Playwright.
    -   Configura el navegador con optimizaciones específicas para WSL, maneja su inicio (`setup`) y cierre (`cleanup`).
    -   Realiza verificaciones del sistema (memoria, procesos zombie).
    -   **Equivalente a**: Las funciones `verificar_sistema`, `setup_browser` y `cleanup` del script original.

-   **`file_utils.py` (El Archivista)**:
    -   **Responsabilidad**: Manejar la búsqueda, validación y selección de archivos.
    -   Contiene funciones para encontrar todos los archivos JSON, analizar su estructura y seleccionar el más adecuado para el procesamiento.
    -   **Equivalente a**: Las funciones `find_all_json_files`, `analyze_json_structure` y `select_best_json_file` del script original.

-   **`data_parser.py` (El Intérprete)**:
    -   **Responsabilidad**: Extraer y limpiar datos desde el contenido HTML.
    -   Contiene métodos estáticos para parsear información específica como nombres de torneos, fechas, resultados y determinar el ganador de un partido.
    -   **Equivalente a**: Las funciones de extracción y limpieza que estaban dentro de las clases `SequentialH2HExtractor` y `RivalryAnalyzer` en el script original (ej. `determine_winner_from_result`, `extract_winner_sets`).

### Módulos de Análisis (`/analysis`):

Aunque no están en la carpeta `scraping`, son componentes cruciales utilizados por el orquestador:

-   **`ranking_manager.py`**: Gestiona la carga y el acceso a los datos de ranking de los jugadores.
-   **`elo_system.py`**: Implementa el sistema de calificación ELO.
-   **`rivalry_analyzer.py`**: Realiza el análisis avanzado de rivalidades transitivas y genera predicciones.

---

## 3. Paso a Paso del Proceso de Scraping (Versión Refactorizada)

A continuación se detalla el flujo de ejecución paso a paso, indicando qué módulo es responsable de cada tarea.

### **Paso 1: Inicialización**

1.  **`h2h_extractor.py`**: Se crea una instancia de `H2HExtractor`.
2.  **`h2h_extractor.py`**: El extractor inicializa sus componentes: `BrowserManager`, `DataParser`, y los módulos de `analysis`.
3.  **`browser_manager.py`**: Se realizan las verificaciones del sistema (`check_system_resources`).

### **Paso 2: Carga de Partidos**

1.  **`h2h_extractor.py`**: Llama a `load_matches()`.
2.  **`file_utils.py`**: La función `select_best_json_file()` busca, analiza y selecciona el archivo JSON más relevante que contiene las URLs de los partidos.
3.  **`h2h_extractor.py`**: Se leen los datos del JSON seleccionado y se crea una cola interna (`matches_queue`) con los partidos a procesar.

### **Paso 3: Inicio del Proceso de Extracción**

1.  **`h2h_extractor.py`**: Se inicia el navegador llamando a `setup()` en su instancia de `BrowserManager`.
2.  **`browser_manager.py`**: Lanza una instancia de Playwright con la configuración optimizada.
3.  **`h2h_extractor.py`**: Comienza a iterar sobre la `matches_queue`, procesando un partido a la vez.

### **Paso 4: Procesamiento de un Partido Individual**

Para cada partido en la cola:

1.  **Navegación (BrowserManager)**:
    -   Se navega a la `match_url` del partido.
    -   Se localiza y se hace clic en el botón "H2H" para cargar la sección de enfrentamientos.

2.  **Extracción (H2HExtractor + BrowserManager)**:
    -   El `H2HExtractor` coordina la extracción.
    -   Localiza las tres secciones principales de H2H (historial jugador 1, historial jugador 2, enfrentamientos directos).
    -   Expande cada sección haciendo clic repetidamente en "Mostrar más" para cargar todo el historial disponible.

3.  **Parseo (DataParser)**:
    -   El `H2HExtractor` pasa el contenido HTML de cada sección a los métodos de `DataParser`.
    -   `DataParser` extrae, limpia y estructura la información de cada partido del historial: fecha, oponente, resultado, torneo, superficie, etc.

### **Paso 5: Enriquecimiento y Análisis (Módulos de `analysis`)**

1.  **`ranking_manager.py`**: Para cada partido extraído del historial, se consulta el ranking del oponente.
2.  **`rivalry_analyzer.py`**:
    -   Se calcula un rating ELO para cada jugador basado en su historial extraído.
    -   Se analizan los oponentes comunes, la forma reciente, la especialización por superficie y otros factores.
    -   Finalmente, se genera una predicción avanzada con un porcentaje de confianza.

### **Paso 6: Consolidación y Almacenamiento**

1.  **`h2h_extractor.py`**: Todos los datos extraídos y analizados se consolidan en un único objeto JSON para el partido.
2.  Este objeto se añade a la lista de resultados (`all_results`).
3.  Una vez procesados todos los partidos, se llama a `save_results()`, que guarda la lista completa en un nuevo archivo JSON en la carpeta `reports/`.

### **Paso 7: Limpieza**

1.  **`h2h_extractor.py`**: Al finalizar (o en caso de interrupción), se llama a `cleanup()`.
2.  **`browser_manager.py`**: Cierra de forma segura la página, el contexto y el navegador para liberar todos los recursos.

Este enfoque modular no solo hace que el código sea más fácil de leer y mantener, sino que también permite probar cada componente de forma aislada, mejorando la robustez y escalabilidad del sistema.
