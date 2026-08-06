# Graphify Report — 2026-07-12

_Generado: 2026-07-12 00:13_

**Nodos:** 1686 | **Edges:** 2753 | **Comunidades:** 91 | Commit: `f88df196193f2af2df049a8cf18a9a7212f6a637`

_(No se encontró reporte anterior para delta.)_

## Top 20 nodos por grado (centralidad)

_Si este nodo se rompe, más componentes se ven afectados._

| Nodo | Grado | Tipo | Archivo fuente |
|---|---|---|---|
| `betplay_combo_builder.py` | 50 | file | `betplay_combo_builder.py` |
| `shadow_book.py` | 41 | file | `shadow_book.py` |
| `IntelligentMLEnhancer` | 38 | class | `Intelligent_ml_enhancer.py` |
| `RivalryAnalyzer` | 37 | class | `analysis/rivalry_analyzer.py` |
| `RankingManager` | 33 | class | `analysis/ranking_manager.py` |
| `combo_confianza_builder.py` | 33 | file | `combo_confianza_builder.py` |
| `edge_calculator.py` | 32 | file | `edge_calculator.py` |
| `H2HExtractor` | 32 | class | `scraping/h2h_extractor.py` |
| `ninja_h2h_parser.py` | 32 | file | `scraping/ninja_h2h_parser.py` |
| `dashboard.py` | 30 | file | `dashboard.py` |
| `main()` | 29 | fn | `betplay_combo_builder.py` |
| `pipeline_tracker.py` | 28 | file | `pipeline_tracker.py` |
| `rivalry_analyzer.py` | 26 | file | `analysis/rivalry_analyzer.py` |
| `SmartLogger` | 26 | class | `utils/logger.py` |
| `generar_tabla_favoritos2.py` | 25 | file | `generar_tabla_favoritos2.py` |
| `DataFrame` | 24 | class | `-` |
| `kambi_tennis.py` | 24 | file | `scraping/kambi_tennis.py` |
| `.generate_advanced_prediction()` | 23 | fn | `analysis/rivalry_analyzer.py` |
| `IntelligentDatasetGenerator` | 22 | class | `generar_dataset_plus.py` |
| `NinjaH2HExtractor` | 22 | class | `scraping/ninja_h2h_parser.py` |

## Huérfanos SDD (archivos sin Nodo)

_Fuente única: `nodos_index.json` — misma que `check_contradictions.py`. No reimplementa lógica._

- Archivos .py rastreados: **62**
- Huérfanos oficiales: **0**
- Índice generado: 2026-07-10T01:50:00

_Cobertura SDD: 100% — ningún archivo sin Nodo._ ✓


## Resumen de comunidades

_Top 5 nodos por grado dentro de cada comunidad (lectura rápida de qué trata el clúster)._

| Comunidad | Nodos | Top representantes |
|---|---|---|
| Community 0 | 64 | `IntelligentMLEnhancer`, `DataFrame`, `.enhance_dataset()`, `.detect_and_clean_anomalies()`, `.intelligent_feature_selection()` |
| Community 1 | 59 | `dashboard.py`, `token_odometer.py`, `parse_sessions()`, `main()`, `panel_atribucion()` |
| Community 2 | 57 | `edge_calculator.py`, `calcular_edge_completo()`, `procesar_archivo_h2h()`, `data_contract.py`, `calcular_recencia_regimen()` |
| Community 3 | 57 | `combo_confianza_builder.py`, `main()`, `_build_portfolio_v2()`, `_extract_and_categorize()`, `_build_anchor_combos()` |
| Community 4 | 51 | `H2HExtractor`, `._process_single_match()`, `._extract_h2h_sections()`, `.cleanup()`, `.save_results()` |
| Community 5 | 49 | `pipeline_tracker.py`, `main()`, `_with_resultado()`, `seccion_27_4_senales()`, `seccion_27_6_temporal()` |
| Community 6 | 44 | `database.py`, `get_db()`, `app.py`, `test_database()`, `init_db()` |
| Community 7 | 43 | `betslip_registrar.py`, `Handler`, `close_snapshot_server.py`, `._handle_check_and_close()`, `cerrar()` |
| Community 8 | 40 | `kambi_tennis.py`, `extract_matches_flashscore_only()`, `extract_matches()`, `_parse_nombre()`, `_build_match_key()` |
| Community 9 | 39 | `detectar_tier()`, `resultados_finales.py`, `config.py`, `tournament_context.py`, `build_tournament_context()` |
| Community 10 | 37 | `trader_ev_tenis.py`, `main()`, `_print_individuales()`, `_portfolio_risk_report()`, `_ev()` |
| Community 11 | 36 | `ZitaScraper`, `main()`, `.extract_matches_from_dom()`, `.navigate_to_flashscore()`, `.extract_tennis_matches()` |
| Community 12 | 36 | `IntelligentDatasetGenerator`, `.generate_superior_dataset()`, `._final_ml_validation()`, `._save_superior_dataset()`, `._generate_executive_report()` |
| Community 13 | 35 | `BrowserManager`, `h2h_extractor.py`, `__init__.py`, `select_best_json_file()`, `file_utils.py` |
| Community 14 | 32 | `betplay_combo_builder.py`, `main()`, `generar_bat_chrome()`, `mostrar_consola()`, `generar_whatsapp_html()` |
| Community 15 | 32 | `validar_con_api.py`, `validar_predicciones()`, `validar_partido_individual()`, `_parse_nombre()`, `obtener_resultado_partido()` |
| Community 16 | 31 | `MLDatasetOrchestrator`, `.run_full_pipeline()`, `.__init__()`, `._initialize_enhancer()`, `._load_latest_dataset()` |
| Community 17 | 30 | `CompleteRankingScraper`, `.run_complete_extraction()`, `.extract_rankings()`, `.extract_complete_player_data()`, `.navigate_to_rankings()` |
| Community 18 | 29 | `NinjaH2HExtractor`, `._analyze_and_consolidate()`, `._process_ronda_futura()`, `._run_playwright_batch_async()`, `extract_match_id_from_url()` |
| Community 19 | 28 | `ComboRegistry`, `.settle_date()`, `_normalize_name()`, `._registry_path()`, `._settle_pierna()` |
| Community 20 | 27 | `rivalry_analyzer.py`, `markov_analyzer.py`, `__init__.py`, `elo_system.py`, `k_factor_efectivo()` |
| Community 21 | 27 | `RankingManager`, `.get_player_info()`, `.normalize_name()`, `.get_ranking_metrics()`, `.get_player_metrics_summary()` |
| Community 22 | 27 | `ZitaScraper`, `main()`, `.navigate_to_flashscore()`, `.extract_tennis_matches()`, `.extract_matches_from_dom()` |
| Community 23 | 26 | `ninja_h2h_parser.py`, `_parse_direct_h2h()`, `_parse_player_history()`, `_clean_player_name()`, `_timestamp_to_date()` |
| Community 24 | 25 | `generar_tabla_favoritos2.py`, `analyze_matches_with_pandas()`, `format_component_status()`, `_load_profitability_data()`, `_load_edge_report()` |
| Community 25 | 23 | `CompleteRankingScraper`, `extraer_historh2h.py`, `.run_complete_extraction()`, `ranking_manager.py`, `main()` |
| Community 26 | 23 | `RivalryAnalyzer`, `.calculate_elo_from_history()`, `.calcular_peso_oponentes_comunes()`, `.estimate_elo_from_rank()`, `._partidos_recientes()` |
| Community 27 | 22 | `PlayerRegistry`, `.resolve()`, `._bootstrap()`, `player_registry.py`, `._register_atp_entry()` |
| Community 28 | 22 | `games_signal_calculator.py`, `procesar_partidos()`, `main()`, `_cargar_thresholds_calibrados()`, `auto_calibrar_thresholds()` |
| Community 29 | 22 | `shadow_book.py`, `_build_record()`, `_match_key()`, `_parse_apellido()`, `_build_sb_id()` |
| Community 30 | 21 | `DataParser`, `.__init__()`, `.load_matches()`, `.normalize_surface()`, `.parse_match_date()` |
| Community 31 | 20 | `.analyze_rivalry()`, `erdos_graph.py`, `distancia_erdos()`, `historial_a_partidos()`, `construir_grafo_victorias()` |
| Community 32 | 20 | `fetch_kambi_outcomes()`, `build_live_combos()`, `build_was_combos()`, `find_outcome()`, `build_safe_combos()` |
| Community 33 | 20 | `report()`, `report_dict()`, `_segment_metrics()`, `_graduated()`, `_pick_status_sb()` |
| Community 34 | 20 | `_load_jsonl()`, `close_snapshot()`, `settle()`, `_save_jsonl()`, `log_picks()` |
| Community 35 | 20 | `hypothesis_tracker.py`, `get_hypothesis()`, `load_hypotheses()`, `get_nodo46_case_count()`, `sprt_from_hypothesis()` |
| Community 36 | 19 | `pattern_audit.py`, `audit_pattern()`, `_match_control()`, `_get_field_value()`, `_values_match()` |
| Community 37 | 18 | `SmartLogger`, `.info()`, `.success()`, `.section()`, `.progress()` |
| Community 38 | 17 | `ResultVerifier`, `main()`, `.verify_all_matches()`, `consultar_resultados_historicos.py`, `.setup_browser()` |
| Community 39 | 17 | `normalization.py`, `normalize_and_weight_scores()`, `normalize_min_max()`, `normalize_with_log_scale()`, `Any` |
| Community 40 | 17 | `session_compiler.py`, `main()`, `_git()`, `_commits_since()`, `_files_changed()` |
| Community 41 | 16 | `DataFrame`, `AdvancedMLFormatter`, `.format_for_algorithm()`, `._format_for_classification()`, `._format_for_regression()` |
| Community 42 | 15 | `EloRatingSystem`, `backtest_nodo28_limpio.py`, `main()`, `.update_ratings()`, `.get_rating()` |
| Community 43 | 14 | `drift_monitor.py`, `daily_drift_report()`, `cusum_brier()`, `psi_score()`, `_default_shadow_dir()` |
| Community 44 | 14 | `.generate_advanced_prediction()`, `.classify_tournament()`, `_enqueue_playwright_candidate()`, `density_confidence()`, `shrink_weights()` |
| Community 45 | 14 | `run_daily.py`, `main()`, `_was_qualifies()`, `_run()`, `_build_daily_brief()` |
| Community 46 | 14 | `combo_governor.py`, `main()`, `_parse_combo_plan()`, `_latest_combo_plans()`, `_betplay_stakes_today()` |
| Community 47 | 13 | `flb_curve.py`, `flb_curve()`, `_in_banda()`, `_load_settled_for_flb()`, `_banda_label()` |
| Community 48 | 13 | `IntelligentMLValidator`, `.validate_ml_readiness()`, `._simulate_ml_training()`, `._analyze_feature_correlations()`, `._analyze_class_separability()` |
| Community 49 | 13 | `player_routes.py`, `test_connectivity()`, `setup_test_driver()`, `test_flashscore_access()`, `connectivity_info()` |
| Community 50 | 13 | `FlashscoreRankingsInspector`, `main()`, `.setup_browser()`, `.navigate_to_url()`, `.handle_cookie_consent()` |
| Community 51 | 12 | `_normalize_name()`, `player_profitability.py`, `_normalize_player_name_for_prof()`, `build_player_profitability()`, `get_player_profitability()` |
| Community 52 | 12 | `.load_rankings()`, `._load_basic_ranking_fallback()`, `._load_complete_ranking_file()`, `.load_basic_rankings()`, `.__init__()` |
| Community 53 | 12 | `build_mega_combos()`, `line_movement_signal()`, `ranking_preserved_blend()`, `cv_edge_guard()`, `dispersion_index()` |
| Community 54 | 12 | `check_contradictions.py`, `main()`, `_get_nodo_files()`, `_extract_nodo_state()`, `_check_claude_md()` |
| Community 55 | 12 | `._process_match()`, `_name_tokens()`, `_token_in_kb()`, `_lookup_player_history_temporal()`, `_fuzzy_name_match()` |
| Community 56 | 11 | `Path`, `build_games_combos()`, `_find_bankroll_from_plans()`, `_find_latest_games_signal()`, `find_latest_trader_plan()` |
| Community 57 | 11 | `close_snapshot_trigger.py`, `main()`, `_log()`, `_open_records()`, `_run_close_snapshot()` |
| Community 58 | 11 | `generar_dataset_plus.py`, `IntelligentFeatureGenerator`, `.__init__()`, `MLConfig`, `.generate_advanced_features()` |
| Community 59 | 11 | `IntelligentDataAnalyzer`, `._deep_analysis()`, `.analyze_class_distribution()`, `.analyze_feature_discriminative_power()`, `.detect_trivial_patterns()` |
| Community 60 | 11 | `pre_game_validator.py`, `validate_file()`, `main()`, `Path`, `create_fixture()` |
| Community 61 | 10 | `.determine_match_winner()`, `.analyze_advanced_player_metrics()`, `.analyze_strength_of_schedule()`, `.analyze_streaks_and_consistency()`, `._win_rate_vs_oponente()` |
| Community 62 | 10 | `._fetch_player_history_from_proxy()`, `_is_main_section_kb()`, `_split_into_h2h_blocks()`, `_parse_sections()`, `fetch_h2h_from_api()` |
| Community 66 | 8 | `AIConfig`, `aplicar_enhancer.py`, `Intelligent_ml_enhancer.py`, `OrchestratorConfig`, `logger.py` |
| Community 63 | 8 | `conformal_band.py`, `conformal_report()`, `conformal_quantile()`, `is_no_bet_conformal()`, `analysis/conformal_band.py — Nodo-68: Banda Conformal  INSTRUMENTO DE MEDICIÓN —` |
| Community 64 | 8 | `rho_empirical.py`, `block_bootstrap_rho()`, `_pairwise_correlation_session()`, `rho_report()`, `analysis/rho_empirical.py — Nodo-65: Bootstrap ρ Empírico Inter-Pick  INSTRUMENT` |
| Community 65 | 8 | `.analyze_surface_specialization()`, `.analizar_contundencia()`, `.analizar_resistencia()`, `_is_gcs_season_active()`, `Nodo-61 D61-F1: verifica si un torneo ganado está en la ventana estacional activ` |
| Community 67 | 8 | `_build_live_combos_legacy()`, `_save_betslip_index()`, `_score_combo()`, `_select_with_cobertura()`, `Score a combo using Portfolio Theory + Markov Regime + Information Theory.` |
| Community 68 | 8 | `.extract_rankings()`, `.analyze_table_structure_corrected()`, `.detect_by_position_and_content()`, `.click_show_more()`, `CORREGIDO: Analizar estructura con detección específica de clases CSS` |
| Community 69 | 8 | `build_index()`, `rebuild_nodos_index.py`, `_parse_nodo()`, `_collect_py_files()`, `Path` |
| Community 70 | 6 | `select_best_json_file()`, `find_all_json_files()`, `analyze_json_structure()`, `🔍 Encontrar todos los archivos JSON en el directorio actual y carpeta data`, `🔬 Analizar la estructura de un archivo JSON para verificar si contiene partidos` |
| Community 71 | 6 | `.extract_complete_player_data()`, `.safe_int_extract()`, `.calculate_momentum_metrics()`, `COMPLETO: Extraer datos de TODOS los jugadores usando mapeo específico por clase`, `Extraer entero de forma segura` |
| Community 72 | 6 | `.validate_weights()`, `show_normalization_transparency()`, `format_weights_distribution()`, `Muestra información sobre los parámetros de normalización utilizados.     Esto p`, `Formats and writes the weights distribution table.` |
| Community 73 | 6 | `WeightManager`, `.calculate_adjusted_weights()`, `.__init__()`, `Gestor de pesos dinámicos con redistribución automática.          RESPONSABILIDA`, `Inicializa el gestor de pesos.                  Args:             weights_config` |
| Community 74 | 5 | `.get_player_ranking()`, `.get_atp_ranking()`, `.get_wta_ranking()`, `Obtener ranking específico ATP`, `Obtener ranking de un jugador con manejo robusto de errores.` |
| Community 75 | 5 | `get_configured_driver()`, `test_driver_configuration()`, `selenium_config.py`, `Configura y devuelve una instancia de Chrome WebDriver optimizada para WSL2.`, `Una función de prueba simple para verificar si el driver se puede crear.` |
| Community 76 | 5 | `normalize_player_name()`, `engineer_features()`, `feature_engineering.py`, `Normaliza el nombre de un jugador para que coincida con las claves del JSON.`, `Extrae características numéricas de columnas complejas de forma dinámica.     Es` |
| Community 77 | 4 | `velocity_monitor.py`, `velocity_zscore()`, `analysis/velocity_monitor.py — Nodo-71: Kyle's λ Velocity Monitor  INSTRUMENTO D`, `Nodo-71: Kyle's λ — velocidad de movimiento de línea.      velocity_i = (odds[i]` |
| Community 80 | 4 | `enviar_combos_telegram()`, `_enviar_telegram()`, `Envía mensaje a Telegram.`, `Envía combos a Telegram link por link con redirect URLs.      Cada combo va como` |
| Community 78 | 4 | `session_budget()`, `check_budget()`, `Presupuesto máximo de inversión por sesión (M-26-2).`, `Pre-sesión: ¿los combos planificados exceden el budget? (M-26-2)     Recorta a l` |
| Community 79 | 4 | `tournament_concentration_ok()`, `discipline_check()`, `Nodo-25 Guard 2: Max N picks from the same tournament in any combo.     Returns`, `Nodo-25 Guard 3: Only picks from trader_plan enter combos.     Returns True if p` |
| Community 81 | 4 | `explicar_ventaja_rival_comun()`, `parse_score()`, `Parsea un resultado como '2-1' en una tupla de enteros (sets_ganados, sets_perdi`, `Genera una explicación textual de por qué se asigna la ventaja a un jugador` |
| Community 82 | 4 | `n8n_push_workflow.py`, `_api()`, `main()`, `n8n_push_workflow.py — Sube el workflow de close-snapshot a n8n via API REST. No` |
| Community 83 | 2 | `find_latest_h2h_file()`, `Finds the most recent h2h_results_enhanced_...json file in the reports directory` |
| Community 84 | 2 | `instalar_selenium_wsl2.sh`, `instalar_selenium_wsl2.sh script` |
| Community 85 | 2 | `__init__.py`, `validation/ — Framework de Validación Pre-Registrada (Nodo-51 F5)  Contiene hipó` |
| Community 86 | 1 | `__init__.py` |
| Community 87 | 1 | `__init__.py` |
| Community 88 | 1 | `__init__.py` |
| Community 89 | 1 | `__init__.py` |
| Community 90 | 1 | `__init__.py` |

---
_graphify report · 2026-07-12 00:13_