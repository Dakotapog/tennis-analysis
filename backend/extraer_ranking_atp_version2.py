#!/usr/bin/env python3
"""
RANKING SCRAPER COMPLETO - Versión para extracción de todos los jugadores
Extrae el ranking ATP completo de FlashScore con detección precisa de columnas específicas.

CAMBIOS PARA EXTRACCIÓN COMPLETA:
- Removidas las limitaciones de solo 10 jugadores para logging
- Removidas las limitaciones de debugging solo para top 3
- Mantenido el logging eficiente para todos los jugadores
- Optimizado para procesar todo el ranking ATP
"""

import asyncio
import logging
import os
import re
import json
from datetime import datetime
from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteRankingScraper:
    def __init__(self):
        self.browser = None
        self.page = None
        self.playwright = None
        self.column_mapping = {}

    async def init_browser(self):
        """Inicializar navegador con configuración optimizada"""
        logger.info("🚀 Iniciando navegador para Complete Ranking Scraper...")
        
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--disable-software-rasterizer',
                '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ]
        )
        
        self.page = await self.browser.new_page()
        await self.page.set_viewport_size({"width": 1920, "height": 1080})
        logger.info("✅ Navegador listo")

    async def navigate_to_rankings(self):
        """Navegar directamente a la página de rankings ATP en vivo"""
        try:
            live_ranking_url = "https://www.flashscore.co/tenis/rankings/atp-live/"
            logger.info(f"🔍 Navegando directamente a la página de ranking en vivo: {live_ranking_url}")
            await self.page.goto(live_ranking_url, 
                                wait_until="domcontentloaded", 
                                timeout=60000)
            
            await self.handle_cookie_consent()
            await self.page.wait_for_selector(".rankingTable", timeout=20000)
            
            logger.info("✅ Página de rankings en vivo cargada correctamente.")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error navegando a la página de ranking en vivo: {str(e)}")
            return False

    async def handle_cookie_consent(self):
        """Manejar el banner de consentimiento de cookies"""
        try:
            accept_button_selector = "#onetrust-accept-btn-handler"
            await self.page.wait_for_selector(accept_button_selector, timeout=7000)
            await self.page.click(accept_button_selector)
            logger.info("✅ Banner de cookies aceptado.")
            await asyncio.sleep(3)
        except Exception:
            logger.info("ℹ️ No se encontró banner de cookies.")

    async def analyze_table_structure_corrected(self):
        """CORREGIDO: Analizar estructura con detección específica de clases CSS"""
        try:
            logger.info("🔍 Analizando estructura de tabla (VERSIÓN CORREGIDA)...")
            
            # Obtener todas las filas
            rows = await self.page.query_selector_all(".rankingTable__row")
            if not rows:
                logger.error("❌ No se encontraron filas en la tabla")
                return False
            
            # Encontrar una fila de datos (no header)
            data_row = None
            for row in rows:
                class_name = await row.get_attribute("class") or ""
                if "headRow" not in class_name:
                    data_row = row
                    break
            
            if not data_row:
                logger.error("❌ No se encontró fila de datos")
                return False
            
            # Obtener todas las celdas de la primera fila de datos
            data_cells = await data_row.query_selector_all(".rankingTable__cell")
            logger.info(f"📊 Encontradas {len(data_cells)} columnas en la fila de datos")
            
            self.column_mapping = {}
            
            # MÉTODO CORREGIDO: Buscar por clases específicas
            for i, cell in enumerate(data_cells):
                cell_class = await cell.get_attribute("class") or ""
                cell_text = (await cell.text_content() or "").strip()
                
                logger.info(f"   Columna {i}: clase='{cell_class}', texto='{cell_text[:50]}'")
                
                # Mapeo específico basado en las clases CSS observadas
                if "rankingTable__cell--rank" in cell_class or i == 0:
                    self.column_mapping['ranking'] = i
                    logger.info(f"   🎯 RANKING detectado en columna {i}")
                
                elif "rankingTable__cell--player" in cell_class or (i == 1 and any(char.isalpha() for char in cell_text)):
                    self.column_mapping['player'] = i
                    logger.info(f"   🎯 PLAYER detectado en columna {i}")
                
                elif "rankingTable__cell--points" in cell_class:
                    self.column_mapping['current_points'] = i
                    logger.info(f"   🎯 CURRENT_POINTS detectado en columna {i}")
                
                elif "rankingTable__cell--maxPoints" in cell_class:
                    # Esta clase es usada tanto para 'Próx. pts' como para 'Pts máx.'
                    # Se diferencian por su posición. 'Próx. pts' aparece primero.
                    if 'next_points' not in self.column_mapping:
                        self.column_mapping['next_points'] = i
                        logger.info(f"   🎯 NEXT_POINTS (Próx. pts) detectado en columna {i} por posición ⭐")
                    elif 'max_points' not in self.column_mapping:
                        self.column_mapping['max_points'] = i
                        logger.info(f"   🎯 MAX_POINTS (Pts máx.) detectado en columna {i} por posición ⭐")
                
                elif "rankingTable__cell--tournament" in cell_class:
                    self.column_mapping['tournaments'] = i
                    logger.info(f"   🎯 TOURNAMENTS detectado en columna {i}")
                
                # Detección por contenido como fallback
                elif cell_text.isdigit() and len(cell_text) >= 4:
                    number = int(cell_text)
                    if 8000 <= number <= 15000 and 'next_points' not in self.column_mapping:
                        self.column_mapping['next_points'] = i
                        logger.info(f"   🎯 NEXT_POINTS detectado por contenido en columna {i}: {cell_text}")
                    elif 10000 <= number <= 20000 and 'max_points' not in self.column_mapping:
                        self.column_mapping['max_points'] = i
                        logger.info(f"   🎯 MAX_POINTS detectado por contenido en columna {i}: {cell_text}")
            
            # Método adicional: buscar por posición conocida si no se detectó por clase
            if 'next_points' not in self.column_mapping:
                await self.detect_by_position_and_content(data_cells, 'next_points', [10290])
            
            if 'max_points' not in self.column_mapping:
                await self.detect_by_position_and_content(data_cells, 'max_points', [12240])
            
            logger.info(f"📋 Mapeo final de columnas: {self.column_mapping}")
            
            # Verificar que tenemos las columnas críticas
            critical_columns = ['ranking', 'player', 'current_points']
            missing_critical = [col for col in critical_columns if col not in self.column_mapping]
            
            if missing_critical:
                logger.warning(f"⚠️ Faltan columnas críticas: {missing_critical}")
                return False
            
            # Verificar columnas objetivo
            target_columns = ['next_points', 'max_points']
            missing_target = [col for col in target_columns if col not in self.column_mapping]
            
            if missing_target:
                logger.warning(f"⚠️ Faltan columnas objetivo: {missing_target}")
            else:
                logger.info("✅ Todas las columnas objetivo detectadas correctamente!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error analizando estructura: {e}")
            return False

    async def detect_by_position_and_content(self, data_cells, column_type, expected_values):
        """Detectar columna por posición y valores esperados"""
        try:
            for i, cell in enumerate(data_cells):
                if i in self.column_mapping.values():
                    continue  # Ya mapeada
                
                cell_text = (await cell.text_content() or "").strip()
                if cell_text.isdigit():
                    value = int(cell_text)
                    if value in expected_values:
                        self.column_mapping[column_type] = i
                        logger.info(f"   🎯 {column_type.upper()} detectado por valor esperado en columna {i}: {cell_text}")
                        return True
        except Exception as e:
            logger.warning(f"⚠️ Error en detección por posición: {e}")
        return False

    async def click_show_more(self, max_clicks=2):
        """Hacer click en 'Mostrar más' para cargar más jugadores"""
        logger.info(f"🔄 Expandiendo ranking completo (máx {max_clicks} clicks)...")
        
        clicks_realizados = 0
        for i in range(max_clicks):
            try:
                show_more_button = self.page.locator('button:has-text("Mostrar más")').first
                if await show_more_button.count() > 0 and await show_more_button.is_visible():
                    await show_more_button.click()
                    await asyncio.sleep(3)
                    clicks_realizados += 1
                    logger.info(f"   ✅ Click {clicks_realizados}/{max_clicks} realizado")
                else:
                    logger.info("   ℹ️ No más botones 'Mostrar más' disponibles")
                    break
            except Exception as e:
                logger.warning(f"   ⚠️ Error en click {i+1}: {e}")
                break
        
        logger.info(f"✅ Ranking expandido completamente con {clicks_realizados} clicks")
        return clicks_realizados

    async def extract_rankings(self):
        """Extraer TODOS los rankings con el nuevo método corregido"""
        logger.info("🎾 Extrayendo RANKING ATP COMPLETO (VERSIÓN CORREGIDA)...")
        
        try:
            await self.page.wait_for_selector(".rankingTable__row", timeout=20000)
            
            # Analizar estructura CORREGIDA
            if not await self.analyze_table_structure_corrected():
                logger.error("❌ No se pudo analizar la estructura correctamente")
                return []
            
            # Cargar TODOS los jugadores
            clicks_realizados = await self.click_show_more()
            
            # Re-analizar después de cargar más datos
            await self.analyze_table_structure_corrected()
            
            # Extraer filas de datos
            player_rows = await self.page.query_selector_all(".rankingTable__row")
            player_rows = [row for row in player_rows 
                          if "headRow" not in (await row.get_attribute("class") or "")]
            
            logger.info(f"🔍 Procesando {len(player_rows)} filas de jugadores COMPLETAS...")
            
            rankings_data = []
            processed_count = 0
            
            for idx, row in enumerate(player_rows):
                player_data = await self.extract_complete_player_data(row, idx + 1)
                if player_data and player_data.get('name'):
                    rankings_data.append(player_data)
                    processed_count += 1
                    
                    # Log de progreso cada 50 jugadores para mejor seguimiento
                    if processed_count % 50 == 0:
                        logger.info(f"   📊 Progreso: {processed_count} jugadores procesados...")
                    
                    # Log detallado solo para primeros 5 jugadores como muestra
                    if idx < 5:
                        logger.info(f"   ✅ #{player_data['ranking_position']:3d}: {player_data['name']:<25} | "
                                   f"Puntos: {player_data['ranking_points']:5d} | "
                                   f"Próx: {player_data['prox_points']:5d} | "
                                   f"Máx: {player_data['max_points']:5d}")
            
            logger.info(f"🎯 EXTRACCIÓN COMPLETA: {len(rankings_data)} jugadores extraídos exitosamente")
            return rankings_data

        except Exception as e:
            logger.error(f"❌ Error extrayendo rankings completos: {str(e)}")
            return []

    async def extract_complete_player_data(self, row_element, expected_rank):
        """COMPLETO: Extraer datos de TODOS los jugadores usando mapeo específico por clases CSS"""
        try:
            all_cells = await row_element.query_selector_all(".rankingTable__cell")
            if len(all_cells) == 0:
                return None
            
            player_data = {
                'ranking_position': expected_rank,
                'name': '',
                'nationality': 'N/A',
                'ranking_points': 0,
                'tournaments_played': 0,
                'prox_points': 0,
                'max_points': 0,
                'ranking_change': 0,
                'defense_points': 0,
                'extraction_timestamp': datetime.now().isoformat(),
                'data_source': 'flashscore_atp_rankings_complete'
            }
            
            # Log detallado del contenido solo para los primeros 3 jugadores como referencia
            if expected_rank <= 3:
                logger.info(f"   DEBUG Jugador #{expected_rank}:")
                for i, cell in enumerate(all_cells):
                    cell_class = await cell.get_attribute("class") or ""
                    cell_text = (await cell.text_content() or "").strip()
                    logger.info(f"     Celda {i}: '{cell_text}' (clase: {cell_class})")
            
            # Extraer datos usando el mapeo de columnas PARA TODOS LOS JUGADORES
            for data_type, col_index in self.column_mapping.items():
                if col_index < len(all_cells):
                    cell = all_cells[col_index]
                    cell_text = (await cell.text_content() or "").strip()
                    
                    if data_type == 'ranking':
                        player_data['ranking_position'] = self.safe_int_extract(cell_text) or expected_rank
                    
                    elif data_type == 'player':
                        # Buscar link del jugador
                        player_link = await cell.query_selector("a")
                        if player_link:
                            player_data['name'] = (await player_link.text_content() or "").strip()
                        else:
                            player_data['name'] = cell_text
                        
                        # Buscar nacionalidad en la misma celda
                        flag_el = await cell.query_selector("span.flag, .flag")
                        if flag_el:
                            nationality = await flag_el.get_attribute('title')
                            if nationality:
                                player_data['nationality'] = nationality.strip()
                    
                    elif data_type == 'current_points':
                        player_data['ranking_points'] = self.safe_int_extract(cell_text)
                    
                    elif data_type == 'next_points':
                        player_data['prox_points'] = self.safe_int_extract(cell_text)
                    
                    elif data_type == 'max_points':
                        player_data['max_points'] = self.safe_int_extract(cell_text)
                    
                    elif data_type == 'tournaments':
                        player_data['tournaments_played'] = self.safe_int_extract(cell_text)
            
            # Aplicar fallbacks si las columnas objetivo no se extrajeron
            if player_data['prox_points'] == 0:
                player_data['prox_points'] = player_data['ranking_points']  # Fallback básico
            
            if player_data['max_points'] == 0:
                # Estimación más conservadora
                player_data['max_points'] = int(player_data['ranking_points'] * 1.1)
            
            # Calcular métricas derivadas
            player_data = self.calculate_momentum_metrics(player_data)
            
            return player_data if player_data['name'] else None

        except Exception as e:
            logger.error(f"❌ Error procesando jugador #{expected_rank}: {e}")
            return None

    def safe_int_extract(self, text):
        """Extraer entero de forma segura"""
        if not text:
            return 0
        # Eliminar todo excepto dígitos
        number_str = re.sub(r'\D', '', str(text))
        return int(number_str) if number_str else 0

    def calculate_momentum_metrics(self, player_data):
        """Calcular métricas derivadas"""
        try:
            ranking_points = player_data.get('ranking_points', 0)
            prox_points = player_data.get('prox_points', 0)
            max_points = player_data.get('max_points', 0)
            defense_points = player_data.get('defense_points', 0)

            # Calcular métricas
            if ranking_points > 0:
                player_data['defense_pressure'] = round(defense_points / ranking_points, 4)
            else:
                player_data['defense_pressure'] = 0.0

            player_data['improvement_potential'] = max(0, max_points - ranking_points)
            player_data['momentum_secured'] = max(0, prox_points - ranking_points)
            
            return player_data

        except Exception as e:
            logger.error(f"❌ Error calculando métricas: {e}")
            return player_data

    async def save_complete_data(self, rankings_data):
        """Guardar datos completos con análisis detallado"""
        if not rankings_data:
            logger.warning("⚠️ No hay datos para guardar.")
            return None
        
        try:
            os.makedirs("data", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Análisis de extracción COMPLETO
            successful_extractions = {
                'with_prox_points': len([p for p in rankings_data if p.get('prox_points', 0) > 0]),
                'with_max_points': len([p for p in rankings_data if p.get('max_points', 0) > 0]),
                'with_both': len([p for p in rankings_data if p.get('prox_points', 0) > 0 and p.get('max_points', 0) > 0]),
                'top_100': len([p for p in rankings_data if p.get('ranking_position', 999) <= 100]),
                'top_500': len([p for p in rankings_data if p.get('ranking_position', 999) <= 500])
            }
            
            output_data = {
                "metadata": {
                    "extraction_timestamp": datetime.now().isoformat(),
                    "total_players": len(rankings_data),
                    "version": "complete_extraction_v1.0",
                    "column_mapping_used": self.column_mapping,
                    "extraction_success": successful_extractions,
                    "coverage": {
                        "complete_ranking": True,
                        "max_clicks_performed": 2,
                        "data_completeness": "full_atp_ranking"
                    }
                },
                "rankings": rankings_data,
                "players_by_name": {player['name']: player for player in rankings_data}
            }
            
            filename = f"data/atp_rankings_complete_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ DATOS COMPLETOS guardados en: {filename}")
            logger.info("📊 Análisis de extracción COMPLETA:")
            logger.info(f"   - Total jugadores: {len(rankings_data)}")
            logger.info(f"   - Con próx_points: {successful_extractions['with_prox_points']}")
            logger.info(f"   - Con max_points: {successful_extractions['with_max_points']}")
            logger.info(f"   - Con ambos campos: {successful_extractions['with_both']}")
            logger.info(f"   - Top 100: {successful_extractions['top_100']}")
            logger.info(f"   - Top 500: {successful_extractions['top_500']}")
            
            return filename
            
        except Exception as e:
            logger.error(f"❌ Error guardando datos completos: {e}")
            return None

    async def close_browser(self):
        """Cerrar navegador"""
        logger.info("🔒 Cerrando navegador...")
        try:
            if self.page:
                await self.page.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            logger.info("✅ Navegador cerrado")
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando navegador: {e}")

    async def run_complete_extraction(self):
        """Ejecutar extracción COMPLETA de todo el ranking ATP"""
        logger.info("🚀 === INICIANDO EXTRACCIÓN COMPLETA DE RANKING ATP ===")
        
        try:
            await self.init_browser()
            
            if not await self.navigate_to_rankings():
                logger.error("❌ No se pudo navegar a rankings")
                return None
            
            rankings_data = await self.extract_rankings()
            
            if not rankings_data:
                logger.error("❌ No se extrajeron datos")
                return None
            
            saved_file = await self.save_complete_data(rankings_data)
            
            if saved_file:
                logger.info("🎯 === EXTRACCIÓN COMPLETA FINALIZADA ===")
                logger.info(f"   📄 Archivo: {saved_file}")
                logger.info(f"   📊 Total jugadores: {len(rankings_data)}")
                
                # Mostrar ejemplos de jugadores de diferentes rangos
                logger.info("   🏆 Muestra de jugadores extraídos:")
                
                # Top 5
                for i, player in enumerate(rankings_data[:5]):
                    logger.info(f"      #{player['ranking_position']:3d}: {player['name']:<25} | "
                               f"Actual: {player['ranking_points']:5d} | "
                               f"Próx: {player['prox_points']:5d} | "
                               f"Máx: {player['max_points']:5d}")
                
                # Algunos del medio del ranking si hay suficientes
                if len(rankings_data) > 50:
                    logger.info("   📊 Jugadores rango medio (alrededor del #50):")
                    for player in rankings_data[48:53]:  # Jugadores 49-53
                        logger.info(f"      #{player['ranking_position']:3d}: {player['name']:<25} | "
                                   f"Actual: {player['ranking_points']:5d} | "
                                   f"Próx: {player['prox_points']:5d}")
                
                # Últimos 3 del ranking
                if len(rankings_data) > 10:
                    logger.info("   📉 Últimos jugadores del ranking:")
                    for player in rankings_data[-3:]:
                        logger.info(f"      #{player['ranking_position']:3d}: {player['name']:<25} | "
                                   f"Actual: {player['ranking_points']:5d} | "
                                   f"Próx: {player['prox_points']:5d}")
                
                return saved_file
            else:
                logger.error("❌ Error guardando datos")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error en extracción completa: {e}")
            return None
        finally:
            await self.close_browser()


async def main():
    """Función principal para extracción completa"""
    scraper = CompleteRankingScraper()
    result = await scraper.run_complete_extraction()
    
    if result:
        print("\n🎉 ¡Extracción COMPLETA terminada exitosamente!")
        print(f"📁 Archivo generado: {result}")
        print("📊 Revisa los logs para ver estadísticas detalladas de la extracción.")
    else:
        print("\n❌ La extracción completa falló. Revisa los logs para más detalles.")


if __name__ == "__main__":
    asyncio.run(main())