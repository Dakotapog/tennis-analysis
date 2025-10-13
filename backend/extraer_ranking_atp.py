#!/usr/bin/env python3
"""
RANKING SCRAPER MEJORADO - Versión con análisis completo de momentum
Extrae el ranking ATP de FlashScore con métricas avanzadas de momentum y presión.

NUEVAS CARACTERÍSTICAS:
- Extrae puntos próximos (PROX PTS), máximos (PTS MAX), cambios de ranking
- Puntos a defender del torneo del año pasado
- Calcula métricas de presión, momentum y potencial de mejora
- JSON optimizado para consumo por extraer_historh2h.py
- Análisis de inteligencia competitiva real
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

class EnhancedRankingScraper:
    def __init__(self):
        self.browser = None
        self.page = None
        self.playwright = None

    async def init_browser(self):
        """Inicializar navegador con configuración optimizada para WSL"""
        logger.info("🚀 Iniciando navegador para Enhanced Ranking Scraper...")
        
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--disable-software-rasterizer',
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding',
                '--disable-features=TranslateUI',
                '--disable-extensions',
                '--no-first-run',
                '--disable-default-apps',
                '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ]
        )
        
        self.page = await self.browser.new_page()
        await self.page.set_viewport_size({"width": 1920, "height": 1080})
        logger.info("✅ Navegador listo")

    async def navigate_to_rankings(self):
        """Navegar a la página de rankings ATP de FlashScore.co con ranking en vivo."""
        try:
            # Usar la URL que incluye el ranking en vivo para obtener todas las columnas
            logger.info("🔍 Navegando a https://www.flashscore.co/tenis/rankings/atp/ (Live Rankings)...")
            await self.page.goto("https://www.flashscore.co/tenis/rankings/atp/", 
                                wait_until="domcontentloaded", 
                                timeout=60000)
            
            # Activar el ranking en vivo si está disponible
            try:
                live_toggle = self.page.locator('input[type="checkbox"]').first
                if await live_toggle.count() > 0:
                    await live_toggle.check()
                    await asyncio.sleep(3)
                    logger.info("✅ Ranking en vivo activado")
            except:
                logger.info("ℹ️ No se encontró toggle de ranking en vivo")
            
            logger.info("✅ Página de rankings cargada")
            await self.handle_cookie_consent()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error navegando a la página de rankings: {str(e)}")
            return False

    async def handle_cookie_consent(self):
        """Manejar el banner de consentimiento de cookies si aparece."""
        try:
            accept_button_selector = "#onetrust-accept-btn-handler"
            await self.page.wait_for_selector(accept_button_selector, timeout=7000)
            await self.page.click(accept_button_selector)
            logger.info("✅ Banner de cookies aceptado.")
            await asyncio.sleep(3)
        except Exception:
            logger.info("ℹ️ No se encontró banner de cookies o ya fue aceptado.")

    async def click_show_more(self, max_clicks=8):
        """🔄 Hacer click en 'Mostrar más' hasta 8 veces en una sección específica"""
        logger.info(f"   🔄 Expandiendo sección de ranking (máx {max_clicks} clicks)...")
        
        clicks_realizados = 0
        max_rank_target = 350

        for i in range(max_clicks):
            try:
                # Extraer el último ranking actual para decidir si continuar
                rows = await self.page.query_selector_all(".rankingTable__row")
                if rows:
                    last_row = rows[-1]
                    last_rank_el = await last_row.query_selector(".rankingTable__cell--rank")
                    if last_rank_el:
                        last_rank_str = await last_rank_el.text_content()
                        last_rank = int(re.sub(r'\D', '', last_rank_str))
                        if last_rank >= max_rank_target:
                            logger.info(f"🎯 Se alcanzó el ranking objetivo de {max_rank_target}. Deteniendo la expansión.")
                            break
                
                show_more_button = self.page.locator('button:has-text("Mostrar más")').first
                if await show_more_button.count() > 0 and await show_more_button.is_visible():
                    await show_more_button.click()
                    await self.page.wait_for_timeout(3000)  # Esperar que carguen más datos
                    clicks_realizados += 1
                    logger.info(f"      ✅ Click {clicks_realizados} en 'Mostrar más' realizado")
                else:
                    logger.info("      ℹ️ No se encontró más botones 'Mostrar más'")
                    break
                    
            except Exception as e:
                logger.warning(f"      ⚠️ Error en click {i+1}: {e}")
                break
        
        logger.info(f"   ✅ Sección de ranking expandida con {clicks_realizados} clicks")
        return clicks_realizados

    async def extract_rankings(self):
        """Extraer la tabla de rankings ATP con todas las métricas avanzadas."""
        logger.info("🎾 Extrayendo rankings ATP con métricas completas de momentum...")
        
        try:
            # Esperar a que la tabla inicial cargue
            rows_selector = ".rankingTable__row"
            logger.info(f"⏳ Esperando por las filas de la tabla con el selector: '{rows_selector}'")
            await self.page.wait_for_selector(rows_selector, timeout=20000)

            # Cargar todos los jugadores hasta el ranking 350
            await self.click_show_more()
            
            player_rows = await self.page.query_selector_all(rows_selector)
            # Omitir la primera fila que es la cabecera (headRow)
            player_rows = [row for row in player_rows if "rankingTable__headRow" not in (await row.get_attribute("class"))]
            
            logger.info(f"🔍 Encontradas {len(player_rows)} filas de jugadores (excluyendo cabecera).")
            
            rankings_data = []
            for idx, row in enumerate(player_rows):
                player_data = await self.extract_enhanced_player_data(row)
                if player_data and player_data.get('name'):
                    # Calcular métricas derivadas
                    player_data = self.calculate_momentum_metrics(player_data)
                    rankings_data.append(player_data)
                    
                    # Log de progreso cada 50 jugadores
                    if (idx + 1) % 50 == 0:
                        logger.info(f"   📊 Procesados {idx + 1}/{len(player_rows)} jugadores...")
            
            logger.info(f"🎯 Total de jugadores extraídos con métricas completas: {len(rankings_data)}")
            return rankings_data

        except Exception as e:
            logger.error(f"❌ Error extrayendo la tabla de rankings: {str(e)}")
            await self.page.screenshot(path="error_screenshot.png")
            logger.info("📸 Screenshot de error guardado en 'error_screenshot.png'")
            return []

    async def extract_enhanced_player_data(self, row_element):
        """Extraer datos completos de una fila de jugador con todas las métricas."""
        try:
            # Selectores básicos
            rank_el = await row_element.query_selector(".rankingTable__cell--rank")
            name_el = await row_element.query_selector(".rankingTable__cell--player a")
            nationality_el = await row_element.query_selector(".rankingTable__cell--player span.flag")
            
            # Selectores para métricas avanzadas
            points_el = await row_element.query_selector(".rankingTable__cell--points")
            tournaments_el = await row_element.query_selector(".rankingTable__cell--tournaments")
            
            # Buscar columnas adicionales para puntos próximos, máximos, cambios, etc.
            all_cells = await row_element.query_selector_all(".rankingTable__cell")
            
            # Extraer datos básicos
            rank = await rank_el.text_content() if rank_el else ""
            name = await name_el.text_content() if name_el else ""
            nationality = await nationality_el.get_attribute('title') if nationality_el else "N/A"
            points = await points_el.text_content() if points_el else ""
            tournaments = await tournaments_el.text_content() if tournaments_el else ""

            if not name:
                return None

            # Inicializar estructura de datos completa
            player_data = {
                # Datos básicos
                'ranking_position': self.safe_int_extract(rank),
                'name': name.strip(),
                'nationality': nationality.strip() if nationality else "N/A",
                'ranking_points': self.safe_int_extract(points),
                'tournaments_played': self.safe_int_extract(tournaments),
                
                # Métricas avanzadas (inicializar en 0, se calcularán después)
                'prox_points': 0,
                'max_points': 0,
                'ranking_change': 0,
                'defense_points': 0,
                
                # Métricas calculadas (se llenan en calculate_momentum_metrics)
                'defense_pressure': 0.0,
                'improvement_potential': 0,
                'momentum_secured': 0,
                
                # Metadata
                'extraction_timestamp': datetime.now().isoformat(),
                'data_source': 'flashscore_atp_rankings'
            }

            # Extraer métricas adicionales de las celdas disponibles
            await self.extract_additional_metrics(row_element, all_cells, player_data)
            
            return player_data

        except Exception as e:
            logger.error(f"❌ Error procesando una fila de jugador: {e}")
            return None

    async def extract_additional_metrics(self, row_element, all_cells, player_data):
        """Extraer métricas adicionales como puntos próximos, máximos, cambios."""
        try:
            # Buscar células con clases específicas o contenido que indique puntos próximos/máximos
            for cell in all_cells:
                cell_class = await cell.get_attribute("class") or ""
                cell_text = await cell.text_content() or ""
                
                # Buscar patrones para identificar diferentes tipos de puntos
                if "nextPoints" in cell_class or "prox" in cell_class.lower():
                    player_data['prox_points'] = self.safe_int_extract(cell_text)
                elif "maxPoints" in cell_class or "max" in cell_class.lower():
                    player_data['max_points'] = self.safe_int_extract(cell_text)
                elif "change" in cell_class or "+" in cell_text or "-" in cell_text:
                    if cell_text.strip() and not cell_text.isdigit():  # Evitar confundir con puntos normales
                        player_data['ranking_change'] = self.safe_signed_int_extract(cell_text)
                
                # Buscar puntos a defender (puede estar en tooltip o data attribute)
                defense_attr = await cell.get_attribute("title")
                if defense_attr and ("defend" in defense_attr.lower() or "defense" in defense_attr.lower()):
                    player_data['defense_points'] = self.safe_int_extract(defense_attr)

            # Si no encontramos puntos próximos/máximos, usar estimaciones conservadoras
            if player_data['prox_points'] == 0:
                player_data['prox_points'] = player_data['ranking_points']
            if player_data['max_points'] == 0:
                player_data['max_points'] = player_data['ranking_points']

        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo métricas adicionales: {e}")

    def safe_int_extract(self, text):
        """Extraer entero de forma segura de un texto."""
        if not text:
            return 0
        number_str = re.sub(r'\D', '', str(text))
        return int(number_str) if number_str else 0

    def safe_signed_int_extract(self, text):
        """Extraer entero con signo de forma segura."""
        if not text:
            return 0
        # Buscar patrón +N o -N
        match = re.search(r'([+-]?\d+)', str(text))
        return int(match.group(1)) if match else 0

    def calculate_momentum_metrics(self, player_data):
        """Calcular métricas derivadas de momentum y presión."""
        try:
            ranking_points = player_data.get('ranking_points', 0)
            prox_points = player_data.get('prox_points', 0)
            max_points = player_data.get('max_points', 0)
            defense_points = player_data.get('defense_points', 0)

            # 1. Presión de defensa (defense_pressure)
            if ranking_points > 0:
                player_data['defense_pressure'] = round(defense_points / ranking_points, 4)
            else:
                player_data['defense_pressure'] = 0.0

            # 2. Potencial de mejora (improvement_potential)
            player_data['improvement_potential'] = max(0, max_points - ranking_points)

            # 3. Momentum asegurado (momentum_secured)
            player_data['momentum_secured'] = max(0, prox_points - ranking_points)

            # 4. Métricas adicionales útiles para análisis
            player_data['points_at_risk_percentage'] = round(player_data['defense_pressure'] * 100, 2)
            player_data['high_defense_pressure'] = player_data['defense_pressure'] > 0.15  # >15% considerado alta presión
            player_data['momentum_positive'] = player_data['momentum_secured'] > 0

            return player_data

        except Exception as e:
            logger.error(f"❌ Error calculando métricas de momentum: {e}")
            return player_data

    async def save_enhanced_rankings_data(self, rankings_data):
        """Guardar datos de rankings mejorados en un solo archivo JSON optimizado."""
        if not rankings_data:
            logger.warning("⚠️ No hay datos de ranking para guardar.")
            return None
        
        try:
            os.makedirs("data", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Crear estructura única optimizada para consumo directo
            output_data = {
                "metadata": {
                    "extraction_timestamp": datetime.now().isoformat(),
                    "total_players": len(rankings_data),
                    "data_source": "flashscore_atp_rankings",
                    "version": "2.0_enhanced",
                    "features_included": [
                        "ranking_position", "ranking_points", "prox_points", "max_points",
                        "ranking_change", "defense_points", "defense_pressure",
                        "improvement_potential", "momentum_secured"
                    ]
                },
                "summary_stats": self.calculate_summary_stats(rankings_data),
                "rankings": rankings_data,
                # Índice por nombre para búsqueda rápida por extraer_historh2h.py
                "players_by_name": {player['name']: player for player in rankings_data}
            }
            
            filename = f"data/atp_rankings_enhanced_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Datos de ranking guardados en: {filename}")
            logger.info(f"📊 Total de jugadores guardados: {len(rankings_data)}")
            
            return filename
            
        except Exception as e:
            logger.error(f"❌ Error guardando los datos del ranking: {str(e)}")
            return None

    def calculate_summary_stats(self, rankings_data):
        """Calcular estadísticas resumidas de los datos extraídos."""
        if not rankings_data:
            return {}
        
        try:
            total_players = len(rankings_data)
            high_pressure_count = sum(1 for p in rankings_data if p.get('high_defense_pressure', False))
            positive_momentum_count = sum(1 for p in rankings_data if p.get('momentum_positive', False))
            
            return {
                "total_players": total_players,
                "players_with_high_defense_pressure": high_pressure_count,
                "players_with_positive_momentum": positive_momentum_count,
                "high_pressure_percentage": round((high_pressure_count / total_players) * 100, 2),
                "positive_momentum_percentage": round((positive_momentum_count / total_players) * 100, 2),
                "avg_defense_pressure": round(sum(p.get('defense_pressure', 0) for p in rankings_data) / total_players, 4),
                "avg_improvement_potential": round(sum(p.get('improvement_potential', 0) for p in rankings_data) / total_players, 2)
            }
        except:
            return {"error": "Could not calculate summary stats"}

    async def close(self):
        """Cerrar el navegador y playwright."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        logger.info("🔚 Navegador cerrado")

async def main():
    """Función principal para ejecutar el Enhanced Ranking Scraper."""
    print("🎾" + "="*70)
    print("🎾 ATP RANKING SCRAPER MEJORADO - ANÁLISIS COMPLETO DE MOMENTUM")
    print("🎾" + "="*70)
    
    scraper = EnhancedRankingScraper()
    
    try:
        await scraper.init_browser()
        
        navigation_success = await scraper.navigate_to_rankings()
        if not navigation_success:
            logger.error("❌ Falló la navegación inicial. Abortando.")
            return
        
        rankings = await scraper.extract_rankings()
        
        if rankings:
            filename = await scraper.save_enhanced_rankings_data(rankings)
            if filename:
                logger.info(f"🎉 ÉXITO: {len(rankings)} jugadores extraídos con métricas completas")
                print("\n📋 EJEMPLO DE DATOS MEJORADOS (TOP 5):")
                for player in rankings[:5]:
                    rank = player['ranking_position']
                    name = player['name']
                    country = player['nationality']
                    points = player['ranking_points']
                    pressure = player['defense_pressure']
                    momentum = player['momentum_secured']
                    potential = player['improvement_potential']
                    
                    print(f"  #{rank} {name} ({country})")
                    print(f"      📊 Puntos: {points} | Momentum: +{momentum} | Potencial: +{potential}")
                    print(f"      ⚡ Presión defensa: {pressure:.1%} | Alto riesgo: {'Sí' if player.get('high_defense_pressure') else 'No'}")
                    print()
                    
                # Mostrar estadísticas resumidas
                if rankings:
                    high_pressure = sum(1 for p in rankings if p.get('high_defense_pressure', False))
                    positive_momentum = sum(1 for p in rankings if p.get('momentum_positive', False))
                    print("📈 RESUMEN DE INTELIGENCIA COMPETITIVA:")
                    print(f"   🔥 Jugadores con alta presión de defensa: {high_pressure}")
                    print(f"   ⚡ Jugadores con momentum positivo: {positive_momentum}")
                    print("   📊 Un solo archivo JSON con estructura optimizada")
                    print("   🔗 Listo para extraer_historh2h.py (usa players_by_name para búsqueda rápida)")
                    
            else:
                logger.error("❌ Error guardando los datos del ranking.")
        else:
            logger.error("❌ No se extrajeron datos del ranking.")
            
    except Exception as e:
        logger.error(f"❌ Error general en Enhanced Ranking Scraper: {str(e)}")
    
    finally:
        await scraper.close()
    
    print("\n✅ Proceso de scraping de ranking mejorado completado.")
    print("🎯 Siguiente paso: Ejecutar extraer_historh2h_version2.py con estos datos")

if __name__ == "__main__":
    asyncio.run(main()) 