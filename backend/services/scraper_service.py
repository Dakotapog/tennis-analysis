# services/scraper_service.py - Servicio de scraping de datos de tenis
# 🔴 ARCHIVO CRÍTICO - Extrae datos de Flashscore - VERSIÓN CORREGIDA

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
import time
import re
from datetime import datetime
import logging
import json

# 📦 IMPORTAR FUNCIONES DE BASE DE DATOS
from models.database import insert_player, get_player_by_name, get_all_players

# 🔧 CONFIGURAR LOGGING
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TennisScraperService:
    """
    Servicio para extraer datos de tenis desde Flashscore
    Versión corregida con selectores actualizados y mejor manejo de errores
    """
    
    def __init__(self):
        self.base_url = "https://www.flashscore.com"
        self.tennis_url = "https://www.flashscore.com/tennis/"
        self.driver = None
        self.max_retries = 3
        self.timeout = 15
        
    def setup_driver(self):
        """
        Configurar Chrome WebDriver para scraping con mejor configuración
        """
        logger.info("🔧 Configurando Chrome WebDriver...")
        
        try:
            # Configuraciones de Chrome mejoradas
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            chrome_options.add_argument("--accept-language=en-US,en;q=0.9")
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument("--allow-running-insecure-content")
            
            # Deshabilitar imágenes para velocidad
            chrome_options.add_experimental_option("prefs", {
                "profile.managed_default_content_settings.images": 2,
                "profile.default_content_setting_values.notifications": 2
            })
            
            # Crear driver
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
            logger.info("✅ Chrome WebDriver configurado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error configurando Chrome WebDriver: {e}")
            return False
    
    def close_driver(self):
        """
        Cerrar el driver de Chrome de forma segura
        """
        try:
            if self.driver:
                self.driver.quit()
                logger.info("🔒 Chrome WebDriver cerrado")
        except Exception as e:
            logger.error(f"⚠️ Error cerrando driver: {e}")
    
    def wait_for_element(self, selector, by=By.CSS_SELECTOR, timeout=None):
        """
        Esperar por un elemento con múltiples selectores de respaldo
        """
        if timeout is None:
            timeout = self.timeout
            
        try:
            wait = WebDriverWait(self.driver, timeout)
            element = wait.until(EC.presence_of_element_located((by, selector)))
            return element
        except TimeoutException:
            logger.warning(f"⏰ Timeout esperando elemento: {selector}")
            return None
    
    def scrape_atp_rankings(self, limit=50):
        """
        Scraping del ranking ATP desde Flashscore con selectores actualizados
        """
        logger.info(f"🎾 Scraping ranking ATP (top {limit})...")
        
        if not self.setup_driver():
            return {'success': False, 'error': 'No se pudo configurar el driver', 'data': []}
        
        try:
            # URL del ranking ATP
            ranking_url = f"{self.tennis_url}rankings/atp/"
            logger.info(f"📍 Accediendo a: {ranking_url}")
            
            self.driver.get(ranking_url)
            time.sleep(5)  # Esperar más tiempo para la carga
            
            # Selectores actualizados para 2025
            possible_selectors = [
                "div[data-testid='wcl-table']",  # Nuevo selector
                ".wcl-table",
                "table.ui-table",
                ".table",
                ".rankings-table",
                "[class*='table']"
            ]
            
            table_element = None
            for selector in possible_selectors:
                table_element = self.wait_for_element(selector, timeout=5)
                if table_element:
                    logger.info(f"✅ Tabla encontrada con selector: {selector}")
                    break
            
            if not table_element:
                logger.error("❌ No se encontró la tabla de rankings")
                return {'success': False, 'error': 'Tabla de rankings no encontrada', 'data': []}
            
            players_data = []
            
            # Selectores actualizados para filas de jugadores
            row_selectors = [
                "div[data-testid='wcl-table-row']",
                ".wcl-table-row",
                "tr.ui-table__row",
                "tr.table__row",
                ".ranking-row",
                "tr[class*='row']"
            ]
            
            player_rows = []
            for selector in row_selectors:
                try:
                    rows = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if rows:
                        player_rows = rows
                        logger.info(f"✅ Filas encontradas con selector: {selector} ({len(rows)} filas)")
                        break
                except:
                    continue
            
            if not player_rows:
                logger.error("❌ No se encontraron filas de jugadores")
                return {'success': False, 'error': 'No se encontraron filas de jugadores', 'data': []}
            
            # Procesar jugadores
            for i, row in enumerate(player_rows[:limit]):
                try:
                    # Múltiples selectores para cada campo
                    ranking_selectors = [
                        "[data-testid='wcl-rank']",
                        ".wcl-rank",
                        ".ui-table__body-cell--rank",
                        ".table__cell--rank",
                        ".rank"
                    ]
                    
                    name_selectors = [
                        "[data-testid='wcl-participant-name']",
                        ".wcl-participant-name",
                        ".ui-table__body-cell--participant",
                        ".table__cell--participant",
                        ".participant-name"
                    ]
                    
                    country_selectors = [
                        "[data-testid='wcl-flag']",
                        ".wcl-flag",
                        ".ui-table__body-cell--country",
                        ".table__cell--country",
                        ".country"
                    ]
                    
                    # Extraer ranking
                    ranking = None
                    for selector in ranking_selectors:
                        try:
                            ranking_element = row.find_element(By.CSS_SELECTOR, selector)
                            ranking_text = ranking_element.text.strip()
                            ranking = int(re.search(r'\d+', ranking_text).group()) if re.search(r'\d+', ranking_text) else None
                            if ranking:
                                break
                        except:
                            continue
                    
                    # Extraer nombre
                    name = None
                    for selector in name_selectors:
                        try:
                            name_element = row.find_element(By.CSS_SELECTOR, selector)
                            name = name_element.text.strip()
                            if name:
                                break
                        except:
                            continue
                    
                    # Si no encontramos nombre, intentar con el texto completo de la fila
                    if not name:
                        name = row.text.strip()
                        # Limpiar el nombre (remover números y símbolos)
                        name = re.sub(r'^\d+\.?\s*', '', name)  # Remover ranking al inicio
                        name = re.sub(r'\s+\d+$', '', name)    # Remover puntos al final
                        name = name.split('\n')[0] if '\n' in name else name
                    
                    # Extraer país
                    country = "Unknown"
                    for selector in country_selectors:
                        try:
                            country_element = row.find_element(By.CSS_SELECTOR, selector)
                            country = country_element.get_attribute('title') or country_element.text.strip() or "Unknown"
                            if country and country != "Unknown":
                                break
                        except:
                            continue
                    
                    # Validar datos mínimos
                    if not ranking:
                        ranking = i + 1  # Usar índice como ranking de respaldo
                    
                    if not name or len(name) < 2:
                        logger.warning(f"⚠️ Nombre inválido en fila {i+1}, saltando...")
                        continue
                    
                    # Extraer edad (opcional)
                    age = None
                    age_selectors = [
                        "[data-testid='wcl-age']",
                        ".wcl-age",
                        ".ui-table__body-cell--age",
                        ".table__cell--age"
                    ]
                    
                    for selector in age_selectors:
                        try:
                            age_element = row.find_element(By.CSS_SELECTOR, selector)
                            age_text = age_element.text.strip()
                            age = int(re.search(r'\d+', age_text).group()) if re.search(r'\d+', age_text) else None
                            if age:
                                break
                        except:
                            continue
                    
                    player_data = {
                        'ranking': ranking,
                        'name': name,
                        'country': country,
                        'age': age
                    }
                    
                    players_data.append(player_data)
                    logger.info(f"✅ Jugador {i+1}: {name} (#{ranking}) - {country}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error procesando jugador {i+1}: {e}")
                    continue
            
            if players_data:
                logger.info(f"📊 Scraping completado: {len(players_data)} jugadores extraídos")
                return {'success': True, 'data': players_data, 'count': len(players_data)}
            else:
                logger.error("❌ No se pudieron extraer datos de jugadores")
                return {'success': False, 'error': 'No se pudieron extraer datos', 'data': []}
            
        except Exception as e:
            logger.error(f"❌ Error en scraping ATP rankings: {e}")
            return {'success': False, 'error': f'Error en scraping: {str(e)}', 'data': []}
        finally:
            self.close_driver()
    
    def scrape_player_matches(self, player_name, limit=10):
        """
        Scraping de partidos recientes de un jugador
        """
        logger.info(f"🎾 Scraping partidos de {player_name}...")
        
        if not self.setup_driver():
            return {'success': False, 'error': 'No se pudo configurar el driver', 'data': []}
        
        try:
            # Buscar jugador en Flashscore
            search_query = player_name.replace(' ', '+')
            search_url = f"{self.base_url}/search/?q={search_query}"
            logger.info(f"🔍 Buscando: {search_url}")
            
            self.driver.get(search_url)
            time.sleep(3)
            
            # Buscar primer resultado
            result_selectors = [
                "[data-testid='search-result-participant']",
                ".search-result__participant",
                ".searchResult__participant",
                ".search-participant",
                "a[href*='tennis']"
            ]
            
            first_result = None
            for selector in result_selectors:
                try:
                    first_result = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if first_result:
                        break
                except:
                    continue
            
            if not first_result:
                logger.error(f"❌ No se encontró el jugador {player_name}")
                return {'success': False, 'error': f'Jugador {player_name} no encontrado', 'data': []}
            
            first_result.click()
            time.sleep(3)
            
            # Buscar sección de resultados
            results_selectors = [
                "a[href*='results']",
                "a[data-testid='results-tab']",
                "a[text()='Results']",
                ".results-tab"
            ]
            
            results_tab = None
            for selector in results_selectors:
                try:
                    results_tab = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if results_tab:
                        break
                except:
                    continue
            
            if results_tab:
                results_tab.click()
                time.sleep(2)
            
            matches_data = []
            
            # Buscar partidos
            match_selectors = [
                "[data-testid='match-row']",
                ".match-row",
                ".tennis-match",
                ".ui-table__row"
            ]
            
            match_rows = []
            for selector in match_selectors:
                try:
                    rows = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if rows:
                        match_rows = rows
                        break
                except:
                    continue
            
            for i, match in enumerate(match_rows[:limit]):
                try:
                    match_text = match.text.strip()
                    if match_text:
                        # Parsear información básica del partido
                        match_data = {
                            'date': 'N/A',
                            'opponent': 'N/A',
                            'score': 'N/A',
                            'tournament': 'N/A',
                            'raw_text': match_text
                        }
                        
                        matches_data.append(match_data)
                        logger.info(f"✅ Partido {i+1}: {match_text[:50]}...")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error procesando partido {i+1}: {e}")
                    continue
            
            return {'success': True, 'data': matches_data, 'count': len(matches_data)}
            
        except Exception as e:
            logger.error(f"❌ Error scraping partidos de {player_name}: {e}")
            return {'success': False, 'error': f'Error scraping partidos: {str(e)}', 'data': []}
        finally:
            self.close_driver()
    
    def save_players_to_database(self, players_data):
        """
        Guardar jugadores scrapeados en la base de datos con mejor manejo de errores
        """
        logger.info("💾 Guardando jugadores en la base de datos...")
        
        if not players_data:
            logger.warning("⚠️ No hay datos de jugadores para guardar")
            return {'success': False, 'saved_count': 0, 'errors': ['No hay datos para guardar']}
        
        saved_count = 0
        errors = []
        
        for player in players_data:
            try:
                # Validar datos del jugador
                if not player.get('name') or len(player['name']) < 2:
                    errors.append(f"Nombre inválido: {player.get('name', 'N/A')}")
                    continue
                
                player_id = insert_player(
                    name=player['name'],
                    ranking=player.get('ranking', 0),
                    country=player.get('country', 'Unknown'),
                    age=player.get('age')
                )
                
                if player_id:
                    saved_count += 1
                    logger.info(f"✅ Guardado: {player['name']}")
                else:
                    errors.append(f"No se pudo guardar: {player['name']}")
                    
            except Exception as e:
                error_msg = f"Error guardando {player.get('name', 'N/A')}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"❌ {error_msg}")
        
        logger.info(f"📊 Guardado completado: {saved_count} jugadores guardados")
        
        return {
            'success': saved_count > 0,
            'saved_count': saved_count,
            'errors': errors,
            'total_attempted': len(players_data)
        }
    
    def scrape_and_save_atp_top(self, limit=50):
        """
        Scraping completo: extraer y guardar top ATP con mejor manejo de errores
        """
        logger.info(f"🚀 Iniciando scraping completo del top {limit} ATP...")
        
        try:
            # Scraping
            scrape_result = self.scrape_atp_rankings(limit)
            
            if not scrape_result['success']:
                logger.error(f"❌ Error en scraping: {scrape_result['error']}")
                return {
                    'success': False,
                    'error': scrape_result['error'],
                    'scraped_count': 0,
                    'saved_count': 0
                }
            
            players_data = scrape_result['data']
            
            if not players_data:
                logger.error("❌ No se pudieron extraer datos")
                return {
                    'success': False,
                    'error': 'No se pudieron extraer datos',
                    'scraped_count': 0,
                    'saved_count': 0
                }
            
            # Guardar en base de datos
            save_result = self.save_players_to_database(players_data)
            
            result = {
                'success': save_result['success'],
                'scraped_count': len(players_data),
                'saved_count': save_result['saved_count'],
                'errors': save_result.get('errors', [])
            }
            
            if save_result['success']:
                logger.info(f"🎉 Scraping completado exitosamente: {save_result['saved_count']} jugadores guardados")
            else:
                logger.error(f"❌ Error guardando datos: {save_result['errors']}")
                result['error'] = 'Error guardando en base de datos'
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en scrape_and_save_atp_top: {e}")
            return {
                'success': False,
                'error': f'Error inesperado: {str(e)}',
                'scraped_count': 0,
                'saved_count': 0
            }

# 🏭 FUNCIONES AUXILIARES MEJORADAS
def get_scraper():
    """
    Crear instancia del scraper
    """
    return TennisScraperService()

def quick_scrape_top_10():
    """
    Función rápida para scrapear el top 10 ATP con manejo de errores
    """
    try:
        scraper = get_scraper()
        return scraper.scrape_and_save_atp_top(10)
    except Exception as e:
        logger.error(f"❌ Error en quick_scrape_top_10: {e}")
        return {
            'success': False,
            'error': f'Error inesperado: {str(e)}',
            'scraped_count': 0,
            'saved_count': 0
        }

def scrape_player(player_name):
    """
    Scrapear información de un jugador específico
    """
    try:
        scraper = get_scraper()
        return scraper.scrape_player_matches(player_name)
    except Exception as e:
        logger.error(f"❌ Error scraping jugador {player_name}: {e}")
        return {
            'success': False,
            'error': f'Error inesperado: {str(e)}',
            'data': []
        }

# 🧪 FUNCIÓN DE PRUEBA MEJORADA
def test_scraper():
    """
    Probar el scraper con datos reales y mejor reporte
    """
    logger.info("🧪 Probando scraper de tenis...")
    
    try:
        # Crear scraper
        scraper = TennisScraperService()
        
        # Probar scraping del top 5
        logger.info("1. Probando scraping del top 5 ATP...")
        result = scraper.scrape_and_save_atp_top(5)
        
        print(f"📊 Resultado del scraping:")
        print(f"   - Éxito: {result['success']}")
        print(f"   - Jugadores scrapeados: {result['scraped_count']}")
        print(f"   - Jugadores guardados: {result['saved_count']}")
        
        if result.get('errors'):
            print(f"   - Errores: {len(result['errors'])}")
            for error in result['errors'][:3]:  # Mostrar solo los primeros 3
                print(f"     * {error}")
        
        if result['success']:
            logger.info("✅ Scraping exitoso")
            
            # Mostrar jugadores guardados
            try:
                players = get_all_players()
                print(f"📈 Jugadores en la base de datos: {len(players)}")
                for player in players[:5]:
                    print(f"   - {player['name']} (#{player['ranking']}) - {player['country']}")
            except Exception as e:
                logger.error(f"❌ Error obteniendo jugadores de la BD: {e}")
        else:
            logger.error(f"❌ Error en scraping: {result.get('error', 'Error desconocido')}")
            
    except Exception as e:
        logger.error(f"❌ Error en test_scraper: {e}")
        print(f"💥 Error general: {e}")

# 🚀 EJECUTAR PRUEBA SI SE EJECUTA DIRECTAMENTE
if __name__ == '__main__':
    test_scraper()