#!/usr/bin/env python3
"""
ZITA SCRAPER V3 - VERSIÓN MEJORADA - Extracción Optimizada de Torneos
Extracción directa de datos de FlashScore sin OCR + URLs para H2H automático
MEJORADO: Extracción más precisa de nombres de torneos con múltiples estrategias
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

class ZitaScraper:
    def __init__(self):
        self.browser = None
        self.page = None
        self.detected_tournaments = []
        # Configuración consistente para todas las capturas
        self.capture_config = {
            "x": 0, 
            "y": 0, 
            "width": 1920, 
            "height": 800
        }
    
    async def init_browser(self):
        """Inicializar navegador con configuración optimizada para WSL"""
        logger.info("🚀 Iniciando navegador Zita...")
        
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
        logger.info("✅ Navegador Zita listo")
    
    async def navigate_to_flashscore(self):
        """Navegar a FlashScore Tennis"""
        try:
            logger.info("📍 Navegando a Flashscore Tennis...")
            await self.page.goto("https://www.flashscore.com/tennis/", 
                                wait_until="domcontentloaded", 
                                timeout=45000)
            
            await asyncio.sleep(3)
            logger.info("✅ Página cargada correctamente")

            # Primero, manejar el banner de cookies que puede bloquear otros elementos
            await self.handle_cookie_consent()
            
            # Intentar hacer click en cuotas si están disponibles
            await self.try_click_odds()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error navegando: {str(e)}")
            return False
    
    async def try_click_odds(self):
        """Intentar hacer click en cuotas si están disponibles"""
        cuotas_selectors = [
            'text="Odds"',
            'text="Cuotas"', 
            '[data-testid*="odds"]',
            'a[href*="odds"]',
            'button:has-text("Odds")',
            '[class*="odds"]',
            'text="Bookmakers"',
            'text="Betting"'
        ]
        
        for selector in cuotas_selectors:
            try:
                element = await self.page.wait_for_selector(selector, timeout=3000)
                if element:
                    await element.scroll_into_view_if_needed()
                    await asyncio.sleep(1)
                    await element.click()
                    logger.info("🎯 Click en cuotas realizado!")
                    await asyncio.sleep(3)
                    return True
            except:
                continue
        
        logger.info("ℹ️  No se encontraron cuotas disponibles, continuando...")
        return False

    async def handle_cookie_consent(self):
        """Manejar el banner de consentimiento de cookies si aparece."""
        try:
            # Selector común para el botón de aceptar cookies
            accept_button_selector = "#onetrust-accept-btn-handler"
            # Esperar un tiempo prudencial por si el banner tarda en aparecer
            button = await self.page.wait_for_selector(accept_button_selector, timeout=7000)
            if button:
                await button.click()
                logger.info("✅ Banner de cookies aceptado.")
                # Esperar a que la acción de click se complete y el banner desaparezca
                await asyncio.sleep(2)
        except Exception:
            # Si no se encuentra el botón después del timeout, se asume que no hay banner
            logger.info("ℹ️ No se encontró banner de cookies o ya fue aceptado.")
    
    async def extract_tennis_matches(self):
        """Extraer partidos de tenis con datos estructurados"""
        logger.info("🎾 Extrayendo partidos de tenis...")
        
        matches_data = []
        
        try:
            # Hacer scroll para cargar contenido dinámico
            await self.page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(2)
            
            # Scroll único para cargar todo el contenido
            logger.info("🔄 Realizando scroll para cargar todos los partidos...")
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(3)  # Espera para que cargue el contenido

            # Extraer todos los partidos de una vez
            current_matches = await self.extract_matches_from_dom()
            if current_matches:
                matches_data.extend(current_matches)
                logger.info(f"   ✅ Extraídos {len(current_matches)} partidos.")
            
            # Eliminar duplicados
            unique_matches = self.remove_duplicates(matches_data)
            logger.info(f"🎯 Total partidos únicos extraídos: {len(unique_matches)}")
            
            return unique_matches
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo partidos: {str(e)}")
            return matches_data
    
    def clean_tournament_name(self, tournament_text):
        """
        MÉTODO MEJORADO: Limpia y normaliza los nombres de los torneos de forma más conservadora.
        """
        if not tournament_text:
            return "Sin Nombre"

        # Eliminar prefijos y sufijos no deseados, pero mantener información clave
        clean_name = tournament_text.strip()
        
        # Eliminar texto de "Live" o "Finished"
        clean_name = re.sub(r'^(LIVE|EN VIVO|FINISHED|FINALIZADO)\s*-\s*', '', clean_name, flags=re.IGNORECASE).strip()

        # Reemplazar múltiples espacios con uno solo
        clean_name = re.sub(r'\s+', ' ', clean_name).strip()
        
        # Capitalizar correctamente para mantener la consistencia
        if clean_name.isupper():
            clean_name = clean_name.title()

        # Asegurar que las siglas importantes estén en mayúsculas
        replacements = {
            r'\bAtp\b': 'ATP',
            r'\bWta\b': 'WTA',
            r'\bItf\b': 'ITF',
            r'\bUtr\b': 'UTR',
        }
        for pattern, replacement in replacements.items():
            clean_name = re.sub(pattern, replacement, clean_name, flags=re.IGNORECASE)

        return clean_name if clean_name else "Torneo Sin Nombre"

    async def extract_matches_from_dom(self):
        """
        MÉTODO REESTRUCTURADO: Extrae torneos y partidos en el orden en que aparecen en el DOM.
        Filtra los torneos de dobles.
        """
        matches = []
        current_tournament = "Sin Torneo Asignado"
        is_doubles_tournament = False
        
        try:
            logger.info("🔍 Procesando elementos de la página en orden...")
            
            # Selector unificado para obtener torneos y partidos en el orden del DOM
            all_elements = await self.page.query_selector_all('.event__title, .event__match')
            logger.info(f"   Encontrados {len(all_elements)} elementos totales (torneos y partidos).")

            for element in all_elements:
                element_class = await element.get_attribute('class') or ''

                # Si es un título de torneo, actualizar el torneo actual y verificar si es de dobles
                if 'event__title' in element_class:
                    tournament_text = await element.text_content()
                    if tournament_text:
                        current_tournament = self.clean_tournament_name(tournament_text)
                        # Comprobar si el torneo es de dobles para omitir sus partidos
                        if 'dobles' in current_tournament.lower() or 'doubles' in current_tournament.lower():
                            is_doubles_tournament = True
                            logger.info(f"🚫 Omitiendo torneo de dobles: {current_tournament}")
                        else:
                            is_doubles_tournament = False
                            logger.info(f"🏆 Nuevo torneo detectado: {current_tournament}")
                
                # Si es un partido, extraer sus datos si no es de un torneo de dobles
                elif 'event__match' in element_class:
                    if not is_doubles_tournament:
                        match_data = await self.extract_single_match(element)
                        if match_data and match_data.get('jugador1') and match_data.get('jugador2'):
                            match_data['torneo'] = current_tournament
                            matches.append(match_data)
                            logger.debug(f"   🎾 Partido añadido a '{current_tournament}': {match_data['jugador1']} vs {match_data['jugador2']}")

            logger.info(f"✅ Extracción completada: {len(matches)} partidos de individuales procesados.")
            return matches

        except Exception as e:
            logger.error(f"❌ Error extrayendo del DOM: {str(e)}")
            return matches
    
    async def extract_single_match(self, element):
        """Extraer datos de un partido individual incluyendo URL y Match ID"""
        try:
            match_data = {
                'torneo': '',
                'jugador1': '',
                'jugador2': '',
                'resultado': '',
                'estado': '',
                'hora_partido': None,
                'cuota1': None,
                'cuota2': None,
                'confianza_promedio': None,
                'screenshot_origen': None,
                'match_url': None,
                'match_id': None,
                'h2h_url': None
            }
            
            # Extraer URL del partido y Match ID
            try:
                match_link_selectors = [
                    'a[href*="/partido/"]',
                    'a[href*="/match/"]',
                    'a[href*="tenis/"]',
                    'a'
                ]
                
                match_link = None
                for selector in match_link_selectors:
                    try:
                        match_link = await element.query_selector(selector)
                        if match_link:
                            href = await match_link.get_attribute('href')
                            if href and ('/partido/' in href or '/match/' in href):
                                break
                    except:
                        continue
                
                if match_link:
                    href = await match_link.get_attribute('href')
                    if href:
                        if href.startswith('/'):
                            href = 'https://www.flashscore.com' + href
                        elif not href.startswith('http'):
                            href = 'https://www.flashscore.com/' + href
                        
                        match_data['match_url'] = href
                        
                        match_id_patterns = [
                            r'/partido/tenis/([^/#]+)',
                            r'/match/([^/#]+)',
                            r'/tenis/([^/#]+)',
                            r'/([A-Za-z0-9]{8,})/?'
                        ]
                        
                        for pattern in match_id_patterns:
                            match_id_search = re.search(pattern, href)
                            if match_id_search:
                                match_data['match_id'] = match_id_search.group(1)
                                if '/partido/tenis/' in href:
                                    match_data['h2h_url'] = f"https://www.flashscore.co/partido/tenis/{match_data['match_id']}/#/h2h/general"
                                break
                
            except Exception as e:
                logger.debug(f"Error extrayendo URL/ID: {str(e)}")
                pass
            
            # Extraer jugadores
            try:
                participant_elements = await element.query_selector_all('.event__participant')
                if len(participant_elements) >= 2:
                    match_data['jugador1'] = (await participant_elements[0].text_content()).strip()
                    match_data['jugador2'] = (await participant_elements[1].text_content()).strip()
                else:
                    player_elements = await element.query_selector_all('[class*="participant"]')
                    if len(player_elements) >= 2:
                        match_data['jugador1'] = (await player_elements[0].text_content()).strip()
                        match_data['jugador2'] = (await player_elements[1].text_content()).strip()
            except:
                pass
            
            # Extraer resultado/score
            try:
                score_element = await element.query_selector('[class*="event__scores"], [class*="score"]')
                if score_element:
                    score_text = await score_element.text_content()
                    if score_text:
                        match_data['resultado'] = re.sub(r'\s+', ' ', score_text).strip()
            except:
                pass
            
            # Extraer estado/tiempo y hora del partido
            try:
                time_element = await element.query_selector('[class*="event__time"], [class*="time"]')
                if time_element:
                    time_text = (await time_element.text_content()).strip()
                    if time_text:
                        # Extraer la hora si el formato es HH:MM
                        if re.match(r'^\d{2}:\d{2}$', time_text):
                            match_data['hora_partido'] = time_text
                        match_data['estado'] = self.improve_status(time_text)
            except:
                pass
            
            # Extraer cuotas con lógica mejorada y selectores específicos
            try:
                # Selector para el contenedor de ambas cuotas
                odds_container_selector = '.event__odds'
                odds_container = await element.query_selector(odds_container_selector)
                
                if odds_container:
                    # Selectores más granulares para cada cuota dentro del contenedor
                    # Flashscore a menudo usa 'home' y 'away' para los participantes
                    odd_elements = await odds_container.query_selector_all('.odd__value, .oddsValue')

                    if len(odd_elements) >= 2:
                        # Extraer y validar primera cuota
                        cuota1_text = await odd_elements[0].text_content()
                        if cuota1_text and cuota1_text.strip():
                            cuota1_clean = re.sub(r'[^\d\.]', '', cuota1_text.strip())
                            if cuota1_clean and cuota1_clean.replace('.', '', 1).isdigit():
                                match_data['cuota1'] = float(cuota1_clean)

                        # Extraer y validar segunda cuota
                        cuota2_text = await odd_elements[1].text_content()
                        if cuota2_text and cuota2_text.strip():
                            cuota2_clean = re.sub(r'[^\d\.]', '', cuota2_text.strip())
                            if cuota2_clean and cuota2_clean.replace('.', '', 1).isdigit():
                                match_data['cuota2'] = float(cuota2_clean)

                # Fallback si la lógica anterior falla: buscar selectores de cuotas individuales
                if not match_data['cuota1'] or not match_data['cuota2']:
                    # Selector para la primera cuota (home)
                    cuota1_el = await element.query_selector('[data-testid*="odds-1"], .event__odd--home, .event__odd--odd1')
                    if cuota1_el and not match_data['cuota1']:
                        cuota1_text = await cuota1_el.text_content()
                        cuota1_clean = re.sub(r'[^\d\.]', '', cuota1_text.strip())
                        if cuota1_clean and cuota1_clean.replace('.', '', 1).isdigit():
                            match_data['cuota1'] = float(cuota1_clean)

                    # Selector para la segunda cuota (away)
                    cuota2_el = await element.query_selector('[data-testid*="odds-2"], .event__odd--away, .event__odd--odd2')
                    if cuota2_el and not match_data['cuota2']:
                        cuota2_text = await cuota2_el.text_content()
                        cuota2_clean = re.sub(r'[^\d\.]', '', cuota2_text.strip())
                        if cuota2_clean and cuota2_clean.replace('.', '', 1).isdigit():
                            match_data['cuota2'] = float(cuota2_clean)

                # Calcular confianza solo si tenemos ambas cuotas válidas
                if match_data['cuota1'] and match_data['cuota2']:
                    try:
                        prob1 = 1 / match_data['cuota1']
                        prob2 = 1 / match_data['cuota2']
                        total_prob = prob1 + prob2
                        if total_prob > 0:
                            match_data['confianza_promedio'] = {
                                "jugador1": round(prob1 / total_prob, 3),
                                "jugador2": round(prob2 / total_prob, 3)
                            }
                    except (ValueError, ZeroDivisionError) as e:
                        logger.debug(f"Error calculando probabilidades: {e}")
                        pass
                        
            except Exception as e:
                logger.debug(f"Error extrayendo cuotas: {str(e)}")
                pass
            
            if not match_data['jugador1'] or not match_data['jugador2']:
                return None
            
            return match_data
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo partido individual: {str(e)}")
            return None
    
    def improve_status(self, status):
        """Mejorar el formato del estado del partido"""
        if not status:
            return "No disponible"
        
        status_lower = status.lower()
        
        if "finished" in status_lower:
            return "Finalizado"
        elif "walkover" in status_lower:
            return "Walkover"
        elif "retired" in status_lower:
            return "Retirado"
        elif "cancelled" in status_lower:
            return "Cancelado"
        elif "postponed" in status_lower:
            return "Pospuesto"
        elif "interrupted" in status_lower:
            return "Interrumpido"
        elif re.search(r'set \d', status_lower):
            return "En vivo"
        elif status.strip() == "-":
            return "Próximamente"
        elif re.match(r'^\d{2}:\d{2}$', status.strip()):
            return f"Programado {status.strip()}"
        else:
            return status.strip()
    
    def remove_duplicates(self, matches_data):
        """Eliminar partidos duplicados"""
        seen = set()
        unique_matches = []
        
        for match in matches_data:
            key = f"{match.get('jugador1', '')}|{match.get('jugador2', '')}"
            if key not in seen and match.get('jugador1') and match.get('jugador2'):
                seen.add(key)
                unique_matches.append(match)
        
        return unique_matches
    
    async def save_matches_data(self, matches_data):
        """Guardar datos en archivo JSON con formato mejorado (agrupado por torneo)"""
        try:
            os.makedirs("data", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/zita_tennis_matches_{timestamp}.json"

            # Agrupar partidos por torneo
            grouped_data = {}
            for match in matches_data:
                tournament_name = match.pop('torneo', 'Sin Torneo Asignado')
                if not tournament_name.strip():
                    tournament_name = 'Sin Torneo Asignado'
                
                if tournament_name not in grouped_data:
                    grouped_data[tournament_name] = []
                
                grouped_data[tournament_name].append(match)

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(grouped_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Datos guardados en: {filename}")
            logger.info(f"📊 Total partidos guardados: {len(matches_data)}")

            # Estadísticas
            with_odds = sum(1 for match in matches_data if match.get('cuota1') and match.get('cuota2'))
            with_urls = sum(1 for match in matches_data if match.get('match_url'))
            with_match_ids = sum(1 for match in matches_data if match.get('match_id'))
            with_h2h_urls = sum(1 for match in matches_data if match.get('h2h_url'))
            
            logger.info(f"🏆 Torneos únicos: {len(grouped_data)}")
            logger.info(f"📊 Partidos con cuotas: {with_odds}/{len(matches_data)}")
            logger.info(f"🔗 Partidos con URLs: {with_urls}/{len(matches_data)}")
            logger.info(f"🆔 Partidos con Match IDs: {with_match_ids}/{len(matches_data)}")
            logger.info(f"🎯 Partidos con URLs H2H: {with_h2h_urls}/{len(matches_data)}")
            
            # Mostrar ejemplos mejorados
            logger.info("📋 EJEMPLOS DE PARTIDOS EXTRAÍDOS (con Torneos Mejorados):")
            
            count = 0
            for tournament, matches in grouped_data.items():
                if count >= 3: break
                logger.info(f"   🏆 {tournament}:")
                for i, match in enumerate(matches[:2], 1):
                    if count >= 6: break
                    logger.info(f"      {i}. {match.get('jugador1', 'N/A')} vs {match.get('jugador2', 'N/A')}")
                    logger.info(f"         📊 Estado: {match.get('estado', 'N/A')}")
                    if match.get('hora_partido'):
                        logger.info(f"         🕒 Hora: {match.get('hora_partido')}")
                    if match.get('cuota1') and match.get('cuota2'):
                        logger.info(f"         💰 Cuotas: {match.get('cuota1')} - {match.get('cuota2')}")
                    if match.get('match_id'):
                        logger.info(f"         🆔 Match ID: {match.get('match_id')}")
                    count += 1
            
            return filename, grouped_data
            
        except Exception as e:
            logger.error(f"❌ Error guardando datos: {str(e)}")
            return None, {}
    
    async def take_screenshot(self, filename_prefix="zita_screenshot"):
        """Tomar captura de pantalla de la página actual"""
        try:
            os.makedirs("screenshots", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"screenshots/{filename_prefix}_{timestamp}.png"
            
            await self.page.screenshot(
                path=screenshot_path,
                clip=self.capture_config
            )
            
            logger.info(f"📸 Captura guardada: {screenshot_path}")
            return screenshot_path
            
        except Exception as e:
            logger.error(f"❌ Error tomando captura: {str(e)}")
            return None
    
    async def close(self):
        """Cerrar navegador y limpiar recursos"""
        try:
            if self.browser:
                await self.browser.close()
            if hasattr(self, 'playwright'):
                await self.playwright.stop()
            logger.info("🔄 Navegador cerrado correctamente")
        except Exception as e:
            logger.error(f"❌ Error cerrando navegador: {str(e)}")

async def main():
    """Función principal"""
    scraper = ZitaScraper()
    
    try:
        logger.info("🎾 === ZITA SCRAPER V3 - VERSIÓN MEJORADA ===")
        logger.info("🚀 Iniciando extracción de datos de FlashScore...")
        
        # Inicializar navegador
        await scraper.init_browser()
        
        # Navegar a FlashScore
        success = await scraper.navigate_to_flashscore()
        if not success:
            logger.error("❌ Falló la navegación inicial")
            return
        
        # Tomar captura inicial
        await scraper.take_screenshot("flashscore_inicial")
        
        # Extraer partidos de tenis
        matches_data = await scraper.extract_tennis_matches()
        
        if not matches_data:
            logger.error("❌ No se extrajeron datos de partidos")
            return
        
        # Tomar captura final
        await scraper.take_screenshot("flashscore_final")
        
        
        # Guardar datos
        filename, grouped_data = await scraper.save_matches_data(matches_data)
        
        if filename:
            logger.info("✅ === EXTRACCIÓN COMPLETADA EXITOSAMENTE ===")
            logger.info(f"📁 Archivo generado: {filename}")
            logger.info(f"🎾 Total partidos: {len(matches_data)}")
            logger.info(f"🏆 Total torneos: {len(grouped_data)}")
        else:
            logger.error("❌ Error guardando los datos")
    
    except Exception as e:
        logger.error(f"❌ Error en ejecución principal: {str(e)}")
    
    finally:
        await scraper.close()

if __name__ == "__main__":
    asyncio.run(main())
