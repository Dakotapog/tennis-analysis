#!/usr/bin/env python3
"""
ZITA SCRAPER V3 - Con extracción de URLs y Match IDs
Extracción directa de datos de FlashScore sin OCR + URLs para H2H automático
NUEVO: Extrae match_url y match_id para cada partido automáticamente
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
    
    async def extract_matches_from_dom(self):
        """Extraer partidos del DOM usando selectores validados"""
        matches = []
        
        try:
            # Usar el selector más efectivo según flashscore_inspector
            match_elements = await self.page.query_selector_all('.event__match')
            logger.info(f"🔍 Encontrados {len(match_elements)} elementos de partidos")
            
            for element in match_elements:
                try:
                    match_data = await self.extract_single_match(element)
                    if match_data and match_data.get('jugador1') and match_data.get('jugador2'):
                        matches.append(match_data)
                except:
                    continue
            
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
                'cuota1': None,
                'cuota2': None,
                'confianza_promedio': None,
                'screenshot_origen': None,
                'match_url': None,           # NUEVO
                'match_id': None,            # NUEVO
                'h2h_url': None             # NUEVO
            }
            
            # NUEVO: Extraer URL del partido y Match ID
            try:
                # Buscar enlaces dentro del elemento que contengan la URL del partido
                match_link_selectors = [
                    'a[href*="/partido/"]',
                    'a[href*="/match/"]',
                    'a[href*="tenis/"]',
                    'a'  # Fallback: cualquier enlace
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
                        # Limpiar y normalizar URL
                        if href.startswith('/'):
                            href = 'https://www.flashscore.com' + href
                        elif not href.startswith('http'):
                            href = 'https://www.flashscore.com/' + href
                        
                        match_data['match_url'] = href
                        
                        # Extraer Match ID usando múltiples patrones
                        match_id_patterns = [
                            r'/partido/tenis/([^/#]+)',
                            r'/match/([^/#]+)',
                            r'/tenis/([^/#]+)',
                            r'/([A-Za-z0-9]{8,})/?'  # Patrón genérico para IDs
                        ]
                        
                        for pattern in match_id_patterns:
                            match_id_search = re.search(pattern, href)
                            if match_id_search:
                                match_data['match_id'] = match_id_search.group(1)
                                # Construir URL H2H directamente
                                if '/partido/tenis/' in href:
                                    match_data['h2h_url'] = f"https://www.flashscore.co/partido/tenis/{match_data['match_id']}/#/h2h/general"
                                break
                        
                        logger.debug(f"🔗 URL extraída: {href}")
                        if match_data['match_id']:
                            logger.debug(f"🆔 Match ID: {match_data['match_id']}")
                
            except Exception as e:
                logger.debug(f"Error extrayendo URL/ID: {str(e)}")
                pass
            
            # Extraer torneo
            try:
                # Buscar el contenedor padre que tenga información del torneo
                tournament_element = await element.query_selector('xpath=ancestor::*[contains(@class, "sportName") or contains(@class, "event__header")]')
                if not tournament_element:
                    # Fallback: buscar en elementos hermanos
                    tournament_element = await element.query_selector('xpath=preceding-sibling::*[contains(@class, "sportName")]')
                
                if tournament_element:
                    tournament_text = await tournament_element.text_content()
                    if tournament_text:
                        # Limpiar nombre del torneo
                        clean_tournament = re.sub(r'\(.*?\)', '', tournament_text).strip()
                        clean_tournament = re.sub(r'\s*-\s*(SINGLES|DOUBLES)', '', clean_tournament, flags=re.IGNORECASE).strip()
                        match_data['torneo'] = clean_tournament
            except:
                pass
            
            # Extraer jugadores usando el selector validado
            try:
                participant_elements = await element.query_selector_all('.event__participant')
                if len(participant_elements) >= 2:
                    match_data['jugador1'] = (await participant_elements[0].text_content()).strip()
                    match_data['jugador2'] = (await participant_elements[1].text_content()).strip()
                else:
                    # Fallback: buscar por otros selectores
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
            
            # Extraer estado/tiempo
            try:
                time_element = await element.query_selector('[class*="event__time"], [class*="time"]')
                if time_element:
                    time_text = await time_element.text_content()
                    if time_text:
                        match_data['estado'] = self.improve_status(time_text.strip())
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
            
            # Si no tenemos jugadores, no es un partido válido
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
        
        # Mapear estados comunes
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
            # Crear clave única basada en jugadores
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

            # Agrupar partidos por torneo para un JSON más estructurado
            grouped_data = {}
            for match in matches_data:
                # Usar un nombre de torneo por defecto si no se encuentra
                tournament_name = match.pop('torneo', 'Sin Torneo Asignado')
                if not tournament_name.strip():
                    tournament_name = 'Sin Torneo Asignado'
                
                if tournament_name not in grouped_data:
                    grouped_data[tournament_name] = []
                
                # El campo 'torneo' ya fue extraído, el resto del match se añade a la lista
                grouped_data[tournament_name].append(match)

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(grouped_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Datos guardados en: {filename}")
            logger.info(f"📊 Total partidos guardados: {len(matches_data)}")

            # Mostrar estadísticas
            with_odds = sum(1 for match in matches_data if match.get('cuota1') and match.get('cuota2'))
            with_urls = sum(1 for match in matches_data if match.get('match_url'))
            with_match_ids = sum(1 for match in matches_data if match.get('match_id'))
            with_h2h_urls = sum(1 for match in matches_data if match.get('h2h_url'))
            
            logger.info(f"🏆 Torneos únicos: {len(grouped_data)}")
            logger.info(f"📊 Partidos con cuotas: {with_odds}/{len(matches_data)}")
            logger.info(f"🔗 Partidos con URLs: {with_urls}/{len(matches_data)}")
            logger.info(f"🆔 Partidos con Match IDs: {with_match_ids}/{len(matches_data)}")
            logger.info(f"🎯 Partidos con URLs H2H: {with_h2h_urls}/{len(matches_data)}")
            
            # Mostrar algunos ejemplos con URLs
            logger.info("📋 EJEMPLOS DE PARTIDOS EXTRAÍDOS (con URLs):")
            
            count = 0
            for tournament, matches in grouped_data.items():
                if count >= 2: break # Mostrar ejemplos de los primeros 2 torneos
                logger.info(f"   🏆 {tournament}:")
                for i, match in enumerate(matches[:2], 1): # Mostrar hasta 2 partidos por torneo
                    if count >= 5: break
                    logger.info(f"      {i}. {match.get('jugador1', 'N/A')} vs {match.get('jugador2', 'N/A')}")
                    logger.info(f"         📊 Estado: {match.get('estado', 'N/A')}")
                    if match.get('cuota1') and match.get('cuota2'):
                        logger.info(f"         💰 Cuotas: {match.get('cuota1')} - {match.get('cuota2')}")
                    if match.get('match_id'):
                        logger.info(f"         🆔 Match ID: {match.get('match_id')}")
                    if match.get('h2h_url'):
                        logger.info(f"         🎯 H2H URL: {match.get('h2h_url')}")
                    count += 1
            
            return filename, grouped_data
            
        except Exception as e:
            logger.error(f"❌ Error guardando datos: {str(e)}")
            return None, None
    
    # NUEVA FUNCIÓN: Generar archivo específico para H2H automático
    async def save_h2h_queue(self, matches_data):
        """Guardar archivo específico para extracción H2H automática"""
        try:
            os.makedirs("data", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/h2h_extraction_queue_{timestamp}.json"
            
            # Crear cola de partidos para H2H solo con partidos que tengan URL H2H
            h2h_queue = []
            for match in matches_data:
                if match.get('h2h_url') and match.get('match_id'):
                    h2h_match = {
                        'match_id': match['match_id'],
                        'h2h_url': match['h2h_url'],
                        'player1': match.get('jugador1', ''),
                        'player2': match.get('jugador2', ''),
                        'tournament': match.get('torneo', ''),
                        'status': 'pending',
                        'extracted_at': None
                    }
                    h2h_queue.append(h2h_match)
            
            queue_data = {
                'extraction_date': datetime.now().strftime("%Y-%m-%d"),
                'total_matches': len(h2h_queue),
                'processed': 0,
                'current_match': 0,
                'matches': h2h_queue
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(queue_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"🎯 Cola H2H guardada: {filename}")
            logger.info(f"📊 Partidos listos para H2H: {len(h2h_queue)}")
            
            return filename, h2h_queue
            
        except Exception as e:
            logger.error(f"❌ Error guardando cola H2H: {str(e)}")
            return None, None
    
    async def close(self):
        """Cerrar navegador"""
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
        logger.info("🔚 Navegador cerrado")

async def main():
    """Función principal de Zita Scraper V3"""
    print("🎾" + "="*60)
    print("🎾 ZITA SCRAPER V3 - EXTRACCIÓN CON URLs Y MATCH IDs")
    print("🎾" + "="*60)
    
    scraper = ZitaScraper()
    
    try:
        # Inicializar navegador
        await scraper.init_browser()
        
        # Navegar a FlashScore
        logger.info("🔍 FASE 1: Navegando a FlashScore...")
        navigation_success = await scraper.navigate_to_flashscore()
        
        if not navigation_success:
            logger.error("❌ Falló la navegación inicial")
            return
        
        # Extraer partidos
        logger.info("🎾 FASE 2: Extrayendo partidos de tenis con URLs...")
        matches_data = await scraper.extract_tennis_matches()
        
        if matches_data:
            # Guardar datos principales
            logger.info("💾 FASE 3: Guardando y estructurando datos...")
            filename, grouped_data = await scraper.save_matches_data(matches_data)
            
            # Guardar cola específica para H2H
            logger.info("🎯 FASE 4: Generando cola para extracción H2H automática...")
            h2h_filename, h2h_queue = await scraper.save_h2h_queue(matches_data)
            
            if filename and grouped_data:
                logger.info(f"🎉 ÉXITO: {len(matches_data)} partidos extraídos y guardados")
                
                if h2h_queue:
                    logger.info(f"🎯 BONUS: {len(h2h_queue)} URLs H2H listas para extracción automática")
                    logger.info(f"📋 Archivo cola H2H: {h2h_filename}")
                    
                    # Mostrar primeras 3 URLs H2H como ejemplo
                    logger.info("🔗 PRIMERAS URLs H2H DETECTADAS:")
                    for i, match in enumerate(h2h_queue[:3], 1):
                        logger.info(f"   {i}. {match['player1']} vs {match['player2']}")
                        logger.info(f"      🆔 ID: {match['match_id']}")
                        logger.info(f"      🎯 H2H: {match['h2h_url']}")
                
            else:
                logger.error("❌ Error guardando o estructurando los datos")
        else:
            logger.error("❌ No se extrajeron partidos")
        
    except Exception as e:
        logger.error(f"❌ Error general en Zita Scraper V3: {str(e)}")
    
    finally:
        await scraper.close()
    
    print("\n🎾 ZITA SCRAPER V3 COMPLETADO")
    print("🚀 Datos y URLs H2H listos para análisis automático")
    print("🎯 Siguiente paso: Usar las URLs H2H para extracción masiva")

if __name__ == "__main__":
    asyncio.run(main())
