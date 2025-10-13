#!/usr/bin/env python3
"""
ZITA SCRAPER V2 - Corregido con mejoras de flashscore_inspector
Extracción directa de datos de FlashScore sin OCR
MEJORAS: Selectores actualizados, mejor detección de torneos, datos estructurados
FIX: Corrección para extraer ambas cuotas correctamente
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
        """Extraer datos de un partido individual"""
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
                'screenshot_origen': None
            }
            
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
            
            # CORRECCIÓN: Extraer cuotas con múltiples selectores y mejor lógica
            try:
                # Intentar múltiples selectores para las cuotas
                odds_selectors = [
                    '[class*="odds"]',
                    '[data-odd]',
                    '[class*="odd-"]',
                    '.event__odds',
                    '[class*="bookmaker"]'
                ]
                
                odds_elements = []
                for selector in odds_selectors:
                    elements = await element.query_selector_all(selector)
                    if elements and len(elements) >= 2:
                        odds_elements = elements
                        break
                
                if len(odds_elements) >= 2:
                    try:
                        # Extraer y validar primera cuota
                        cuota1_text = await odds_elements[0].text_content()
                        if cuota1_text and cuota1_text.strip():
                            cuota1_clean = re.sub(r'[^\d\.]', '', cuota1_text.strip())
                            if cuota1_clean and cuota1_clean.replace('.', '').isdigit():
                                match_data['cuota1'] = float(cuota1_clean)
                        
                        # Extraer y validar segunda cuota
                        cuota2_text = await odds_elements[1].text_content()
                        if cuota2_text and cuota2_text.strip():
                            cuota2_clean = re.sub(r'[^\d\.]', '', cuota2_text.strip())
                            if cuota2_clean and cuota2_clean.replace('.', '').isdigit():
                                match_data['cuota2'] = float(cuota2_clean)
                        
                        # Calcular confianza solo si tenemos ambas cuotas válidas
                        if match_data['cuota1'] and match_data['cuota2']:
                            prob1 = 1 / match_data['cuota1']
                            prob2 = 1 / match_data['cuota2']
                            total_prob = prob1 + prob2
                            if total_prob > 0:
                                match_data['confianza_promedio'] = {
                                    "jugador1": round(prob1 / total_prob, 3),
                                    "jugador2": round(prob2 / total_prob, 3)
                                }
                    except ValueError as ve:
                        # Error al convertir a float, mantener None
                        logger.debug(f"Error convirtiendo cuotas a float: {ve}")
                        pass
                    except ZeroDivisionError:
                        # División por cero en cálculo de probabilidades
                        logger.debug("Error: división por cero en cálculo de probabilidades")
                        pass
                
                # Fallback: buscar cuotas en el elemento padre o hermanos
                if not match_data['cuota1'] or not match_data['cuota2']:
                    try:
                        parent_element = await element.query_selector('xpath=..')
                        if parent_element:
                            parent_odds = await parent_element.query_selector_all('[class*="odds"], [data-odd]')
                            if len(parent_odds) >= 2:
                                for i, odd_elem in enumerate(parent_odds[:2]):
                                    odd_text = await odd_elem.text_content()
                                    if odd_text:
                                        odd_clean = re.sub(r'[^\d\.]', '', odd_text.strip())
                                        if odd_clean and odd_clean.replace('.', '').isdigit():
                                            if i == 0 and not match_data['cuota1']:
                                                match_data['cuota1'] = float(odd_clean)
                                            elif i == 1 and not match_data['cuota2']:
                                                match_data['cuota2'] = float(odd_clean)
                    except:
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
        seen = set()
        unique_matches = []

        for match in matches_data:
            j1 = match.get('jugador1', '').strip().lower()
            j2 = match.get('jugador2', '').strip().lower()
            torneo = match.get('torneo', '').strip().lower()
            estado = match.get('estado', '').strip().lower()

        # Normalizar orden alfabético para evitar duplicados invertidos
        players_key = '|'.join(sorted([j1, j2]))
        key = f"{torneo}|{players_key}|{estado}"

        if key not in seen and j1 and j2:
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
            
            logger.info(f"🏆 Torneos únicos: {len(grouped_data)}")
            logger.info(f"📊 Partidos con cuotas: {with_odds}/{len(matches_data)}")
            
            # Mostrar algunos ejemplos
            logger.info("📋 EJEMPLOS DE PARTIDOS EXTRAÍDOS (agrupados por torneo):")
            
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
                    count += 1
            
            return filename, grouped_data
            
        except Exception as e:
            logger.error(f"❌ Error guardando datos: {str(e)}")
            return None, None
    
    async def close(self):
        """Cerrar navegador"""
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
        logger.info("🔚 Navegador cerrado")

def generate_zita_report(grouped_data, filename):
    """Generar reporte final de Zita con datos agrupados por torneo"""
    
    all_matches = [match for matches in grouped_data.values() for match in matches]
    total_matches = len(all_matches)
    total_tournaments = len(grouped_data)

    print("\n" + "="*70)
    print("🎾 ZITA SCRAPER - REPORTE FINAL")
    print("="*70)
    
    print(f"""
🎯 EXTRACCIÓN COMPLETADA:
✅ {total_matches} partidos extraídos exitosamente
✅ Datos estructurados y agrupados por {total_tournaments} torneos
✅ Archivo JSON mejorado generado: {filename}

📊 ESTADÍSTICAS DETALLADAS:""")
    
    states = {}
    with_odds = 0
    
    for match in all_matches:
        # Contar por estado
        state = match.get('estado', 'Sin estado')
        states[state] = states.get(state, 0) + 1
        
        # Contar cuotas
        if match.get('cuota1') and match.get('cuota2'):
            with_odds += 1
    
    print(f"   🏆 Torneos únicos detectados: {total_tournaments}")
    print(f"   📊 Partidos con cuotas: {with_odds}/{total_matches}")
    print(f"   📈 Estados diferentes: {len(states)}")
    
    # Top 5 torneos
    print("\n🏆 TOP 5 TORNEOS (por número de partidos):")
    # Crear un diccionario de torneos con su conteo de partidos
    tournaments_counts = {name: len(matches) for name, matches in grouped_data.items()}
    sorted_tournaments = sorted(tournaments_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (tournament, count) in enumerate(sorted_tournaments[:5], 1):
        print(f"   {i}. {tournament[:50]}... ({count} partidos)")
    
    # Estados más comunes
    print("\n📊 ESTADOS MÁS COMUNES:")
    sorted_states = sorted(states.items(), key=lambda x: x[1], reverse=True)
    for i, (state, count) in enumerate(sorted_states[:5], 1):
        print(f"   {i}. {state}: {count} partidos")
    
    print(f"""
📁 ARCHIVO GENERADO:
   💾 {filename}

🚀 PRÓXIMOS PASOS:
   1️⃣ Analizar el nuevo formato JSON (agrupado por torneo)
   2️⃣ Implementar filtros y análisis por torneo
   3️⃣ Crear API para consultar datos por torneo
   4️⃣ Desarrollar análisis predictivo con datos más estructurados

💡 ZITA SCRAPER - VENTAJAS:
   ✅ Extracción rápida y eficiente
   ✅ Datos con formato JSON mejorado y más útil
   ✅ Compatible con WSL y sistemas Linux
   ✅ Sin dependencias de OCR

🎯 CONCLUSIÓN: Zita Scraper extrajo exitosamente {total_matches} partidos
   de {total_tournaments} torneos con una estructura de datos mejorada.
""")

async def main():
    """Función principal de Zita Scraper"""
    print("🎾" + "="*60)
    print("🎾 ZITA SCRAPER V2 - EXTRACCIÓN DIRECTA OPTIMIZADA")
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
        logger.info("🎾 FASE 2: Extrayendo partidos de tenis...")
        matches_data = await scraper.extract_tennis_matches()
        
        if matches_data:
            # Guardar datos
            logger.info("💾 FASE 3: Guardando y estructurando datos...")
            filename, grouped_data = await scraper.save_matches_data(matches_data)
            
            if filename and grouped_data:
                # Generar reporte final con los datos agrupados
                generate_zita_report(grouped_data, filename)
                
                logger.info(f"🎉 ÉXITO: {len(matches_data)} partidos extraídos y guardados en formato mejorado")
            else:
                logger.error("❌ Error guardando o estructurando los datos")
        else:
            logger.error("❌ No se extrajeron partidos")
        
    except Exception as e:
        logger.error(f"❌ Error general en Zita Scraper: {str(e)}")
    
    finally:
        await scraper.close()
    
    print("\n🎾 ZITA SCRAPER COMPLETADO")
    print("🚀 Datos listos para análisis y procesamiento")

if __name__ == "__main__":
    asyncio.run(main())