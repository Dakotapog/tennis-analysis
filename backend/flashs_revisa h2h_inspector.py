#!/usr/bin/env python3
"""
FlashScore Inspector Avanzado - Arma Poderosa para Análisis H2H
Analiza la estructura específica de páginas de partidos para extraer datos H2H
URL de prueba: https://www.flashscore.co/partido/tenis/IXQP3WfJ/#/resumen-del-partido/resumen-del-partido
"""

import asyncio
import logging
import json
from datetime import datetime
from playwright.async_api import async_playwright
import re

# Configurar logging detallado
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FlashScoreAdvancedInspector:
    def __init__(self):
        self.browser = None
        self.page = None
        self.playwright = None
        self.inspection_data = {}
        
    async def setup_browser(self):
        """Configurar navegador optimizado para inspección"""
        try:
            logger.info("🚀 Iniciando Inspector Avanzado...")
            
            self.playwright = await async_playwright().start()
            
            # Opciones optimizadas para WSL
            browser_args = [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-accelerated-2d-canvas',
                '--no-first-run',
                '--no-zygote',
                '--single-process',
                '--disable-gpu',
                '--disable-software-rasterizer',
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding',
                '--disable-features=TranslateUI,VizDisplayCompositor',
                '--disable-extensions',
                '--disable-default-apps',
                '--disable-sync',
                '--metrics-recording-only',
                '--no-default-browser-check',
                '--allow-running-insecure-content',
                '--disable-web-security',
                '--disable-blink-features=AutomationControlled',
                '--ignore-certificate-errors',
                '--ignore-ssl-errors',
                '--ignore-certificate-errors-spki-list',
                '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ]
            
            self.browser = await self.playwright.chromium.launch(
                headless=True,   # Headless para WSL
                args=browser_args,
                slow_mo=100,     # Optimizado para WSL
                timeout=60000
            )
            
            context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                locale='es-ES'
            )
            
            self.page = await context.new_page()
            self.page.set_default_timeout(30000)
            
            logger.info("✅ Inspector configurado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error configurando inspector: {str(e)}")
            return False
    
    async def navigate_and_analyze(self, url):
        """Navegar a URL específica y analizar estructura con manejo robusto"""
        try:
            logger.info(f"🌐 Navegando a: {url}")
            
            # Navegar con múltiples intentos
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    await asyncio.sleep(5)  # Esperar carga completa
                    
                    # Verificar que la página se cargó
                    title = await self.page.title()
                    if title and len(title) > 5:
                        logger.info(f"✅ Página cargada: {title}")
                        break
                    else:
                        raise Exception("Página no cargada correctamente")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Intento {attempt + 1}/{max_retries} falló: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(3)
                        continue
                    else:
                        raise
            
            logger.info("📊 Página cargada, iniciando análisis...")
            
            # 1. Analizar estructura general de la página
            await self.analyze_page_structure()
            
            # 2. Buscar y analizar tabs/pestañas
            await self.analyze_tabs()
            
            # 3. Intentar acceder a sección H2H
            await self.find_h2h_section()
            
            # 4. Analizar estructura de datos H2H
            await self.analyze_h2h_structure()
            
            # 5. Tomar screenshots para documentación
            await self.take_inspection_screenshots()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error durante análisis: {str(e)}")
            return False
    
    async def analyze_page_structure(self):
        """Analizar estructura general de la página"""
        try:
            logger.info("🔍 Analizando estructura general...")
            
            # Obtener título de la página
            title = await self.page.title()
            logger.info(f"   📄 Título: {title}")
            
            # Buscar elementos principales
            main_selectors = [
                'main',
                '[role="main"]',
                '.container',
                '#main',
                '.content',
                '.match-container',
                '.game-container'
            ]
            
            structure_info = {}
            
            for selector in main_selectors:
                try:
                    elements = await self.page.query_selector_all(selector)
                    if elements:
                        structure_info[selector] = len(elements)
                        logger.info(f"   🎯 {selector}: {len(elements)} elementos")
                except:
                    continue
            
            # Buscar tabs/navegación
            tab_selectors = [
                '.tabs',
                '.tab',
                '.navigation',
                '.nav-tab',
                '[role="tab"]',
                '.menu-item',
                '.tab-item',
                'a[href*="h2h"]',
                'a[href*="enfrentamiento"]',
                'button[data-testid*="tab"]'
            ]
            
            tabs_info = {}
            for selector in tab_selectors:
                try:
                    elements = await self.page.query_selector_all(selector)
                    if elements:
                        tabs_info[selector] = len(elements)
                        # Obtener texto de los primeros 3 elementos
                        for i, elem in enumerate(elements[:3]):
                            try:
                                text = await elem.text_content()
                                if text and text.strip():
                                    tabs_info[f"{selector}_text_{i+1}"] = text.strip()
                            except:
                                continue
                except:
                    continue
            
            self.inspection_data['page_structure'] = structure_info
            self.inspection_data['tabs_structure'] = tabs_info
            
            logger.info(f"   ✅ Estructura analizada: {len(structure_info)} contenedores, {len(tabs_info)} elementos de navegación")
            
        except Exception as e:
            logger.error(f"❌ Error analizando estructura: {e}")
    
    async def analyze_tabs(self):
        """Analizar y encontrar pestañas específicamente"""
        try:
            logger.info("🗂️ Analizando pestañas/tabs...")
            
            # Selectores específicos para tabs
            specific_tab_selectors = [
                'a[href*="#/h2h"]',
                'a[href*="#/enfrentamiento"]',
                'button:has-text("H2H")',
                'button:has-text("Enfrentamiento")',
                'a:has-text("H2H")',
                'a:has-text("Enfrentamiento")',
                '.tab:has-text("H2H")',
                '[data-testid*="h2h"]'
            ]
            
            found_tabs = {}
            
            # Buscar todos los enlaces y botones
            all_links = await self.page.query_selector_all('a, button')
            logger.info(f"   🔗 Total enlaces/botones encontrados: {len(all_links)}")
            
            h2h_candidates = []
            
            for i, link in enumerate(all_links):
                try:
                    text = await link.text_content()
                    href = await link.get_attribute('href')
                    onclick = await link.get_attribute('onclick')
                    data_testid = await link.get_attribute('data-testid')
                    class_name = await link.get_attribute('class')
                    
                    # Buscar indicadores de H2H
                    indicators = [
                        'h2h', 'enfrentamiento', 'historial', 'head', 'cara',
                        'previous', 'anterior', 'matches', 'partidos'
                    ]
                    
                    text_lower = text.lower() if text else ""
                    href_lower = href.lower() if href else ""
                    
                    for indicator in indicators:
                        if (indicator in text_lower or 
                            indicator in href_lower or 
                            (data_testid and indicator in data_testid.lower()) or
                            (class_name and indicator in class_name.lower())):
                            
                            candidate = {
                                'index': i,
                                'text': text.strip() if text else "",
                                'href': href,
                                'onclick': onclick,
                                'data_testid': data_testid,
                                'class': class_name,
                                'indicator_found': indicator
                            }
                            h2h_candidates.append(candidate)
                            break
                            
                except:
                    continue
            
            self.inspection_data['h2h_candidates'] = h2h_candidates
            
            logger.info(f"   🎯 Candidatos H2H encontrados: {len(h2h_candidates)}")
            for i, candidate in enumerate(h2h_candidates[:5]):  # Mostrar primeros 5
                logger.info(f"      {i+1}. '{candidate['text']}' -> {candidate['href']} ({candidate['indicator_found']})")
            
        except Exception as e:
            logger.error(f"❌ Error analizando tabs: {e}")
    
    async def find_h2h_section(self):
        """Intentar encontrar y acceder a la sección H2H con mejor manejo de errores"""
        try:
            logger.info("🎯 Buscando sección H2H...")
            
            # URLs posibles para H2H basadas en la estructura de FlashScore
            match_id = self.extract_match_id_from_url(self.page.url)
            if match_id:
                h2h_urls = [
                    f"https://www.flashscore.co/partido/tenis/{match_id}/#/h2h/general",
                    f"https://www.flashscore.co/partido/tenis/{match_id}/#/enfrentamientos-directos", 
                    f"https://www.flashscore.co/partido/tenis/{match_id}/#/h2h",
                    f"https://www.flashscore.co/partido/tenis/{match_id}/#/historial",
                    f"https://www.flashscore.co/partido/tenis/{match_id}/#/head-to-head"
                ]
                
                logger.info(f"   🔗 Match ID detectado: {match_id}")
                
                for i, url in enumerate(h2h_urls):
                    try:
                        logger.info(f"   🌐 Probando URL {i+1}/{len(h2h_urls)}: {url}")
                        
                        # Navegar con timeout más corto
                        await self.page.goto(url, wait_until="domcontentloaded", timeout=20000)
                        await asyncio.sleep(4)
                        
                        # Verificar si hay contenido H2H
                        h2h_content = await self.check_h2h_content()
                        if h2h_content:
                            logger.info(f"   ✅ Sección H2H encontrada en: {url}")
                            self.inspection_data['h2h_url'] = url
                            self.inspection_data['h2h_access_method'] = 'direct_url'
                            return True
                        else:
                            logger.info(f"   ❌ Sin contenido H2H en: {url}")
                            
                    except Exception as e:
                        logger.debug(f"   ❌ Error probando {url}: {e}")
                        continue
            
            # Si no funcionan las URLs directas, buscar clics en tabs
            logger.info("   🖱️ Intentando acceso por tabs...")
            
            # Volver a página original primero
            original_url = self.page.url.split('#')[0]
            await self.page.goto(original_url, wait_until="domcontentloaded", timeout=20000)
            await asyncio.sleep(3)
            
            if 'h2h_candidates' in self.inspection_data:
                for i, candidate in enumerate(self.inspection_data['h2h_candidates'][:3]):  # Solo primeros 3
                    try:
                        logger.info(f"   🖱️ Intento click {i+1}: '{candidate['text']}'")
                        
                        # Intentar diferentes métodos de click
                        click_methods = []
                        
                        if candidate['href']:
                            click_methods.append(('href', f'a[href="{candidate["href"]}"]'))
                        
                        if candidate['text']:
                            click_methods.append(('text', f'text="{candidate["text"]}"'))
                            
                        if candidate['data_testid']:
                            click_methods.append(('testid', f'[data-testid="{candidate["data_testid"]}"]'))
                        
                        for method_name, selector in click_methods:
                            try:
                                logger.info(f"      Probando {method_name}: {selector}")
                                await self.page.click(selector, timeout=10000)
                                await asyncio.sleep(3)
                                
                                # Verificar contenido
                                h2h_content = await self.check_h2h_content()
                                if h2h_content:
                                    logger.info(f"   ✅ H2H accesible via {method_name}: '{candidate['text']}'")
                                    self.inspection_data['h2h_access_method'] = f'click_{method_name}'
                                    self.inspection_data['h2h_element'] = candidate
                                    return True
                                    
                            except Exception as e:
                                logger.debug(f"      Error con {method_name}: {e}")
                                continue
                                
                    except Exception as e:
                        logger.debug(f"   ❌ Error con candidato {i+1}: {e}")
                        continue
            
            logger.warning("   ⚠️ No se pudo acceder a sección H2H")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error buscando H2H: {e}")
            return False
    
    async def check_h2h_content(self):
        """Verificar si hay contenido H2H en la página actual"""
        try:
            # Selectores para contenido H2H
            h2h_content_selectors = [
                '.h2h',
                '.head2head',
                '.match-history',
                '.historial',
                '.enfrentamiento',
                '[data-testid*="h2h"]',
                '.previous-matches',
                '.player-matches'
            ]
            
            for selector in h2h_content_selectors:
                elements = await self.page.query_selector_all(selector)
                if elements:
                    logger.info(f"      ✅ Contenido H2H detectado: {selector} ({len(elements)} elementos)")
                    return True
            
            # Buscar por texto indicativo
            h2h_texts = [
                'últimos partidos',
                'historial',
                'enfrentamientos',
                'previous matches',
                'head to head',
                'vs',
                'ganó',
                'perdió'
            ]
            
            page_content = await self.page.content()
            page_text = await self.page.text_content('body')
            
            for text in h2h_texts:
                if text.lower() in page_text.lower():
                    logger.info(f"      ✅ Texto H2H detectado: '{text}'")
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error verificando contenido H2H: {e}")
            return False
    
    async def analyze_h2h_structure(self):
        """Analizar estructura específica de datos H2H"""
        try:
            logger.info("📊 Analizando estructura de datos H2H...")
            
            # Selectores para diferentes tipos de datos H2H
            h2h_data_selectors = {
                'match_rows': [
                    '.match-row', '.partido', '.game-row', 'tr[class*="match"]',
                    '.h2h-match', '.historial-partido', '[data-testid*="match"]'
                ],
                'player_sections': [
                    '.player-section', '.jugador', '.player-stats',
                    '.player-matches', '.player-history'
                ],
                'load_more_buttons': [
                    'button:has-text("más")', 'button:has-text("more")',
                    '.load-more', '.mostrar-mas', '[data-testid*="load"]'
                ],
                'results': [
                    '.result', '.resultado', '.score', '.marcador',
                    '.win', '.loss', '.ganado', '.perdido'
                ],
                'dates': [
                    '.date', '.fecha', '.time', '.when', '.cuándo'
                ],
                'opponents': [
                    '.opponent', '.rival', '.vs', '.contra'
                ]
            }
            
            structure_analysis = {}
            
            for category, selectors in h2h_data_selectors.items():
                category_data = {}
                
                for selector in selectors:
                    try:
                        elements = await self.page.query_selector_all(selector)
                        if elements:
                            category_data[selector] = {
                                'count': len(elements),
                                'sample_data': []
                            }
                            
                            # Obtener datos de muestra
                            for i, elem in enumerate(elements[:3]):
                                try:
                                    text = await elem.text_content()
                                    html = await elem.inner_html()
                                    
                                    sample = {
                                        'text': text.strip() if text else "",
                                        'html_length': len(html),
                                        'has_links': 'href=' in html,
                                        'has_images': '<img' in html
                                    }
                                    category_data[selector]['sample_data'].append(sample)
                                    
                                except:
                                    continue
                                    
                            logger.info(f"   🎯 {category} - {selector}: {len(elements)} elementos")
                            
                    except:
                        continue
                
                if category_data:
                    structure_analysis[category] = category_data
            
            self.inspection_data['h2h_structure'] = structure_analysis
            
            # Análisis específico de patrones de datos
            await self.analyze_data_patterns()
            
        except Exception as e:
            logger.error(f"❌ Error analizando estructura H2H: {e}")
    
    async def analyze_data_patterns(self):
        """Analizar patrones específicos de datos de partidos"""
        try:
            logger.info("🔍 Analizando patrones de datos...")
            
            # Buscar patrones de resultados (6-4, 6-2, etc.)
            page_text = await self.page.text_content('body')
            
            # Patrones regex para datos de tenis
            patterns = {
                'tennis_scores': r'\d{1,2}-\d{1,2}',
                'dates': r'\d{1,2}[\./]\d{1,2}[\./]\d{2,4}',
                'player_names': r'[A-Z][a-z]+ [A-Z][a-z]+',
                'tournaments': r'[A-Z][a-z]+ \d{4}',
                'surfaces': r'(Arcilla|Hierba|Dura|Clay|Grass|Hard)'
            }
            
            pattern_analysis = {}
            
            for pattern_name, pattern in patterns.items():
                matches = re.findall(pattern, page_text)
                if matches:
                    pattern_analysis[pattern_name] = {
                        'count': len(matches),
                        'samples': matches[:5],  # Primeros 5 ejemplos
                        'unique_count': len(set(matches))
                    }
                    logger.info(f"   📈 {pattern_name}: {len(matches)} coincidencias")
            
            self.inspection_data['data_patterns'] = pattern_analysis
            
        except Exception as e:
            logger.error(f"❌ Error analizando patrones: {e}")
    
    def extract_match_id_from_url(self, url):
        """Extraer match ID de URL"""
        try:
            # Patrón para FlashScore: /partido/tenis/IXQP3WfJ/
            match = re.search(r'/partido/tenis/([^/]+)', url)
            if match:
                return match.group(1)
            
            # Patrón alternativo
            match = re.search(r'/match/([^/]+)', url)
            if match:
                return match.group(1)
                
            return None
        except:
            return None
    
    async def take_inspection_screenshots(self):
        """Tomar screenshots para documentación"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Screenshot de página completa
            await self.page.screenshot(
                path=f"inspection_full_{timestamp}.png",
                full_page=True
            )
            
            # Screenshot de viewport
            await self.page.screenshot(
                path=f"inspection_viewport_{timestamp}.png"
            )
            
            logger.info(f"📸 Screenshots guardados: inspection_*_{timestamp}.png")
            
        except Exception as e:
            logger.error(f"❌ Error tomando screenshots: {e}")
    
    async def generate_report(self):
        """Generar reporte completo de inspección"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flashscore_inspection_report_{timestamp}.json"
            
            report = {
                'inspection_info': {
                    'timestamp': datetime.now().isoformat(),
                    'url_analyzed': self.page.url,
                    'inspector_version': '2.0_advanced'
                },
                'findings': self.inspection_data
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📋 Reporte guardado: {filename}")
            
            # Mostrar resumen en consola
            self.print_inspection_summary()
            
        except Exception as e:
            logger.error(f"❌ Error generando reporte: {e}")
    
    def print_inspection_summary(self):
        """Imprimir resumen de inspección"""
        try:
            print("\n" + "="*80)
            print("🔍 REPORTE DE INSPECCIÓN AVANZADA")
            print("="*80)
            
            if 'h2h_candidates' in self.inspection_data:
                candidates = self.inspection_data['h2h_candidates']
                print(f"🎯 Candidatos H2H encontrados: {len(candidates)}")
                for i, c in enumerate(candidates[:3]):
                    print(f"   {i+1}. '{c['text']}' -> {c['href']}")
            
            if 'h2h_url' in self.inspection_data:
                print(f"✅ URL H2H detectada: {self.inspection_data['h2h_url']}")
            
            if 'h2h_structure' in self.inspection_data:
                structure = self.inspection_data['h2h_structure']
                print(f"📊 Categorías de datos H2H: {len(structure)}")
                for category, data in structure.items():
                    total_elements = sum(sel_data['count'] for sel_data in data.values())
                    print(f"   {category}: {total_elements} elementos")
            
            if 'data_patterns' in self.inspection_data:
                patterns = self.inspection_data['data_patterns']
                print(f"🔍 Patrones detectados: {len(patterns)}")
                for pattern, data in patterns.items():
                    print(f"   {pattern}: {data['count']} coincidencias")
            
            print("="*80)
            
        except Exception as e:
            logger.error(f"Error imprimiendo resumen: {e}")
    
    async def cleanup(self):
        """Limpiar recursos"""
        try:
            if self.page:
                await self.page.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            logger.info("🧹 Inspector limpiado")
        except Exception as e:
            logger.warning(f"⚠️ Error limpiando: {e}")

async def main():
    """Función principal de inspección"""
    inspector = FlashScoreAdvancedInspector()
    
    # URL del partido para analizar
    TEST_URL = "https://www.flashscore.co/partido/tenis/IXQP3WfJ/#/resumen-del-partido/resumen-del-partido"
    
    try:
        logger.info("🔍 Iniciando Inspección Avanzada de FlashScore")
        
        # Configurar navegador
        if not await inspector.setup_browser():
            logger.error("❌ No se pudo configurar inspector")
            return
        
        # Navegar y analizar
        if not await inspector.navigate_and_analyze(TEST_URL):
            logger.error("❌ Error durante análisis")
            return
        
        # Generar reporte
        await inspector.generate_report()
        
        logger.info("✅ Inspección completada exitosamente")
        
        # Mantener navegador abierto para debug (comentado para WSL)
        # logger.info("🔍 Navegador abierto para inspección manual...")
        # logger.info("🔍 Presiona Ctrl+C para cerrar")
        
        # try:
        #     while True:
        #         await asyncio.sleep(1)
        # except KeyboardInterrupt:
        #     logger.info("🛑 Cerrando inspector...")
        
    except Exception as e:
        logger.error(f"❌ Error durante inspección: {str(e)}")
        
    finally:
        await inspector.cleanup()

if __name__ == "__main__":
    print("🔍" + "="*80)
    print("🔍 FLASHSCORE INSPECTOR AVANZADO - ARMA PODEROSA")
    print("🔍 Análisis detallado de estructura H2H")
    print("🔍" + "="*80)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Inspección interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")