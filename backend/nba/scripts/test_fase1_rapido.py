#!/usr/bin/env python3
"""
🏀 PRUEBA RÁPIDA - FASE 1
Verificar extracción de partidos NBA

Autor: David Alberto Coronado Tabares
"""

import asyncio
import logging
from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_nba_extraction():
    """Prueba rápida de extracción NBA"""
    
    logger.info("🏀 INICIANDO PRUEBA RÁPIDA...")
    
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(
        headless=True,
        args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
    )
    
    page = await browser.new_page()
    
    try:
        # 1. Navegar
        logger.info("🌐 Navegando a FlashScore NBA...")
        await page.goto("https://www.flashscore.co/baloncesto/usa/nba/", timeout=30000)
        await asyncio.sleep(3)
        
        # 2. Cookies
        try:
            btn = await page.wait_for_selector("#onetrust-accept-btn-handler", timeout=5000)
            if btn:
                await btn.click()
                await asyncio.sleep(1)
        except:
            pass
        
        # 3. Buscar tab "Partidos de hoy"
        logger.info("🔍 Buscando tab 'Partidos de hoy'...")
        
        today_selectors = [
            "div.tabs__ear",
            ".tabs__ear",
            "[class*='tabs__ear']"
        ]
        
        today_tab = None
        for selector in today_selectors:
            try:
                today_tab = await page.query_selector(selector)
                if today_tab:
                    text = await today_tab.text_content()
                    logger.info(f"   ✅ Tab encontrado: '{text.strip()}'")
                    if 'hoy' in text.lower() or 'today' in text.lower():
                        await today_tab.click()
                        logger.info(f"   ✅ Click en '{text.strip()}'")
                        await asyncio.sleep(3)
                        break
            except:
                continue
        
        # 4. Contar partidos
        logger.info("\n📊 CONTANDO PARTIDOS...")
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(2)
        
        matches = await page.query_selector_all('.event__match')
        logger.info(f"   Total elementos: {len(matches)}")
        
        # 5. Analizar primeros 10 partidos
        logger.info("\n🔍 ANALIZANDO PRIMEROS 10 PARTIDOS:")
        
        nba_count = 0
        other_count = 0
        
        nba_teams = [
            'celtics', 'lakers', 'warriors', 'heat', 'knicks', 'bulls', 'nets',
            '76ers', 'sixers', 'mavericks', 'mavs', 'clippers', 'bucks', 'suns',
            'nuggets', 'raptors', 'hawks', 'cavaliers', 'cavs', 'pelicans', 'jazz',
            'magic', 'pacers', 'hornets', 'pistons', 'rockets', 'kings', 'spurs',
            'thunder', 'timberwolves', 'wolves', 'trail blazers', 'blazers',
            'grizzlies', 'wizards'
        ]
        
        for i, match in enumerate(matches[:10], 1):
            try:
                participants = await match.query_selector_all('[class*="participant"]')
                if len(participants) >= 2:
                    home = (await participants[0].text_content()).strip().lower()
                    away = (await participants[1].text_content()).strip().lower()
                    
                    # Verificar si es NBA
                    is_nba = any(team in home or team in away for team in nba_teams)
                    
                    if is_nba:
                        nba_count += 1
                        logger.info(f"   ✅ {i}. NBA: {home.title()} vs {away.title()}")
                    else:
                        other_count += 1
                        logger.info(f"   ❌ {i}. OTRA LIGA: {home.title()} vs {away.title()}")
            except:
                continue
        
        # 6. Buscar sección NBA específica
        logger.info("\n🔍 BUSCANDO HEADER 'NBA'...")
        
        nba_header_selectors = [
            'strong:has-text("NBA")',
            'strong.wcl-simpleText_2t3pL:has-text("NBA")',
            '[data-testid*="text"]:has-text("NBA")',
            '.headerLeague__title-text:has-text("NBA")'
        ]
        
        for selector in nba_header_selectors:
            try:
                header = await page.query_selector(selector)
                if header:
                    text = await header.text_content()
                    classes = await header.get_attribute('class')
                    logger.info(f"   ✅ Header encontrado: {selector}")
                    logger.info(f"      Texto: {text}")
                    logger.info(f"      Classes: {classes}")
                    break
            except:
                continue
        
        # 7. Resumen
        logger.info("\n" + "="*60)
        logger.info("📊 RESUMEN:")
        logger.info(f"   Total partidos visibles: {len(matches)}")
        logger.info(f"   Partidos NBA (muestra): {nba_count}/10")
        logger.info(f"   Otras ligas (muestra): {other_count}/10")
        logger.info("="*60)
        
        # Screenshot
        await page.screenshot(path="test_nba_page.png")
        logger.info("\n📸 Screenshot guardado: test_nba_page.png")
        
        # Recomendación
        if other_count > 0:
            logger.info("\n⚠️  PROBLEMA DETECTADO:")
            logger.info("   Se están extrayendo partidos de otras ligas")
            logger.info("   Necesitas filtrar SOLO la sección NBA")
        else:
            logger.info("\n✅ TODO CORRECTO:")
            logger.info("   Solo partidos NBA detectados")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    
    finally:
        await browser.close()
        await playwright.stop()
    
    logger.info("\n🏁 PRUEBA COMPLETADA")

if __name__ == "__main__":
    asyncio.run(test_nba_extraction())
