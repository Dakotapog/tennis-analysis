"""
📜 SCRIPT DE CONSULTA DE RESULTADOS HISTÓRICOS
Este script permite verificar los resultados finales de partidos a partir de un archivo
de pronósticos H2H específico. Es útil para generar archivos de resultados
reales para datos históricos.

Uso:
python consultar_resultados_historicos.py --file RUTA_AL_ARCHIVO.json
"""

import json
import asyncio
import logging
from datetime import datetime
from playwright.async_api import async_playwright
from pathlib import Path
import argparse

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResultVerifier:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None
        self.matches_to_verify = []
        self.verification_results = []

    async def setup_browser(self):
        """🚀 Configurar el navegador Playwright."""
        logger.info("🚀 Configurando el navegador...")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
        )
        context = await self.browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )
        self.page = await context.new_page()
        logger.info("✅ Navegador configurado.")

    def load_matches_to_verify(self, input_file: str):
        """📂 Cargar los partidos desde el archivo JSON especificado."""
        logger.info(f"📂 Cargando partidos para verificación desde: {input_file}")
        
        if not Path(input_file).exists():
            logger.error(f"❌ El archivo especificado no existe: {input_file}")
            return False
            
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.matches_to_verify = data.get('partidos', [])
            if not self.matches_to_verify:
                logger.error("❌ El archivo JSON no contiene una lista de 'partidos'.")
                return False

            logger.info(f"✅ Cargados {len(self.matches_to_verify)} partidos para verificar.")
            return True
        except Exception as e:
            logger.error(f"❌ Error cargando el archivo JSON de resultados: {e}")
            return False

    async def get_final_result(self, match_url):
        """📊 Extraer el resultado final de la página del partido con lógica de ganador mejorada."""
        try:
            logger.info(f"  -> Navegando a {match_url} para obtener resultado final.")
            await self.page.goto(match_url, wait_until='domcontentloaded', timeout=40000)

            try:
                consent_button = self.page.locator("#onetrust-accept-btn-handler")
                if await consent_button.is_visible(timeout=5000):
                    await consent_button.click()
                    logger.info("  ✅ Banner de consentimiento de cookies aceptado.")
                    await self.page.wait_for_timeout(2000)
            except Exception:
                logger.info("  ℹ️ No se encontró o no fue necesario hacer clic en el banner de consentimiento.")

            await self.page.wait_for_timeout(5000)

            status_selector = ".detailScore__status"
            status_text = (await self.page.locator(status_selector).inner_text()).strip()
            
            is_finished = any(s in status_text.lower() for s in ["finalizado", "finished", "terminado", "retired"])
            if not is_finished:
                logger.warning(f"  ⚠️ El partido en {match_url} no ha finalizado. Estado: {status_text}")
                return None, None

            home_player_element = self.page.locator(".duelParticipant__home").first
            away_player_element = self.page.locator(".duelParticipant__away").first
            
            home_player_name = (await home_player_element.locator(".participant__participantName").first.text_content() or "").strip()
            away_player_name = (await away_player_element.locator(".participant__participantName").first.text_content() or "").strip()

            home_score_selector = ".detailScore__wrapper span:nth-child(1)"
            away_score_selector = ".detailScore__wrapper span:nth-child(3)"
            home_score_str = await self.page.locator(home_score_selector).inner_text()
            away_score_str = await self.page.locator(away_score_selector).inner_text()
            
            winner = None
            final_score_suffix = ""

            try:
                home_sets = int(home_score_str)
                away_sets = int(away_score_str)
                if home_sets > away_sets:
                    winner = home_player_name
                elif away_sets > home_sets:
                    winner = away_player_name
            except (ValueError, TypeError):
                logger.warning(f"  ⚠️ No se pudo convertir el marcador '{home_score_str}-{away_score_str}'.")

            if not winner:
                if await home_player_element.locator(".duelParticipant--winner, .participant__participantName--winner").count() > 0:
                    winner = home_player_name
                elif await away_player_element.locator(".duelParticipant--winner, .participant__participantName--winner").count() > 0:
                    winner = away_player_name

            if "retired" in status_text.lower():
                final_score_suffix = " (Retiro)"

            if not winner:
                logger.error(f"  ❌ No se pudo determinar un ganador para el partido en {match_url}.")

            final_score = f"{home_score_str}-{away_score_str}{final_score_suffix}"
            logger.info(f"  ✅ Resultado final: {final_score}. Ganador: {winner}")
            return final_score, winner

        except Exception as e:
            logger.error(f"  ❌ Error extrayendo el resultado final de {match_url}: {e}")
            return None, None

    async def verify_all_matches(self):
        """🔄 Procesar todos los partidos y guardar sus resultados."""
        logger.info("🚀 Iniciando la consulta de resultados históricos...")
        
        for i, match_data in enumerate(self.matches_to_verify):
            logger.info("-" * 60)
            logger.info(f"🔎 Consultando Partido {i+1}/{len(self.matches_to_verify)}: {match_data['jugador1']} vs {match_data['jugador2']}")

            final_score, actual_winner = await self.get_final_result(match_data['match_url'])
            
            result = {
                'match_info': {
                    'jugador1': match_data['jugador1'],
                    'jugador2': match_data['jugador2'],
                    'match_url': match_data['match_url']
                },
                'actual_result': {
                    'final_score': final_score,
                    'actual_winner': actual_winner
                },
                'verification': {
                    'status': 'Verified' if actual_winner else 'Not Finished or Error'
                }
            }
            self.verification_results.append(result)
            await self.page.wait_for_timeout(3000)

    def save_verification_results(self):
        """💾 Guardar los resultados de la consulta en un archivo JSON."""
        if not self.verification_results:
            logger.warning("⚠️ No hay resultados de consulta para guardar.")
            return

        output_dir = Path('reports')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_dir / f"resultados_finales_{timestamp}.json"

        output_data = {
            'metadata': {
                'consultation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_matches_processed': len(self.verification_results),
            },
            'detailed_results': self.verification_results
        }

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info("=" * 60)
            logger.info("🏁 CONSULTA COMPLETADA 🏁")
            logger.info(f"💾 Resultados guardados en: {filename}")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"❌ Error guardando los resultados de la consulta: {e}")

    async def cleanup(self):
        """🧹 Limpieza de recursos de Playwright."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        logger.info("🧹 Recursos del navegador liberados.")

async def main(input_file):
    verifier = ResultVerifier()
    try:
        if not verifier.load_matches_to_verify(input_file):
            return
        
        await verifier.setup_browser()
        await verifier.verify_all_matches()
        verifier.save_verification_results()

    except Exception as e:
        logger.error(f"❌ Ocurrió un error inesperado en la ejecución principal: {e}")
    finally:
        await verifier.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consulta resultados de partidos a partir de un archivo JSON de pronósticos.")
    parser.add_argument('--file', type=str, required=True, help="Ruta al archivo JSON de entrada (ej: reports/h2h_results_enhanced_20250811_170726.json).")
    
    args = parser.parse_args()
    
    asyncio.run(main(args.file))
