"""
🔬 SCRIPT DE VERIFICACIÓN DE RESULTADOS FINALES
Este script carga los resultados de un análisis H2H previo, navega a las URLs de los partidos
y extrae el resultado final para compararlo con la predicción generada.
"""

import json
import asyncio
import logging
from datetime import datetime
from playwright.async_api import async_playwright
from pathlib import Path
import re

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_latest_h2h_results_file():
    """🔍 Encontrar el archivo de resultados H2H más reciente en la carpeta 'reports'."""
    logger.info("🔍 Buscando el archivo de resultados H2H más reciente en 'reports/'...")
    
    reports_dir = Path('reports')
    if not reports_dir.exists() or not reports_dir.is_dir():
        logger.error(f"❌ El directorio '{reports_dir}' no existe.")
        return None

    # Buscar archivos que sigan el patrón 'h2h_results_enhanced_*.json'
    results_files = list(reports_dir.glob('h2h_results_enhanced_*.json'))
    
    if not results_files:
        logger.error("❌ No se encontraron archivos 'h2h_results_enhanced_*.json' en la carpeta 'reports'.")
        return None
    
    # Encontrar el más reciente por fecha de modificación
    latest_file = max(results_files, key=lambda p: p.stat().st_mtime)
    
    logger.info(f"📂 Archivo de resultados seleccionado: {latest_file}")
    return str(latest_file)

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

    def load_matches_to_verify(self):
        """📂 Cargar los partidos desde el archivo JSON de resultados H2H."""
        logger.info("📂 Cargando partidos para verificación...")
        
        h2h_results_file = find_latest_h2h_results_file()
        if not h2h_results_file:
            return False
            
        try:
            with open(h2h_results_file, 'r', encoding='utf-8') as f:
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

            # --- PASO 1: Manejar el banner de consentimiento de cookies ---
            try:
                consent_button = self.page.locator("#onetrust-accept-btn-handler")
                if await consent_button.is_visible(timeout=5000):
                    await consent_button.click()
                    logger.info("  ✅ Banner de consentimiento de cookies aceptado.")
                    await self.page.wait_for_timeout(2000)
            except Exception:
                logger.info("  ℹ️ No se encontró o no fue necesario hacer clic en el banner de consentimiento.")

            await self.page.wait_for_timeout(5000)

            # --- PASO 2: Verificar si el partido ha finalizado ---
            status_selector = ".detailScore__status"
            status_text = (await self.page.locator(status_selector).inner_text()).strip()
            
            is_finished = any(s in status_text.lower() for s in ["finalizado", "finished", "terminado", "retired"])
            if not is_finished:
                logger.warning(f"  ⚠️ El partido en {match_url} no ha finalizado. Estado: {status_text}")
                return None, None

            # --- PASO 3: Extraer nombres de jugadores y marcador ---
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

            # --- PASO 4: Lógica de Detección de Ganador Jerárquica ---

            # 1. MÉTODO PRINCIPAL: Determinar ganador por el marcador de sets.
            try:
                home_sets = int(home_score_str)
                away_sets = int(away_score_str)
                if home_sets > away_sets:
                    winner = home_player_name
                    logger.info(f"  ℹ️ Ganador detectado por marcador: {winner} ({home_sets}-{away_sets})")
                elif away_sets > home_sets:
                    winner = away_player_name
                    logger.info(f"  ℹ️ Ganador detectado por marcador: {winner} ({home_sets}-{away_sets})")
            except (ValueError, TypeError):
                logger.warning(f"  ⚠️ No se pudo convertir el marcador '{home_score_str}-{away_score_str}' a números. Usando métodos de respaldo.")

            # 2. RESPALDO: Buscar por clase de ganador en el contenedor del participante.
            if not winner:
                try:
                    if await home_player_element.locator(".duelParticipant--winner, .participant__participantName--winner").count() > 0:
                        winner = home_player_name
                        logger.info(f"  ℹ️ Ganador detectado por clase CSS en jugador local: {winner}")
                    elif await away_player_element.locator(".duelParticipant--winner, .participant__participantName--winner").count() > 0:
                        winner = away_player_name
                        logger.info(f"  ℹ️ Ganador detectado por clase CSS en jugador visitante: {winner}")
                except Exception as e:
                    logger.warning(f"  ⚠️ Error buscando ganador con clase CSS: {e}")

            # 3. RESPALDO: Buscar por negrita (<strong>).
            if not winner:
                try:
                    if await home_player_element.locator(".participant__participantName strong").count() > 0:
                        winner = home_player_name
                        logger.info(f"  ℹ️ Ganador detectado por etiqueta <strong> en jugador local: {winner}")
                    elif await away_player_element.locator(".participant__participantName strong").count() > 0:
                        winner = away_player_name
                        logger.info(f"  ℹ️ Ganador detectado por etiqueta <strong> en jugador visitante: {winner}")
                except Exception as e:
                    logger.warning(f"  ⚠️ Error buscando ganador con etiqueta <strong>: {e}")

            # 4. MANEJO ESPECIAL: Partido por retiro.
            if "retired" in status_text.lower():
                final_score_suffix = " (Retiro)"
                if not winner:
                    logger.info("  ℹ️ Partido por retiro, intentando extraer ganador del texto de estado.")
                    try:
                        # Estrategia: Extraer el nombre del jugador retirado del texto de estado.
                        # El formato suele ser "STATUS / RETIRED - PLAYER NAME".
                        match = re.search(r'retired\s*-\s*(.*)', status_text, re.IGNORECASE)
                        if match:
                            retired_player_name = match.group(1).strip()
                            logger.info(f"  ℹ️ Jugador retirado detectado en estado: '{retired_player_name}'")

                            # Normalizar nombres para una comparación robusta.
                            retired_norm = retired_player_name.lower().replace('.', '').strip()
                            home_norm = home_player_name.lower().replace('.', '').strip()
                            away_norm = away_player_name.lower().replace('.', '').strip()

                            # El ganador es el oponente del jugador retirado.
                            if retired_norm in home_norm or home_norm in retired_norm:
                                winner = away_player_name
                                logger.info(f"  ✅ Ganador por retiro: {winner}")
                            elif retired_norm in away_norm or away_norm in retired_norm:
                                winner = home_player_name
                                logger.info(f"  ✅ Ganador por retiro: {winner}")
                            else:
                                logger.warning(f"  ⚠️ Nombre retirado ('{retired_norm}') no coincide con jugadores ('{home_norm}', '{away_norm}').")
                        
                        if not winner:
                            logger.warning("  ⚠️ No se pudo confirmar el ganador en un partido por retiro por los métodos primarios o de estado.")

                    except Exception as e:
                        logger.error(f"  ❌ Error procesando estado de retiro: {e}")


            if not winner:
                logger.error(f"  ❌ No se pudo determinar un ganador para el partido en {match_url}. Estado: {status_text}")

            final_score = f"{home_score_str}-{away_score_str}{final_score_suffix}"

            logger.info(f"  ✅ Resultado final: {final_score}. Ganador: {winner}")
            return final_score, winner

        except Exception as e:
            logger.error(f"  ❌ Error extrayendo el resultado final de {match_url}: {e}")
            error_path = f"error_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            await self.page.screenshot(path=error_path)
            logger.error(f"  📸 Screenshot de error guardado en: {error_path}")
            return None, None

    async def verify_all_matches(self):
        """🔄 Procesar todos los partidos y verificar las predicciones."""
        logger.info("🚀 Iniciando la verificación de predicciones...")
        
        for i, match_data in enumerate(self.matches_to_verify):
            logger.info("-" * 60)
            logger.info(f"🔎 Verificando Partido {i+1}/{len(self.matches_to_verify)}: {match_data['jugador1']} vs {match_data['jugador2']}")

            prediction_info = match_data.get('ranking_analysis', {}).get('prediction', {})
            predicted_winner = prediction_info.get('favored_player')
            prediction_confidence = prediction_info.get('confidence')

            if not predicted_winner:
                logger.warning("  ⚠️ No se encontró predicción para este partido. Saltando.")
                continue

            final_score, actual_winner = await self.get_final_result(match_data['match_url'])
            
            hit = None
            if actual_winner:
                # Lógica de comparación de nombres mejorada
                # Toma la primera palabra del nombre predicho (ej: "Rublev" de "Rublev A.")
                predicted_name_part = predicted_winner.split(' ')[0]
                # Comprueba si esa parte está en el nombre completo del ganador real
                hit = predicted_name_part.lower() in actual_winner.lower()

            result = {
                'match_info': {
                    'jugador1': match_data['jugador1'],
                    'jugador2': match_data['jugador2'],
                    'match_url': match_data['match_url']
                },
                'prediction': {
                    'predicted_winner': predicted_winner,
                    'confidence': prediction_confidence
                },
                'actual_result': {
                    'final_score': final_score,
                    'actual_winner': actual_winner
                },
                'verification': {
                    'hit': hit,
                    'status': 'Verified' if actual_winner else 'Not Finished or Error'
                }
            }
            self.verification_results.append(result)
            
            if hit is not None:
                logger.info(f"  🎯 Predicción: {predicted_winner} | Ganador Real: {actual_winner} -> {'✅ ACIERTO' if hit else '❌ FALLO'}")
            
            await self.page.wait_for_timeout(3000) # Pausa para no sobrecargar

    def save_verification_results(self):
        """💾 Guardar los resultados de la verificación en un archivo JSON."""
        if not self.verification_results:
            logger.warning("⚠️ No hay resultados de verificación para guardar.")
            return

        output_dir = Path('reports')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_dir / f"resultados_finales_{timestamp}.json"

        # Calcular estadísticas de aciertos
        total_verified = sum(1 for r in self.verification_results if r['verification']['hit'] is not None)
        total_hits = sum(1 for r in self.verification_results if r['verification']['hit'] is True)
        accuracy = (total_hits / total_verified * 100) if total_verified > 0 else 0

        output_data = {
            'metadata': {
                'verification_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_matches_processed': len(self.verification_results),
                'total_matches_verified': total_verified,
            },
            'summary': {
                'total_hits': total_hits,
                'total_misses': total_verified - total_hits,
                'accuracy_percentage': round(accuracy, 2)
            },
            'detailed_results': self.verification_results
        }

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info("=" * 60)
            logger.info("🏁 VERIFICACIÓN COMPLETADA 🏁")
            logger.info(f"💾 Resultados guardados en: {filename}")
            logger.info(f"📊 Tasa de Aciertos: {accuracy:.2f}% ({total_hits}/{total_verified})")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"❌ Error guardando los resultados de la verificación: {e}")

    async def cleanup(self):
        """🧹 Limpieza de recursos de Playwright."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        logger.info("🧹 Recursos del navegador liberados.")

async def main():
    verifier = ResultVerifier()
    try:
        if not verifier.load_matches_to_verify():
            return
        
        await verifier.setup_browser()
        await verifier.verify_all_matches()
        verifier.save_verification_results()

    except Exception as e:
        logger.error(f"❌ Ocurrió un error inesperado en la ejecución principal: {e}")
    finally:
        await verifier.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
