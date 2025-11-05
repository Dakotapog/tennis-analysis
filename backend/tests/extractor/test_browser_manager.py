"""
🧪 Tests para BrowserManager
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from scraping.browser_manager import BrowserManager


class TestBrowserManager:
    """Suite de tests para BrowserManager."""
    
    @pytest.fixture
    def browser_manager(self):
        """Instancia limpia de BrowserManager."""
        return BrowserManager()
    
    # ═══════════════════════════════════════════════════════════════
    # HAPPY PATH
    # ═══════════════════════════════════════════════════════════════
    
    def test_init_creates_instance(self, browser_manager):
        """Test: Inicialización crea instancia correctamente."""
        assert browser_manager.playwright is None
        assert browser_manager.browser is None
        assert browser_manager.context is None
        assert browser_manager.page is None
        assert isinstance(browser_manager.is_wsl, bool)
    
    def test_check_wsl_returns_boolean(self, browser_manager):
        """Test: _check_wsl retorna booleano."""
        result = browser_manager._check_wsl()
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_setup_success(self, browser_manager):
        """Test: Setup configura navegador exitosamente."""
        with patch('scraping.browser_manager.async_playwright') as mock_playwright:
            # Mock de Playwright
            mock_pw = AsyncMock()
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_page = AsyncMock()
            
            mock_pw.start = AsyncMock(return_value=mock_pw)
            mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_browser.new_context = AsyncMock(return_value=mock_context)
            mock_context.new_page = AsyncMock(return_value=mock_page)
            mock_page.on = MagicMock()
            mock_page.set_extra_http_headers = AsyncMock()
            
            mock_playwright.return_value = mock_pw
            
            # Ejecutar setup
            result = await browser_manager.setup()
            
            # Validaciones
            assert result == mock_page
            assert browser_manager.playwright == mock_pw
            assert browser_manager.browser == mock_browser
            assert browser_manager.context == mock_context
            assert browser_manager.page == mock_page
    
    @pytest.mark.asyncio
    async def test_cleanup_closes_all_resources(self, browser_manager):
        """Test: Cleanup cierra todos los recursos."""
        # Mock de recursos
        browser_manager.page = AsyncMock()
        browser_manager.context = AsyncMock()
        browser_manager.browser = AsyncMock()
        browser_manager.playwright = AsyncMock()
        
        # Ejecutar cleanup
        await browser_manager.cleanup()
        
        # Validar que se llamaron los métodos close
        browser_manager.page.close.assert_called_once()
        browser_manager.context.close.assert_called_once()
        browser_manager.browser.close.assert_called_once()
        browser_manager.playwright.stop.assert_called_once()
    
    # ═══════════════════════════════════════════════════════════════
    # EDGE CASES
    # ═══════════════════════════════════════════════════════════════
    
    @pytest.mark.asyncio
    async def test_cleanup_with_none_resources(self, browser_manager):
        """Test: Cleanup no falla con recursos None."""
        browser_manager.page = None
        browser_manager.context = None
        browser_manager.browser = None
        browser_manager.playwright = None
        
        # No debería lanzar excepción
        await browser_manager.cleanup()
        
        assert browser_manager.page is None
    
    @pytest.mark.asyncio
    async def test_cleanup_handles_errors_gracefully(self, browser_manager):
        """Test: Cleanup maneja errores sin propagar excepciones."""
        # Mock que lanza error
        browser_manager.page = AsyncMock()
        browser_manager.page.close.side_effect = Exception("Test error")
        
        # No debería propagar la excepción
        await browser_manager.cleanup()
    
    def test_kill_zombie_chrome_processes(self, browser_manager):
        """Test: Limpia procesos Chrome zombies."""
        with patch('scraping.browser_manager.psutil.process_iter') as mock_process_iter:
            mock_proc = MagicMock()
            mock_proc.info = {'pid': 123, 'name': 'chrome'}
            mock_proc.kill = MagicMock()
            
            mock_process_iter.return_value = [mock_proc]
            
            killed = browser_manager._kill_zombie_chrome_processes()
            
            assert killed >= 0
            assert isinstance(killed, int)
    
    # ═══════════════════════════════════════════════════════════════
    # ERROR HANDLING
    # ═══════════════════════════════════════════════════════════════
    
    @pytest.mark.asyncio
    async def test_setup_raises_on_playwright_failure(self, browser_manager):
        """Test: Setup propaga excepción si Playwright falla."""
        with patch('scraping.browser_manager.async_playwright') as mock_playwright:
            mock_playwright.side_effect = Exception("Playwright failed")
            
            with pytest.raises(Exception, match="Playwright failed"):
                await browser_manager.setup()
    
    @pytest.mark.asyncio
    async def test_setup_cleans_up_on_failure(self, browser_manager):
        """Test: Setup llama cleanup si falla."""
        with patch('scraping.browser_manager.async_playwright') as mock_playwright:
            mock_pw = AsyncMock()
            mock_pw.start = AsyncMock(side_effect=Exception("Start failed"))
            mock_playwright.return_value = mock_pw
            
            with patch.object(browser_manager, 'cleanup', new_callable=AsyncMock) as mock_cleanup:
                with pytest.raises(Exception):
                    await browser_manager.setup()
                
                # Validar que se llamó cleanup
                mock_cleanup.assert_called_once()
    
    def test_check_system_resources_logs_info(self, browser_manager):
        """Test: check_system_resources ejecuta sin errores."""
        with patch('scraping.browser_manager.psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = MagicMock(available=2 * 1024**3)  # 2GB
        
            # Verificar que no lanza excepción
            browser_manager.check_system_resources()
        
            # Verificar que se llamó al mock
            mock_memory.assert_called_once()
