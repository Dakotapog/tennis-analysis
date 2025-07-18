#!/bin/bash
# install_selenium_wsl2.sh
# Este script instala Google Chrome y todas las dependencias necesarias en un entorno Debian/Ubuntu (WSL).

echo "🚀 Iniciando instalación de dependencias para Selenium en WSL..."

# Actualizar la lista de paquetes
sudo apt-get update

echo "🔧 Instalando dependencias base (wget, unzip)..."
sudo apt-get install -y wget unzip

echo "🔧 Instalando Google Chrome Stable..."
# Descargar el paquete oficial de Chrome
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
# Instalar el paquete
sudo dpkg -i google-chrome-stable_current_amd64.deb
# Corregir dependencias rotas que puedan surgir de la instalación de Chrome
sudo apt-get install -f -y

echo "🔧 Verificando la instalación de Chrome..."
google-chrome --version

echo "🧹 Limpiando archivos descargados..."
rm google-chrome-stable_current_amd64.deb

echo "✅ ¡Instalación completada! Tu entorno WSL está listo para Selenium."