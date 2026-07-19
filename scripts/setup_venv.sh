#!/bin/bash
# ==============================================================================
# SCRIPT AUTOMATIZADO DE CONFIGURAÇÃO DE AMBIENTE VIRTUAL E COMPILAÇÃO C++
# Miniola - Spec-Driven Development (SDD) & Multi-Plataforma
# ==============================================================================

set -e

echo "═════════════════════════════════════════════════════════════"
echo "MINIOLA - CONFIGURAÇÃO DE AMBIENTE VIRTUAL (VENV) E BUILD C++"
echo "═════════════════════════════════════════════════════════════"

# 1. Checar Python 3
if ! command -v python3 &> /dev/null; then
    echo "[ERRO] python3 não foi encontrado. Instale o Python 3 antes de continuar."
    exit 1
fi

PYTHON_VER=$(python3 --version)
echo "Python detectado: $PYTHON_VER"
echo "Arquitetura detectada: $(uname -m)"

# 2. Criar ambiente virtual
echo ""
echo "[1/4] Criando ambiente virtual em ./venv..."
python3 -m venv venv

# 3. Ativar e instalar dependências
echo "[2/4] Instalando dependências e ferramentas de build via pip..."
source venv/bin/activate
pip install --upgrade pip setuptools wheel --quiet
pip install -r requirements.txt

# 4. Compilar motor de visão C++ via pybind11
echo ""
echo "[3/4] Checando dependências de sistema para build C++..."
if ! pkg-config --exists opencv4 2>/dev/null && [ ! -f "/usr/include/opencv4/opencv2/opencv.hpp" ] && [ ! -f "/usr/include/opencv2/opencv.hpp" ]; then
    echo "══════════════════════════════════════════════════════════════════════════"
    echo " [FALHA NO BUILD C++] Headers C++ do OpenCV não foram encontrados!"
    echo " O pacote pip 'opencv-python-headless' instala apenas os bindings Python,"
    echo " mas o compilador g++ precisa dos headers C++ (opencv2/opencv.hpp)."
    echo ""
    echo " PARA RESOLVER NO SEU LINUX (Mac Mini / Raspberry Pi):"
    echo " Abra o terminal e execute o comando abaixo para instalar libopencv-dev:"
    echo " 👉 sudo apt update && sudo apt install -y libopencv-dev pkg-config"
    echo "══════════════════════════════════════════════════════════════════════════"
    exit 1
fi

echo "[3/4] Compilando extensão nativa C++ (miniola_cv) para esta arquitetura..."
python3 setup.py build_ext --inplace

# 5. Validar conformidade SDD e suíte de testes
echo ""
echo "[4/4] Executando validação de especificações e testes unitários..."
python3 scripts/check_specs.py
python3 -m unittest discover -s tests

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "[SUCESSO] Ambiente virtual (venv) configurado e motor C++ compilado!"
echo "════════════════════════════════════════════════════════════════════"
echo " ➡️  Para começar a usar no seu dia a dia, execute no terminal:"
echo "      source venv/bin/activate"
echo ""
echo " ➡️  Para iniciar o Anotador Visual na porta 5001:"
echo "      python3 tools/anotador_amostras.py --port 5001"
echo ""
echo " ➡️  Para iniciar o Scanner de Captura (modo mock para teste sem câmera):"
echo "      python3 miniola.py --camera mock"
echo "══════════════════════════════════════════════════════════════════════════"
