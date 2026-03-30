#!/bin/bash

echo "========================================"
echo " INICIANDO SISTEMA MINIOLA"
echo "========================================"

# Navega até a pasta de execução
cd /home/felipe/miniola/miniola_py

# Atualiza os arquivos diretamente da branch de trabalho
echo "[SISTEMA] Sincronizando código fonte..."
git pull origin desenvolvimento

# Ativa o isolamento de bibliotecas
source venv/bin/activate

# Executa o software de digitalização
echo "[SISTEMA] Ligando Camera Module 3 e Motor de Captura..."
python3 miniola.py