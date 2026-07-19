#!/bin/bash
# ==============================================================================
# SCRIPT DE INICIALIZAÇÃO RÁPIDA - MINIOLA
# ==============================================================================

echo "================================================================="
echo " INICIANDO SISTEMA MINIOLA"
echo "================================================================="

# 1. Determina a raiz do projeto de forma dinâmica
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$ROOT_DIR"
echo "[SISTEMA] Raiz do projeto detectada: $ROOT_DIR"

# 2. Sincroniza o código fonte da branch atual
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "main")
echo "[SISTEMA] Sincronizando código fonte (branch: $CURRENT_BRANCH)..."
git pull origin "$CURRENT_BRANCH" || echo "[WARN] Falha ao sincronizar com o remoto (verifique conexão ou commits). Continuando..."

# 3. Ativa o ambiente virtual Python
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "[ERRO] Ambiente virtual não encontrado em $ROOT_DIR/venv. Execute primeiro: ./scripts/setup_venv.sh"
    exit 1
fi

# 4. Executa o software principal
echo "[SISTEMA] Iniciando motor de captura e painel Miniola..."
python3 miniola.py "$@"
