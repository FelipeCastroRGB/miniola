# Miniola - Scanner de Preservação Audiovisual (35mm)

O **Miniola** é uma estação de trabalho de baixo custo e alta precisão para a digitalização e inspeção de películas cinematográficas de 35mm. Desenvolvido para operar em mesas de revisão manuais, o sistema utiliza uma **Raspberry Pi 5/4** e a **Camera Module 3** para capturar quadros sincronizados através da detecção de perfurações.

## Arquitetura e Decisões Técnicas

O projeto foi reconstruído para ser resiliente a falhas de hardware e atualizações de sistema operacional, alinhado aos padrões de preservação (FIAF/AMIA):

* **Proteção de Dados (RAM Drive):** O sistema utiliza uma repartição em memória volátil (`tmpfs`) para o diretório de captura. Isso evita o desgaste do cartão SD e garante a velocidade necessária para gravar frames em 60fps sem latência de escrita.
* **Decoupled Preview (Headless):** Para evitar erros de dependência de monitores físicos (`kms`/`pykms`), o software utiliza um *mock* de sistema no topo do código. O visionamento é feito via rede (Flask), tornando o scanner independente de periféricos HDMI.
* **Foco Motorizado por Gradação:** Controle fino sobre a lente da Camera Module 3, permitindo ajustes milimétricos na emulsão da película via comandos de terminal.

---

## Guia de Instalação (SOP)

Este protocolo deve ser seguido em caso de formatação ou nova unidade no laboratório.

### 1. Dependências do Sistema Operacional
Execute no terminal para instalar os headers nativos e bibliotecas de câmera:
```bash
sudo apt update
sudo apt install libcap-dev libgnutls28-dev python3-libcamera -y
```

### 2. Configuração do Escudo de Hardware (RAM Drive)
Crie o ponto de montagem para o armazenamento temporário:
```bash
mkdir -p ~/miniola/miniola_py/captura
```

Abra o arquivo de configuração de partições (`sudo nano /etc/fstab`) e adicione esta linha ao final:
```text
tmpfs /home/felipe/miniola/miniola_py/captura tmpfs defaults,noatime,size=1024M 0 0
```

Ative a montagem imediata:
```bash
sudo systemctl daemon-reload && sudo mount -a
```

### 3. Ambiente Python (3.13+)
Configure o ambiente virtual permitindo o uso dos pacotes de sistema:
```bash
cd ~/miniola/miniola_py
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Configuração do Atalho (Alias)
Para que o comando `miniola` funcione em qualquer diretório:
```bash
echo "alias miniola='~/miniola/miniola_py/start.sh'" >> ~/.bashrc
source ~/.bashrc
```

---

## Operação e Atalhos

Ao digitar `miniola`, o script `start.sh` fará o `git pull` automático da branch de desenvolvimento e iniciará os serviços.

### Comandos do Painel de Controle (Terminal):
```text
k / l   : Ajustar Foco (Longe / Perto)
j [v]   : Definir passo do motor de foco (ex: j 0.05)
f / p   : Iniciar Gravação / Pausar
r       : Resetar contadores e limpar RAM Drive
e [v]   : Ajustar Exposição (Shutter Speed)
g [v]   : Ajustar Ganho Analógico
o       : Auto-Threshold (Algoritmo Otsu)
w / s   : Mover área de leitura (ROI) para Cima / Baixo
a / d   : Mover área de leitura (ROI) para Esquerda / Direita
< / >   : Ajustar linha de gatilho vertical
```

---

## Estrutura do Repositório
* `miniola.py`: Núcleo do sistema (Câmera, Flask e Lógica de Detecção).
* `start.sh`: Script de boot, atualização e ativação do venv.
* `/captura`: Ponto de montagem RAM (Mapeado como `CAPTURE_PATH`).

---

### Notas de Manutenção (Log de 15/03/2026)
> **Resiliência Headless:** Foi implementado um *Mock* para `sys.modules["pykms"]` no topo do `miniola.py`. Esta alteração resolve o erro de `ModuleNotFoundError` em ambientes sem monitor físico no Raspberry Pi OS (Bookworm).
