# Miniola 

A **Miniola** é um projeto de código aberto focado no desenvolvimento de um dispositivo de baixo custo para a inspeção de películas cinematográficas. Utilizando um sistema de transporte contínuo *sprocketless* (sem roletes dentados), o projeto tem como objetivo fornecer subsídios para a preservação do patrimônio audiovisual. Sua criação nasce da vontade de facilitar o acesso ao conteúdo de filmes em película para pesquisadores e arquivistas, além de disponibilizar um conjunto de ferramentas analíticas voltadas à compreensão do estado de conservação dos materiais.

Concebido para ser acessível e modular, o projeto possui uma estrutura cujas peças fundamentais são desenhadas para fabricação em impressoras 3D pequenas e de baixo custo. Além disso, o sistema foi pensado para atender a diferentes realidades, operando sem dependência de um hardware específico e suportando requisitos computacionais flexíveis, bem como câmeras com variadas configurações.

Estado atual de testes de hardware: 

Computadores

**Raspberry Pi 4 (1 GB)**
**Raspberry Pi 5 (2 GB)**
**Mac Mini 2012 (8 GB) - Linus Mint**

Câmeras

**Raspberry Pi Camera Module 3**
**Raspberry Pi Camera Module 2**
**Raspberry Pi Camera Module 1.3**
**XIMEA MQ042MG-CM**

---

## Estrutura do repositorio

Arquivos principais agora ficam na **raiz do repositorio**:

- `miniola.py`: ponto de entrada principal.
- `process.py`: pos-processamento (gera MP4/ProRes a partir dos frames).
- `miniola_debug.py`: variante de depuracao (opcional)
- `requirements.txt`: dependencias Python.

---

## Guia de instalação

### 1) Dependências de Sistema (Multi-Plataforma)

O Miniola compila extensões C++ nativas (`miniola_cv`) e utiliza o **FFMPEG** para montagem final de vídeo sincronizado com áudio (`process.py`). Em qualquer plataforma Linux (**Mac Mini / MiniPCs `x86_64`** ou **Raspberry Pi `arm64`**), instale os pacotes obrigatórios:

```bash
sudo apt update
sudo apt install -y libcap-dev libgnutls28-dev python3-libcamera git python3-dev python3-venv build-essential ffmpeg libopencv-dev pkg-config
```

### 2) Clonagem

```bash
git clone -b desenvolvimento https://github.com/FelipeCastroRGB/miniola.git
cd ~/miniola
```

### 3) Criação dos Diretórios de Captura

O Miniola precisa de pastas específicas para salvar os quadros temporários e os vídeos processados:
```bash
mkdir -p ~/miniola/capturas
mkdir -p ~/miniola/output
```

### 4) Ambiente Python (Dependências Base)

```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install .
```

### 5) Instalação Específica da Câmera

O Miniola suporta diferentes modelos de câmera, com flexibilidade de hardware. Após instalar as dependências base, escolha e instale a opção de câmera que você for utilizar:

#### Opção A: Câmera Ximea (MQ042MG-CM)
Se o scanner utilizar a câmera industrial Ximea, instale o SDK oficial da fabricante (`ximea_api`). Como o pacote difere entre processadores ARM e Intel/AMD (x86_64), escolha o comando correto para a sua máquina:

- **Para Mac Mini / MiniPCs / Linux (`x86_64`)**:
> *Nota: Se o link principal de download (`kb.ximea.com`) der erro de rota/firewall, acesse a página oficial de downloads (`https://www.ximea.com/support/wiki/apis/XIMEA_Linux_Software_Package`) pelo navegador e baixe o arquivo `XIMEA_Linux_SP.tgz` para a pasta do projeto.*
```bash
# Ou via terminal tentando o servidor de atualizações beta x64:
wget -O XIMEA_Linux_SP.tgz https://updates.ximea.com/public/ximea_linux_x64_sp_beta.tgz || wget -O XIMEA_Linux_SP.tgz https://updates.ximea.com/public/ximea_linux_sp_beta.tgz
tar -xzf XIMEA_Linux_SP.tgz
cd package && ./install -cam_usb30
```

- **Para Raspberry Pi 5 / 4 (`arm64`)**:
```bash
wget -O XIMEA_Linux_SP.tgz https://updates.ximea.com/public/ximea_linux_arm_sp_beta.tgz
tar -xzf XIMEA_Linux_SP.tgz
cd package && ./install -cam_usb30
```

#### Opção B: Raspberry Pi Camera Module 3
Se você estiver utilizando a câmera padrão do ecossistema Raspberry Pi, instale as dependências correspondentes:
```bash
pip install picamera2 python-prctl
```
> **IMPORTANTE: Aumento do Buffer USB (Obrigatório para Ximea)**
> Para que a câmera atinja a taxa máxima de quadros (FPS) em USB 3.0 sem engasgos ou perda de pacotes, você **deve** aumentar a memória do buffer USB. No Raspberry Pi OS, faça o seguinte:
>
> **Temporário (perde ao reiniciar):**
> ```bash
> echo 1000 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb
> ```
>
> **Permanente (Recomendado):**
> Adicione o parâmetro `usbcore.usbfs_memory_mb=1000` ao final da única linha existente no arquivo `/boot/firmware/cmdline.txt`.
> ```bash
> sudo nano /boot/firmware/cmdline.txt
> ```
> *(Após editar e salvar com `Ctrl+O` > `Enter` > `Ctrl+X`, reinicie o sistema).*

### 6) Configuração do RAM Drive para captura (`tmpfs`) - Opcional

> **Nota de Hardware**: No **Raspberry Pi**, esta montagem em `tmpfs` é **muito recomendada** para gravação em cartões MicroSD. Em **MiniPCs ou computadores com SSDs de alta velocidade (NVMe/SATA)**, este passo é **opcional**, já que a gravação direta no disco costuma ser rápida o suficiente.

Se for configurar o RAM drive (`tmpfs`), edite o arquivo `fstab` (`sudo nano /etc/fstab`) e adicione ao final (lembre-se de substituir `SEU_USUARIO` pelo seu usuário real):

```text
tmpfs /home/SEU_USUARIO/miniola/capturas tmpfs defaults,noatime,size=1024M 0 0
```

Aplicar montagem:

```bash
sudo systemctl daemon-reload && sudo mount -a
```

---

## Operação

Para iniciar o sistema, ative o ambiente virtual e execute o script principal:

```bash
source venv/bin/activate
python3 miniola.py
```

---

## Pos-processamento de video (`process.py`)

Exemplos:

```bash
python3 process.py
python3 process.py --format prores
python3 process.py --format both --fps 18
python3 process.py --verify-frames
```

Por padrao, o script tenta ler frames em:

1. `./capturas`
2. `./captura` (fallback legado)

As saidas e relatorios sao gravados em `./output`.

---

## Nota de manutencao

**Resiliencia headless:** foi implementado um mock para `sys.modules["pykms"]` no topo do `miniola.py`, evitando `ModuleNotFoundError` em ambientes sem monitor fisico no Raspberry Pi OS (Bookworm).

**Fallback de motor de visao:** se `miniola_cv` falhar na compilacao (por ex., dependencias nativas ausentes), o `miniola.py` entra automaticamente em modo Python nativo. O sistema funciona, porem com menor desempenho de processamento.
