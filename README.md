# Miniola - Scanner de Preservacao Audiovisual (35mm)

O **Miniola** é um dispositivo de baixo custo desenhado para inspeção de películas cinematograficas com o objetivo de preservar o patriônio audiovisual e democraziar o acesso às ferramentas de visionamento dos materiaies em filme. O projeto utiliza Raspberry Pi + Camera Module para capturar quadros sincronizados por detecção de perfuraçôes via OpenCV, assim como extração de som óptico AV e DV.

> Estado atual de hardware: migrado para **Raspberry Pi 4 B (1 GB)** para maior desempenho de OpenCV.

---

## Estrutura atual do repositorio (branch `desenvolvimento`)

Arquivos principais agora ficam na **raiz do repositorio**:

- `miniola.py`: ponto de entrada principal.
- `process.py`: pos-processamento (gera MP4/ProRes a partir dos frames).
- `miniola_debug.py`: variante de depuracao (opcional).
- `start.sh`: script de boot e atualizacao (`git pull` + execucao).
- `requirements.txt`: dependencias Python.

---

## Guia de instalacao limpa (Raspberry Pi OS Bookworm)

### 1) Dependencias do sistema

```bash
sudo apt update
sudo apt install libcap-dev libgnutls28-dev python3-libcamera git python3-dev build-essential ffmpeg libopencv-dev pkg-config -y
```

### 2) Clonagem

```bash
git clone -b desenvolvimento https://github.com/FelipeCastroRGB/miniola.git
cd ~/miniola
```

### 3) RAM Drive para captura (`tmpfs`)

```bash
mkdir -p ~/miniola/capturas
mkdir -p ~/miniola/output
```

Edite `sudo nano /etc/fstab` e adicione ao final:

```text
tmpfs /home/felipe/miniola/capturas tmpfs defaults,noatime,size=1024M 0 0
```

Aplicar montagem:

```bash
sudo systemctl daemon-reload && sudo mount -a
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

#### Opção A: Raspberry Pi Camera Module 3 (Padrão)
Se você estiver utilizando a câmera padrão do ecossistema Raspberry Pi, instale as dependências correspondentes:
```bash
pip install picamera2 python-prctl
```

#### Opção B: Câmera Ximea (MQ042MG-CM)
Se o scanner utilizar a câmera industrial Ximea, instale o SDK oficial da fabricante (que inclui os drivers e a biblioteca em Python `ximea_api`). Acesse a documentação oficial da Ximea para sistemas ARM/Linux ou execute:
```bash
wget -O XIMEA_Linux_SP.tgz https://updates.ximea.com/public/ximea_linux_arm_sp_beta.tgz
tar -xzf XIMEA_Linux_SP.tgz
cd package
./install -cam_usb30
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

### 6) Atalho de execucao

```bash
chmod +x ~/miniola/start.sh
echo "alias miniola='~/miniola/start.sh'" >> ~/.bashrc
source ~/.bashrc
```

---

## Operacao

Ao executar `miniola`, o `start.sh` faz:

1. `git pull origin desenvolvimento`
2. ativa `venv`
3. inicia `python3 miniola.py`

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
