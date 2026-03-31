# Miniola - Scanner de Preservacao Audiovisual (35mm)

O **Miniola** e uma estacao de baixo custo para digitalizacao e inspecao de peliculas cinematograficas. O projeto utiliza Raspberry Pi + Camera Module 3 para capturar quadros sincronizados por deteccao de perfuracoes.

> Estado atual de hardware: migrado para **Raspberry Pi 4 B (1 GB)** para maior desempenho de OpenCV.

---

## Estrutura atual do repositorio (branch `desenvolvimento`)

Arquivos principais agora ficam na **raiz do repositorio**:

- `miniola.py`: ponto de entrada principal.
- `process.py`: pos-processamento (gera MP4/ProRes a partir dos frames).
- `start.sh`: script de boot e atualizacao (`git pull` + execucao).
- `requirements.txt`: dependencias Python.
- `miniola_py/`: camada legada de compatibilidade temporaria.

---

## Guia de instalacao limpa (Raspberry Pi OS Bookworm)

### 1) Dependencias do sistema

```bash
sudo apt update
sudo apt install libcap-dev libgnutls28-dev python3-libcamera git python3-dev build-essential ffmpeg -y
```

### 2) Clonagem

```bash
git clone -b desenvolvimento https://github.com/FelipeCastroRGB/miniola.git
cd ~/miniola
```

### 3) RAM Drive para captura (`tmpfs`)

```bash
mkdir -p ~/miniola/capturas
```

Edite `sudo nano /etc/fstab` e adicione ao final:

```text
tmpfs /home/felipe/miniola/capturas tmpfs defaults,noatime,size=1024M 0 0
```

Aplicar montagem:

```bash
sudo systemctl daemon-reload && sudo mount -a
```

### 4) Ambiente Python (venv com pacotes de sistema)

```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 5) Atalho de execucao

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

## Compatibilidade legada

- `miniola_py/start.sh` foi mantido como ponte para `~/miniola/start.sh`.
- `miniola.py` e `process.py` na raiz funcionam como entrada oficial.

---

## Nota de manutencao

Foi aplicado mock para `sys.modules["pykms"]` em `miniola.py` para evitar erro `ModuleNotFoundError` em ambiente headless.
