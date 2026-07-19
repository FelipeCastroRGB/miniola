# SPEC-006: Suporte Multi-Plataforma para MiniPCs e Mac Mini (`x86_64` Linux)

| Metadado | Valor |
| :--- | :--- |
| **ID da Especificação** | `SPEC-006` |
| **Status** | `Completed` |
| **Autor** | Equipe Miniola |
| **Data de Criação** | 2026-07-19 |
| **Última Atualização** | 2026-07-19 |

---

## 1. Contexto e Objetivo
O Miniola nasceu rodando no Raspberry Pi 5 e 4. Embora o Raspberry Pi 5 seja excelente e compacto, existem cenários onde o usuário ou instituição arquivística já possui computadores compactos potentes — como um **Mac Mini Late 2012** rodando Linux (`x86_64`), Intel NUCs ou MiniPCs industriais com portas USB 3.0 rápidas e discos SSD internos.

Rodar em um MiniPC x86_64 sem modificações manuais elimina a dependência exclusiva de placas Raspberry Pi, aumenta a folga de memória RAM (8GB a 16GB+) e de processamento multicore, e permite conectar câmeras Ximea por USB 3.0 ou câmeras genéricas UVC / arquivos de simulação (Mock Playback) para testes e digitalização na bancada do usuário.

Esta é a **primeira especificação estrutural e evolutiva do nosso fluxo de Spec-Driven Development (SDD)**, estabelecendo a camada de abstração de plataforma (HAL) no Miniola.

## 2. Requisitos Funcionais
- `[RF-01]`: O sistema deve detectar a arquitetura host (`platform.machine()`) no topo do `miniola.py` para aplicar automaticamente o perfil de hardware apropriado (`arm64`/`aarch64` vs. `x86_64`).
- `[RF-02]`: O mock de subsistemas de vídeo headless exclusivos do Raspberry Pi (`sys.modules["pykms"] = MagicMock()`) deve ser isolado com segurança para não gerar conflitos com drivers de vídeo ou ambientes de janela no Linux x86_64.
- `[RF-03]`: A leitura térmica na rota `/status` deve verificar dinamicamente a presença de sensores térmicos do host (tentando `/sys/class/thermal/thermal_zone0/temp` ou paths genéricos do kernel Linux x86_64) sem falhar se o arquivo não existir (`try/except` robusto retornando `0.0`).
- `[RF-04]`: O sistema deve suportar provedores de câmera adicionais em `cameras/`:
  - Provedor `uvc` (`cameras/uvc.py` ou integrado no seletor) usando `cv2.VideoCapture` com controle UVC padrão via OpenCV, permitindo que qualquer webcam ou câmera industrial UVC USB 3.0 funcione no Mac Mini.
  - Provedor `mock` (`cameras/mock.py`) que gera quadros sintéticos ou lê um arquivo de vídeo/imagem (`playback`) para permitir que o desenvolvedor execute, teste o gatilho e altere especificações no PC local sem nenhum hardware de scanner conectado.
- `[RF-05]`: O comando de desligamento (`off`) no painel deve checar permissões ou utilizar chamadas genéricas de encerramento (`sudo poweroff` / `systemctl poweroff`) compatíveis com distribuições Debian/Ubuntu em x86_64.

## 3. Requisitos Não-Funcionais e Performance
- `[RNF-01]`: O suporte x86_64 não deve introduzir nenhum overhead computacional ou checagem de plataforma repetitiva dentro do loop crítico de visão ou no método `process_frame`. A detecção e configuração do perfil de hardware devem ocorrer uma única vez na inicialização do script.
- `[RNF-02]`: A compilação da extensão C++ (`setup.py`) deve funcionar de forma perfeitamente transparente e nativa tanto com `gcc`/`g++` em `arm64` quanto em `x86_64`.

---

## 4. Matriz de Impacto Multi-Plataforma

| Plataforma | Ações e Comportamentos do Perfil |
| :--- | :--- |
| **Raspberry Pi 5/4 (`arm64` / `aarch64`)** | Perfil `rpi`: Ativa mock de `pykms` em ambiente sem monitor, restringe buffer de processamento para aliviar USB 3.0 em modelos RPi 4, e utiliza o provedor `pi` (`picamera2`) como alternativa ao `ximea`. |
| **Mac Mini / MiniPCs (`x86_64`)** | Perfil `minipc_x86_64`: Pula inicializações exclusivas do ecossistema Pi (`picamera2`), ativa suporte nativo ao SDK Ximea Linux para arquitetura x86_64 ou câmeras `uvc`/`mock`. Em controladoras USB 3.0 xHCI (ex: Mac Mini Late 2012 com processador Ivy Bridge), a largura de banda do sensor Ximea deve usar negociação dinâmica (`auto_bandwidth_calculation=1`) e o `processo_escrita_disco` deve usar obrigatoriamente gravação nativa C++ `cv2.imwrite` com libjpeg-turbo, evitando gargalos de I/O na CPU/USB durante capturas com `REC` ativo. Requer `ffmpeg` instalado via `apt` para compilação final em `process.py`. |

---

## 5. Arquitetura e Design Técnico

### 5.1. Componentes e Arquivos Modificados
- `miniola.py`: Inclusão de verificação de arquitetura com `import platform`, tratamento seguro em `/status` para sensores térmicos x86_64 e expansão dos argumentos de `--camera` (`choices=['pi', 'ximea', 'uvc', 'mock']`).
- `cameras/__init__.py`: Seletor modular que carrega `ximea`, `pi` (somente em ARM ou com tratamento de import), `uvc` e `mock`.
- `cameras/mock.py`: Implementação da classe `MockCameraProvider` herdando de `CameraProvider` (gera quadros com 4 perfurações simuladas e fenda de som para testes contínuos de SDD).

### 5.2. Estrutura do Perfil de Hardware (Exemplo Lógico em `miniola.py`)
```python
import platform

ARQUITETURA_HOST = platform.machine().lower()
IS_RPI = "arm" in ARQUITETURA_HOST or "aarch64" in ARQUITETURA_HOST

if IS_RPI:
    # Ajustes exclusivos de Raspberry Pi
    sys.modules["pykms"] = MagicMock()
    sys.modules["kms"] = MagicMock()
```

---

## 6. Critérios de Aceitação e Plano de Verificação

### 6.1. Verificação Automatizada (`tests/`)
- [x] A suíte de testes em `tests/` pode rodar nativamente em qualquer Linux x86_64 ou Mac Mini, instanciando o provedor `mock` e validando o pipeline de visão em C++ (`miniola_cv.cpp`).
- [x] O valider `check_specs.py` confirma a especificação SPEC-006 e suas métricas.

### 6.2. Verificação Operacional no Mac Mini / MiniPC
- [x] Executar `python3 miniola.py --camera mock` no Linux x86_64 inicia o servidor Flask sem erros de `pykms` ou `Picamera2`, permitindo abrir o dashboard no navegador, ajustar o crop via web e simular gravações assíncronas.
