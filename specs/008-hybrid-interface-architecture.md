# SPEC-008: Arquitetura Híbrida de Interfaces (Web Dashboard & App Nativo Desktop)

| Metadado | Valor |
| :--- | :--- |
| **ID da Especificação** | `SPEC-008` |
| **Status** | `Draft` |
| **Autor** | Equipe Miniola |
| **Data de Criação** | 2026-07-19 |
| **Última Atualização** | 2026-07-19 |

---

## 1. Contexto e Objetivo
O **Miniola** opera em ambientes e hardwares heterogêneos de preservação audiovisual, dividindo-se em dois cenários principais no laboratório:

1. **Modo Headless (Embarcado em Raspberry Pi 5/4):** O hardware de digitalização fica acoplado à mesa enroladeira sem monitor conectado. Rodar um ambiente desktop (Wayland/X11) no Raspberry Pi consome entre 600 MB e 900 MB de RAM e compete pelos ciclos de CPU/GPU com o motor crítico C++ (`miniola_cv`) e a fila assíncrona de gravação no RAM drive (`tmpfs` de 1 GB da `SPEC-004`). Portanto, no RPi, o sistema operacional deve rodar sem ambiente gráfico (*Debian Headless / RPi OS Lite*), sendo controlado remotamente via rede.
2. **Modo Workstation de Bancada (Mac Mini ou MiniPCs `x86_64` com Monitor Dedicado):** O computador de digitalização fica sobre a bancada com um monitor de alta resolução conectado localmente. Nesse cenário, o arquivista necessita de **latência visual zero** para ajuste físico fino da rosca da lente (foco óptico sub-pixel) e alinhamento mecânico da película, além de inspeção dos pixels brutos capturados sem perda de nitidez ou artefatos de compressão de rede (JPEG/MJPEG/WebRTC).

Para atender com máxima excelência a ambos os cenários sem duplicar lógica de negócio ou comprometer a performance do processamento a 120+ FPS, esta especificação estabelece a **Arquitetura Híbrida de Interfaces do Miniola**: um núcleo backend desacoplado (`miniola-core`) capaz de alimentar tanto um **Web Dashboard otimizado** (para operação remota e headless) quanto um **App Nativo Desktop em PySide6/Qt** (para estações de bancada ou clientes remotos de alta performance).

---

## 2. Requisitos Funcionais

- `[RF-01] Desacoplamento do Core (Serviço Agnóstico)`: O loop principal do Miniola (`miniola.py` refatorado) deve separar estritamente a lógica de captura e controle da camada de apresentação, expondo chamadas unificadas de transporte (`REC`, `PAUSE`), calibração (`set_crop`, `offset`), e telemetria (temperatura, FPS, pitch instantâneo) através de uma API REST/WebSocket ou memória compartilhada.
- `[RF-02] Modo Headless & Web Dashboard Renovado (SPEC-005)`: Para uso em sistemas sem ambiente gráfico (como o Raspberry Pi Headless), o Miniola deve continuar servindo o Web Dashboard no navegador, recebendo as seguintes evoluções de UI/UX profissional:
  - **CSS Grid Responsivo & Paleta Dark Pro (`#181818`)**: Substituição da geração de mosaicos estáticos (`np.vstack`/`np.hstack`) por layout web limpo com guias vetoriais de enquadramento (4-perf, 3-perf, 16mm).
  - **Scopes de Engenharia**: Exibição clara do Histograma de Luminância (0-255 IRE), Waveform monitor e barra cinza da Fenda de Áudio Óptico (`AUDIO_READ_W`) com medidor VU de pico.
  - **Zebra de Exposição Comutável**: Alerta visual de superexposição (clipping > 245) com proporção de aspecto real preservada, operando como camada (toggle) sobre o vídeo principal sem esticar ou distorcer a prévia.
- `[RF-03] App Nativo Desktop Multiplataforma (PySide6 / Qt6)`: O sistema deve disponibilizar uma aplicação desktop nativa para Linux, macOS e Windows operando em dois modos de conexão:
  - **Modo Local (Memória RAM Direta / Latência Zero)**: Quando executado na mesma máquina que o scanner (Workstation x86_64), o app acessa diretamente os arrays NumPy dos buffers da câmera (`ultimo_frame_bruto` e `ultimo_frame_binario`) na memória, renderizando via GPU (`QImage`/OpenGL) a 60 FPS cravados sem passar por compressão ou codificação JPEG.
  - **Modo Remoto (Cliente LAN)**: Quando executado no computador pessoal ou notebook do arquivista apontando para o IP de um Miniola Headless (RPi), o app atua como cliente de rede, consumindo a API e renderizando os gráficos de telemetria (curva de pitch em tempo real e osciloscópio) nativamente com aceleração gráfica local.
- `[RF-04] Alinhamento Óptico Sub-Pixel (Foco na Bancada)`: No modo local, o App Nativo deve permitir zoom digital instâneo na ROI das perfurações e na fenda de áudio para conferência milimétrica do foco da lente e nitidez dos furos antes do início da sessão de captura.

---

## 3. Requisitos Não-Funcionais e Performance

- `[RNF-01] Pegada de Memória no Modo Headless (RPi)`: Em Raspberry Pi 5/4 rodando Debian Headless sem X11/Wayland, o consumo total de memória do serviço base do Miniola + motor C++ (`miniola_cv`) não deve ultrapassar 350 MB de RAM, garantindo a disponibilidade e estabilidade do RAM Drive `tmpfs` de 1 GB (`SPEC-004`).
- `[RNF-02] Latência Visual e Renderização Local`: No App Nativo em Modo Local, o tempo decorrido entre a disponibilização de um quadro binarizado/bruto pelo C++ e a sua renderização na tela do monitor não deve exceder 16 milissegundos (60 Hz fluidos), com zero artefatos de compressão.
- `[RNF-03] Intocabilidade da Vazão C++`: Nem o streaming web MJPEG nem a renderização do App Nativo em PySide6 podem bloquear ou atrasar a thread de captura do motor `miniola_cv.cpp`, preservando o processamento em tempo real de 120 FPS a 180 FPS.

---

## 4. Matriz de Impacto Multi-Plataforma

| Plataforma | Comportamento Esperado / Restrições Específicas |
| :--- | :--- |
| **Raspberry Pi 5/4 (`arm64`)** | Operação recomendada em **Modo Headless (Sem Ambiente Gráfico X11/Wayland)** para economizar RAM e recursos da CPU ARM. O controle e monitoramento são realizados via **Web Dashboard Renovado** no navegador ou conectando o **App Nativo Remoto** a partir de um PC de controle na rede local. |
| **Mac Mini / MiniPCs (`x86_64`)** | Operação ideal como **Workstation de Bancada com Monitor Dedicado**. Execução local do **App Nativo Desktop (`PySide6`)** com latência zero via acesso direto à memória dos frames, aproveitando aceleração de GPU e otimizações nativas x86_64 sem overhead de codificação de stream de rede. |

---

## 5. Arquitetura e Design Técnico

### 5.1. Divisão em Camadas (Desacoplamento)
Para permitir a convivência harmoniosa das duas interfaces, a estrutura evoluirá para:

```
[ Provedores de Câmera (`cameras/`) ] ---> [ Motor C++ (`src/miniola_cv.cpp`) ]
                                                    |
                                                    v
                                      [ Core Desacoplado (`miniola.py`) ]
                                         /                          \
                        (Filas e Memória Direta)         (API REST / WebSockets / MJPEG)
                                       /                              \
                                       v                               v
                     [ App Nativo Desktop (`PySide6`) ]      [ Web Dashboard (`index.html`) ]
                     - Workstation Local (Latência Zero)     - RPi Headless / Acesso Browser
                     - Cliente Remoto em LAN                 - CSS Grid & Scopes Profissionais
```

### 5.2. Módulos e Componentes Alvo
- `web/` (`routes.py` e gerador de stream): Extração das rotas `@app.route` do `miniola.py` (Item 2.1 do `TODO_PROXIMA_SESSAO.md`), mantendo compatibilidade com a `SPEC-005`.
- `workers/recorder.py`: Isolamento do processo de gravação assíncrona (`processo_escrita_disco`).
- `gui/` (Futuro módulo de desktop nativo): Pacote PySide6/Qt6 contendo componentes visuais (`scopes.py`, `live_view.py`, `transport_bar.py`) conectáveis ao core.

---

## 6. Critérios de Aceitação e Plano de Verificação

### 6.1. Verificação Automatizada (`tests/` & `check_specs.py`)
- [ ] A checagem `python3 scripts/check_specs.py` não deve apontar erros, validando que a `SPEC-008` cumpre todas as regras do SDD e da Matriz Multi-Plataforma.
- [ ] Os testes unitários em `tests/` devem confirmar que o isolamento do core não quebra as rotas e contratos da `SPEC-005`.

### 6.2. Verificação Operacional / Bancada
- [ ] No Raspberry Pi em modo Headless, verificar com `htop` / `free -m` que o consumo de memória RAM permanece abaixo de 350 MB sem interface gráfica rodando no sistema operacional.
- [ ] No Mac Mini / PC de bancada com monitor, validar que o preview visual da câmera exibe os quadros sem compressão e com resposta instantânea ao ajuste de foco mecânico da lente.
