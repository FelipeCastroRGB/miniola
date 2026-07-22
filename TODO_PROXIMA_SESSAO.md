# Roadmap & Checklist para Próxima Sessão (Miniola)

Este documento centraliza a varredura arquitetural e técnica do repositório **Miniola**, definindo as prioridades de refatoração, conformidade multi-plataforma e melhorias de performance para a próxima sessão de desenvolvimento.

---

## 1. Provedores de Câmera (`cameras/`) [SPEC-002 & SPEC-006]

- [ ] **1.1. Implementar Provedor `uvc` (`cameras/uvc.py`)**:
  - Criar classe `UVCCameraAdapter(CameraProvider)` utilizando `cv2.VideoCapture(index, cv2.CAP_V4L2)` para permitir que qualquer webcam ou câmera industrial USB genérica conecte no Mac Mini ou PC Linux sem depender do SDK da Ximea.
- [ ] **1.2. Implementar Provedor `mock` (`cameras/mock.py`)**:
  - Criar classe `MockCameraProvider(CameraProvider)` capaz de gerar quadros sintéticos contínuos a 120 FPS (com perfurações animadas descendo pela ROI e fenda de áudio de teste) ou reproduzir um arquivo de vídeo preliminar (`playback`). Isso permitirá testar gatilhos, rotas web e gravação assíncrona em PCs de desenvolvimento sem nenhum hardware conectado.
- [ ] **1.3. Otimizar Seletor Dinâmico (`cameras/__init__.py`)**:
  - Evitar importações incondicionais no topo (`from .pi import PiCameraAdapter`), carregando os módulos sob demanda via `get_camera_provider(name)` para que sistemas `x86_64` não tentem checar dependências exclusivas de Raspberry Pi na inicialização.

---

## 2. Modularização e Desacoplamento do Core (`miniola.py`) [SPEC-005]

- [ ] **2.1. Extrair Servidor Web/Dashboard (`web/` ou `routes.py`)**:
  - O arquivo `miniola.py` possui mais de 880 linhas englobando visão computacional, fila IPC, terminal interativo e rotas do Flask. Isolar as rotas (`/video_feed`, `/status`, `/cmd`, `/status_stream`) e o gerador de streaming em um módulo dedicado melhorará a manutenibilidade e a clareza.
- [ ] **2.2. Extrair Worker de Gravação Assíncrona (`workers/recorder.py`)**:
  - Mover `processo_escrita_disco(fila_in)` e funções de encerramento de sessão de áudio óptico (`abrir_sessao_audio_optico`, `fechar_sessao_audio_optico`) para um módulo trabalhador isolado (`workers/`), reforçando o desacoplamento entre o loop de captura de alta prioridade (`Core 0/1`) e o loop de I/O em disco (`Core 2/3`).

---

## 3. Inteligência de Pós-Processamento (`process.py`) [SPEC-001 & SPEC-004]

- [ ] **3.1. Detecção Automática do Tipo de Obturador nos Metadados**:
  - Atualmente, para evitar distorção vertical de *breathing* em câmeras Global Shutter (`Ximea`), o usuário ou script passa a flag `--disable-rs-comp` ao acionar o `process.py`. Inserir o campo `"shutter_type": "global" | "rolling"` no cabeçalho dos arquivos `miniola_tracking_{sid}.jsonl` para que o `process.py` aplique a compensação de Rolling Shutter automaticamente apenas quando necessário.
- [ ] **3.2. Validação da Pipeline de Áudio Óptico (Filtro Anti-Aliasing Slit)**:
  - Adicionar testes automatizados cobrindo a conversão e interpolação cúbica (`CubicSpline`) do áudio óptico de densidade variável e área variável gerado nos arquivos `.f32` sidecar.

---

## 4. Expansão da Suíte de Testes (`tests/`) e SDD

- [ ] **4.1. Testes de Concorrência e Isolamento de Buffer**:
  - Criar teste unitário em `tests/` que simule leituras concorrentes em threads separadas para garantir que o `.copy()` no retorno dos provedores de câmera evite condições de corrida (*race conditions*) contra a conversão do OpenCV/Flask.
- [ ] **4.2. Benchmark de Gravação Assíncrona (`libjpeg-turbo`)**:
  - Criar teste de estresse em `tests/` que enfileire 300 quadros BGR na `multiprocessing.Queue` e meça o tempo de vazão via `cv2.imwrite`, garantindo que o tempo médio permaneça estritamente abaixo do limite de 5 milissegundos por fotograma (`[RNF-02]` da `SPEC-004`).
- [ ] **4.3. Criação de Especificação para Ferramenta de Anotação (`specs/007-visual-annotation-tool.md`)**:
  - Revisar e completar a especificação `SPEC-007` (caso esteja em rascunho ou pendente de expansão) para cobrir a interface de anotação visual de defeitos de filme.

---

## 5. Referências e Padrões (`referencias/`)

- [ ] **5.1. Mapeamento de Normas na Documentação**:
  - Conectar explicitamente os parâmetros de áudio óptico (notch de 90Hz/180Hz, passa-baixa de 7kHz no `process.py`) com os documentos `referencias/fadgi-audio-adc-performance.md` e `referencias/kodak-filmmakers-reference-guide.md` em notas explicativas ou na `SPEC-003`.

---

## 6. Redesign Profissional de UI/UX do Dashboard (`templates/index.html` & `miniola.py`) [SPEC-005]

A interface web atual (`index.html` e `generate_dashboard`) foi construída como um mosaico rígido de buffers OpenCV combinados verticalmente/horizontalmente (`np.vstack` / `np.hstack`). Para elevar o Miniola ao nível dos softwares de scanners profissionais de arquivo (como **Blackmagic Cintel**, **Lasergraphics ScanStation/Director** e **DFT Scanity/Spirit**), implementaremos um redesign visual completo:

- [ ] **6.1. Correção da Distorção da Imagem da Zebra**:
  - **Diagnóstico**: Atualmente em `miniola.py` (linha 715), o frame da Zebra (`ultimo_crop_preview`) é forçado para as dimensões estáticas de `(100, 280)px`. Isso esmaga e estica verticalmente qualquer proporção de corte na tela.
  - **Solução**: Modificar a renderização da Zebra para preservar estritamente a proporção de aspecto natural da película (`aspect ratio` calculado dinamicamente ou mantendo escala uniforme `min(ratio_w, ratio_h)`). Além disso, permitir que a Zebra seja ativada como um **overlay comutável (*toggle button*)** diretamente por cima do feed de vídeo principal de alta resolução, como ocorre nos monitores do Cintel e do DaVinci Resolve.
- [ ] **6.2. Arquitetura de Layout Baseada em Scopes Profissionais (GUI/UX)**:
  - **Desacoplamento do Mosaico OpenCV (`display: grid`)**: Em vez de gerar um único blocão JPEG gigante no Python com todos os painéis grudados, desmembrar o dashboard em componentes independentes no `templates/index.html` utilizando **CSS Grid / Flexbox responsivo**:
    - **Viewport Principal (Film Gate View)**: Centralizado, com guias de enquadramento limpas (*Framing Guides* profissionais: 4-perf 35mm, 3-perf, 16mm Academy) e indicadores de foco/exposição.
    - **Painel de Scopes de Engenharia (Luminance Histogram & Waveform)**: Redesenhar o histograma com escala logarítmica/linear clara, marcações de IRE (0 a 100 IRE / 0 a 255 8-bit / 10-bit), e gradiente de cor de advertência de clipping (vermelho nos extremos 0 e 255, verde na zona segura).
    - **Painel de Áudio Óptico (Optical Sound Track & VU Meter)**: Exibir a tira cinza da trilha de áudio com indicador de balanço lateral e medidor visual de pico de modulação.
- [ ] **6.3. Padronização de Design System (Cores, Fontes e Ergonomia de Estúdio)**:
  - **Paleta Dark Pro**: Adotar uma paleta cinza-escuro neutra e de alto contraste inspirada em softwares de color grading/telecine (`#181818` para o fundo da janela, `#262626` para cards de painel, `#3a3a3a` para bordas de contenção, `#00d26a` para status OK/Safe, e `#ff3333` para gravação ativa `REC`).
  - **Tipografia Técnica e Legível**: Padronizar as fontes utilizando famílias modernas e limpas via Google Fonts ou variáveis CSS (`Inter` / `Outfit` para a interface geral, e `JetBrains Mono` ou `Roboto Mono` para telemetria numérica, FPS, temperatura e posições XY de crop).
  - **Controle de Transporte (*Transport Console*)**: Redesenhar a barra de botões inferiores em estilo console físico (Botão `REC` vermelho pulsante bem nítido, `PAUSE`, controles milimétricos de Crop, e botões táteis para alteração de padrão Bayer `0-3` e ganho/exposição do sensor ao vivo sem precisar digitar no terminal).
