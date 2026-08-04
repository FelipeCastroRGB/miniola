# SPEC-003: Extração e Síntese de Áudio Óptico (Captura ao Vivo e Pós-Processamento)

| Metadado | Valor |
| :--- | :--- |
| **ID da Especificação** | `SPEC-003` |
| **Status** | `Completed` |
| **Autor** | Equipe Miniola |
| **Data de Criação** | 2026-07-19 |
| **Última Atualização** | 2026-07-19 |

---

## 1. Contexto e Objetivo
O filme de 35mm frequentemente carrega uma pista de som óptico (densidade variável ou área variável) na lateral adjacente às perfurações. Para preservar o som junto com a imagem sem precisar de um cabeçote magnético ou leitor laser dedicado, o Miniola captura amostras visuais de luminância da pista de áudio a cada quadro em alta velocidade e, no pós-processamento, reconstrói o sinal de áudio digital sincronizado em arquivo WAV.

Esta especificação define o pipeline de extração de som ao vivo (geração do sidecar `.f32` + metadados `.json`) no `miniola.py`/`miniola_cv.cpp` e a síntese/alinhamento em `process.py`.

## 2. Requisitos Funcionais
- `[RF-01]`: O sistema deve definir uma região de extração de áudio na lateral da ROI de perfuração (`AUDIO_X_OFFSET = 50`, `AUDIO_READ_W = 96`).
- `[RF-02]`: Durante a gravação (`GRAVANDO = True`), o motor de visão (em C++) deve extrair as linhas da pista de áudio para cada quadro processado e gerar um pacote `audio_chunk` de luminância.
- `[RF-03]`: Os chunks de áudio extraídos devem ser enviados para a `fila_gravacao` de forma assíncrona com mensagem `{"type": "audio_chunk", "data": chunk}`.
- `[RF-04]`: O processo de gravação deve gerenciar uma sessão de áudio (`abrir_sessao_audio_optico`, `fechar_sessao_audio_optico`) e escrever sequencialmente as amostras brutas de ponto flutuante em `capturas/miniola_audio_{session_id}.f32`.
- `[RF-05]`: Ao encerrar a gravação (`rec_stop`), o sistema deve salvar um arquivo sidecar contendo metadados de sincronismo (`capturas/miniola_audio_{session_id}.json`), incluindo taxa de amostragem calculada com base no FPS de projeção (`fps_projecao`) e no pitch calculado (`samples_per_frame = pitch * 4`).
- `[RF-06]`: No pós-processamento (`process.py --extract-audio`), o script deve ler o `.f32` e `.json`, aplicar processamento de sinal (filtragem de ruído, equalização/normalização) e gerar um arquivo `WAV` sincronizado com o vídeo final na pasta `output/`.

## 3. Requisitos Não-Funcionais e Performance
- `[RNF-01]`: A extração da amostra visual da fenda de áudio no C++ (`miniola_cv.cpp`) deve ser vetorizada pela média das colunas em cada linha horizontal da fenda, operando sem alocação dinâmica contínua.
- `[RNF-02]`: O fluxo contínuo de escrita de chunks no arquivo `.f32` deve ser não-bloqueante no processo isolado do disco, com `flush()` ao final de cada sessão.

---

## 4. Matriz de Impacto Multi-Plataforma

| Plataforma | Comportamento Esperado / Restrições Específicas |
| :--- | :--- |
| **Raspberry Pi 5/4 (`arm64`)** | O processamento em `process.py` utiliza `scipy.signal` e `noisereduce`. No RPi, a filtragem de ruído espectral pesado pode levar alguns segundos por minuto de áudio; o script deve exibir barra de progresso no terminal e evitar consumo excessivo de RAM (processamento em blocos se necessário). |
| **Mac Mini / MiniPCs (`x86_64`)** | Com CPU multicore e vetorização AVX, a conversão `.f32` -> `.wav` com redução de ruído roda instantaneamente, permitindo até mesmo aplicar filtros mais sofisticados ou taxas de sobreamostragem maiores. |

---

## 5. Arquitetura e Design Técnico

### 5.1. Componentes e Arquivos Modificados
- `src/miniola_cv.cpp`: Método `process_frame` extrai a fenda de áudio nas coordenadas `(audio_x, slit_y - h/2, audio_w, h)` e retorna array 1D no campo `audio_chunk`.
- `miniola.py`: Funções `abrir_sessao_audio_optico`, `fechar_sessao_audio_optico` e interceptação no `processo_escrita_disco` para despejar amostras binárias no arquivo `.f32`.
- `process.py`: Módulo de reconstrução e compilação FFmpeg/WAV.

### 5.2. Estrutura do Arquivo de Metadados de Áudio (`.json`)
```json
{
  "version": 1,
  "session_id": "20260719T153000Z",
  "started_at_utc": "20260719T153000Z",
  "closed_at_utc": "20260719T153100Z",
  "close_reason": "manual_stop",
  "mode": "variable_density",
  "fps_projecao": 24.0,
  "samples_per_frame": 780,
  "source_sample_rate": 18720,
  "frames_with_audio": 1200,
  "total_samples": 936000,
  "raw_path": "miniola_audio_20260719T153000Z.f32"
}
```

---

## 6. Critérios de Aceitação e Plano de Verificação

### 6.1. Verificação Automatizada (`tests/`)
- [x] O framework de testes `tests/` verifica se um frame simulado com variação senoidal na fenda de áudio gera um `audio_chunk` flutuante com a frequência esperada.
- [x] O linter de specs (`check_specs.py`) aprova a estrutura da SPEC-003.

### 6.2. Verificação em Hardware / Operação
- [x] Ao iniciar e parar uma gravação no painel do scanner, os arquivos `miniola_audio_{sid}.f32` e `miniola_audio_{sid}.json` são criados em `capturas/` com tamanho proporcional ao tempo e ao pitch médio.
- [x] Executar `python3 process.py --extract-audio` gera com sucesso um arquivo de áudio limpo em `output/`.
