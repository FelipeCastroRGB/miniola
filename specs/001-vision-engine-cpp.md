# SPEC-001: Motor de Visão Computacional em C++ (miniola_cv)

| Metadado | Valor |
| :--- | :--- |
| **ID da Especificação** | `SPEC-001` |
| **Status** | `Completed` |
| **Autor** | Equipe Miniola |
| **Data de Criação** | 2026-07-19 |
| **Última Atualização** | 2026-07-19 |

---

## 1. Contexto e Objetivo
O Miniola inspeciona filmes cinematográficos de 35mm em alta velocidade. Para capturar quadros sincronizados de forma estável, o sistema analisa a fenda da perfuração em tempo real (120 FPS+). Fazer binarização, busca de contornos e cálculos geométricos complexos em puro Python (NumPy/OpenCV Python) satura o interpretador e derruba a taxa de quadros no Raspberry Pi.

Esta especificação define o comportamento do motor nativo otimizado em C++ (`miniola_cv.cpp`) que se conecta ao Python via `pybind11`, bem como o contrato de fallback em Python puro caso a extensão C++ não esteja compilada.
Conforme estabelecido pela **SPEC-011**, o papel primordial deste módulo passa a ser o de **Extrator de Fase (Frame Picker)**, abandonando a responsabilidade de medir a velocidade do transporte.

## 2. Requisitos Funcionais
- `[RF-01]`: O motor de visão deve recortar a região de interesse (ROI) da perfuração a partir do quadro RAW da câmera.
- `[RF-02]`: O motor deve aplicar threshold de binarização (`THRESH_VAL`) na escala de cinza da ROI reduzida (`ESCALA_CV = 0.5`) para detecção rápida.
- `[RF-03]`: Deve encontrar contornos, filtrar por proporção (`0.2 < w/h < 2.5`) e área (`200 < area < 10000`) para identificar furos válidos de 35mm.
- `[RF-04]`: Deve detectar se uma perfuração cruzou a linha de gatilho Y (`LINHA_GATILHO_Y +- MARGEM_GATILHO`).
- `[RF-05]`: A cada ciclo de 4 perfurações (padrão 35mm = 4 furos por fotograma), o motor deve calcular o centro X e Y do quadro (`cx_a`, `cy_a`), calcular o pitch instantâneo/médio e sinalizar `capturar = True`.
- `[RF-06]`: Deve computar o encolhimento percentual atual do filme (`encolhimento_atual_pct`) comparando o pitch médio com `PITCH_PADRAO_PX`.
- `[RF-07]`: O motor atua estritamente como um Seletor de Fase (Frame Picker). A sinalização de alinhamento (`perfuracao_na_linha`) não deve mais ser usada como entrada para a malha de velocidade PID do motor.

## 3. Requisitos Não-Funcionais e Performance
- `[RNF-01]`: O tempo de execução por quadro (`tempo_ms_ciclo`) na chamada `process_frame` deve ser inferior a 3 ms em Raspberry Pi 5 e inferior a 1 ms em x86_64.
- `[RNF-02]`: A alocação de memória dentro do C++ deve reutilizar buffers estáticos de contorno ou vetores internos, evitando chamadas contínuas de `malloc`/`new`.

---

## 4. Matriz de Impacto Multi-Plataforma

| Plataforma | Comportamento Esperado / Restrições Específicas |
| :--- | :--- |
| **Raspberry Pi 5/4 (`arm64`)** | Compilado via `setup.py` ligando com `/usr/include/opencv4` do sistema Debian Bookworm. Em RPi 4, a resolução do sensor é cortada (`RES_W=1420, RES_H=880`) antes de enviar ao C++ para evitar gargalo USB. |
| **Mac Mini / MiniPCs (`x86_64`)** | Compilado via `setup.py` usando `pkg-config opencv4`. Aproveita otimizações nativas de CPU Intel/AMD, permitindo processar a resolução integral ou matrizes maiores com latência praticamente zero. |

---

## 5. Arquitetura e Design Técnico

### 5.1. Componentes e Arquivos Modificados
- `src/miniola_cv.cpp`: Módulo C++ com a classe `ScannerVision` exportada via `pybind11`.
- `setup.py`: Script de build `build_ext` configurado para buscar flags e bibliotecas do OpenCV via `pkgconfig` ou fallback padrão.
- `miniola.py`: Importação com bloco `try/except ImportError` para definir `CV_ENGINE = "C++ [Pybind11]"` ou `"Python [Nativo]"`.

### 5.2. Contrato da API (`pybind11`)
```python
class ScannerVision:
    def __init__(self) -> None: ...
    def reset_ciclo(self) -> None: ...
    def process_frame(
        self,
        frame_raw: np.ndarray,
        lx: int, ly: int, lw: int, lh: int,
        thresh_val: int, linha_gatilho_y: int, margem_gatilho: int,
        pitch_padrao_px: float,
        capturar_audio: bool, audio_x: int, audio_w: int, slit_y: int
    ) -> dict: ...
```
O dicionário de retorno do `process_frame` deve conter:
- `capturar`: `bool`
- `achou_furo`: `bool`
- `perfuracao_na_linha`: `bool`
- `contador_perfs_ciclo`: `int`
- `cx_a`, `cy_a`: `int` (coordenadas globais para o centro do fotograma)
- `encolhimento_atual_pct`: `float`
- `ultimo_pitch_medio`: `float`
- `pitch_instantaneo`: `float`
- `binary_small`: `np.ndarray` (imagem binarizada para o dashboard)
- `debug_visual`: `list[dict]` (retângulos com coordenadas e cores dos furos detectados)
- `audio_chunk`: `np.ndarray` (amostra de áudio extraída, se `capturar_audio` ativo)

---

## 6. Critérios de Aceitação e Plano de Verificação

### 6.1. Verificação Automatizada (`tests/`)
- [x] O módulo C++ compila com sucesso executando `python3 setup.py build_ext --inplace`.
- [x] O teste de bancada `test_vision_engine.py` passa ao processar quadros sintéticos simulando 4 perfurações sequenciais, validando que o gatilho dispara no 4º furo.

### 6.2. Verificação em Hardware / Operação
- [x] Ao alternar o comando `motor` no painel do `miniola.py`, o sistema transita entre C++ e Python sem travar e relatando o motor no cabeçalho do painel.
