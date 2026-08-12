# SPEC-011: Modos de Captura Avançados (Telecine Híbrido)

| Metadado | Valor |
| :--- | :--- |
| **ID da Especificação** | `SPEC-011` |
| **Status** | `Draft` |
| **Autor** | Antigravity (IA) |
| **Data de Criação** | 2026-08-08 |
| **Última Atualização** | 2026-08-08 |

---

## 1. Contexto e Objetivo
Historicamente, a Miniola dependia da Visão Computacional (CV) tanto para o alinhamento óptico quanto para determinar a velocidade do transporte. Essa dependência ("Tight Coupling") causava instabilidades mecânicas sempre que a CV perdia um quadro.
Com a integração do Encoder Rotativo (SPEC-010), o sistema ganha consciência posicional absoluta. O objetivo desta especificação é definir a arquitetura do **Modo Telecine Contínuo Híbrido**, que separa as responsabilidades entre Mecânica (Velocidade) e Óptica (Fase/Alinhamento), criando um sistema resiliente a filmes danificados e encolhidos.

## 2. A Arquitetura Híbrida (Desacoplada)

A solução baseia-se na separação estrita de domínios:

### 2.1. O Domínio Mecânico (Estabilidade de Velocidade)
- O transporte do filme é governado **exclusivamente** pelo Encoder físico através da malha PID em `core/motor_controller.py`.
- O PID ignora sumariamente o que a câmera "enxerga", garantindo uma tração suave (efeito flywheel/volante de inércia) independente do estado físico das perfurações do filme.
- O Encoder fornece a métrica de `current_mm_s`.

### 2.2. O Domínio Óptico (Estabilidade de Fase e Alinhamento)
- A câmera roda em modo "Oversampling" (alta taxa de quadros, ex: 120 fps) com luz contínua.
- O motor C++ (`miniola_cv.cpp`) atua **exclusivamente** como um extrator de quadros (Frame Picker).
- Ao detectar a linha exata da perfuração, o C++ seleciona aquele quadro perfeito e o extrai, descartando os intermediários (borrados ou fora de esquadro).
- Isso garante imunidade ao "Encolhimento do Filme" (Shrinkage), pois o sistema fotografa o que realmente está na janela (Optical Registration) em vez de confiar cegamente na matemática do encoder que falha se o filme estiver encolhido.

### 2.3. Interpolação de Software (Dead-Reckoning)
- Caso o OpenCV sofra um "dropframe" ou uma perfuração esteja completamente rasgada, o software python injetará a "Previsão" baseada no Encoder.
- Se o filme avançar o equivalente a 1 frame (ex: 4.75mm para 35mm) sem que o OpenCV tenha reportado sucesso, o sistema força a gravação do quadro naquele momento exato, usando a medição física do encoder como fallback. Isso impede a "perda de sincronia".

### 2.4. Modo Playback PLL (Phase-Locked Loop Inverso)
- Para garantir o visionamento em tempo real (ex: 24 fps) sem instabilidades, a lógica mestre-escravo é invertida.
- A **Câmera atua como Mestre**: Roda continuamente a uma taxa de quadros fixa (ex: 24.0 fps) gerada pelo seu próprio relógio (Free Run).
- O **Motor atua como Escravo**: O PID do `motor_controller.py` tenta manter a velocidade em 24 fps e usa a informação do OpenCV como "Erro de Fase".
- A cada quadro, o OpenCV verifica se a perfuração está acima ou abaixo da `LINHA_GATILHO_Y`. Esse desvio em pixels é convertido para milímetros e injetado no PID como correção de fase (`update_phase_error()`), forçando o motor a acelerar ou frear sutilmente para travar a perfuração no centro da câmera.

## 3. Modificações de Arquivos Associadas
- `core/motor_controller.py`: Eliminação do `notify_perforation` como variável do PID. Adoção da lógica de interpolação.
- `src/miniola_cv.cpp`: O FPS meter acoplado ao gatilho é desabilitado. O OpenCV apenas carimba o quadro atual (Timestamps e Flags) e manda para a fila de gravação `process.py`.
- `miniola.py`: Orquestra o fluxo de gravação contínuo mesclando as informações do Encoder com o evento do OpenCV.

## 4. Requisitos Futuros (Roadmap)
- **Modo Roda-Livre:** Implementação de botão no Dashboard que corta a corrente dos motores TMC2209 (`EN=1`) permitindo manipulação manual.
- **Modo Scanner (Stop-Motion RGB):** Captura passo-a-passo (4 perfurações -> para -> fotos -> avança) para digitalização com câmera monocromática e luzes RGB.
