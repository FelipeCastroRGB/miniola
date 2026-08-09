# SPEC-010: Integração de Transporte Motorizado (SKR Pico / TMC2209)

| Metadado | Valor |
| :--- | :--- |
| **ID da Especificação** | `SPEC-010` |
| **Status** | `Draft` |
| **Autor** | Antigravity (IA) |
| **Data de Criação** | 2026-07-26 |
| **Última Atualização** | 2026-07-26 |

---

## 1. Contexto e Objetivo
O Miniola atualmente atua como um sistema de captura passivo ou sincronizado manualmente. Para automatizar o processo de digitalização de película e garantir uma tração suave, constante e tracionada com precisão, o sistema requer a integração de um transporte motorizado. 
O hardware escolhido é a placa **BTT SKR PICO V1.0** (baseada no microcontrolador RP2040) em conjunto com drivers **TMC2209** integrados, controlando 2 motores de passo **NEMA 17 (42BYGH23-A-21DH)**. 
O objetivo desta especificação é definir como o Miniola se comunicará com esta placa via USB para controlar a velocidade e direção dos motores, mantendo a compatibilidade multi-plataforma e sem onerar a CPU que processa a visão computacional.

## 2. Requisitos Funcionais (O que o sistema deve fazer)
- `[RF-01]`: O sistema deve ser capaz de enviar comandos de Iniciar, Parar, Avançar, Recuar e Definir Velocidade para a placa controladora via porta serial (USB).
- `[RF-02]`: O dashboard web do Miniola (`miniola.py`) deve expor controles de interface (botões e sliders) para o transporte motorizado.
- `[RF-03]`: A comunicação serial deve ser não-bloqueante para evitar atrasos no loop principal de captura de quadros.
- `[RF-04]`: O sistema deve permitir a configuração da porta serial (`/dev/tty*` ou COM) via linha de comando ou arquivo de configuração.

## 3. Requisitos Não-Funcionais e Performance
- `[RNF-01]`: A taxa de captura do sensor (`fps_cam`) não deve ser afetada pela comunicação com os motores. O envio de comandos deve ocorrer em uma thread ou processo separado (ou via chamadas assíncronas de baixo custo).
- `[RNF-02]`: A placa controladora deve garantir a precisão dos pulsos (steps) em hardware real-time (no RP2040), garantindo tração suave e operação silenciosa (StealthChop no TMC2209).

---

## 4. Matriz de Impacto Multi-Plataforma

Descreva o comportamento e as restrições em cada perfil de hardware suportado pelo Miniola:

| Plataforma | Comportamento Esperado / Restrições Específicas |
| :--- | :--- |
| **Raspberry Pi 5/4 (`arm64`)** | Conexão USB-C nativa ou via GPIO UART. A porta padrão será tipicamente `/dev/ttyACM0` ou `/dev/ttyUSB0`. Mantém-se o paradigma de gravação em `tmpfs`. |
| **Mac Mini / MiniPCs (`x86_64`)** | Conexão exclusivamente via USB. A porta padrão será algo como `/dev/tty.usbmodem*` (macOS) ou `/dev/ttyACM0` (Linux). Nenhuma diferença no código lógico de controle, apenas o nome da porta. |

---

## 5. Arquitetura e Design Técnico

### 5.1. Componentes e Arquivos Modificados
- `core/motor_controller.py` **[NOVO]**: Módulo em Python responsável por instanciar a comunicação Serial e expor métodos de alto nível (`start()`, `stop()`, `set_speed()`).
- `miniola.py`: Instanciará o `motor_controller` e o passará (ou invocará) a partir do servidor web/dashboard, e conectará os endpoints da API web aos comandos do motor.
- `templates/` e `web/`: Inclusão de botões "Transporte" (Avançar/Recuar, Stop, Slider de Velocidade) no Dashboard (interface web).
- **Firmware da Placa**: A SKR Pico V1.0 rodará um firmware **C++ customizado**. Este firmware receberá comandos seriais simples via UART (via cabo USB-C) do host (RPi ou PC) e traduzirá esses comandos em sinais STEP/DIR para os drivers TMC2209 no microcontrolador RP2040 em tempo real. Adicionalmente, o firmware fará a leitura de um **Encoder Rotativo E38S6G5-600B-G24N (sinal NPN)** diretamente conectado aos seus pinos GPIO, e reportará a contagem de pulsos (posição) via porta serial de volta ao host.

### 5.2. Contratos e Estruturas de Dados

```python
# Contrato sugerido para o motor_controller.py
class MotorController:
    def __init__(self, port: str, baudrate: int = 115200):
        pass

    def connect(self) -> bool:
        pass

    def disconnect(self):
        pass

    def move_forward(self, speed: float):
        """ speed em mm/s ou rpm """
        pass

    def move_backward(self, speed: float):
        pass

    def stop(self):
        pass
```

---

## 6. Critérios de Aceitação e Plano de Verificação

### 6.1. Verificação Automatizada / Bancada (`tests/`)
- [ ] O teste unitário `test_motor_controller.py` deve mockar a porta serial e verificar se os comandos corretos (ex: strings G-code ou protocolo binário) são enviados para o buffer.
- [ ] A checagem `python3 scripts/check_specs.py` não deve apontar erros nesta spec.

### 6.2. Verificação Manual / Hardware
- [ ] Conectar a SKR Pico via USB e verificar se o Miniola consegue abrir a porta serial sem crashar.
- [ ] Clicar no botão "Avançar" no Dashboard e confirmar se o comando é transmitido e os NEMA 17 giram sem afetar o FPS da câmera.

---

## 7. Fase 2: Controle Duplo e Dinâmica de Rolos (Tração Direta)

Na Fase 2, o sistema passará a controlar 2 motores (Feed-in e Take-up) cujas velocidades físicas mudam constantemente devido à variação do diâmetro dos rolos de filme. A arquitetura escolhida (Opção A) define que:

### 7.1. Medição de Velocidade via Encoder
O hardware utilizará um **Encoder Rotativo E38S6G5-600B-G24N (600 PPR)** acoplado a um rolete por onde o filme passa. O rolete tem um **diâmetro de 30.5 mm**. O firmware da SKR Pico fará a contagem dos pulsos (usando interrupções rápidas via PIO/registradores no pino de UART) e o módulo Python converterá para velocidade linear (mm/s) através da fórmula:
`Velocidade (mm/s) = (Pulsos_Lidos / 600) * (PI * 30.5) / Tempo_Decorrido`
*Nota: Qualquer acoplamento anterior do cálculo de velocidade com a Visão Computacional (OpenCV) foi **completamente removido** da malha mecânica. Conforme definido na **SPEC-011**, a responsabilidade da velocidade é 100% do Encoder físico.*

### 7.2. Malha de Controle (PID) em Python e Aceleração (S-Curve)
A "mola matemática" será implementada em Python. O módulo `motor_controller.py` implementará um loop PID (Proporcional-Integral-Derivativo) **estritamente isolado da ótica**. Utilizando a velocidade linear calculada pelo encoder como Variável de Processo (PV). 

A velocidade alvo (Setpoint) durante a Gravação (REC) será derivada diretamente da variável global `fps_motor` (configurada via comando `mfps` no terminal), convertida para mm/s. O "sistema de playback" (sincronia estrita via `FPS_PROJECAO`) está temporariamente desabilitado para privilegiar a cadência e tração contínua da película, e a captura da câmera (`fps_cam`) agora roda totalmente desacoplada da velocidade física dos rolos.

**Aceleração (Soft Start):** Para evitar solavancos e o rompimento do filme no momento de ligar os motores, a meta do PID não sobe subitamente (degrau). Ela passa por uma rampa matemática (Smoothstep / S-Curve) durante os primeiros N segundos da captura, garantindo um "Ease-In" mecânico liso. Ao comparar com a velocidade desejada real-time, o PID ajustará a velocidade dos motores enviando comandos diferenciais (F) via porta serial para compensar o enchimento/esvaziamento dos rolos.

### 7.3. Tensão de Emergência (StallGuard)
O firmware C++ da SKR Pico fará a leitura constante do *StallGuard 4* (sensor de carga mecânica sem sensor embutido nos drivers TMC2209) medindo o *back-EMF*. Caso o filme enrosque, o pico de tensão mecânica será detectado instantaneamente pelo firmware, que cortará a corrente dos motores (emergência) e notificará o host via USB, protegendo a película de rompimento.
