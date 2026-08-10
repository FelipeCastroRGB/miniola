# SPEC-012: Controle de Iluminação LED (Painel) via SKR Pico

| Metadado | Valor |
| :--- | :--- |
| **ID da Especificação** | `SPEC-012` |
| **Status** | `Draft` |
| **Autor** | Antigravity (IA) |
| **Data de Criação** | 2026-08-09 |
| **Última Atualização** | 2026-08-09 |

---

## 1. Contexto e Objetivo
Para realizar a digitalização dos fotogramas, o Miniola necessita de uma fonte de iluminação (Painel LED) para projetar a luz através da película. Em vez de utilizar uma fonte de alimentação externa sem controle de software, o objetivo é ligar o painel LED diretamente a uma das portas de potência da controladora BTT SKR Pico (ex: porta do cartucho de aquecimento HE0) e usar o sinal PWM da placa para dimerizar a intensidade de luz.

## 2. Requisitos Funcionais (O que o sistema deve fazer)
- `[RF-01]`: O firmware C++ da SKR Pico deve receber o comando serial `L <valor>` (de 0 a 255) para alterar o PWM da porta de potência, ajustando o brilho do LED.
- `[RF-02]`: A interface Web (Dashboard) deverá ter um controle deslizante (slider) para ajustar o nível da iluminação em tempo real.
- `[RF-03]`: A biblioteca `motor_controller.py` deve abstrair a conversão do comando e o envio via Serial.

## 3. Requisitos Não-Funcionais e Performance
- `[RNF-01]`: O envio de comandos de LED não deve engasgar a leitura do encoder nem a velocidade de processamento das perfurações.

---

## 4. Matriz de Impacto Multi-Plataforma

| Plataforma | Comportamento Esperado / Restrições Específicas |
| :--- | :--- |
| **Todas** | Comunicação Serial já abstrata. Independente de OS. |

---

## 5. Arquitetura e Design Técnico

### 5.1. Componentes Modificados
- `firmware/src/main.cpp`: Inclusão do `#define LED_PIN 23` (Porta de Heater 0, suporta até ~5A, ideal para o painel LED) e parse do comando `L`.
- `core/motor_controller.py`: Novo método `set_led_brightness(self, level)`.
- `web/routes.py`: Nova rota `/api/set_light` que recebe POST ou GET e repassa ao MotorController.
- `templates/index.html`: Novo Slider na UI para controle interativo.
