#include <Arduino.h>
#include <TMCStepper.h>

// Definições de pinos do Eixo X na SKR Pico V1.0 (RP2040)
#define X_STEP_PIN 11
#define X_DIR_PIN  10
#define X_EN_PIN   12
#define X_RX_PIN   9
#define X_TX_PIN   8

// Parâmetros do Motor e Driver
#define R_SENSE 0.11f // Resistor de sense (padrão na SKR Pico)
#define DRIVER_ADDRESS 0b00 // Endereço UART do Eixo X

// Inicializa a instância do driver TMC2209 utilizando a Serial2 (UART1) de hardware do RP2040
TMC2209Stepper driver(&Serial2, R_SENSE, DRIVER_ADDRESS);

bool is_moving = false;
int step_delay = 1000; // Delay em microssegundos (controla a velocidade)

void setup() {
    // Inicializa a porta serial USB para comunicação com o PC/RPi
    Serial.begin(115200);
    
    // Configura os pinos de controle do motor
    pinMode(X_STEP_PIN, OUTPUT);
    pinMode(X_DIR_PIN, OUTPUT);
    pinMode(X_EN_PIN, OUTPUT);
    
    // Desativa o motor inicialmente (LOW = habilitado, HIGH = desabilitado)
    digitalWrite(X_EN_PIN, HIGH);
    
    delay(2000); // Aguarda o host montar a porta USB
    
    // Configura a porta Serial2 (UART1) para conversar com o driver TMC2209
    Serial2.setRX(X_RX_PIN);
    Serial2.setTX(X_TX_PIN);
    Serial2.begin(115200);
    
    // Pequeno atraso para estabilização elétrica
    delay(1000);
    
    // Configuração do TMC2209 via UART
    driver.begin();
    driver.toff(5);         // Liga o driver
    driver.rms_current(800); // Define a corrente para 800mA (seguro para o NEMA 17 1A)
    driver.microsteps(16);   // 16 micropassos para suavidade
    driver.pwm_autoscale(true);
    driver.en_spreadCycle(false); // Força o modo StealthChop (super silencioso)
    
    // Habilita a saída de potência para as bobinas
    digitalWrite(X_EN_PIN, LOW);
    
    Serial.println("Miniola - Teste de Motor SKR Pico Inicializado.");
    Serial.println("Envie 'F' para rodar pra frente, 'R' para rodar pra trás, 'S' para parar.");
}

void loop() {
    // Lê os comandos via USB
    if (Serial.available() > 0) {
        char cmd = Serial.read();
        if (cmd == 'F' || cmd == 'f') {
            digitalWrite(X_DIR_PIN, HIGH);
            is_moving = true;
            Serial.println("Movendo: Frente");
        } 
        else if (cmd == 'R' || cmd == 'r') {
            digitalWrite(X_DIR_PIN, LOW);
            is_moving = true;
            Serial.println("Movendo: Reverso");
        } 
        else if (cmd == 'S' || cmd == 's') {
            is_moving = false;
            Serial.println("Movendo: Parado");
        }
    }

    // Gera os pulsos STEP se o motor estiver mandado se mover
    if (is_moving) {
        digitalWrite(X_STEP_PIN, HIGH);
        delayMicroseconds(10); // Pulso mínimo de trigger
        digitalWrite(X_STEP_PIN, LOW);
        delayMicroseconds(step_delay); // Intervalo entre os passos (velocidade)
    }
}
