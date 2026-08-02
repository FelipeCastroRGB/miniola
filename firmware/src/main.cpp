#include <Arduino.h>
#include <TMCStepper.h>

// --- PINOUT DA SKR PICO V1.0 (RP2040) ---
// Eixo X (Feed-in / Freio Sutil)
#define X_STEP_PIN 11
#define X_DIR_PIN  10
#define X_EN_PIN   12

// Eixo Y (Take-up / Puxador Principal)
#define Y_STEP_PIN 6
#define Y_DIR_PIN  5
#define Y_EN_PIN   7

// UART Compartilhada dos Drivers (Serial2 no Arduino-Pico)
#define UART_RX_PIN 9
#define UART_TX_PIN 8

// Endereços UART (Na SKR Pico V1.0, X=0, Z=1, Y=2, E0=3)
#define DRIVER_ADDRESS_X 0b00 // 0
#define DRIVER_ADDRESS_Y 0b10 // 2 (ESTE ERA O BUG!)
#define R_SENSE 0.11f

// Instâncias dos Drivers
TMC2209Stepper driverX(&Serial2, R_SENSE, DRIVER_ADDRESS_X);
TMC2209Stepper driverY(&Serial2, R_SENSE, DRIVER_ADDRESS_Y);

// --- VARIÁVEIS DE ESTADO E CONTROLE ---
bool is_moving = false;
bool manual_mode = false;
bool safety_stop_triggered = false;

// Velocidade em Passos por Segundo (Hz). Valores negativos invertem a direção.
long speed_X = 0; 
long speed_Y = 0;

// Variáveis para controle não-bloqueante de passos (micros())
unsigned long last_step_time_X = 0;
unsigned long last_step_time_Y = 0;
unsigned long step_interval_X = 0; 
unsigned long step_interval_Y = 0;

// StallGuard Threshold
const int SG_THRESHOLD_Y = 50; // Limiar de tensão. Abaixo disso = filme travado. Ajustável depois.
unsigned long last_sg_check = 0;

void setup() {
    Serial.begin(115200);
    delay(2000); // Aguarda host montar a porta USB

    // Pinos de Direção e Step
    pinMode(X_STEP_PIN, OUTPUT); pinMode(X_DIR_PIN, OUTPUT); pinMode(X_EN_PIN, OUTPUT);
    pinMode(Y_STEP_PIN, OUTPUT); pinMode(Y_DIR_PIN, OUTPUT); pinMode(Y_EN_PIN, OUTPUT);

    // Inicia motores desligados
    digitalWrite(X_EN_PIN, HIGH);
    digitalWrite(Y_EN_PIN, HIGH);

    // Inicializa a UART (Serial2)
    Serial2.setRX(UART_RX_PIN);
    Serial2.setTX(UART_TX_PIN);
    Serial2.begin(115200);
    delay(500);

    // Configuração Driver X (Feed-in)
    driverX.begin();
    driverX.toff(5);
    driverX.rms_current(100, 0.0); // 100mA. 0 de corrente em repouso.
    driverX.microsteps(16);
    driverX.pwm_autoscale(true);
    driverX.en_spreadCycle(false); // StealthChop

    // Configuração Driver Y (Take-up)
    driverY.begin();
    driverY.toff(5);
    // Modo Pancake Frio: 100mA rodando. 0.0 em repouso = ZERO calor (desligado eletricamente ao parar).
    driverY.rms_current(100, 0.0); 
    driverY.microsteps(16);
    driverY.pwm_autoscale(true);
    driverY.en_spreadCycle(false);
    
    // Configura o StallGuard no Motor Y
    driverY.TCOOLTHRS(0xFFFFF); // Habilita medição SG em baixa velocidade
    driverY.SGTHRS(50); // Sensibilidade do StallGuard (0-255)

    // Liga as saídas de potência
    digitalWrite(X_EN_PIN, LOW);
    digitalWrite(Y_EN_PIN, LOW);

    Serial.println("Miniola Dual-Motor Firmware Iniciado (PID Ready)");
}

void parse_serial_command() {
    static String cmd_buffer = "";
    while (Serial.available() > 0) {
        char c = Serial.read();
        if (c == '\n') {
            String cmd = cmd_buffer;
            cmd_buffer = "";
            cmd.trim();

            if (cmd.startsWith("V ")) {
                int space_idx = cmd.indexOf(' ', 2);
                if (space_idx > 0) {
                    speed_X = cmd.substring(2, space_idx).toInt();
                    speed_Y = cmd.substring(space_idx + 1).toInt();
                    step_interval_X = (speed_X != 0) ? 1000000UL / abs(speed_X) : 0;
                    step_interval_Y = (speed_Y != 0) ? 1000000UL / abs(speed_Y) : 0;
                    digitalWrite(X_EN_PIN, LOW);
                    digitalWrite(Y_EN_PIN, LOW);
                    digitalWrite(X_DIR_PIN, (speed_X > 0) ? HIGH : LOW);
                    digitalWrite(Y_DIR_PIN, (speed_Y > 0) ? HIGH : LOW);
                    is_moving = true;
                    manual_mode = false;
                    safety_stop_triggered = false;
                }
            } 
            else if (cmd == "F" || cmd == "f") {
                speed_X = 0; 
                speed_Y = 2000;
                step_interval_X = 0;
                step_interval_Y = 1000000UL / speed_Y;
                digitalWrite(Y_DIR_PIN, HIGH);
                digitalWrite(X_EN_PIN, HIGH); 
                digitalWrite(Y_EN_PIN, LOW);  
                is_moving = true;
                manual_mode = true;
                safety_stop_triggered = false;
                Serial.println("Manobra: Frente (Y puxa)");
            } 
            else if (cmd == "R" || cmd == "r") {
                speed_X = -2000;
                speed_Y = 0;
                step_interval_X = 1000000UL / abs(speed_X);
                step_interval_Y = 0;
                digitalWrite(X_DIR_PIN, LOW);
                digitalWrite(Y_EN_PIN, HIGH); 
                digitalWrite(X_EN_PIN, LOW);  
                is_moving = true;
                manual_mode = true;
                safety_stop_triggered = false;
                Serial.println("Manobra: Reverso (X puxa)");
            } 
            else if (cmd == "S" || cmd == "s") {
                digitalWrite(X_EN_PIN, LOW);
                digitalWrite(Y_EN_PIN, LOW);
                speed_X = 0;
                speed_Y = 0;
                step_interval_X = 0;
                step_interval_Y = 0;
                is_moving = false;
                Serial.println("Manobra: Parado");
            }
        } else {
            cmd_buffer += c;
        }
    }
}

void loop() {
    parse_serial_command();

    if (safety_stop_triggered) {
        return; // Trava tudo até receber novo comando de parada/reset
    }

    unsigned long current_micros = micros();

    // Pulso Motor X
    if (is_moving && step_interval_X > 0) {
        if (current_micros - last_step_time_X >= step_interval_X) {
            last_step_time_X = current_micros;
            digitalWrite(X_STEP_PIN, HIGH);
            delayMicroseconds(2);
            digitalWrite(X_STEP_PIN, LOW);
        }
    }

    // Pulso Motor Y
    if (is_moving && step_interval_Y > 0) {
        if (current_micros - last_step_time_Y >= step_interval_Y) {
            last_step_time_Y = current_micros;
            digitalWrite(Y_STEP_PIN, HIGH);
            delayMicroseconds(2);
            digitalWrite(Y_STEP_PIN, LOW);
        }
    }

    // Temporariamente desativado: O StallGuard precisa de calibração fina
    // baseada na nova corrente (100mA). Estava disparando falsos positivos.
    /*
    if (is_moving && (millis() - last_sg_check > 50)) {
        last_sg_check = millis();
        uint16_t sg_result = driverY.SG_RESULT();
        if (sg_result > 0 && sg_result < SG_THRESHOLD_Y) {
            is_moving = false;
            safety_stop_triggered = true;
            speed_X = 0; speed_Y = 0;
            Serial.print("!STALL! EMERGENCIA Y. SG=");
            Serial.println(sg_result);
        }
    }
    */
}
