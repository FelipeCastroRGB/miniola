#include <Arduino.h>
#include <TMCStepper.h>

// --- PINOUT DA SKR PICO V1.0 (RP2040) ---
// Eixo X (Feed-in / Freio Sutil)
#define X_STEP_PIN 11
#define X_DIR_PIN 10
#define X_EN_PIN 12

// Eixo Y (Take-up / Puxador Principal)
#define Y_STEP_PIN 6
#define Y_DIR_PIN 5
#define Y_EN_PIN 7

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

// Velocidade Alvo e Atual (Hz) para Rampa de Aceleração
long target_speed_X = 0;
long target_speed_Y = 0;
long current_speed_X = 0;
long current_speed_Y = 0;
unsigned long last_accel_time = 0;

// Variáveis para controle não-bloqueante de passos (micros())
unsigned long last_step_time_X = 0;
unsigned long last_step_time_Y = 0;
unsigned long step_interval_X = 0;
unsigned long step_interval_Y = 0;

// StallGuard Threshold
const int SG_THRESHOLD_Y =
    50; // Limiar de tensão. Abaixo disso = filme travado. Ajustável depois.
unsigned long last_sg_check = 0;

void setup() {
  Serial.begin(115200);
  delay(2000); // Aguarda host montar a porta USB

  // Pinos de Direção e Step
  pinMode(X_STEP_PIN, OUTPUT);
  pinMode(X_DIR_PIN, OUTPUT);
  pinMode(X_EN_PIN, OUTPUT);
  pinMode(Y_STEP_PIN, OUTPUT);
  pinMode(Y_DIR_PIN, OUTPUT);
  pinMode(Y_EN_PIN, OUTPUT);

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
  driverX.rms_current(800,
                      0.0); // Aumentado para 200mA para garantir torque suave
  driverX.microsteps(16);
  driverX.pwm_autoscale(true);
  driverX.en_spreadCycle(false); // StealthChop

  // Configuração Driver Y (Take-up)
  driverY.begin();
  driverY.toff(5);
  driverY.rms_current(
      800, 0.0); // Aumentado para 250mA para puxar o filme sem tremer
  driverY.microsteps(16);
  driverY.pwm_autoscale(true);
  driverY.en_spreadCycle(false);

  // Configura o StallGuard no Motor Y
  driverY.TCOOLTHRS(0xFFFFF); // Habilita medição SG em baixa velocidade
  driverY.SGTHRS(50);         // Sensibilidade do StallGuard (0-255)

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
          target_speed_X = cmd.substring(2, space_idx).toInt();
          target_speed_Y = cmd.substring(space_idx + 1).toInt();
          digitalWrite(X_EN_PIN, LOW);
          digitalWrite(Y_EN_PIN, LOW);
          is_moving = true;
          manual_mode = false;
          safety_stop_triggered = false;
        }
      } else if (cmd.startsWith("F") || cmd.startsWith("f")) {
        int space_idx = cmd.indexOf(' ');
        target_speed_Y =
            (space_idx > 0) ? cmd.substring(space_idx + 1).toInt() : 2000;
        target_speed_X = 0;
        digitalWrite(X_EN_PIN, HIGH);
        digitalWrite(Y_EN_PIN, LOW);
        is_moving = true;
        manual_mode = true;
        safety_stop_triggered = false;
        Serial.println("Manobra: Frente (Y puxa)");
      } else if (cmd.startsWith("R") || cmd.startsWith("r")) {
        int space_idx = cmd.indexOf(' ');
        long spd =
            (space_idx > 0) ? cmd.substring(space_idx + 1).toInt() : 2000;
        target_speed_X = -spd;
        target_speed_Y = 0;
        digitalWrite(Y_EN_PIN, HIGH);
        digitalWrite(X_EN_PIN, LOW);
        is_moving = true;
        manual_mode = true;
        safety_stop_triggered = false;
        Serial.println("Manobra: Reverso (X puxa)");
      } else if (cmd == "S" || cmd == "s") {
        digitalWrite(X_EN_PIN, LOW);
        digitalWrite(Y_EN_PIN, LOW);
        target_speed_X = 0;
        target_speed_Y = 0;
        Serial.println("Manobra: Parada Iniciada");
      }
    } else {
      cmd_buffer += c;
    }
  }
}

void loop() {
  parse_serial_command();

  if (safety_stop_triggered)
    return;

  // --- LÓGICA DE RAMPA DE ACELERAÇÃO ---
  if (millis() - last_accel_time >= 10) {
    last_accel_time = millis();
    // Se estiver no PID, acelera rápido para não atrasar o controle matemático.
    // Se for manual, rampa suave de 0.5s.
    long accel_step =
        manual_mode ? 40 : 10000; // 40 Hz a cada 10ms = 2000 Hz em 0.5s

    // Eixo Y
    if (current_speed_Y < target_speed_Y) {
      current_speed_Y = min(current_speed_Y + accel_step, target_speed_Y);
    } else if (current_speed_Y > target_speed_Y) {
      current_speed_Y = max(current_speed_Y - accel_step, target_speed_Y);
    }

    // Eixo X
    if (current_speed_X < target_speed_X) {
      current_speed_X = min(current_speed_X + accel_step, target_speed_X);
    } else if (current_speed_X > target_speed_X) {
      current_speed_X = max(current_speed_X - accel_step, target_speed_X);
    }

    // Atualiza Direção fisicamente baseada no sinal da velocidade atual
    if (current_speed_X != 0)
      digitalWrite(X_DIR_PIN, (current_speed_X > 0) ? HIGH : LOW);
    if (current_speed_Y != 0)
      digitalWrite(Y_DIR_PIN, (current_speed_Y > 0) ? HIGH : LOW);

    // Recalcula os intervalos (Hz -> Microssegundos)
    step_interval_X =
        (current_speed_X != 0) ? 1000000UL / abs(current_speed_X) : 0;
    step_interval_Y =
        (current_speed_Y != 0) ? 1000000UL / abs(current_speed_Y) : 0;

    // Parada total
    if (current_speed_X == 0 && current_speed_Y == 0 && target_speed_X == 0 &&
        target_speed_Y == 0) {
      is_moving = false;
    }
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
