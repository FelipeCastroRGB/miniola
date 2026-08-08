import serial
import threading
import time
import logging

class FilmTransportPID:
    GAUGES = {
        '35mm': {'pitch': 4.75},
        '16mm': {'pitch': 7.62},
        '8mm': {'pitch': 3.81},
        'super8': {'pitch': 4.23}
    }

    def __init__(self, port='/dev/ttyACM0', baudrate=115200, gauge='35mm'):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.connected = False
        
        # Parâmetros Mecânicos e Estado
        self.gauge = gauge
        self.pitch = self.GAUGES.get(gauge, self.GAUGES['35mm'])['pitch']
        self.target_fps = 18.0
        self.target_mm_s = self.target_fps * self.pitch
        self.current_mm_s = 0.0
        
        # Setup do Encoder e Rolete
        self.roller_diameter = 30.5
        self.roller_circumference = 3.14159 * self.roller_diameter
        self.encoder_ppr = 600.0
        self.last_encoder_pulses = 0
        self.last_encoder_time = 0.0
        self.encoder_distance_accumulated = 0.0 # Distância percorrida desde a última perfuração vista
        
        # Ganhos do PID (Reduzidos para evitar trancos físicos - Soft Start)
        self.Kp = 5.0   # Mola bem macia
        self.Ki = 0.5   # Memória lenta
        self.Kd = 0.0   # ZERO! A derivada com encoder via USB gera ruído brutal (trancos)
        
        self.smoothed_adjustment = 0.0
        
        self.error_sum = 0.0
        self.last_error = 0.0
        self.last_pid_time = time.time()
        
        # Velocidade Base Inicial (Passos por segundo - Hz)
        self.base_speed_y = 1000 # Take-up puxa
        self.base_speed_x = 200  # Feed-in segura levemente (tensão passiva)
        
        self.is_running_pid = False
        self.thread = None
        self.lock = threading.Lock()

    def connect(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.connected = True
            logging.info(f"Conectado à SKR Pico em {self.port}")
            return True
        except serial.SerialException as e:
            logging.error(f"Erro ao conectar na placa SKR Pico: {e}")
            self.connected = False
            return False

    def disconnect(self):
        self.stop()
        if self.serial and self.serial.is_open:
            self.serial.close()
        self.connected = False

    def send_command(self, cmd: str):
        if self.connected and self.serial.is_open:
            try:
                self.serial.write((cmd + '\n').encode('utf-8'))
            except Exception as e:
                logging.error(f"Erro de escrita serial: {e}")

    # --- CONTROLES MANUAIS (Acionamento independente do PID) ---
    def manual_forward(self, speed=2000):
        self.stop_pid()
        self.send_command(f"F {speed}")

    def manual_reverse(self, speed=2000):
        self.stop_pid()
        self.send_command(f"R {speed}")

    def stop(self):
        self.stop_pid()
        self.send_command("S")

    # --- SINCRONIA ÓPTICA (Híbrida - SPEC-011) ---
    def sync_optical_phase(self):
        """
        Chamado pelo OpenCV quando uma perfuração válida cruza a linha de gatilho perfeitamente.
        Isso zera a distância mecânica acumulada, atrelando a fase física à fase óptica.
        """
        with self.lock:
            self.encoder_distance_accumulated = 0.0

    def get_accumulated_distance(self):
        """
        Retorna quantos milímetros o filme andou desde o último furo lido pela câmera.
        Usado pelo orquestrador para forçar a captura se o OpenCV falhar (Dead-Reckoning).
        """
        with self.lock:
            return self.encoder_distance_accumulated
    # --- LOOP PID (Mola Matemática) ---
    def start_pid(self):
        if not self.connected:
            return
        
        self.is_running_pid = True
        self.error_sum = 0.0
        self.last_error = 0.0
        self.target_mm_s = self.target_fps * self.pitch
        self.ramped_target = 0.0 # Começa do zero (Soft Start)
        self.current_mm_s = 0.0
        self.smoothed_adjustment = 0.0
        self.last_encoder_time = time.time()
        self.last_encoder_pulses = 0
        
        self.thread = threading.Thread(target=self._pid_loop, daemon=True)
        self.thread.start()

    def stop_pid(self):
        self.is_running_pid = False
        if self.thread:
            self.thread.join(timeout=0.5)
            self.thread = None

    def _pid_loop(self):
        while self.is_running_pid:
            now = time.time()
            dt = now - self.last_pid_time
            if dt <= 0:
                dt = 0.01
                
            with self.lock:
                # Lê TODAS as mensagens pendentes da placa
                latest_pulses = None
                while self.serial.in_waiting > 0:
                    try:
                        line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                        if "!STALL!" in line:
                            logging.critical(f"Emergência na placa SKR: {line}")
                            self.is_running_pid = False
                        elif line.startswith("E "):
                            latest_pulses = int(line.split(" ")[1])
                    except Exception as e:
                        pass
                
                # Só calcula a velocidade no final do lote, neutralizando o "USB Jitter" (lag)
                if latest_pulses is not None:
                    delta_p = latest_pulses - self.last_encoder_pulses
                    delta_t = now - self.last_encoder_time
                    
                    if delta_t > 0.01: # Evita divisão por zero ou spikes irreais
                        distance = (delta_p / self.encoder_ppr) * self.roller_circumference
                        self.encoder_distance_accumulated += abs(distance)
                        
                        inst_mm_s = abs(distance / delta_t)
                        # Filtro Passa-Baixa (80% passado, 20% atual) para suavizar a leitura
                        self.current_mm_s = (self.current_mm_s * 0.8) + (inst_mm_s * 0.2)
                        
                    self.last_encoder_pulses = latest_pulses
                    self.last_encoder_time = now

                # Se passou muito tempo sem pulso novo, o filme parou
                if (time.time() - self.last_encoder_time) > 0.5:
                    self.current_mm_s = 0.0
                
                # Soft Start: Rampa a velocidade alvo suavemente (15 mm/s a cada ciclo de 50ms)
                if self.ramped_target < self.target_mm_s:
                    self.ramped_target = min(self.target_mm_s, self.ramped_target + 3.0)
                
                error = self.ramped_target - self.current_mm_s
                
                self.error_sum += error * dt
                # Limite anti-windup
                self.error_sum = max(-1000, min(1000, self.error_sum))
                
                # Equação PID baseada no erro de Velocidade Linear (Kd removido)
                raw_adjustment = (self.Kp * error) + (self.Ki * self.error_sum)
                
                # Filtro na saída para que a placa SKR não receba degraus violentos a cada 50ms
                self.smoothed_adjustment = (self.smoothed_adjustment * 0.8) + (raw_adjustment * 0.2)
                
                self.last_error = error
                self.last_pid_time = now
                
                # Calcula as novas velocidades
                new_speed_y = int(self.base_speed_y + self.smoothed_adjustment)
                new_speed_x = self.base_speed_x 
                
                # Evita reversões acidentais e velocidades perigosas (>3000 Hz trava o motor)
                new_speed_y = max(100, min(2500, new_speed_y))
                
                # Só envia o comando para a SKR se a velocidade mudou mais de 5Hz (evita micro-agitações)
                if not hasattr(self, 'last_sent_speed') or abs(new_speed_y - self.last_sent_speed) > 5:
                    cmd = f"V {new_speed_x} {new_speed_y}"
                    self.send_command(cmd)
                    self.last_sent_speed = new_speed_y
            
            time.sleep(0.05) # 20Hz update rate para os motores
