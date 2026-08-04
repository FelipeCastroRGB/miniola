import serial
import threading
import time
import logging

class FilmTransportPID:
    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.connected = False
        
        # Parâmetros Mecânicos e Estado
        self.target_fps = 18.0
        self.current_fps = 0.0
        self.last_perf_time = 0.0
        
        # Ganhos do PID (Ajustáveis durante calibração)
        self.Kp = 50.0  # Mola: Ação proporcional ao erro
        self.Ki = 2.0   # Memória: Compensa diâmetro mudando lentamente
        self.Kd = 10.0  # Amortecedor: Evita solavancos
        
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

    # --- FEEDBACK DO OPENCV ---
    def notify_perforation(self):
        """Chamado pelo OpenCV toda vez que uma perfuração cruza a linha de gatilho"""
        now = time.perf_counter()
        with self.lock:
            if self.last_perf_time > 0:
                delta_t = now - self.last_perf_time
                if delta_t > 0:
                    inst_fps = 1.0 / delta_t
                    # Filtro passa-baixa simples para suavizar o FPS
                    self.current_fps = (self.current_fps * 0.7) + (inst_fps * 0.3)
            self.last_perf_time = now

    # --- LOOP PID (Mola Matemática) ---
    def start_pid(self):
        if not self.connected:
            return
        
        self.is_running_pid = True
        self.error_sum = 0.0
        self.last_error = 0.0
        self.current_fps = self.target_fps # Assume início perfeito
        self.last_perf_time = time.perf_counter()
        
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
                # Se passou muito tempo sem furo, o FPS zerou (filme parou ou enroscou)
                if (time.perf_counter() - self.last_perf_time) > 0.5:
                    self.current_fps = 0.0
                
                error = self.target_fps - self.current_fps
                
                self.error_sum += error * dt
                # Limite anti-windup
                self.error_sum = max(-1000, min(1000, self.error_sum))
                
                d_error = (error - self.last_error) / dt
                
                # Equação PID
                adjustment = (self.Kp * error) + (self.Ki * self.error_sum) + (self.Kd * d_error)
                
                self.last_error = error
                self.last_pid_time = now
                
                # Calcula as novas velocidades (Y puxa, X segura a tensão constante)
                new_speed_y = int(self.base_speed_y + adjustment)
                new_speed_x = self.base_speed_x # Feed-in fixo leve ou proporcional se necessário
                
                # Evita reversões acidentais no modo PID
                new_speed_y = max(100, new_speed_y)
                
                cmd = f"V {new_speed_x} {new_speed_y}"
                self.send_command(cmd)
                
            # Lê alertas da placa (como o Safety Stop do StallGuard)
            if self.serial.in_waiting > 0:
                line = self.serial.readline().decode('utf-8').strip()
                if "!STALL!" in line:
                    logging.critical(f"Emergência na placa SKR: {line}")
                    self.is_running_pid = False
            
            time.sleep(0.05) # 20Hz update rate para os motores
