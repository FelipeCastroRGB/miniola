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
        
        # Ganhos do PID (Restaurados após a remoção do ruído USB)
        self.Kp = 5.0   # Mola (Reação imediata)
        self.Ki = 1.0   # Memória (Busca o FPS exato mais rápido)
        self.Kd = 0.0   # ZERO! A derivada com encoder via USB gera ruído brutal (trancos)
        
        self.smoothed_adjustment = 0.0
        self.encoder_history = [] # Janela deslizante para cálculo de velocidade
        
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
        # No 35mm, 1 Frame = 4 furos. Logo, a velocidade física (mm/s) tem que ser multiplicada por 4!
        self.target_mm_s = self.target_fps * (self.pitch * 4) 
        self.ramped_target = 0.0 # Começa do zero (Soft Start)
        self.current_mm_s = 0.0
        self.smoothed_adjustment = 0.0
        self.last_encoder_time = time.time()
        self.last_encoder_pulses = 0
        self.encoder_history = []
        
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
                
                # Cálculo da Distância (Dead-Reckoning) a cada tick
                if latest_pulses is not None:
                    delta_p_tick = latest_pulses - self.last_encoder_pulses
                    dist_tick = (delta_p_tick / self.encoder_ppr) * self.roller_circumference
                    self.encoder_distance_accumulated += abs(dist_tick)
                    
                    self.last_encoder_pulses = latest_pulses
                    self.last_encoder_time = now
                    
                    # Janela Deslizante de Velocidade (Anti-Jitter da USB do Windows)
                    self.encoder_history.append((now, latest_pulses))
                    # Mantém apenas os últimos 300ms de histórico
                    while len(self.encoder_history) > 1 and (now - self.encoder_history[0][0]) > 0.3:
                        self.encoder_history.pop(0)
                        
                    if len(self.encoder_history) >= 2:
                        old_time, old_pulses = self.encoder_history[0]
                        window_dt = now - old_time
                        if window_dt > 0.1: # Pelo menos 100ms para estabilidade matemática
                            delta_p_win = latest_pulses - old_pulses
                            dist_win = (delta_p_win / self.encoder_ppr) * self.roller_circumference
                            self.current_mm_s = abs(dist_win / window_dt)

                # Se passou muito tempo sem pulso novo, o filme parou
                if (time.time() - self.last_encoder_time) > 0.5:
                    self.current_mm_s = 0.0
                    self.encoder_history.clear()
                
                # Soft Start mais agressivo: Sobe 20 mm/s a cada ciclo de 50ms para não demorar demais
                if self.ramped_target < self.target_mm_s:
                    self.ramped_target = min(self.target_mm_s, self.ramped_target + 20.0)
                
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
                
                # O limite máximo subiu para 15000 Hz, pois 24fps reais exigem quase 10000 Hz no motor
                new_speed_y = max(100, min(15000, new_speed_y))
                
                # Vamos usar o comando "F" (Forward Manual) da placa em vez de "V"!
                # Por que? O comando "V" liga o Motor X (Feed-in) que tem 800mA de força.
                # Se o Motor X estiver ligado a 200Hz, ele vira um freio de mão puxado
                # brigando contra o Y, causando os mini trancos terríveis (cogging).
                # O comando "F" desliga a energia do Motor X (Deixa em Banguela/Free-wheel)
                # permitindo que o Y puxe o filme liso!
                if not hasattr(self, 'last_sent_speed') or abs(new_speed_y - getattr(self, 'last_sent_speed', 0)) > 10:
                    cmd = f"F {new_speed_y}"
                    self.send_command(cmd)
                    self.last_sent_speed = new_speed_y
                    
                    # Print de telemetria apenas quando houver atualização real para a placa
                    print(f"[PID] Tgt: {self.ramped_target:.1f} | Cur: {self.current_mm_s:.1f} | Err: {error:.1f} | Spd_Y: {new_speed_y}")
            
            time.sleep(0.05) # 20Hz update rate para os motores
