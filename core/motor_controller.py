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
        self.roller_diameter = 26.6
        self.roller_circumference = 3.14159 * self.roller_diameter
        # O Encoder E38S6G5 tem 600 PPR, mas o RP2040 lê em quadratura completa (4 bordas por pulso)
        self.encoder_ppr = 2400.0
        self.last_encoder_pulses = 0
        self.last_encoder_time = 0.0
        self.encoder_distance_accumulated = 0.0 # Distância percorrida desde a última perfuração vista
        
        # Ganhos do PID (Otimizados para suavidade extrema baseada em Feed-Forward)
        self.Kp = 0.1   # Quase zero! Ignora oscilações rápidas (Jitter da USB)
        self.Ki = 1.5   # Memória que corrige a variação lenta do diâmetro do rolo
        self.Kd = 0.0   # ZERO! A derivada com encoder via USB gera ruído brutal
        
        self.smoothed_adjustment = 0.0
        self.encoder_history = [] # Janela deslizante para cálculo de velocidade
        
        # PLL (Phase-Locked Loop)
        self.phase_error_mm = 0.0
        self.Kp_phase = 2.5 # Ganho de correção de fase (mm/s por mm de erro)
        
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

    # --- CONTROLE DE FOCO (Atuador Z) ---
    def focus_in(self, speed=500):
        self.send_command(f"Z {speed}")

    def focus_out(self, speed=500):
        self.send_command(f"Z -{speed}")

    def focus_stop(self):
        self.send_command("Z 0")

    # --- CONTROLE DE ILUMINAÇÃO (Painel LED) ---
    def set_led_brightness(self, level: int):
        level = max(0, min(255, int(level)))
        self.send_command(f"L {level}")

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

    def update_phase_error(self, error_px: float, pixels_per_mm: float):
        """
        Recebe o erro de fase (distância do furo detectado até a linha de gatilho).
        Atualiza o Phase-Locked Loop.
        """
        if pixels_per_mm > 0:
            with self.lock:
                self.phase_error_mm = error_px / pixels_per_mm
    # --- LOOP PID (Mola Matemática) ---
    def start_pid(self, target_fps=None):
        if not self.connected:
            return
        
        self.is_running_pid = True
        self.error_sum = 0.0
        self.last_error = 0.0
        
        # Usa o FPS passado (ex: da câmera) ou o default 18.0
        _fps = target_fps if target_fps is not None else self.target_fps
        # No 35mm, 1 Frame = 4 furos. Logo, a velocidade física (mm/s) tem que ser multiplicada por 4!
        self.target_mm_s = _fps * (self.pitch * 4) 
        
        self.ramped_target = 0.0 # Começa do zero (Soft Start)
        self.current_mm_s = 0.0
        self.smoothed_adjustment = 0.0
        
        self.last_sent_speed = None # Gatilho para o primeiro comando F
        self.last_encoder_time = time.time()
        self.pid_start_time = time.time() # Para calcular a curva S de aceleração
        self.last_encoder_pulses = 0
        self.encoder_history = []
        
        self.thread = threading.Thread(target=self._pid_loop, daemon=True)
        self.thread.start()

    def stop_pid(self):
        if self.is_running_pid:
            self.is_running_pid = False
            if hasattr(self, 'thread') and self.thread is not None:
                if threading.current_thread() != self.thread:
                    self.thread.join(timeout=0.5)
            self.send_command("S")

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
                    # Mantém apenas os últimos 500ms de histórico para uma janela mais estável
                    while len(self.encoder_history) > 1 and (now - self.encoder_history[0][0]) > 0.5:
                        self.encoder_history.pop(0)
                        
                    if len(self.encoder_history) >= 2:
                        old_time, old_pulses = self.encoder_history[0]
                        window_dt = now - old_time
                        if window_dt > 0.1: # Pelo menos 100ms para estabilidade
                            delta_p_win = latest_pulses - old_pulses
                            dist_win = (delta_p_win / self.encoder_ppr) * self.roller_circumference
                            measured_mm_s = abs(dist_win / window_dt)
                            # Suaviza a leitura de velocidade via EMA para mitigar o jitter da USB
                            if self.current_mm_s == 0.0:
                                self.current_mm_s = measured_mm_s
                            else:
                                self.current_mm_s = (self.current_mm_s * 0.7) + (measured_mm_s * 0.3)

                # Se passou muito tempo sem pulso novo, o filme parou
                if (time.time() - self.last_encoder_time) > 0.5:
                    self.current_mm_s = 0.0
                    self.encoder_history.clear()
                
                # --- Aceleração em Curva S (Smoothstep) ---
                # Garante que o filme arranque suavemente e atinja a velocidade final sem trancos
                tempo_decorrido = now - self.pid_start_time
                duracao_rampa = 3.0 # 3 Segundos para atingir velocidade final
                
                if tempo_decorrido < duracao_rampa:
                    t = tempo_decorrido / duracao_rampa
                    s_curve = t * t * (3.0 - 2.0 * t) # Fórmula matemática do Smoothstep
                    self.ramped_target = self.target_mm_s * s_curve
                else:
                    self.ramped_target = self.target_mm_s

                # Injeta a compensação do PLL (Phase-Locked Loop) se a rampa já completou a maior parte
                if tempo_decorrido > 1.0: # Dá 1 segundo pro motor estabilizar o arranque antes de plugar a fase
                    phase_correction = self.Kp_phase * self.phase_error_mm
                    # Limitar a correção de fase para não dar solavancos extremos
                    phase_correction = max(-self.target_mm_s * 0.2, min(self.target_mm_s * 0.2, phase_correction))
                    self.ramped_target += phase_correction
                
                error = self.ramped_target - self.current_mm_s
                
                self.error_sum += error * dt
                # Limite anti-windup (Aumentado absurdamente para suportar altas velocidades se o FF errar)
                self.error_sum = max(-15000, min(15000, self.error_sum))
                
                # FEED-FORWARD: Multiplicador ajustado para a velocidade real.
                # ~9000 Hz gera ~456 mm/s num núcleo médio de carretel. Multiplicador ~ 20.0
                feed_forward = self.ramped_target * 20.0
                
                # Equação PID baseada no erro de Velocidade Linear
                raw_adjustment = feed_forward + (self.Kp * error) + (self.Ki * self.error_sum)
                
                # Filtro na saída ultra pesado (90% do valor anterior) para planificar a curva
                self.smoothed_adjustment = (self.smoothed_adjustment * 0.9) + (raw_adjustment * 0.1)
                
                self.last_error = error
                self.last_pid_time = now
                # Calcula as novas velocidades
                new_speed_y = int(self.smoothed_adjustment)
                
                # O limite máximo subiu para 15000 Hz, pois 24fps reais exigem quase 10000 Hz no motor
                new_speed_y = max(100, min(15000, new_speed_y))
                
                # === SLIP DETECTION (E-STOP) ===
                # Se a velocidade exigida for alta (>2000Hz) mas o encoder estiver marcando
                # 0 de velocidade real por mais de 1.0 segundo contínuo, a fita arrebentou ou escorregou!
                if new_speed_y > 2000 and self.current_mm_s < 5.0:
                    if not hasattr(self, 'slip_timer'):
                        self.slip_timer = now
                    elif (now - self.slip_timer) > 1.0:
                        print(f"\n[E-STOP] ALARME CRITICO! Filme arrebentou ou patinou no encoder! Parada de Emergência acionada!\n")
                        self.send_command("S") # Manda comando absoluto de parada para a SKR
                        self.stop()
                        self.stop_pid()
                        return # Aborta a thread do PID imediatamente
                else:
                    self.slip_timer = now # Reseta o timer de segurança se tudo estiver normal
                # O comando "F" bloqueava a placa por 5ms (UART para o driverX). 
                # Agora usamos o comando "U" (Update) recém criado no C++ para setar o target de forma imediata!
                # O primeiro comando DEVE ser F para o firmware C++ ativar o driver e is_moving=true
                if self.last_sent_speed is None:
                    cmd = f"F {new_speed_y}"
                    self.send_command(cmd)
                    self.last_sent_speed = new_speed_y
                elif abs(new_speed_y - self.last_sent_speed) > 15:
                    cmd = f"U {new_speed_y}"
                    self.send_command(cmd)
                    self.last_sent_speed = new_speed_y
                    
                    # Print de telemetria apenas quando houver atualização real para a placa
                    print(f"[PID] Tgt: {self.ramped_target:.1f} | Cur: {self.current_mm_s:.1f} | Err: {error:.1f} | Spd_Y: {new_speed_y}")
            
            time.sleep(0.05) # 20Hz update rate para os motores
