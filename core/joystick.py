import evdev
import threading
import logging
import time

class GamepadController:
    def __init__(self, motor_controller, on_rec_toggle=None):
        self.motor = motor_controller
        self.on_rec_toggle = on_rec_toggle
        self.device = None
        self.running = False
        self.thread = None
        
        self.max_speed = 3500 # Velocidade máxima no gatilho totalmente apertado
        
        # Estado atual para não mandar stop mil vezes no serial
        self.is_moving_fwd = False
        self.is_moving_rev = False
        self.last_fwd_speed = 0
        self.last_rev_speed = 0
        
    def find_gamepad(self):
        try:
            devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
            for device in devices:
                name = device.name.lower()
                # Procura nomes comuns em controles Knup / Genéricos
                if "joystick" in name or "gamepad" in name or "controller" in name or "knup" in name or "usb" in name:
                    if "keyboard" not in name and "mouse" not in name:
                        return device
        except Exception:
            pass
        return None

    def start(self):
        self.device = self.find_gamepad()
        if not self.device:
            print("[GAMEPAD] Nenhum controle encontrado. Modo terminal padrão.")
            return False
            
        print(f"[GAMEPAD] Controle conectado com SUCESSO: {self.device.name}")
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def _loop(self):
        try:
            for event in self.device.read_loop():
                if not self.running:
                    break
                
                # --- GATILHOS E MANCHES ANALÓGICOS (ABS_Y e ABS_RZ) ---
                if event.type == evdev.ecodes.EV_ABS:
                    
                    if event.code in [evdev.ecodes.ABS_Y, evdev.ecodes.ABS_RZ]:
                        val = event.value
                        
                        # Cria uma "Deadzone" ao redor do centro (115 a 140).
                        # Dentro dessa área, cravamos a velocidade perfeitamente em 50% (1750 Hz),
                        # garantindo estabilidade no modo Cruzeiro (ignora tremedeira e drift da mola).
                        if 115 <= val <= 140:
                            speed = int(self.max_speed * 0.5)
                        else:
                            # Fora do centro, o valor varia linearmente
                            # 0 (CIMA) = Vel Máx | 255 (BAIXO) = Quase parando
                            speed = int(((255 - val) / 255.0) * self.max_speed)
                        
                        
                        if speed < 150: # Se puxou todo pra baixo, freia o motor
                            if self.is_moving_fwd or self.is_moving_rev:
                                self.motor.stop()
                                self.is_moving_fwd = False
                                self.is_moving_rev = False
                                self.last_fwd_speed = 0
                                self.last_rev_speed = 0
                        else:
                            # ABS_Y puxa o filme pra frente, ABS_RZ puxa pra trás
                            if event.code == evdev.ecodes.ABS_Y:
                                if abs(speed - self.last_fwd_speed) > 100 or not self.is_moving_fwd:
                                    self.motor.manual_forward(max(500, speed))
                                    self.last_fwd_speed = speed
                                    self.is_moving_fwd = True
                                    self.is_moving_rev = False
                            elif event.code == evdev.ecodes.ABS_RZ:
                                if abs(speed - self.last_rev_speed) > 100 or not self.is_moving_rev:
                                    self.motor.manual_reverse(max(500, speed))
                                    self.last_rev_speed = speed
                                    self.is_moving_rev = True
                                    self.is_moving_fwd = False

                # --- BOTÕES DIGITAIS ---
                elif event.type == evdev.ecodes.EV_KEY:
                    
                    # BTN_PINKIE: Avançar Filme (Motor Receptor / Play)
                    if event.code == evdev.ecodes.BTN_PINKIE:
                        if event.value == 1: # Pressionado
                            self.motor.manual_forward(self.max_speed)
                            self.is_moving_fwd = True
                        elif event.value == 0: # Solto
                            self.motor.stop()
                            self.is_moving_fwd = False
                            
                    # BTN_TOP2: Rebobinar Filme (Motor Doador)
                    elif event.code == evdev.ecodes.BTN_TOP2:
                        if event.value == 1:
                            self.motor.manual_reverse(self.max_speed)
                            self.is_moving_rev = True
                        elif event.value == 0:
                            self.motor.stop()
                            self.is_moving_rev = False

                    # Start Button para REC/PID (Mantido padrão)
                    elif event.code in [evdev.ecodes.BTN_START, 297, 315] and event.value == 1:
                        if self.on_rec_toggle:
                            self.on_rec_toggle()

        except Exception as e:
            print(f"[GAMEPAD] Desconectado ou erro de leitura: {e}")
            self.running = False
