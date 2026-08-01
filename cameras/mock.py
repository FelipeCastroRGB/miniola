import time
import platform
import numpy as np
import cv2
from .base import CameraProvider

class MockCameraProvider(CameraProvider):
    def __init__(self, video_path=None):
        arch = platform.machine().lower()
        if "arm" in arch or "aarch64" in arch:
            raise RuntimeError(f"O provedor mock não é suportado e está isolado de sistemas ARM64 ({arch}). Use-o apenas em x86_64.")
            
        self.video_path = video_path
        self.cap = None
        self.is_running = False
        self.width = 1920
        self.height = 1080
        self.fps = 120
        self.frame_time = 1.0 / self.fps
        self.last_frame_time = 0
        
        # Variáveis para simulação sintética
        self.perf_y = 0
        self.perf_height = 80
        self.perf_width = 100
        self.perf_gap = 200 # Distância entre perfurações
        
    def start(self, res_w, res_h, fps, shutter_speed, gain, lens_position, offset_x=0, offset_y=0):
        self.width = res_w
        self.height = res_h
        self.fps = fps if fps > 0 else 120
        self.frame_time = 1.0 / self.fps
        
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print(f"[Mock] Falha ao abrir {self.video_path}, caindo para modo sintético.")
                self.cap = None

        # OTIMIZAÇÃO DE PERFORMANCE: Preparar fundo base para evitar recálculo de numpy/desenhos a 120 FPS
        self.base_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.base_frame[:] = (30, 30, 30)
        self.film_x_start = self.width // 4
        self.film_x_end = self.width * 3 // 4
        cv2.rectangle(self.base_frame, (self.film_x_start, 0), (self.film_x_end, self.height), (10, 10, 10), -1)
        
        self.perf_x_left = self.film_x_start + 20
        self.perf_x_right = self.film_x_end - self.perf_width - 20
        
        self.audio_slit_x = self.perf_x_left + self.perf_width + 40
        self.audio_slit_w = 40
        
        # Buffer de ruído estendido para permitir rolar a janela (Slice) ao invés de recalcular
        self.base_noise = np.random.randint(50, 200, (self.height * 2, self.audio_slit_w, 3), dtype=np.uint8)

        self.is_running = True
        self.last_frame_time = time.time()
        print(f"[Mock] Câmera iniciada a {self.width}x{self.height} @ {self.fps} FPS")

    def get_frame(self):
        if not self.is_running:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Controle de tempo para manter o FPS
        now = time.time()
        elapsed = now - self.last_frame_time
        if elapsed < self.frame_time:
            time.sleep(self.frame_time - elapsed)
        self.last_frame_time = time.time()

        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                # Fazer loop no vídeo
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if frame is not None:
                # Opcional: redimensionar se necessário
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                return frame

        # Modo sintético: usa o base_frame pré-alocado
        frame = self.base_frame.copy()
        
        # Desenhar perfurações
        y = self.perf_y - self.perf_height
        while y < self.height:
            if y + self.perf_height > 0:
                cv2.rectangle(frame, (self.perf_x_left, int(y)), (self.perf_x_left + self.perf_width, int(y) + self.perf_height), (255, 255, 255), -1)
                cv2.rectangle(frame, (self.perf_x_right, int(y)), (self.perf_x_right + self.perf_width, int(y) + self.perf_height), (255, 255, 255), -1)
            y += self.perf_gap
            
        # Fenda de áudio: aplicar ruído copiando do buffer em vez de gerar na hora
        noise_offset_y = np.random.randint(0, self.height)
        frame[0:self.height, self.audio_slit_x:self.audio_slit_x+self.audio_slit_w] = self.base_noise[noise_offset_y:noise_offset_y+self.height, :]
        
        # Deslocar as perfurações para o próximo frame (simula avanço do filme)
        # Velocidade: 5 pixels por frame
        self.perf_y += 5
        if self.perf_y > self.perf_gap:
            self.perf_y -= self.perf_gap
            
        return frame

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        print("[Mock] Câmera parada")

    def set_exposure(self, value):
        pass # No-op para mock

    def set_gain(self, value):
        pass # No-op

    def set_fps(self, value):
        self.fps = value
        if self.fps > 0:
            self.frame_time = 1.0 / self.fps

    def set_focus(self, value):
        pass

    def autofocus_cycle(self):
        pass

    def capture_metadata(self):
        return {"mock": True, "fps": self.fps, "perf_y": self.perf_y}

    def set_white_balance(self, kr, kb):
        pass
