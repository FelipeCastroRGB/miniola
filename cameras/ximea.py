from .base import CameraProvider

class XimeaAdapter(CameraProvider):
    def __init__(self):
        self.cam = None
        try:
            from ximea import xiapi # type: ignore
            self.cam = xiapi.Camera()
        except ImportError:
            print("[WARN] ximea_api não está instalado. Modo de câmera Ximea inativo.")

    def start(self, res_w, res_h, fps, shutter_speed, gain, lens_position):
        if not self.cam: return
        try:
            self.cam.open_device()
            # Ximea config:
            self.cam.set_imgdataformat('XI_RGB24')
            
            # Limite de Banda Seguro para o Raspberry Pi 4
            # (Evita o "failed with status 5" / "Camera has been reset" por excesso de tráfego/energia)
            try:
                self.cam.set_limit_bandwidth(1200) # 1200 Mbps (~150 MB/s) - Suficiente para 1536x864@90fps
            except Exception as e:
                print(f"[WARN] Não foi possível limitar a banda: {e}")

            self.cam.set_exposure(shutter_speed) # em us
            self.cam.set_gain(gain) # em dB
            
            # Resolução pode ser travada no hardware/ROI ANTES do FPS!
            # (Limitar o frame primeiro impede erro caso o FPS desejado só seja possível no crop)
            try:
                self.cam.set_width(res_w)
                self.cam.set_height(res_h)
            except Exception as e:
                print(f"[WARN] Falha ao definir resolução Ximea {res_w}x{res_h}: {e}")

            # Controle de FPS
            try:
                self.cam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FRAME_RATE')
                self.cam.set_framerate(fps)
            except Exception as e:
                print(f"[WARN] Falha ao definir Framerate Ximea ({fps} FPS): {e}")

            self.cam.start_acquisition()
            from ximea import xiapi
            self.img = xiapi.Image()
            print("[SISTEMA] Câmera Ximea MQ042MG-CM Inicializada com Sucesso.")
        except Exception as e:
            print(f"[ERRO] Falha ao iniciar Ximea: {e}")
            self.cam = None

    def stop(self):
        if self.cam:
            try:
                self.cam.stop_acquisition()
                self.cam.close_device()
            except: pass

    def get_frame(self):
        if not self.cam: return None
        try:
            # Timeout explícito de 1000ms (evita que o loop do Python quebre imediatamente)
            self.cam.get_image(self.img, timeout=1000)
            return self.img.get_image_data_numpy()
        except Exception as e:
            # O Erro 45 (Timeout) significa que o frame não chegou pelo cabo USB.
            # Um pequeno sleep evita que o log do terminal seja "spammado" infinitamente.
            import time
            time.sleep(0.1)
            return None

    def set_exposure(self, value):
        if self.cam:
            try: self.cam.set_exposure(value)
            except: pass

    def set_gain(self, value):
        if self.cam:
            try: self.cam.set_gain(value)
            except: pass

    def set_fps(self, value):
        if self.cam:
            try: self.cam.set_framerate(value)
            except: pass

    def set_focus(self, value):
        # Lente C-Mount manual, não fazemos nada
        pass

    def autofocus_cycle(self):
        # Não aplicável
        return False

    def capture_metadata(self):
        # Dummy para painel não quebrar
        return {}
