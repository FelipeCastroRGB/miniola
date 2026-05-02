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
            try:
                try: self.cam.set_limit_bandwidth_mode('XI_OFF')
                except: pass
                try: self.cam.set_auto_bandwidth_calculation('XI_OFF')
                except: pass
                
                self.cam.set_limit_bandwidth(1000) # 1000 Mbps (~125 MB/s)
            except Exception as e:
                print(f"[WARN] Não foi possível limitar a banda: {e}")

            self.cam.set_exposure(shutter_speed) # em us
            self.cam.set_gain(gain) # em dB
            
            # Resolução
            try:
                self.cam.set_width(res_w)
                self.cam.set_height(res_h)
            except Exception as e:
                print(f"[WARN] Falha ao definir resolução Ximea {res_w}x{res_h}: {e}")

            # Deixamos a câmera em Free-Run (rodando o mais rápido possível dentro do limite de banda)
            # Impor um Framerate estava causando o ERROR 11 (Invalid Arguments)

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
