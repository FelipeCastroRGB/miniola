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
            # Voltando para RGB24! Como cravamos o limite em 1500 Mbps abaixo,
            # vamos tentar deixar a Ximea entregar as cores lindas originais (com White Balance automático).
            self.cam.set_imgdataformat('XI_RGB24')
            
            # Limite de Banda Seguro
            # Subimos para 1500 Mbps (~187 MB/s) - Suporta os 90 FPS em RAW8 com folga, 
            # sem chegar perto dos 2800 Mbps que crasham o Raspberry Pi.
            try:
                self.cam.set_param('auto_bandwidth_calculation', 0)
            except: pass
            try:
                self.cam.set_limit_bandwidth(1500) 
            except: pass

            self.cam.set_exposure(shutter_speed) # em us
            self.cam.set_gain(gain) # em dB
            
            # Resolução nativa ou Crop
            try:
                self.cam.set_width(res_w)
                self.cam.set_height(res_h)
            except Exception as e:
                print(f"[WARN] Falha ao definir resolução Ximea {res_w}x{res_h}: {e}")

            # Tenta definir o FPS desejado. Se a matemática interna da Ximea rejeitar (ERROR 11),
            # deixamos em FREE_RUN (vai rodar no máximo que a banda de 1500 Mbps permitir)
            try:
                self.cam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FRAME_RATE')
                self.cam.set_framerate(fps)
            except Exception as e:
                print(f"[WARN] Falha ao definir Framerate ({fps} FPS), ativando FREE_RUN: {e}")
                try:
                    self.cam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FREE_RUN')
                except: pass

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
            self.cam.get_image(self.img, timeout=1000)
            data = self.img.get_image_data_numpy()
            
            # Se for RAW8 (matriz 2D), debayerizamos na CPU com OpenCV
            if len(data.shape) == 2:
                import cv2
                # A MQ042CG-CM geralmente usa padrão BGGR ou RGGB. O BG2BGR cobre a maioria dos casos.
                data = cv2.cvtColor(data, cv2.COLOR_BayerBG2BGR)
                
            return data
        except Exception as e:
            err_str = str(e)
            import time
            time.sleep(0.1)
            # Se a câmera morrer (Erro 49 = Desconectado), paramos de tentar ler para evitar o Spam
            if "49" in err_str or "disconnect" in err_str.lower():
                print("[ERRO FATAL] A câmera desconectou do barramento USB (Status 5 / Erro 49).")
                self.cam = None
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
