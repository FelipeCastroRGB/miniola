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
            self.cam.set_exposure(shutter_speed) # em us
            self.cam.set_gain(gain) # em dB
            
            # Controle de FPS
            self.cam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FRAME_RATE')
            self.cam.set_framerate(fps)

            # Resolução pode ser travada no hardware/ROI
            # Algumas câmeras requerem incrementos específicos, então try/except é prudente
            try:
                self.cam.set_width(res_w)
                self.cam.set_height(res_h)
            except Exception as e:
                print(f"[WARN] Falha ao definir resolução Ximea {res_w}x{res_h}: {e}")

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
            self.cam.get_image(self.img)
            return self.img.get_image_data_numpy()
        except:
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
