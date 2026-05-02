from .base import CameraProvider
import time

class PiCameraAdapter(CameraProvider):
    def __init__(self):
        try:
            from picamera2 import Picamera2 # type: ignore
            self.picam2 = Picamera2()
        except ImportError:
            self.picam2 = None
            print("[WARN] picamera2 não está instalado. Modo de câmera Raspberry Pi inativo.")

    def start(self, res_w, res_h, fps, shutter_speed, gain, lens_position):
        if not self.picam2: return
        config = self.picam2.create_video_configuration(main={"size": (res_w, res_h), "format": "RGB888"})
        self.picam2.configure(config)
        self.picam2.set_controls({
            "ExposureTime": shutter_speed, 
            "AnalogueGain": gain, 
            "FrameRate": fps, 
            "LensPosition": lens_position,
            "ScalerCrop": (0, 0, 4608, 2592) # Trava o FOV Total
        })
        self.picam2.start()

    def stop(self):
        if self.picam2:
            self.picam2.stop()

    def get_frame(self):
        if not self.picam2: return None
        return self.picam2.capture_array()

    def set_exposure(self, value):
        if self.picam2:
            self.picam2.set_controls({"ExposureTime": value})

    def set_gain(self, value):
        if self.picam2:
            self.picam2.set_controls({"AnalogueGain": value})

    def set_fps(self, value):
        if self.picam2:
            self.picam2.set_controls({"FrameRate": value})

    def set_focus(self, value):
        if self.picam2:
            self.picam2.set_controls({"LensPosition": value})

    def autofocus_cycle(self):
        if not self.picam2: return False
        try:
            self.picam2.set_controls({"AfMode": 1, "AfRange": 2})
            time.sleep(0.5)
            self.picam2.autofocus_cycle()
            return True
        except Exception as e:
            print(f"[ÓPTICA] Erro no Autofoco nativo: {e}")
            self.picam2.set_controls({"AfMode": 0, "AfRange": 0})
            return False

    def capture_metadata(self):
        if not self.picam2: return {}
        return self.picam2.capture_metadata()
