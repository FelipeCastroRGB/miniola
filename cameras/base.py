class CameraProvider:
    def start(self, res_w, res_h, fps, shutter_speed, gain, lens_position, offset_x=0, offset_y=0):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def get_frame(self):
        raise NotImplementedError

    def set_exposure(self, value):
        raise NotImplementedError

    def set_gain(self, value):
        raise NotImplementedError

    def set_fps(self, value):
        raise NotImplementedError

    def set_focus(self, value):
        raise NotImplementedError

    def autofocus_cycle(self):
        raise NotImplementedError

    def capture_metadata(self):
        raise NotImplementedError

    def set_white_balance(self, kr, kg, kb):
        raise NotImplementedError

    def set_gamma(self, gamma_y, gamma_c):
        raise NotImplementedError

    def set_contrast(self, value):
        raise NotImplementedError

    def set_sharpness(self, value):
        raise NotImplementedError
