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
            # O RGB24 sufocou a CPU do Raspberry Pi (10 FPS). Voltamos para o formato RAW8 veloz!
            self.cam.set_imgdataformat('XI_RAW8')
            
            # ATIVANDO CORES NO MODO RAW!
            # Para não ficar desbotado/cinza, ligamos o White Balance direto no sensor de hardware.
            try:
                self.cam.set_param('auto_wb', 1) # Tenta ligar WB Automático
            except:
                try:
                    # Se falhar o automático, forçamos um ganho manual (Red e Blue)
                    self.cam.set_param('wb_kr', 2.0) # Booster de Vermelho
                    self.cam.set_param('wb_kb', 2.0) # Booster de Azul
                except: pass
            
            # Limite de Banda Extremo (2000 Mbps) para atingir os 160 FPS
            # Estamos tirando o limite de 1500 para permitir a taxa maciça de dados do RAW8.
            try:
                self.cam.set_param('auto_bandwidth_calculation', 0)
            except: pass
            try:
                self.cam.set_limit_bandwidth(2000) 
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
            # deixamos em FREE_RUN (vai rodar no máximo que a banda de 2000 Mbps permitir)
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
            # Retorna a matriz pura (RAW8 = 2D Array). 
            # NÃO FAZEMOS DEBAYER AQUI para salvar CPU e conseguir 160 FPS.
            return self.img.get_image_data_numpy()
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
