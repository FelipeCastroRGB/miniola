from .base import CameraProvider

class XimeaAdapter(CameraProvider):
    def __init__(self):
        self.cam = None
        try:
            from ximea import xiapi # type: ignore
            self.cam = xiapi.Camera()
        except ImportError:
            print("[WARN] ximea_api não está instalado. Modo de câmera Ximea inativo.")

    def start(self, res_w, res_h, fps, shutter_speed, gain, lens_position, offset_x=0, offset_y=0):
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
                    self.cam.set_param('wb_kr', 1.0) # Booster de Vermelho
                    self.cam.set_param('wb_kb', 1.0) # Booster de Azul
                except: pass
            
            # Limite de Banda Extremo (2000 Mbps) para atingir os 160 FPS
            # Estamos tirando o limite de 1500 para permitir a taxa maciça de dados do RAW8.
            try:
                self.cam.set_param('auto_bandwidth_calculation', 0)
            except: pass
            try:
                self.cam.set_limit_bandwidth(2200) # 1800 é o limite físico seguro do chip VL805 do Pi4
            except: pass

            # Força o Global Shutter explicitamente para evitar distorções de movimento em alta velocidade
            try:
                self.cam.set_param('shutter_type', 0) # 0 = XI_SHUTTER_GLOBAL
            except Exception as e:
                pass

            self.cam.set_exposure(shutter_speed) # em us
            self.cam.set_gain(gain) # em dB
            
            # Resolução nativa ou Crop
            try:
                self.cam.set_width(res_w)
                self.cam.set_height(res_h)
                self.cam.set_offsetX(offset_x)
                self.cam.set_offsetY(offset_y)
            except Exception as e:
                print(f"[WARN] Falha ao definir Geometria Ximea (Res:{res_w}x{res_h} Offset:{offset_x},{offset_y}): {e}")

            # Tenta definir o FPS desejado. Se a matemática interna da Ximea rejeitar (ERROR 11),
            # nós deixamos no modo de Framerate (seguro) mas com o valor máximo possível suportado.
            try:
                self.cam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FRAME_RATE')
                self.cam.set_framerate(fps)
            except Exception as e:
                print(f"[WARN] Falha ao definir Framerate alvo de {fps} FPS: {e}")
                # NÃO ATIVAR FREE_RUN (Causa estouro de Banda e desliga a câmera no RPi4 com Status 5!)

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
            
            # Checagem de Hardware de Drop Frames (Baseado no contador do Sensor)
            current_nframe = self.img.nframe
            if hasattr(self, 'last_nframe'):
                diff = current_nframe - self.last_nframe
                if diff > 1:
                    print(f"[ALERTA DE HARDWARE] DROP FRAME DETECTADO NA USB! Perdemos {diff - 1} frames entre o quadro {self.last_nframe} e {current_nframe}.")
            self.last_nframe = current_nframe
            
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

    def set_white_balance(self, kr: float, kb: float):
        if self.cam:
            try:
                self.cam.set_param('auto_wb', 0)
                self.cam.set_param('wb_kr', kr)
                self.cam.set_param('wb_kb', kb)
                print(f"[XIMEA] White Balance Manual Aplicado: R={kr} B={kb}")
            except Exception as e:
                print(f"[WARN] Falha ao ajustar White Balance: {e}")
