from .base import CameraProvider

class XimeaAdapter(CameraProvider):
    def __init__(self):
        self.cam = None
        try:
            from ximea import xiapi # type: ignore
            self.cam = xiapi.Camera()
        except ImportError:
            print("[WARN] ximea_api não está instalado. Modo de câmera Ximea inativo.")

    def start(self, res_w, res_h, fps, shutter_speed, gain, lens_position, offset_x=0, offset_y=0, color_mode='raw'):
        if not self.cam: return
        try:
            self.cam.open_device()
            self.color_mode = color_mode
            
            # Formato de Imagem (RAW8 para OpenCV Debayer, RGB24 para Ximea ISP)
            try:
                if self.color_mode == 'rgb':
                    self.cam.set_imgdataformat('XI_RGB24')
                    # Ativa AWB (Auto White Balance) apenas no modo RGB por padrão
                    self.cam.set_param('auto_wb', 1)
                else:
                    self.cam.set_imgdataformat('XI_RAW8')
            except Exception as e:
                print(f"[WARN] Falha ao definir modo {self.color_mode.upper()} na Ximea: {e}")
            
            # ATIVANDO CORES NO MODO RAW!
            # Para não ficar desbotado/cinza, ligamos o White Balance direto no sensor de hardware.
            try:
                self.cam.set_param('auto_wb', 0) # Desliga WB Automático para evitar Flicker de Cor
                self.cam.set_param('wb_kr', 1.5) # Booster de Vermelho (estimativa de filme incandescente/padrão)
                self.cam.set_param('wb_kg', 1.5) # Booster de Verde
                self.cam.set_param('wb_kb', 1.5) # Booster de Azul
                self.cam.set_param('gammaY', 1.0) # COMPRESSÃO DE SOMBRAS (Luminosity Gamma) -> Salva os tons escuros no RAW8!
            except: pass
            
            # Cálculo de Banda Automático via Hardware
            try:
                self.cam.set_param('auto_bandwidth_calculation', 1)
            except Exception:
                try: self.cam.set_limit_bandwidth(2200)
                except: pass
                
            # AUMENTO DO BUFFER:
            # O Python sofre pequenos "solavancos" de latência devido ao Garbage Collector e o GIL (Global Interpreter Lock).
            # Se a fila de buffers da câmera for muito pequena (ex: 4 frames), qualquer pausa de 30ms no Python a 120FPS fará a fila estourar!
            # Aumentar para 50-100 buffers na RAM absorve essas pausas tranquilamente.
            try:
                self.cam.set_param('acq_transport_buffer_size', 1048576 * 4) # Aumenta tamanho do pacote USB
                self.cam.set_param('buffers_queue_size', 50) 
            except Exception as e:
                print(f"[WARN] Não foi possível aumentar o buffers_queue_size: {e}")

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

            # Tenta definir o FPS desejado. Se rejeitado pelo cálculo da Ximea, mantém no modo Free Run automático.
            try:
                self.cam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FRAME_RATE')
                self.cam.set_framerate(fps)
            except Exception as e:
                print(f"[WARN] Falha ao definir Framerate fixo de {fps} FPS ({e}). Usando modo Free Run/Auto...")
                try:
                    self.cam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FREE_RUN')
                except: pass

            self.cam.start_acquisition()
            from ximea import xiapi
            self.img = xiapi.Image()
            device_name = "MQ042MG-CM"
            try:
                name_bytes = self.cam.get_device_name()
                device_name = name_bytes.decode('utf-8') if isinstance(name_bytes, bytes) else name_bytes
            except: pass
            print(f"[SISTEMA] Câmera Ximea {device_name} Inicializada com Sucesso.")
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
            self.cam.get_image(self.img, timeout=2000)
            
            # Checagem de Hardware de Drop Frames (Baseado no contador do Sensor)
            current_nframe = self.img.nframe
            if hasattr(self, 'last_nframe'):
                diff = current_nframe - self.last_nframe
                if diff > 1:
                    print(f"[ALERTA DE HARDWARE] DROP FRAME DETECTADO NA USB! Perdemos {diff - 1} frames entre o quadro {self.last_nframe} e {current_nframe}.")
            self.last_nframe = current_nframe
            
            # Retorna apenas uma VIEW do array em vez de clonar a memória (evita saturar o Garbage Collector do Python)
            # Como a Câmera Ximea reutiliza os buffers internos, isso reduzirá o tempo do loop principal.
            arr = self.img.get_image_data_numpy()
            
            # Garantir que o array seja estritamente 2D para que o len(shape) == 2 do debayer funcione
            if len(arr.shape) == 3 and arr.shape[2] == 1:
                arr = arr.squeeze(2)
            elif len(arr.shape) == 3 and arr.shape[2] > 1:
                if getattr(self, 'color_mode', 'raw') == 'raw':
                    arr = arr[:, :, 0]
                # Se for RGB, mantemos arr.shape == (H, W, 3) intacto.
                
            return arr
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

    def set_white_balance(self, kr: float, kg: float, kb: float):
        if self.cam:
            try:
                self.cam.set_param('auto_wb', 0)
                self.cam.set_param('wb_kr', kr)
                self.cam.set_param('wb_kg', kg)
                self.cam.set_param('wb_kb', kb)
                print(f"[XIMEA ISP] White Balance Manual Aplicado no Hardware: R={kr} G={kg} B={kb}")
            except Exception as e:
                print(f"[WARN] Falha ao ajustar White Balance na Ximea: {e}")

    def set_gamma(self, gamma_y: float, gamma_c: float):
        if self.cam:
            try:
                self.cam.set_param('gammaY', gamma_y)
                self.cam.set_param('gammaC', gamma_c)
                print(f"[XIMEA] Gamma Atualizado: Y={gamma_y} C={gamma_c}")
            except Exception as e:
                print(f"[WARN] Falha ao ajustar Gamma: {e}")

    def set_contrast(self, value: float):
        # A maioria das câmeras Ximea gerencia contraste linearmente via Gamma ou LUTs.
        # Caso exista suporte direto no modelo específico, aplicamos aqui.
        if self.cam:
            try:
                self.cam.set_param('contrast', value)
                print(f"[XIMEA] Contrast Atualizado: {value}")
            except Exception as e:
                print(f"[WARN] Câmera não suporta parâmetro de contraste direto: {e}")

    def set_sharpness(self, value: float):
        if self.cam:
            try:
                self.cam.set_param('sharpness', value)
                print(f"[XIMEA] Sharpness Atualizado: {value}")
            except Exception as e:
                print(f"[WARN] Câmera não suporta parâmetro de sharpness direto: {e}")

    def load_hardware_lut(self, lut_array):
        # A Ximea suporta LUT Enable, LUT Index e LUT Value
        if not self.cam:
            return False
            
        try:
            # O Erro 41 indica que não podemos alterar LUT_EN com a câmera rodando.
            # Pausamos a aquisição rapidamente, injetamos a curva, e voltamos a rodar.
            try:
                self.cam.stop_acquisition()
            except:
                pass
                
            self.cam.set_param('LUTEnable', 0)
            
            # lut_array no OpenCV tem shape (1, 256, 3).
            # Precisamos extrair uma curva 1D (usaremos o canal verde, índice 1)
            if len(lut_array.shape) == 3:
                curve = lut_array[0, :, 1]
            else:
                curve = lut_array
                
            for i in range(min(256, len(curve))):
                self.cam.set_param('LUTIndex', i)
                val = int(curve[i])
                self.cam.set_param('LUTValue', val)
                
            self.cam.set_param('LUTEnable', 1)
            
            try:
                self.cam.start_acquisition()
            except:
                pass
                
            print(f"[XIMEA] LUT de Hardware Carregado com Sucesso!")
            return True
        except Exception as e:
            print(f"[XIMEA] Falha ao injetar LUT no Hardware. Usando LUT via Python/OpenCV. Erro: {e}")
            try:
                self.cam.start_acquisition() # Garante que volte a rodar em caso de erro
            except:
                pass
            return False
