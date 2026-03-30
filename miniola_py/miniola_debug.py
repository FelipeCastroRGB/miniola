import sys 
from unittest.mock import MagicMock

# --- CONFIGURAÇÃO DE AMBIENTE E HARDWARE ---
sys.modules["pykms"] = MagicMock()
sys.modules["kms"] = MagicMock()

from flask import Flask, Response, request 
from picamera2 import Picamera2 
import cv2 
import numpy as np 
import threading 
import multiprocessing as mp 
import time 
import logging 
import shutil
import os

app = Flask(__name__) # Flask para o Dashboard (Roda no Core 0)
log = logging.getLogger('werkzeug') # Desativa os logs de requisição do Flask para não poluir o console
log.setLevel(logging.ERROR) 

CAPTURE_PATH = "capturas"
if not os.path.exists(CAPTURE_PATH): os.makedirs(CAPTURE_PATH)

picam2 = Picamera2()
shutter_speed, gain, fps_cam = 600, 1.0, 70
foco_atual, passo_foco = 14.5, 0.5
config = picam2.create_video_configuration(main={"size": (1080, 720), "format": "RGB888"})
picam2.configure(config)
picam2.set_controls({"ExposureTime": shutter_speed, "AnalogueGain": gain, "FrameRate": fps_cam, "LensPosition": foco_atual})
picam2.start()

# --- GEOMETRIA DO ROI E ESTADO ---
GRAVANDO = False
CALIBRANDO = False           # Trava de segurança da tela
ROI_X, ROI_Y = 25, 10
ROI_W, ROI_H = 80, 700
# --- LÓGICA DE GATILHO SIMPLIFICADA ---
LINHA_GATILHO_Y = 110  # Posição Y relativa DENTRO da ROI
MARGEM_GATILHO = 23    # Margem de disparo (px para cima e para baixo)
THRESH_VAL = 239 # Valor do threshold para binarização
MODO_DETECCAO = '2D'  # Inicia com o findContours que já está funcionando
PITCH_PADRAO_PX = 85.0  # CALIBRE AQUI: Quantos pixels tem o pitch de um filme NOVO na sua lente?
# --- PARÂMETROS DO CROP ---
OFFSET_X = 470 
CROP_W, CROP_H = 918, 612 

contador_perfs_ciclo = 0
frame_count = 0
perfuracao_na_linha = False
ultimo_frame_bruto = None
ultimo_frame_binario = None
ultimo_crop_preview = np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
lista_contornos_debug = []
fps_real_proc = 0.0
tempo_ms_ciclo = 0.0
encolhimento_atual_pct = 0.0

# --- FILA DE MULTIPROCESSAMENTO ---
# Fila compartilhada entre o Scanner (Thread) e o Gravador (Processo)
fila_gravacao = mp.Queue(maxsize=30) 

def processo_escrita_disco(fila_in):
    """ Este processo roda isolado em outro núcleo da CPU para não travar o Scanner """
    print("[SISTEMA] Processo de gravação (Núcleo Isolado) iniciado.")
    while True:
        item = fila_in.get()
        if item is None: break
        
        img_rgb, filename = item
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

def processar_captura(frame, cx_global, cy_global, n_frame):
    global OFFSET_X, CROP_W, CROP_H, ultimo_crop_preview, GRAVANDO
    
    # Cálculo rápido de coordenadas usando o centro global calculado
    fx, fy = cx_global + OFFSET_X, cy_global
    x1, y1 = max(0, int(fx - (CROP_W // 2))), max(0, int(fy - (CROP_H // 2)))
    x2, y2 = min(frame.shape[1], x1 + CROP_W), min(frame.shape[0], y1 + CROP_H)
    
    crop = frame[y1:y2, x1:x2]
    
    if crop.size > 0:
        ultimo_crop_preview = crop
        if GRAVANDO:
            filename = f"{CAPTURE_PATH}/miniola_{n_frame:06d}.jpg"
            try:
                # Envia para o processo isolado
                fila_gravacao.put((crop.copy(), filename), block=False)
            except:
                pass # Fila cheia, pula o frame silenciosamente para não travar

# --- PAINEL DE CONTROLE ---
def painel_controle():
    global frame_count, GRAVANDO, LINHA_GATILHO_Y, MARGEM_GATILHO, ROI_X, CROP_H, CROP_W, ROI_Y, ROI_W, ROI_H, THRESH_VAL
    global foco_atual, passo_foco, shutter_speed, gain, fps_cam, OFFSET_X, contador_perfs_ciclo, CALIBRANDO
    time.sleep(2)
    print("\n" + "═"*45)
    print("   MINIOLA - PAINEL DE CONTROLE")
    print("═"*45)
    print("   GATILHO:   ly (Linha na ROI)| mg (Margem)")
    print("   SISTEMA:   rec (Gravar)| r (Reset Tudo)| rc (Realinhar Ciclo)")
    print("   ÓPTICA:    k/l (Foco Manual)| j [val] (Passo)| af (Auto Foco)")
    print("   EXPOSIÇÃO: e [val] (Shutter Speed)| g [val] (Gain)| fps [val] (Frame Rate)")
    print("   CROP:   ch (Altura)| cw (Largura)")
    print("   SISTEMA:   rec (Gravar)| r (Reset)| rc (Realinhar) | md (Modos) | off (Desligar) | cal (Calibrar)")
    print("   TRESHOLD:   t")
    print("   ROI: w, a, s, d (Move ROI)| rx, ry, rw, rh [val] (Ajuste direto da ROI)")
    print("═"*45)
    while True:
        try:
            entrada = input("\n>> ").split()
            if not entrada: continue
            cmd = entrada[0].lower()
            val = float(entrada[1]) if len(entrada) > 1 else 0
            
            if cmd == 'w': ROI_Y = max(0, ROI_Y - 5)
            elif cmd == 's': ROI_Y = min(720 - ROI_H, ROI_Y + 5)
            elif cmd == 'a': ROI_X = max(0, ROI_X - 5)
            elif cmd == 'd': ROI_X = min(1080 - ROI_W, ROI_X + 5)
            elif cmd == 'rx': ROI_X = int(val)
            elif cmd == 'ry': ROI_Y = int(val)
            elif cmd == 'rw': ROI_W = int(val)
            elif cmd == 'rh': ROI_H = int(val)
            elif cmd == 'ch': CROP_H = int(val)
            elif cmd == 'cw': CROP_W = int(val)
            elif cmd == 'ly': 
                LINHA_GATILHO_Y = int(val)
                print(f"[GATILHO] Linha ajustada para: {LINHA_GATILHO_Y}px dentro da ROI")
            elif cmd == 'mg':
                MARGEM_GATILHO = int(val)
                print(f"[GATILHO] Margem ajustada para: +-{MARGEM_GATILHO}px")
            elif cmd == 'ox': OFFSET_X = int(val)
            elif cmd == 'l':
                foco_atual = round(foco_atual + passo_foco, 2)
                picam2.set_controls({"LensPosition": foco_atual})
            elif cmd == 'k':
                foco_atual = max(0.0, round(foco_atual - passo_foco, 2))
                picam2.set_controls({"LensPosition": foco_atual})
            # --- NOVO: AUTOFOCO DE TIRO ÚNICO (MODO MACRO / FULL RANGE) ---
            elif cmd == 'af':
                print("[ÓPTICA] Destravando lente e ativando Modo Macro... (Aguarde)")
                try:
                    # 1. AfMode: 1 (Auto). 
                    # AfRange: 2 (Full). Isso diz ao algoritmo para ignorar o limite de 12.0 
                    # e empurrar o motor até o limite físico do Macro (perto de 15.0).
                    picam2.set_controls({"AfMode": 1, "AfRange": 2})
                    
                    # Dá 0.5 segundos para o ISP processar e energizar o motor
                    time.sleep(0.5) 
                    
                    print("[ÓPTICA] Iniciando varredura profunda...")
                    picam2.autofocus_cycle()
                    
                    # Lê os metadados do exato momento pós-foco
                    metadados = picam2.capture_metadata()
                    
                    if "LensPosition" in metadados:
                        foco_atual = round(metadados["LensPosition"], 2)
                        
                        # 4. Tranca a lente de volta no modo Manual (0) e ancora na nova posição.
                        # (O AfRange volta para 0 por segurança, já que estamos em manual).
                        picam2.set_controls({"AfMode": 0, "LensPosition": foco_atual, "AfRange": 0})
                        
                        print(f"[ÓPTICA] Sucesso! Foco Macro cravado em: {foco_atual}")
                    else:
                        print("[ÓPTICA] Varredura concluída, mas posição não relatada pelo sensor.")
                        picam2.set_controls({"AfMode": 0, "AfRange": 0}) 
                        
                except Exception as e:
                    print(f"[ÓPTICA] Erro no Autofoco nativo: {e}")
                    picam2.set_controls({"AfMode": 0, "AfRange": 0}) 
            # -----------------------------------------------------------------
            elif cmd == 'e': 
                shutter_speed = int(val); picam2.set_controls({"ExposureTime": shutter_speed})
            elif cmd == 'cal':
                CALIBRANDO = True
                print("[SISTEMA] MODO DE CALIBRAÇÃO ATIVADO!")
                print("Vá para o navegador, clique no Live View e arraste para desenhar a linha do Pitch.")
            elif cmd == 'off':
                print("[SISTEMA] Encerrando processos e desligando a Raspberry Pi de forma segura...")
                time.sleep(1)
                os.system("sudo poweroff")
            elif cmd == 'g': 
                gain = val; picam2.set_controls({"AnalogueGain": gain})
            elif cmd == 'fps':
                fps_cam = int(val); picam2.set_controls({"FrameRate": fps_cam})
            elif cmd == 't': THRESH_VAL = int(val)
            elif cmd == 'rec': GRAVANDO = not GRAVANDO
            elif cmd == 'rc': 
                contador_perfs_ciclo = 0
                print("[SISTEMA] Fase realinhada! Ciclo forçado para 0/4.")
            elif cmd == 'md':
                global MODO_DETECCAO
                if MODO_DETECCAO == '2D': MODO_DETECCAO = '1D'
                elif MODO_DETECCAO == '1D': MODO_DETECCAO = 'MIX'
                else: MODO_DETECCAO = '2D'
                print(f"[SISTEMA] Motor de Visão alterado para: {MODO_DETECCAO}")
            elif cmd == 'r': 
                frame_count = 0
                for f in os.listdir(CAPTURE_PATH): os.remove(os.path.join(CAPTURE_PATH, f))
                print("RAM DRIVE LIMPO.")
        except Exception as e: print(f"Erro: {e}")

# --- LÓGICA DO SCANNER OTIMIZADA (MOTORES ISOLADOS) ---
def logica_scanner():
    cap_array = picam2.capture_array
    cv_cvt = cv2.cvtColor
    cv_resize = cv2.resize
    cv_thresh = cv2.threshold
    cv_find = cv2.findContours
    get_time = time.perf_counter
    
    global frame_count, ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug
    global contador_perfs_ciclo, perfuracao_na_linha, fps_real_proc, tempo_ms_ciclo
    global MODO_DETECCAO, encolhimento_atual_pct, PITCH_PADRAO_PX

    ESCALA_CV = 0.5 
    skip_ui = 0

    while True:
        t_inicio = get_time()
        
        frame_raw = cap_array()
        if frame_raw is None: continue
        
        lx, ly, lw, lh = ROI_X, ROI_Y, ROI_W, ROI_H
        roi_color = frame_raw[ly:ly+lh, lx:lx+lw]
        
        roi_gray = cv_cvt(roi_color, cv2.COLOR_RGB2GRAY)
        roi_small = cv_resize(roi_gray, (0, 0), fx=ESCALA_CV, fy=ESCALA_CV) 

        _, binary_small = cv_thresh(roi_small, THRESH_VAL, 255, cv2.THRESH_BINARY) 
        
        # Variáveis limpas para o quadro atual
        perfs_neste_frame = []
        debug_visual = []
        furo_detectado_agora = False

        # ==========================================================
        # MOTOR 1: 2D - FINDCONTOURS (VELOCIDADE + ESTABILIDADE MULTI-FURO)
        # ==========================================================
        if MODO_DETECCAO == '2D':
            # RETR_LIST continua aqui para garantir FPS alto
            contours, _ = cv_find(binary_small, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            limite_superior = LINHA_GATILHO_Y - MARGEM_GATILHO
            limite_inferior = LINHA_GATILHO_Y + MARGEM_GATILHO
            
            # Voltamos a guardar todos os furos válidos do frame para fazer a média
            furos_validos = []
            
            for cnt in contours:
                x_s, y_s, w_s, h_s = cv2.boundingRect(cnt)
                
                # Cálculo rápido de área para não pesar a CPU
                area_aprox = (w_s * h_s) * 4 
                
                if 200 < area_aprox < 10000 and 0.2 < (w_s / h_s) < 2.5:
                    cy_roi = (y_s * 2) + ((h_s * 2) // 2)
                    cx_global = (x_s * 2) + (w_s * 2 // 2) + lx
                    cy_global = cy_roi + ly
                    
                    acionou = limite_superior <= cy_roi <= limite_inferior
                    cor = (0, 0, 255) if acionou else (0, 255, 0)
                    
                    furos_validos.append({
                        'cy_roi': cy_roi, 
                        'cx_g': cx_global, 
                        'cy_g': cy_global, 
                        'acionou': acionou
                    })
                    
                    debug_visual.append({'rect': (x_s*2+lx, y_s*2+ly, w_s*2, h_s*2), 'color': cor})

            # Ordena os furos de cima para baixo
            furos_validos.sort(key=lambda p: p['cy_roi'])
            
            if furos_validos and furos_validos[0]['acionou']:
                furo_detectado_agora = True
                if not perfuracao_na_linha:
                    contador_perfs_ciclo += 1
                    perfuracao_na_linha = True
                    
                    if contador_perfs_ciclo >= 4:
                        # --- PROJEÇÃO MULTI-PONTO PURA + ENCOLHIMENTO ---
                        qtd = min(4, len(furos_validos))
                        pts = furos_validos[0:qtd]
                        
                        cx_a = int(sum(p['cx_g'] for p in pts) / qtd)
                        
                        if qtd > 1:
                            # 1. Mede o Pitch instantâneo
                            soma_pitch = 0
                            for i in range(1, qtd):
                                soma_pitch += (pts[i]['cy_g'] - pts[i-1]['cy_g'])
                            pitch_instantaneo = soma_pitch / (qtd - 1)
                            
                            # 2. CÁLCULO DE ENCOLHIMENTO (A cada 10 frames gravados)
                            if frame_count % 10 == 0 and pitch_instantaneo > 0:
                                calc_pct = (1.0 - (pitch_instantaneo / PITCH_PADRAO_PX)) * 100.0
                                encolhimento_atual_pct = max(-5.0, min(10.0, calc_pct))
                            
                            # 3. Projeção Virtual Geométrica (Crava o centro da tela)
                            soma_centros_y = 0
                            for i in range(qtd):
                                multiplicador = 1.5 - i 
                                soma_centros_y += (pts[i]['cy_g'] + (multiplicador * pitch_instantaneo))
                                
                            cy_a = int(soma_centros_y / qtd)
                        else:
                            cy_a = int(pts[0]['cy_g'] + 150) 
                        
                        processar_captura(frame_raw, cx_a, cy_a, frame_count)
                        frame_count += 1
                        contador_perfs_ciclo = 0

        # ==========================================================
        # MOTOR 2: 1D - CENTRO DE MASSA CRAVADO
        # ==========================================================
        elif MODO_DETECCAO == '1D':
            # 1. Gatilho 1D super rápido
            projecao_y = np.sum(binary_small, axis=1)
            limiar_luz = 255 * 8
            linhas_claras = np.where(projecao_y > limiar_luz)[0]
            
            if len(linhas_claras) > 0:
                furos_1d = []
                bloco_atual = [linhas_claras[0]]
                
                for i in range(1, len(linhas_claras)):
                    if linhas_claras[i] - linhas_claras[i-1] <= 3:
                        bloco_atual.append(linhas_claras[i])
                    else:
                        furos_1d.append(bloco_atual)
                        bloco_atual = [linhas_claras[i]]
                furos_1d.append(bloco_atual) 
                
                # Pega o primeiro bloco de linhas (o furo mais alto)
                furo_alvo = furos_1d[0]
                
                # 2. O "MIX": Matemática 2D aplicada sobre o array 1D
                # Extrai a quantidade de luz (peso) de cada linha deste furo específico
                pesos_luz = projecao_y[furo_alvo]
                
                # Calcula a média ponderada (Centro de Massa). Isso é precisão sub-pixel!
                # Se uma borda piscar, o peso brutal do miolo do furo absorve o erro.
                y_pico_small_float = np.average(furo_alvo, weights=pesos_luz)
                y_pico_real = int(y_pico_small_float / ESCALA_CV)
                
                limite_sup = LINHA_GATILHO_Y - MARGEM_GATILHO
                limite_inf = LINHA_GATILHO_Y + MARGEM_GATILHO
                
                cor_1d = (0, 0, 255) if (limite_sup <= y_pico_real <= limite_inf) else (0, 255, 0)
                debug_visual.append({'rect': (lx, y_pico_real - 5 + ly, lw, 10), 'color': cor_1d})
                
                if limite_sup <= y_pico_real <= limite_inf:
                    furo_detectado_agora = True
                    
                    if not perfuracao_na_linha:
                        contador_perfs_ciclo += 1
                        perfuracao_na_linha = True
                        
                        if contador_perfs_ciclo >= 4:
                            cx_a = lx + (lw // 2)
                            cy_a = ly + y_pico_real
                            
                            processar_captura(frame_raw, cx_a, cy_a, frame_count)
                            frame_count += 1
                            contador_perfs_ciclo = 0

        # ==========================================================
        # MOTOR 3: O HÍBRIDO (RADAR 1D + SNIPER 2D)
        # ==========================================================
        elif MODO_DETECCAO == 'MIX':
            # 1. RADAR 1D: Rastreamento ultrarrápido contínuo (Garante FPS alto)
            projecao_y = np.sum(binary_small, axis=1)
            limiar_luz = 255 * 8
            linhas_claras = np.where(projecao_y > limiar_luz)[0]
            
            if len(linhas_claras) > 0:
                furos_1d = []
                bloco_atual = [linhas_claras[0]]
                for i in range(1, len(linhas_claras)):
                    if linhas_claras[i] - linhas_claras[i-1] <= 3:
                        bloco_atual.append(linhas_claras[i])
                    else:
                        furos_1d.append(bloco_atual)
                        bloco_atual = [linhas_claras[i]]
                furos_1d.append(bloco_atual) 
                
                furo_alvo = furos_1d[0]
                y_pico_small = int(np.mean(furo_alvo))
                y_pico_real_1d = int(y_pico_small / ESCALA_CV)
                
                limite_sup = LINHA_GATILHO_Y - MARGEM_GATILHO
                limite_inf = LINHA_GATILHO_Y + MARGEM_GATILHO
                
                # O Radar 1D detectou que o furo cruzou a margem?
                if limite_sup <= y_pico_real_1d <= limite_inf:
                    furo_detectado_agora = True
                    cor_mix = (0, 0, 255)
                    
                    # 2. SNIPER 2D: Acorda APENAS neste frame para extrair a geometria estável
                    contours, _ = cv_find(binary_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    melhor_cy_2d = y_pico_real_1d # Fallback de segurança
                    melhor_cx_2d = lx + (lw // 2)
                    
                    for cnt in contours:
                        area = cv2.contourArea(cnt) * 4 
                        if 300 < area < 10000:
                            x_s, y_s, w_s, h_s = cv2.boundingRect(cnt)
                            if 0.4 < (w_s/h_s) < 2.5:
                                cy_roi_2d = (y_s * 2) + ((h_s * 2) // 2)
                                
                                # Confirma se o contorno 2D é o mesmo buraco que o 1D achou
                                if abs(cy_roi_2d - y_pico_real_1d) < 30: 
                                    melhor_cy_2d = cy_roi_2d
                                    melhor_cx_2d = (x_s * 2) + (w_s * 2 // 2) + lx
                                    
                                    # Desenha a caixa geométrica vermelha do 2D
                                    debug_visual.append({'rect': (x_s*2+lx, y_s*2+ly, w_s*2, h_s*2), 'color': cor_mix})
                                    break 

                    if not perfuracao_na_linha:
                        contador_perfs_ciclo += 1
                        perfuracao_na_linha = True
                        
                        if contador_perfs_ciclo >= 4:
                            # 3. O TIRO: Usa a coordenada matemática estável do 2D para o Crop!
                            cy_a = ly + melhor_cy_2d
                            cx_a = melhor_cx_2d
                            
                            processar_captura(frame_raw, cx_a, cy_a, frame_count)
                            frame_count += 1
                            contador_perfs_ciclo = 0
                else:
                    # Fora da zona, usa só a barra verde do 1D (CPU descansa, FPS sobe)
                    debug_visual.append({'rect': (lx, y_pico_real_1d - 5 + ly, lw, 10), 'color': (0, 255, 0)})

        # ==========================================================
        # FECHAMENTO DO GATILHO E UI (Comum aos dois motores)
        # ==========================================================
        if not furo_detectado_agora:
            perfuracao_na_linha = False

        skip_ui += 1
        if skip_ui >= 3:
            ultimo_frame_bruto = frame_raw 
            ultimo_frame_binario = binary_small
            lista_contornos_debug = debug_visual
            skip_ui = 0
        
        t_fim = get_time()
        tempo_ms_ciclo = (t_fim - t_inicio) * 1000.0
        fps_real_proc = 1.0 / (t_fim - t_inicio) if (t_fim - t_inicio) > 0 else 0

# --- FLASK: DASHBOARD SIMPLIFICADO + HISTOGRAMA + ZEBRA ESTÁTICO ---
def generate_dashboard():
    global perfuracao_na_linha
    while True:
        time.sleep(0.06) 
        if ultimo_frame_bruto is None: continue
        
        # --- PAINEL ESQUERDO (p_live): LIVE VIEW LIMPO ---
        p_live = cv2.resize(ultimo_frame_bruto.copy(), (640, 420))
        sx, sy = 640/1080, 420/720
        
        # Desenhos da Geometria da ROI
        cv2.rectangle(p_live, (int(ROI_X*sx), int(ROI_Y*sy)), (int((ROI_X+ROI_W)*sx), int((ROI_Y+ROI_H)*sy)), (150, 150, 150), 1)
        cor_gatilho = (0, 0, 255) if perfuracao_na_linha else (0, 255, 0)
        
        y_gl = ROI_Y + LINHA_GATILHO_Y
        cv2.line(p_live, (int(ROI_X*sx), int(y_gl*sy)), (int((ROI_X+ROI_W)*sx), int(y_gl*sy)), cor_gatilho, 3)
        cv2.line(p_live, (int(ROI_X*sx), int((y_gl - MARGEM_GATILHO)*sy)), (int((ROI_X+ROI_W)*sx), int((y_gl - MARGEM_GATILHO)*sy)), (50, 50, 50), 1)
        cv2.line(p_live, (int(ROI_X*sx), int((y_gl + MARGEM_GATILHO)*sy)), (int((ROI_X+ROI_W)*sx), int((y_gl + MARGEM_GATILHO)*sy)), (50, 50, 50), 1)

        for item in lista_contornos_debug:
            x, y, w, h = item['rect']; cv2.rectangle(p_live, (int(x*sx), int(y*sy)), (int((x+w)*sx), int((y+h)*sy)), item['color'], 2)
        
        # --- PAINEL DIREITO (p_bin): BINÁRIO E HISTOGRAMA ---
        p_bin = np.zeros((420, 640, 3), dtype=np.uint8)
        
        if ultimo_frame_binario is not None:
            bin_res = cv2.resize(cv2.cvtColor(ultimo_frame_binario, cv2.COLOR_GRAY2RGB), (240, 420))
            p_bin[0:420, 50:290] = bin_res
            
            if ultimo_crop_preview is not None and ultimo_crop_preview.size > 0:
                gray_crop = cv2.cvtColor(ultimo_crop_preview, cv2.COLOR_RGB2GRAY)
                hist = cv2.calcHist([gray_crop], [0], None, [256], [0, 256])
                cv2.normalize(hist, hist, 0, 150, cv2.NORM_MINMAX)
                
                grafico_h = np.zeros((150, 256, 3), dtype=np.uint8)
                cv2.rectangle(grafico_h, (0, 0), (256, 150), (30, 30, 30), -1)
                
                for x in range(256):
                    valor_y = int(hist[x][0])
                    cor_linha = (255, 255, 255) if x > 200 else (150, 255, 150)
                    cv2.line(grafico_h, (x, 150), (x, 150 - valor_y), cor_linha, 1)
                
                cv2.line(grafico_h, (10, 0), (10, 150), (0, 0, 255), 1)
                cv2.line(grafico_h, (245, 0), (245, 150), (0, 0, 255), 1)
                
                pos_y_hist = 135
                pos_x_hist = 330
                p_bin[pos_y_hist : pos_y_hist+150, pos_x_hist : pos_x_hist+256] = grafico_h
                cv2.putText(p_bin, "HISTOGRAMA", (pos_x_hist, pos_y_hist - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # --- PAINEL INFERIOR (p_inf): FOTOGRAMA ESTÁTICO COM ZEBRA ---
        p_inf = np.zeros((300, 1280, 3), dtype=np.uint8)
        
        if ultimo_crop_preview is not None and ultimo_crop_preview.size > 0:
            crop_preview = cv2.resize(ultimo_crop_preview.copy(), (400, 280))
            luma = cv2.cvtColor(crop_preview, cv2.COLOR_RGB2GRAY)
            
            zebra_overlay = crop_preview.copy()
            zebra_overlay[luma > 245] = [0, 0, 255] # Estouro = Vermelho
            zebra_overlay[luma < 10] = [255, 0, 0]  # Crush = Azul
            
            # Centralizando a foto (1280 / 2) - (400 / 2) = 440
            pos_y_zebra = 10
            pos_x_zebra = 50 
            
            p_inf[pos_y_zebra : pos_y_zebra+280, pos_x_zebra : pos_x_zebra+400] = zebra_overlay
            
            # Fundo preto semi-transparente para o texto ficar legível
            cv2.rectangle(p_inf, (pos_x_zebra, pos_y_zebra), (pos_x_zebra + 370, pos_y_zebra + 25), (0, 0, 0), -1)
            cv2.putText(p_inf, "ZEBRA (VERMELHO=ALTAS / AZUL=BAIXAS)", (pos_x_zebra + 5, pos_y_zebra + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Monta a imagem final (restaurando o vstack)
        dashboard = np.vstack((np.hstack((p_live, p_bin)), p_inf))
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(dashboard, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/status')
def get_status():
    global GRAVANDO, contador_perfs_ciclo, frame_count, fps_real_proc, tempo_ms_ciclo
    global ROI_X, ROI_Y, ROI_W, ROI_H, CROP_W, CROP_H, OFFSET_X
    global foco_atual, shutter_speed, gain, fps_cam, THRESH_VAL, LINHA_GATILHO_Y, MARGEM_GATILHO
    global encolhimento_atual_pct, CALIBRANDO
    
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            cpu_temp = float(f.read()) / 1000.0
    except:
        cpu_temp = 0.0

    # --- LEITURA DE DISCO E ARQUIVOS (Ultrarrápida) ---
    total_arquivos = sum(1 for _ in os.scandir(CAPTURE_PATH))
    uso_disco = shutil.disk_usage(CAPTURE_PATH)
    espaco_livre_mb = uso_disco.free / (1024 * 1024)
    espaco_total_mb = uso_disco.total / (1024 * 1024)
    
    return {
        "rec": "GRAVANDO" if GRAVANDO else "PARADO", 
        "cor": "#ff0000" if GRAVANDO else "#00ff00",
        "ciclo": f"{contador_perfs_ciclo}/4", 
        "total": frame_count,
        "fps_proc": f"{fps_real_proc:.1f} FPS", 
        "ms_ciclo": f"{tempo_ms_ciclo:.1f} ms",
        "queue": fila_gravacao.qsize(),
        "temp": f"{cpu_temp:.1f} °C",
        "arquivos": total_arquivos,  # <--- ENVIANDO TOTAL DE FOTOS
        "espaco": f"{espaco_livre_mb:.0f}MB", # <--- ENVIANDO ESPAÇO LIVRE
        "foco": f"{foco_atual:.2f}",
        "exp": shutter_speed,
        "gain": f"{gain:.1f}",
        "fps_cam": fps_cam,
        "shrink": f"{encolhimento_atual_pct:.1f}%",
        "calibrando": CALIBRANDO,
        "thresh": THRESH_VAL,
        "roi_x": ROI_X, "roi_y": ROI_Y, "roi_w": ROI_W, "roi_h": ROI_H,
        "crop_w": CROP_W, "crop_h": CROP_H, "ox": OFFSET_X,
        "gatilho_y": LINHA_GATILHO_Y, "margem": MARGEM_GATILHO
    }


# --- ROTA DE CALIBRAÇÃO ÓPTICA ---
@app.route('/calibrar')
def calibrar():
    global PITCH_PADRAO_PX, CALIBRANDO
    try:
        px = float(request.args.get('px'))
        mm = float(request.args.get('mm'))
        pixels_por_mm = px / mm
        
        PITCH_PADRAO_PX = pixels_por_mm * 4.74  # 4.74mm é o pitch do 35mm positivo
        CALIBRANDO = False  # Trava a tela novamente
        
        print(f"\n[SISTEMA] Calibração Óptica Concluída! 1mm = {pixels_por_mm:.2f}px")
        print(f"[SISTEMA] Novo Pitch Padrão de 35mm cravado em: {PITCH_PADRAO_PX:.1f}px")
        return "OK"
    except Exception as e:
        CALIBRANDO = False
        return f"Erro: {e}"

@app.route('/')
def index():
    return """
    <html><body style='background:#0a0a0a; color:#eee; font-family:monospace; margin:0; overflow-x:hidden;'>
        
        <div style='display:flex; background:#111; padding:10px; border-bottom:1px solid #333; justify-content:space-around; font-size:16px;'>
            <span id='m'>--</span> | 
            CICLO: <b id='c'>0/4</b> |
            DISCO: <b id='arq' style='color:#0f0'>0</b> imgs (<b id='esp' style='color:#0aa'>-</b> livres) | 
            FRAMES: <b id='f'>0</b> | 
            PROC: <b id='fps_proc' style='color:#0ff'>0.0 FPS</b> (<b id='ms_ciclo' style='color:#ff0'>0.0 ms</b>) | 
            TEMP: <b id='t_cpu' style='color:#f90'>0.0 °C</b>
        </div>
        
<div style='display:flex; background:#1a1a1a; padding:6px 10px; border-bottom:1px solid #444; justify-content:space-between; font-size:12px; color:#aaa;'>
            <span><b>ÓPTICA:</b> Foco <span id='v_foco' style='color:#fff'>-</span> | Exp <span id='v_exp' style='color:#fff'>-</span> | Gain <span id='v_gain' style='color:#fff'>-</span> | FPS Cam <span id='v_fpscam' style='color:#fff'>-</span></span>
            <span><b>VISÃO:</b> Thresh <span id='v_thresh' style='color:#fff'>-</span> | Gatilho:<span id='v_gatilho' style='color:#fff'>-</span> &plusmn;<span id='v_margem' style='color:#fff'>-</span></span>
            <span><b>GEOMETRIA:</b> CROP(W:<span id='v_cw' style='color:#fff'>-</span> H:<span id='v_ch' style='color:#fff'>-</span>) | OX:<span id='v_ox' style='color:#fff'>-</span></span>
            <span><b>PRESERVAÇÃO:</b> SHRINKAGE: <b id='v_shrink' style='color:#f0f; font-size:14px;'>0.0%</b></span>
        </div>

        <div style='display:flex; height:88vh; width:100vw;'>
            <div style='flex:7; position:relative; background:#000; display:flex; justify-content:center; align-items:center;' id='video_container'>
                <div id='canvas_wrapper' style='position:relative; display:inline-block;'>
                    <img id='video_img' src="/video_feed" style="max-width:100%; max-height:88vh; display:block;">
                    <canvas id="paquimetro" style="position:absolute; top:0; left:0; width:100%; height:100%; z-index:10; pointer-events:none;"></canvas>
                </div>
            </div>
            
            <div style='flex:3; background:#050505; border-left:1px solid #333; display:flex; align-items:center;'>
                <img src="/preview_feed" style="width:100%; border:1px solid #0f0;">
            </div>
        </div>
        
        <script>
            const canvas = document.getElementById('paquimetro');
            const ctx = canvas.getContext('2d');
            const videoImg = document.getElementById('video_img');
            const wrapper = document.getElementById('canvas_wrapper');
            
            let isDrawing = false;
            let startX, startY;
            let modoCalibracao = false;

            function syncCanvasSize() {
                // Sincroniza a resolução matemática do canvas com o tamanho físico da imagem na tela do seu navegador
                canvas.width = videoImg.clientWidth;
                canvas.height = videoImg.clientHeight;
            }
            window.addEventListener('resize', syncCanvasSize);
            videoImg.addEventListener('load', syncCanvasSize);

            // Atualização contínua de status via Flask
            setInterval(() => {
                fetch('/status').then(r => r.json()).then(d => {
// --- ATUALIZAÇÕES DO PAINEL SUPERIOR ---
                    const m = document.getElementById('m'); m.innerText = d.rec; m.style.color = d.cor;
                    document.getElementById('c').innerText = d.ciclo; 
                    document.getElementById('f').innerText = d.total;
                    document.getElementById('arq').innerText = d.arquivos;
                    document.getElementById('esp').innerText = d.espaco;
                    document.getElementById('fps_proc').innerText = d.fps_proc; 
                    document.getElementById('t_cpu').innerText = d.temp; 
                    
                    // --- ATUALIZAÇÕES DA BARRA INFERIOR DE TELEMETRIA ---
                    document.getElementById('v_foco').innerText = d.foco;
                    document.getElementById('v_exp').innerText = d.exp;
                    document.getElementById('v_gain').innerText = d.gain;
                    document.getElementById('v_fpscam').innerText = d.fps_cam;
                    
                    document.getElementById('v_thresh').innerText = d.thresh;
                    document.getElementById('v_gatilho').innerText = d.gatilho_y;
                    document.getElementById('v_margem').innerText = d.margem;
                    
                    document.getElementById('v_cw').innerText = d.crop_w;
                    document.getElementById('v_ch').innerText = d.crop_h;
                    document.getElementById('v_ox').innerText = d.ox;
                    
                    if(d.shrink) document.getElementById('v_shrink').innerText = d.shrink;

                    // Gatilho do Terminal: Libera a tela para o clique
                    if (d.calibrando && !modoCalibracao) {
                        modoCalibracao = true;
                        syncCanvasSize();
                        canvas.style.pointerEvents = 'auto'; // Habilita o clique
                        canvas.style.cursor = 'crosshair';
                        wrapper.style.outline = '4px solid #f90'; // Borda laranja
                    } else if (!d.calibrando && modoCalibracao) {
                        modoCalibracao = false;
                        canvas.style.pointerEvents = 'none'; // Trava o clique
                        wrapper.style.outline = 'none';
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                    }
                });
            }, 250);

            // Ações de Desenho do Mouse
            canvas.addEventListener('mousedown', (e) => {
                if(!modoCalibracao) return;
                isDrawing = true;
                const rect = canvas.getBoundingClientRect();
                startX = e.clientX - rect.left;
                startY = e.clientY - rect.top;
            });

            canvas.addEventListener('mousemove', (e) => {
                if(!isDrawing) return;
                const rect = canvas.getBoundingClientRect();
                const currentX = e.clientX - rect.left;
                const currentY = e.clientY - rect.top;
                
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.beginPath();
                ctx.moveTo(startX, startY);
                ctx.lineTo(currentX, currentY);
                ctx.strokeStyle = '#00ff00';
                ctx.lineWidth = 2;
                ctx.stroke();
            });

            canvas.addEventListener('mouseup', (e) => {
                if(!isDrawing) return;
                isDrawing = false;
                const rect = canvas.getBoundingClientRect();
                const endX = e.clientX - rect.left;
                const endY = e.clientY - rect.top;

                // 1. O painel completo de mosaico gerado no Python tem 1280x720.
                const ratioX = 1280 / canvas.width;
                const ratioY = 720 / canvas.height;
                
                const distX_mosaico = Math.abs(endX - startX) * ratioX;
                const distY_mosaico = Math.abs(endY - startY) * ratioY;
                
                // 2. O Live View Colorido (onde você deve desenhar) ocupa um bloco de 640x420,
                // que é um redimensionamento do sensor original de 1080x720.
                const escalaReversaX = 1080 / 640;
                const escalaReversaY = 720 / 420;
                
                const distX_camera = distX_mosaico * escalaReversaX;
                const distY_camera = distY_mosaico * escalaReversaY;
                
                const distRealPixels = Math.sqrt(Math.pow(distX_camera, 2) + Math.pow(distY_camera, 2));

                const mm = prompt("Linha aferida! Qual a medida real em milímetros? (Pitch 35mm = 4.74)");
                
                if (mm && !isNaN(mm) && mm > 0) {
                    fetch(`/calibrar?px=${distRealPixels}&mm=${mm}`).then(() => {
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                    });
                } else {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                }
            });
        </script>
    </body></html>
    """

@app.route('/preview_feed')
def preview_feed():
    def generate_preview():
        while True:
            files = sorted([f for f in os.listdir(CAPTURE_PATH) if f.endswith('.jpg')])
            last_frames = files[-120:] if len(files) > 0 else []
            if not last_frames: time.sleep(0.5); continue
            for frame_file in last_frames:
                img = cv2.imread(os.path.join(CAPTURE_PATH, frame_file))
                if img is None: continue
                _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(1/24)
    return Response(generate_preview(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed(): return Response(generate_dashboard(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 1. Inicia o Processo Isolado para Gravação (Ocupa o Core 2)
    mp.Process(target=processo_escrita_disco, args=(fila_gravacao,), daemon=True).start()
    
    # 2. Inicia as Threads principais
    threading.Thread(target=painel_controle, daemon=True).start()
    threading.Thread(target=logica_scanner, daemon=True).start()
    
    # 3. Roda o Flask
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)