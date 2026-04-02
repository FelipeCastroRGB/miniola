import sys 
from unittest.mock import MagicMock

# --- CONFIGURAÇÃO DE AMBIENTE E HARDWARE ---
sys.modules["pykms"] = MagicMock()
sys.modules["kms"] = MagicMock()

try:
    import miniola_cv
    CV_ENGINE = "C++ [Pybind11]"
    scanner_cv = miniola_cv.ScannerVision()
except ImportError:
    CV_ENGINE = "Python [Nativo]"
    scanner_cv = None

from flask import Flask, Response, request, render_template, send_from_directory, jsonify
from picamera2 import Picamera2 
import cv2 
import numpy as np 
import threading 
import multiprocessing as mp 
import time 
import logging 
import shutil
import os
import subprocess
import glob

app = Flask(__name__) # Flask para o Dashboard (Roda no Core 0)
log = logging.getLogger('werkzeug') # Desativa os logs de requisição do Flask para não poluir o console
log.setLevel(logging.ERROR) 

CAPTURE_PATH = "capturas"
if not os.path.exists(CAPTURE_PATH): os.makedirs(CAPTURE_PATH)

picam2 = Picamera2()
shutter_speed, gain, fps_cam = 600, 1.0, 90
foco_atual, passo_foco = 14.5, 0.5

# Resolução única: HIGH (1536x864) — recalibrar ROI/CROP no hardware
RES_W, RES_H = 1536, 864

config = picam2.create_video_configuration(main={"size": (RES_W, RES_H), "format": "RGB888"})
picam2.configure(config)
picam2.set_controls({
    "ExposureTime": shutter_speed, 
    "AnalogueGain": gain, 
    "FrameRate": fps_cam, 
    "LensPosition": foco_atual,
    "ScalerCrop": (0, 0, 4608, 2592) # Trava o FOV Total
})
picam2.start()

# --- GEOMETRIA DO ROI E ESTADO ---
GRAVANDO = False
CALIBRANDO = False           # Trava de segurança da tela
PROCESSANDO_VIDEO = False    # Alerta o scanner para hibernar
FPS_PROJECAO = 24.0          # FPS de reprodução do filme (independente do fps_cam do sensor!)
ROI_X, ROI_Y = 200, 10
ROI_W, ROI_H = 80, 840
# --- LÓGICA DE GATILHO SIMPLIFICADA ---
LINHA_GATILHO_Y = 110  # Posição Y relativa DENTRO da ROI
MARGEM_GATILHO = 23    # Margem de disparo (px para cima e para baixo)
THRESH_VAL = 239 # Valor do threshold para binarização
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
        cv2.imwrite(filename, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 99])

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

# --- LABORATÓRIO BACKGROUND ---
def disparar_processamento():
    global PROCESSANDO_VIDEO
    PROCESSANDO_VIDEO = True
    print("\n[LABORATÓRIO] 🧪 Injetando químicos! Compilador FFmpeg iniciado e Scanner adormecido...")
    try:
        proc = subprocess.run([sys.executable, "process.py", "--fps", str(FPS_PROJECAO)], capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[LABORATÓRIO] FFFmpeg abortou ou frames estão faltando!\nLOG DE ERRO:\n{proc.stderr}\n{proc.stdout}")
        else:
            print(f"[LABORATÓRIO] Rolo finalizado! Disponível na Galeria Web.")
    except Exception as e:
        print(f"[LABORATÓRIO] ERRO FATAL no processamento: {e}")
    
    PROCESSANDO_VIDEO = False
    print("[SISTEMA] Scanner acordado de volta à vida.")

# --- PAINEL DE CONTROLE ---
def painel_controle():
    global frame_count, GRAVANDO, LINHA_GATILHO_Y, MARGEM_GATILHO, ROI_X, CROP_H, CROP_W, ROI_Y, ROI_W, ROI_H, THRESH_VAL
    global foco_atual, passo_foco, shutter_speed, gain, fps_cam, OFFSET_X, contador_perfs_ciclo, CALIBRANDO
    global ultimo_pitch_medio, PITCH_PADRAO_PX, CV_ENGINE, FPS_PROJECAO
    time.sleep(2)
    print("\n" + "═"*45)
    print(f"   MINIOLA - PAINEL DE CONTROLE  |  MOTOR DE VISÃO: {CV_ENGINE}")
    print("═"*45)
    print("   GATILHO:   ly (Linha na ROI)| mg (Margem)")
    print("   SISTEMA:   rec (Gravar)| r (Reset Capturas)| rc (Realinhar Ciclo)| proc (Gerar MP4)| rout (Limpar Vídeos)| pfps [val] (FPS Projeção)")
    print("   ÓPTICA:    k/l (Foco Manual)| j [val] (Passo)| af (Auto Foco)")
    print("   EXPOSIÇÃO: e [val] (Shutter Speed)| g [val] (Gain)| fps [val] (Frame Rate)")
    print("   CROP:      ch (Altura)| cw (Largura)| ox [val] (Offset X)")
    print("   ROI:       w/a/s/d (Move ROI)| rx/ry/rw/rh [val] (Ajuste direto)")
    print("   MEDIÇÃO:   cal (Calibrar)| setcal [val] (Cal. Dinâmica)")
    print("   MOTOR:     motor (Alterna C++ <-> Python)| t [val] (Threshold)")
    print("   OUTROS:    off (Desligar)")
    print("═"*45)
    while True:
        try:
            entrada = input("\n>> ").split()
            if not entrada: continue
            cmd = entrada[0].lower()
            
            val = 0
            if len(entrada) > 1:
                try: val = float(entrada[1])
                except ValueError: pass
            
            if cmd == 'motor':
                if scanner_cv is None:
                    print("[MOTOR] Módulo C++ não está compilado. Impossível alternar.")
                elif CV_ENGINE == "C++ [Pybind11]":
                    CV_ENGINE = "Python [Nativo]"
                    print(f"[MOTOR] ⚡ Motor alternado para: {CV_ENGINE}")
                else:
                    CV_ENGINE = "C++ [Pybind11]"
                    scanner_cv.reset_ciclo()
                    print(f"[MOTOR] ⚡ Motor alternado para: {CV_ENGINE}")
            elif cmd == 'w': ROI_Y = max(0, ROI_Y - 5)
            elif cmd == 's': ROI_Y = min(RES_H - ROI_H, ROI_Y + 5)
            elif cmd == 'a': ROI_X = max(0, ROI_X - 5)
            elif cmd == 'd': ROI_X = min(RES_W - ROI_W, ROI_X + 5)
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
            # --- NOVO: AFERIÇÃO ESTRUTURAL (CALIBRAÇÃO COM REFERÊNCIA) ---
            elif cmd == 'setcal':
                if 'ultimo_pitch_medio' in globals() and ultimo_pitch_medio > 0:
                    encolhimento_referencia = float(val) if len(entrada) > 1 else 0.0
                    
                    # Calcula o fator de escala real do filme de teste
                    fator_escala = 1.0 - (encolhimento_referencia / 100.0)
                    
                    # Projeta o novo Padrão (0%) cravado na óptica atual
                    PITCH_PADRAO_PX = ultimo_pitch_medio / fator_escala
                    
                    print(f"\n[METROLOGIA] CALIBRAÇÃO DINÂMICA CONCLUÍDA!")
                    print(f"-> Filme Referência utilizado: {encolhimento_referencia}% de encolhimento.")
                    print(f"-> Novo Padrão (0%): {PITCH_PADRAO_PX:.2f}px")
                else:
                    print("[ERRO] Deixe o filme de referência rodar e estabilizar no dashboard antes de calibrar.")
            elif cmd == 'g': 
                gain = val; picam2.set_controls({"AnalogueGain": gain})
            elif cmd == 'fps':
                fps_cam = int(val); picam2.set_controls({"FrameRate": fps_cam})
            elif cmd == 't': THRESH_VAL = int(val)
            elif cmd == 'rec': GRAVANDO = not GRAVANDO
            elif cmd == 'proc': 
                if not PROCESSANDO_VIDEO:
                    threading.Thread(target=disparar_processamento, daemon=True).start()
                else: print("[ERRO] FFmpeg já está encodando no núcleo adjacente.")
            elif cmd == 'rc': 
                contador_perfs_ciclo = 0
                if CV_ENGINE == "C++ [Pybind11]" and scanner_cv is not None:
                    scanner_cv.reset_ciclo()
                print("[SISTEMA] Fase realinhada! Ciclo forçado para 0/4.")
            elif cmd == 'rout': 
                print("[SISTEMA] Queimando os acetatos da prateleira (Limpeza de Filmes Renderizados)...")
                if os.path.exists('output'):
                    for f in os.listdir('output'): 
                        if f.endswith('.mp4'): os.remove(os.path.join('output', f))
                print("-> GALERIA DE MP4 LIMPA.")
            elif cmd == 'pfps':
                FPS_PROJECAO = float(val)
                print(f"[LABORATÓRIO] FPS de Projeção definido para {FPS_PROJECAO} fps.")
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
    global encolhimento_atual_pct, PITCH_PADRAO_PX, ultimo_pitch_medio

    ESCALA_CV = 0.5 
    skip_ui = 0
    buffer_pitches = []  
    ultimo_pitch_medio = 0.0 # <-- INICIALIZADO AQUI
    buffer_tempos = []

    while True:
        if PROCESSANDO_VIDEO:
            time.sleep(1.0)
            continue
            
        t_inicio = get_time()
        
        frame_raw = cap_array()
        if frame_raw is None: continue
        
        lx, ly, lw, lh = ROI_X, ROI_Y, ROI_W, ROI_H
        
        if CV_ENGINE == "C++ [Pybind11]":
            ret = scanner_cv.process_frame(
                frame_raw, lx, ly, lw, lh,
                THRESH_VAL, LINHA_GATILHO_Y, MARGEM_GATILHO, PITCH_PADRAO_PX
            )
            binary_small = ret["binary_small"]
            
            # C++ retorna uma lista de arrays, precisamos normalizar num format dict pra UI python funcionar sem quebrar
            debug_visual = []
            for item in ret["debug_visual"]:
                debug_visual.append({'rect': item['rect'], 'color': item['color']})
            
            perfuracao_na_linha = ret["perfuracao_na_linha"]
            contador_perfs_ciclo = ret["contador_perfs_ciclo"]
            encolhimento_atual_pct = ret["encolhimento_atual_pct"]
            
            if ret["ultimo_pitch_medio"] > 0:
                ultimo_pitch_medio = ret["ultimo_pitch_medio"]
            
            furo_detectado_agora = ret["achou_furo"]

            if ret["capturar"]:
                processar_captura(frame_raw, ret["cx_a"], ret["cy_a"], frame_count)
                frame_count += 1
                
        else:
            roi_color = frame_raw[ly:ly+lh, lx:lx+lw]
            roi_gray = cv_cvt(roi_color, cv2.COLOR_RGB2GRAY)
            roi_small = cv_resize(roi_gray, (0, 0), fx=ESCALA_CV, fy=ESCALA_CV) 
            _, binary_small = cv_thresh(roi_small, THRESH_VAL, 255, cv2.THRESH_BINARY) 
            
            perfs_neste_frame = []
            debug_visual = []
            furo_detectado_agora = False
            contours, _ = cv_find(binary_small, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            limite_superior = LINHA_GATILHO_Y - MARGEM_GATILHO
            limite_inferior = LINHA_GATILHO_Y + MARGEM_GATILHO
            
            furos_validos = []
            
            for cnt in contours:
                x_s, y_s, w_s, h_s = cv2.boundingRect(cnt)
                area_aprox = (w_s * h_s) * 4 
                
                if 200 < area_aprox < 10000 and 0.2 < (w_s / h_s) < 2.5:
                    cy_roi = (y_s * 2) + ((h_s * 2) // 2)
                    cx_global = (x_s * 2) + (w_s * 2 // 2) + lx
                    cy_global = cy_roi + ly
                    
                    acionou = limite_superior <= cy_roi <= limite_inferior
                    cor = (0, 0, 255) if acionou else (0, 255, 0)
                    
                    furos_validos.append({'cy_roi': cy_roi, 'cx_g': cx_global, 'cy_g': cy_global, 'acionou': acionou})
                    debug_visual.append({'rect': (x_s*2+lx, y_s*2+ly, w_s*2, h_s*2), 'color': cor})

            furos_validos.sort(key=lambda p: p['cy_roi'])
            
            if furos_validos and furos_validos[0]['acionou']:
                furo_detectado_agora = True
                if not perfuracao_na_linha:
                    contador_perfs_ciclo += 1
                    perfuracao_na_linha = True
                    
                    if contador_perfs_ciclo >= 4:
                        qtd = min(4, len(furos_validos))
                        pts = furos_validos[0:qtd]
                        cx_a = int(sum(p['cx_g'] for p in pts) / qtd)
                        
                        if qtd > 1:
                            soma_pitch = 0
                            for i in range(1, qtd):
                                soma_pitch += (pts[i]['cy_g'] - pts[i-1]['cy_g'])
                            pitch_instantaneo = soma_pitch / (qtd - 1)
                            
                            if pitch_instantaneo > 0:
                                buffer_pitches.append(pitch_instantaneo)
                                if len(buffer_pitches) >= 10:
                                    pitch_medio = sum(buffer_pitches) / len(buffer_pitches)
                                    ultimo_pitch_medio = pitch_medio 
                                    calc_pct = (1.0 - (pitch_medio / PITCH_PADRAO_PX)) * 100.0
                                    encolhimento_atual_pct = max(-5.0, min(10.0, calc_pct))
                                    buffer_pitches.clear()    
                            
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
        inst_ms = (t_fim - t_inicio) * 1000.0
        
        # Média Móvel Deslizante Estrita (A média EMA matemática subestimava severamente a cadência!)
        buffer_tempos.append(inst_ms)
        if len(buffer_tempos) > 30:
            buffer_tempos.pop(0)
            
        tempo_ms_ciclo = sum(buffer_tempos) / len(buffer_tempos)
        fps_real_proc = 1000.0 / tempo_ms_ciclo if tempo_ms_ciclo > 0 else 0

# --- FLASK: DASHBOARD SIMPLIFICADO + HISTOGRAMA + ZEBRA ESTÁTICO ---
def generate_dashboard():
    global perfuracao_na_linha
    while True:
        time.sleep(0.06) 
        if ultimo_frame_bruto is None: continue
        
        # --- PAINEL ESQUERDO (p_live): LIVE VIEW LIMPO ---
        p_live = cv2.resize(ultimo_frame_bruto.copy(), (640, 420))
        sx, sy = 640/RES_W, 420/RES_H
        
        # Desenhos da Geometria da ROI
        cv2.rectangle(p_live, (int(ROI_X*sx), int(ROI_Y*sy)), (int((ROI_X+ROI_W)*sx), int((ROI_Y+ROI_H)*sy)), (150, 150, 150), 1)
        cor_gatilho = (0, 0, 255) if perfuracao_na_linha else (0, 255, 0)
        
        y_gl = ROI_Y + LINHA_GATILHO_Y
        cv2.line(p_live, (int(ROI_X*sx), int(y_gl*sy)), (int((ROI_X+ROI_W)*sx), int(y_gl*sy)), cor_gatilho, 3)
        cv2.line(p_live, (int(ROI_X*sx), int((y_gl - MARGEM_GATILHO)*sy)), (int((ROI_X+ROI_W)*sx), int((y_gl - MARGEM_GATILHO)*sy)), (50, 50, 50), 1)
        cv2.line(p_live, (int(ROI_X*sx), int((y_gl + MARGEM_GATILHO)*sy)), (int((ROI_X+ROI_W)*sx), int((y_gl + MARGEM_GATILHO)*sy)), (50, 50, 50), 1)

        for item in lista_contornos_debug:
            x, y, w, h = item['rect']; cv2.rectangle(p_live, (int(x*sx), int(y*sy)), (int((x+w)*sx), int((y+h)*sy)), item['color'], 2)
        
        # --- PAINEL DIREITO (p_bin): SÓ BINÁRIO ---
        p_bin = np.zeros((420, 640, 3), dtype=np.uint8)
        
        if ultimo_frame_binario is not None:
            bin_res = cv2.resize(cv2.cvtColor(ultimo_frame_binario, cv2.COLOR_GRAY2RGB), (240, 420))
            p_bin[0:420, 50:290] = bin_res

        # --- PAINEL INFERIOR (p_inf): ZEBRA + HISTOGRAMA LADO A LADO ---
        p_inf = np.zeros((300, 1280, 3), dtype=np.uint8)
        
        if ultimo_crop_preview is not None and ultimo_crop_preview.size > 0:
            # --- ZEBRA (esquerda do painel inferior) ---
            crop_preview = cv2.resize(ultimo_crop_preview.copy(), (400, 280))
            luma = cv2.cvtColor(crop_preview, cv2.COLOR_RGB2GRAY)
            
            zebra_overlay = crop_preview.copy()
            zebra_overlay[luma > 245] = [0, 0, 255]  # Estouro = Vermelho
            zebra_overlay[luma < 10]  = [255, 0, 0]  # Crush = Azul
            
            pos_y_zebra, pos_x_zebra = 10, 50
            p_inf[pos_y_zebra : pos_y_zebra+280, pos_x_zebra : pos_x_zebra+400] = zebra_overlay
            cv2.rectangle(p_inf, (pos_x_zebra, pos_y_zebra), (pos_x_zebra + 370, pos_y_zebra + 25), (0, 0, 0), -1)
            cv2.putText(p_inf, "ZEBRA (VERM=ALTAS / AZUL=BAIXAS)", (pos_x_zebra + 5, pos_y_zebra + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # --- HISTOGRAMA (ao lado da Zebra, maior) ---
            gray_crop = cv2.cvtColor(ultimo_crop_preview, cv2.COLOR_RGB2GRAY)
            hist = cv2.calcHist([gray_crop], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, 270, cv2.NORM_MINMAX)  # Altura máxima = 270px
            
            HIST_W, HIST_H = 512, 280  # Histograma 2x mais largo e altura total do painel
            grafico_h = np.zeros((HIST_H, HIST_W, 3), dtype=np.uint8)
            cv2.rectangle(grafico_h, (0, 0), (HIST_W, HIST_H), (20, 20, 20), -1)
            
            # Desenha as barras interpolando 256 bins -> 512px de largura
            for i in range(256):
                x0 = i * 2
                x1 = x0 + 2
                valor_y = int(hist[i][0])
                cor = (255, 255, 255) if i > 200 else (80, 200, 80)
                cv2.rectangle(grafico_h, (x0, HIST_H), (x1, HIST_H - valor_y), cor, -1)
            
            # Linhas de referência (preto/branco absoluto)
            cv2.line(grafico_h, (20, 0), (20, HIST_H), (0, 80, 255), 1)
            cv2.line(grafico_h, (490, 0), (490, HIST_H), (0, 80, 255), 1)
            cv2.putText(grafico_h, "0", (5, HIST_H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 255), 1)
            cv2.putText(grafico_h, "255", (476, HIST_H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 255), 1)
            
            pos_x_hist = 500
            pos_y_hist = 10
            p_inf[pos_y_hist : pos_y_hist + HIST_H, pos_x_hist : pos_x_hist + HIST_W] = grafico_h
            cv2.putText(p_inf, "HISTOGRAMA (LUMINANCIA)", (pos_x_hist, pos_y_hist - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Monta a imagem final
        dashboard = np.vstack((np.hstack((p_live, p_bin)), p_inf))
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(dashboard, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/status')
def get_status():
    global GRAVANDO, contador_perfs_ciclo, frame_count, fps_real_proc, tempo_ms_ciclo
    global ROI_X, ROI_Y, ROI_W, ROI_H, CROP_W, CROP_H, OFFSET_X
    global foco_atual, shutter_speed, gain, fps_cam, THRESH_VAL, LINHA_GATILHO_Y, MARGEM_GATILHO
    # --- MEDICÃO DE HARDWARE (Linux /proc) ---
    cpu_percent, ram_percent, cpu_temp = 0.0, 0.0, 0.0
    try:
        # CPU Load: /proc/stat
        with open('/proc/stat', 'r') as f:
            fields = [float(column) for column in f.readline().strip().split()[1:]]
        idle, total = fields[3], sum(fields)
        cpu_percent = 100.0 * (1.0 - idle / total)
        
        # RAM Usage: /proc/meminfo
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
        mem = {line.split(':')[0]: int(line.split(':')[1].split()[0]) for line in lines[:32]}
        ram_percent = 100.0 * (1.0 - mem['MemAvailable'] / mem['MemTotal'])

        # Temp: /sys/class/thermal
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            cpu_temp = float(f.read()) / 1000.0
    except:
        pass # Fallback para Windows/Dev
    
    # --- LEITURA DE DISCO E ARQUIVOS (Ultrarrápida) ---
    total_arquivos = sum(1 for _ in os.scandir(CAPTURE_PATH))
    uso_disco = shutil.disk_usage(CAPTURE_PATH)
    espaco_livre_mb = uso_disco.free / (1024 * 1024)
    espaco_total_mb = uso_disco.total / (1024 * 1024)
    
    return {
        "processando": PROCESSANDO_VIDEO, 
        "cpu": f"{cpu_percent:.1f}%",
        "ram": f"{ram_percent:.1f}%",
        "temp": f"{cpu_temp:.1f}°C",
        "rec": "GRAVANDO" if GRAVANDO else "PARADO", 
        "cor": "#ff0000" if GRAVANDO else "#00ff00",
        "ciclo": f"{contador_perfs_ciclo}/4", 
        "total": frame_count,
        "fps_proc": f"{fps_real_proc:.1f} FPS", 
        "ms_ciclo": f"{tempo_ms_ciclo:.1f} ms",
        "queue": fila_gravacao.qsize(),
        "arquivos": total_arquivos,  
        "espaco": f"{espaco_livre_mb:.0f}MB", 
        "foco": f"{foco_atual:.2f}",
        "exp": shutter_speed,
        "gain": f"{gain:.1f}",
        "fps_cam": fps_cam,
        "shrink": f"{encolhimento_atual_pct:.1f}%",
        "calibrando": CALIBRANDO,
        "thresh": THRESH_VAL,
        "roi_x": ROI_X, "roi_y": ROI_Y, "roi_w": ROI_W, "roi_h": ROI_H,
        "crop_w": CROP_W, "crop_h": CROP_H, "ox": OFFSET_X,
        "gatilho_y": LINHA_GATILHO_Y, "margem": MARGEM_GATILHO,
        "res_w": RES_W, "res_h": RES_H,
        "fps_projecao": FPS_PROJECAO
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

@app.route('/api/process', methods=['POST'])
def api_process():
    if not PROCESSANDO_VIDEO:
        threading.Thread(target=disparar_processamento, daemon=True).start()
        return jsonify({"status": "started"})
    return jsonify({"status": "already_running"}), 400

@app.route('/api/videos', methods=['GET'])
def api_videos():
    if not os.path.exists('output'): return jsonify([])
    # Varre a pasta de renderizados e ordena do mais novo para o mais antigo
    arquivos = glob.glob('output/*.mp4')
    arquivos.sort(key=os.path.getctime, reverse=True)
    return jsonify([os.path.basename(f) for f in arquivos])

@app.route('/output/<path:filename>')
def serve_video(filename):
    return send_from_directory('output', filename)

@app.route('/')
def index():
    return render_template('index.html')

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