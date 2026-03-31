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
sistema_logs = ["[SISTEMA] Miniola iniciada e aguardando comandos."]

def registrar_log(msg):
    global sistema_logs
    agora = time.strftime("%H:%M:%S")
    mensagem_formatada = f"[{agora}] {msg}"
    sistema_logs.append(mensagem_formatada)
    if len(sistema_logs) > 6: # Mantém apenas as últimas 6 mensagens para não pesar
        sistema_logs.pop(0)
    print(mensagem_formatada) # Mantém o print no terminal caso ele esteja aberto

CAPTURE_PATH = "capturas"
if not os.path.exists(CAPTURE_PATH): os.makedirs(CAPTURE_PATH)

picam2 = Picamera2()
shutter_speed, gain, fps_cam = 600, 1.0, 70
foco_atual, passo_foco = 14.5, 0.5
HDR_ATIVO = 0 # 0 = Desligado, 1 = Ativado (V3 IMX708)

# --- GEOMETRIA DUAL-STREAM (Proporção 3:2 cravada) ---
RES_W_MAIN, RES_H_MAIN = 3888, 2592  # 4K para Gravação (Quase 10MP)
RES_W_LORES, RES_H_LORES = 720, 480  # 480p para Visão Computacional

FATOR_ESCALA_X = RES_W_MAIN / RES_W_LORES
FATOR_ESCALA_Y = RES_H_MAIN / RES_H_LORES

# Cria a configuração com os DUAS saídas simultâneas
config = picam2.create_video_configuration(
    main={"size": (RES_W_MAIN, RES_H_MAIN), "format": "RGB888"},
    lores={"size": (RES_W_LORES, RES_H_LORES), "format": "YUV420"}
)
picam2.configure(config)

picam2.set_controls({
    "ExposureTime": shutter_speed, 
    "AnalogueGain": gain, 
    "FrameRate": fps_cam, 
    "LensPosition": foco_atual,
    "HdrMode": HDR_ATIVO 
})
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
PITCH_PADRAO_PX = 85.0  # CALIBRE AQUI: Quantos pixels tem o pitch de um filme NOVO na sua lente?
# --- PARÂMETROS DO CROP ---
OFFSET_X = 220 
CROP_W, CROP_H = 400, 266 

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

def processar_captura(cx_global, cy_global, n_frame):
    global OFFSET_X, CROP_W, CROP_H, ultimo_crop_preview, GRAVANDO
    global FATOR_ESCALA_X, FATOR_ESCALA_Y, ultimo_frame_bruto
    
    # 1. Puxa a matriz 4K gigante (Operação rápida)
    frame_high = picam2.capture_array("main")
    
    # 2. Transpõe as coordenadas para a escala 4K e "fatia" a matriz (Slice é quase instantâneo)
    fx = (cx_global + OFFSET_X) * FATOR_ESCALA_X
    fy = cy_global * FATOR_ESCALA_Y
    cw_real = int(CROP_W * FATOR_ESCALA_X)
    ch_real = int(CROP_H * FATOR_ESCALA_Y)
    
    x1, y1 = max(0, int(fx - (cw_real // 2))), max(0, int(fy - (ch_real // 2)))
    x2, y2 = min(frame_high.shape[1], x1 + cw_real), min(frame_high.shape[0], y1 + ch_real)
    crop_4k = frame_high[y1:y2, x1:x2]
    
    # 3. GERA O PREVIEW LEVE PARA O PAINEL WEB (Sem usar o 4K)
    if ultimo_frame_bruto is not None:
        px1 = max(0, int((cx_global + OFFSET_X) - (CROP_W // 2)))
        py1 = max(0, int(cy_global - (CROP_H // 2)))
        px2 = min(ultimo_frame_bruto.shape[1], px1 + CROP_W)
        py2 = min(ultimo_frame_bruto.shape[0], py1 + CROP_H)
        crop_leve = ultimo_frame_bruto[py1:py2, px1:px2]
        
        if crop_leve.size > 0:
            ultimo_crop_preview = cv2.resize(crop_leve, (400, 280))

    # 4. Envia a matriz pesada para o processo isolado (Core 2) gravar
    if crop_4k.size > 0 and GRAVANDO:
        filename = f"{CAPTURE_PATH}/miniola_{n_frame:06d}.jpg"
        try:
            fila_gravacao.put((crop_4k.copy(), filename), block=False)
        except:
            pass

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

    while True:
        t_inicio = get_time()
        
        # 1. Puxa a matriz bruta em YUV420
        frame_yuv = cap_array("lores")
        if frame_yuv is None: continue
        
        # 2. A MÁGICA DO YUV: O canal Y (Preto e Branco) é a metade superior da matriz!
        # Isso custa ZERO ciclos de CPU para o OpenCV.
        frame_gray_completo = frame_yuv[:RES_H_LORES, :RES_W_LORES]
        
        # 3. Recorta a ROI na imagem já em tons de cinza
        lx, ly, lw, lh = ROI_X, ROI_Y, ROI_W, ROI_H
        roi_gray = frame_gray_completo[ly:ly+lh, lx:lx+lw]
        
        # (Opcional) Podemos manter o resize do ESCALA_CV se quiser, ou tirar se o 480p já for pequeno o suficiente
        roi_small = cv_resize(roi_gray, (0, 0), fx=ESCALA_CV, fy=ESCALA_CV) 
        
        # 4. Binariza direto
        _, binary_small = cv_thresh(roi_small, THRESH_VAL, 255, cv2.THRESH_BINARY) 
        
        # Variáveis limpas para o quadro atual
        perfs_neste_frame = []
        debug_visual = []
        furo_detectado_agora = False
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
                        
                        # 2. CÁLCULO DE ENCOLHIMENTO (Lotes de 10 amostras)
                        if pitch_instantaneo > 0:
                            buffer_pitches.append(pitch_instantaneo)
                            
                            # Quando atingir 10 leituras válidas (Resposta rápida e estável)
                            if len(buffer_pitches) >= 10:
                                pitch_medio = sum(buffer_pitches) / len(buffer_pitches)
                                ultimo_pitch_medio = pitch_medio # Salva para o comando setcal
                                
                                calc_pct = (1.0 - (pitch_medio / PITCH_PADRAO_PX)) * 100.0
                                encolhimento_atual_pct = max(-5.0, min(10.0, calc_pct))
                                
                                buffer_pitches.clear() # Limpa a memória para o próximo lote    
                        
                        # 3. Projeção Virtual Geométrica (Crava o centro da tela)
                        soma_centros_y = 0
                        for i in range(qtd):
                            multiplicador = 1.5 - i 
                            soma_centros_y += (pts[i]['cy_g'] + (multiplicador * pitch_instantaneo))
                            
                        cy_a = int(soma_centros_y / qtd)
                    else:
                        cy_a = int(pts[0]['cy_g'] + 150) 
                    
                    processar_captura(cx_a, cy_a, frame_count)
                    frame_count += 1
                    contador_perfs_ciclo = 0

        # ==========================================================
        # FECHAMENTO DO GATILHO E UI (Comum aos dois motores)
        # ==========================================================
        if not furo_detectado_agora:
            perfuracao_na_linha = False

        skip_ui += 1
        if skip_ui >= 3:
            # Só gasta CPU convertendo cor 1 vez a cada 3 frames!
            ultimo_frame_bruto = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2RGB_I420) 
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
        
        # Escala dinâmica baseada na nova resolução LORES
        sx, sy = 640/RES_W_LORES, 420/RES_H_LORES
        
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
    
@app.route('/api/logs')
def api_logs():
    return {"logs": sistema_logs}

# --- ROTA DA API PARA OS BOTÕES TOUCH ---
@app.route('/api/comando', methods=['POST'])
def api_comando():
    global GRAVANDO, foco_atual, passo_foco, shutter_speed, gain, fps_cam, THRESH_VAL
    global ROI_X, ROI_Y, ROI_W, ROI_H, CROP_W, CROP_H, OFFSET_X, LINHA_GATILHO_Y, MARGEM_GATILHO
    global PITCH_PADRAO_PX, ultimo_pitch_medio, contador_perfs_ciclo, frame_count, CALIBRANDO, HDR_ATIVO
    
    try:
        dados = request.get_json()
        cmd = dados.get('cmd')
        val = float(dados.get('val', 0.0))

        # --- AÇÕES DE SISTEMA ---
        if cmd == 'rec': GRAVANDO = not GRAVANDO
        elif cmd == 'rc': contador_perfs_ciclo = 0
        elif cmd == 'r': 
            frame_count = 0
            for f in os.listdir(CAPTURE_PATH): os.remove(os.path.join(CAPTURE_PATH, f))
        elif cmd == 'off': os.system("sudo poweroff")
        # (Dentro da def api_comando, adicione isso junto com a ÓPTICA)
        elif cmd == 'hdr':
            global HDR_ATIVO
            HDR_ATIVO = 1 if HDR_ATIVO == 0 else 0
            picam2.set_controls({"HdrMode": HDR_ATIVO})
            return {"status": "ok", "msg": f"HDR alterado para: {'ON' if HDR_ATIVO else 'OFF'}"}
        # --- ÓPTICA ---
        elif cmd == 'foco': 
            foco_atual = val; picam2.set_controls({"LensPosition": foco_atual})
        elif cmd == 'exp': 
            shutter_speed = int(val); picam2.set_controls({"ExposureTime": shutter_speed})
        elif cmd == 'gain': 
            gain = val; picam2.set_controls({"AnalogueGain": gain})
        elif cmd == 'fps': 
            fps_cam = int(val); picam2.set_controls({"FrameRate": fps_cam})
        elif cmd == 'af':
            # Motor de Autofoco Macro Nativo
            picam2.set_controls({"AfMode": 1, "AfRange": 2})
            time.sleep(0.5)
            picam2.autofocus_cycle()
            metadados = picam2.capture_metadata()
            if "LensPosition" in metadados:
                foco_atual = round(metadados["LensPosition"], 2)
                picam2.set_controls({"AfMode": 0, "LensPosition": foco_atual, "AfRange": 0})
        
        # --- VISÃO E GATILHO ---
        elif cmd == 'thresh': THRESH_VAL = int(val)
        elif cmd == 'ly': LINHA_GATILHO_Y = int(val)
        elif cmd == 'mg': MARGEM_GATILHO = int(val)
        
        # --- GEOMETRIA E CROP ---
        elif cmd == 'ox': OFFSET_X = int(val)
        elif cmd == 'rx': ROI_X = int(val)
        elif cmd == 'ry': ROI_Y = int(val)
        elif cmd == 'rw': ROI_W = int(val)
        elif cmd == 'rh': ROI_H = int(val)
        elif cmd == 'cw': CROP_W = int(val)
        elif cmd == 'ch': CROP_H = int(val)
        
        # --- JOYSTICK DA ROI ---
        elif cmd == 'w': ROI_Y = max(0, ROI_Y - 5)
        elif cmd == 's': ROI_Y = min(720 - ROI_H, ROI_Y + 5)
        elif cmd == 'a': ROI_X = max(0, ROI_X - 5)
        elif cmd == 'd': ROI_X = min(1080 - ROI_W, ROI_X + 5)
        
        # --- METROLOGIA ---
        elif cmd == 'cal':
            CALIBRANDO = True
            return {"status": "ok", "msg": "Modo Calibração ON: Desenhe a linha no Live View!"}
        elif cmd == 'setcal':
            if ultimo_pitch_medio > 0:
                fator_escala = 1.0 - (val / 100.0)
                PITCH_PADRAO_PX = ultimo_pitch_medio / fator_escala
                return {"status": "ok", "msg": f"Calibrado para {val}%"}
            else:
                return {"status": "erro", "msg": "Aguarde o filme rodar primeiro."}
        
        return {"status": "ok"}
    except Exception as e:
        return {"status": "erro", "msg": str(e)}

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
            <span><b>PRESERVAÇÃO</b> ENCOLHIMENTO: <b id='v_shrink' style='color:#f0f; font-size:14px;'>0.0%</b></span>
        </div>

        <div style='background:#1a1a1a; padding:10px; display:flex; flex-wrap:wrap; gap:10px; justify-content:space-around; border-bottom:1px solid #444; font-size:11px;'>
            
        <div style='background:#222; padding:8px; border-radius:5px; border:1px solid #333;'>
                <b style='color:#aaa; display:block; margin-bottom:5px;'>SISTEMA & METROLOGIA</b>
                <div style='display:flex; gap:5px; margin-bottom:5px;'>
                    <button onclick="enviarCmd('rec')" style='background:#d00; color:#fff; border:none; padding:6px 12px; font-weight:bold; cursor:pointer; border-radius:3px;'>REC/STOP</button>
                    <button onclick="enviarCmd('rc')" style='background:#444; color:#fff; border:none; padding:6px; cursor:pointer; border-radius:3px;'>Realinhar Ciclo</button>
                    <button onclick="enviarCmd('r')" style='background:#444; color:#fff; border:none; padding:6px; cursor:pointer; border-radius:3px;'>Limpar RAM</button>
                    <button onclick="if(confirm('Desligar a Miniola?')) enviarCmd('off')" style='background:#600; color:#fff; border:none; padding:6px; cursor:pointer; border-radius:3px;'>OFF</button>
                    <button onclick="enviarCmd('hdr')" style='width:100%; margin-top:5px; background:#44a; color:#fff; border:none; padding:4px; cursor:pointer; border-radius:3px;'>LIGAR/DESLIGAR HDR (V3)</button>
                </div>
                <div style='display:flex; gap:5px;'>
                    <button onclick="enviarCmd('cal')" style='background:#f90; color:#000; border:none; padding:6px; font-weight:bold; cursor:pointer; border-radius:3px;'>CAL (Óptica)</button>
                    <input type="number" id="in_setcal" step="0.1" placeholder="Ref %" style="width:60px; background:#111; color:#0f0; border:1px solid #555; text-align:center; margin-left:10px;">
                    <button onclick="enviarInput('setcal', 'in_setcal')" style='background:#0055ff; color:#fff; border:none; padding:6px; cursor:pointer; border-radius:3px;'>SETCAL (Dinâmico)</button>
                </div>
            </div>

            <div style='background:#222; padding:8px; border-radius:5px; border:1px solid #333;'>
                <b style='color:#aaa; display:block; margin-bottom:5px;'>ÓPTICA (Lente & Sensor)</b>
                <div style='display:grid; grid-template-columns: 1fr 1fr; gap:5px;'>
                    <div><input type="number" id="in_foco" step="0.1" placeholder="Foco" style="width:50px; background:#111; color:#fff; border:1px solid #555;"> <button onclick="enviarInput('foco', 'in_foco')">Set</button></div>
                    <div><input type="number" id="in_exp" placeholder="Exp(us)" style="width:50px; background:#111; color:#fff; border:1px solid #555;"> <button onclick="enviarInput('exp', 'in_exp')">Set</button></div>
                    <div><input type="number" id="in_gain" step="0.1" placeholder="Gain" style="width:50px; background:#111; color:#fff; border:1px solid #555;"> <button onclick="enviarInput('gain', 'in_gain')">Set</button></div>
                    <div><input type="number" id="in_fps" placeholder="FPS" style="width:50px; background:#111; color:#fff; border:1px solid #555;"> <button onclick="enviarInput('fps', 'in_fps')">Set</button></div>
                </div>
                <button onclick="enviarCmd('af')" style='width:100%; margin-top:5px; background:#880; color:#fff; border:none; padding:4px; cursor:pointer; border-radius:3px;'>AUTO-FOCO MACRO</button>
            </div>

            <div style='background:#222; padding:8px; border-radius:5px; border:1px solid #333;'>
                <b style='color:#aaa; display:block; margin-bottom:5px;'>VISÃO & GATILHO</b>
                <div style='display:grid; grid-template-columns: 1fr; gap:5px;'>
                    <div><input type="number" id="in_thresh" placeholder="Thresh" style="width:50px; background:#111; color:#fff; border:1px solid #555;"> <button onclick="enviarInput('thresh', 'in_thresh')">Thresh</button></div>
                    <div><input type="number" id="in_ly" placeholder="Linha Y" style="width:50px; background:#111; color:#fff; border:1px solid #555;"> <button onclick="enviarInput('ly', 'in_ly')">Linha Y</button></div>
                    <div><input type="number" id="in_mg" placeholder="Margem" style="width:50px; background:#111; color:#fff; border:1px solid #555;"> <button onclick="enviarInput('mg', 'in_mg')">Margem</button></div>
                </div>
            </div>

            <div style='background:#222; padding:8px; border-radius:5px; border:1px solid #333; display:flex; gap:10px;'>
                <div>
                    <b style='color:#aaa; display:block; margin-bottom:5px;'>CROP E OFFSET</b>
                    <div style='display:grid; grid-template-columns: 1fr; gap:5px;'>
                        <div><input type="number" id="in_cw" placeholder="Crop W" style="width:50px; background:#111; color:#fff; border:1px solid #555;"> <button onclick="enviarInput('cw', 'in_cw')">W</button></div>
                        <div><input type="number" id="in_ch" placeholder="Crop H" style="width:50px; background:#111; color:#fff; border:1px solid #555;"> <button onclick="enviarInput('ch', 'in_ch')">H</button></div>
                        <div><input type="number" id="in_ox" placeholder="Offset X" style="width:50px; background:#111; color:#fff; border:1px solid #555;"> <button onclick="enviarInput('ox', 'in_ox')">OX</button></div>
                    </div>
                </div>
                <div>
                    <b style='color:#aaa; display:block; margin-bottom:5px;'>MOVER ROI</b>
                    <div style='display:grid; grid-template-columns: 30px 30px 30px; gap:2px; text-align:center;'>
                        <span></span><button onclick="enviarCmd('w')">W</button><span></span>
                        <button onclick="enviarCmd('a')">A</button><button onclick="enviarCmd('s')">S</button><button onclick="enviarCmd('d')">D</button>
                    </div>
                </div>
            </div>
        </div>

        <div style='background:#000; border-bottom:1px solid #333; padding:5px 10px; font-family:monospace; font-size:11px; height:80px; overflow-y:hidden; display:flex; flex-direction:column; justify-content:flex-end;'>
            <div id="terminal_web" style='color:#0f0; line-height:1.4;'>
                Carregando logs do sistema...
            </div>
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

            // --- FUNÇÕES DA API DO PAINEL ---
            function enviarCmd(comando, valor=0) {
                fetch('/api/comando', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ cmd: comando, val: valor })
                }).then(r => r.json()).then(res => {
                    if(res.status === 'erro') alert("Erro no sistema: " + res.msg);
                    else if(res.msg) alert(res.msg); // Mensagens de sucesso como Setcal
                });
            }

            function enviarInput(comando, id_campo) {
                let valor = document.getElementById(id_campo).value;
                if(valor !== "") enviarCmd(comando, parseFloat(valor));
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
                    fetch('/api/logs').then(r => r.json()).then(dados => {
                        document.getElementById('terminal_web').innerHTML = dados.logs.join('<br>');
                    });
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
    # 1. ARQUIVISTA: Processo Isolado para gravação pesada
    mp.Process(target=processo_escrita_disco, args=(fila_gravacao,), daemon=True).start()
    
    # 2. MOTOR ÓPTICO: Thread de Visão ininterrupta
    threading.Thread(target=logica_scanner, daemon=True).start()
    
    # 3. COMUNICADOR: Roda o Flask na Thread Principal
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)