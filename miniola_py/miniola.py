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

shutter_speed, gain, fps_cam = 600, 1.0, 75
foco_atual, passo_foco = 14.5, 0.5
HDR_ATIVO = 0 # 0 = Desligado, 1 = Ativado

# --- TABELA DE RESOLUÇÕES NATIVAS 16:9 ---
MODOS_RES = {
    "VGA": (854, 480),     
    "HD": (1280, 720),     
    "HIGH": (1536, 864)    
}
MODO_ATUAL = "HIGH"

# Unificamos as variáveis: Visão e Gravação agora são o mesmo espelho
RES_W, RES_H = MODOS_RES[MODO_ATUAL]
RES_W_LORES, RES_H_LORES = RES_W, RES_H 
F_X, F_Y = 1.0, 1.0 

# Inicialização de segurança para o Dashboard não abrir em preto
ultimo_frame_bruto = np.zeros((RES_H, RES_W, 3), dtype=np.uint8)
ultimo_frame_binario = np.zeros((RES_H//2, RES_W//2, 1), dtype=np.uint8)

picam2 = Picamera2()

# --- CONFIGURAÇÃO DE STREAM ÚNICO ---
config = picam2.create_video_configuration(
    main={"size": (RES_W, RES_H), "format": "YUV420"}
)
picam2.configure(config)

picam2.set_controls({
    "ExposureTime": shutter_speed, 
    "AnalogueGain": gain, 
    "FrameRate": fps_cam, 
    "LensPosition": foco_atual,
    "HdrMode": HDR_ATIVO,
    "ScalerCrop": (0, 0, 4608, 2592) # Trava o FOV Total do sensor V3
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
perfuracao_na_linha = False      # trava interna para não contar o mesmo furo várias vezes
trigger_visual_ate = 0.0         # pulso curto só para indicar o gatilho no dashboard
ultimo_frame_bruto = None
ultimo_frame_binario = None
ultimo_crop_preview = np.zeros((280, 400), dtype=np.uint8)
lista_contornos_debug = []
fps_real_proc = 0.0
tempo_ms_ciclo = 0.0
encolhimento_atual_pct = 0.0

# --- FILA DE MULTIPROCESSAMENTO ---
# Fila compartilhada entre o Scanner (Thread) e o Gravador (Processo)
fila_gravacao = mp.Queue(maxsize=30) 

def processo_escrita_disco(fila_in):
    print("[SISTEMA] Gravador 4K Ativado no Core 2.")
    while True:
        item = fila_in.get()
        if item is None: break
        
        f_yuv, coords, filename = item
        x1, y1, x2, y2 = coords
        
        # Converte YUV para BGR (OpenCV faz isso muito rápido no Core isolado)
        img_bgr = cv2.cvtColor(f_yuv, cv2.COLOR_YUV2BGR_I420)
        crop = img_bgr[y1:y2, x1:x2]
        
        cv2.imwrite(filename, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

def processar_captura(f_yuv, cx_g, cy_g, n_f):
    global OFFSET_X, CROP_W, CROP_H, ultimo_crop_preview, GRAVANDO

    # 1. Geometria Direta (F_X e F_Y são 1.0, então a conta é simples)
    # cx_g e cy_g já estão na resolução correta!
    x1 = max(0, int((cx_g + OFFSET_X) - (CROP_W // 2)))
    y1 = max(0, int(cy_g - (CROP_H // 2)))
    x2 = min(RES_W, x1 + CROP_W)
    y2 = min(RES_H, y1 + CROP_H)
    
    # 2. PREVIEW PARA DASHBOARD: Extrai o canal Y (Luma) para o Zebra
    # f_yuv[:RES_H, :RES_W] é a matriz de brilho
    crop_v = f_yuv[y1:y2, x1:x2]
    if crop_v.size > 0:
        # Redimensiona para o dashboard manter o padrão visual 16:9
        ultimo_crop_preview = cv2.resize(crop_v, (400, 225)) 

    # 3. ENVIO PARA O CORE 2 (Gravação)
    if GRAVANDO:
        filename = f"{CAPTURE_PATH}/miniola_{n_f:06d}.jpg"
        try:
            # Enviamos o frame YUV e as coordenadas do crop
            # O Core 2 fará a conversão YUV -> BGR e o salvamento em JPEG
            fila_gravacao.put((f_yuv.copy(), (x1, y1, x2, y2), filename), block=False)
        except mp.queues.Full:
            pass # Evita travar o scanner se o disco estiver lento

def logica_scanner():
    # Cache de funções para ganho de performance
    cap_array = picam2.capture_array
    get_time = time.perf_counter
    
    global frame_count, ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug
    global contador_perfs_ciclo, perfuracao_na_linha, fps_real_proc, tempo_ms_ciclo
    global encolhimento_atual_pct, PITCH_PADRAO_PX, ultimo_pitch_medio, trigger_visual_ate

    ESCALA_CV = 0.5 
    skip_ui = 0
    buffer_pitches = []  
    ultimo_pitch_medio = 0.0
    
    # --- FILTRO DE DEBOUNCE ---
    ultimo_furo_tempo = 0
    MIN_INTERVALO_FURO = 0.05 # 50ms (Evita contar o mesmo furo duas vezes em alta velocidade)
    PULSO_TRIGGER_S = 0.12    # Mantém o gatilho vermelho por pouco tempo só no evento real

    registrar_log("[SISTEMA] Motor de Visão Nativo HIGH em 16:9 iniciado.")

    while True:
        try:
            t_inicio = get_time()
            
            # 1. Puxa o frame único (1536x864)
            f_main = cap_array("main")
            if f_main is None: continue
            
            # 2. Canal Y (P&B) para o OpenCV
            frame_gray_completo = f_main[:RES_H, :RES_W]
            
            # 3. ROI e Binarização
            lx, ly, lw, lh = ROI_X, ROI_Y, ROI_W, ROI_H
            roi_gray = frame_gray_completo[ly:ly+lh, lx:lx+lw]
            roi_small = cv2.resize(roi_gray, (0, 0), fx=ESCALA_CV, fy=ESCALA_CV) 
            _, binary_small = cv2.threshold(roi_small, THRESH_VAL, 255, cv2.THRESH_BINARY)
            
            furo_detectado_agora = False
            contours, _ = cv2.findContours(binary_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            l_sup, l_inf = LINHA_GATILHO_Y - MARGEM_GATILHO, LINHA_GATILHO_Y + MARGEM_GATILHO
            furos_validos = []
            debug_visual = []
            h_lim, w_lim = binary_small.shape[:2]

            for cnt in contours:
                x_s, y_s, w_s, h_s = cv2.boundingRect(cnt)

                # Ignora contornos grudados nas bordas da ROI, que costumam deixar o gatilho sempre ativo
                if x_s <= 0 or y_s <= 0 or (x_s + w_s) >= (w_lim - 1) or (y_s + h_s) >= (h_lim - 1):
                    continue

                area = (w_s * h_s) * 4 # Compensa o resize 0.5
                if 200 < area < 10000 and 0.3 < (w_s / h_s) < 2.5:
                    cy_roi = (y_s * 2) + h_s  # Centro Y na ROI
                    cx_g = (x_s * 2) + w_s + lx
                    cy_g = cy_roi + ly
                    acionou = l_sup <= cy_roi <= l_inf
                    furos_validos.append({'cy_roi': cy_roi, 'cx_g': cx_g, 'cy_g': cy_g, 'acionou': acionou})
                    debug_visual.append({'rect': (x_s*2+lx, y_s*2+ly, w_s*2, h_s*2), 'color': (0,255,0) if acionou else (0,0,255)})

            furos_validos.sort(key=lambda p: p['cy_roi'])
            furos_na_janela = [p for p in furos_validos if p['acionou']]
            furo_gatilho = min(furos_na_janela, key=lambda p: abs(p['cy_roi'] - LINHA_GATILHO_Y)) if furos_na_janela else None
            
            # --- LÓGICA DE GATILHO + PITCH (METROLOGIA) ---
            if furo_gatilho is not None:
                agora = get_time()
                furo_detectado_agora = True
                
                # Só conta quando há transição real de fora -> dentro da janela
                if not perfuracao_na_linha and (agora - ultimo_furo_tempo) > MIN_INTERVALO_FURO:
                    contador_perfs_ciclo += 1
                    perfuracao_na_linha = True
                    ultimo_furo_tempo = agora
                    trigger_visual_ate = agora + PULSO_TRIGGER_S
                    
                    # 1. Medição de Pitch (Só fazemos no momento do gatilho para poupar CPU)
                    qtd = len(furos_validos)
                    pitch_instantaneo = 0
                    if qtd > 1:
                        soma_dist = 0
                        for i in range(1, qtd):
                            soma_dist += (furos_validos[i]['cy_g'] - furos_validos[i-1]['cy_g'])
                        pitch_instantaneo = soma_dist / (qtd - 1)
                        
                        if pitch_instantaneo > 0:
                            buffer_pitches.append(pitch_instantaneo)
                            if len(buffer_pitches) >= 10:
                                pitch_medio = sum(buffer_pitches) / 10
                                ultimo_pitch_medio = pitch_medio
                                calc_pct = (1.0 - (pitch_medio / PITCH_PADRAO_PX)) * 100.0
                                encolhimento_atual_pct = max(-5.0, min(10.0, calc_pct))
                                buffer_pitches.clear()

                    # 2. Disparo da Captura (Ciclo de 4)
                    if contador_perfs_ciclo >= 4:
                        # Projeção geométrica para centralizar o frame
                        if pitch_instantaneo > 0:
                            soma_y = 0
                            for i in range(min(4, qtd)):
                                soma_y += (furos_validos[i]['cy_g'] + ((1.5 - i) * pitch_instantaneo))
                            cy_a = int(soma_y / min(4, qtd))
                        else:
                            cy_a = int(furo_gatilho['cy_g'] + 150)
                        
                        processar_captura(f_main, furo_gatilho['cx_g'], cy_a, frame_count)
                        frame_count += 1
                        contador_perfs_ciclo = 0

            if not furo_detectado_agora:
                perfuracao_na_linha = False

            # --- ATUALIZAÇÃO DA UI (ESSENCIAL PARA NÃO FICAR PRETO) ---
            skip_ui += 1
            if skip_ui >= 3:
                # Converte o frame atual para RGB para o Dashboard
                ultimo_frame_bruto = cv2.cvtColor(f_main, cv2.COLOR_YUV2RGB_I420) 
                ultimo_frame_binario = binary_small
                lista_contornos_debug = debug_visual
                skip_ui = 0
            
            t_fim = get_time()
            fps_real_proc = 1.0 / (t_fim - t_inicio)
            tempo_ms_ciclo = (t_fim - t_inicio) * 1000.0

        except Exception as e:
            # Se der erro, registra mas não deixa a thread morrer
            registrar_log(f"ERRO SCANNER: {str(e)}")
            time.sleep(0.1)

# --- FLASK: DASHBOARD SIMPLIFICADO + HISTOGRAMA + ZEBRA ESTÁTICO ---
def generate_dashboard():
    global trigger_visual_ate
    while True:
        time.sleep(0.06) 
        if ultimo_frame_bruto is None: continue
        
        # --- PAINEL ESQUERDO (p_live): LIVE VIEW LIMPO ---
        p_live = cv2.resize(ultimo_frame_bruto.copy(), (640, 360)) # 360 mantém o 16:9
        
        
        sx, sy = 640/RES_W, 360/RES_H # Agora que é 1:1, RES_W é 1536
        
        # Desenhos da Geometria da ROI
        cv2.rectangle(p_live, (int(ROI_X*sx), int(ROI_Y*sy)), (int((ROI_X+ROI_W)*sx), int((ROI_Y+ROI_H)*sy)), (150, 150, 150), 1)
        cor_gatilho = (0, 0, 255) if time.perf_counter() < trigger_visual_ate else (0, 255, 0)
        
        y_gl = ROI_Y + LINHA_GATILHO_Y
        cv2.line(p_live, (int(ROI_X*sx), int(y_gl*sy)), (int((ROI_X+ROI_W)*sx), int(y_gl*sy)), cor_gatilho, 3)
        cv2.line(p_live, (int(ROI_X*sx), int((y_gl - MARGEM_GATILHO)*sy)), (int((ROI_X+ROI_W)*sx), int((y_gl - MARGEM_GATILHO)*sy)), (50, 50, 50), 1)
        cv2.line(p_live, (int(ROI_X*sx), int((y_gl + MARGEM_GATILHO)*sy)), (int((ROI_X+ROI_W)*sx), int((y_gl + MARGEM_GATILHO)*sy)), (50, 50, 50), 1)

        for item in lista_contornos_debug:
            x, y, w, h = item['rect']; cv2.rectangle(p_live, (int(x*sx), int(y*sy)), (int((x+w)*sx), int((y+h)*sy)), item['color'], 2)
        
        # --- PAINEL DIREITO (p_bin): BINÁRIO E HISTOGRAMA ---
        p_bin = np.zeros((360, 640, 3), dtype=np.uint8)
        
        if ultimo_frame_binario is not None:
            bin_res = cv2.resize(cv2.cvtColor(ultimo_frame_binario, cv2.COLOR_GRAY2RGB), (240, 360))
            p_bin[0:360, 50:290] = bin_res
            
            if ultimo_crop_preview is not None and ultimo_crop_preview.size > 0:
                # --- DETECTOR DE CANAIS ---
                if len(ultimo_crop_preview.shape) == 3:
                    # Se for RGB (3 canais), converte para cinza
                    gray_crop = cv2.cvtColor(ultimo_crop_preview, cv2.COLOR_RGB2GRAY)
                else:
                    # Se já for Cinza/YUV-Y (1 canal), usa direto!
                    gray_crop = ultimo_crop_preview
                
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
        
        # --- PAINEL INFERIOR (p_inf): FOTOGRAMA ESTÁTICO COM ZEBRA ---
        if ultimo_crop_preview is not None and ultimo_crop_preview.size > 0:
            # 1. Normaliza a entrada para Cinza (1 canal)
            if len(ultimo_crop_preview.shape) == 3:
                luma = cv2.cvtColor(ultimo_crop_preview, cv2.COLOR_RGB2GRAY)
            else:
                luma = ultimo_crop_preview 
            
            # Redimensiona para 16:9 (400x225 é o par de 16:9)
            luma_res = cv2.resize(luma, (400, 225))
            
            # 3. Cria a base colorida para o Zebra (converte 1 canal -> 3 canais)
            zebra_overlay = cv2.cvtColor(luma_res, cv2.COLOR_GRAY2RGB)
            
            # 4. Aplica as cores de aviso (Estouro/Sombras)
            zebra_overlay[luma_res > 245] = [255, 0, 0] # Vermelho (High)
            zebra_overlay[luma_res < 10]  = [0, 0, 255] # Azul (Low)
            
            pos_y_z, pos_x_z = 40, 50 
            p_inf[pos_y_z : pos_y_z+225, pos_x_z : pos_x_z+400] = zebra_overlay
            
            # Fundo preto semi-transparente para o texto ficar legível
            cv2.rectangle(p_inf, (pos_x_z, pos_y_z), (pos_x_z + 370, pos_y_z + 25), (0, 0, 0), -1)
            cv2.putText(p_inf, "ZEBRA (VERMELHO=ALTAS / AZUL=BAIXAS)", (pos_x_z + 5, pos_y_z + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
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
        

# --- TROCA DE RESOLUÇÃO (Modo Stream Único Nativo) ---
        elif cmd == 'res':
            global RES_W, RES_H, RES_W_LORES, RES_H_LORES, F_X, F_Y, MODO_ATUAL
            escolha = dados.get('val_str')
            
            if escolha in MODOS_RES:
                registrar_log(f"Engatando marcha: {escolha}")
                MODO_ATUAL = escolha
                
                # 1. Atualizamos a resolução de Captura
                RES_W, RES_H = MODOS_RES[escolha]
                
                # 2. Sincronizamos a "Visão" com a Captura (Essencial para não quebrar o resto do código)
                RES_W_LORES, RES_H_LORES = RES_W, RES_H
                
                # 3. Zeramos o fator de escala (1 pixel na visão = 1 pixel no sensor)
                F_X, F_Y = 1.0, 1.0 
                
                picam2.stop()
                nova_conf = picam2.create_video_configuration(
                    main={"size": (RES_W, RES_H), "format": "YUV420"}
                )
                picam2.configure(nova_conf)
                
                # Aplicamos o FOV travado e o FrameRate alto
                picam2.set_controls({
                    "ExposureTime": shutter_speed, 
                    "AnalogueGain": gain, 
                    "FrameRate": fps_cam, 
                    "LensPosition": foco_atual,
                    "HdrMode": HDR_ATIVO, 
                    "ScalerCrop": (0, 0, 4608, 2592)
                })
                picam2.start()
                return {"status": "ok", "msg": f"Resolução {escolha} Ativa"}
        
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

            <div style='background:#222; padding:8px; border-radius:5px; border:1px solid #444; margin-top:10px;'>
                <b style='color:#f90; display:block; margin-bottom:5px; font-size:10px;'>MARCHAS (Resolução Main 3:2)</b>
                <div style='display:grid; grid-template-columns: repeat(5, 1fr); gap:4px;'>
                    <button onclick="enviarRes('VGA')" style='background:#333; color:#fff; border:1px solid #555; padding:8px 2px; cursor:pointer; border-radius:3px; font-weight:bold;'>VGA</button>
                    <button onclick="enviarRes('HD')"  style='background:#333; color:#fff; border:1px solid #555; padding:8px 2px; cursor:pointer; border-radius:3px; font-weight:bold;'>HD</button>
                    <button onclick="enviarRes('FHD')" style='background:#333; color:#fff; border:1px solid #555; padding:8px 2px; cursor:pointer; border-radius:3px; font-weight:bold;'>FHD</button>
                    <button onclick="enviarRes('2K')"  style='background:#333; color:#fff; border:1px solid #555; padding:8px 2px; cursor:pointer; border-radius:3px; font-weight:bold;'>2K</button>
                    <button onclick="enviarRes('4K')"  style='background:#44a; color:#fff; border:1px solid #555; padding:8px 2px; cursor:pointer; border-radius:3px; font-weight:bold;'>4K</button>
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

            function enviarRes(modo) {
                fetch('/api/comando', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ cmd: 'res', val_str: modo }) // Note o val_str aqui!
                }).then(r => r.json()).then(res => {
                    if(res.status === 'erro') alert("Erro ao trocar resolução: " + res.msg);
                    else registrarLogLocal("Sistema: Resolução alterada para " + modo);
                });
            }

            // Pequena função para ajudar a ver o feedback na hora
            function registrarLogLocal(msg) {
                console.log(msg);
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