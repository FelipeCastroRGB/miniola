import sys
from unittest.mock import MagicMock
import os

# --- CONFIGURAÇÃO DE AMBIENTE E HARDWARE ---
sys.modules["pykms"] = MagicMock()
sys.modules["kms"] = MagicMock()

from flask import Flask, Response 
from picamera2 import Picamera2 
import cv2
import numpy as np
import threading
import multiprocessing as mp # MULTIPROCESSAMENTO
import time
import logging
import shutil

app = Flask(__name__)
log = logging.getLogger('werkzeug')
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
ROI_X, ROI_Y = 25, 10
ROI_W, ROI_H = 80, 700

# --- LÓGICA DE GATILHO SIMPLIFICADA ---
LINHA_GATILHO_Y = 110  # Posição Y relativa DENTRO da ROI
MARGEM_GATILHO = 23    # Margem de disparo (px para cima e para baixo)
THRESH_VAL = 239 # Valor do threshold para binarização

# --- PARÂMETROS DO CROP ---
OFFSET_X = 470 # Deslocamento X do centro global em relação ao centro da ROI (ajuste fino para alinhar o crop com a posição real dos furos)
CROP_W, CROP_H = 918, 612 # Tamanho do crop final (ajustado para capturar a área do fotograma)

contador_perfs_ciclo = 0
frame_count = 0
perfuracao_na_linha = False
ultimo_frame_bruto = None
ultimo_frame_binario = None
ultimo_crop_preview = np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
lista_contornos_debug = []
fps_real_proc = 0.0
tempo_ms_ciclo = 0.0

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
    global foco_atual, passo_foco, shutter_speed, gain, fps_cam, OFFSET_X, contador_perfs_ciclo
    time.sleep(2)
    print("\n" + "═"*45)
    print("   MINIOLA - PAINEL DE CONTROLE")
    print("═"*45)
    print("   GATILHO:   ly (Linha na ROI)| mg (Margem)")
    print("   SISTEMA:   rec (Gravar)| r (Reset Tudo)| rc (Realinhar Ciclo)")
    print("   ÓPTICA:    k/l (Foco Manual)| j [val] (Passo)| af (Auto Foco)")
    print("   EXPOSIÇÃO: e [val] (Shutter Speed)| g [val] (Gain)| fps [val] (Frame Rate)")
    print("   CROP:   ch (Altura)| cw (Largura)")
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
            elif cmd == 'af':
                print("[ÓPTICA] Iniciando varredura de Auto Foco (Aguarde...)")
                # AfMode: 1 (Auto), AfTrigger: 1 (Start)
                picam2.set_controls({"AfMode": 1, "AfTrigger": 1})
                
                # Dá 3 segundos para o motor da lente ir e voltar procurando o contraste
                time.sleep(3) 
                
                # AfMode: 0 (Manual). Trava a lente na posição encontrada (Padrão de Arquivo)
                picam2.set_controls({"AfMode": 0})
                
                # Lê os sensores internos da câmera para descobrir onde o motor parou
                controles_atuais = picam2.get_controls()
                if "LensPosition" in controles_atuais:
                    foco_atual = round(controles_atuais["LensPosition"], 2)
                print(f"[ÓPTICA] Foco cravado e travado na posição: {foco_atual}")
            elif cmd == 'e': 
                shutter_speed = int(val); picam2.set_controls({"ExposureTime": shutter_speed})
            elif cmd == 'g': 
                gain = val; picam2.set_controls({"AnalogueGain": gain})
            elif cmd == 'fps':
                fps_cam = int(val); picam2.set_controls({"FrameRate": fps_cam})
            elif cmd == 't': THRESH_VAL = int(val)
            elif cmd == 'rec': GRAVANDO = not GRAVANDO
            elif cmd == 'rc': 
                contador_perfs_ciclo = 0
                print("[SISTEMA] Fase realinhada! Ciclo forçado para 0/4.")
            elif cmd == 'r': 
                frame_count = 0
                for f in os.listdir(CAPTURE_PATH): os.remove(os.path.join(CAPTURE_PATH, f))
                print("RAM DRIVE LIMPO.")
        except Exception as e: print(f"Erro: {e}")

# --- LÓGICA DO SCANNER OTIMIZADA ---
def logica_scanner():
    cap_array = picam2.capture_array
    cv_cvt = cv2.cvtColor
    cv_resize = cv2.resize
    cv_thresh = cv2.threshold
    cv_find = cv2.findContours
    get_time = time.perf_counter
    
    global frame_count, ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug
    global contador_perfs_ciclo, perfuracao_na_linha, fps_real_proc, tempo_ms_ciclo

    ESCALA_CV = 0.5 # Reduz a resolução para acelerar o processamento de contornos (ajuste fino para manter a detecção confiável) 
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
        contours, _ = cv_find(binary_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        perfs_neste_frame = []
        debug_visual = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt) * 4 
            if 300 < area < 10000:
                x_s, y_s, w_s, h_s = cv2.boundingRect(cnt)
                if 0.4 < (w_s/h_s) < 2.5:
                    # Coordenadas relativas apenas à ROI
                    cy_roi = (y_s * 2) + h_s 
                    
                    # Coordenadas globais só para passar para o Crop depois
                    cx_global = (x_s * 2) + w_s + lx
                    cy_global = cy_roi + ly
                    
                    perfs_neste_frame.append({
                        'cy_roi': cy_roi, 
                        'cx_global': cx_global, 
                        'cy_global': cy_global
                    })
                    debug_visual.append({'rect': (x_s*2+lx, y_s*2+ly, w_s*2, h_s*2), 'color': (0, 255, 0)})

        # Ordena as perfurações pela posição Y dentro da ROI
        perfs_neste_frame.sort(key=lambda p: p['cy_roi'])
        furo_detectado_agora = False
        
        if perfs_neste_frame:
            # Avalia DENTRO do espaço da ROI
            if abs(perfs_neste_frame[0]['cy_roi'] - LINHA_GATILHO_Y) <= MARGEM_GATILHO:
                furo_detectado_agora = True
                if not perfuracao_na_linha:
                    contador_perfs_ciclo += 1
                    perfuracao_na_linha = True
                    
                    if contador_perfs_ciclo >= 4:
                        # Pega até 4 furos (o que estiver visível na ROI)
                        qtd_furos_visiveis = min(4, len(perfs_neste_frame))
                        pts = perfs_neste_frame[0:qtd_furos_visiveis]
                        
                        # Calcula o centro com base no que a câmera está vendo
                        cx_a = int(sum(p['cx_global'] for p in pts) / qtd_furos_visiveis)
                        cy_a = int(sum(p['cy_global'] for p in pts) / qtd_furos_visiveis)
                        
                        processar_captura(frame_raw, cx_a, cy_a, frame_count)
                        frame_count += 1
                        
                        # OBRIGATÓRIO: Zera o ciclo sempre que bater 4!
                        contador_perfs_ciclo = 0
        
        if not furo_detectado_agora:
            perfuracao_na_linha = False

        # THROTTLING DA UI
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
            cv2.putText(p_inf, "CROP ESTATICO (VERMELHO=ESTOURO / AZUL=CRUSH)", (pos_x_zebra + 5, pos_y_zebra + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Monta a imagem final (restaurando o vstack)
        dashboard = np.vstack((np.hstack((p_live, p_bin)), p_inf))
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(dashboard, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/status')
def get_status():
    global GRAVANDO, contador_perfs_ciclo, frame_count, fps_real_proc, tempo_ms_ciclo
    global ROI_X, ROI_Y, ROI_W, ROI_H, CROP_W, CROP_H, OFFSET_X
    global foco_atual, shutter_speed, gain, fps_cam, THRESH_VAL, LINHA_GATILHO_Y, MARGEM_GATILHO
    
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
        "thresh": THRESH_VAL,
        "roi_x": ROI_X, "roi_y": ROI_Y, "roi_w": ROI_W, "roi_h": ROI_H,
        "crop_w": CROP_W, "crop_h": CROP_H, "ox": OFFSET_X,
        "gatilho_y": LINHA_GATILHO_Y, "margem": MARGEM_GATILHO
    }

@app.route('/')
def index():
    return """
    <html><body style='background:#0a0a0a; color:#eee; font-family:monospace; margin:0;'>
        
        <div style='display:flex; background:#111; padding:10px; border-bottom:1px solid #333; justify-content:space-around; font-size:16px;'>
            <span id='m'>--</span> | 
            CICLO: <b id='c'>0/4</b> |
            DISCO: <b id='arq' style='color:#0f0'>0</b> imgs (<b id='esp' style='color:#0aa'>-</b> livres) | 
            FRAMES: <b id='f'>0</b> | 
            PROC: <b id='fps_proc' style='color:#0ff'>0.0 FPS</b> (<b id='ms_ciclo' style='color:#ff0'>0.0 ms</b>) | 
            QUEUE: <b id='q' style='color:#f0f'>0</b>/30 |
            TEMP: <b id='t_cpu' style='color:#f90'>0.0 °C</b>
        </div>
        
        <div style='display:flex; background:#1a1a1a; padding:6px 10px; border-bottom:1px solid #444; justify-content:space-between; font-size:12px; color:#aaa;'>
            <span><b>ÓPTICA:</b> Foco <span id='v_foco' style='color:#fff'>-</span> | Exp <span id='v_exp' style='color:#fff'>-</span> | Gain <span id='v_gain' style='color:#fff'>-</span> | Cam <span id='v_fps_cam' style='color:#fff'>-</span>fps</span>
            <span><b>VISÃO:</b> Thresh <span id='v_thresh' style='color:#fff'>-</span> | Gatilho Y:<span id='v_gatilho' style='color:#fff'>-</span> &plusmn;<span id='v_margem' style='color:#fff'>-</span></span>
            <span><b>GEOMETRIA:</b> ROI(X:<span id='v_rx' style='color:#fff'>-</span> Y:<span id='v_ry' style='color:#fff'>-</span> W:<span id='v_rw' style='color:#fff'>-</span> H:<span id='v_rh' style='color:#fff'>-</span>) | Crop(W:<span id='v_cw' style='color:#fff'>-</span> H:<span id='v_ch' style='color:#fff'>-</span>) | OX:<span id='v_ox' style='color:#fff'>-</span></span>
        </div>

        <div style='display:flex; height:88vh;'>
            <div style='flex:2; border-right:1px solid #333;'><img src="/video_feed" style="width:100%;"></div>
            <div style='flex:1; background:#000;'><img src="/preview_feed" style="width:100%; border:1px solid #0f0;"></div>
        </div>
        
        <script>
            setInterval(() => {
                fetch('/status').then(r => r.json()).then(d => {
                    // Atualiza Barra Principal
                    const m = document.getElementById('m'); m.innerText = d.rec; m.style.color = d.cor;
                    document.getElementById('c').innerText = d.ciclo; 
                    document.getElementById('f').innerText = d.total;
                    document.getElementById('arq').innerText = d.arquivos;
                    document.getElementById('esp').innerText = d.espaco;
                    document.getElementById('fps_proc').innerText = d.fps_proc; 
                    document.getElementById('ms_ciclo').innerText = d.ms_ciclo;
                    document.getElementById('q').innerText = d.queue;
                    document.getElementById('t_cpu').innerText = d.temp; // <--- INJETA A TEMPERATURA AQUI
                    
                    // Atualiza Barra de Telemetria
                    document.getElementById('v_foco').innerText = d.foco;
                    document.getElementById('v_exp').innerText = d.exp;
                    document.getElementById('v_gain').innerText = d.gain;
                    document.getElementById('v_fps_cam').innerText = d.fps_cam;
                    document.getElementById('v_thresh').innerText = d.thresh;
                    document.getElementById('v_gatilho').innerText = d.gatilho_y;
                    document.getElementById('v_margem').innerText = d.margem;
                    document.getElementById('v_rx').innerText = d.roi_x;
                    document.getElementById('v_ry').innerText = d.roi_y;
                    document.getElementById('v_rw').innerText = d.roi_w;
                    document.getElementById('v_rh').innerText = d.roi_h;
                    document.getElementById('v_cw').innerText = d.crop_w;
                    document.getElementById('v_ch').innerText = d.crop_h;
                    document.getElementById('v_ox').innerText = d.ox;
                });
            }, 250);
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