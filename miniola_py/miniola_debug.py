import sys
from unittest.mock import MagicMock
import os

# --- MOCKS PARA AMBIENTE SEM HARDWARE ---
sys.modules["pykms"] = MagicMock()
sys.modules["kms"] = MagicMock()

from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import threading
import time
import logging

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR) 

if not os.path.exists('capturas'): os.makedirs('capturas')

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1080, 720), "format": "RGB888"})
picam2.configure(config)
picam2.set_controls({"ExposureTime": 450, "AnalogueGain": 1.0, "FrameRate": 60, "LensPosition": 15.0})
picam2.start()

# --- GEOMETRIA E CONTROLE ---
GRAVANDO = False
ROI_X, ROI_Y = 215, 50
ROI_W, ROI_H = 80, 600
LINHA_RESET_Y = 150  # A linha de gatilho (Trigger Line)
THRESH_VAL = 110
OFFSET_X = 260
CROP_W, CROP_H = 440, 330 

# Variáveis de Estado de Contagem
contador_perfs_ciclo = 0
perfuracao_na_linha = False
frame_count = 0
ultimo_frame_bruto = None
ultimo_frame_binario = None
ultimo_crop_preview = np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
lista_contornos_debug = []
pos_ancora_debug = None

def processar_captura(frame, cx, cy, n_frame):
    global OFFSET_X, CROP_W, CROP_H, ultimo_crop_preview, GRAVANDO
    fx, fy = cx + OFFSET_X, cy
    x1, y1 = max(0, int(fx - (CROP_W // 2))), max(0, int(fy - (CROP_H // 2)))
    x2, y2 = min(frame.shape[1], x1 + CROP_W), min(frame.shape[0], y1 + CROP_H)
    crop = frame[y1:y2, x1:x2].copy()
    
    if crop.size > 0:
        ultimo_crop_preview = crop
        if GRAVANDO:
            cv2.imwrite(f"capturas/miniola_{n_frame:06d}.jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
    return (x1, y1, x2, y2)

def painel_controle():
    global frame_count, GRAVANDO, LINHA_RESET_Y, ROI_X, ROI_Y, ROI_W, ROI_H, OFFSET_X, CROP_W, CROP_H, THRESH_VAL, contador_perfs_ciclo
    time.sleep(2)
    print("\n" + "═"*45)
    print("   MINIOLA v6.0 - CONTADOR DE 4 PERFURAÇÕES")
    print("═"*45)
    print("   ly [v] : Linha de Gatilho (Contagem)")
    print("   t [v]  : Threshold (Contraste)")
    print("   ox [v] : Offset X (Horizontal)")
    print("   rec    : Iniciar/Parar Gravação")
    print("   r      : Resetar contador de frames")
    print("═"*45)

    while True:
        try:
            entrada = input("\nComando >> ").split()
            if not entrada: continue
            cmd, val = entrada[0].lower(), int(entrada[1]) if len(entrada) > 1 else 0
            
            if cmd == 't': THRESH_VAL = val
            elif cmd == 'ly': LINHA_RESET_Y = val
            elif cmd == 'ox': OFFSET_X = val
            elif cmd == 'cw': CROP_W = val
            elif cmd == 'ch': CROP_H = val
            elif cmd == 'rx': ROI_X = val
            elif cmd == 'ry': ROI_Y = val
            elif cmd == 'rec': GRAVANDO = not GRAVANDO
            elif cmd == 'r': 
                frame_count = 0
                contador_perfs_ciclo = 0
            elif cmd == 'clean':
                for f in os.listdir('capturas'): os.remove(os.path.join('capturas', f))
        except: pass

def logica_scanner():
    global frame_count, ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug
    global contador_perfs_ciclo, perfuracao_na_linha, pos_ancora_debug

    while True:
        frame_raw = picam2.capture_array()
        if frame_raw is None: continue
        
        gray = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2GRAY)
        roi = gray[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
        _, binary = cv2.threshold(roi, THRESH_VAL, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        perfs_neste_frame, debug_visual = [], []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            # Filtro de forma baseado em padrões de preservação audiovisual
            if 150 < area < 9000 and 0.4 < (w/h) < 2.5:
                cx, cy = x + (w//2) + ROI_X, y + (h//2) + ROI_Y
                perfs_neste_frame.append({'cx': cx, 'cy': cy, 'y_topo': y + ROI_Y})
                debug_visual.append({'rect': (x+ROI_X, y+ROI_Y, w, h), 'color': (0, 255, 0)})

        # Ordena do topo para o fundo
        perfs_neste_frame.sort(key=lambda p: p['cy'])

        # --- LÓGICA DE CONTAGEM E GATILHO ---
        line_y_abs = ROI_Y + LINHA_RESET_Y
        
        # Detecta se a perfuração mais alta cruzou a linha (subindo)
        if perfs_neste_frame:
            topo_mais_alto = perfs_neste_frame[0]['cy']
            
            # Se a perfuração cruzou a linha para cima
            if topo_mais_alto < line_y_abs and not perfuracao_na_linha:
                contador_perfs_ciclo += 1
                perfuracao_na_linha = True # Trava para não contar o mesmo furo várias vezes
                
                # Se completou o ciclo de 4 furos (1 quadro SMPTE)
                if contador_perfs_ciclo >= 4:
                    if len(perfs_neste_frame) >= 4:
                        # Calcula o centroide das 4 perfs atuais para estabilidade
                        grupo = perfs_neste_frame[0:4]
                        cx_a = int(np.mean([p['cx'] for p in grupo]))
                        cy_a = int(np.mean([p['cy'] for p in grupo]))
                        pos_ancora_debug = (cx_a, cy_a)
                        
                        processar_captura(frame_raw, cx_a, cy_a, frame_count)
                        frame_count += 1
                    
                    contador_perfs_ciclo = 0 # Reinicia o ciclo
            
            # Reset da trava: só permite contar o próximo furo quando o atual sair da zona da linha
            elif topo_mais_alto > (line_y_abs + 20): # Margem de "gap filling"
                perfuracao_na_linha = False

        ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug = frame_raw, binary, debug_visual
        time.sleep(0.002)

def generate_dashboard():
    while True:
        if ultimo_frame_bruto is None: time.sleep(0.1); continue
        p_live = cv2.resize(ultimo_frame_bruto.copy(), (640, 420))
        sx, sy = 640/1080, 420/720
        
        # ROI e Linha de Gatilho
        cv2.rectangle(p_live, (int(ROI_X*sx), int(ROI_Y*sy)), (int((ROI_X+ROI_W)*sx), int((ROI_Y+ROI_H)*sy)), (150, 150, 150), 1)
        y_gl = ROI_Y + LINHA_RESET_Y
        cv2.line(p_live, (int(ROI_X*sx), int(y_gl*sy)), (int((ROI_X+ROI_W)*sx), int(y_gl*sy)), (0, 0, 255), 2)
        
        for item in lista_contornos_debug:
            x, y, w, h = item['rect']
            cv2.rectangle(p_live, (int(x*sx), int(y*sy)), (int((x+w)*sx), int((y+h)*sy)), item['color'], 2)
        
        if pos_ancora_debug:
            cv2.drawMarker(p_live, (int(pos_ancora_debug[0]*sx), int(pos_ancora_debug[1]*sy)), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
            fx_c, fy_c = pos_ancora_debug[0] + OFFSET_X, pos_ancora_debug[1]
            cx1, cy1 = int((fx_c - CROP_W//2)*sx), int((fy_c - CROP_H//2)*sy)
            cx2, cy2 = int((fx_c + CROP_W//2)*sx), int((fy_c + CROP_H//2)*sy)
            cv2.rectangle(p_live, (cx1, cy1), (cx2, cy2), (255, 255, 0), 1)

        p_bin = np.zeros((420, 640, 3), dtype=np.uint8)
        bin_z = cv2.resize(cv2.cvtColor(ultimo_frame_binario, cv2.COLOR_GRAY2RGB), (240, 420))
        p_bin[0:420, 200:440] = bin_z

        # Preview do Crop Final
        p_prev = np.zeros((480, 1280, 3), dtype=np.uint8)
        if ultimo_crop_preview is not None:
            hc, wc = ultimo_crop_preview.shape[:2]
            nw = int(460 * (wc/hc))
            p_prev[10:470, (1280-nw)//2 : (1280-nw)//2 + nw] = cv2.resize(ultimo_crop_preview, (nw, 460))
        
        cor_rec = (0, 0, 255) if GRAVANDO else (0, 255, 0)
        # Exibe o contador de ciclo (1 a 4) para feedback visual
        txt_rec = f"REC: {frame_count:05d} | CICLO: {contador_perfs_ciclo}/4"
        cv2.putText(p_prev, txt_rec, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor_rec, 3)

        dashboard = np.vstack((np.hstack((p_live, p_bin)), p_prev))
        _, buffer = cv2.imencode('.jpg', dashboard, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed(): return Response(generate_dashboard(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status(): return f"Furos no Ciclo: {contador_perfs_ciclo} | Total Frames: {frame_count}"

@app.route('/')
def index():
    return """
    <html><body style='background:#000; color:#0f0; text-align:center; font-family:monospace; margin:0;'>
    <div style='background:#111; padding:10px;'><span id='st'>...</span></div>
    <img src="/video_feed" style="height:92vh; border:1px solid #333;">
    <script>setInterval(() => { fetch('/status').then(r => r.text()).then(t => { document.getElementById('st').innerText = t; }); }, 150);</script>
    </body></html>
    """

if __name__ == '__main__':
    try:
        threading.Thread(target=painel_controle, daemon=True).start()
        threading.Thread(target=logica_scanner, daemon=True).start()
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except: picam2.stop()