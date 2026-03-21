import sys
from unittest.mock import MagicMock
import os
import shutil

# --- MOCKS ---
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

CAPTURE_PATH = 'capturas'
if not os.path.exists(CAPTURE_PATH): os.makedirs(CAPTURE_PATH)

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1080, 720), "format": "RGB888"})
picam2.configure(config)
picam2.set_controls({"ExposureTime": 500, "AnalogueGain": 1.0, "FrameRate": 60, "LensPosition": 15.0})
picam2.start()

# --- GEOMETRIA ---
GRAVANDO = False
ESTADO_ATUAL = "BUSCAR_QUADRO"
ROI_X, ROI_Y = 215, 50
ROI_W, ROI_H = 80, 600
LINHA_RESET_Y = 100  # Gatilho Y
THRESH_VAL = 110
OFFSET_X = 260
CROP_W, CROP_H = 440, 330

frame_count = 0
ultimo_frame_bruto = None
ultimo_frame_binario = None
ultimo_crop_disco = np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
lista_contornos_debug = []
pos_ancora_debug = None

# --- LÓGICA DE CAPTURA ---
def processar_captura(frame, cx, cy, n_frame):
    global OFFSET_X, CROP_W, CROP_H, GRAVANDO
    fx, fy = cx + OFFSET_X, cy
    x1, y1 = max(0, int(fx - (CROP_W // 2))), max(0, int(fy - (CROP_H // 2)))
    x2, y2 = min(frame.shape[1], x1 + CROP_W), min(frame.shape[0], y1 + CROP_H)
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size > 0 and GRAVANDO:
        caminho = os.path.join(CAPTURE_PATH, f"frame_{n_frame:06d}.jpg")
        cv2.imwrite(caminho, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return (x1, y1, x2, y2)

# --- TERMINAL ---
def painel_controle():
    global frame_count, GRAVANDO, LINHA_RESET_Y, ROI_X, ROI_Y, OFFSET_X, THRESH_VAL
    time.sleep(2)
    print("\n" + "═"*45)
    print("   MINIOLA v3.9.2 - PREVIEW DE DISCO ATIVO")
    print("═"*45)
    print("   ly [v] : Ajustar Gatilho Y (Vermelho)")
    print("   rec    : Iniciar Gravação")
    print("   clean  : Limpar Capturas")
    print("═"*45)

    while True:
        try:
            entrada = input("\nComando >> ").split()
            if not entrada: continue
            cmd, val = entrada[0].lower(), int(entrada[1]) if len(entrada) > 1 else 0
            if cmd == 'ly': LINHA_RESET_Y = val
            elif cmd == 'rx': ROI_X = val
            elif cmd == 'ry': ROI_Y = val
            elif cmd == 'ox': OFFSET_X = val
            elif cmd == 't': THRESH_VAL = val
            elif cmd == 'rec': GRAVANDO = not GRAVANDO
            elif cmd == 'r': frame_count = 0
            elif cmd == 'clean':
                shutil.rmtree(CAPTURE_PATH); os.makedirs(CAPTURE_PATH)
                print(">> Capturas apagadas.")
        except: pass

# --- LOOP DE VISÃO ---
def logica_scanner():
    global frame_count, ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug
    global ESTADO_ATUAL, pos_ancora_debug, LINHA_RESET_Y

    while True:
        frame_raw = picam2.capture_array()
        if frame_raw is None: continue
        
        gray = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2GRAY)
        roi = gray[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
        _, binary = cv2.threshold(roi, THRESH_VAL, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        perfs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if 150 < area < 9000:
                perfs.append({'cx': x + (w//2) + ROI_X, 'cy': y + (h//2) + ROI_Y})
        
        perfs.sort(key=lambda p: p['cy'])

        # Atualiza âncora visual sempre
        if len(perfs) >= 4:
            grupo = perfs[0:4]
            cx_a, cy_a = int(np.mean([p['cx'] for p in grupo])), int(np.mean([p['cy'] for p in grupo]))
            pos_ancora_debug = (cx_a, cy_a)

            if ESTADO_ATUAL == "BUSCAR_QUADRO":
                processar_captura(frame_raw, cx_a, cy_a, frame_count)
                frame_count += 1
                ESTADO_ATUAL = "ESPERAR_SAIDA"
        
        # Lógica de Reset: Se a perfuração mais alta subir além da linha de gatilho
        if ESTADO_ATUAL == "ESPERAR_SAIDA" and len(perfs) > 0:
            if perfs[0]['cy'] < (ROI_Y + LINHA_RESET_Y): # Se o furo cruzar a linha p/ cima
                ESTADO_ATUAL = "BUSCAR_QUADRO"

        ultimo_frame_bruto, ultimo_frame_binario = frame_raw, binary
        time.sleep(0.005)

# --- DASHBOARD COM PREVIEW DE DISCO ---
def generate_dashboard():
    global ultimo_crop_disco
    while True:
        if ultimo_frame_bruto is None: time.sleep(0.1); continue
        
        # 1. Topo: Live e Binário
        p_live = cv2.resize(ultimo_frame_bruto.copy(), (640, 420))
        sx, sy = 640/1080, 420/720
        cv2.rectangle(p_live, (int(ROI_X*sx), int(ROI_Y*sy)), (int((ROI_X+ROI_W)*sx), int((ROI_Y+ROI_H)*sy)), (150, 150, 150), 1)
        cv2.line(p_live, (int(ROI_X*sx), int((ROI_Y+LINHA_RESET_Y)*sy)), (int((ROI_X+ROI_W)*sx), int((ROI_Y+LINHA_RESET_Y)*sy)), (0,0,255), 2)
        if pos_ancora_debug:
            cv2.drawMarker(p_live, (int(pos_ancora_debug[0]*sx), int(pos_ancora_debug[1]*sy)), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)

        p_bin = np.zeros((420, 640, 3), dtype=np.uint8)
        bin_z = cv2.resize(cv2.cvtColor(ultimo_frame_binario, cv2.COLOR_GRAY2RGB), (240, 420))
        p_bin[0:420, 200:440] = bin_z

        # 2. Base: Preview dos arquivos Gravados (24fps simulado)
        p_prev = np.zeros((480, 1280, 3), dtype=np.uint8)
        files = sorted([f for f in os.listdir(CAPTURE_PATH) if f.endswith('.jpg')])
        
        if files:
            # Pega o último arquivo gravado para mostrar no preview
            img_path = os.path.join(CAPTURE_PATH, files[-1])
            try:
                img_disk = cv2.imread(img_path)
                if img_disk is not None:
                    h, w = img_disk.shape[:2]
                    nw = int(460 * (w/h))
                    res = cv2.resize(img_disk, (nw, 460))
                    p_prev[10:470, (1280-nw)//2 : (1280-nw)//2 + nw] = res
            except: pass

        cor = (0, 0, 255) if GRAVANDO else (0, 255, 0)
        cv2.putText(p_prev, f"{'REC' if GRAVANDO else 'READY'} - FRAMES: {len(files)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor, 3)

        dashboard = np.vstack((np.hstack((p_live, p_bin)), p_prev))
        _, buffer = cv2.imencode('.jpg', dashboard, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed(): return Response(generate_dashboard(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status(): return f"STATUS: {ESTADO_ATUAL} | GATILHO Y: {ROI_Y + LINHA_RESET_Y}"

@app.route('/')
def index():
    return """
    <html><body style='background:#000; color:#0f0; text-align:center; font-family:monospace; margin:0;'>
    <div style='background:#111; padding:10px;'><span id='st'>...</span></div>
    <img src="/video_feed" style="height:90vh;">
    <script>setInterval(() => { fetch('/status').then(r => r.text()).then(t => { document.getElementById('st').innerText = t; }); }, 200);</script>
    </body></html>
    """

if __name__ == '__main__':
    threading.Thread(target=painel_controle, daemon=True).start()
    threading.Thread(target=logica_scanner, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)