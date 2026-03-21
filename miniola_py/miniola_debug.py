import sys
from unittest.mock import MagicMock
import os

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
LINHA_RESET_Y = 300  # Linha de contagem (Odômetro)
THRESH_VAL = 110
OFFSET_X = 260
CROP_W, CROP_H = 440, 330 

# Variáveis do Odômetro
perfs_contadas = 0
ids_vistos = set() # Para não contar a mesma perfuração duas vezes
frame_count = 0
ultimo_frame_bruto = None
ultimo_frame_binario = None
ultimo_crop_salvo = np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
lista_contornos_debug = []
pos_ancora_debug = None

def processar_captura(frame, cx, cy, n_frame):
    global OFFSET_X, CROP_W, CROP_H, ultimo_crop_salvo, GRAVANDO
    fx, fy = cx + OFFSET_X, cy
    x1, y1 = max(0, int(fx - (CROP_W // 2))), max(0, int(fy - (CROP_H // 2)))
    x2, y2 = min(frame.shape[1], x1 + CROP_W), min(frame.shape[0], y1 + CROP_H)
    crop = frame[y1:y2, x1:x2].copy()
    
    if crop.size > 0:
        ultimo_crop_salvo = crop
        if GRAVANDO:
            cv2.imwrite(f"capturas/miniola_{n_frame:06d}.jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
    return (x1, y1, x2, y2)

def painel_controle():
    global frame_count, GRAVANDO, LINHA_RESET_Y, ROI_X, ROI_Y, OFFSET_X, perfs_contadas
    time.sleep(2)
    print("\n" + "═"*45)
    print("   MINIOLA v4.0 - ODÔMETRO DE 4 PERFURAÇÕES")
    print("═"*45)
    print("   ly [v] : Posicao da Linha de Contagem")
    print("   rec    : Toggle Gravação")
    print("   r      : Zerar Contador de Frames")
    print("   p 0    : Zerar Odômetro de Perfs")
    print("═"*45)

    while True:
        try:
            entrada = input("\nComando >> ").split()
            if not entrada: continue
            cmd, val = entrada[0].lower(), int(entrada[1]) if len(entrada) > 1 else 0
            if cmd == 'ly': LINHA_RESET_Y = val
            elif cmd == 'ox': OFFSET_X = val
            elif cmd == 'rec': GRAVANDO = not GRAVANDO
            elif cmd == 'p': perfs_contadas = val
            elif cmd == 'r': frame_count = 0
            elif cmd == 'clean':
                for f in os.listdir('capturas'): os.remove(os.path.join('capturas', f))
        except: pass

def logica_scanner():
    global frame_count, ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug
    global perfs_contadas, ids_vistos, pos_ancora_debug

    while True:
        frame_raw = picam2.capture_array()
        if frame_raw is None: continue
        
        gray = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2GRAY)
        roi = gray[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
        _, binary = cv2.threshold(roi, THRESH_VAL, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        perfs_neste_frame = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if 150 < area < 9000 and 0.4 < (w/h) < 2.5:
                # Criamos um ID baseado na posição Y aproximada para rastrear a perf
                cx, cy = x + (w//2) + ROI_X, y + (h//2) + ROI_Y
                perfs_neste_frame.append({'cx': cx, 'cy': cy, 'id': cy // 5}) # ID rústico por slot de 5px

        perfs_neste_frame.sort(key=lambda p: p['cy'])

        # --- LÓGICA DO ODÔMETRO ---
        gatilho_y_absoluto = ROI_Y + LINHA_RESET_Y
        
        for p in perfs_neste_frame:
            # Se a perfuração está ACIMA da linha e ainda não foi contada neste ciclo
            if p['cy'] < gatilho_y_absoluto and p['id'] not in ids_vistos:
                perfs_contadas += 1
                ids_vistos.add(p['id'])
                print(f">> Perf Contada: {perfs_contadas}/4")

        # Limpa IDs que já sumiram do topo para permitir re-contagem no próximo ciclo
        ids_vistos = {pid for pid in ids_vistos if any(abs(p['id'] - pid) < 10 for p in perfs_neste_frame)}

        # --- DISPARO POR LOTE DE 4 ---
        if perfs_contadas >= 4:
            if len(perfs_neste_frame) >= 4:
                # Tira a foto baseada no grupo atual (centroide das 4 perfs em tela)
                grupo = perfs_neste_frame[0:4]
                cx_a, cy_a = int(np.mean([p['cx'] for p in grupo])), int(np.mean([p['cy'] for p in grupo]))
                pos_ancora_debug = (cx_a, cy_a)
                
                processar_captura(frame_raw, cx_a, cy_a, frame_count)
                frame_count += 1
                perfs_contadas = 0 # Reseta o odômetro
                ids_vistos.clear()

        ultimo_frame_bruto, ultimo_frame_binario = frame_raw, binary
        time.sleep(0.002)

def generate_dashboard():
    while True:
        if ultimo_frame_bruto is None: time.sleep(0.1); continue
        p_live = cv2.resize(ultimo_frame_bruto.copy(), (640, 420))
        sx, sy = 640/1080, 420/720
        
        # Linha de Contagem (Azul claro para diferenciar do antigo reset)
        y_gl = ROI_Y + LINHA_RESET_Y
        cv2.line(p_live, (int(ROI_X*sx), int(y_gl*sy)), (int((ROI_X+ROI_W)*sx), int(y_gl*sy)), (255, 255, 0), 2)
        cv2.putText(p_live, f"ODOM: {perfs_contadas}/4", (int(ROI_X*sx), int(y_gl*sy)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if pos_ancora_debug:
            cv2.drawMarker(p_live, (int(pos_ancora_debug[0]*sx), int(pos_ancora_debug[1]*sy)), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)

        # Dashboard e Preview
        p_bin = cv2.resize(cv2.cvtColor(ultimo_frame_binario, cv2.COLOR_GRAY2RGB), (640, 420))
        p_prev = np.zeros((480, 1280, 3), dtype=np.uint8)
        if ultimo_crop_salvo is not None:
            hc, wc = ultimo_crop_salvo.shape[:2]
            nw = int(460 * (wc/hc))
            p_prev[10:470, (1280-nw)//2 : (1280-nw)//2 + nw] = cv2.resize(ultimo_crop_salvo, (nw, 460))
        
        cor_rec = (0, 0, 255) if GRAVANDO else (0, 255, 0)
        cv2.putText(p_prev, f"REC: {frame_count:05d} | PERFS: {perfs_contadas}/4", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor_rec, 3)

        dashboard = np.vstack((np.hstack((p_live, p_bin)), p_prev))
        _, buffer = cv2.imencode('.jpg', dashboard, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed(): return Response(generate_dashboard(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status(): return f"CONTAGEM: {perfs_contadas}/4 | FRAMES: {frame_count}"

@app.route('/')
def index():
    return """
    <html><body style='background:#000; color:#0f0; text-align:center; font-family:monospace; margin:0;'>
    <div style='background:#111; padding:10px;'><span id='st'>...</span></div>
    <img src="/video_feed" style="height:92vh;">
    <script>setInterval(() => { fetch('/status').then(r => r.text()).then(t => { document.getElementById('st').innerText = t; }); }, 150);</script>
    </body></html>
    """

if __name__ == '__main__':
    threading.Thread(target=painel_controle, daemon=True).start()
    threading.Thread(target=logica_scanner, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)