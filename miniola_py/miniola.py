from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import threading
import sys
import time
import logging
import os

# 1. Configuração e Silenciamento
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR) 

# --- CONFIGURAÇÃO DE CAMINHO (RAM DRIVE) ---
# Usamos o caminho absoluto para garantir a gravação no tmpfs (RAM) [cite: 2026-02-28]
CAPTURE_PATH = "/home/felipe/miniola_py/captura"
if not os.path.exists(CAPTURE_PATH):
    os.makedirs(CAPTURE_PATH)

picam2 = Picamera2()

# 2. Configuração de Hardware (800x600: O "Sweet Spot" para RPi)
WIDTH, HEIGHT = 800, 600
config = picam2.create_video_configuration(main={"size": (WIDTH, HEIGHT), "format": "RGB888"})
picam2.configure(config)

# Valores iniciais (Recuperados da v2.7 funcional)
shutter_speed = 10000  # 10ms
gain = 1.0
fps = 45

picam2.set_controls({"ExposureTime": shutter_speed, "AnalogueGain": gain, "FrameRate": fps})
picam2.start()

# --- GEOMETRIA DINÂMICA (Adaptada para 800x600) ---
ROI_X, ROI_Y = 250, 40
ROI_W, ROI_H = 300, 120  
LINHA_X, MARGEM = 400, 15
THRESH_VAL = 110

# Estado Global
contador_perf = 0
frame_count = 0
furo_na_linha = False
modo_gravacao = False
ultimo_frame_bruto = None
ultimo_frame_binario = None
lista_contornos_debug = []

# --- THREAD: PAINEL DE CONTROLE (Comandos Dinâmicos) ---

def painel_controle():
    global contador_perf, frame_count, modo_gravacao, THRESH_VAL, ultimo_frame_bruto
    global ROI_X, ROI_Y, LINHA_X, shutter_speed, gain
    
    print("\n" + "="*45)
    print("  MINIOLA CONTROL v2.9 - ESTABILIZADA")
    print("="*45)
    print("  MOVER ROI:   w/s (C/B) | a/d (E/D)")
    print("  GATILHO:     < / >  (Linha vermelha)")
    print("  IMAGEM:      e/g (Exp/Gain) | t (Thresh) | o (Otsu)")
    print("  CAPTURA:     f (Gravar) | p (Pausar) | r (Reset)")
    print("="*45)

    while True:
        try:
            entrada = input(">> ").split()
            if not entrada: continue
            cmd = entrada[0].lower() # Aceita tanto maiúsculo quanto minúsculo [cite: 2026-02-28]
            
            if cmd == 'w': ROI_Y = max(0, ROI_Y - 5)
            elif cmd == 's': ROI_Y = min(HEIGHT - ROI_H, ROI_Y + 5)
            elif cmd == 'a': ROI_X = max(0, ROI_X - 5)
            elif cmd == 'd': ROI_X = min(WIDTH - ROI_W, ROI_X + 5)
            elif cmd == '>': LINHA_X = min(ROI_X + ROI_W, LINHA_X + 5)
            elif cmd == '<': LINHA_X = max(ROI_X, LINHA_X - 5)
            elif cmd == 'o':
                if ultimo_frame_bruto is not None:
                    gray = cv2.cvtColor(ultimo_frame_bruto, cv2.COLOR_RGB2GRAY)
                    roi_a = gray[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
                    val, _ = cv2.threshold(roi_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    THRESH_VAL = int(val * 1.71)
                    print(f"[AUTO] Threshold: {THRESH_VAL}")
            elif cmd == 'e' and len(entrada) > 1:
                shutter_speed = int(entrada[1])
                picam2.set_controls({"ExposureTime": shutter_speed})
            elif cmd == 'g' and len(entrada) > 1:
                gain = float(entrada[1])
                picam2.set_controls({"AnalogueGain": gain})
            elif cmd == 't' and len(entrada) > 1:
                THRESH_VAL = int(entrada[1])
            
            # COMANDOS DE GRAVAÇÃO ATUALIZADOS
            elif cmd == 'f': 
                modo_gravacao = True; print("GRAVANDO NO RAM DRIVE...")
            elif cmd == 'p': 
                modo_gravacao = False; print("PAUSADO.")
            elif cmd == 'r': 
                contador_perf = frame_count = 0; print("ESTATÍSTICAS RESETADAS.")
        except Exception as e: print(f"Erro: {e}")

# --- THREAD: LÓGICA DO SCANNER (Processamento Otimizado) ---

def logica_scanner():
    global contador_perf, frame_count, furo_na_linha, ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug
    while True:
        frame_raw = picam2.capture_array()
        if frame_raw is None: 
            time.sleep(0.01)
            continue
            
        gray = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2GRAY)
        roi = gray[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
        
        _, binary = cv2.threshold(roi, THRESH_VAL, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        furo_agora = False
        temp_contornos = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if 50 < area < 5000:
                cx_global = x + ROI_X + (w // 2)
                temp_contornos.append({'rect': (x, y, w, h), 'color': (0, 255, 0)})
                if abs(cx_global - LINHA_X) < MARGEM:
                    furo_agora = True
                    if not furo_na_linha:
                        contador_perf += 1
                        furo_na_linha = True
                        if contador_perf % 4 == 0:
                            frame_count += 1
                            if modo_gravacao:
                                fbgr = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)
                                # Salvamento direto no caminho absoluto do RAM Drive [cite: 2026-02-28]
                                cv2.imwrite(f"{CAPTURE_PATH}/frame_{frame_count:06d}.jpg", fbgr)
            else:
                temp_contornos.append({'rect': (x, y, w, h), 'color': (0, 0, 255)})

        if not furo_agora: furo_na_linha = False
        ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug = frame_raw, binary, temp_contornos
        
        time.sleep(0.005) 

# --- FLASK: VISIONAMENTO COM TELEMETRIA ---

def generate_frames():
    while True:
        if ultimo_frame_bruto is None:
            time.sleep(0.1); continue
        
        # Cria uma cópia para evitar que o desenho ocorra enquanto a lógica do scanner atualiza [cite: 2025-12-23]
        vis_base = cv2.cvtColor(ultimo_frame_bruto, cv2.COLOR_RGB2BGR)
        
        cor_gat = (0, 255, 0) if furo_na_linha else (0, 0, 255)
        cv2.rectangle(vis_base, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (150, 150, 150), 2)
        cv2.line(vis_base, (LINHA_X, ROI_Y), (LINHA_X, ROI_Y + ROI_H), cor_gat, 3)
        
        for item in lista_contornos_debug:
            x, y, w, h = item['rect']
            cv2.rectangle(vis_base, (x + ROI_X, y + ROI_Y), (x + w + ROI_X, y + h + ROI_Y), item['color'], 2)
        
        vis = cv2.rotate(vis_base, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        cor_m = (0, 255, 0) if modo_gravacao else (0, 255, 255)
        cv2.putText(vis, f"MODO: {'GRAVANDO' if modo_gravacao else 'VISIONAMENTO'}", (20, 35), 1, 1.2, cor_m, 2)
        cv2.putText(vis, f"PERF: {contador_perf} | FR: {frame_count}", (20, 70), 1, 1.2, (255,255,255), 2)

        ret, buffer = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.01)

# --- ROTA: AO VIVO ROLO ---

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- NOVA ROTA: PREVIEW DO ROLO (Últimos frames gravados) ---

@app.route('/preview_feed')
def preview_feed():
    def generate_preview():
        while True:
            # Lista os últimos 48 frames (2 segundos a 24fps) para o loop de preview 
            files = sorted([f for f in os.listdir(CAPTURE_PATH) if f.endswith('.jpg')])
            last_frames = files[-48:] if len(files) > 0 else []
            
            if not last_frames:
                time.sleep(0.5); continue

            for frame_file in last_frames:
                path = os.path.join(CAPTURE_PATH, frame_file)
                try:
                    with open(path, "rb") as f:
                        frame_data = f.read()
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                    time.sleep(1/24) # Simula a cadência de 24fps 
                except:
                    continue
    return Response(generate_preview(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- INTERFACE ATUALIZADA (Painel Duplo) ---

@app.route('/')
def index():
    return """<html>
    <head><title>MINIOLA v3.0</title></head>
    <body style='background:#111; color:#0f0; text-align:center; font-family:monospace; margin:0; padding:10px;'>
        <div style="display: flex; justify-content: space-around; align-items: flex-start;">
            <div>
                <h3>AO VIVO (AJUSTE)</h3>
                <img src="/video_feed" style="height:75vh; border:2px solid #333;">
            </div>
            <div>
                <h3>PREVIEW (24 FPS)</h3>
                <img src="/preview_feed" style="height:75vh; border:2px solid #0f0;">
                <p style="color:#aaa;">Visualização do material no RAM Drive</p>
            </div>
        </div>
        <p>Controles via Console: 'f' Gravar | 'p' Pausar | 'r' Reset</p>
    </body></html>"""

if __name__ == '__main__':
    threading.Thread(target=painel_controle, daemon=True).start()
    threading.Thread(target=logica_scanner, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)