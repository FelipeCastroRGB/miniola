from flask import Flask, Response
from picamera2 import Picamera2
from picamera2.previews import NullPreview
import cv2
import numpy as np
import threading
import sys
import time
import logging
import os

# 1. Configuração e Silenciamento de Logs
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR) 

# Caminho para o RAM Drive
CAPTURE_PATH = "/home/felipe/miniola_py/captura"
if not os.path.exists(CAPTURE_PATH):
    os.makedirs(CAPTURE_PATH)

# 2. Inicialização da Câmera (Modo Headless)
picam2 = Picamera2()
WIDTH, HEIGHT = 800, 600
config = picam2.create_video_configuration(main={"size": (WIDTH, HEIGHT), "format": "RGB888"})
picam2.configure(config)

shutter_speed, gain, fps = 2000, 1.0, 30
picam2.set_controls({"ExposureTime": shutter_speed, "AnalogueGain": gain, "FrameRate": fps})
picam2.start_preview(NullPreview()) # Evita erros de driver de monitor (KMS)

# --- ESTADO GLOBAL E GEOMETRIA ---
ROI_X, ROI_Y, ROI_W, ROI_H = 250, 40, 300, 120  
LINHA_X, MARGEM, THRESH_VAL = 400, 15, 110
contador_perf, frame_count = 0, 0
furo_na_linha, modo_gravacao = False, False

# Sincronização de Threads
frame_lock = threading.Lock() # Trava para evitar tela preta
ultimo_frame_bruto = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
lista_contornos_debug = []

# --- THREAD: PAINEL DE CONTROLE ---

def painel_controle():
    global contador_perf, frame_count, modo_gravacao, THRESH_VAL
    global ROI_X, ROI_Y, LINHA_X, shutter_speed, gain
    
    print("\n" + "="*45)
    print("  MINIOLA CONTROL v2.8.8 - RECONSTRUÍDO")
    print("="*45)
    print("  MOVER ROI:   w/s (C/B) | a/d (E/D)")
    print("  GATILHO:     < / >  (Linha vermelha)")
    print("  IMAGEM:      e/g (Exp/Gain) | t (Thresh) | o (Otsu)")
    print("  CAPTURA:     S (Start) | P (Pause) | R (Reset)")
    print("="*45)

    while True:
        try:
            entrada = input(">> ").split()
            if not entrada: continue
            cmd = entrada[0]
            
            # Movimentação
            if cmd == 'w': ROI_Y = max(0, ROI_Y - 5)
            elif cmd == 's': ROI_Y = min(HEIGHT - ROI_H, ROI_Y + 5)
            elif cmd == 'a': ROI_X = max(0, ROI_X - 5)
            elif cmd == 'd': ROI_X = min(WIDTH - ROI_W, ROI_X + 5)
            elif cmd == '>': LINHA_X = min(ROI_X + ROI_W, LINHA_X + 5)
            elif cmd == '<': LINHA_X = max(ROI_X, LINHA_X - 5)
            
            # Ajustes de Imagem
            elif cmd == 'e' and len(entrada) > 1:
                shutter_speed = int(entrada[1])
                picam2.set_controls({"ExposureTime": shutter_speed})
            elif cmd == 'g' and len(entrada) > 1:
                gain = float(entrada[1])
                picam2.set_controls({"AnalogueGain": gain})
            elif cmd == 't' and len(entrada) > 1:
                THRESH_VAL = int(entrada[1])
                
            # Controle de Gravação (Case Sensitive conforme v2.7)
            elif cmd == 'S': modo_gravacao = True; print("GRAVANDO...")
            elif cmd == 'P': modo_gravacao = False; print("PAUSADO.")
            elif cmd == 'R': contador_perf = frame_count = 0; print("RESETADO.")
        except Exception as e: print(f"Erro no console: {e}")

# --- THREAD: LÓGICA DO SCANNER ---

def logica_scanner():
    global contador_perf, frame_count, furo_na_linha, ultimo_frame_bruto, lista_contornos_debug
    while True:
        frame_raw = picam2.capture_array()
        if frame_raw is None: 
            time.sleep(0.01); continue
            
        # Atualização segura do frame para o Flask
        with frame_lock:
            ultimo_frame_bruto = frame_raw.copy()
            
        # Processamento de Imagem
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
                                cv2.imwrite(f"{CAPTURE_PATH}/frame_{frame_count:06d}.jpg", fbgr)
            else:
                temp_contornos.append({'rect': (x, y, w, h), 'color': (0, 0, 255)})

        if not furo_agora: furo_na_linha = False
        
        with frame_lock:
            lista_contornos_debug = temp_contornos
            
        time.sleep(0.005) 

# --- FLASK: VISIONAMENTO SEGURO ---

def generate_frames():
    while True:
        # Pega uma cópia estática e segura do frame [cite: 2026-02-28]
        with frame_lock:
            img_local = ultimo_frame_bruto.copy()
            contornos_local = lista_contornos_debug.copy()
            furo_local = furo_na_linha
            modo_local = modo_gravacao
            perf_local = contador_perf
            fr_local = frame_count

        vis = cv2.cvtColor(img_local, cv2.COLOR_RGB2BGR)
        
        # Desenhos Telemetria
        cor_gat = (0, 255, 0) if furo_local else (0, 0, 255)
        cv2.rectangle(vis, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (150, 150, 150), 2)
        cv2.line(vis, (LINHA_X, ROI_Y), (LINHA_X, ROI_Y + ROI_H), cor_gat, 3)
        
        for item in contornos_local:
            x, y, w, h = item['rect']
            cv2.rectangle(vis, (x + ROI_X, y + ROI_Y), (x + w + ROI_X, y + h + ROI_Y), item['color'], 2)
        
        vis = cv2.rotate(vis, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Overlays de Status
        cor_status = (0, 255, 0) if modo_local else (0, 255, 255)
        cv2.putText(vis, f"MODO: {'GRAVANDO' if modo_local else 'VISIONAMENTO'}", (20, 35), 1, 1.2, cor_status, 2)
        cv2.putText(vis, f"PERF: {perf_local} | FR: {fr_local}", (20, 70), 1, 1.2, (255,255,255), 2)

        ret, buffer = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.04) # 25 FPS para aliviar a rede e a CPU do Pi

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """<html><body style='background:#000; color:#0f0; text-align:center; font-family:monospace;'>
              <h3>MINIOLA v2.8.8</h3><img src="/video_feed" style="height:88vh; border:1px solid #333;">
              </body></html>"""

if __name__ == '__main__':
    threading.Thread(target=painel_controle, daemon=True).start()
    threading.Thread(target=logica_scanner, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)