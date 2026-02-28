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

picam2 = Picamera2()

# 2. Configuração de Hardware (720p para detalhamento de perfuração)
WIDTH, HEIGHT = 1280, 720
config = picam2.create_video_configuration(main={"size": (WIDTH, HEIGHT), "format": "RGB888"})
picam2.configure(config)

# Valores iniciais
shutter_speed = 20000
gain = 1.0
fps = 30

picam2.set_controls({"ExposureTime": shutter_speed, "AnalogueGain": gain, "FrameRate": fps})
picam2.start()

# --- GEOMETRIA DINÂMICA (Pode ser alterada via teclado) ---
ROI_X, ROI_Y = 400, 50
ROI_W, ROI_H = 480, 150  
LINHA_X, MARGEM = 640, 20
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
    print("  MINIOLA CONTROL v2.6 - COMANDOS AO VIVO")
    print("="*45)
    print("  MOVER ROI:   w/s (Cima/Baixo) | a/d (Esq/Dir)")
    print("  GATILHO:     < / >  (Mover linha vermelha)")
    print("  IMAGEM:      e/g (Exp/Gain) | t (Thresh) | o (Otsu)")
    print("  CAPTURA:     S (Start) | P (Pause) | R (Reset)")
    print("="*45)

    while True:
        try:
            entrada = input(">> ").split()
            if not entrada: continue
            cmd = entrada[0].lower()
            
            # Movimentação da ROI (Passos de 10px)
            if cmd == 'w': ROI_Y = max(0, ROI_Y - 10)
            elif cmd == 's': ROI_Y = min(HEIGHT - ROI_H, ROI_Y + 10)
            elif cmd == 'a': ROI_X = max(0, ROI_X - 10)
            elif cmd == 'd': ROI_X = min(WIDTH - ROI_W, ROI_X + 10)
            
            # Movimentação da Linha de Gatilho
            elif cmd == '>': LINHA_X = min(ROI_X + ROI_W, LINHA_X + 5)
            elif cmd == '<': LINHA_X = max(ROI_X, LINHA_X - 5)

            # Ajustes Técnicos
            elif cmd == 'o': # Auto-Otsu
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
            
            # Controle de Sessão
            elif cmd == 'S': 
                if not os.path.exists("captura"): os.makedirs("captura")
                modo_gravacao = True; print("GRAVANDO...")
            elif cmd == 'P': modo_gravacao = False; print("PAUSADO.")
            elif cmd == 'R': contador_perf = frame_count = 0; print("RESETADO.")

        except Exception as e: print(f"Erro: {e}")

# --- THREAD: LÓGICA DO SCANNER ---

def logica_scanner():
    global contador_perf, frame_count, furo_na_linha, ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug
    while True:
        frame_raw = picam2.capture_array()
        if frame_raw is None: continue
            
        gray = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2GRAY)
        roi = gray[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
        
        _, binary = cv2.threshold(roi, THRESH_VAL, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        furo_agora = False
        temp_contornos = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if 100 < area < 10000:
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
                                cv2.imwrite(f"captura/frame_{frame_count:06d}.jpg", fbgr)
            else:
                temp_contornos.append({'rect': (x, y, w, h), 'color': (0, 0, 255)})

        if not furo_agora: furo_na_linha = False
        ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug = frame_raw, binary, temp_contornos

# --- FLASK: VISIONAMENTO COM TELEMETRIA ---

def generate_frames():
    while True:
        if ultimo_frame_bruto is None:
            time.sleep(0.01); continue
        
        vis_base = cv2.cvtColor(ultimo_frame_bruto, cv2.COLOR_RGB2BGR)
        
        # Desenha Telemetria (ROI e Gatilho)
        cor_gat = (0, 255, 0) if furo_na_linha else (0, 0, 255)
        cv2.rectangle(vis_base, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (150, 150, 150), 2)
        cv2.line(vis_base, (LINHA_X, ROI_Y), (LINHA_X, ROI_Y + ROI_H), cor_gat, 3)
        
        for item in lista_contornos_debug:
            x, y, w, h = item['rect']
            cv2.rectangle(vis_base, (x + ROI_X, y + ROI_Y), (x + w + ROI_X, y + h + ROI_Y), item['color'], 2)
        
        # Rotação Vertical
        vis = cv2.rotate(vis_base, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Overlays de Status
        cor_m = (0, 255, 0) if modo_gravacao else (0, 255, 255)
        cv2.putText(vis, f"MODO: {'GRAVANDO' if modo_gravacao else 'VISIONAMENTO'}", (20, 40), 1, 1.5, cor_m, 2)
        cv2.putText(vis, f"PERF: {contador_perf} | FR: {frame_count}", (20, 80), 1, 1.5, (255,255,255), 2)

        ret, buffer = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """<html><body style='background:#111; color:#0f0; text-align:center; font-family:monospace;'>
              <h2>MINIOLA v2.6</h2><img src="/video_feed" style="height:85vh; border:1px solid #444;">
              </body></html>"""

if __name__ == '__main__':
    threading.Thread(target=painel_controle, daemon=True).start()
    threading.Thread(target=logica_scanner, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)