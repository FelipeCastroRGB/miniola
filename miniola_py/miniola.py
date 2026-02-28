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

# 1. Configuração e Silenciamento
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR) 

# Caminho absoluto para o RAM Drive
BASE_DIR = "/home/felipe/miniola_py"
CAPTURE_PATH = os.path.join(BASE_DIR, "captura")
if not os.path.exists(CAPTURE_PATH):
    os.makedirs(CAPTURE_PATH)

picam2 = Picamera2()
WIDTH, HEIGHT = 800, 600
config = picam2.create_video_configuration(main={"size": (WIDTH, HEIGHT), "format": "RGB888"})
picam2.configure(config)

shutter_speed, gain, fps = 1000, 1.0, 30
picam2.set_controls({"ExposureTime": shutter_speed, "AnalogueGain": gain, "FrameRate": fps})
picam2.start_preview(NullPreview())

# --- GEOMETRIA E ESTADO ---
ROI_X, ROI_Y, ROI_W, ROI_H = 250, 40, 300, 120  
LINHA_X, MARGEM, THRESH_VAL = 400, 15, 110
contador_perf, frame_count = 0, 0
furo_na_linha, modo_gravacao = False, False
ultimo_frame_bruto = None

# --- MELHORIA: PAINEL DE CONTROLE VISÍVEL ---

def exibir_menu():
    print("\n" + "="*45)
    print("  MINIOLA CONTROL v2.8.4 - ACTIVE")
    print("="*45)
    print("  ROI:  [w,a,s,d] | GATILHO: [< , >]")
    print("  IMG:  [e] Exp  | [g] Gain | [t] Thresh")
    print("  OP:   [S] START | [P] PAUSE | [R] RESET")
    print("="*45)

def painel_controle():
    global contador_perf, frame_count, modo_gravacao, THRESH_VAL
    global ROI_X, ROI_Y, LINHA_X, shutter_speed, gain
    
    exibir_menu()

    while True:
        try:
            # .strip() remove espaços extras que podem vir do terminal
            entrada = input(">> ").strip().split()
            if not entrada: continue
            
            cmd_original = entrada[0]
            cmd = cmd_original.lower()
            
            # Movimentação do ROI (Apenas minúsculos)
            if cmd == 'w': ROI_Y = max(0, ROI_Y - 5)
            elif cmd == 's' and cmd_original == 's': ROI_Y = min(HEIGHT - ROI_H, ROI_Y + 5)
            elif cmd == 'a': ROI_X = max(0, ROI_X - 5)
            elif cmd == 'd': ROI_X = min(WIDTH - ROI_W, ROI_X + 5)
            
            # Comandos de GATILHO (Sensíveis a maiúsculas para evitar erro)
            elif cmd_original == 'S': 
                modo_gravacao = True
                print("\n[!] STATUS: GRAVAÇÃO INICIADA")
            elif cmd_original == 'P': 
                modo_gravacao = False
                print("\n[!] STATUS: GRAVAÇÃO PAUSADA")
            elif cmd_original == 'R': 
                contador_perf = frame_count = 0
                print("\n[!] STATUS: CONTADORES ZERADOS")
                
            # Ajustes de Imagem
            elif cmd == '>' : LINHA_X = min(ROI_X + ROI_W, LINHA_X + 5)
            elif cmd == '<' : LINHA_X = max(ROI_X, LINHA_X - 5)
            elif cmd == 't' and len(entrada) > 1:
                THRESH_VAL = int(entrada[1])
                print(f"Threshold ajustado: {THRESH_VAL}")
            elif cmd == 'e' and len(entrada) > 1:
                shutter_speed = int(entrada[1])
                picam2.set_controls({"ExposureTime": shutter_speed})
            elif cmd == 'g' and len(entrada) > 1:
                gain = float(entrada[1])
                picam2.set_controls({"AnalogueGain": gain})
                
        except Exception as e: 
            print(f"Erro no comando: {e}")

# --- LÓGICA DO SCANNER COM FEEDBACK DE ESCRITA ---

def logica_scanner():
    global contador_perf, frame_count, furo_na_linha, ultimo_frame_bruto
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
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 5000:
                x, y, w, h = cv2.boundingRect(cnt)
                cx_global = x + ROI_X + (w // 2)
                if abs(cx_global - LINHA_X) < MARGEM:
                    furo_agora = True
                    if not furo_na_linha:
                        contador_perf += 1
                        furo_na_linha = True
                        if contador_perf % 4 == 0:
                            frame_count += 1
                            if modo_gravacao:
                                # Feedback de preservação AMIA
                                filename = f"frame_{frame_count:06d}.jpg"
                                full_path = os.path.join(CAPTURE_PATH, filename)
                                fbgr = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)
                                cv2.imwrite(full_path, fbgr)
                                # Este print confirma no terminal a gravação real
                                sys.stdout.write(f"\rCapturando: {filename} | Perf: {contador_perf}")
                                sys.stdout.flush()

        if not furo_agora: furo_na_linha = False
        ultimo_frame_bruto = frame_raw
        time.sleep(0.005) 

# --- FLASK E EXECUÇÃO ---
@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if ultimo_frame_bruto is not None:
                vis = cv2.cvtColor(ultimo_frame_bruto, cv2.COLOR_RGB2BGR)
                vis = cv2.rotate(vis, cv2.ROTATE_90_COUNTERCLOCKWISE)
                ret, buffer = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<html><body style='background:#000;color:#0f0;'><img src='/video_feed' style='height:90vh;'></body></html>"

if __name__ == '__main__':
    # Usamos sys.stdout.write para garantir que o menu apareça antes do input
    threading.Thread(target=painel_controle, daemon=True).start()
    threading.Thread(target=logica_scanner, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)