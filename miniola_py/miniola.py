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

# 1. Configuração e Silenciamento de Logs irrelevantes
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR) 

# Caminho absoluto para o RAM Drive (Garante persistência na sessão)
BASE_DIR = "/home/felipe/miniola_py"
CAPTURE_PATH = os.path.join(BASE_DIR, "captura")
if not os.path.exists(CAPTURE_PATH):
    os.makedirs(CAPTURE_PATH)

# Inicialização da Câmera (Modo Headless para evitar erro de KMS)
picam2 = Picamera2()
WIDTH, HEIGHT = 800, 600
config = picam2.create_video_configuration(main={"size": (WIDTH, HEIGHT), "format": "RGB888"})
picam2.configure(config)

shutter_speed, gain, fps = 1000, 1.0, 30
picam2.set_controls({"ExposureTime": shutter_speed, "AnalogueGain": gain, "FrameRate": fps})
picam2.start_preview(NullPreview())

# --- ESTADO E GEOMETRIA ---
ROI_X, ROI_Y, ROI_W, ROI_H = 250, 40, 300, 120  
LINHA_X, MARGEM, THRESH_VAL = 400, 15, 110
contador_perf, frame_count = 0, 0
furo_na_linha, modo_gravacao = False, False
ultimo_frame_bruto = None

# --- PAINEL DE CONTROLE COM FORÇA DE SAÍDA (FLUSH) ---

def exibir_menu():
    os.system('clear') # Limpa o lixo do terminal antes de iniciar
    menu = f"""
    {"="*45}
    MINIOLA CONTROL v2.8.5 - ESTABILIZADO
    {"="*45}
    MOVER ROI:   [w, a, s, d] (minúsculos)
    GATILHO:     [< , >]
    IMAGEM:      [e] Exp | [g] Gain | [t] Thresh
    COMANDOS:    [y] GRAVAR | [P] PAUSAR | [R] RESET
    {"="*45}
    """
    print(menu, flush=True)

def painel_controle():
    global contador_perf, frame_count, modo_gravacao, THRESH_VAL
    global ROI_X, ROI_Y, LINHA_X, shutter_speed, gain
    
    exibir_menu()

    while True:
        try:
            # Pegamos a entrada bruta para diferenciar 's' de 'S'
            entrada_raw = input(">> ").strip()
            if not entrada_raw: continue
            
            # Comandos de GRAVAÇÃO (Exigem Shift/Caps Lock)
            if entrada_raw == 'y':
                modo_gravacao = True
                print("\n[!] STATUS: GRAVAÇÃO ATIVADA (RAM DRIVE)", flush=True)
            elif entrada_raw == 'P':
                modo_gravacao = False
                print("\n[!] STATUS: GRAVAÇÃO PAUSADA", flush=True)
            elif entrada_raw == 'R':
                contador_perf = frame_count = 0
                print("\n[!] STATUS: CONTADORES ZERADOS", flush=True)
            
            # Comandos de MOVIMENTAÇÃO (Apenas minúsculos)
            elif entrada_raw == 'w': ROI_Y = max(0, ROI_Y - 5)
            elif entrada_raw == 's': ROI_Y = min(HEIGHT - ROI_H, ROI_Y + 5)
            elif entrada_raw == 'a': ROI_X = max(0, ROI_X - 5)
            elif entrada_raw == 'd': ROI_X = min(WIDTH - ROI_W, ROI_X + 5)
            
            # Comandos de AJUSTE (Gatilho e Imagem)
            elif entrada_raw == '>': LINHA_X = min(ROI_X + ROI_W, LINHA_X + 5)
            elif entrada_raw == '<': LINHA_X = max(ROI_X, LINHA_X - 5)
            
            # Comandos com argumentos (ex: t 150)
            partes = entrada_raw.split()
            if len(partes) > 1:
                cmd = partes[0].lower()
                val = partes[1]
                if cmd == 't': THRESH_VAL = int(val); print(f"Threshold: {THRESH_VAL}", flush=True)
                elif cmd == 'e': 
                    shutter_speed = int(val)
                    picam2.set_controls({"ExposureTime": shutter_speed})
                    print(f"Exp: {shutter_speed}", flush=True)
                elif cmd == 'g':
                    gain = float(val)
                    picam2.set_controls({"AnalogueGain": gain})
                    print(f"Gain: {gain}", flush=True)
                    
        except Exception as e: 
            print(f"Erro: {e}", flush=True)

# --- LÓGICA DO SCANNER ---

def logica_scanner():
    global contador_perf, frame_count, furo_na_linha, ultimo_frame_bruto
    while True:
        frame_raw = picam2.capture_array()
        if frame_raw is None: 
            time.sleep(0.01); continue
            
        gray = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2GRAY)
        roi = gray[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
        _, binary = cv2.threshold(roi, THRESH_VAL, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        furo_agora = False
        for cnt in contours:
            if 50 < cv2.contourArea(cnt) < 5000:
                x, y, w, h = cv2.boundingRect(cnt)
                if abs((x + ROI_X + w//2) - LINHA_X) < MARGEM:
                    furo_agora = True
                    if not furo_na_linha:
                        contador_perf += 1
                        furo_na_linha = True
                        if contador_perf % 4 == 0:
                            frame_count += 1
                            if modo_gravacao:
                                path = os.path.join(CAPTURE_PATH, f"frame_{frame_count:06d}.jpg")
                                cv2.imwrite(path, cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR))
                                sys.stdout.write(f"\r[SCANNING] Frame: {frame_count} | Perf: {contador_perf}")
                                sys.stdout.flush()

        if not furo_agora: furo_na_linha = False
        ultimo_frame_bruto = frame_raw
        time.sleep(0.005) 

# --- FLASK E STARTUP ---

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if ultimo_frame_bruto is not None:
                vis = cv2.cvtColor(ultimo_frame_bruto, cv2.COLOR_RGB2BGR)
                vis = cv2.rotate(vis, cv2.ROTATE_90_COUNTERCLOCKWISE)
                ret, buffer = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<html><body style='background:#000;color:#0f0;text-align:center;'><h3>MINIOLA v2.8.5</h3><img src='/video_feed' style='height:85vh;'></body></html>"

if __name__ == '__main__':
    threading.Thread(target=painel_controle, daemon=True).start()
    threading.Thread(target=logica_scanner, daemon=True).start()
    # Desativamos o reloader para não duplicar as threads e o menu
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)