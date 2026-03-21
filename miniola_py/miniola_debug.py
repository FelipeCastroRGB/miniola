import sys
from unittest.mock import MagicMock
import os
import shutil

# --- MOCKS PARA AMBIENTE HEADLESS (Essencial para o Pi) ---
sys.modules["pykms"] = MagicMock()
sys.modules["kms"] = MagicMock()

from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import threading
import time
import logging

# 1. Configuração e Pastas
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR) 

PASTA_CAPTURA = 'capturas'
if not os.path.exists(PASTA_CAPTURA): os.makedirs(PASTA_CAPTURA)

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1080, 720), "format": "RGB888"})
picam2.configure(config)
picam2.set_controls({
    "ExposureTime": 450, 
    "AnalogueGain": 1.0, 
    "FrameRate": 60, 
    "LensPosition": 15.0
})
picam2.start()

# --- VARIÁVEIS DE CONTROLE E GEOMETRIA ---
GRAVANDO = False
ESTADO_ATUAL = "BUSCAR_QUADRO" # Estados: "BUSCAR_QUADRO" ou "ESPERAR_SAIDA"

# Geometria do Filme (Ajuste conforme sua montagem)
ROI_Y, ROI_H = 50, 600    
ROI_X, ROI_W = 215, 80  
THRESH_VAL = 110
LINHA_RESET_Y = 120       # Quando a perfuração base sobe além deste Y, o sistema rearma
OFFSET_X = 260            # Distância do centro das perfs até o centro do filme
CROP_W, CROP_H = 440, 330 # Tamanho do fotograma final

# Variáveis de Runtime
frame_count = 0
ultimo_pitch_estimado = 95 
ultimo_frame_bruto = None
ultimo_frame_binario = None
ultimo_crop_preview = np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
lista_contornos_debug = []
pos_ancora_debug = None
box_crop_debug = None

# --- FUNÇÃO DE REGISTRO E CROP ---
def processar_captura(frame, cx, cy, n_frame):
    global OFFSET_X, CROP_W, CROP_H, ultimo_crop_preview, GRAVANDO
    
    fx, fy = cx + OFFSET_X, cy
    x1, y1 = max(0, int(fx - (CROP_W // 2))), max(0, int(fy - (CROP_H // 2)))
    x2, y2 = min(frame.shape[1], x1 + CROP_W), min(frame.shape[0], y1 + CROP_H)
    
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size > 0:
        ultimo_crop_preview = crop
        if GRAVANDO:
            caminho = os.path.join(PASTA_CAPTURA, f"miniola_{n_frame:06d}.jpg")
            cv2.imwrite(caminho, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
    return (x1, y1, x2, y2)

# --- THREAD: INTERFACE DE COMANDO (TERMINAL) ---
def painel_controle():
    global frame_count, THRESH_VAL, OFFSET_X, GRAVANDO, PASTA_CAPTURA, LINHA_RESET_Y
    time.sleep(2)
    print("\n" + "═"*45)
    print("   MINIOLA CONSOLIDATED v3.7 - CONTROLE")
    print("" + "═"*45)
    print("   rec        : Ligar/Desligar Gravação")
    print("   clean      : Apagar todas as fotos da pasta")
    print("   r          : Zerar contador de frames")
    print("   t [val]    : Ajustar Threshold (0-255)")
    print("   ox [val]   : Ajustar Offset Horizontal")
    print("   ly [val]   : Linha de Reset Y (atual: " + str(LINHA_RESET_Y) + ")")
    print("   exit       : Fechar scanner")
    print("   " + "═"*45)

    while True:
        try:
            entrada = input("\nComando >> ").split()
            if not entrada: continue
            cmd = entrada[0].lower()
            
            if cmd == 'rec':
                GRAVANDO = not GRAVANDO
                print(f">> GRAVAÇÃO: {'ON' if GRAVANDO else 'OFF'}")
            elif cmd == 'clean':
                for f in os.listdir(PASTA_CAPTURA):
                    os.remove(os.path.join(PASTA_CAPTURA, f))
                print(">> [OK] Pasta de capturas limpa.")
            elif cmd == 'r':
                frame_count = 0
                print(">> Contador resetado.")
            elif cmd == 't' and len(entrada) > 1:
                THRESH_VAL = int(entrada[1])
            elif cmd == 'ox' and len(entrada) > 1:
                OFFSET_X = int(entrada[1])
            elif cmd == 'ly' and len(entrada) > 1:
                LINHA_RESET_Y = int(entrada[1])
            elif cmd == 'exit':
                os._exit(0)
        except Exception as e: print(f"Erro: {e}")

# --- THREAD: LÓGICA DE VISÃO COMPUTACIONAL ---
def logica_scanner():
    global frame_count, ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug
    global ultimo_pitch_estimado, ESTADO_ATUAL, pos_ancora_debug, box_crop_debug

    while True:
        frame_raw = picam2.capture_array()
        if frame_raw is None: continue
        
        gray = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2GRAY)
        ry, rx = max(0, min(ROI_Y, 700)), max(0, min(ROI_X, 1000))
        roi = gray[ry:ry+ROI_H, rx:rx+ROI_W]
        _, binary = cv2.threshold(roi, THRESH_VAL, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 1. Detecção de Perfurações
        perfs_reais, debug_visual = [], []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if 150 < area < 9000 and 0.4 < (w/h) < 2.5:
                perfs_reais.append({'cx': x + (w//2) + rx, 'cy': y + (h//2) + ry})
                debug_visual.append({'rect': (x+rx, y+ry, w, h), 'color': (0, 255, 0)})

        perfs_reais.sort(key=lambda p: p['cy'])
        
        # 2. Gap Filling (Resiliência)
        perfs_finais = []
        if len(perfs_reais) >= 2:
            ultimo_pitch_estimado = int(np.median([perfs_reais[i+1]['cy'] - perfs_reais[i]['cy'] for i in range(len(perfs_reais)-1)]))
            for i in range(len(perfs_reais)):
                perfs_finais.append(perfs_reais[i])
                if i < len(perfs_reais) - 1:
                    gap = perfs_reais[i+1]['cy'] - perfs_reais[i]['cy']
                    if gap > (ultimo_pitch_estimado * 1.5):
                        for f in range(1, round(gap / ultimo_pitch_estimado)):
                            cy_v = perfs_reais[i]['cy'] + (f * ultimo_pitch_estimado)
                            perfs_finais.append({'cx': perfs_reais[i]['cx'], 'cy': cy_v})
                            debug_visual.append({'rect': (perfs_reais[i]['cx']-15, cy_v-15, 30, 30), 'color': (0, 255, 255)})
        else: perfs_finais = perfs_reais

        # 3. Máquina de Estados e Registro
        if ESTADO_ATUAL == "BUSCAR_QUADRO":
            if len(perfs_finais) >= 4:
                grupo = perfs_finais[0:4]
                cx_a, cy_a = int(np.mean([p['cx'] for p in grupo])), int(np.mean([p['cy'] for p in grupo]))
                box_crop_debug = processar_captura(frame_raw, cx_a, cy_a, frame_count)
                pos_ancora_debug = (cx_a, cy_a)
                frame_count += 1
                ESTADO_ATUAL = "ESPERAR_SAIDA"
        elif ESTADO_ATUAL == "ESPERAR_SAIDA":
            if len(perfs_reais) > 0 and perfs_reais[-1]['cy'] < (ry + LINHA_RESET_Y):
                ESTADO_ATUAL = "BUSCAR_QUADRO"
                pos_ancora_debug = None

        ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug = frame_raw, binary, debug_visual
        time.sleep(0.002)

# --- GERADOR DO SUPER-DASHBOARD (1280x900) ---
def generate_dashboard():
    while True:
        if ultimo_frame_bruto is None: time.sleep(0.1); continue
        
        # Topo Esquerda: Live (640x420)
        p_live = cv2.resize(ultimo_frame_bruto.copy(), (640, 420))
        sx, sy = 640/1080, 420/720
        for item in lista_contornos_debug:
            x, y, w, h = item['rect']
            cv2.rectangle(p_live, (int(x*sx), int(y*sy)), (int((x+w)*sx), int((y+h)*sy)), item['color'], 2)
        if pos_ancora_debug:
            cv2.drawMarker(p_live, (int(pos_ancora_debug[0]*sx), int(pos_ancora_debug[1]*sy)), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)

        # Topo Direita: Binário (640x420)
        p_bin = np.zeros((420, 640, 3), dtype=np.uint8)
        bin_zoom = cv2.resize(cv2.cvtColor(ultimo_frame_binario, cv2.COLOR_GRAY2RGB), (200, 420))
        p_bin[0:420, 220:420] = bin_zoom
        cv2.putText(p_bin, "ANALISE DE PERFURACOES", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Base: Preview Estabilizado (1280x480)
        p_prev = np.zeros((480, 1280, 3), dtype=np.uint8)
        h_c, w_c = ultimo_crop_preview.shape[:2]
        nw = int(460 * (w_c/h_c))
        res_crop = cv2.resize(ultimo_crop_preview, (nw, 460))
        p_prev[10:470, (1280-nw)//2 : (1280-nw)//2 + nw] = res_crop
        
        # UI de Status no Preview
        if GRAVANDO:
            cv2.circle(p_prev, (40, 40), 12, (0, 0, 255), -1)
            cv2.putText(p_prev, f"GRAVANDO: {frame_count:05d}", (65, 52), cv2.FONT_HERSHEY_BOLD, 1.1, (0, 0, 255), 3)
        else:
            cv2.putText(p_prev, "STANDBY - PREVIEW APENAS", (30, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)

        final = np.vstack((np.hstack((p_live, p_bin)), p_prev))
        _, buffer = cv2.imencode('.jpg', final, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed(): return Response(generate_dashboard(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status(): return f"{'● REC' if GRAVANDO else 'READY'} | Frames: {frame_count} | {ESTADO_ATUAL}"

@app.route('/')
def index():
    return """
    <html>
        <body style='background:#000; color:#0f0; text-align:center; font-family:monospace; margin:0;'>
            <div style='background:#111; padding:10px; border-bottom:2px solid #333;'>
                <span id='st' style='font-size:24px;'>Aguardando...</span>
            </div>
            <img src="/video_feed" style="height:90vh; border:2px solid #222; margin-top:5px;">
            <script>
                setInterval(() => {
                    fetch('/status').then(r => r.text()).then(t => { 
                        let el = document.getElementById('st'); el.innerText = t;
                        el.style.color = t.includes('REC') ? 'red' : '#0f0';
                    });
                }, 100);
            </script>
        </body>
    </html>
    """

if __name__ == '__main__':
    try:
        threading.Thread(target=painel_controle, daemon=True).start()
        threading.Thread(target=logica_scanner, daemon=True).start()
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except: picam2.stop()