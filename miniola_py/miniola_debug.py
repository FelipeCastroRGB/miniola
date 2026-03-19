import sys
from unittest.mock import MagicMock

# --- FIX: Mock para ambiente headless ---
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

picam2 = Picamera2()

# Configuração de Hardware conforme seu último ajuste
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.set_controls({
    "ExposureTime": 400,
    "AnalogueGain": 1.0,
    "FrameRate": 60,
    "LensPosition": 15.0 
})
picam2.start()

# --- GEOMETRIA AJUSTADA (Alinhando ROI e Gatilho) ---
ROI_Y, ROI_H = 100, 300
ROI_X, ROI_W = 215, 60  
# A LINHA_X agora fica dentro do ROI (215 + 30 = 245)
LINHA_X, MARGEM = 245, 12 
THRESH_VAL = 110

contador_perf = 0
frame_count = 0
furo_na_linha = False
ultimo_pitch_estimado = 80 

ultimo_frame_bruto = None
ultimo_frame_binario = None
lista_contornos_debug = []

def painel_controle():
    global contador_perf, frame_count, THRESH_VAL, ultimo_frame_bruto, ROI_X, ROI_Y, LINHA_X
    time.sleep(2)
    print("\n" + "="*45)
    print("  MINIOLA CALIBRATION CENTER v2.3")
    print("="*45)
    print("  lx [val] : Ajustar Linha (DICA: Tente 245)")
    print("  r        : Resetar Contagem")
    print("  t [val]  : Ajustar Threshold")
    print("="*45)

    while True:
        try:
            entrada = input("\nComando >> ").split()
            if not entrada: continue
            cmd = entrada[0].lower()
            if cmd == 'r':
                contador_perf, frame_count = 0, 0
                print(">> [OK] Zerado.")
            elif cmd == 'lx' and len(entrada) > 1:
                LINHA_X = int(entrada[1])
                print(f">> [GATILHO] Linha X em: {LINHA_X}")
            elif cmd == 't' and len(entrada) > 1:
                THRESH_VAL = int(entrada[1])
            # ... (outros comandos podem ser mantidos aqui)
        except Exception as e: print(f">> [ERRO]: {e}")

def logica_scanner():
    global contador_perf, frame_count, furo_na_linha, ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug
    global ultimo_pitch_estimado, THRESH_VAL, ROI_X, ROI_Y, LINHA_X

    while True:
        frame_raw = picam2.capture_array()
        if frame_raw is None: continue
        
        gray = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2GRAY)
        ry, rx = max(0, min(ROI_Y, 420)), max(0, min(ROI_X, 580))
        roi = gray[ry:ry+ROI_H, rx:rx+ROI_W]
        
        _, binary = cv2.threshold(roi, THRESH_VAL, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        perfs_reais, temp_contornos = [], []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if 400 < area < 8500 and 0.5 < (w/h) < 2.0:
                perfs_reais.append({'cx': x + (w//2), 'rect': (x, y, w, h)})
                temp_contornos.append({'rect': (x, y, w, h), 'color': (0, 255, 0)})
            elif area > 100:
                temp_contornos.append({'rect': (x, y, w, h), 'color': (0, 0, 255)})

        perfs_reais.sort(key=lambda x: x['cx'])
        perfs_finais_cx = []
        
        # --- FIX: Divisão por zero e Gap Filling ---
        if len(perfs_reais) >= 2:
            diffs = [perfs_reais[i+1]['cx'] - perfs_reais[i]['cx'] for i in range(len(perfs_reais)-1)]
            ultimo_pitch_estimado = max(1, int(np.mean(diffs))) 
            
            for i in range(len(perfs_reais)):
                perfs_finais_cx.append(perfs_reais[i]['cx'])
                if i < len(perfs_reais) - 1:
                    gap = perfs_reais[i+1]['cx'] - perfs_reais[i]['cx']
                    if gap > (ultimo_pitch_estimado * 1.6):
                        for f in range(1, round(gap / ultimo_pitch_estimado)):
                            cx_v = perfs_reais[i]['cx'] + (f * ultimo_pitch_estimado)
                            perfs_finais_cx.append(cx_v)
                            x_v, y_v, w_v, h_v = perfs_reais[i]['rect']
                            temp_contornos.append({'rect': (cx_v-(w_v//2), y_v, w_v, h_v), 'color': (0, 255, 255)})
        else:
            perfs_finais_cx = [p['cx'] for p in perfs_reais]

        # --- LÓGICA DE GATILHO ---
        furo_agora = False
        for cx in perfs_finais_cx:
            # Importante: somamos rx para comparar com a LINHA_X que é global
            if abs((cx + rx) - LINHA_X) < MARGEM:
                furo_agora = True
                if not furo_na_linha:
                    contador_perf += 1
                    furo_na_linha = True
                    # Padrão 35mm: 4 perfurações por frame
                    if contador_perf % 4 == 0: 
                        frame_count += 1
                        print(f">> Frame {frame_count} detectado!")
        
        if not furo_agora: 
            furo_na_linha = False
            
        ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug = frame_raw, binary, temp_contornos
        time.sleep(0.005)

def generate_frames():
    while True:
        if ultimo_frame_bruto is None:
            time.sleep(0.01); continue
        vis = ultimo_frame_bruto.copy()
        ry, rx = max(0, min(ROI_Y, 420)), max(0, min(ROI_X, 580))
        
        # Desenha o ROI e a Linha de Gatilho
        cv2.rectangle(vis, (rx, ry), (rx + ROI_W, ry + ROI_H), (255, 255, 255), 1)
        cor_linha = (0, 255, 0) if furo_na_linha else (0, 0, 255) # Vermelho se livre, Verde se gatilhado
        cv2.line(vis, (LINHA_X, ry), (LINHA_X, ry + ROI_H), cor_linha, 2)
        
        for item in lista_contornos_debug:
            x, y, w, h = item['rect']
            cv2.rectangle(vis, (x + rx, y + ry), (x + w + rx, y + h + ry), item['color'], 2)
        
        bin_rgb = cv2.cvtColor(ultimo_frame_binario, cv2.COLOR_GRAY2RGB)
        canvas_bin = np.zeros_like(vis)
        canvas_bin[ry:ry+ROI_H, rx:rx+ROI_W] = bin_rgb
        output = np.hstack((vis, canvas_bin))
        _, buffer = cv2.imencode('.jpg', output, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status(): return f"{frame_count} Frames | {contador_perf} Perfs"

@app.route('/')
def index():
    return """
    <html>
        <body style='background:#000; color:#0f0; text-align:center; font-family:monospace;'>
            <h2 style='color:#ff0'>MINIOLA DEBUG v2.3</h2>
            <div id='val' style='font-size:40px; margin-bottom:10px;'>0 Frames | 0 Perfs</div>
            <img src="/video_feed" style="width:95%; border:2px solid #333;">
            <script>
                setInterval(() => {
                    fetch('/status').then(r => r.text()).then(t => { document.getElementById('val').innerText = t; });
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
    except KeyboardInterrupt: pass
    finally: picam2.stop()