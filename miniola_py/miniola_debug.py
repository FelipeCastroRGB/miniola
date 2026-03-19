from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import threading
import time
import logging

# 1. Configuração do Flask e Silenciamento de Logs
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR) 

picam2 = Picamera2()

# 2. Configuração de Hardware
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)

picam2.set_controls({
    "ExposureTime": 1000,
    "AnalogueGain": 1.0,
    "FrameRate": 60,
    "LensPosition": 0.0  # Foco inicial (infinito)
})
picam2.start()

# --- GEOMETRIA E VARIÁVEIS DE ESTADO (Agora globais para ajuste dinâmico) ---
ROI_Y, ROI_H = 100, 60
ROI_X, ROI_W = 200, 240  
LINHA_X, MARGEM = 320, 10
THRESH_VAL = 110

# Variáveis de Contagem e Resiliência
contador_perf = 0
frame_count = 0
furo_na_linha = False
ultimo_pitch_estimado = 80 

ultimo_frame_bruto = None
ultimo_frame_binario = None
lista_contornos_debug = []

# --- THREAD: PAINEL DE CONTROLE (Terminal) ---
def painel_controle():
    global contador_perf, frame_count, THRESH_VAL, ultimo_frame_bruto
    global ROI_X, ROI_Y, LINHA_X
    time.sleep(2)
    
    print("\n" + "="*45)
    print("  MINIOLA COMMAND CENTER v2.1")
    print("="*45)
    print("  r         : Reseta contadores")
    print("  a         : Auto-ajuste Threshold")
    print("  t [val]   : Threshold (0-255)")
    print("  v [val]   : Foco/LensPosition (ex: v 5.5)")
    print("  x [val]   : Mover ROI Horizontalmente")
    print("  y [val]   : Mover ROI Verticalmente")
    print("  lx [val]  : Ajustar Linha de Gatilho X")
    print("  e/g/f     : Exposure/Gain/FrameRate")
    print("="*45)

    while True:
        try:
            entrada = input("\nComando >> ").split()
            if not entrada: continue
            cmd = entrada[0].lower()
            
            if cmd == 'r':
                contador_perf = 0
                frame_count = 0
                print(">> [OK] Zerado.")
                
            elif cmd == 'a':
                if ultimo_frame_bruto is not None:
                    gray = cv2.cvtColor(ultimo_frame_bruto, cv2.COLOR_RGB2GRAY)
                    roi_analise = gray[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
                    val_otsu, _ = cv2.threshold(roi_analise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    THRESH_VAL = int(val_otsu * 1.5)
                    print(f">> [AUTO] Threshold: {THRESH_VAL}")

            elif cmd == 't' and len(entrada) > 1:
                THRESH_VAL = int(entrada[1])
                print(f">> [SET] Threshold: {THRESH_VAL}")

            elif cmd == 'v' and len(entrada) > 1:
                val = float(entrada[1])
                picam2.set_controls({"LensPosition": val})
                print(f">> [SET] Foco (LensPosition): {val}")

            elif cmd == 'x' and len(entrada) > 1:
                ROI_X = int(entrada[1])
                print(f">> [ROI] Nova posição X: {ROI_X}")

            elif cmd == 'y' and len(entrada) > 1:
                ROI_Y = int(entrada[1])
                print(f">> [ROI] Nova posição Y: {ROI_Y}")

            elif cmd == 'lx' and len(entrada) > 1:
                LINHA_X = int(entrada[1])
                print(f">> [GATILHO] Nova Linha X: {LINHA_X}")

            elif cmd == 'e' and len(entrada) > 1:
                val = int(entrada[1])
                picam2.set_controls({"ExposureTime": val})
                print(f">> [SET] Exposure: {val}")

            elif cmd == 'g' and len(entrada) > 1:
                val = float(entrada[1])
                picam2.set_controls({"AnalogueGain": val})
                print(f">> [SET] Gain: {val}")

            elif cmd == 'f' and len(entrada) > 1:
                val = int(entrada[1])
                picam2.set_controls({"FrameRate": val})
                print(f">> [SET] FrameRate: {val}")

        except Exception as e:
            print(f">> [ERRO]: {e}")

# --- THREAD: LÓGICA DO SCANNER ---
def logica_scanner():
    global contador_perf, frame_count, furo_na_linha, ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug
    global ultimo_pitch_estimado, THRESH_VAL, ROI_X, ROI_Y, LINHA_X

    while True:
        frame_raw = picam2.capture_array()
        if frame_raw is None: continue
            
        gray = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2GRAY)
        
        # Clipping de segurança para o ROI não sair do frame (640x480)
        ry = max(0, min(ROI_Y, 480 - ROI_H))
        rx = max(0, min(ROI_X, 640 - ROI_W))
        
        roi = gray[ry:ry+ROI_H, rx:rx+ROI_W]
        
        _, binary = cv2.threshold(roi, THRESH_VAL, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        perfs_reais = []
        temp_contornos = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            if 400 < area < 8500 and 0.5 < (w/h) < 2.0:
                cx_local = x + (w // 2)
                perfs_reais.append({'cx': cx_local, 'rect': (x, y, w, h)})
                temp_contornos.append({'rect': (x, y, w, h), 'color': (0, 255, 0)}) # VERDE
            else:
                if area > 100:
                    temp_contornos.append({'rect': (x, y, w, h), 'color': (0, 0, 255)}) # VERMELHO

        perfs_reais.sort(key=lambda x: x['cx'])

        perfs_finais_cx = []
        if len(perfs_reais) >= 2:
            distancias = [perfs_reais[i+1]['cx'] - perfs_reais[i]['cx'] for i in range(len(perfs_reais)-1)]
            ultimo_pitch_estimado = int(np.mean(distancias))
            
            for i in range(len(perfs_reais)):
                perfs_finais_cx.append(perfs_reais[i]['cx'])
                if i < len(perfs_reais) - 1:
                    gap = perfs_reais[i+1]['cx'] - perfs_reais[i]['cx']
                    if gap > (ultimo_pitch_estimado * 1.6):
                        num_faltantes = round(gap / ultimo_pitch_estimado) - 1
                        for f in range(1, num_faltantes + 1):
                            cx_virtual = perfs_reais[i]['cx'] + (f * ultimo_pitch_estimado)
                            perfs_finais_cx.append(cx_virtual)
                            x_v, y_v, w_v, h_v = perfs_reais[i]['rect']
                            temp_contornos.append({'rect': (cx_virtual - (w_v//2), y_v, w_v, h_v), 'color': (0, 255, 255)}) # AMARELO
        else:
            perfs_finais_cx = [p['cx'] for p in perfs_reais]

        furo_agora = False
        for cx in perfs_finais_cx:
            cx_global = cx + rx
            if abs(cx_global - LINHA_X) < MARGEM:
                furo_agora = True
                if not furo_na_linha:
                    contador_perf += 1
                    furo_na_linha = True
                    if contador_perf % 4 == 0:
                        frame_count += 1
        
        if not furo_agora: furo_na_linha = False
            
        ultimo_frame_bruto = frame_raw
        ultimo_frame_binario = binary
        lista_contornos_debug = temp_contornos
        time.sleep(0.005)

# --- FLASK: VISUALIZAÇÃO ---
def generate_frames():
    global ROI_X, ROI_Y, LINHA_X
    while True:
        if ultimo_frame_bruto is None:
            time.sleep(0.01); continue
        
        vis = ultimo_frame_bruto.copy()
        
        # Clipping para desenho
        ry = max(0, min(ROI_Y, 480 - ROI_H))
        rx = max(0, min(ROI_X, 640 - ROI_W))

        cor_gat = (0, 255, 0) if furo_na_linha else (200, 200, 200)
        cv2.rectangle(vis, (rx, ry), (rx + ROI_W, ry + ROI_H), (100, 100, 100), 1)
        cv2.line(vis, (LINHA_X, ry), (LINHA_X, ry + ROI_H), cor_gat, 2)

        for item in lista_contornos_debug:
            x, y, w, h = item['rect']
            cv2.rectangle(vis, (x + rx, y + ry), (x + w + rx, y + h + ry), item['color'], 2)

        bin_rgb = cv2.cvtColor(ultimo_frame_binario, cv2.COLOR_GRAY2RGB)
        canvas_bin = np.zeros_like(vis)
        canvas_bin[ry:ry+ROI_H, rx:rx+ROI_W] = bin_rgb
        
        output = np.hstack((vis, canvas_bin))
        ret, buffer = cv2.imencode('.jpg', output, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    return f"{frame_count} Frames | {contador_perf} Perfs"

@app.route('/')
def index():
    return """
    <html>
        <body style='background:#000; color:#0f0; text-align:center; font-family:monospace;'>
            <h2 style='color:#ff0'>MINIOLA DEBUG - CONTROL PANEL</h2>
            <div id='val' style='font-size:40px; margin-bottom:10px;'>0 Frames | 0 Perfs</div>
            <img src="/video_feed" style="width:95%; border:2px solid #333;">
            <div style='margin-top:20px;'>
                <span style='color:#0f0'>[VERDE]: Real</span> 
                <span style='color:#ff0; margin-left:20px;'>[AMARELO]: Virtual (Gap)</span>
                <span style='color:#f00; margin-left:20px;'>[VERMELHO]: Ruído</span>
            </div>
            <script>
                setInterval(() => {
                    fetch('/status').then(r => r.text()).then(t => { 
                        document.getElementById('val').innerText = t; 
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
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()