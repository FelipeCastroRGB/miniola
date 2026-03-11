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
CAPTURE_PATH = "/home/felipe/miniola_py/captura"
if not os.path.exists(CAPTURE_PATH):
    os.makedirs(CAPTURE_PATH)

picam2 = Picamera2()

# 2. Configuração de Hardware (800x600)
WIDTH, HEIGHT = 800, 600
config = picam2.create_video_configuration(main={"size": (WIDTH, HEIGHT), "format": "RGB888"})
picam2.configure(config)

# Valores iniciais 
shutter_speed = 1800
gain = 1.0
fps = 60

picam2.set_controls({"ExposureTime": shutter_speed, "AnalogueGain": gain, "FrameRate": fps})
picam2.start()

# --- GEOMETRIA DINÂMICA (NATIVA: CÂMERA VERTICAL) ---
# Ajustado para buscar perfurações na lateral agora que a câmera está orientada [cite: 2026-02-28]
ROI_X, ROI_Y = 5, 5  
ROI_W, ROI_H = 100, 580 # ROI agora é uma faixa vertical na lateral [cite: 2026-02-28]
LINHA_Y, MARGEM = 300, 15 # Gatilho agora monitora a passagem vertical (Y) [cite: 2026-02-28]
THRESH_VAL = 230
CROP_Y1, CROP_Y2 = 0, 600  
CROP_X1, CROP_X2 = 0, 800  

# Estado Global
contador_perf = 0
frame_count = 0
furo_na_linha = False
modo_gravacao = False
ultimo_frame_bruto = None
ultimo_frame_binario = None
lista_contornos_debug = []

# --- THREAD: PAINEL DE CONTROLE ---

def painel_controle():
    global contador_perf, frame_count, modo_gravacao, THRESH_VAL, ultimo_frame_bruto
    global ROI_X, ROI_Y, LINHA_Y, shutter_speed, gain
    
    print("\n" + "="*45)
    print("  MINIOLA CONTROL v3.0 - CÂMERA ORIENTADA")
    print("="*45)
    print("  MOVER ROI:    w/s (C/B) | a/d (E/D)")
    print("  GATILHO:     < / >  (Linha Gatilho)")
    print("  IMAGEM:       e/g (Exp/Gain) | t (Thresh) | o (Otsu)")
    print("  CAPTURA:      f (Gravar) | p (Pausar) | r (Reset)")
    print("="*45)

    while True:
        try:
            entrada = input(">> ").split()
            if not entrada: continue
            cmd = entrada[0].lower()
            
            if cmd == 'w': ROI_Y = max(0, ROI_Y - 5)
            elif cmd == 's': ROI_Y = min(HEIGHT - ROI_H, ROI_Y + 5)
            elif cmd == 'a': ROI_X = max(0, ROI_X - 5)
            elif cmd == 'd': ROI_X = min(WIDTH - ROI_W, ROI_X + 5)
            elif cmd == '>': LINHA_Y = min(ROI_Y + ROI_H, LINHA_Y + 5)
            elif cmd == '<': LINHA_Y = max(ROI_Y, LINHA_Y - 5)
            elif cmd == 'o':
                if ultimo_frame_bruto is not None:
                    gray = cv2.cvtColor(ultimo_frame_bruto, cv2.COLOR_RGB2GRAY)
                    roi_a = gray[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
                    val, _ = cv2.threshold(roi_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    THRESH_VAL = int(val * 1.55)
                    print(f"[AUTO] Threshold: {THRESH_VAL}")
            elif cmd == 'e' and len(entrada) > 1:
                shutter_speed = int(entrada[1])
                picam2.set_controls({"ExposureTime": shutter_speed})
            elif cmd == 'g' and len(entrada) > 1:
                gain = float(entrada[1])
                picam2.set_controls({"AnalogueGain": gain})
            elif cmd == 't' and len(entrada) > 1:
                THRESH_VAL = int(entrada[1])
            elif cmd == 'f': 
                modo_gravacao = True; print("GRAVANDO NO RAM DRIVE...")
            elif cmd == 'p': 
                modo_gravacao = False; print("PAUSADO.")
            elif cmd == 'r': 
                contador_perf = frame_count = 0
                os.system(f"rm -rf {CAPTURE_PATH}/*.jpg")
                print("ESTATÍSTICAS E RAM DRIVE LIMPOS.")
        except Exception as e: print(f"Erro: {e}")

# --- THREAD: LÓGICA DO SCANNER ---

def logica_scanner():
    global contador_perf, frame_count, furo_na_linha, ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug
    
    while True:
        frame_raw_completo = picam2.capture_array()
        if frame_raw_completo is None: 
            time.sleep(0.01); continue
            
        frame_raw = frame_raw_completo[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]
        gray = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2GRAY)
        roi = gray[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
        
        _, binary = cv2.threshold(roi, THRESH_VAL, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        furo_agora = False
        temp_contornos = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            if 500 < area < 8000 and 0.6 < (w/h) < 1.6:
                cy_global = y + ROI_Y + (h // 2) # Monitorando o centro Y [cite: 2026-02-28]
                temp_contornos.append({'rect': (x, y, w, h), 'color': (0, 255, 0)}) 
                
                if abs(cy_global - LINHA_Y) < MARGEM:
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
                if area > 150:
                    temp_contornos.append({'rect': (x, y, w, h), 'color': (0, 0, 255)})

        if not furo_agora: furo_na_linha = False
            
        ultimo_frame_bruto = frame_raw
        ultimo_frame_binario = binary
        lista_contornos_debug = temp_contornos
        time.sleep(0.005)

# --- FLASK: VISIONAMENTO ---

def generate_frames():
    while True:
        if ultimo_frame_bruto is None:
            time.sleep(0.1); continue
        
        vis_base = cv2.cvtColor(ultimo_frame_bruto, cv2.COLOR_RGB2BGR)
        
        # Telemetria (sem rotação) [cite: 2026-02-28]
        cor_gat = (0, 255, 0) if furo_na_linha else (0, 0, 255)
        cv2.rectangle(vis_base, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (150, 150, 150), 2)
        cv2.line(vis_base, (ROI_X, LINHA_Y), (ROI_X + ROI_W, LINHA_Y), cor_gat, 3)
        
        for item in lista_contornos_debug:
            x, y, w, h = item['rect']
            cv2.rectangle(vis_base, (x + ROI_X, y + ROI_Y), (x + w + ROI_X, y + h + ROI_Y), item['color'], 2)
        
        # Redimensionamento para o navegador
        altura_alvo = 600 
        h_orig, w_orig = vis_base.shape[:2]
        proporcao = w_orig / h_orig
        largura_final = int(altura_alvo * proporcao)
        
        vis_light = cv2.resize(vis_base, (largura_final, altura_alvo), interpolation=cv2.INTER_AREA)
        ret, buffer = cv2.imencode('.jpg', vis_light, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.04) 

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/preview_feed')
def preview_feed():
    def generate_preview():
        while True:
            files = sorted([f for f in os.listdir(CAPTURE_PATH) if f.endswith('.jpg')])
            last_frames = files[-96:] if len(files) > 0 else [] 
            
            if not last_frames:
                time.sleep(0.5); continue

            for frame_file in last_frames:
                path = os.path.join(CAPTURE_PATH, frame_file)
                try:
                    img = cv2.imread(path) # Sem rotação [cite: 2026-02-28]
                    ret, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    time.sleep(1/24)
                except: continue
    return Response(generate_preview(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    return {
        "modo": "GRAVANDO" if modo_gravacao else "VISIONAMENTO",
        "cor": "#ff0000" if modo_gravacao else "#00ff00",
        "perf": contador_perf,
        "frames": frame_count
    }

@app.route('/')
def index():
    return """
    <html>
        <body style='background:#111; color:#eee; font-family:monospace; display:flex; flex-direction:column; align-items:center;'>
            <div style="background:#222; padding:10px; border-radius:5px; margin:10px; width:80%; display:flex; justify-content:space-around; border:1px solid #444;">
                <span id="modo" style="font-weight:bold;">MODO: ---</span>
                <span>PERFURAÇÕES: <b id="perf">0</b></span>
                <span>FRAMES SALVOS: <b id="fr">0</b></span>
            </div>
            
            <div style="display:flex; gap:20px; justify-content:center; width:100%;">
                <div style="text-align:center;">
                    <p>AO VIVO (NATIVO)</p>
                    <div style="background:#000; width:450px; height:600px; border:2px solid #333; display:flex; align-items:center; justify-content:center;">
                        <img src="/video_feed" style="max-width:100%; max-height:100%; object-fit:contain;">
                    </div>
                </div>
                
                <div style="text-align:center;">
                    <p>PREVIEW (GRAVADO)</p>
                    <div style="background:#000; width:450px; height:600px; border:2px solid #0f0; display:flex; align-items:center; justify-content:center;">
                        <img src="/preview_feed" style="max-width:100%; max-height:100%; object-fit:contain;">
                    </div>
                </div>
            </div>

            <script>
                setInterval(() => {
                    fetch('/status').then(r => r.json()).then(data => {
                        const m = document.getElementById('modo');
                        m.innerText = "MODO: " + data.modo;
                        m.style.color = data.cor;
                        document.getElementById('perf').innerText = data.perf;
                        document.getElementById('fr').innerText = data.frames;
                    });
                }, 500); 
            </script>
        </body>
    </html>
    """

if __name__ == '__main__':
    threading.Thread(target=painel_controle, daemon=True).start()
    threading.Thread(target=logica_scanner, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)