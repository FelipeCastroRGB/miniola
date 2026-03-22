import sys
from unittest.mock import MagicMock
import os

# --- MOCKS PARA AMBIENTE SEM HARDWARE ---
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

# --- CONFIGURAÇÃO DE CAMINHOS ---
CAPTURE_PATH = "capturas"
if not os.path.exists(CAPTURE_PATH): os.makedirs(CAPTURE_PATH)

picam2 = Picamera2()

# --- CONFIGURAÇÃO INICIAL DE HARDWARE ---
shutter_speed = 450
gain = 1.0
fps_cam = 60
foco_atual = 15.0
passo_foco = 0.5

config = picam2.create_video_configuration(main={"size": (1080, 720), "format": "RGB888"})
picam2.configure(config)
picam2.set_controls({
    "ExposureTime": shutter_speed, 
    "AnalogueGain": gain, 
    "FrameRate": fps_cam, 
    "LensPosition": foco_atual
})
picam2.start()

# --- GEOMETRIA E ESTADO ---
GRAVANDO = False
ROI_X, ROI_Y = 215, 50
ROI_W, ROI_H = 80, 600
LINHA_RESET_Y = 150 
THRESH_VAL = 110
OFFSET_X = 260
CROP_W, CROP_H = 440, 330 

contador_perfs_ciclo = 0
frame_count = 0
perfuracao_na_linha = False
ultimo_frame_bruto = None
ultimo_frame_binario = None
ultimo_crop_preview = np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
lista_contornos_debug = []
pos_ancora_debug = None

# --- LÓGICA DE CAPTURA ---

def processar_captura(frame, cx, cy, n_frame):
    global OFFSET_X, CROP_W, CROP_H, ultimo_crop_preview, GRAVANDO
    fx, fy = cx + OFFSET_X, cy
    x1, y1 = max(0, int(fx - (CROP_W // 2))), max(0, int(fy - (CROP_H // 2)))
    x2, y2 = min(frame.shape[1], x1 + CROP_W), min(frame.shape[0], y1 + CROP_H)
    
    # Converte para BGR para salvar corretamente
    crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_RGB2BGR)
    
    if crop.size > 0:
        ultimo_crop_preview = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) # Para o dashboard
        if GRAVANDO:
            cv2.imwrite(f"{CAPTURE_PATH}/miniola_{n_frame:06d}.jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 98])

# --- PAINEL DE CONTROLE ATUALIZADO ---

def painel_controle():
    global frame_count, GRAVANDO, LINHA_RESET_Y, ROI_X, ROI_Y, THRESH_VAL
    global foco_atual, passo_foco, shutter_speed, gain, fps_cam, contador_perfs_ciclo
    
    time.sleep(2)
    print("\n" + "═"*45)
    print("   MINIOLA v7.0 - CONTROLE DE PRESERVAÇÃO")
    print("═"*45)
    print("   FOCO:    k (Longe) | l (Perto) | j [passo]")
    print("   IMAGEM:  e [exp]   | g [gain]  | fps [valor]")
    print("   GATILHO: ly [y]    | t [thresh]| ox [offset]")
    print("   CAPTURA: rec (On/Off) | r (Reset)")
    print("═"*45)

    while True:
        try:
            entrada = input("\n>> ").split()
            if not entrada: continue
            cmd = entrada[0].lower()
            val = float(entrada[1]) if len(entrada) > 1 else 0
            
            # Controles de Imagem e Foco
            if cmd == 'l':
                foco_atual = round(foco_atual + passo_foco, 2)
                picam2.set_controls({"LensPosition": foco_atual})
            elif cmd == 'k':
                foco_atual = max(0.0, round(foco_atual - passo_foco, 2))
                picam2.set_controls({"LensPosition": foco_atual})
            elif cmd == 'j': passo_foco = val
            elif cmd == 'e': 
                shutter_speed = int(val)
                picam2.set_controls({"ExposureTime": shutter_speed})
            elif cmd == 'g': 
                gain = val
                picam2.set_controls({"AnalogueGain": gain})
            elif cmd == 'fps':
                fps_cam = int(val)
                picam2.set_controls({"FrameRate": fps_cam})
            
            # Controles de Gatilho
            elif cmd == 't': THRESH_VAL = int(val)
            elif cmd == 'ly': LINHA_RESET_Y = int(val)
            elif cmd == 'ox': OFFSET_X = int(val)
            
            # Sistema
            elif cmd == 'rec': GRAVANDO = not GRAVANDO
            elif cmd == 'r': 
                frame_count = 0
                contador_perfs_ciclo = 0
                for f in os.listdir(CAPTURE_PATH): os.remove(os.path.join(CAPTURE_PATH, f))
        except Exception as e: print(f"Erro comando: {e}")

def logica_scanner():
    global frame_count, ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug
    global contador_perfs_ciclo, perfuracao_na_linha, pos_ancora_debug

    while True:
        frame_raw = picam2.capture_array()
        if frame_raw is None: continue
        
        gray = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2GRAY)
        roi = gray[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
        _, binary = cv2.threshold(roi, THRESH_VAL, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        perfs_neste_frame, debug_visual = [], []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if 150 < area < 9000 and 0.4 < (w/h) < 2.5:
                cx, cy = x + (w//2) + ROI_X, y + (h//2) + ROI_Y
                perfs_neste_frame.append({'cx': cx, 'cy': cy})
                debug_visual.append({'rect': (x+ROI_X, y+ROI_Y, w, h), 'color': (0, 255, 0)})

        perfs_neste_frame.sort(key=lambda p: p['cy'])
        line_y_abs = ROI_Y + LINHA_RESET_Y
        
        if perfs_neste_frame:
            topo_mais_alto = perfs_neste_frame[0]['cy']
            if topo_mais_alto < line_y_abs and not perfuracao_na_linha:
                contador_perfs_ciclo += 1
                perfuracao_na_linha = True
                if contador_perfs_ciclo >= 4:
                    if len(perfs_neste_frame) >= 4:
                        grupo = perfs_neste_frame[0:4]
                        cx_a = int(np.mean([p['cx'] for p in grupo]))
                        cy_a = int(np.mean([p['cy'] for p in grupo]))
                        pos_ancora_debug = (cx_a, cy_a)
                        processar_captura(frame_raw, cx_a, cy_a, frame_count)
                        frame_count += 1
                    contador_perfs_ciclo = 0
            elif topo_mais_alto > (line_y_abs + 30): 
                perfuracao_na_linha = False

        ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug = frame_raw, binary, debug_visual
        time.sleep(0.001)

# --- FLASK: GENERATORS ---

def generate_dashboard():
    while True:
        if ultimo_frame_bruto is None: time.sleep(0.1); continue
        
        # Painel Superior: Live e Binário
        p_live = cv2.resize(ultimo_frame_bruto.copy(), (640, 420))
        sx, sy = 640/1080, 420/720
        
        y_gl = ROI_Y + LINHA_RESET_Y
        cv2.line(p_live, (int(ROI_X*sx), int(y_gl*sy)), (int((ROI_X+ROI_W)*sx), int(y_gl*sy)), (0, 0, 255), 2)
        
        if pos_ancora_debug:
            fx_c, fy_c = pos_ancora_debug[0] + OFFSET_X, pos_ancora_debug[1]
            cv2.rectangle(p_live, (int((fx_c-CROP_W//2)*sx), int((fy_c-CROP_H//2)*sy)), 
                          (int((fx_c+CROP_W//2)*sx), int((fy_c+CROP_H//2)*sy)), (255, 255, 0), 1)

        p_bin = np.zeros((420, 640, 3), dtype=np.uint8)
        bin_res = cv2.resize(cv2.cvtColor(ultimo_frame_binario, cv2.COLOR_GRAY2RGB), (240, 420))
        p_bin[0:420, 200:440] = bin_res

        # Painel Inferior: Status e Último Crop
        p_inf = np.zeros((300, 1280, 3), dtype=np.uint8)
        if ultimo_crop_preview is not None:
            prev_res = cv2.resize(ultimo_crop_preview, (400, 280))
            p_inf[10:290, 440:840] = prev_res
        
        # Telemetria Estilo SMPTE
        info = f"EXP: {shutter_speed}us | GAIN: {gain} | FOCUS: {foco_atual} | FPS: {fps_cam}"
        cv2.putText(p_inf, info, (20, 40), 2, 0.7, (180, 180, 180), 1)
        
        dashboard = np.vstack((np.hstack((p_live, p_bin)), p_inf))
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(dashboard, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/preview_feed')
def preview_feed():
    def generate_preview():
        while True:
            files = sorted([f for f in os.listdir(CAPTURE_PATH) if f.endswith('.jpg')])
            last_frames = files[-48:] if len(files) > 0 else [] # Últimos 2 segundos a 24fps
            if not last_frames:
                time.sleep(0.5); continue
            for frame_file in last_frames:
                img = cv2.imread(os.path.join(CAPTURE_PATH, frame_file))
                if img is None: continue
                _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(1/24)
    return Response(generate_preview(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed(): return Response(generate_dashboard(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    return {
        "rec": "GRAVANDO" if GRAVANDO else "STOP",
        "cor": "#ff0000" if GRAVANDO else "#00ff00",
        "ciclo": f"{contador_perfs_ciclo}/4",
        "total": frame_count
    }

@app.route('/')
def index():
    return """
    <html><body style='background:#050505; color:#eee; font-family:monospace; margin:0; overflow:hidden;'>
        <div style='display:flex; background:#111; padding:10px; border-bottom:1px solid #333; justify-content:space-around;'>
            <span id='m' style='font-weight:bold;'>MODE: --</span>
            <span>CICLO: <b id='c'>0/4</b></span>
            <span>FRAMES: <b id='f'>0</b></span>
        </div>
        <div style='display:flex; flex-direction:row; height:90vh;'>
            <div style='flex:2; border-right:1px solid #333;'>
                <p style='text-align:center; font-size:10px; color:#666;'>DASHBOARD (LIVE / BINARY / LAST CROP)</p>
                <img src="/video_feed" style="width:100%; object-fit:contain;">
            </div>
            <div style='flex:1; background:#000;'>
                <p style='text-align:center; font-size:10px; color:#0f0;'>PREVIEW 24FPS (ULTIMOS CAPTURADOS)</p>
                <img src="/preview_feed" style="width:100%; border:1px solid #0f0;">
            </div>
        </div>
        <script>
            setInterval(() => {
                fetch('/status').then(r => r.json()).then(d => {
                    const m = document.getElementById('m');
                    m.innerText = d.rec; m.style.color = d.cor;
                    document.getElementById('c').innerText = d.ciclo;
                    document.getElementById('f').innerText = d.total;
                });
            }, 200);
        </script>
    </body></html>
    """

if __name__ == '__main__':
    threading.Thread(target=painel_controle, daemon=True).start()
    threading.Thread(target=logica_scanner, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)