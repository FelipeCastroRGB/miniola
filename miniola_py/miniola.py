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
# Usamos o caminho absoluto para garantir a gravação no tmpfs (RAM) 
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
fps = 45

picam2.set_controls({"ExposureTime": shutter_speed, "AnalogueGain": gain, "FrameRate": fps})
picam2.start()

# --- GEOMETRIA DINÂMICA (Adaptada para 800x600) ---
ROI_X, ROI_Y = 5, 5  # Posição inicial do ROI (ajustada para o novo tamanho)
ROI_W, ROI_H = 300, 40  # Tamanho do ROI (ajustado para o novo tamanho)  
LINHA_X, MARGEM = 100, 15
THRESH_VAL = 110
CROP_Y1, CROP_Y2 = 110, 580  # Ajuste o corte vertical
CROP_X1, CROP_X2 = 250, 550  # Ajuste o corte horizontal (elimina bordas pretas)

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
            cmd = entrada[0].lower() # Aceita tanto maiúsculo quanto minúsculo
            
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
                    THRESH_VAL = int(val * 1.5)
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
                contador_perf = frame_count = 0
                os.system(f"rm -rf {CAPTURE_PATH}/*.jpg")
                print("ESTATÍSTICAS E RAM DRIVE LIMPOS.")
        except Exception as e: print(f"Erro: {e}")

# --- THREAD: LÓGICA DO SCANNER (Processamento Otimizado) ---

def logica_scanner():
    global contador_perf, frame_count, furo_na_linha, ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug
    
    while True:
        frame_raw_completo = picam2.capture_array()
        if frame_raw_completo is None: 
            time.sleep(0.01)
            continue
            
        # 1. APLICAÇÃO DO CROP (v3.1)
        # Corta a imagem bruta para processar apenas a área do filme e perfurações [cite: 2025-12-23]
        # Isso economiza CPU e remove bordas pretas desnecessárias [cite: 2026-02-28]
        frame_raw = frame_raw_completo[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]
        
        # Converte para cinza e extrai o ROI dentro da imagem já cortada
        gray = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2GRAY)
        roi = gray[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
        
        _, binary = cv2.threshold(roi, THRESH_VAL, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        furo_agora = False
        temp_contornos = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 2. FILTRO DE RUÍDO APRIMORADO (v3.1)
            # Aumentamos o limite para 300px e adicionamos filtro de proporção (aspect ratio)
            # Isso elimina poeira e pequenos artefatos vermelhos no ROI 
            if 500 < area < 8000 and 0.6 < (w/h) < 1.6:
                cx_global = x + ROI_X + (w // 2)
                temp_contornos.append({'rect': (x, y, w, h), 'color': (0, 255, 0)}) # Verde: Perfuração válida
                
                # Lógica de Gatilho (Trigger)
                if abs(cx_global - LINHA_X) < MARGEM:
                    furo_agora = True
                    if not furo_na_linha:
                        contador_perf += 1
                        furo_na_linha = True
                        
                        # A cada 4 perfurações (Padrão 35mm), salvamos 1 frame
                        if contador_perf % 4 == 0:
                            frame_count += 1
                            if modo_gravacao:
                                # Salva a imagem CORTADA para o Master de Preservação
                                fbgr = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)
                                cv2.imwrite(f"{CAPTURE_PATH}/frame_{frame_count:06d}.jpg", fbgr)
            else:
                # Desenha em vermelho apenas objetos médios que não são furos
                # Ignora ruídos minúsculos (< 100px) para manter a tela limpa
                if area > 150:
                    temp_contornos.append({'rect': (x, y, w, h), 'color': (0, 0, 255)})

        if not furo_agora: 
            furo_na_linha = False
            
        # Atualiza globais para o streaming (v3.1 usa frame_raw já com crop)
        ultimo_frame_bruto = frame_raw
        ultimo_frame_binario = binary
        lista_contornos_debug = temp_contornos
        
        time.sleep(0.005)

# --- FLASK: VISIONAMENTO COM TELEMETRIA ---

def generate_frames():
    while True:
        if ultimo_frame_bruto is None:
            time.sleep(0.1); continue
        
        # Cria a base de visualização a partir do frame já cortado [cite: 2026-02-28]
        vis_base = cv2.cvtColor(ultimo_frame_bruto, cv2.COLOR_RGB2BGR)
        
        # Desenha telemetria
        cor_gat = (0, 255, 0) if furo_na_linha else (0, 0, 255)
        cv2.rectangle(vis_base, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (150, 150, 150), 2)
        cv2.line(vis_base, (LINHA_X, ROI_Y), (LINHA_X, ROI_Y + ROI_H), cor_gat, 3)
        
        for item in lista_contornos_debug:
            x, y, w, h = item['rect']
            cv2.rectangle(vis_base, (x + ROI_X, y + ROI_Y), (x + w + ROI_X, y + h + ROI_Y), item['color'], 2)
        
        vis = cv2.rotate(vis_base, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # --- REDIMENSIONAMENTO PROPORCIONAL (v3.1d) ---
        altura_alvo = 500 # Você define a altura; a largura se ajusta sozinha [cite: 2026-02-28]
        h_orig, w_orig = vis.shape[:2]
        proporcao = w_orig / h_orig
        largura_final = int(altura_alvo * proporcao)
        
        # 2. Otimização de Wi-Fi: Redimensiona o streaming e baixa qualidade 
        vis_light = cv2.resize(vis, (largura_final, altura_alvo), interpolation=cv2.INTER_AREA)
        ret, buffer = cv2.imencode('.jpg', vis_light, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.04) # Limita streaming a ~25fps para poupar CPU 

# --- ROTA: AO VIVO ROLO ---

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- NOVA ROTA: PREVIEW DO ROLO (Últimos frames gravados) ---

@app.route('/preview_feed')
def preview_feed():
    def generate_preview():
        while True:
            files = sorted([f for f in os.listdir(CAPTURE_PATH) if f.endswith('.jpg')])
            last_frames = files[-96:] if len(files) > 0 else [] # 4 segundos 
            
            if not last_frames:
                time.sleep(0.5); continue

            for frame_file in last_frames:
                path = os.path.join(CAPTURE_PATH, frame_file)
                try:
                    img = cv2.imread(path)
                    # ROTAÇÃO 90 GRAUS À ESQUERDA
                    img_rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    ret, buffer = cv2.imencode('.jpg', img_rot, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
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

# --- INTERFACE ATUALIZADA (Painel Duplo) ---

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
            
            <div style="display:flex; gap:20px;">
                <div style="text-align:center;">
                    <p>AO VIVO (AJUSTE)</p>
                    <img src="/video_feed" style="border:2px solid #333; height:550px; object-fit:contain;">
                </div>
                <div style="text-align:center;">
                    <p>PREVIEW (ESTABILIDADE)</p>
                    <img src="/preview_feed" style="border:2px solid #0f0; height:550px; object-fit:contain;">
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