from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import threading
import sys
import time
import logging

# 1. Configuração do Flask e Silenciamento de Logs
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR) 

picam2 = Picamera2()

# 2. Configuração de Hardware (VGA para equilíbrio entre Foco e Performance)
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)

picam2.set_controls({
    "ExposureTime": 1000,
    "AnalogueGain": 1.0,
    "FrameRate": 60
})
picam2.start()

# --- GEOMETRIA E VARIÁVEIS DE ESTADO ---
# Restringimos o X para evitar o "breu" das laterais da máquina
ROI_Y, ROI_H = 100, 60
ROI_X, ROI_W = 200, 240  
LINHA_X, MARGEM = 320, 10
THRESH_VAL = 110

contador = 0
furo_na_linha = False
ultimo_frame_bruto = None
ultimo_frame_binario = None
lista_contornos_debug = []

# --- THREAD: PAINEL DE CONTROLE (Terminal) ---

def painel_controle():
    global contador, THRESH_VAL, ultimo_frame_bruto
    time.sleep(2)
    
    print("\n" + "="*45)
    print("  MINIOLA TERMINAL CONTROL v1.6.3")
    print("  Comandos + [Enter]:")
    print("  r       : Reseta contador")
    print("  a       : Auto-ajuste de Threshold (Otsu na ROI)")
    print("  e [val] : ExposureTime (ex: e 800)")
    print("  g [val] : AnalogueGain (ex: g 2.0)")
    print("  f [val] : FrameRate    (ex: f 60)")
    print("  t [val] : Threshold manual (ex: t 120)")
    print("="*45)

    while True:
        try:
            entrada = input("\nComando >> ").split()
            if not entrada: continue
            cmd = entrada[0].lower()
            
            if cmd == 'r':
                contador = 0
                print(">> [OK] Contador zerado.")
                
            elif cmd == 'a':
                if ultimo_frame_bruto is not None:
                    gray = cv2.cvtColor(ultimo_frame_bruto, cv2.COLOR_RGB2GRAY)
                    # Analisa apenas a "janela" central de perfurações
                    roi_analise = gray[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
                    val_otsu, _ = cv2.threshold(roi_analise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    # Margem de 10% para garantir que a binarização seja robusta
                    THRESH_VAL = int(val_otsu * 1.71)
                    print(f">> [AUTO] Valor Otsu: {int(val_otsu)} | Aplicado: {THRESH_VAL}")
                else:
                    print(">> [ERRO] Sem frame para análise.")

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
            elif cmd == 't' and len(entrada) > 1:
                THRESH_VAL = int(entrada[1])
                print(f">> [SET] Threshold: {THRESH_VAL}")
        except Exception as e:
            print(f">> [ERRO]: {e}")

# --- THREAD: LÓGICA DO SCANNER ---

def logica_scanner():
    global contador, furo_na_linha, ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug
    while True:
        frame_raw = picam2.capture_array()
        if frame_raw is None: continue
            
        gray = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2GRAY)
        # O processamento ocorre apenas dentro da "Caixa de Densitometria"
        roi = gray[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
        
        _, binary = cv2.threshold(roi, THRESH_VAL, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        furo_agora = False
        temp_contornos = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            if 100 < area < 6000:
                # Centro X global (relativo ao frame de 640px)
                centro_x_global = x + ROI_X + (w // 2)
                temp_contornos.append({'rect': (x, y, w, h), 'color': (0, 255, 0)})
                
                if abs(centro_x_global - LINHA_X) < MARGEM:
                    furo_agora = True
                    if not furo_na_linha:
                        contador += 1
                        furo_na_linha = True
            else:
                temp_contornos.append({'rect': (x, y, w, h), 'color': (0, 0, 255)})

        if not furo_agora:
            furo_na_linha = False
            
        ultimo_frame_bruto = frame_raw
        ultimo_frame_binario = binary
        lista_contornos_debug = temp_contornos

# --- FLASK: VISUALIZAÇÃO ---

def generate_frames():
    while True:
        if ultimo_frame_bruto is None or ultimo_frame_binario is None:
            time.sleep(0.01)
            continue
        
        vis = ultimo_frame_bruto.copy()
        
        # Desenha a ROI restrita (Caixa Cinza)
        cor_gat = (0, 255, 0) if furo_na_linha else (0, 0, 255)
        cv2.rectangle(vis, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (150, 150, 150), 1)
        cv2.line(vis, (LINHA_X, ROI_Y), (LINHA_X, ROI_Y + ROI_H), cor_gat, 2)

        # Desenha contornos compensando o deslocamento da ROI
        for item in lista_contornos_debug:
            x, y, w, h = item['rect']
            cv2.rectangle(vis, (x + ROI_X, y + ROI_Y), (x + w + ROI_X, y + h + ROI_Y), item['color'], 2)

        # Monta o canvas binário respeitando a posição central
        bin_rgb = cv2.cvtColor(ultimo_frame_binario, cv2.COLOR_GRAY2RGB)
        canvas_bin = np.zeros_like(vis)
        canvas_bin[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W] = bin_rgb
        
        output = np.hstack((vis, canvas_bin))
        ret, buffer = cv2.imencode('.jpg', output, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/count')
def get_count():
    return str(contador)

@app.route('/')
def index():
    return """
    <html>
        <body style='background:#111; color:#0f0; text-align:center; font-family:monospace;'>
            <h2>MINIOLA DENSITOMETRY MODE</h2>
            <div id='val' style='font-size:100px;'>0</div>
            <img src="/video_feed" style="width:100%; max-width:1200px; border:1px solid #444;">
            <script>
                setInterval(() => {
                    fetch('/count').then(r => r.text()).then(t => { 
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