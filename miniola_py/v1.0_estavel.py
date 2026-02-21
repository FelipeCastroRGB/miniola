from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import threading
import sys
import time

app = Flask(__name__)
picam2 = Picamera2()

# 1. MÁXIMA QUALIDADE PARA FOCO
# Usando 640x480 para manter a proporção nativa do sensor ov5647
config = picam2.create_video_configuration(main={"size": (640, 720), "format": "RGB888"})
picam2.configure(config)

picam2.set_controls({
    "ExposureTime": 1200, # Aumentado para compensar o ganho baixo
    "AnalogueGain": 2.0,  # Ganho mínimo = Imagem limpa sem ruído
    "FrameRate": 60       # Baixamos o FPS para priorizar a exposição e qualidade
})
picam2.start()

# --- GEOMETRIA TEMPORÁRIA (Ajustada para 640x480) ---
ROI_Y, ROI_H = 140, 60   # ROI maior para facilitar a visualização do foco
LINHA_X, MARGEM = 320, 15
THRESH_VAL = 206

contador = 0
furo_na_linha = False
ultimo_frame_bruto = None
ultimo_frame_binario = None
lista_contornos_debug = []

def escutar_teclado():
    global contador
    while True:
        comando = sys.stdin.read(1)
        if comando.lower() == 'r':
            contador = 0
            print("\n[RESET] Contador zerado.")

def logica_scanner():
    global contador, furo_na_linha, ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug
    while True:
        frame_raw = picam2.capture_array()
        if frame_raw is None: continue
            
        gray = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2GRAY)
        roi = gray[ROI_Y:ROI_Y+ROI_H, :]
        
        _, binary = cv2.threshold(roi, THRESH_VAL, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        furo_agora = False
        temp_contornos = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Área ajustada para a nova resolução de 640px
            if 500 < area < 2000:
                centro_x = x + (w // 2)
                temp_contornos.append({'rect': (x, y, w, h), 'color': (0, 255, 0)})
                if abs(centro_x - LINHA_X) < MARGEM:
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

def generate_frames():
    while True:
        if ultimo_frame_bruto is None or ultimo_frame_binario is None:
            time.sleep(0.01)
            continue
        
        vis = ultimo_frame_bruto.copy()
        
        # Desenho de guias em alta definição
        cv2.rectangle(vis, (0, ROI_Y), (640, ROI_Y + ROI_H), (255, 255, 255), 1)
        cv2.line(vis, (LINHA_X, ROI_Y), (LINHA_X, ROI_Y + ROI_H), (0, 255, 255), 2)

        for item in lista_contornos_debug:
            x, y, w, h = item['rect']
            cv2.rectangle(vis, (x, y + ROI_Y), (x + w, y + h + ROI_Y), item['color'], 2)

        # Output lado a lado (1280x480 para inspeção detalhada)
        bin_rgb = cv2.cvtColor(ultimo_frame_binario, cv2.COLOR_GRAY2RGB)
        canvas_bin = np.zeros_like(vis)
        canvas_bin[ROI_Y:ROI_Y+ROI_H, :] = bin_rgb
        
        output = np.hstack((vis, canvas_bin))

        # Qualidade 95 para inspeção de foco manual
        ret, buffer = cv2.imencode('.jpg', output, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
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
            <h2>MINIOLA FOCUS MODE (640x480)</h2>
            <div id='val' style='font-size:80px;'>0</div>
            <img src="/video_feed" style="width:100%; max-width:1280px; border:2px solid #555;">
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
        threading.Thread(target=escutar_teclado, daemon=True).start()
        threading.Thread(target=logica_scanner, daemon=True).start()
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()