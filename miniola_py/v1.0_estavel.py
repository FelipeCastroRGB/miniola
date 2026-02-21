from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import threading
import sys
import time

app = Flask(__name__)
picam2 = Picamera2()

# 1. Configuração de Hardware (160x120)
config = picam2.create_video_configuration(main={"size": (640, 240), "format": "RGB888"})
picam2.configure(config)

picam2.set_controls({
    "ExposureTime": 400,
    "AnalogueGain": 4.0,
    "FrameRate": 120 
})
picam2.start()

# --- GEOMETRIA E CALIBRAÇÃO ---
ROI_Y, ROI_H = 15, 15
LINHA_X, MARGEM = 80, 5
THRESH_VAL = 110

contador = 0
furo_na_linha = False
ultimo_frame_bruto = None
ultimo_frame_binario = None
lista_contornos_debug = [] # Buffer para passar os contornos ao preview

# --- FUNÇÕES ---

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
            
            # Filtro de área
            if 10 < area < 400:
                centro_x = x + (w // 2)
                # Marcar como 'detectado' (Verde)
                temp_contornos.append({'rect': (x, y, w, h), 'color': (0, 255, 0)})
                
                if abs(centro_x - LINHA_X) < MARGEM:
                    furo_agora = True
                    if not furo_na_linha:
                        contador += 1
                        furo_na_linha = True
            else:
                # Marcar como 'ruído/rejeitado' (Vermelho) - apenas para diagnóstico
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
        
        # 1. Desenhar a Linha de Gatilho e ROI
        cor_gatilho = (0, 255, 0) if furo_na_linha else (0, 0, 255)
        cv2.rectangle(vis, (0, ROI_Y), (160, ROI_Y + ROI_H), (100, 100, 100), 1)
        cv2.line(vis, (LINHA_X, ROI_Y), (LINHA_X, ROI_Y + ROI_H), cor_gatilho, 1)

        # 2. Desenhar os quadradinhos de detecção (Debug)
        for item in lista_contornos_debug:
            x, y, w, h = item['rect']
            # Ajustamos o Y para a posição global do frame (somando ROI_Y)
            cv2.rectangle(vis, (x, y + ROI_Y), (x + w, y + h + ROI_Y), item['color'], 1)

        # 3. Lado Binário
        bin_rgb = cv2.cvtColor(ultimo_frame_binario, cv2.COLOR_GRAY2RGB)
        canvas_bin = np.zeros_like(vis)
        canvas_bin[ROI_Y:ROI_Y+ROI_H, :] = bin_rgb
        
        # 4. Saída duplicada
        output = np.hstack((vis, canvas_bin))
        output_res = cv2.resize(output, (640, 240), interpolation=cv2.INTER_NEAREST)

        ret, buffer = cv2.imencode('.jpg', output_res, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- ROTAS FLASK E MAIN PERMANECEM IGUAIS ---

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
        <body style='background:#000; color:#0f0; text-align:center; font-family:monospace;'>
            <h2>MINIOLA CONTOUR DEBUG</h2>
            <div id='val' style='font-size:100px;'>0</div>
            <img src="/video_feed" style="width:95%; border:1px solid #444;">
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