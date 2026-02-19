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
config = picam2.create_video_configuration(main={"size": (160, 120), "format": "RGB888"})
picam2.configure(config)

picam2.set_controls({
    "ExposureTime": 400,
    "AnalogueGain": 4.0,
    "FrameRate": 120 
})
picam2.start()

# --- GEOMETRIA (Ajuste aqui enquanto olha o preview) ---
ROI_Y, ROI_H = 20, 15
LINHA_X, MARGEM = 80, 8
THRESH_VAL = 110

contador = 0
furo_na_linha = False
ultimo_frame_bruto = None
ultimo_frame_binario = None

# --- FUNÇÕES ---

def escutar_teclado():
    global contador
    while True:
        comando = sys.stdin.read(1)
        if comando.lower() == 'r':
            contador = 0
            print("\n[RESET] Contador zerado.")

def logica_scanner():
    global contador, furo_na_linha, ultimo_frame_bruto, ultimo_frame_binario
    while True:
        frame_raw = picam2.capture_array()
        if frame_raw is None: continue
            
        gray = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2GRAY)
        roi = gray[ROI_Y:ROI_Y+ROI_H, :]
        
        _, binary = cv2.threshold(roi, THRESH_VAL, 255, cv2.THRESH_BINARY)
        # Opcional: descomente a linha abaixo se houver muito ruído no binário
        # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        furo_agora = False
        for cnt in contours:
            if 10 < cv2.contourArea(cnt) < 400:
                x, y, w, h = cv2.boundingRect(cnt)
                centro_x = x + (w // 2)
                
                if abs(centro_x - LINHA_X) < MARGEM:
                    furo_agora = True
                    if not furo_na_linha:
                        contador += 1
                        furo_na_linha = True
                    break

        if not furo_agora:
            furo_na_linha = False
            
        ultimo_frame_bruto = frame_raw
        ultimo_frame_binario = binary

def generate_frames():
    while True:
        if ultimo_frame_bruto is None or ultimo_frame_binario is None:
            time.sleep(0.01)
            continue
        
        # 1. Base colorida para o preview
        vis = ultimo_frame_bruto.copy()
        
        # 2. Desenha a área do ROI (Retângulo cinza) e a Linha de Gatilho
        cor_gatilho = (0, 255, 0) if furo_na_linha else (0, 0, 255)
        cv2.rectangle(vis, (0, ROI_Y), (160, ROI_Y + ROI_H), (200, 200, 200), 1)
        cv2.line(vis, (LINHA_X, ROI_Y), (LINHA_X, ROI_Y + ROI_H), cor_gatilho, 2)

        # 3. Prepara o lado Binário (Fatia do ROI)
        bin_rgb = cv2.cvtColor(ultimo_frame_binario, cv2.COLOR_GRAY2RGB)
        # Criamos um fundo preto para alinhar o binário com o original
        canvas_bin = np.zeros_like(vis)
        canvas_bin[ROI_Y:ROI_Y+ROI_H, :] = bin_rgb
        
        # 4. Empilha e Redimensiona para o navegador ver melhor (320x240 total)
        merged = np.hstack((vis, canvas_bin))
        output = cv2.resize(merged, (640, 240), interpolation=cv2.INTER_NEAREST)

        ret, buffer = cv2.imencode('.jpg', output, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- ROTAS E MAIN ---

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
            <h2>MINIOLA DIAGNOSTICS (160x120)</h2>
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