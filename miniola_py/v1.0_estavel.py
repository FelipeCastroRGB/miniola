from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import threading
import sys
import time

app = Flask(__name__)
picam2 = Picamera2()

# 1. Configuração de Hardware (160x120 para máxima velocidade)
config = picam2.create_video_configuration(main={"size": (160, 120), "format": "RGB888"})
picam2.configure(config)

picam2.set_controls({
    "ExposureTime": 400,
    "AnalogueGain": 4.0,
    "FrameRate": 150 
})
picam2.start()

# --- GEOMETRIA ---
ROI_Y, ROI_H = 20, 15
LINHA_X, MARGEM = 80, 8
THRESH_VAL = 110

contador = 0
furo_na_linha = False
ultimo_frame_bruto = None

# --- FUNÇÕES DE SUPORTE ---

def escutar_teclado():
    """Monitora o terminal para resetar o contador."""
    global contador
    print("[SISTEMA] Teclado ativo: Pressione 'r' para resetar.")
    while True:
        comando = sys.stdin.read(1)
        if comando.lower() == 'r':
            contador = 0
            print("\n[RESET] Contador zerado via teclado.")

def logica_scanner():
    """Thread de alta prioridade para contagem de perfurações."""
    global contador, furo_na_linha, ultimo_frame_bruto
    
    while True:
        frame_raw = picam2.capture_array()
        if frame_raw is None:
            continue
            
        # Extração de cinza rápida
        gray = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2GRAY)
        roi = gray[ROI_Y:ROI_Y+ROI_H, :]
        
        # Threshold simples
        _, binary = cv2.threshold(roi, THRESH_VAL, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        furo_agora = False
        for cnt in contours:
            if 10 < cv2.contourArea(cnt) < 300:
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

def generate_frames():
    """Streaming de vídeo otimizado para o Flask."""
    while True:
        if ultimo_frame_bruto is None:
            time.sleep(0.01)
            continue
        
        # Upscale barato apenas para visualização no navegador
        vis = cv2.resize(ultimo_frame_bruto, (320, 240), interpolation=cv2.INTER_NEAREST)
        
        # Desenho da linha de gatilho no preview
        cor = (0, 255, 0) if furo_na_linha else (0, 0, 255)
        cv2.line(vis, (LINHA_X*2, 0), (LINHA_X*2, 240), cor, 2)

        ret, buffer = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 25])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- ROTAS FLASK ---

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
            <h1>MINIOLA HIGH-SPEED</h1>
            <div id='val' style='font-size:120px;'>0</div>
            <img src="/video_feed" style="width:80%; border:2px solid #333;">
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

# --- EXECUÇÃO ---

if __name__ == '__main__':
    try:
        # Inicia as threads antes do Flask
        t1 = threading.Thread(target=escutar_teclado, daemon=True)
        t2 = threading.Thread(target=logica_scanner, daemon=True)
        
        t1.start()
        t2.start()
        
        print("[SISTEMA] Servidor iniciado em http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        print("[OK] Câmera liberada.")