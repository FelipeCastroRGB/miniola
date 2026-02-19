from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import threading
import sys
import time

app = Flask(__name__)
picam2 = Picamera2()

# 1. Configuração de Hardware Otimizada
config = picam2.create_video_configuration(main={"size": (320, 240), "format": "RGB888"})
picam2.configure(config)

picam2.set_controls({
    "ExposureTime": 500,
    "AnalogueGain": 2.0,
    "FrameRate": 100 # Reduzido levemente para estabilidade térmica no Pi Zero
})
picam2.start()

# --- GEOMETRIA E CALIBRAÇÃO ---
ROI_Y, ROI_H = 35, 20
LINHA_X, MARGEM = 160, 5
THRESH_VAL = 100 # Ajuste isso se o furo não ficar branco no binário

contador = 0
furo_na_linha = False
ultimo_frame_bruto = None
ultimo_frame_binario = None

# Kernel global para economizar CPU
KERNEL = np.ones((5,5), np.uint8)

def logica_scanner():
    """Thread dedicada exclusivamente à contagem das perfurações."""
    global contador, furo_na_linha, ultimo_frame_bruto, ultimo_frame_binario
    
    while True:
        frame_raw = picam2.capture_array()
        # Conversão rápida para escala de cinza
        gray = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2GRAY)
        
        # Recorte do ROI
        roi = gray[ROI_Y:ROI_Y+ROI_H, :]
        
        # Processamento de imagem
        _, binary = cv2.threshold(roi, THRESH_VAL, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, KERNEL)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        furo_agora = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 30 < area < 600:
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
            
        # Atualiza os buffers globais para o Flask usar quando quiser
        ultimo_frame_bruto = frame_raw
        ultimo_frame_binario = binary

def generate_frames():
    """Gera o streaming de vídeo para o navegador."""
    while True:
        if ultimo_frame_bruto is None or ultimo_frame_binario is None:
            time.sleep(0.01)
            continue
        
        # Criar cópia para desenho
        vis = ultimo_frame_bruto.copy()
        
        # Desenhos técnicos de alinhamento
        cor_linha = (0, 255, 0) if furo_na_linha else (0, 0, 255)
        cv2.line(vis, (LINHA_X, ROI_Y), (LINHA_X, ROI_Y + ROI_H), cor_linha, 2)
        cv2.rectangle(vis, (0, ROI_Y), (320, ROI_Y + ROI_H), (100, 100, 100), 1)

        # Preparar visualização Lado a Lado
        binary_rgb = cv2.cvtColor(ultimo_frame_binario, cv2.COLOR_GRAY2RGB)
        display_binary = np.zeros_like(vis)
        display_binary[ROI_Y:ROI_Y+ROI_H, :] = binary_rgb
        
        output = np.hstack((vis, display_binary))
        
        # Qualidade 30 para não travar o Wi-Fi do Pi Zero
        ret, buffer = cv2.imencode('.jpg', output, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
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
        <body style='background:#000; color:#0f0; text-align:center; font-family:monospace;'>
            <h1 style='margin-bottom:0;'>MINIOLA V1.3 OPTIMIZED</h1>
            <div id='val' style='font-size:120px;'>0</div>
            <img src="/video_feed" style="width:90%; border:2px solid #333;">
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

def escutar_teclado():
    global contador
    while True:
        comando = sys.stdin.read(1)
        if comando.lower() == 'r':
            contador = 0
            print("\n[RESET] Contador zerado.")

if __name__ == '__main__':
    try:
        # Inicia as threads de suporte
        threading.Thread(target=logica_scanner, daemon=True).start()
        threading.Thread(target=escutar_teclado, daemon=True).start()
        
        print("[SISTEMA] Iniciando servidor Flask...")
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n[ENCERRANDO] Finalizando...")
    finally:
        picam2.stop()
        print("[OK] Câmera liberada.")