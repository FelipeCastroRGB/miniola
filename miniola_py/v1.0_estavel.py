from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import threading
import sys
import time

app = Flask(__name__)
picam2 = Picamera2()

# Configuração de captura focada
config = picam2.create_video_configuration(main={"size": (320, 240)})
picam2.configure(config)

# Exposição curta para evitar borrão (motion blur)
picam2.set_controls({
    "ExposureTime": 800,
    "AnalogueGain": 1.0,
    "FrameRate": 200 
})
picam2.start()

# --- GEOMETRIA E CALIBRAÇÃO ---
ROI_Y, ROI_H = 35, 20   # Área de busca das perfurações
LINHA_X = 160          # Gatilho central
MARGEM = 5            # Tolerância para captura em velocidade

contador = 0
furo_na_linha = False

def generate_frames():
    global contador, furo_na_linha
    while True:
        frame_raw = picam2.capture_array()
        frame = cv2.cvtColor(frame_raw, cv2.COLOR_BGRA2BGR)

        roi = frame[ROI_Y:ROI_Y+ROI_H, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 1. Suavização para eliminar ruído eletrônico (crucial no Pi Zero)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # 2. Threshold Binário com valor mais alto (Busca o brilho real do furo)
        # Se o furo for a parte mais clara, ele sobreviverá a 200.
        _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

        # 3. Limpeza morfológica agressiva (Remove o "chuvisco" que sobrou)
        kernel = np.ones((5,5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        furo_agora = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 4. Filtro Geométrico Rígido (Padrão de perfuração)
            if 30 < area < 500:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w)/h
                
                # Perfurações são quase quadradas (0.8 a 1.4)
                if 0.1 < aspect_ratio < 2.5:                
                    centro_x = x + (w // 2)
                    
                    # Só desenha se passar em todos os filtros
                    cv2.rectangle(frame, (x, y + ROI_Y), (x + w, y + h + ROI_Y), (255, 255, 255), 2)

                    if abs(centro_x - LINHA_X) < MARGEM:
                        furo_agora = True
                        if not furo_na_linha:
                            contador += 1
                            furo_na_linha = True
                            break

        if not furo_agora:
            furo_na_linha = False

        # --- VISUALIZAÇÃO PROFISSIONAL ---
        # Guia de alinhamento (Linha vertical tênue)
        cor_guia = (0, 255, 0) if furo_na_linha else (100, 100, 100)
        cv2.line(frame, (LINHA_X, ROI_Y), (LINHA_X, ROI_Y + ROI_H), cor_guia, 1)
        
        # Marcação do limite do ROI
        cv2.rectangle(frame, (0, ROI_Y), (320, ROI_Y + ROI_H), (50, 50, 50), 1)

        # LOGICA DE VISUALIZAÇÃO LADO A LADO ---
        # 1. Criamos uma versão colorida da imagem binária para podermos juntar com a original
        binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # 2. Como a 'binary' é apenas o ROI, vamos criar uma imagem preta do tamanho do frame
        # e colar o processamento nela para alinhar visualmente
        display_binary = np.zeros_like(frame)
        display_binary[ROI_Y:ROI_Y+ROI_H, :] = binary_rgb

        # 3. Junta as duas imagens (Original na esquerda, Binária na direita)
        # O frame final terá 640x240
        output_duplo = np.hstack((frame, display_binary))

        # Encode JPEG (ajustado para a nova largura)
        ret, buffer = cv2.imencode('.jpg', output_duplo, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/count')
def get_count():
    return str(contador)

@app.route('/')
def index():
    # HTML com o contador gigante e a imagem embaixo para monitoramento
    return """
    <html>
        <body style='background:#000; color:#0f0; text-align:center; font-family:monospace;'>
            <h1 style='margin-bottom:0;'>MINIOLA MONITOR</h1>
            <div id='val' style='font-size:100px;'>0</div>
            <div style="margin-top: 10px;">
                <img src="/video_feed" style="width:80%; border:2px solid #222;">
            </div>
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
            print("\n[RESET] Contador zerado via teclado.")

if __name__ == '__main__':
    threading.Thread(target=escutar_teclado, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)