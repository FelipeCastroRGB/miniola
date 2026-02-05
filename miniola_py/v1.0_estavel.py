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
    "ExposureTime": 100,
    "AnalogueGain": 5.0,
    "FrameRate": 80 
})
picam2.start()

# --- GEOMETRIA E CALIBRAÇÃO ---
ROI_Y, ROI_H = 0, 60   # Área de busca das perfurações
LINHA_X = 160          # Gatilho central
MARGEM = 18            # Tolerância para captura em velocidade
ADAPTIVE_BLOCK = 11    # Sensibilidade: vizinhança de pixels (deve ser ímpar)
ADAPTIVE_C = 2         # Sensibilidade: quanto maior, mais ignora cinzas claros

contador = 0
furo_na_linha = False

def generate_frames():
    global contador, furo_na_linha
    while True:
        # 1. Captura e conversão
        frame_raw = picam2.capture_array()
        frame = cv2.cvtColor(frame_raw, cv2.COLOR_BGRA2BGR)

        # 2. PROCESSAMENTO ADAPTATIVO (Ideal para filmes transparentes)
        roi = frame[ROI_Y:ROI_Y+ROI_H, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # O Adaptive Threshold detecta bordas por contraste local, não por brilho fixo
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, ADAPTIVE_BLOCK, ADAPTIVE_C
        )

        # 3. FILTRO MORFOLÓGICO (Limpa "chuviscos" no suporte transparente)
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 4. CONTORNO E GATILHO
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        furo_agora = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filtro de Área e Proporção (Perfuração é aproximadamente quadrada/retangular)
            if 150 < area < 1200:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w)/h
                
                if 0.7 < aspect_ratio < 1.7:
                    centro_x = x + (w // 2)

                    # Se o contorno passar no teste, desenha um quadrado branco discreto no ROI
                    cv2.rectangle(frame, (x, y + ROI_Y), (x + w, y + h + ROI_Y), (255, 255, 255), 1)

                    if abs(centro_x - LINHA_X) < MARGEM:
                        furo_agora = True
                        if not furo_na_linha:
                            contador += 1
                            furo_na_linha = True
                            break # Já contou, pode sair do loop de contornos

        if not furo_agora:
            furo_na_linha = False

        # --- VISUALIZAÇÃO PROFISSIONAL ---
        # Guia de alinhamento (Linha vertical tênue)
        cor_guia = (0, 255, 0) if furo_na_linha else (100, 100, 100)
        cv2.line(frame, (LINHA_X, ROI_Y), (LINHA_X, ROI_Y + ROI_H), cor_guia, 1)
        
        # Marcação do limite do ROI
        cv2.rectangle(frame, (0, ROI_Y), (320, ROI_Y + ROI_H), (50, 50, 50), 1)

        # Informações fora da área de visão central (Canto superior)
        cv2.putText(frame, f"PERF: {contador}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Encode JPEG com compressão para manter o FPS estável no navegador
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
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