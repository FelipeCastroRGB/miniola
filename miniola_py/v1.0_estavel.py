from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import threading
import sys

app = Flask(__name__)
picam2 = Picamera2()

# Resolução de captura focada (320x240 é o ponto ideal para o Pi Zero)
config = picam2.create_video_configuration(main={"size": (320, 240)})
picam2.configure(config)

# SHUTTER CURTO: 1000us (1ms) é ideal para "congelar" o filme em movimento
picam2.set_controls({
    "ExposureTime": 100,
    "AnalogueGain": 3.0,
    "FrameRate": 80 # Tentamos extrair o máximo de velocidade do sensor
})
picam2.start()

# --- GEOMETRIA RESTRITA (ROI) ---
# Aqui definimos a fatia exata. No 320x240, os furos devem passar por aqui:
ROI_Y, ROI_H = 0, 40  # Ajuste para que a fatia vermelha cubra SÓ os furos
LINHA_X = 160         # Gatilho central
MARGEM = 15           # Janela de captura para altas velocidades

contador = 0
furo_na_linha = False

def generate_frames():
    global contador, furo_na_linha
    while True:
        # 1. Captura rápida e conversão imediata para BGR (3 canais)
        frame_raw = picam2.capture_array()
        frame = cv2.cvtColor(frame_raw, cv2.COLOR_BGRA2BGR)

        # 2. PROCESSAMENTO RESTRITO: Cortamos a imagem ANTES de qualquer filtro
        # Isso faz o OpenCV trabalhar com uma imagem minúscula de 320x50 pixels
        roi = frame[ROI_Y:ROI_Y+ROI_H, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Binarização direta (Furo Branco)
        # Usamos um threshold fixo para não perder tempo calculando Otsu
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # 3. CONTORNO E GATILHO
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        furo_agora = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 150 < area < 1000:
                x, y, w, h = cv2.boundingRect(cnt)
                centro_x = x + (w // 2)

                # Feedback visual no frame original (dentro da área do ROI)
                cv2.rectangle(frame, (x, y + ROI_Y), (x + w, y + h + ROI_Y), (255, 255, 255), 2)

                if abs(centro_x - LINHA_X) < MARGEM:
                    furo_agora = True
                    if not furo_na_linha:
                        contador += 1
                        furo_na_linha = True

        if not furo_agora:
            furo_na_linha = False

        # --- VISUALIZAÇÃO OTIMIZADA ---
        # Desenha a linha de gatilho e a zona do ROI
        cor_linha = (0, 0, 255) if furo_na_linha else (0, 255, 0)
        cv2.line(frame, (LINHA_X, 0), (LINHA_X, 240), cor_linha, 2)
        cv2.rectangle(frame, (0, ROI_Y), (320, ROI_Y + ROI_H), (0, 0, 255), 1)

        cv2.putText(frame, f"PERF: {contador}", (10, 40), 1, 2, (0, 255, 0), 2)

        # Encode JPEG com qualidade média para priorizar FPS no navegador
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 20])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<html><body style='background:#000; color:white; text-align:center;'><img src='/video_feed' style='width:90%; border:1px solid #333;'></body></html>"

def escutar_teclado():
    global contador
    print("--- COMANDOS DO TECLADO ---")
    print("Pressione 'r' + Enter para RESETAR o contador")
    print("---------------------------")
    while True:
        comando = sys.stdin.read(1) # Lê um caractere por vez
        if comando.lower() == 'r':
            contador = 0
            print("\n[INFO] Contador resetado para zero!")

if __name__ == '__main__':
    # Inicia a thread que cuida do teclado antes de abrir o Flask
    threading.Thread(target=escutar_teclado, daemon=True).start()

    # Roda o servidor Flask (bloqueante)
    app.run(host='0.0.0.0', port=5000, threaded=True)