from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import threading
import sys
import time

app = Flask(__name__)
picam2 = Picamera2()

# 1. RESOLUÇÃO MÍNIMA PARA MÁXIMA VELOCIDADE
# 160x120 é o "sweet spot" para o Pi Zero 2 W em alta frequência
config = picam2.create_video_configuration(main={"size": (160, 120), "format": "RGB888"})
picam2.configure(config)

picam2.set_controls({
    "ExposureTime": 400, # Ajuste conforme a intensidade do seu LED
    "AnalogueGain": 4.0,
    "FrameRate": 150      # Alvo de 150 FPS reais
})
picam2.start()

# --- GEOMETRIA RECALIBRADA (Para 160x120) ---
ROI_Y, ROI_H = 20, 15  # ROI ultra-focado
LINHA_X, MARGEM = 80, 8
THRESH_VAL = 110

contador = 0
furo_na_linha = False
ultimo_frame_bruto = None
ultimo_frame_binario = None

def logica_scanner():
    """Lógica minimalista para não perder frames."""
    global contador, furo_na_linha, ultimo_frame_bruto, ultimo_frame_binario
    
    while True:
        frame_raw = picam2.capture_array()
        
        # Extração rápida do canal de cinza
        gray = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2GRAY)
        roi = gray[ROI_Y:ROI_Y+ROI_H, :]
        
        # Threshold simples sem filtros extras (Velocidade Pura)
        _, binary = cv2.threshold(roi, THRESH_VAL, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        furo_agora = False
        for cnt in contours:
            # Em 160x120, as áreas são menores. Ajustado de 30 para 10.
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
        ultimo_frame_binario = binary

def generate_frames():
    while True:
        if ultimo_frame_bruto is None:
            time.sleep(0.01)
            continue
        
        # Stream em baixa qualidade para não roubar CPU da lógica
        vis = cv2.resize(ultimo_frame_bruto, (320, 240), interpolation=cv2.INTER_NEAREST)
        
        ret, buffer = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 20])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ... (Rotas Flask e Teclado permanecem as mesmas) ...

if __name__ == '__main__':
    try:
        threading.Thread(target=logica_scanner, daemon=True).start()
        threading.Thread(target=escutar_teclado, daemon=True).start()
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()