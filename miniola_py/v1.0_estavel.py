from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import threading
import sys

app = Flask(__name__)
picam2 = Picamera2()

# Configuração de captura otimizada para velocidade
config = picam2.create_video_configuration(main={"size": (320, 240), "format": "RGB888"})
picam2.configure(config)

picam2.set_controls({
    "ExposureTime": 500,
    "AnalogueGain": 2.0,
    "FrameRate": 120 # O Zero 2 W estabiliza melhor entre 80-120fps
})
picam2.start()

ROI_Y, ROI_H = 35, 20
LINHA_X, MARGEM = 160, 5

contador = 0
furo_na_linha = False
ultimo_frame_processado = None

# Kernel global para evitar recriação em cada loop
KERNEL = np.ones((5,5), np.uint8)

def logica_scanner():
    """Thread dedicada apenas à contagem, sem interface, para máxima velocidade."""
    global contador, furo_na_linha, ultimo_frame_processado
    
    while True:
        frame_raw = picam2.capture_array()
        # Trabalhar direto no canal de brilho se possível, ou conversão rápida
        gray = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2GRAY)
        
        # ROI focado
        roi = gray[ROI_Y:ROI_Y+ROI_H, :]
        
        # Threshold direto (mais rápido que adaptativo ou com blur)
        _, binary = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)
        
        # Limpeza morfológica rápida
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, KERNEL)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        furo_agora = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 30 < area < 500:
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
            
        # Prepara o frame de visualização APENAS uma vez por ciclo de contagem
        # Reduzimos o desenho aqui
        ultimo_frame_processado = (frame_raw, binary)

def generate_frames():
    while True:
        if ultimo_frame_processado is None:
            continue
        
        frame, binary = ultimo_frame_processado
        
        # Desenhos mínimos apenas para o Stream
        vis = frame.copy()
        cv2.line(vis, (LINHA_X, ROI_Y), (LINHA_X, ROI_Y + ROI_H), (0, 255, 0), 1)
        
        # Stack horizontal otimizado
        binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        display_binary = np.zeros_like(vis)
        display_binary[ROI_Y:ROI_Y+ROI_H, :] = binary_rgb
        
        output = np.hstack((vis, display_binary))
        
        # Encode com qualidade menor (25) para não sobrecarregar o Wi-Fi/CPU
        ret, buffer = cv2.imencode('.jpg', output, [int(cv2.IMWRITE_JPEG_QUALITY), 25])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- MENSAGEM DE COMMIT ---
# perf: separa lógica de contagem e visualização em threads e otimiza filtros