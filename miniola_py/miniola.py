import os
import time
import cv2
import numpy as np
import threading
import logging
from datetime import datetime
from flask import Flask, Response
from picamera2 import Picamera2

# 1. CONFIGURAÇÕES DE SESSÃO E ARQUIVAMENTO
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_PATH = f"capturas/sessao_{SESSION_ID}"
os.makedirs(BASE_PATH, exist_ok=True)

# 2. CONFIGURAÇÃO DO FLASK (Silenciando logs excessivos)
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# 3. CONFIGURAÇÃO DA CÂMERA (Foco em Shutter Rápido para minimizar Rolling Shutter)
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"})
picam2.configure(config)

picam2.set_controls({
    "ExposureTime": 800,  # Exp. curta = menos 'jelly effect' no movimento manual
    "AnalogueGain": 2.0,   # Ajustar conforme a fonte de luz (backlight)
    "FrameRate": 60        # Alta taxa para precisão na detecção de perfuração
})
picam2.start()

# --- GEOMETRIA E ESTADO GLOBAL ---
ROI_Y, ROI_H = 100, 60
ROI_X, ROI_W = 200, 240  
LINHA_X, MARGEM = 320, 10
THRESH_VAL = 110

# Variáveis de Controle
contador_perfuracoes = 0
contador_fotogramas = 0
furo_na_linha = False
MODO_GRAVACAO = False
take_atual_path = ""

# Buffers para Visualização
ultimo_frame_bruto = None
ultimo_frame_binario = None
lista_contornos_debug = []

# --- FUNÇÕES DE SUPORTE ---

def salvar_fotograma_async(frame, num_frame, path):
    """Salva a imagem em thread separada para não travar a detecção."""
    if not path: return
    filename = f"{path}/frame_{num_frame:06d}.jpg"
    # Convertendo RGB (picamera) para BGR (opencv) para salvar
    cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

# --- THREAD: PAINEL DE CONTROLE (Terminal) ---

def painel_controle():
    global contador_perfuracoes, THRESH_VAL, MODO_GRAVACAO, take_atual_path, contador_fotogramas
    time.sleep(2)
    
    print("\n" + "="*50)
    print("      MINIOLA: INSTRUMENTO DE VISIONAMENTO v2.0")
    print("="*50)
    print(" [S] - INICIAR CAPTURA (Start Take)")
    print(" [P] - PARAR CAPTURA (Stop/Pause)")
    print(" [R] - RESETAR Contador de Perfurações")
    print(" [A] - AUTO-AJUSTE de Threshold (Otsu na ROI)")
    print(" [T] [val] - Threshold Manual (ex: t 120)")
    print(" [E] [val] - Exposure Time (ex: e 1000)")
    print("="*50)

    while True:
        try:
            entrada = input("\nMINIOLA >> ").split()
            if not entrada: continue
            cmd = entrada[0].lower()
            
            if cmd == 's':
                if not MODO_GRAVACAO:
                    timestamp = datetime.now().strftime("%H%M%S")
                    take_atual_path = f"{BASE_PATH}/take_{timestamp}"
                    os.makedirs(take_atual_path, exist_ok=True)
                    contador_fotogramas = 0
                    MODO_GRAVACAO = True
                    print(f">> [GRAVANDO] Take iniciado em: {take_atual_path}")
                else:
                    print(">> [AVISO] Já existe uma gravação em curso.")

            elif cmd == 'p':
                if MODO_GRAVACAO:
                    MODO_GRAVACAO = False
                    print(f">> [PARADO] Take finalizado. Total de frames: {contador_fotogramas}")
                else:
                    print(">> [AVISO] Não há gravação ativa.")

            elif cmd == 'r':
                contador_perfuracoes = 0
                print(">> [OK] Contador de perfurações zerado.")

            elif cmd == 'a':
                if ultimo_frame_bruto is not None:
                    gray = cv2.cvtColor(ultimo_frame_bruto, cv2.COLOR_RGB2GRAY)
                    roi_analise = gray[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
                    val_otsu, _ = cv2.threshold(roi_analise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    THRESH_VAL = int(val_otsu * 1.5) # Ajuste empírico para luz de fundo
                    print(f">> [AUTO] Threshold ajustado para: {THRESH_VAL}")

            elif cmd == 't' and len(entrada) > 1:
                THRESH_VAL = int(entrada[1])
                print(f">> [SET] Threshold: {THRESH_VAL}")

            elif cmd == 'e' and len(entrada) > 1:
                picam2.set_controls({"ExposureTime": int(entrada[1])})
                print(f">> [SET] Exposure: {entrada[1]}")

        except Exception as e:
            print(f">> [ERRO]: {e}")

# --- THREAD: LÓGICA DO SCANNER ---

def logica_scanner():
    global contador_perfuracoes, contador_fotogramas, furo_na_linha
    global ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug
    
    while True:
        frame_raw = picam2.capture_array()
        if frame_raw is None: continue
            
        # Processamento focado na ROI das perfurações
        gray = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2GRAY)
        roi = gray[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
        _, binary = cv2.threshold(roi, THRESH_VAL, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        furo_agora = False
        temp_contornos = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filtro de tamanho para evitar ruído (ajustar conforme a distância da câmera)
            if 150 < area < 8000:
                centro_x_global = x + ROI_X + (w // 2)
                temp_contornos.append({'rect': (x, y, w, h), 'color': (0, 255, 0)})
                
                # Verifica se a perfuração cruzou a linha de gatilho
                if abs(centro_x_global - LINHA_X) < MARGEM:
                    furo_agora = True
                    if not furo_na_linha:
                        contador_perfuracoes += 1
                        furo_na_linha = True
                        
                        # GATILHO: A cada 4 perfurações (Padrão 35mm 4-perf)
                        if MODO_GRAVACAO and (contador_perfuracoes % 4 == 0):
                            contador_fotogramas += 1
                            # Captura em thread separada para manter a fluidez
                            threading.Thread(
                                target=salvar_fotograma_async, 
                                args=(frame_raw.copy(), contador_fotogramas, take_atual_path),
                                daemon=True
                            ).start()
            else:
                temp_contornos.append({'rect': (x, y, w, h), 'color': (0, 0, 255)})

        if not furo_agora:
            furo_na_linha = False
            
        ultimo_frame_bruto = frame_raw
        ultimo_frame_binario = binary
        lista_contornos_debug = temp_contornos

# --- FLASK: INTERFACE DE VISIONAMENTO ---

def generate_frames():
    while True:
        if ultimo_frame_bruto is None:
            time.sleep(0.01)
            continue
        
        vis = ultimo_frame_bruto.copy()
        
        # Overlay de Status
        cor_status = (0, 255, 0) if MODO_GRAVACAO else (200, 200, 200)
        txt_modo = "GRAVANDO" if MODO_GRAVACAO else "VISIONAMENTO"
        cv2.putText(vis, f"MODO: {txt_modo}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, cor_status, 2)
        cv2.putText(vis, f"PERF: {contador_perfuracoes}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis, f"FRAMES: {contador_fotogramas}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Desenho da ROI e Linha de Gatilho
        cor_linha = (0, 255, 0) if furo_na_linha else (0, 0, 255)
        cv2.rectangle(vis, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (255, 255, 255), 1)
        cv2.line(vis, (LINHA_X, ROI_Y), (LINHA_X, ROI_Y + ROI_H), cor_linha, 3)

        # Encode para Streaming
        ret, buffer = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
        <body style='background:#111; color:#fff; text-align:center; font-family:sans-serif;'>
            <h1>MINIOLA DIGITAL VIEWER</h1>
            <img src="/video_feed" style="width:80%; border:2px solid #333;">
            <p>Use o terminal para comandos de captura (S/P) e ajustes técnicos.</p>
        </body>
    </html>
    """

if __name__ == '__main__':
    try:
        threading.Thread(target=painel_controle, daemon=True).start()
        threading.Thread(target=logica_scanner, daemon=True).start()
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()