from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import threading
import sys
import time
import logging
import os

# 1. Configuração do Flask e Silenciamento de Logs
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR) 

picam2 = Picamera2()

# 2. Configuração de Hardware (Aumentamos a resolução para ver o fotograma)
WIDTH, HEIGHT = 1280, 720
config = picam2.create_video_configuration(main={"size": (WIDTH, HEIGHT), "format": "RGB888"})
picam2.configure(config)

# Valores iniciais da câmera
shutter_speed = 20000
gain = 1.0
fps = 30

picam2.set_controls({
    "ExposureTime": shutter_speed,
    "AnalogueGain": gain,
    "FrameRate": fps
})
picam2.start()

# --- GEOMETRIA E VARIÁVEIS DE ESTADO ---
# Valores ajustados para a resolução maior e baseados na sua imagem (perfurações à direita)
ROI_Y, ROI_H = 50, 80
ROI_X, ROI_W = 850, 150  
LINHA_X, MARGEM = 920, 15
THRESH_VAL = 110

# Contadores e Flags
contador_perf = 0
frame_count = 0
furo_na_linha = False
modo_gravacao = False
ultimo_frame_bruto = None
ultimo_frame_binario = None
lista_contornos_debug = []

# --- THREAD: PAINEL DE CONTROLE (Terminal) ---

def painel_controle():
    global contador_perf, frame_count, modo_gravacao, THRESH_VAL, ultimo_frame_bruto
    global shutter_speed, gain, fps
    time.sleep(2)
    
    print("\n" + "="*45)
    print("  MINIOLA VISIONAMENTO & CAPTURA v2.5")
    print("  Baseado no motor de Densitometria")
    print("="*45)
    print("  COMANDOS DE CAPTURA:")
    print("  s       : Iniciar Captura (Gravação)")
    print("  p       : Pausar Captura")
    print("  r       : Reseta contadores (Perf/Frame)")
    print("-" * 45)
    print("  COMANDOS TÉCNICOS:")
    print("  a       : Auto-ajuste de Threshold (Otsu na ROI)")
    print("  t [val] : Threshold manual (ex: t 120)")
    print("  e [val] : ExposureTime (ex: e 20000)")
    print("  g [val] : AnalogueGain (ex: g 2.0)")
    print("  f [val] : FrameRate    (ex: f 30)")
    print("="*45)

    while True:
        try:
            entrada = input("\nComando >> ").split()
            if not entrada: continue
            cmd = entrada[0].lower()
            
            if cmd == 's':
                if not os.path.exists("captura"): os.makedirs("captura")
                modo_gravacao = True
                print(">> [CAPTURA] GRAVANDO!")
            elif cmd == 'p':
                modo_gravacao = False
                print(">> [CAPTURA] PAUSADA.")
            elif cmd == 'r':
                contador_perf = 0
                frame_count = 0
                print(">> [OK] Contadores zerados.")
                
            elif cmd == 'a':
                if ultimo_frame_bruto is not None:
                    gray = cv2.cvtColor(ultimo_frame_bruto, cv2.COLOR_RGB2GRAY)
                    roi_analise = gray[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
                    val_otsu, _ = cv2.threshold(roi_analise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    THRESH_VAL = int(val_otsu * 1.71) # Margem de segurança original
                    print(f">> [AUTO] Valor Otsu: {int(val_otsu)} | Aplicado: {THRESH_VAL}")
                else:
                    print(">> [ERRO] Sem frame para análise.")

            elif cmd == 'e' and len(entrada) > 1:
                val = int(entrada[1])
                shutter_speed = val
                picam2.set_controls({"ExposureTime": val})
                print(f">> [SET] Exposure: {val}")
            elif cmd == 'g' and len(entrada) > 1:
                val = float(entrada[1])
                gain = val
                picam2.set_controls({"AnalogueGain": val})
                print(f">> [SET] Gain: {val}")
            elif cmd == 'f' and len(entrada) > 1:
                val = int(entrada[1])
                fps = val
                picam2.set_controls({"FrameRate": val})
                print(f">> [SET] FrameRate: {val}")
            elif cmd == 't' and len(entrada) > 1:
                THRESH_VAL = int(entrada[1])
                print(f">> [SET] Threshold: {THRESH_VAL}")
        except Exception as e:
            print(f">> [ERRO]: {e}")

# --- THREAD: LÓGICA DO SCANNER ---

def logica_scanner():
    global contador_perf, frame_count, furo_na_linha, ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug
    while True:
        # Usa o array, compatível com sua versão atual do picamera2
        frame_raw = picam2.capture_array()
        if frame_raw is None: continue
            
        gray = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2GRAY)
        roi = gray[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
        
        _, binary = cv2.threshold(roi, THRESH_VAL, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        furo_agora = False
        temp_contornos = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Ajustei a área mínima/máxima pois a resolução agora é maior
            if 100 < area < 8000:
                centro_x_global = x + ROI_X + (w // 2)
                temp_contornos.append({'rect': (x, y, w, h), 'color': (0, 255, 0)})
                
                if abs(centro_x_global - LINHA_X) < MARGEM:
                    furo_agora = True
                    if not furo_na_linha:
                        contador_perf += 1
                        furo_na_linha = True
                        
                        # Lógica de Captura: 1 Frame a cada 4 perfurações
                        if contador_perf % 4 == 0:
                            frame_count += 1
                            if modo_gravacao:
                                frame_bgr = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)
                                cv2.imwrite(f"captura/frame_{frame_count:06d}.jpg", frame_bgr)
            else:
                temp_contornos.append({'rect': (x, y, w, h), 'color': (0, 0, 255)})

        if not furo_agora:
            furo_na_linha = False
            
        ultimo_frame_bruto = frame_raw
        ultimo_frame_binario = binary
        lista_contornos_debug = temp_contornos

# --- FLASK: VISUALIZAÇÃO ---

def generate_frames():
    while True:
        if ultimo_frame_bruto is None:
            time.sleep(0.01)
            continue
        
        # 1. Rotaciona a imagem para visionamento na vertical
        vis_base = cv2.cvtColor(ultimo_frame_bruto, cv2.COLOR_RGB2BGR)
        vis = cv2.rotate(vis_base, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # 2. Status na Tela (Desenhados após a rotação para ficarem no topo da tela)
        cor_status = (0, 255, 0) if modo_gravacao else (0, 255, 255)
        txt_modo = "GRAVANDO" if modo_gravacao else "VISIONAMENTO"
        cv2.putText(vis, f"MODO: {txt_modo}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_status, 2)
        cv2.putText(vis, f"PERF: {contador_perf} | FRAME: {frame_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(vis, f"EXP: {shutter_speed} | THRESH: {THRESH_VAL}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        # 3. Envia o Frame
        ret, buffer = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
        <body style='background:#111; color:#0f0; text-align:center; font-family:monospace;'>
            <h2>MINIOLA VIEWER v2.5</h2>
            <img src="/video_feed" style="height:85vh; border:1px solid #444;">
            <p>Use o Terminal (via SSH ou direto no Pi) para controle (S/P/R/A/E/T/G)</p>
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