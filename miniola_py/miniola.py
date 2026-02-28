import cv2
import numpy as np
import time
import os
import threading
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2

# --- CONFIGURAÇÕES TÉCNICAS INICIAIS ---
WIDTH, HEIGHT = 1280, 720 
ROI_X, ROI_Y = 850, 50   
ROI_W, ROI_H = 150, 80   
LINHA_X = 920            

# --- ESTADO GLOBAL ---
perf_count = 0
frame_count = 0
modo_gravacao = False
ultimo_frame_bruto = None
# Parâmetros de Imagem
shutter_speed = 20000 
gain = 1.0            
# Parâmetros de Detecção
thresh_val = 60          
modo_otsu = False        # Começa desativado para controle manual

app = Flask(__name__)
picam2 = Picamera2()

def logica_scanner():
    global perf_count, frame_count, ultimo_frame_bruto, shutter_speed, gain, thresh_val
    
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (WIDTH, HEIGHT)})
    picam2.configure(config)
    picam2.start()
    
    # Trava controles automáticos para consistência de arquivo
    picam2.set_controls({"ExposureTime": shutter_speed, "AnalogueGain": gain})
    
    perfuracao_detectada = False

    while True:
        frame = picam2.capture_array()
        ultimo_frame_bruto = frame.copy()
        
        # Processamento de Detecção
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        roi = gray[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
        
        # Lógica de Threshold Dinâmico (Otsu) ou Fixo
        if modo_otsu:
            # Otsu calcula o limiar ideal baseado no histograma da ROI
            ret, thresh_img = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            current_thresh = ret
        else:
            current_thresh = thresh_val
        
        # Coluna de gatilho para contagem
        coluna_gatilho = roi[:, LINHA_X - ROI_X]
        media_brilho = np.mean(coluna_gatilho)

        # Gatilho de Perfuração
        if media_brilho < current_thresh and not perfuracao_detectada:
            perfuracao_detectada = True
            perf_count += 1
            if perf_count % 4 == 0:
                frame_count += 1
                if modo_gravacao:
                    if not os.path.exists("captura"): os.makedirs("captura")
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"captura/frame_{frame_count:06d}.jpg", frame_bgr)
        
        elif media_brilho > current_thresh + 20:
            perfuracao_detectada = False

def generate_frames():
    while True:
        if ultimo_frame_bruto is None:
            time.sleep(0.1)
            continue
            
        vis_base = cv2.cvtColor(ultimo_frame_bruto, cv2.COLOR_RGB2BGR)
        vis = cv2.rotate(vis_base, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        status_cor = (0, 255, 0) if modo_gravacao else (0, 255, 255)
        txt_modo = "GRAVANDO" if modo_gravacao else "VISIONAMENTO"
        txt_otsu = "OTSU: ON" if modo_otsu else f"THRESH: {thresh_val}"
        
        # Overlays Técnicos para a Mesa de Enroladeira
        cv2.putText(vis, f"MODO: {txt_modo}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_cor, 2)
        cv2.putText(vis, f"PERF: {perf_count} | FR: {frame_count}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(vis, f"{txt_otsu} | EXP: {shutter_speed}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        _, buffer = cv2.imencode('.jpg', vis)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template_string("""
        <html><body style="background: #000; color: #ccc; text-align: center; font-family: monospace;">
            <h2 style="color: #0ff;">MINIOLA DIGITAL v2.4 - DEBUG MODE</h2>
            <img src="{{ url_for('video_feed') }}" style="height: 82vh; border: 1px solid #333;">
            <div style="margin-top: 5px;">Comandos: S/P (Grav), E/D (Exp), G/H (Gain), T/Y (Thresh), O (Otsu)</div>
        </body></html>
    """)

def terminal_control():
    global modo_gravacao, perf_count, frame_count, shutter_speed, gain, thresh_val, modo_otsu
    print("\n--- COMANDOS DE BANCADA ---")
    print("S/P: Start/Pause | R: Reset | O: Toggle OTSU")
    print("E/D: Exp +/- | G/H: Gain +/- | T/Y: Thresh +/-")
    
    while True:
        cmd = input(">> ").lower()
        if cmd == 's': modo_gravacao = True
        elif cmd == 'p': modo_gravacao = False
        elif cmd == 'r': perf_count = frame_count = 0
        elif cmd == 'o': 
            modo_otsu = not modo_otsu
            print(f">> MODO OTSU: {modo_otsu}")
        elif cmd == 'e': 
            shutter_speed = max(100, shutter_speed - 2000)
            picam2.set_controls({"ExposureTime": shutter_speed})
        elif cmd == 'd': 
            shutter_speed += 2000
            picam2.set_controls({"ExposureTime": shutter_speed})
        elif cmd == 'g':
            gain = max(1.0, gain - 0.2); picam2.set_controls({"AnalogueGain": gain})
        elif cmd == 'h':
            gain = min(12.0, gain + 0.2); picam2.set_controls({"AnalogueGain": gain})
        elif cmd == 't':
            thresh_val = max(10, thresh_val - 5)
        elif cmd == 'y':
            thresh_val = min(240, thresh_val + 5)
        elif cmd == 'q': os._exit(0)

if __name__ == '__main__':
    t1 = threading.Thread(target=logica_scanner, daemon=True)
    t2 = threading.Thread(target=terminal_control, daemon=True)
    t1.start(); t2.start()
    app.run(host='0.0.0.0', port=5000, threaded=True)