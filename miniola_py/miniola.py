import cv2
import numpy as np
import time
from flask import Flask, Response, render_template_string
import threading
from picamera2 import Picamera2
from picamera2.outputs import NumpyOutput

# --- CONFIGURAÇÕES TÉCNICAS (AMIA/SMPTE Standards) ---
# Resolução ampliada para evitar cortes no fotograma
WIDTH, HEIGHT = 1280, 720 
# ROI ajustada para as perfurações (agora considerando a rotação visual)
# Nota: A lógica de detecção ainda corre no frame horizontal original
ROI_X, ROI_Y = 850, 50   # Movido para a direita conforme sua imagem
ROI_W, ROI_H = 150, 80   
LINHA_X = 920            # Linha de gatilho dentro da ROI
THRESH_VAL = 60          # Sensibilidade do preto da perfuração

# --- ESTADO GLOBAL ---
perf_count = 0
frame_count = 0
modo_gravacao = False
ultimo_frame_bruto = None

app = Flask(__name__)

# --- LÓGICA DO SCANNER (DETECÇÃO) ---
def logica_scanner():
    global perf_count, frame_count, ultimo_frame_bruto
    
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (WIDTH, HEIGHT)})
    picam2.configure(config)
    picam2.start()
    
    output = NumpyOutput()
    perfuracao_detectada = False

    while True:
        picam2.capture_array(output)
        frame = output.array
        ultimo_frame_bruto = frame.copy()
        
        # Processamento para detecção (Grayscale + ROI)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        roi = gray[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
        
        # Média de brilho na linha vertical de gatilho
        coluna_gatilho = roi[:, LINHA_X - ROI_X]
        media_brilho = np.mean(coluna_gatilho)

        # Lógica de Gatilho (Schmitt Trigger simples)
        if media_brilho < THRESH_VAL and not perfuracao_detectada:
            perfuracao_detectada = True
            perf_count += 1
            # A cada 4 perfurações (35mm padrão), contamos 1 frame
            if perf_count % 4 == 0:
                frame_count += 1
                if modo_gravacao:
                    cv2.imwrite(f"captura/frame_{frame_count:06d}.jpg", frame)
        
        elif media_brilho > THRESH_VAL + 20:
            perfuracao_detectada = False

# --- INTERFACE VISUAL (FLASK + OPENCV) ---
def generate_frames():
    while True:
        if ultimo_frame_bruto is None:
            time.sleep(0.1)
            continue
            
        # 1. Rotaciona a imagem para visionamento vertical (Personagens em pé)
        # Usamos ROTATE_90_CLOCKWISE ou COUNTERCLOCKWISE dependendo da sua câmera
        vis = cv2.rotate(ultimo_frame_bruto, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h_vis, w_vis = vis.shape[:2]

        # 2. Desenho de Overlays (Precisamos transpor as coordenadas da ROI para a imagem rotacionada)
        # Para facilitar o visionamento, desenhamos info fixas no topo
        status_cor = (0, 255, 0) if modo_gravacao else (0, 255, 255)
        txt_modo = "GRAVANDO" if modo_gravacao else "VISIONAMENTO"
        
        cv2.putText(vis, f"MODO: {txt_modo}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_cor, 2)
        cv2.putText(vis, f"PERF: {perf_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(vis, f"FRAMES: {frame_count}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        # 3. Desenho da Caixa de ROI (Transposta para a visão rotacionada)
        # Nota: Simplificamos o desenho para focar no fotograma
        cv2.rectangle(vis, (50, 50), (150, 150), (255, 0, 0), 2) # Guia visual simples

        _, buffer = cv2.imencode('.jpg', vis)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template_string("""
        <html>
            <body style="background: #111; color: white; text-align: center; font-family: sans-serif;">
                <h1>MINIOLA DIGITAL VIEWER</h1>
                <img src="{{ url_for('video_feed') }}" style="height: 85vh; border: 2px solid #444;">
                <p>Use o terminal para comandos de captura (S/P) e ajustes técnicos.</p>
            </body>
        </html>
    """)

# --- CONTROLE VIA TERMINAL ---
def terminal_control():
    global modo_gravacao, perf_count, frame_count
    print("\n--- CONTROLE MINIOLA ATIVO ---")
    print("S: Start Gravacao | P: Pause/Stop | R: Reset Contadores | Q: Sair")
    while True:
        cmd = input("Comando: ").lower()
        if cmd == 's':
            import os
            if not os.path.exists("captura"): os.makedirs("captura")
            modo_gravacao = True
            print(">> CAPTURA INICIADA")
        elif cmd == 'p':
            modo_gravacao = False
            print(">> CAPTURA PAUSADA")
        elif cmd == 'r':
            perf_count = 0
            frame_count = 0
            print(">> CONTADORES ZERADOS")

if __name__ == '__main__':
    t1 = threading.Thread(target=logica_scanner, daemon=True)
    t2 = threading.Thread(target=terminal_control, daemon=True)
    t1.start()
    t2.start()
    app.run(host='0.0.0.0', port=5000, threaded=True)