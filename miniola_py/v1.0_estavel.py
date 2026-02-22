from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import threading
import sys
import time
import logging # Import necessário para silenciar o log

# 1. Configuração do Flask e Silenciamento de Logs
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR) # <--- ISSO VAI LIMPAR SEU TERMINAL

picam2 = Picamera2()

# ... (restante das configurações de hardware e variáveis permanecem iguais)

def painel_controle():
    """Thread para alterar parâmetros via terminal de forma limpa."""
    global contador, THRESH_VAL
    
    # Pequeno delay para o menu aparecer depois dos logs iniciais do sistema
    time.sleep(2) 
    
    print("\n" + "="*40)
    print("  PAINEL DE CONTROLE MINIOLA ATIVO")
    print("  (Digite o comando e aperte Enter)")
    print("="*40)

    while True:
        # O input() agora vai funcionar sem ser interrompido pelos logs
        entrada = input("\nComando (e, g, f, t, r) >> ").split()
        if not entrada: continue
        
        try:
            cmd = entrada[0].lower()
            if cmd == 'r':
                contador = 0
                print(">> [RESET] Contador zerado.")
            elif cmd == 'e' and len(entrada) > 1:
                val = int(entrada[1])
                picam2.set_controls({"ExposureTime": val})
                print(f">> [SET] Exposure: {val}")
            elif cmd == 'g' and len(entrada) > 1:
                val = float(entrada[1])
                picam2.set_controls({"AnalogueGain": val})
                print(f">> [SET] Gain: {val}")
            elif cmd == 't' and len(entrada) > 1:
                THRESH_VAL = int(entrada[1])
                print(f">> [SET] Threshold: {THRESH_VAL}")
            elif cmd == 'f' and len(entrada) > 1:
                val = int(entrada[1])
                picam2.set_controls({"FrameRate": val})
                print(f">> [SET] FrameRate: {val}")
        except Exception as e:
            print(f">> [ERRO]: {e}")

# --- RESTANTE DA LÓGICA (logica_scanner, generate_frames, routes) ---

def logica_scanner():
    global contador, furo_na_linha, ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug
    while True:
        frame_raw = picam2.capture_array()
        if frame_raw is None: continue
            
        gray = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2GRAY)
        roi = gray[ROI_Y:ROI_Y+ROI_H, :]
        
        _, binary = cv2.threshold(roi, THRESH_VAL, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        furo_agora = False
        temp_contornos = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if 100 < area < 5000:
                centro_x = x + (w // 2)
                temp_contornos.append({'rect': (x, y, w, h), 'color': (0, 255, 0)})
                if abs(centro_x - LINHA_X) < MARGEM:
                    furo_agora = True
                    if not furo_na_linha:
                        contador += 1
                        furo_na_linha = True
            else:
                temp_contornos.append({'rect': (x, y, w, h), 'color': (0, 0, 255)})

        if not furo_agora: furo_na_linha = False
        ultimo_frame_bruto = frame_raw
        ultimo_frame_binario = binary
        lista_contornos_debug = temp_contornos

def generate_frames():
    while True:
        if ultimo_frame_bruto is None:
            time.sleep(0.01)
            continue
        vis = ultimo_frame_bruto.copy()
        cv2.rectangle(vis, (0, ROI_Y), (640, ROI_Y + ROI_H), (255, 255, 255), 1)
        cv2.line(vis, (LINHA_X, ROI_Y), (LINHA_X, ROI_Y + ROI_H), (0, 255, 255), 2)
        for item in lista_contornos_debug:
            x, y, w, h = item['rect']
            cv2.rectangle(vis, (x, y + ROI_Y), (x + w, y + h + ROI_Y), item['color'], 2)
        
        bin_rgb = cv2.cvtColor(ultimo_frame_binario, cv2.COLOR_GRAY2RGB)
        canvas_bin = np.zeros_like(vis)
        canvas_bin[ROI_Y:ROI_Y+ROI_H, :] = bin_rgb
        output = np.hstack((vis, canvas_bin))
        ret, buffer = cv2.imencode('.jpg', output, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/count')
def get_count(): return str(contador)

@app.route('/')
def index():
    return """
    <html>
        <body style='background:#111; color:#0f0; text-align:center; font-family:monospace;'>
            <h2>MINIOLA LIVE TUNING</h2>
            <div id='val' style='font-size:80px;'>0</div>
            <img src="/video_feed" style="width:100%; max-width:1200px;">
            <script>
                setInterval(() => {
                    fetch('/count').then(r => r.text()).then(t => { document.getElementById('val').innerText = t; });
                }, 100);
            </script>
        </body>
    </html>
    """

if __name__ == '__main__':
    try:
        # Iniciamos o painel de controle em uma thread
        threading.Thread(target=painel_controle, daemon=True).start()
        threading.Thread(target=logica_scanner, daemon=True).start()
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt: pass
    finally: picam2.stop()