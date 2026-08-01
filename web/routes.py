import os
import glob
import shutil
import cv2
import time
import threading
import subprocess
import sys
import numpy as np
from flask import Blueprint, request, jsonify, render_template, Response, send_from_directory
from core.state import state

bp = Blueprint('main', __name__)

# --- STREAMS ---
def generate_dashboard():
    while True:
        time.sleep(0.06)
        try:
            if state.ultimo_frame_bruto is None:
                p_vazio = np.zeros((420, 640, 3), dtype=np.uint8)
                cv2.putText(p_vazio, "SEM SINAL DA CAMERA / CONECTANDO...", (130, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                cv2.putText(p_vazio, f"Modo atual: {state.CAMERA_MODE.upper()}", (230, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
                _, buffer = cv2.imencode('.jpg', p_vazio, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                continue
            
            ratio_w = 640 / state.RES_W
            ratio_h = 420 / state.RES_H
            scale = min(ratio_w, ratio_h)
            new_w = int(state.RES_W * scale)
            new_h = int(state.RES_H * scale)
            
            if len(state.ultimo_frame_bruto.shape) == 2:
                p_live_color = cv2.cvtColor(state.ultimo_frame_bruto, state.BAYER_MODE)
                p_live_resized = cv2.resize(p_live_color, (new_w, new_h))
            else:
                p_live_resized = cv2.resize(state.ultimo_frame_bruto.copy(), (new_w, new_h))
        except Exception as e:
            print(f"[ERRO DASHBOARD] Falha ao renderizar p_live: {e}")
            time.sleep(0.1)
            continue
            
        sx, sy = scale, scale
        off_x = (640 - new_w) // 2
        off_y = (420 - new_h) // 2
        
        def px(val): return off_x + int(val * sx)
        def py(val): return off_y + int(val * sy)
        
        p_live = np.zeros((420, 640, 3), dtype=np.uint8)
        p_live[off_y:off_y+new_h, off_x:off_x+new_w] = p_live_resized
        
        cv2.rectangle(p_live, (px(state.ROI_X), py(state.ROI_Y)), (px(state.ROI_X+state.ROI_W), py(state.ROI_Y+state.ROI_H)), (150, 150, 150), 1)
        
        a_x = state.ROI_X + state.ROI_W + state.AUDIO_X_OFFSET
        cv2.rectangle(p_live, (px(a_x), py(state.ROI_Y)), (px(a_x + state.AUDIO_READ_W), py(state.ROI_Y+state.ROI_H)), (0, 255, 255), 1)
        cor_gatilho = (0, 0, 255) if state.perfuracao_na_linha else (0, 255, 0)
        
        y_gl = state.ROI_Y + state.LINHA_GATILHO_Y
        cv2.line(p_live, (px(state.ROI_X), py(y_gl)), (px(state.ROI_X+state.ROI_W), py(y_gl)), cor_gatilho, 3)
        cv2.line(p_live, (px(state.ROI_X), py(y_gl - state.MARGEM_GATILHO)), (px(state.ROI_X+state.ROI_W), py(y_gl - state.MARGEM_GATILHO)), (50, 50, 50), 1)
        cv2.line(p_live, (px(state.ROI_X), py(y_gl + state.MARGEM_GATILHO)), (px(state.ROI_X+state.ROI_W), py(y_gl + state.MARGEM_GATILHO)), (50, 50, 50), 1)

        for item in state.lista_contornos_debug:
            x, y, w, h = item['rect']
            cv2.rectangle(p_live, (px(x), py(y)), (px(x+w), py(y+h)), item['color'], 2)
        
        p_bin = np.zeros((420, 640, 3), dtype=np.uint8)
        cv2.putText(p_bin, "PERFURACOES", (10, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)
        if state.ultimo_frame_binario is not None:
            bin_res = cv2.resize(cv2.cvtColor(state.ultimo_frame_binario, cv2.COLOR_GRAY2RGB), (270, 400))
            p_bin[20:420, 10:280] = bin_res

        cv2.line(p_bin, (310, 0), (310, 420), (40, 40, 40), 1)
        ax_raw = state.ROI_X + state.ROI_W + state.AUDIO_X_OFFSET
        aw_raw = max(1, state.AUDIO_READ_W)
        ay_raw = max(0, state.ROI_Y)
        ah_raw = max(1, min(state.RES_H - ay_raw, state.ROI_H))
        safe_ax = max(0, ax_raw)
        safe_aw = min(aw_raw, state.RES_W - safe_ax)

        if state.ultimo_frame_bruto is not None and safe_aw > 0 and ah_raw > 0:
            audio_strip = state.ultimo_frame_bruto[ay_raw : ay_raw + ah_raw, safe_ax : safe_ax + safe_aw]
            if audio_strip.size > 0:
                audio_gray = audio_strip if len(audio_strip.shape) == 2 else cv2.cvtColor(audio_strip, cv2.COLOR_RGB2GRAY)
                audio_preview = cv2.resize(cv2.cvtColor(audio_gray, cv2.COLOR_GRAY2RGB), (140, 400))
                p_bin[20:420, 330:470] = audio_preview
                cv2.putText(p_bin, "PISTA AUDIO [Escala de Cinza]", (330, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 220, 80), 1)
        else:
            cv2.putText(p_bin, "PISTA AUDIO", (330, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)
            cv2.putText(p_bin, "(sem frame)", (330, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1)

        p_inf = np.zeros((300, 1280, 3), dtype=np.uint8)
        if state.ultimo_crop_preview is not None and state.ultimo_crop_preview.size > 0:
            h_raw, w_raw = state.ultimo_crop_preview.shape[:2]
            aspect_ratio = w_raw / float(h_raw) if h_raw > 0 else 1.0
            crop_w_view = max(10, int(280 * aspect_ratio))
            
            if len(state.ultimo_crop_preview.shape) == 2:
                crop_color = cv2.cvtColor(state.ultimo_crop_preview, state.BAYER_MODE)
                crop_preview_color = cv2.resize(crop_color, (crop_w_view, 280))
                luma = cv2.resize(state.ultimo_crop_preview, (crop_w_view, 280))
            else:
                crop_preview_color = cv2.resize(state.ultimo_crop_preview.copy(), (crop_w_view, 280))
                luma = cv2.cvtColor(crop_preview_color, cv2.COLOR_RGB2GRAY)
            
            zebra_overlay = crop_preview_color.copy()
            zebra_overlay[luma > 245] = [0, 0, 255] 
            zebra_overlay[luma < 10]  = [255, 0, 0] 
            
            pos_y_zebra, pos_x_zebra = 10, 50
            p_inf[pos_y_zebra : pos_y_zebra+280, pos_x_zebra : pos_x_zebra+crop_w_view] = zebra_overlay
            cv2.rectangle(p_inf, (pos_x_zebra, pos_y_zebra), (pos_x_zebra + crop_w_view + 40, pos_y_zebra + 25), (0, 0, 0), -1)
            cv2.putText(p_inf, "ZEBRA", (pos_x_zebra + 5, pos_y_zebra + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            hist = cv2.calcHist([luma], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, 270, cv2.NORM_MINMAX)
            
            HIST_W, HIST_H = 512, 280
            grafico_h = np.zeros((HIST_H, HIST_W, 3), dtype=np.uint8)
            cv2.rectangle(grafico_h, (0, 0), (HIST_W, HIST_H), (20, 20, 20), -1)
            
            for i in range(256):
                x0 = i * 2; x1 = x0 + 2
                valor_y = int(hist.ravel()[i])
                cor = (255, 255, 255) if i > 200 else (80, 200, 80)
                cv2.rectangle(grafico_h, (x0, HIST_H), (x1, HIST_H - valor_y), cor, -1)
            
            cv2.line(grafico_h, (20, 0), (20, HIST_H), (0, 80, 255), 1)
            cv2.line(grafico_h, (490, 0), (490, HIST_H), (0, 80, 255), 1)
            cv2.putText(grafico_h, "0", (5, HIST_H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 255), 1)
            cv2.putText(grafico_h, "255", (476, HIST_H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 255), 1)
            
            pos_x_hist, pos_y_hist = 500, 10
            p_inf[pos_y_hist : pos_y_hist + HIST_H, pos_x_hist : pos_x_hist + HIST_W] = grafico_h
            cv2.putText(p_inf, "HISTOGRAMA (LUMINANCIA)", (pos_x_hist, pos_y_hist - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        dashboard = np.vstack((np.hstack((p_live, p_bin)), p_inf))
        _, buffer = cv2.imencode('.jpg', dashboard, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_preview():
    while True:
        files = sorted([f for f in os.listdir("capturas") if f.endswith('.jpg')])
        last_frames = files[-120:] if len(files) > 0 else []
        if not last_frames: 
            time.sleep(0.5)
            continue
        for frame_file in last_frames:
            img = cv2.imread(os.path.join("capturas", frame_file))
            if img is None: continue
            _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1/24)

# --- ROUTES ---
@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/video_feed')
def video_feed():
    return Response(generate_dashboard(), mimetype='multipart/x-mixed-replace; boundary=frame')

@bp.route('/preview_feed')
def preview_feed():
    return Response(generate_preview(), mimetype='multipart/x-mixed-replace; boundary=frame')

@bp.route('/set_crop')
def set_crop():
    try:
        x_web = float(request.args.get('x', 0))
        y_web = float(request.args.get('y', 0))
        w_web = float(request.args.get('w', 0))
        h_web = float(request.args.get('h', 0))
        
        cx_web = x_web + (w_web / 2)
        cy_web = y_web + (h_web / 2)
        cx_furo_web = state.ROI_X + (state.ROI_W / 2)
        cy_furo_web = state.ROI_Y + state.LINHA_GATILHO_Y
        
        state.OFFSET_X = int(cx_web - cx_furo_web)
        state.OFFSET_Y_CROP = int(cy_web - cy_furo_web)
        state.CROP_W = int(w_web)
        state.CROP_H = int(h_web)
        
        if state.CROP_W % 2 != 0: state.CROP_W += 1
        if state.CROP_H % 2 != 0: state.CROP_H += 1
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)})

@bp.route('/status')
def get_status():
    cpu_percent, ram_percent, cpu_temp = 0.0, 0.0, 0.0
    try:
        with open('/proc/stat', 'r') as f: fields = [float(column) for column in f.readline().strip().split()[1:]]
        idle, total = fields[3], sum(fields)
        cpu_percent = 100.0 * (1.0 - idle / total)
        with open('/proc/meminfo', 'r') as f: lines = f.readlines()
        mem = {line.split(':')[0]: int(line.split(':')[1].split()[0]) for line in lines[:32]}
        ram_percent = 100.0 * (1.0 - mem['MemAvailable'] / mem['MemTotal'])
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f: cpu_temp = float(f.read()) / 1000.0
    except: pass 
    
    total_arquivos = sum(1 for _ in os.scandir("capturas"))
    uso_disco = shutil.disk_usage("capturas")
    espaco_livre_mb = uso_disco.free / (1024 * 1024)
    
    return {
        "processando": state.PROCESSANDO_VIDEO, "cpu": f"{cpu_percent:.1f}%", "ram": f"{ram_percent:.1f}%", "temp": f"{cpu_temp:.1f}°C",
        "rec": "GRAVANDO" if state.GRAVANDO else "PARADO", "cor": "#ff0000" if state.GRAVANDO else "#00ff00",
        "ciclo": f"{state.contador_perfs_ciclo}/4", "total": state.frame_count, "fps_proc": f"{state.fps_real_proc:.1f} FPS", "ms_ciclo": f"{state.tempo_ms_ciclo:.1f} ms",
        "queue": 0, "arquivos": total_arquivos, "espaco": f"{espaco_livre_mb:.0f}MB", "foco": f"{state.foco_atual:.2f}",
        "exp": state.shutter_speed, "gain": f"{state.gain:.1f}", "fps_cam": state.fps_cam, "shrink": f"{state.encolhimento_atual_pct:.2f}%",
        "calibrando": state.CALIBRANDO, "thresh": state.THRESH_VAL,
        "roi_x": state.ROI_X, "roi_y": state.ROI_Y, "roi_w": state.ROI_W, "roi_h": state.ROI_H, "crop_w": state.CROP_W, "crop_h": state.CROP_H, "ox": state.OFFSET_X,
        "oy": state.OFFSET_Y_CROP, "gatilho_y": state.LINHA_GATILHO_Y, "margem": state.MARGEM_GATILHO, "res_w": state.RES_W, "res_h": state.RES_H, "fps_projecao": state.FPS_PROJECAO,
        "motor_cor": "PIL/RGB" if state.HAS_PIL else "cv2/BGR-fallback",
    }

@bp.route('/calibrar')
def calibrar():
    try:
        px = float(request.args.get('px'))
        mm = float(request.args.get('mm'))
        pixels_por_mm = px / mm
        state.PITCH_PADRAO_PX = pixels_por_mm * 4.74  
        state.CALIBRANDO = False  
        return "OK"
    except Exception as e:
        state.CALIBRANDO = False
        return f"Erro: {e}"

def disparar_processamento():
    state.PROCESSANDO_VIDEO = True
    print("\n[Compilador FFmpeg iniciado e Scanner pausado]")
    try:
        cmd = [sys.executable, "process.py", "--fps", str(state.FPS_PROJECAO), "--extract-audio"]
        if state.CAMERA_MODE == "ximea":
            cmd.append("--disable-rs-comp")
            
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"[FFmpeg abortou ou frames estão faltando!]\nLOG DE ERRO:\n{proc.stderr}\n{proc.stdout}")
        else:
            print(f"[Finalizado! Disponível na Galeria Web.]")
    except Exception as e:
        print(f"[ERRO FATAL no processamento: {e}]")
    
    state.PROCESSANDO_VIDEO = False
    print("[SISTEMA] Scanner acordado de volta à vida.")

@bp.route('/api/process', methods=['POST'])
def api_process():
    if not state.PROCESSANDO_VIDEO:
        threading.Thread(target=disparar_processamento, daemon=True).start()
        return jsonify({"status": "started"})
    return jsonify({"status": "already_running"}), 400

@bp.route('/api/videos', methods=['GET'])
def api_videos():
    if not os.path.exists('output'): return jsonify([])
    arquivos = glob.glob('output/*.mp4')
    arquivos.sort(key=os.path.getctime, reverse=True)
    return jsonify([os.path.basename(f) for f in arquivos])

@bp.route('/output/<path:filename>')
def serve_video(filename): return send_from_directory('../output', filename)
