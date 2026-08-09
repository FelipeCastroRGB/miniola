import sys 
import platform
from unittest.mock import MagicMock

# --- CONFIGURAÇÃO DE AMBIENTE E HARDWARE ---
if platform.machine() not in ('aarch64', 'armv7l'):
    sys.modules["pykms"] = MagicMock()
    sys.modules["kms"] = MagicMock()

try:
    import miniola_cv  # type: ignore
    CV_ENGINE = "C++ [Pybind11]"
    scanner_cv = miniola_cv.ScannerVision()
except ImportError:
    CV_ENGINE = "Python [Nativo]"
    scanner_cv = None

from flask import Flask, Response, request, render_template, send_from_directory, jsonify  # type: ignore
import argparse
from cameras import get_camera_provider 
from core.motor_controller import FilmTransportPID
from core.joystick import GamepadController
import cv2 
import numpy as np 
import threading 
import multiprocessing as mp 
import time 
import logging 
import shutil
import os
import subprocess
import glob
import json
from datetime import datetime, timezone
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    PILImage = None  # type: ignore

app = Flask(__name__) # Flask para o Dashboard (Roda no Core 0)
log = logging.getLogger('werkzeug') # Desativa os logs de requisição do Flask para não poluir o console
log.setLevel(logging.ERROR) 

CAPTURE_PATH = "capturas"
if not os.path.exists(CAPTURE_PATH): os.makedirs(CAPTURE_PATH)

# Parser de Argumentos
parser = argparse.ArgumentParser(description="Miniola Scanner")
parser.add_argument('--camera', type=str, default='ximea', choices=['pi', 'ximea', 'mock'], help='Qual hardware de câmera usar (pi, ximea ou mock)')
args = parser.parse_args()

CAMERA_MODE = args.camera

# Controle de Motores (SKR Pico)
motor = FilmTransportPID()
motor.connect()

def toggle_rec():
    global GRAVANDO, fila_gravacao, ultimo_pitch_medio, PITCH_PADRAO_PX, AUDIO_CAPTURE_ENABLED, FPS_PROJECAO, fps_motor
    if not GRAVANDO:
        motor.start_pid(target_fps=fps_motor)
        sid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        try:
            p_val = ultimo_pitch_medio if ultimo_pitch_medio > 0 else PITCH_PADRAO_PX
            fila_gravacao.put({
                "type": "rec_start", "session_id": sid,
                "audio_enabled": AUDIO_CAPTURE_ENABLED, "fps_projecao": FPS_PROJECAO,
                "pitch_padrao": p_val
            }, block=True, timeout=2)
        except Exception as e: 
            print(f"[ERRO] Falha ao iniciar REC: {e}")
            return
        GRAVANDO = True
        print(f"\n[SISTEMA] REC ON | Sessão: {sid}\n>> ", end="", flush=True)
    else:
        GRAVANDO = False
        motor.stop_pid()
        motor.stop()
        try: fila_gravacao.put({"type": "rec_stop"}, block=True, timeout=2)
        except Exception as e: print(f"[WARN] REC OFF sem confirmação: {e}")
        print("\n[SISTEMA] REC OFF\n>> ", end="", flush=True)

gamepad = GamepadController(motor, on_rec_toggle=toggle_rec)
gamepad.start()

shutter_speed, gain, fps_cam = 1000, 1.0, 80
fps_motor = 18.0
foco_atual, passo_foco = 14.5, 0.5


# Resolução: Corte de Hardware para aliviar a porta USB do Raspberry Pi 4
# Largura 1420 acomoda a imagem (1388) e Altura 880 acomoda a ROI (840).
# Redução de 32% no tráfego de dados!
RES_W, RES_H = 1420, 880

# Offsets Exatos do CamTool do Usuário
CAM_OFFSET_X = 272
CAM_OFFSET_Y = 224

# --- SOFTWARE ISP (LUT) ---
WB_R, WB_G, WB_B = 1.0, 1.0, 1.0
GAMMA_Y, GAMMA_C = 1.0, 1.0
CONTRAST = 0.0

def build_color_lut(r, g, b, gy, gc, contrast):
    """Constrói a tabela de pré-computação (LUT) do ISP para o OpenCV aplicar instantaneamente"""
    lut = np.zeros((1, 256, 3), dtype=np.uint8)
    f_c = (259 * (contrast + 255)) / (255 * (259 - contrast)) if contrast != 0 else 1.0
    for i in range(256):
        val = i / 255.0
        # Gamma simples
        g_val = val ** (1.0 / gy) if gy > 0 else val
        # WB multipliers
        b_val, g_val, r_val = g_val * b, g_val * g, g_val * r
        # Re-scale e Contrast
        b_idx = f_c * (b_val * 255 - 128) + 128
        g_idx = f_c * (g_val * 255 - 128) + 128
        r_idx = f_c * (r_val * 255 - 128) + 128
        # Store in LUT (OpenCV usa BGR)
        lut[0, i, 0] = np.clip(b_idx, 0, 255)
        lut[0, i, 1] = np.clip(g_idx, 0, 255)
        lut[0, i, 2] = np.clip(r_idx, 0, 255)
    return lut

PIPELINE_LUT = build_color_lut(WB_R, WB_G, WB_B, GAMMA_Y, GAMMA_C, CONTRAST)

print(f"[SISTEMA] Inicializando provedor de câmera: {args.camera.upper()}")
camera = get_camera_provider(args.camera)
camera.start(RES_W, RES_H, fps_cam, shutter_speed, gain, foco_atual, CAM_OFFSET_X, CAM_OFFSET_Y)

# Padrão Bayer Padrão (Pode ser alterado dinamicamente via painel)
# Mudando para RG2BGR porque o crop no sensor altera o alinhamento da matriz Bayer, causando a imagem rosa!
BAYER_MODE = cv2.COLOR_BayerBG2BGR

# --- GEOMETRIA DO ROI E ESTADO ---
GRAVANDO = False
CALIBRANDO = False           # Trava de segurança da tela
PROCESSANDO_VIDEO = False    # Alerta o scanner para hibernar
FPS_PROJECAO = 24.0          # FPS de reprodução do filme (independente do fps_cam do sensor!)
ROI_X, ROI_Y = 200, 10
ROI_W, ROI_H = 80, 840
# --- LÓGICA DE GATILHO SIMPLIFICADA ---
LINHA_GATILHO_Y = 110  # Posição Y relativa DENTRO da ROI
MARGEM_GATILHO = 23    # Margem de disparo (px para cima e para baixo)
THRESH_VAL = 239 # Valor do threshold para binarização
PITCH_PADRAO_PX = 195.0  # CALIBRE AQUI: Quantos pixels tem o pitch de um filme NOVO na sua lente?
# --- PARÂMETROS DO CROP ---
OFFSET_X = 470 
OFFSET_Y_CROP = 0 # Deslocamento Y relativo à âncora (linha de gatilho)
CROP_W, CROP_H = 918, 612 

# --- EXTRAÇÃO DE ÁUDIO ÓTICO (CAPTURA AO VIVO) ---
AUDIO_X_OFFSET = 50      # Distância da borda direita da ROI perfuração até a pista de som
AUDIO_CAPTURE_ENABLED = True
AUDIO_CAPTURE_MODE = "variable_density"
AUDIO_READ_W = 96


contador_perfs_ciclo = 0
frame_count = 0
perfuracao_na_linha = False
ultimo_frame_bruto = None
ultimo_frame_binario = None
ultimo_crop_preview = np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
lista_contornos_debug = []
fps_real_proc = 0.0
tempo_ms_ciclo = 0.0
encolhimento_atual_pct = 0.0

# --- FILA DE MULTIPROCESSAMENTO ---
fila_gravacao = mp.Queue(maxsize=30) 
ultimo_pitch_medio = 0.0

def abrir_sessao_audio_optico(session_id: str, fps_projecao: float, pitch: float):
    raw_name = f"miniola_audio_{session_id}.f32"
    meta_name = f"miniola_audio_{session_id}.json"
    raw_path = os.path.join(CAPTURE_PATH, raw_name)
    meta_path = os.path.join(CAPTURE_PATH, meta_name)

    try:
        raw_fp = open(raw_path, "wb")
    except Exception as e:
        print(f"[AUDIO] Falha ao abrir sidecar RAW: {e}")
        return None

    fps_seguro = fps_projecao if fps_projecao > 0 else 24.0
    sessao = {
        "session_id": session_id,
        "started_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "raw_name": raw_name,
        "raw_path": raw_path,
        "meta_path": meta_path,
        "raw_fp": raw_fp,
        "mode": AUDIO_CAPTURE_MODE,
        "fps_projecao": float(fps_seguro),
        "pitch": pitch,
        "read_w": int(AUDIO_READ_W),
        "y_center": None,
        "x_left": None,
        "x_right": None,
        "frames_with_audio": 0,
        "total_samples": 0,
    }
    print(f"[AUDIO] Sessão ótica iniciada: {meta_name}")
    return sessao

def fechar_sessao_audio_optico(sessao, motivo: str):
    if not sessao: return

    raw_fp = sessao.get("raw_fp")
    if raw_fp is not None and not raw_fp.closed:
        raw_fp.flush()
        raw_fp.close()

    pitch_calculado = sessao.get("pitch", PITCH_PADRAO_PX)
    
    meta = {
        "version": 1,
        "session_id": sessao.get("session_id"),
        "started_at_utc": sessao.get("started_at_utc"),
        "closed_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "close_reason": motivo,
        "mode": sessao.get("mode"),
        "fps_projecao": sessao.get("fps_projecao"),
        "samples_per_frame": int(pitch_calculado * 4), # No 35mm, 1 frame de tempo vale 4 perfurações de espaço
        "source_sample_rate": int(round(sessao.get("fps_projecao", 24.0) * (pitch_calculado * 4))),
        "search_side": sessao.get("search_side"),
        "search_width": sessao.get("search_w"),
        "read_width": sessao.get("read_w"),
        "read_height": sessao.get("read_h"),
        "frames_with_audio": sessao.get("frames_with_audio"),
        "total_samples": sessao.get("total_samples"),
        "raw_path": sessao.get("raw_name"),
    }

    try:
        with open(sessao.get("meta_path"), "w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=2)
        print(f"[AUDIO] Sessão encerrada: {os.path.basename(sessao.get('meta_path'))} | {meta['frames_with_audio']} frames, {meta['total_samples']} samples")
    except Exception as e:
        print(f"[AUDIO] Falha ao salvar metadados da sessão: {e}")

def processo_escrita_disco(fila_in):
    print("[SISTEMA] Processo de gravação (Núcleo Isolado) iniciado.")
    sessao_audio = None
    arquivo_tracking = None
    while True:
        item = fila_in.get()
        if item is None:
            fechar_sessao_audio_optico(sessao_audio, "shutdown")
            if arquivo_tracking: arquivo_tracking.close()
            break

        msg_type = item.get("type", "frame") if isinstance(item, dict) else "frame"

        if msg_type == "set_bayer":
            global BAYER_MODE
            BAYER_MODE = item.get("mode")
            continue
        elif msg_type == "set_lut":
            global PIPELINE_LUT
            PIPELINE_LUT = item.get("lut")
            continue

        if msg_type == "audio_chunk":
            if sessao_audio is not None:
                chunk = item.get("data")
                if chunk is not None and chunk.size > 0:
                    chunk.tofile(sessao_audio["raw_fp"])
                    sessao_audio["total_samples"] += int(chunk.size)
            continue

        if msg_type == "rec_start":
            fechar_sessao_audio_optico(sessao_audio, "restart")
            sessao_audio = None
            ultimo_frame_processado = -1

            if item.get("audio_enabled", True):
                sid = item.get("session_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                p_pitch = float(item.get("pitch_padrao", PITCH_PADRAO_PX))
                sessao_audio = abrir_sessao_audio_optico(sid, float(item.get("fps_projecao", 24.0)), p_pitch)

            # Abre arquivo de telemetria para a nova sessão
            if arquivo_tracking: arquivo_tracking.close()
            sid = item.get("session_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            tracking_path = os.path.join(CAPTURE_PATH, f"miniola_tracking_{sid}.jsonl")
            arquivo_tracking = open(tracking_path, "w", encoding="utf-8")
            print(f"[TRACKING] Arquivo de telemetria criado: {os.path.basename(tracking_path)}")
            continue

        if msg_type == "rec_stop":
            fechar_sessao_audio_optico(sessao_audio, "manual_stop")
            sessao_audio = None
            if arquivo_tracking:
                arquivo_tracking.close()
                arquivo_tracking = None
            continue

        # picamera2 com "RGB888" entrega BGR na memória (comportamento libcamera).
        # O frame precisa ser convertido BGR→RGB antes de qualquer encoder que assuma RGB.
        if isinstance(item, dict):
            img_bgr = item.get("img_bgr")
            filename = item.get("filename")
        else:
            try: img_bgr, filename = item
            except Exception: continue

        if img_bgr is None or not filename: continue

        # Debayer Assíncrono: O loop principal da câmera manda o RAW8 cru e não gasta tempo.
        # É este processo isolado (que roda em outro núcleo do processador) que faz o trabalho pesado de debayer.
        if len(img_bgr.shape) == 2:
            img_bgr = cv2.cvtColor(img_bgr, BAYER_MODE)
            img_bgr = cv2.LUT(img_bgr, PIPELINE_LUT)

        # Salva como JPEG com cores corretas usando libjpeg-turbo C++ nativo (cv2.imwrite):
        # A velocidade de escrita cai de ~35ms para ~3ms por quadro, evitando que o buffer de memória do Python
        # sature a controladora USB 3.0 e cause queda de pacotes (dropframes) na câmera Ximea.
        cv2.imwrite(filename, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # Gravar as coordenadas matemáticas de registro deste fotograma
        if arquivo_tracking and "cy" in item:
            log_linha = json.dumps({
                "frame": item.get("frame_index"),
                "cx": item.get("cx"),
                "cy": item.get("cy"),
                "ox": item.get("ox"),
                "oy": item.get("oy"),
                "cw": item.get("cw"),
                "ch": item.get("ch"),
                "pitch_inst": item.get("pitch_inst", -1.0)
            })
            arquivo_tracking.write(log_linha + "\n")

def processar_captura(frame, cx_global, cy_global, n_frame, pitch_inst=-1.0):
    global OFFSET_X, OFFSET_Y_CROP, CROP_W, CROP_H, ultimo_crop_preview, GRAVANDO
    
    fx, fy = cx_global + OFFSET_X, cy_global + OFFSET_Y_CROP
    x1, y1 = max(0, int(fx - (CROP_W // 2))), max(0, int(fy - (CROP_H // 2)))
    # SPEC-002: Alinhamento da malha Bayer! Força o crop em coordenadas pares para evitar inversão (Zebra verde/roxa)
    x1 = (x1 // 2) * 2
    y1 = (y1 // 2) * 2
    x2, y2 = min(frame.shape[1], x1 + CROP_W), min(frame.shape[0], y1 + CROP_H)
    
    crop = frame[y1:y2, x1:x2]
    
    if crop.size > 0:
        ultimo_crop_preview = crop
        if GRAVANDO:
            filename = f"{CAPTURE_PATH}/miniola_{n_frame:06d}.jpg"
            try:
                fila_gravacao.put(
                    {
                        "type": "frame",
                        "img_bgr": frame.copy(),  # Envia o frame INTEIRO (overscan) para o disco
                        "filename": filename,
                        "frame_index": n_frame,
                        "cx": float(cx_global),
                        "cy": float(cy_global),
                        "ox": int(OFFSET_X),
                        "oy": int(OFFSET_Y_CROP),
                        "cw": int(CROP_W),
                        "ch": int(CROP_H),
                        "pitch_inst": float(pitch_inst)
                    },
                    block=False,
                )
            except Exception as e:
                print(f"[WARN] Fila de gravação cheia, frame {n_frame} descartado: {e}")

def disparar_processamento():
    global PROCESSANDO_VIDEO
    PROCESSANDO_VIDEO = True
    print("\n[Compilador FFmpeg iniciado e Scanner pausado")
    try:
        cmd = [sys.executable, "process.py", "--fps", str(FPS_PROJECAO), "--extract-audio"]
        if args.camera == "ximea":
            cmd.append("--disable-rs-comp")
            
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"[FFFmpeg abortou ou frames estão faltando!\nLOG DE ERRO:\n{proc.stderr}\n{proc.stdout}")
        else:
            print(f"[Finalizado! Disponível na Galeria Web.")
    except Exception as e:
        print(f"[ERRO FATAL no processamento: {e}")
    
    PROCESSANDO_VIDEO = False
    print("[SISTEMA] Scanner acordado de volta à vida.")

def painel_controle():
    global frame_count, GRAVANDO, LINHA_GATILHO_Y, MARGEM_GATILHO, ROI_X, CROP_H, CROP_W, ROI_Y, ROI_W, ROI_H, THRESH_VAL
    global foco_atual, passo_foco, shutter_speed, gain, fps_cam, OFFSET_X, contador_perfs_ciclo, CALIBRANDO
    global ultimo_pitch_medio, PITCH_PADRAO_PX, CV_ENGINE, FPS_PROJECAO, AUDIO_X_OFFSET, AUDIO_READ_W, fps_motor
    global BAYER_MODE, WB_R, WB_G, WB_B, GAMMA_Y, GAMMA_C, CONTRAST, PIPELINE_LUT
    
    def print_menu():
        print("\n" + "═"*60)
        print(f"   MINIOLA - PAINEL DE CONTROLE  |  MOTOR: {CV_ENGINE}")
        print("═"*60)
        print(" [SISTEMA]   rec (Gravar) | r (Zerar) | proc (Encodar MP4) | rout (Limpar Vídeos)")
        print(" [IMAGEM]    e [val] (Shutter) | g [val] (Gain) | fps [val] (FPS Cam)")
        print(" [COR]       wb [R] [G] [B] | gamma [Y] [C] | contrast [val] | sharp [val] | bayer [0-3]")
        print(" [FOCO]      k/l (Foco -/+) | af (Auto Foco) | j [val] (Passo Foco)")
        print(" [TRACKING]  ly (Linha) | mg (Margem) | t [val] (Limiar/Thresh)")
        print(" [GEOMETRIA] w/a/s/d (Move ROI) | rx/ry/rw/rh [val] (Modifica ROI)")
        print(" [CROP]      ch [val] (Alt) | cw [val] (Larg) | ox [val] (Offset X)")
        print(" [METROLOGIA]cal (Calibrar) | setcal [val] (Cal. Dinâmica)")
        print(" [MOTOR]     mf [vel] (Avanço) | mb [vel] (Reverso) | ms (Parar) | mfps [val] | motor (C++/Py)")
        print(" [ÁUDIO]     ax [val] (Offset X) | aw [val] (Largura) | pfps [val] (FPS Proj.)")
        print(" [OUTROS]    h (Menu) | off (Desligar)")
        print("═"*60)
        
    time.sleep(2)
    print_menu()
    
    while True:
        try:
            entrada = input("\n>> ").split()
            if not entrada: continue
            cmd = entrada[0].lower()
            
            val = 0
            if len(entrada) > 1:
                try: val = float(entrada[1])
                except ValueError: pass
            
            if cmd == 'h' or cmd == 'help':
                print_menu()
            elif cmd == 'motor':
                if scanner_cv is None:
                    print("[MOTOR] Módulo C++ não está compilado. Impossível alternar.")
                elif CV_ENGINE == "C++ [Pybind11]":
                    CV_ENGINE = "Python [Nativo]"
                    print(f"[MOTOR] Motor alternado para: {CV_ENGINE}")
                else:
                    CV_ENGINE = "C++ [Pybind11]"
                    scanner_cv.reset_ciclo()
                    print(f"[MOTOR] Motor alternado para: {CV_ENGINE}")
            elif cmd == 'bayer':
                if len(entrada) >= 2:
                    modo = int(entrada[1])
                    if modo == 0: BAYER_MODE = cv2.COLOR_BayerBG2BGR
                    elif modo == 1: BAYER_MODE = cv2.COLOR_BayerGB2BGR
                    elif modo == 2: BAYER_MODE = cv2.COLOR_BayerRG2BGR
                    elif modo == 3: BAYER_MODE = cv2.COLOR_BayerGR2BGR
                    fila_gravacao.put({"type": "set_bayer", "mode": BAYER_MODE})
                    print(f"[COR] Padrão Bayer alterado para o modo {modo}")
            elif cmd == 'wb':
                if len(entrada) >= 4:
                    WB_R, WB_G, WB_B = float(entrada[1]), float(entrada[2]), float(entrada[3])
                    PIPELINE_LUT = build_color_lut(WB_R, WB_G, WB_B, GAMMA_Y, GAMMA_C, CONTRAST)
                    fila_gravacao.put({"type": "set_lut", "lut": PIPELINE_LUT})
                    print(f"[ISP] White Balance atualizado para R:{WB_R} G:{WB_G} B:{WB_B}")
                else:
                    print("[ERRO] Uso: wb [R] [G] [B]. Exemplo: wb 1.5 1.0 1.5")
            elif cmd == 'gamma':
                if len(entrada) >= 3:
                    GAMMA_Y, GAMMA_C = float(entrada[1]), float(entrada[2])
                elif len(entrada) == 2:
                    GAMMA_Y = GAMMA_C = float(entrada[1])
                else:
                    print("[ERRO] Uso: gamma [Y] [C]. Exemplo: gamma 1.0 1.0")
                    continue
                PIPELINE_LUT = build_color_lut(WB_R, WB_G, WB_B, GAMMA_Y, GAMMA_C, CONTRAST)
                fila_gravacao.put({"type": "set_lut", "lut": PIPELINE_LUT})
                print(f"[ISP] Gamma atualizado para Y:{GAMMA_Y} C:{GAMMA_C}")
            elif cmd == 'contrast':
                CONTRAST = val
                PIPELINE_LUT = build_color_lut(WB_R, WB_G, WB_B, GAMMA_Y, GAMMA_C, CONTRAST)
                fila_gravacao.put({"type": "set_lut", "lut": PIPELINE_LUT})
                print(f"[ISP] Contraste atualizado para {CONTRAST}")
            elif cmd == 'sharp':
                camera.set_sharpness(val)
            elif cmd == 'w': ROI_Y = max(0, ROI_Y - 5)
            elif cmd == 's': ROI_Y = min(RES_H - ROI_H, ROI_Y + 5)
            elif cmd == 'a': ROI_X = max(0, ROI_X - 5)
            elif cmd == 'd': ROI_X = min(RES_W - ROI_W, ROI_X + 5)
            elif cmd == 'rx': ROI_X = int(val)
            elif cmd == 'ry': ROI_Y = int(val)
            elif cmd == 'rw': ROI_W = int(val)
            elif cmd == 'rh': ROI_H = int(val)
            elif cmd == 'ch': CROP_H = int(val)
            elif cmd == 'cw': CROP_W = int(val)
            elif cmd == 'ly': 
                LINHA_GATILHO_Y = int(val)
                print(f"[GATILHO] Linha ajustada para: {LINHA_GATILHO_Y}px dentro da ROI")
            elif cmd == 'mg':
                MARGEM_GATILHO = int(val)
                print(f"[GATILHO] Margem ajustada para: +-{MARGEM_GATILHO}px")
            elif cmd == 'ox': OFFSET_X = int(val)
            elif cmd == 'ax': AUDIO_X_OFFSET = int(val)
            elif cmd == 'aw': AUDIO_READ_W = int(val)
            elif cmd == 'l':
                foco_atual = round(foco_atual + passo_foco, 2)
                camera.set_focus(foco_atual)
            elif cmd == 'k':
                foco_atual = max(0.0, round(foco_atual - passo_foco, 2))
                camera.set_focus(foco_atual)
            elif cmd == 'af':
                print("[ÓPTICA] Destravando lente e ativando Modo Macro... (Aguarde)")
                try:
                    print("[ÓPTICA] Iniciando varredura profunda...")
                    if camera.autofocus_cycle():
                        metadados = camera.capture_metadata()
                        if "LensPosition" in metadados:
                            foco_atual = round(metadados["LensPosition"], 2)
                            print(f"[ÓPTICA] Sucesso! Foco Macro cravado em: {foco_atual}")
                            camera.set_focus(foco_atual)
                        else:
                            print("[ÓPTICA] Varredura concluída, mas posição não relatada pelo sensor.")
                    else:
                        print("[ÓPTICA] Autofoco não suportado pela câmera atual.")
                except Exception as e:
                    print(f"[ÓPTICA] Erro no Autofoco nativo: {e}")
            elif cmd == 'e': 
                shutter_speed = int(val); camera.set_exposure(shutter_speed)
            elif cmd == 'cal':
                CALIBRANDO = True
                print("[SISTEMA] MODO DE CALIBRAÇÃO ATIVADO!")
                print("Vá para o navegador, clique no Live View e arraste para desenhar a linha do Pitch.")
            elif cmd == 'off':
                print("[SISTEMA] Encerrando processos e desligando a Raspberry Pi de forma segura...")
                time.sleep(1)
                os.system("sudo poweroff")
            elif cmd == 'setcal':
                if ultimo_pitch_medio > 0:
                    encolhimento_referencia = float(val) if len(entrada) > 1 else 0.0
                    fator_escala = 1.0 - (encolhimento_referencia / 100.0)
                    PITCH_PADRAO_PX = ultimo_pitch_medio / fator_escala
                    print(f"\n[METROLOGIA] CALIBRAÇÃO DINÂMICA CONCLUÍDA!")
                    print(f"-> Filme Referência utilizado: {encolhimento_referencia}% de encolhimento.")
                    print(f"-> Novo Padrão (0%): {PITCH_PADRAO_PX:.2f}px")
                else: print("[ERRO] Deixe o filme de referência rodar e estabilizar no dashboard antes de calibrar.")
            elif cmd == 'g': gain = val; camera.set_gain(gain)
            elif cmd == 'fps': fps_cam = int(val); camera.set_fps(fps_cam)
            elif cmd == 'mfps': 
                fps_motor = float(val)
                print(f"[MOTOR] Velocidade Alvo de Captura definida para {fps_motor} fps.")
            elif cmd == 't': THRESH_VAL = int(val)
            elif cmd == 'mf': 
                spd = int(val) if val > 0 else 2000
                motor.manual_forward(spd)
            elif cmd == 'mb': 
                spd = int(val) if val > 0 else 2000
                motor.manual_reverse(spd)
            elif cmd == 'ms': motor.stop()
            elif cmd == 'rec':
                toggle_rec()
            elif cmd == 'proc': 
                if not PROCESSANDO_VIDEO: threading.Thread(target=disparar_processamento, daemon=True).start()
                else: print("[ERRO] FFmpeg já está encodando.")
            elif cmd == 'rc': 
                contador_perfs_ciclo = 0
                if CV_ENGINE == "C++ [Pybind11]" and scanner_cv is not None: scanner_cv.reset_ciclo()
                print("[SISTEMA] Fase realinhada! Ciclo forçado para 0/4.")
            elif cmd == 'rout': 
                print("[SISTEMA] Queimando os acetatos da prateleira (Limpeza de Filmes Renderizados)...")
                if os.path.exists('output'):
                    for f in os.listdir('output'): 
                        if f.endswith('.mp4'): os.remove(os.path.join('output', f))
                print("-> GALERIA DE MP4 LIMPA.")
            elif cmd == 'pfps':
                FPS_PROJECAO = float(val)
                print(f"FPS de Projeção definido para {FPS_PROJECAO} fps.")
            elif cmd == 'r': 
                frame_count = 0
                for f in os.listdir(CAPTURE_PATH): os.remove(os.path.join(CAPTURE_PATH, f))
                print("RAM DRIVE LIMPO.")
        except Exception as e: print(f"Erro: {e}")

def logica_scanner():
    cap_array = camera.get_frame
    cv_cvt = cv2.cvtColor
    cv_resize = cv2.resize
    cv_thresh = cv2.threshold
    cv_find = cv2.findContours
    get_time = time.perf_counter
    
    global frame_count, ultimo_frame_bruto, ultimo_frame_binario, lista_contornos_debug
    global contador_perfs_ciclo, perfuracao_na_linha, fps_real_proc, tempo_ms_ciclo
    global encolhimento_atual_pct, PITCH_PADRAO_PX, ultimo_pitch_medio, AUDIO_X_OFFSET

    ESCALA_CV = 0.5 
    skip_ui = 0
    buffer_pitches = []  
    buffer_tempos = []

    while True:
        if PROCESSANDO_VIDEO:
            time.sleep(1.0)
            continue
            
        t_inicio = get_time()
        frame_raw = cap_array()
        if frame_raw is None: continue
        
        lx, ly, lw, lh = ROI_X, ROI_Y, ROI_W, ROI_H
        
        if CV_ENGINE == "C++ [Pybind11]":
            slit_y = ROI_Y + (ROI_H // 2)
            audio_x = ROI_X + ROI_W + AUDIO_X_OFFSET

            ret = scanner_cv.process_frame(
                frame_raw, lx, ly, lw, lh,
                THRESH_VAL, LINHA_GATILHO_Y, MARGEM_GATILHO, PITCH_PADRAO_PX,
                (GRAVANDO and AUDIO_CAPTURE_ENABLED), audio_x, AUDIO_READ_W, slit_y
            )
            binary_small = ret["binary_small"]
            
            audio_chunk = ret.get("audio_chunk")
            if audio_chunk is not None and audio_chunk.size > 0:
                try: fila_gravacao.put({"type": "audio_chunk", "data": audio_chunk}, block=False)
                except Exception as e: print(f"[WARN] Fila cheia, chunk de áudio descartado: {e}")

            debug_visual = []
            if "debug_visual" in ret:
                for item in ret["debug_visual"]:
                    debug_visual.append({'rect': item['rect'], 'color': item['color']})
            
            perfuracao_na_linha = ret["perfuracao_na_linha"]
            contador_perfs_ciclo = ret["contador_perfs_ciclo"]
            encolhimento_atual_pct = ret["encolhimento_atual_pct"]
            
            if ret.get("ultimo_pitch_medio", 0) > 0:
                ultimo_pitch_medio = ret["ultimo_pitch_medio"]
            furo_detectado_agora = ret["achou_furo"]

            if ret["capturar"]:
                motor.sync_optical_phase()
                p_inst = ret.get("pitch_instantaneo", -1.0)
                processar_captura(frame_raw, ret["cx_a"], ret["cy_a"], frame_count, p_inst)
                frame_count += 1
        else:
            roi_color = frame_raw[ly:ly+lh, lx:lx+lw]
            roi_gray = cv_cvt(roi_color, cv2.COLOR_RGB2GRAY)
            roi_small = cv_resize(roi_gray, (0, 0), fx=ESCALA_CV, fy=ESCALA_CV) 
            _, binary_small = cv_thresh(roi_small, THRESH_VAL, 255, cv2.THRESH_BINARY) 
            
            debug_visual = []
            furo_detectado_agora = False
            contours, _ = cv_find(binary_small, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            limite_superior = LINHA_GATILHO_Y - MARGEM_GATILHO
            limite_inferior = LINHA_GATILHO_Y + MARGEM_GATILHO
            
            furos_validos = []
            for cnt in contours:
                x_s, y_s, w_s, h_s = cv2.boundingRect(cnt)
                area_aprox = (w_s * h_s) * 4 
                if 200 < area_aprox < 10000 and 0.2 < (w_s / h_s) < 2.5:
                    cy_roi = (y_s * 2) + ((h_s * 2) // 2)
                    cx_global = (x_s * 2) + (w_s * 2 // 2) + lx
                    cy_global = cy_roi + ly
                    acionou = limite_superior <= cy_roi <= limite_inferior
                    cor = (0, 0, 255) if acionou else (0, 255, 0)
                    furos_validos.append({'cy_roi': cy_roi, 'cx_g': cx_global, 'cy_g': cy_global, 'acionou': acionou})
                    debug_visual.append({'rect': (x_s*2+lx, y_s*2+ly, w_s*2, h_s*2), 'color': cor})

            furos_validos.sort(key=lambda p: p['cy_roi'])
            if furos_validos and furos_validos[0]['acionou']:
                furo_detectado_agora = True
                if not perfuracao_na_linha:
                    contador_perfs_ciclo += 1
                    perfuracao_na_linha = True
                    if contador_perfs_ciclo >= 4:
                        qtd = min(4, len(furos_validos))
                        pts = furos_validos[0:qtd]
                        cx_a = int(sum(p['cx_g'] for p in pts) / qtd)
                        if qtd > 1:
                            soma_pitch = 0
                            for i in range(1, qtd): soma_pitch += (pts[i]['cy_g'] - pts[i-1]['cy_g'])
                            pitch_instantaneo = soma_pitch / (qtd - 1)
                            if pitch_instantaneo > 0:
                                buffer_pitches.append(pitch_instantaneo)
                                if len(buffer_pitches) >= 10:
                                    pitch_medio = sum(buffer_pitches) / len(buffer_pitches)
                                    ultimo_pitch_medio = pitch_medio 
                                    calc_pct = (1.0 - (pitch_medio / PITCH_PADRAO_PX)) * 100.0
                                    encolhimento_atual_pct = max(-5.0, min(10.0, calc_pct))
                                    buffer_pitches.clear()    
                            soma_centros_y = 0
                            for i in range(qtd):
                                multiplicador = 1.5 - i 
                                soma_centros_y += (pts[i]['cy_g'] + (multiplicador * pitch_instantaneo))
                            cy_a = int(soma_centros_y / qtd)
                        else: cy_a = int(pts[0]['cy_g'] + 150) 
                        processar_captura(frame_raw, cx_a, cy_a, frame_count)
                        frame_count += 1
                        contador_perfs_ciclo = 0

        if not furo_detectado_agora: perfuracao_na_linha = False
        
        # --- DEAD-RECKONING (Interpolação Preditiva - SPEC-011) ---
        # Se o filme andou fisicamente mais que um Pitch inteiro e o OpenCV não capturou nada,
        # significa que a perfuração estava rasgada ou houve dropframe. Forçamos a captura!
        distancia_acumulada = motor.get_accumulated_distance()
        pitch_seguro = PITCH_PADRAO_PX if PITCH_PADRAO_PX > 0 else 19.0
        
        if GRAVANDO and (distancia_acumulada >= pitch_seguro):
            motor.sync_optical_phase() # Zera o acumulador para o próximo quadro
            #print(f"[ALERTA] Interpolação Forçada (Furo Perdido)! Dist: {distancia_acumulada:.1f}mm")
            
            cy_teorico = int(LINHA_GATILHO_Y + ly)
            cx_teorico = int(lx + (lw // 2))
            
            processar_captura(frame_raw, cx_teorico, cy_teorico, frame_count, ultimo_pitch_medio)
            frame_count += 1
            if CV_ENGINE == "C++ [Pybind11]":
                scanner_cv.reset_ciclo()
        # ----------------------------------------------------------

        skip_ui += 1
        if skip_ui >= 3:
            ultimo_frame_bruto = frame_raw 
            ultimo_frame_binario = binary_small
            lista_contornos_debug = debug_visual
            skip_ui = 0
        
        t_fim = get_time()
        inst_ms = (t_fim - t_inicio) * 1000.0
        buffer_tempos.append(inst_ms)
        if len(buffer_tempos) > 30: buffer_tempos.pop(0)
        tempo_ms_ciclo = sum(buffer_tempos) / len(buffer_tempos)
        fps_real_proc = 1000.0 / tempo_ms_ciclo if tempo_ms_ciclo > 0 else 0

if __name__ == '__main__':
    from core.state import state
    from web.app import create_app
    
    mp.Process(target=processo_escrita_disco, args=(fila_gravacao,), daemon=True).start()
    threading.Thread(target=logica_scanner, daemon=True).start()
    threading.Thread(target=painel_controle, daemon=True).start()
    
    app_web = create_app(state)
    app_web.run(host='0.0.0.0', port=5000, threaded=True)
