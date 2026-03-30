import cv2
import numpy as np
import os
import glob
from tqdm import tqdm  # Biblioteca para a barra de progresso [cite: 2026-02-28]

# --- CONFIGURAÇÃO ---
INPUT_DIR = "/home/felipe/miniola_py/captura"
OUTPUT_VIDEO = "/home/felipe/miniola_py/valida_estabilidade.mp4"
ROI_X_PROCESS = 60  # Ajuste para a largura da sua perfuração [cite: 2026-02-28]

def processar_com_progresso():
    # Coleta e ordena os frames do RAM Drive [cite: 2026-02-28]
    frames = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))
    if not frames: 
        print("Erro: Nenhum frame encontrado na pasta de captura.")
        return

    # Lê o primeiro frame para definir dimensões [cite: 2026-02-28]
    primeiro = cv2.imread(frames[0])
    h, w = primeiro.shape[:2]
    # Split Screen: Original + 10px borda + Estabilizado [cite: 2026-02-28]
    size_split = ((w * 2) + 10, h)

    # Configura o gravador de vídeo (MP4 padrão para o Pi) [cite: 2026-02-28]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 24.0, size_split)

    print(f"\n[MINIOLA PROCESS] Iniciando Estabilização e QC...")
    
    # A mágica da barra de progresso acontece aqui [cite: 2026-02-28]
    for f_path in tqdm(frames, desc="Processando Frames", unit="fr"):
        img_raw = cv2.imread(f_path)
        if img_raw is None: continue

        # --- LÓGICA DE ESTABILIZAÇÃO (Vertical Shift) --- [cite: 2025-12-23]
        gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        roi = gray[:, :ROI_X_PROCESS]
        _, thresh = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_estabilizada = img_raw.copy()
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cy = int(M["m01"] / M["m00"])
                desloc_y = (h // 2) - cy
                matriz = np.float32([[1, 0, 0], [0, 1, desloc_y]])
                img_estabilizada = cv2.warpAffine(img_raw, matriz, (w, h))

        # --- CONSTRUÇÃO DO CANVAS DE COMPARAÇÃO --- [cite: 2026-02-28]
        canvas = np.zeros((h, (w * 2) + 10, 3), dtype=np.uint8)
        canvas[:, :w] = img_raw
        canvas[:, w+10:] = img_estabilizada
        
        # Legendas para o Quality Control (QC) [cite: 2025-12-23]
        cv2.putText(canvas, "RAW", (15, 35), 1, 1.5, (0,0,255), 2)
        cv2.putText(canvas, "ESTABILIZADO", (w + 25, 35), 1, 1.5, (0,255,0), 2)

        out.write(canvas)

    out.release()
    print(f"\n[SUCESSO] Vídeo de conferência gerado: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    processar_com_progresso()