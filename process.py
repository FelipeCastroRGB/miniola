import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
import cv2
import numpy as np

try:
    import scipy.signal as sp_signal
    from scipy.interpolate import CubicSpline
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    sp_signal = None  # type: ignore
    CubicSpline = None  # type: ignore

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False
    nr = None  # type: ignore


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
AUDIO_SIDECAR_GLOB = "miniola_audio_*.json"

# No filme 35mm, a trilha ótica está fisicamente 21 fotogramas à frente
# da janela de projeção. Para sincronizar áudio e vídeo é necessário
# aparar esse offset do início do WAV antes de muxar.
FILM_35MM_AUDIO_ADVANCE_FRAMES = 21


def read_frame_as_grayscale(path: Path) -> np.ndarray | None:
    try:
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return frame
    except Exception:
        pass
    try:
        img = Image.open(path).convert("L")
        return np.array(img)
    except Exception:
        return None


def load_tracking_data(input_dir: Path) -> dict[int, dict]:
    """Lê o arquivo de telemetria mais recente (se existir) para estabilização."""
    tracking_files = list(input_dir.glob("miniola_tracking_*.jsonl"))
    if not tracking_files:
        return {}
    tracking_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    data = {}
    with open(tracking_files[0], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                row = json.loads(line)
                data[row["frame"]] = row
            except Exception:
                pass
    return data
    try:
        img = Image.open(path).convert("L")
        return np.array(img)
    except Exception:
        return None


def extract_audio_from_frames(
    frames: list[Path],
    roi: tuple[int, int, int, int],
    audio_mode: str,
    sample_rate: int,
    frame_rate: float,
) -> tuple[np.ndarray, dict]:
    roi_x, roi_y, roi_w, roi_h = roi
    samples_per_frame = int(sample_rate / frame_rate)
    total_samples = len(frames) * samples_per_frame
    audio_signal = np.zeros(total_samples, dtype=np.float32)
    processed_samples = 0

    for i, frame_path in enumerate(frames):
        gray = read_frame_as_grayscale(frame_path)
        if gray is None:
            processed_samples += samples_per_frame
            continue

        h, w = gray.shape
        rx = max(0, min(roi_x, w - 1))
        ry = max(0, min(roi_y, h - 1))
        rw = max(1, min(roi_w, w - rx))
        rh = max(1, min(roi_h, h - ry))

        strip = gray[ry : ry + rh, rx : rx + rw]

        if strip.size == 0:
            processed_samples += samples_per_frame
            continue

        if audio_mode == "variable_density":
            row = np.mean(strip, axis=0).astype(np.float32)
            row = (255 - row) / 255.0
        else:
            row = np.mean(strip, axis=1).astype(np.float32)
            row = (255 - row) / 255.0

        interpolated = np.interp(
            np.linspace(0, len(row) - 1, samples_per_frame),
            np.arange(len(row)),
            row,
        )
        audio_signal[processed_samples : processed_samples + samples_per_frame] = interpolated
        processed_samples += samples_per_frame
    # Remove o DC offset
    audio_signal = audio_signal - np.mean(audio_signal)
    audio_signal = np.clip(audio_signal, -1.0, 1.0)
    normalized = (audio_signal * 32767).astype(np.int16)

    stats = {
        "total_frames": len(frames),
        "processed_frames": processed_samples // samples_per_frame,
        "total_samples": total_samples,
        "sample_rate": sample_rate,
        "frame_rate": frame_rate,
        "audio_mode": audio_mode,
        "roi": {"x": roi_x, "y": roi_y, "w": roi_w, "h": roi_h},
    }
    return normalized, stats


def try_extract_audio_from_sidecar(input_dir: Path, sample_rate: int) -> tuple[np.ndarray, dict] | None:
    sidecar_meta_files = sorted(
        input_dir.glob(AUDIO_SIDECAR_GLOB),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not sidecar_meta_files:
        return None

    for meta_path in sidecar_meta_files:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        raw_ref = meta.get("raw_path")
        if raw_ref:
            raw_path = Path(raw_ref)
            if not raw_path.is_absolute():
                raw_path = (meta_path.parent / raw_path).resolve()
        else:
            raw_path = meta_path.with_suffix(".f32")

        if not raw_path.exists():
            continue

        try:
            signal = np.fromfile(raw_path, dtype=np.float32)
        except Exception:
            continue

        if signal.size == 0:
            continue

        source_sample_rate = float(meta.get("source_sample_rate") or 0.0)
        if source_sample_rate <= 0:
            fps_projecao = float(meta.get("fps_projecao") or 0.0)
            samples_per_frame = int(meta.get("samples_per_frame") or 0)
            if fps_projecao > 0 and samples_per_frame > 0:
                source_sample_rate = fps_projecao * samples_per_frame

        if source_sample_rate <= 0:
            source_sample_rate = float(sample_rate)

        if abs(source_sample_rate - sample_rate) > 1e-6:
            # === DIGITAL ANTI-ALIASING (THE VIRTUAL SLIT) ===
            # Esmaga os grãos fotográficos pontiagudos antes da viagem no tempo.
            # Um moving average atua reduzindo ruído branco em baixas frequências do source
            # que, de outra forma, colidiriam (aliasing) contra a voz humana em alta frequência na interpolação.
            if HAS_SCIPY:
                # Uma janela de Butterworth atua suavemente bloqueando agudos destrutivos
                # Simulando uma fenda de aprox 75μm que corta o grão microscópico da prata fotográfica
                nyq_raw = source_sample_rate / 2.0
                # Aumentado para 7000Hz para salvar os transientes e sibilâncias da voz
                cutoff_raw = min(nyq_raw * 0.8, 7000)
                if cutoff_raw > 0 and cutoff_raw < nyq_raw:
                    sos_aa = sp_signal.butter(4, cutoff_raw, 'lp', fs=source_sample_rate, output='sos')
                    signal = sp_signal.sosfiltfilt(sos_aa, signal)
            else:
                # Fallback: Um boxcar filter puramente espacial (Emulation of a wide physical lens Slit)
                lens_width = max(2, int(source_sample_rate / 6000))
                signal = np.convolve(signal, np.ones(lens_width)/lens_width, mode='same')

            # === CUBIC SPLINE DYNAMIC INTERPOLATION ===
            # Usa curvas Bézier contínuas em vez do serrilhado linear `np.interp` para esticar fita (M=6x).
            out_samples = max(1, int(round(signal.size * (sample_rate / source_sample_rate))))
            if HAS_SCIPY:
                x_old = np.linspace(0, signal.size - 1, signal.size)
                x_new = np.linspace(0, signal.size - 1, out_samples)
                cs = CubicSpline(x_old, signal)
                signal = cs(x_new).astype(np.float32)
            else:
                signal = np.interp(
                    np.linspace(0, signal.size - 1, out_samples),
                    np.arange(signal.size),
                    signal,
                ).astype(np.float32)

        if HAS_SCIPY:
            print(f"[AUDIO] Aplicando Masterização: High-Pass(40Hz), Low-Pass(7000Hz), Notch Filtros(90Hz, 180Hz)")

            # 1. High-Pass (Corta 'rumble' de sub-grave mecânico < 40Hz)
            sos_hp = sp_signal.butter(4, 40, 'hp', fs=sample_rate, output='sos')
            signal = sp_signal.sosfiltfilt(sos_hp, signal)

            # 2. Notch 90Hz (Remove buzz/robótico caso a lâmpada/obturador pulse em 90 FPS)
            b_notch, a_notch = sp_signal.iirnotch(90.0, 30.0, sample_rate)
            signal = sp_signal.filtfilt(b_notch, a_notch, signal)

            # 3. Notch 180Hz (Harmônico)
            b_notch2, a_notch2 = sp_signal.iirnotch(180.0, 30.0, sample_rate)
            signal = sp_signal.filtfilt(b_notch2, a_notch2, signal)

            # 4. Low-Pass (Corta estridência, arranhaões e poeira óptica > 7000Hz)
            sos_lp = sp_signal.butter(4, 7000, 'lp', fs=sample_rate, output='sos')
            signal = sp_signal.sosfiltfilt(sos_lp, signal)
        else:
            print("[WARN] Biblioteca 'scipy' não detectada! Masterização de cinema pulada. Para ter o áudio super limpo, instale: pip install scipy")
            signal = signal - np.mean(signal) # Fallback: DC offset apenas
            kernel_size = max(3, int(sample_rate / 8000))
            if kernel_size > 0: signal = np.convolve(signal, np.ones(kernel_size)/kernel_size, mode='same')

        if HAS_NOISEREDUCE:
            print("[AUDIO] Aplicando Spectral Gating Estacionário (prop_decrease=0.15)...")
            # stationary=True impede que a fase da voz seja destruída dinamicamente pelo algoritmo.
            # prop_decrease conservador para preservar consoantes acústicas reais do som.
            signal = nr.reduce_noise(y=signal, sr=sample_rate, prop_decrease=0.5, stationary=True)
        else:
            print("[WARN] Biblioteca 'noisereduce' não detectada! Para embutir redução de ruídos, instale: pip install noisereduce")
        
        # Maximização Transparente (Normalize 0.95%)
        peak = np.max(np.abs(signal))
        if peak > 0: signal = signal * (0.95 / peak)

        audio_signal = np.clip(signal, -1.0, 1.0)
        normalized = (audio_signal * 32767).astype(np.int16)
        stats = {
            "source": "live_sidecar",
            "meta_path": str(meta_path),
            "raw_path": str(raw_path),
            "total_samples": int(normalized.size),
            "sample_rate": sample_rate,
            "source_sample_rate": source_sample_rate,
            "audio_mode": meta.get("mode", "unknown"),
            "frames_with_audio": int(meta.get("frames_with_audio", 0)),
            "samples_per_frame": int(meta.get("samples_per_frame", 0)),
            "session_id": meta.get("session_id"),
        }
        return normalized, stats

    return None


def write_wav(path: Path, audio_data: np.ndarray, sample_rate: int) -> None:
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())


def natural_sort_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name.lower())
    return tuple(int(p) if p.isdigit() else p for p in parts)


def extract_last_number(path: Path) -> int | None:
    matches = re.findall(r"(\d+)", path.stem)
    if not matches:
        return None
    return int(matches[-1])


def list_frames(input_dir: Path) -> list[Path]:
    frames = [p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file()]
    return sorted(frames, key=natural_sort_key)


def detect_missing_indices(frames: Iterable[Path]) -> list[int]:
    """Retorna lista de índices numéricos absolutos ausentes na sequência (não paths de arquivo)."""
    numeric_indices = [extract_last_number(frame) for frame in frames]
    numeric_indices = [idx for idx in numeric_indices if idx is not None]
    if len(numeric_indices) < 2:
        return []
    missing: list[int] = []
    for previous, current in zip(numeric_indices, numeric_indices[1:]):
        if current - previous > 1:
            missing.extend(range(previous + 1, current))
    return missing


def build_concat_manifest(frames: list[Path], fps: float, manifest_path: Path) -> None:
    frame_duration = 1.0 / fps
    lines: list[str] = []
    
    if not frames: return

    # [AEO-Light] Metadata Headers (Futuro módulo de extração via área ótica do filme)
    lines.append("# [AEO_SYNC_INFO] Mode=Constant_LipSync")
    
    for i in range(len(frames) - 1):
        current_frame = frames[i]
        next_frame = frames[i+1]
        
        idx_curr = extract_last_number(current_frame)
        idx_next = extract_last_number(next_frame)
        
        duration = frame_duration
        if idx_curr is not None and idx_next is not None:
            diff = idx_next - idx_curr
            if diff > 1:
                # Compensação Cinematográfica: Gap detectado! 
                # Congele este frame multiplicando sua duração na tela para não dessincronizar o áudio ótico futuro.
                duration = frame_duration * diff
                
        lines.append(f"file {shlex.quote(str(current_frame.resolve()))}")
        lines.append(f"duration {duration:.10f}")

    # O concat demuxer recomenda repetir o último frame sem "duration" no encerramento.
    lines.append(f"file {shlex.quote(str(frames[-1].resolve()))}")
    lines.append(f"duration {frame_duration:.10f}")
    lines.append(f"file {shlex.quote(str(frames[-1].resolve()))}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_ffmpeg() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg não encontrado no PATH. Instale o ffmpeg antes de executar o process.py.")
    return ffmpeg_path


def build_ffmpeg_command(
    ffmpeg_path: str,
    manifest_path: Path,
    fps: float,
    output_path: Path,
    output_type: str,
) -> list[str]:
    base = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(manifest_path),
        "-r",
        f"{fps}",
    ]

    if output_type == "mp4":
        return base + [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]

    if output_type == "prores":
        return base + [
            "-c:v",
            "prores_ks",
            "-profile:v",
            "3",
            "-pix_fmt",
            "yuv422p10le",
            str(output_path),
        ]

    raise ValueError(f"Tipo de saída não suportado: {output_type}")


def build_ffmpeg_mux_command(
    ffmpeg_path: str,
    video_path: Path,
    audio_path: Path,
    fps: float,
    audio_advance_frames: int,
    output_path: Path,
) -> list[str]:
    """Gera comando ffmpeg para muxar vídeo + áudio com compensação do offset ótico 35mm.

    O som no 35mm está 'audio_advance_frames' fotogramas à frente da imagem
    correspondente. Aparamos esse offset do início do WAV com 'atrim' para que
    o primeiro sample audível coincida com o quadro 1 do vídeo.
    O vídeo é copiado sem reencoding (-c:v copy) para máxima velocidade e qualidade.
    """
    delay_seconds = audio_advance_frames / fps
    return [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-af", f"atrim=start={delay_seconds:.6f},asetpts=PTS-STARTPTS",
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "256k",
        "-movflags", "+faststart",
        str(output_path),
    ]


def probe_first_frame(path: Path) -> tuple[int, int]:
    try:
        # Abertura extremamente rápida sem decodificar a bagagem do JPEG no O(N) de tempo.
        with Image.open(path) as img:
            return img.size
    except Exception as e:
        raise RuntimeError(f"Não foi possível ler as dimensões do primeiro frame: {path}") from e


def render_stabilized_video_stream(
    ffmpeg_path: str,
    frames: list[Path],
    tracking_data: dict[int, dict],
    fps: float,
    disable_rs_comp: bool,
    outputs: list[tuple[Path, str]],
):
    """Lê frames, recorta e alinha perfeitamente usando sub-pixel warpAffine, e envia pro ffmpeg via pipe."""
    # Descobre tamanho final do crop com base no primeiro frame rastreado
    ref_track = None
    for f in frames:
        idx = int(f.stem.split('_')[-1])
        if idx in tracking_data:
            ref_track = tracking_data[idx]
            break
            
    if not ref_track:
        raise RuntimeError("Nenhum dado de tracking casou com os frames encontrados.")

    crop_w, crop_h = ref_track["cw"], ref_track["ch"]
    
    valid_pitches = []
    for f in frames:
        idx = int(f.stem.split('_')[-1])
        if idx in tracking_data:
            p = tracking_data[idx].get("pitch_inst", -1.0)
            if p > 0:
                valid_pitches.append(p)
                
    pitch_padrao = sum(valid_pitches) / len(valid_pitches) if valid_pitches else -1.0
    
    # Extrai array de cx para aplicar Filtro Gaussiano de passa-baixa
    # Isso elimina o jitter de alta frequência (tremor do limiar binário)
    # mas mantém o weave natural e suave do filme.
    raw_cx_array = []
    last_valid_cx = ref_track["cx"]
    for f in frames:
        idx = int(f.stem.split('_')[-1])
        if idx in tracking_data:
            last_valid_cx = tracking_data[idx]["cx"]
        raw_cx_array.append(last_valid_cx)
        
    try:
        import scipy.ndimage as ndimage
        smoothed_cx = ndimage.gaussian_filter1d(raw_cx_array, sigma=4.0)
        print("[ESTABILIZAÇÃO] Filtro passa-baixa aplicado no Eixo X (Suavização de Threshold).")
    except ImportError:
        smoothed_cx = raw_cx_array
        print("[WARN] Scipy não instalado. Filtro de suavização no Eixo X ignorado.")
    
    print(f"[ESTABILIZAÇÃO] Iniciando ancoragem na perfuração. Crop: {crop_w}x{crop_h}")
    if pitch_padrao > 0 and not disable_rs_comp:
        print(f"[ESTABILIZAÇÃO] Compensação de Rolling Shutter ativada (Pitch Padrão: {pitch_padrao:.2f}px)")
    elif disable_rs_comp:
        print(f"[ESTABILIZAÇÃO] Compensação de Rolling Shutter DESATIVADA (Modo Global Shutter).")

    # Monta comando FFmpeg recebendo RAW de stdin e gerando MÚLTIPLAS saídas simultâneas
    cmd = [
        ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{crop_w}x{crop_h}", "-r", str(fps),
        "-i", "-"
    ]
    
    for out_path, out_type in outputs:
        if out_type == "mp4":
            cmd.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", str(out_path)])
        elif out_type == "prores":
            cmd.extend(["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", str(out_path)])

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    
    try:
        for i, frame_path in enumerate(frames):
            if i % 100 == 0:
                print(f"[ESTABILIZAÇÃO] Processando frame {i+1}/{len(frames)}...")
                
            img = cv2.imread(str(frame_path))
            if img is None: continue
                
            f_idx = int(frame_path.stem.split('_')[-1])
            track = tracking_data.get(f_idx)
            
            scale_y = 1.0
            cx = smoothed_cx[i]
            
            if track:
                cy, ox = track["cy"], track["ox"]
                oy = track.get("oy", 0) # Fallback para vídeos gravados antes do Crop Dinâmico
                cw, ch = track.get("cw", crop_w), track.get("ch", crop_h)
                
                # Para sensores Rolling Shutter (ex: Raspberry Pi V3), usamos o stretch vertical.
                # Para sensores Global Shutter (ex: XIMEA), desativamos para evitar "vertical breathing".
                pitch_inst = track.get("pitch_inst", -1.0)
                if pitch_padrao > 0 and pitch_inst > 0 and not disable_rs_comp:
                    scale_y = pitch_padrao / pitch_inst
            else:
                # Fallback no centro se faltar tracking
                cy, ox, oy = img.shape[0] / 2, 0, 0
                cx = img.shape[1] / 2
                cw, ch = crop_w, crop_h
                
            center_x, center_y = cx + ox, cy + oy
            
            # Matriz Afim: Translação X, e (Escala Y + Translação Y)
            # Para que o center_y original caia exatamente no meio do crop_h após o redimensionamento.
            tx = cw / 2.0 - center_x
            ty = ch / 2.0 - (scale_y * center_y)
            
            # warpAffine aplica shift sub-pixel e correção de stretch do rolling shutter ao mesmo tempo!
            M = np.float32([[1.0, 0.0, tx], [0.0, scale_y, ty]])
            dst = cv2.warpAffine(img, M, (cw, ch), flags=cv2.INTER_LINEAR)
            
            proc.stdin.write(dst.tobytes())
    finally:
        if proc.stdin: proc.stdin.close()
        proc.wait()
        
    if proc.returncode != 0:
        raise RuntimeError("Erro na renderização estabilizada com FFmpeg.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera vídeo MP4/ProRes a partir dos frames de captura do Miniola.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Diretório com os frames capturados. Se omitido, tenta ./capturas e depois ./captura.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "output"),
        help="Diretório de saída dos arquivos processados.",
    )
    parser.add_argument("--name", default="miniola_scan", help="Nome base dos arquivos de saída.")
    parser.add_argument("--fps", type=float, default=24.0, help="Frames por segundo de saída.")
    parser.add_argument(
        "--format",
        choices=("mp4", "prores", "both"),
        default="mp4",
        help="Formato de saída desejado.",
    )
    parser.add_argument(
        "--verify-frames",
        action="store_true",
        help="Lê cada frame com OpenCV e descarta arquivos corrompidos.",
    )
    parser.add_argument(
        "--extract-audio",
        action="store_true",
        help="Extrai áudio da trilha ótica e salva WAV (prioriza sidecar ao vivo, fallback em ROI).",
    )
    parser.add_argument(
        "--disable-rs-comp",
        action="store_true",
        help="Desativa compensação de Rolling Shutter (Ideal para câmeras Global Shutter como XIMEA).",
    )
    parser.add_argument(
        "--audio-roi",
        default="0,0,0,0",
        help="ROI da trilha ótica no frame (x,y,w,h). Ex: 1200,100,150,600. Usa auto-detect se omitido.",
    )
    parser.add_argument(
        "--audio-mode",
        choices=("variable_density", "variable_area"),
        default="variable_density",
        help="Modo de trilha ótica: variable_density (DFFF) ou variable_area (VA).",
    )
    parser.add_argument(
        "--audio-sample-rate",
        type=int,
        default=48000,
        help="Taxa de amostragem do WAV gerado (padrão: 48000).",
    )
    parser.add_argument(
        "--audio-advance-frames",
        type=int,
        default=FILM_35MM_AUDIO_ADVANCE_FRAMES,
        help=(
            f"Fotogramas de avanço da trilha ótica em relação à imagem "
            f"(padrão: {FILM_35MM_AUDIO_ADVANCE_FRAMES} para 35mm)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).parent
    if args.input_dir:
        input_dir = Path(args.input_dir).expanduser().resolve()
    else:
        preferred = script_dir / "capturas"
        legacy = script_dir / "captura"
        input_dir = preferred if preferred.exists() else legacy
        input_dir = input_dir.expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.fps <= 0:
        print("[ERRO] O valor de --fps deve ser maior que zero.")
        return 1

    if not input_dir.exists():
        print(f"[ERRO] Diretório de entrada não existe: {input_dir}")
        return 1

    frames = list_frames(input_dir)
    if not frames:
        print(f"[ERRO] Nenhum frame encontrado em: {input_dir}")
        return 1

    if args.verify_frames:
        print("[INFO] Verificação minuciosa multi-thread ativada (isso examina o miolo dos JPEGs)...")
        valid_frames: list[Path] = []
        dropped = 0
        
        def is_valid_frame(p: Path) -> bool:
            return cv2.imread(str(p)) is not None

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(is_valid_frame, frames))
            
        for frame, is_valid in zip(frames, results):
            if is_valid:
                valid_frames.append(frame)
            else:
                dropped += 1
                
        frames = valid_frames
        print(f"[INFO] Verificação concluída: {len(frames)} frames válidos, {dropped} descartados.")
        if not frames:
            print("[ERRO] Todos os frames foram descartados na verificação.")
            return 1

    width, height = probe_first_frame(frames[0])

    ffmpeg = ensure_ffmpeg()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    audio_output_path: Path | None = None
    audio_stats: dict = {}
    if args.extract_audio:
        print("[INFO] Extraindo trilha ótica...")
        sidecar_result = try_extract_audio_from_sidecar(input_dir, args.audio_sample_rate)
        if sidecar_result is not None:
            audio_data, audio_stats = sidecar_result
            print(f"[INFO] Sidecar ótico detectado: {Path(audio_stats['meta_path']).name}")
        else:
            roi_parts = [int(x.strip()) for x in args.audio_roi.split(",")]
            if len(roi_parts) == 4 and roi_parts[2] > 0 and roi_parts[3] > 0:
                roi: tuple[int, int, int, int] = (roi_parts[0], roi_parts[1], roi_parts[2], roi_parts[3])
                print(f"[INFO] ROI configurada: {roi}")
            else:
                auto_x = max(0, width - 200)
                roi = (auto_x, 0, 180, height)
                print(f"[INFO] ROI auto-detectada (lateral direita): {roi}")
            audio_data, audio_stats = extract_audio_from_frames(
                frames, roi, args.audio_mode, args.audio_sample_rate, args.fps
            )
        wav_path = output_dir / f"{args.name}_{timestamp}.wav"
        write_wav(wav_path, audio_data, args.audio_sample_rate)
        audio_output_path = wav_path
        print(f"[INFO] WAV salvo: {wav_path.name} ({len(audio_data)} samples)")

    missing_indices = detect_missing_indices(frames)
    if missing_indices:
        print(f"[WARN] Detectados {len(missing_indices)} índices ausentes na sequência numérica.")

    manifest_path = output_dir / f".{args.name}_{timestamp}.frames.txt"
    build_concat_manifest(frames, args.fps, manifest_path)

    tracking_data = load_tracking_data(input_dir)
    outputs: list[Path] = []
    output_types = ("mp4", "prores") if args.format == "both" else (args.format,)
    extension_map = {"mp4": "mp4", "prores": "mov"}

    try:
        if tracking_data:
            print("[INFO] Telemetria (Registro Óptico) detectada! Usando ancoragem sub-pixel.")
            plan_outputs = []
            for output_type in output_types:
                output_path = output_dir / f"{args.name}_{timestamp}.{extension_map[output_type]}"
                plan_outputs.append((output_path, output_type))
                outputs.append(output_path)
        
            render_stabilized_video_stream(ffmpeg, frames, tracking_data, args.fps, args.disable_rs_comp, plan_outputs)
        else:
            print("[INFO] Sem telemetria detectada. Processando concatenação nativa rápida.")
            for output_type in output_types:
                output_path = output_dir / f"{args.name}_{timestamp}.{extension_map[output_type]}"
                cmd = build_ffmpeg_command(ffmpeg, manifest_path, args.fps, output_path, output_type)
                print(f"[INFO] Gerando arquivo {output_type.upper()}: {output_path.name}")
                subprocess.run(cmd, check=True)
                outputs.append(output_path)

    finally:
        if manifest_path.exists():
            manifest_path.unlink()

    # --- MUX VÍDEO + ÁUDIO COM OFFSET 35MM ---
    muxed_outputs: list[Path] = []
    if audio_output_path is not None and outputs:
        delay_s = args.audio_advance_frames / args.fps
        print(
            f"\n[INFO] Muxando vídeo + áudio com offset 35mm: "
            f"{args.audio_advance_frames} frames @ {args.fps} fps = {delay_s:.3f}s"
        )
        for video_path in outputs:
            # ProRes já é formato de arquivo (MOV), suporta AAC normalmente.
            # Se precisar de PCM em ProRes, trocar -c:a por pcm_s24le.
            muxed_ext = video_path.suffix
            muxed_path = output_dir / f"{video_path.stem}_com_audio{muxed_ext}"
            cmd_mux = build_ffmpeg_mux_command(
                ffmpeg, video_path, audio_output_path,
                args.fps, args.audio_advance_frames, muxed_path
            )
            subprocess.run(cmd_mux, check=True)
            muxed_outputs.append(muxed_path)
            print(f"[INFO] Muxado: {muxed_path.name}")

    report: dict = {
        "created_at_utc": timestamp,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "name": args.name,
        "fps": args.fps,
        "total_frames": len(frames),
        "frame_size": {"width": width, "height": height},
        "missing_indices_count": len(missing_indices),
        "missing_indices_preview": missing_indices[:50],
        "outputs": [str(path) for path in outputs],
        "muxed_outputs": [str(path) for path in muxed_outputs],
    }
    if audio_output_path:
        report["audio"] = {
            "wav_path": str(audio_output_path),
            "audio_advance_frames": args.audio_advance_frames,
            "audio_advance_seconds": round(args.audio_advance_frames / args.fps, 6),
            "stats": audio_stats,
        }
    report_path = output_dir / f"{args.name}_{timestamp}.report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n[SUCESSO] Processamento concluído.")
    print(f"[INFO] Relatório: {report_path}")
    for output_path in outputs:
        print(f"[INFO] Vídeo (mudo): {output_path}")
    for muxed_path in muxed_outputs:
        print(f"[INFO] Vídeo + Áudio sincronizado: {muxed_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
