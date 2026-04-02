import argparse
import json
import math
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


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


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


def extract_audio_from_frames(
    frames: list[Path],
    roi: tuple[int, int, int, int],
    audio_mode: str,
    sample_rate: int,
    frame_rate: float,
    debug_dir: Path | None = None,
) -> tuple[np.ndarray, dict]:
    roi_x, roi_y, roi_w, roi_h = roi
    nominal_raw_rate = int(frame_rate * roi_h)
    total_raw = len(frames) * roi_h
    raw_signal = np.zeros(total_raw, dtype=np.float32)
    write_pos = 0

    debug_strips: list[np.ndarray] = []
    processed_frames = 0

    for frame_path in frames:
        gray = read_frame_as_grayscale(frame_path)
        if gray is None:
            continue

        h, w = gray.shape

        if roi_x >= w or roi_y >= h or roi_x + roi_w <= 0 or roi_y + roi_h <= 0:
            continue

        rx = max(0, roi_x)
        ry = max(0, roi_y)
        rw = max(1, min(roi_w, w - rx))
        rh = max(1, min(roi_h, h - ry))

        strip = gray[ry : ry + rh, rx : rx + rw]

        if strip.size == 0:
            continue

        if audio_mode == "variable_area":
            # O adaptive threshold amplifica granulação do filme.
            # Para VA, um Otsu global por frame tende a gerar menos chiado.
            blurred = cv2.GaussianBlur(strip, (5, 5), 0)
            _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            bright_mean = float(np.mean(strip[otsu == 255])) if np.any(otsu == 255) else 0.0
            dark_mean = float(np.mean(strip[otsu == 0])) if np.any(otsu == 0) else 0.0
            # Região útil da trilha costuma ser a faixa mais clara.
            binary = otsu if bright_mean >= dark_mean else cv2.bitwise_not(otsu)

            kernel = np.ones((3, 3), dtype=np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            row_widths = np.zeros(rh, dtype=np.float32)
            for y in range(rh):
                row = binary[y]
                dark_cols = np.where(row > 0)[0]
                if dark_cols.size > 0:
                    row_widths[y] = dark_cols[-1] - dark_cols[0] + 1
                else:
                    row_widths[y] = 0.0

            # Suaviza serrilhado de grão/perfuração na direção temporal.
            if rh >= 5:
                kernel_1d = np.ones(5, dtype=np.float32) / 5.0
                row_widths = np.convolve(row_widths, kernel_1d, mode="same")

            max_w = rw
            row_widths = np.clip(row_widths, 0, max_w)
            signal_row = (row_widths / max_w) * 2.0 - 1.0
            debug_strips.append(binary)
        else:
            # Em trilha de densidade variável, cada linha vertical do frame
            # representa um instante de áudio.
            row = np.mean(strip, axis=1).astype(np.float32) / 255.0
            signal_row = (1.0 - row) * 2.0 - 1.0

        raw_signal[write_pos : write_pos + rh] = signal_row
        write_pos += rh
        processed_frames += 1

    if processed_frames == 0:
        return np.zeros(sample_rate, dtype=np.int16), {}

    actual_len = write_pos
    raw_signal = raw_signal[:actual_len]

    if debug_dir and debug_strips:
        max_rows = min(len(debug_strips), 500)
        dbg_chunks = []
        for s in debug_strips[:max_rows]:
            dbg_chunks.append(s)
        dbg_strip = np.concatenate(dbg_chunks, axis=0)
        dbg_path = debug_dir / "debug_audio_strip.png"
        cv2.imwrite(str(dbg_path), dbg_strip)

    signal = raw_signal.astype(np.float32)
    signal -= np.mean(signal)

    # Bloqueio de DC estável: y[n] = x[n] - x[n-1] + R*y[n-1]
    # Reduz graves artificiais sem destruir formantes de voz.
    dc_r = 0.995
    if len(signal) > 1:
        dc_filtered = np.zeros_like(signal)
        prev_x = signal[0]
        prev_y = 0.0
        for i in range(1, len(signal)):
            x = signal[i]
            y = x - prev_x + dc_r * prev_y
            dc_filtered[i] = y
            prev_x = x
            prev_y = y
        signal = dc_filtered

    duration_s = processed_frames / frame_rate
    target_samples = max(1, int(round(duration_s * sample_rate)))
    final = np.interp(
        np.linspace(0, actual_len - 1, target_samples),
        np.arange(actual_len),
        signal,
    ).astype(np.float32)

    # Filtro passa-faixa simples para voz (remove rumble e ruído agudo).
    def one_pole_highpass(data: np.ndarray, cutoff_hz: float, fs: int) -> np.ndarray:
        if cutoff_hz <= 0 or fs <= 0:
            return data
        dt = 1.0 / fs
        rc = 1.0 / (2.0 * math.pi * cutoff_hz)
        alpha = rc / (rc + dt)
        out = np.zeros_like(data)
        prev_x = data[0]
        prev_y = 0.0
        for i in range(1, len(data)):
            x = data[i]
            y = alpha * (prev_y + x - prev_x)
            out[i] = y
            prev_x = x
            prev_y = y
        return out

    def one_pole_lowpass(data: np.ndarray, cutoff_hz: float, fs: int) -> np.ndarray:
        if cutoff_hz <= 0 or fs <= 0:
            return data
        dt = 1.0 / fs
        rc = 1.0 / (2.0 * math.pi * cutoff_hz)
        alpha = dt / (rc + dt)
        out = np.zeros_like(data)
        out[0] = data[0]
        for i in range(1, len(data)):
            out[i] = out[i - 1] + alpha * (data[i] - out[i - 1])
        return out

    final = one_pole_highpass(final, cutoff_hz=90.0, fs=sample_rate)
    final = one_pole_lowpass(final, cutoff_hz=5500.0, fs=sample_rate)

    peak = float(np.percentile(np.abs(final), 99.5))
    if peak > 1e-6:
        final = final / peak
    final = np.clip(final, -1.0, 1.0)

    final_int = (final * 32767).astype(np.int16)

    stats: dict = {
        "total_frames": len(frames),
        "processed_frames": processed_frames,
        "total_samples": len(final_int),
        "sample_rate": sample_rate,
        "frame_rate": frame_rate,
        "audio_mode": audio_mode,
        "roi": {"x": roi_x, "y": roi_y, "w": roi_w, "h": roi_h},
        "raw_rate_nominal": nominal_raw_rate,
        "raw_rate_effective": int(round(actual_len / duration_s)) if duration_s > 0 else nominal_raw_rate,
        "dc_removed": True,
        "highpass_applied": True,
        "lowpass_applied": True,
        "normalization": "peak_p99_5",
    }
    if debug_dir:
        stats["debug_strip_path"] = str(debug_dir / "debug_audio_strip.png")

    return final_int, stats


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


def probe_first_frame(path: Path) -> tuple[int, int]:
    try:
        # Abertura extremamente rápida sem decodificar a bagagem do JPEG no O(N) de tempo.
        with Image.open(path) as img:
            return img.size
    except Exception as e:
        raise RuntimeError(f"Não foi possível ler as dimensões do primeiro frame: {path}") from e


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
        help="Extrai áudio da trilha ótica dos frames e salva como WAV.",
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
        "--debug-audio",
        action="store_true",
        help="Salva debug_audio_strip.png (trilha binarizada) no output para calibrar a ROI.",
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
        roi: tuple[int, int, int, int]
        try:
            roi_parts = [int(x.strip()) for x in args.audio_roi.split(",")]
        except ValueError:
            roi_parts = []

        if (
            len(roi_parts) == 4
            and roi_parts[0] >= 0
            and roi_parts[1] >= 0
            and roi_parts[2] > 0
            and roi_parts[3] > 0
        ):
            roi = (roi_parts[0], roi_parts[1], roi_parts[2], roi_parts[3])
            print(f"[INFO] ROI configurada: {roi}")
        else:
            auto_x = max(0, width - 200)
            roi = (auto_x, 0, 180, height)
            print(f"[INFO] ROI auto-detectada (lateral direita): {roi}")

        roi_x, roi_y, roi_w, roi_h = roi
        if roi_x + roi_w > width or roi_y + roi_h > height:
            print(f"[ERRO] ROI {roi} fora dos limites do frame ({width}x{height}). Ajuste --audio-roi.")
            return 1
        if roi_w < 5:
            print(f"[ERRO] ROI largura ({roi_w}) muito pequena para extrair audio. Use w >= 5.")
            return 1

        debug_dir: Path | None = None
        if args.debug_audio:
            debug_dir = output_dir
            print("[INFO] Debug habilitado: debug_audio_strip.png sera gerado.")

        audio_data, audio_stats = extract_audio_from_frames(
            frames, roi, args.audio_mode, args.audio_sample_rate, args.fps, debug_dir
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

    outputs: list[Path] = []
    output_types = ("mp4", "prores") if args.format == "both" else (args.format,)
    extension_map = {"mp4": "mp4", "prores": "mov"}

    try:
        for output_type in output_types:
            output_path = output_dir / f"{args.name}_{timestamp}.{extension_map[output_type]}"
            cmd = build_ffmpeg_command(ffmpeg, manifest_path, args.fps, output_path, output_type)
            print(f"[INFO] Gerando arquivo {output_type.upper()}: {output_path.name}")
            subprocess.run(cmd, check=True)
            outputs.append(output_path)
    finally:
        if manifest_path.exists():
            manifest_path.unlink()

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
    }
    if audio_output_path:
        report["audio"] = {
            "wav_path": str(audio_output_path),
            "stats": audio_stats,
        }
    report_path = output_dir / f"{args.name}_{timestamp}.report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n[SUCESSO] Processamento concluído.")
    print(f"[INFO] Relatório: {report_path}")
    for output_path in outputs:
        print(f"[INFO] Saída: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
