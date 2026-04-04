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


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
AUDIO_SIDECAR_GLOB = "miniola_audio_*.json"


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
) -> tuple[np.ndarray, dict]:
    roi_x, roi_y, roi_w, roi_h = roi
    samples_per_frame = int(sample_rate / frame_rate)
    total_samples = len(frames) * samples_per_frame
    audio_signal = np.zeros(total_samples, dtype=np.float32)
    processed = 0

    for i, frame_path in enumerate(frames):
        gray = read_frame_as_grayscale(frame_path)
        if gray is None:
            processed += samples_per_frame
            continue

        h, w = gray.shape
        rx = max(0, min(roi_x, w - 1))
        ry = max(0, min(roi_y, h - 1))
        rw = max(1, min(roi_w, w - rx))
        rh = max(1, min(roi_h, h - ry))

        strip = gray[ry : ry + rh, rx : rx + rw]

        if strip.size == 0:
            processed += samples_per_frame
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
        audio_signal[processed : processed + samples_per_frame] = interpolated
        processed += samples_per_frame
    # Remove o DC offset
    audio_signal = audio_signal - np.mean(audio_signal)
    audio_signal = np.clip(audio_signal, -1.0, 1.0)
    normalized = (audio_signal * 32767).astype(np.int16)

    stats = {
        "total_frames": len(frames),
        "processed_frames": processed // samples_per_frame,
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
            out_samples = max(1, int(round(signal.size * (sample_rate / source_sample_rate))))
            signal = np.interp(
                np.linspace(0, signal.size - 1, out_samples),
                np.arange(signal.size),
                signal,
            ).astype(np.float32)

        # Remove o DC offset inerente à extração ótica para centralizar a onda em zero
        signal = signal - np.mean(signal)
        
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
        help="Extrai áudio da trilha ótica e salva WAV (prioriza sidecar ao vivo, fallback em ROI).",
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
