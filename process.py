import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
import cv2


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


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
    missing_indices = detect_missing_indices(frames)
    if missing_indices:
        print(f"[WARN] Detectados {len(missing_indices)} índices ausentes na sequência numérica.")

    ffmpeg = ensure_ffmpeg()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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

    report = {
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
    report_path = output_dir / f"{args.name}_{timestamp}.report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n[SUCESSO] Processamento concluído.")
    print(f"[INFO] Relatório: {report_path}")
    for output_path in outputs:
        print(f"[INFO] Saída: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())