from pathlib import Path
import shutil
import subprocess

import yaml

CONFIG_PATH = Path("configs/train_config.yaml")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    config = load_config()
    extract_cfg = ((config.get("data_preparation") or {}).get("extract_frames") or {}).copy()

    input_dir = Path(extract_cfg.get("input_dir", "data/raw"))
    output_dir = Path(extract_cfg.get("output_dir", "data/interim/frames"))
    sample_fps = float(extract_cfg.get("sample_fps", 2.0))
    image_ext = str(extract_cfg.get("image_ext", "jpg"))

    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be > 0, got: {sample_fps}")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg is not installed or not available in PATH")

    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS])
    if not video_paths:
        raise FileNotFoundError(f"No video files found in: {input_dir}")

    total_saved = 0
    print(f"Input directory: {input_dir}")
    print(f"Videos found: {len(video_paths)}")
    print(f"Sampling FPS: {sample_fps:.2f}")
    print(f"Output directory: {output_dir}")

    for video_path in video_paths:
        stem = video_path.stem
        output_pattern = output_dir / f"{stem}_frame_%06d.{image_ext}"

        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={sample_fps}",
            str(output_pattern),
        ]

        before = len(list(output_dir.glob(f"{stem}_frame_*.{image_ext}")))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        after = len(list(output_dir.glob(f"{stem}_frame_*.{image_ext}")))

        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed for {video_path}\n"
                f"stdout: {proc.stdout.strip()}\n"
                f"stderr: {proc.stderr.strip()}"
            )

        saved = max(after - before, 0)
        total_saved += saved
        print(f"- {video_path.name}: saved {saved} frames")

    print(f"Total saved frames: {total_saved}")


if __name__ == "__main__":
    main()
