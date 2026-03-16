from pathlib import Path

import cv2
import yaml

CONFIG_PATH = Path("configs/train_config.yaml")


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    config = load_config()
    extract_cfg = ((config.get("data_preparation") or {}).get("extract_frames") or {}).copy()

    input_video = Path(extract_cfg.get("input_video", "data/raw/input.mp4"))
    output_dir = Path(extract_cfg.get("output_dir", "data/interim/frames"))
    sample_fps = float(extract_cfg.get("sample_fps", 2.0))
    image_ext = str(extract_cfg.get("image_ext", "jpg"))

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step = max(int(round(source_fps / sample_fps)), 1) if sample_fps > 0 else 1

    frame_idx = 0
    saved_idx = 0
    stem = input_video.stem

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_step == 0:
            out_name = f"{stem}_frame_{saved_idx:06d}.{image_ext}"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()

    print(f"Source FPS: {source_fps:.2f}")
    print(f"Sampling FPS: {sample_fps:.2f}")
    print(f"Saved frames: {saved_idx}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
