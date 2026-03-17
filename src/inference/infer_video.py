from pathlib import Path

import cv2
import yaml

try:
    from src.inference.utils import draw_detections, load_model, run_inference
except ModuleNotFoundError:
    from utils import draw_detections, load_model, run_inference

CONFIG_PATH = "configs/train_config.yaml"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def process_video(model, conf: float, input_path: Path, output_path: Path) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pred = run_inference(model, frame, conf=conf)[0]
        out_frame = draw_detections(frame, pred)
        writer.write(out_frame)
        frame_count += 1

    cap.release()
    writer.release()
    print(f"Saved: {output_path} | frames: {frame_count}")


def main() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    inference_config = config.get("inference") or {}
    video_config = inference_config.get("video") or {}

    input_dir = Path(video_config.get("input_dir", "examples/input"))
    output_dir = Path(video_config.get("output_dir", "examples/output"))
    output_suffix = str(video_config.get("output_suffix", "_pred"))
    output_ext = str(video_config.get("output_ext", ".mp4"))

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    video_paths = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS])
    if not video_paths:
        raise FileNotFoundError(f"No videos found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(inference_config["weights"])
    conf = float(inference_config.get("conf", 0.25))

    processed = 0
    for video_path in video_paths:
        out_name = f"{video_path.stem}{output_suffix}{output_ext}"
        out_path = output_dir / out_name
        process_video(model, conf, video_path, out_path)
        processed += 1

    print(f"Processed videos: {processed}")


if __name__ == "__main__":
    main()
