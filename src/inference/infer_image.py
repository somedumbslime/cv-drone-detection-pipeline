from pathlib import Path

import cv2
import yaml

try:
    from src.inference.utils import draw_detections, load_model, run_inference
except ModuleNotFoundError:
    from utils import draw_detections, load_model, run_inference

CONFIG_PATH = "configs/train_config.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    inference_config = config.get("inference") or {}
    image_config = inference_config.get("image") or {}

    input_dir = Path(image_config.get("input_dir", "examples/input"))
    output_dir = Path(image_config.get("output_dir", "examples/output"))
    output_suffix = str(image_config.get("output_suffix", "_bbox"))

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(inference_config["weights"])
    conf = float(inference_config.get("conf", 0.25))

    processed = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        predictions = run_inference(model, image, conf=conf)
        rendered = draw_detections(image, predictions[0])

        out_name = f"{image_path.stem}{output_suffix}{image_path.suffix}"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), rendered)
        processed += 1
        print(f"Saved: {out_path}")

    print(f"Processed images: {processed}")


if __name__ == "__main__":
    main()
