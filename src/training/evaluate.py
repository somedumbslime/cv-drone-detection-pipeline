import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
from ultralytics import YOLO

CONFIG_PATH = Path("configs/train_config.yaml")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_split_dir(dataset_yaml_path: Path, split_name: str) -> Path:
    if not dataset_yaml_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {dataset_yaml_path}")

    with dataset_yaml_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f) or {}

    root = Path(data_cfg.get("path", dataset_yaml_path.parent))
    split_value = data_cfg.get(split_name)
    if split_value is None:
        raise ValueError(f"Split '{split_name}' is missing in {dataset_yaml_path}")

    split_path = Path(split_value)
    return split_path if split_path.is_absolute() else root / split_path


def count_images(folder: Path) -> int:
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def extract_metrics(results) -> dict:
    metrics: dict[str, float] = {}

    for key, value in (getattr(results, "results_dict", {}) or {}).items():
        if isinstance(value, (int, float)):
            metrics[key] = float(value)

    box = getattr(results, "box", None)
    if box is not None:
        key_map = {
            "mp": "precision",
            "mr": "recall",
            "map50": "map50",
            "map75": "map75",
            "map": "map50_95",
        }
        for attr, out_key in key_map.items():
            value = getattr(box, attr, None)
            if value is not None:
                metrics[out_key] = float(value)

    speed = getattr(results, "speed", None)
    if isinstance(speed, dict):
        for key, value in speed.items():
            if isinstance(value, (int, float)):
                metrics[f"speed_{key}_ms"] = float(value)

    return metrics


def main() -> None:
    config = load_config()
    eval_cfg = (config.get("evaluate") or {}).copy()

    weights_path = Path(eval_cfg.get("weights", "models/weights/best.pt"))
    data_path = Path(eval_cfg.get("data", "configs/dataset.yaml"))
    split_name = str(eval_cfg.get("split", "test"))
    imgsz = int(eval_cfg.get("imgsz", 640))
    batch = int(eval_cfg.get("batch", 16))
    device = str(eval_cfg.get("device", "auto"))
    output_path = Path(eval_cfg.get("output", "metrics.json"))

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    split_dir = resolve_split_dir(data_path, split_name)
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    model = YOLO(str(weights_path))
    results = model.val(
        data=str(data_path),
        split=split_name,
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=False,
    )

    metrics = extract_metrics(results)
    metrics.update(
        {
            "weights": str(weights_path).replace("\\", "/"),
            "data": str(data_path).replace("\\", "/"),
            "split": split_name,
            "split_images_dir": str(split_dir).replace("\\", "/"),
            "num_split_images": count_images(split_dir),
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved metrics to {output_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
