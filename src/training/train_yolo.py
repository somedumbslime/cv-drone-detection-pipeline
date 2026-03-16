from pathlib import Path

import yaml
from ultralytics import YOLO

CONFIG_PATH = Path("configs/train_config.yaml")


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    config = load_config()
    train_config = (config.get("train") or {}).copy()

    model_name = train_config.pop("model", "yolo11n.pt")
    model = YOLO(model_name)
    model.train(**train_config)


if __name__ == "__main__":
    main()
