import yaml
from ultralytics import YOLO


def main() -> None:
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    train_config = (config.get("train") or {}).copy()
    model_name = train_config.pop("model")
    model = YOLO(model_name)
    model.train(**train_config)


if __name__ == "__main__":
    main()
