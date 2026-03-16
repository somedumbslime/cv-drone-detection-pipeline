from pathlib import Path
import shutil

import yaml
from ultralytics import YOLO

CONFIG_PATH = Path("configs/train_config.yaml")


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    config = load_config()
    export_cfg = (config.get("export") or {}).copy()

    weights = export_cfg.get("weights", "models/weights/best.pt")
    imgsz = int(export_cfg.get("imgsz", 640))
    export_format = str(export_cfg.get("format", "onnx"))
    dynamic = bool(export_cfg.get("dynamic", False))
    simplify = bool(export_cfg.get("simplify", True))
    output_dir = Path(export_cfg.get("output_dir", "models/onnx"))

    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    exported = model.export(format=export_format, imgsz=imgsz, dynamic=dynamic, simplify=simplify)
    exported_path = Path(str(exported))

    target_name = "model.onnx" if export_format == "onnx" else exported_path.name
    target_path = output_dir / target_name

    if exported_path.resolve() != target_path.resolve():
        shutil.copy2(exported_path, target_path)

    print(f"Exported model: {target_path}")


if __name__ == "__main__":
    main()
