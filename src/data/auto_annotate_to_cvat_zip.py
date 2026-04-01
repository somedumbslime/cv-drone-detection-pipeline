from pathlib import Path
import shutil
import tempfile
import zipfile

import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def resolve_model_source(model_ref: str) -> str:
    candidate = Path(model_ref)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    root_candidate = (PROJECT_ROOT / candidate).resolve()
    if root_candidate.exists():
        return str(root_candidate)

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)

    return model_ref


def load_class_names(dataset_yaml_path: Path, model: YOLO) -> list[str]:
    names: dict[int, str] = {}

    if dataset_yaml_path.exists():
        with dataset_yaml_path.open("r", encoding="utf-8") as f:
            dataset_cfg = yaml.safe_load(f) or {}

        raw_names = dataset_cfg.get("names", {})
        if isinstance(raw_names, dict):
            names = {int(k): str(v) for k, v in raw_names.items()}
        elif isinstance(raw_names, list):
            names = {idx: str(v) for idx, v in enumerate(raw_names)}

    if not names:
        raw_model_names = getattr(model, "names", {})
        if isinstance(raw_model_names, dict):
            names = {int(k): str(v) for k, v in raw_model_names.items()}
        elif isinstance(raw_model_names, list):
            names = {idx: str(v) for idx, v in enumerate(raw_model_names)}

    if not names:
        raise ValueError("Unable to resolve class names from dataset config or model.")

    max_idx = max(names.keys())
    return [names.get(idx, f"class_{idx}") for idx in range(max_idx + 1)]


def list_images(input_dir: Path) -> list[Path]:
    return sorted([p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def make_safe_filename(input_dir: Path, image_path: Path) -> str:
    rel = image_path.relative_to(input_dir)
    stem = rel.with_suffix("").as_posix().replace("/", "__")
    suffix = image_path.suffix.lower() if image_path.suffix else ".jpg"
    return f"{stem}{suffix}"


def predict_yolo_labels(
    model: YOLO,
    image_paths: list[Path],
    conf: float,
    iou: float,
    device: str,
) -> tuple[dict[Path, list[str]], int]:
    labels_by_image: dict[Path, list[str]] = {}
    total_boxes = 0

    for idx, image_path in enumerate(image_paths, start=1):
        prediction = model.predict(
            source=str(image_path),
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )[0]

        lines: list[str] = []
        boxes = prediction.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                xc, yc, bw, bh = [float(v) for v in box.xywhn[0].tolist()]
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                bw = max(0.0, min(1.0, bw))
                bh = max(0.0, min(1.0, bh))
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        labels_by_image[image_path] = lines
        total_boxes += len(lines)

        if idx % 50 == 0 or idx == len(image_paths):
            print(f"Processed {idx}/{len(image_paths)} images")

    return labels_by_image, total_boxes


def write_cvat_yolo_zip(
    input_dir: Path,
    image_paths: list[Path],
    labels_by_image: dict[Path, list[str]],
    class_names: list[str],
    output_zip: Path,
) -> None:
    with tempfile.TemporaryDirectory(prefix="cvat_yolo_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        data_dir = tmp_root / "obj_train_data"
        data_dir.mkdir(parents=True, exist_ok=True)

        train_entries: list[str] = []

        for image_path in image_paths:
            out_image_name = make_safe_filename(input_dir, image_path)
            out_image_path = data_dir / out_image_name
            out_label_path = data_dir / f"{Path(out_image_name).stem}.txt"

            shutil.copy2(image_path, out_image_path)

            lines = labels_by_image.get(image_path, [])
            label_content = "\n".join(lines)
            if label_content:
                label_content += "\n"
            out_label_path.write_text(label_content, encoding="utf-8")

            train_entries.append(f"data/obj_train_data/{out_image_name}")

        (tmp_root / "obj.names").write_text("\n".join(class_names) + "\n", encoding="utf-8")
        (tmp_root / "train.txt").write_text("\n".join(train_entries) + "\n", encoding="utf-8")
        (tmp_root / "obj.data").write_text(
            f"classes = {len(class_names)}\n"
            "train = train.txt\n"
            "names = obj.names\n"
            "backup = backup/\n",
            encoding="utf-8",
        )

        output_zip.parent.mkdir(parents=True, exist_ok=True)
        if output_zip.exists():
            output_zip.unlink()

        with zipfile.ZipFile(output_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(tmp_root.rglob("*")):
                if file_path.is_file():
                    zf.write(file_path, arcname=file_path.relative_to(tmp_root).as_posix())


def main() -> None:
    cfg = load_config()
    prep_cfg = (cfg.get("data_preparation") or {}).copy()
    semi_cfg = (prep_cfg.get("semi_annotation") or {}).copy()

    default_input_dir = ((prep_cfg.get("deduplicate") or {}).get("output_dir") or "data/interim/unique_frames")
    input_dir = resolve_path(semi_cfg.get("input_dir", default_input_dir))

    default_zip = input_dir.parent / f"{input_dir.name}_yolo_detection_1_0.zip"
    output_zip = resolve_path(semi_cfg.get("output_zip", str(default_zip)))

    inference_cfg = (cfg.get("inference") or {}).copy()
    evaluate_cfg = (cfg.get("evaluate") or {}).copy()

    weights_ref = (
        semi_cfg.get("weights")
        or evaluate_cfg.get("weights")
        or inference_cfg.get("weights")
        or "models/weights/best.pt"
    )
    model_source = resolve_model_source(str(weights_ref))

    conf = float(semi_cfg.get("conf", inference_cfg.get("conf", 0.25)))
    iou = float(semi_cfg.get("iou", 0.5))
    device = str(semi_cfg.get("device", evaluate_cfg.get("device", "cpu")))

    dataset_yaml_ref = semi_cfg.get("data") or inference_cfg.get("data") or cfg.get("train", {}).get("data") or "configs/dataset.yaml"
    dataset_yaml_path = resolve_path(dataset_yaml_ref)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = list_images(input_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in: {input_dir}")

    model = YOLO(model_source)
    class_names = load_class_names(dataset_yaml_path, model)

    labels_by_image, total_boxes = predict_yolo_labels(
        model=model,
        image_paths=image_paths,
        conf=conf,
        iou=iou,
        device=device,
    )
    write_cvat_yolo_zip(
        input_dir=input_dir,
        image_paths=image_paths,
        labels_by_image=labels_by_image,
        class_names=class_names,
        output_zip=output_zip,
    )

    images_with_boxes = sum(1 for p in image_paths if labels_by_image.get(p))
    print(f"Images processed: {len(image_paths)}")
    print(f"Images with detections: {images_with_boxes}")
    print(f"Total predicted boxes: {total_boxes}")
    print(f"Model source: {model_source}")
    print(f"Archive created: {output_zip}")
    print("Import this zip into CVAT as: YOLO Detection 1.0")


if __name__ == "__main__":
    main()
