from pathlib import Path
import random
import shutil

import yaml

CONFIG_PATH = Path("configs/train_config.yaml")
DATASET_CONFIG_PATH = Path("configs/dataset.yaml")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_class_names(cvat_root: Path) -> list[str]:
    names_path = cvat_root / "obj.names"
    if not names_path.exists():
        raise FileNotFoundError(f"Class names file not found: {names_path}")

    names = [line.strip() for line in names_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"No class names found in: {names_path}")

    return names


def resolve_images(cvat_root: Path) -> list[Path]:
    train_txt = cvat_root / "train.txt"
    images: list[Path] = []

    if train_txt.exists():
        for raw_line in train_txt.read_text(encoding="utf-8").splitlines():
            rel = raw_line.strip().replace("\\", "/")
            if not rel:
                continue

            rel_no_data = rel[5:] if rel.startswith("data/") else rel
            candidates = [
                cvat_root / rel,
                cvat_root / rel_no_data,
                cvat_root / "obj_train_data" / Path(rel).name,
            ]
            src = next((p for p in candidates if p.exists()), None)
            if src is None:
                raise FileNotFoundError(f"Image not found for entry: {rel}")
            images.append(src)

        return images

    data_dir = cvat_root / "obj_train_data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Not found: {data_dir}")

    return sorted([p for p in data_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])


def split_images(image_paths: list[Path], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> dict[str, list[Path]]:
    shuffled = image_paths[:]
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def reset_output(yolo_root: Path) -> None:
    shutil.rmtree(yolo_root / "images", ignore_errors=True)
    shutil.rmtree(yolo_root / "labels", ignore_errors=True)

    for name in ["train.txt", "val.txt", "test.txt", "classes.txt", "dataset.yaml"]:
        path = yolo_root / name
        if path.exists():
            path.unlink()


def create_dirs(yolo_root: Path) -> None:
    for split_name in ["train", "val", "test"]:
        (yolo_root / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / split_name).mkdir(parents=True, exist_ok=True)


def copy_split_files(yolo_root: Path, split_name: str, image_paths: list[Path]) -> int:
    missing_labels = 0
    rel_list: list[str] = []

    for image_path in image_paths:
        dst_image = yolo_root / "images" / split_name / image_path.name
        shutil.copy2(image_path, dst_image)

        src_label = image_path.with_suffix(".txt")
        dst_label = yolo_root / "labels" / split_name / f"{image_path.stem}.txt"
        if src_label.exists():
            shutil.copy2(src_label, dst_label)
        else:
            dst_label.write_text("", encoding="utf-8")
            missing_labels += 1

        rel_list.append(f"images/{split_name}/{image_path.name}")

    content = "\n".join(rel_list)
    if content:
        content += "\n"
    (yolo_root / f"{split_name}.txt").write_text(content, encoding="utf-8")

    return missing_labels


def write_dataset_yaml(dataset_path: Path, yolo_root: Path, class_names: list[str]) -> None:
    data = {
        "path": str(yolo_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(class_names)},
    }

    with dataset_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def main() -> None:
    config = load_config()
    prep_cfg = (config.get("data_preparation") or {}).copy()

    split_cfg = (prep_cfg.get("split") or {}).copy()
    train_ratio = float(split_cfg.get("train", 0.7))
    val_ratio = float(split_cfg.get("val", 0.2))
    test_ratio = float(split_cfg.get("test", 0.1))
    seed = int(split_cfg.get("seed", 42))

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split coefficients must sum to 1.0, got: {ratio_sum}")

    cvat_root = Path(prep_cfg.get("cvat_root", "data/processed/CVAT"))
    yolo_root = Path(prep_cfg.get("yolo_root", "data/processed/YOLO"))

    if not cvat_root.exists():
        raise FileNotFoundError(f"CVAT directory not found: {cvat_root}")

    class_names = read_class_names(cvat_root)
    image_paths = resolve_images(cvat_root)
    splits = split_images(image_paths, train_ratio, val_ratio, test_ratio, seed)

    reset_output(yolo_root)
    create_dirs(yolo_root)

    missing_labels = 0
    missing_labels += copy_split_files(yolo_root, "train", splits["train"])
    missing_labels += copy_split_files(yolo_root, "val", splits["val"])
    missing_labels += copy_split_files(yolo_root, "test", splits["test"])

    (yolo_root / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")

    write_dataset_yaml(yolo_root / "dataset.yaml", yolo_root, class_names)
    write_dataset_yaml(DATASET_CONFIG_PATH, yolo_root, class_names)

    print(f"Total images: {len(image_paths)}")
    print(f"train/val/test: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    print(f"Missing label files: {missing_labels}")
    print(f"YOLO dataset saved to: {yolo_root}")
    print(f"Updated dataset config: {DATASET_CONFIG_PATH}")


if __name__ == "__main__":
    main()
