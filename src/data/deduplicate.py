from pathlib import Path
import shutil

import cv2
import numpy as np
import yaml

CONFIG_PATH = Path("configs/train_config.yaml")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def average_hash(image: np.ndarray, hash_size: int = 8) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    return resized > resized.mean()


def hamming_distance(hash_a: np.ndarray, hash_b: np.ndarray) -> int:
    return int(np.count_nonzero(hash_a != hash_b))


def main() -> None:
    config = load_config()
    dedup_cfg = ((config.get("data_preparation") or {}).get("deduplicate") or {}).copy()

    input_dir = Path(dedup_cfg.get("input_dir", "data/interim/frames"))
    output_dir = Path(dedup_cfg.get("output_dir", "data/interim/unique_frames"))
    hash_size = int(dedup_cfg.get("hash_size", 8))
    threshold = int(dedup_cfg.get("hamming_threshold", 5))

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
    unique_hashes: list[np.ndarray] = []
    unique_count = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        current_hash = average_hash(image, hash_size=hash_size)
        is_duplicate = any(hamming_distance(current_hash, existing_hash) <= threshold for existing_hash in unique_hashes)

        if not is_duplicate:
            shutil.copy2(image_path, output_dir / image_path.name)
            unique_hashes.append(current_hash)
            unique_count += 1

    print(f"Input images: {len(image_paths)}")
    print(f"Unique images: {unique_count}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
