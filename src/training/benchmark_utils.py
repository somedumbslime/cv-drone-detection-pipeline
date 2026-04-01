from pathlib import Path
import random
import shutil
import time
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _import_onnxruntime():
    try:
        import onnxruntime as ort  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "onnxruntime is required for ONNX benchmarking. Install it with: pip install onnxruntime (or onnxruntime-gpu)."
        ) from exc
    return ort


def detect_project_root(start: Path | None = None) -> Path:
    base = (start or Path.cwd()).resolve()
    candidates = [base, base.parent, base.parent.parent]
    for candidate in candidates:
        if (candidate / "configs" / "train_config.yaml").exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate project root from cwd: {base}")


def parse_val_metrics(results) -> dict[str, Any]:
    metrics = {}
    box = getattr(results, "box", None)
    if box is not None:
        for attr, name in [("mp", "precision"), ("mr", "recall"), ("map50", "map50"), ("map", "map50_95")]:
            value = getattr(box, attr, None)
            if value is not None:
                metrics[name] = float(value)
    return metrics


def model_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024) if path.exists() else float("nan")


def rel_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root)).replace("\\", "/")
    except Exception:
        return path.resolve().as_posix()


def _resolve_data_root(dataset_cfg: dict, dataset_yaml_path: Path, project_root: Path) -> Path:
    root = Path(dataset_cfg.get("path", dataset_yaml_path.parent))
    if not root.is_absolute():
        root = (project_root / root).resolve()
    return root


def write_runtime_dataset_yaml(source_yaml: Path, target_yaml: Path, project_root: Path) -> Path:
    with source_yaml.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f) or {}

    root = _resolve_data_root(data_cfg, source_yaml, project_root=project_root)
    data_cfg["path"] = root.as_posix()

    target_yaml.parent.mkdir(parents=True, exist_ok=True)
    with target_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_cfg, f, allow_unicode=True, sort_keys=False)

    return target_yaml


def get_split_images(
    dataset_yaml_path: Path,
    split_name: str,
    project_root: Path,
    max_images: int | None = None,
) -> list[Path]:
    with dataset_yaml_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f) or {}

    root = _resolve_data_root(data_cfg, dataset_yaml_path, project_root=project_root)
    split_rel = data_cfg.get(split_name)
    if split_rel is None:
        raise ValueError(f"Split '{split_name}' not found in {dataset_yaml_path}")

    split_dir = (root / split_rel).resolve()
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    images = sorted([p for p in split_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    return images if max_images is None else images[:max_images]


def read_unicode_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def preprocess_for_onnx(image_path: Path, size: int) -> np.ndarray:
    image = read_unicode_image(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))[None, ...]
    return image


def _torch_device_name(device_name: str) -> str:
    d = str(device_name).lower().strip()
    if d.isdigit():
        return f"cuda:{d}"
    if d.startswith("cuda"):
        return d
    return "cpu"


def resolve_model_source(model_ref: str, project_root: Path) -> str:
    candidate = Path(str(model_ref))
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)

    root_candidate = (project_root / candidate).resolve()
    if root_candidate.exists():
        return str(root_candidate)

    return str(model_ref)


def resolve_existing_weights(weights_path: Path, project_root: Path) -> Path:
    candidate = Path(weights_path).resolve()
    if candidate.exists():
        return candidate

    fallback_candidates: list[Path] = []
    for pattern in [
        "models/weights/*.pt",
        "runs/train/**/weights/best.pt",
        "runs/train/**/weights/last.pt",
    ]:
        fallback_candidates.extend([p.resolve() for p in project_root.glob(pattern) if p.is_file()])

    if not fallback_candidates:
        raise FileNotFoundError(
            f"Weights not found: {candidate}. No fallback *.pt files in models/weights or runs/train."
        )

    fallback = max(fallback_candidates, key=lambda p: p.stat().st_mtime)
    print(f"[WARN] Weights not found: {candidate}")
    print(f"[WARN] Using fallback weights: {fallback}")
    return fallback


def _onnx_providers_for_device(device_name: str) -> list[str]:
    ort = _import_onnxruntime()
    available = ort.get_available_providers()
    d = str(device_name).lower().strip()
    want_cuda = d.isdigit() or d.startswith("cuda")

    if want_cuda:
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                f"CUDAExecutionProvider is unavailable in ONNX Runtime. Available providers: {available}"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    return ["CPUExecutionProvider"]


def _onnx_expected_dtype(input_type: str):
    t = (input_type or "").lower()
    if "float16" in t:
        return np.float16
    if "double" in t:
        return np.float64
    return np.float32


def benchmark_pt(weights: Path, tensors: list[np.ndarray], device_name: str, warmup: int, runs: int) -> dict[str, Any]:
    dev_name = _torch_device_name(device_name)
    dev = torch.device(dev_name if (dev_name.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    model = YOLO(str(weights), task="detect").model.to(dev).eval()
    torch_tensors = [torch.from_numpy(t).to(dev) for t in tensors]

    with torch.no_grad():
        for i in range(warmup):
            _ = model(torch_tensors[i % len(torch_tensors)])
        if dev.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for i in range(runs):
            _ = model(torch_tensors[i % len(torch_tensors)])
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000.0 / runs
    return {
        "runtime": f"torch:{dev.type}",
        "latency_ms": latency_ms,
        "fps": (1000.0 / latency_ms) if latency_ms > 0 else float("nan"),
        "input_dtype": "float32",
    }


def benchmark_onnx(model_path: Path, tensors: list[np.ndarray], device_name: str, warmup: int, runs: int) -> dict[str, Any]:
    ort = _import_onnxruntime()
    providers = _onnx_providers_for_device(device_name)
    sess = ort.InferenceSession(str(model_path), providers=providers)

    input_meta = sess.get_inputs()[0]
    input_name = input_meta.name
    expected_dtype = _onnx_expected_dtype(getattr(input_meta, "type", "tensor(float)"))
    cast_tensors = [np.asarray(t, dtype=expected_dtype) for t in tensors]

    for i in range(warmup):
        _ = sess.run(None, {input_name: cast_tensors[i % len(cast_tensors)]})

    t0 = time.perf_counter()
    for i in range(runs):
        _ = sess.run(None, {input_name: cast_tensors[i % len(cast_tensors)]})
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000.0 / runs
    return {
        "runtime": "onnxruntime:" + ",".join(sess.get_providers()),
        "latency_ms": latency_ms,
        "fps": (1000.0 / latency_ms) if latency_ms > 0 else float("nan"),
        "input_dtype": str(expected_dtype).replace("<class '", "").replace("'>", ""),
    }


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def _axis_starts(length: int, tile: int, overlap: float) -> list[int]:
    tile = max(1, int(tile))
    if length <= tile:
        return [0]

    step = max(1, int(round(tile * (1.0 - overlap))))
    starts = list(range(0, max(length - tile, 0) + 1, step))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def build_tile_windows(image_w: int, image_h: int, tile_w: int, tile_h: int, overlap: float) -> list[tuple[int, int, int, int]]:
    xs = _axis_starts(image_w, tile_w, overlap)
    ys = _axis_starts(image_h, tile_h, overlap)

    windows: list[tuple[int, int, int, int]] = []
    for y1 in ys:
        for x1 in xs:
            x2 = min(x1 + tile_w, image_w)
            y2 = min(y1 + tile_h, image_h)
            windows.append((x1, y1, x2, y2))
    return windows


def read_yolo_boxes_abs(label_path: Path, image_w: int, image_h: int) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return boxes

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(float(parts[0]))
                xc = float(parts[1]) * image_w
                yc = float(parts[2]) * image_h
                bw = float(parts[3]) * image_w
                bh = float(parts[4]) * image_h
            except ValueError:
                continue

            if bw <= 0 or bh <= 0:
                continue

            x1 = _clip(xc - bw / 2.0, 0.0, float(image_w))
            y1 = _clip(yc - bh / 2.0, 0.0, float(image_h))
            x2 = _clip(xc + bw / 2.0, 0.0, float(image_w))
            y2 = _clip(yc + bh / 2.0, 0.0, float(image_h))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((cls_id, x1, y1, x2, y2))

    return boxes


def _intersection_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _xyxy_to_yolo_line(cls_id: int, x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> str:
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0

    xc_n = _clip(xc / img_w, 0.0, 1.0)
    yc_n = _clip(yc / img_h, 0.0, 1.0)
    bw_n = _clip(bw / img_w, 0.0, 1.0)
    bh_n = _clip(bh / img_h, 0.0, 1.0)

    return f"{cls_id} {xc_n:.6f} {yc_n:.6f} {bw_n:.6f} {bh_n:.6f}"


def _write_jpg_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def _guess_split_dirs(dataset_cfg: dict, dataset_yaml_path: Path, split_name: str, project_root: Path) -> tuple[Path, Path]:
    root = _resolve_data_root(dataset_cfg, dataset_yaml_path, project_root=project_root)
    split_rel = dataset_cfg.get(split_name)
    if split_rel is None:
        raise ValueError(f"Split '{split_name}' not found in {dataset_yaml_path}")

    images_dir = Path(split_rel)
    if not images_dir.is_absolute():
        images_dir = (root / images_dir).resolve()

    img_posix = images_dir.as_posix()
    if "/images/" in img_posix:
        labels_dir = Path(img_posix.replace("/images/", "/labels/")).resolve()
    elif img_posix.endswith("/images"):
        labels_dir = (images_dir.parent / "labels").resolve()
    else:
        labels_dir = (root / "labels" / split_name).resolve()

    return images_dir, labels_dir


def tile_one_split(
    images_dir: Path,
    labels_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    tile_w: int,
    tile_h: int,
    overlap: float,
    keep_ratio: float,
    min_box_px: int,
    empty_to_pos_ratio: float,
    rng: random.Random,
) -> dict[str, int]:
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "source_images": 0,
        "tiles_total": 0,
        "positive_tiles": 0,
        "empty_tiles": 0,
        "saved_tiles": 0,
        "saved_positive": 0,
        "saved_empty": 0,
    }

    image_paths = sorted([p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])

    for image_path in image_paths:
        image = read_unicode_image(image_path)
        if image is None:
            continue

        stats["source_images"] += 1
        image_h, image_w = image.shape[:2]

        rel_id = image_path.relative_to(images_dir).with_suffix("").as_posix().replace("/", "__")
        label_path = labels_dir / image_path.relative_to(images_dir).with_suffix(".txt")

        boxes = read_yolo_boxes_abs(label_path, image_w=image_w, image_h=image_h)
        windows = build_tile_windows(image_w=image_w, image_h=image_h, tile_w=tile_w, tile_h=tile_h, overlap=overlap)

        candidates = []
        for tile_idx, (tx1, ty1, tx2, ty2) in enumerate(windows):
            tile_w_local = tx2 - tx1
            tile_h_local = ty2 - ty1
            tile_label_lines: list[str] = []

            for cls_id, bx1, by1, bx2, by2 in boxes:
                inter = _intersection_xyxy((bx1, by1, bx2, by2), (tx1, ty1, tx2, ty2))
                if inter is None:
                    continue

                inter_area = (inter[2] - inter[0]) * (inter[3] - inter[1])
                box_area = max((bx2 - bx1) * (by2 - by1), 1e-6)
                ratio = inter_area / box_area
                if ratio < keep_ratio:
                    continue

                ix1, iy1, ix2, iy2 = inter
                inter_w = ix2 - ix1
                inter_h = iy2 - iy1
                if inter_w < min_box_px or inter_h < min_box_px:
                    continue

                lx1 = ix1 - tx1
                ly1 = iy1 - ty1
                lx2 = ix2 - tx1
                ly2 = iy2 - ty1

                if lx2 <= lx1 or ly2 <= ly1:
                    continue

                tile_label_lines.append(
                    _xyxy_to_yolo_line(
                        cls_id=cls_id,
                        x1=lx1,
                        y1=ly1,
                        x2=lx2,
                        y2=ly2,
                        img_w=tile_w_local,
                        img_h=tile_h_local,
                    )
                )

            candidates.append((tile_idx, tx1, ty1, tx2, ty2, tile_label_lines))

        stats["tiles_total"] += len(candidates)

        positives = [c for c in candidates if len(c[5]) > 0]
        empties = [c for c in candidates if len(c[5]) == 0]

        stats["positive_tiles"] += len(positives)
        stats["empty_tiles"] += len(empties)

        if positives:
            n_empty = int(round(len(positives) * max(empty_to_pos_ratio, 0.0)))
            n_empty = min(len(empties), max(0, n_empty))
        else:
            n_empty = 1 if (len(empties) > 0 and empty_to_pos_ratio > 0) else 0

        chosen_empties = rng.sample(empties, n_empty) if n_empty > 0 and n_empty < len(empties) else empties[:n_empty]
        selected = positives + chosen_empties

        for tile_idx, tx1, ty1, tx2, ty2, tile_label_lines in selected:
            tile_img = image[ty1:ty2, tx1:tx2]
            if tile_img.size == 0:
                continue

            tile_stem = f"{rel_id}__t{tile_idx:03d}"
            out_img = out_images_dir / f"{tile_stem}.jpg"
            out_lbl = out_labels_dir / f"{tile_stem}.txt"

            _write_jpg_unicode(out_img, tile_img)
            out_lbl.parent.mkdir(parents=True, exist_ok=True)
            with out_lbl.open("w", encoding="utf-8") as f:
                if tile_label_lines:
                    f.write("\n".join(tile_label_lines) + "\n")

            stats["saved_tiles"] += 1
            if tile_label_lines:
                stats["saved_positive"] += 1
            else:
                stats["saved_empty"] += 1

    return stats


def build_tiled_dataset(
    source_dataset_yaml: Path,
    tile_root: Path,
    tile_w: int,
    tile_h: int,
    overlap: float,
    keep_ratio: float,
    min_box_px: int,
    empty_to_pos_ratio: float,
    seed: int,
    project_root: Path,
) -> Path:
    with source_dataset_yaml.open("r", encoding="utf-8") as f:
        dataset_cfg = yaml.safe_load(f) or {}

    if tile_root.exists():
        shutil.rmtree(tile_root)
    tile_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    split_keys = [s for s in ("train", "val", "test") if dataset_cfg.get(s) is not None]
    split_stats: dict[str, dict[str, int]] = {}

    for split_name in split_keys:
        images_dir, labels_dir = _guess_split_dirs(dataset_cfg, source_dataset_yaml, split_name, project_root=project_root)
        out_images_dir = tile_root / "images" / split_name
        out_labels_dir = tile_root / "labels" / split_name

        stats = tile_one_split(
            images_dir=images_dir,
            labels_dir=labels_dir,
            out_images_dir=out_images_dir,
            out_labels_dir=out_labels_dir,
            tile_w=tile_w,
            tile_h=tile_h,
            overlap=overlap,
            keep_ratio=keep_ratio,
            min_box_px=min_box_px,
            empty_to_pos_ratio=empty_to_pos_ratio,
            rng=rng,
        )
        split_stats[split_name] = stats

    tiled_cfg = {
        "path": tile_root.as_posix(),
        "nc": dataset_cfg.get("nc"),
        "names": dataset_cfg.get("names"),
    }
    for split_name in split_keys:
        tiled_cfg[split_name] = f"images/{split_name}"

    tiled_dataset_yaml = tile_root / "dataset.yaml"
    with tiled_dataset_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(tiled_cfg, f, allow_unicode=True, sort_keys=False)

    print("Tiling summary:")
    for split_name in split_keys:
        s = split_stats[split_name]
        print(
            f"- {split_name}: images={s['source_images']}, tiles={s['tiles_total']}, "
            f"saved={s['saved_tiles']} (pos={s['saved_positive']}, empty={s['saved_empty']})"
        )

    return tiled_dataset_yaml
