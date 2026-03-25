from pathlib import Path
from typing import Any

import cv2
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_CONFIG = PROJECT_ROOT / "configs" / "dataset.yaml"

# BGR palette for readable class-colored boxes
_COLOR_PALETTE = [(60, 200, 50)]


def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def run_inference(model: YOLO, image, conf: float = 0.25) -> Any:
    return model.predict(source=image, conf=conf, verbose=False)


def _resolve_data_config_path(data_config_path: str | None) -> Path:
    if data_config_path:
        raw = Path(data_config_path)
    else:
        raw = DEFAULT_DATASET_CONFIG

    if raw.is_absolute():
        return raw

    cwd_candidate = (Path.cwd() / raw).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (PROJECT_ROOT / raw).resolve()


def load_class_names(data_config_path: str | None = None) -> dict[int, str]:
    """Load class names from YOLO dataset yaml (`names` field)."""
    path = _resolve_data_config_path(data_config_path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    names = cfg.get("names", {})
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(v) for idx, v in enumerate(names)}
    return {}


def _prediction_class_name(prediction, cls_id: int) -> str:
    names = getattr(prediction, "names", {})
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, list) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def _class_color(cls_id: int) -> tuple[int, int, int]:
    return _COLOR_PALETTE[cls_id % len(_COLOR_PALETTE)]


def _fit_label_to_box(
    class_name: str,
    score: float,
    max_text_width: int,
    font,
    base_scale: float,
) -> tuple[str, float, int, int, int, int]:
    """Fit label text into available width by reducing scale/verbosity."""
    max_text_width = max(max_text_width, 16)
    candidates = [f"{class_name} {score:.2f}", class_name, class_name[:10]]

    for label in candidates:
        scale = base_scale
        while scale >= 0.35:
            thickness = max(1, int(round(scale * 2)))
            (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
            if tw <= max_text_width:
                return label, scale, thickness, tw, th, baseline
            scale -= 0.03

    # Last fallback: always keep class name visible
    fallback = class_name if class_name else "object"
    scale = 0.35
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(fallback, font, scale, thickness)
    return fallback, scale, thickness, tw, th, baseline


def draw_detections(
    image, prediction, class_names: dict[int, str] | None = None
) -> Any:
    """Draw detections with tidy labels sized/placed per bbox."""
    rendered = image.copy()
    boxes = prediction.boxes
    if boxes is None or len(boxes) == 0:
        return rendered

    h, w = rendered.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        box_w = x2 - x1
        box_h = y2 - y1

        cls_id = int(box.cls[0].item())
        score = float(box.conf[0].item())

        if class_names and cls_id in class_names:
            class_name = class_names[cls_id]
        else:
            class_name = _prediction_class_name(prediction, cls_id)

        # Scale styles with box size (cleaner look on small/large boxes)
        line_thickness = max(1, int(round(min(box_w, box_h) / 90)))
        base_scale = min(max(min(box_w, box_h) / 180.0, 0.42), 1.0)
        color = _class_color(cls_id)

        # Main bbox
        cv2.rectangle(
            rendered, (x1, y1), (x2, y2), color, line_thickness, lineType=cv2.LINE_AA
        )

        # Label fitted to frame width (class name must stay visible even for tiny boxes)
        label, font_scale, text_thickness, tw, th, baseline = _fit_label_to_box(
            class_name, score, max_text_width=w - 12, font=font, base_scale=base_scale
        )

        pad_x = 4
        label_h = th + baseline + 6
        label_w = tw + 2 * pad_x

        # Prefer label above bbox; then below bbox; final fallback to nearest visible area
        top_y1 = y1 - label_h - 2
        top_y2 = y1 - 2

        # Keep label visible in frame width
        max_bg_x1 = max(0, w - label_w - 1)
        bg_x1 = max(0, min(x1, max_bg_x1))
        bg_x2 = min(w - 1, bg_x1 + label_w)

        if top_y1 >= 0:
            bg_y1, bg_y2 = top_y1, top_y2
        elif y2 + label_h + 2 <= h - 1:
            bg_y1 = y2 + 2
            bg_y2 = y2 + 2 + label_h
        else:
            bg_y1 = max(0, min(y1 + 2, h - label_h - 1))
            bg_y2 = min(h - 1, bg_y1 + label_h)

        text_x = bg_x1 + pad_x
        text_y = bg_y2 - baseline - 2

        # Keep coords in frame bounds
        bg_x1 = max(0, min(bg_x1, w - 1))
        bg_y1 = max(0, min(bg_y1, h - 1))
        bg_x2 = max(0, min(bg_x2, w - 1))
        bg_y2 = max(0, min(bg_y2, h - 1))

        if bg_x2 <= bg_x1 or bg_y2 <= bg_y1:
            continue

        cv2.rectangle(rendered, (bg_x1, bg_y1), (bg_x2, bg_y2), color, thickness=-1)
        cv2.putText(
            rendered,
            label,
            (text_x, max(text_y, th + baseline + 1)),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            lineType=cv2.LINE_AA,
        )

    return rendered


def prediction_to_dict(
    prediction, class_names: dict[int, str] | None = None
) -> dict[str, Any]:
    result = {"boxes": []}
    boxes = prediction.boxes
    if boxes is None:
        return result

    for box in boxes:
        cls_id = int(box.cls[0].item())
        if class_names and cls_id in class_names:
            class_name = class_names[cls_id]
        else:
            class_name = _prediction_class_name(prediction, cls_id)

        result["boxes"].append(
            {
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": float(box.conf[0].item()),
                "xyxy": [float(v) for v in box.xyxy[0].tolist()],
            }
        )

    return result


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
