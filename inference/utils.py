from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def run_inference(model: YOLO, image, conf: float = 0.25) -> Any:
    return model.predict(source=image, conf=conf, verbose=False)


def draw_detections(image, prediction) -> Any:
    rendered = image.copy()
    boxes = prediction.boxes
    if boxes is None:
        return rendered

    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cls_id = int(box.cls[0].item())
        score = float(box.conf[0].item())
        label = f"{cls_id}:{score:.2f}"

        cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(rendered, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return rendered


def prediction_to_dict(prediction) -> dict[str, Any]:
    result = {"boxes": []}
    boxes = prediction.boxes
    names = prediction.names
    if boxes is None:
        return result

    for box in boxes:
        cls_id = int(box.cls[0].item())
        result["boxes"].append(
            {
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)),
                "confidence": float(box.conf[0].item()),
                "xyxy": [float(v) for v in box.xyxy[0].tolist()],
            }
        )

    return result


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
