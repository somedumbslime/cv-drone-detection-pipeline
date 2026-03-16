from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile

from api.schemas import PredictResponse

# Reuse inference helpers from shared module.
from inference.utils import load_model, prediction_to_dict, run_inference

CONFIG_PATH = Path("config/config.yaml")


def get_default_weights() -> Path:
    if not CONFIG_PATH.exists():
        return Path("models/weights/yolo_best.pt")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    inference_config = config.get("inference") or {}
    return Path(inference_config.get("weights", "models/weights/yolo_best.pt"))


DEFAULT_WEIGHTS = get_default_weights()

app = FastAPI(title="Drone Detection API", version="0.1.0")
_model = None


@app.on_event("startup")
def startup_event() -> None:
    global _model
    if DEFAULT_WEIGHTS.exists():
        _model = load_model(str(DEFAULT_WEIGHTS))


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "weights": str(DEFAULT_WEIGHTS),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail=f"Model is not loaded. Put weights at {DEFAULT_WEIGHTS}")

    content = await file.read()
    image_array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    pred = run_inference(_model, image)[0]
    result = prediction_to_dict(pred)
    return PredictResponse(**result)
