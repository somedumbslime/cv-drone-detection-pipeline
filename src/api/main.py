from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile

try:
    from .schemas import PredictResponse
    from ..inference.utils import load_model, prediction_to_dict, run_inference
except ImportError:
    from src.api.schemas import PredictResponse
    from src.inference.utils import load_model, prediction_to_dict, run_inference

CONFIG_PATH = Path("configs/train_config.yaml")


def get_weights_path() -> Path:
    if not CONFIG_PATH.exists():
        return Path("models/weights/best.pt")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    api_cfg = config.get("api") or {}
    inference_cfg = config.get("inference") or {}
    return Path(api_cfg.get("weights", inference_cfg.get("weights", "models/weights/best.pt")))


WEIGHTS_PATH = get_weights_path()
_model = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _model
    if WEIGHTS_PATH.exists():
        _model = load_model(str(WEIGHTS_PATH))
    yield


app = FastAPI(title="Drone Detection API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "weights": str(WEIGHTS_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail=f"Model is not loaded. Put weights at {WEIGHTS_PATH}")

    content = await file.read()
    image_array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    prediction = run_inference(_model, image)[0]
    result = prediction_to_dict(prediction)
    return PredictResponse(**result)
