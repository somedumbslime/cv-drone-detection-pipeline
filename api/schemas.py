from pydantic import BaseModel


class BBoxPrediction(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    xyxy: list[float]


class PredictResponse(BaseModel):
    boxes: list[BBoxPrediction]
