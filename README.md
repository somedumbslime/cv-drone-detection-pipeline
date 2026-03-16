# CV Drone Detection Pipeline

End-to-end computer vision pipeline for drone detection: dataset preparation, annotation flow, YOLO training, model export to ONNX, inference scripts, and a lightweight FastAPI service.

## Project Overview

This repository demonstrates a practical ML engineering workflow for object detection without heavy production overhead.

## Architecture

```text
Dataset
   |
   v
Annotation (CVAT)
   |
   v
Training (YOLO)
   |
   v
Model Export (ONNX)
   |
   v
Inference
   |
   v
FastAPI Service
```

Detailed architecture notes: `docs/architecture.md`

## Dataset

- Raw images: `data/raw/`
- Annotation files: `data/annotations/`
- Processed splits: `data/processed/`

See dataset documentation in `docs/dataset.md`.

Convert CVAT export to YOLO format (with train/val/test split from `config/config.yaml`):

```bash
python scripts/convert_cvat_to_yolo.py
```

## Training

Main script: `training/train_yolo.py`

Example:

```bash
python training/train_yolo.py
```

Configs:

- `config/config.yaml` - all runtime settings (training + inference)
- `config/dataset.yaml` - YOLO dataset paths and classes

## Inference

Image inference:

```bash
python inference/infer_image.py
```

Video inference:

```bash
python inference/infer_video.py
```

## API

Run API:

```bash
uvicorn api.main:app --reload
```

Endpoint:

- `POST /predict` with multipart field `file` (image)

Response contains predicted bounding boxes, classes, and confidence scores.

## How To Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare dataset and labels.
3. Train model with `training/train_yolo.py`.
4. Export model to ONNX (optional for deployment).
5. Run inference scripts or start API.

## Results

Add to this section:

- qualitative detection examples (`examples/output/`)
- validation metrics (precision, recall, mAP)
- speed/latency notes for inference

## Tech Stack

Python, PyTorch, Ultralytics YOLO, OpenCV, ONNX, FastAPI
