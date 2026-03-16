# Architecture

[Українська версія](architecture_ua.md)

## Pipeline Overview

This repository demonstrates an end-to-end computer vision workflow for drone object detection:

1. Raw videos/images are stored in `data/raw/`.
2. Frames are extracted into `data/interim/`.
3. Near-duplicate frames are removed.
4. Data is annotated in CVAT.
5. CVAT export is converted to YOLO dataset format and split into train/val/test.
6. YOLO model is trained and best weights are saved in `models/weights/`.
7. Trained weights are exported to ONNX in `models/onnx/`.
8. Inference is available via scripts (`src/inference/`) and FastAPI (`src/api/`).

## Data Flow

```text
Raw videos / images
        -> Frame extraction
        -> Deduplication
        -> CVAT annotation
        -> YOLO dataset conversion + split
        -> YOLO training
        -> ONNX export
        -> Image/Video inference
        -> FastAPI /predict endpoint
```

## Automated vs Manual Steps

Automated:

- frame extraction (`src/data/extract_frames.py`)
- deduplication (`src/data/deduplicate.py`)
- dataset conversion and split (`src/data/prepare_dataset.py`)
- training (`src/training/train_yolo.py`)
- evaluation (`src/training/evaluate.py`)
- ONNX export (`src/training/export_onnx.py`)
- inference scripts and API

Manual:

- source data collection
- labeling decisions and QA in CVAT
- class taxonomy updates
- model error analysis and iteration

## Design Decisions

- Keep the architecture simple and interview-defensible.
- Prioritize clear ML pipeline stages over extra infrastructure.
- Use one detector family (YOLO) to focus on data quality and reproducibility.
- Keep API as a minimal serving layer for model predictions.

