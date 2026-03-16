# Architecture

The project follows a simple and interview-friendly ML pipeline:

1. Collect drone images and videos in diverse environments.
2. Annotate objects in CVAT and export labels in YOLO format.
3. Validate and split dataset into train/val/test.
4. Train a YOLO detector.
5. Export best model to ONNX for portable inference.
6. Run inference via scripts (image/video) and expose a REST API.

## Components

- `data/`: dataset stages (raw, annotations, processed)
- `training/`: training entrypoint and configs
- `inference/`: reusable inference logic and CLIs
- `api/`: FastAPI serving layer
- `models/`: trained and exported model artifacts

## Design Goals

- Keep implementation practical and readable.
- Show full ML lifecycle for CV portfolio use.
- Avoid unnecessary infrastructure complexity.
