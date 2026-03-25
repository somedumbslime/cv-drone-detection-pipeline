# RAPTOR-AI (MilTech, UAV)

[Українська версія](README_UA.md)

<p align="center">
  <img src="assets/raptor-ai.gif" alt="Raptor AI"/>
</p>

End-to-end computer vision project focused on drone detection: dataset preparation, CVAT annotation workflow, YOLO training, ONNX export, inference scripts, and FastAPI serving.

<p align="center">
  <img src="assets/soldiers.gif" alt="Soldier detection demo" />
</p>

## Table of Contents

- [Project Overview](#project-overview)
- [What This Project Demonstrates](#what-this-project-demonstrates)
- [Architecture](#architecture)
- [Dataset And Labeling Workflow](#dataset-and-labeling-workflow)
- [Training Pipeline](#training-pipeline)
- [Benchmark Results (Full Test Split, GPU)](#benchmark-results-full-test-split-gpu)
- [Inference](#inference)
- [API](#api)
- [Repository Structure](#repository-structure)
- [Example Results](#example-results)
- [Business Value](#business-value)
- [Challenges Solved](#challenges-solved)
- [What I Learned](#what-i-learned)
- [How To Run](#how-to-run)

## Project Overview

This project is designed as a CV-oriented ML engineering pipeline, not a web product. The core focus is data workflow, model training, reproducible evaluation, and lightweight deployment.

Practical focus: automate object detection in drone imagery to reduce manual review time and speed up monitoring workflows.

## What This Project Demonstrates

- dataset preparation from raw media
- deduplication and data quality checks
- perceptual-hash based duplicate removal before annotation to reduce label noise
- CVAT-to-YOLO annotation conversion with train/val/test split
- YOLO model training and evaluation
- ONNX export for deployment-ready inference
- FastAPI endpoint for image prediction

## Architecture

Ukrainian docs: `docs/architecture_ua.md`, `docs/dataset_ua.md`

```text
Raw videos / images
        ↓
Frame extraction
        ↓
Deduplication
        ↓
Annotation workflow (CVAT)
        ↓
Dataset split
        ↓
YOLO training
        ↓
Model export (ONNX)
        ↓
Image / video inference
        ↓
FastAPI API
```

## Dataset And Labeling Workflow

- Source media is stored in `data/raw/`
- Frames are extracted to `data/interim/`
- Near-duplicates are removed before annotation
- Images are annotated in CVAT
- CVAT export is converted into YOLO format in `data/processed/YOLO/`
- Split coefficients are configured in `configs/train_config.yaml`

Detailed notes: `docs/dataset.md`

## Training Pipeline

- Training config: `configs/train_config.yaml` (`train` section)
- Dataset config: `configs/dataset.yaml`
- Training script: `src/training/train_yolo.py`
- Evaluation script: `src/training/evaluate.py`
- Metrics output: `metrics.json`
- Full training/export/benchmark notebook: `notebooks/model_pipeline_benchmark.ipynb`
  - GPU benchmark table: ONNX FP32 comparison across model variants

## Benchmark Results (Full Test Split, GPU)

Benchmark setup:

- GPU: NVIDIA GeForce GTX 1080
- Frameworks: PyTorch + ONNX Runtime (`CUDAExecutionProvider`)
- Evaluation data: full `test` split from `configs/dataset.yaml`
- Input resolution: `640x640`
- Benchmark batch size: `1` (single-image latency measurement)

| model | size_mb | precision | recall | map50 | map50_95 | latency_ms | fps | input_dtype |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `yolo12s (onnx.fp32)` | 10.09 | 0.4940 | 0.3491 | 0.3540 | 0.1648 | 9.332 | 107.16 | `numpy.float32` |
| `yolo26n (onnx.fp32)` | 9.35 | 0.7426 | 0.5385 | 0.6079 | 0.3226 | 8.936 | 111.91 | `numpy.float32` |

Key takeaways:

- `yolo26n (onnx.fp32)` outperforms `yolo12s (onnx.fp32)` on this setup.
- Quality is significantly higher for `yolo26n` (`map50 +0.2539`, `map50_95 +0.1578`).
- Runtime is also better for `yolo26n` (`8.936 ms` vs `9.332 ms`, `111.91 FPS` vs `107.16 FPS`) with smaller model size.

Recommended deployment choice for edge inference:

- Primary model: `yolo26n (onnx.fp32)` for best latency/FPS to quality balance in this benchmark.
- Secondary option: `yolo12s (onnx.fp32)` if project constraints require that exact baseline variant.

## Inference

- Image inference: `src/inference/infer_image.py`
- Video inference: `src/inference/infer_video.py`
- Common utilities: `src/inference/utils.py`

Inference parameters (weights, confidence, I/O paths) are in `configs/train_config.yaml`.

## API

FastAPI app: `src/api/main.py`

Endpoints:

- `GET /health`
- `POST /predict` (multipart image file)

Response includes class id, class name, confidence, and bbox coordinates.

## Repository Structure

```text
drone-object-detection-pipeline
│
├ README.md
├ requirements.txt
├ .gitignore
│
├ docs
│   ├ architecture.md
│   ├ architecture_ua.md
│   ├ dataset.md
│   └ dataset_ua.md
│
├ configs
│   ├ dataset.yaml
│   └ train_config.yaml
│
├ data
│   ├ raw
│   ├ interim
│   ├ annotations
│   └ processed
│
├ notebooks
│   ├ dataset_analysis.ipynb
│   └ model_pipeline_benchmark.ipynb
│
├ src
│   ├ data
│   │   ├ extract_frames.py
│   │   ├ deduplicate.py
│   │   └ prepare_dataset.py
│   ├ training
│   │   ├ train_yolo.py
│   │   ├ evaluate.py
│   │   └ export_onnx.py
│   ├ inference
│   │   ├ infer_image.py
│   │   ├ infer_video.py
│   │   └ utils.py
│   └ api
│       ├ main.py
│       └ schemas.py
│
├ models
│   ├ weights
│   └ onnx
│
├ examples
│   ├ input
│   └ output
│
└ assets
    └ architecture.png
```

## Example Results

Visual examples (soldier detection):

| Input | Detection Output |
| --- | --- |
| ![Input sample 01](examples/input/sample_01.jpg) | ![Output sample 01](examples/output/sample_01_bbox.jpg) |
| ![Input sample 02](examples/input/sample_02.jpg) | ![Output sample 02](examples/output/sample_02_bbox.jpg) |

## Business Value

This project is positioned as an ML pipeline that solves practical monitoring tasks:

- reduces manual analysis workload for drone footage
- increases throughput for near real-time detection workflows
- provides reproducible model evaluation for safer model updates
- exposes detection through API, enabling integration into existing systems

For CV/HR review, this demonstrates full pipeline ownership: data preparation, annotation workflow, model training, export to deployment format, inference benchmarking, and API serving.

## Challenges Solved

- converting CVAT export into consistent YOLO train/val/test structure
- keeping split reproducible using configurable coefficients and seed
- separating training, inference, and API concerns into clear modules
- exposing the same model logic via scripts and API

## What I Learned

- data quality (dedup + annotation consistency) has direct impact on detector quality
- structured configs simplify reproducible ML experimentation
- deployment-friendly artifacts (ONNX + API) improve project completeness for CV
- small, clear pipelines are easier to defend in interviews than over-engineered stacks

## How To Run

### System Requirements

- Install Python dependencies:

```bash
pip install -r requirements.txt
```

- Install `ffmpeg` and ensure it is available in `PATH` (`ffmpeg -version` should work).
- For large-scale annotation, run a local CVAT container (Docker) and use it as the labeling workspace.

### Data Preparation

1. Put source videos into `data/raw/`.
2. Extract frames and remove near-duplicates:

```bash
python src/data/extract_frames.py
python src/data/deduplicate.py
```

3. Create a CVAT task and annotate the prepared frames.
4. Export annotations in YOLO format. If direct YOLO export is unavailable in your CVAT setup, normalize the exported data with:

```bash
python src/data/prepare_dataset.py
```

5. Run dataset analysis:

```bash
jupyter notebook notebooks/dataset_analysis.ipynb
```

### Model Building

1. Configure parameters in `configs/train_config.yaml` and `configs/dataset.yaml`.
2. Train the `.pt` model:

```bash
python src/training/train_yolo.py
```

3. Evaluate on `test` split:

```bash
python src/training/evaluate.py
```

4. Export to `.onnx`:

```bash
python src/training/export_onnx.py
```

### Benchmark

Use the notebook to compare `yolo12s (onnx.fp32)` and `yolo26n (onnx.fp32)`:

```bash
jupyter notebook notebooks/model_pipeline_benchmark.ipynb
```

### Additional

- Batch inference for images/videos:

```bash
python src/inference/infer_image.py
python src/inference/infer_video.py
```

- Run FastAPI backend:

```bash
python -m uvicorn src.api.main:app --reload
```

