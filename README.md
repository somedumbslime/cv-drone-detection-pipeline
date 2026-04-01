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
- [Benchmark Results](#benchmark-results)
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
  - consolidated benchmark table across dataset versions and leakage settings

## Benchmark Results

Benchmark setup:

- Evaluation data: full `test` split from `configs/dataset.yaml`
- Input resolution: `640x640`
- Benchmark batch size: `1` (single-image latency measurement)
- Runtime environment differs across experiments (GPU and CPU); use `runtime` column for direct context.

| experiment | train_frames | leakage_status | model | size_mb | precision | recall | map50 | map50_95 | latency_ms | fps | runtime | input_dtype |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `5800_frames_legacy` | 5800 | likely leakage | `yolo12s (onnx.fp32)` | 10.09 | 0.4940 | 0.3491 | 0.3540 | 0.1648 | 9.332 | 107.16 | `onnxruntime:CUDAExecutionProvider` | `numpy.float32` |
| `5800_frames_legacy` | 5800 | likely leakage | `yolo26n (onnx.fp32)` | 9.35 | 0.7426 | 0.5385 | 0.6079 | 0.3226 | 8.936 | 111.91 | `onnxruntime:CUDAExecutionProvider` | `numpy.float32` |
| `7500_frames_old_split` | 7500 | before leakage fix | `yolo26n (onnx.fp32)` | 9.35 | 0.8203 | 0.6171 | 0.7387 | 0.4440 | 26.860 | 37.23 | `onnxruntime:CPUExecutionProvider` | `numpy.float32` |
| `7500_frames_old_split` | 7500 | before leakage fix | `yolo26n (onnx.fp16)` | 4.74 | 0.8325 | 0.6113 | 0.7379 | 0.4438 | 27.370 | 36.54 | `onnxruntime:CPUExecutionProvider` | `numpy.float16` |
| `7500_frames_old_split` | 7500 | before leakage fix | `yolo26n (.pt)` | 5.15 | 0.7387 | 0.5397 | 0.6143 | 0.3373 | 96.408 | 10.37 | `torch:cpu` | `float32` |
| `7500_frames_fixed_split` | 7500 | after leakage fix | `yolo26n (onnx.fp16)` | 4.74 | 0.6364 | 0.4599 | 0.4977 | 0.2603 | 26.886 | 37.19 | `onnxruntime:CPUExecutionProvider` | `numpy.float16` |
| `7500_frames_fixed_split` | 7500 | after leakage fix | `yolo26n (onnx.fp32)` | 9.35 | 0.6388 | 0.4565 | 0.4972 | 0.2611 | 28.935 | 34.56 | `onnxruntime:CPUExecutionProvider` | `numpy.float32` |
| `7500_frames_fixed_split` | 7500 | after leakage fix | `yolo26n (.pt)` | 5.15 | 0.6336 | 0.4741 | 0.5056 | 0.2617 | 108.369 | 9.23 | `torch:cpu` | `float32` |

Key takeaways:

- Split leakage had a major impact on quality metrics. On the same 7500-frame dataset, `yolo26n (onnx.fp32)` dropped from `map50 0.7387` to `0.4972` after leakage-safe split.
- After fixing leakage, quality metrics across formats became much closer and more realistic.
- On CPU edge setup, `yolo26n (onnx.fp16)` is the best speed/size compromise (`37.19 FPS`, `4.74 MB`) with minor quality gap vs `.pt`.
- Runtime numbers across GPU and CPU experiments should not be compared directly without matching providers/hardware.

Recommended deployment choice for edge inference:

- Default CPU edge model: `yolo26n (onnx.fp16)`.
- If maximum quality is required and lower FPS is acceptable: `yolo26n (.pt)`.
- If GPU ONNX provider is available (`CUDAExecutionProvider`), re-run benchmark in the same environment before final model freeze.

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

Use the notebook to reproduce the consolidated benchmark table (dataset versions, leakage status, and model formats):

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

