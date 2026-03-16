# Dataset

## Data Sources

The project uses drone-related visual media (videos/images) that are converted into labeled object detection samples.

## Dataset Stages

- `data/raw/`: source media before processing
- `data/interim/`: extracted and cleaned frames before final packaging
- `data/annotations/`: annotation artifacts exported from CVAT
- `data/processed/`: final YOLO-ready dataset splits

## Annotation Workflow

1. Extract candidate frames.
2. Remove near-duplicate images.
3. Upload selected frames to CVAT.
4. Annotate bounding boxes.
5. Export labels in YOLO-compatible format.
6. Convert export to standard YOLO folder layout and split train/val/test.

## Classes

Class names are defined in `configs/dataset.yaml` and synchronized by `src/data/prepare_dataset.py` based on CVAT export metadata.

## Split Strategy

Split ratios are configured in `configs/train_config.yaml` under `data_preparation.split`.

Default:

- train: 70%
- val: 20%
- test: 10%

The split is reproducible using a fixed random seed.

## Quality Controls

- verify image-label pairing during dataset preparation
- generate empty label files only when annotation file is missing
- ensure YOLO folder structure integrity (`images/*`, `labels/*`)

## Notes For Interviews

Be ready to explain:

- why deduplication improves generalization
- why train/val/test split is needed
- how CVAT exports are transformed to training-ready data
- known dataset limitations and class imbalance risks
