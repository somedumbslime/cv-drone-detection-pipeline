# Dataset Notes

## Data Sources

Document where drone images/videos were collected from and license constraints.

## Annotation

- Tool: CVAT
- Output format: YOLO (txt)
- Primary class: `drone`
- Optional classes: `bird`, `helicopter` (if relevant)

## Quality Rules

- Bounding box tightly covers visible object.
- Skip ambiguous far-away objects.
- Keep consistent class naming.

## Split Strategy

Recommended split:

- train: 70%
- val: 20%
- test: 10%

Ensure scene diversity across splits (weather, background, altitude, camera angle).

## Versioning

Track dataset versions with changelog entries whenever labels or classes are updated.
