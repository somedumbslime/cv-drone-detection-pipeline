# Drone Object Detection Pipeline

[English version](README.md)

End-to-end проєкт з computer vision, сфокусований на детекції дронів: підготовка датасету, workflow розмітки в CVAT, навчання YOLO, експорт у ONNX, скрипти інференсу та FastAPI API.

## Огляд Проєкту

Цей репозиторій побудований як ML engineering pipeline для CV, а не як вебпродукт. Основний фокус: потік даних, навчання моделі, відтворювана оцінка та легке розгортання.

Практичний фокус: автоматизувати детекцію об'єктів на дрон-кадрах, щоб зменшити час ручного перегляду та пришвидшити моніторинг.

## Що Демонструє Цей Проєкт

- підготовку датасету із сирих медіа
- дедуплікацію та базові перевірки якості даних
- конвертацію розмітки CVAT -> YOLO зі split train/val/test
- навчання та оцінку YOLO-моделі
- експорт у ONNX для deployment-інференсу
- FastAPI endpoint для прогнозу по зображенню

## Архітектура

```text
Сирі відео / зображення
        ↓
Витяг кадрів
        ↓
Дедуплікація
        ↓
Workflow розмітки (CVAT)
        ↓
Розбиття датасету
        ↓
Навчання YOLO
        ↓
Експорт моделі (ONNX)
        ↓
Інференс на зображеннях / відео
        ↓
FastAPI API
```

Детальніше: `docs/architecture_ua.md`

## Workflow Датасету Та Розмітки

- Сирі медіа зберігаються в `data/raw/`
- Кадри витягуються у `data/interim/`
- Майже дублікати видаляються до етапу розмітки
- Зображення розмічаються у CVAT
- CVAT-експорт конвертується у YOLO-формат в `data/processed/YOLO/`
- Коефіцієнти split налаштовуються в `configs/train_config.yaml`

Детальніше: `docs/dataset_ua.md`

## Pipeline Навчання

- Конфіг навчання: `configs/train_config.yaml` (секція `train`)
- Конфіг датасету: `configs/dataset.yaml`
- Скрипт навчання: `src/training/train_yolo.py`
- Скрипт оцінки: `src/training/evaluate.py`
- Вихід метрик: `metrics.json`
- Повний notebook для train/export/benchmark: `notebooks/model_pipeline_benchmark.ipynb`
  - GPU-таблиця порівняння: `.pt`, `.onnx`, `.onnx (fp16)`

## Результати Бенчмарку (Повний Test Split, GPU)

Налаштування бенчмарку:

- GPU: NVIDIA GeForce GTX 1080
- Фреймворки: PyTorch + ONNX Runtime (`CUDAExecutionProvider`)
- Дані оцінки: повний `test` split із `configs/dataset.yaml`
- Роздільна здатність входу: `640x640`
- Batch size для бенчмарку: `1` (latency на одному зображенні)

| model_format | path | size_mb | precision | recall | map50 | map50_95 | latency_ms | fps | runtime | input_dtype |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `.onnx` | `models/onnx/model.onnx` | 10.09 | 0.4940 | 0.3491 | 0.3540 | 0.1648 | 9.418 | 106.18 | onnxruntime:CUDAExecutionProvider,CPUExecutionProvider | `numpy.float32` |
| `.pt` | `models/weights/best.pt` | 5.19 | 0.5126 | 0.3467 | 0.3558 | 0.1658 | 13.718 | 72.90 | torch:cuda | `float32` |
| `.onnx (fp16)` | `models/onnx/model.fp16.onnx` | 5.09 | 0.4944 | 0.3529 | 0.3563 | 0.1656 | 13.829 | 72.31 | onnxruntime:CUDAExecutionProvider,CPUExecutionProvider | `numpy.float16` |

Ключові висновки:

- `.onnx` (FP32) найшвидша модель у цьому середовищі: близько `1.46x` FPS відносно `.pt`.
- Просідання якості від `.pt` до `.onnx` невелике (`mAP50`: `-0.0018`, `mAP50-95`: `-0.0010`) і прийнятне для багатьох real-time сценаріїв.
- `.onnx (fp16)` не швидша за `.onnx` FP32 на GTX 1080 (Pascal без Tensor Cores), тому FP16 тут не найкращий варіант.

Рекомендація для edge deployment:

- Основна модель: `.onnx` FP32 (найкращий баланс latency/FPS та якості).
- Резерв/еталон: `.pt` для baseline-порівнянь під час навчання.

## Інференс

- Інференс зображення: `src/inference/infer_image.py`
- Інференс відео: `src/inference/infer_video.py`
- Спільні утиліти: `src/inference/utils.py`

Параметри інференсу (ваги, confidence, шляхи I/O) задаються в `configs/train_config.yaml`.

## API

FastAPI застосунок: `src/api/main.py`

Endpoints:

- `GET /health`
- `POST /predict` (multipart image file)

Відповідь містить class id, class name, confidence і координати bbox.

## Структура Репозиторію

```text
drone-object-detection-pipeline
│
├ README.md
├ README_UA.md
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

## Як Запустити

1. Встановити залежності:

```bash
pip install -r requirements.txt
```

2. Підготувати датасет із CVAT-експорту:

```bash
python src/data/prepare_dataset.py
```

3. Навчити модель:

```bash
python src/training/train_yolo.py
```

4. Оцінити модель на test split:

```bash
python src/training/evaluate.py
```

5. Експортувати в ONNX:

```bash
python src/training/export_onnx.py
```

6. Запустити скрипти інференсу:

```bash
python src/inference/infer_image.py
python src/inference/infer_video.py
```

7. Запустити API:

```bash
python -m uvicorn src.api.main:app --reload
```

## Бізнес-Цінність

Цей проєкт позиціонується як ML pipeline, що вирішує практичні задачі моніторингу:

- зменшує ручне навантаження під час аналізу дрон-відео
- підвищує пропускну здатність у near real-time сценаріях детекції
- забезпечує відтворювану оцінку моделі для безпечних оновлень
- надає API для інтеграції детекції в існуючі системи

Для CV/HR це демонструє повний цикл володіння ML-рішенням: підготовка даних, workflow розмітки, навчання, експорт deployment-артефактів, бенчмарк інференсу та API-сервінг.

## Які Проблеми Вирішено

- конвертація CVAT-експорту в стабільну YOLO-структуру train/val/test
- відтворюваний split через коефіцієнти та seed у конфігу
- розділення data/training/inference/api на зрозумілі модулі
- використання єдиної моделі як у скриптах, так і в API

## Що Я Вивчив

- якість даних (дедуп + консистентна розмітка) напряму впливає на якість детектора
- структуровані конфіги спрощують відтворювані ML-експерименти
- deployment-артефакти (ONNX + API) роблять CV-проєкт завершеним
- невеликі й чіткі pipeline легше захищати на співбесіді, ніж переускладнені стеки

