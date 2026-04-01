# Drone Object Detection Pipeline

[English version](README.md)

<p align="center">
  <img src="assets/raptor-ai.gif" alt="Raptor AI"/>
</p>

End-to-end проєкт з computer vision, сфокусований на детекції дронів: підготовка датасету, workflow розмітки в CVAT, навчання YOLO, експорт у ONNX, скрипти інференсу та FastAPI API.

<p align="center">
  <img src="assets/soldiers.gif" alt="Soldier detection demo" />
</p>

## Зміст

- [Огляд Проєкту](#огляд-проєкту)
- [Що Демонструє Цей Проєкт](#що-демонструє-цей-проєкт)
- [Архітектура](#архітектура)
- [Workflow Датасету Та Розмітки](#workflow-датасету-та-розмітки)
- [Pipeline Навчання](#pipeline-навчання)
- [Результати Бенчмарку](#результати-бенчмарку)
- [Інференс](#інференс)
- [API](#api)
- [Структура Репозиторію](#структура-репозиторію)
- [Приклади Результатів](#приклади-результатів)
- [Бізнес-Цінність](#бізнес-цінність)
- [Які Проблеми Вирішено](#які-проблеми-вирішено)
- [Що Я Вивчив](#що-я-вивчив)
- [Як Запустити](#як-запустити)

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
- Дублікати видаляються до етапу розмітки
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
  - зведена таблиця бенчмарку між версіями датасету та станом leakage

## Результати Бенчмарку

Налаштування бенчмарку:

- Дані оцінки: повний `test` split із `configs/dataset.yaml`
- Роздільна здатність входу: `640x640`
- Batch size для бенчмарку: `1` (latency на одному зображенні)
- Середовища запуску відрізняються між експериментами (GPU і CPU); для коректного контексту дивись колонку `runtime`.

| experiment | train_frames | leakage_status | model | size_mb | precision | recall | map50 | map50_95 | latency_ms | fps | runtime | input_dtype |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `5800_frames_legacy` | 5800 | ймовірний leakage | `yolo12s (onnx.fp32)` | 10.09 | 0.4940 | 0.3491 | 0.3540 | 0.1648 | 9.332 | 107.16 | `onnxruntime:CUDAExecutionProvider` | `numpy.float32` |
| `5800_frames_legacy` | 5800 | ймовірний leakage | `yolo26n (onnx.fp32)` | 9.35 | 0.7426 | 0.5385 | 0.6079 | 0.3226 | 8.936 | 111.91 | `onnxruntime:CUDAExecutionProvider` | `numpy.float32` |
| `7500_frames_old_split` | 7500 | до виправлення leakage | `yolo26n (onnx.fp32)` | 9.35 | 0.8203 | 0.6171 | 0.7387 | 0.4440 | 26.860 | 37.23 | `onnxruntime:CPUExecutionProvider` | `numpy.float32` |
| `7500_frames_old_split` | 7500 | до виправлення leakage | `yolo26n (onnx.fp16)` | 4.74 | 0.8325 | 0.6113 | 0.7379 | 0.4438 | 27.370 | 36.54 | `onnxruntime:CPUExecutionProvider` | `numpy.float16` |
| `7500_frames_old_split` | 7500 | до виправлення leakage | `yolo26n (.pt)` | 5.15 | 0.7387 | 0.5397 | 0.6143 | 0.3373 | 96.408 | 10.37 | `torch:cpu` | `float32` |
| `7500_frames_fixed_split` | 7500 | після виправлення leakage | `yolo26n (onnx.fp16)` | 4.74 | 0.6364 | 0.4599 | 0.4977 | 0.2603 | 26.886 | 37.19 | `onnxruntime:CPUExecutionProvider` | `numpy.float16` |
| `7500_frames_fixed_split` | 7500 | після виправлення leakage | `yolo26n (onnx.fp32)` | 9.35 | 0.6388 | 0.4565 | 0.4972 | 0.2611 | 28.935 | 34.56 | `onnxruntime:CPUExecutionProvider` | `numpy.float32` |
| `7500_frames_fixed_split` | 7500 | після виправлення leakage | `yolo26n (.pt)` | 5.15 | 0.6336 | 0.4741 | 0.5056 | 0.2617 | 108.369 | 9.23 | `torch:cpu` | `float32` |

Ключові висновки:

- Leakage у split суттєво завищував якість. На тому самому датасеті 7500 кадрів `yolo26n (onnx.fp32)` знизилась із `map50 0.7387` до `0.4972` після leakage-safe split.
- Після виправлення leakage метрики між форматами стали значно ближчими та реалістичнішими.
- Для CPU edge-сценарію `yolo26n (onnx.fp16)` дає найкращий компроміс швидкість/розмір (`37.19 FPS`, `4.74 MB`) при невеликій втраті якості відносно `.pt`.
- Показники runtime між GPU та CPU експериментами не варто порівнювати напряму без однакового заліза та провайдерів.

Рекомендація для edge deployment:

- Базовий варіант для CPU edge: `yolo26n (onnx.fp16)`.
- Якщо потрібна максимальна якість і прийнятний нижчий FPS: `yolo26n (.pt)`.
- Якщо доступний GPU ONNX провайдер (`CUDAExecutionProvider`), перед фінальним вибором моделі треба повторити бенчмарк у тому самому середовищі.

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

## Приклади Результатів

Візуальні приклади (детекція солдатів):

| Вхід | Результат детекції |
| --- | --- |
| ![Input sample 01](examples/input/sample_01.jpg) | ![Output sample 01](examples/output/sample_01_bbox.jpg) |
| ![Input sample 02](examples/input/sample_02.jpg) | ![Output sample 02](examples/output/sample_02_bbox.jpg) |

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

## Як Запустити

### Системні Вимоги

- Встановити Python-залежності:

```bash
pip install -r requirements.txt
```

- Встановити `ffmpeg` і переконатися, що він доступний у `PATH` (команда `ffmpeg -version` має працювати).
- Для розмітки великих обсягів даних запустити локальний контейнер CVAT (Docker) і використовувати його як середовище анотації.

### Підготовка Даних

1. Завантажити сирі відео у `data/raw/`.
2. Витягнути кадри та прибрати майже дублікати:

```bash
python src/data/extract_frames.py
python src/data/deduplicate.py
```

3. Створити Task у CVAT і виконати розмітку кадрів.
4. Експортувати розмітку у YOLO-формат. Якщо у вашому CVAT прямий YOLO-експорт недоступний, нормалізувати структуру датасету скриптом:

```bash
python src/data/prepare_dataset.py
```

5. Виконати аналіз датасету:

```bash
jupyter notebook notebooks/dataset_analysis.ipynb
```

### Створення Моделі

1. Налаштувати параметри у `configs/train_config.yaml` та `configs/dataset.yaml`.
2. Навчити `.pt` модель:

```bash
python src/training/train_yolo.py
```

3. Оцінити модель на `test` split:

```bash
python src/training/evaluate.py
```

4. Експортувати модель у `.onnx`:

```bash
python src/training/export_onnx.py
```

### Бенчмарк

Використати notebook для відтворення зведеного бенчмарку (версії датасету, стан leakage та формати моделі):

```bash
jupyter notebook notebooks/model_pipeline_benchmark.ipynb
```

### Додатково

- Пакетний інференс зображень/відео:

```bash
python src/inference/infer_image.py
python src/inference/infer_video.py
```

- Запуск FastAPI бекенду:

```bash
python -m uvicorn src.api.main:app --reload
```
