# Synthetic 2D Point Classification

## Описание

Этот проект показывает полный учебный ML-пайплайн на синтетических 2D-данных.
Он генерирует два класса точек внутри кругов, обучает `GradientBoostingClassifier`,
считает метрики, сохраняет модель и строит визуализации.

Проект намеренно остается простым: здесь нет бизнес-данных, внешнего API или
искусственно раздутой инфраструктуры.

## Что демонстрирует проект

- генерацию синтетических данных;
- бинарную классификацию;
- train/test split;
- `GradientBoostingClassifier`;
- оценку качества;
- сохранение модели;
- визуализацию результатов;
- тесты.

## Структура проекта

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── src/
│   └── point_classifier/
│       ├── __init__.py
│       ├── data.py
│       ├── metrics.py
│       ├── model.py
│       ├── pipeline.py
│       └── visualization.py
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_pipeline.py
├── artifacts/
│   ├── models/
│   └── plots/
└── reports/
```

## Установка

Требуется Python 3.10+.

```bash
python -m pip install -r requirements.txt
```

## Запуск

Базовый запуск:

```bash
python main.py
```

После запуска будут созданы:

- `artifacts/models/gradient_boosting_model.joblib`
- `artifacts/plots/*.png`
- `reports/metrics.json`

## Пример CLI-команды

```bash
python main.py --n-samples 10000 --radius 1.0 --shift 1.5 --test-size 0.25 --random-state 42
```

Дополнительно можно переопределить пути вывода:

```bash
python main.py --model-path artifacts/models/custom_model.joblib --plots-dir artifacts/plots --metrics-path reports/metrics.json
```

## Метрики

Во время запуска проект считает и выводит:

- accuracy;
- precision;
- recall;
- f1-score;
- confusion matrix;
- classification report.

Основные результаты сохраняются в `reports/metrics.json`.

## Визуализации

Проект сохраняет несколько PNG-файлов в `artifacts/plots/`:

- `synthetic_dataset.png` — исходные данные;
- `train_test_split.png` — тренировочная и тестовая выборки;
- `test_predictions.png` — предсказания модели на тесте;
- `decision_boundary.png` — граница решений модели.
- `process_overview.png` — обзор всей цепочки: данные, split, предсказания, decision boundary, confusion matrix и итоговые метрики.

Если нужен один файл, который объясняет весь запуск целиком, открывайте именно `process_overview.png`.

## Тесты

Запуск тестов:

```bash
pytest
```

Тесты проверяют:

- корректность генерации данных;
- бинарные метки классов `{0, 1}`;
- форму признаков;
- обучение модели;
- создание `metrics.json`;
- сохранение модели;
- сохранение графиков.

## Ограничения проекта

Это учебный проект на синтетических данных.
Он не решает прикладную бизнес-задачу, не использует реальные признаки и не
претендует на production-ready ML-систему.

## Возможные улучшения

- подбор гиперпараметров;
- сравнение нескольких моделей;
- cross-validation;
- MLflow;
- Docker;
- GitHub Actions.
