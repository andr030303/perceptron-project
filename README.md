# Perceptron Project

Учебный проект: однослойный перцептрон для бинарной классификации.

## Возможности

- Класс `Perceptron` с методами `activate`, `predict`, `train` и `score`.
- Генерация синтетических 2D данных с двумя классами.
- Проверка на Iris (два класса, два признака).
- CLI для запуска обучения.
- Визуализация границы решения и количества ошибок по эпохам.
- Тесты `pytest`, проверки `ruff` и `mypy`.
- GitHub Actions CI.

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## Установка

```bash
uv sync --all-groups
```

## Запуск

Синтетические данные (200 объектов, lr=0.1, 50 эпох):

```bash
uv run perceptron --samples 200 --lr 0.1 --epochs 50
```

Или:

```bash
uv run python -m perceptron_project.perceptron --samples 200 --lr 0.1 --epochs 50
```

Запуск на Iris:

```bash
uv run perceptron --dataset iris --epochs 50
```

Не показывать графики:

```bash
uv run perceptron --no-plots
```

Справка по параметрам:

```bash
uv run perceptron --help
```

## Проверки

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict src/
```

## Структура

```text
perceptron-project/
├── .github/workflows/ci.yml
├── src/perceptron_project/
│   ├── __init__.py
│   ├── perceptron.py
│   ├── data.py
│   └── visualization.py
├── tests/
│   ├── test_perceptron.py
│   └── test_data.py
├── pyproject.toml
├── README.md
└── .gitignore
```

## Как работает перцептрон

Для входного вектора `x` вычисляется:

`z = sum(w_i * x_i) + bias`

Затем применяется пороговая активация: `1`, если `z >= 0`, иначе `0`.

При ошибке веса обновляются по правилу:

`w_i += learning_rate * (target - prediction) * x_i`

Bias обновляется аналогично, но без умножения на признак.

## Примечание

`--samples` означает общее количество синтетических объектов, которое делится примерно поровну между классами. `score` в CLI показывает точность на данных, использованных для обучения; это не независимая тестовая оценка.
