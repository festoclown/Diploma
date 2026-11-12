# Запуск экспериментов

Python 3.11+. Команды выполняются из `code/`.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev]'
```

## Воспроизведение на CPU

```bash
python scripts/run_paper_experiments.py --out reports/paper_check
```

Входные массивы включены в `fixtures/paper_inputs.zip` (около 3 МБ). Загрузка корпусов и моделей не требуется. Команда пересчитывает основную серию и дополнительные проверки, сверяет исходные решения, строит таблицы и графики. Результаты сохраняются в указанной папке. Описание этапов: [карта экспериментов](docs/README.md).

[Лёгкий ноутбук](notebooks/light_experiments.ipynb) позволяет запускать проверки отдельно. Для новых исходных данных используйте [инструкцию запуска](docs/README.md), для текстов установите `'.[text,dev]'`. GPU-вариант требует совместимых CuPy и cuML.

## Обновление рисунков и слайдов

После CPU-запуска:

```bash
python scripts/build_experiment_figures.py \
  --results reports/paper_check/main \
  --diagnostics reports/paper_check/diagnostics \
  --refinement reports/paper_check/refinement \
  --out ../paper/figures

cd ../slides
tectonic main.tex
cd ../paper
tectonic main.tex
```

Для сборки LaTeX нужен Tectonic. Проверка программного пакета: `python -m pytest -q` из `code/`. Данные, модели, результаты и журналы запусков исключены из Git. Методика: [протокол](docs/README.md).
