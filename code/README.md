# Запуск экспериментов

Python 3.11+. Команды выполняются из `code/`.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev]'
```

Для текстов установите `'.[text,dev]'`. GPU-конфиги требуют совместимых CuPy и cuML в CUDA-окружении.

## Эксперимент

```bash
CONFIG=configs/experiments/pilot.nip
python scripts/show_config.py --config "$CONFIG"
python scripts/prepare_data.py --config "$CONFIG"
python scripts/build_candidates.py --config "$CONFIG"
python scripts/run_experiment.py --config "$CONFIG"
python scripts/summarize_results.py --results results/pilot --out reports/pilot_first
```

Для другого опыта замените CONFIG и имя каталога `results`. Каталог отчёта должен быть новым. Лёгкие GPU-конфиги: `gpu_pilot`, `gpu_digits_light`, `gpu_digits_db_light`, `gpu_breast_cancer_light`. Для серий с M4 предварительно выполните `prepare_data.py`, `build_candidates.py` и `tune_mixture.py` на соответствующем CPU/GPU development-конфиге.

Dry Bean требует `download_data.py`. Тексты требуют загрузки данных и модели, затем `encode_texts.py`. Полная последовательность: [quickstart](docs/quickstart.md).

## Ноутбук и диагностика

[Ноутбук экспериментов](notebooks/research_experiments_colab.ipynb) содержит постановку, запуск и графики.

```bash
python scripts/analyze_switches.py --results results --out reports/switching_first
python scripts/plot_switches.py --report reports/switching_first
python -m pytest -q
```

Решения сохраняются в `results/`, CSV и PNG — в `reports/`. Больший проверочный ARI и меньшая потеря regret предпочтительны. Повторы разметки одного набора не заменяют независимые задачи. Определения: [протокол](docs/protocol.md). Параметры: [конфиги](docs/configs.md).
