# Порядок ручного запуска

Команды выполняются из `Diploma/code/`. Лёгкие пилоты уже выполнены; эти инструкции позволяют повторить стадии либо запустить новые серии. Расширенные серии не считаются проведёнными только из-за наличия конфигураций.

## 1. Окружение

Для CPU нужен Python 3.11+ и зависимости проекта:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev]'
```

Для текстов добавьте `python -m pip install '.[text,dev]'`. Для CUDA используйте Linux/WSL с совместимыми драйвером, CUDA, CuPy и cuML. Практический способ — актуальный контейнер RAPIDS; после его запуска установите этот проект внутри контейнера. Не устанавливайте произвольную сборку `cuml` или `cupy` поверх несовместимого CUDA окружения. Версию среды сохраните после установки: `python -m pip freeze > environment.lock.txt`. macOS не служит средой запуска CUDA.

Проверка кода и конфигурации перед длинными сериями:

```bash
python scripts/show_config.py --config configs/experiments/gpu_pilot.nip
python -m pytest
```

Тесты проверяют реализацию на CPU и не запускают GPU-серии. Совместимость cuML проверяется пилотом на целевой машине.

## 2. Быстрый старт без текста

При доступной A100 начните с `gpu_pilot.nip`; на CPU — с `pilot.nip`. Это технические прогоны без M4 и без выводов для статьи.

```bash
python scripts/prepare_data.py --config configs/experiments/gpu_pilot.nip
python scripts/build_candidates.py --config configs/experiments/gpu_pilot.nip
python scripts/run_experiment.py --config configs/experiments/gpu_pilot.nip
python scripts/summarize_results.py --results results/gpu_pilot --out reports/gpu_pilot
```

Digits встроен в scikit-learn, отдельная загрузка не нужна. Для Dry Bean явная команда использует сеть и сохраняет исходный архивный hash:

```bash
python scripts/download_data.py --config configs/experiments/gpu_tabular.nip
python scripts/prepare_data.py --config configs/experiments/gpu_tabular.nip
python scripts/build_candidates.py --config configs/experiments/gpu_tabular.nip
```

Первая команда в смешанной сетке ничего не скачивает для Digits. Основная табличная серия с M4 запускается после заморозки веса; см. следующий раздел.

## 3. Заморозить M4 и провести основную проверку

CPU и CUDA имеют раздельные пулы и раздельные замороженные веса. Для CUDA:

```bash
python scripts/prepare_data.py --config configs/experiments/gpu_development.nip
python scripts/build_candidates.py --config configs/experiments/gpu_development.nip
python scripts/tune_mixture.py --config configs/experiments/gpu_development.nip
python scripts/prepare_data.py --config configs/experiments/gpu_synthetic.nip
python scripts/build_candidates.py --config configs/experiments/gpu_synthetic.nip
python scripts/run_experiment.py --config configs/experiments/gpu_synthetic.nip
python scripts/run_experiment.py --config configs/experiments/gpu_tabular.nip
```

GPU development использует 18 гауссовых наборов и создаёт `artifacts/frozen_mixture_cuda.json`. GPU confirmation включает 80 синтетических наборов, в том числе невыпуклые moons. Пул табличных данных должен быть построен согласно разделу 2. Для CPU соответствующие конфиги: `development.nip`, `synthetic.nip`, `tabular.nip`; файл веса — `artifacts/frozen_mixture.json`. На Breast Cancer используйте `gpu_tabular_binary.nip` или `tabular_binary.nip`: там малые бюджеты и нет M4.

## 4. Текст как проверка переноса

```bash
python scripts/download_data.py --config configs/experiments/gpu_text.nip
python scripts/download_models.py --config configs/experiments/gpu_text.nip
python scripts/prepare_data.py --config configs/experiments/gpu_text.nip
python scripts/encode_texts.py --config configs/experiments/gpu_text.nip
python scripts/build_candidates.py --config configs/experiments/gpu_text.nip
python scripts/run_experiment.py --config configs/experiments/gpu_text.nip
```

Первые две команды используют сеть; остальные берут локальные данные. `model.device: cuda` переносит кодирование текстов на GPU, `compute.backend: cuda` переносит PCA, построение кластеров, расстояния, внутренние оценки и пакетный ARI. Менять только один из параметров значит получить смешанную вычислительную схему. CPU-версия — `text.nip`.

Серии чувствительности `text_db.nip`, `text_fallback.nip` сейчас предназначены для CPU-пулов. Для GPU-копий нужно создать собственные NIP-файлы с тем же блоком `compute`, `model` и `candidates`, что у `gpu_text.nip`, и уникальными `name`. Не сравнивайте результаты разных пулов как парные повторы одного эксперимента.

## 5. Время и сбои

Точное время на A100 пока не измерено. После пилота замерьте подготовку одного пула и одно условие выбора, затем оцените полную серию с учётом числа наборов, сценариев и повторов. Узкое место может сместиться к CPU-подготовке данных, загрузке или записи. Одна плотная матрица `float32` для 4 000 объектов занимает около 64 MB; промежуточные массивы увеличивают пик памяти. Спектральный метод и HDBSCAN могут быть существенно дороже KMeans. Планирование приведено в [числовых данных и GPU](datasets_gpu.md).

Завершённые кеши проверяются и переиспользуются. Неполный каталог с `status: running` автоматически не удаляется: изучите исключение, сохраните материалы для отладки и повторите стадию в отдельном каталоге. Не запускайте две записи в один и тот же артефакт параллельно.
