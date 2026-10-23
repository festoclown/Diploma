# NIP-конфиги

Парсер — [spairet/nip](https://github.com/spairet/nip); синтаксис сверялся с [руководством автора](https://github.com/spairet/nip/tree/main/doc). Файл `.nip` загружается через `nip.load`. Без `@` получается словарь, с `@` — итератор вариантов. Наш загрузчик обрабатывает оба случая.

```text
dataset:
  kind: synthetic
  geometry: @ [separated, overlapping, elongated]
  balance: @ [balanced, imbalanced]
  seed: @ [100, 101]
  n: 2000
  d: 10
```

Это 3×2×2=12 вариантов датасета. Обычный список `budgets: [20, 50]` остаётся одним списком; цикл бюджетов выполняет движок. Оператор `@` нужен только там, где требуется отдельная конфигурация серии/датасета.

## Наследование проекта

`extends: ../base.nip` читается **нашим** `config.py`. Путь относителен текущему файлу. Родитель должен содержать один вариант. Словари объединяются рекурсивно, списки заменяются полностью. Циклы наследования запрещены. В корне итогового конфига обязательны `name`, `phase`, `dataset`, `model`, `compute`, `candidates`, `selection`, `evaluation`.

Не используются NIP-теги для создания модели или запуска функции: разбор конфига должен только получать параметры. Все расчёты запускаются явной командой соответствующей стадии. Для проверки итоговых значений предусмотрен `scripts/show_config.py`.

## Серии

| Файл | Вариантов | Назначение |
|---|---:|---|
| experiments/pilot.nip | 1 | Малый технический пилот, без M4 |
| experiments/development.nip | 18 | Только настройка веса M4 |
| experiments/synthetic.nip | 60 | Основная синтетическая проверка |
| experiments/text.nip | 3 | Основные реальные корпуса |
| experiments/text_db.nip | 3 | Замена силуэта на DB |
| experiments/text_fallback.nip | 3 | Отдельная политика вырожденной разметки |
| experiments/tabular.nip | 2 | Digits и Dry Bean на CPU |
| experiments/tabular_binary.nip | 1 | Двоичный набор на CPU |
| experiments/gpu_pilot.nip | 1 | CUDA-пилот на moons |
| experiments/gpu_development.nip | 18 | Настройка M4 для CUDA-пула |
| experiments/gpu_synthetic.nip | 80 | Синтетика, включая moons |
| experiments/gpu_tabular.nip | 2 | Digits и Dry Bean на CUDA |
| experiments/gpu_tabular_binary.nip | 1 | Двоичный набор на CUDA |
| experiments/gpu_text.nip | 3 | Текстовый перенос на CUDA |

E1 (случайная разметка) и E2 (покрытие) выполняются одним основным конфигом: три `scenarios`. E4 — сохранённая диагностика во всех сериях. Восьми правил × 4 бюджетов × 3 сценариев × 30 повторений достаточно для 2880 строк выбора на корпус при отсутствии пропусков. Это 360 условий разметки, каждое общее для всех методов.

## Параметры

| Блок | Поля и смысл |
|---|---|
| name | Безопасное имя подкаталога `results`; при новом исследовательском варианте задавайте новое имя |
| phase | pilot/development/confirmation/sensitivity; защищает разделение seed и настройку M4 |
| dataset, synthetic | geometry, balance, n, d, seed; число истинных компонент в текущей реализации фиксировано: 8 |
| dataset, text | name, max_documents, subsample_seed, source_revision |
| dataset, tabular | name, max_samples, subsample_seed; Digits, Dry Bean, Breast Cancer |
| model | repo, revision, max_length, batch_size, device; нормировка эмбеддингов всегда включена |
| compute | backend: cpu или cuda, без скрытого fallback |
| candidates | algorithms (kmeans, hdbscan, spectral), k, параметры методов, min_coverage, projection_dim |
| selection.methods | Подмножество M0–M7 |
| selection.budgets | Размеры размеченного L; берутся префиксы общей случайной последовательности |
| selection.scenarios | random/missing_classes/biased |
| selection.repeats | Число повторений получения разметки |
| selection.repeat_seed_start | Начальная основа seed; потоки детерминированно выводятся через SHA256 |
| selection.bootstrap_draws | B, число парных bootstrap-выборок, минимум 2 |
| selection.internal | silhouette или davies_bouldin |
| selection.fixed_window | tau для M5 |
| selection.adaptive_multiplier | c для M6 |
| selection.bias_weight | Относительный вес представленных классов в biased |
| selection.fallback | Явная общая политика вырожденных случаев |
| evaluation | split_seed, test_fraction (по группам), near_optimal_delta, large_loss_threshold |

Для внешних метрик зафиксирован ARI. Подмена его на NMI без пересмотра формул, преобразования шкалы и документации не предусмотрена. Сетка весов M4 закреплена в `experiment.LAMBDA_GRID`.

Вся итоговая конфигурация сохраняется в `results/.../manifest.json`. Хеш путей разделяет варианты. Идентичные данные и пулы используются повторно в сериях чувствительности; полученная разметка также совпадает, если не менялись dataset, сценарий и seed. Смена размера корпуса создаёт другую идентичность датасета и другой поток разметки.

Перебор порогов выполняется функцией `clusterization.switching.sweep` в разделе диагностики ноутбука. Используются готовые разметки и бутстрепные оценки. Новое построение кластеров для каждого порога не требуется.
