# Источники и привязка к реализации

Ссылки на программные источники сверялись 12 октября 2026 года. Это техническая библиография проекта; она не заменяет обзор новизны исследовательской темы.

| Источник | Что используется |
|---|---|
| [spairet/nip](https://github.com/spairet/nip) и [руководство](https://github.com/spairet/nip/tree/main/doc) | `nip.load`, файлы .nip и оператор сетки @ |
| [nip-config 0.12.4](https://pypi.org/project/nip-config/0.12.4/) | Имя и версия устанавливаемого пакета |
| [AG News на Hugging Face](https://huggingface.co/datasets/fancyzhx/ag_news) | Конкретный источник данных text/label |
| [Banking77, данные авторов](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data) | Официальные train.csv/test.csv, category |
| [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset) | Загрузчик корпуса и удаление служебных частей |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Выбранный эмбеддер |
| [SentenceTransformer API](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html) | Локальная загрузка, encode, нормировка |
| [Hugging Face snapshot download](https://huggingface.co/docs/huggingface_hub/guides/download) | Скачивание версии модели в отдельной стадии |
| [cuML HDBSCAN](https://docs.nvidia.com/cuml/latest/api/generated/cuml.cluster.hdbscan.HDBSCAN/) | GPU кластеризация по плотности и метка шума |
| [cuML KMeans](https://docs.rapids.ai/api/cuml/stable/api/generated/cuml.cluster.KMeans/) | GPU центроидная кластеризация |
| [cuML SpectralClustering](https://docs.rapids.ai/api/cuml/stable/api/generated/cuml.cluster.SpectralClustering/) | GPU спектральная кластеризация |
| [RAPIDS install](https://docs.rapids.ai/install/) | Совместимая CUDA-среда |
| [UCI Dry Bean](https://archive.ics.uci.edu/dataset/602/dry%2B) | Быстрый табличный набор с классами |
| [scikit-learn Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) | Небольшой реальный числовой набор |

Математические основания: ARI — поправка индекса Rand на случайное совпадение; silhouette — сравнение внутрикластерного расстояния с ближайшим другим кластером; Davies–Bouldin — отношение разброса к разделению кластеров. Реализация внешних проверочных метрик использует sklearn, а ускоренный ARI с кратностями сопоставляется с sklearn в подготовленных тестах.

Исследовательские предшественники и формулировка гипотез сохранены в Obsidian в заметке «Выбор кластерного разбиения по ограниченной внешней информации — план экспериментов». Сам факт наличия M6 в коде не доказывает его научную новизну.
