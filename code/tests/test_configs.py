from pathlib import Path
from clusterization.config import load_configs, validate

ROOT = Path(__file__).resolve().parents[1]


def test_nip_iterators_expand_expected_dataset_grids():
    for name, count in [("pilot", 1), ("development", 18), ("synthetic", 60), ("text", 3),
                        ("text_db", 3), ("text_fallback", 3),
                        ("tabular", 2), ("tabular_binary", 1),
                        ("gpu_pilot", 1), ("gpu_digits_light", 1), ("gpu_digits_db_light", 1),
                        ("gpu_breast_cancer_light", 1), ("gpu_development", 18),
                        ("gpu_synthetic", 80), ("gpu_tabular", 2),
                        ("gpu_tabular_binary", 1), ("gpu_text", 3)]:
        configs = load_configs(ROOT / "configs" / "experiments" / f"{name}.nip")
        assert len(configs) == count
        for config in configs:
            validate(config)
