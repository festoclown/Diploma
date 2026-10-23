"""Thin explicit stage dispatch. Importing this module starts no work."""

import argparse
from pathlib import Path

from .config import load_configs, validate


def main(stage: str, project_root: Path) -> None:
    parser = argparse.ArgumentParser(description=f"clusterization: {stage}")
    parser.add_argument("--root", type=Path, default=project_root)
    parser.add_argument("--config", type=Path, required=True, help="NIP file, relative to project root")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    configs = load_configs(config_path)
    for config in configs:
        validate(config)
    if stage == "tune":
        from .tuning import tune
        print(tune(root, configs))
        return
    for index, config in enumerate(configs):
        print(f"[{index + 1}/{len(configs)}] {stage}: {config['dataset']}", flush=True)
        if stage == "download_data":
            from .downloads import download_data
            path = download_data(root, config["dataset"])
        elif stage == "download_models":
            from .downloads import download_model
            path = download_model(root, config["model"]) if config["dataset"]["kind"] == "text" else None
        elif stage == "prepare":
            from .data import prepare
            path = prepare(root, config)
        elif stage == "encode":
            from .embeddings import encode
            path = encode(root, config)
        elif stage == "build":
            from .candidates import build_pool
            path = build_pool(root, config)
        elif stage == "run":
            from .experiment import run
            path = run(root, config)
        else:
            raise ValueError(stage)
        print(path if path else "Not applicable for synthetic data", flush=True)
