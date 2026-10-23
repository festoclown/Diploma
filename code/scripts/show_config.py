"""Display expanded NIP variants without executing pipeline stages."""

import argparse
import json
from pathlib import Path
from clusterization.config import load_configs, validate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    configs = load_configs(args.config)
    for config in configs:
        validate(config)
    print(json.dumps(configs, indent=2, ensure_ascii=False))
    print(f"Total variants: {len(configs)}")
