from pathlib import Path
from clusterization.cli import main

if __name__ == "__main__":
    main("encode", Path(__file__).resolve().parents[1])
