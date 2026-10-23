from pathlib import Path
from clusterization.cli import main

if __name__ == "__main__":
    main("download_data", Path(__file__).resolve().parents[1])
