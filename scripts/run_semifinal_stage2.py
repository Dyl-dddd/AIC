"""Second-stage 4090 inference only: lower confidence floor and 1280px ablation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_semifinal_stage1 import main

if __name__ == "__main__":
    main(stage=2)
