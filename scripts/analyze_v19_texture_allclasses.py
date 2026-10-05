"""Run the V19 texture-reranker audit across all eight permitted classes."""

from pathlib import Path

from scripts import analyze_v19_texture_reranker as experiment


experiment.TARGETS = (
    "jieba", "zonglie", "jiaza", "yiwuyaru", "huashang", "mamianmakeng",
    "yanghuatiepi", "gunyin",
)
experiment.OUT = Path(experiment.ROOT / "runs/semifinal/v19_texture_allclasses_20261002/results.json")


if __name__ == "__main__":
    experiment.main()
