from types import SimpleNamespace

import numpy as np

from scripts.preflight_semifinal_retrain_v2 import normalized_names
from scripts.select_semifinal_checkpoint_v2 import eligible, summarize_metrics


def test_checkpoint_gate_requires_ap_gain_and_recall_retention():
    baseline = {"crop_dev": {"map50_macro": 0.4, "recall_macro": 0.5}}
    assert eligible({"crop_dev": {"map50_macro": 0.42, "recall_macro": 0.495}}, baseline, 0.01, 0.01)
    assert not eligible({"crop_dev": {"map50_macro": 0.405, "recall_macro": 0.495}}, baseline, 0.01, 0.01)
    assert not eligible({"crop_dev": {"map50_macro": 0.42, "recall_macro": 0.48}}, baseline, 0.01, 0.01)


def test_normalized_names_sorts_numeric_keys():
    assert normalized_names({"names": {"1": "zonglie", "0": "jieba"}}) == ("jieba", "zonglie")


def test_summary_excludes_qilie_and_counts_absent_class_as_zero(tmp_path):
    data = tmp_path / "data.yaml"
    data.write_text("names: {0: jieba, 1: zonglie, 2: qilie}\n", encoding="utf-8")
    box = SimpleNamespace(
        ap_class_index=np.array([0, 2]),
        r=np.array([0.6, 0.9]),
        ap50=np.array([0.4, 0.8]),
        ap=np.array([[0.3, 0.2], [0.7, 0.6]]),
    )
    metrics = SimpleNamespace(names={0: "jieba", 1: "zonglie", 2: "qilie"}, box=box)
    result = summarize_metrics(metrics, data)
    assert result["classes"] == ["jieba", "zonglie"]
    assert result["recall_macro"] == 0.3
    assert result["map50_macro"] == 0.2
