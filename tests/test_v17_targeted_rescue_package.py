from pathlib import Path

import yaml

from scripts.run_v17_targeted_rescue_pipeline import focus_label, prepare_focus_dataset
from steel_defect.classes import CLASS_NAMES


def test_focus_label(tmp_path: Path):
    label = tmp_path / "sample.txt"
    label.write_text("0 0.5 0.5 0.01 0.01\n", encoding="utf-8")
    assert focus_label(label)
    label.write_text("1 0.5 0.5 0.01 0.25\n", encoding="utf-8")
    assert focus_label(label)
    label.write_text("3 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    assert focus_label(label)
    label.write_text("8 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    assert not focus_label(label)


def test_prepare_focus_dataset_preserves_source(tmp_path: Path):
    source = tmp_path / "data/semifinal_yolo_balanced_v3"
    for subfolder in ("images/train", "labels/train", "images/dev", "labels/dev", "metadata"):
        (source / subfolder).mkdir(parents=True)
    yaml_payload = {"path": source.as_posix(), "train": "images/train", "val": "images/dev",
                    "nc": 9, "names": {index: name for index, name in enumerate(CLASS_NAMES)}}
    (source / "steel_defect.yaml").write_text(yaml.safe_dump(yaml_payload), encoding="utf-8")
    (source / "metadata/summary.json").write_text('{"global_repeat":3}', encoding="utf-8")
    for index in range(100):
        (source / "images/train" / f"sample{index:03d}.jpg").write_bytes(b"x")
        label = "0 0.5 0.5 0.01 0.01\n" if index < 10 else "8 0.5 0.5 0.3 0.3\n"
        (source / "labels/train" / f"sample{index:03d}.txt").write_text(label, encoding="utf-8")
    (source / "images/dev/dev.jpg").write_bytes(b"x")
    (source / "labels/dev/dev.txt").write_text("8 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    result = prepare_focus_dataset(tmp_path)
    assert result["base_images"] == 100
    assert result["focus_duplicates"] == 10
    assert result["dev_images"] == 1
    target = tmp_path / "data/semifinal_yolo_targeted_v17"
    assert (target / "images/train/sample000__focus_v17.jpg").is_file()
    assert (target / "labels/train/sample000__focus_v17.txt").is_file()
    assert (source / "images/train/sample000.jpg").is_file()
    assert prepare_focus_dataset(tmp_path) == result
