import importlib.util
from pathlib import Path
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v15", ROOT / "generate_v15_class_probes.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def rows(scores):
    names = ("jieba", "jiaza", "mamianmakeng", "gunyin")
    return [
        {"image_id": "x.jpg", "category_name": name, "bbox": [i, 0, 1, 1], "score": score}
        for i, (name, score) in enumerate(zip(names, scores, strict=True))
    ]


def test_class_probe_changes_only_target():
    old = rows((0.1, 0.2, 0.3, 0.4))
    new = rows((0.9, 0.8, 0.7, 0.6))
    output = MODULE.build_class_probe(old, new, "jiaza")
    assert [item["score"] for item in output] == [0.1, 0.8, 0.3, 0.4]
    assert MODULE.identity_digest(output) == MODULE.identity_digest(old)


def test_blend_changes_only_three_target_classes():
    old = rows((0.25, 0.25, 0.25, 0.25))
    new = rows((1.0, 1.0, 1.0, 1.0))
    output = MODULE.build_blend(old, new, 0.5)
    assert [round(item["score"], 6) for item in output] == [0.5, 0.5, 0.5, 0.25]
    assert MODULE.identity_digest(output) == MODULE.identity_digest(old)


def test_submit_zip_has_one_root_json():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "submission.json"
        source.write_text("[]\n", encoding="utf-8")
        archive = root / "submit.zip"
        MODULE.package_submission(source, archive)
        with zipfile.ZipFile(archive) as handle:
            assert handle.namelist() == ["submission.json"]
            assert handle.testzip() is None
