from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steel_defect.governance import connected_groups, check_evaluation_scope, validate_split_manifest, frozen_records
from steel_defect.metrics import evaluate_predictions
from steel_defect.voc import VocRecord
from scripts.match_update_epochs import estimate_updates


def entry(key, path, sha):
    return {"id": key, "path": path, "sha256": sha, "pixel_sha256": sha}


def test_connected_source_and_content_chain():
    entries = [entry("a", "C123_V1_F1.jpg", "1"), entry("b", "C123_V2_F2.jpg", "2"),
               entry("c", "copy.jpg", "2"), entry("test:d", "copy2.jpg", "1")]
    groups, _ = connected_groups(entries)
    assert len(set(groups.values())) == 1


def test_near_hash_is_isolation_relation():
    import base64
    import numpy as np
    thumbnail = base64.b64encode(np.arange(256, dtype=np.uint8).tobytes()).decode()
    entries = [dict(entry("a", "a.jpg", "a"), width=8, height=8, dhash="0000000000000000"),
               dict(entry("b", "b.jpg", "b"), width=8, height=8, dhash="0000000000000001")]
    entries[0]["thumbnail"] = entries[1]["thumbnail"] = thumbnail
    groups, near = connected_groups(entries)
    assert groups["a"] == groups["b"] and len(near) == 1


def test_flat_texture_dhash_does_not_merge_unrelated_images():
    entries = [dict(entry("a", "a.jpg", "a"), width=8, height=8, dhash="0"),
               dict(entry("b", "b.jpg", "b"), width=8, height=8, dhash="0")]
    groups, near = connected_groups(entries)
    assert groups["a"] != groups["b"] and not near


def test_test_prefix_alone_does_not_claim_duplicate():
    entries = [dict(entry("a", "coil-Raw01-f1.jpg", "a"), kind="train"),
               dict(entry("b", "coil-Raw02-f2.jpg", "b"), kind="test")]
    groups, _ = connected_groups(entries)
    assert groups["a"] != groups["b"]


def test_final_scope_guards():
    manifest = {"splits": {"final": ["a"]}, "roles": {"final": "final_holdout"}}
    with pytest.raises(ValueError, match="scope"):
        check_evaluation_scope(manifest, "final", [], tune=False, limit=0)
    with pytest.raises(ValueError, match="forbids"):
        check_evaluation_scope(manifest, "final", ["a"], tune=True, limit=0)
    with pytest.raises(ValueError, match="signature"):
        check_evaluation_scope(manifest, "final", ["a"], tune=False, limit=0)
    check_evaluation_scope(manifest, "final", ["a"], tune=False, limit=0,
                           freeze={"weights_sha256": "w", "split_sha256": "s", "evaluation_policy": {}},
                           weights_sha="w", manifest_sha="s", policy={})


def test_group_leakage_rejected():
    with pytest.raises(ValueError, match="leakage"):
        validate_split_manifest({"splits": {"train": ["a"], "dev": ["b"]},
                                 "groups": {"train": ["g"], "dev": ["g"]}})


def test_missing_source_rejected(tmp_path):
    manifest = {"splits": {"train": ["a.jpg"]}, "groups": {"train": ["g"]}}
    with pytest.raises(ValueError, match="missing source"):
        frozen_records([], tmp_path, manifest)


def pred(image="a", cls="defect", box=None, score=.9):
    return {"image_id": image, "category_name": cls, "bbox": box or [0, 0, 10, 10], "score": score}


def test_gold_perfect_duplicate_wrong_class_and_empty_images():
    gt = {"a": {"defect": [[0, 0, 10, 10]]}, "empty": {}}
    perfect = evaluate_predictions([pred()], gt, ["defect", "absent"])
    assert perfect["summary"]["map50_macro"] == 1
    output = evaluate_predictions([pred(), pred(score=.8), pred(image="empty", cls="absent")], gt, ["defect", "absent"])
    assert output["summary"]["false_positives_per_image"] == 1
    assert output["per_class"]["defect"]["true_positives"] == 1
    wrong = evaluate_predictions([pred(cls="absent")], gt, ["defect", "absent"])
    assert wrong["summary"]["map50_macro"] == 0
    assert wrong["summary"]["false_positives_per_image"] == .5
    empty = evaluate_predictions([pred(image="empty")], {"empty": {}}, ["defect"])
    assert empty["summary"]["false_positives_per_image"] == 1


@pytest.mark.parametrize("prediction", [pred(image="unknown"), pred(cls="unknown"),
                         pred(score=float("nan")), pred(box=[0, 0, float("inf"), 10]),
                         pred(box=[0, 0, 30, 10]), pred(box=[10, 0, 0, 10])])
def test_invalid_predictions_fail_closed(prediction):
    with pytest.raises(ValueError):
        evaluate_predictions([prediction], {"a": {}}, ["defect"], image_sizes={"a": (20, 20)})


def test_update_estimator_accounts_for_accumulation():
    results = [estimate_updates(640, b, 2, nbs=64, warmup_epochs=0) for b in (1, 2, 4)]
    assert [r["estimated_optimizer_calls"] for r in results] == [20, 20, 20]
    assert results[0]["microbatches"] == 1280
    assert estimate_updates(40, 2, 25, nbs=2, warmup_epochs=0)["estimated_optimizer_calls"] == 500


def test_audit_does_not_count_final_validation_as_epoch(tmp_path):
    from types import SimpleNamespace
    from steel_defect.training_audit import OptimizerAudit
    audit = OptimizerAudit()
    audit.path = tmp_path / "steps.jsonl"
    trainer = SimpleNamespace(epoch=0, accumulate=1, batch_size=2,
                              args=SimpleNamespace(nbs=2), optimizer=SimpleNamespace(param_groups=[{"lr": .001}]))
    audit.epoch(trainer)
    audit.epoch(trainer)
    assert audit.epochs == 1


def test_queue_dry_run_is_read_only(tmp_path, monkeypatch):
    from scripts import run_experiments
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text("experiments:\n  - name: demo\n    data: not_present.yaml\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_experiments.py", "--matrix", str(matrix), "--dry-run"])
    run_experiments.main()
    assert not matrix.with_suffix(".state.json").exists()


def test_best_checkpoint_alone_is_not_complete(tmp_path):
    from scripts.run_experiments import completed_run
    folder = tmp_path / "demo" / "weights"
    folder.mkdir(parents=True)
    (folder / "best.pt").write_bytes(b"partial checkpoint")
    assert not completed_run({"project": str(tmp_path), "name": "demo", "data": "not_present.yaml"})


def test_model_initialization_failure_is_recorded(tmp_path, monkeypatch):
    import json
    from scripts import train
    config = {**train.DEFAULTS, "data": "dummy", "device": "cpu", "project": str(tmp_path), "name": "broken"}
    monkeypatch.setattr(train, "resolve_config", lambda: config)
    monkeypatch.setattr(train, "platform_manifest", lambda c: {"config": c})
    monkeypatch.setattr(train, "dataset_signature", lambda p: {"sha256": "dummy"})
    def broken_model(*args):
        raise ValueError("initialization failed")
    monkeypatch.setattr(train, "YOLO", broken_model)
    with pytest.raises(ValueError, match="initialization"):
        train.main()
    saved = json.loads(next((tmp_path / "_manifests").glob("*.json")).read_text(encoding="utf-8"))
    assert saved["status"] == "failed" and "initialization failed" in saved["error"]


def test_optimizer_hook_counts_updates_not_batch_calls(tmp_path):
    import torch
    from types import SimpleNamespace
    from steel_defect.training_audit import OptimizerAudit
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=.1)
    audit = OptimizerAudit()
    optimizer.register_step_post_hook(audit.step)
    for i in range(8):
        parameter.square().backward()
        if i % 4 == 3:
            optimizer.step()
            optimizer.zero_grad()
    assert audit.updates == 2


def test_learning_gate_fails_closed():
    from scripts.run_m0 import learning_gate
    good = {"summary": {"map50_macro": .95, "recall_macro": .95}}
    assert learning_gate(good, 800, True)
    assert not learning_gate(good, 40, True)
    assert not learning_gate(good, 800, False)
    assert not learning_gate(good, 800, True, numerical_healthy=False)
    assert not learning_gate({"summary": {"map50_macro": float("nan"), "recall_macro": 1}}, 800, True)


def test_bn_freeze_preserves_statistics_but_trains_affine():
    import torch
    from steel_defect.training_audit import freeze_bn_statistics
    model = torch.nn.Sequential(torch.nn.BatchNorm2d(3))
    model.train()
    before = model[0].running_mean.clone()
    freeze_bn_statistics(model)
    result = model(torch.randn(2, 3, 8, 8) + 10)
    result.square().mean().backward()
    assert torch.equal(model[0].running_mean, before)
    assert model[0].weight.requires_grad and model[0].weight.grad is not None


def test_frozen_source_hash_detects_mutation(tmp_path):
    from steel_defect.governance import digest_file
    image, xml = tmp_path / "a.jpg", tmp_path / "a.xml"
    image.write_bytes(b"image-v1")
    xml.write_bytes(b"label-v1")
    record = VocRecord(image, xml, 10, 10, ())
    manifest = {"splits": {"train": ["a.jpg"]}, "groups": {"train": ["g"]},
                "records": {"a.jpg": {"group": "g", "sha256": digest_file(image), "xml_sha256": digest_file(xml)}}}
    assert frozen_records([record], tmp_path, manifest)["train"][0].group_id == "g"
    xml.write_bytes(b"label-v2")
    with pytest.raises(ValueError, match="changed"):
        frozen_records([record], tmp_path, manifest)


def test_rejected_visible_fragment_never_becomes_background(tmp_path):
    import numpy as np
    from types import SimpleNamespace
    from scripts.prepare_data import process_record
    from steel_defect.voc import Annotation
    from steel_defect.image_io import imwrite
    path = tmp_path / "test.jpg"
    imwrite(path, np.zeros((64, 128), dtype=np.uint8))
    record = VocRecord(path, path.with_suffix(".xml"), 128, 64,
                       (Annotation("defect", 0, (60, 10, 100, 50)),))
    args = SimpleNamespace(view_mode="tiles", tile_size=64, overlap=0, min_visibility=.35,
                           seed=42, negative_ratio=1, output=tmp_path / "out", deblur=False,
                           clahe=False, deblur_threshold=85)
    metadata, _ = process_record(record, "train", args, ["defect"])
    assert len(metadata) == 1
    assert metadata[0]["origin"] == [64, 0] and metadata[0]["objects"] == 1


def test_zero_positive_loss_collapse_is_rejected(tmp_path):
    import torch
    from types import SimpleNamespace
    from steel_defect.training_audit import OptimizerAudit
    audit = OptimizerAudit()
    audit.batch_path = tmp_path / "batches.jsonl"
    audit.assignment = {"gt_objects": 1, "foreground_anchors": 0}
    trainer = SimpleNamespace(epoch=0, loss=torch.tensor(0.0))
    for _ in range(24):
        audit.batch(trainer)
    with pytest.raises(FloatingPointError, match="zero loss"):
        audit.batch(trainer)


def test_m1_refuses_failed_learning_gate(tmp_path):
    import json
    from scripts.run_v2_baselines import check_gate
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"scope_complete": True, "gate": {"passed": False}}), encoding="utf-8")
    with pytest.raises(ValueError, match="not passed"):
        check_gate(gate)
