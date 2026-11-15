"""Group tests for T14: label map, suspicion ranking, review apply, export, end-to-end run."""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src import main as runner
from src.common.offline import disable_offline_guard
from src.common.review import REVIEW_COLUMNS, AuditLog, load_review_decisions
from src.t14.annotate import score
from src.t14.data import map_labels
from src.t14.review_export import apply_review, from_yolo, to_yolo

LABEL_MAP = {0: "person", 2: "car", 3: "motorcycle", 11: "stop sign"}
TAXONOMY = {"car", "pedestrian", "motorbike", "sign"}


def _ctx(tmp_path=None, thr=0.25):
    ctx = SimpleNamespace(param=lambda k, d=None: {"conf_threshold": thr}.get(k, d), stats={})
    if tmp_path is not None:
        ctx.output_dir = tmp_path
        ctx.out = lambda name: tmp_path / name
        ctx.audit = AuditLog(tmp_path / "audit_log.jsonl")
    return ctx


def _boxes(rows):
    df = pd.DataFrame(rows, columns=["image_id", "class", "x", "y", "w", "h", "conf"])
    df.insert(0, "box_id", df["image_id"] + "#" + df.groupby("image_id").cumcount().astype(str))
    return df.assign(model_class_id=0, path="images/a.jpg", split="dev", width=320, height=180)


# ---------------------------------------------------------------- label map (failure mode 1)
def test_label_map_goes_through_names_not_ids():
    # model id 0 is `person`; reading it as taxonomy id 0 would silently export `car`
    assert map_labels(pd.Series([0, 2, 3, 11]), LABEL_MAP, TAXONOMY).tolist() == ["pedestrian", "car", "motorbike", "sign"]


def test_unknown_model_class_id_fails_loud():
    with pytest.raises(ValueError):
        map_labels(pd.Series([0, 99]), LABEL_MAP, TAXONOMY)


# ---------------------------------------------------------------- filter + rank
def test_small_midconf_box_ranked_above_big_confident_box():
    s = score(_ctx(), _boxes([("a", "car", 0, 0, 100, 60, 0.95), ("a", "sign", 150, 10, 8, 8, 0.30)])).set_index("box_id")
    assert s.loc["a#1", "score"] > s.loc["a#0", "score"]
    assert "small_box" in s.loc["a#1", "reason"]


def test_small_box_threshold_scales_with_image_size():
    big = _boxes([("a", "car", 0, 0, 40, 40, 0.9)]).assign(width=1280, height=720)   # 1600 px^2
    assert "small_box" in score(_ctx(), big).set_index("box_id").loc["a#0", "reason"]
    assert "small_box" not in score(_ctx(), _boxes([("a", "car", 0, 0, 40, 40, 0.9)])).set_index("box_id").loc["a#0", "reason"]


def test_box_below_threshold_not_kept_and_image_row_added():
    s = score(_ctx(), _boxes([("a", "car", 0, 0, 50, 50, 0.10)])).set_index("box_id")
    assert not s.loc["a#0", "keep"]
    assert "img:a" in s.index and not s.loc["img:a", "keep"]     # image-level check for missed objects


# ---------------------------------------------------------------- review + export
def _reviewed(tmp_path, rows):
    full = [{c: "" for c in REVIEW_COLUMNS} | r for r in rows]
    pd.DataFrame(full, columns=REVIEW_COLUMNS).to_csv(tmp_path / "review_queue_reviewed.csv", index=False)


def test_review_reject_edit_add_are_applied_and_audited(tmp_path):
    scored = score(_ctx(), _boxes([("a", "car", 0, 0, 100, 60, 0.95), ("a", "pedestrian", 150, 10, 30, 12, 0.40),
                                   ("a", "sign", 200, 100, 9, 9, 0.50)]))
    who = {"reviewer": "tester"}
    _reviewed(tmp_path, [who | {"item_id": "a#0", "reviewer_decision": "reject"},
                         who | {"item_id": "a#1", "reviewer_decision": "edit", "override_value": "class=motorbike"},
                         who | {"item_id": "img:a", "reviewer_decision": "edit", "override_value": "add=sign,280,38,18,17"}])
    ctx = _ctx(tmp_path)
    final = apply_review(ctx, scored).set_index("box_id")
    assert "a#0" not in final.index                               # deleted
    assert final.loc["a#1", "class"] == "motorbike"               # class fixed
    assert final.loc["a#add0", ["x", "y", "w", "h"]].tolist() == [280, 38, 18, 17]   # missed box added
    assert "a#2" in final.index                                   # untouched box survives
    assert [e["action"] for e in ctx.audit.read()] == ["review:reject", "review:edit", "review:add"]
    assert all(e["actor"] == "tester" for e in ctx.audit.read())


def test_review_row_without_reviewer_fails(tmp_path):
    _reviewed(tmp_path, [{"item_id": "a#0", "reviewer_decision": "reject"}])
    with pytest.raises(ValueError):
        load_review_decisions(tmp_path / "review_queue_reviewed.csv")


def test_yolo_roundtrip_within_half_pixel():
    box, W, H = (88.91, 102.9, 39.3, 25.53), 320, 180
    norm = [round(v, 6) for v in to_yolo(*box, W, H)]              # 6 decimals, as written to the label file
    assert all(0 <= v <= 1 for v in norm)
    assert all(abs(a - b) < 0.5 for a, b in zip(from_yolo(*norm, W, H), box))


# ---------------------------------------------------------------- end to end
@pytest.mark.skipif(not Path("data/real/t14_bdd/manifest.csv").exists(),
                    reason="real BDD pack is gitignored: build it with python -m tools.make_t14_bdd")
def test_t14_pipeline_end_to_end(tmp_path):
    try:
        rc = runner.main(["--track", "t14", "--output", str(tmp_path), "--offline", "--seed", "42"])
    finally:
        disable_offline_guard()
    assert rc == 0
    pred = pd.read_csv(tmp_path / "predictions.csv")
    assert set(pred["class"]) <= TAXONOMY and len(pred) > 0
    assert (pred["w"] > 0).all() and (pred["x"] >= 0).all()
    queue = pd.read_csv(tmp_path / "review_queue.csv")
    assert len(queue) == 30
    assert queue["item_id"].str.startswith("img:").sum() == 6    # image_budget: missed-object checks get a seat
    for name in ("contact_sheet.html", "coco.json", "yolo/classes.txt", "audit_log.jsonl"):
        assert (tmp_path / name).exists(), name
    decision = json.loads((tmp_path / "decision.json").read_text())
    assert decision["status"] == "HOLD" and decision["reason_code"] == "REVIEW_PENDING"   # nobody reviewed yet
