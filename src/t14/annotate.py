"""Confidence filter + suspicion score (higher = review first).

Box rows get a score from uncertainty signals. Extra image-level rows (box_id "img:<image_id>")
ask the reviewer to look for MISSED objects: confidence on the boxes we have says nothing
about boxes the model never produced (failure mode: high conf, small objects missing).
"""
from __future__ import annotations

import pandas as pd

SMALL_FRAC = 150 / (320 * 180)   # share of the image: 150 px^2 on the 320x180 fixture (the evaluator's
                                 # small-object slice), 2400 px^2 on 1280x720 BDD frames
MID_CONF = (0.25, 0.6)      # initial guess; tune on dev only, freeze before eval
WEIGHTS = {"mid_conf": 1.0, "near_thr": 0.5, "small_box": 1.0, "class_conflict": 1.5, "ped_wide": 1.0}


def iou(a, b) -> float:
    x1, y1 = max(a["x"], b["x"]), max(a["y"], b["y"])
    x2, y2 = min(a["x"] + a["w"], b["x"] + b["w"]), min(a["y"] + a["h"], b["y"] + b["h"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def score(ctx, boxes: pd.DataFrame, images: pd.DataFrame | None = None) -> pd.DataFrame:
    thr = float(ctx.param("conf_threshold", 0.25))
    b = boxes.reset_index(drop=True).copy()
    b["keep"] = b["conf"] >= thr
    small = b["w"] * b["h"] < SMALL_FRAC * b["width"] * b["height"]

    conflict = pd.Series(False, index=b.index)       # kept boxes of different class overlapping
    for _, g in b[b["keep"]].groupby("image_id"):
        idx = list(g.index)
        for i, p in enumerate(idx):
            for q in idx[i + 1:]:
                if g.at[p, "class"] != g.at[q, "class"] and iou(g.loc[p], g.loc[q]) > 0.3:
                    conflict[p] = conflict[q] = True

    sig = pd.DataFrame({
        "mid_conf": b["conf"].between(*MID_CONF),
        "near_thr": (b["conf"] - thr).abs() < 0.1,
        "small_box": small,
        "class_conflict": conflict,
        "ped_wide": (b["class"] == "pedestrian") & (b["w"] > b["h"]),   # likely a motorbike
    })
    b["score"] = sum(WEIGHTS[k] * sig[k] for k in WEIGHTS) * (1 - b["conf"] / 2)
    b["reason"] = [";".join(k for k in sig.columns if row[k]) or "low_risk" for row in sig.to_dict("records")]

    # Image-level rows: include images with zero predictions (taken from images.csv when given)
    src = images if images is not None else boxes
    img = src[["image_id", "path", "split", "width", "height"]].drop_duplicates("image_id").reset_index(drop=True)
    kept = b[b["keep"]]
    n_all = img["image_id"].map(kept.groupby("image_id").size()).fillna(0)
    n_small = img["image_id"].map(kept[small[kept.index]].groupby("image_id").size()).fillna(0)
    img["box_id"] = "img:" + img["image_id"].astype(str)
    img["score"] = 1.0 * (n_small == 0) + 0.5 * (n_all <= 2)
    img["reason"] = "image_check_missing_small"
    img["keep"] = False                                # not a box; never exported

    out = pd.concat([b, img], ignore_index=True)
    out["keep"] = out["keep"].astype(bool)
    return out[["box_id"] + [c for c in out.columns if c != "box_id"]]
