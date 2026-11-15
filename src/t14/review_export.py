"""T14 human review apply + export (YOLO / COCO) + QA gate + release decision.

Reviewer decisions (src/common/review.py): approve | reject | edit | escalate | skip.
  approve                      keep the box (also rescues a box below the confidence threshold)
  reject                       DELETE the box
  edit + override_value        on a box row:   class=motorbike | box=x,y,w,h | class=car;box=x,y,w,h
                               on an img: row: add=sign,280,38,18,17|pedestrian,50,90,10,30  (missed boxes)
  escalate / skip              no change; decide() holds the release
"""
from __future__ import annotations

import json

import pandas as pd
import yaml

from ..common.decision import Decision
from ..common.review import load_review_decisions

COLORS = {"car": "red", "pedestrian": "lime", "motorbike": "orange", "sign": "cyan"}
STATE = ["class", "x", "y", "w", "h", "keep"]


def _is_img(ids: pd.Series) -> pd.Series:
    return ids.astype(str).str.startswith("img:")


def boxes_by_item(scored: pd.DataFrame) -> dict:
    """item_id -> boxes drawn on the contact sheet. Box row: that box. img: row: all kept boxes."""
    def draw(records):
        return [{"x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"], "color": COLORS.get(r["class"], "red"),
                 "label": f'{r["class"]} {r["conf"]:.2f}'} for r in records]
    boxes = scored[~_is_img(scored["box_id"])]
    out = {r["box_id"]: draw([r]) for r in boxes.to_dict("records")}
    kept = {img: draw(g.to_dict("records")) for img, g in boxes[boxes["keep"].astype(bool)].groupby("image_id")}
    for r in scored[_is_img(scored["box_id"])].to_dict("records"):
        out[r["box_id"]] = kept.get(r["image_id"], [])
    return out


def _parse(value: str) -> dict:
    return dict(p.split("=", 1) for p in str(value).split(";") if "=" in p)


def _state(boxes: pd.DataFrame, mask) -> dict:
    return json.loads(boxes.loc[mask, STATE].iloc[0].to_json())


def apply_review(ctx, scored: pd.DataFrame) -> pd.DataFrame:
    """Apply <output>/review_queue_reviewed.csv if present. Returns the kept boxes (image rows dropped)."""
    boxes = scored[~_is_img(scored["box_id"])].copy()
    meta = scored.drop_duplicates("image_id").set_index("image_id")
    path = ctx.output_dir / "review_queue_reviewed.csv"
    diffs = []
    if path.exists():
        rev = load_review_decisions(path)            # raises on unknown decision / missing reviewer
        ctx.stats["n_reviewed"] = len(rev)
        for r in rev.to_dict("records"):
            item, decision = r["item_id"], r["reviewer_decision"]
            if item.startswith("img:"):
                if decision != "edit":
                    continue
                img = item[4:]
                if img not in meta.index:
                    raise ValueError(f"review row {item}: unknown image")
                m = meta.loc[img]
                specs = [s for s in _parse(r["override_value"]).get("add", "").split("|") if s.strip()]
                for k, spec in enumerate(specs):
                    parts = [p.strip() for p in spec.split(",")]
                    if len(parts) != 5:
                        raise ValueError(f"review row {item}: add= needs class,x,y,w,h, got {spec!r}")
                    new = {"box_id": f"{img}#add{k}", "image_id": img, "class": parts[0],
                           "x": float(parts[1]), "y": float(parts[2]), "w": float(parts[3]), "h": float(parts[4]),
                           "conf": 1.0, "keep": True, "path": m["path"], "split": m["split"],
                           "width": m["width"], "height": m["height"], "reason": "added_by_reviewer"}
                    boxes = pd.concat([boxes, pd.DataFrame([new])], ignore_index=True)
                    diffs.append({"action": "add", "item_id": new["box_id"], "before": None,
                                  "after": {c: new[c] for c in STATE}, "reviewer": r["reviewer"], "note": r["review_note"]})
                continue
            mask = boxes["box_id"] == item
            if not mask.any() or decision in ("escalate", "skip"):
                continue
            before = _state(boxes, mask)
            if decision == "reject":
                boxes.loc[mask, "keep"] = False
            else:                                     # approve / edit
                if decision == "edit":
                    p = _parse(r["override_value"])
                    if "class" in p:
                        boxes.loc[mask, "class"] = p["class"].strip()
                    if "box" in p:
                        xywh = [float(t) for t in p["box"].split(",")]
                        if len(xywh) != 4:
                            raise ValueError(f"review row {item}: box= needs x,y,w,h")
                        boxes.loc[mask, ["x", "y", "w", "h"]] = xywh
                boxes.loc[mask, "keep"] = True
            after = _state(boxes, mask)
            if after != before:
                diffs.append({"action": decision, "item_id": item, "before": before, "after": after,
                              "reviewer": r["reviewer"], "note": r["review_note"]})
    for d in diffs:
        ctx.audit.log(actor=d["reviewer"], action=f"review:{d['action']}", item_id=d["item_id"],
                      before=d["before"], after=d["after"], reason=d["note"])
    pd.DataFrame(diffs, columns=["action", "item_id", "before", "after", "reviewer", "note"]).to_csv(
        ctx.out("review_diff.csv"), index=False)
    ctx.stats["n_review_changes"] = len(diffs)
    return boxes[boxes["keep"].astype(bool)].reset_index(drop=True)


def to_yolo(x, y, w, h, W, H) -> tuple:
    """Pixel xywh (top-left origin) -> normalised YOLO cx, cy, w, h."""
    return (x + w / 2) / W, (y + h / 2) / H, w / W, h / H


def from_yolo(cx, cy, nw, nh, W, H) -> tuple:
    return (cx - nw / 2) * W, (cy - nh / 2) * H, nw * W, nh * H


def export(ctx, final: pd.DataFrame) -> pd.DataFrame:
    """Clip to image, write YOLO + COCO, record QA. Returns the valid boxes (= predictions.csv)."""
    tax = yaml.safe_load((ctx.input_dir / "taxonomy.yaml").read_text(encoding="utf-8"))["classes"]
    tax = {int(k): v for k, v in tax.items()}
    name2id = {v: k for k, v in tax.items()}
    f = final.copy()
    x2, y2 = (f["x"] + f["w"]).clip(upper=f["width"]), (f["y"] + f["h"]).clip(upper=f["height"])
    f["x"], f["y"] = f["x"].clip(lower=0), f["y"].clip(lower=0)
    f["w"], f["h"] = x2 - f["x"], y2 - f["y"]
    invalid = ~f["class"].isin(list(name2id)) | ~(f["w"] > 0) | ~(f["h"] > 0)
    ok = f[~invalid].reset_index(drop=True)

    labels = ctx.out("yolo/labels/.keep").parent
    for old in labels.glob("*.txt"):                  # our own previous export; avoid stale labels
        old.unlink()
    (labels.parent / "classes.txt").write_text("\n".join(tax[k] for k in sorted(tax)) + "\n", encoding="utf-8")
    for img, g in ok.groupby("image_id"):
        lines = []
        for r in g.to_dict("records"):
            cx, cy, nw, nh = to_yolo(r["x"], r["y"], r["w"], r["h"], r["width"], r["height"])
            lines.append(f"{name2id[r['class']]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (labels / f"{img}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    img_rows = ok.drop_duplicates("image_id").to_dict("records")
    img_idx = {r["image_id"]: i for i, r in enumerate(img_rows)}
    coco = {
        "images": [{"id": i, "file_name": str(r["path"]), "width": int(r["width"]), "height": int(r["height"])}
                   for i, r in enumerate(img_rows)],
        "categories": [{"id": k, "name": v} for k, v in sorted(tax.items())],
        "annotations": [{"id": n, "image_id": img_idx[r["image_id"]], "category_id": name2id[r["class"]],
                         "bbox": [float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])],
                         "area": float(r["w"] * r["h"]), "iscrowd": 0} for n, r in enumerate(ok.to_dict("records"))],
    }
    ctx.out("coco.json").write_text(json.dumps(coco), encoding="utf-8")

    qa = {"n_export": len(ok), "n_invalid": int(invalid.sum()),
          "invalid_rate": float(invalid.mean()) if len(f) else 0.0}
    ctx.stats["qa"] = qa
    if qa["n_invalid"]:
        ctx.audit.log(actor="pipeline", action="qa_invalid_dropped", after=f.loc[invalid, "box_id"].tolist())
    return ok


def decide(ctx, final: pd.DataFrame) -> Decision:
    owner = ctx.config.get("owner", "TODO-owner")
    qa = ctx.stats.get("qa", {})
    ev = [str(ctx.output_dir / n) for n in ("predictions.csv", "coco.json", "review_diff.csv", "audit_log.jsonl")]

    def mk(status, code, reason):
        return Decision(status=status, reason_code=code, reason=reason, owner=owner, scope="t14", evidence=ev, metrics=qa)

    if qa.get("n_invalid", 1) > 0:
        return mk("REWORK", "INVALID_LABELS", f"{qa.get('n_invalid', '?')} invalid boxes found at export QA")
    reviewed, queue = ctx.output_dir / "review_queue_reviewed.csv", ctx.output_dir / "review_queue.csv"
    if not reviewed.exists():
        return mk("HOLD", "REVIEW_PENDING", "Review queue has not been reviewed yet")
    dec = pd.read_csv(reviewed, dtype=str).fillna("")
    done = dict(zip(dec["item_id"], dec["reviewer_decision"].str.strip().str.lower()))
    todo = pd.read_csv(queue, dtype=str)["item_id"].tolist() if queue.exists() else list(done)
    pending = [i for i in todo if done.get(i, "") in ("", "escalate")]
    if pending:
        return mk("HOLD", "REVIEW_INCOMPLETE", f"{len(pending)} queued items unreviewed or escalated")
    return mk("GO", "QA_PASSED", "All queued items reviewed; 0 invalid labels after export")
