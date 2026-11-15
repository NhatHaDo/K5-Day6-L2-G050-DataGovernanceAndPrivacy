"""Build a T14 input pack from real BDD100K images + REAL YOLO inference (run once, offline after weights exist).

    python -m tools.make_t14_bdd --images data/images --weights models_optional/yolo26n.pt --out data/real/t14_bdd

Output has the same layout as fixtures/t14_auto_annotation, so the pipeline runs unchanged:
    python -m src.main --config configs/lab.yaml --track t14 --input data/real/t14_bdd --output out/poc_bdd --offline --seed 42

Ground truth (gt/gt_boxes.csv) is converted from the dataset's own YOLO labels (--labels), mapped by
class NAME to the project taxonomy. The pipeline must never read gt/ (GT guard); only src.evaluate does:
    python -m src.evaluate --track t14 --gt data/real/t14_bdd/gt/gt_boxes.csv --pred out/poc_bdd/predictions.csv
Real people / plates: the output lives under data/real/ (gitignored). Never commit it.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.common.io import sha256_file
from src.common.manifest import build_manifest, write_manifest

SOURCE = "kaggle:a7madmostafa/bdd100k-yolo (BDD100K images)"
BDD_LICENSE = "BDD100K License: non-commercial research/education only (https://doc.bdd100k.com/license.html)"
MODEL_LICENSE = "AGPL-3.0 (Ultralytics); commercial closed-source use needs an Enterprise license"
KEEP_IDS = {0: "person", 2: "car", 3: "motorcycle", 11: "stop sign"}   # COCO ids that map to the project taxonomy
TAXONOMY = {0: "car", 1: "pedestrian", 2: "motorbike", 3: "sign"}
# Label ids of the Kaggle YOLO export, same order as data/data.yaml; re-checked at build time (car agreement below).
BDD_NAMES = ["person", "rider", "car", "bus", "truck", "bike", "motor", "traffic light", "traffic sign", "train"]
# COCO `person` also fires on riders, so both map to pedestrian. Other BDD classes are dropped, like their COCO twins.
BDD_TO_TAXONOMY = {"person": "pedestrian", "rider": "pedestrian", "car": "car", "motor": "motorbike", "traffic sign": "sign"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--labels", default="data/labels", help="YOLO txt labels of the same images (ground truth)")
    ap.add_argument("--weights", default="models_optional/yolo26n.pt")
    ap.add_argument("--out", default="data/real/t14_bdd")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--conf", type=float, default=0.05, help="low on purpose: the pipeline applies the real threshold")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    weights = Path(a.weights)
    if not weights.exists():           # fail loud, never auto-download
        raise SystemExit(f"weights not found: {weights} (download once before class, record sha256)")
    src = sorted(Path(a.images).glob("*.jpg"))
    if len(src) < a.n:
        raise SystemExit(f"need {a.n} images in {a.images}, found {len(src)}")

    rng = np.random.default_rng(a.seed)
    chosen = [src[i] for i in sorted(rng.choice(len(src), a.n, replace=False))]
    eval_idx = set(rng.choice(a.n, a.n_eval, replace=False).tolist())

    from PIL import Image
    from ultralytics import YOLO       # import BEFORE wiping the old pack: a missing dependency must not destroy it

    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)

    model = YOLO(str(weights))
    rows, preds, gt = [], [], []
    for i, p in enumerate(chosen):
        iid = f"bdd_{i:02d}"
        rel = f"images/{iid}.jpg"
        shutil.copy2(p, out / rel)
        W, H = Image.open(out / rel).size
        rows.append({"image_id": iid, "path": rel, "width": W, "height": H,
                     "split": "eval" if i in eval_idx else "dev", "source_file": p.name})
        label = Path(a.labels) / f"{p.stem}.txt"
        if not label.exists():
            raise SystemExit(f"no ground-truth label file for {p.name}: {label}")
        for line in label.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            c, cx, cy, nw, nh = line.split()
            name = BDD_TO_TAXONOMY.get(BDD_NAMES[int(c)])      # IndexError on an unknown id: fail loud
            if name:
                gt.append({"image_id": iid, "class": name, "x": round((float(cx) - float(nw) / 2) * W, 2),
                           "y": round((float(cy) - float(nh) / 2) * H, 2), "w": round(float(nw) * W, 2),
                           "h": round(float(nh) * H, 2), "split": rows[-1]["split"]})
        r = model.predict(str(out / rel), conf=a.conf, verbose=False, device="cpu")[0]
        for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            if int(c) in KEEP_IDS:
                preds.append({"image_id": iid, "class_id": int(c), "x": round(x1, 2), "y": round(y1, 2),
                              "w": round(x2 - x1, 2), "h": round(y2 - y1, 2), "conf": round(s, 3)})

    images = pd.DataFrame(rows)
    images.drop(columns="source_file").to_csv(out / "images.csv", index=False)
    images[["image_id", "source_file"]].to_csv(out / "source_files.csv", index=False)
    pd.DataFrame(preds, columns=["image_id", "class_id", "x", "y", "w", "h", "conf"]).to_csv(
        out / "model_predictions_fixture.csv", index=False)

    (out / "taxonomy.yaml").write_text("# Project taxonomy (export target)\n" + yaml.safe_dump({"classes": TAXONOMY}, sort_keys=False),
                                       encoding="utf-8")
    card = {
        "model_name": "ultralytics/yolo26n", "version": f"yolo26n.pt sha256={sha256_file(weights)}",
        "license": MODEL_LICENSE, "inference": f"REAL inference, CPU, conf>={a.conf}, other COCO classes dropped",
        "label_map": dict(KEEP_IDS),
        "notes": "COCO label map, NOT the project taxonomy ids. COCO has only 'stop sign', so most BDD traffic signs are missed.",
    }
    (out / "model_card_fixture.yaml").write_text("# REAL model inference on real images\n" + yaml.safe_dump(card, sort_keys=False),
                                                 encoding="utf-8")

    files = ["images.csv", "model_predictions_fixture.csv", "taxonomy.yaml", "model_card_fixture.yaml", *images["path"]]
    split = dict(zip(images["path"], images["split"]))
    write_manifest(build_manifest(out, files, SOURCE, BDD_LICENSE, split=split), out / "manifest.csv")
    (out / "gt").mkdir()
    gt = pd.DataFrame(gt, columns=["image_id", "class", "x", "y", "w", "h", "split"])
    gt.to_csv(out / "gt" / "gt_boxes.csv", index=False)
    write_manifest(build_manifest(out, ["gt/gt_boxes.csv"], SOURCE + " YOLO labels", BDD_LICENSE), out / "gt" / "manifest.csv")

    # Class-order self-check: confident model `car` boxes must mostly sit on a GT `car`.
    from src.t14.annotate import iou
    cars = [p for p in preds if p["class_id"] == 2 and p["conf"] >= 0.6]
    gt_cars = {k: g.to_dict("records") for k, g in gt[gt["class"] == "car"].groupby("image_id")}
    hit = sum(any(iou(p, g) >= 0.5 for g in gt_cars.get(p["image_id"], [])) for p in cars)
    print(f"[t14_bdd] label-order check: {hit}/{len(cars)} confident model cars match a GT car")
    if not cars or hit / len(cars) < 0.5:
        raise SystemExit("GT class order looks wrong (BDD_NAMES): fix it before using gt/gt_boxes.csv")
    print("          gt per class:", gt["class"].value_counts().to_dict(),
          "| eval:", gt[gt["split"] == "eval"]["class"].value_counts().to_dict())
    print(f"[t14_bdd] {len(images)} images ({(images.split == 'eval').sum()} eval), {len(preds)} boxes -> {out}")
    print("          per class:", pd.DataFrame(preds)["class_id"].map(KEEP_IDS).value_counts().to_dict() if preds else {})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
