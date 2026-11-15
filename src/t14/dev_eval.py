"""Internal check on the DEV split of the real BDD pack (for tuning thresholds/weights).
NOT the reported score: that is src.evaluate on the eval split.

    python -m src.t14.dev_eval out/baseline/predictions.csv out/poc/predictions.csv
"""
import sys
from pathlib import Path

import pandas as pd

from ..metrics.detection import detection_prf

GOLD = Path("data/real/t14_bdd/gt/gt_boxes.csv")     # BDD labels; only the dev split is used here


def main(paths: list[str]) -> int:
    if not GOLD.exists():
        print(f"{GOLD} not found: build the pack first (python -m tools.make_t14_bdd)", file=sys.stderr)
        return 2
    gold = pd.read_csv(GOLD)
    gold = gold[gold["split"] == "dev"].reset_index(drop=True)      # tune on dev only, never on eval
    small = gold[gold["w"] * gold["h"] < 150].reset_index(drop=True)
    for path in paths:
        pred = pd.read_csv(path)
        pred = pred[pred["image_id"].isin(gold["image_id"])].reset_index(drop=True)
        r = detection_prf(gold, pred, 0.5, True)
        rs = detection_prf(small, pred, 0.5, True)["recall"] if len(small) else None
        print(f"{path}: F1={r['f1']:.3f} P={r['precision']} R={r['recall']} small_recall={rs} "
              f"n_gt={len(gold)} n_images={gold['image_id'].nunique()} (dev split, BDD labels)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
