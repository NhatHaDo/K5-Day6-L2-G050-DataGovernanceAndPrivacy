"""T14 data loader: input pack tables + model->taxonomy label map (validated, fails loud).

Inputs (data/real/t14_bdd, built by tools/make_t14_bdd.py): images.csv, model_predictions_fixture.csv,
taxonomy.yaml, model_card_fixture.yaml. Never reads gt/.
"""
from __future__ import annotations

import pandas as pd
import yaml

# Model label names (COCO-style, see model_card_fixture.yaml) -> project taxonomy names.
# Map by NAME, never by numeric id: model id 0 is `person`, taxonomy id 0 is `car`.
MODEL_TO_TAXONOMY = {"person": "pedestrian", "car": "car", "motorcycle": "motorbike", "stop sign": "sign"}


def map_labels(class_ids: pd.Series, label_map: dict, taxonomy: set) -> pd.Series:
    """Model class_id -> taxonomy class name. Unknown id or unmapped name -> ValueError."""
    unknown = sorted(set(class_ids) - set(label_map))
    if unknown:
        raise ValueError(f"class_id not in model label_map: {unknown}")
    names = class_ids.map(label_map).map(MODEL_TO_TAXONOMY)
    bad = sorted(set(class_ids[names.isna() | ~names.isin(taxonomy)]))
    if bad:
        raise ValueError(f"model class ids {bad} have no project-taxonomy mapping")
    return names


def load(ctx) -> dict:
    d = ctx.input_dir
    images = pd.read_csv(d / "images.csv")
    pred = pd.read_csv(d / "model_predictions_fixture.csv")
    taxonomy = set(yaml.safe_load((d / "taxonomy.yaml").read_text(encoding="utf-8"))["classes"].values())
    card = yaml.safe_load((d / "model_card_fixture.yaml").read_text(encoding="utf-8"))
    label_map = {int(k): v for k, v in card["label_map"].items()}

    pred = pred.rename(columns={"class_id": "model_class_id"})
    pred["class"] = map_labels(pred["model_class_id"], label_map, taxonomy)
    pred = pred.merge(images[["image_id", "path", "split", "width", "height"]], on="image_id", how="left")
    if pred["path"].isna().any():
        raise ValueError("predictions reference image_id values missing from images.csv")
    pred.insert(0, "box_id", pred["image_id"].astype(str) + "#" + pred.groupby("image_id").cumcount().astype(str))
    if ctx.audit:
        ctx.audit.log(actor="pipeline", action="label_map", after=MODEL_TO_TAXONOMY,
                      reason=f"{card.get('model_name')} {card.get('version')}")
    return {"boxes": pred, "images": images, "taxonomy": taxonomy, "model_card": card}
