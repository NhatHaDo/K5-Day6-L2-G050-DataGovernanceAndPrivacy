"""T14 baseline: auto-label at the default threshold and export WITHOUT human review.

    python -m src.main --config configs/lab.yaml --track t14 --pipeline src.t14.baseline \
        --output out/baseline --run-name baseline --offline --seed 42
"""
from __future__ import annotations

from ..common.context import RunContext
from ..common.decision import Decision, write_decision
from ..pipeline import load


def run(ctx: RunContext) -> None:
    boxes = load(ctx)["boxes"]                        # same provenance check + label map as the PoC
    out = boxes[boxes["conf"] >= float(ctx.param("conf_threshold", 0.25))]
    ctx.stats["n_items"] = len(out)
    out[["image_id", "class", "x", "y", "w", "h", "conf"]].to_csv(ctx.out("predictions.csv"), index=False)
    write_decision(Decision(status="HOLD", reason_code="BASELINE_NO_REVIEW",
                            reason="Auto-labels exported without human review: never release on this",
                            owner=ctx.config.get("owner", "baseline"), scope=ctx.track,
                            evidence=[str(ctx.out("predictions.csv"))]),
                   ctx.out("decision.json"), ctx.out("decision_log.csv"))
