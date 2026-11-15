"""YOUR PoC PIPELINE — this is the file your group owns.

Lab contract = 5 automated steps, run by `python -m src.main ...`:
    01 load      input + provenance check (manifest.csv)
    02 process   your algorithm / model (the research part)
    03 rank      prioritise for a fixed review budget B
    04 review    human-in-the-loop: queue -> reviewer decisions -> overrides + audit log
    05 decide    GO / HOLD / REWORK (or track equivalent) with reason_code + owner, write outputs

Required output: out/<pred_file> with the columns listed for your track in
docs/tracks/TXX.md (also in src/metrics/tracks.py::TRACKS). Then evaluate with
    python -m src.evaluate --track tXX --pred out/<pred_file>

Rules: never read fixtures/*/gt/ here (the GT guard will stop you); seed everything via ctx.rng;
no network calls; fail loudly on missing models instead of downloading.
See src/examples/random_baseline.py for a minimal end-to-end wiring example.
"""
from __future__ import annotations

import pandas as pd

from .common.contact_sheet import render_contact_sheet
from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.manifest import verify_manifest
from .common.review import make_review_queue, write_review_queue
from .metrics.tracks import get_track
from .t14 import annotate as A
from .t14 import data as D
from .t14 import review_export as R


# ---------------------------------------------------------------- 01 load
def load(ctx: RunContext) -> dict:
    if ctx.track != "t14":      # this group's pipeline only implements T14
        raise NotImplementedError(f"src.pipeline implements t14 only, not {ctx.track}")
    manifest = ctx.input_dir / "manifest.csv"
    problems = verify_manifest(manifest) if manifest.exists() else [{"problem": "no_manifest"}]
    if problems:
        ctx.audit.log(actor="pipeline", action="provenance_problems", item_id=str(manifest), after=problems[:20])
        if ctx.config.get("strict_provenance", True):
            raise ValueError(f"provenance check failed ({len(problems)} problems), first: {problems[:3]}")
    return D.load(ctx)          # boxes with taxonomy class names (label map validated)


# ---------------------------------------------------------------- 02 process
def process(ctx: RunContext, data: dict) -> pd.DataFrame:
    """Confidence filter + suspicion score; adds image-level rows for missed-object checks."""
    return A.score(ctx, data["boxes"], data["images"])


# ---------------------------------------------------------------- 03 rank
def rank(ctx: RunContext, scored: pd.DataFrame) -> pd.DataFrame:
    """Top-B most suspicious boxes / images -> review_queue.csv + offline HTML contact sheet."""
    budget = int(ctx.param("budget", 30))
    # Reserve part of the budget for image-level "missed object" checks: box scores alone crowd them out.
    is_img = scored["box_id"].astype(str).str.startswith("img:")
    top_img = scored[is_img].sort_values("score", ascending=False, kind="mergesort").head(int(ctx.param("image_budget", 6)))
    top_box = scored[~is_img].sort_values("score", ascending=False, kind="mergesort").head(budget - len(top_img))
    q = make_review_queue(pd.concat([top_box, top_img]), id_col="box_id", score_col="score", budget=budget,
                          reason_col="reason", evidence_col="path")
    write_review_queue(q, ctx.out("review_queue.csv"))
    render_contact_sheet(q, ctx.out("contact_sheet.html"), image_root=ctx.input_dir,
                         boxes_by_item=R.boxes_by_item(scored), title="T14 review queue")
    return q


# ---------------------------------------------------------------- 04 review
def review(ctx: RunContext, scored: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    """Apply <output>/review_queue_reviewed.csv (approve / edit / delete / add), then export YOLO + COCO + QA.

    Returns the final valid boxes, which run() writes to predictions.csv.
    """
    return R.export(ctx, R.apply_review(ctx, scored))


# ---------------------------------------------------------------- 05 decide
def decide(ctx: RunContext, final: pd.DataFrame) -> Decision:
    """REWORK on invalid labels, HOLD while review is pending/escalated, else GO (policy fixed before eval)."""
    return R.decide(ctx, final)


def run(ctx: RunContext) -> None:
    spec = get_track(ctx.track)
    data = load(ctx)
    scored = process(ctx, data)
    ctx.stats["n_items"] = len(scored)
    queue = rank(ctx, scored)
    final = review(ctx, scored, queue)
    final[spec.pred_columns + [c for c in final.columns if c not in spec.pred_columns]].to_csv(ctx.out(spec.pred_file), index=False)
    write_decision(decide(ctx, final), ctx.out("decision.json"), ctx.out("decision_log.csv"))
