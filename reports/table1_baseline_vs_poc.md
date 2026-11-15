Baseline: `baseline` (git b0ce8bf, seed 42) · PoC: `poc` (git b84f491, seed 42)

| Metric (unit) | Slice | Baseline | PoC | Δ abs | Δ rel | Better | N |
|---|---|---|---|---|---|---|---|
| **Detection F1@IoU0.5** (class-aware greedy match) | all | 0.4433 | 0.4263 | -0.0170 | -3.8% | higher | 287 |
| Precision (TP/(TP+FP)) | all | 0.8571 | 0.8710 | 0.0138 | +1.6% | higher | None |
| Recall (TP/(TP+FN)) | all | 0.2990 | 0.2822 | -0.0168 | -5.6% | higher | None |
| Invalid-label rate (exported boxes with unknown class or out-of-bounds / exported boxes) | all | 0.0000 | 0.0000 | 0.0000 | baseline=0 | lower | None |
| Runtime / 100 items (s) | all | 0.0500 | 0.0285 | | | lower | |
