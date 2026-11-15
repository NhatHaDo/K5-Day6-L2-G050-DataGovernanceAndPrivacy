Baseline: `baseline` (git 0e9e23b, seed 42) · PoC: `poc` (git 0e9e23b, seed 42)

| Metric (unit) | Slice | Baseline | PoC | Δ abs | Δ rel | Better | N |
|---|---|---|---|---|---|---|---|
| **Detection F1@IoU0.5** (class-aware greedy match) | all | 0.4381 | 0.4381 | 0.0000 | +0.0% | higher | 287 |
| Precision (TP/(TP+FP)) | all | 0.6000 | 0.6000 | 0.0000 | +0.0% | higher | None |
| Recall (TP/(TP+FN)) | all | 0.3449 | 0.3449 | 0.0000 | +0.0% | higher | None |
| Invalid-label rate (exported boxes with unknown class or out-of-bounds / exported boxes) | all | 0.0000 | 0.0000 | 0.0000 | baseline=0 | lower | None |
| Runtime / 100 items (s) | all | 0.0191 | 0.1070 | | | lower | |
