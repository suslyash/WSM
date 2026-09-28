# WSM Stage 6 Detailed Plan

Status: **ACTIVE**.

This is a stage-specific detail file extracted from the former monolithic `docs/PLAN.md`.
Read it only when the active task belongs to this stage or when historical plan detail is explicitly needed.
Current project status is authoritative in [../PROGRESS_EN.md](../PROGRESS_EN.md).

### Stage 6 — Ablations and claims audit

Required:

1. F1 sparse MTL;
2. + task-aware fusion;
3. + direct pseudo-supervision;
4. + uncertainty/reliability;
5. + semantic evidence;
6. + RA-STCH;
7. remove each modality;
8. shuffled/mismatched pseudo-target negative control;
9. corpus probe;
10. equal-parameter control if size grows materially.

Log DEV scores, UAR/MF1, ECE/Brier, pseudo coverage by class, task gradient norms/cosines, negative-transfer deltas, and gate distributions. Use 3 seeds for ablations.

If a dual-annotated audit subset exists, measure missing-label precision/recall/calibration but do not use it for ordinary training.

Gate: each claim maps to an ablation; negative results are retained; final config/thresholds are immutable.
