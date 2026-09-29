# WSM Stage 7 Detailed Plan

Status: **ACTIVE — candidate/config/seed freeze in progress; Final Test locked**.

This is a stage-specific detail file extracted from the former monolithic `docs/PLAN.md`.
Read it only when the active task belongs to this stage or when historical plan detail is explicitly needed.
Current project status is authoritative in [../PROGRESS_EN.md](../PROGRESS_EN.md).

### Stage 7 — Final evaluation

Manager-frozen compared methods before final-seed execution:

1. frozen temporal audio;
2. full R4 trial012;
3. equal-parameter shared-fusion trial012.

Frozen seeds for all compared methods: `42,43,44,45,46`. Existing accepted 42-44 checkpoints are reused. Only seeds45/46 require new training after TASK-007A passes review.

Statistical plan is to be frozen before new runs: five-seed mean, sample standard deviation, and two-sided 95% Student-t confidence interval (df=4), plus paired same-seed deltas/95% CIs for audio vs full R4, audio vs shared fusion, and full R4 vs shared fusion.

Final Test remains locked until all five checkpoints for every method are DEV-selected and frozen. Mandatory epoch-level Test monitoring outputs from training are not selection evidence and must not be inspected/used before the separate final-Test task.

After freeze:

1. train final compared methods on at least 5 seeds;
2. choose every checkpoint only by DEV/Mean_Score;
3. run separate TEST_NONE/SOFT/HARD evaluation once;
4. report per-task UAR/MF1/Score and Mean_Score;
5. report mean, standard deviation, confidence interval, and paired comparison where applicable;
6. trace every table entry to config, commit, checkpoint, and MLflow run.

Gate: no Test-driven revision; complete traceability; claims match annotation evidence.
